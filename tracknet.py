"""
TrackNet-style ball tracker — encoder-decoder heatmap regression for small ball detection.

Architecture based on TrackNetV3 with U-Net skip connections:
  - Input: 3 consecutive RGB frames + 2 motion diff maps ->11-channel tensor (640×360)
  - Output: 1-channel sigmoid heatmap
  - Ball location: extracted via weighted centroid of peak region

Usage:
  Inference:  wrapper = TrackNetInference("tracknet_basketball.pt", device="cuda:0")
              result  = wrapper.predict(frame)   # feed every frame
  Training:   python tracknet.py --train --data data/tracknet_labels/
"""

import argparse
import csv
import gc
import math
import os
import random as pyrandom

from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

INPUT_W, INPUT_H = 640, 360    # TrackNet canonical resolution

# Top-K candidate extraction from the heatmap. The argmax alone is not enough:
# when the ball is near a face, the face can briefly win confidence, while the
# real ball sits as a secondary peak. Returning multiple candidates lets the
# validator pick the one best aligned with the predicted trajectory.
TN_TOPK = 3
TN_NMS_RADIUS = 20        # model-space pixels suppressed around each peak
TN_RUNNER_UP_CONF = 0.30  # runner-ups must clear this to be returned

# Motion-streak candidate extraction — fast ball detection
# A basketball moving at high speed creates an elongated streak in the diff map.
# Aspect ratio distinguishes a streak (ball) from a blob (shoe, logo, hand).
TN_STREAK_MIN_ASPECT  = 2.0   # min long/short side ratio to count as a streak
TN_STREAK_MIN_AREA    = 40    # min px area at model res (avoids noise specs)
TN_STREAK_MAX_AREA    = 600   # max px area (larger = player limb, not ball)
TN_STREAK_MIN_INTENS  = 0.12  # min mean diff intensity to suppress static noise
TN_STREAK_CONF        = 0.45  # pseudo-confidence assigned to streak candidates
                               # (lower than heatmap peak; validator scores by trajectory)

# Visibility classification head
VIS_LAMBDA = 0.1          # weight of vis BCE loss relative to heatmap focal loss
                          # Keep low so heatmap training dominates while vis_head bootstraps
VIS_THRESH_INFER = 0.25   # below this vis_prob at inference → return [] (no ball)

# V4 output-level motion alignment: penalise predictions on stationary background
# during training, closing the train/inference gap left by the motion gate in
# postprocess_heatmap. Targets the precision problem (FP on court logos, shoes).
# Weight kept low — focal loss must still dominate so heatmap regression is stable.
MOTION_PENALTY = 0.15     # weight of stationary-background penalty loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1,
                 stride=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=pad, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TrackNetV3(nn.Module):
    """Encoder-decoder CNN with U-Net skip connections.

    Input:  (B, 11, 360, 640) — 3 consecutive RGB frames + 2 motion diff maps (V4)
            Channels 0-8:  frame_{t-2}, frame_{t-1}, frame_t  (RGB each)
            Channel 9:     |frame_{t-1} - frame_{t-2}|  (grayscale motion)
            Channel 10:    |frame_t     - frame_{t-1}|  (grayscale motion)
    Output: (B, 1, 360, 640)  — heatmap sigmoid (or logits during training)

    Skip connections (U-Net style) are the key difference from the plain
    encoder-decoder: they pass fine spatial detail from the encoder to the
    decoder, which is critical for localising a ~5px ball in the heatmap.
    """

    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.conv1 = ConvBlock(11, 64)   # 9 RGB + 2 motion diff maps (V4)
        self.conv2 = ConvBlock(64, 64)          # ->skip1 (64ch, full res)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)        # ->skip2 (128ch, /2 res)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)        # ->skip3 (256ch, /4 res)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 256)
        self.conv9 = ConvBlock(256, 256)
        self.conv10 = ConvBlock(256, 256)       # bottleneck (256ch, /8 res)
        self.bottleneck_dropout = nn.Dropout2d(p=0.2)

        # --- Visibility classification head ---
        # Runs off the bottleneck (global view of frame) to predict whether
        # the ball is visible at all. Suppresses FP on fully-occluded frames.
        # AdaptiveAvgPool → flatten → 256→64→1 (logit, no sigmoid here)
        self.vis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # --- Decoder (with skip connections) ---
        # After ups1: 256ch. cat with skip3(256ch) ->512ch
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        # After ups2: 256ch. cat with skip2(128ch) ->384ch
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(384, 128)
        self.conv15 = ConvBlock(128, 128)
        # After ups3: 128ch. cat with skip1(64ch) ->192ch
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(192, 64)
        self.conv17 = ConvBlock(64, 64)
        # 1-channel output: direct heatmap regression
        self.conv18 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip1 = x                               # (B, 64, H, W)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        skip2 = x                               # (B, 128, H/2, W/2)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        skip3 = x                               # (B, 256, H/4, W/4)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        bottleneck = self.bottleneck_dropout(x)

        # Visibility head — branches off bottleneck before decoder
        vis_logit = self.vis_head(bottleneck)   # (B, 1)

        x = self.ups1(bottleneck)
        x = torch.cat([x, skip3], dim=1)        # 256+256=512
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = torch.cat([x, skip2], dim=1)        # 256+128=384
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = torch.cat([x, skip1], dim=1)        # 128+64=192
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x, vis_logit                     # (heatmap logits, vis logit)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Output layer produces raw logits — near-zero init keeps initial
        # predictions close to sigmoid(0)=0.5 instead of the extreme values
        # (0.01 or 0.99) that Kaiming init produces, avoiding massive initial
        # loss from the (pred^2 * log(1-pred)) term on background pixels.
        nn.init.normal_(self.conv18.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv18.bias, -1.0)  # sigmoid(-1.0)=0.27, calibrated no-ball prior
        # -0.5 (sigmoid=0.38) was too weak: with ~0.17% positive pixels the model
        # starts with a large focal loss on background, slowing the first few epochs.
        # -1.0 (sigmoid=0.27) is better calibrated without being as extreme as
        # the old -2.0 (which was set when data was ~50/50 and over-suppressed
        # valid ball detections in the early training phase).


# ---------------------------------------------------------------------------
# Heatmap post-processing — extract ball (x, y) from model output
# ---------------------------------------------------------------------------

def postprocess_heatmap(heatmap: np.ndarray,
                        scale_x: float = 1.0,
                        scale_y: float = 1.0,
                        local_radius: int = 10,
                        motion_map: np.ndarray | None = None,
                        motion_floor: float = 0.3,
                        motion_dilate: int = 15) -> list:
    """Convert sigmoid heatmap -> list of up to TN_TOPK ball candidates.

    Each candidate is (x, y, conf) in original frame coordinates. The global
    peak is always returned first; subsequent peaks are found by NMS-masking
    a TN_NMS_RADIUS window around each accepted peak and must clear
    TN_RUNNER_UP_CONF to be included. Returns [] if the top peak is below 0.1.

    Motion gate (V4-style): if `motion_map` is supplied, the heatmap is
    multiplied by (floor + (1 - floor) * normalized_motion) before peak
    extraction. This kills high-confidence peaks on stationary distractors
    (orange shoes, court logos) while preserving real detections that briefly
    pause. The diff map is dilated so a peak slightly off-center from the
    motion streak still gets credit for being near motion.

    Args:
        heatmap: (1, H, W) or (H, W) float32 in [0, 1] — sigmoid output
        scale_x: frame_w / INPUT_W
        scale_y: frame_h / INPUT_H
        local_radius: half-size of local window for centroid (model pixels)
        motion_map: optional (1, H, W) or (H, W) float32 in [0, 1] — |f_t - f_{t-1}|
                    averaged over RGB (same as the model input diff channel)
        motion_floor: minimum gate multiplier in pure-static regions (0..1).
                      0.3 means a stationary high-conf peak is scaled to 30%;
                      a moving peak is unchanged.
        motion_dilate: dilation kernel size applied to the motion map before
                       gating, so nearby peaks share credit with the motion streak.
    """
    if heatmap.ndim == 3:
        heatmap = heatmap[0]
    heatmap = heatmap.reshape(INPUT_H, INPUT_W).astype(np.float32)
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 1.0)

    if motion_map is not None:
        m = motion_map[0] if motion_map.ndim == 3 else motion_map
        m = m.reshape(INPUT_H, INPUT_W).astype(np.float32)
        # Dilate so a peak a few px away from the motion streak still counts.
        if motion_dilate > 0:
            k = np.ones((motion_dilate, motion_dilate), dtype=np.float32)
            m = cv2.dilate(m, k)
        # Normalize to [0, 1] via a robust percentile (frame-by-frame diffs
        # are tiny in magnitude; .max() alone is noisy).
        m_hi = float(np.percentile(m, 99.0))
        if m_hi > 1e-6:
            m = np.clip(m / m_hi, 0.0, 1.0)
        else:
            m = np.zeros_like(m)
        gate = motion_floor + (1.0 - motion_floor) * m
        heatmap = heatmap * gate

    candidates: list = []
    # First peak uses the detection floor (0.1); subsequent peaks must clear
    # the higher runner-up threshold.
    threshold = 0.1
    for k in range(TN_TOPK):
        peak_val = float(heatmap.max())
        if peak_val < threshold:
            break

        flat_idx = int(heatmap.argmax())
        peak_y, peak_x = divmod(flat_idx, INPUT_W)

        y0 = max(0, peak_y - local_radius)
        y1 = min(INPUT_H, peak_y + local_radius + 1)
        x0 = max(0, peak_x - local_radius)
        x1 = min(INPUT_W, peak_x + local_radius + 1)
        patch = heatmap[y0:y1, x0:x1]
        mask = patch >= 0.5 * peak_val
        if mask.any():
            ys, xs = np.where(mask)
            weights = patch[ys, xs]
            total = float(weights.sum())
            cx = float((xs * weights).sum() / total + x0) * scale_x
            cy = float((ys * weights).sum() / total + y0) * scale_y
        else:
            cx = float(peak_x) * scale_x
            cy = float(peak_y) * scale_y
        candidates.append((cx, cy, peak_val))

        # NMS: zero out a window around this peak so it can't win again
        ny0 = max(0, peak_y - TN_NMS_RADIUS)
        ny1 = min(INPUT_H, peak_y + TN_NMS_RADIUS + 1)
        nx0 = max(0, peak_x - TN_NMS_RADIUS)
        nx1 = min(INPUT_W, peak_x + TN_NMS_RADIUS + 1)
        heatmap[ny0:ny1, nx0:nx1] = 0.0

        threshold = TN_RUNNER_UP_CONF

    # --- Motion-streak candidates ---
    # Fast-moving balls create elongated blobs in the diff map. Use connected
    # component analysis on the diff map to find them and add as candidates
    # if they pass aspect-ratio / area / intensity filters. This fires only
    # when the ball moves too fast for the heatmap to produce a strong peak.
    if motion_map is not None:
        streak_candidates = _find_motion_streaks(motion_map, scale_x, scale_y)
        for (sx, sy, sconf) in streak_candidates:
            # Deduplicate: skip if within NMS radius of an existing candidate
            too_close = False
            for (cx, cy, _) in candidates:
                dist = math.hypot(sx / scale_x - cx / scale_x,
                                  sy / scale_y - cy / scale_y)
                if dist < TN_NMS_RADIUS:
                    too_close = True
                    break
            if not too_close and len(candidates) < TN_TOPK:
                candidates.append((sx, sy, sconf))

    return candidates


def _find_motion_streaks(motion_map: np.ndarray,
                         scale_x: float = 1.0,
                         scale_y: float = 1.0) -> list:
    """Find elongated motion-streak candidates in the diff map.

    Returns a list of (x, y, conf) tuples (at output coordinate scale).
    Empty list if no qualifying streaks found.
    """
    m = motion_map[0] if motion_map.ndim == 3 else motion_map
    m = m.reshape(INPUT_H, INPUT_W).astype(np.float32)

    # Normalise to [0, 1] using a robust 99th-percentile cap so a single bright
    # pixel doesn't collapse everything else to near-zero.
    m_hi = float(np.percentile(m, 99.0))
    if m_hi < 1e-6:
        return []
    m_norm = np.clip(m / m_hi, 0.0, 1.0)

    # Threshold to binary mask
    binary = (m_norm >= TN_STREAK_MIN_INTENS).astype(np.uint8)
    if binary.sum() == 0:
        return []

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    streak_candidates = []
    for label_id in range(1, num_labels):  # skip background (0)
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < TN_STREAK_MIN_AREA or area > TN_STREAK_MAX_AREA:
            continue

        # Check mean intensity within the component (filter out dim noise)
        mask = (labels == label_id)
        mean_intens = float(m_norm[mask].mean())
        if mean_intens < TN_STREAK_MIN_INTENS:
            continue

        # Aspect ratio via PCA on the component pixels
        ys, xs = np.where(mask)
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        if len(pts) < 5:
            continue
        mean_pt = pts.mean(axis=0)
        cov = np.cov((pts - mean_pt).T)
        if cov.ndim < 2:
            continue
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))[::-1]
        if eigvals[1] < 1e-6:
            continue
        aspect = float(eigvals[0] / eigvals[1]) ** 0.5  # sqrt so it's a length ratio
        if aspect < TN_STREAK_MIN_ASPECT:
            continue

        # Centroid of the component
        cx_model = float(centroids[label_id, 0])
        cy_model = float(centroids[label_id, 1])
        streak_candidates.append((cx_model * scale_x, cy_model * scale_y, TN_STREAK_CONF))

    return streak_candidates


# ---------------------------------------------------------------------------
# Gaussian heatmap generation (for training targets)
# ---------------------------------------------------------------------------

def generate_heatmap(cx: int, cy: int, w: int = INPUT_W, h: int = INPUT_H,
                     sigma: float = 3.16) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (cx, cy).

    Sigma=3.16 = sqrt(10) per TrackNetV1 paper (sigma^2=10).
    FWHM ≈ 2.35 * 3.16 ≈ 7.4px at model resolution 640x360.

    Uses a small patch (4σ radius) instead of a full meshgrid — ~100x fewer
    exponentials per call. Called 23K+ times per epoch so this matters.

    Returns (h, w) float32 array with values in [0, 1].
    """
    if cx < 0 or cy < 0:
        return np.zeros((h, w), dtype=np.float32)

    hmap = np.zeros((h, w), dtype=np.float32)
    radius = int(4 * sigma + 0.5)  # beyond 4σ the Gaussian is < 0.0003
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    hmap[y0:y1, x0:x1] = np.exp(-((xx - cx)**2 + (yy - cy)**2)
                                  / (2 * sigma**2))
    return hmap


# ---------------------------------------------------------------------------
# CenterNet focal loss — heatmap regression (Zhou et al. 2019)
# ---------------------------------------------------------------------------

def vis_bce_loss(vis_logit: torch.Tensor, vis_flag: torch.Tensor) -> torch.Tensor:
    """BCE loss for the visibility classification head.

    vis_logit: (B, 1) raw logit
    vis_flag:  (B,)   int tensor, 1=ball visible, 0=no ball
    """
    return torch.nn.functional.binary_cross_entropy_with_logits(
        vis_logit.squeeze(1), vis_flag.float())


def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet focal loss (Zhou et al. 2019, Objects as Points).

    Positive = only the peak pixel(s) where target == 1. The Gaussian decay
    around the peak is handled by the negative term's (1-Y)^4 weighting,
    which gracefully reduces the penalty near the center.

    Each term is averaged over its own pixel count so that the ~4 positive
    pixels aren't drowned out by ~1.8M negative pixels.
    """
    pred = torch.sigmoid(pred_logits)
    # Paper: positive = only the peak (Y == 1), not the full Gaussian blob.
    pos_mask = (target >= 0.9999).float()
    neg_mask = (target < 0.9999).float()

    pos_loss = -(torch.pow(1.0 - pred, 2.0)
                 * torch.log(pred.clamp(min=1e-6))) * pos_mask
    neg_loss = -(torch.pow(1.0 - target, 4.0)
                 * torch.pow(pred, 2.0)
                 * torch.log((1.0 - pred).clamp(min=1e-6))) * neg_mask

    num_pos = pos_mask.sum().clamp(min=1)
    num_neg = neg_mask.sum().clamp(min=1)
    return pos_loss.sum() / num_pos + neg_loss.sum() / num_neg


# ---------------------------------------------------------------------------
# PE (Positioning Error) evaluation — TrackNet paper metric
# ---------------------------------------------------------------------------

def evaluate_pe(model: "TrackNetV3", loader, device: str,
                pe_thresh: float = 10.0,
                conf_thresh: float = 0.5) -> dict:
    """Compute val_loss + precision/recall/F1 in a single val pass.

    TP: model predicts ball AND predicted center ≤ pe_thresh px from GT.
    FP: model predicts ball but no GT, or distance > pe_thresh.
    FN: model predicts no ball but GT exists.

    Also tracks per-game TP/FP/FN for diagnostic breakdowns.
    """
    tp = fp = fn = 0
    total_loss = 0.0
    margins = []                      # peak - highest-distractor on visible frames
    vis_correct = vis_total = 0       # visibility head accuracy
    game_stats: dict[str, dict] = {}  # {game: {tp, fp, fn}}
    model.eval()
    use_amp = device != "cpu"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for inp, target, vis_flags, games in loader:
            inp    = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            vis_flags_dev = vis_flags.to(device, non_blocking=True)
            out, vis_logit = model(inp)  # (B, 1, H, W) logits, (B, 1) vis logit
            total_loss += (focal_loss(out, target)
                           + VIS_LAMBDA * vis_bce_loss(vis_logit, vis_flags_dev)).item()
            # Vis head accuracy
            vis_pred = (torch.sigmoid(vis_logit.squeeze(1)) >= 0.5).cpu()
            vis_correct += (vis_pred == vis_flags.bool()).sum().item()
            vis_total   += vis_flags.shape[0]
            pred_np   = torch.sigmoid(out).cpu().numpy()
            target_np = target.cpu().numpy()
            vis_np    = vis_flags.cpu().numpy()
            for b in range(pred_np.shape[0]):
                game = games[b]
                if game not in game_stats:
                    game_stats[game] = {"tp": 0, "fp": 0, "fn": 0}
                has_ball = int(vis_np[b]) > 0
                gt_hmap  = target_np[b, 0]
                if has_ball:
                    gt_y, gt_x = divmod(int(gt_hmap.argmax()), INPUT_W)
                else:
                    gt_x = gt_y = -1
                pred_hmap   = pred_np[b, 0]
                pred_conf   = float(pred_hmap.max())
                pred_detect = pred_conf >= conf_thresh
                pr_y, pr_x  = divmod(int(pred_hmap.argmax()), INPUT_W)

                # Peak-to-distractor margin
                if has_ball:
                    y0 = max(0, pr_y - 10)
                    y1 = min(INPUT_H, pr_y + 11)
                    x0 = max(0, pr_x - 10)
                    x1 = min(INPUT_W, pr_x + 11)
                    masked = pred_hmap.copy()
                    masked[y0:y1, x0:x1] = 0.0
                    distractor_conf = float(masked.max())
                    margins.append(pred_conf - distractor_conf)

                if pred_detect and has_ball:
                    dist = math.hypot(pr_x - gt_x, pr_y - gt_y)
                    if dist <= pe_thresh:
                        tp += 1
                        game_stats[game]["tp"] += 1
                    else:
                        fp += 1
                        fn += 1
                        game_stats[game]["fp"] += 1
                        game_stats[game]["fn"] += 1
                elif pred_detect:
                    fp += 1
                    game_stats[game]["fp"] += 1
                elif has_ball:
                    fn += 1
                    game_stats[game]["fn"] += 1
    avg_loss   = total_loss / max(len(loader), 1)
    avg_margin = sum(margins) / max(len(margins), 1) if margins else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Per-game F1
    per_game = {}
    for g, s in sorted(game_stats.items()):
        g_prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        g_rec  = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
        per_game[g] = {"precision": g_prec, "recall": g_rec, "f1": g_f1,
                       "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]}

    vis_acc = vis_correct / max(vis_total, 1)
    return {"val_loss": avg_loss, "precision": prec, "recall": rec,
            "f1": f1, "margin": avg_margin,
            "tp": tp, "fp": fp, "fn": fn,
            "vis_acc": vis_acc,
            "per_game": per_game}


def _wandb_prediction_samples(model, val_loader, device, n_samples=8):
    """Generate W&B image table with GT vs predicted heatmap overlays."""
    model.eval()
    images = []
    inp, target, _vis, _game = next(iter(val_loader))
    inp = inp.to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device != "cpu")):
        hmap_logits, _ = model(inp)
        pred = torch.sigmoid(hmap_logits).cpu().numpy()
    target_np = target.numpy()
    for b in range(min(n_samples, inp.shape[0])):
        frame = (inp[b, 6:9].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gt_hmap = (target_np[b, 0] * 255).astype(np.uint8)
        pr_hmap = (pred[b, 0] * 255).astype(np.uint8)
        gt_color = cv2.applyColorMap(gt_hmap, cv2.COLORMAP_JET)
        pr_color = cv2.applyColorMap(pr_hmap, cv2.COLORMAP_JET)
        gt_overlay = cv2.addWeighted(frame, 0.6, gt_color, 0.4, 0)
        pr_overlay = cv2.addWeighted(frame, 0.6, pr_color, 0.4, 0)
        combined = np.hstack([gt_overlay, pr_overlay])
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        images.append(wandb.Image(combined, caption=f"sample_{b} GT|Pred"))
    return images


# ---------------------------------------------------------------------------
# Dataset — sequences of 3 consecutive frames with ball center labels
# ---------------------------------------------------------------------------

def _parse_frame_number(path: str) -> int:
    """Extract frame number from path like .../frame_000123.jpg ->123."""
    stem = Path(path).stem
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return -1


class TrackNetDataset(Dataset):
    """Load (frame_t-2, frame_t-1, frame_t) triples with heatmap targets.

    Expects a CSV file with columns: frame_path, visibility, x, y
    Rows must be in sequential order within each video clip.

    Uses lazy loading (reads frames on demand) to support large multi-game
    datasets (40K+ frames) that won't fit in RAM.
    """

    def __init__(self, csv_path: str, sigma: float = 3.16,
                 oversample_visible: bool = False,
                 augment: bool = False,
                 data_root: str | None = None):
        self.sigma = sigma
        self.augment = augment
        self.entries = []       # list of (path, vis, x, y)
        self.sequences = []     # list of (idx_t-2, idx_t-1, idx_t) triples

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fp = row["frame_path"]
                # Remap relative 'data\...' paths to an external data_root
                if data_root is not None:
                    fp = fp.replace("\\", "/")
                    if fp.startswith("data/"):
                        fp = str(Path(data_root) / fp[len("data/"):])
                self.entries.append((
                    fp,
                    int(row["visibility"]),
                    float(row["x"]) if row["visibility"] != "0" else -1.0,
                    float(row["y"]) if row["visibility"] != "0" else -1.0,
                ))

        # Build valid 3-frame sequences, respecting game boundaries.
        # A boundary is detected when: (a) frame numbers are non-consecutive,
        # or (b) the parent directory (game name) changes.
        visible_seqs = []
        empty_seqs = []
        for i in range(2, len(self.entries)):
            i0, i1, i2 = i - 2, i - 1, i

            # Check for game boundaries — don't mix frames from different games
            p0 = Path(self.entries[i0][0])
            p1 = Path(self.entries[i1][0])
            p2 = Path(self.entries[i2][0])

            if p0.parent != p1.parent or p1.parent != p2.parent:
                continue  # different games

            n0 = _parse_frame_number(str(p0))
            n1 = _parse_frame_number(str(p1))
            n2 = _parse_frame_number(str(p2))
            if n0 < 0 or n1 != n0 + 1 or n2 != n1 + 1:
                continue  # non-consecutive frames

            seq = (i0, i1, i2)
            if self.entries[i2][1] > 0:
                visible_seqs.append(seq)
            else:
                empty_seqs.append(seq)

        if oversample_visible and visible_seqs:
            # Balance to ~50/50 visible vs empty for equal gradient contribution
            repeat = max(1, round(len(empty_seqs) / max(len(visible_seqs), 1)))
            self.sequences = empty_seqs + visible_seqs * repeat
            print(f"  TrackNet dataset: {len(visible_seqs)} visible, "
                  f"{len(empty_seqs)} empty -> oversampled {repeat}x "
                  f"-> {len(self.sequences)} total sequences")
        else:
            self.sequences = visible_seqs + empty_seqs
            if visible_seqs:
                print(f"  TrackNet dataset: {len(visible_seqs)} visible, "
                      f"{len(empty_seqs)} empty ->{len(self.sequences)} total")

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, path: str) -> tuple:
        """Load a frame ->((3, H, W) float32 [0,1], orig_w, orig_h).

        Checks for a pre-resized cache first (saved by --preprocess).
        Cache is already 640×360 so no resize needed — ~3-4x faster loading.
        A sidecar .txt file stores original dims for correct label scaling.
        """
        p = Path(path)
        # .npy cache: uint8 (H,W,3) array + sidecar .txt for orig dims
        # numpy load is ~5x faster than JPEG decode — critical on Windows
        cache_npy  = p.parent / f"{p.stem}_640x360.npy"
        cache_dims = p.parent / f"{p.stem}_640x360.txt"

        if cache_npy.exists() and cache_dims.exists():
            img = np.load(str(cache_npy))          # uint8 (H,W,3), ~0.2ms
            orig_w, orig_h = map(int, cache_dims.read_text().split())
            return (np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0,
                    orig_w, orig_h)

        img = cv2.imread(path)
        if img is None:
            return (np.zeros((3, INPUT_H, INPUT_W), dtype=np.float32), INPUT_W, INPUT_H)
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        return (np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0,
                orig_w, orig_h)

    def __getitem__(self, idx):
        i0, i1, i2 = self.sequences[idx]
        path0, _, _, _ = self.entries[i0]
        path1, _, _, _ = self.entries[i1]
        path2, vis2, x2, y2 = self.entries[i2]

        f0, _, _       = self._load_frame(path0)
        f1, _, _       = self._load_frame(path1)
        f2, orig_w, orig_h = self._load_frame(path2)

        # Scale label coords from original frame resolution ->model resolution
        if vis2 > 0 and x2 >= 0 and y2 >= 0:
            cx = x2 * INPUT_W / orig_w
            cy = y2 * INPUT_H / orig_h
        else:
            cx, cy = -1.0, -1.0

        # --- Augmentation ---
        if self.augment:
            # Horizontal flip (50%)
            if pyrandom.random() < 0.5:
                f0 = np.flip(f0, axis=2).copy()
                f1 = np.flip(f1, axis=2).copy()
                f2 = np.flip(f2, axis=2).copy()
                if cx >= 0:
                    cx = INPUT_W - 1 - cx

            # Color jitter (HSV) + contrast — consistent across all 3 frames
            # Hue shift +/-15 deg, saturation scale 0.7-1.4, value scale 0.8-1.2
            hue_shift = pyrandom.uniform(-15.0, 15.0)
            sat_scale = pyrandom.uniform(0.5, 1.6)   # wider: 0.7-1.4 → 0.5-1.6 for cross-game generalization
            val_scale = pyrandom.uniform(0.8, 1.2)
            contrast = pyrandom.uniform(0.75, 1.3)   # wider: 0.9-1.1 → 0.75-1.3 (SimCLR/RandAugment range)

            def _apply_hsv_contrast(f):
                # f: (3, H, W) float32 in BGR order, values in [0,1]
                f_hwc = np.transpose(f, (1, 2, 0))
                img_uint8 = np.clip(f_hwc * 255.0, 0, 255).astype(np.uint8)
                hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
                # OpenCV hue range: 0..179
                h_shift_cv2 = int(hue_shift * 179.0 / 360.0)
                hsv[..., 0] = (hsv[..., 0].astype(np.int32) + h_shift_cv2) % 180
                hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
                hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
                bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
                bgr = np.clip((bgr - 0.5) * contrast + 0.5, 0.0, 1.0)
                return np.transpose(bgr.astype(np.float32), (2, 0, 1))

            f0 = _apply_hsv_contrast(f0)
            f1 = _apply_hsv_contrast(f1)
            f2 = _apply_hsv_contrast(f2)

            # Brightness jitter [0.7, 1.3]
            b = pyrandom.uniform(0.7, 1.3)
            f0 = np.clip(f0 * b, 0.0, 1.0)
            f1 = np.clip(f1 * b, 0.0, 1.0)
            f2 = np.clip(f2 * b, 0.0, 1.0)

            # Independent Gaussian noise per frame (std=0.02)
            noise = np.random.normal(0, 0.02, (3, *f0.shape)).astype(np.float32)
            f0 = np.clip(f0 + noise[0], 0.0, 1.0)
            f1 = np.clip(f1 + noise[1], 0.0, 1.0)
            f2 = np.clip(f2 + noise[2], 0.0, 1.0)

            # Motion blur (30% chance) — apply same kernel to all frames
            if pyrandom.random() < 0.3:
                ksize = pyrandom.choice([5, 7, 9, 11])
                kernel = np.zeros((ksize, ksize), dtype=np.float32)
                if pyrandom.random() < 0.5:
                    kernel[ksize // 2, :] = 1.0 / ksize   # horizontal
                else:
                    kernel[:, ksize // 2] = 1.0 / ksize   # vertical
                # Apply to each frame (HWC float32)
                f0_hwc = np.transpose(f0, (1, 2, 0))
                f1_hwc = np.transpose(f1, (1, 2, 0))
                f2_hwc = np.transpose(f2, (1, 2, 0))
                f0_hwc = cv2.filter2D(f0_hwc, -1, kernel)
                f1_hwc = cv2.filter2D(f1_hwc, -1, kernel)
                f2_hwc = cv2.filter2D(f2_hwc, -1, kernel)
                f0 = np.transpose(f0_hwc, (2, 0, 1)).astype(np.float32)
                f1 = np.transpose(f1_hwc, (2, 0, 1)).astype(np.float32)
                f2 = np.transpose(f2_hwc, (2, 0, 1)).astype(np.float32)

        # Motion diff maps (V4): |frame_{t-1} - frame_{t-2}| and |frame_t - frame_{t-1}|
        # Computed AFTER augmentation so they reflect the augmented frames.
        # Averaged over RGB channels ->(1, H, W) motion magnitude in [0, 1].
        diff01 = np.abs(f1 - f0).mean(axis=0, keepdims=True)  # (1, H, W)
        diff12 = np.abs(f2 - f1).mean(axis=0, keepdims=True)  # (1, H, W)
        inp = np.concatenate([f0, f1, f2, diff01, diff12], axis=0)   # (11, H, W)

        if cx >= 0 and cy >= 0 and vis2 > 0:
            hmap = generate_heatmap(round(cx), round(cy), INPUT_W, INPUT_H,
                                    self.sigma)
        else:
            hmap = np.zeros((INPUT_H, INPUT_W), dtype=np.float32)
        hmap = hmap[np.newaxis, :, :]   # (1, H, W)

        game = Path(path2).parent.name

        return (torch.from_numpy(inp),
                torch.from_numpy(hmap),
                torch.tensor(1 if (vis2 > 0 and cx >= 0) else 0),
                game)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tracknet(data_dir: str, epochs: int = 100, batch_size: int = 8,
                   device: str = "cuda:0",
                   output_dir: str = "runs/tracknet/weights",
                   num_workers: int = 4,
                   resume: bool = False,
                   accum_steps: int = 2,
                   data_root: str | None = None,
                   seed: int = 42,
                   train_csv: str | None = None,
                   val_csv: str | None = None):
    """Train TrackNetV3 on basketball frame sequences.

    Args:
        accum_steps: gradient accumulation steps. Effective batch size is
                     batch_size * accum_steps (default 8*2=16).
        data_root: root directory for frame data. When set, CSV paths like
                   'data\\...' are remapped to data_root/... for fast local I/O.
        seed: random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    pyrandom.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed: {seed}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_csv = train_csv or str(Path(data_dir) / "train.csv")
    val_csv   = val_csv   or str(Path(data_dir) / "val.csv")

    print("Building training dataset ...")
    train_ds = TrackNetDataset(train_csv, oversample_visible=False, augment=True,
                               data_root=data_root)
    val_ds   = (TrackNetDataset(val_csv, augment=False, data_root=data_root)
                if Path(val_csv).exists() else None)

    _pw = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    )
    val_loader = (DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    ) if val_ds else None)

    # cuDNN benchmark: caches fastest conv algorithm for fixed input size.
    # Free 10-20% speedup since every input is always 640×360.
    torch.backends.cudnn.benchmark = True

    model = TrackNetV3().to(device)
    import platform
    if platform.system() != "Windows":
        try:
            model = torch.compile(model)
            print("  torch.compile: enabled")
        except Exception as e:
            print(f"  torch.compile: skipped ({e})")
    else:
        print("  torch.compile: skipped (Windows/no Triton)")

    # AdamW: decoupled weight decay generalises better than L2 in Adam
    # (Loshchilov & Hutter 2019), especially with cosine schedulers.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=1e-3)
    # lr: 1e-4 empirically best (F1=0.524). 2e-4 overshoots on this heatmap task.
    # wd: raised from 1e-4 → 1e-3 (Loshchilov & Hutter 2019 recommend 0.01–0.05;
    # 1e-4 is below the lower bound). verified_train.csv is ~7.5k sequences —
    # a small clean dataset where overfitting risk is higher than label-noise risk,
    # so stronger regularization is now correct.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    # CosineAnnealingWarmRestarts: LR restarts at epoch 20, then 60, then 140...
    # Each cycle is T_mult=2x longer than the last.
    # T_0 raised 10→20: with ~7.5k sequences at batch 8 (~940 steps/epoch), T_0=10
    # gave only 9,400 gradient steps before the first restart — cosine had decayed
    # to ~1e-5 by epoch 5, leaving the model at near-zero LR for the second half
    # before it had converged. SGDR (Loshchilov & Hutter 2017) recommends T_0
    # large enough to reach a reasonable local minimum in the first cycle.
    # T_0=20 gives ~18,800 steps before restart, matching that intent.

    use_amp = device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1        = 0.0
    no_improve     = 0       # epochs since best F1 improved
    early_stop_pat = 15
    # 15 (was 12): warm restarts cause a brief F1 dip when LR spikes at the
    # start of each new cycle. Patience must exceed the recovery time (~3-5
    # epochs) to avoid stopping in the middle of a promising restart.
    start_epoch    = 0

    # Load best-ever F1 for overall tracking
    overall_best_path = out_path / "best_overall.pt"
    overall_best_f1 = 0.0
    if overall_best_path.exists():
        try:
            meta = torch.load(str(out_path / "best_overall_meta.pt"),
                              map_location="cpu", weights_only=True)
            overall_best_f1 = meta.get("f1", 0.0)
            print(f"  Best overall F1 so far: {overall_best_f1:.3f}")
        except Exception:
            pass

    # Resume from last checkpoint if requested
    ckpt_path = out_path / "last_ckpt.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  Resume: {len(missing)} new keys init from scratch "
                  f"(e.g. vis_head — expected after architecture update)")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        except ValueError:
            print("  Optimizer/scheduler state incompatible (architecture changed) "
                  "— resetting optimizer, keeping model weights")
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1     = ckpt.get("best_f1", 0.0)
        no_improve  = ckpt.get("no_improve", 0)
        print(f"  Resumed from epoch {start_epoch} "
              f"(best_f1={best_f1:.3f}, no_improve={no_improve})")
    elif resume:
        print("  No checkpoint found — starting from scratch")

    eff_batch = batch_size * accum_steps
    print(f"  batch={batch_size} x accum={accum_steps} ->effective batch {eff_batch}")
    print(f"  AMP fp16: {'enabled' if use_amp else 'disabled'}")

    # --- W&B init ---
    wandb_run_id = None
    if _HAS_WANDB:
        if resume and ckpt_path.exists():
            wandb_run_id = ckpt.get("wandb_run_id")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "uvambb-cv"),
            entity=os.environ.get("WANDB_ENTITY"),
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            name=f"tracknet-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config={
                "model": "TrackNetV3",
                "epochs": epochs,
                "batch_size": batch_size,
                "accum_steps": accum_steps,
                "effective_batch": eff_batch,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "scheduler": "CosineAnnealingWarmRestarts",
                "optimizer": "AdamW",
                "amp": use_amp,
                "early_stop_patience": early_stop_pat,
                "motion_penalty": MOTION_PENALTY,
                "vis_lambda": VIS_LAMBDA,
                "data_dir": data_dir,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds) if val_ds else 0,
            },
        )
        wandb_run_id = wandb.run.id

    for epoch in range(start_epoch, epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        motion_pen_total = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inp, target, vis_flags, _game) in enumerate(train_loader):
            inp       = inp.to(device, non_blocking=True)
            target    = target.to(device, non_blocking=True)
            vis_flags = vis_flags.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                hmap_out, vis_out = model(inp)

                # V4 output-level motion alignment loss:
                # Penalise the model for firing on stationary background regions.
                # diff12 (input channel 10) is near-zero on static distractors
                # (court logos, shoes) — we want predictions there to also be
                # near-zero. This closes the training/inference gap: inference
                # already suppresses these via the motion_floor gate in
                # postprocess_heatmap; now training does the same explicitly.
                diff12 = inp[:, 10:11, :, :]                              # (B,1,H,W)
                m_max  = diff12.amax(dim=[2, 3], keepdim=True).clamp(min=1e-5)
                m_norm = (diff12 / m_max).clamp(0.0, 1.0)                 # per-sample norm
                static_bg = (1.0 - m_norm) * (target < 0.1).float()      # stationary + no ball
                motion_pen = (torch.sigmoid(hmap_out) * static_bg).mean()

                loss = (focal_loss(hmap_out, target)
                        + VIS_LAMBDA    * vis_bce_loss(vis_out, vis_flags)
                        + MOTION_PENALTY * motion_pen) / accum_steps

            scaler.scale(loss).backward()

            # Step optimizer every accum_steps batches (or at end of epoch)
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss     += loss.item() * accum_steps  # undo the /accum_steps for logging
            motion_pen_total += motion_pen.item()

            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  "
                      f"batch {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item() * accum_steps:.4f}")

        avg_train      = train_loss      / max(len(train_loader), 1)
        avg_motion_pen = motion_pen_total / max(len(train_loader), 1)

        # --- Validate + PE metric in a single pass ---
        if val_loader:
            pe = evaluate_pe(model, val_loader, device,
                             pe_thresh=10.0, conf_thresh=0.5)
            print(f"Epoch {epoch+1}/{epochs}  "
                  f"train_loss={avg_train:.4f}  motion_pen={avg_motion_pen:.4f}  "
                  f"val_loss(diag)={pe['val_loss']:.4f}  "
                  f"prec={pe['precision']:.3f}  rec={pe['recall']:.3f}  "
                  f"F1={pe['f1']:.3f}  margin={pe['margin']:.3f}  "
                  f"vis_acc={pe['vis_acc']:.3f}  "
                  f"(TP={pe['tp']} FP={pe['fp']} FN={pe['fn']})")
            for g, gs in pe.get("per_game", {}).items():
                print(f"  {g}: prec={gs['precision']:.3f} rec={gs['recall']:.3f} "
                      f"F1={gs['f1']:.3f} (TP={gs['tp']} FP={gs['fp']} FN={gs['fn']})")
            if _HAS_WANDB:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": avg_train,
                    "train/motion_penalty": avg_motion_pen,
                    "val/loss_diagnostic": pe["val_loss"],
                    "val/precision": pe["precision"],
                    "val/recall": pe["recall"],
                    "val/f1": pe["f1"],
                    "val/margin": pe["margin"],
                    "val/tp": pe["tp"],
                    "val/fp": pe["fp"],
                    "val/fn": pe["fn"],
                    "val/vis_acc": pe["vis_acc"],
                    "lr": scheduler.get_last_lr()[0],
                }
                for g, gs in pe.get("per_game", {}).items():
                    log_dict[f"val/{g}/f1"] = gs["f1"]
                    log_dict[f"val/{g}/precision"] = gs["precision"]
                    log_dict[f"val/{g}/recall"] = gs["recall"]
                if (epoch + 1) % 5 == 0:
                    try:
                        log_dict["predictions"] = _wandb_prediction_samples(
                            model, val_loader, device)
                    except Exception as e:
                        print(f"  W&B prediction viz skipped: {e}")
                wandb.log(log_dict)

            if pe["f1"] > best_f1:
                best_f1    = pe["f1"]
                no_improve = 0
                torch.save(model.state_dict(), str(out_path / "best_session.pt"))
                print(f"  -> Saved best session model (F1={best_f1:.3f})")
                # Update overall best if this session beat it
                if best_f1 > overall_best_f1:
                    overall_best_f1 = best_f1
                    torch.save(model.state_dict(), str(out_path / "best_overall.pt"))
                    torch.save({"f1": best_f1, "epoch": epoch + 1},
                               str(out_path / "best_overall_meta.pt"))
                    print(f"  -> New best overall model! (F1={overall_best_f1:.3f})")
                if _HAS_WANDB:
                    art = wandb.Artifact(
                        "tracknet-best", type="model",
                        description=f"TrackNet F1={best_f1:.3f} epoch={epoch+1}",
                        metadata={"f1": best_f1, "epoch": epoch + 1},
                    )
                    art.add_file(str(out_path / "best_session.pt"))
                    wandb.log_artifact(art)
            else:
                no_improve += 1
                print(f"  No F1 improvement ({no_improve}/{early_stop_pat})")
        else:
            print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_train:.4f}")
            if _HAS_WANDB:
                wandb.log({"epoch": epoch + 1, "train/loss": avg_train,
                           "lr": scheduler.get_last_lr()[0]})

        scheduler.step()

        # Save resumable checkpoint
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "scaler":     scaler.state_dict(),
            "best_f1":    best_f1,
            "no_improve": no_improve,
            "wandb_run_id": wandb_run_id,
        }, str(out_path / "last_ckpt.pt"))

        gc.collect()
        torch.cuda.empty_cache()

        if val_loader and no_improve >= early_stop_pat:
            print(f"\nEarly stopping: F1 hasn't improved for "
                  f"{early_stop_pat} epochs (best F1={best_f1:.3f})")
            break

    print(f"\nTraining complete.")
    print(f"  Best this session: {out_path / 'best_session.pt'} (F1={best_f1:.3f})")
    print(f"  Best overall:      {out_path / 'best_overall.pt'} (F1={overall_best_f1:.3f})")
    if _HAS_WANDB:
        wandb.finish()
    return str(out_path / "best_overall.pt")


# ---------------------------------------------------------------------------
# Inference wrapper — used by main.py pipeline
# ---------------------------------------------------------------------------

class TrackNetInference:
    """Stateful inference wrapper: feed frames one-by-one, get ball coords.

    Maintains a 3-frame sliding window internally.  Returns None until
    at least 3 frames have been pushed.

    Usage:
        tracker = TrackNetInference("tracknet_basketball.pt", "cuda:0")
        for frame in video_frames:
            result = tracker.predict(frame)
            if result is not None:
                det_dict = result   # {"tid": -1, "box": [...], "cls": 0, "conf": ...}
    """

    CLS_BALL = 0
    DEFAULT_RADIUS = 15     # pixels in original frame, for bbox synthesis

    def __init__(self, weights_path: str, device: str = "cuda:0",
                 conf_thresh: float = 0.5, ball_radius: int = 15,
                 vis_thresh: float = VIS_THRESH_INFER):
        self.device = device
        self.conf_thresh = conf_thresh
        self.ball_radius = ball_radius
        self.vis_thresh = vis_thresh
        self._frame_buf: deque = deque(maxlen=3)
        self._frame_w: int = 0
        self._frame_h: int = 0

        self.model = TrackNetV3()
        if Path(weights_path).exists():
            state = torch.load(weights_path, map_location=device,
                               weights_only=True)
            # strict=False: tolerate architecture differences between old/new
            # checkpoints (e.g. before/after adding skip connections)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"TrackNet: {len(missing)} missing keys "
                      f"(new architecture — retrain recommended)")
            if unexpected:
                print(f"TrackNet: {len(unexpected)} unexpected keys ignored")
            print(f"TrackNet: loaded weights from {weights_path}")
        else:
            print(f"WARNING: TrackNet weights not found at {weights_path} "
                  f"— using random init (train first!)")

        self.model.to(device)
        self.model.eval()

    def predict(self, frame: np.ndarray) -> list:
        """Feed a frame and return up to TN_TOPK ball detection candidates.

        Always call this for EVERY frame (including skipped ones) so the
        3-frame sliding window stays in sync with the video.

        Returns a list of detection dicts (possibly empty). The first element
        is the argmax; subsequent ones are NMS-suppressed runner-ups that
        cleared TN_RUNNER_UP_CONF. Downstream (BallValidator._score) picks the
        winner by confidence × trajectory alignment.
        """
        h, w = frame.shape[:2]
        self._frame_w = w
        self._frame_h = h

        resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        self._frame_buf.append(resized)

        if len(self._frame_buf) < 3:
            return []

        frames = [f.astype(np.float32) / 255.0 for f in self._frame_buf]
        f0 = np.transpose(frames[0], (2, 0, 1))
        f1 = np.transpose(frames[1], (2, 0, 1))
        f2 = np.transpose(frames[2], (2, 0, 1))
        diff01 = np.abs(f1 - f0).mean(axis=0, keepdims=True)
        diff12 = np.abs(f2 - f1).mean(axis=0, keepdims=True)
        inp = np.concatenate([f0, f1, f2, diff01, diff12], axis=0)
        inp = np.expand_dims(inp, axis=0)
        tensor = torch.from_numpy(inp).to(self.device)

        # Single forward pass — TTA (flip average) costs ~15ms/frame for
        # negligible F1 gain given TrackNet's 3-frame temporal context already
        # handles most flip ambiguity.
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=self.device != "cpu"):
            hmap_logits, vis_logit = self.model(tensor)
            out = torch.sigmoid(hmap_logits)

        # Visibility gate: if the head is confident there's no ball, skip
        # candidate extraction entirely — suppresses FP on occluded frames.
        vis_prob = float(torch.sigmoid(vis_logit).squeeze())
        if vis_prob < self.vis_thresh:
            return []

        heatmap = out[0].cpu().numpy()

        scale_x = w / INPUT_W
        scale_y = h / INPUT_H

        candidates = postprocess_heatmap(heatmap, scale_x, scale_y,
                                         motion_map=diff12)
        r = self.ball_radius
        dets: list = []
        for cx, cy, conf in candidates:
            if conf < self.conf_thresh:
                continue
            box = [
                max(0, int(cx - r)),
                max(0, int(cy - r)),
                min(w, int(cx + r)),
                min(h, int(cy + r)),
            ]
            dets.append({
                "tid": -1,
                "box": box,
                "cls": self.CLS_BALL,
                "conf": conf,
            })
        return dets

    def reset(self):
        """Clear the frame buffer (e.g. when switching videos)."""
        self._frame_buf.clear()


# ---------------------------------------------------------------------------
# CLI — standalone training
# ---------------------------------------------------------------------------

def preprocess_frames(data_dir: str) -> None:
    """Pre-resize all frames to 640×360 and save as .npy binary arrays.

    .npy loads ~5x faster than JPEG decode — eliminates the data loading
    bottleneck on Windows where multiprocessing spawn overhead is high.
    Disk cost: ~691KB per frame (vs ~20KB JPEG) — ~19GB for 28K frames.
    One-time cost; run once before training.
    """
    paths: set = set()
    for split in ("train.csv", "val.csv"):
        p = Path(data_dir) / split
        if not p.exists():
            continue
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                paths.add(row["frame_path"])

    print(f"Pre-processing {len(paths)} frames ->.npy cache ...")
    print(f"  Estimated disk: {len(paths) * 691 / 1024:.0f} MB")
    done = skipped = 0
    for i, path in enumerate(sorted(paths)):
        src       = Path(path)
        dst_npy   = src.parent / f"{src.stem}_640x360.npy"
        dst_dims  = src.parent / f"{src.stem}_640x360.txt"
        if dst_npy.exists() and dst_dims.exists():
            skipped += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        np.save(str(dst_npy), img)               # saves uint8 (H,W,3)
        dst_dims.write_text(f"{orig_w} {orig_h}")
        done += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(paths)} ...")
    print(f"Done: {done} saved as .npy, {skipped} already cached.")


def sweep_conf_thresh(
    weights: str,
    data_dir: str,
    val_csv: str | None = None,
    device: str = "cuda:0",
    num_workers: int = 4,
    batch_size: int = 8,
    data_root: str | None = None,
    conf_min: float = 0.10,
    conf_max: float = 0.90,
    conf_step: float = 0.05,
) -> None:
    """Sweep conf_thresh on the val set and report F1 for each value.

    Loads the model once, then re-evaluates with each threshold — fast
    because evaluate_pe only does inference + metric accumulation.
    """
    val_csv = val_csv or str(Path(data_dir) / "val.csv")
    if not Path(val_csv).exists():
        print(f"ERROR: val CSV not found: {val_csv}")
        return
    if not Path(weights).exists():
        print(f"ERROR: weights not found: {weights}")
        return

    print(f"Loading model from {weights} ...")
    model = TrackNetV3().to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  {len(missing)} missing keys (architecture mismatch)")
    model.eval()
    torch.backends.cudnn.benchmark = True

    print(f"Loading val dataset: {val_csv}")
    val_ds = TrackNetDataset(val_csv, augment=False, data_root=data_root)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )

    thresholds = []
    t = conf_min
    while t <= conf_max + 1e-9:
        thresholds.append(round(t, 4))
        t += conf_step

    print(f"\nSweeping conf_thresh {conf_min:.2f} -> {conf_max:.2f} "
          f"(step {conf_step:.2f})  |  {len(thresholds)} values\n")
    header = f"{'conf':>6}  {'prec':>6}  {'rec':>6}  {'F1':>6}  {'TP':>6}  {'FP':>6}  {'FN':>6}"
    print(header)
    print("-" * len(header))

    best_f1, best_conf = 0.0, 0.0
    results = []
    for thresh in thresholds:
        pe = evaluate_pe(model, val_loader, device,
                         pe_thresh=10.0, conf_thresh=thresh)
        f1   = pe["f1"]
        prec = pe["precision"]
        rec  = pe["recall"]
        tp, fp, fn = pe["tp"], pe["fp"], pe["fn"]
        marker = " <--" if f1 > best_f1 else ""
        print(f"{thresh:>6.2f}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}"
              f"  {tp:>6}  {fp:>6}  {fn:>6}{marker}")
        results.append((thresh, prec, rec, f1, tp, fp, fn))
        if f1 > best_f1:
            best_f1   = f1
            best_conf = thresh

    print("-" * len(header))
    print(f"\nBest conf_thresh = {best_conf:.2f}  (F1 = {best_f1:.3f})")
    print(f"\nTo apply: set TRACKNET_CONF_THRESH = {best_conf} in main.py")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="TrackNet ball tracker")
    parser.add_argument("--train", action="store_true",
                        help="Train TrackNet on basketball data")
    parser.add_argument("--preprocess", action="store_true",
                        help="Pre-resize all frames to 640x360 for faster training")
    parser.add_argument("--data", default="data/tracknet_merged/",
                        help="Directory containing train.csv / val.csv")
    parser.add_argument("--data-root",
                        default=os.environ.get("UVAMBB_DATA_ROOT"),
                        help="Root directory for frame data (or set UVAMBB_DATA_ROOT in .env)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size per step (default 8, effective 16 with accum=2)")
    parser.add_argument("--accum", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch * accum)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader num_workers (default 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last_ckpt.pt checkpoint")
    parser.add_argument("--seed", type=int, default=6050,
                        help="Random seed for reproducibility (default 6050)")
    parser.add_argument("--train-csv", default=None,
                        help="Override train CSV path (default: <data>/train.csv)")
    parser.add_argument("--val-csv", default=None,
                        help="Override val CSV path (default: <data>/val.csv)")
    parser.add_argument("--sweep-conf", action="store_true",
                        help="Sweep conf_thresh on val set to find optimal threshold")
    parser.add_argument("--weights", default="runs/tracknet/weights/best_overall.pt",
                        help="Weights for --sweep-conf (default: best_overall.pt)")
    parser.add_argument("--conf-min",  type=float, default=0.10)
    parser.add_argument("--conf-max",  type=float, default=0.90)
    parser.add_argument("--conf-step", type=float, default=0.05)
    args = parser.parse_args()

    if args.preprocess:
        preprocess_frames(args.data)
    elif args.sweep_conf:
        sweep_conf_thresh(
            weights=args.weights,
            data_dir=args.data,
            val_csv=args.val_csv,
            device=args.device,
            num_workers=args.workers,
            batch_size=args.batch,
            data_root=args.data_root,
            conf_min=args.conf_min,
            conf_max=args.conf_max,
            conf_step=args.conf_step,
        )
    elif args.train:
        train_tracknet(args.data, epochs=args.epochs, batch_size=args.batch,
                       device=args.device,
                       num_workers=args.workers, resume=args.resume,
                       accum_steps=args.accum,
                       data_root=args.data_root,
                       seed=args.seed,
                       train_csv=args.train_csv,
                       val_csv=args.val_csv)
    else:
        print("Use --train, --sweep-conf, or --preprocess. See --help for options.")
