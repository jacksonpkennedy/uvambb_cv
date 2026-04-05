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
import random as pyrandom

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

INPUT_W, INPUT_H = 640, 360    # TrackNet canonical resolution


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

        x = self.ups1(x)
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
        return x                                # always raw logits

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


# ---------------------------------------------------------------------------
# Heatmap post-processing — extract ball (x, y) from model output
# ---------------------------------------------------------------------------

def postprocess_heatmap(heatmap: np.ndarray,
                        scale_x: float = 1.0,
                        scale_y: float = 1.0,
                        local_radius: int = 10) -> tuple:
    """Convert sigmoid heatmap ->ball center (x, y, conf) in original coords.

    Finds the global peak, then computes a weighted centroid in a local window
    around that peak. This prevents drifting between two competing hot spots
    (e.g. ball and hoop both activating) which caused frame-to-frame jumping.

    Args:
        heatmap: (1, H, W) or (H, W) float32 in [0, 1] — sigmoid output
        scale_x: frame_w / INPUT_W
        scale_y: frame_h / INPUT_H
        local_radius: half-size of local window for centroid (model pixels)

    Returns:
        (x, y, conf) in original frame coordinates, or (None, None, 0.0)
    """
    if heatmap.ndim == 3:
        heatmap = heatmap[0]
    heatmap = heatmap.reshape(INPUT_H, INPUT_W)

    # Slight blur to suppress single-pixel noise before finding peak
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 1.0)

    conf = float(heatmap.max())
    if conf < 0.1:
        return None, None, 0.0

    # Find peak pixel
    flat_idx = int(heatmap.argmax())
    peak_y, peak_x = divmod(flat_idx, INPUT_W)

    # Weighted centroid in local window around peak only
    # This isolates the ball blob and ignores other hot regions
    y0 = max(0, peak_y - local_radius)
    y1 = min(INPUT_H, peak_y + local_radius + 1)
    x0 = max(0, peak_x - local_radius)
    x1 = min(INPUT_W, peak_x + local_radius + 1)

    patch = heatmap[y0:y1, x0:x1]
    threshold = 0.5 * conf
    mask = patch >= threshold
    if mask.any():
        ys, xs = np.where(mask)
        weights = patch[ys, xs]
        total = float(weights.sum())
        cx = float((xs * weights).sum() / total + x0) * scale_x
        cy = float((ys * weights).sum() / total + y0) * scale_y
        return cx, cy, conf

    # Fallback: raw peak location
    return float(peak_x) * scale_x, float(peak_y) * scale_y, conf


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

def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet focal loss for heatmap regression.

    For ball pixels (GT > 0): penalises by (1-pred)^2 so confident correct
    predictions are down-weighted.
    For background pixels: penalises by (1-GT)^4 * pred^2 so near-GT pixels
    (which are almost-ball by the Gaussian spread) are treated leniently.
    Normalised by number of positive pixels to remain scale-stable.
    """
    pred = torch.sigmoid(pred_logits)
    pos_mask = (target >= 0.01).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = -(torch.pow(1.0 - pred, 2.0)
                 * torch.log(pred.clamp(min=1e-6))) * pos_mask
    neg_loss = -(torch.pow(1.0 - target, 4.0)
                 * torch.pow(pred, 2.0)
                 * torch.log((1.0 - pred).clamp(min=1e-6))) * neg_mask

    # Standard CenterNet normalization: both terms divided by num_pos.
    # The (1-GT)^4 * pred^2 focal weighting already suppresses easy negatives —
    # dividing neg by num_total on top of that made background penalty ~1700x
    # weaker per-pixel, so the model could never learn to suppress false positives.
    num_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


# ---------------------------------------------------------------------------
# PE (Positioning Error) evaluation — TrackNet paper metric
# ---------------------------------------------------------------------------

def evaluate_pe(model: "TrackNetV3", loader, device: str,
                pe_thresh: float = 5.0,
                conf_thresh: float = 0.5) -> dict:
    """Compute val_loss + precision/recall/F1 in a single val pass.

    TP: model predicts ball AND predicted center ≤ pe_thresh px from GT.
    FP: model predicts ball but no GT, or distance > pe_thresh.
    FN: model predicts no ball but GT exists.
    """
    tp = fp = fn = 0
    total_loss = 0.0
    model.eval()
    use_amp = device != "cpu"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for inp, target, vis_flags in loader:
            inp    = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            out    = model(inp)          # (B, 1, H, W) logits
            total_loss += focal_loss(out, target).item()
            pred_np   = torch.sigmoid(out).cpu().numpy()
            target_np = target.cpu().numpy()
            vis_np    = vis_flags.cpu().numpy()
            for b in range(pred_np.shape[0]):
                has_ball = int(vis_np[b]) > 0
                gt_hmap  = target_np[b, 0]
                if has_ball:
                    gt_y, gt_x = divmod(int(gt_hmap.argmax()), INPUT_W)
                else:
                    gt_x = gt_y = -1
                pred_hmap   = pred_np[b, 0]
                pred_conf   = float(pred_hmap.max())
                pred_detect = pred_conf >= conf_thresh
                if pred_detect:
                    pr_y, pr_x = divmod(int(pred_hmap.argmax()), INPUT_W)
                if pred_detect and has_ball:
                    dist = math.hypot(pr_x - gt_x, pr_y - gt_y)
                    if dist <= pe_thresh:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif pred_detect:
                    fp += 1
                elif has_ball:
                    fn += 1
    avg_loss = total_loss / max(len(loader), 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"val_loss": avg_loss, "precision": prec, "recall": rec,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn}


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

            # Brightness jitter [0.7, 1.3]
            b = pyrandom.uniform(0.7, 1.3)
            f0 = np.clip(f0 * b, 0.0, 1.0)
            f1 = np.clip(f1 * b, 0.0, 1.0)
            f2 = np.clip(f2 * b, 0.0, 1.0)

            # Independent Gaussian noise per frame (std=0.02)
            # Single allocation for all 3 frames — ~3x faster than separate calls
            noise = np.random.normal(0, 0.02, (3, *f0.shape)).astype(np.float32)
            f0 = np.clip(f0 + noise[0], 0.0, 1.0)
            f1 = np.clip(f1 + noise[1], 0.0, 1.0)
            f2 = np.clip(f2 + noise[2], 0.0, 1.0)

            # Motion blur (30% chance) — simulates fast ball during shots/passes
            if pyrandom.random() < 0.3:
                ksize = pyrandom.choice([5, 7, 9, 11])
                kernel = np.zeros((ksize, ksize), dtype=np.float32)
                if pyrandom.random() < 0.5:
                    kernel[ksize // 2, :] = 1.0 / ksize   # horizontal
                else:
                    kernel[:, ksize // 2] = 1.0 / ksize   # vertical
                f2_hwc = np.transpose(f2, (1, 2, 0))
                f2_hwc = cv2.filter2D(f2_hwc, -1, kernel)
                f2 = np.transpose(f2_hwc, (2, 0, 1)).astype(np.float32)

        # Motion diff maps (V4): |frame_{t-1} - frame_{t-2}| and |frame_t - frame_{t-1}|
        # Computed AFTER augmentation so they reflect the augmented frames.
        # Averaged over RGB channels ->(1, H, W) motion magnitude in [0, 1].
        diff01 = np.abs(f1 - f0).mean(axis=0, keepdims=True)  # (1, H, W)
        diff12 = np.abs(f2 - f1).mean(axis=0, keepdims=True)  # (1, H, W)
        inp = np.concatenate([f0, f1, f2, diff01, diff12], axis=0)   # (11, H, W)

        if cx >= 0 and cy >= 0 and vis2 > 0:
            hmap = generate_heatmap(int(cx), int(cy), INPUT_W, INPUT_H,
                                    self.sigma)
        else:
            hmap = np.zeros((INPUT_H, INPUT_W), dtype=np.float32)
        hmap = hmap[np.newaxis, :, :]   # (1, H, W)

        return (torch.from_numpy(inp),
                torch.from_numpy(hmap),
                torch.tensor(1 if (vis2 > 0 and cx >= 0) else 0))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tracknet(data_dir: str, epochs: int = 100, batch_size: int = 8,
                   device: str = "cuda:0",
                   output_dir: str = "runs/tracknet/weights",
                   num_workers: int = 4,
                   resume: bool = False,
                   accum_steps: int = 2,
                   data_root: str | None = None):
    """Train TrackNetV3 on basketball frame sequences.

    Args:
        accum_steps: gradient accumulation steps. Effective batch size is
                     batch_size * accum_steps (default 8*2=16).
        data_root: root directory for frame data. When set, CSV paths like
                   'data\\...' are remapped to data_root/... for fast local I/O.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_csv = str(Path(data_dir) / "train.csv")
    val_csv   = str(Path(data_dir) / "val.csv")

    print("Building training dataset ...")
    train_ds = TrackNetDataset(train_csv, oversample_visible=True, augment=True,
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
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1        = 0.0
    no_improve     = 0       # epochs since best F1 improved
    early_stop_pat = 15
    start_epoch    = 0

    # Resume from last checkpoint if requested
    ckpt_path = out_path / "last_ckpt.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
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

    for epoch in range(start_epoch, epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inp, target, _vis) in enumerate(train_loader):
            inp    = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out  = model(inp)
                loss = focal_loss(out, target) / accum_steps

            scaler.scale(loss).backward()

            # Step optimizer every accum_steps batches (or at end of epoch)
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * accum_steps  # undo the /accum_steps for logging

            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  "
                      f"batch {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item() * accum_steps:.4f}")

        avg_train = train_loss / max(len(train_loader), 1)

        # --- Validate + PE metric in a single pass ---
        if val_loader:
            pe = evaluate_pe(model, val_loader, device,
                             pe_thresh=5.0, conf_thresh=0.5)
            print(f"Epoch {epoch+1}/{epochs}  "
                  f"train_loss={avg_train:.4f}  val_loss={pe['val_loss']:.4f}  "
                  f"prec={pe['precision']:.3f}  rec={pe['recall']:.3f}  "
                  f"F1={pe['f1']:.3f}  "
                  f"(TP={pe['tp']} FP={pe['fp']} FN={pe['fn']})")
            if pe["f1"] > best_f1:
                best_f1    = pe["f1"]
                no_improve = 0
                torch.save(model.state_dict(), str(out_path / "best.pt"))
                print(f"  -> Saved best model (F1={best_f1:.3f})")
            else:
                no_improve += 1
                print(f"  No F1 improvement ({no_improve}/{early_stop_pat})")
        else:
            print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_train:.4f}")

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
        }, str(out_path / "last_ckpt.pt"))

        gc.collect()
        torch.cuda.empty_cache()

        if val_loader and no_improve >= early_stop_pat:
            print(f"\nEarly stopping: F1 hasn't improved for "
                  f"{early_stop_pat} epochs (best F1={best_f1:.3f})")
            break

    print(f"\nTraining complete. Best model: {out_path / 'best.pt'}")
    return str(out_path / "best.pt")


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
                 conf_thresh: float = 0.5, ball_radius: int = 15):
        self.device = device
        self.conf_thresh = conf_thresh
        self.ball_radius = ball_radius
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

    def predict(self, frame: np.ndarray) -> dict | None:
        """Feed a frame and return a ball detection dict or None.

        Always call this for EVERY frame (including skipped ones) so the
        3-frame sliding window stays in sync with the video.

        Returns:
            {"tid": -1, "box": [x1,y1,x2,y2], "cls": 0, "conf": float}
            or None if ball not detected / not enough frames buffered.
        """
        h, w = frame.shape[:2]
        self._frame_w = w
        self._frame_h = h

        resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        self._frame_buf.append(resized)

        if len(self._frame_buf) < 3:
            return None

        # Build 11-channel input: 9 RGB + 2 motion diff maps (V4)
        frames = [f.astype(np.float32) / 255.0 for f in self._frame_buf]
        f0 = np.transpose(frames[0], (2, 0, 1))     # (3, H, W) t-2
        f1 = np.transpose(frames[1], (2, 0, 1))     # (3, H, W) t-1
        f2 = np.transpose(frames[2], (2, 0, 1))     # (3, H, W) t
        diff01 = np.abs(f1 - f0).mean(axis=0, keepdims=True)  # (1, H, W)
        diff12 = np.abs(f2 - f1).mean(axis=0, keepdims=True)  # (1, H, W)
        inp = np.concatenate([f0, f1, f2, diff01, diff12], axis=0)  # (11, H, W)
        inp = np.expand_dims(inp, axis=0)            # (1, 11, H, W)
        tensor = torch.from_numpy(inp).to(self.device)

        # TTA: average original + horizontally-flipped heatmaps.
        # Flip along the width axis (dim=-1), run both through the model,
        # then flip the second prediction back before averaging.
        tensor_flip = torch.flip(tensor, dims=[-1])
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=self.device != "cpu"):
            out_orig = torch.sigmoid(self.model(tensor))       # (1,1,H,W)
            out_flip = torch.sigmoid(self.model(tensor_flip))  # (1,1,H,W)
        out_flip_back = torch.flip(out_flip, dims=[-1])
        heatmap = ((out_orig + out_flip_back) / 2.0)[0].cpu().numpy()  # (1,H,W)

        scale_x = w / INPUT_W
        scale_y = h / INPUT_H

        cx, cy, conf = postprocess_heatmap(heatmap, scale_x, scale_y)

        if cx is None or conf < self.conf_thresh:
            return None

        r = self.ball_radius
        box = [
            max(0, int(cx - r)),
            max(0, int(cy - r)),
            min(w, int(cx + r)),
            min(h, int(cy + r)),
        ]

        return {
            "tid": -1,
            "box": box,
            "cls": self.CLS_BALL,
            "conf": conf,
        }

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackNet ball tracker")
    parser.add_argument("--train", action="store_true",
                        help="Train TrackNet on basketball data")
    parser.add_argument("--preprocess", action="store_true",
                        help="Pre-resize all frames to 640x360 for faster training")
    parser.add_argument("--data", default="data/tracknet_merged/",
                        help="Directory containing train.csv / val.csv")
    parser.add_argument("--data-root",
                        default=r"C:\Users\wanns\Desktop\Personal Project Data",
                        help="Root directory for frame data (remaps CSV 'data\\' paths)")
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
    args = parser.parse_args()

    if args.preprocess:
        preprocess_frames(args.data)
    elif args.train:
        train_tracknet(args.data, epochs=args.epochs, batch_size=args.batch,
                       device=args.device,
                       num_workers=args.workers, resume=args.resume,
                       accum_steps=args.accum,
                       data_root=args.data_root)
    else:
        print("Use --train or --preprocess. See --help for options.")
