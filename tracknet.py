"""
TrackNet-style ball tracker — encoder-decoder heatmap regression for small ball detection.

Architecture based on TrackNetV3 with U-Net skip connections:
  - Input: 3 consecutive RGB frames stacked → 9-channel tensor (640×360)
  - Output: 1-channel sigmoid heatmap
  - Ball location: extracted via weighted centroid of peak region

Usage:
  Inference:  wrapper = TrackNetInference("tracknet_basketball.pt", device="cuda:0")
              result  = wrapper.predict(frame)   # feed every frame
  Training:   python tracknet.py --train --data data/tracknet_labels/
"""

import argparse
import csv
import math
import os
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
                 stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class TrackNetV3(nn.Module):
    """Encoder-decoder CNN with U-Net skip connections.

    Input:  (B, 9, 360, 640)  — 3 consecutive RGB frames
    Output: (B, 1, 360, 640)  — heatmap sigmoid (or logits during training)

    Skip connections (U-Net style) are the key difference from the plain
    encoder-decoder: they pass fine spatial detail from the encoder to the
    decoder, which is critical for localising a ~5px ball in the heatmap.
    """

    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.conv1 = ConvBlock(9, 64)
        self.conv2 = ConvBlock(64, 64)          # → skip1 (64ch, full res)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)        # → skip2 (128ch, /2 res)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)        # → skip3 (256ch, /4 res)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)       # bottleneck (512ch, /8 res)

        # --- Decoder (with skip connections) ---
        # After ups1: 512ch. cat with skip3(256ch) → 768ch
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(768, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        # After ups2: 256ch. cat with skip2(128ch) → 384ch
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(384, 128)
        self.conv15 = ConvBlock(128, 128)
        # After ups3: 128ch. cat with skip1(64ch) → 192ch
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
        x = torch.cat([x, skip3], dim=1)        # 512+256=768
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

        if self.training:
            return x                            # raw logits for bce_with_logits
        return torch.sigmoid(x)                 # (B, 1, H, W) in [0, 1]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.05, 0.05)
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
    """Convert sigmoid heatmap → ball center (x, y, conf) in original coords.

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
                     sigma: float = 2.5) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (cx, cy).

    Sigma=2.5 matches TrackNetV3 paper (~10px in the original 1280x720 res,
    scaled to 640x360 → ~5px, FWHM ≈ 2.35*2.5 ≈ 6px).

    Returns (h, w) float32 array with values in [0, 1].
    """
    if cx < 0 or cy < 0:
        return np.zeros((h, w), dtype=np.float32)

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    hmap = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return hmap


# ---------------------------------------------------------------------------
# Dataset — sequences of 3 consecutive frames with ball center labels
# ---------------------------------------------------------------------------

def _parse_frame_number(path: str) -> int:
    """Extract frame number from path like .../frame_000123.jpg → 123."""
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

    def __init__(self, csv_path: str, sigma: float = 2.5,
                 oversample_visible: bool = False,
                 augment: bool = False):
        self.sigma = sigma
        self.augment = augment
        self.entries = []       # list of (path, vis, x, y)
        self.sequences = []     # list of (idx_t-2, idx_t-1, idx_t) triples

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append((
                    row["frame_path"],
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
            repeat = max(1, len(empty_seqs) // max(len(visible_seqs), 1))
            self.sequences = empty_seqs + visible_seqs * repeat
            print(f"  TrackNet dataset: {len(visible_seqs)} visible, "
                  f"{len(empty_seqs)} empty → oversampled {repeat}x "
                  f"→ {len(self.sequences)} total sequences")
        else:
            self.sequences = visible_seqs + empty_seqs
            if visible_seqs:
                print(f"  TrackNet dataset: {len(visible_seqs)} visible, "
                      f"{len(empty_seqs)} empty → {len(self.sequences)} total")

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, path: str) -> tuple:
        """Load a frame → ((3, H, W) float32 [0,1], orig_w, orig_h)."""
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

        # Scale label coords from original frame resolution → model resolution
        if vis2 > 0 and x2 >= 0 and y2 >= 0:
            cx = x2 * INPUT_W / orig_w
            cy = y2 * INPUT_H / orig_h
        else:
            cx, cy = -1.0, -1.0

        # --- Augmentation ---
        flip = False
        brightness = 1.0
        if self.augment:
            # Horizontal flip (50%)
            if pyrandom.random() < 0.5:
                flip = True
                f0 = np.flip(f0, axis=2).copy()
                f1 = np.flip(f1, axis=2).copy()
                f2 = np.flip(f2, axis=2).copy()
                if cx >= 0:
                    cx = INPUT_W - 1 - cx

            # Brightness jitter [0.7, 1.3]
            brightness = pyrandom.uniform(0.7, 1.3)
            f0 = np.clip(f0 * brightness, 0.0, 1.0)
            f1 = np.clip(f1 * brightness, 0.0, 1.0)
            f2 = np.clip(f2 * brightness, 0.0, 1.0)

            # Gaussian noise (std=0.02)
            noise = np.random.normal(0, 0.02, f2.shape).astype(np.float32)
            f0 = np.clip(f0 + noise, 0.0, 1.0)
            f1 = np.clip(f1 + noise, 0.0, 1.0)
            f2 = np.clip(f2 + noise, 0.0, 1.0)

            # Motion blur (30% chance) — simulates fast ball during shots/passes
            # Randomly horizontal or vertical kernel (ball moves in both directions)
            if pyrandom.random() < 0.3:
                ksize = pyrandom.choice([5, 7, 9, 11])
                if pyrandom.random() < 0.5:
                    kernel = np.zeros((ksize, ksize), dtype=np.float32)
                    kernel[ksize // 2, :] = 1.0 / ksize   # horizontal
                else:
                    kernel = np.zeros((ksize, ksize), dtype=np.float32)
                    kernel[:, ksize // 2] = 1.0 / ksize   # vertical
                # Apply only to f2 (target frame) — blur simulates the ball mid-flight
                # Transpose to (H,W,3) for cv2.filter2D, then back to (3,H,W)
                f2_hwc = np.transpose(f2, (1, 2, 0))
                f2_hwc = cv2.filter2D(f2_hwc, -1, kernel)
                f2 = np.transpose(f2_hwc, (2, 0, 1)).astype(np.float32)

        inp = np.concatenate([f0, f1, f2], axis=0)   # (9, H, W)

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
# Mixup collation helper
# ---------------------------------------------------------------------------

class MixupCollate:
    """Collate function with mixup augmentation — picklable for multiprocessing."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def __call__(self, batch):
        inps, hmaps, vis_flags = zip(*batch)
        inps      = torch.stack(inps)
        hmaps     = torch.stack(hmaps)
        vis_flags = torch.stack(vis_flags)

        if self.alpha > 0 and pyrandom.random() < 0.5:
            lam = np.random.beta(self.alpha, self.alpha)
            idx = torch.randperm(inps.size(0))
            inps  = lam * inps  + (1 - lam) * inps[idx]
            hmaps = lam * hmaps + (1 - lam) * hmaps[idx]

        return inps, hmaps, vis_flags


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tracknet(data_dir: str, epochs: int = 100, batch_size: int = 16,
                   lr: float = 1e-3, device: str = "cuda:0",
                   output_dir: str = "runs/tracknet/weights",
                   num_workers: int = 4):
    """Train TrackNetV3 on basketball frame sequences."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_csv = str(Path(data_dir) / "train.csv")
    val_csv   = str(Path(data_dir) / "val.csv")

    print("Building training dataset ...")
    train_ds = TrackNetDataset(train_csv, sigma=2.5,
                               oversample_visible=True, augment=True)
    val_ds   = (TrackNetDataset(val_csv, sigma=2.5, augment=False)
                if Path(val_csv).exists() else None)

    # Lazy loading: use num_workers for parallel disk I/O
    # persistent_workers keeps worker processes alive between epochs
    _pw = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
        collate_fn=MixupCollate(alpha=0.5),
    )
    val_loader = (DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    ) if val_ds else None)

    model = TrackNetV3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    use_amp = device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # pos_weight=200: ball pixels are ~0.004% of heatmap (1 pixel / 640*360)
    # High weight ensures model can't just predict all-zeros and get low loss
    pos_weight = torch.tensor([200.0], device=device)
    neg_weight = torch.ones(1, device=device)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch_idx, (inp, target, _vis) in enumerate(train_loader):
            inp    = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out  = model(inp)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    out, target,
                    weight=torch.where(target > 0.5, pos_weight, neg_weight),
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Gradient clipping prevents exploding gradients on hard samples
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  "
                      f"batch {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}")

        avg_train = train_loss / max(len(train_loader), 1)

        # --- Validate ---
        avg_val = float("inf")
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for inp, target, _vis in val_loader:
                    inp    = inp.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    out    = model(inp)
                    loss   = nn.functional.binary_cross_entropy_with_logits(
                        out, target,
                        weight=torch.where(target > 0.5, pos_weight, neg_weight),
                    )
                    val_loss += loss.item()
            avg_val = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch+1}/{epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        scheduler.step(avg_val if val_loader else avg_train)

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), str(out_path / "best.pt"))
            print(f"  → Saved best model (val_loss={avg_val:.4f})")

        # Save latest checkpoint
        torch.save(model.state_dict(), str(out_path / "last.pt"))

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

        # Stack 3 frames → (1, 9, H, W) tensor
        stacked = np.concatenate(list(self._frame_buf), axis=2)
        stacked = stacked.astype(np.float32) / 255.0
        inp = np.transpose(stacked, (2, 0, 1))      # (9, H, W)
        inp = np.expand_dims(inp, axis=0)            # (1, 9, H, W)
        tensor = torch.from_numpy(inp).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)                 # (1, 1, H, W) sigmoid

        heatmap = out[0].cpu().numpy()               # (1, H, W) float32 [0,1]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackNet ball tracker")
    parser.add_argument("--train", action="store_true",
                        help="Train TrackNet on basketball data")
    parser.add_argument("--data", default="data/tracknet_labels/",
                        help="Directory containing train.csv / val.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader num_workers")
    args = parser.parse_args()

    if args.train:
        train_tracknet(args.data, epochs=args.epochs, batch_size=args.batch,
                       lr=args.lr, device=args.device,
                       num_workers=args.workers)
    else:
        print("Use --train to start training. See --help for options.")
