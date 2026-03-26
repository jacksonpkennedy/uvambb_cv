"""
TrackNet-style ball tracker — encoder-decoder heatmap regression for small ball detection.

Architecture based on TrackNet (yastrebksv/TrackNet unofficial PyTorch implementation):
  - Input: 3 consecutive RGB frames stacked → 9-channel tensor (640×360)
  - Output: heatmap (256 classes per pixel, reshaped to 360×640)
  - Ball location: extracted via thresholding + Hough circle detection

Usage:
  Inference:  wrapper = TrackNetInference("tracknet_basketball.pt", device="cuda:0")
              result  = wrapper.predict(frame)   # feed every frame
  Training:   python tracknet.py --train --data data/tracknet_labels/
"""

import argparse
import csv
import math
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
    """Encoder-decoder CNN: 9-ch input → 256-class heatmap output.

    The 256 output channels are treated as a per-pixel classification
    (softmax over 256 bins).  During inference the argmax gives a
    confidence-like heatmap that is thresholded + circle-detected.
    """

    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.conv1 = ConvBlock(9, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

        # --- Decoder ---
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        # 1-channel sigmoid output: direct heatmap regression
        self.conv18 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return torch.sigmoid(x)    # (B, 1, H, W) in [0, 1]

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
                        scale_y: float = 1.0) -> tuple:
    """Convert sigmoid heatmap → ball center (x, y, conf) in original coords.

    Args:
        heatmap: (1, H, W) or (H, W) float32 in [0, 1] — sigmoid output
        scale_x: frame_w / INPUT_W
        scale_y: frame_h / INPUT_H

    Returns:
        (x, y, conf) in original frame coordinates, or (None, None, 0.0)
    """
    if heatmap.ndim == 3:
        heatmap = heatmap[0]       # (1, H, W) → (H, W)
    heatmap = heatmap.reshape(INPUT_H, INPUT_W)

    # Peak confidence
    conf = float(heatmap.max())
    if conf < 0.1:
        return None, None, 0.0

    # Find peak location
    hmap_u8 = (heatmap * 255).astype(np.uint8)
    ret, binary = cv2.threshold(hmap_u8, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=2, minRadius=2, maxRadius=7,
    )

    if circles is not None and len(circles) >= 1:
        cx = float(circles[0][0][0]) * scale_x
        cy = float(circles[0][0][1]) * scale_y
        return cx, cy, conf

    # Fallback: argmax if Hough fails but peak is strong
    if conf >= 0.3:
        flat_idx = int(heatmap.argmax())
        py, px = divmod(flat_idx, INPUT_W)
        cx = float(px) * scale_x
        cy = float(py) * scale_y
        return cx, cy, conf

    return None, None, 0.0


# ---------------------------------------------------------------------------
# Gaussian heatmap generation (for training targets)
# ---------------------------------------------------------------------------

def generate_heatmap(cx: int, cy: int, w: int = INPUT_W, h: int = INPUT_H,
                     sigma: float = 5.0) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (cx, cy).

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

class TrackNetDataset(Dataset):
    """Load (frame_t-2, frame_t-1, frame_t) triples with heatmap targets.

    Expects a CSV file with columns: frame_path, visibility, x, y
    Rows must be in sequential order within each video clip.
    """

    def __init__(self, csv_path: str, sigma: float = 5.0):
        self.sigma = sigma
        self.entries = []       # list of (path, vis, x, y)
        self.sequences = []     # list of (idx_t-2, idx_t-1, idx_t) triples

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append((
                    row["frame_path"],
                    int(row["visibility"]),
                    float(row["x"]) if row["visibility"] != "0" else -1,
                    float(row["y"]) if row["visibility"] != "0" else -1,
                ))

        # Build valid 3-frame sequences (consecutive indices)
        for i in range(2, len(self.entries)):
            self.sequences.append((i - 2, i - 1, i))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        i0, i1, i2 = self.sequences[idx]

        frames = []
        for i in (i0, i1, i2):
            path = self.entries[i][0]
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
            img = cv2.resize(img, (INPUT_W, INPUT_H))
            frames.append(img)

        # Stack 3 frames → (H, W, 9)
        stacked = np.concatenate(frames, axis=2).astype(np.float32) / 255.0
        # → (9, H, W)
        inp = np.transpose(stacked, (2, 0, 1))

        # Target heatmap for the LAST frame (t)
        _, vis, x, y = self.entries[i2]
        if vis > 0 and x >= 0 and y >= 0:
            # Scale coordinates to heatmap resolution
            hmap = generate_heatmap(int(x), int(y), INPUT_W, INPUT_H,
                                    self.sigma)
        else:
            hmap = np.zeros((INPUT_H, INPUT_W), dtype=np.float32)

        # Target: (1, H, W) float32 heatmap for BCE loss
        target = hmap[np.newaxis, :, :]   # (1, H, W)

        return torch.from_numpy(inp), torch.from_numpy(target)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tracknet(data_dir: str, epochs: int = 30, batch_size: int = 4,
                   lr: float = 1e-3, device: str = "cuda:0",
                   output_dir: str = "runs/tracknet/weights"):
    """Train TrackNetV3 on basketball frame sequences."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_csv = str(Path(data_dir) / "train.csv")
    val_csv = str(Path(data_dir) / "val.csv")

    train_ds = TrackNetDataset(train_csv)
    val_ds = TrackNetDataset(val_csv) if Path(val_csv).exists() else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
                  if val_ds else None)

    model = TrackNetV3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Weighted BCE: ball pixels are rare (~0.01% of image), so weight positives higher
    pos_weight = torch.tensor([20.0], device=device)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch_idx, (inp, target) in enumerate(train_loader):
            inp = inp.to(device)
            target = target.to(device)          # (B, 1, H, W) float32

            out = model(inp)                    # (B, 1, H, W) sigmoid
            # Weighted BCE: upweight ball pixels (pos_weight=20) since
            # they're <0.01% of the heatmap. Without this, model learns
            # to predict all-zeros.
            loss = nn.functional.binary_cross_entropy(
                out, target,
                weight=torch.where(target > 0.5, pos_weight, torch.ones(1, device=device))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  "
                      f"batch {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}")

        avg_train = train_loss / max(len(train_loader), 1)

        # --- Validate ---
        avg_val = float("inf")
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inp, target in val_loader:
                    inp = inp.to(device)
                    target = target.to(device)
                    out = model(inp)
                    loss = nn.functional.binary_cross_entropy(
                        out, target,
                        weight=torch.where(target > 0.5, pos_weight, torch.ones(1, device=device))
                    )
                    val_loss += loss.item()
            avg_val = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch+1}/{epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), str(out_path / "best.pt"))
            print(f"  → Saved best model (val_loss={avg_val:.4f})")

        # Save latest
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
            self.model.load_state_dict(state)
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

        # Resize for model input
        resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        self._frame_buf.append(resized)

        if len(self._frame_buf) < 3:
            return None

        # Stack 3 frames → (1, 9, H, W) tensor
        stacked = np.concatenate(list(self._frame_buf), axis=2)
        stacked = stacked.astype(np.float32) / 255.0
        inp = np.transpose(stacked, (2, 0, 1))     # (9, H, W)
        inp = np.expand_dims(inp, axis=0)           # (1, 9, H, W)
        tensor = torch.from_numpy(inp).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)                # (1, 1, H, W) sigmoid

        heatmap = out[0].cpu().numpy()              # (1, H, W) float32 [0,1]

        scale_x = w / INPUT_W
        scale_y = h / INPUT_H

        cx, cy, conf = postprocess_heatmap(heatmap, scale_x, scale_y)

        if cx is None or conf < self.conf_thresh:
            return None

        # Synthesize bounding box
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.train:
        train_tracknet(args.data, epochs=args.epochs, batch_size=args.batch,
                       lr=args.lr, device=args.device)
    else:
        print("Use --train to start training. See --help for options.")
