# UVA Men's Basketball Computer Vision Analysis

## Project Overview

A basketball analysis pipeline that detects, tracks, and annotates players, referees, the basketball, and the hoop from game footage using a fine-tuned YOLO11s model with ByteTrack multi-object tracking, TrackNet heatmap-based ball detection with test-time augmentation, Kalman-filtered ball trajectory smoothing, pose estimation, jersey number OCR, temporal interpolation with outlier rejection, and automatic team classification.

**Team Members:**
- Jackson Kennedy
- Nathan Wan
- Nathan Todd
- Hudson Noyes
- James Sweat

---

## Pipeline Architecture

```
                              ┌─────────────────────────────────┐
                              │         VIDEO INPUT             │
                              │   1280x720 @ 60fps (mp4)        │
                              └───────────────┬─────────────────┘
                                              │
                              ┌───────────────▼─────────────────┐
                              │        FRAME SCHEDULER          │
                              │  Process every frame            │
                              │  (configurable frame_skip=1)    │
                              └───────────────┬─────────────────┘
                                              │
          ┌───────────────────────────────────┼──────────────────────┐
          │                                   │                      │
   ┌──────▼──────────────┐      ┌─────────────▼──────────────┐      │
   │  FINE-TUNED YOLO11s │      │     TRACKNET (PRIMARY)     │      │
   │  Detection+Tracking │      │  3-frame heatmap regression │      │
   │  4 cls + ByteTrack  │      │  11-ch input (RGB + motion) │      │
   │                     │      │  Runs EVERY frame for        │      │
   │  Players, Refs,     │      │  sliding window sync         │      │
   │  Hoops              │      │  TTA (horizontal flip avg)   │      │
   └──────┬──────────────┘      └─────────────┬──────────────┘      │
          │                                   │                      │
          │                        ┌──────────▼───────────────┐      │
          │                        │     BALL VALIDATION      │      │
          │                        │  Court region → Teleport │      │
          │                        │  → Backboard constraint  │      │
          │                        └──────────┬───────────────┘      │
          │                                   │                      │
          │                        ┌──────────▼───────────────┐      │
          │                        │   KALMAN FILTER          │      │
          │                        │   4-state constant-vel   │      │
          │                        │   Smooths trajectory     │      │
          │                        └──────────┬───────────────┘      │
          │                                   │                      │
          └────────────────────────┬──────────┘──────────────────────┘
                                   │
          ┌────────────────────────┼──────────────────────────────┐
          │                        │                              │
   ┌──────▼──────────────┐  ┌─────▼────────────────┐  ┌─────────▼──────────────┐
   │  PLAYER PROCESSING  │  │    BALL TRACKING     │  │   POSE ESTIMATION      │
   │                     │  │                      │  │                        │
   │  Re-ID (jersey+prox)│  │  BallTracker (Kalman)│  │   YOLO11n-pose         │
   │  Jersey OCR         │  │  BallInterpolator    │  │   Every processed      │
   │  Team classify      │  │  (gap filling)       │  │   frame, cached        │
   │  Velocity tracker   │  │                      │  │                        │
   └──────┬──────────────┘  └──────────┬───────────┘  └───────────────┬────────┘
          │                            │                              │
          └────────────────────────────┼──────────────────────────────┘
                                       │
                              ┌────────▼────────────────────────┐
                              │         ANNOTATION              │
                              │  Draw boxes, skeletons, labels  │
                              │  Team colors, jersey numbers    │
                              └────────┬────────────────────────┘
                                       │
                              ┌────────▼────────────────────────┐
                              │      INTERPOLATION BUFFER       │
                              │  20-frame look-ahead            │
                              │  Outlier rejection (1-3 spikes) │
                              │  Linear + arc gap filling       │
                              └────────┬────────────────────────┘
                                       │
                              ┌────────▼────────────────────────┐
                              │        VIDEO OUTPUT             │
                              │   Annotated MP4 @ original fps  │
                              └─────────────────────────────────┘
```

### Component Summary

| Component | Implementation | Details |
|---|---|---|
| Detection | YOLO11s (fine-tuned) | Fine-tuned for players/hoop/ref (4 classes) |
| Tracking | ByteTrack | Custom config — 4s track buffer, low thresholds for ball |
| Ball Detection | TrackNet (primary) | Encoder-decoder heatmap regression with U-Net skip connections on 3 consecutive frames (11-ch input: 9 RGB + 2 motion diff maps) with TTA |
| Ball Validation | BallValidator | Court region, teleport rejection, backboard constraint |
| Ball Interpolation | BallInterpolator | 20-frame look-ahead buffer, relative outlier rejection (1-3 frame spikes), linear + parabolic gap filling |
| Ball Tracker | BallTracker | Picks highest-confidence detection, Kalman filter (constant-velocity model) for trajectory smoothing, maintains velocity history |
| Pose Estimation | YOLO11n-pose | 17-point COCO skeleton, every processed frame |
| Jersey OCR | EasyOCR | Adaptive frequency (every 30 frames unconfirmed, 90 confirmed), majority voting |
| Player Re-ID | TemporalReIDBuffer | Jersey-based matching (ignores distance) + proximity fallback (300px), 10s memory |
| Velocity Tracker | VelocityTracker | Preserves player IDs through overlapping bounding boxes |
| Team Classification | TeamClassifier | K-means (k=2) on HSV jersey color histograms, auto-detects home vs away |

### Detection Classes

| ID | Class | Color (BGR) | Notes |
|---|---|---|---|
| 0 | basketball | Red (0,0,255) | Oversampled 8x during training, TrackNet primary detector |
| 1 | hoop | White (255,255,255) | Cross-hair marker at center |
| 2 | player | Team color (auto) | Pose skeleton + jersey OCR, color-coded by team after classification |
| 3 | referee | Dark Blue (180,60,20) | No pose estimation |

---

## Project Structure

```
uvambb_cv/
├── main.py                          # Full pipeline — fine-tuning + inference
├── tracknet.py                      # TrackNet model — training, inference, TTA, focal loss
├── auto_label_tracknet.py           # Auto-label video frames for TrackNet training (YOLO + SAHI)
├── convert_labels.py                # Convert YOLO-seg annotations → TrackNet CSV format
├── merge_labels.py                  # Merge per-game TrackNet labels with per-game 85/15 split
├── clean_labels.py                  # Re-verify auto-labels at higher YOLO confidence (removes bad labels)
├── verify_labels.py                 # Visual inspection — draws labeled positions on random sample of frames
├── bytetrack_players.yaml           # ByteTrack tracker config
├── requirements.txt                 # Python dependencies
├── data/
│   ├── custom_annotations/          # Roboflow export (YOLOv11 format)
│   │   ├── data.yaml                # Dataset config (nc=4, class names)
│   │   ├── train/                   # ~391 training images + labels
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── valid/                   # 97 validation images + labels (80/20 split)
│   │       ├── images/
│   │       └── labels/
│   ├── tracknet_merged/             # Merged training data from all games (~28K frames)
│   │   ├── train.csv                # Training labels (85% per game)
│   │   └── val.csv                  # Validation labels (last 15% per game)
│   ├── tracknet_autolabels/         # Auto-labeled frames (game_01)
│   ├── tracknet_autolabels_02/      # Auto-labeled frames (game_02)
│   ├── tracknet_autolabels_03/      # Auto-labeled frames (game_03)
│   ├── tracknet_labels/             # Converted YOLO → TrackNet labels (from Roboflow data)
│   └── game_01.mp4                  # Game footage
├── runs/
│   ├── detect/train/weights/        # Fine-tuned YOLO weights
│   │   ├── best.pt                  # Best val mAP checkpoint
│   │   ├── best.engine              # TensorRT engine (after --export-tensorrt)
│   │   └── last.pt                  # Latest epoch checkpoint
│   └── tracknet/weights/            # TrackNet trained weights
│       ├── best.pt                  # Best val F1 checkpoint
│       └── last_ckpt.pt             # Full resumable checkpoint (model + optimizer + scheduler + scaler)
└── output/
    └── annotated.mp4                # Annotated output video
```

---

## Installation & Setup

**Requirements:** Python 3.10+, CUDA GPU recommended

```bash
pip install ultralytics        # YOLO11 detection + pose
pip install opencv-python      # Video I/O + drawing
pip install numpy torch        # Core dependencies
pip install easyocr            # Jersey number recognition
pip install sahi               # Sliced inference (used in auto-labeling only)
```

---

## Usage

### Fine-tune on your dataset

```bash
python main.py --finetune [--epochs 50] [--batch 8]
```

This will:
1. Oversample basketball images 8x to balance class distribution
2. Train YOLO11s with transfer learning from COCO weights
3. Save best weights to `runs/detect/train/weights/best.pt`
4. Early stopping with patience=10 (stops if val mAP plateaus)

### Run inference

```bash
python main.py --video data/game_01.mp4 [--weights path/to/best.pt]
```

Automatically loads `runs/detect/train/weights/best.pt` if no weights specified.

### Train TrackNet (ball detection)

```bash
# Step 1: Auto-label frames from video(s)
python auto_label_tracknet.py --video data/game_01.mp4 data/game_02.mp4 --max-frames 5000

# Step 2: Merge labels from all games (per-game 85/15 val split)
python merge_labels.py

# Step 3: (Optional) Clean labels — re-verify at higher confidence, removes ~20% bad labels
python clean_labels.py --data data/tracknet_merged --weights runs/detect/train/weights/best.pt

# Step 4: (Optional) Visually verify labels — check random sample of frames
python verify_labels.py --data data/tracknet_merged --n 100

# Step 5: Pre-process frames to .npy cache (5x faster loading on Windows)
python tracknet.py --preprocess --data data/tracknet_merged

# Step 6: Train
python tracknet.py --train --data data/tracknet_merged --epochs 100 --batch 8 --resume
```

### Export to TensorRT (2-3x faster inference)

```bash
python main.py --export-tensorrt [--weights path/to/best.pt]
```

One-time export — engines are GPU-specific and auto-loaded on subsequent runs.

### Disable TrackNet

```bash
python main.py --video data/game_01.mp4 --no-tracknet
```

---

## Training Configuration

### YOLO Fine-tuning

| Parameter | Value | Rationale |
|---|---|---|
| Base model | yolo11s.pt | Small model, fast training, COCO pretrained |
| Image size | 960 | Balance between detail and VRAM |
| Batch size | 8 | Fits in GPU VRAM |
| Gradient accum (nbs) | 64 | Effective batch=64 for smoother training |
| Epochs | 50 (max) | Early stopping typically fires at 20-30 |
| Patience | 10 | Fast convergence with small dataset |
| Learning rate | 0.001 → 0.00001 | Cosine decay (lrf=0.01) |
| Dropout | 0.10 | Regularization for small dataset |
| Basketball oversample | 8x | 159 → ~1,272 annotations |

### Augmentation Strategy

Aggressive color augmentation to force shape-based learning over color shortcuts:
- **HSV jitter**: h=0.15, s=0.7, v=0.4 — strong hue/saturation shifts prevent the model from relying on "orange = basketball" and force it to learn the round shape and seam texture
- **Geometric**: mosaic, mixup (0.15), copy-paste (0.3), random erasing (0.4), rotation (10°), translation, scale, shear, perspective warp, horizontal flip
- **No Roboflow augmentation** — YOLO's on-the-fly augmentation generates infinite variations per epoch vs. a fixed augmented set

### TrackNet Training

| Parameter | Value | Rationale |
|---|---|---|
| Architecture | TrackNetV3 with U-Net skip connections | 11-ch input (3 RGB frames + 2 motion diff maps) → 1-ch heatmap, 256-ch bottleneck |
| Input resolution | 640x360 | TrackNet canonical size, fast inference |
| Target | 2D Gaussian heatmap (sigma=3.16 = sqrt(10)) | Ball center as soft probability map, per TrackNetV1 paper |
| Loss | CenterNet focal loss (Zhou et al. 2019) | Adaptive weighting for extreme class imbalance — pos normalized by num_pos, neg normalized by num_total |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) | Decoupled weight decay (Loshchilov & Hutter 2019) |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) | Smooth LR decay, crash-safe (only decreases) |
| Gradient accumulation | batch=8 x accum=2 = effective 16 | Larger effective batch without extra VRAM |
| Mixed precision | AMP fp16 (autocast + GradScaler) | ~2x throughput on RTX 4070 |
| Early stopping | Patience 15 on val F1 | Stops when PE F1 metric plateaus |
| Weight init | Kaiming He (fan_out, relu) | Proper init for Conv-BN-ReLU blocks |
| Augmentation | Horizontal flip, brightness jitter, Gaussian noise (per-frame), motion blur (30%) | Data variety without distorting ball position |
| Oversampling | Visible sequences repeated to match empty count | Addresses ~60% empty frames in training data |
| Inference TTA | Horizontal flip + average | Eliminates left/right asymmetry, ~1-3% precision gain |
| Evaluation metric | PE (Positioning Error) at 5px threshold | Precision, recall, F1 computed on predicted vs GT ball center distance |
| Training data | ~28K frames from 3 games (merged) | Auto-labeled with YOLO+SAHI, per-game last-15% val split |
| Frame cache | .npy binary (pre-resized 640x360) | ~5x faster than JPEG decode, critical on Windows |
| Checkpoint | Full state (model + optimizer + scheduler + scaler + best_f1) | Crash-safe resume with `--resume` |

---

## ByteTrack Configuration

```yaml
track_high_thresh: 0.25    # Low to catch ball at low confidence
track_low_thresh: 0.05     # Very low rescue threshold for ball during passes
new_track_thresh: 0.35     # Low to pick up ball quickly
track_buffer: 240           # 4 seconds at 60fps
match_thresh: 0.70          # Loose for fast ball movement
```

---

## Performance Optimizations

| Optimization | Category | Impact | Accuracy Cost |
|---|---|---|---|
| Reduce imgsz 1920 → 1280 | Resolution | ~35% faster per call | None (video is 720p) |
| `@torch.inference_mode()` | Memory | Reduces overhead | None |
| TensorRT export (`.engine`) | Compilation | 2-3x per model | None |
| TrackNet TTA (flip average) | Inference | ~2x TrackNet forward pass | Positive (+1-3% precision) |
| Kalman filter smoothing | Post-processing | Negligible cost | Positive (reduces jitter) |
| cuDNN benchmark mode | Training | 10-20% training speedup | None (fixed input size) |
| .npy frame cache | Training I/O | ~5x faster data loading | None |
| Gradient accumulation | Training | Larger effective batch, same VRAM | None |
| AMP fp16 mixed precision | Training | ~2x throughput | Negligible |
| Pose every processed frame | Inference | Full skeleton coverage | None |
| Reduced OCR frequency | CPU savings | Less EasyOCR | Slightly slower jersey convergence |
| Deque for interpolation buffer | Data structure | O(1) vs O(n) pop | None |

---

## Ball Detection Pipeline

The basketball is the hardest object to detect (small, fast-moving, similar color to shoes and skin). TrackNet is the sole ball detector in the inference pipeline:

```
  TrackNet (3-frame heatmap + TTA)   ← runs EVERY frame
        │
        ├── heatmap peak ≥ conf threshold? ──YES──→ BallValidator → Kalman → BallInterpolator
        │
        NO → no detection this frame (Kalman predicts state forward)
```

### Why TrackNet over YOLO for ball detection

YOLO sees one frame at a time — the ball is ~15-25px with motion blur, often indistinguishable from shoes, bald heads, and court markings in a single frame. TrackNet stacks 3 consecutive frames (11-channel input: 9 RGB + 2 motion diff maps) and learns *motion patterns*: a blurry streak across 3 frames is a strong signal even when a single frame is ambiguous. It outputs a heatmap (probability per pixel) rather than a bounding box, which is better suited for localizing a single small point.

### Post-processing
1. **Gaussian blur** (5x5, sigma=1.0) — suppresses single-pixel noise
2. **Weighted centroid** — local window (10px radius) around peak, weighted by heatmap intensity above 50% of peak confidence
3. **Fallback** — raw peak pixel location if centroid computation fails

### Validation (BallValidator)
Rejects false positives with 3 checks:
1. **Court region** — ball must be within court column bounds (with 5% horizontal padding), below court bottom rejected
2. **Teleport rejection** — max 450px/frame movement (scaled to resolution), with grace period reset after 10 frames unseen
3. **Backboard constraint** — if ball was near hoop recently, reject detections above and behind the hoop (crowd area)

### Tracking (BallTracker) with Kalman Filter
- Picks highest-confidence detection per frame
- **Kalman filter** (4-state constant-velocity model: x, y, vx, vy) smooths frame-to-frame jitter
  - Measurement noise R=15, process noise Q=0.5
  - Auto-resets on teleport-size jumps to avoid filter divergence
  - Smoothed box rebuilt around filtered center
- 5-frame velocity history (uses raw centers to avoid Kalman feedback loop)

### Temporal Interpolation (BallInterpolator)
Post-game look-ahead analysis with 20-frame sliding buffer:

**Outlier Rejection** — removes 1-3 frame detection spikes using a *relative* test:
- Compare spike's jump distance to the neighbor-to-neighbor distance (how far the real ball moved)
- If the spike jumps 3x+ further than the neighbors moved apart (+ 20px margin), it's a false positive
- This catches nearby false positives (shirts, shoes 50-80px away) that absolute thresholds miss

**Gap Filling** — interpolates up to 8 consecutive missing frames:
- Linear interpolation for short gaps (<=3 frames)
- Parabolic arc interpolation for longer gaps (>3 frames) — models gravity for shots/lobs
- Distance sanity check prevents interpolation between different objects

---

## Auto-labeling Pipeline

TrackNet requires sequential frame labels (CSV with frame_path, visibility, x, y). The auto-labeling pipeline generates these from raw game video:

1. **auto_label_tracknet.py** — runs YOLO (fine-tuned) + SAHI (COCO sports ball) on each frame with full BallValidator + BallTracker filtering. SAHI is used here (not in main inference) because it improves label recall for the training set.
2. **merge_labels.py** — combines labels from multiple games into `data/tracknet_merged/`, using a per-game last-15% val split to ensure each split has representative data from all games.
3. **clean_labels.py** — re-verifies every "visible" label by running YOLO at conf >= 0.50. If no ball is detected within 30px of the labeled position, the label is demoted to invisible. Addresses ~20% mislabel rate from low-confidence auto-labeling.
4. **verify_labels.py** — draws green circles at labeled positions on a random sample of frames and saves annotated images for manual visual inspection.
5. **convert_labels.py** — converts YOLO-seg polygon annotations (from Roboflow) to TrackNet CSV format, extracting centroids from polygon vertices. Deduplicates augmented frames.

---

## Player Tracking & Re-ID

### Jersey OCR
- EasyOCR with digit-only allowlist on upper 55% of player bbox (torso crop)
- **Adaptive frequency**: every 30 frames for unconfirmed players, every 90 for confirmed
- **Majority voting**: number must have 2+ reads AND be the plurality candidate to lock
- Confirmed numbers persist permanently, transferred on track ID remaps

### Player Re-Identification
Two-tier matching when players reappear after going off-camera:
1. **Jersey-based (primary)** — if new track's confirmed jersey matches a lost track's jersey, force re-association regardless of distance. Handles players returning from opposite side of court.
2. **Proximity fallback** — nearest lost track within 300px (for unconfirmed players)
3. **10-second memory** (600 frames at 60fps) — long enough for timeouts and substitutions when jersey matching is available

### Team Classification
- Automatic home/away detection using K-means (k=2) on HSV color histograms from upper-torso jersey crops
- Collects 100 player samples, then clusters into two teams
- New players classified by nearest cluster center
- Bounding boxes color-coded by team (orange vs cyan)

---

## Known Limitations

- Ball tracking degrades during fast passes and heavy occlusion
- Jersey OCR accuracy depends on camera resolution and player orientation
- Court ROI polygon is hardcoded — needs adjustment for different camera angles
- Pose estimation can occasionally snap to nearby referees despite filtering
- Team classification assumes two distinct jersey colors — may struggle with similar uniforms
- Auto-labeled training data has inherent noise (~20% error rate before cleaning)

---

---

*Last Updated: April 2026*
