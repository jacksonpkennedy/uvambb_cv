# UVA Men's Basketball Computer Vision Analysis

## Project Overview

A basketball analysis pipeline that detects, tracks, and annotates players, referees, the basketball, and the hoop from game footage using a fine-tuned YOLO11s model with ByteTrack multi-object tracking, pose estimation, jersey number OCR, SAHI-enhanced ball detection, temporal interpolation with outlier rejection, and automatic team classification.

**Team Members:**
- Jackson Kennedy
- Nathan Wan
- Nathan Todd

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
                              │  Process every 2nd frame        │
                              │  (skipped frames reuse results) │
                              └───────────────┬─────────────────┘
                                              │
                 ┌────────────────────────────┼────────────────────────────┐
                 │                            │                            │
    ┌────────────▼────────────┐  ┌────────────▼────────────┐  ┌───────────▼───────────┐
    │   FINE-TUNED YOLO11s    │  │   COCO FALLBACK (11n)   │  │   SAHI SLICED INFER   │
    │   Detection + Tracking  │  │   "sports ball" cls 32  │  │   Targeted 128px crop │
    │   4 classes + ByteTrack │  │   Only if no ball found │  │   Only if still no    │
    │                         │  │                         │  │   ball after COCO     │
    │   Players, Refs, Hoops  │  │   Ball candidates       │  │   Ball candidates     │
    └────────┬────────────────┘  └────────────┬────────────┘  └───────────┬───────────┘
             │                                │                           │
             └────────────────────────────────┼───────────────────────────┘
                                              │
                              ┌───────────────▼─────────────────┐
                              │       BALL VALIDATION           │
                              │  Size → Aspect → Court region   │
                              │  → Teleport → Backboard         │
                              └───────────────┬─────────────────┘
                                              │
                 ┌────────────────────────────┼────────────────────────────┐
                 │                            │                            │
    ┌────────────▼────────────┐  ┌────────────▼────────────┐  ┌───────────▼───────────┐
    │   PLAYER PROCESSING     │  │   BALL TRACKING          │  │   POSE ESTIMATION     │
    │                         │  │                          │  │                       │
    │  Re-ID (jersey + prox)  │  │  BallTracker (velocity)  │  │   YOLO11n-pose        │
    │  Jersey OCR (EasyOCR)   │  │  BallReID (stable IDs)   │  │   Every 3rd processed │
    │  Team classify (K-means)│  │  BallInterpolator (gaps)  │  │   frame, cached       │
    │  Velocity tracker       │  │                          │  │                       │
    └────────┬────────────────┘  └────────────┬────────────┘  └───────────┬───────────┘
             │                                │                           │
             └────────────────────────────────┼───────────────────────────┘
                                              │
                              ┌───────────────▼─────────────────┐
                              │         ANNOTATION              │
                              │  Draw boxes, skeletons, labels  │
                              │  Team colors, jersey numbers    │
                              └───────────────┬─────────────────┘
                                              │
                              ┌───────────────▼─────────────────┐
                              │      INTERPOLATION BUFFER       │
                              │  20-frame look-ahead            │
                              │  Outlier rejection (1-3 spikes) │
                              │  Linear + arc gap filling       │
                              └───────────────┬─────────────────┘
                                              │
                              ┌───────────────▼─────────────────┐
                              │        VIDEO OUTPUT             │
                              │   Annotated MP4 @ original fps  │
                              └─────────────────────────────────┘
```

### Component Summary

| Component | Implementation | Details |
|---|---|---|
| Detection | YOLO11s (fine-tuned) + COCO pretrained | Fine-tuned for players/hoop/ref, COCO "sports ball" (class 32) as conditional fallback |
| Tracking | ByteTrack | Custom config — 4s track buffer, low thresholds for ball |
| Pose Estimation | YOLO11n-pose | 17-point COCO skeleton, every 3rd processed frame (cached) |
| Jersey OCR | EasyOCR | Adaptive frequency (every 30 frames unconfirmed, 90 confirmed), majority voting |
| Ball Recovery | SAHI (COCO model) | Conditional — only when primary + COCO both miss, targeted crop + full-frame fallback |
| Ball Validation | BallValidator | 5-layer filter: size, aspect ratio, court region, teleport rejection, backboard constraint |
| Ball Interpolation | BallInterpolator | 20-frame look-ahead buffer, relative outlier rejection (1-3 frame spikes), linear + parabolic gap filling |
| Ball Tracker | BallTracker | Velocity-adaptive search zone with direction preference (5-frame history, 80-500px radius) |
| Ball Re-ID | BallReID | Keeps ball track ID stable across occlusions (4s buffer, 500px radius) |
| Player Re-ID | TemporalReIDBuffer | Jersey-based matching (ignores distance) + proximity fallback (300px), 10s memory |
| Velocity Tracker | VelocityTracker | Preserves player IDs through overlapping bounding boxes |
| Team Classification | TeamClassifier | K-means (k=2) on HSV jersey color histograms, auto-detects home vs away |

### Detection Classes

| ID | Class | Color (BGR) | Notes |
|---|---|---|---|
| 0 | basketball | Red (0,0,255) | Oversampled 8x during training, dual-model detection (fine-tuned + COCO) |
| 1 | hoop | White (255,255,255) | Cross-hair marker at center |
| 2 | player | Team color (auto) | Pose skeleton + jersey OCR, color-coded by team after classification |
| 3 | referee | Dark Blue (180,60,20) | No pose estimation |

---

## Project Structure

```
uvambb_cv/
├── main.py                          # Full pipeline — fine-tuning + inference
├── bytetrack_players.yaml           # ByteTrack tracker config
├── data/
│   ├── custom_annotations/          # Roboflow export (YOLOv11 format)
│   │   ├── data.yaml                # Dataset config (nc=4, class names)
│   │   ├── train/                   # ~391 training images + labels
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── valid/                   # 97 validation images + labels (80/20 split)
│   │       ├── images/
│   │       └── labels/
│   ├── frames/                      # Extracted frames from game 1
│   ├── frames2/                     # Extracted frames from game 2 (first 20 min, every 15th frame)
│   └── game_01.mp4                  # Game footage
├── runs/detect/train/weights/       # Fine-tuned model weights
│   ├── best.pt                      # Best val mAP checkpoint
│   ├── best.engine                  # TensorRT engine (after --export-tensorrt)
│   └── last.pt                      # Latest epoch checkpoint
└── output/
    └── annotated.mp4                # Annotated output video
```

---

## Installation & Setup

**Requirements:** Python 3.10+, CUDA GPU recommended

```bash
conda activate personal

pip install ultralytics        # YOLO11 detection + pose
pip install opencv-python      # Video I/O + drawing
pip install numpy torch        # Core dependencies
pip install easyocr            # Jersey number recognition
pip install sahi               # Sliced inference for small ball detection
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

### Export to TensorRT (2-3x faster inference)

```bash
python main.py --export-tensorrt [--weights path/to/best.pt]
```

One-time export — engines are GPU-specific and auto-loaded on subsequent runs.

### Disable SAHI (faster, less ball detection)

```bash
python main.py --video data/game_01.mp4 --no-sahi
```

### Resume training from checkpoint

```python
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True)
```

---

## Training Configuration

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

The pipeline is optimized to process a full 2-hour game (432K frames at 60fps) within ~24 hours:

```
  BEFORE                               AFTER
  ~~~~~~                               ~~~~~
  ~2.0 s/frame                         ~0.2 s/frame  (estimated)
  ~10 days per game                    ~24 hours per game
  3-4 model calls/frame (unconditional) 1-2 model calls/frame (conditional)
  imgsz=1920 (upscaling 720p video)    imgsz=1280 (native resolution)
```

| Optimization | Category | Impact | Accuracy Cost |
|---|---|---|---|
| Reduce imgsz 1920 → 1280 | Resolution | ~35% faster per call | None (video is 720p) |
| Conditional COCO fallback | Inference skip | Skips ~50%+ frames | None |
| Conditional SAHI | Inference skip | Skips most frames | None |
| `@torch.inference_mode()` | Memory | Reduces overhead | None |
| TensorRT export (`.engine`) | Compilation | 2-3x per model | None |
| Frame skipping (every 2nd) | Temporal | 2x total | Negligible at 60fps |
| Pose every 3rd processed frame | Inference skip | ~67% less pose | Negligible |
| YOLO11n for COCO fallback | Model size | ~2x when it runs | Minimal |
| Reduced OCR frequency | CPU savings | Less EasyOCR | Slightly slower convergence |
| Deque for interpolation buffer | Data structure | O(1) vs O(n) pop | None |

---

## Ball Detection Pipeline

The basketball is the hardest object to detect (small, fast-moving, similar color to shoes and skin). The pipeline uses a cascading approach — each stage only fires when the previous one fails:

```
  Fine-tuned YOLO11s          ← runs every processed frame
        │
        ├── ball found? ──YES──→ skip to BallValidator
        │
        NO
        │
  COCO "sports ball" (11n)   ← only when fine-tuned misses
        │
        ├── ball found? ──YES──→ skip to BallValidator
        │
        NO
        │
  SAHI sliced inference      ← only when both miss
        │
        ├── targeted crop (if last position known)
        └── full-frame scan (if ball lost >60 frames, every 5th frame)
```

### Detection Sources
1. **Fine-tuned model** — detects ball at conf >= 0.25 (BallValidator catches false positives)
2. **COCO pretrained model** — `yolo11n.pt` "sports ball" class (32) at conf >= 0.20, only when fine-tuned model finds no ball
3. **SAHI targeted** — 128px sliced inference around last known position, only when both primary detectors miss
4. **SAHI full-frame** — when ball lost >60 frames, scans entire frame every 5th frame to re-acquire

### Validation (BallValidator)
Rejects false positives with 5 checks in order:
1. **Size sanity** — 50–2500px² at 1080p (1.5x allowed near hoops)
2. **Aspect ratio** — max 1.8:1 — basketball is round/square, shoes are elongated
3. **Court region** — ball must be within court bounds (with 5% padding)
4. **Teleport rejection** — max 450px/frame movement, scaled by gap duration
5. **Backboard constraint** — if ball was near hoop recently, reject detections above and behind the hoop (crowd area). Allows airballs to pass through horizontally at court level.

### Tracking (BallTracker)
Velocity-adaptive search zone with direction preference:
- 5-frame velocity history averaged for stable estimates
- Base search radius: 80px (stationary) scaling up to 500px (fast passes)
- 1.5x bonus multiplier for detections in the direction of travel
- Falls back to largest detection only when nothing in search zone

### Temporal Interpolation (BallInterpolator)
Post-game look-ahead analysis with 20-frame sliding buffer:

**Outlier Rejection** — removes 1–3 frame detection spikes using a *relative* test:
- Compare spike's jump distance to the neighbor-to-neighbor distance (how far the real ball moved)
- If the spike jumps 3x+ further than the neighbors moved apart (+ 20px margin), it's a false positive
- This catches nearby false positives (shirts, shoes 50-80px away) that absolute thresholds miss
- Example: ball stationary (neighbors 5px apart), detection jumps 60px to a shirt for 1 frame → rejected because 60 > 5×3+20

**Gap Filling** — interpolates up to 8 consecutive missing frames:
- Linear interpolation for short gaps (≤3 frames)
- Parabolic arc interpolation for longer gaps (>3 frames) — models gravity for shots/lobs
- Distance sanity check prevents interpolation between different objects

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

---

## Team

- Jackson Kennedy
- Nathan Wan
- Nathan Todd

---

*Last Updated: March 2026*
