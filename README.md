# UVA Men's Basketball Computer Vision Analysis

## Project Overview

A basketball analysis pipeline that detects, tracks, and annotates players, referees, the basketball, and the hoop from game footage using a fine-tuned YOLO11s model with ByteTrack multi-object tracking, pose estimation, jersey number OCR, and SAHI-enhanced ball detection.

**Team Members:**
- Jackson Kennedy
- Nathan Wan
- Nathan Todd

---

## Pipeline Architecture

| Component | Implementation | Details |
|---|---|---|
| Detection | YOLO11s (fine-tuned) + COCO pretrained | Fine-tuned model for players/hoop/ref, COCO "sports ball" (class 32) as secondary ball detector |
| Tracking | ByteTrack | Custom config tuned for basketball — 4s track buffer, low thresholds for ball |
| Pose Estimation | YOLO11n-pose | 17-point COCO skeleton, players only (excludes referees) |
| Jersey OCR | EasyOCR | Reads jersey numbers from torso crops, confirms after 2 consistent reads |
| Ball Recovery | SAHI (COCO model) | Targeted sliced inference around last known ball position every frame + full-frame scan to re-acquire |
| Ball Validation | BallValidator | 3-layer filter: size sanity, court region check, teleport rejection |
| Ball Re-ID | Custom | Keeps ball track ID stable across occlusions (4s buffer, 500px radius) |
| Player Re-ID | Custom | Re-associates lost player tracks (3s buffer, 300px radius) |
| Velocity Tracker | Custom | Preserves player IDs through overlapping bounding boxes |

### Detection Classes

| ID | Class | Color (BGR) | Notes |
|---|---|---|---|
| 0 | basketball | Red (0,0,255) | Oversampled 8x during training, dual-model detection (fine-tuned + COCO) |
| 1 | hoop | White (255,255,255) | Cross-hair marker at center |
| 2 | player | Green (0,255,0) | Pose skeleton + jersey OCR |
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
│   │   ├── train/                   # ~9,600 training images + labels (incl. oversampled)
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── val/                     # 364 validation images + labels
│   │       ├── images/
│   │       └── labels/
│   ├── frames/                      # Extracted frames from game footage
│   └── game_01.mp4                  # Game footage
├── runs/detect/train/weights/       # Fine-tuned model weights
│   ├── best.pt                      # Best val mAP checkpoint
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
python main.py --finetune [--epochs 200] [--batch 8]
```

This will:
1. Oversample basketball images 8x to balance class distribution
2. Train YOLO11s with transfer learning from COCO weights
3. Save best weights to `runs/detect/train/weights/best.pt`
4. Early stopping with patience=30 (stops if val mAP plateaus)

### Run inference

```bash
python main.py --video data/game_01.mp4 [--weights path/to/best.pt]
```

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
| Epochs | 200 (max) | Early stopping typically fires at 130-180 |
| Patience | 30 | 242 val images = smooth metrics |
| Learning rate | 0.001 → 0.00001 | Cosine decay (lrf=0.01) |
| Dropout | 0.10 | Regularization for ~1,200 image dataset |
| Basketball oversample | 8x | 159 → ~1,272 annotations |

### Augmentation

Mosaic, mixup (0.15), copy-paste (0.3), random erasing (0.4), moderate HSV jitter (h=0.02, s=0.5, v=0.4), rotation (10 deg), translation, scale, shear, perspective warp, horizontal flip. Augmentation kept moderate to preserve color distinction between basketball (orange) and skin tones.

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

## Ball Detection Pipeline

The basketball is the hardest object to detect (small, fast-moving, similar to bald heads). The pipeline uses a multi-stage approach:

1. **Fine-tuned model** — detects ball at conf ≥ 0.20 (low threshold, BallValidator catches false positives)
2. **COCO pretrained model** — runs `yolo11s.pt` with "sports ball" class (32) at conf ≥ 0.15 as a secondary detector
3. **SAHI targeted** — zoomed-in sliced inference (128px slices, 35% overlap) around last known ball position using COCO model every frame
4. **SAHI full-frame** — when ball is lost, scans entire frame every 10th frame to re-acquire
5. **BallValidator** — rejects false positives with size check (50-2500px² at 1080p), court region check, and teleport rejection (450px/frame max)
6. **BallReID** — keeps track ID stable across occlusions
7. **Visual** — bounding box with shrinking ellipse comet trail

---

## Known Limitations

- Ball tracking degrades during fast passes and heavy occlusion
- Jersey OCR accuracy depends on camera resolution and player orientation
- Court ROI polygon is hardcoded — needs adjustment for different camera angles
- Pose estimation can occasionally snap to nearby referees despite filtering

---

## Team

- Jackson Kennedy
- Nathan Wan
- Nathan Todd

---

*Last Updated: February 2026*
