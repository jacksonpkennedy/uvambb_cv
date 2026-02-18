# UVA Men's Basketball Computer Vision Analysis

## Project Overview

This project builds a full basketball analysis pipeline on top of YOLOE-26 to automatically convert raw game footage into actionable play data. The system moves through four phases — detection, spatial mapping, event recognition, and play classification — turning raw pixels into a structured play log and box score.

**Team Members:**
- Jackson Kennedy
- Nathan Wan
- Nathan Todd

---

## Pipeline Architecture

### Phase 1 — Detection & Tracking (Vision Layer)

Converts raw video frames into timestamped x,y coordinate streams for every tracked object.

| Component | Choice | Rationale |
|---|---|---|
| Base Model | YOLOE-26 (open-vocabulary) | NMS-free inference, STAL for small-object (ball) detection, up to 43% faster on CPU than prior versions |
| Text Prompts | `"basketball"`, `"player"`, `"hoop"`, `"referee"` | Open-vocabulary head — no custom class retraining needed |
| Tracker | ByteTrack or BoT-SORT | High-FPS re-association prevents ball ID loss during fast passes |
| Pose Head | YOLOE-26-Pose (joint keypoints) | Distinguishes shots from passes via elbow/knee extension angles |

**Output:** Per-frame bounding boxes + track IDs + 17-keypoint skeletons for all players and the ball.

---

### Phase 2 — Spatial Mapping (Homography)

Camera perspective distorts all pixel coordinates. This phase warps them onto a canonical 2D top-down court.

**Steps:**

1. Run a secondary YOLOE-26-Pose model trained on court keypoints (baseline corners, sideline corners, 3-point arc endpoints).
2. Solve for a 3×3 Homography Matrix **H** using the detected court keypoints and their known real-world positions.
3. Apply the transform to every tracked object every frame:

```
[x, y, 1]ᵀ = H · [u, v, 1]ᵀ
```

**Output:** True court coordinates (feet or meters) for every player and the ball — enabling real distances, speeds, and spacing calculations.

---

### Phase 3 — Event Recognition (Logic Layer)

Spatio-temporal triggers convert coordinate streams into discrete basketball events.

| Event | Trigger Logic |
|---|---|
| **Shot** | Player pose shows arm extension + ball trajectory follows a parabolic arc toward the hoop zone |
| **Rebound** | Ball enters hoop zone → trajectory reverses → ball enters a player bounding box |
| **Pass** | Ball leaves Player A bounding box at high velocity → enters Player B bounding box |
| **Screen / Pick** | Player stops within 1–2 ft of a teammate for ≥ 1 second while a third player drives |
| **Crash / Contact** | Two or more player boxes have IoU > 0.5 with high relative velocity |
| **Out of Bounds** | Ball court coordinates exit court boundary polygon |

**Output:** Timestamped event log — e.g., `[t=14.3s, PASS, player_id=5 → player_id=11]`.

---

### Phase 4 — Play Classification (Analyze Layer)

Sequences of events are fed into a temporal model to classify full offensive/defensive plays.

**Model options:**
- **State Machine** — deterministic rules for common sets (Pick and Roll, DHO, etc.)
- **Transformer** — learns play patterns from labeled sequence data

**Example rule (Pick and Roll):**
```
IF player[PG] holds position at top
AND player[C] moves toward player[PG] and stops (Screen event)
AND player[PG] drives baseline or presses mid-range
THEN classify → PICK AND ROLL
```

**Output:** Structured JSON play log + box score CSV.

```json
{
  "play_id": 42,
  "type": "Pick and Roll",
  "ball_handler": 5,
  "screener": 33,
  "outcome": "Layup (Made)",
  "timestamp_start": "Q2 08:14",
  "timestamp_end": "Q2 08:09"
}
```

---

## Technical Stack

| Layer | Tools |
|---|---|
| Detection & Pose | YOLOE-26, YOLOE-26-Pose (Ultralytics) |
| Tracking | ByteTrack / BoT-SORT |
| Homography | OpenCV `findHomography`, `perspectiveTransform` |
| Event Logic | Custom rule engine (Python) |
| Play Classification | PyTorch Transformer or `scikit-learn` state machine |
| Data Processing | NumPy, Pandas |
| Video I/O | OpenCV, FFmpeg |
| Visualization | OpenCV, Matplotlib |
| Language | Python 3.10+ |

---

## Project Structure

```
uvambb_cv/
├── README.md
├── src/
│   ├── detection/           # YOLOE-26 inference + ByteTrack integration
│   ├── homography/          # Court keypoint detection + H-matrix solving
│   ├── events/              # Spatio-temporal event triggers
│   ├── play_classifier/     # Temporal model for play classification
│   ├── statistics/          # Box score and play log export
│   └── utils/               # Shared helpers (video I/O, coordinate math)
├── models/                  # Model weights (YOLOE-26, court keypoint model)
├── data/                    # Sample game footage and labeled datasets
├── notebooks/               # Exploratory analysis and prototyping
└── tests/                   # Unit tests
```

---

## Installation & Setup

**Requirements:** Python 3.10+, CUDA GPU recommended, FFmpeg

```bash
pip install ultralytics          # YOLOE-26 + YOLOE-26-Pose
pip install opencv-python
pip install numpy pandas matplotlib scikit-learn
pip install torch torchvision
```

---

## Usage

```python
from src.detection import YOLOETracker
from src.homography import CourtMapper
from src.events import EventDetector
from src.play_classifier import PlayClassifier

# 1. Track players and ball
tracker = YOLOETracker(prompts=["basketball", "player", "hoop", "referee"])
tracks = tracker.process("game_video.mp4")

# 2. Map pixel coords to court coords
mapper = CourtMapper()
court_tracks = mapper.transform(tracks)

# 3. Detect events
detector = EventDetector()
events = detector.detect(court_tracks)

# 4. Classify plays and export
classifier = PlayClassifier()
plays = classifier.classify(events)
plays.to_json("play_log.json")
```

---

## Known Limitations

- Ball tracking degrades during pile-ups or off-screen moments
- Homography accuracy depends on camera angle — fixed broadcast angles work best
- Play classification quality scales with the size of the labeled play dataset

---

## Team

- Jackson Kennedy
- Nathan Wan
- Nathan Todd

---

*Last Updated: February 2026*
