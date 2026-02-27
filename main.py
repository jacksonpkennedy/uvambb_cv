"""
UVA Men's Basketball CV Pipeline — Detection + Tracking + Pose

Usage:
  Fine-tune : python main.py --finetune [--epochs 250] [--batch 16]
  Inference : python main.py [--video data/game_01.mp4] [--weights path/to/best.pt]

Dataset (data/custom_annotations/data.yaml):
  nc: 4
  names: [basketball, hoop, players, referee]   # IDs must match CLS_* constants below
"""

import argparse
import math
import re
import shutil
import time
from collections import deque
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return 'cuda:0' when a GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU detected: {name}  ({vram:.1f} GB VRAM)  → using cuda:0")
        return "cuda:0"
    print("WARNING: CUDA not available — falling back to CPU (will be slow).")
    return "cpu"


# ---------------------------------------------------------------------------
# Config — class IDs must match names order in data/custom_annotations/data.yaml
# ---------------------------------------------------------------------------

CLS_BALL   = 0   # "basketball"
CLS_HOOP   = 1   # "hoop"
CLS_PLAYER = 2   # "players"
CLS_REF    = 3   # "referee"

LABEL_PLAYER = "player"        # internal tag for TrackMemory / pose association

C_PLAYER = (0,   255,   0)     # BGR green
C_BALL   = (0,     0, 255)     # red
C_HOOP   = (255, 255, 255)     # white
C_REF    = (180,  60,  20)     # dark blue

REID_BUFFER_FRAMES = 180    # frames to remember lost player tracks (~3s at 60fps)
REID_DIST_THRESH   = 300    # max pixel distance to re-associate a lost player track

BALL_REID_BUFFER   = 240    # frames to remember lost ball track (~4s at 60fps)
BALL_REID_DIST     = 500    # ball moves fast — allow wider re-association radius

IOU_CRASH_THRESH   = 0.40
VEL_HISTORY_LEN    = 15
VEL_EXIT_THRESH    = 80

# SAHI — targeted inference around ball's last known position
SAHI_CROP_PAD      = 250    # pixels to pad around last known ball center for SAHI crop
SAHI_CONF_THRESH   = 0.10   # lower conf OK — we're zooming into a small region
SAHI_SLICE_SIZE    = 160    # slice size within the cropped region
SAHI_OVERLAP_RATIO = 0.25   # overlap between slices within the crop
SAHI_MAX_LOST      = 60     # max frames since last ball sighting to run targeted SAHI

# Jersey OCR
OCR_INTERVAL       = 30     # run OCR every N frames (performance vs accuracy)
OCR_CONFIRM_COUNT  = 2      # need this many consistent reads to lock a jersey number


_G = (0, 255, 0)
SKELETON_EDGES = [
    (0,  1, _G), (0,  2, _G), (1,  3, _G), (2,  4, _G),
    (5,  6, _G), (5,  7, _G), (7,  9, _G), (6,  8, _G),
    (8, 10, _G), (5, 11, _G), (6, 12, _G), (11, 12, _G),
    (11,13, _G), (13, 15, _G), (12, 14, _G), (14, 16, _G),
]
KP_COLORS      = [_G] * 17
KP_CONF_THRESH = 0.30


# ---------------------------------------------------------------------------
# Track Memory — persist IDs across brief occlusions
# ---------------------------------------------------------------------------

class TrackMemory:
    def __init__(self, max_age: int):
        self.max_age   = max_age
        self._tracks: dict = {}

    def update(self, frame_idx: int, detections: list):
        seen_ids = {d["tid"] for d in detections}

        for d in detections:
            self._tracks[d["tid"]] = {
                "box":   d["box"],
                "label": d["label"],
                "frame": frame_idx,
            }

        ghost_dets = []
        dead = []
        for tid, info in self._tracks.items():
            if tid in seen_ids:
                continue
            age = frame_idx - info["frame"]
            if age <= self.max_age:
                ghost_dets.append({
                    "tid":   tid,
                    "box":   info["box"],
                    "label": info["label"],
                    "ghost": True,
                    "age":   age,
                })
            else:
                dead.append(tid)

        for tid in dead:
            del self._tracks[tid]

        return detections, ghost_dets


# ---------------------------------------------------------------------------
# Temporal Re-ID Buffer
# ---------------------------------------------------------------------------

class TemporalReIDBuffer:
    def __init__(self):
        self._lost:     dict = {}
        self._id_remap: dict = {}

    def update(self, frame_idx: int, raw_dets: list) -> list:
        expired = [
            tid for tid, (fl, _, _) in self._lost.items()
            if frame_idx - fl > REID_BUFFER_FRAMES
        ]
        for tid in expired:
            del self._lost[tid]

        for d in raw_dets:
            tid = d["tid"]
            if tid in self._id_remap:
                d["tid"] = self._id_remap[tid]
                continue
            cx = (d["box"][0] + d["box"][2]) / 2.0
            cy = (d["box"][1] + d["box"][3]) / 2.0
            best_old_tid = None
            best_dist    = REID_DIST_THRESH
            for old_tid, (_, ox, oy) in self._lost.items():
                dist = math.hypot(cx - ox, cy - oy)
                if dist < best_dist:
                    best_dist    = dist
                    best_old_tid = old_tid
            if best_old_tid is not None:
                self._id_remap[tid] = best_old_tid
                del self._lost[best_old_tid]
                d["tid"] = best_old_tid

        return raw_dets

    def register_lost(self, frame_idx: int, ghost_dets: list):
        for d in ghost_dets:
            if d.get("age", 0) == 1:
                cx = (d["box"][0] + d["box"][2]) / 2.0
                cy = (d["box"][1] + d["box"][3]) / 2.0
                self._lost[d["tid"]] = (frame_idx, cx, cy)


# ---------------------------------------------------------------------------
# Velocity Tracker — overlap/eclipse ID preservation
# ---------------------------------------------------------------------------

def bbox_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class VelocityTracker:
    def __init__(self):
        self._positions:       dict = {}
        self._overlapping:     dict = {}
        self._vel_at_overlap:  dict = {}

    def update(self, frame_idx: int, real_dets: list) -> dict:
        for d in real_dets:
            tid = d["tid"]
            cx  = (d["box"][0] + d["box"][2]) / 2.0
            cy  = (d["box"][1] + d["box"][3]) / 2.0
            if tid not in self._positions:
                self._positions[tid] = deque(maxlen=VEL_HISTORY_LEN)
            self._positions[tid].append((frame_idx, cx, cy))

        current_overlapping: set = set()
        n = len(real_dets)
        for i in range(n):
            for j in range(i + 1, n):
                iou = bbox_iou(real_dets[i]["box"], real_dets[j]["box"])
                if iou >= IOU_CRASH_THRESH:
                    key = frozenset({real_dets[i]["tid"], real_dets[j]["tid"]})
                    current_overlapping.add(key)
                    if key not in self._overlapping:
                        self._overlapping[key] = frame_idx
                        for d in [real_dets[i], real_dets[j]]:
                            vx, vy = self._compute_velocity(d["tid"])
                            self._vel_at_overlap[d["tid"]] = (vx, vy)

        remap: dict = {}
        current_tids = {d["tid"] for d in real_dets}
        ended_keys = [k for k in self._overlapping if k not in current_overlapping]
        for key in ended_keys:
            del self._overlapping[key]
            for tid in key:
                if tid in current_tids or tid not in self._vel_at_overlap:
                    self._vel_at_overlap.pop(tid, None)
                    continue
                vx, vy = self._vel_at_overlap.pop(tid)
                hist = self._positions.get(tid)
                if not hist:
                    continue
                _, last_cx, last_cy = hist[-1]
                pred_cx = last_cx + vx
                pred_cy = last_cy + vy
                for d in real_dets:
                    dcx = (d["box"][0] + d["box"][2]) / 2.0
                    dcy = (d["box"][1] + d["box"][3]) / 2.0
                    if math.hypot(dcx - pred_cx, dcy - pred_cy) < VEL_EXIT_THRESH:
                        remap[d["tid"]] = tid
                        break

        return remap

    def _compute_velocity(self, tid: int):
        hist = self._positions.get(tid)
        if hist is None or len(hist) < 2:
            return 0.0, 0.0
        window = list(hist)[-5:]
        fi0, x0, y0 = window[0]
        fi1, x1, y1 = window[-1]
        dt = fi1 - fi0
        if dt == 0:
            return 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt


# ---------------------------------------------------------------------------
# Jersey Number OCR — identify players by jersey number
# ---------------------------------------------------------------------------

class JerseyOCR:
    """Crop the torso region of player bboxes, run EasyOCR, and cache jersey numbers."""

    def __init__(self):
        self._reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        self._confirmed: dict[int, str]       = {}   # tid → locked jersey number
        self._candidates: dict[int, dict]     = {}   # tid → {number: count, ...}

    def get_jersey(self, tid: int) -> str | None:
        return self._confirmed.get(tid)

    def scan(self, frame: np.ndarray, detections: list):
        """Run OCR on player crops. Call every OCR_INTERVAL frames."""
        for d in detections:
            tid = d["tid"]
            if tid in self._confirmed:
                continue
            box = d["box"]
            # Crop upper 55% of bbox (torso/jersey area)
            x1, y1, x2, y2 = box
            h = y2 - y1
            crop_y2 = y1 + int(h * 0.55)
            # Slight horizontal padding inward to avoid arms
            w = x2 - x1
            crop_x1 = x1 + int(w * 0.15)
            crop_x2 = x2 - int(w * 0.15)
            if crop_x2 <= crop_x1 or crop_y2 <= y1:
                continue
            crop = frame[max(0, y1):crop_y2, max(0, crop_x1):crop_x2]
            if crop.size == 0:
                continue

            results = self._reader.readtext(crop, allowlist="0123456789",
                                            paragraph=False, min_size=10)
            for (_, text, conf) in results:
                text = text.strip()
                # Filter to valid jersey numbers (1-99)
                if not re.match(r"^\d{1,2}$", text):
                    continue
                num = int(text)
                if num < 0 or num > 99:
                    continue
                jersey = str(num)
                if tid not in self._candidates:
                    self._candidates[tid] = {}
                self._candidates[tid][jersey] = self._candidates[tid].get(jersey, 0) + 1
                # Lock once we see the same number enough times
                if self._candidates[tid][jersey] >= OCR_CONFIRM_COUNT:
                    self._confirmed[tid] = jersey
                    self._candidates.pop(tid, None)
                    break

    def transfer_id(self, old_tid: int, new_tid: int):
        """When Re-ID remaps a track, carry the jersey number over."""
        if old_tid in self._confirmed:
            self._confirmed[new_tid] = self._confirmed[old_tid]


# ---------------------------------------------------------------------------
# Ball Tracker — interpolation + trajectory prediction for ball occlusions
# ---------------------------------------------------------------------------

class BallTracker:
    """Track ball detections — pick best detection per frame, no interpolation.
    Stores last known position so SAHI can target that region."""

    def __init__(self):
        self._last_box: list | None = None
        self._last_seen_frame: int = -999

    @property
    def last_center(self) -> tuple | None:
        """Return (cx, cy) of last known ball position, or None."""
        if self._last_box is None:
            return None
        b = self._last_box
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @property
    def frames_since_seen(self) -> int:
        return self._frames_since

    def update(self, frame_idx: int, ball_dets: list) -> list:
        """Return only real ball detections (no predictions)."""
        self._frames_since = frame_idx - self._last_seen_frame
        if not ball_dets:
            return []
        # Pick largest detection (most likely the real ball, not noise)
        best = max(ball_dets, key=lambda d: (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1]))
        self._last_box = best["box"]
        self._last_seen_frame = frame_idx
        return [best]


class BallReID:
    """Keep the ball's track ID stable across occlusions.
    When ByteTrack assigns a new tid to the ball after a gap,
    remap it back to the original tid."""

    def __init__(self):
        self._canonical_tid: int | None = None   # the "real" ball track ID
        self._last_center: tuple | None = None
        self._last_frame: int = -999

    def update(self, frame_idx: int, ball_dets: list) -> list:
        if not ball_dets:
            return ball_dets

        for d in ball_dets:
            tid = d["tid"]
            if tid == -1:
                # SAHI detection without a tracker ID — assign canonical
                if self._canonical_tid is not None:
                    d["tid"] = self._canonical_tid
                continue

            cx = (d["box"][0] + d["box"][2]) / 2.0
            cy = (d["box"][1] + d["box"][3]) / 2.0

            if self._canonical_tid is None:
                # First ball detection ever
                self._canonical_tid = tid
            elif tid != self._canonical_tid:
                # ByteTrack assigned a new ID — check if it's the same ball
                gap = frame_idx - self._last_frame
                if gap <= BALL_REID_BUFFER and self._last_center is not None:
                    dist = math.hypot(cx - self._last_center[0],
                                      cy - self._last_center[1])
                    if dist < BALL_REID_DIST:
                        d["tid"] = self._canonical_tid
                    else:
                        # Too far — accept new track as the ball
                        self._canonical_tid = tid
                else:
                    # Too long since last seen — accept new ID
                    self._canonical_tid = tid

            self._last_center = (cx, cy)
            self._last_frame = frame_idx

        return ball_dets


# ---------------------------------------------------------------------------
# SAHI — targeted inference around ball's last known position
# ---------------------------------------------------------------------------

def run_sahi_ball_targeted(
    sahi_model: AutoDetectionModel,
    frame: np.ndarray,
    center: tuple,
    existing_ball_dets: list,
) -> list:
    """Run SAHI on a small crop around the ball's last known position.
    Much faster than full-frame slicing — only processes the region
    where the ball is likely to be."""
    cx, cy = center
    h, w = frame.shape[:2]

    # Crop region around last known ball center
    x1 = max(0, int(cx - SAHI_CROP_PAD))
    y1 = max(0, int(cy - SAHI_CROP_PAD))
    x2 = min(w, int(cx + SAHI_CROP_PAD))
    y2 = min(h, int(cy + SAHI_CROP_PAD))

    if x2 - x1 < 50 or y2 - y1 < 50:
        return []

    crop = frame[y1:y2, x1:x2]

    result = get_sliced_prediction(
        image=crop,
        detection_model=sahi_model,
        slice_height=SAHI_SLICE_SIZE,
        slice_width=SAHI_SLICE_SIZE,
        overlap_height_ratio=SAHI_OVERLAP_RATIO,
        overlap_width_ratio=SAHI_OVERLAP_RATIO,
        postprocess_type="NMS",
        postprocess_match_threshold=0.40,
        verbose=0,
    )

    sahi_balls: list = []
    for pred in result.object_prediction_list:
        if int(pred.category.id) != CLS_BALL:
            continue
        if pred.score.value < SAHI_CONF_THRESH:
            continue
        bb = pred.bbox
        # Remap crop coordinates back to full frame
        box = [int(bb.minx) + x1, int(bb.miny) + y1,
               int(bb.maxx) + x1, int(bb.maxy) + y1]
        # Skip if it largely overlaps an existing ball detection
        dup = False
        for ed in existing_ball_dets:
            if bbox_iou(box, ed["box"]) > 0.30:
                dup = True
                break
        if not dup:
            sahi_balls.append({"tid": -1, "box": box, "cls": CLS_BALL})

    return sahi_balls


# ---------------------------------------------------------------------------
# Court ROI — polygon check
# ---------------------------------------------------------------------------

def build_court_roi(frame_w: int, frame_h: int) -> np.ndarray:
    pts = np.array([
        [frame_w * 0.13, frame_h * 0.88],
        [frame_w * 0.87, frame_h * 0.88],
        [frame_w * 0.75, frame_h * 0.19],
        [frame_w * 0.25, frame_h * 0.19],
    ], dtype=np.float32)
    return pts.reshape((-1, 1, 2)).astype(np.int32)


def in_court_roi(roi_poly: np.ndarray, box: list) -> bool:
    foot_x = (box[0] + box[2]) / 2.0
    foot_y = float(box[3])
    return cv2.pointPolygonTest(roi_poly, (foot_x, foot_y), False) >= 0


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _contrast_text(bg_color):
    """Return black or white text depending on background brightness."""
    brightness = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    return (0, 0, 0) if brightness > 140 else (255, 255, 255)


def _label_bg(frame, x: int, y: int, text: str, bg_color, scale=0.48, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = 3
    cv2.rectangle(frame, (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad), bg_color, -1)
    txt_color = _contrast_text(bg_color)
    cv2.putText(frame, text, (x, y), font, scale, txt_color, thick, cv2.LINE_AA)


def draw_player(frame, box, label: str, color=None):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw_color = color if color is not None else C_PLAYER
    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 3)
    _label_bg(frame, x1, y1 - 3, label, draw_color, scale=0.55, thick=2)


def draw_skeleton(frame: np.ndarray, kps_xy: np.ndarray, kps_conf: np.ndarray):
    for (i, j, color) in SKELETON_EDGES:
        if kps_conf[i] >= KP_CONF_THRESH and kps_conf[j] >= KP_CONF_THRESH:
            x1, y1 = int(kps_xy[i][0]), int(kps_xy[i][1])
            x2, y2 = int(kps_xy[j][0]), int(kps_xy[j][1])
            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    for k in range(min(len(kps_xy), 17)):
        if kps_conf[k] >= KP_CONF_THRESH:
            x, y = int(kps_xy[k][0]), int(kps_xy[k][1])
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 4, KP_COLORS[k], -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 4, (0, 0, 0),    1,  cv2.LINE_AA)


def associate_poses(real_dets: list, ref_boxes: list, pose_result) -> dict:
    """Match pose skeletons to player detections only.
    Excludes any pose that overlaps more with a referee than with any player."""
    associations = {}
    if (pose_result is None
            or pose_result.keypoints is None
            or pose_result.boxes is None
            or len(pose_result.boxes.xyxy) == 0):
        return associations

    p_boxes  = pose_result.boxes.xyxy.cpu().numpy()
    kps_xy   = pose_result.keypoints.xy.cpu().numpy()
    kps_conf_raw = pose_result.keypoints.conf
    if kps_conf_raw is not None:
        kps_conf = kps_conf_raw.cpu().numpy()
    else:
        kps_conf = np.ones((len(kps_xy), 17), dtype=np.float32)

    track_candidates = [
        (d["box"], d["tid"])
        for d in real_dets
        if d["label"] == LABEL_PLAYER
    ]

    for i, p_box in enumerate(p_boxes):
        p_box_list = p_box.tolist()

        # Check if this pose overlaps more with a ref than any player
        best_ref_iou = max(
            (bbox_iou(p_box_list, rb) for rb in ref_boxes),
            default=0.0
        )

        best_iou = 0.12
        best_tid = None
        for (y_box, tid) in track_candidates:
            iou = bbox_iou(p_box_list, y_box)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        # Only assign if player overlap beats ref overlap
        if best_tid is not None and best_iou > best_ref_iou:
            associations[best_tid] = (kps_xy[i], kps_conf[i])

    return associations


def draw_ball(frame, box, label="BASKETBALL"):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BALL, 3, cv2.LINE_AA)
    _label_bg(frame, x1, y1 - 3, label, C_BALL)


def draw_hoop(frame, box):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_HOOP, 3, cv2.LINE_AA)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.drawMarker(frame, (cx, cy), C_HOOP, cv2.MARKER_CROSS, 14, 3, cv2.LINE_AA)
    _label_bg(frame, x1, y1 - 3, "HOOP", C_HOOP)


def draw_ref(frame, box, tid: int):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_REF, 3, cv2.LINE_AA)
    _label_bg(frame, x1, y1 - 3, f"REF#{tid}", C_REF)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def _oversample_class(data_dir: Path, cls_id: int, cls_name: str, factor: int = 3):
    """Duplicate images containing a specific class to balance the dataset.
    Copies both image and label files with a unique suffix.
    Skips if oversampled copies already exist."""
    train_imgs  = data_dir / "train" / "images"
    train_lbls  = data_dir / "train" / "labels"
    if not train_imgs.exists() or not train_lbls.exists():
        return 0

    copied = 0
    for lbl_path in train_lbls.glob("*.txt"):
        # Check if this label file contains the target class
        text = lbl_path.read_text()
        has_cls = any(
            line.strip().startswith(f"{cls_id} ")
            for line in text.splitlines()
            if line.strip()
        )
        if not has_cls:
            continue

        stem = lbl_path.stem
        # Skip files that are already oversampled copies
        if f"_os{cls_name}" in stem:
            continue

        # Find matching image (try common extensions)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = train_imgs / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        # Check if copies already exist from a previous run
        first_copy_lbl = train_lbls / f"{stem}_os{cls_name}_1.txt"
        if first_copy_lbl.exists():
            continue

        # Create N-1 extra copies (original + copies = factor total)
        for k in range(1, factor):
            new_stem = f"{stem}_os{cls_name}_{k}"
            shutil.copy2(img_path, train_imgs / f"{new_stem}{img_path.suffix}")
            shutil.copy2(lbl_path, train_lbls / f"{new_stem}.txt")
            copied += 1

    return copied


def finetune(epochs: int = 200, batch: int = 8, imgsz: int = 960):
    base_dir  = Path(__file__).resolve().parent
    data_yaml = str(base_dir / "data" / "custom_annotations" / "data.yaml")

    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}\n"
            "Export your Roboflow dataset (YOLO format) to data/custom_annotations/\n"
            "data.yaml must define: nc=4, names=[basketball, hoop, players, referee]"
        )

    # Oversample basketball class — it's small and rare in frames,
    # so the model under-learns it compared to players/refs.
    data_dir = base_dir / "data" / "custom_annotations"
    # basketball=159, hoop=596, player=8032, referee=2174 annotations
    # factor=8 → basketball appears 8x → ~1,272 annotations (closer to hoop/ref)
    n = _oversample_class(data_dir, CLS_BALL, "ball", factor=8)
    if n:
        print(f"Oversampled basketball: +{n} copies added to training set")
    else:
        print("Basketball oversampling: already done or no ball annotations found")

    device = get_device()
    print(f"Fine-tuning YOLO11s  |  epochs={epochs}  batch={batch}  "
          f"imgsz={imgsz}  nbs=64 (grad accum)  device={device}")

    model = YOLO("yolo11s.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(base_dir / "runs" / "detect"),
        name="train",
        exist_ok=True,
        pretrained=True,
        # Gradient accumulation — simulate effective batch of 64
        # actual batch=8 × 8 accumulation steps = 64 effective batch
        # smooths gradients without extra VRAM
        nbs=64,
        # Learning rate
        lr0=0.001,
        lrf=0.01,              # final LR = lr0 * lrf
        warmup_epochs=5,       # ~1,200 images — converges faster, shorter warmup OK
        # Regularization
        dropout=0.10,          # less dropout needed with more data
        weight_decay=0.0005,   # L2 regularization
        # Augmentation — moderate, ~1,200 train images is a reasonable dataset
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=5.0,           # slight rotation — court camera is fairly stable
        translate=0.15,
        scale=0.5,             # moderate scale variation
        shear=1.0,
        perspective=0.0003,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,            # combine 4 images — still useful for occlusion
        mixup=0.15,            # lighter blending — enough data for real examples
        copy_paste=0.2,        # paste objects for synthetic occlusion
        erasing=0.3,           # random erase for occlusion robustness
        crop_fraction=1.0,
        # Early stopping — 242 val images = smooth metrics, can stop sooner
        patience=30,
    )

    best_pt = base_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
    print(f"\nFine-tuning complete.  Best weights: {best_pt}")
    print(f"Run inference: python main.py --video data/game_01.mp4")
    return str(best_pt)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(video_path: str, out_dir: str, weights: str | None = None,
        use_sahi: bool = True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}  |  {frame_w}x{frame_h}  "
          f"|  {fps:.1f} fps  |  {total_frames} frames")

    device = get_device()
    if device.startswith("cuda"):
        infer_imgsz = 1920
        use_half    = True
    else:
        infer_imgsz = 640
        use_half    = False
        print(f"NOTE: Running on CPU with imgsz={infer_imgsz}.")

    # Resolve weights — default to last fine-tune run
    base_dir     = Path(__file__).resolve().parent
    weights_path = weights or str(
        base_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
    )
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"No fine-tuned model found at: {weights_path}\n"
            "Run first: python main.py --finetune"
        )

    print(f"Loading detection model: {Path(weights_path).name} ...")
    model = YOLO(weights_path)

    print("Loading YOLO11n-pose (skeleton estimation) ...")
    pose_model = YOLO("yolo11n-pose.pt")

    # SAHI — sliced inference for small ball detection
    sahi_model = None
    if use_sahi:
        print("Loading SAHI sliced-inference model (ball detection) ...")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",          # compatible with YOLO11
            model_path=weights_path,
            confidence_threshold=SAHI_CONF_THRESH,
            device=device,
        )

    roi_poly      = build_court_roi(frame_w, frame_h)
    ghost_max_age = int(fps * 1.5)
    memory        = TrackMemory(max_age=ghost_max_age)
    reid_buffer   = TemporalReIDBuffer()
    vel_tracker   = VelocityTracker()
    tracker_cfg   = str(base_dir / "bytetrack_players.yaml")

    print("Loading EasyOCR (jersey number reader) ...")
    jersey_ocr    = JerseyOCR()
    ball_tracker  = BallTracker()
    ball_reid     = BallReID()

    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        str(Path(out_dir) / "annotated.mp4"), fourcc, fps, (frame_w, frame_h)
    )

    frame_idx = 0
    print("Processing frames ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_frame_start = time.perf_counter()

        # ---- 1. Detection & tracking -----------------------------------------
        raw_dets       = []
        ball_dets      = []
        hoop_dets      = []
        ref_dets       = []

        det_res = model.track(
            frame, persist=True, tracker=tracker_cfg,
            conf=0.15, iou=0.45,
            imgsz=infer_imgsz, device=device, half=use_half,
            verbose=False,
        )[0]

        if det_res.boxes is not None and det_res.boxes.id is not None:
            for box, tid, cls_id, conf in zip(
                    det_res.boxes.xyxy, det_res.boxes.id,
                    det_res.boxes.cls, det_res.boxes.conf):
                box_list = list(map(int, box.tolist()))
                tid      = int(tid)
                cls_id   = int(cls_id)
                conf_val = float(conf)

                if cls_id == CLS_PLAYER:
                    if conf_val < 0.30:
                        continue
                    if (box_list[3] - box_list[1]) < frame_h * 0.08:
                        continue
                    if not in_court_roi(roi_poly, box_list):
                        continue
                    raw_dets.append({"tid": tid, "box": box_list, "label": LABEL_PLAYER})
                elif cls_id == CLS_BALL:
                    ball_dets.append({"tid": tid, "box": box_list, "cls": CLS_BALL})
                elif cls_id == CLS_HOOP:
                    if conf_val >= 0.30:
                        hoop_dets.append({"tid": tid, "box": box_list, "cls": CLS_HOOP})
                elif cls_id == CLS_REF:
                    if conf_val >= 0.30:
                        ref_dets.append({"tid": tid, "box": box_list, "cls": CLS_REF})

        # ---- 1b. SAHI targeted detection around last known ball position ------
        if sahi_model is not None and not ball_dets:
            center = ball_tracker.last_center
            if center is not None and ball_tracker.frames_since_seen < SAHI_MAX_LOST:
                sahi_balls = run_sahi_ball_targeted(
                    sahi_model, frame, center, ball_dets)
                ball_dets.extend(sahi_balls)

        # ---- 2. Re-ID + memory + velocity ------------------------------------
        raw_dets              = reid_buffer.update(frame_idx, raw_dets)
        real_dets, ghost_dets = memory.update(frame_idx, raw_dets)
        reid_buffer.register_lost(frame_idx, ghost_dets)

        vel_remap = vel_tracker.update(frame_idx, real_dets)
        if vel_remap:
            for d in real_dets:
                if d["tid"] in vel_remap:
                    old_tid = d["tid"]
                    d["tid"] = vel_remap[old_tid]
                    jersey_ocr.transfer_id(vel_remap[old_tid], d["tid"])

        # ---- 3. Jersey OCR (every N frames) ----------------------------------
        if frame_idx % OCR_INTERVAL == 0:
            jersey_ocr.scan(frame, real_dets)

        # ---- 4. Pose estimation (players only, exclude refs) ------------------
        ref_boxes = [d["box"] for d in ref_dets]
        pose_res = pose_model.predict(
            frame, verbose=False, device=device,
            conf=0.20, imgsz=infer_imgsz, half=use_half,
        )[0]
        pose_map = associate_poses(real_dets, ref_boxes, pose_res)

        # ---- 5. Ball tracking + Re-ID (keep ball ID stable across occlusions) -
        ball_dets    = ball_reid.update(frame_idx, ball_dets)
        ball_to_draw = ball_tracker.update(frame_idx, ball_dets)

        # ---- 6. Draw ball / hoop / ref ---------------------------------------
        for det in ball_to_draw:
            draw_ball(frame, det["box"])

        for det in hoop_dets:
            draw_hoop(frame, det["box"])

        for det in ref_dets:
            draw_ref(frame, det["box"], det["tid"])

        # ---- 7. Draw real players + skeletons --------------------------------
        for d in real_dets:
            tid = d["tid"]
            jersey = jersey_ocr.get_jersey(tid)
            label = f"#{jersey}" if jersey else f"#{tid}"
            draw_player(frame, d["box"], label)
            if tid in pose_map:
                draw_skeleton(frame, *pose_map[tid])

        out_video.write(frame)
        frame_idx += 1

        t_frame = time.perf_counter() - t_frame_start
        if frame_idx <= 5 or frame_idx % 30 == 0:
            n_jerseys = sum(1 for d in real_dets if jersey_ocr.get_jersey(d["tid"]))
            print(f"  frame {frame_idx}/{total_frames}  "
                  f"| {t_frame:.2f}s/frame  "
                  f"| players={len(real_dets)}  "
                  f"| jerseys={n_jerseys}  "
                  f"| poses={len(pose_map)}  "
                  f"| ball={'YES' if ball_to_draw else 'no'}")

    cap.release()
    out_video.release()
    print(f"\nDone — {frame_idx} frames processed.")
    print(f"  Annotated video : {Path(out_dir) / 'annotated.mp4'}")
    print(f"  Jersey numbers found: {dict(jersey_ocr._confirmed)}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UVA MBB CV Pipeline")
    parser.add_argument("--video",    default="data/game_02test.mp4")
    parser.add_argument("--out",      default="output")
    parser.add_argument("--weights",  default=None,
                        help="Fine-tuned weights path "
                             "(default: runs/detect/train/weights/best.pt)")
    parser.add_argument("--finetune", action="store_true",
                        help="Train on data/custom_annotations/ then run inference")
    parser.add_argument("--epochs",   type=int, default=200)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--no-sahi",  action="store_true",
                        help="Disable SAHI sliced inference for ball detection")
    args = parser.parse_args()

    if args.finetune:
        best_pt = finetune(epochs=args.epochs, batch=args.batch)
        run(args.video, args.out, weights=best_pt, use_sahi=not args.no_sahi)
    else:
        run(args.video, args.out, weights=args.weights, use_sahi=not args.no_sahi)
