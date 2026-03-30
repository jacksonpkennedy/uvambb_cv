"""
UVA Men's Basketball CV Pipeline — Detection + Tracking + Pose

Usage:
  Fine-tune : python main.py --finetune [--epochs 50] [--batch 16]
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

from tracknet import TrackNetInference


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

REID_BUFFER_FRAMES = 600    # frames to remember lost player tracks (~10s at 60fps)
REID_DIST_THRESH   = 300    # max pixel distance to re-associate a lost player track

BALL_REID_BUFFER   = 240    # frames to remember lost ball track (~4s at 60fps)
BALL_REID_DIST     = 500    # ball moves fast — allow wider re-association radius

IOU_CRASH_THRESH   = 0.40
VEL_HISTORY_LEN    = 15
VEL_EXIT_THRESH    = 80

# SAHI — targeted inference around ball's last known position
SAHI_CROP_PAD      = 400    # pixels to pad around last known ball center for SAHI crop
SAHI_CONF_THRESH   = 0.10   # lower conf OK — we're zooming into a small region
SAHI_SLICE_SIZE    = 128    # smaller slices = more zoom on small ball
SAHI_OVERLAP_RATIO = 0.35   # more overlap = less chance ball splits across slice edges
SAHI_MAX_LOST      = 60     # max frames since last ball sighting to run targeted SAHI

# TrackNet — heatmap-based ball detection using 3 consecutive frames
TRACKNET_WEIGHTS     = "runs/tracknet/weights/best.pt"  # path to trained TrackNet weights
TRACKNET_CONF_THRESH = 0.3                        # heatmap peak confidence threshold (lower for early training)
TRACKNET_BALL_RADIUS = 15                          # pixels, for bbox synthesis from center point

# COCO pretrained ball detection — used by SAHI fallback (class 32)
COCO_BALL_CLS      = 32     # "sports ball" in COCO
COCO_BALL_CONF     = 0.20   # confidence threshold for COCO ball detections

# Jersey OCR
OCR_INTERVAL       = 30     # run OCR every N frames (performance vs accuracy)
OCR_CONFIRM_COUNT  = 2      # need this many consistent reads to lock a jersey number

# Ball validation — reject false positives (bald heads, off-court objects)
BALL_COURT_X_PAD     = 0.05    # horizontal slack beyond court edges (fraction of frame_w)
BALL_MAX_BBOX_AREA   = 2500    # max ball bbox area in pixels (at 1080p) — rejects bald heads
BALL_MIN_BBOX_AREA   = 50      # min ball bbox area — rejects noise
BALL_MAX_TELEPORT_PX = 450     # max pixels ball can move per frame (at 1080p, 60fps)
BALL_TELEPORT_GRACE  = 10      # frames since last sighting before teleport check resets

# Ball interpolation — fill gaps in detection using confirmed neighbors
BALL_INTERP_BUFFER   = 20      # frames to buffer for look-ahead (~333ms at 60fps)
BALL_INTERP_MAX_GAP  = 8       # max consecutive missing frames to interpolate


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
        self._lost:     dict = {}          # tid → (frame_lost, cx, cy)
        self._id_remap: dict = {}
        self._jersey_ocr: "JerseyOCR | None" = None

    def set_jersey_ocr(self, jersey_ocr: "JerseyOCR"):
        self._jersey_ocr = jersey_ocr

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

            # --- Jersey-based matching (high priority, ignores distance) ---
            if self._jersey_ocr is not None:
                new_jersey = self._jersey_ocr.get_jersey(tid)
                if new_jersey is not None:
                    for old_tid in list(self._lost.keys()):
                        old_jersey = self._jersey_ocr.get_jersey(old_tid)
                        if old_jersey == new_jersey:
                            best_old_tid = old_tid
                            break

            # --- Proximity fallback (if no jersey match) ---
            if best_old_tid is None:
                for old_tid, (_, ox, oy) in self._lost.items():
                    dist = math.hypot(cx - ox, cy - oy)
                    if dist < best_dist:
                        best_dist    = dist
                        best_old_tid = old_tid

            if best_old_tid is not None:
                self._id_remap[tid] = best_old_tid
                del self._lost[best_old_tid]
                d["tid"] = best_old_tid
                if self._jersey_ocr is not None:
                    self._jersey_ocr.transfer_id(best_old_tid, best_old_tid)

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

    def needs_scan(self, tid: int) -> bool:
        """Unconfirmed players need more frequent OCR."""
        return tid not in self._confirmed

    def scan(self, frame: np.ndarray, detections: list):
        """Run OCR on player crops."""
        for d in detections:
            tid = d["tid"]
            if tid in self._confirmed:
                continue
            box = d["box"]
            x1, y1, x2, y2 = box
            h = y2 - y1
            crop_y2 = y1 + int(h * 0.55)
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
                if not re.match(r"^\d{1,2}$", text):
                    continue
                num = int(text)
                if num < 1 or num > 99:
                    continue
                jersey = str(num)
                if tid not in self._candidates:
                    self._candidates[tid] = {}
                self._candidates[tid][jersey] = self._candidates[tid].get(jersey, 0) + 1
                # Lock: needs OCR_CONFIRM_COUNT reads AND must be the plurality
                cands = self._candidates[tid]
                top_jersey = max(cands, key=cands.get)
                if cands[top_jersey] >= OCR_CONFIRM_COUNT and top_jersey == jersey:
                    self._confirmed[tid] = jersey
                    self._candidates.pop(tid, None)
                    break

    def transfer_id(self, old_tid: int, new_tid: int):
        """When Re-ID remaps a track, carry the jersey number over."""
        if old_tid in self._confirmed:
            self._confirmed[new_tid] = self._confirmed[old_tid]


# ---------------------------------------------------------------------------
# Team Classification — K-means on jersey color histograms
# ---------------------------------------------------------------------------

TEAM_SAMPLE_COUNT = 100   # player crops to collect before clustering
TEAM_COLORS = [(255, 140, 0), (0, 180, 255)]  # BGR: orange (team A), cyan (team B)

class TeamClassifier:
    """Auto-detect two teams from jersey colors using K-means clustering."""

    def __init__(self):
        self._samples: list  = []     # list of (tid, color_hist)
        self._labels:  dict  = {}     # tid → 0 or 1 (team index)
        self._centers         = None  # K-means cluster centers
        self._ready           = False

    def is_ready(self) -> bool:
        return self._ready

    def get_team(self, tid: int) -> int | None:
        return self._labels.get(tid)

    def get_team_color(self, tid: int):
        team = self._labels.get(tid)
        if team is not None:
            return TEAM_COLORS[team]
        return None

    def collect(self, frame: np.ndarray, detections: list):
        """Collect upper-torso color histograms from player detections."""
        if self._ready:
            return
        for d in detections:
            tid = d["tid"]
            if any(s[0] == tid for s in self._samples):
                continue  # already sampled this tid
            box = d["box"]
            x1, y1, x2, y2 = box
            h = y2 - y1
            w = x2 - x1
            # Crop upper-middle torso (avoid arms, head)
            crop_y1 = y1 + int(h * 0.15)
            crop_y2 = y1 + int(h * 0.55)
            crop_x1 = x1 + int(w * 0.20)
            crop_x2 = x2 - int(w * 0.20)
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue
            crop = frame[max(0, crop_y1):crop_y2, max(0, crop_x1):crop_x2]
            if crop.size == 0:
                continue
            hist = self._color_hist(crop)
            self._samples.append((tid, hist))

        if len(self._samples) >= TEAM_SAMPLE_COUNT:
            self._cluster()

    def _color_hist(self, crop: np.ndarray) -> np.ndarray:
        """Compute a normalized HSV hue+saturation histogram."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 16 hue bins, 8 saturation bins = 128-dim feature
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _cluster(self):
        """Run K-means (k=2) on collected histograms."""
        data = np.array([s[1] for s in self._samples], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    100, 0.2)
        _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10,
                                        cv2.KMEANS_PP_CENTERS)
        self._centers = centers
        for i, (tid, _) in enumerate(self._samples):
            self._labels[tid] = int(labels[i][0])
        self._ready = True
        print(f"  Team classifier ready ({len(self._samples)} samples, 2 clusters)")

    def classify(self, frame: np.ndarray, det: dict) -> int | None:
        """Classify a new player detection into team 0 or 1."""
        if not self._ready or self._centers is None:
            return None
        tid = det["tid"]
        if tid in self._labels:
            return self._labels[tid]
        box = det["box"]
        x1, y1, x2, y2 = box
        h = y2 - y1
        w = x2 - x1
        crop_y1 = y1 + int(h * 0.15)
        crop_y2 = y1 + int(h * 0.55)
        crop_x1 = x1 + int(w * 0.20)
        crop_x2 = x2 - int(w * 0.20)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        crop = frame[max(0, crop_y1):crop_y2, max(0, crop_x1):crop_x2]
        if crop.size == 0:
            return None
        hist = self._color_hist(crop)
        # Assign to nearest cluster center
        dists = [np.linalg.norm(hist - c) for c in self._centers]
        team = int(np.argmin(dists))
        self._labels[tid] = team
        return team


# ---------------------------------------------------------------------------
# Ball Tracker — interpolation + trajectory prediction for ball occlusions
# ---------------------------------------------------------------------------

class BallTracker:
    """Track ball detections — pick best detection per frame.
    Simplified for TrackNet: typically 1 detection per frame.
    When multiple sources (TrackNet + SAHI), picks by confidence.
    Stores last known position so SAHI can target that region."""

    _VEL_HISTORY = 5       # frames of velocity history to average

    def __init__(self):
        self._last_box: list | None = None
        self._last_seen_frame: int = -999
        self._frames_since: int = 0
        self._vel_history: deque = deque(maxlen=self._VEL_HISTORY)
        self._last_cx: float = 0.0
        self._last_cy: float = 0.0

    @property
    def last_center(self) -> tuple | None:
        if self._last_box is None:
            return None
        b = self._last_box
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @property
    def frames_since_seen(self) -> int:
        return self._frames_since

    def update(self, frame_idx: int, ball_dets: list) -> list:
        """Return best ball detection. Picks highest confidence."""
        self._frames_since = frame_idx - self._last_seen_frame
        if not ball_dets:
            return []

        # Pick highest confidence detection
        best = max(ball_dets, key=lambda d: d.get("conf", 0.5))
        best_cx = (best["box"][0] + best["box"][2]) / 2.0
        best_cy = (best["box"][1] + best["box"][3]) / 2.0

        # Update velocity history
        if self._last_box is not None and self._frames_since <= 3:
            gap = max(1, self._frames_since)
            vx = (best_cx - self._last_cx) / gap
            vy = (best_cy - self._last_cy) / gap
            self._vel_history.append((vx, vy))
        elif self._frames_since > 10:
            self._vel_history.clear()

        self._last_box = best["box"]
        self._last_cx = best_cx
        self._last_cy = best_cy
        self._last_seen_frame = frame_idx
        return [best]


class BallInterpolator:
    """Buffer frames and fill ball detection gaps using confirmed neighbors.
    Since this is post-game review, we can look ahead before finalizing output.
    Uses a sliding window of BALL_INTERP_BUFFER frames."""

    def __init__(self, max_teleport: float):
        self._buffer: deque = deque()  # deque of (frame_image, ball_box_or_None)
        self._max_teleport = max_teleport

    def push(self, frame: np.ndarray, ball_box: list | None, label: str | None = None):
        """Add a frame and its ball detection (or None) to the buffer."""
        self._buffer.append((frame, ball_box, label))

    def _interpolate_box(self, box_a: list, box_b: list, t: float) -> list:
        """Linearly interpolate between two boxes. t=0 → box_a, t=1 → box_b."""
        return [int(box_a[i] + (box_b[i] - box_a[i]) * t) for i in range(4)]

    def _interpolate_box_arc(self, box_a: list, box_b: list, t: float,
                             gap_len: int) -> list:
        """Quadratic interpolation with gravity for arcing ball trajectories.
        Adds a downward parabolic offset to the Y axis — peaks at t=0.5."""
        # Linear base for all 4 coords
        box = [box_a[i] + (box_b[i] - box_a[i]) * t for i in range(4)]
        # Gravity arc: ball rises then falls. Peak offset at midpoint.
        # Scale arc height by horizontal distance and gap length.
        cx_a = (box_a[0] + box_a[2]) / 2.0
        cy_a = (box_a[1] + box_a[3]) / 2.0
        cx_b = (box_b[0] + box_b[2]) / 2.0
        cy_b = (box_b[1] + box_b[3]) / 2.0
        horiz_dist = abs(cx_b - cx_a)
        # Arc height: proportional to horizontal distance, capped
        arc_height = min(horiz_dist * 0.3, gap_len * 15.0)
        # Parabolic offset: negative because Y increases downward in image coords
        # Ball goes UP (y decreases) then comes back down
        y_offset = -arc_height * 4.0 * t * (1.0 - t)
        box[1] += y_offset  # y1
        box[3] += y_offset  # y2
        return [int(v) for v in box]

    def _box_dist(self, box_a: list, box_b: list) -> float:
        """Distance between box centers."""
        cx_a = (box_a[0] + box_a[2]) / 2.0
        cy_a = (box_a[1] + box_a[3]) / 2.0
        cx_b = (box_b[0] + box_b[2]) / 2.0
        cy_b = (box_b[1] + box_b[3]) / 2.0
        return math.hypot(cx_a - cx_b, cy_a - cy_b)

    def _reject_outliers(self):
        """Remove short detection spikes that are likely false positives.
        If a detection at frame N is far from both neighbors (N-1 and N+1)
        but those neighbors are close to each other, N is an outlier.
        Works for spikes of 1-3 frames."""
        n = len(self._buffer)
        if n < 3:
            return

        # How far to search for neighbor detections (allows gaps)
        LOOK = 8

        # Pass 1: mark single-frame outliers
        for i in range(1, n - 1):
            curr = self._buffer[i][1]
            if curr is None:
                continue
            prev_box = None
            for j in range(i - 1, max(i - LOOK, -1), -1):
                if self._buffer[j][1] is not None:
                    prev_box = self._buffer[j][1]
                    break
            next_box = None
            for j in range(i + 1, min(i + LOOK, n)):
                if self._buffer[j][1] is not None:
                    next_box = self._buffer[j][1]
                    break

            if prev_box is None or next_box is None:
                continue

            dist_to_prev = self._box_dist(curr, prev_box)
            dist_to_next = self._box_dist(curr, next_box)
            dist_prev_next = self._box_dist(prev_box, next_box)

            # Outlier if BOTH conditions hold:
            # 1. Current jumped away from both neighbors more than they
            #    moved relative to each other (ratio test)
            # 2. Neighbors are reasonably close (didn't both teleport)
            min_jump = max(dist_to_prev, dist_to_next)
            # Spike jumps 3x+ further than the neighbors moved apart
            is_spike = (min_jump > dist_prev_next * 3.0 + 20.0 and
                        dist_to_prev > 30.0 and dist_to_next > 30.0)
            if is_spike:
                frame_img = self._buffer[i][0]
                self._buffer[i] = (frame_img, None, None)

        # Pass 2: mark 2-frame spike outliers (A→B→B→A pattern)
        for i in range(1, n - 2):
            curr = self._buffer[i][1]
            curr_next = self._buffer[i + 1][1]
            if curr is None or curr_next is None:
                continue

            if self._box_dist(curr, curr_next) > self._max_teleport * 0.3:
                continue  # not a coherent spike

            prev_box = None
            for j in range(i - 1, max(i - LOOK, -1), -1):
                if self._buffer[j][1] is not None:
                    prev_box = self._buffer[j][1]
                    break
            next_box = None
            for j in range(i + 2, min(i + 2 + LOOK, n)):
                if self._buffer[j][1] is not None:
                    next_box = self._buffer[j][1]
                    break

            if prev_box is None or next_box is None:
                continue

            dist_spike_prev = self._box_dist(curr, prev_box)
            dist_spike_next = self._box_dist(curr_next, next_box)
            dist_prev_next = self._box_dist(prev_box, next_box)

            min_jump = max(dist_spike_prev, dist_spike_next)
            is_spike = (min_jump > dist_prev_next * 3.0 + 20.0 and
                        dist_spike_prev > 30.0 and dist_spike_next > 30.0)
            if is_spike:
                frame_a = self._buffer[i][0]
                frame_b = self._buffer[i + 1][0]
                self._buffer[i] = (frame_a, None, None)
                self._buffer[i + 1] = (frame_b, None, None)

        # Pass 3: mark 3-frame spike outliers (A→B→B→B→A pattern)
        for i in range(1, n - 3):
            frames = [self._buffer[i + k][1] for k in range(3)]
            if any(f is None for f in frames):
                continue
            if (self._box_dist(frames[0], frames[1]) > self._max_teleport * 0.3 or
                    self._box_dist(frames[1], frames[2]) > self._max_teleport * 0.3):
                continue

            prev_box = None
            for j in range(i - 1, max(i - LOOK, -1), -1):
                if self._buffer[j][1] is not None:
                    prev_box = self._buffer[j][1]
                    break
            next_box = None
            for j in range(i + 3, min(i + 3 + LOOK, n)):
                if self._buffer[j][1] is not None:
                    next_box = self._buffer[j][1]
                    break

            if prev_box is None or next_box is None:
                continue

            dist_spike_prev = self._box_dist(frames[0], prev_box)
            dist_spike_next = self._box_dist(frames[2], next_box)
            dist_prev_next = self._box_dist(prev_box, next_box)

            min_jump = max(dist_spike_prev, dist_spike_next)
            is_spike = (min_jump > dist_prev_next * 3.0 + 20.0 and
                        dist_spike_prev > 30.0 and dist_spike_next > 30.0)
            if is_spike:
                for k in range(3):
                    frame_img = self._buffer[i + k][0]
                    self._buffer[i + k] = (frame_img, None, None)

    def _fill_gaps(self):
        """Reject outlier spikes, then fill gaps with interpolated positions."""
        self._reject_outliers()
        n = len(self._buffer)
        i = 0
        while i < n:
            if self._buffer[i][1] is not None:
                i += 1
                continue
            # Found a gap start — find gap end
            gap_start = i
            while i < n and self._buffer[i][1] is None:
                i += 1
            gap_end = i  # first frame after gap with detection (or end of buffer)
            gap_len = gap_end - gap_start

            if gap_len > BALL_INTERP_MAX_GAP:
                continue  # gap too long, don't interpolate

            # Need detections on BOTH sides
            if gap_start == 0 or gap_end >= n:
                continue
            box_before = self._buffer[gap_start - 1][1]
            box_after  = self._buffer[gap_end][1]
            if box_before is None or box_after is None:
                continue

            # Check distance is reasonable
            dist = self._box_dist(box_before, box_after)
            if dist > self._max_teleport * (gap_len + 1):
                continue  # ball moved too far, likely different object

            # Interpolate each gap frame (arc for longer gaps, linear for short)
            use_arc = gap_len > 3
            for j in range(gap_start, gap_end):
                t = (j - gap_start + 1) / (gap_len + 1)
                if use_arc:
                    interp_box = self._interpolate_box_arc(
                        box_before, box_after, t, gap_len)
                else:
                    interp_box = self._interpolate_box(box_before, box_after, t)
                frame_img = self._buffer[j][0]
                self._buffer[j] = (frame_img, interp_box, None)  # label=None → "INTERP"

    def pop_ready(self) -> list:
        """Return finalized frames that are safe to write.
        Keeps BALL_INTERP_BUFFER//2 frames buffered for look-ahead."""
        half = BALL_INTERP_BUFFER // 2
        if len(self._buffer) < BALL_INTERP_BUFFER:
            return []
        # Fill gaps in current buffer
        self._fill_gaps()
        # Pop frames that are far enough from the buffer edge
        ready = []
        while len(self._buffer) > half:
            frame_img, ball_box, label = self._buffer.popleft()
            ready.append((frame_img, ball_box, label))
        return ready

    def flush(self) -> list:
        """Flush all remaining frames at end of video."""
        self._fill_gaps()
        ready = []
        while self._buffer:
            frame_img, ball_box, label = self._buffer.popleft()
            ready.append((frame_img, ball_box, label))
        return ready


# ---------------------------------------------------------------------------
# Ball Validator — reject false positives (off-court, teleports, backboard)
# ---------------------------------------------------------------------------

class BallValidator:
    """Validate ball detections: court boundary, teleport rejection, backboard."""

    def __init__(self, roi_poly: np.ndarray, frame_w: int, frame_h: int):
        self._roi_poly = roi_poly
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._last_center: tuple | None = None
        self._last_frame: int = -999
        self._near_hoop_frame: int = -999      # last frame ball was near a hoop
        self._near_hoop_box: list | None = None # the hoop box it was near
        self._max_teleport = BALL_MAX_TELEPORT_PX * (frame_w / 1920.0)

    def filter(self, frame_idx: int, ball_dets: list,
               hoop_dets: list) -> list:
        """Return only plausible ball detections."""
        valid = []
        for d in ball_dets:
            box = d["box"]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            # --- Check 1: Court column ---
            if not ball_in_court_region(self._roi_poly, cx, cy, self._frame_w):
                continue

            # --- Check 2: Teleport rejection ---
            if self._last_center is not None:
                gap = frame_idx - self._last_frame
                if 0 < gap <= BALL_TELEPORT_GRACE:
                    dist = math.hypot(cx - self._last_center[0],
                                      cy - self._last_center[1])
                    if dist > self._max_teleport * gap:
                        continue

            # --- Check 3: Backboard constraint ---
            # If ball was near a hoop recently, reject detections that jump
            # ABOVE the hoop into the crowd. Ball can go past horizontally
            # (airball/bounce) but can't teleport upward into the stands.
            if (self._near_hoop_box is not None and
                    frame_idx - self._near_hoop_frame <= 30):
                hoop_top = self._near_hoop_box[1]  # top Y of hoop box
                # Ball can't be significantly above the hoop AND past it
                # horizontally — that's the crowd/backboard area
                hoop_cx = (self._near_hoop_box[0] + self._near_hoop_box[2]) / 2.0
                court_cx = self._frame_w / 2.0
                above_hoop = cy < hoop_top - self._frame_h * 0.03
                if above_hoop:
                    if hoop_cx < court_cx and cx < hoop_cx:
                        continue  # above & behind left hoop
                    elif hoop_cx >= court_cx and cx > hoop_cx:
                        continue  # above & behind right hoop

            valid.append(d)

        # Update history with the best valid detection
        if valid:
            best = max(valid, key=lambda d: (d["box"][2] - d["box"][0]) *
                                             (d["box"][3] - d["box"][1]))
            bcx = (best["box"][0] + best["box"][2]) / 2.0
            bcy = (best["box"][1] + best["box"][3]) / 2.0
            self._last_center = (bcx, bcy)
            self._last_frame = frame_idx
            # Track if ball is near a hoop (for backboard constraint)
            for h in hoop_dets:
                if self._near_any_hoop(bcx, bcy, [h]):
                    self._near_hoop_frame = frame_idx
                    self._near_hoop_box = h["box"]
                    break

        return valid

    def _near_any_hoop(self, cx: float, cy: float, hoop_dets: list) -> bool:
        margin = self._frame_w * 0.08
        for h in hoop_dets:
            hcx = (h["box"][0] + h["box"][2]) / 2.0
            hcy = (h["box"][1] + h["box"][3]) / 2.0
            if abs(cx - hcx) < margin and abs(cy - hcy) < margin:
                return True
        return False


# ---------------------------------------------------------------------------
# SAHI — targeted inference around ball's last known position
# ---------------------------------------------------------------------------

def run_sahi_ball_targeted(
    sahi_model: AutoDetectionModel,
    frame: np.ndarray,
    center: tuple,
    existing_ball_dets: list,
    ball_cls_id: int = COCO_BALL_CLS,
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
        if int(pred.category.id) != ball_cls_id:
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
            sahi_balls.append({"tid": -1, "box": box, "cls": CLS_BALL, "conf": pred.score.value})

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


def ball_in_court_region(roi_poly: np.ndarray, cx: float, cy: float,
                         frame_w: int) -> bool:
    """Check if ball is within the court or above it (airborne).

    Rules:
      - Ball BELOW the court bottom edge → reject (crowd/bench area)
      - Ball WITHIN or ABOVE the court → accept if x is within court column
      - Small vertical slack below bottom edge for balls near the baseline

    The court ROI trapezoid has 4 points:
      [0] bottom-left, [1] bottom-right, [2] top-right, [3] top-left
    """
    pts = roi_poly.reshape(-1, 2).astype(np.float64)
    bl, br, tr, tl = pts[0], pts[1], pts[2], pts[3]

    top_y = min(tl[1], tr[1])
    bot_y = max(bl[1], br[1])
    pad_x = frame_w * BALL_COURT_X_PAD
    # Small vertical slack below the baseline (ball can bounce near edge)
    pad_y = (bot_y - top_y) * 0.05

    # Reject anything below the court bottom + slack (crowd/bench)
    if cy > bot_y + pad_y:
        return False

    # For ball within or above the court, check x is in the court column
    if cy <= top_y:
        # Above the court — use top edge x-span
        left_x  = tl[0] - pad_x
        right_x = tr[0] + pad_x
    else:
        # Within the court trapezoid — interpolate left/right edges
        t = (cy - top_y) / (bot_y - top_y)
        left_x  = tl[0] + t * (bl[0] - tl[0]) - pad_x
        right_x = tr[0] + t * (br[0] - tr[0]) + pad_x

    return left_x <= cx <= right_x


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


def draw_ball(frame, box, label="BALL"):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BALL, 2, cv2.LINE_AA)
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


def finetune(epochs: int = 50, batch: int = 8, imgsz: int = 960):
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
        # Augmentation — aggressive color jitter to force shape learning over color.
        hsv_h=0.15,            # strong hue shift — force model to learn shape over color
        hsv_s=0.7,             # aggressive saturation — weaken color reliance
        hsv_v=0.4,             # brightness jitter — handles shadows, glare
        degrees=10.0,          # slight rotation — preserve court spatial context
        translate=0.2,
        scale=0.5,             # moderate scale variation
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,            # no vertical flip — preserves court=bottom, crowd=top
        fliplr=0.5,
        mosaic=1.0,            # combine 4 images — occlusion + context variety
        mixup=0.15,            # light blending
        copy_paste=0.3,        # paste objects — synthetic occlusion
        erasing=0.4,           # random erase — partial occlusion robustness
        crop_fraction=1.0,
        # Early stopping — 242 val images = smooth metrics, can stop sooner
        patience=10,
    )

    best_pt = base_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
    print(f"\nFine-tuning complete.  Best weights: {best_pt}")
    print(f"Run inference: python main.py --video data/game_01.mp4")
    return str(best_pt)


# ---------------------------------------------------------------------------
# TensorRT export — one-time conversion for faster inference
# ---------------------------------------------------------------------------

def _trt_engine_path(pt_path: str, imgsz: int) -> Path:
    """Return the expected TensorRT engine path for a given .pt model."""
    p = Path(pt_path)
    return p.parent / f"{p.stem}.engine"


def export_tensorrt(weights: str | None = None, imgsz: int = 1280):
    """Export all pipeline models to TensorRT .engine format.
    Run once per GPU — engines are hardware-specific."""
    device = get_device()
    if not device.startswith("cuda"):
        print("ERROR: TensorRT export requires a CUDA GPU.")
        return

    base_dir = Path(__file__).resolve().parent
    weights_path = weights or str(
        base_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
    )

    models_to_export = [
        (weights_path, "Fine-tuned detection"),
        ("yolo11n-pose.pt", "Pose estimation"),
        ("yolo11n.pt", "COCO ball fallback"),
    ]

    for pt_path, label in models_to_export:
        engine_path = _trt_engine_path(pt_path, imgsz)
        if engine_path.exists():
            print(f"  {label}: {engine_path} already exists, skipping")
            continue
        print(f"  Exporting {label}: {pt_path} → TensorRT (imgsz={imgsz}) ...")
        m = YOLO(pt_path)
        m.export(format="engine", imgsz=imgsz, half=True, device=0)
        print(f"  → {engine_path}")

    print("\nTensorRT export complete. Re-run inference to use engines automatically.")


def _load_model(pt_path: str, imgsz: int, label: str) -> YOLO:
    """Load a YOLO model, preferring TensorRT engine if available."""
    engine_path = _trt_engine_path(pt_path, imgsz)
    if engine_path.exists():
        print(f"Loading {label}: {engine_path.name} (TensorRT)")
        return YOLO(str(engine_path))
    print(f"Loading {label}: {Path(pt_path).name}")
    return YOLO(pt_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run(video_path: str, out_dir: str, weights: str | None = None,
        use_sahi: bool = True, use_tracknet: bool = True,
        tracknet_weights: str | None = None):
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
        infer_imgsz = 1280
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

    model      = _load_model(weights_path, infer_imgsz, "detection model")
    pose_model = _load_model("yolo11n-pose.pt", infer_imgsz, "pose model")

    # TrackNet — primary ball detector (heatmap regression on 3 consecutive frames)
    tracknet_model = None
    if use_tracknet:
        tn_weights = tracknet_weights or TRACKNET_WEIGHTS
        tracknet_model = TrackNetInference(
            tn_weights, device=device,
            conf_thresh=TRACKNET_CONF_THRESH,
            ball_radius=TRACKNET_BALL_RADIUS,
        )

    # SAHI — sliced inference fallback for ball detection
    sahi_model = None
    if use_sahi:
        print("Loading SAHI sliced-inference model (COCO sports ball) ...")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",          # compatible with YOLO11
            model_path="yolo11s.pt",      # COCO pretrained — sports ball class
            confidence_threshold=SAHI_CONF_THRESH,
            device=device,
        )

    roi_poly      = build_court_roi(frame_w, frame_h)
    ghost_max_age = int(fps * 0.5)   # 0.5s — was 1.5s, caused stale boxes in empty space
    memory        = TrackMemory(max_age=ghost_max_age)
    reid_buffer   = TemporalReIDBuffer()
    vel_tracker   = VelocityTracker()
    tracker_cfg   = str(base_dir / "bytetrack_players.yaml")

    print("Loading EasyOCR (jersey number reader) ...")
    jersey_ocr    = JerseyOCR()
    reid_buffer.set_jersey_ocr(jersey_ocr)
    ball_tracker   = BallTracker()
    ball_validator = BallValidator(roi_poly, frame_w, frame_h)
    team_clf      = TeamClassifier()
    ball_interp   = BallInterpolator(max_teleport=BALL_MAX_TELEPORT_PX * (frame_w / 1920.0))

    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        str(Path(out_dir) / "annotated.mp4"), fourcc, fps, (frame_w, frame_h)
    )

    frame_idx = 0
    frame_skip = 1       # process every frame (no skip)
    pose_interval = 1    # run pose every processed frame
    pose_counter = 0     # counter for processed frames
    last_ball_box = None  # cache ball result for skipped frames
    # Cache last-processed-frame annotations so skipped frames aren't bare
    cached_real_dets: list = []
    cached_ghost_dets: list = []
    cached_hoop_dets: list = []
    cached_ref_dets: list = []
    cached_pose_map: dict = {}
    print("Processing frames ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames: on non-processed frames, still run TrackNet (needs
        # every frame for its 3-frame sliding window) but skip YOLO/pose/OCR.
        # Redraw cached annotations so the output doesn't flicker.
        if frame_idx % frame_skip != 0:
            tracknet_ball_box = None
            if tracknet_model is not None:
                tn_det = tracknet_model.predict(frame)
                if tn_det is not None:
                    tracknet_ball_box = tn_det["box"]

            # Redraw all non-ball annotations from the last processed frame
            for det in cached_hoop_dets:
                draw_hoop(frame, det["box"])
            for det in cached_ref_dets:
                draw_ref(frame, det["box"], det["tid"])
            for d in cached_real_dets:
                tid = d["tid"]
                jersey = jersey_ocr.get_jersey(tid)
                label = f"#{jersey}" if jersey else f"#{tid}"
                team_color = team_clf.get_team_color(tid)
                draw_player(frame, d["box"], label, color=team_color)
                if tid in cached_pose_map:
                    draw_skeleton(frame, *cached_pose_map[tid])
            for d in cached_ghost_dets:
                tid = d["tid"]
                jersey = jersey_ocr.get_jersey(tid)
                label = f"#{jersey}" if jersey else f"#{tid}"
                team_color = team_clf.get_team_color(tid)
                draw_player(frame, d["box"], label, color=team_color)

            skip_ball = tracknet_ball_box if tracknet_ball_box else last_ball_box
            skip_label = "BALL (TrackNet)" if tracknet_ball_box else None
            ball_interp.push(frame, skip_ball, skip_label)
            for fin_frame, fin_ball_box, fin_label in ball_interp.pop_ready():
                if fin_ball_box is not None:
                    draw_ball(fin_frame, fin_ball_box, label=fin_label or "BALL (INTERP)")
                out_video.write(fin_frame)
            frame_idx += 1
            continue

        t_frame_start = time.perf_counter()

        # ---- 1. Detection & tracking -----------------------------------------
        raw_dets       = []
        ball_dets      = []
        hoop_dets      = []
        ref_dets       = []
        ball_source    = "none"

        det_res = model.track(
            frame, persist=True, tracker=tracker_cfg,
            conf=0.10, iou=0.50,
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
                elif cls_id == CLS_HOOP:
                    if conf_val >= 0.30:
                        hoop_dets.append({"tid": tid, "box": box_list, "cls": CLS_HOOP})
                elif cls_id == CLS_REF:
                    if conf_val >= 0.30:
                        ref_dets.append({"tid": tid, "box": box_list, "cls": CLS_REF})

        # ---- 1a. TrackNet primary ball detection --------------------------------
        # TrackNet uses 3 consecutive frames (temporal context) for heatmap
        # regression — much better at small, fast-moving balls than YOLO.
        if tracknet_model is not None:
            tn_det = tracknet_model.predict(frame)
            if tn_det is not None:
                ball_dets.append(tn_det)
                ball_source = "TrackNet"

        # ---- 1b. SAHI fallback — when TrackNet misses the ball ----------------
        if sahi_model is not None and not ball_dets:
            center = ball_tracker.last_center
            if center is not None and ball_tracker.frames_since_seen < SAHI_MAX_LOST:
                # COCO SAHI — sports ball class
                sahi_balls = run_sahi_ball_targeted(
                    sahi_model, frame, center, ball_dets,
                    ball_cls_id=COCO_BALL_CLS)
                ball_dets.extend(sahi_balls)
            elif center is None or ball_tracker.frames_since_seen >= SAHI_MAX_LOST:
                # No prior position or ball lost too long — run full-frame SAHI
                # every 5 frames to re-acquire without killing performance
                if frame_idx % 5 == 0:
                    result = get_sliced_prediction(
                        image=frame,
                        detection_model=sahi_model,
                        slice_height=320,
                        slice_width=320,
                        overlap_height_ratio=0.25,
                        overlap_width_ratio=0.25,
                        postprocess_type="NMS",
                        postprocess_match_threshold=0.40,
                        verbose=0,
                    )
                    for pred in result.object_prediction_list:
                        if int(pred.category.id) != COCO_BALL_CLS:
                            continue
                        if pred.score.value < SAHI_CONF_THRESH:
                            continue
                        bb = pred.bbox
                        box = [int(bb.minx), int(bb.miny),
                               int(bb.maxx), int(bb.maxy)]
                        ball_dets.append({"tid": -1, "box": box, "cls": CLS_BALL, "conf": pred.score.value})
            if ball_dets:
                ball_source = "SAHI"

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
                    jersey_ocr.transfer_id(old_tid, d["tid"])

        # ---- 3. Jersey OCR (adaptive frequency) --------------------------------
        # Every 10 frames for unconfirmed players, confirmed ones skip inside scan()
        has_unconfirmed = any(jersey_ocr.needs_scan(d["tid"]) for d in real_dets)
        ocr_interval = 30 if has_unconfirmed else 90
        if frame_idx % ocr_interval == 0:
            jersey_ocr.scan(frame, real_dets)

        # ---- 4. Pose estimation (players only, exclude refs) ------------------
        # Run pose every pose_interval processed frames; reuse cached result otherwise
        pose_counter += 1
        if pose_counter % pose_interval == 0:
            ref_boxes = [d["box"] for d in ref_dets]
            pose_res = pose_model.predict(
                frame, verbose=False, device=device,
                conf=0.20, imgsz=infer_imgsz, half=use_half,
            )[0]
            cached_pose_map = associate_poses(real_dets, ref_boxes, pose_res)
        pose_map = cached_pose_map

        # ---- 5. Ball validation + tracking ------------------------------------
        ball_dets    = ball_validator.filter(frame_idx, ball_dets, hoop_dets)
        ball_to_draw = ball_tracker.update(frame_idx, ball_dets)

        # ---- 6. Draw hoop / ref / players (everything except ball) ------------
        for det in hoop_dets:
            draw_hoop(frame, det["box"])

        for det in ref_dets:
            draw_ref(frame, det["box"], det["tid"])

        # Collect team color samples (before classifier is ready)
        if not team_clf.is_ready():
            team_clf.collect(frame, real_dets)

        for d in real_dets:
            tid = d["tid"]
            jersey = jersey_ocr.get_jersey(tid)
            label = f"#{jersey}" if jersey else f"#{tid}"
            team_color = team_clf.get_team_color(tid)
            if team_color is None and team_clf.is_ready():
                team_clf.classify(frame, d)
                team_color = team_clf.get_team_color(tid)
            draw_player(frame, d["box"], label, color=team_color)
            if tid in pose_map:
                draw_skeleton(frame, *pose_map[tid])

        # Cache annotations for skipped frames (prevents flicker)
        cached_real_dets = real_dets
        cached_ghost_dets = ghost_dets
        cached_hoop_dets = hoop_dets
        cached_ref_dets = ref_dets
        cached_pose_map = pose_map

        # ---- 7. Buffer frame for ball interpolation --------------------------
        # Ball is NOT drawn yet — the interpolator will fill gaps first.
        ball_box = ball_to_draw[0]["box"] if ball_to_draw else None
        last_ball_box = ball_box  # cache for skipped frames
        ball_label = f"BALL ({ball_source})" if ball_box else None
        ball_interp.push(frame, ball_box, ball_label)

        # Write finalized frames (interpolator releases them after look-ahead)
        for fin_frame, fin_ball_box, fin_label in ball_interp.pop_ready():
            if fin_ball_box is not None:
                draw_ball(fin_frame, fin_ball_box, label=fin_label or "BALL (INTERP)")
            out_video.write(fin_frame)

        frame_idx += 1

        t_frame = time.perf_counter() - t_frame_start
        if frame_idx % 30 == 0 or frame_idx <= 5:
            n_jerseys = sum(1 for d in real_dets if jersey_ocr.get_jersey(d["tid"]))
            print(f"  frame {frame_idx}/{total_frames}  "
                  f"| {t_frame:.2f}s/frame  "
                  f"| players={len(real_dets)}  "
                  f"| jerseys={n_jerseys}  "
                  f"| poses={len(pose_map)}  "
                  f"| ball={ball_source if ball_to_draw else 'no'}")

    # Flush remaining buffered frames
    for fin_frame, fin_ball_box, fin_label in ball_interp.flush():
        if fin_ball_box is not None:
            draw_ball(fin_frame, fin_ball_box, label=fin_label or "BALL (INTERP)")
        out_video.write(fin_frame)

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
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--no-sahi",  action="store_true",
                        help="Disable SAHI sliced inference for ball detection")
    parser.add_argument("--no-tracknet", action="store_true",
                        help="Disable TrackNet ball detection (fall back to SAHI only)")
    parser.add_argument("--tracknet-weights", default=None,
                        help="Path to TrackNet weights (default: tracknet_basketball.pt)")
    parser.add_argument("--finetune-tracknet", action="store_true",
                        help="Train TrackNet then run inference")
    parser.add_argument("--tracknet-data", type=str, default=None,
                        help="TrackNet label directory (default: auto-converts YOLO labels)")
    parser.add_argument("--export-tensorrt", action="store_true",
                        help="Export all models to TensorRT .engine format (run once per GPU)")
    args = parser.parse_args()

    run_kwargs = dict(
        use_sahi=not args.no_sahi,
        use_tracknet=not args.no_tracknet,
        tracknet_weights=args.tracknet_weights,
    )

    if args.export_tensorrt:
        export_tensorrt(weights=args.weights)
    elif args.finetune_tracknet:
        from tracknet import train_tracknet
        if args.tracknet_data:
            tn_data_dir = args.tracknet_data
        else:
            from convert_labels import convert
            print("Converting YOLO labels → TrackNet format ...")
            convert("data/custom_annotations", "data/tracknet_labels")
            tn_data_dir = "data/tracknet_labels"
        print(f"\nTraining TrackNet on {tn_data_dir} ...")
        tn_best = train_tracknet(tn_data_dir,
                                 epochs=args.epochs, batch_size=args.batch)
        run_kwargs["tracknet_weights"] = tn_best
        run(args.video, args.out, weights=args.weights, **run_kwargs)
    elif args.finetune:
        best_pt = finetune(epochs=args.epochs, batch=args.batch)
        run(args.video, args.out, weights=best_pt, **run_kwargs)
    else:
        run(args.video, args.out, weights=args.weights, **run_kwargs)
