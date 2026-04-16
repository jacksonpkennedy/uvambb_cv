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

# TrackNet — heatmap-based ball detection using 3 consecutive frames
TRACKNET_WEIGHTS     = "runs/tracknet/weights/best_overall.pt"  # best across all training sessions
TRACKNET_CONF_THRESH = 0.5                        # heatmap peak confidence threshold (lower for early training)
TRACKNET_BALL_RADIUS = 15                          # pixels, for bbox synthesis from center point

# Jersey OCR
OCR_INTERVAL       = 30     # run OCR every N frames (performance vs accuracy)
OCR_CONFIRM_COUNT  = 2      # need this many consistent reads to lock a jersey number

# Ball validation — reject false positives (bald heads, off-court objects)
BALL_COURT_X_PAD     = 0.05    # horizontal slack beyond court edges (fraction of frame_w)
BALL_MAX_BBOX_AREA   = 2500    # max ball bbox area in pixels (at 1080p) — rejects bald heads
BALL_MIN_BBOX_AREA   = 50      # min ball bbox area — rejects noise
BALL_MAX_TELEPORT_PX = 200    # max pixels ball can move per frame (at 1080p, 60fps)
BALL_TELEPORT_GRACE  = 10      # frames since last sighting before teleport check resets
BALL_TEMPORAL_WINDOW = 5       # look-back window for temporal consistency
BALL_TEMPORAL_MIN    = 2       # require N of WINDOW recent frames in same region
BALL_TEMPORAL_RADIUS = 120     # max px spread to count as "same region" (at 1080p)
BALL_VELOCITY_HISTORY = 5      # frames of velocity history for motion prediction

# Ball interpolation — fill gaps in detection using confirmed neighbors
BALL_INTERP_BUFFER   = 20      # frames to buffer for look-ahead (~333ms at 60fps)
BALL_INTERP_MAX_GAP  = 8       # max consecutive missing frames to interpolate

# Ball head-lock rejection — use pose keypoints to reject bald-head false positives
BALL_HEADLOCK_WINDOW   = 20    # frames of offset history to evaluate (~333ms at 60fps)
BALL_HEADLOCK_MAX_VAR  = 6.0   # max px std-dev of offset to count as "locked" to a head
BALL_HEADLOCK_HEAD_KPS = (0, 1, 2, 3, 4)  # COCO: nose, left_eye, right_eye, left_ear, right_ear
BALL_HEADLOCK_PROX     = 80    # px — ball must be this close to a head kp to start tracking offset
HEADLOCK_PERSIST_FRAMES = 5    # consecutive head-locks before clearing track state (soft reject otherwise)

# Court visibility detection
COURT_MIN_LINES      = 4       # min Hough lines to consider court visible
COURT_WOOD_RATIO     = 0.15    # min ratio of wood-colored pixels (tan/brown floor)

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
    TrackNet is the sole ball detector; picks highest-confidence detection,
    maintains velocity history for the interpolator, and applies a Kalman
    filter (constant-velocity model) to smooth frame-to-frame jitter."""

    _VEL_HISTORY = 5       # frames of velocity history to average

    def __init__(self):
        self._last_box: list | None = None
        self._last_seen_frame: int = -999
        self._frames_since: int = 0
        self._vel_history: deque = deque(maxlen=self._VEL_HISTORY)
        self._last_cx: float = 0.0
        self._last_cy: float = 0.0

        # Kalman filter state (4-state constant-velocity: x, y, vx, vy)
        self._kf_x: np.ndarray | None = None   # (4,1) state vector
        self._kf_P: np.ndarray | None = None   # (4,4) covariance
        # Transition model: x_t = F * x_{t-1}
        self._kf_F = np.array([[1,0,1,0],[0,1,0,1],
                                [0,0,1,0],[0,0,0,1]], dtype=np.float64)
        # Observation model: z = H * x  (we observe x,y only)
        self._kf_H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        self._kf_R = np.eye(2, dtype=np.float64) * 15.0  # measurement noise
        # Process noise: low on position (model is fairly accurate), higher
        # on velocity (ball accelerates during shots/bounces/passes, so the
        # constant-velocity assumption is violated — KF needs to trust
        # measurements more during acceleration).
        self._kf_Q = np.diag([0.5, 0.5, 4.0, 4.0]).astype(np.float64)

    def _kf_init(self, cx: float, cy: float) -> None:
        self._kf_x = np.array([[cx],[cy],[0.0],[0.0]], dtype=np.float64)
        self._kf_P = np.eye(4, dtype=np.float64) * 100.0

    def _kf_predict(self) -> None:
        if self._kf_x is None:
            return
        self._kf_x = self._kf_F @ self._kf_x
        self._kf_P = self._kf_F @ self._kf_P @ self._kf_F.T + self._kf_Q

    def _kf_update(self, cx: float, cy: float) -> tuple[float, float]:
        """Update Kalman state with measurement, return smoothed (cx, cy)."""
        if self._kf_x is None:
            self._kf_init(cx, cy)
            return cx, cy
        z = np.array([[cx],[cy]], dtype=np.float64)
        S = self._kf_H @ self._kf_P @ self._kf_H.T + self._kf_R
        K = self._kf_P @ self._kf_H.T @ np.linalg.inv(S)
        self._kf_x = self._kf_x + K @ (z - self._kf_H @ self._kf_x)
        self._kf_P = (np.eye(4) - K @ self._kf_H) @ self._kf_P
        return float(self._kf_x[0, 0]), float(self._kf_x[1, 0])

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
        """Pick best detection and apply Kalman smoothing.

        Teleport/motion filtering is handled upstream in BallValidator — this
        class only picks highest-confidence candidate and smooths jitter.
        Kalman reset threshold is velocity-aware so acceleration doesn't
        cause constant resets.
        """
        self._frames_since = frame_idx - self._last_seen_frame

        if not ball_dets:
            self._kf_predict()
            return []

        # Pick best (validator already did teleport/head filtering)
        best = max(ball_dets, key=lambda d: d.get("conf", 0.5))
        raw_cx = (best["box"][0] + best["box"][2]) / 2.0
        raw_cy = (best["box"][1] + best["box"][3]) / 2.0

        # Velocity-aware Kalman reset: the KF uses a constant-velocity model,
        # so during acceleration (shot release, bounce, pass start) the
        # measurement legitimately diverges from the prediction. Scale the
        # reset threshold by current KF-estimated speed so we don't reset
        # every time the ball accelerates.
        if self._kf_x is not None and self._frames_since <= 5:
            kf_cx, kf_cy = float(self._kf_x[0, 0]), float(self._kf_x[1, 0])
            kf_vx, kf_vy = float(self._kf_x[2, 0]), float(self._kf_x[3, 0])
            kf_speed = math.hypot(kf_vx, kf_vy)
            reset_thresh = max(BALL_MAX_TELEPORT_PX, kf_speed * 2.5)
            if np.hypot(raw_cx - kf_cx, raw_cy - kf_cy) > reset_thresh:
                self._kf_x = None

        self._kf_predict()
        smooth_cx, smooth_cy = self._kf_update(raw_cx, raw_cy)

        bw, bh = best["box"][2] - best["box"][0], best["box"][3] - best["box"][1]
        smoothed_box = [
            int(smooth_cx - bw / 2), int(smooth_cy - bh / 2),
            int(smooth_cx + bw / 2), int(smooth_cy + bh / 2),
        ]

        best = dict(best)
        best["box"] = smoothed_box

        # Update velocity history (used for any external consumers / logging)
        if self._last_box is not None:
            gap = max(1, self._frames_since)
            self._vel_history.append(
                ((raw_cx - self._last_cx) / gap, (raw_cy - self._last_cy) / gap))

        self._last_box = smoothed_box
        self._last_cx, self._last_cy = raw_cx, raw_cy
        self._last_seen_frame = frame_idx
        return [best]

class BallInterpolator:
    """Buffer frames and fill ball detection gaps using confirmed neighbors.
    Since this is post-game review, we can look ahead before finalizing output.
    Uses a sliding window of BALL_INTERP_BUFFER frames.

    Stability requirement: interpolation only uses endpoints that pass a
    majority vote — MIN_STABLE of STABLE_WINDOW recent frames must have
    detections clustering within STABLE_RADIUS. This prevents interpolating
    toward a head or false positive that the model briefly locks onto.
    """

    STABLE_WINDOW = 8   # frames to look back/forward for stability check
    MIN_STABLE = 2      # require 2 of 8 frames to agree (tolerates ~44% recall)
    STABLE_RADIUS = 180  # max px spread within a stable cluster (at 1080p)

    def __init__(self, max_teleport: float):
        self._buffer: deque = deque()  # deque of (frame_image, ball_box_or_None)
        self._max_teleport = max_teleport
        # Scale stable radius proportionally to teleport (both are resolution-dependent)
        self._stable_radius = self.STABLE_RADIUS * (max_teleport / BALL_MAX_TELEPORT_PX)

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

    def _is_stable_before(self, idx: int) -> bool:
        """Check if MIN_STABLE of STABLE_WINDOW frames ending at idx have
        detections clustering within STABLE_RADIUS (majority-vote, gaps OK)."""
        if idx < 0 or self._buffer[idx][1] is None:
            return False
        ref = self._buffer[idx][1]
        count = 0
        start = max(idx - self.STABLE_WINDOW + 1, 0)
        for j in range(start, idx + 1):
            box = self._buffer[j][1]
            if box is not None and self._box_dist(ref, box) <= self._stable_radius:
                count += 1
        return count >= self.MIN_STABLE

    def _is_stable_after(self, idx: int) -> bool:
        """Check if MIN_STABLE of STABLE_WINDOW frames starting at idx have
        detections clustering within STABLE_RADIUS (majority-vote, gaps OK)."""
        n = len(self._buffer)
        if idx >= n or self._buffer[idx][1] is None:
            return False
        ref = self._buffer[idx][1]
        count = 0
        end = min(idx + self.STABLE_WINDOW, n)
        for j in range(idx, end):
            box = self._buffer[j][1]
            if box is not None and self._box_dist(ref, box) <= self._stable_radius:
                count += 1
        return count >= self.MIN_STABLE

    def _reject_physics_outliers(self):
        """Reject detections that break the ball's physical trajectory.

        Works on sliding windows of consecutive detections. Fits a quadratic
        to (t, x) and (t, y) separately — a quadratic with a≈0 is a straight
        line, so this handles both arcing shots AND straight passes.
        Rejects frames whose residual is >3× the window median (and >30px
        absolute, so we don't nitpick tight fits).
        """
        WINDOW = 5
        n = len(self._buffer)
        if n < WINDOW:
            return
        for i in range(n - WINDOW + 1):
            boxes = [self._buffer[i + k][1] for k in range(WINDOW)]
            if any(b is None for b in boxes):
                continue  # need full window for fit
            centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
            ts = np.arange(WINDOW, dtype=np.float64)
            xs = np.array([c[0] for c in centers], dtype=np.float64)
            ys = np.array([c[1] for c in centers], dtype=np.float64)
            try:
                px = np.polyfit(ts, xs, 2)
                py = np.polyfit(ts, ys, 2)
            except (np.linalg.LinAlgError, ValueError):
                continue
            xres = np.abs(xs - np.polyval(px, ts))
            yres = np.abs(ys - np.polyval(py, ts))
            res = np.hypot(xres, yres)
            med = float(np.median(res))
            if med < 10.0:
                continue  # fit is tight; small residuals are normal
            for k in range(WINDOW):
                if res[k] > med * 3.0 and res[k] > 30.0:
                    frame_img = self._buffer[i + k][0]
                    self._buffer[i + k] = (frame_img, None, None)

    def _velocity_at(self, idx: int, look_back: int = 3) -> tuple[float, float] | None:
        """Estimate velocity (px/frame) at buffer index idx from the last
        few detections BEFORE and including idx. Returns None if not enough
        data."""
        points = []
        j = idx
        while j >= 0 and len(points) < look_back + 1:
            box = self._buffer[j][1]
            if box is not None:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                points.append((j, cx, cy))
            j -= 1
        if len(points) < 2:
            return None
        # Average velocity across consecutive pairs
        vxs, vys = [], []
        for a, b in zip(points[1:], points[:-1]):
            dt = b[0] - a[0]
            if dt <= 0:
                continue
            vxs.append((b[1] - a[1]) / dt)
            vys.append((b[2] - a[2]) / dt)
        if not vxs:
            return None
        return (sum(vxs) / len(vxs), sum(vys) / len(vys))

    def _fill_gaps(self):
        """Reject outlier spikes, then fill gaps between two stable endpoints.
        Single-sided extrapolation is intentionally not done — projecting the
        last velocity forward when the ball stops being detected was flinging
        the box toward the frame edge on lost tracks.
        """
        self._reject_outliers()
        self._reject_physics_outliers()
        n = len(self._buffer)
        i = 0
        while i < n:
            if self._buffer[i][1] is not None:
                i += 1
                continue
            gap_start = i
            while i < n and self._buffer[i][1] is None:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_len > BALL_INTERP_MAX_GAP:
                continue
            if gap_start == 0 or gap_end >= n:
                continue

            box_before = self._buffer[gap_start - 1][1]
            box_after = self._buffer[gap_end][1]
            if box_before is None or box_after is None:
                continue

            # Gaps only fill when BOTH endpoints are stable real detections.
            # Single-sided extrapolation was flying the box off-screen when a
            # fast ball went undetected (pass end / off-court).
            if not (self._is_stable_before(gap_start - 1)
                    and self._is_stable_after(gap_end)):
                continue

            dist = self._box_dist(box_before, box_after)
            if dist > self._max_teleport * (gap_len + 1):
                continue
            use_arc = gap_len > 3
            for j in range(gap_start, gap_end):
                t = (j - gap_start + 1) / (gap_len + 1)
                if use_arc:
                    interp_box = self._interpolate_box_arc(
                        box_before, box_after, t, gap_len)
                else:
                    interp_box = self._interpolate_box(box_before, box_after, t)
                frame_img = self._buffer[j][0]
                self._buffer[j] = (frame_img, interp_box, None)

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
# Head-Lock Detector — reject detections fused to a player's skull
# ---------------------------------------------------------------------------

class HeadLockDetector:
    """Track whether a ball candidate is rigidly attached to a head keypoint.

    For each player, we record the offset (dx, dy) from their nearest head
    keypoint to the ball center every frame.  If that offset stays nearly
    constant (std-dev < threshold) over BALL_HEADLOCK_WINDOW frames, the
    "ball" is really a part of the player's body (e.g. bald head).

    A real ball near a head — during a shot, pass, or catch — always has
    independent motion so the offset drifts, breaking the lock.
    """

    def __init__(self, frame_w: int):
        self._scale = frame_w / 1920.0
        self._prox = BALL_HEADLOCK_PROX * self._scale
        self._max_var = BALL_HEADLOCK_MAX_VAR * self._scale
        # Per-player offset history: tid → deque of (dx, dy)
        self._offsets: dict[int, deque] = {}

    def is_head_locked(self, bcx: float, bcy: float,
                       pose_map: dict) -> bool:
        """Return True if (bcx, bcy) appears fused to any player's head."""
        active_tids = set()

        for tid, (kps_xy, kps_conf) in pose_map.items():
            # Find the nearest high-confidence head keypoint
            best_dist = self._prox
            best_kp = None
            for kp_idx in BALL_HEADLOCK_HEAD_KPS:
                if kps_conf[kp_idx] < KP_CONF_THRESH:
                    continue
                kx, ky = float(kps_xy[kp_idx][0]), float(kps_xy[kp_idx][1])
                if kx == 0.0 and ky == 0.0:
                    continue  # undetected keypoint
                dist = math.hypot(bcx - kx, bcy - ky)
                if dist < best_dist:
                    best_dist = dist
                    best_kp = (kx, ky)

            if best_kp is None:
                # Ball not near this player's head — clear their history
                self._offsets.pop(tid, None)
                continue

            active_tids.add(tid)
            dx = bcx - best_kp[0]
            dy = bcy - best_kp[1]

            if tid not in self._offsets:
                self._offsets[tid] = deque(maxlen=BALL_HEADLOCK_WINDOW)
            self._offsets[tid].append((dx, dy))

            buf = self._offsets[tid]
            if len(buf) >= BALL_HEADLOCK_WINDOW:
                dxs = [o[0] for o in buf]
                dys = [o[1] for o in buf]
                std = math.hypot(np.std(dxs), np.std(dys))
                if std < self._max_var:
                    return True

        # Prune players no longer relevant
        for tid in list(self._offsets.keys()):
            if tid not in active_tids:
                del self._offsets[tid]

        return False

    def is_near_head(self, bcx: float, bcy: float, pose_map: dict) -> bool:
        """Instant proximity check — no history required. Returns True if
        the point is within BALL_HEADLOCK_PROX of any visible head keypoint.
        Used to guard cold-start jumps (which can't wait 20 frames for
        is_head_locked to trigger)."""
        for tid, (kps_xy, kps_conf) in pose_map.items():
            for kp_idx in BALL_HEADLOCK_HEAD_KPS:
                if kps_conf[kp_idx] < KP_CONF_THRESH:
                    continue
                kx, ky = float(kps_xy[kp_idx][0]), float(kps_xy[kp_idx][1])
                if kx == 0.0 and ky == 0.0:
                    continue
                if math.hypot(bcx - kx, bcy - ky) < self._prox:
                    return True
        return False

    def clear(self):
        """Reset all offset histories (e.g. on track loss)."""
        self._offsets.clear()


# ---------------------------------------------------------------------------
# Ball Validator — reject false positives (off-court, teleports, backboard)
# ---------------------------------------------------------------------------

class BallValidator:
    """Validate ball detections with the following ordered checks:
      1. Court region (polygon column test)
      2. Velocity-aware teleport, with cold-start pathway (guarded by
         head-proximity — cold-start jumps onto a head are rejected)
      3. Bbox size consistency vs recent accepted detections
      4. Backboard constraint (when ball was recently near a hoop)
      5. Best-candidate selection: confidence × trajectory alignment
      6. Trajectory coherence on re-acquisition (velocity-projection),
         falls back to spatial clustering when no velocity is available
      7. Hand/wrist proximity for stationary balls (rejects false
         positives that can't be physically held; only fires when pose
         data for at least one player's wrist is present in-frame)
      8. Head-lock rejection (20-frame rigid-offset detector)
    """

    def __init__(self, roi_poly: np.ndarray, frame_w: int, frame_h: int):
        self._roi_poly = roi_poly
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._last_center: tuple | None = None
        self._last_frame: int = -999
        self._near_hoop_frame: int = -999      # last frame ball was near a hoop
        self._near_hoop_box: list | None = None # the hoop box it was near
        self._scale = frame_w / 1920.0
        self._max_teleport = BALL_MAX_TELEPORT_PX * self._scale
        self._temporal_radius = BALL_TEMPORAL_RADIUS * self._scale
        self._recent_candidates: deque = deque(maxlen=BALL_TEMPORAL_WINDOW)
        self._headlock = HeadLockDetector(frame_w)
        # Velocity tracking: list of (vx, vy) per-frame deltas
        self._velocity_history: deque = deque(maxlen=BALL_VELOCITY_HISTORY)
        # Bbox-size consistency: diagonals of last N accepted boxes
        self._recent_sizes: deque = deque(maxlen=10)
        # Consecutive head-lock hits — only clear state after persistent lock
        self._headlock_consec: int = 0
        # Consecutive bbox-size-band violations — single-frame noise passes
        self._size_violations: int = 0

    def filter(self, frame_idx: int, ball_dets: list,
               hoop_dets: list, pose_map: dict | None = None) -> list:
        """Return only plausible ball detections."""
        speed = self._current_speed()

        # --- Per-detection gates: court, teleport (w/ cold-start), size, backboard ---
        valid = []
        for d in ball_dets:
            box = d["box"]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            # Check 1: Court column
            if not ball_in_court_region(self._roi_poly, cx, cy, self._frame_w):
                continue

            # Check 2: Velocity-aware teleport, with cold-start pathway
            # Standard: allowed = max(base, speed * 2.5) * gap.
            # Cold-start: when velocity is ~0 we have no history to justify a
            # large jump. Pass/shot starts legitimately exceed the base limit.
            # Allow a one-off jump up to cold_start_limit, BUT reject if the
            # landing point is directly on a head keypoint (that is not a
            # pass start — it's a head-lock blip).
            if self._last_center is not None:
                gap = frame_idx - self._last_frame
                dist = math.hypot(cx - self._last_center[0],
                                  cy - self._last_center[1])
                if 0 < gap <= BALL_TELEPORT_GRACE:
                    allowed = max(self._max_teleport, speed * 2.5) * gap
                    if dist > allowed:
                        cold_start = speed < 15.0 * self._scale
                        # Cold-start is a one-off allowance, not a per-frame
                        # budget — cap the absolute limit so gap >=3 can't
                        # permit 1000+ px jumps across the frame.
                        cold_start_limit = min(700.0 * self._scale,
                                               350.0 * self._scale * gap)
                        if not (cold_start and dist <= cold_start_limit):
                            continue
                        # Cold-start head-proximity guard
                        if pose_map and self._headlock.is_near_head(
                                cx, cy, pose_map):
                            continue

            # Check 3: Bbox size consistency (skip until we have baseline).
            # Widened band (2.5x / 0.4x) and single-frame violations pass —
            # only reject on the second consecutive out-of-band hit, so motion
            # blur or partial occlusion doesn't drop a valid track.
            if len(self._recent_sizes) >= 3:
                bw, bh = box[2] - box[0], box[3] - box[1]
                curr_diag = math.hypot(bw, bh)
                sizes_sorted = sorted(self._recent_sizes)
                median_diag = sizes_sorted[len(sizes_sorted) // 2]
                if median_diag > 0 and (
                        curr_diag > median_diag * 2.5 or
                        curr_diag < median_diag * 0.4):
                    self._size_violations += 1
                    if self._size_violations >= 2:
                        continue
                else:
                    self._size_violations = 0

            # Check 4: Backboard constraint (only when recently near hoop)
            if (self._near_hoop_box is not None and
                    frame_idx - self._near_hoop_frame <= 30):
                hoop_top = self._near_hoop_box[1]
                hoop_cx = (self._near_hoop_box[0] + self._near_hoop_box[2]) / 2.0
                court_cx = self._frame_w / 2.0
                above_hoop = cy < hoop_top - self._frame_h * 0.03
                if above_hoop:
                    if hoop_cx < court_cx and cx < hoop_cx:
                        continue
                    elif hoop_cx >= court_cx and cx > hoop_cx:
                        continue

            valid.append(d)

        # --- Best-candidate selection: confidence × trajectory alignment ---
        # When trajectory info is available, prefer detections along the
        # predicted path. When unavailable (cold-start / no velocity), fall
        # back to raw confidence. This lets a lower-conf on-trajectory
        # detection beat a higher-conf off-trajectory head-lock.
        best_det = None
        if valid:
            best_det = max(valid, key=lambda d: self._score(d, frame_idx, speed))
            bcx = (best_det["box"][0] + best_det["box"][2]) / 2.0
            bcy = (best_det["box"][1] + best_det["box"][3]) / 2.0
            self._recent_candidates.append((frame_idx, bcx, bcy))
        else:
            self._recent_candidates.append((frame_idx, None, None))
            return []

        # --- Re-acquisition check: trajectory coherence OR spatial cluster ---
        # Active track (within grace + within allowed motion): accept now.
        # Otherwise require either (a) trajectory-consistent re-acquisition,
        # or (b) spatial cluster of recent candidates for stationary balls.
        continuing_track = False
        if self._last_center is not None:
            gap = frame_idx - self._last_frame
            dist = math.hypot(bcx - self._last_center[0],
                              bcy - self._last_center[1])
            allowed = max(self._max_teleport, speed * 2.5) * gap
            if gap <= BALL_TELEPORT_GRACE and dist <= allowed:
                continuing_track = True

        if not continuing_track:
            have_velocity = (speed > 5.0 * self._scale and
                             self._last_center is not None)
            if have_velocity:
                # Project last known position along average velocity
                avg_vx = sum(v[0] for v in self._velocity_history) / len(self._velocity_history)
                avg_vy = sum(v[1] for v in self._velocity_history) / len(self._velocity_history)
                gap = frame_idx - self._last_frame
                pred_x = self._last_center[0] + avg_vx * gap
                pred_y = self._last_center[1] + avg_vy * gap
                pred_dist = math.hypot(bcx - pred_x, bcy - pred_y)
                # Uncertainty grows with gap. Wider than a rigid projection
                # so rebounds and deflections can be re-acquired — trajectory
                # alignment is a prior, not a hard constraint.
                tolerance = self._max_teleport * max(1.5, gap * 1.0)
                if pred_dist > tolerance:
                    return []
            else:
                # No velocity info — spatial clustering (stationary ball case)
                nearby = 0
                for _, rx, ry in self._recent_candidates:
                    if rx is not None:
                        if math.hypot(bcx - rx, bcy - ry) <= self._temporal_radius:
                            nearby += 1
                if nearby < BALL_TEMPORAL_MIN:
                    return []

        # --- Wrist proximity for stationary balls ---
        # A stationary ball should be near someone's wrist (held). This only
        # fires when: ball is stationary, pose data exists, AND at least one
        # player has a detectable wrist in-frame. Missing pose → skip (don't
        # penalize without evidence).
        if pose_map and speed < 8.0 * self._scale:
            if not self._near_any_wrist(bcx, bcy, pose_map):
                return []

        # --- Head-lock rejection (soft: drop detection, keep state) ---
        # A single head-lock hit is likely a brief face-overlap during a pass
        # — hard-resetting state causes a visible annotation glitch on the
        # next frame. Only clear the track when the lock persists for
        # HEADLOCK_PERSIST_FRAMES consecutive frames.
        if pose_map and self._headlock.is_head_locked(bcx, bcy, pose_map):
            self._headlock_consec += 1
            if self._headlock_consec >= HEADLOCK_PERSIST_FRAMES:
                self._headlock.clear()
                self._last_center = None
                self._last_frame = -999
                self._recent_candidates.clear()
                self._velocity_history.clear()
                self._recent_sizes.clear()
                self._headlock_consec = 0
            return []
        self._headlock_consec = 0

        # --- Accept: update state ---
        if self._last_center is not None:
            gap = frame_idx - self._last_frame
            if 0 < gap <= 3:
                vx = (bcx - self._last_center[0]) / gap
                vy = (bcy - self._last_center[1]) / gap
                self._velocity_history.append((vx, vy))
        else:
            self._velocity_history.clear()

        # Record accepted bbox size
        bw, bh = (best_det["box"][2] - best_det["box"][0],
                  best_det["box"][3] - best_det["box"][1])
        self._recent_sizes.append(math.hypot(bw, bh))

        self._last_center = (bcx, bcy)
        self._last_frame = frame_idx
        for h in hoop_dets:
            if self._near_any_hoop(bcx, bcy, [h]):
                self._near_hoop_frame = frame_idx
                self._near_hoop_box = h["box"]
                break
        return [best_det]

    def _current_speed(self) -> float:
        """Average speed (px/frame) from recent velocity history."""
        if not self._velocity_history:
            return 0.0
        speeds = [math.hypot(vx, vy) for vx, vy in self._velocity_history]
        return sum(speeds) / len(speeds)

    def _score(self, det: dict, frame_idx: int, speed: float) -> float:
        """Score a detection: confidence × trajectory alignment.

        If no trajectory info (no last_center, or no velocity history, or
        velocity too low to project meaningfully) return raw conf. Otherwise
        multiply conf by 0.5-1.0 depending on how well the detection matches
        the predicted position.
        """
        conf = det.get("conf", 0.0)
        if (self._last_center is None or not self._velocity_history or
                speed < 5.0 * self._scale):
            return conf
        avg_vx = sum(v[0] for v in self._velocity_history) / len(self._velocity_history)
        avg_vy = sum(v[1] for v in self._velocity_history) / len(self._velocity_history)
        gap = max(1, frame_idx - self._last_frame)
        pred_x = self._last_center[0] + avg_vx * gap
        pred_y = self._last_center[1] + avg_vy * gap
        cx = (det["box"][0] + det["box"][2]) / 2.0
        cy = (det["box"][1] + det["box"][3]) / 2.0
        dist = math.hypot(cx - pred_x, cy - pred_y)
        tolerance = max(self._max_teleport, speed * 2.5) * gap
        alignment = max(0.0, 1.0 - dist / tolerance) if tolerance > 0 else 0.0
        return conf * (0.5 + 0.5 * alignment)

    def _near_any_wrist(self, bcx: float, bcy: float, pose_map: dict) -> bool:
        """Return True if any player has a visible wrist within proximity,
        OR if NO wrist is detectable in-frame at all (conservative — don't
        penalize when pose data is missing).

        COCO keypoints 9 = left_wrist, 10 = right_wrist.
        """
        WRIST_KPS = (9, 10)
        WRIST_PROX = 80.0 * self._scale
        any_wrist_visible = False
        for _, (kps_xy, kps_conf) in pose_map.items():
            for kp_idx in WRIST_KPS:
                if kps_conf[kp_idx] < KP_CONF_THRESH:
                    continue
                wx, wy = float(kps_xy[kp_idx][0]), float(kps_xy[kp_idx][1])
                if wx == 0.0 and wy == 0.0:
                    continue
                any_wrist_visible = True
                if math.hypot(bcx - wx, bcy - wy) <= WRIST_PROX:
                    return True
        # If no wrist was detectable at all, don't penalize
        return not any_wrist_visible

    def _near_any_hoop(self, cx: float, cy: float, hoop_dets: list) -> bool:
        margin = self._frame_w * 0.08
        for h in hoop_dets:
            hcx = (h["box"][0] + h["box"][2]) / 2.0
            hcy = (h["box"][1] + h["box"][3]) / 2.0
            if abs(cx - hcx) < margin and abs(cy - hcy) < margin:
                return True
        return False


# ---------------------------------------------------------------------------
# Court ROI — polygon check
# ---------------------------------------------------------------------------

def is_court_visible(frame: np.ndarray) -> bool:
    """Detect whether the basketball court is visible in this frame.

    Wood floor color ratio only — the Hough line check (~8ms/frame) was
    removed. Broadcast camera is almost always on court, and the court ROI
    polygon already filters off-court ball detections. HSV check is ~1ms.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4))  # 4× downsample is enough
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    wood_mask = cv2.inRange(hsv, (10, 30, 120), (30, 180, 240))
    wood_ratio = wood_mask.sum() / 255.0 / (wood_mask.shape[0] * wood_mask.shape[1])
    return wood_ratio >= COURT_WOOD_RATIO


def build_court_roi(frame_w: int, frame_h: int) -> np.ndarray:
    """Hardcoded fallback ROI — used when auto-detection fails."""
    pts = np.array([
        [frame_w * 0.13, frame_h * 0.88],
        [frame_w * 0.87, frame_h * 0.88],
        [frame_w * 0.75, frame_h * 0.19],
        [frame_w * 0.25, frame_h * 0.19],
    ], dtype=np.float32)
    return pts.reshape((-1, 1, 2)).astype(np.int32)


def detect_court_roi(cap: cv2.VideoCapture, n_frames: int = 30,
                     frame_w: int = 0, frame_h: int = 0) -> np.ndarray:
    """Auto-detect court ROI from the first N frames using wood floor color.

    Accumulates wood-color masks, finds the largest contour, and returns
    a convex hull polygon. Falls back to hardcoded ROI on failure.
    Rewinds the capture to frame 0 when done.
    """
    orig_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    w = frame_w or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = frame_h or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    accum = np.zeros((h, w), dtype=np.float32)
    count = 0

    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Detect multiple court surface colors:
        # - Wood floor (tan/brown): H=10-30, S=30-180, V=120-240
        # - Blue paint (keys/sidelines): H=90-130, S=40-255, V=80-240
        # - Red/orange paint: H=0-10 or 160-180, S=50-255, V=80-240
        # - Gray/white paint (lines): H=0-180, S=0-40, V=160-255
        wood   = cv2.inRange(hsv, (10, 30, 120), (30, 180, 240))
        blue   = cv2.inRange(hsv, (90, 40, 80), (130, 255, 240))
        red_lo = cv2.inRange(hsv, (0, 50, 80), (10, 255, 240))
        red_hi = cv2.inRange(hsv, (160, 50, 80), (180, 255, 240))
        white  = cv2.inRange(hsv, (0, 0, 160), (180, 40, 255))
        court_mask = wood | blue | red_lo | red_hi | white
        # Exclude very dark regions (crowd/stands tend to be darker)
        accum += (court_mask > 0).astype(np.float32)
        count += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, orig_pos)

    if count == 0:
        print("[court ROI] No frames read, using hardcoded fallback")
        return build_court_roi(w, h)

    # Threshold: pixel is "court" if detected as wood in >40% of sampled frames
    avg = accum / count
    binary = (avg > 0.4).astype(np.uint8) * 255

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[court ROI] No court contour found, using hardcoded fallback")
        return build_court_roi(w, h)

    largest = max(contours, key=cv2.contourArea)
    min_area = w * h * 0.10  # court should be at least 10% of frame
    if cv2.contourArea(largest) < min_area:
        print("[court ROI] Court contour too small, using hardcoded fallback")
        return build_court_roi(w, h)

    hull = cv2.convexHull(largest)
    print(f"[court ROI] Auto-detected from {count} frames "
          f"({len(hull)} hull points, area={cv2.contourArea(hull):.0f}px)")
    return hull


def in_court_roi(roi_poly: np.ndarray, box: list) -> bool:
    foot_x = (box[0] + box[2]) / 2.0
    foot_y = float(box[3])
    return cv2.pointPolygonTest(roi_poly, (foot_x, foot_y), False) >= 0


def ball_in_court_region(roi_poly: np.ndarray, cx: float, cy: float,
                         frame_w: int) -> bool:
    """
    Check if ball is within the horizontal 'box' of the court.
    Ignores height (y) to allow for high-arcing 3-pointers.
    """
    if roi_poly is None or len(roi_poly) < 3:
        return True

    pts = roi_poly.reshape(-1, 2)
    
    # 1. Get the absolute widest points of the court in the frame
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    
    # 2. Get the very bottom of the court to filter out the crowd/floor below
    bot_y = np.max(pts[:, 1])

    # 3. Apply padding
    pad_x = frame_w * BALL_COURT_X_PAD
    pad_y = 50  # pixels of slack below the court
    
    # REJECT if it's below the floor (shoes/referee feet/floor reflections)
    if cy > bot_y + pad_y:
        return False

    # ACCEPT if it is within the horizontal span, no matter how high it is
    # This ensures high arcs are never clipped by perspective tapering
    return (min_x - pad_x) <= cx <= (max_x + pad_x)


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

    # Enable W&B for ultralytics (auto-detected if wandb is installed)
    import os
    os.environ.setdefault("WANDB_PROJECT", os.environ.get("WANDB_PROJECT", "uvambb-cv"))

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
        use_tracknet: bool = True,
        tracknet_weights: str | None = None,
        no_pipeline: bool = False,
        use_yolo_ball: bool = False,
        out_name: str = "annotated"):
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
        infer_imgsz = 960   # 1280 → 960: ~20ms saved per YOLO call, no visible
                             # difference for large objects (player/ref/hoop).
                             # Ball detection uses TrackNet at fixed 640×360.
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

    roi_poly      = detect_court_roi(cap, n_frames=30,
                                      frame_w=frame_w, frame_h=frame_h)
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
        str(Path(out_dir) / f"{out_name}.mp4"), fourcc, fps, (frame_w, frame_h)
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
                tn_dets = tracknet_model.predict(frame)
                if is_court_visible(frame) and tn_dets:
                    tracknet_ball_box = tn_dets[0]["box"]

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
            skip_label = "BALL" if tracknet_ball_box else None
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
                elif cls_id == CLS_BALL and use_yolo_ball:
                    if conf_val >= 0.25:
                        ball_dets.append({"tid": -1, "box": box_list,
                                          "cls": CLS_BALL, "conf": conf_val})
                        ball_source = "YOLO"

        # ---- 1a. Court visibility check ----------------------------------------
        court_visible = is_court_visible(frame)

        # ---- 1b. TrackNet primary ball detection --------------------------------
        # TrackNet uses 3 consecutive frames (temporal context) for heatmap
        # regression — much better at small, fast-moving balls than YOLO.
        # Skip ball detection when court is not visible (closeups/replays).
        if tracknet_model is not None:
            tn_dets = tracknet_model.predict(frame)
            if court_visible and tn_dets:
                ball_dets.extend(tn_dets)
                ball_source = "TrackNet"

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
        if no_pipeline:
            # Raw mode: skip validator / Kalman tracker / interpolator.
            # Draw the highest-conf detection directly.
            ball_to_draw = (sorted(ball_dets, key=lambda d: d.get("conf", 0),
                                   reverse=True)[:1]
                            if ball_dets else [])
        else:
            ball_dets    = ball_validator.filter(frame_idx, ball_dets, hoop_dets,
                                                pose_map=pose_map)
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

        # ---- 7. Ball output (interpolated pipeline or raw) ---------------------
        ball_box = ball_to_draw[0]["box"] if ball_to_draw else None
        last_ball_box = ball_box
        ball_label = f"BALL ({ball_source})" if ball_box else None
        if no_pipeline:
            # Draw immediately — no interpolation buffer
            if ball_box is not None:
                draw_ball(frame, ball_box, label=ball_label or "BALL")
        else:
            # Ball is NOT drawn yet — the interpolator will fill gaps first.
            ball_interp.push(frame, ball_box, ball_label)

        if no_pipeline:
            out_video.write(frame)
        else:
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

    if not no_pipeline:
        # Flush remaining buffered frames
        for fin_frame, fin_ball_box, fin_label in ball_interp.flush():
            if fin_ball_box is not None:
                draw_ball(fin_frame, fin_ball_box, label=fin_label or "BALL (INTERP)")
            out_video.write(fin_frame)

    out_path = Path(out_dir) / f"{out_name}.mp4"
    cap.release()
    out_video.release()
    print(f"\nDone — {frame_idx} frames processed.")
    print(f"  Annotated video : {out_path}")
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
    parser.add_argument("--no-tracknet", action="store_true",
                        help="Disable TrackNet ball detection")
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
            from scripts.convert_labels import convert
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
