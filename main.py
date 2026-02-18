"""
UVA Men's Basketball CV Pipeline
Usage: python main.py [--video data/game_01.mp4] [--out output/]
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
from ultralytics import YOLO, YOLOE


# ---------------------------------------------------------------------------
# Device + model helpers
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


def ensure_model(model_name: str) -> str:
    """
    Guard against OneDrive Files-On-Demand stubs.
    If the local file can't be read, delete it so ultralytics re-downloads.
    """
    p = Path(model_name)
    if p.exists():
        try:
            with open(p, "rb") as f:
                f.read(4)
        except OSError:
            print(f"WARNING: {model_name} is unreadable (OneDrive stub). Deleting ...")
            p.unlink()
            return p.name
    return model_name


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# YOLOE handles ball, hoop, and referee; players are detected by yolo11s
YOLOE_PROMPTS = ["basketball", "hoop", "referee"]
LABEL_BALL    = "basketball"
LABEL_HOOP    = "hoop"
LABEL_REF     = "referee"

# YOLO11s player label (court persons excluding refs)
LABEL_PLAYER  = "player"

# Annotation colours (BGR)
C_PLAYER = (0,   255,   0)    # green
C_BALL   = (0,   165, 255)    # orange
C_HOOP   = (0,   215, 255)    # gold
C_REF    = (200, 200, 200)    # light grey

# Court dimensions in feet (NCAA half-court)
COURT_W_FT = 50.0
COURT_H_FT = 47.0

# ---------------------------------------------------------------------------
# Pose skeleton constants (COCO 17 keypoints, 0-indexed)
# ---------------------------------------------------------------------------
#  0 nose | 1 L-eye | 2 R-eye | 3 L-ear | 4 R-ear
#  5 L-shoulder | 6 R-shoulder | 7 L-elbow | 8 R-elbow
#  9 L-wrist | 10 R-wrist | 11 L-hip | 12 R-hip
# 13 L-knee | 14 R-knee | 15 L-ankle | 16 R-ankle

# Each entry: (kp_a, kp_b, bgr_color) — all green to match player boxes
_G = (0, 255, 0)
SKELETON_EDGES = [
    (0,  1,  _G),   # nose – L-eye
    (0,  2,  _G),   # nose – R-eye
    (1,  3,  _G),   # L-eye – L-ear
    (2,  4,  _G),   # R-eye – R-ear
    (5,  6,  _G),   # shoulders
    (5,  7,  _G),   # L-shoulder – L-elbow
    (7,  9,  _G),   # L-elbow – L-wrist
    (6,  8,  _G),   # R-shoulder – R-elbow
    (8,  10, _G),   # R-elbow – R-wrist
    (5,  11, _G),   # L-shoulder – L-hip
    (6,  12, _G),   # R-shoulder – R-hip
    (11, 12, _G),   # hips
    (11, 13, _G),   # L-hip – L-knee
    (13, 15, _G),   # L-knee – L-ankle
    (12, 14, _G),   # R-hip – R-knee
    (14, 16, _G),   # R-knee – R-ankle
]

KP_COLORS = [_G] * 17  # all keypoints same green

KP_CONF_THRESH = 0.30    # skip keypoints below this confidence

# Ball trail spike guard — if the ball was missing for more than this many
# frames before reappearing, the trail is cleared.  A real fast pass is
# tracked continuously; a gap means detector loss, not actual movement.
BALL_TRAIL_GAP_THRESH = 3   # frames

# Max plausible ball displacement between consecutive detections (pixels).
# Detections that jump farther than this are treated as false positives and
# cause the trail to be cleared rather than connected.
BALL_MAX_JUMP_PX = 150

# Court-visibility check — HSV colour ranges for hardwood floor and court lines.
# If the fraction of pixels inside the court ROI that match either range falls
# below COURT_COLOR_THRESH the frame is treated as a non-court shot (crowd,
# replay, close-up, timeout, etc.) and all analysis is suppressed.
_COURT_HSV_FLOOR_LO = np.array([  8,  15, 100], dtype=np.uint8)  # warm tan/maple
_COURT_HSV_FLOOR_HI = np.array([ 40, 200, 245], dtype=np.uint8)
_COURT_HSV_LINE_LO  = np.array([  0,   0, 175], dtype=np.uint8)  # white markings
_COURT_HSV_LINE_HI  = np.array([180,  65, 255], dtype=np.uint8)
COURT_COLOR_THRESH  = 0.20   # 20 % of ROI pixels must look like court

# ---------------------------------------------------------------------------
# Jersey Memory — OCR voting per track ID
# ---------------------------------------------------------------------------

OCR_INTERVAL  = 30    # minimum frames between OCR attempts per track
OCR_MIN_VOTES = 2     # votes needed to confirm a jersey number


class JerseyMemory:
    """
    Associates jersey numbers with ByteTrack IDs via OCR voting.

    Every OCR_INTERVAL frames, crops the chest region of each player,
    runs EasyOCR (digits only), and casts a vote.  Once OCR_MIN_VOTES
    consistent reads are accumulated, the number is "confirmed" and
    shown in the annotation instead of the raw track ID.
    """

    def __init__(self, reader: easyocr.Reader):
        self._reader    = reader
        self._votes:     dict = {}   # tid → {number_str: count}
        self._confirmed: dict = {}   # tid → confirmed number_str
        self._last_ocr:  dict = {}   # tid → frame_idx of last attempt

    # ------------------------------------------------------------------
    def maybe_ocr(self, frame_idx: int, tid: int,
                  frame: np.ndarray, box: list):
        """Run OCR on this player's crop if enough frames have passed."""
        if frame_idx - self._last_ocr.get(tid, -OCR_INTERVAL - 1) < OCR_INTERVAL:
            return
        self._last_ocr[tid] = frame_idx

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        h, w = y2 - y1, x2 - x1
        if h < 24 or w < 12:
            return

        # Crop the chest/number region: centre width strip, upper-middle height
        cx1 = max(0, x1 + w // 5)
        cx2 = min(frame.shape[1], x2 - w // 5)
        cy1 = max(0, y1 + int(h * 0.20))
        cy2 = max(0, y1 + int(h * 0.65))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return

        # Upscale tiny crops so OCR has enough pixels to work with
        min_dim = min(crop.shape[:2])
        if min_dim < 40:
            scale = max(2, 60 // min_dim)
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        results = self._reader.readtext(
            crop,
            allowlist="0123456789",
            detail=1,
            paragraph=False,
        )
        for (_, text, conf) in results:
            text = text.strip()
            if text and 1 <= len(text) <= 2 and conf >= 0.50:
                self._cast_vote(tid, text)
                break

    # ------------------------------------------------------------------
    def _cast_vote(self, tid: int, number: str):
        votes = self._votes.setdefault(tid, {})
        votes[number] = votes.get(number, 0) + 1
        best = max(votes, key=votes.get)
        if votes[best] >= OCR_MIN_VOTES:
            self._confirmed[tid] = best

    # ------------------------------------------------------------------
    def label(self, tid: int) -> str:
        """Return '#23' if confirmed, else '#<track_id>'."""
        return f"#{self._confirmed[tid]}" if tid in self._confirmed else f"#{tid}"


# ---------------------------------------------------------------------------
# Track Memory — persist IDs across brief occlusions / missed detections
# ---------------------------------------------------------------------------

class TrackMemory:
    """
    Re-emits recently-seen tracks as ghost detections when the detector
    misses them for a few frames.  Ghosts are drawn but NOT used for
    collision event detection.
    """

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
                "cx_ft": d["cx_ft"],
                "cy_ft": d["cy_ft"],
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
                    "cx_ft": info["cx_ft"],
                    "cy_ft": info["cy_ft"],
                    "ghost": True,
                    "age":   age,
                })
            else:
                dead.append(tid)

        for tid in dead:
            del self._tracks[tid]

        return detections, ghost_dets


# ---------------------------------------------------------------------------
# Phase 2 helpers — Homography
# ---------------------------------------------------------------------------

def build_homography(frame_w: int, frame_h: int) -> np.ndarray:
    """
    Pixel court corners → real-world court coords (feet).

    Coordinate system:
      x : sideline-to-sideline 0–50 ft (left → right)
      y : 0 = near baseline (bottom of frame), 47 = half-court line (top)

    TODO: Replace src_pts with actual keypoints from your footage.
    """
    src_pts = np.float32([
        [frame_w * 0.08, frame_h * 0.92],
        [frame_w * 0.92, frame_h * 0.92],
        [frame_w * 0.80, frame_h * 0.15],
        [frame_w * 0.20, frame_h * 0.15],
    ])
    dst_pts = np.float32([
        [0,          0         ],
        [COURT_W_FT, 0         ],
        [COURT_W_FT, COURT_H_FT],
        [0,          COURT_H_FT],
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H


def pixel_to_court(H: np.ndarray, px: float, py: float):
    pt = np.array([[[px, py]]], dtype=np.float32)
    court = cv2.perspectiveTransform(pt, H)
    return float(court[0][0][0]), float(court[0][0][1])


# ---------------------------------------------------------------------------
# Court ROI — crowd filter
# ---------------------------------------------------------------------------

def build_court_roi(frame_w: int, frame_h: int) -> np.ndarray:
    """
    Returns a polygon (int32, shape [4,1,2]) covering the *inbounds* court area.
    Inset from build_homography corners so sideline coaches and front-row fans
    whose feet touch just outside the boundary lines are excluded.

    Calibrate together with build_homography when you have real footage.
    """
    pts = np.array([
        [frame_w * 0.13, frame_h * 0.88],   # bottom-left  (inset from 0.08, 0.92)
        [frame_w * 0.87, frame_h * 0.88],   # bottom-right (inset from 0.92, 0.92)
        [frame_w * 0.75, frame_h * 0.19],   # top-right    (inset from 0.80, 0.15)
        [frame_w * 0.25, frame_h * 0.19],   # top-left     (inset from 0.20, 0.15)
    ], dtype=np.float32)
    return pts.reshape((-1, 1, 2)).astype(np.int32)


# Tolerance in feet around the inbounds court area.
# Keeps players near the boundary lines; rejects coaches (further outside).
_COURT_TOL_FT = 1.5


def in_court_roi(roi_poly: np.ndarray, box: list) -> bool:
    """Return True if the foot position (bottom-center) of bbox is inside the court polygon."""
    foot_x = (box[0] + box[2]) / 2.0
    foot_y = float(box[3])   # bottom edge ≈ foot position
    return cv2.pointPolygonTest(roi_poly, (foot_x, foot_y), False) >= 0


def court_color_score(frame: np.ndarray, roi_poly: np.ndarray) -> float:
    """
    Returns the fraction of pixels inside the court ROI that match either
    hardwood-floor tones or white court-line tones.

    A score >= COURT_COLOR_THRESH means the court floor is likely in view.
    Crowd shots, replays, and close-ups score well below the threshold because
    they lack the distinctive maple-floor colour and bright white markings.
    """
    h, w = frame.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_poly], 255)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    floor_mask = cv2.inRange(hsv, _COURT_HSV_FLOOR_LO, _COURT_HSV_FLOOR_HI)
    line_mask  = cv2.inRange(hsv, _COURT_HSV_LINE_LO,  _COURT_HSV_LINE_HI)
    court_px   = cv2.bitwise_and(cv2.bitwise_or(floor_mask, line_mask), roi_mask)

    roi_total = int(np.count_nonzero(roi_mask))
    if roi_total == 0:
        return 0.0
    return int(np.count_nonzero(court_px)) / roi_total


# ---------------------------------------------------------------------------
# Phase 3 helpers — Event Detection
# ---------------------------------------------------------------------------

HOOP_ZONE_FT         = {"x": (20.0, 30.0), "y": (0.0, 6.0)}
IOU_CRASH_THRESH      = 0.40
PASS_VELOCITY_THRESH  = 15.0
PASS_MIN_DT           = 0.50


def bbox_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def in_hoop_zone(cx, cy):
    return (HOOP_ZONE_FT["x"][0] <= cx <= HOOP_ZONE_FT["x"][1] and
            HOOP_ZONE_FT["y"][0] <= cy <= HOOP_ZONE_FT["y"][1])


def detect_events(frame_idx, fps, ball_prev, ball_curr,
                  real_player_boxes, real_track_ids, ball_hoop_dwell):
    events = []
    ts = round(frame_idx / fps, 2)

    for i in range(len(real_player_boxes)):
        for j in range(i + 1, len(real_player_boxes)):
            if bbox_iou(real_player_boxes[i], real_player_boxes[j]) >= IOU_CRASH_THRESH:
                events.append({"type": "CRASH", "t": ts,
                                "players": [real_track_ids[i], real_track_ids[j]]})

    if ball_prev is not None and ball_curr is not None:
        prev_fidx, prev_cx, prev_cy = ball_prev
        curr_fidx, curr_cx, curr_cy = ball_curr
        dt = (curr_fidx - prev_fidx) / fps

        if ball_hoop_dwell >= 3 and not in_hoop_zone(curr_cx, curr_cy):
            events.append({"type": "REBOUND", "t": ts})

        if dt >= PASS_MIN_DT:
            vel = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy) / dt
            if vel > PASS_VELOCITY_THRESH:
                events.append({"type": "PASS", "t": ts,
                                "velocity_ft_s": round(vel, 1)})

    return events


# ---------------------------------------------------------------------------
# Phase 4 — Simple Play State Machine
# ---------------------------------------------------------------------------

class PlayClassifier:
    def __init__(self):
        self.buffer  = []
        self.plays   = []
        self.play_id = 0

    def update(self, event):
        self.buffer.append(event)
        self._try_classify()

    def _try_classify(self):
        types = [e["type"] for e in self.buffer]

        if "SCREEN" in types and ("PASS" in types or "DRIVE" in types):
            self._emit("Pick and Roll")
            return

        passes = [e for e in self.buffer if e["type"] == "PASS"]
        if len(passes) >= 2 and passes[-1]["t"] - passes[0]["t"] < 4.0:
            self._emit("Fast Break")
            return

        if (len(self.buffer) >= 2 and
                self.buffer[-1]["t"] - self.buffer[-2]["t"] > 8.0):
            self.buffer.clear()

    def _emit(self, play_type):
        self.play_id += 1
        self.plays.append({
            "play_id": self.play_id,
            "type":    play_type,
            "t_start": self.buffer[0]["t"],
            "t_end":   self.buffer[-1]["t"],
            "events":  list(self.buffer),
        })
        self.buffer.clear()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_skeleton(frame: np.ndarray,
                  kps_xy: np.ndarray,
                  kps_conf: np.ndarray):
    """
    Draw COCO-17 skeleton on frame.

    kps_xy   : (17, 2) float array of pixel coordinates
    kps_conf : (17,)   float array of per-keypoint confidence scores
    """
    # Bones
    for (i, j, color) in SKELETON_EDGES:
        if kps_conf[i] >= KP_CONF_THRESH and kps_conf[j] >= KP_CONF_THRESH:
            x1, y1 = int(kps_xy[i][0]), int(kps_xy[i][1])
            x2, y2 = int(kps_xy[j][0]), int(kps_xy[j][1])
            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Keypoints
    for k in range(min(len(kps_xy), 17)):
        if kps_conf[k] >= KP_CONF_THRESH:
            x, y = int(kps_xy[k][0]), int(kps_xy[k][1])
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 4, KP_COLORS[k], -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 4, (0, 0, 0),    1,  cv2.LINE_AA)


def associate_poses(real_dets: list, ghost_dets: list,
                    pose_result) -> dict:
    """
    Match each pose detection to the player track with the highest IoU.
    Returns {tid: (kps_xy np.ndarray, kps_conf np.ndarray)}.
    """
    associations = {}

    if (pose_result is None
            or pose_result.keypoints is None
            or pose_result.boxes is None
            or len(pose_result.boxes.xyxy) == 0):
        return associations

    p_boxes  = pose_result.boxes.xyxy.cpu().numpy()          # (N, 4)
    kps_xy   = pose_result.keypoints.xy.cpu().numpy()        # (N, 17, 2)

    kps_conf_raw = pose_result.keypoints.conf
    if kps_conf_raw is not None:
        kps_conf = kps_conf_raw.cpu().numpy()                # (N, 17)
    else:
        kps_conf = np.ones((len(kps_xy), 17), dtype=np.float32)

    # Build unified track list from real + ghost player detections
    track_candidates = [
        (d["box"], d["tid"])
        for d in (real_dets + ghost_dets)
        if d["label"] == LABEL_PLAYER
    ]

    for i, p_box in enumerate(p_boxes):
        best_iou = 0.12
        best_tid = None
        for (y_box, tid) in track_candidates:
            iou = bbox_iou(p_box.tolist(), y_box)
            if iou > best_iou:
                best_iou = iou
                best_tid  = tid
        if best_tid is not None:
            associations[best_tid] = (kps_xy[i], kps_conf[i])

    return associations


def draw_player(frame, box, label: str,
                ghost: bool = False, age: int = 0, max_age: int = 45):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if ghost:
        alpha = max(0.25, 1.0 - age / max(1, max_age))
        color = (0, int(200 * alpha), int(255 * alpha))
        seg = 12
        corners = [
            (x1,       y1,       x1 + seg, y1      ),
            (x2 - seg, y1,       x2,       y1      ),
            (x1,       y2,       x1 + seg, y2      ),
            (x2 - seg, y2,       x2,       y2      ),
            (x1,       y1,       x1,       y1 + seg),
            (x1,       y2 - seg, x1,       y2      ),
            (x2,       y1,       x2,       y1 + seg),
            (x2,       y2 - seg, x2,       y2      ),
        ]
        for c in corners:
            cv2.line(frame, (c[0], c[1]), (c[2], c[3]), color, 2)
        cv2.putText(frame, f"{label}?", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# Drawing helpers — ball, hoop, trail
# ---------------------------------------------------------------------------

def _label_bg(frame, x: int, y: int, text: str, color, scale=0.48, thick=1):
    """Draw text over a tight dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = 2
    cv2.rectangle(frame, (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_ball(frame, box, label="BALL"):
    """Orange bounding box + label for the basketball."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BALL, 2, cv2.LINE_AA)
    _label_bg(frame, x1, y1 - 3, label, C_BALL)


def draw_ball_ghost(frame, box, age: int, ghost_limit: int):
    """Dashed corner-bracket ghost box for ball when temporarily missed."""
    x1, y1, x2, y2 = map(int, box)
    alpha = max(0.20, 1.0 - age / max(1, ghost_limit))
    color = (0, int(165 * alpha), int(255 * alpha))
    seg = 10
    for c in [(x1, y1, x1+seg, y1), (x2-seg, y1, x2, y1),
              (x1, y2, x1+seg, y2), (x2-seg, y2, x2, y2),
              (x1, y1, x1, y1+seg), (x1, y2-seg, x1, y2),
              (x2, y1, x2, y1+seg), (x2, y2-seg, x2, y2)]:
        cv2.line(frame, c[:2], c[2:], color, 2, cv2.LINE_AA)


def draw_hoop(frame, box):
    """Gold bounding box + crosshair for the hoop/net."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_HOOP, 2, cv2.LINE_AA)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.drawMarker(frame, (cx, cy), C_HOOP, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    _label_bg(frame, x1, y1 - 3, "HOOP", C_HOOP)


def draw_ref(frame, box, tid: int):
    """Grey bounding box for referees."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_REF, 1, cv2.LINE_AA)
    _label_bg(frame, x1, y1 - 3, f"REF#{tid}", C_REF)


def draw_trail(frame, trail: deque):
    """Fading orange polyline tracing the ball's path."""
    pts = list(trail)
    n   = len(pts)
    if n < 2:
        return
    for i in range(1, n):
        alpha = i / n                              # brightest at newest point
        color = (0, int(100 * alpha), int(255 * alpha))
        cv2.line(frame, pts[i - 1], pts[i], color,
                 max(1, int(3 * alpha)), cv2.LINE_AA)
    # Bright dot at the head of the trail
    cv2.circle(frame, pts[-1], 4, C_BALL, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(video_path: str, out_dir: str):
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

    # --- Load models ----------------------------------------------------------
    print("Loading YOLO11s (player tracking) ...")
    player_model = YOLO(ensure_model("yolo11s.pt"))

    print("Loading YOLOE-26 (ball + hoop detection) ...")
    ball_model = YOLOE(ensure_model("yoloe-26l-seg.pt"))
    ball_model.set_classes(YOLOE_PROMPTS, ball_model.get_text_pe(YOLOE_PROMPTS))

    print("Loading YOLO11n-pose (skeleton estimation) ...")
    pose_model = YOLO("yolo11n-pose.pt")

    print("Loading EasyOCR (jersey number recognition) ...")
    ocr_reader = easyocr.Reader(
        ["en"],
        gpu=device.startswith("cuda"),
        verbose=False,
    )

    # --- Homography + court ROI -----------------------------------------------
    H        = build_homography(frame_w, frame_h)
    roi_poly = build_court_roi(frame_w, frame_h)

    # --- State ----------------------------------------------------------------
    ball_prev        = None
    ball_curr        = None
    ball_px          = None        # last-known pixel centre (cx, cy)
    ball_box         = None        # last-known bounding box [x1,y1,x2,y2]
    ball_trail       = deque(maxlen=int(fps * 0.75))  # ~45 frames @ 60 fps
    ball_last_frame  = -9999
    ball_hoop_dwell  = 0
    ball_stale_limit = int(fps * 2.0)
    ball_ghost_limit = int(fps * 1.0)  # keep ghost ball for 1 s

    all_events = []
    classifier = PlayClassifier()

    ghost_max_age = int(fps * 1.5)
    memory        = TrackMemory(max_age=ghost_max_age)
    jersey_mem    = JerseyMemory(ocr_reader)

    # Court visibility: if no court players are detected for this many frames
    # the camera has likely cut away (crowd shot, replay, timeout close-up).
    # All annotations are suppressed until players reappear.
    COURT_EMPTY_THRESH = int(fps * 1.0)   # 1 s of emptiness = off-court view
    court_empty_frames = 0

    # --- Tracker config paths -------------------------------------------------
    base_dir         = Path(__file__).resolve().parent
    player_tracker   = str(base_dir / "bytetrack_players.yaml")
    ball_tracker     = str(base_dir / "bytetrack_custom.yaml")

    # --- Output writer --------------------------------------------------------
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

        # ---- 1. Player tracking — YOLO11s (person class, court ROI filter) --
        p_res = player_model.track(
            frame, persist=True, tracker=player_tracker,
            classes=[0], conf=0.35, iou=0.50,
            imgsz=1920, device=device, half=True,
            verbose=False,
        )[0]

        raw_dets = []
        if p_res.boxes is not None and p_res.boxes.id is not None:
            for box, tid in zip(p_res.boxes.xyxy, p_res.boxes.id):
                box_list = list(map(int, box.tolist()))
                # Crowd members in the stands appear much smaller than court players.
                # Require box height >= 8% of frame height to reject distant spectators.
                if (box_list[3] - box_list[1]) < frame_h * 0.08:
                    continue
                # Skip anyone whose foot position falls outside the court polygon.
                if not in_court_roi(roi_poly, box_list):
                    continue
                tid    = int(tid)
                cx_px  = (box_list[0] + box_list[2]) / 2
                cy_px  = (box_list[1] + box_list[3]) / 2
                cx_ft, cy_ft = pixel_to_court(H, cx_px, cy_px)
                # Reject anyone mapped outside the inbounds court area — coaches
                # and front-row fans project beyond the court boundary lines.
                if not (-_COURT_TOL_FT <= cx_ft <= COURT_W_FT + _COURT_TOL_FT and
                        -_COURT_TOL_FT <= cy_ft <= COURT_H_FT + _COURT_TOL_FT):
                    continue
                raw_dets.append({
                    "tid": tid, "box": box_list,
                    "label": LABEL_PLAYER, "cx_ft": cx_ft, "cy_ft": cy_ft,
                })

        real_dets, ghost_dets = memory.update(frame_idx, raw_dets)

        # Court visibility — two independent checks, both must pass:
        #   1. Colour check: enough court-floor / white-line pixels in the ROI
        #      → instantly rejects crowd shots, replays, and close-ups.
        #   2. Player-presence check: no court players seen for > COURT_EMPTY_THRESH
        #      → rejects sustained non-court periods even if floor tones bleed through.
        court_floor_visible = court_color_score(frame, roi_poly) >= COURT_COLOR_THRESH
        if raw_dets:
            court_empty_frames = 0
        else:
            court_empty_frames += 1
        court_visible = court_floor_visible and (court_empty_frames < COURT_EMPTY_THRESH)

        # ---- 2. Pose estimation on current frame ----------------------------
        pose_res = pose_model.predict(
            frame, verbose=False, device=device,
            conf=0.20, imgsz=1920, half=False,
        )[0]
        pose_map = associate_poses(real_dets, ghost_dets, pose_res)

        if court_visible:
            # ---- 3. Ball + hoop — YOLOE -------------------------------------
            b_res = ball_model.track(
                frame, persist=True, tracker=ball_tracker,
                conf=0.07, iou=0.70,
                imgsz=1920, device=device, half=False,
                verbose=False,
            )[0]

            ball_detected = False
            if b_res.boxes is not None and len(b_res.boxes.xyxy) > 0:
                b_ids = b_res.boxes.id  # track IDs (may be None first frame)
                for i_b, (box, cls_id) in enumerate(
                        zip(b_res.boxes.xyxy, b_res.boxes.cls)):
                    label    = (YOLOE_PROMPTS[int(cls_id)]
                                if int(cls_id) < len(YOLOE_PROMPTS) else "?")
                    box_list = list(map(int, box.tolist()))
                    cx_px    = (box_list[0] + box_list[2]) / 2
                    cy_px    = (box_list[1] + box_list[3]) / 2
                    cx_ft, cy_ft = pixel_to_court(H, cx_px, cy_px)
                    b_tid = int(b_ids[i_b]) if (b_ids is not None) else i_b

                    if label == LABEL_BALL:
                        ball_detected        = True
                        prev_ball_last_frame = ball_last_frame  # save before overwrite
                        ball_last_frame      = frame_idx
                        if (ball_curr is None or
                                (frame_idx - ball_curr[0]) / fps >= PASS_MIN_DT):
                            ball_prev = ball_curr
                        ball_curr = (frame_idx, cx_ft, cy_ft)
                        new_px    = (int(cx_px), int(cy_px))
                        # Clear trail when ball was missing for several frames —
                        # the gap is detector loss, not real movement.
                        if (frame_idx - prev_ball_last_frame) > BALL_TRAIL_GAP_THRESH:
                            ball_trail.clear()
                        # Clear trail when the ball jumps an implausible distance
                        # between detections — almost certainly a false positive.
                        elif (ball_px is not None and
                              math.hypot(new_px[0] - ball_px[0],
                                         new_px[1] - ball_px[1]) > BALL_MAX_JUMP_PX):
                            ball_trail.clear()
                        ball_px  = new_px
                        ball_box = box_list
                        ball_trail.append(ball_px)
                        if in_hoop_zone(cx_ft, cy_ft):
                            ball_hoop_dwell += 1
                        else:
                            ball_hoop_dwell = 0
                        draw_ball(frame, ball_box)

                    elif label == LABEL_HOOP:
                        draw_hoop(frame, box_list)

                    elif label == LABEL_REF:
                        draw_ref(frame, box_list, b_tid)

            # Ghost ball
            if not ball_detected and ball_box is not None:
                age = frame_idx - ball_last_frame
                if age <= ball_ghost_limit:
                    draw_ball_ghost(frame, ball_box, age, ball_ghost_limit)

            # Ball trajectory trail (drawn over everything)
            draw_trail(frame, ball_trail)

            # Stale ball reset
            if not ball_detected and (frame_idx - ball_last_frame) > ball_stale_limit:
                ball_prev = ball_curr = ball_px = ball_box = None
                ball_trail.clear()
                ball_hoop_dwell = 0

            # ---- 4. Draw real players ---------------------------------------
            real_player_boxes = []
            real_track_ids    = []

            for d in real_dets:
                tid = d["tid"]
                real_player_boxes.append(d["box"])
                real_track_ids.append(tid)

                jersey_mem.maybe_ocr(frame_idx, tid, frame, d["box"])
                lbl = jersey_mem.label(tid)

                draw_player(frame, d["box"], lbl)

                if tid in pose_map:
                    draw_skeleton(frame, *pose_map[tid])

            # ---- 5. Ghost players (visual only) -----------------------------
            for d in ghost_dets:
                tid = d["tid"]
                lbl = jersey_mem.label(tid)
                draw_player(frame, d["box"], lbl,
                            ghost=True, age=d["age"], max_age=ghost_max_age)
                if tid in pose_map:
                    draw_skeleton(frame, *pose_map[tid])

            # ---- 6. Event detection -----------------------------------------
            events = detect_events(frame_idx, fps, ball_prev, ball_curr,
                                   real_player_boxes, real_track_ids,
                                   ball_hoop_dwell)
            for ev in events:
                all_events.append(ev)
                classifier.update(ev)
                print(f"  [t={ev['t']:.1f}s] {ev['type']}")

        out_video.write(frame)
        frame_idx += 1

        if frame_idx % 300 == 0:
            n_ghost = len(ghost_dets)
            print(f"  ... {frame_idx}/{total_frames} frames  "
                  f"| real={len(real_player_boxes)}  ghost={n_ghost}  "
                  f"| poses={len(pose_map)}")

    cap.release()
    out_video.release()

    play_log = {
        "video":        video_path,
        "fps":          fps,
        "total_frames": frame_idx,
        "events":       all_events,
        "plays":        classifier.plays,
    }
    log_path = Path(out_dir) / "play_log.json"
    with open(log_path, "w") as f:
        json.dump(play_log, f, indent=2)

    print(f"\nDone.")
    print(f"  Annotated video : {Path(out_dir) / 'annotated.mp4'}")
    print(f"  Play log        : {log_path}")
    print(f"  Events detected : {len(all_events)}")
    print(f"  Plays classified: {len(classifier.plays)}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UVA MBB CV Pipeline")
    parser.add_argument("--video", default="data/game_01test.mp4")
    parser.add_argument("--out",   default="output")
    args = parser.parse_args()
    run(args.video, args.out)
