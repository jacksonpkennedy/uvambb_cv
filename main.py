"""
UVA Men's Basketball CV Pipeline — Detection + Tracking + Pose
Usage:
  Inference : python main.py [--video data/game_01.mp4] [--out output/]
  Finetune  : python main.py --finetune [--epochs 50] [--batch 8]
  Custom    : python main.py --custom-model runs/detect/train/weights/best.pt --video ...
"""

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
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

LABEL_PLAYER  = "player"

# Custom fine-tuned model class IDs (data.yaml: basketball=0, hoop=1, players=2, referee=3)
CUSTOM_CLS_BALL    = 0
CUSTOM_CLS_HOOP    = 1
CUSTOM_CLS_PLAYER  = 2
CUSTOM_CLS_REF     = 3

# Annotation colours (BGR)
C_PLAYER    = (0,   255,   0)    # green
C_BALL      = (0,   165, 255)    # orange
C_HOOP      = (0,   215, 255)    # gold
C_REF       = (200, 200, 200)    # light grey

# Temporal re-ID
REID_BUFFER_FRAMES = 10
REID_DIST_THRESH   = 150

# Velocity tracker / overlap detection
IOU_CRASH_THRESH = 0.40
VEL_HISTORY_LEN  = 15
VEL_EXIT_THRESH  = 80

# Pose skeleton constants (COCO 17 keypoints)
_G = (0, 255, 0)
SKELETON_EDGES = [
    (0,  1,  _G), (0,  2,  _G), (1,  3,  _G), (2,  4,  _G),
    (5,  6,  _G), (5,  7,  _G), (7,  9,  _G), (6,  8,  _G),
    (8,  10, _G), (5,  11, _G), (6,  12, _G), (11, 12, _G),
    (11, 13, _G), (13, 15, _G), (12, 14, _G), (14, 16, _G),
]
KP_COLORS = [_G] * 17
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
# Temporal Re-ID Buffer (Feature 2)
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
# Velocity Tracker — overlap/eclipse ID preservation (Feature 4)
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
# Court ROI — lightweight polygon check (no HSV colour scoring)
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

def _label_bg(frame, x: int, y: int, text: str, color, scale=0.48, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = 2
    cv2.rectangle(frame, (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_player(frame, box, label: str,
                ghost: bool = False, age: int = 0, max_age: int = 45,
                color=None):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if ghost:
        alpha = max(0.25, 1.0 - age / max(1, max_age))
        ghost_color = (0, int(200 * alpha), int(255 * alpha))
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
            cv2.line(frame, (c[0], c[1]), (c[2], c[3]), ghost_color, 2)
        cv2.putText(frame, f"{label}?", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, ghost_color, 1)
    else:
        draw_color = color if color is not None else C_PLAYER
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)


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


def associate_poses(real_dets: list, ghost_dets: list, pose_result) -> dict:
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


def draw_ball(frame, box, label="BALL"):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BALL, 2, cv2.LINE_AA)
    _label_bg(frame, x1, y1 - 3, label, C_BALL)


def draw_hoop(frame, box):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_HOOP, 2, cv2.LINE_AA)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.drawMarker(frame, (cx, cy), C_HOOP, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    _label_bg(frame, x1, y1 - 3, "HOOP", C_HOOP)


def draw_ref(frame, box, tid: int):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_REF, 1, cv2.LINE_AA)
    _label_bg(frame, x1, y1 - 3, f"REF#{tid}", C_REF)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(epochs: int = 50, batch: int = 8, imgsz: int = 1920):
    base_dir = Path(__file__).resolve().parent
    data_yaml = str(base_dir / "data" / "custom_annotations" / "data.yaml")

    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}\n"
            "Place your Roboflow export in data/custom_annotations/."
        )

    device = get_device()

    print(f"Fine-tuning YOLO11s on custom dataset")
    print(f"  data.yaml : {data_yaml}")
    print(f"  epochs    : {epochs}")
    print(f"  batch     : {batch}")
    print(f"  imgsz     : {imgsz}")
    print(f"  device    : {device}")

    model = YOLO(ensure_model("yolo11s.pt"))
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
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1,
    )

    best_pt = base_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
    print(f"\nFine-tuning complete.")
    print(f"  Best weights: {best_pt}")
    print(f"\nRun the pipeline with:")
    print(f"  python main.py --custom-model {best_pt} --video data/game_01test.mp4")

    return str(best_pt)


# ---------------------------------------------------------------------------
# Main pipeline — detection + tracking + pose (no OCR, no events, no play log)
# ---------------------------------------------------------------------------

def run(video_path: str, out_dir: str, custom_model: str | None = None):
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
    use_custom = custom_model is not None

    if device.startswith("cuda"):
        infer_imgsz = 1920
        use_half    = True
    else:
        infer_imgsz = 640
        use_half    = False
        print(f"NOTE: Running on CPU with imgsz={infer_imgsz}.")

    # --- Load models ----------------------------------------------------------
    if use_custom:
        print(f"Loading CUSTOM model ({custom_model}) — 4-class detection ...")
        unified_model = YOLO(ensure_model(custom_model))
        player_model  = None
        ball_model    = None
    else:
        print("Loading YOLO11s (player tracking) ...")
        player_model = YOLO(ensure_model("yolo11s.pt"))

        print("Loading YOLOE-26 (ball + hoop detection) ...")
        ball_model = YOLOE(ensure_model("yoloe-26l-seg.pt"))
        ball_model.set_classes(YOLOE_PROMPTS, ball_model.get_text_pe(YOLOE_PROMPTS))
        unified_model = None

    print("Loading YOLO11n-pose (skeleton estimation) ...")
    pose_model = YOLO("yolo11n-pose.pt")

    # --- Court ROI (polygon check only, no HSV) ------------------------------
    roi_poly = build_court_roi(frame_w, frame_h)

    # --- State ----------------------------------------------------------------
    ghost_max_age = int(fps * 1.5)
    memory        = TrackMemory(max_age=ghost_max_age)
    reid_buffer   = TemporalReIDBuffer()
    vel_tracker   = VelocityTracker()

    # --- Tracker configs ------------------------------------------------------
    base_dir       = Path(__file__).resolve().parent
    player_tracker = str(base_dir / "bytetrack_players.yaml")
    ball_tracker   = str(base_dir / "bytetrack_custom.yaml")

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

        t_frame_start = time.perf_counter()

        # ---- 1. Detection & tracking -----------------------------------------
        raw_dets       = []
        ball_hoop_dets = []

        if use_custom:
            u_res = unified_model.track(
                frame, persist=True, tracker=player_tracker,
                conf=0.30, iou=0.50,
                imgsz=infer_imgsz, device=device, half=use_half,
                verbose=False,
            )[0]
            if u_res.boxes is not None and u_res.boxes.id is not None:
                for box, tid, cls_id in zip(
                        u_res.boxes.xyxy, u_res.boxes.id, u_res.boxes.cls):
                    box_list = list(map(int, box.tolist()))
                    tid      = int(tid)
                    cls_id   = int(cls_id)

                    if cls_id == CUSTOM_CLS_PLAYER:
                        if (box_list[3] - box_list[1]) < frame_h * 0.08:
                            continue
                        if not in_court_roi(roi_poly, box_list):
                            continue
                        raw_dets.append({
                            "tid": tid, "box": box_list, "label": LABEL_PLAYER,
                        })
                    else:
                        ball_hoop_dets.append({
                            "tid": tid, "box": box_list, "cls": cls_id,
                        })
        else:
            # YOLO11s for players
            p_res = player_model.track(
                frame, persist=True, tracker=player_tracker,
                classes=[0], conf=0.35, iou=0.50,
                imgsz=infer_imgsz, device=device, half=use_half,
                verbose=False,
            )[0]
            if p_res.boxes is not None and p_res.boxes.id is not None:
                for box, tid in zip(p_res.boxes.xyxy, p_res.boxes.id):
                    box_list = list(map(int, box.tolist()))
                    if (box_list[3] - box_list[1]) < frame_h * 0.08:
                        continue
                    if not in_court_roi(roi_poly, box_list):
                        continue
                    raw_dets.append({
                        "tid": int(tid), "box": box_list, "label": LABEL_PLAYER,
                    })

            # YOLOE for ball/hoop/ref
            b_res = ball_model.track(
                frame, persist=True, tracker=ball_tracker,
                conf=0.07, iou=0.70,
                imgsz=infer_imgsz, device=device, half=use_half,
                verbose=False,
            )[0]
            if b_res.boxes is not None and len(b_res.boxes.xyxy) > 0:
                b_ids = b_res.boxes.id
                for i_b, (box, cls_id) in enumerate(
                        zip(b_res.boxes.xyxy, b_res.boxes.cls)):
                    box_list = list(map(int, box.tolist()))
                    b_tid = int(b_ids[i_b]) if (b_ids is not None) else i_b
                    yoloe_label = (YOLOE_PROMPTS[int(cls_id)]
                                   if int(cls_id) < len(YOLOE_PROMPTS) else "?")
                    if yoloe_label == LABEL_BALL:
                        cls_mapped = CUSTOM_CLS_BALL
                    elif yoloe_label == LABEL_HOOP:
                        cls_mapped = CUSTOM_CLS_HOOP
                    elif yoloe_label == LABEL_REF:
                        cls_mapped = CUSTOM_CLS_REF
                    else:
                        continue
                    ball_hoop_dets.append({
                        "tid": b_tid, "box": box_list, "cls": cls_mapped,
                    })

        # ---- 2. Re-ID + memory + velocity ------------------------------------
        raw_dets = reid_buffer.update(frame_idx, raw_dets)
        real_dets, ghost_dets = memory.update(frame_idx, raw_dets)
        reid_buffer.register_lost(frame_idx, ghost_dets)

        vel_remap = vel_tracker.update(frame_idx, real_dets)
        if vel_remap:
            for d in real_dets:
                if d["tid"] in vel_remap:
                    d["tid"] = vel_remap[d["tid"]]

        # ---- 3. Pose estimation ----------------------------------------------
        pose_res = pose_model.predict(
            frame, verbose=False, device=device,
            conf=0.20, imgsz=infer_imgsz, half=use_half,
        )[0]
        pose_map = associate_poses(real_dets, ghost_dets, pose_res)

        # ---- 4. Draw ball / hoop / ref ---------------------------------------
        for det in ball_hoop_dets:
            if det["cls"] == CUSTOM_CLS_BALL:
                draw_ball(frame, det["box"])
            elif det["cls"] == CUSTOM_CLS_HOOP:
                draw_hoop(frame, det["box"])
            elif det["cls"] == CUSTOM_CLS_REF:
                draw_ref(frame, det["box"], det["tid"])

        # ---- 5. Draw real players + skeletons --------------------------------
        for d in real_dets:
            tid = d["tid"]
            draw_player(frame, d["box"], f"#{tid}")
            if tid in pose_map:
                draw_skeleton(frame, *pose_map[tid])

        # ---- 6. Ghost players ------------------------------------------------
        for d in ghost_dets:
            tid = d["tid"]
            draw_player(frame, d["box"], f"#{tid}",
                        ghost=True, age=d["age"], max_age=ghost_max_age)
            if tid in pose_map:
                draw_skeleton(frame, *pose_map[tid])

        # ---- Write frame -----------------------------------------------------
        out_video.write(frame)
        frame_idx += 1

        t_frame = time.perf_counter() - t_frame_start
        if frame_idx <= 5 or frame_idx % 30 == 0:
            print(f"  frame {frame_idx}/{total_frames}  "
                  f"| {t_frame:.2f}s/frame  "
                  f"| players={len(real_dets)}  ghost={len(ghost_dets)}  "
                  f"| poses={len(pose_map)}")

    cap.release()
    out_video.release()

    print(f"\nDone — {frame_idx} frames processed.")
    print(f"  Annotated video : {Path(out_dir) / 'annotated.mp4'}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UVA MBB CV Pipeline")
    parser.add_argument("--video", default="data/game_01test.mp4")
    parser.add_argument("--out",   default="output")
    parser.add_argument("--custom-model", default=None,
                        help="Path to fine-tuned YOLO weights (replaces YOLO11s + YOLOE)")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune YOLO11s on data/custom_annotations/ dataset")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (for --finetune)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (for --finetune)")
    args = parser.parse_args()

    if args.finetune:
        best_pt = finetune(epochs=args.epochs, batch=args.batch)
        run(args.video, args.out, custom_model=best_pt)
    else:
        run(args.video, args.out, custom_model=args.custom_model)
