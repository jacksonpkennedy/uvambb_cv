"""
UVA Men's Basketball CV Pipeline
Usage: python main.py [--video data/game_01.mp4] [--out output/]
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPTS = ["basketball", "player", "hoop", "referee"]

# Court dimensions in feet (NCAA half-court)
COURT_W_FT = 50.0
COURT_H_FT = 47.0

# ---------------------------------------------------------------------------
# Phase 2 helpers — Homography
# ---------------------------------------------------------------------------

def build_homography(frame_w: int, frame_h: int) -> np.ndarray:
    """
    Build H-matrix from pixel court corners → real-world court coords (feet).

    TODO: Replace src_pts with actual detected court keypoints from your footage.
          These defaults assume a standard broadcast angle for game_01.mp4.
    """
    src_pts = np.float32([
        [frame_w * 0.08, frame_h * 0.92],   # bottom-left baseline
        [frame_w * 0.92, frame_h * 0.92],   # bottom-right baseline
        [frame_w * 0.80, frame_h * 0.15],   # top-right sideline
        [frame_w * 0.20, frame_h * 0.15],   # top-left sideline
    ])

    dst_pts = np.float32([
        [0,          COURT_H_FT],   # bottom-left
        [COURT_W_FT, COURT_H_FT],   # bottom-right
        [COURT_W_FT, 0           ],  # top-right
        [0,          0           ],  # top-left
    ])

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H


def pixel_to_court(H: np.ndarray, px: float, py: float):
    """Map a single pixel (px, py) to court coords (x_ft, y_ft)."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    court = cv2.perspectiveTransform(pt, H)
    return float(court[0][0][0]), float(court[0][0][1])


# ---------------------------------------------------------------------------
# Phase 3 helpers — Event Detection
# ---------------------------------------------------------------------------

HOOP_ZONE_FT = {"x": (20, 30), "y": (0, 5)}   # tweak to your court setup
IOU_CRASH_THRESH = 0.40
PASS_VELOCITY_THRESH = 15.0   # ft/s


def bbox_iou(a, b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def in_hoop_zone(cx, cy):
    return (HOOP_ZONE_FT["x"][0] <= cx <= HOOP_ZONE_FT["x"][1] and
            HOOP_ZONE_FT["y"][0] <= cy <= HOOP_ZONE_FT["y"][1])


def detect_events(frame_idx: int, fps: float,
                  ball_history: dict, player_boxes: list, track_ids: list) -> list:
    """
    Emit events based on spatio-temporal triggers.

    ball_history: {frame_idx: (cx_ft, cy_ft)}
    player_boxes: list of [x1,y1,x2,y2] in pixels for current frame
    track_ids:    list of int IDs matching player_boxes
    """
    events = []
    ts = round(frame_idx / fps, 2)

    # --- Crash / Contact: IoU between two player boxes > threshold
    for i in range(len(player_boxes)):
        for j in range(i + 1, len(player_boxes)):
            if bbox_iou(player_boxes[i], player_boxes[j]) >= IOU_CRASH_THRESH:
                events.append({
                    "type": "CRASH",
                    "t": ts,
                    "players": [track_ids[i], track_ids[j]],
                })

    # --- Rebound: ball enters hoop zone then exits into a player box
    frames = sorted(ball_history.keys())
    if len(frames) >= 2:
        prev_pos = ball_history[frames[-2]]
        curr_pos = ball_history[frames[-1]]
        was_in_hoop = in_hoop_zone(*prev_pos)
        now_out = not in_hoop_zone(*curr_pos)
        if was_in_hoop and now_out:
            events.append({"type": "REBOUND", "t": ts})

    # --- Pass: high ball velocity between frames
    if len(frames) >= 2:
        p0 = ball_history[frames[-2]]
        p1 = ball_history[frames[-1]]
        dt = (frames[-1] - frames[-2]) / fps
        if dt > 0:
            vel = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt
            if vel > PASS_VELOCITY_THRESH:
                events.append({"type": "PASS", "t": ts, "velocity_ft_s": round(vel, 1)})

    return events


# ---------------------------------------------------------------------------
# Phase 4 — Simple Play State Machine
# ---------------------------------------------------------------------------

class PlayClassifier:
    """
    State machine: accumulates events and emits play classifications.
    Extend with more rules or swap in a Transformer model.
    """

    def __init__(self):
        self.buffer = []
        self.plays = []
        self.play_id = 0

    def update(self, event: dict):
        self.buffer.append(event)
        self._try_classify()

    def _try_classify(self):
        types = [e["type"] for e in self.buffer]

        # Pick and Roll: SCREEN → PASS or DRIVE sequence
        if "SCREEN" in types and ("PASS" in types or "DRIVE" in types):
            self._emit("Pick and Roll")
            return

        # Fast Break: multiple PASS events in quick succession
        passes = [e for e in self.buffer if e["type"] == "PASS"]
        if len(passes) >= 2:
            dt = passes[-1]["t"] - passes[0]["t"]
            if dt < 4.0:
                self._emit("Fast Break")
                return

        # Flush buffer after 8 s without a classification
        if self.buffer and (self.buffer[-1]["t"] - self.buffer[0]["t"]) > 8.0:
            self.buffer.clear()

    def _emit(self, play_type: str):
        self.play_id += 1
        self.plays.append({
            "play_id": self.play_id,
            "type": play_type,
            "t_start": self.buffer[0]["t"],
            "t_end": self.buffer[-1]["t"],
            "events": list(self.buffer),
        })
        self.buffer.clear()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(video_path: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}  |  {frame_w}x{frame_h}  |  {fps:.1f} fps  |  {total_frames} frames")

    # Phase 1 — Load YOLOE-26
    print("Loading YOLOE-26 ...")
    model = YOLOE("yoloe-26l-seg.pt")   # downloads weights on first run
    model.set_classes(PROMPTS, model.get_text_pe(PROMPTS))

    # Phase 2 — Homography
    H = build_homography(frame_w, frame_h)

    # Phase 3 / 4 — Event + Play state
    ball_history = {}   # {frame_idx: (cx_ft, cy_ft)}
    all_events = []
    classifier = PlayClassifier()

    # Annotated video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        str(Path(out_dir) / "annotated.mp4"), fourcc, fps, (frame_w, frame_h)
    )

    frame_idx = 0
    print("Processing frames ...")

    for result in model.track(video_path, stream=True, tracker="bytetrack.yaml",
                               persist=True, verbose=False):
        frame = result.orig_img.copy()

        player_boxes = []
        track_ids = []

        if result.boxes is not None and result.boxes.id is not None:
            for box, cls_id, tid in zip(result.boxes.xyxy,
                                        result.boxes.cls,
                                        result.boxes.id):
                x1, y1, x2, y2 = map(int, box.tolist())
                tid = int(tid)
                label = PROMPTS[int(cls_id)] if int(cls_id) < len(PROMPTS) else "?"

                cx_px, cy_px = (x1 + x2) / 2, (y1 + y2) / 2
                cx_ft, cy_ft = pixel_to_court(H, cx_px, cy_px)

                if label == "basketball":
                    ball_history[frame_idx] = (cx_ft, cy_ft)
                    cv2.circle(frame, (int(cx_px), int(cy_px)), 10, (0, 140, 255), -1)
                elif label == "player":
                    player_boxes.append([x1, y1, x2, y2])
                    track_ids.append(tid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"#{tid}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Phase 3 — events
        events = detect_events(frame_idx, fps, ball_history, player_boxes, track_ids)
        for ev in events:
            all_events.append(ev)
            classifier.update(ev)
            print(f"  [t={ev['t']:.1f}s] {ev['type']}")

        out_video.write(frame)
        frame_idx += 1

        if frame_idx % 300 == 0:
            print(f"  ... {frame_idx}/{total_frames} frames processed")

    cap.release()
    out_video.release()

    # Export
    play_log = {
        "video": video_path,
        "fps": fps,
        "total_frames": frame_idx,
        "events": all_events,
        "plays": classifier.plays,
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
    parser.add_argument("--video", default="data/game_01.mp4", help="Input video path")
    parser.add_argument("--out",   default="output",           help="Output directory")
    args = parser.parse_args()

    run(args.video, args.out)
