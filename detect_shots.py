"""
Shot Detection Script
Reads output/tracking.json and output video, detects shot attempts based on
ball trajectory, speed, arc, and proximity to hoop.

Usage:
  python detect_shots.py --tracking output/tracking.json --video run2.mp4 --out shot_output.mp4
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters — tweak these to tune shot detection
# ---------------------------------------------------------------------------

# Minimum ball speed (pixels/frame) to consider the ball "in flight"
MIN_SPEED = 4.0

# Minimum number of consecutive frames the ball must be tracked as "in flight"
MIN_FLIGHT_FRAMES = 10

# Maximum distance (pixels) from ball to hoop center at peak approach
# to count as a shot attempt
MAX_HOOP_DIST = 120

# Minimum upward velocity (negative y, since y increases downward) that must
# appear at some point during the trajectory (arc requirement)
MIN_UPWARD_VY = -3.0   # ball must move up by at least this many px/frame

# The ball must spend some time moving upward before it can approach the hoop
# (filters out passes that happen to pass near the hoop)
MIN_UPWARD_FRAMES = 3

# How many frames before the closest-approach frame to look for upward motion
ARC_LOOKBACK_FRAMES = 60

# Minimum horizontal travel (pixels) during the flight — avoids false positives
# from a stationary ball jiggling near the hoop
MIN_HORIZONTAL_TRAVEL = 30

# Cooldown (frames) after a shot is detected before another can be registered
SHOT_COOLDOWN_FRAMES = 90

# Overlay display duration (frames)
OVERLAY_DURATION_FRAMES = 90

# ---------------------------------------------------------------------------


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_tracking(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_ball_trajectory(frames: dict) -> list[dict]:
    """Return sorted list of {frame_id, center, hoop_center} for frames with ball."""
    traj = []
    for fid_str, fdata in frames.items():
        if fdata["ball"] is None:
            continue
        ball_cx, ball_cy = fdata["ball"]["center"]
        hoop_center = None
        if fdata["hoops"]:
            hoop_center = fdata["hoops"][0]["center"]
        traj.append({
            "frame_id": int(fid_str),
            "bx": ball_cx,
            "by": ball_cy,
            "hoop": hoop_center,
        })
    traj.sort(key=lambda x: x["frame_id"])
    return traj


def find_shots(traj: list[dict]) -> list[dict]:
    """
    Scan the trajectory for shot attempts.

    A shot is a window of consecutive (or near-consecutive) ball detections
    where:
      1. Ball moves fast enough (MIN_SPEED avg px/frame)
      2. Ball travels upward at some point (arc)
      3. Ball comes within MAX_HOOP_DIST pixels of the hoop
      4. Sufficient horizontal travel occurred
    """
    shots = []
    last_shot_frame = -SHOT_COOLDOWN_FRAMES

    n = len(traj)
    i = 0
    while i < n:
        # Slide a window starting at i
        window = [traj[i]]
        j = i + 1
        while j < n:
            gap = traj[j]["frame_id"] - traj[j - 1]["frame_id"]
            if gap > 5:   # allow up to 5-frame gap in detection (occlusion)
                break
            window.append(traj[j])
            j += 1

        if len(window) >= MIN_FLIGHT_FRAMES:
            shot = evaluate_window(window, last_shot_frame)
            if shot:
                shots.append(shot)
                last_shot_frame = shot["peak_frame"]

        i = j if j > i + 1 else i + 1

    return shots


def evaluate_window(window: list[dict], last_shot_frame: int) -> dict | None:
    """Check if this continuous ball-tracking window contains a shot."""
    if len(window) < MIN_FLIGHT_FRAMES:
        return None

    # Compute per-frame velocities
    vx_list, vy_list, speeds = [], [], []
    for k in range(1, len(window)):
        dt = window[k]["frame_id"] - window[k - 1]["frame_id"]
        if dt == 0:
            continue
        vx = (window[k]["bx"] - window[k - 1]["bx"]) / dt
        vy = (window[k]["by"] - window[k - 1]["by"]) / dt
        vx_list.append(vx)
        vy_list.append(vy)
        speeds.append(math.hypot(vx, vy))

    if not speeds:
        return None

    avg_speed = sum(speeds) / len(speeds)
    if avg_speed < MIN_SPEED:
        return None

    # Horizontal travel
    total_hx = abs(window[-1]["bx"] - window[0]["bx"])
    if total_hx < MIN_HORIZONTAL_TRAVEL:
        return None

    # Find closest approach to hoop
    best_dist = float("inf")
    best_idx = -1
    for k, pt in enumerate(window):
        if pt["hoop"] is None:
            continue
        d = dist((pt["bx"], pt["by"]), pt["hoop"])
        if d < best_dist:
            best_dist = d
            best_idx = k

    if best_idx < 0 or best_dist > MAX_HOOP_DIST:
        return None

    # Arc check — look for upward motion (negative vy) in ARC_LOOKBACK_FRAMES
    # before the closest approach frame
    peak_fid = window[best_idx]["frame_id"]
    upward_count = 0
    lookback_start = max(0, best_idx - ARC_LOOKBACK_FRAMES)
    for vy in vy_list[lookback_start : best_idx]:
        if vy < MIN_UPWARD_VY:
            upward_count += 1

    if upward_count < MIN_UPWARD_FRAMES:
        return None

    # Cooldown check
    if peak_fid - last_shot_frame < SHOT_COOLDOWN_FRAMES:
        return None

    hoop_center = window[best_idx]["hoop"]
    return {
        "start_frame": window[0]["frame_id"],
        "end_frame": window[-1]["frame_id"],
        "peak_frame": peak_fid,
        "closest_dist_px": round(best_dist, 1),
        "avg_speed_px_per_frame": round(avg_speed, 2),
        "upward_frames": upward_count,
        "hoop_center": hoop_center,
        "ball_at_peak": (window[best_idx]["bx"], window[best_idx]["by"]),
    }


def render_video(
    video_path: str,
    out_path: str,
    frames_data: dict,
    shots: list[dict],
    fps: float,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Build a set: frame_id -> shot info for quick lookup
    active_shots: dict[int, dict] = {}  # frame_id -> shot
    for shot in shots:
        for fid in range(shot["peak_frame"], shot["peak_frame"] + OVERLAY_DURATION_FRAMES):
            active_shots[fid] = shot

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw "Shot taken." overlay if this frame is within an active shot window
        if frame_idx in active_shots:
            shot = active_shots[frame_idx]
            label = "Shot taken."
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            pad = 10
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            box_x1, box_y1 = 20, 20
            box_x2 = box_x1 + tw + pad * 2
            box_y2 = box_y1 + th + baseline + pad * 2
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
            cv2.putText(
                frame, label,
                (box_x1 + pad, box_y2 - pad - baseline),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video saved to {out_path}")


def export_csv(shots: list[dict], out_path: str, fps: float):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "shot_number",
            "start_frame", "end_frame", "peak_frame",
            "start_time_s", "end_time_s", "peak_time_s",
            "closest_dist_px", "avg_speed_px_per_frame",
            "upward_frames", "hoop_center_x", "hoop_center_y",
            "ball_x_at_peak", "ball_y_at_peak",
        ])
        writer.writeheader()
        for i, shot in enumerate(shots, 1):
            hx, hy = shot["hoop_center"] if shot["hoop_center"] else (None, None)
            bx, by = shot["ball_at_peak"]
            writer.writerow({
                "shot_number": i,
                "start_frame": shot["start_frame"],
                "end_frame": shot["end_frame"],
                "peak_frame": shot["peak_frame"],
                "start_time_s": round(shot["start_frame"] / fps, 3),
                "end_time_s": round(shot["end_frame"] / fps, 3),
                "peak_time_s": round(shot["peak_frame"] / fps, 3),
                "closest_dist_px": shot["closest_dist_px"],
                "avg_speed_px_per_frame": shot["avg_speed_px_per_frame"],
                "upward_frames": shot["upward_frames"],
                "hoop_center_x": hx,
                "hoop_center_y": hy,
                "ball_x_at_peak": bx,
                "ball_y_at_peak": by,
            })
    print(f"CSV saved to {out_path}  ({len(shots)} shots)")


def main():
    parser = argparse.ArgumentParser(description="Detect shot attempts from tracking.json")
    parser.add_argument("--tracking", default="output/tracking.json", help="Path to tracking JSON")
    parser.add_argument("--video", default=None, help="Source video path (optional; needed for annotated output)")
    parser.add_argument("--out", default="output/shots_annotated.mp4", help="Output annotated video path")
    parser.add_argument("--csv", default="output/shots.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading tracking data from {args.tracking} ...")
    data = load_tracking(args.tracking)
    fps = data["fps"]
    frames = data["frames"]

    print(f"  {len(frames)} frames, {fps:.1f} fps, resolution {data['frame_width']}x{data['frame_height']}")

    traj = build_ball_trajectory(frames)
    print(f"  {len(traj)} frames with ball detected")

    shots = find_shots(traj)
    print(f"\nShots detected: {len(shots)}")
    for i, s in enumerate(shots, 1):
        t = s["peak_frame"] / fps
        print(f"  Shot {i}: frame {s['peak_frame']} ({t:.2f}s)  dist_to_hoop={s['closest_dist_px']}px  speed={s['avg_speed_px_per_frame']}px/f")

    # Export CSV regardless of video
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    export_csv(shots, args.csv, fps)

    # Render annotated video if a source video is provided
    video_src = args.video or data.get("video")
    if video_src and Path(video_src).exists():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        render_video(video_src, args.out, frames, shots, fps)
    else:
        print(f"Video source not found ({video_src}), skipping video export.")


if __name__ == "__main__":
    main()
