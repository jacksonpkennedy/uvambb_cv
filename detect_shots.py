"""
Shot + Board Crash Detection Script
Reads output/tracking.json and source video, detects shot attempts and then
identifies which players from each team crashed the boards on each shot.

A player "crashes the boards" if:
  - They were NOT already under the hoop at shot peak (effort-based)
  - They moved toward the hoop by MIN_CRASH_DISPLACEMENT pixels during the
    CRASH_WINDOW_FRAMES after the shot, regardless of where the ball ends up

Usage:
  python detect_shots_crashes.py --tracking output/tracking.json --video run2.mp4
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Shot detection hyperparameters
# ---------------------------------------------------------------------------

# Minimum ball speed (pixels/frame) to consider the ball "in flight"
MIN_SPEED = 1.0

# Minimum number of consecutive frames the ball must be tracked as "in flight"
MIN_FLIGHT_FRAMES = 8

# Maximum distance (pixels) from ball to hoop center at peak approach
# to count as a shot attempt
MAX_HOOP_DIST = 600

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
MIN_HORIZONTAL_TRAVEL = 8

# Cooldown (frames) after a shot is detected before another can be registered
SHOT_COOLDOWN_FRAMES = 60

# Far-shot arc detector: minimum horizontal travel (px) for a 3-pt / long-2 arc
FAR_SHOT_HORIZ_TRAVEL = 80

# Far-shot arc detector: ball must start at least this far (px) from hoop —
# filters out layups that happen to have a small arc
FAR_SHOT_MIN_RELEASE_DIST = 180

# Far-shot arc detector: minimum rise of the arc peak above the avg of
# release and hoop y positions (px, upward = decreasing y in image coords)
MIN_ARC_HEIGHT_PX = 25

# Overlay display duration (frames)
OVERLAY_DURATION_FRAMES = 90

# Pause duration (in seconds) for shot highlight
PAUSE_DURATION_SEC = 5

# ---------------------------------------------------------------------------
# Board crash detection hyperparameters
# ---------------------------------------------------------------------------

# Frames after shot peak to watch for crashing activity (~1.5s at 60fps)
CRASH_WINDOW_FRAMES = 90

# Players starting within this distance of the hoop count as crashers
# automatically — they're already in rebounding position (inside the paint).
CRASH_ZONE_DIST = 200

# Players starting farther than this are too far to be crashing the boards
# (e.g., standing at the 3-point line).  They won't be counted even if they
# drift a few pixels toward the hoop during normal offensive flow.
MAX_CRASH_START_DIST = 270

# A player in the 200-270px zone must close at least this many pixels toward
# the hoop to be counted as crashing (effort threshold).
MIN_CRASH_DISPLACEMENT = 40

# Players within this distance AND who barely moved are excluded — they're
# standing directly under the basket and are not making an effort play.
UNDER_BASKET_DIST   = 65
UNDER_BASKET_MOVE   = 15

# ---------------------------------------------------------------------------


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def find_release_frame(window: list[dict], best_idx: int) -> int:
    """
    Scan backward from best_idx to find the start of the ball's final arc.

    Strategy: walk backward through (frame, dist_to_hoop) pairs.  The first
    step we find where the distance *increased* (ball momentarily moved away
    from the hoop) and the *subsequent* distances decrease consistently is the
    boundary between pre-shot ball-handling and the actual shot arc.  Returning
    the frame of that boundary gives an accurate release estimate without being
    fooled by dribbling/passing earlier in the window.
    """
    search_start = max(0, best_idx - 150)

    dist_pairs: list[tuple[int, float]] = []
    for k in range(search_start, best_idx + 1):
        if window[k]["hoop"] is not None:
            d = dist((window[k]["bx"], window[k]["by"]), window[k]["hoop"])
            dist_pairs.append((k, d))

    if len(dist_pairs) < 4:
        return window[search_start]["frame_id"]

    # Scan backward: find the most recent step where dist increased, after
    # which the ball mostly approaches the hoop (consistent shot trajectory).
    for i in range(len(dist_pairs) - 2, 0, -1):
        k,      d      = dist_pairs[i]
        k_prev, d_prev = dist_pairs[i - 1]
        if d_prev >= d:          # distance didn't increase here — keep going back
            continue
        # Distance increased at this step.  Check that from here onward the
        # ball mostly closes on the hoop (confirming this is the shot start).
        tail = [dd for _, dd in dist_pairs[i:]]
        if len(tail) < 3:
            continue
        dec = sum(1 for j in range(1, len(tail)) if tail[j] <= tail[j - 1] + 8)
        if dec > 0.65 * (len(tail) - 1):
            return window[k]["frame_id"]

    return window[dist_pairs[0][0]]["frame_id"]


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
            if gap > 10:   # allow larger gaps for long shots / occlusion
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
        print("Reject: too_short", len(window))
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
        print("Reject: no_speeds", len(window))
        return None

    avg_speed = sum(speeds) / len(speeds)
    # Allow slower speeds if trajectory is long (e.g., 3pt arcs with dt>1)
    if avg_speed < MIN_SPEED:
        if len(window) < 12:
            print("Reject: low_speed", len(window))
            return None

    # Check if distance to hoop is mostly decreasing
    dists = []
    for pt in window:
        if pt["hoop"] is not None:
            dists.append(dist((pt["bx"], pt["by"]), pt["hoop"]))

    decreasing_count = 0
    for i in range(1, len(dists)):
        if dists[i] <= dists[i-1]:
            decreasing_count += 1

    # Check velocity trend (vy increasing over time due to gravity)
    vy_trend = 0
    for i in range(1, len(vy_list)):
        if vy_list[i] >= vy_list[i-1]:
            vy_trend += 1

    # Find first close approach to hoop (not the global minimum).
    # Once the ball reaches within 40px of the hoop and then bounces
    # away by 35px+, we stop — this prevents post-shot ball activity
    # from overriding the actual shot peak.
    best_dist = float("inf")
    best_idx = -1
    for k, pt in enumerate(window):
        if pt["hoop"] is None:
            continue
        d = dist((pt["bx"], pt["by"]), pt["hoop"])
        if d < best_dist:
            best_dist = d
            best_idx = k
        # Ball reached the hoop and has now rebounded away — stop here
        if k > best_idx and best_dist < 40 and d > best_dist + 35:
            break

    if best_idx < 0 or best_dist > MAX_HOOP_DIST:
        print("Reject: no_hoop_approach", len(window))
        return None

    peak_fid = window[best_idx]["frame_id"]

    # --- Far-shot / 3-point arc detector ---
    # Uses the full window so it can find the arc peak even when it occurs
    # long before the ball reaches the hoop (typical of 3-pointers).
    if len(window) >= 20 and dists:
        # Per-frame vy over the full window
        full_vy = []
        for k in range(1, len(window)):
            dt = window[k]["frame_id"] - window[k - 1]["frame_id"]
            if dt > 0:
                full_vy.append((window[k]["by"] - window[k - 1]["by"]) / dt)

        # Find arc peak: vy transitions from clearly negative (going up)
        # to clearly non-negative (going down / levelling off).
        # Use a 2-frame smoothed comparison to reduce noise.
        arc_peak_k = -1
        for k in range(2, len(full_vy) - 1):
            avg_before = sum(full_vy[max(0, k - 2):k]) / max(1, min(k, 2))
            avg_after  = sum(full_vy[k:min(len(full_vy), k + 2)]) / 2
            if avg_before < -0.8 and avg_after > -0.2:
                arc_peak_k = k
                break

        hx_travel    = abs(window[-1]["bx"] - window[0]["bx"])
        ys           = [pt["by"] for pt in window]
        release_dist = dists[0]
        # Arc height: how much higher (in image y) is the peak vs the average
        # of the release and hoop endpoints (upward = smaller y value).
        endpoint_avg_y = (window[0]["by"] + window[best_idx]["by"]) / 2
        arc_height     = endpoint_avg_y - min(ys)   # positive means ball rose

        if (
            arc_peak_k > 2 and
            arc_peak_k < len(full_vy) - 2 and      # peak is interior, not noise
            hx_travel    > FAR_SHOT_HORIZ_TRAVEL and
            arc_height   > MIN_ARC_HEIGHT_PX and
            release_dist > FAR_SHOT_MIN_RELEASE_DIST and
            min(dists)   < 80 and                  # ball must actually reach the hoop
            peak_fid - last_shot_frame >= SHOT_COOLDOWN_FRAMES
        ):
            return {
                "start_frame": window[0]["frame_id"],
                "end_frame":   window[-1]["frame_id"],
                "peak_frame":  peak_fid,
                "closest_dist_px":        round(min(dists), 1),
                "avg_speed_px_per_frame": round(avg_speed, 2),
                "peak_detected": True,
                "hoop_center":   window[best_idx]["hoop"],
                "ball_at_peak":  (window[best_idx]["bx"], window[best_idx]["by"]),
                "release_frame": find_release_frame(window, best_idx),
                "window": window,
            }

    # --- SIMPLIFIED: recent-window arc detector ---
    # Extended lookback (40 frames) so it catches the arc peak on mid-range
    # shots and shorter 3s where the ball is still arcing near the hoop.
    if best_idx >= 0:
        end_idx = best_idx
        start_idx = max(0, end_idx - 40)
        recent = window[start_idx:end_idx+1]

        rd = [dist((pt["bx"], pt["by"]), pt["hoop"]) for pt in recent if pt["hoop"] is not None]

        if len(rd) >= 6:
            # distance must mostly decrease
            dec = sum(1 for i in range(1, len(rd)) if rd[i] <= rd[i-1] + 3)

            # velocities
            rvy = []
            for k in range(1, len(recent)):
                dt = recent[k]["frame_id"] - recent[k-1]["frame_id"]
                if dt > 0:
                    rvy.append((recent[k]["by"] - recent[k-1]["by"]) / dt)

            # detect sign change (arc)
            sign_change = False
            for i in range(1, len(rvy)):
                if rvy[i-1] < -0.05 and rvy[i] > 0.05:
                    sign_change = True
                    break

            if (
                dec > 0.6 * (len(rd) - 1) and
                sign_change and
                min(rd) < 80 and
                max(rd) - min(rd) > 200 and  # ball must travel far toward hoop (filters close-in noise)
                peak_fid - last_shot_frame >= SHOT_COOLDOWN_FRAMES
            ):
                return {
                    "start_frame": recent[0]["frame_id"],
                    "end_frame": recent[-1]["frame_id"],
                    "peak_frame": peak_fid,
                    "closest_dist_px": round(min(rd), 1),
                    "avg_speed_px_per_frame": round(avg_speed, 2),
                    "peak_detected": True,
                    "hoop_center": window[best_idx]["hoop"],
                    "ball_at_peak": (window[best_idx]["bx"], window[best_idx]["by"]),
                    "release_frame": find_release_frame(window, best_idx),
                    "window": recent,
                }

    # Ensure closest approach is near the end (ball actually reaches hoop)
    if best_idx < int(0.1 * len(window)):
        print("Reject: early_peak", len(window))
        return None

    # --- Subwindow-based detection around closest approach ---
    if best_idx >= 0 and len(window) > 10:
        start_idx = max(0, best_idx - 40)
        sub = window[start_idx:best_idx+1]

        sub_dists = []
        for pt in sub:
            if pt["hoop"] is not None:
                sub_dists.append(dist((pt["bx"], pt["by"]), pt["hoop"]))

        if len(sub_dists) > 5:
            sub_total_drop = sub_dists[0] - min(sub_dists)
            sub_vy = []
            for k in range(1, len(sub)):
                dt = sub[k]["frame_id"] - sub[k-1]["frame_id"]
                if dt > 0:
                    sub_vy.append((sub[k]["by"] - sub[k-1]["by"]) / dt)

            has_up = any(v < -0.1 for v in sub_vy)
            has_down = any(v > 0.1 for v in sub_vy)

            peak_idx = -1
            for i in range(1, len(sub_vy)):
                if sub_vy[i-1] < -0.1 and sub_vy[i] >= -0.05:
                    peak_idx = i
                    break

            mono_before = 0
            mono_after = 0
            if peak_idx > 2:
                for i in range(1, peak_idx):
                    if sub_dists[i] <= sub_dists[i-1] + 2:
                        mono_before += 1
                for i in range(peak_idx+1, len(sub_dists)):
                    if sub_dists[i] <= sub_dists[i-1] + 2:
                        mono_after += 1

            strong_mono = (
                peak_idx > 2 and
                mono_before > 0.3 * max(1, peak_idx-1) and
                mono_after > 0.3 * max(1, len(sub_dists)-peak_idx-1)
            )

            if (
                has_up and has_down and
                strong_mono and
                min(sub_dists) < 80 and
                sub_total_drop > 50 and
                peak_fid - last_shot_frame >= SHOT_COOLDOWN_FRAMES
            ):
                return {
                    "start_frame": sub[0]["frame_id"],
                    "end_frame": sub[-1]["frame_id"],
                    "peak_frame": peak_fid,
                    "closest_dist_px": round(min(sub_dists), 1),
                    "avg_speed_px_per_frame": round(avg_speed, 2),
                    "peak_detected": False,
                    "hoop_center": window[best_idx]["hoop"],
                    "ball_at_peak": (window[best_idx]["bx"], window[best_idx]["by"]),
                    "release_frame": find_release_frame(window, best_idx),
                    "window": sub,
                }

    # --- Require strong final approach to hoop ---
    if len(dists) >= 7:
        final_dists = dists[-7:]
    else:
        final_dists = dists

    if min(final_dists) > 250:
        print("Reject: final_dists", len(window))
        return None

    if len(dists) > 5:
        total_drop = max(dists) - min(dists)
        if total_drop < 200:
            print("Reject: total_drop", len(window))
            return None

    # Ball must actually reach the hoop — blocks noise/tracking artifacts
    if min(dists) > 80:
        print("Reject: never_reached_hoop", len(window))
        return None

    # Arc check: Peak detection (vy sign change: negative -> positive)
    peak_detected = False

    for k in range(1, len(vy_list)):
        prev_vy = vy_list[k - 1]
        curr_vy = vy_list[k]
        if prev_vy < -1.5 and curr_vy > -1:
            peak_detected = True
            break

    if not peak_detected:
        ys = [pt["by"] for pt in window]
        min_y = min(ys)
        avg_y = sum(ys) / len(ys)
        if min_y < avg_y - 25:
            peak_detected = True

    if not peak_detected:
        strong_traj = (
            len(dists) > 10 and
            decreasing_count > 0.6 * (len(dists) - 1) and
            vy_trend > 0.6 * (len(vy_list) - 1)
        )
        strong_finish = (
            min(dists) < 80 and
            (dists[0] - min(dists)) > 80
        )
        if strong_traj and strong_finish:
            peak_detected = True
        else:
            print("Reject: arc_check", len(window))
            return None

    # Cooldown check
    peak_fid = window[best_idx]["frame_id"]
    if peak_fid - last_shot_frame < SHOT_COOLDOWN_FRAMES:
        print("Reject: cooldown", len(window))
        return None

    hoop_center = window[best_idx]["hoop"]

    return {
        "start_frame": window[0]["frame_id"],
        "end_frame": window[-1]["frame_id"],
        "peak_frame": peak_fid,
        "closest_dist_px": round(best_dist, 1),
        "avg_speed_px_per_frame": round(avg_speed, 2),
        "peak_detected": True,
        "hoop_center": hoop_center,
        "ball_at_peak": (window[best_idx]["bx"], window[best_idx]["by"]),
        "release_frame": find_release_frame(window, best_idx),
        "window": window,
    }


# ---------------------------------------------------------------------------
# Board crash detection
# ---------------------------------------------------------------------------

def detect_board_crashes(shots: list[dict], frames: dict) -> None:
    """
    For each shot, identify players who crashed the boards and tag them by team.
    Modifies each shot dict in-place, adding a 'crashes' key.

    Crash criteria — a player counts if they were close to the basket to start
    OR moved toward it once the ball was in the air:

      AUTOMATIC CRASH (already in position):
        initial_dist <= CRASH_ZONE_DIST (200px) AND not standing still directly
        under the basket (initial_dist > UNDER_BASKET_DIST OR moved > UNDER_BASKET_MOVE)

      EFFORT CRASH (moved toward basket):
        initial_dist in (CRASH_ZONE_DIST, MAX_CRASH_START_DIST] (200–270px)
        AND displacement_toward >= MIN_CRASH_DISPLACEMENT (40px)

      EXCLUDED:
        initial_dist > MAX_CRASH_START_DIST (270px) — too far away (3-point line+)

    Team assignment uses majority-vote of the 'team' field across all frames
    the player appears in during the crash window, to reduce misclassification
    from the first-frame snapshot.
    """
    from collections import Counter

    for shot in shots:
        peak = shot["peak_frame"]
        hoop = shot["hoop_center"]

        if hoop is None:
            shot["crashes"] = {"team_0": [], "team_1": [], "unknown": []}
            continue

        window_end = peak + CRASH_WINDOW_FRAMES

        # Collect each tracked player's first position, final position, bbox at
        # peak, and all team-label observations (for majority-vote assignment).
        player_initial:     dict[int, tuple]      = {}   # tid -> (cx, cy, box)
        player_final:       dict[int, tuple]      = {}   # tid -> (cx, cy)
        player_team_votes:  dict[int, list]       = {}   # tid -> [team, ...]

        for fid in range(peak, window_end + 1):
            fdata = frames.get(str(fid))
            if fdata is None:
                continue
            for p in fdata.get("players", []):
                tid    = p["tid"]
                cx, cy = p["center"]
                team   = p.get("team")
                box    = p.get("box")
                if tid not in player_initial:
                    player_initial[tid] = (cx, cy, box)
                player_final[tid] = (cx, cy)
                if team is not None:
                    player_team_votes.setdefault(tid, []).append(team)

        crashers: dict[str, list] = {"team_0": [], "team_1": [], "unknown": []}

        for tid, (ix, iy, ibox) in player_initial.items():
            if tid not in player_final:
                continue
            fx, fy = player_final[tid]

            initial_dist      = dist((ix, iy), hoop)
            final_dist        = dist((fx, fy), hoop)
            displacement_toward = initial_dist - final_dist

            # Skip players too far from the basket to be crashing (3-point line+)
            if initial_dist > MAX_CRASH_START_DIST:
                continue

            # Skip players standing still directly under the basket — they
            # didn't contest; they were just already there
            if initial_dist < UNDER_BASKET_DIST and displacement_toward < UNDER_BASKET_MOVE:
                continue

            in_crash_zone = initial_dist <= CRASH_ZONE_DIST
            effort_crash  = displacement_toward >= MIN_CRASH_DISPLACEMENT

            if not (in_crash_zone or effort_crash):
                continue

            # Majority-vote team label across the crash window
            votes = player_team_votes.get(tid, [])
            if votes:
                team = Counter(votes).most_common(1)[0][0]
            else:
                team = None

            bucket = f"team_{team}" if team in (0, 1) else "unknown"
            crashers[bucket].append({
                "tid":          tid,
                "initial_dist": round(initial_dist, 1),
                "displacement": round(displacement_toward, 1),
                "peak_box":     ibox,
                "peak_pos":     [round(ix), round(iy)],
            })

        shot["crashes"] = crashers


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

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

    # Build a set: frame_id -> shot info for quick lookup.
    # Overlay spans release_frame → peak_frame + OVERLAY_DURATION_FRAMES so it
    # shows during the full flight and through the crash window afterward.
    active_shots: dict[int, dict] = {}  # frame_id -> shot
    for shot in shots:
        start = shot.get("release_frame", shot["peak_frame"])
        end   = shot["peak_frame"] + OVERLAY_DURATION_FRAMES
        for fid in range(start, end):
            active_shots[fid] = shot

    shot_windows = {shot["peak_frame"]: shot for shot in shots}

    pause_frames = int(fps * PAUSE_DURATION_SEC)
    crash_pause_frames = int(fps * 2.0)   # 2-second crash highlight pauses
    shown_shots = set()       # shots that already triggered the release-frame pause
    shown_crash_shots = set() # shots that already triggered the post-peak crash highlight

    frame_idx = 0
    prev_prob = 0.0
    debug_rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Probability bar initialization ---
        shot_prob = prev_prob * 0.9

        # --- Compute shot probability (simple heuristic) ---
        fdata = frames_data.get(str(frame_idx))
        if fdata and fdata.get("ball") and fdata.get("hoops"):
            bx, by = fdata["ball"]["center"]
            hx, hy = fdata["hoops"][0]["center"]

            d = dist((bx, by), (hx, hy))
            prob_dist = max(0, 1 - d / 300)

            if frame_idx > 0:
                prev = frames_data.get(str(frame_idx - 1))
                if prev and prev.get("ball"):
                    _, prev_y = prev["ball"]["center"]
                    vy = by - prev_y
                    prob_arc = 1.0 if vy < 0 else 0.3
                else:
                    prob_arc = 0.0
            else:
                prob_arc = 0.0

            raw_prob = 0.6 * prob_dist + 0.4 * prob_arc
            shot_prob = 0.85 * prev_prob + 0.15 * raw_prob
            prev_prob = shot_prob

        # Draw overlay if this frame is within an active shot window
        if frame_idx in active_shots:
            shot = active_shots[frame_idx]
            crashes   = shot.get("crashes", {})
            t1_crash  = len(crashes.get("team_0", []))
            t2_crash  = len(crashes.get("team_1", []))
            unk_crash = len(crashes.get("unknown", []))

            # Trigger pause only once per shot (at release frame — when player shoots)
            _trigger_fid = shot.get("release_frame", shot["peak_frame"])
            if frame_idx == _trigger_fid and shot["peak_frame"] not in shown_shots:
                shown_shots.add(shot["peak_frame"])

                for _ in range(pause_frames):
                    pause_frame = frame.copy()

                    traj_window = shot_windows.get(shot["peak_frame"], {}).get("window", [])

                    # Draw ball and hoop markers
                    fdata = frames_data.get(str(frame_idx))
                    if fdata and fdata.get("ball") and fdata.get("hoops"):
                        bx, by = fdata["ball"]["center"]
                        hx, hy = fdata["hoops"][0]["center"]

                        cv2.line(pause_frame, (int(bx), int(by)), (int(hx), int(hy)), (0, 255, 0), 3)
                        cv2.circle(pause_frame, (int(bx), int(by)), 6, (0, 0, 255), -1)
                        cv2.circle(pause_frame, (int(hx), int(hy)), 8, (255, 0, 0), 2)

                    # Draw trajectory trail
                    if traj_window:
                        idx = None
                        for i, pt in enumerate(traj_window):
                            if pt["frame_id"] >= frame_idx:
                                idx = i
                                break
                        if idx is None:
                            idx = len(traj_window) - 1

                        trail = traj_window[max(0, idx-10):idx+1]
                        for i in range(1, len(trail)):
                            p1 = (int(trail[i-1]["bx"]), int(trail[i-1]["by"]))
                            p2 = (int(trail[i]["bx"]), int(trail[i]["by"]))
                            cv2.line(pause_frame, p1, p2, (0, 255, 255), 2)

                        if len(trail) >= 2:
                            p1 = trail[-2]
                            p2 = trail[-1]
                            vx = p2["bx"] - p1["bx"]
                            vy = p2["by"] - p1["by"]
                            scale = 3
                            end_point = (int(p2["bx"] + vx * scale), int(p2["by"] + vy * scale))
                            cv2.arrowedLine(pause_frame, (int(p2["bx"]), int(p2["by"])), end_point, (255, 0, 255), 2, tipLength=0.3)

                    # Pause overlay box — includes crash counts
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    y0   = 40
                    dy   = 20

                    unk_suffix = f"  (+{unk_crash} unknown)" if unk_crash else ""
                    pause_texts = [
                        "SHOT DETECTED",
                        f"Dist to hoop: {shot['closest_dist_px']} px",
                        f"Avg speed:    {shot['avg_speed_px_per_frame']} px/f",
                        f"Peak frame:   {shot['peak_frame']}",
                        f"Team 1 crashes: {t1_crash}",
                        f"Team 2 crashes: {t2_crash}" + unk_suffix,
                    ]

                    box_h_pause = y0 + len(pause_texts) * dy + 12
                    cv2.rectangle(pause_frame, (20, 20), (320, box_h_pause), (255, 255, 255), -1)
                    for i, text in enumerate(pause_texts):
                        txt_color = (0, 0, 0)
                        if i == 4 and t1_crash > 0:
                            txt_color = (20, 100, 20)   # dark green for Team 1 crashes
                        elif i == 5 and t2_crash > 0:
                            txt_color = (160, 80, 0)    # dark blue-ish for Team 2 crashes
                        cv2.putText(pause_frame, text, (30, y0 + i*dy),
                                    font, 0.5, txt_color, 2, cv2.LINE_AA)

                    # Probability bar
                    bar_x1, bar_y1 = 10, 50
                    bar_x2, bar_y2 = 30, h - 50
                    cv2.rectangle(pause_frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (200,200,200), 1)
                    fill_height = int((bar_y2 - bar_y1) * shot_prob)
                    fill_y1 = bar_y2 - fill_height
                    color = (0,255,0) if shot_prob > 0.6 else (0,200,0)
                    cv2.rectangle(pause_frame, (bar_x1, fill_y1), (bar_x2, bar_y2), color, -1)
                    cv2.putText(pause_frame, "P", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                    # Debug overlay (bottom-left)
                    debug_x = 10
                    debug_y = h - 120
                    line_h  = 18
                    fdata_dbg = frames_data.get(str(frame_idx))
                    debug_texts = []
                    if fdata_dbg and fdata_dbg.get("ball") and fdata_dbg.get("hoops"):
                        bx_dbg, by_dbg = fdata_dbg["ball"]["center"]
                        hx_dbg, hy_dbg = fdata_dbg["hoops"][0]["center"]
                        d_dbg = dist((bx_dbg, by_dbg), (hx_dbg, hy_dbg))
                        if frame_idx > 0:
                            prev_dbg = frames_data.get(str(frame_idx - 1))
                            if prev_dbg and prev_dbg.get("ball"):
                                _, prev_y_dbg = prev_dbg["ball"]["center"]
                                vy_dbg = by_dbg - prev_y_dbg
                            else:
                                vy_dbg = 0
                        else:
                            vy_dbg = 0
                        debug_texts = [
                            f"dist: {d_dbg:.1f} / {MAX_HOOP_DIST}",
                            f"vy: {vy_dbg:.2f} (up if <0)",
                            f"prob: {shot_prob:.2f}",
                            f"threshold(prob): 0.60",
                            f"min_speed: {MIN_SPEED}",
                            f"min_travel: {MIN_HORIZONTAL_TRAVEL}",
                        ]
                    else:
                        debug_texts = ["no ball/hoop"]

                    box_h = line_h * len(debug_texts) + 10
                    cv2.rectangle(pause_frame, (debug_x, debug_y - box_h), (260, debug_y), (255, 255, 255), -1)
                    for i, txt in enumerate(debug_texts):
                        cv2.putText(
                            pause_frame, txt,
                            (debug_x + 5, debug_y - box_h + 15 + i * line_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
                        )

                    # Frame counter on pause frames
                    pause_label = f"Frame {frame_idx}"
                    (pl_w, pl_h), pl_base = cv2.getTextSize(
                        pause_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                    )
                    pl_x = w - pl_w - 12
                    pl_y = pl_h + 10
                    cv2.rectangle(pause_frame, (pl_x - 4, 4), (w - 6, pl_y + pl_base + 4), (0, 0, 0), -1)
                    cv2.putText(pause_frame, pause_label, (pl_x, pl_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                    out.write(pause_frame)

            # Post-peak crash highlight pauses — 2 sec per team showing who crashed
            if frame_idx == shot["peak_frame"] and shot["peak_frame"] not in shown_crash_shots:
                shown_crash_shots.add(shot["peak_frame"])
                crashes = shot.get("crashes", {})

                TEAM_CONFIGS = [
                    ("team_0", "Team 1 Crashers", (50, 200, 50)),   # green
                    ("team_1", "Team 2 Crashers", (50, 50, 230)),   # red
                ]

                for team_key, team_label, color in TEAM_CONFIGS:
                    crasher_list = crashes.get(team_key, [])
                    count = len(crasher_list)
                    count_text = f"{count} player{'s' if count != 1 else ''} crashed the boards"

                    for _ in range(crash_pause_frames):
                        cf = frame.copy()

                        # Draw bounding box (or circle fallback) around each crasher
                        for c in crasher_list:
                            box = c.get("peak_box")
                            pos = c.get("peak_pos")
                            if box:
                                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                                cv2.rectangle(cf, (x1, y1), (x2, y2), color, 4)
                                lbl = f"#{c['tid']}"
                                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                                cv2.rectangle(cf, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
                                cv2.putText(cf, lbl, (x1 + 3, y1 - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                            elif pos:
                                cv2.circle(cf, (pos[0], pos[1]), 45, color, 3)

                        # Centered team label banner
                        lbl_font = cv2.FONT_HERSHEY_SIMPLEX
                        (bw, bh), _ = cv2.getTextSize(team_label, lbl_font, 1.1, 3)
                        bx = (w - bw) // 2
                        by = 70
                        cv2.rectangle(cf, (bx - 14, by - bh - 14), (bx + bw + 14, by + 12), (0, 0, 0), -1)
                        cv2.putText(cf, team_label, (bx, by), lbl_font, 1.1, color, 3, cv2.LINE_AA)

                        # Count sub-label
                        (cw, ch), _ = cv2.getTextSize(count_text, lbl_font, 0.65, 2)
                        cx_lbl = (w - cw) // 2
                        cy_lbl = by + ch + 18
                        cv2.putText(cf, count_text, (cx_lbl, cy_lbl),
                                    lbl_font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                        # Frame counter (top-right)
                        fc_label = f"Frame {frame_idx}"
                        (fl_w, fl_h), fl_base = cv2.getTextSize(fc_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        fl_x = w - fl_w - 12
                        fl_y = fl_h + 10
                        cv2.rectangle(cf, (fl_x - 4, 4), (w - 6, fl_y + fl_base + 4), (0, 0, 0), -1)
                        cv2.putText(cf, fc_label, (fl_x, fl_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                        out.write(cf)

            # Regular (non-pause) overlay — white box top-left with crash counts
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness  = 2
            pad        = 8

            reg_lines = [
                "Shot taken.",
                f"Team 1 crashes: {t1_crash}",
                f"Team 2 crashes: {t2_crash}",
            ]
            if unk_crash:
                reg_lines.append(f"(+{unk_crash} unassigned)")

            line_sizes  = [cv2.getTextSize(l, font, font_scale, thickness) for l in reg_lines]
            box_w       = max(sz[0][0] for sz in line_sizes) + pad * 2
            line_h_reg  = max(sz[0][1] for sz in line_sizes) + 6
            box_h_reg   = line_h_reg * len(reg_lines) + pad * 2

            box_x1, box_y1 = 20, 20
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h_reg
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)

            for li, line in enumerate(reg_lines):
                ty = box_y1 + pad + (li + 1) * line_h_reg - 4
                cv2.putText(frame, line, (box_x1 + pad, ty),
                            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # --- Draw probability bar (left side) ---
        bar_x1, bar_y1 = 10, 50
        bar_x2, bar_y2 = 30, h - 50
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (200,200,200), 1)
        fill_height = int((bar_y2 - bar_y1) * shot_prob)
        fill_y1 = bar_y2 - fill_height
        color = (0,255,0) if shot_prob > 0.6 else (0,200,0)
        cv2.rectangle(frame, (bar_x1, fill_y1), (bar_x2, bar_y2), color, -1)
        cv2.putText(frame, "P", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Debug overlay (bottom-left, every frame)
        debug_x = 10
        debug_y = h - 120
        line_h  = 18

        fdata_dbg  = frames_data.get(str(frame_idx))
        debug_texts = []

        if fdata_dbg and fdata_dbg.get("ball") and fdata_dbg.get("hoops"):
            bx_dbg, by_dbg = fdata_dbg["ball"]["center"]
            hx_dbg, hy_dbg = fdata_dbg["hoops"][0]["center"]
            d_dbg = dist((bx_dbg, by_dbg), (hx_dbg, hy_dbg))

            if frame_idx > 0:
                prev_dbg = frames_data.get(str(frame_idx - 1))
                if prev_dbg and prev_dbg.get("ball"):
                    _, prev_y_dbg = prev_dbg["ball"]["center"]
                    vy_dbg = by_dbg - prev_y_dbg
                else:
                    vy_dbg = 0
            else:
                vy_dbg = 0

            debug_texts = [
                f"dist: {d_dbg:.1f} / {MAX_HOOP_DIST}",
                f"vy: {vy_dbg:.2f} (up if <0)",
                f"prob: {shot_prob:.2f}",
                f"threshold(prob): 0.60",
                f"min_speed: {MIN_SPEED}",
                f"min_travel: {MIN_HORIZONTAL_TRAVEL}",
            ]
        else:
            debug_texts = ["no ball/hoop"]

        if frame_idx % 5 == 0:
            print(f"Frame {frame_idx}: " + " | ".join(debug_texts))
            if frame_idx == 0:
                debug_rows = []
            if fdata_dbg and fdata_dbg.get("ball") and fdata_dbg.get("hoops"):
                debug_rows.append({
                    "frame": frame_idx,
                    "dist": round(d_dbg, 2),
                    "vy": round(vy_dbg, 2),
                    "prob": round(shot_prob, 2),
                })

        box_h = line_h * len(debug_texts) + 10
        cv2.rectangle(frame, (debug_x, debug_y - box_h), (260, debug_y), (255, 255, 255), -1)
        for i, txt in enumerate(debug_texts):
            cv2.putText(
                frame, txt,
                (debug_x + 5, debug_y - box_h + 15 + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )

        # Frame counter — top-right corner
        frame_label = f"Frame {frame_idx}"
        (fl_w, fl_h), fl_base = cv2.getTextSize(
            frame_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        fl_x = w - fl_w - 12
        fl_y = fl_h + 10
        cv2.rectangle(frame, (fl_x - 4, 4), (w - 6, fl_y + fl_base + 4), (0, 0, 0), -1)
        cv2.putText(frame, frame_label, (fl_x, fl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Export debug CSV
    debug_csv_path = "output/debug_metrics.csv"
    with open(debug_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "dist", "vy", "prob"])
        writer.writeheader()
        for row in debug_rows:
            writer.writerow(row)
    print(f"Debug CSV saved to {debug_csv_path}")
    print(f"Video saved to {out_path}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(shots: list[dict], out_path: str, fps: float):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "shot_number",
            "start_frame", "end_frame", "peak_frame",
            "start_time_s", "end_time_s", "peak_time_s",
            "closest_dist_px", "avg_speed_px_per_frame",
            "peak_detected",
            "hoop_center_x", "hoop_center_y",
            "ball_x_at_peak", "ball_y_at_peak",
            "team_1_crashes", "team_2_crashes",
            "unknown_crashes", "total_crashes",
        ])
        writer.writeheader()
        for i, shot in enumerate(shots, 1):
            hx, hy = shot["hoop_center"] if shot["hoop_center"] else (None, None)
            bx, by = shot["ball_at_peak"]
            crashes = shot.get("crashes", {})
            t1 = len(crashes.get("team_0", []))
            t2 = len(crashes.get("team_1", []))
            uk = len(crashes.get("unknown", []))
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
                "peak_detected": shot.get("peak_detected", False),
                "hoop_center_x": hx,
                "hoop_center_y": hy,
                "ball_x_at_peak": bx,
                "ball_y_at_peak": by,
                "team_1_crashes": t1,
                "team_2_crashes": t2,
                "unknown_crashes": uk,
                "total_crashes": t1 + t2 + uk,
            })
    print(f"CSV saved to {out_path}  ({len(shots)} shots)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect shots + board crashes from tracking.json"
    )
    parser.add_argument("--tracking", default="output/tracking.json",
                        help="Path to tracking JSON")
    parser.add_argument("--video", default=None,
                        help="Source video path (needed for annotated output)")
    parser.add_argument("--out", default="output/shots_crashes_annotated.mp4",
                        help="Output annotated video path")
    parser.add_argument("--csv", default="output/shots_crashes.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading tracking data from {args.tracking} ...")
    data   = load_tracking(args.tracking)
    fps    = data["fps"]
    frames = data["frames"]

    print(f"  {len(frames)} frames, {fps:.1f} fps, "
          f"resolution {data['frame_width']}x{data['frame_height']}")

    traj = build_ball_trajectory(frames)
    print(f"  {len(traj)} frames with ball detected")

    shots = find_shots(traj)
    print(f"\nShots detected: {len(shots)}")

    detect_board_crashes(shots, frames)

    for i, s in enumerate(shots, 1):
        t       = s["peak_frame"] / fps
        crashes = s.get("crashes", {})
        t1      = len(crashes.get("team_0", []))
        t2      = len(crashes.get("team_1", []))
        uk      = len(crashes.get("unknown", []))
        print(f"  Shot {i}: frame {s['peak_frame']} ({t:.2f}s)  "
              f"dist={s['closest_dist_px']}px  speed={s['avg_speed_px_per_frame']}px/f  "
              f"crashes -> Team1={t1}  Team2={t2}  unknown={uk}")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    export_csv(shots, args.csv, fps)

    video_src = args.video or data.get("video")
    if video_src and Path(video_src).exists():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        render_video(video_src, args.out, frames, shots, fps)
    else:
        print(f"Video source not found ({video_src}), skipping video export.")


if __name__ == "__main__":
    main()
