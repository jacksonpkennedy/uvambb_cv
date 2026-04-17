"""effective_capture_rate.py — Interactive video labeler for capture rate metric.

Shows each frame of an annotated video. You classify every frame with a keypress.

Keys
----
  D  — Direct detection  (TrackNet found the ball, labeled "BALL ...")
  I  — Interpolated      (gap-filled by interpolator, labeled "BALL (INTERP)")
  F  — Fail              (ball IS on court but system missed it entirely)
  N  — No ball           (ball not visible / out of frame — skip, not counted)
  ←  — Go back one frame (undo last label)
  →  — Skip forward without labeling  (treated as N)
  Q  — Quit and save progress

Metric
------
  Effective Capture Rate = (Direct + Interpolated) / (Direct + Interpolated + Fail)
  "No ball" frames are excluded from numerator and denominator.

Progress auto-saves to .ecr_progress.json and resumes on restart.

Usage
-----
  python scripts/effective_capture_rate.py --video output/annotated.mp4
  python scripts/effective_capture_rate.py --video output/annotated.mp4 --reset
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROGRESS_FILE = ".ecr_progress.json"

LABEL_DIRECT = "direct"
LABEL_INTERP = "interp"
LABEL_FAIL   = "fail"
LABEL_WRONG  = "wrong"
LABEL_NOBALL = "noball"

KEY_MAP = {
    ord('d'): LABEL_DIRECT,
    ord('D'): LABEL_DIRECT,
    ord('i'): LABEL_INTERP,
    ord('I'): LABEL_INTERP,
    ord('f'): LABEL_FAIL,
    ord('F'): LABEL_FAIL,
    ord('w'): LABEL_WRONG,
    ord('W'): LABEL_WRONG,
    ord('h'): LABEL_WRONG,
    ord('H'): LABEL_WRONG,
    ord('n'): LABEL_NOBALL,
    ord('N'): LABEL_NOBALL,
}

COLOR_MAP = {
    LABEL_DIRECT: (0, 255, 0),     # green
    LABEL_INTERP: (255, 200, 0),   # cyan-ish
    LABEL_FAIL:   (0, 0, 255),     # red
    LABEL_WRONG:  (0, 100, 255),   # orange
    LABEL_NOBALL: (120, 120, 120), # grey
}

ARROW_LEFT  = 2424832   # Windows left arrow
ARROW_RIGHT = 2555904   # Windows right arrow
ARROW_LEFT_MAC  = 63234
ARROW_RIGHT_MAC = 63235


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",  required=True, help="Annotated video from main.py")
    p.add_argument("--reset",  action="store_true", help="Ignore saved progress")
    p.add_argument("--scale",  type=float, default=1.5,
                   help="Display scale factor (default 1.5)")
    p.add_argument("--distances", default=None,
                   help=("Optional CSV mapping frames -> distance_px. "
                         "Columns accepted: frame_index,distance_px  OR  frame_path,distance_px"))
    return p.parse_args()


def load_progress(video_path, reset):
    key = str(Path(video_path).resolve())
    if reset or not Path(PROGRESS_FILE).exists():
        return key, {}
    with open(PROGRESS_FILE) as f:
        data = json.load(f)
    if data.get("video") != key:
        return key, {}
    return key, {int(k): v for k, v in data.get("labels", {}).items()}


def save_progress(video_key, labels):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"video": video_key, "labels": {str(k): v for k, v in labels.items()}}, f)


def _parse_distances_csv(path):
    """Parse a CSV mapping frames -> distance_px. Returns dict{frame_index: distance}.

    Accepted formats:
      frame_index,distance_px
      frame_path,distance_px   (extracts trailing integer from filename stem)
      pred_x,pred_y,label_x,label_y  (computes euclidean distance)
    """
    from csv import DictReader
    from math import hypot
    import re

    dmap = {}
    p = Path(path)
    if not p.exists():
        print(f"Distances CSV not found: {path}")
        return dmap

    with p.open(newline='') as f:
        rdr = DictReader(f)
        for row in rdr:
            try:
                if 'frame_index' in row and row.get('frame_index'):
                    idx = int(row['frame_index'])
                    dist = float(row.get('distance_px') or row.get('distance') or 0.0)
                    dmap[idx] = dist
                    continue

                if 'frame_path' in row and row.get('frame_path'):
                    stem = Path(row['frame_path']).stem
                    # Try trailing digits
                    m = re.search(r"(\d+)$", stem)
                    if m:
                        idx = int(m.group(1))
                        dist = float(row.get('distance_px') or row.get('distance') or 0.0)
                        dmap[idx] = dist
                        continue

                # Compute from pred/label coords if available
                if row.get('pred_x') and row.get('pred_y') and row.get('label_x') and row.get('label_y'):
                    px = float(row['pred_x']); py = float(row['pred_y'])
                    lx = float(row['label_x']); ly = float(row['label_y'])
                    dist = hypot(px - lx, py - ly)
                    # frame_index or frame_path optional
                    if 'frame_index' in row and row.get('frame_index'):
                        dmap[int(row['frame_index'])] = dist
                    elif 'frame_path' in row and row.get('frame_path'):
                        m = re.search(r"(\d+)$", Path(row['frame_path']).stem)
                        if m:
                            dmap[int(m.group(1))] = dist
                    continue
            except Exception:
                # ignore problematic rows but continue
                continue
    return dmap


def load_distance_records(path):
    """Load detailed distance records from CSV into dict{frame_index: record}.

    Record keys: pred_x, pred_y, label_x, label_y, distance_px
    """
    from csv import DictReader
    from math import hypot
    import re

    records = {}
    if not path:
        return records
    p = Path(path)
    if not p.exists():
        return records
    with p.open(newline='') as f:
        rdr = DictReader(f)
        for row in rdr:
            try:
                idx = None
                if row.get('frame_index'):
                    idx = int(row['frame_index'])
                elif row.get('frame_path'):
                    m = re.search(r"(\d+)$", Path(row['frame_path']).stem)
                    if m:
                        idx = int(m.group(1))
                if idx is None:
                    continue
                def _f(k):
                    v = row.get(k)
                    return float(v) if v not in (None, "") else None
                px = _f('pred_x')
                py = _f('pred_y')
                lx = _f('label_x')
                ly = _f('label_y')
                dist = None
                if px is not None and py is not None and lx is not None and ly is not None:
                    dist = hypot(px - lx, py - ly)
                elif row.get('distance_px') or row.get('distance'):
                    dist = float(row.get('distance_px') or row.get('distance'))
                records[int(idx)] = {
                    'pred_x': px,
                    'pred_y': py,
                    'label_x': lx,
                    'label_y': ly,
                    'distance_px': dist,
                }
            except Exception:
                continue
    return records


def save_distance_records(path, records):
    """Write distance records mapping to CSV (overwrites file).

    Fields: frame_index,pred_x,pred_y,label_x,label_y,distance_px
    """
    from csv import DictWriter

    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['frame_index', 'pred_x', 'pred_y', 'label_x', 'label_y', 'distance_px']
    with p.open('w', newline='') as f:
        w = DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx in sorted(records.keys()):
            r = records[idx]
            w.writerow({
                'frame_index': idx,
                'pred_x': '' if r.get('pred_x') is None else r.get('pred_x'),
                'pred_y': '' if r.get('pred_y') is None else r.get('pred_y'),
                'label_x': '' if r.get('label_x') is None else r.get('label_x'),
                'label_y': '' if r.get('label_y') is None else r.get('label_y'),
                'distance_px': '' if r.get('distance_px') is None else r.get('distance_px'),
            })


def _map_display_to_original(x_disp, y_disp, orig_w, orig_h, disp_w, disp_h, win_name):
    """Map a click at display coords to original frame pixel coords.

    Uses cv2.getWindowImageRect when available to compensate for any
    window-level scaling (e.g., maximize or DPI scaling). Falls back to
    using the configured `args.scale` ratio (disp_w/orig_w).
    """
    try:
        if hasattr(cv2, 'getWindowImageRect'):
            rx, ry, win_w, win_h = cv2.getWindowImageRect(win_name)
            if win_w > 0 and win_h > 0:
                # Map: x_original = x_click * (orig_w / win_w)
                return int(x_disp * (orig_w / win_w)), int(y_disp * (orig_h / win_h))
    except Exception:
        pass
    # Fallback: assume display image is disp_w x disp_h as rendered
    sx = disp_w / float(orig_w) if orig_w else 1.0
    sy = disp_h / float(orig_h) if orig_h else 1.0
    # Prevent division by zero
    if sx == 0 or sy == 0:
        return int(x_disp), int(y_disp)
    return int(x_disp / sx), int(y_disp / sy)


def _wrong_weight(distance_px: float) -> float:
    """Map a pixel distance -> weight according to the user's policy.

        Cases (new):
            - Near Perfect:    d < 10 px  -> w = 1.0
            - Close Miss:     10 <= d < 25 px -> w = 0.7
            - Far Miss:       25 <= d < 50 px -> w = 0.2
            - Hallucination:  d >= 50 px -> w = -1.0
    """
    if distance_px is None:
        return 0.0
    try:
        d = float(distance_px)
    except Exception:
        return 0.0
    if d < 10.0:
        return 1.0
    if d < 25.0:
        return 0.7
    if d < 50.0:
        return 0.2
    return -1.0


def compute_stats(labels, total_frames, distances_csv: str | None = None, distances_map: dict | None = None):
    counts = {LABEL_DIRECT: 0, LABEL_INTERP: 0,
              LABEL_FAIL: 0, LABEL_WRONG: 0, LABEL_NOBALL: 0}
    for v in labels.values():
        if v in counts:
            counts[v] += 1
    labeled   = sum(counts.values())
    captured  = counts[LABEL_DIRECT] + counts[LABEL_INTERP]

    # Recall denominator: everything where ball was present (captured + missed)
    recall_denom = captured + counts[LABEL_FAIL]
    recall    = captured / recall_denom if recall_denom else 0.0

    # Original precision (unweighted)
    prec_denom   = captured + counts[LABEL_WRONG]
    precision = captured / prec_denom  if prec_denom  else 0.0

    # Weighted precision: use either provided distances_map or distances CSV
    if distances_map is not None:
        dist_map = distances_map
    else:
        dist_map = _parse_distances_csv(distances_csv) if distances_csv else {}
    wrong_penalty_sum = 0.0
    wrong_counts = {"near": 0, "close": 0, "far": 0, "hallu": 0, "unknown": 0}
    for idx, lbl in labels.items():
        if lbl != LABEL_WRONG:
            continue
        # distances_map uses integer frame indices; labels keys are ints
        d = dist_map.get(int(idx))
        w = _wrong_weight(d) if d is not None else 0.0
        penalty = 1.0 - w
        wrong_penalty_sum += penalty
        if d is None:
            wrong_counts['unknown'] += 1
        elif d < 10.0:
            wrong_counts['near'] += 1
        elif d < 25.0:
            wrong_counts['close'] += 1
        elif d < 50.0:
            wrong_counts['far'] += 1
        else:
            wrong_counts['hallu'] += 1

    weighted_prec_denom = captured + wrong_penalty_sum
    weighted_precision = captured / weighted_prec_denom if weighted_prec_denom else 0.0

    ecr_f1    = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    weighted_ecr_f1 = 2 * weighted_precision * recall / (weighted_precision + recall) if (weighted_precision + recall) else 0.0
    return counts, labeled, recall_denom, prec_denom, captured, recall, precision, ecr_f1, weighted_precision, weighted_ecr_f1, wrong_counts


def draw_overlay(frame, idx, total, labels, scale, distances_csv=None, distances_map=None):
    counts, labeled, recall_denom, prec_denom, captured, recall, precision, ecr_f1, \
        weighted_precision, weighted_ecr_f1, wrong_counts = compute_stats(labels, total, distances_csv, distances_map)
    remaining = total - labeled

    h, w = frame.shape[:2]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)

    # Frame counter
    cv2.putText(frame, f"Frame {idx+1} / {total}   rem: {remaining}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Recall / Precision / ECR F1
    r_col  = (0, 220, 0) if recall    >= 0.7 else (0, 180, 255) if recall    >= 0.5 else (0, 80, 255)
    p_col  = (0, 220, 0) if weighted_precision >= 0.7 else (0, 180, 255) if weighted_precision >= 0.5 else (0, 80, 255)
    f1_col = (0, 220, 0) if weighted_ecr_f1   >= 0.7 else (0, 180, 255) if weighted_ecr_f1   >= 0.5 else (0, 80, 255)
    cv2.putText(frame, f"Recall: {recall*100:.1f}%",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, r_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Prec (w): {weighted_precision*100:.1f}%",
                (155, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, p_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"ECR F1 (w): {weighted_ecr_f1*100:.1f}%",
                (290, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, f1_col, 2, cv2.LINE_AA)

    # Counts right side
    summary = (f"D:{counts[LABEL_DIRECT]}  "
               f"I:{counts[LABEL_INTERP]}  "
               f"F:{counts[LABEL_FAIL]}  "
               f"W:{counts[LABEL_WRONG]}  "
               f"N:{counts[LABEL_NOBALL]}")
    cv2.putText(frame, summary,
                (w - 380, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Key guide bottom bar
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    guide = "D=Direct  I=Interp  F=Fail(miss)  W=Wrong(FP)  H=Halluc  N=NoBall  <- Undo  -> Skip  Q=Quit"
    cv2.putText(frame, guide,
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # Show current frame's existing label if already set
    if idx in labels:
        lbl = labels[idx]
        col = COLOR_MAP[lbl]
        cv2.putText(frame, f"[{lbl.upper()}]",
                    (w - 140, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {args.video}  |  {total} frames  |  {fps:.1f} fps")

    # Read all frames into memory (fast seek, no re-decode on undo)
    print("Loading frames into memory ...")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    total = len(frames)
    print(f"Loaded {total} frames.")

    video_key, labels = load_progress(args.video, args.reset)

    # Start from first unlabeled frame
    idx = 0
    while idx < total and idx in labels:
        idx += 1

    if idx >= total:
        print("All frames already labeled.")
    else:
        print(f"Resuming from frame {idx+1} / {total}")

    win = "Effective Capture Rate Labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    save_every = 50

    # Load any existing distance records (detailed pred/label entries)
    distances_path = args.distances or ".ecr_distances.csv"
    distance_records = load_distance_records(distances_path) if distances_path else {}
    # distances_map is simple mapping idx->distance_px used for stats
    distances_map = {k: v.get('distance_px') for k, v in distance_records.items()}

    while idx < total:
        frame = frames[idx].copy()

        # Scale for display
        h, w = frame.shape[:2]
        disp_w = int(w * args.scale)
        disp_h = int(h * args.scale)
        frame = cv2.resize(frame, (disp_w, disp_h))

        draw_overlay(frame, idx, total, labels, args.scale, args.distances, distances_map)
        cv2.imshow(win, frame)

        key = cv2.waitKeyEx(0)

        if key in (ord('q'), ord('Q')):
            break

        elif key in KEY_MAP:
            # Special-case WRONG label: allow interactive click capture to record
            # predicted and true locations (two clicks) so we can compute distance.
            if KEY_MAP[key] == LABEL_WRONG:
                # Instructions shown in overlay; capture two clicks on the displayed (resized) frame
                clicks = []

                def _on_mouse(evt, x, y, flags, param):
                    if evt == cv2.EVENT_LBUTTONDOWN:
                        clicks.append((x, y))

                cv2.setMouseCallback(win, _on_mouse)
                # Wait for two clicks or ESC to cancel. You can also press H to
                # mark a hallucination (no true ball) instead of clicking a true
                # location. If you clicked the predicted location first, it will
                # be saved with the hallucination record.
                cancel = False
                mark_hallu = False
                while True:
                    tmp = frame.copy()
                    for i, (cx, cy) in enumerate(clicks):
                        col = (0, 255, 0) if i == 0 else (0, 0, 255)
                        cv2.circle(tmp, (int(cx), int(cy)), 6, col, -1)
                        cv2.putText(tmp, f"{i+1}", (int(cx) + 8, int(cy) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    instr = "Click PRED then TRUE (2 clicks). H=Halluc ESC=Cancel"
                    cv2.putText(tmp, instr, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.imshow(win, tmp)
                    k2 = cv2.waitKeyEx(20)
                    if len(clicks) >= 2:
                        break
                    if k2 == 27:  # ESC
                        cancel = True
                        break
                    if k2 in (ord('h'), ord('H')):
                        mark_hallu = True
                        break
                cv2.setMouseCallback(win, lambda *a, **k: None)
                if cancel:
                    # don't label; stay on this frame
                    continue
                if mark_hallu:
                    # Record hallucination: optional pred click saved if provided
                    if len(clicks) >= 1:
                        px_disp, py_disp = clicks[0]
                        px, py = _map_display_to_original(px_disp, py_disp, w, h, disp_w, disp_h, win)
                    else:
                        px = None; py = None
                    lx = None; ly = None
                    dist_px = float('inf')
                    distance_records[idx] = {
                        'pred_x': px,
                        'pred_y': py,
                        'label_x': lx,
                        'label_y': ly,
                        'distance_px': dist_px,
                    }
                    distances_map[idx] = dist_px
                    try:
                        save_distance_records(distances_path, distance_records)
                    except Exception:
                        pass
                    labels[idx] = LABEL_WRONG
                    idx += 1
                    if idx % save_every == 0:
                        save_progress(video_key, labels)
                    continue
                # Convert clicks from display coords back to original frame pixels
                px_disp, py_disp = clicks[0]
                lx_disp, ly_disp = clicks[1]
                px, py = _map_display_to_original(px_disp, py_disp, w, h, disp_w, disp_h, win)
                lx, ly = _map_display_to_original(lx_disp, ly_disp, w, h, disp_w, disp_h, win)
                from math import hypot
                dist_px = hypot(px - lx, py - ly)
                # Save record and update map
                distance_records[idx] = {
                    'pred_x': px,
                    'pred_y': py,
                    'label_x': lx,
                    'label_y': ly,
                    'distance_px': float(dist_px),
                }
                distances_map[idx] = float(dist_px)
                # Persist distances to CSV
                try:
                    save_distance_records(distances_path, distance_records)
                except Exception:
                    pass
                labels[idx] = LABEL_WRONG
                idx += 1
                if idx % save_every == 0:
                    save_progress(video_key, labels)
            else:
                labels[idx] = KEY_MAP[key]
                idx += 1
                if idx % save_every == 0:
                    save_progress(video_key, labels)

        elif key in (ARROW_LEFT, ARROW_LEFT_MAC, 81, 2):
            # Undo: go back one frame and remove its label
            if idx > 0:
                idx -= 1
                labels.pop(idx, None)

        elif key in (ARROW_RIGHT, ARROW_RIGHT_MAC, 83, 3):
            # Skip: treat as no ball
            labels[idx] = LABEL_NOBALL
            idx += 1

    cv2.destroyAllWindows()
    save_progress(video_key, labels)

    # ── Final report ──────────────────────────────────────────────────────────
    counts, labeled, recall_denom, prec_denom, captured, recall, precision, ecr_f1, \
        weighted_precision, weighted_ecr_f1, wrong_counts = compute_stats(labels, total, args.distances, distances_map)

    print()
    print("=" * 52)
    print("EFFECTIVE CAPTURE RATE  —  Final Report")
    print("=" * 52)
    print(f"Frames labeled : {labeled} / {total}")
    print()
    print(f"  Direct detection : {counts[LABEL_DIRECT]:>5}")
    print(f"  Interpolated     : {counts[LABEL_INTERP]:>5}")
    print(f"  Fail  (FN miss)  : {counts[LABEL_FAIL]:>5}   hurts recall")
    print(f"  Wrong (FP box)   : {counts[LABEL_WRONG]:>5}   hurts precision")
    print(f"  No ball (skip)   : {counts[LABEL_NOBALL]:>5}   excluded")
    print()
    print(f"  Capture Recall    : {recall*100:.1f}%  "
          f"({captured} / {recall_denom}  captured / ball-present)")
    print(f"  Capture Precision : {precision*100:.1f}%  "
          f"({captured} / {prec_denom}  captured / box-shown)")
    print(f"  Weighted Precision: {weighted_precision*100:.1f}%  (distance-weighted wrong penalties)")
    print(f"  ECR F1            : {ecr_f1*100:.1f}%")
    print(f"  Weighted ECR F1   : {weighted_ecr_f1*100:.1f}%")
    print()
    print("  Wrong breakdown: ")
    print(f"    near (d<10) — near-perfect / likely correct: {wrong_counts.get('near',0)}")
    print(f"    close miss (10<=d<25) — still useful: {wrong_counts.get('close',0)}")
    print(f"    far miss (25<=d<50) — low utility: {wrong_counts.get('far',0)}")
    print(f"    hallucination (d>=50) — unusable: {wrong_counts.get('hallu',0)}")
    print(f"    unknown distance: {wrong_counts.get('unknown',0)}")
    print("=" * 52)


if __name__ == "__main__":
    main()
