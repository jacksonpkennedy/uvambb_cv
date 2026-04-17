"""label_check.py — Interactive sequential label viewer.

Shows manually-reviewed frames in order (game + frame number) with the
stored label drawn as a red dot.  Use arrow keys to step through and
identify where labels go bad.

Labels are stored in 640x360 model space.  The image is always resized
to 640x360 before drawing so coordinates are correct regardless of the
source frame resolution on disk.

Controls
--------
  →  / D / Space        Next frame
  ←  / A                Previous frame
  Shift+→  / Page Down  Skip forward  N frames (default 100)
  Shift+←  / Page Up    Skip backward N frames
  G                     Jump to frame number (enter in terminal)
  Q / ESC               Quit

Usage
-----
  python scripts/label_check.py
  python scripts/label_check.py --game game_01          # one game only
  python scripts/label_check.py --all                   # include invisible frames too
  python scripts/label_check.py --skip 50               # change skip size
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


MODEL_W, MODEL_H = 640, 360

ARROW_LEFT  = 2424832
ARROW_RIGHT = 2555904
ARROW_LEFT_MAC  = 63234
ARROW_RIGHT_MAC = 63235
PAGE_UP   = 2162688   # Windows Shift+← or PgUp
PAGE_DOWN = 2228224   # Windows Shift+→ or PgDn
PAGE_UP2  = 2097152   # alternative PgUp keycode
PAGE_DOWN2 = 2163712  # alternative PgDn keycode


def resolve_frame_path(fp_str):
    p = Path(fp_str)
    if p.exists():
        return p
    p2 = Path("data") / p
    if p2.exists():
        return p2
    return p  # return anyway; caller checks .exists()


def load_fix_log(fix_log_path):
    """Return {(game, stem): last_row} from fix_log.csv."""
    fix_map = {}
    with open(fix_log_path, newline='') as f:
        for row in csv.DictReader(f):
            fix_map[(row['game'], row['stem'])] = row
    return fix_map


def get_key(fp_str):
    fp = fp_str.replace('\\', '/')
    parts = fp.split('/')
    if len(parts) < 2:
        return None, None
    return parts[-2], parts[-1].rsplit('.', 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default=None, help='Filter to one game (e.g. game_01)')
    parser.add_argument('--all', action='store_true', help='Include invisible frames too')
    parser.add_argument('--csv', default='data/tracknet_merged/train.csv')
    parser.add_argument('--fix-log', default='data/tracknet_merged/fix_log.csv')
    parser.add_argument('--skip', type=int, default=100,
                        help='Number of frames to jump with Page Up/Down (default 100)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    fix_log_path = Path(args.fix_log)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        sys.exit(1)

    with open(csv_path, newline='') as f:
        all_rows = list(csv.DictReader(f))

    # Filter to manually reviewed frames
    if fix_log_path.exists():
        fix_map = load_fix_log(fix_log_path)
        verified_keys = set(fix_map.keys())
        rows = [r for r in all_rows if get_key(r['frame_path']) in
                {(g, s) for g, s in verified_keys}]
        print(f"Loaded {len(rows)} manually reviewed frames from fix_log.")
    else:
        print("Warning: fix_log.csv not found — showing all train rows.")
        rows = all_rows
        fix_map = {}

    # Optional game filter
    if args.game:
        rows = [r for r in rows if get_key(r['frame_path'])[0] == args.game]
        print(f"Filtered to {args.game}: {len(rows)} rows.")

    # Optionally restrict to visible balls
    if not args.all:
        rows = [r for r in rows if r['visibility'] == '1']
        print(f"Visible only: {len(rows)} rows.")

    if not rows:
        print("No rows to show.")
        sys.exit(0)

    # Sort chronologically: game then frame number
    def sort_key(r):
        g, s = get_key(r['frame_path'])
        num = int(''.join(filter(str.isdigit, s))) if s else 0
        return (g or '', num)

    rows.sort(key=sort_key)
    total = len(rows)
    print(f"Showing {total} frames sequentially. Use ← → to navigate, Q to quit.")

    skip = args.skip
    win = "Label Check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    idx = 0

    while True:
        row = rows[idx]
        game, stem = get_key(row['frame_path'])
        full_path = resolve_frame_path(row['frame_path'])

        if full_path.exists():
            img = cv2.imread(str(full_path))
        else:
            img = None

        if img is None:
            img = np.zeros((MODEL_H, MODEL_W, 3), np.uint8)
            cv2.putText(img, f"Image not found: {row['frame_path']}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Always resize to model space — labels are stored in 640x360 coordinates
        img = cv2.resize(img, (MODEL_W, MODEL_H))
        h, w = MODEL_H, MODEL_W

        # Draw label dot
        vis = row.get('visibility', '1')
        lx = float(row.get('x', row.get('label_x', -1)))
        ly = float(row.get('y', row.get('label_y', -1)))

        if vis == '1' and lx >= 0 and ly >= 0:
            draw_x = int(lx)
            draw_y = int(ly)
            # Crosshair for easier centering check
            cv2.circle(img, (draw_x, draw_y), 8, (0, 0, 255), 2)
            cv2.circle(img, (draw_x, draw_y), 2, (0, 0, 255), -1)
            cv2.line(img, (draw_x - 14, draw_y), (draw_x + 14, draw_y), (0, 0, 255), 1)
            cv2.line(img, (draw_x, draw_y - 14), (draw_x, draw_y + 14), (0, 0, 255), 1)
            coord_text = f"({draw_x}, {draw_y})"
        else:
            coord_text = "invisible"

        # HUD bar
        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.putText(img, f"[{idx+1}/{total}]  {game}  {stem}  label={coord_text}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, h - 26), (w, h), (20, 20, 20), -1)
        cv2.putText(img, f"← → step    PgUp/PgDn skip {skip}    G jump    Q quit",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow(win, img)
        key = cv2.waitKeyEx(0)

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ARROW_RIGHT, ARROW_RIGHT_MAC, ord('d'), ord('D'), ord(' '), 83, 3):
            idx = min(idx + 1, total - 1)
        elif key in (ARROW_LEFT, ARROW_LEFT_MAC, ord('a'), ord('A'), 81, 2):
            idx = max(idx - 1, 0)
        elif key in (PAGE_DOWN, PAGE_DOWN2, ord('n'), ord('N')):
            idx = min(idx + skip, total - 1)
        elif key in (PAGE_UP, PAGE_UP2, ord('p'), ord('P')):
            idx = max(idx - skip, 0)
        elif key in (ord('g'), ord('G')):
            cv2.destroyWindow(win)
            try:
                n = int(input(f"Jump to frame # (1–{total}): "))
                idx = max(0, min(total - 1, n - 1))
            except ValueError:
                pass
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1280, 720)

    cv2.destroyAllWindows()
    print(f"Stopped at frame {idx+1}/{total}  ({game} / {stem})")


if __name__ == "__main__":
    main()
