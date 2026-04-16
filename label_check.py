"""label_dheck.py

Create a 100-sample label-check set from the disagreement audit CSVs and
produce annotated images matching the repository's false_negatives style
(green circle at model prediction + "MODEL (conf)").

Usage examples:
  python label_dheck.py --n 100
  python label_dheck.py --n 100 --min-conf 0.2 --copy-orig

Outputs:
  label_check/false_negatives/{game}/... (annotated JPGs)
  label_check/images/{game}/... (optional copied originals)
  label_check/manifest.csv

This script is saved to session memory; paste it into the repo root as
`label_dheck.py` to run.
"""

import argparse
import csv
import os
import sys
import random
import shutil
from pathlib import Path

try:
    import cv2
except Exception as e:
    print("OpenCV not found. Install with: pip install opencv-python", file=sys.stderr)
    raise

INPUT_W = 640
INPUT_H = 360


def read_audit_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for r in reader:
            if len(r) < 4:
                continue
            frame_path = r[0].strip()
            try:
                px = float(r[1])
                py = float(r[2])
                conf = float(r[3])
            except Exception:
                continue
            rows.append({'frame_path': frame_path, 'pred_x': px, 'pred_y': py, 'pred_conf': conf, 'source_csv': str(path)})
    return rows


def resolve_frame_path(frame_path):
    p = Path(frame_path)
    if p.exists():
        return str(p)
    # Try relative to cwd
    p2 = Path.cwd() / frame_path
    if p2.exists():
        return str(p2)
    # Try cleaning backslashes
    p3 = Path(frame_path.replace('\\', '/'))
    if p3.exists():
        return str(p3)
    return None


def group_by_game(rows):
    groups = {}
    for r in rows:
        p = Path(r['frame_path'])
        game = p.parent.name
        groups.setdefault(game, []).append(r)
    return groups


def stratified_sample(groups, n, rng):
    sampled = []
    games = list(groups.keys())
    if not games:
        return []
    base = n // len(games)
    remainder = n - base * len(games)
    for g in games:
        items = groups[g]
        k = min(len(items), base)
        if k > 0:
            sampled += rng.sample(items, k)
    # pool remaining
    pool = []
    for g in games:
        remaining = [it for it in groups[g] if it not in sampled]
        pool += remaining
    k = min(len(pool), remainder)
    if k > 0:
        sampled += rng.sample(pool, k)
    # fill with random choices if still short
    if len(sampled) < n:
        all_rows = []
        for g in games:
            all_rows += groups[g]
        while len(sampled) < n and all_rows:
            sampled.append(rng.choice(all_rows))
    return sampled[:n]


def annotate_and_save(row, out_base, copy_orig=False):
    frame_path = resolve_frame_path(row['frame_path'])
    if frame_path is None:
        print(f"Missing frame: {row['frame_path']}", file=sys.stderr)
        return None
    img = cv2.imread(frame_path)
    if img is None:
        print(f"Failed to load: {frame_path}", file=sys.stderr)
        return None
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    x = int(round(row['pred_x']))
    y = int(round(row['pred_y']))
    x = max(0, min(INPUT_W - 1, x))
    y = max(0, min(INPUT_H - 1, y))
    color = (0, 255, 0)
    cv2.circle(img, (x, y), 8, color, 2, lineType=cv2.LINE_AA)
    text = f"MODEL ({row['pred_conf']:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.4
    ft = 1
    (tw, th), baseline = cv2.getTextSize(text, font, fs, ft)
    pos_x = x + 10
    if pos_x + tw > INPUT_W:
        pos_x = x - 10 - tw
        if pos_x < 0:
            pos_x = max(0, INPUT_W - tw - 1)
    pos_y = y
    if pos_y - th < 0:
        pos_y = th
    cv2.putText(img, text, (pos_x, pos_y), font, fs, color, ft, lineType=cv2.LINE_AA)
    p = Path(row['frame_path'])
    game = p.parent.name
    stem = p.name
    out_dir = Path(out_base) / 'false_negatives' / game
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / stem
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        print(f"Failed to write: {out_path}", file=sys.stderr)
        return None
    if copy_orig:
        img_dir = Path(out_base) / 'images' / game
        img_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(frame_path, img_dir / stem)
        except Exception as e:
            print(f"Failed to copy original: {e}", file=sys.stderr)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both')
    parser.add_argument('--min-conf', type=float, default=0.0)
    parser.add_argument('--outdir', default='label_check')
    parser.add_argument('--copy-orig', action='store_true', help='Copy original frames to label_check/images/')
    args = parser.parse_args()

    paths = []
    if args.split in ('train', 'both'):
        t = Path('output/disagreements/train_false_negatives.csv')
        if t.exists():
            paths.append(t)
    if args.split in ('val', 'both'):
        v = Path('output/disagreements/val_false_negatives.csv')
        if v.exists():
            paths.append(v)
    if not paths:
        print("No audit CSVs found under output/disagreements/", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in paths:
        rows += read_audit_csv(p)

    rows = [r for r in rows if r['pred_conf'] >= args.min_conf]
    rows_valid = []
    for r in rows:
        if resolve_frame_path(r['frame_path']) is not None:
            rows_valid.append(r)
    if not rows_valid:
        print("No valid rows after resolving frame paths", file=sys.stderr)
        sys.exit(1)

    groups = group_by_game(rows_valid)
    rng = random.Random(args.seed)
    sampled = stratified_sample(groups, args.n, rng)

    manifest_path = Path(args.outdir) / 'manifest.csv'
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['frame_path', 'annotated_path', 'pred_x', 'pred_y', 'pred_conf', 'source_csv'])
        saved = 0
        for row in sampled:
            outp = annotate_and_save(row, args.outdir, copy_orig=args.copy_orig)
            if outp:
                writer.writerow([row['frame_path'], outp, row['pred_x'], row['pred_y'], row['pred_conf'], row['source_csv']])
                saved += 1

    print(f"Wrote {saved} annotated images and manifest to {args.outdir}")


if __name__ == '__main__':
    main()
