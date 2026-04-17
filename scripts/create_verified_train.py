"""
Create data/tracknet_merged/verified_train.csv and verified_val.csv

Strategy: for each verified frame (in fix_log.csv), include it as the TARGET
(frame_t) along with its 2 preceding context frames from the full dataset.
This satisfies TrackNet's 3-consecutive-frame requirement while ensuring every
training TARGET uses a manually-verified label.

Context frames (t-2, t-1) use whatever label is already in the full CSV —
they only exist to give the model temporal context; the loss is evaluated on
frame_t only, so their label quality doesn't affect training targets.

COORDINATE SPACE NOTE
---------------------
fix_log.csv stores x,y in MODEL space (0–640, 0–360) because interactive_label.py
divides mouse clicks by DISPLAY_SCALE=2 before saving.

TrackNetDataset.__getitem__ scales labels via:
    cx = x * INPUT_W / orig_w

where orig_w is the frame's on-disk resolution (e.g. 1280). If we write model-space
coordinates directly, the training code halves them again → Gaussian lands at
(x/2, y/2) → actively wrong training signal.

Fix: scale fix_log coordinates from model space → original frame space when writing,
so the training code's division brings them back to model space correctly.
"""
import csv
from collections import defaultdict
from pathlib import Path

import cv2


MODEL_W, MODEL_H = 640, 360


def build_fix_map(fix_log_path):
    """Return {(game, stem): fix_row} — last edit action wins per frame."""
    fix_map = {}
    with open(fix_log_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["game"], row["stem"])
            fix_map[key] = row
    return fix_map


def parse_frame_num(fp_str):
    stem = Path(fp_str.replace("\\", "/")).stem
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return -1


def get_game_stem(fp_str):
    fp = fp_str.replace("\\", "/")
    parts = fp.split("/")
    if len(parts) < 2:
        return None, None
    return parts[-2], parts[-1].rsplit(".", 1)[0]


def build_full_index(csv_path):
    """Return {game: {frame_num: row}} from the full train/val CSV."""
    index = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            game, stem = get_game_stem(row["frame_path"])
            if game is None:
                continue
            num = parse_frame_num(row["frame_path"])
            if num >= 0:
                index[game][num] = row
    return index


def get_frame_resolution(game_index):
    """Read one sample frame per game to determine on-disk resolution.

    Returns (orig_w, orig_h) by reading the first frame that exists on disk.
    Falls back to MODEL_W x MODEL_H if no frame can be read.
    """
    for row in list(game_index.values())[:20]:  # try first 20
        fp = row["frame_path"].replace("\\", "/")
        p = Path(fp)
        if not p.exists():
            p = Path("data") / p
        if p.exists():
            # Try .txt sidecar first (fast, written by --preprocess)
            sidecar = p.parent / f"{p.stem}_640x360.txt"
            if sidecar.exists():
                try:
                    w, h = map(int, sidecar.read_text().split())
                    return w, h
                except Exception:
                    pass
            # Fall back to reading image header
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    return MODEL_W, MODEL_H  # fallback (no coordinate scaling needed)


def build_verified_split(full_csv, fix_map, out_path):
    """
    For each verified frame in fix_map that exists in full_csv, emit 3 rows:
      frame_{n-2}: context (label from full_csv, original frame space — unchanged)
      frame_{n-1}: context (label from full_csv, original frame space — unchanged)
      frame_{n}:   target  (label from fix_map, scaled from model → orig frame space)

    Rows are written in (game, frame_num) order so the dataset builder sees
    consecutive frame numbers in sequence.
    """
    index = build_full_index(full_csv)

    # Determine per-game on-disk resolution for coordinate scaling
    game_resolution = {}
    for game, game_index in index.items():
        w, h = get_frame_resolution(game_index)
        game_resolution[game] = (w, h)
        print(f"  {game}: on-disk resolution = {w}x{h}  "
              f"(scale fix_log coords by {w/MODEL_W:.2f}x, {h/MODEL_H:.2f}x)")

    to_write = {}  # {(game, num): row}

    matched = 0
    skipped_no_context = 0
    skipped_not_in_full = 0

    for (game, stem), fix_row in fix_map.items():
        if game not in index:
            skipped_not_in_full += 1
            continue
        game_index = index[game]
        num = parse_frame_num(stem)
        if num < 0:
            continue

        if num not in game_index:
            skipped_not_in_full += 1
            continue

        if (num - 2) not in game_index or (num - 1) not in game_index:
            skipped_no_context += 1
            continue

        # Context frames: use full-CSV labels unchanged (already in orig frame space)
        for ctx_num in (num - 2, num - 1):
            if (game, ctx_num) not in to_write:
                to_write[(game, ctx_num)] = game_index[ctx_num]

        # Target frame: override label with fix_map values, scaled to orig frame space
        orig_w, orig_h = game_resolution.get(game, (MODEL_W, MODEL_H))
        scale_x = orig_w / MODEL_W
        scale_y = orig_h / MODEL_H

        target_row = dict(game_index[num])
        target_row["visibility"] = fix_row["visibility"]
        if fix_row["visibility"] == "1" and float(fix_row.get("x", -1)) >= 0:
            # Scale model-space coords → original frame space so training code
            # (which does cx = x * INPUT_W / orig_w) produces correct model coords
            target_row["x"] = str(round(float(fix_row["x"]) * scale_x, 1))
            target_row["y"] = str(round(float(fix_row["y"]) * scale_y, 1))
        else:
            target_row["x"] = fix_row.get("x", "-1")
            target_row["y"] = fix_row.get("y", "-1")

        to_write[(game, num)] = target_row
        matched += 1

    print(f"  Verified targets matched: {matched}")
    print(f"  Skipped (not in full CSV): {skipped_not_in_full}")
    print(f"  Skipped (no t-2/t-1 context in full CSV): {skipped_no_context}")

    rows_sorted = [row for _, row in
                   sorted(to_write.items(), key=lambda kv: (kv[0][0], kv[0][1]))]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "visibility", "x", "y"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    return matched, len(rows_sorted)


def main():
    root = Path("data/tracknet_merged")
    fix_map = build_fix_map(root / "fix_log.csv")
    print(f"fix_log unique (game, stem) pairs: {len(fix_map)}")

    for split in ("train", "val"):
        print(f"\nBuilding verified_{split}.csv ...")
        targets, total_rows = build_verified_split(
            root / f"{split}.csv",
            fix_map,
            root / f"verified_{split}.csv",
        )
        print(f"  -> verified_{split}.csv: {total_rows} rows "
              f"(~{targets} target sequences + context)")


if __name__ == "__main__":
    main()
