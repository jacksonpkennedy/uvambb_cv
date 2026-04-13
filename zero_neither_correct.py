"""Zero out labels for position_mismatch frames where NEITHER label nor model is correct.

Workflow: during position_mismatch review, move images where neither the red
circle (label) nor the green circle (model) is correct into a '_neither/'
subfolder. Keep that subfolder organized by game:

    output/disagreements/position_mismatch/
        game_01/            <- kept (model or label correct)
        game_02/            <- kept
        _neither/
            game_01/        <- neither correct → zero the label
            game_02/
            ...

This script walks the _neither/ tree and zeros the matching rows in
train.csv / val.csv. Matching is by (game, stem) to avoid cross-game
collisions.

Alternative: pass --list path/to/file.txt where each line is "game/stem"
(e.g., "game_02/frame_001578").

Backs up train.csv / val.csv to data/tracknet_merged/backups/<timestamp>/
before mutating.

Usage:
    python zero_neither_correct.py
    python zero_neither_correct.py --neither-dir output/disagreements/position_mismatch/_neither
    python zero_neither_correct.py --list neither_correct.txt
"""
import argparse
import csv
from pathlib import Path

from csv_backup import backup_csvs


def collect_from_dir(neither_dir: Path) -> set[tuple[str, str]]:
    """Walk {neither_dir}/{game}/{stem}.jpg -> {(game, stem), ...}."""
    pairs = set()
    if not neither_dir.exists():
        return pairs
    for game_dir in neither_dir.iterdir():
        if not game_dir.is_dir():
            continue
        game = game_dir.name
        for img in game_dir.glob("*.jpg"):
            pairs.add((game, img.stem))
    return pairs


def collect_from_list(list_path: Path) -> set[tuple[str, str]]:
    """Parse 'game/stem' or 'game\\stem' lines from a text file."""
    pairs = set()
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace("\\", "/")
            if "/" not in line:
                print(f"  [skip] bad line (no game prefix): {line}")
                continue
            game, stem = line.rsplit("/", 1)
            stem = Path(stem).stem  # strip any .jpg/.png extension
            pairs.add((Path(game).name, stem))
    return pairs


def _key(path: str) -> tuple[str, str]:
    p = Path(path)
    return (p.parent.name, p.stem.replace("_640x360", ""))


def zero_rows(csv_path: Path, targets: set[tuple[str, str]]) -> dict:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    zeroed = 0
    already_zero = 0
    for row in rows:
        if _key(row["frame_path"]) in targets:
            if int(row["visibility"]) > 0:
                row["visibility"] = "0"
                row["x"] = "-1"
                row["y"] = "-1"
                zeroed += 1
            else:
                already_zero += 1

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_path", "visibility", "x", "y"])
        w.writeheader()
        w.writerows(rows)

    return {"zeroed": zeroed, "already_zero": already_zero, "total": len(rows)}


def main():
    parser = argparse.ArgumentParser(
        description="Zero labels for 'neither correct' position_mismatch frames")
    parser.add_argument("--data", default="data/tracknet_merged")
    parser.add_argument("--neither-dir",
                        default="output/disagreements/position_mismatch/_neither",
                        help="Folder of images marked 'neither correct' "
                             "(expects {game}/{stem}.jpg layout)")
    parser.add_argument("--list", dest="list_path", default=None,
                        help="Alternative: text file with 'game/stem' per line")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if args.list_path:
        targets = collect_from_list(Path(args.list_path))
        src_desc = f"list file {args.list_path}"
    else:
        targets = collect_from_dir(Path(args.neither_dir))
        src_desc = f"folder {args.neither_dir}"

    if not targets:
        print(f"No targets found in {src_desc}. Nothing to do.")
        return

    print(f"Loaded {len(targets)} (game, stem) target(s) from {src_desc}")

    if not args.no_backup:
        backup_csvs(args.data, tag="zero_neither")

    total_zeroed = 0
    for split in ("train", "val"):
        csv_path = Path(args.data) / f"{split}.csv"
        if not csv_path.exists():
            continue
        stats = zero_rows(csv_path, targets)
        total_zeroed += stats["zeroed"]
        print(f"{split}: zeroed {stats['zeroed']}, "
              f"already zero {stats['already_zero']} / {stats['total']} rows")

    print(f"\nTotal zeroed: {total_zeroed}")
    print("Next: retrain with  python tracknet.py --train --epochs 100 --batch 8")


if __name__ == "__main__":
    main()
