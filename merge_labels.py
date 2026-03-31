"""
Merge all TrackNet label sources into data/tracknet_merged/

Strategy: split per-game so val always contains a representative mix of
visible/empty frames from each game. The last 15% of each game's rows go
to val, the rest to train. This avoids landing val entirely in a halftime
or dead-ball section.

Sources:
  - data/tracknet_labels/          (manually verified YOLO-converted labels)
  - data/tracknet_autolabels/      (game_01 auto-labels)
  - data/tracknet_autolabels_02/   (game_02 auto-labels)
  - data/tracknet_autolabels_03/   (game_03 auto-labels)

Output:
  - data/tracknet_merged/train.csv
  - data/tracknet_merged/val.csv
"""

import csv
from pathlib import Path

SOURCES = [
    "data/tracknet_labels/train.csv",
    "data/tracknet_labels/val.csv",
    "data/tracknet_autolabels/train.csv",
    "data/tracknet_autolabels/val.csv",
    "data/tracknet_autolabels_02/train.csv",
    "data/tracknet_autolabels_02/val.csv",
    "data/tracknet_autolabels_03/train.csv",
    "data/tracknet_autolabels_03/val.csv",
]

OUTPUT_DIR = Path("data/tracknet_merged")
VAL_SPLIT  = 0.15


def _game_name(row: dict) -> str:
    """Extract the game directory name from a frame path."""
    return Path(row["frame_path"]).parent.name


def main():
    all_rows = []
    for src in SOURCES:
        p = Path(src)
        if not p.exists():
            print(f"  [skip] {src} — not found")
            continue
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        vis = sum(1 for r in rows if r["visibility"] == "1")
        print(f"  [load] {src}: {len(rows)} rows, {vis} visible")
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: No label sources found. Run auto_label_tracknet.py first.")
        return

    # Group rows by game, preserving sequential order within each game
    games: dict[str, list] = {}
    for row in all_rows:
        g = _game_name(row)
        games.setdefault(g, []).append(row)

    print(f"\nFound {len(games)} game(s): {list(games.keys())}")

    train_rows = []
    val_rows   = []

    for game, rows in games.items():
        n       = len(rows)
        val_n   = max(1, int(n * VAL_SPLIT))
        # Take the last val_n rows of each game as val (guaranteed to be
        # late-game active play, not halftime/pre-game dead sections)
        t_rows  = rows[: n - val_n]
        v_rows  = rows[n - val_n :]
        t_vis   = sum(1 for r in t_rows if r["visibility"] == "1")
        v_vis   = sum(1 for r in v_rows  if r["visibility"] == "1")
        print(f"  {game}: {len(t_rows)} train ({t_vis} vis), "
              f"{len(v_rows)} val ({v_vis} vis)")
        train_rows.extend(t_rows)
        val_rows.extend(v_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in [("train", train_rows), ("val", val_rows)]:
        out_path = OUTPUT_DIR / f"{split}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["frame_path", "visibility", "x", "y"])
            writer.writeheader()
            writer.writerows(rows)
        vis = sum(1 for r in rows if r["visibility"] == "1")
        print(f"  [{split}] {len(rows)} rows, {vis} visible "
              f"({vis / max(len(rows), 1) * 100:.1f}%) → {out_path}")

    total = len(train_rows) + len(val_rows)
    print(f"\nTotal: {total} rows across {len(games)} games.")
    print("Now retrain:")
    print("  python tracknet.py --train --data data/tracknet_merged "
          "--epochs 100 --batch 16 --workers 4")


if __name__ == "__main__":
    main()
