"""Sync disagreement CSVs to match the images currently in the review folders.

After manual review (deleting images you decided against), run this to
prune the disagreement CSVs so apply_fixes only acts on rows whose image
is still present.

Expects the new find_disagreements layout with game subfolders:
  output/disagreements/
    false_negatives/{game}/{stem}.jpg
    false_positives/{game}/{stem}.jpg
    position_mismatch/{game}/{stem}.jpg

Backs up the original CSVs to output/disagreements/backups/<timestamp>/
before overwriting.

Usage:
    python sync_disagreements.py
    python sync_disagreements.py --disagreements output/disagreements
"""
import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


CATEGORIES = {
    "false_negatives": "false_negatives",
    "false_positives": "false_positives",
    "position_mismatch": "position_mismatch",
}


def _key(path: str) -> tuple[str, str]:
    p = Path(path)
    return (p.parent.name, p.stem.replace("_640x360", ""))


def collect_kept_pairs(category_dir: Path) -> set[tuple[str, str]]:
    """Walk {category}/{game}/{stem}.jpg and return {(game, stem), ...}."""
    pairs = set()
    if not category_dir.exists():
        return pairs
    for game_dir in category_dir.iterdir():
        if not game_dir.is_dir():
            continue
        game = game_dir.name
        for img in game_dir.glob("*.jpg"):
            pairs.add((game, img.stem))
    return pairs


def sync_csv(csv_path: Path, kept: set[tuple[str, str]]) -> tuple[int, int]:
    """Filter a disagreement CSV in place. Returns (kept_count, dropped_count)."""
    if not csv_path.exists():
        return (0, 0)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if _key(r["frame_path"]) in kept]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    # Caller reports total before/after
    return len(rows), 0  # dropped computed by caller


def backup_disagreements(disagree_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = disagree_dir / "backups" / ts
    bdir.mkdir(parents=True, exist_ok=True)
    for csv_file in disagree_dir.glob("*.csv"):
        shutil.copy2(csv_file, bdir / csv_file.name)
    print(f"[backup] disagreement CSVs -> {bdir}")
    return bdir


def main():
    parser = argparse.ArgumentParser(
        description="Sync disagreement CSVs to the current review folder state")
    parser.add_argument("--disagreements", default="output/disagreements")
    args = parser.parse_args()

    disagree_dir = Path(args.disagreements)
    if not disagree_dir.exists():
        print(f"Error: {disagree_dir} not found.")
        return

    backup_disagreements(disagree_dir)

    for category, folder in CATEGORIES.items():
        kept = collect_kept_pairs(disagree_dir / folder)
        print(f"\n{category}: {len(kept)} image(s) kept in folder")

        for split in ("train", "val"):
            csv_path = disagree_dir / f"{split}_{category}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path) as f:
                before = sum(1 for _ in csv.DictReader(f))
            after, _ = sync_csv(csv_path, kept)
            print(f"  {split}: {before} -> {after} (dropped {before - after})")

    print("\nDone. Disagreement CSVs now match review folder contents.")


if __name__ == "__main__":
    main()
