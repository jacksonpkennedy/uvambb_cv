"""Apply label fixes based on find_disagreements.py output.

Reads the disagreement CSVs and patches the training/val CSVs:
  - false_positives: removes the label (visibility=0, x=-1, y=-1)
  - position_mismatch: optionally updates position to model's prediction
  - false_negatives: optionally adds labels where model detected a ball

Usage:
    # Conservative: only remove bad labels
    python apply_fixes.py --data data/tracknet_merged

    # Also fix positions where model and label disagree by >20px
    python apply_fixes.py --data data/tracknet_merged --fix-positions

    # Also recover missed labels (review output/disagreements/false_negatives/ first!)
    python apply_fixes.py --data data/tracknet_merged --add-missed --min-conf 0.7

    # All fixes
    python apply_fixes.py --data data/tracknet_merged --fix-positions --add-missed
"""
import argparse
import csv
import shutil
from pathlib import Path


def load_false_positives(disagree_dir: Path, split: str) -> set:
    """Frame paths where label says ball but model doesn't see one."""
    fp_set = set()
    p = disagree_dir / f"{split}_false_positives.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                fp_set.add(row["frame_path"])
    return fp_set


def load_position_mismatches(disagree_dir: Path, split: str) -> dict:
    """Frame paths where both see ball but positions differ."""
    fixes = {}
    p = disagree_dir / f"{split}_position_mismatch.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                fixes[row["frame_path"]] = {
                    "pred_x": float(row["pred_x"]),
                    "pred_y": float(row["pred_y"]),
                    "dist": float(row["distance_px"]),
                }
    return fixes


def load_false_negatives(disagree_dir: Path, split: str) -> dict:
    """Frame paths where model sees ball but label says none."""
    missed = {}
    p = disagree_dir / f"{split}_false_negatives.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                missed[row["frame_path"]] = {
                    "pred_x": float(row["pred_x"]),
                    "pred_y": float(row["pred_y"]),
                    "pred_conf": float(row["pred_conf"]),
                }
    return missed


def patch_csv(csv_path: str, fp_set: set, pos_fixes: dict, missed: dict,
              fix_positions: bool, add_missed: bool,
              min_dist: float, min_conf: float) -> dict:
    """Patch a training CSV in place. Returns stats."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    removed = 0
    repositioned = 0
    added = 0

    for row in rows:
        fp = row["frame_path"]

        # 1) Remove false positives (bad labels)
        if fp in fp_set and int(row["visibility"]) > 0:
            row["visibility"] = "0"
            row["x"] = "-1"
            row["y"] = "-1"
            removed += 1
            continue

        # 2) Fix position mismatches (use model's prediction)
        if fix_positions and fp in pos_fixes:
            fix = pos_fixes[fp]
            if fix["dist"] >= min_dist and int(row["visibility"]) > 0:
                row["x"] = f"{fix['pred_x']:.1f}"
                row["y"] = f"{fix['pred_y']:.1f}"
                repositioned += 1
                continue

        # 3) Recover false negatives (add missed labels)
        if add_missed and fp in missed:
            m = missed[fp]
            if int(row["visibility"]) == 0 and m["pred_conf"] >= min_conf:
                row["visibility"] = "1"
                row["x"] = f"{m['pred_x']:.1f}"
                row["y"] = f"{m['pred_y']:.1f}"
                added += 1

    # Write back
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "visibility", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)

    return {"removed": removed, "repositioned": repositioned,
            "added": added, "total": len(rows)}


def main():
    parser = argparse.ArgumentParser(
        description="Apply label fixes from disagreement analysis")
    parser.add_argument("--data", default="data/tracknet_merged",
                        help="Directory with train.csv / val.csv")
    parser.add_argument("--disagreements", default="output/disagreements",
                        help="Directory with disagreement CSVs from find_disagreements.py")
    parser.add_argument("--fix-positions", action="store_true",
                        help="Update position mismatches to model predictions")
    parser.add_argument("--add-missed", action="store_true",
                        help="Add labels for false negatives (model found ball, no label). "
                             "Review output/disagreements/false_negatives/ visually first!")
    parser.add_argument("--min-dist", type=float, default=20.0,
                        help="Only fix positions with distance >= this (default 20px)")
    parser.add_argument("--min-conf", type=float, default=0.7,
                        help="Only add missed labels above this model confidence (default 0.7)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating .bak backup of original CSVs")
    args = parser.parse_args()

    disagree_dir = Path(args.disagreements)
    if not disagree_dir.exists():
        print(f"Error: {disagree_dir} not found. Run find_disagreements.py first.")
        return

    for split in ("train", "val"):
        csv_path = Path(args.data) / f"{split}.csv"
        if not csv_path.exists():
            print(f"Skipping {split}: {csv_path} not found")
            continue

        # Backup original
        if not args.no_backup:
            bak = csv_path.with_suffix(f".csv.bak")
            shutil.copy2(csv_path, bak)
            print(f"Backed up {csv_path} -> {bak}")

        # Load disagreements
        fp_set = load_false_positives(disagree_dir, split)
        pos_fixes = load_position_mismatches(disagree_dir, split)
        missed = load_false_negatives(disagree_dir, split)

        print(f"\n{split.upper()}:")
        print(f"  False positives to remove:    {len(fp_set)}")
        print(f"  Position mismatches found:    {len(pos_fixes)}"
              + (f" (fixing >= {args.min_dist:.0f}px)" if args.fix_positions else " (skipping, use --fix-positions)"))
        print(f"  False negatives found:        {len(missed)}"
              + (f" (adding conf >= {args.min_conf:.2f})" if args.add_missed else " (skipping, use --add-missed)"))

        stats = patch_csv(str(csv_path), fp_set, pos_fixes, missed,
                          args.fix_positions, args.add_missed,
                          args.min_dist, args.min_conf)

        print(f"  -> Removed {stats['removed']} bad labels")
        if args.fix_positions:
            print(f"  -> Repositioned {stats['repositioned']} labels")
        if args.add_missed:
            print(f"  -> Added {stats['added']} missed labels")
        print(f"  -> Total rows: {stats['total']}")

    print("\nDone! Next: retrain with cleaned labels:")
    print("  python tracknet.py --train --epochs 100 --batch 8")


if __name__ == "__main__":
    main()
