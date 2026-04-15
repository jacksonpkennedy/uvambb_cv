"""Apply label fixes based on find_disagreements.py output.

Reads the disagreement CSVs and patches the training/val CSVs:
  - false_positives: removes the label (visibility=0, x=-1, y=-1)
  - position_mismatch: optionally updates position to model's prediction
  - false_negatives: optionally adds labels where model detected a ball

Rows are keyed by (game, stem) so the same frame number in different games
is handled independently (prevents cross-game stem collisions).

Per-category --max-*-frame flags support PARTIAL manual review: set a cutoff
for any category you didn't finish reviewing. Fixes are only applied to
frames with stem-number <= cutoff for that category. Categories you fully
reviewed don't need a cutoff.

Usage:
    # Fully reviewed all three categories:
    python apply_fixes.py --data data/tracknet_merged --fix-positions --add-missed

    # Only reviewed position_mismatch up to frame 3513:
    python apply_fixes.py --data data/tracknet_merged \
        --fix-positions --add-missed --max-pm-frame 3513

    # Reviewed everything EXCEPT false_negatives past frame 5000:
    python apply_fixes.py --data data/tracknet_merged \
        --fix-positions --add-missed --max-fn-frame 5000
"""
import argparse
import csv
import os
import re
from pathlib import Path

from csv_backup import backup_csvs
import fix_log

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


_DIGIT_RE = re.compile(r"(\d+)(?!.*\d)")


def _key(path: str) -> tuple[str, str]:
    """(game, stem) identifier — unique across games."""
    p = Path(path)
    stem = p.stem.replace("_640x360", "")
    return (p.parent.name, stem)


def _frame_num(stem: str) -> int:
    m = _DIGIT_RE.search(stem)
    return int(m.group(1)) if m else -1


def _under_cutoff(stem: str, cutoff: int | None) -> bool:
    """True if this row should be touched given the per-category cutoff."""
    if cutoff is None:
        return True
    return _frame_num(stem) <= cutoff


def load_false_positives(disagree_dir: Path, split: str) -> set[tuple[str, str]]:
    fp_set = set()
    p = disagree_dir / f"{split}_false_positives.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                fp_set.add(_key(row["frame_path"]))
    return fp_set


def load_position_mismatches(disagree_dir: Path,
                             split: str) -> dict[tuple[str, str], dict]:
    fixes = {}
    p = disagree_dir / f"{split}_position_mismatch.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                fixes[_key(row["frame_path"])] = {
                    "pred_x": float(row["pred_x"]),
                    "pred_y": float(row["pred_y"]),
                    "dist": float(row["distance_px"]),
                }
    return fixes


def load_false_negatives(disagree_dir: Path,
                         split: str) -> dict[tuple[str, str], dict]:
    missed = {}
    p = disagree_dir / f"{split}_false_negatives.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                missed[_key(row["frame_path"])] = {
                    "pred_x": float(row["pred_x"]),
                    "pred_y": float(row["pred_y"]),
                    "pred_conf": float(row["pred_conf"]),
                }
    return missed


def patch_csv(csv_path: str,
              fp_set: set, pos_fixes: dict, missed: dict,
              fix_positions: bool, add_missed: bool,
              min_dist: float, min_conf: float,
              max_fp_frame: int | None,
              max_pm_frame: int | None,
              max_fn_frame: int | None) -> tuple[dict, list[dict]]:
    """Patch a CSV in place and return (stats, log_entries). Log entries
    record every action so merge_labels.py can replay them later."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    removed = 0
    repositioned = 0
    added = 0
    skipped_fp = 0
    skipped_pm = 0
    skipped_fn = 0
    log_entries: list[dict] = []

    for row in rows:
        k = _key(row["frame_path"])
        game, stem = k

        # 1) Remove false positives (bad labels)
        if k in fp_set and int(row["visibility"]) > 0:
            if _under_cutoff(stem, max_fp_frame):
                row["visibility"] = "0"
                row["x"] = "-1"
                row["y"] = "-1"
                removed += 1
                log_entries.append({"game": game, "stem": stem, "action": "remove"})
                continue
            else:
                skipped_fp += 1

        # 2) Fix position mismatches (use model's prediction)
        if fix_positions and k in pos_fixes:
            fix = pos_fixes[k]
            if fix["dist"] >= min_dist and int(row["visibility"]) > 0:
                if _under_cutoff(stem, max_pm_frame):
                    row["x"] = f"{fix['pred_x']:.1f}"
                    row["y"] = f"{fix['pred_y']:.1f}"
                    repositioned += 1
                    log_entries.append({
                        "game": game, "stem": stem, "action": "reposition",
                        "visibility": 1,
                        "x": f"{fix['pred_x']:.1f}",
                        "y": f"{fix['pred_y']:.1f}",
                    })
                    continue
                else:
                    skipped_pm += 1

        # 3) Recover false negatives (add missed labels)
        if add_missed and k in missed:
            m = missed[k]
            if int(row["visibility"]) == 0 and m["pred_conf"] >= min_conf:
                if _under_cutoff(stem, max_fn_frame):
                    row["visibility"] = "1"
                    row["x"] = f"{m['pred_x']:.1f}"
                    row["y"] = f"{m['pred_y']:.1f}"
                    added += 1
                    log_entries.append({
                        "game": game, "stem": stem, "action": "add",
                        "visibility": 1,
                        "x": f"{m['pred_x']:.1f}",
                        "y": f"{m['pred_y']:.1f}",
                    })
                else:
                    skipped_fn += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "visibility", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)

    stats = {"removed": removed, "repositioned": repositioned, "added": added,
             "skipped_fp": skipped_fp, "skipped_pm": skipped_pm,
             "skipped_fn": skipped_fn, "total": len(rows)}
    return stats, log_entries


def main():
    parser = argparse.ArgumentParser(
        description="Apply label fixes from disagreement analysis")
    parser.add_argument("--data", default="data/tracknet_merged")
    parser.add_argument("--disagreements", default="output/disagreements")
    parser.add_argument("--fix-positions", action="store_true")
    parser.add_argument("--add-missed", action="store_true")
    parser.add_argument("--min-dist", type=float, default=15.0)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--max-fp-frame", type=int, default=None,
                        help="Only apply false-positive removal to stems with "
                             "frame number <= this (for partial FP review)")
    parser.add_argument("--max-pm-frame", type=int, default=None,
                        help="Only apply position fixes to stems with "
                             "frame number <= this (for partial PM review)")
    parser.add_argument("--max-fn-frame", type=int, default=None,
                        help="Only add missed labels for stems with "
                             "frame number <= this (for partial FN review)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip timestamped backup (NOT recommended)")
    args = parser.parse_args()

    disagree_dir = Path(args.disagreements)
    if not disagree_dir.exists():
        print(f"Error: {disagree_dir} not found. Run find_disagreements.py first.")
        return

    if not args.no_backup:
        backup_csvs(args.data, tag="apply_fixes")

    total = {"removed": 0, "repositioned": 0, "added": 0,
             "skipped_fp": 0, "skipped_pm": 0, "skipped_fn": 0}

    for split in ("train", "val"):
        csv_path = Path(args.data) / f"{split}.csv"
        if not csv_path.exists():
            print(f"Skipping {split}: {csv_path} not found")
            continue

        fp_set = load_false_positives(disagree_dir, split)
        pos_fixes = load_position_mismatches(disagree_dir, split)
        missed = load_false_negatives(disagree_dir, split)

        print(f"\n{split.upper()}:")
        print(f"  False positives to remove:    {len(fp_set)}"
              + (f" [cutoff <= {args.max_fp_frame}]" if args.max_fp_frame else ""))
        print(f"  Position mismatches found:    {len(pos_fixes)}"
              + (f" (fixing >= {args.min_dist:.0f}px)" if args.fix_positions else " (skipping)")
              + (f" [cutoff <= {args.max_pm_frame}]" if args.max_pm_frame else ""))
        print(f"  False negatives found:        {len(missed)}"
              + (f" (adding conf >= {args.min_conf:.2f})" if args.add_missed else " (skipping)")
              + (f" [cutoff <= {args.max_fn_frame}]" if args.max_fn_frame else ""))

        stats, log_entries = patch_csv(
            str(csv_path), fp_set, pos_fixes, missed,
            args.fix_positions, args.add_missed,
            args.min_dist, args.min_conf,
            args.max_fp_frame, args.max_pm_frame, args.max_fn_frame)

        # Persist log so a future re-merge replays these corrections
        fix_log.append(args.data, log_entries)

        for k in total:
            total[k] += stats[k]

        print(f"  -> Removed {stats['removed']} bad labels"
              + (f" (skipped {stats['skipped_fp']} past cutoff)" if stats['skipped_fp'] else ""))
        if args.fix_positions:
            print(f"  -> Repositioned {stats['repositioned']} labels"
                  + (f" (skipped {stats['skipped_pm']} past cutoff)" if stats['skipped_pm'] else ""))
        if args.add_missed:
            print(f"  -> Added {stats['added']} missed labels"
                  + (f" (skipped {stats['skipped_fn']} past cutoff)" if stats['skipped_fn'] else ""))
        print(f"  -> Total rows: {stats['total']}")

    if _HAS_WANDB:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "uvambb-cv"),
            entity=os.environ.get("WANDB_ENTITY"),
            job_type="label-fix",
            name="apply-fixes",
        )
        wandb.log({
            "fixes/removed": total["removed"],
            "fixes/repositioned": total["repositioned"],
            "fixes/added": total["added"],
        })
        art = wandb.Artifact("tracknet-labels", type="dataset")
        for split in ("train", "val"):
            p = Path(args.data) / f"{split}.csv"
            if p.exists():
                art.add_file(str(p))
        wandb.log_artifact(art)
        wandb.finish()

    print("\nDone! Next: retrain with cleaned labels:")
    print("  python tracknet.py --train --epochs 100 --batch 8")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
