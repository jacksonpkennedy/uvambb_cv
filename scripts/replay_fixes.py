"""Replay the persisted fix log onto the current merged train/val CSVs.

Normally merge_labels.py does this automatically at the end of a merge, but
this standalone script is useful for:
  - Testing replay without re-running the full merge
  - Re-applying fixes after manually editing CSVs
  - Debugging replay issues (lots of "missing" entries usually means the
    log references frames that aren't in the merged set anymore)

Usage:
    python replay_fixes.py [--data data/tracknet_merged]
"""
import argparse
import csv
from pathlib import Path

import scripts.fix_log as fix_log
from scripts.csv_backup import backup_csvs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tracknet_merged")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    log_file = fix_log.log_path(data_dir)
    if not log_file.exists():
        print(f"No fix log at {log_file}. Nothing to replay.")
        return

    entries = fix_log.read_entries(data_dir)
    print(f"Loaded {len(entries)} fix-log entries from {log_file}")

    if not args.no_backup:
        backup_csvs(args.data, tag="replay_fixes")

    total = {"remove": 0, "reposition": 0, "add": 0, "zero": 0, "missing": 0}
    for split in ("train", "val"):
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        stats = fix_log.apply_to_rows(rows, entries)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame_path", "visibility", "x", "y"])
            w.writeheader()
            w.writerows(rows)
        print(f"  [{split}] {stats}")
        for k in total:
            total[k] += stats[k]

    print(f"\nTotals: {total}")
    if total["missing"]:
        print(f"  NOTE: {total['missing']} entries referenced frames not in "
              "the current merged set — this is normal if those frames were "
              "removed from source data.")


if __name__ == "__main__":
    main()
