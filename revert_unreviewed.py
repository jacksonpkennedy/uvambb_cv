"""Revert position repositioning for frames past the manual review cutoff.

The user manually reviewed position_mismatch images in stem-alphabetical order
up to frame number 3513 (the last bad frame listed). Any reposition applied
to a frame with stem number > 3513 was NOT manually reviewed and should be
reverted to the backup values.

Also cross-checks the zero-out operation: reports any visibility=1 -> 0
transition whose frame stem number is > 3513 (those should not have been
zeroed since they were not reviewed).
"""
import csv
import re
import shutil
from pathlib import Path

REVIEW_CUTOFF = 3513  # last manually reviewed frame number
DATA_DIR = Path("data/tracknet_merged")

_stem_re = re.compile(r"(\d+)(?!.*\d)")  # last run of digits in stem


def frame_num(frame_path: str) -> int:
    stem = Path(frame_path).stem
    m = _stem_re.search(stem)
    return int(m.group(1)) if m else -1


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_path", "visibility", "x", "y"])
        w.writeheader()
        w.writerows(rows)


def process(split: str) -> None:
    cur_path = DATA_DIR / f"{split}.csv"
    bak_path = DATA_DIR / f"{split}.csv.bak"
    if not bak_path.exists():
        print(f"  [skip] {bak_path} not found")
        return

    cur = load_rows(cur_path)
    bak = load_rows(bak_path)

    # Save a pre-revert snapshot so we never destroy the current state blindly
    pre_revert = cur_path.with_suffix(".csv.prerevert")
    shutil.copy2(cur_path, pre_revert)

    bak_by_path = {r["frame_path"]: r for r in bak}

    reverted_pos = 0
    kept_pos_reviewed = 0
    zero_out_in_review = 0
    zero_out_past_review = []
    unchanged_pos_past = 0

    for row in cur:
        b = bak_by_path.get(row["frame_path"])
        if b is None:
            continue
        num = frame_num(row["frame_path"])

        # Case A: backup had label, current was zeroed (bad-label removal)
        if int(b["visibility"]) > 0 and int(row["visibility"]) == 0:
            if num > REVIEW_CUTOFF:
                zero_out_past_review.append(row["frame_path"])
            else:
                zero_out_in_review += 1
            continue

        # Case B: both have labels, but x/y differ => position was repositioned
        if (int(b["visibility"]) > 0 and int(row["visibility"]) > 0
                and (row["x"] != b["x"] or row["y"] != b["y"])):
            if num > REVIEW_CUTOFF:
                # revert
                row["x"] = b["x"]
                row["y"] = b["y"]
                reverted_pos += 1
            else:
                kept_pos_reviewed += 1

    write_rows(cur_path, cur)

    print(f"\n{split.upper()}:")
    print(f"  Position fixes reverted (frame > {REVIEW_CUTOFF}): {reverted_pos}")
    print(f"  Position fixes kept (frame <= {REVIEW_CUTOFF}):    {kept_pos_reviewed}")
    print(f"  Zero-outs in reviewed range (frame <= {REVIEW_CUTOFF}): {zero_out_in_review}")
    print(f"  Zero-outs past review cutoff (frame > {REVIEW_CUTOFF}):  {len(zero_out_past_review)}")
    if zero_out_past_review:
        print("    First 10 suspect zero-outs:")
        for p in zero_out_past_review[:10]:
            print(f"      {Path(p).stem}")
    print(f"  Pre-revert snapshot saved to {pre_revert.name}")


def main():
    for split in ("train", "val"):
        process(split)
    print("\nDone. If the revert looks wrong, restore from .csv.prerevert.")


if __name__ == "__main__":
    main()
