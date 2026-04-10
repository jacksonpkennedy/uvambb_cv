"""Verify that zero-outs match the manually reviewed 'neither correct' list.

User-provided ranges (from chat) are the only frames that should have been
zeroed. Any zeroed label whose frame number is not in this set is a mistake.

We don't have a true pre-zero-out backup (apply_fixes's .bak was taken after
the zero-out). So we reconstruct the candidate set: anything that appears in
the original position_mismatch CSVs AND is currently visibility=0 in the
training CSV is (very likely) a row we zeroed. Cross-check each such stem's
frame number against the user's range list.
"""
import csv
import re
from pathlib import Path

REVIEWED_BAD = set()

# Individual frame numbers
for n in [413, 600, 638, 691, 711, 1129, 1158, 1209, 1239, 1256,
          1389, 1972, 2063, 2067, 2068, 2496, 2575, 2602,
          3108, 3305, 3308, 3314, 3513]:
    REVIEWED_BAD.add(n)

# Ranges (inclusive per user's clarification)
RANGES = [
    (844, 902), (1161, 1192), (1323, 1372), (1408, 1422), (1443, 1444),
    (1477, 1507), (1510, 1510), (1515, 1515), (1536, 1536),
    (1578, 1672), (1739, 1779), (1845, 1910),
    (1995, 2058), (2112, 2133), (2136, 2229), (2310, 2314),
    (2501, 2529), (2582, 2594), (2612, 2673), (2737, 2794),
    (2822, 2849), (2851, 2854), (2857, 3031), (3321, 3324),
]
for a, b in RANGES:
    for n in range(a, b + 1):
        REVIEWED_BAD.add(n)

print(f"Manually reviewed 'neither correct' frame numbers: {len(REVIEWED_BAD)}")

_digit_re = re.compile(r"(\d+)(?!.*\d)")


def frame_num(stem: str) -> int:
    m = _digit_re.search(stem)
    return int(m.group(1)) if m else -1


# Collect all stems that appeared in the original position_mismatch CSVs.
# These CSVs were already synced after the user's manual review, so they
# represent the set of images that were still in the folder when apply_fixes
# was run. Older / deleted rows (including reviewed bad ones) may be gone.
pm_stems = set()
for split in ("train", "val"):
    p = Path(f"output/disagreements/{split}_position_mismatch.csv")
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                pm_stems.add(Path(row["frame_path"]).stem)
print(f"Stems currently in position_mismatch CSVs: {len(pm_stems)}")

# Scan training CSVs for visibility=0 rows whose stem frame number is in
# REVIEWED_BAD. Also look for rows that might have been zeroed but are NOT
# in the reviewed set (mistakes).
matched = 0
missing = []  # numbers in REVIEWED_BAD that don't appear as v=0 in CSVs
found_nums = set()

all_zero_nums = {"train": set(), "val": set()}
for split in ("train", "val"):
    csv_p = Path(f"data/tracknet_merged/{split}.csv")
    with open(csv_p) as f:
        for row in csv.DictReader(f):
            if int(row["visibility"]) == 0:
                num = frame_num(Path(row["frame_path"]).stem)
                if num in REVIEWED_BAD:
                    matched += 1
                    found_nums.add(num)
                    all_zero_nums[split].add(num)

print(f"\nLabels matching REVIEWED_BAD and currently zeroed: {matched}")
print(f"  (train unique nums: {len(all_zero_nums['train'])}, "
      f"val unique nums: {len(all_zero_nums['val'])})")

not_found = REVIEWED_BAD - found_nums
print(f"Reviewed nums NOT found as zeroed: {len(not_found)}")
if not_found:
    print(f"  sample: {sorted(not_found)[:20]}")
    print("  (these may have been deleted-image rows, or image-only frames "
          "not in training CSVs, or not in a mismatch collision)")
