"""Visualise auto-annotated ball labels for manual verification.

Draws a green circle at the labeled (x, y) on a random sample of frames
and saves them to a folder for quick visual inspection.

Usage:
    python verify_labels.py --data data/tracknet_merged --n 100
"""
import argparse
import csv
import random
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tracknet_merged",
                        help="Directory containing train.csv / val.csv")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of random visible frames to check")
    parser.add_argument("--output", default="label_check",
                        help="Output directory for annotated frames")
    parser.add_argument("--split", default="train",
                        help="Which split to check (train or val)")
    args = parser.parse_args()

    csv_path = Path(args.data) / f"{args.split}.csv"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all visible entries
    visible = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if int(row["visibility"]) > 0:
                visible.append(row)

    sample = random.sample(visible, min(args.n, len(visible)))
    print(f"Checking {len(sample)} random visible frames from {csv_path}")

    bad = 0
    for i, row in enumerate(sample):
        img = cv2.imread(row["frame_path"])
        if img is None:
            print(f"  MISSING: {row['frame_path']}")
            bad += 1
            continue

        x, y = float(row["x"]), float(row["y"])
        h, w = img.shape[:2]

        # Draw crosshair at labeled position
        cx, cy = int(x), int(y)
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), 2)     # green circle
        cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

        # Add text with coordinates
        label = f"({cx}, {cy})"
        cv2.putText(img, label, (cx + 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save
        fname = f"{i:04d}_{Path(row['frame_path']).stem}.jpg"
        cv2.imwrite(str(out_dir / fname), img)

    print(f"Saved {len(sample) - bad} annotated frames to {out_dir}/")
    print(f"Open the folder and scroll through — look for:")
    print(f"  - Green circles NOT on the ball (wrong label)")
    print(f"  - Ball visible but labeled as invisible (missing from this check)")
    print(f"  - Labels on players' hands/shoes instead of ball")


if __name__ == "__main__":
    main()
