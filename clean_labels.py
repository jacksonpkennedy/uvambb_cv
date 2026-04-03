"""Clean auto-labeled TrackNet CSVs by re-verifying with YOLO at higher confidence.

For each "visible" label, runs YOLO on the frame and checks whether it detects
a ball within 30px of the labeled position at conf >= 0.50. If not, the label
is demoted to invisible (visibility=0, x=-1, y=-1).

Usage:
    python clean_labels.py --data data/tracknet_merged --weights runs/detect/train/weights/best.pt
"""
import argparse
import csv
import math
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# Must match the ball class ID used in your fine-tuned YOLO model
CLS_BALL = 0
VERIFY_CONF = 0.50      # minimum YOLO confidence to trust a label
VERIFY_DIST_PX = 30.0   # max distance (px) between YOLO detection and label


def clean_csv(csv_path: str, model: YOLO, device: str) -> dict:
    """Re-verify labels in a CSV. Overwrites in place with cleaned labels."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    total_vis = 0
    removed = 0

    for i, row in enumerate(rows):
        if int(row["visibility"]) == 0:
            continue
        total_vis += 1

        img = cv2.imread(row["frame_path"])
        if img is None:
            row["visibility"] = "0"
            row["x"] = "-1"
            row["y"] = "-1"
            removed += 1
            continue

        lx, ly = float(row["x"]), float(row["y"])

        # Run YOLO at higher confidence
        results = model.predict(img, conf=VERIFY_CONF, iou=0.50,
                                verbose=False, device=device)
        det = results[0]

        # Check if any ball detection is near the labeled position
        found_match = False
        if det.boxes is not None and len(det.boxes) > 0:
            for box, cls_id, conf in zip(
                    det.boxes.xyxy, det.boxes.cls, det.boxes.conf):
                if int(cls_id) != CLS_BALL:
                    continue
                if float(conf) < VERIFY_CONF:
                    continue
                bx = float(box[0] + box[2]) / 2.0
                by = float(box[1] + box[3]) / 2.0
                dist = math.hypot(bx - lx, by - ly)
                if dist <= VERIFY_DIST_PX:
                    found_match = True
                    break

        if not found_match:
            row["visibility"] = "0"
            row["x"] = "-1"
            row["y"] = "-1"
            removed += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)} checked, {removed} removed so far ...")

    # Write back
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "visibility", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)

    return {"total_visible": total_vis, "removed": removed,
            "kept": total_vis - removed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tracknet_merged",
                        help="Directory containing train.csv / val.csv")
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt",
                        help="Fine-tuned YOLO weights")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading YOLO from {args.weights} ...")
    model = YOLO(args.weights)

    for split in ("train", "val"):
        csv_path = Path(args.data) / f"{split}.csv"
        if not csv_path.exists():
            continue
        print(f"\nCleaning {csv_path} ...")
        stats = clean_csv(str(csv_path), model, args.device)
        print(f"  {split}: {stats['total_visible']} visible labels → "
              f"kept {stats['kept']}, removed {stats['removed']} "
              f"({stats['removed']/max(stats['total_visible'],1)*100:.1f}% bad)")


if __name__ == "__main__":
    main()
