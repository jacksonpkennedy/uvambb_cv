"""
Convert YOLO segmentation labels → TrackNet training CSV.

Reads basketball polygon annotations (class 0) from YOLO-seg format,
extracts centroids, and writes sequential CSVs for TrackNet training.

YOLO-seg format:  class x1 y1 x2 y2 ... xN yN  (normalized coords)
TrackNet CSV:     frame_path, visibility, x, y   (pixel coords at 640×360)

Usage:
    python convert_labels.py
    python convert_labels.py --data data/custom_annotations --out data/tracknet_labels
"""

import argparse
import csv
import re
from pathlib import Path

import cv2

BALL_CLS = 0
TRACKNET_W, TRACKNET_H = 640, 360


def extract_ball_center(label_path: Path, img_w: int, img_h: int
                        ) -> tuple | None:
    """Read YOLO-seg label, find basketball polygon, return centroid in pixels.

    Returns (cx_pixel, cy_pixel) at TrackNet resolution, or None if no ball.
    """
    if not label_path.exists():
        return None

    text = label_path.read_text().strip()
    if not text:
        return None

    for line in text.splitlines():
        parts = line.strip().split()
        cls_id = int(parts[0])
        if cls_id != BALL_CLS:
            continue

        # Remaining values are normalized x,y polygon vertices
        coords = list(map(float, parts[1:]))
        if len(coords) < 4:
            continue

        xs = coords[0::2]
        ys = coords[1::2]
        # Centroid of polygon (normalized)
        cx_norm = sum(xs) / len(xs)
        cy_norm = sum(ys) / len(ys)

        # Convert to TrackNet resolution
        cx = cx_norm * TRACKNET_W
        cy = cy_norm * TRACKNET_H
        return cx, cy

    return None


def get_frame_number(filename: str) -> int:
    """Extract frame number from filenames like 'frame_0036_jpg.rf.xxx.jpg'."""
    match = re.search(r"frame_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def convert(data_dir: str, output_dir: str):
    """Convert YOLO-seg annotations to TrackNet CSV format."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid"):
        img_dir = data_path / split / "images"
        lbl_dir = data_path / split / "labels"

        if not img_dir.exists():
            print(f"Skipping {split}: {img_dir} not found")
            continue

        # Gather all images, sorted by frame number for sequential ordering
        images = sorted(img_dir.glob("*.jpg"), key=lambda p: p.name)
        print(f"{split}: found {len(images)} images")

        # Deduplicate by frame number — Roboflow augmentations create
        # multiple versions of the same frame. For TrackNet we want one
        # entry per original frame.
        seen_frames = {}
        for img_path in images:
            frame_num = get_frame_number(img_path.name)
            if frame_num < 0:
                continue
            # Keep the first augmentation variant per frame
            if frame_num not in seen_frames:
                seen_frames[frame_num] = img_path

        # Sort by frame number for sequential ordering
        sorted_frames = sorted(seen_frames.items())
        print(f"  {len(sorted_frames)} unique frames after deduplication")

        csv_name = "train.csv" if split == "train" else "val.csv"
        csv_path = out_path / csv_name

        # Read one image to get dimensions (for label denormalization)
        sample_img = cv2.imread(str(sorted_frames[0][1]))
        img_h, img_w = sample_img.shape[:2]

        ball_count = 0
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["frame_path", "visibility",
                                                "x", "y"])
            writer.writeheader()

            for frame_num, img_path in sorted_frames:
                # Find matching label file
                lbl_name = img_path.stem + ".txt"
                lbl_path = lbl_dir / lbl_name

                center = extract_ball_center(lbl_path, img_w, img_h)

                if center is not None:
                    cx, cy = center
                    writer.writerow({
                        "frame_path": str(img_path),
                        "visibility": 1,
                        "x": f"{cx:.1f}",
                        "y": f"{cy:.1f}",
                    })
                    ball_count += 1
                else:
                    writer.writerow({
                        "frame_path": str(img_path),
                        "visibility": 0,
                        "x": "-1",
                        "y": "-1",
                    })

        print(f"  Wrote {csv_path}  "
              f"({ball_count}/{len(sorted_frames)} frames with ball)")

    print(f"\nDone. TrackNet labels saved to {out_path}/")
    print("Train with:  python tracknet.py --train --data "
          f"{out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YOLO-seg labels to TrackNet CSV format")
    parser.add_argument("--data", default="data/custom_annotations",
                        help="YOLO dataset root (with train/valid subdirs)")
    parser.add_argument("--out", default="data/tracknet_labels",
                        help="Output directory for TrackNet CSVs")
    args = parser.parse_args()
    convert(args.data, args.out)
