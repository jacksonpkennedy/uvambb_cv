"""
Auto-label consecutive video frames for TrackNet training using YOLO + SAHI.

Extracts consecutive frames from game videos, runs the existing fine-tuned
YOLO model + SAHI fallback to detect the basketball, and outputs TrackNet-
compatible CSV files (frame_path, visibility, x, y).

Usage:
    python auto_label_tracknet.py --video data/game_01.mp4 [--max-frames 5000]
    python auto_label_tracknet.py --video data/game_01.mp4 data/game_02.mp4
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

# Reuse constants from main pipeline
from main import (
    CLS_BALL, CLS_HOOP,
    COCO_BALL_CLS, COCO_BALL_CONF,
    SAHI_CONF_THRESH, SAHI_SLICE_SIZE, SAHI_OVERLAP_RATIO,
    SAHI_CROP_PAD, SAHI_MAX_LOST,
    BALL_MAX_BBOX_AREA, BALL_MIN_BBOX_AREA,
    BALL_MAX_TELEPORT_PX, BALL_TELEPORT_GRACE,
    build_court_roi, ball_in_court_region,
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def detect_ball_frame(model, sahi_model, frame, roi_poly,
                      frame_w, frame_h, device, infer_imgsz, use_half,
                      last_center, frames_since_seen):
    """Run YOLO + SAHI on a single frame. Return (cx, cy, conf) or None."""
    ball_dets = []

    # --- YOLO detection (no tracking, just single-frame) ---
    det_res = model.predict(
        frame, conf=0.10, iou=0.50,
        imgsz=infer_imgsz, device=device, half=use_half,
        verbose=False,
    )[0]

    if det_res.boxes is not None:
        for box, cls_id, conf in zip(
                det_res.boxes.xyxy, det_res.boxes.cls, det_res.boxes.conf):
            cls_id = int(cls_id)
            conf_val = float(conf)
            if cls_id == CLS_BALL:
                box_list = list(map(int, box.tolist()))
                area = (box_list[2] - box_list[0]) * (box_list[3] - box_list[1])
                if BALL_MIN_BBOX_AREA <= area <= BALL_MAX_BBOX_AREA:
                    cx = (box_list[0] + box_list[2]) / 2.0
                    cy = (box_list[1] + box_list[3]) / 2.0
                    if ball_in_court_region(roi_poly, cx, cy, frame_w):
                        ball_dets.append((cx, cy, conf_val))

    # --- SAHI fallback ---
    if not ball_dets and sahi_model is not None:
        if last_center is not None and frames_since_seen < SAHI_MAX_LOST:
            # Targeted SAHI around last known position
            lx, ly = last_center
            x1 = max(0, int(lx - SAHI_CROP_PAD))
            y1 = max(0, int(ly - SAHI_CROP_PAD))
            x2 = min(frame_w, int(lx + SAHI_CROP_PAD))
            y2 = min(frame_h, int(ly + SAHI_CROP_PAD))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                result = get_sliced_prediction(
                    image=crop,
                    detection_model=sahi_model,
                    slice_height=SAHI_SLICE_SIZE,
                    slice_width=SAHI_SLICE_SIZE,
                    overlap_height_ratio=SAHI_OVERLAP_RATIO,
                    overlap_width_ratio=SAHI_OVERLAP_RATIO,
                    verbose=0,
                )
                for pred in result.object_prediction_list:
                    if int(pred.category.id) != COCO_BALL_CLS:
                        continue
                    if pred.score.value < COCO_BALL_CONF:
                        continue
                    bb = pred.bbox
                    # Convert crop-relative back to frame coords
                    cx = x1 + (bb.minx + bb.maxx) / 2.0
                    cy = y1 + (bb.miny + bb.maxy) / 2.0
                    if ball_in_court_region(roi_poly, cx, cy, frame_w):
                        ball_dets.append((cx, cy, pred.score.value))
        else:
            # Full-frame SAHI (expensive, only when ball is lost)
            result = get_sliced_prediction(
                image=frame,
                detection_model=sahi_model,
                slice_height=SAHI_SLICE_SIZE,
                slice_width=SAHI_SLICE_SIZE,
                overlap_height_ratio=SAHI_OVERLAP_RATIO,
                overlap_width_ratio=SAHI_OVERLAP_RATIO,
                verbose=0,
            )
            for pred in result.object_prediction_list:
                if int(pred.category.id) != COCO_BALL_CLS:
                    continue
                if pred.score.value < COCO_BALL_CONF:
                    continue
                bb = pred.bbox
                cx = (bb.minx + bb.maxx) / 2.0
                cy = (bb.miny + bb.maxy) / 2.0
                if ball_in_court_region(roi_poly, cx, cy, frame_w):
                    ball_dets.append((cx, cy, pred.score.value))

    if not ball_dets:
        return None

    # Pick highest confidence detection
    best = max(ball_dets, key=lambda d: d[2])

    # Teleport rejection against last known position
    if last_center is not None and frames_since_seen <= BALL_TELEPORT_GRACE:
        max_dist = BALL_MAX_TELEPORT_PX * (frame_w / 1920.0) * frames_since_seen
        dist = ((best[0] - last_center[0]) ** 2 +
                (best[1] - last_center[1]) ** 2) ** 0.5
        if dist > max_dist:
            return None

    return best


@torch.inference_mode()
def auto_label_video(video_path: str, output_dir: str,
                     weights: str, max_frames: int = 5000,
                     conf_thresh: float = 0.15,
                     start_frame: int = 0):
    """Extract consecutive frames from a video and auto-label ball positions."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}  |  {frame_w}x{frame_h}  |  "
          f"{fps:.0f}fps  |  {total_frames} frames")

    device = get_device()
    infer_imgsz = 1280 if device.startswith("cuda") else 640
    use_half = device.startswith("cuda")

    # Load models
    print("Loading fine-tuned YOLO model ...")
    model = YOLO(weights)
    model.to(device)

    print("Loading SAHI model (COCO sports ball) ...")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="yolo11s.pt",
        confidence_threshold=SAHI_CONF_THRESH,
        device=device,
    )

    roi_poly = build_court_roi(frame_w, frame_h)

    # Output setup
    video_name = Path(video_path).stem
    out_path = Path(output_dir)
    frames_dir = out_path / "frames" / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    labels = []         # list of (frame_path, visibility, x, y)
    last_center = None
    frames_since_seen = 999
    detected_count = 0

    print(f"Auto-labeling {min(max_frames, total_frames - start_frame)} "
          f"frames starting at frame {start_frame} ...")

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = start_frame + i

        # Save frame as image
        frame_filename = f"frame_{frame_num:06d}.jpg"
        frame_path = str(frames_dir / frame_filename)
        cv2.imwrite(frame_path, frame)

        # Detect ball
        result = detect_ball_frame(
            model, sahi_model, frame, roi_poly,
            frame_w, frame_h, device, infer_imgsz, use_half,
            last_center, frames_since_seen,
        )

        if result is not None and result[2] >= conf_thresh:
            cx, cy, conf = result
            labels.append((frame_path, 1, round(cx, 1), round(cy, 1)))
            last_center = (cx, cy)
            frames_since_seen = 0
            detected_count += 1
        else:
            labels.append((frame_path, 0, -1, -1))
            frames_since_seen += 1

        if (i + 1) % 200 == 0 or i == 0:
            pct = detected_count / (i + 1) * 100
            print(f"  frame {i+1}/{max_frames}  |  "
                  f"detected: {detected_count}/{i+1} ({pct:.0f}%)")

    cap.release()

    pct = detected_count / max(len(labels), 1) * 100
    print(f"\nDone: {len(labels)} frames, "
          f"{detected_count} with ball ({pct:.0f}% recall)")

    return labels


def write_csvs(all_labels: list, output_dir: str, val_split: float = 0.15):
    """Split labels into train/val CSVs. Keep sequences contiguous."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Split by contiguous chunks (don't shuffle — that breaks sequences)
    n = len(all_labels)
    val_size = int(n * val_split)
    # Take val from a random contiguous block
    val_start = random.randint(0, max(n - val_size, 0))
    val_end = val_start + val_size

    val_labels = all_labels[val_start:val_end]
    train_labels = all_labels[:val_start] + all_labels[val_end:]

    for split_name, split_labels in [("train", train_labels), ("val", val_labels)]:
        csv_path = out_path / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_path", "visibility", "x", "y"])
            for row in split_labels:
                writer.writerow(row)
        detected = sum(1 for r in split_labels if r[1] > 0)
        print(f"  {split_name}: {len(split_labels)} frames, "
              f"{detected} with ball → {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label video frames for TrackNet training")
    parser.add_argument("--video", nargs="+", required=True,
                        help="Path(s) to game video(s)")
    parser.add_argument("--weights", type=str,
                        default="runs/detect/train/weights/best.pt",
                        help="Fine-tuned YOLO weights")
    parser.add_argument("--max-frames", type=int, default=5000,
                        help="Max frames to extract per video")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame to start extraction from")
    parser.add_argument("--conf-thresh", type=float, default=0.15,
                        help="Min confidence to accept a detection as label")
    parser.add_argument("--output-dir", type=str,
                        default="data/tracknet_autolabels",
                        help="Output directory for frames and CSVs")
    args = parser.parse_args()

    all_labels = []
    for video in args.video:
        labels = auto_label_video(
            video, args.output_dir, args.weights,
            max_frames=args.max_frames,
            conf_thresh=args.conf_thresh,
            start_frame=args.start_frame,
        )
        all_labels.extend(labels)

    print(f"\nTotal: {len(all_labels)} frames across {len(args.video)} video(s)")
    write_csvs(all_labels, args.output_dir)
    print("\nDone! Train TrackNet with:")
    print(f"  python main.py --finetune-tracknet "
          f"--tracknet-data {args.output_dir}")


if __name__ == "__main__":
    main()
