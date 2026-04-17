"""
Auto-label consecutive video frames for TrackNet training using YOLO + SAHI.

Uses the SAME filtering pipeline as main.py (BallValidator + BallTracker)
to ensure labels match the quality of the main inference output.

Usage:
    python auto_label_tracknet.py --video data/game_01.mp4 [--max-frames 5000]
    python auto_label_tracknet.py --video data/game_01.mp4 data/game_02.mp4
    python auto_label_tracknet.py --relabel  # re-run on already-extracted frames
"""

import argparse
import csv
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

# Reuse validator/tracker classes and ball constants from the main pipeline
from main import (
    CLS_BALL, CLS_HOOP,
    BALL_MAX_BBOX_AREA, BALL_MIN_BBOX_AREA,
    BALL_MAX_TELEPORT_PX,
    BallValidator, BallTracker,
    build_court_roi,
)

# SAHI constants (inlined here — SAHI was removed from main.py inference pipeline
# but is still useful for auto-labeling since it improves label recall)
SAHI_CROP_PAD      = 400    # pixels to pad around last known ball center for SAHI crop
SAHI_CONF_THRESH   = 0.10   # lower conf OK — we're zooming into a small region
SAHI_SLICE_SIZE    = 128    # smaller slices = more zoom on small ball
SAHI_OVERLAP_RATIO = 0.35
SAHI_MAX_LOST      = 60     # max frames since last ball sighting to run targeted SAHI
COCO_BALL_CLS      = 32     # "sports ball" in COCO
COCO_BALL_CONF     = 0.20

# Minimum confidence for fine-tuned YOLO ball detections
YOLO_BALL_CONF = 0.25


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _run_sahi_ball_targeted(sahi_model, frame, center, existing_ball_dets):
    """Run SAHI on a small crop around the ball's last known position."""
    cx, cy = center
    h, w = frame.shape[:2]

    x1 = max(0, int(cx - SAHI_CROP_PAD))
    y1 = max(0, int(cy - SAHI_CROP_PAD))
    x2 = min(w, int(cx + SAHI_CROP_PAD))
    y2 = min(h, int(cy + SAHI_CROP_PAD))

    if x2 - x1 < 50 or y2 - y1 < 50:
        return []

    crop = frame[y1:y2, x1:x2]
    result = get_sliced_prediction(
        image=crop,
        detection_model=sahi_model,
        slice_height=SAHI_SLICE_SIZE,
        slice_width=SAHI_SLICE_SIZE,
        overlap_height_ratio=SAHI_OVERLAP_RATIO,
        overlap_width_ratio=SAHI_OVERLAP_RATIO,
        postprocess_type="NMS",
        postprocess_match_threshold=0.40,
        verbose=0,
    )

    sahi_balls = []
    for pred in result.object_prediction_list:
        if int(pred.category.id) != COCO_BALL_CLS:
            continue
        if pred.score.value < SAHI_CONF_THRESH:
            continue
        bb = pred.bbox
        box = [int(bb.minx) + x1, int(bb.miny) + y1,
               int(bb.maxx) + x1, int(bb.maxy) + y1]
        sahi_balls.append({
            "tid": -1, "box": box,
            "cls": CLS_BALL, "conf": pred.score.value,
        })
    return sahi_balls


def detect_ball_frame(model, sahi_model, frame, frame_idx,
                      frame_w, frame_h, device, infer_imgsz, use_half,
                      ball_validator, ball_tracker):
    """Run YOLO + SAHI on a single frame with full validation pipeline.

    Returns (cx, cy, conf) or None.
    """
    ball_dets = []
    hoop_dets = []

    # --- YOLO detection (no tracking, single-frame predict) ---
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
            box_list = list(map(int, box.tolist()))

            if cls_id == CLS_BALL and conf_val >= YOLO_BALL_CONF:
                w = box_list[2] - box_list[0]
                h = box_list[3] - box_list[1]
                area = w * h
                if area < BALL_MIN_BBOX_AREA or area > BALL_MAX_BBOX_AREA:
                    continue
                if max(w, h) / max(min(w, h), 1) > 1.8:
                    continue
                ball_dets.append({
                    "tid": -1, "box": box_list,
                    "cls": CLS_BALL, "conf": conf_val,
                })
            elif cls_id == CLS_HOOP and conf_val >= 0.30:
                hoop_dets.append({
                    "tid": -1, "box": box_list, "cls": CLS_HOOP,
                })

    # --- SAHI fallback (only when YOLO misses) ---
    if not ball_dets and sahi_model is not None:
        center = ball_tracker.last_center
        frames_since = ball_tracker.frames_since_seen

        if center is not None and frames_since < SAHI_MAX_LOST:
            sahi_balls = _run_sahi_ball_targeted(
                sahi_model, frame, center, ball_dets)
            ball_dets.extend(sahi_balls)
        elif center is None or frames_since >= SAHI_MAX_LOST:
            # Full-frame SAHI every 5 frames when ball is lost
            if frame_idx % 5 == 0:
                result = get_sliced_prediction(
                    image=frame,
                    detection_model=sahi_model,
                    slice_height=320, slice_width=320,
                    overlap_height_ratio=0.25,
                    overlap_width_ratio=0.25,
                    postprocess_type="NMS",
                    postprocess_match_threshold=0.40,
                    verbose=0,
                )
                for pred in result.object_prediction_list:
                    if int(pred.category.id) != COCO_BALL_CLS:
                        continue
                    if pred.score.value < COCO_BALL_CONF:
                        continue
                    bb = pred.bbox
                    box = [int(bb.minx), int(bb.miny),
                           int(bb.maxx), int(bb.maxy)]
                    ball_dets.append({
                        "tid": -1, "box": box,
                        "cls": CLS_BALL, "conf": pred.score.value,
                    })

    # --- Apply the SAME filters as the main pipeline ---
    ball_dets = ball_validator.filter(frame_idx, ball_dets, hoop_dets)
    ball_to_draw = ball_tracker.update(frame_idx, ball_dets)

    if not ball_to_draw:
        return None

    best = ball_to_draw[0]
    cx = (best["box"][0] + best["box"][2]) / 2.0
    cy = (best["box"][1] + best["box"][3]) / 2.0
    return (cx, cy, best.get("conf", 0.5))


@torch.inference_mode()
def auto_label_video(video_path: str, output_dir: str,
                     weights: str, max_frames: int = 5000,
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
    ball_validator = BallValidator(roi_poly, frame_w, frame_h)
    ball_tracker = BallTracker()

    video_name = Path(video_path).stem
    out_path = Path(output_dir)
    frames_dir = out_path / "frames" / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    labels = []
    detected_count = 0

    print(f"Auto-labeling {min(max_frames, total_frames - start_frame)} "
          f"frames starting at frame {start_frame} ...")

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = start_frame + i

        frame_filename = f"frame_{frame_num:06d}.jpg"
        frame_path = str(frames_dir / frame_filename)
        cv2.imwrite(frame_path, frame)

        result = detect_ball_frame(
            model, sahi_model, frame, frame_num,
            frame_w, frame_h, device, infer_imgsz, use_half,
            ball_validator, ball_tracker,
        )

        if result is not None:
            cx, cy, conf = result
            labels.append((frame_path, 1, round(cx, 1), round(cy, 1)))
            detected_count += 1
        else:
            labels.append((frame_path, 0, -1, -1))

        if (i + 1) % 200 == 0 or i == 0:
            pct = detected_count / (i + 1) * 100
            print(f"  frame {i+1}/{max_frames}  |  "
                  f"detected: {detected_count}/{i+1} ({pct:.0f}%)")

    cap.release()

    pct = detected_count / max(len(labels), 1) * 100
    print(f"\nDone: {len(labels)} frames, "
          f"{detected_count} with ball ({pct:.0f}% recall)")

    return labels


@torch.inference_mode()
def relabel_existing_frames(frames_dir: str, output_dir: str,
                            weights: str):
    """Re-run ball detection on already-extracted frame images."""
    frames_path = Path(frames_dir)
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    game_dirs = sorted([d for d in frames_path.iterdir() if d.is_dir()])
    if not game_dirs:
        raise FileNotFoundError(f"No game subdirectories found in {frames_dir}")

    device = get_device()
    infer_imgsz = 1280 if device.startswith("cuda") else 640
    use_half = device.startswith("cuda")

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

    all_labels = []

    for game_dir in game_dirs:
        frame_files = sorted(game_dir.glob("frame_*.jpg"))
        if not frame_files:
            print(f"Skipping {game_dir.name}: no frames found")
            continue

        print(f"\n{game_dir.name}: {len(frame_files)} frames")

        sample = cv2.imread(str(frame_files[0]))
        frame_h, frame_w = sample.shape[:2]
        roi_poly = build_court_roi(frame_w, frame_h)
        ball_validator = BallValidator(roi_poly, frame_w, frame_h)
        ball_tracker = BallTracker()

        detected_count = 0

        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                all_labels.append((str(frame_file), 0, -1, -1))
                ball_tracker.update(i, [])
                continue

            result = detect_ball_frame(
                model, sahi_model, frame, i,
                frame_w, frame_h, device, infer_imgsz, use_half,
                ball_validator, ball_tracker,
            )

            if result is not None:
                cx, cy, conf = result
                all_labels.append((str(frame_file), 1, round(cx, 1), round(cy, 1)))
                detected_count += 1
            else:
                all_labels.append((str(frame_file), 0, -1, -1))

            if (i + 1) % 200 == 0 or i == 0:
                pct = detected_count / (i + 1) * 100
                print(f"  frame {i+1}/{len(frame_files)}  |  "
                      f"detected: {detected_count}/{i+1} ({pct:.0f}%)")

        pct = detected_count / max(len(frame_files), 1) * 100
        print(f"  {game_dir.name}: {detected_count}/{len(frame_files)} "
              f"with ball ({pct:.0f}%)")

    return all_labels


def write_csvs(all_labels: list, output_dir: str, val_split: float = 0.15):
    """Split labels into train/val CSVs. Keep sequences contiguous."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n = len(all_labels)
    val_size = int(n * val_split)
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
              f"{detected} with ball -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label video frames for TrackNet training")
    parser.add_argument("--video", nargs="*", default=None,
                        help="Path(s) to game video(s)")
    parser.add_argument("--relabel", action="store_true",
                        help="Re-run detection on existing extracted frames")
    parser.add_argument("--weights", type=str,
                        default="runs/detect/train/weights/best.pt",
                        help="Fine-tuned YOLO weights")
    parser.add_argument("--max-frames", type=int, default=5000,
                        help="Max frames to extract per video")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame to start extraction from")
    parser.add_argument("--output-dir", type=str,
                        default="data/tracknet_autolabels",
                        help="Output directory for frames and CSVs")
    args = parser.parse_args()

    if args.relabel:
        frames_dir = str(Path(args.output_dir) / "frames")
        all_labels = relabel_existing_frames(
            frames_dir, args.output_dir, args.weights,
        )
    elif args.video:
        all_labels = []
        for video in args.video:
            labels = auto_label_video(
                video, args.output_dir, args.weights,
                max_frames=args.max_frames,
                start_frame=args.start_frame,
            )
            all_labels.extend(labels)
    else:
        parser.error("Provide --video or --relabel")
        return

    print(f"\nTotal: {len(all_labels)} frames")
    write_csvs(all_labels, args.output_dir)
    print("\nDone! Train TrackNet with:")
    print(f"  python main.py --finetune-tracknet "
          f"--tracknet-data {args.output_dir}")


if __name__ == "__main__":
    main()
