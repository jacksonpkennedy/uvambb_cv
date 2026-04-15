"""Generate 5 annotated demo videos showing progressive model improvement.

Outputs go to output/demos/:
  01_base_yolo.mp4          — yolo11s.pt (COCO pretrained, no fine-tuning)
  02_finetuned_yolo.mp4     — fine-tuned YOLO, all 4 classes, raw detections
  03_finetuned_yolo_pipe.mp4 — fine-tuned YOLO + pipeline/gate logic
  04_tracknet_raw.mp4        — fine-tuned YOLO (player/ref/hoop) + TrackNet, raw
  05_full_pipeline.mp4       — fine-tuned YOLO + TrackNet + full pipeline (final)

Usage:
    python generate_demos.py
    python generate_demos.py --video data/game_04_TEST_clip.mp4
    python generate_demos.py --video data/game_04_TEST_clip.mp4 --only 4 5
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FINETUNED_WEIGHTS   = "runs/detect/train/weights/best.pt"
TRACKNET_WEIGHTS    = "runs/tracknet/weights/best_session.pt"
DEFAULT_VIDEO       = "data/game_04_TEST_clip.mp4"
OUT_DIR             = "output/demos"

# COCO class IDs used by base yolo11s.pt
COCO_PERSON         = 0
COCO_SPORTS_BALL    = 32

# Colours (BGR)
C_PLAYER  = (0, 255, 0)
C_BALL    = (0, 0, 255)
C_HOOP    = (255, 255, 255)
C_REF     = (180, 60, 20)


# ---------------------------------------------------------------------------
# Mode 1 — Base YOLO (COCO pretrained, no fine-tuning)
# ---------------------------------------------------------------------------

def run_base_yolo(video_path: str, out_dir: str, out_name: str) -> None:
    """Render yolo11s.pt detections raw — shows zero basketball-specific knowledge."""
    print(f"\n{'='*60}")
    print(f"Mode 1: Base YOLO (yolo11s.pt — COCO pretrained)")
    print(f"{'='*60}")

    model  = YOLO("yolo11s.pt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    imgsz  = 1280 if device.startswith("cuda") else 640
    half   = device.startswith("cuda")

    cap = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        str(Path(out_dir) / f"{out_name}.mp4"), fourcc, fps, (frame_w, frame_h))

    frame_idx = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.predict(
            frame, verbose=False, device=device,
            conf=0.25, imgsz=imgsz, half=half,
        )[0]

        if res.boxes is not None:
            names = model.names
            for box, cls_id, conf in zip(
                    res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                cls_id = int(cls_id)
                conf   = float(conf)
                x1, y1, x2, y2 = map(int, box.tolist())
                label  = names[cls_id]

                if cls_id == COCO_SPORTS_BALL:
                    color = C_BALL
                    label = f"ball? ({label} {conf:.2f})"
                elif cls_id == COCO_PERSON:
                    color = C_PLAYER
                    label = f"person ({conf:.2f})"
                else:
                    color = (120, 120, 120)
                    label = f"{label} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out_video.write(frame)
        frame_idx += 1
        if frame_idx % 60 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  frame {frame_idx}/{total}  ({elapsed:.0f}s elapsed)")

    cap.release()
    out_video.release()
    print(f"  → {Path(out_dir) / out_name}.mp4")


# ---------------------------------------------------------------------------
# Modes 2–5 — delegate to main.run()
# ---------------------------------------------------------------------------

def run_mode(label: str, video: str, out_dir: str, out_name: str,
             weights: str, use_tracknet: bool, tracknet_weights: str | None,
             no_pipeline: bool, use_yolo_ball: bool) -> None:
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    # Import here so any import-time side effects happen after print
    from main import run
    run(
        video_path       = video,
        out_dir          = out_dir,
        weights          = weights,
        use_tracknet     = use_tracknet,
        tracknet_weights = tracknet_weights,
        no_pipeline      = no_pipeline,
        use_yolo_ball    = use_yolo_ball,
        out_name         = out_name,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 5 demo comparison videos")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--out",   default=OUT_DIR)
    parser.add_argument("--only",  nargs="+", type=int, default=None,
                        help="Run only specific modes, e.g. --only 4 5")
    args = parser.parse_args()

    video   = args.video
    out_dir = args.out
    only    = set(args.only) if args.only else {1, 2, 3, 4, 5}

    if not Path(video).exists():
        raise FileNotFoundError(f"Video not found: {video}")
    if not Path(FINETUNED_WEIGHTS).exists():
        raise FileNotFoundError(
            f"Fine-tuned YOLO weights not found: {FINETUNED_WEIGHTS}\n"
            "Run: python main.py --finetune")
    if not Path(TRACKNET_WEIGHTS).exists():
        raise FileNotFoundError(
            f"TrackNet weights not found: {TRACKNET_WEIGHTS}\n"
            "Run: python tracknet.py --train ...")

    t_total = time.perf_counter()

    if 1 in only:
        run_base_yolo(video, out_dir, "01_base_yolo")

    if 2 in only:
        run_mode(
            label            = "Mode 2: Fine-tuned YOLO — all 4 classes, raw detections",
            video            = video,
            out_dir          = out_dir,
            out_name         = "02_finetuned_yolo",
            weights          = FINETUNED_WEIGHTS,
            use_tracknet     = False,
            tracknet_weights = None,
            no_pipeline      = True,
            use_yolo_ball    = True,
        )

    if 3 in only:
        run_mode(
            label            = "Mode 3: Fine-tuned YOLO — all 4 classes + pipeline/gate logic",
            video            = video,
            out_dir          = out_dir,
            out_name         = "03_finetuned_yolo_pipe",
            weights          = FINETUNED_WEIGHTS,
            use_tracknet     = False,
            tracknet_weights = None,
            no_pipeline      = False,
            use_yolo_ball    = True,
        )

    if 4 in only:
        run_mode(
            label            = "Mode 4: Fine-tuned YOLO (player/ref/hoop) + TrackNet — raw",
            video            = video,
            out_dir          = out_dir,
            out_name         = "04_tracknet_raw",
            weights          = FINETUNED_WEIGHTS,
            use_tracknet     = True,
            tracknet_weights = TRACKNET_WEIGHTS,
            no_pipeline      = True,
            use_yolo_ball    = False,
        )

    if 5 in only:
        run_mode(
            label            = "Mode 5: Full pipeline — fine-tuned YOLO + TrackNet + all gates",
            video            = video,
            out_dir          = out_dir,
            out_name         = "05_full_pipeline",
            weights          = FINETUNED_WEIGHTS,
            use_tracknet     = True,
            tracknet_weights = TRACKNET_WEIGHTS,
            no_pipeline      = False,
            use_yolo_ball    = False,
        )

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"All requested modes complete in {elapsed/60:.1f} min")
    print(f"Output directory: {Path(out_dir).resolve()}")
    for name in ["01_base_yolo", "02_finetuned_yolo", "03_finetuned_yolo_pipe",
                 "04_tracknet_raw", "05_full_pipeline"]:
        p = Path(out_dir) / f"{name}.mp4"
        if p.exists():
            print(f"  ✓ {name}.mp4")


if __name__ == "__main__":
    main()
