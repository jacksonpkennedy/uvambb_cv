"""Generate 5 annotated demo videos showing progressive model improvement.

Outputs go to output/demos/:
  01_base_yolo.mp4          — yolo11s.pt (COCO pretrained, no fine-tuning)
  02_finetuned_yolo.mp4     — fine-tuned YOLO, all 4 classes, raw detections
  03_finetuned_yolo_pipe.mp4 — fine-tuned YOLO + pipeline/gate logic
  04_tracknet_raw.mp4        — fine-tuned YOLO (player/ref/hoop) + TrackNet, raw
  05_full_pipeline.mp4       — fine-tuned YOLO + TrackNet + full pipeline (final)

Usage:
    python generate_demos.py --all
    python generate_demos.py --video data/game_04_TEST_clip.mp4 --mode 4 --motion-diff
    python generate_demos.py --video data/game_04_TEST_clip.mp4 --mode 4 --motion-diff --motion-overlay
"""
import argparse
import time
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tracknet import INPUT_W, INPUT_H

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FINETUNED_WEIGHTS   = "runs/detect/train/weights/best.pt"
TRACKNET_WEIGHTS    = "runs/tracknet/weights/best_session.pt"
DEFAULT_VIDEO       = "data/game_04_TEST_clip.mp4"
OUT_DIR             = "output/demos"
DEFAULT_MOTION_OUT  = "output/demos/motion_diff"

# COCO class IDs used by base yolo11s.pt
COCO_PERSON         = 0
COCO_SPORTS_BALL    = 32

# Colours (BGR)
C_PLAYER  = (0, 255, 0)
C_BALL    = (0, 0, 255)
C_HOOP    = (255, 255, 255)
C_REF     = (180, 60, 20)


# ---------------------------------------------------------------------------
# Motion-diff helpers (TrackNet triplet diffs)
# ---------------------------------------------------------------------------

def _norm_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D float array to uint8 using min/max scaling."""
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return (np.zeros_like(arr) * 0).astype(np.uint8)
    norm = (arr - mn) / (mx - mn)
    return (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)


def _compute_triplet_diffs(f0: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> tuple:
    """Compute TrackNet-style diff01 and diff12 from three BGR uint8 frames.

    Frames are expected as HxWx3 uint8. Returns two float arrays (H,W) in [0,1].
    """
    a0 = f0.astype(np.float32) / 255.0
    a1 = f1.astype(np.float32) / 255.0
    a2 = f2.astype(np.float32) / 255.0
    diff01 = np.abs(a1 - a0).mean(axis=2)
    diff12 = np.abs(a2 - a1).mean(axis=2)
    return diff01, diff12


def _compute_consecutive_diff(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    a0 = prev.astype(np.float32) / 255.0
    a1 = cur.astype(np.float32) / 255.0
    return np.abs(a1 - a0).mean(axis=2)


def _colorize_diff(gray_uint8: np.ndarray) -> np.ndarray:
    """Apply JET colormap to a grayscale uint8 HxW map -> BGR HxWx3 uint8."""
    return cv2.applyColorMap(gray_uint8, cv2.COLORMAP_JET)


def make_motion_diff_for_video(src_video_path: str,
                               out_dir: str,
                               out_name: str,
                               diff_type: str = "triplet",
                               overlay: bool = False,
                               save_npy: bool = False) -> str:
    """Create a side-by-side motion-diff video for `src_video_path`.

    - diff_type: 'triplet' (uses frames i-2,i-1,i and uses diff12) or 'consecutive'
    - overlay: if True, blend colored diff over original frame instead of side-by-side
    - save_npy: when using 'triplet', saves per-triplet 11-channel numpy inputs
    Returns path to written side-by-side video.
    """
    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {src_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    side_name = f"{out_name}_sidebyside.mp4"
    side_path = out_dir_p / side_name
    out_w = frame_w if overlay else frame_w * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(side_path), fourcc, fps, (out_w, frame_h))

    # rolling buffer for frames
    buf = []
    idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buf.append(frame)
        # ensure buffer not unbounded
        if len(buf) > 3:
            buf.pop(0)

        # compute diff if possible
        diff_color_resized = None
        if diff_type == "triplet" and len(buf) >= 3:
            f0, f1, f2 = buf[-3], buf[-2], buf[-1]
            # compute diffs at TrackNet resolution, then upsample to original frame size
            f0_res = cv2.resize(f0, (INPUT_W, INPUT_H))
            f1_res = cv2.resize(f1, (INPUT_W, INPUT_H))
            f2_res = cv2.resize(f2, (INPUT_W, INPUT_H))
            diff01, diff12 = _compute_triplet_diffs(f0_res, f1_res, f2_res)
            g = _norm_to_uint8(diff12)  # use diff12 to represent motion at current frame
            c = _colorize_diff(g)
            diff_color_resized = cv2.resize(c, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            if save_npy:
                inp11 = np.concatenate([
                    np.transpose(f0_res.astype(np.float32) / 255.0, (2, 0, 1)),
                    np.transpose(f1_res.astype(np.float32) / 255.0, (2, 0, 1)),
                    np.transpose(f2_res.astype(np.float32) / 255.0, (2, 0, 1)),
                    diff01[np.newaxis, ...],
                    diff12[np.newaxis, ...],
                ], axis=0)
                npy_name = f"{out_name}_frame{idx:06d}_input11.npy"
                np.save(str(out_dir_p / npy_name), inp11)
        elif diff_type == "consecutive" and len(buf) >= 2:
            prev, cur = buf[-2], buf[-1]
            prev_res = cv2.resize(prev, (INPUT_W, INPUT_H))
            cur_res = cv2.resize(cur, (INPUT_W, INPUT_H))
            diff_map = _compute_consecutive_diff(prev_res, cur_res)
            g = _norm_to_uint8(diff_map)
            c = _colorize_diff(g)
            diff_color_resized = cv2.resize(c, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

        # fallback for early frames: use black diff
        if diff_color_resized is None:
            diff_color_resized = np.zeros_like(frame)

        if overlay:
            out_frame = cv2.addWeighted(frame, 0.6, diff_color_resized, 0.4, 0)
        else:
            out_frame = np.hstack([frame, diff_color_resized])

        writer.write(out_frame)
        written += 1
        idx += 1

    writer.release()
    cap.release()
    print(f"Saved motion-diff side-by-side: {side_path}  (frames: {written})")
    return str(side_path)


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
    parser.add_argument("--all", "-a", action="store_true", help="Run all demo modes (1-5)")
    parser.add_argument("--mode", nargs="+", type=int, choices=[1,2,3,4,5],
                        help="Run specific demo modes, e.g. --mode 4")
    parser.add_argument("--only",  nargs="+", type=int, default=None,
                        help="(legacy) same as --mode")
    # Motion diff flags
    parser.add_argument("--motion-diff", action="store_true",
                        help="Produce TrackNet-style motion-diff side-by-side videos per mode")
    parser.add_argument("--motion-diff-type", choices=["triplet", "consecutive"],
                        default="triplet", help="Diff computation method (triplet recommended)")
    parser.add_argument("--motion-out", default=DEFAULT_MOTION_OUT,
                        help="Output dir for motion-diff side-by-side videos")
    parser.add_argument("--motion-overlay", action="store_true",
                        help="Overlay colored diff onto original frames instead of side-by-side")
    parser.add_argument("--motion-save-npy", action="store_true",
                        help="Save per-triplet 11-channel numpy inputs when using triplet diffs")
    parser.add_argument("--only-modes", nargs="+", type=int,
                        help=argparse.SUPPRESS)  # kept for backward compat in some workflows
    args = parser.parse_args()

    video   = args.video
    out_dir = args.out

    # Determine modes to run
    if args.all:
        modes_to_run = {1,2,3,4,5}
    elif args.mode:
        modes_to_run = set(args.mode)
    elif args.only:
        modes_to_run = set(args.only)
    else:
        # default: all modes as before
        modes_to_run = {1,2,3,4,5}

    # Conditional weight checks only for modes that require them
    if any(m in modes_to_run for m in (2,3,4,5)):
        if not Path(FINETUNED_WEIGHTS).exists():
            raise FileNotFoundError(
                f"Fine-tuned YOLO weights not found: {FINETUNED_WEIGHTS}\n"
                "Run: python main.py --finetune")
    if any(m in modes_to_run for m in (4,5)):
        if not Path(TRACKNET_WEIGHTS).exists():
            raise FileNotFoundError(
                f"TrackNet weights not found: {TRACKNET_WEIGHTS}\n"
                "Run: python tracknet.py --train ...")

    t_total = time.perf_counter()

    # Helper to optionally run motion-diff generation for a produced demo
    def maybe_make_motion(name: str):
        if args.motion_diff:
            try:
                make_motion_diff_for_video(
                    src_video_path = video,
                    out_dir        = args.motion_out,
                    out_name       = name,
                    diff_type      = args.motion_diff_type,
                    overlay        = args.motion_overlay,
                    save_npy       = args.motion_save_npy,
                )
            except Exception as e:
                print(f"[warn] motion-diff generation failed for {name}: {e}")

    if 1 in modes_to_run:
        run_base_yolo(video, out_dir, "01_base_yolo")
        maybe_make_motion("01_base_yolo")

    if 2 in modes_to_run:
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
        maybe_make_motion("02_finetuned_yolo")

    if 3 in modes_to_run:
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
        maybe_make_motion("03_finetuned_yolo_pipe")

    if 4 in modes_to_run:
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
        maybe_make_motion("04_tracknet_raw")

    if 5 in modes_to_run:
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
        maybe_make_motion("05_full_pipeline")

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