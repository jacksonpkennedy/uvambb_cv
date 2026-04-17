"""Find disagreements between TrackNet predictions and training labels.

Compares the trained model's predictions against the CSV labels to find:
  1. FALSE NEGATIVES: label says no ball, but model detects one (missed labels)
  2. FALSE POSITIVES: label says ball, but model doesn't detect one (bad labels)
  3. POSITION MISMATCH: both say ball, but positions differ >15px (wrong position)

Usage:
    python find_disagreements.py --data data/tracknet_merged --weights runs/tracknet/weights/best_session.pt
    python find_disagreements.py --data data/tracknet_merged --weights runs/tracknet/weights/best_session.pt --visualize
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def _count_csv(path: str) -> int:
    """Count data rows in a CSV (excluding header). Returns 0 if file missing."""
    p = Path(path)
    if not p.exists():
        return 0
    with open(p) as f:
        return max(sum(1 for _ in f) - 1, 0)

from tracknet import (
    TrackNetV3, INPUT_W, INPUT_H, postprocess_heatmap,
)


CONF_THRESH = 0.5       # model confidence threshold
DIST_THRESH_PX = 15.0   # max distance before flagging position mismatch


def load_model(weights_path: str, device: str) -> TrackNetV3:
    model = TrackNetV3()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_frame(path: str) -> tuple:
    """Load and resize a frame to model resolution. Returns (tensor, orig_w, orig_h)."""
    p = Path(path)
    # Try .npy cache first
    cache_npy = p.parent / f"{p.stem}_640x360.npy"
    cache_dims = p.parent / f"{p.stem}_640x360.txt"

    if cache_npy.exists() and cache_dims.exists():
        img = np.load(str(cache_npy))
        orig_w, orig_h = map(int, cache_dims.read_text().split())
        return np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0, orig_w, orig_h

    img = cv2.imread(path)
    if img is None:
        return None, 0, 0
    orig_h, orig_w = img.shape[:2]
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    return np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0, orig_w, orig_h


def predict_triplet(model, f0, f1, f2, device):
    """Run model on a 3-frame triplet. Returns (cx, cy, conf) at model res or None."""
    diff01 = np.abs(f1 - f0).mean(axis=0, keepdims=True)
    diff12 = np.abs(f2 - f1).mean(axis=0, keepdims=True)
    inp = np.concatenate([f0, f1, f2, diff01, diff12], axis=0)  # (11, H, W)
    tensor = torch.from_numpy(inp).unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device != "cpu"):
        hmap_logits, _ = model(tensor)
        out = torch.sigmoid(hmap_logits)
    heatmap = out[0].cpu().numpy()  # (1, H, W)

    candidates = postprocess_heatmap(heatmap)
    if not candidates:
        return None
    cx, cy, conf = candidates[0]
    if conf < CONF_THRESH:
        return None
    return cx, cy, conf


def parse_frame_number(path: str) -> int:
    stem = Path(path).stem
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return -1


@torch.inference_mode()
def find_disagreements(data_dir: str, weights: str, visualize: bool = False,
                       max_vis: int = 200, data_root: str | None = None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model from {weights} ...")
    model = load_model(weights, device)

    out_dir = Path("output/disagreements")
    if visualize:
        import shutil
        for sub in ("false_negatives", "false_positives", "position_mismatch"):
            sub_dir = out_dir / sub
            if sub_dir.exists():
                for child in sub_dir.iterdir():
                    if child.is_dir() and child.name != "_neither":
                        try:
                            shutil.rmtree(child)
                        except PermissionError:
                            print(f"  [warn] Could not delete {child} — close any open Explorer/viewer windows and retry")
            sub_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        csv_path = Path(data_dir) / f"{split}.csv"
        if not csv_path.exists():
            print(f"Skipping {split}: {csv_path} not found")
            continue

        entries = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                fp = row["frame_path"]
                if data_root is not None:
                    fp = fp.replace("\\", "/")
                    if fp.startswith("data/"):
                        fp = str(Path(data_root) / fp[len("data/"):])
                entries.append((
                    fp,
                    int(row["visibility"]),
                    float(row["x"]) if row["visibility"] != "0" else -1.0,
                    float(row["y"]) if row["visibility"] != "0" else -1.0,
                ))

        false_neg = []   # label=no ball, model=ball (missed labels)
        false_pos = []   # label=ball, model=no ball (bad labels)
        pos_mismatch = []  # both say ball, positions differ

        vis_counts = {"false_negatives": 0, "false_positives": 0, "position_mismatch": 0}
        checked = 0

        for i in range(2, len(entries)):
            i0, i1, i2 = i - 2, i - 1, i

            # Check game boundaries
            p0 = Path(entries[i0][0])
            p1 = Path(entries[i1][0])
            p2 = Path(entries[i2][0])
            if p0.parent != p1.parent or p1.parent != p2.parent:
                continue

            n0 = parse_frame_number(str(p0))
            n1 = parse_frame_number(str(p1))
            n2 = parse_frame_number(str(p2))
            if n0 < 0 or n1 != n0 + 1 or n2 != n1 + 1:
                continue

            # Load frames
            f0, _, _ = load_frame(entries[i0][0])
            f1, _, _ = load_frame(entries[i1][0])
            f2, orig_w, orig_h = load_frame(entries[i2][0])

            if f0 is None or f1 is None or f2 is None:
                continue

            # Get label for frame t (the prediction target)
            _, vis, lx, ly = entries[i2]
            # Scale label to model resolution
            if vis > 0 and lx >= 0 and ly >= 0:
                label_cx = lx * INPUT_W / orig_w
                label_cy = ly * INPUT_H / orig_h
                has_label = True
            else:
                label_cx = label_cy = -1
                has_label = False

            # Get model prediction at model resolution
            pred = predict_triplet(model, f0, f1, f2, device)
            has_pred = pred is not None

            category = None
            if has_pred and not has_label:
                false_neg.append((entries[i2][0], pred[0], pred[1], pred[2]))
                category = "false_negatives"
            elif has_label and not has_pred:
                false_pos.append((entries[i2][0], label_cx, label_cy))
                category = "false_positives"
            elif has_label and has_pred:
                dist = math.hypot(pred[0] - label_cx, pred[1] - label_cy)
                if dist > DIST_THRESH_PX:
                    pos_mismatch.append((entries[i2][0], label_cx, label_cy,
                                        pred[0], pred[1], dist))
                    category = "position_mismatch"

            # Save visualization
            if visualize and category and vis_counts[category] < max_vis:
                img = cv2.imread(entries[i2][0])
                if img is not None:
                    img = cv2.resize(img, (INPUT_W, INPUT_H))
                    if category == "false_negatives":
                        # Green circle = model prediction (no label exists)
                        cv2.circle(img, (int(pred[0]), int(pred[1])), 8,
                                   (0, 255, 0), 2)
                        cv2.putText(img, f"MODEL ({pred[2]:.2f})",
                                    (int(pred[0]) + 10, int(pred[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 255, 0), 1)
                    elif category == "false_positives":
                        # Red circle = label position (model doesn't see ball)
                        cv2.circle(img, (int(label_cx), int(label_cy)), 8,
                                   (0, 0, 255), 2)
                        cv2.putText(img, "LABEL (no model det)",
                                    (int(label_cx) + 10, int(label_cy)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 0, 255), 1)
                    elif category == "position_mismatch":
                        # Red = label, Green = model
                        cv2.circle(img, (int(label_cx), int(label_cy)), 8,
                                   (0, 0, 255), 2)
                        cv2.circle(img, (int(pred[0]), int(pred[1])), 8,
                                   (0, 255, 0), 2)
                        cv2.line(img, (int(label_cx), int(label_cy)),
                                 (int(pred[0]), int(pred[1])), (255, 255, 0), 1)
                        cv2.putText(img, f"dist={dist:.0f}px",
                                    (int(pred[0]) + 10, int(pred[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 0), 1)

                    src_p = Path(entries[i2][0])
                    game = src_p.parent.name
                    game_dir = out_dir / category / game
                    game_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(game_dir / f"{src_p.stem}.jpg"), img)
                    vis_counts[category] += 1

            checked += 1
            if checked % 2000 == 0:
                print(f"  [{split}] {checked} sequences checked ...")

        print(f"\n=== {split.upper()} ({checked} sequences) ===")
        print(f"  False negatives (missed labels):  {len(false_neg)}")
        print(f"  False positives (bad labels):     {len(false_pos)}")
        print(f"  Position mismatches (>{DIST_THRESH_PX:.0f}px): {len(pos_mismatch)}")

        # Write disagreement CSVs for review
        out_csv_dir = Path("output/disagreements")
        out_csv_dir.mkdir(parents=True, exist_ok=True)

        with open(out_csv_dir / f"{split}_false_negatives.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_path", "pred_x", "pred_y", "pred_conf"])
            for row in false_neg:
                w.writerow(row)

        with open(out_csv_dir / f"{split}_false_positives.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_path", "label_x", "label_y"])
            for row in false_pos:
                w.writerow(row)

        with open(out_csv_dir / f"{split}_position_mismatch.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_path", "label_x", "label_y",
                         "pred_x", "pred_y", "distance_px"])
            for row in pos_mismatch:
                w.writerow(row)

    if visualize:
        print(f"\nSaved visualizations to {out_dir}/")
    print(f"Saved disagreement CSVs to output/disagreements/")

    # Log audit results to W&B for tracking label quality over iterations
    if _HAS_WANDB:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "uvambb-cv"),
            entity=os.environ.get("WANDB_ENTITY"),
            job_type="label-audit",
            name=f"audit-{Path(weights).stem}",
        )
        wandb.log({
            "audit/false_negatives": sum(
                _count_csv(f"output/disagreements/{s}_false_negatives.csv")
                for s in ("train", "val")),
            "audit/false_positives": sum(
                _count_csv(f"output/disagreements/{s}_false_positives.csv")
                for s in ("train", "val")),
            "audit/position_mismatches": sum(
                _count_csv(f"output/disagreements/{s}_position_mismatch.csv")
                for s in ("train", "val")),
        })
        art = wandb.Artifact("disagreements", type="label-audit")
        for csv_file in Path("output/disagreements").glob("*.csv"):
            art.add_file(str(csv_file))
        wandb.log_artifact(art)
        wandb.finish()

    print("\nNext steps:")
    print("  1. python interactive_label.py   (review + fix in one session)")
    print("  2. python merge_labels.py")
    print("  3. python tracknet.py --train --data data/tracknet_merged --epochs 40 --batch 8")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Find disagreements between TrackNet and labels")
    parser.add_argument("--data", default="data/tracknet_merged",
                        help="Directory with train.csv / val.csv")
    parser.add_argument("--weights", default="runs/tracknet/weights/best_session.pt",
                        help="Trained TrackNet weights (use best_session.pt for cleaning)")
    parser.add_argument("--data-root",
                        default=os.environ.get("UVAMBB_DATA_ROOT"),
                        help="Root directory for frame data (or set UVAMBB_DATA_ROOT in .env)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated images of disagreements")
    parser.add_argument("--max-vis", type=int, default=2000,
                        help="Max visualizations per category")
    args = parser.parse_args()

    find_disagreements(args.data, args.weights,
                       visualize=args.visualize, max_vis=args.max_vis,
                       data_root=args.data_root)
    
'''
==============================================================================
ITERATION WORKFLOW
==============================================================================

0) BEFORE YOU START — prerequisites
   - A trained TrackNet checkpoint must exist. Use best_session.pt (current
     iteration's best), NOT best_overall.pt — the all-time best may be from
     an older model with different failure modes.
   - merge_labels.py must have been run so data/tracknet_merged/train.csv
     and val.csv exist.

1) OPTIONAL: add more frames and re-merge
   python auto_label_tracknet.py --video data/game_01.mp4 \
       --output-dir data/tracknet_autolabels_01b \
       --start-frame 5000 --max-frames 10000
   python merge_labels.py

   Note: adding raw auto-labeled frames temporarily drops F1 because they
   carry noise. Steps 2-3 win it back.

2) RUN AUDIT
   python find_disagreements.py --visualize --max-vis 2000

   Flags:
     --data PATH           (default data/tracknet_merged)
     --weights PATH        (default runs/tracknet/weights/best_session.pt)
     --visualize           save annotated JPGs into output/disagreements/
     --max-vis N           per-category visualization cap (default 2000)

   Internal thresholds (edit as file constants if needed):
     CONF_THRESH    = 0.5   model confidence to count as a detection
     DIST_THRESH_PX = 15    distance before flagging a position mismatch

   Outputs:
     output/disagreements/
       {train,val}_false_negatives.csv     label=none, model=ball
       {train,val}_false_positives.csv     label=ball, model=none
       {train,val}_position_mismatch.csv   both see ball, positions differ
       false_negatives/{game}/{stem}.jpg   annotated at model resolution
       false_positives/{game}/{stem}.jpg
       position_mismatch/{game}/{stem}.jpg  red=label, green=model

   Narrowing the review set: raise CONF_THRESH (e.g. 0.7) to see only
   high-confidence FN/PM, or raise DIST_THRESH_PX (e.g. 25) to filter
   out near-miss PM.

3) INTERACTIVE REVIEW
   python interactive_label.py

   Opens an OpenCV window — loads all disagreement CSVs, shows one frame at
   a time. Two-stage per frame: Review then Preview (see result before saving).
   Writes directly to fix_log.csv — no sync/apply step needed.

   Controls (Review stage):
     Click         Set custom ball position (cyan dot)
     Right-click   Clear custom position
     A             Accept → add/reposition at clicked or model pos
     R             Remove/zero → bad label or neither-correct
     K             Keep → no change, skip
     N / →         Next frame (skip without action)
     P / ←         Previous frame
     U             Undo last fix (only unqueued entries)
     Q / ESC       Save and quit

   Controls (Preview stage):
     Y / Enter     Confirm and log the fix
     ESC / N       Cancel, back to review

   Category semantics:
     false_negatives  A=add label (there is a ball) R or K=skip (no ball)p
     false_positives  A/R=remove bad label  K=keep(real ball)
     position_mismatch  A=reposition  R=zero(neither)  K=keep label

   Filter to one category:
     python interactive_label.py --category false_negatives
     python interactive_label.py --category false_positives
     python interactive_label.py --category position_mismatch

   Resume is automatic (progress saved to output/disagreements/.review_progress.json).
   Force restart: python interactive_label.py --reset-progress

4) MERGE AND RETRAIN
   python merge_labels.py
   python tracknet.py --train --data data/tracknet_merged --epochs 40 --batch 8

   The .npy preprocessing cache does NOT need regeneration — it stores
   resized frames, not labels. Label changes take effect next epoch.

==============================================================================
ROLLBACK / RECOVERY
==============================================================================

merge_labels.py replays fix_log.csv after each merge, so any interactive_label
actions are preserved across re-merges.

List backups:
   ls data/tracknet_merged/backups/

Restore a specific backup:
   cp data/tracknet_merged/backups/<YYYYMMDD_HHMMSS_tag>/*.csv \
      data/tracknet_merged/

==============================================================================
CAVEATS AND FAQ
==============================================================================

- CROSS-GAME STEM COLLISIONS: solved by {game}/{stem}.jpg layout. Every
  lookup keys on (game, stem) tuple.

- WHICH WEIGHTS FOR AUDIT: best_session.pt, not best_overall.pt. The
  all-time best may be from an older iteration with different failure
  modes that don't match current data.

- VAL SET LABELS MATTER. Don't skip val rows — a wrong val label poisons
  F1 every epoch. merge_labels.py splits deterministically (last 15% per
  game), so a row stays in val across re-merges unless source order
  changes. Clean val as aggressively as train.

- FALSE NEGATIVE COUNT LOOKS HUGE? Expected on freshly-merged auto-labeled
  data. The auto-labeler leaves visibility=0 on anything it wasn't sure
  about; the model then sees balls in those frames and flags them as FN.
  Review normally — most are genuine misses.

- AUDIT IS SLOW? --max-vis caps image writes, not CSV rows. The bottleneck
  is running the model on every 3-frame triplet (~10 min/10k frames on GPU).

- DO I RERUN PREPROCESS AFTER CLEANING? No. Preprocess caches resized frame
  images; label edits don't touch those files. Rerun only when adding NEW
  frames or changing INPUT_W/H.

- WHEN TO STOP ITERATING? F1 plateaus for 2-3 iterations in a row, or the
  audit finds <1% disagreements. Label noise is then below the model's own
  noise floor.
'''
