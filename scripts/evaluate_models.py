"""Generate evaluation statistics for the presentation.

Outputs:
  - YOLO: per-class F1 at optimal confidence threshold (Roboflow val set)
  - YOLO on TrackNet val: apples-to-apples comparison on same game footage
    - TrackNet: F1/precision/recall swept across heatmap thresholds (PE ≤ 10px)

Usage:
    python evaluate_models.py
"""
import csv
import math
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

YOLO_WEIGHTS     = "runs/detect/train/weights/best.pt"
YOLO_DATA        = "data/custom_annotations/data.yaml"
TRACKNET_WEIGHTS = "runs/tracknet/weights/best_overall.pt"
TRACKNET_DATA    = "data/tracknet_merged"
PE_THRESH        = 10.0     # pixels at model resolution (reporting threshold)
INPUT_W, INPUT_H = 640, 360 # TrackNet model resolution
YOLO_BALL_CLS    = 0        # 'basketball' class index in fine-tuned YOLO

# Import architectural constant so summary can reference it
import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from tracknet import VIS_THRESH_INFER


# ---------------------------------------------------------------------------
# YOLO — per-class F1 at optimal confidence threshold
# ---------------------------------------------------------------------------

def eval_yolo():
    print("\n" + "="*60)
    print("YOLO — per-class F1 at optimal confidence threshold")
    print("="*60)

    model   = YOLO(YOLO_WEIGHTS)
    results = model.val(data=YOLO_DATA, imgsz=960, verbose=False)

    names  = model.names                 # {0: 'basketball', 1: 'hoop', ...}
    map50  = float(results.box.map50)

    # curves_results[1] is the F1 curve: (conf_thresholds, per_class_f1, xlabel, ylabel)
    px, py, _, _ = results.box.curves_results[1]  # px:(1000,) conf, py:(nc,1000) F1
    py = np.array(py)

    print(f"\n  Overall mAP50: {map50:.3f}")
    print(f"\n  Per-class F1 at optimal confidence threshold:")
    print(f"  {'Class':<14} {'Best F1':>8}  {'@ conf':>8}  {'AP50':>8}")
    print(f"  {'-'*44}")

    per_class_ap50 = results.box.maps   # per-class mAP50

    summary = {}
    for i, cls_name in names.items():
        if i >= py.shape[0]:
            continue
        f1_curve_i  = py[i]
        best_idx    = int(np.argmax(f1_curve_i))
        best_f1     = float(f1_curve_i[best_idx])
        best_conf   = float(px[best_idx])
        ap50_i      = float(per_class_ap50[i])
        print(f"  {cls_name:<14} {best_f1:>8.3f}  {best_conf:>8.3f}  {ap50_i:>8.3f}")
        summary[cls_name] = {"f1": best_f1, "conf": best_conf, "ap50": ap50_i}

    # Overall F1 at the threshold that maximises mean F1
    mean_f1_curve = py.mean(axis=0)
    best_mean_idx = int(np.argmax(mean_f1_curve))
    best_mean_f1  = float(mean_f1_curve[best_mean_idx])
    best_mean_conf = float(px[best_mean_idx])
    print(f"\n  Mean F1 across classes: {best_mean_f1:.3f}  @ conf {best_mean_conf:.3f}")

    return map50, summary, best_mean_f1, best_mean_conf


# ---------------------------------------------------------------------------
# TrackNet — threshold sweep (single inference pass)
# ---------------------------------------------------------------------------

def eval_tracknet():
    print("\n" + "="*60)
    print("TrackNet — heatmap threshold sweep (PE ≤ 5px)")
    print("="*60)

    import sys
    sys.path.insert(0, ".")
    from tracknet import (TrackNetV3, TrackNetDataset, INPUT_W, INPUT_H,
                          postprocess_heatmap, VIS_THRESH_INFER)
    from torch.utils.data import DataLoader

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = TrackNetV3()
    state = torch.load(TRACKNET_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    val_csv    = str(Path(TRACKNET_DATA) / "val.csv")
    val_ds     = TrackNetDataset(val_csv, augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=(device != "cpu"))

    # --- Single inference pass: collect per-frame data ---
    # records: (vis_prob, best_hmap_conf, best_cx, best_cy, has_ball, gt_cx, gt_cy)
    # vis_prob is ARCHITECTURAL (vis_head) — applied as hard gate at VIS_THRESH_INFER.
    # conf_thresh is swept to find the optimal operating point.
    records = []
    use_amp = device != "cpu"

    print(f"  Running inference on {len(val_ds)} val samples...")
    print(f"  vis_head gate: vis_prob < {VIS_THRESH_INFER} → no detection (architectural)")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for inp, target, vis_flags, _ in val_loader:
            inp    = inp.to(device, non_blocking=True)
            hmap_logits, vis_logit = model(inp)
            preds    = torch.sigmoid(hmap_logits).cpu().numpy()   # (B,1,H,W)
            vis_prob = torch.sigmoid(vis_logit).squeeze(1).cpu().numpy()  # (B,)
            tgt_np   = target.cpu().numpy()
            vis_np   = vis_flags.cpu().numpy()

            for b in range(preds.shape[0]):
                has_ball  = int(vis_np[b]) > 0
                pred_hmap = preds[b, 0]
                vp        = float(vis_prob[b])

                # Top candidate from heatmap (used for dist if it clears thresholds)
                candidates = postprocess_heatmap(pred_hmap.copy())
                best_conf  = candidates[0][2] if candidates else 0.0
                best_cx    = candidates[0][0] if candidates else 0.0
                best_cy    = candidates[0][1] if candidates else 0.0

                if has_ball:
                    gt_hmap    = tgt_np[b, 0]
                    gt_y, gt_x = divmod(int(gt_hmap.argmax()), INPUT_W)
                else:
                    gt_x = gt_y = -1

                records.append((vp, best_conf, best_cx, best_cy,
                                has_ball, float(gt_x), float(gt_y)))

    # --- Sweep conf thresholds (vis_thresh is fixed — it's architectural) ---
    thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]
    print(f"\n  {'Thresh':>7}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}  "
          f"{'TP':>6}  {'FP':>6}  {'FN':>6}")
    print(f"  {'-'*55}")

    best_f1 = best_thresh = 0.0
    best_stats = {}

    for thresh in thresholds:
        tp = fp = fn = 0
        for vp, best_conf, best_cx, best_cy, has_ball, gt_x, gt_y in records:
            # vis_head gate: architectural, fixed at VIS_THRESH_INFER
            vis_pass = vp >= VIS_THRESH_INFER
            detected = vis_pass and (best_conf >= thresh)

            if detected and has_ball:
                dist = math.hypot(best_cx - gt_x, best_cy - gt_y)
                if dist <= PE_THRESH:
                    tp += 1
                else:
                    fp += 1; fn += 1
            elif detected:
                fp += 1
            elif has_ball:
                fn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        marker = " <-- peak" if f1 > best_f1 else ""
        print(f"  {thresh:>7.2f}  {f1:>7.3f}  {prec:>7.3f}  {rec:>7.3f}  "
              f"{tp:>6}  {fp:>6}  {fn:>6}{marker}")

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh
            best_stats  = {"f1": f1, "precision": prec, "recall": rec,
                           "tp": tp, "fp": fp, "fn": fn, "threshold": thresh}

    print(f"\n  Best F1: {best_f1:.3f}  @ conf {best_thresh:.2f} "
          f"(vis_gate={VIS_THRESH_INFER})")
    return best_stats


# ---------------------------------------------------------------------------
# YOLO evaluated on TrackNet val set — apples-to-apples comparison
# ---------------------------------------------------------------------------

def eval_yolo_on_tracknet_val():
    print("\n" + "="*60)
    print("YOLO — evaluated on TrackNet val set (same game footage)")
    print("="*60)

    model  = YOLO(YOLO_WEIGHTS)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load TrackNet val CSV
    val_csv = Path(TRACKNET_DATA) / "val.csv"
    rows = []
    with open(val_csv, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    total    = len(rows)
    visible  = sum(1 for r in rows if r["visibility"] == "1")
    print(f"  Val set: {total} frames, {visible} with ball")

    # Collect (pred_conf, has_ball, dist_model_res) for each frame.
    # PE threshold is defined at model res (640x360); scale GT coords
    # down to model res for a consistent comparison with TrackNet.
    records = []
    print(f"  Running YOLO inference on {total} frames...")

    for i, row in enumerate(rows):
        fp       = row["frame_path"].replace("\\", "/")
        has_ball = row["visibility"] == "1"
        gt_x     = float(row["x"]) if has_ball else -1
        gt_y     = float(row["y"]) if has_ball else -1

        if not Path(fp).exists():
            records.append((0.0, has_ball, None))
            continue

        res = model.predict(fp, verbose=False, device=device,
                            conf=0.01, imgsz=960, classes=[YOLO_BALL_CLS])[0]

        # Best basketball detection by confidence
        best_conf = 0.0
        best_cx = best_cy = -1.0
        if res.boxes is not None and len(res.boxes):
            confs = res.boxes.conf.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            idx   = int(confs.argmax())
            best_conf = float(confs[idx])
            x1, y1, x2, y2 = boxes[idx]
            # Get original frame dims to scale GT to model res
            h, w  = res.orig_shape
            best_cx = ((x1 + x2) / 2) * INPUT_W / w
            best_cy = ((y1 + y2) / 2) * INPUT_H / h

        # Scale GT to model resolution
        if has_ball and gt_x >= 0:
            import cv2
            img = cv2.imread(fp)
            if img is not None:
                orig_h, orig_w = img.shape[:2]
                gt_mx = gt_x * INPUT_W / orig_w
                gt_my = gt_y * INPUT_H / orig_h
                dist  = math.hypot(best_cx - gt_mx, best_cy - gt_my) if best_conf > 0 else None
            else:
                dist = None
        else:
            dist = None

        records.append((best_conf, has_ball, dist))

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{total} frames done")

    # Sweep confidence thresholds
    thresholds = [round(t, 2) for t in np.arange(0.05, 0.95, 0.05)]
    print(f"\n  {'Thresh':>7}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}  "
          f"{'TP':>6}  {'FP':>6}  {'FN':>6}")
    print(f"  {'-'*55}")

    best_f1 = best_thresh = 0.0
    best_stats = {}

    for thresh in thresholds:
        tp = fp = fn = 0
        for pred_conf, has_ball, dist in records:
            detected = pred_conf >= thresh
            if detected and has_ball:
                if dist is not None and dist <= PE_THRESH:
                    tp += 1
                else:
                    fp += 1; fn += 1
            elif detected:
                fp += 1
            elif has_ball:
                fn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        marker = " <-- peak" if f1 > best_f1 else ""
        print(f"  {thresh:>7.2f}  {f1:>7.3f}  {prec:>7.3f}  {rec:>7.3f}  "
              f"{tp:>6}  {fp:>6}  {fn:>6}{marker}")

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh
            best_stats  = {"f1": f1, "precision": prec, "recall": rec,
                           "tp": tp, "fp": fp, "fn": fn, "threshold": thresh}

    print(f"\n  Best F1: {best_f1:.3f}  @ conf {best_thresh:.2f}")
    return best_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    map50, per_class, mean_f1, mean_conf = eval_yolo()
    yolo_tn_stats = eval_yolo_on_tracknet_val()
    tn_stats = eval_tracknet()

    print("\n" + "="*60)
    print("SUMMARY FOR PRESENTATION")
    print("="*60)
    print(f"\nYOLO on Roboflow val (curated)  —  mAP50: {map50:.3f}  |  Mean F1: {mean_f1:.3f} @ conf {mean_conf:.2f}")
    print(f"\n  {'Class':<14} {'F1':>6}  {'AP50':>6}  {'Best conf':>10}")
    for cls, s in per_class.items():
        print(f"  {cls:<14} {s['f1']:>6.3f}  {s['ap50']:>6.3f}  {s['conf']:>10.3f}")

    print(f"\nYOLO on game footage val (same set as TrackNet)  —  ball only, PE ≤ {PE_THRESH}px")
    print(f"  F1:        {yolo_tn_stats['f1']:.3f}")
    print(f"  Precision: {yolo_tn_stats['precision']:.3f}")
    print(f"  Recall:    {yolo_tn_stats['recall']:.3f}")
    print(f"  Threshold: {yolo_tn_stats['threshold']:.2f}")
    print(f"  TP={yolo_tn_stats['tp']}  FP={yolo_tn_stats['fp']}  FN={yolo_tn_stats['fn']}")

    print(f"\nTrackNet on game footage val  —  ball only, PE ≤ {PE_THRESH}px")
    print(f"  F1:        {tn_stats['f1']:.3f}")
    print(f"  Precision: {tn_stats['precision']:.3f}")
    print(f"  Recall:    {tn_stats['recall']:.3f}")
    print(f"  Threshold: {tn_stats['threshold']:.2f}")
    print(f"  TP={tn_stats['tp']}  FP={tn_stats['fp']}  FN={tn_stats['fn']}")
