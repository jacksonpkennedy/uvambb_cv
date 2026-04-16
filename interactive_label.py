"""Interactive point-and-click labeler for TrackNet disagreements.

Replaces the keep/delete folder workflow with a single session that writes
directly to fix_log.csv. After reviewing, run:
  python merge_labels.py
  python tracknet.py --train --data data/tracknet_merged --epochs 40 --batch 8

Two-stage workflow per frame:
  Stage 1 — REVIEW: see current annotations, optionally click a position
  Stage 2 — PREVIEW: see what the frame will look like after the fix
             Confirm with Y/Enter or cancel with ESC/N

Controls (Stage 1 — Review):
  Click           Set custom ball position (cyan dot)
  Right-click     Clear custom position
  A               Accept → use clicked pos if set, else model prediction
  R               Remove/zero → mark as no-ball
  K               Keep → leave label unchanged, move to next
  N / →           Next frame (no action)
  P / ←           Previous frame
  U               Undo last fix_log entry (only unqueued entries)
  Q / ESC         Save and quit

Controls (Stage 2 — Preview):
  Y / Enter       Confirm and log the fix
  N / ESC         Cancel, return to stage 1

Category semantics:
  false_negatives  A=add ball at model/clicked pos  R=skip(no action)  K=skip
  false_positives  A/R=remove bad label             K=keep(label real)
  position_mismatch A=reposition to model/clicked   R=zero(neither)    K=keep label
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DISPLAY_SCALE = 2
MODEL_W, MODEL_H = 640, 360
DISPLAY_W, DISPLAY_H = MODEL_W * DISPLAY_SCALE, MODEL_H * DISPLAY_SCALE

PROGRESS_FILE = "output/disagreements/.review_progress.json"
DATA_DIR = "data/tracknet_merged"


# ---------------------------------------------------------------------------
# Load disagreement entries from CSVs
# ---------------------------------------------------------------------------

def _key_from_path(frame_path: str) -> tuple[str, str]:
    p = Path(frame_path)
    return (p.parent.name, p.stem)


def load_all_entries(disagree_dir: Path, category_filter: str | None) -> list[dict]:
    entries = []
    category_configs = [
        "false_negatives",
        "false_positives",
        "position_mismatch",
    ]
    for split in ("train", "val"):
        for cat in category_configs:
            if category_filter and cat != category_filter:
                continue
            csv_path = disagree_dir / f"{split}_{cat}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    fp = row["frame_path"].replace("\\", "/")
                    game, stem = _key_from_path(fp)
                    entries.append({
                        "frame_path": fp,
                        "category":   cat,
                        "split":      split,
                        "game":       game,
                        "stem":       stem,
                        "label_x":    float(row.get("label_x", -1) or -1),
                        "label_y":    float(row.get("label_y", -1) or -1),
                        "pred_x":     float(row.get("pred_x",  -1) or -1),
                        "pred_y":     float(row.get("pred_y",  -1) or -1),
                        "pred_conf":  float(row.get("pred_conf", 0) or 0),
                    })
    return entries


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    p = Path(PROGRESS_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"reviewed": {}}


def save_progress(progress: dict) -> None:
    p = Path(PROGRESS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(progress, indent=2))


def entry_key(e: dict) -> str:
    return f"{e['game']}/{e['stem']}/{e['category']}"


# ---------------------------------------------------------------------------
# fix_log integration
# ---------------------------------------------------------------------------

_fix_log_queue: list[dict] = []


def _flush_fix_log() -> None:
    if not _fix_log_queue:
        return
    sys.path.insert(0, ".")
    import fix_log as fl
    fl.append(DATA_DIR, _fix_log_queue)
    print(f"  Written {len(_fix_log_queue)} entries to fix_log.csv")
    _fix_log_queue.clear()


def _queue_fix(entry: dict, action: str, cx: float = -1, cy: float = -1) -> None:
    e = {"game": entry["game"], "stem": entry["stem"], "action": action}
    if action in ("reposition", "add"):
        e["visibility"] = 1
        e["x"] = round(cx, 1)
        e["y"] = round(cy, 1)
    _fix_log_queue.append(e)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_vis_image(entry: dict, disagree_dir: Path) -> np.ndarray | None:
    """Load pre-annotated disagreement JPG, or build from raw frame."""
    cat_dir = disagree_dir / entry["category"] / entry["game"]
    vis_path = cat_dir / f"{entry['stem']}.jpg"
    if vis_path.exists():
        img = cv2.imread(str(vis_path))
        if img is not None:
            return cv2.resize(img, (MODEL_W, MODEL_H))

    img = cv2.imread(entry["frame_path"])
    if img is None:
        return None
    img = cv2.resize(img, (MODEL_W, MODEL_H))
    lx, ly = int(entry["label_x"]), int(entry["label_y"])
    px, py = int(entry["pred_x"]),  int(entry["pred_y"])
    if entry["label_x"] >= 0:
        cv2.circle(img, (lx, ly), 8, (0, 0, 255), 2)
    if entry["pred_x"] >= 0:
        cv2.circle(img, (px, py), 8, (0, 255, 0), 2)
    return img


def _load_raw_frame(entry: dict) -> np.ndarray | None:
    """Load clean frame with no annotations (for preview)."""
    img = cv2.imread(entry["frame_path"])
    if img is None:
        return None
    return cv2.resize(img, (MODEL_W, MODEL_H))


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------

CAT_COLORS = {
    "false_negatives":   (50, 255, 50),
    "false_positives":   (50, 50, 255),
    "position_mismatch": (50, 220, 255),
}
CAT_SHORT = {
    "false_negatives":   "FN — model detected ball, no label",
    "false_positives":   "FP — label has ball, model missed",
    "position_mismatch": "PM — position mismatch (red=label, green=model)",
}
CAT_HINT = {
    "false_negatives":   "A=add  R=skip(no action)  K=skip  N=next  P=prev  U=undo  Q=quit",
    "false_positives":   "A/R=remove bad label  K=keep(real ball)  N=next  P=prev  U=undo  Q=quit",
    "position_mismatch": "A=reposition  R=zero(neither)  K=keep label  N=next  P=prev  U=undo  Q=quit",
}


def _draw_bar(img: np.ndarray, lines: list[tuple[str, tuple]]) -> None:
    """Draw a semi-transparent top bar with text lines."""
    bar_h = 16 + 17 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (MODEL_W, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (6, 14 + 17 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_review(base: np.ndarray, entry: dict,
                idx: int, total: int, n_done: int,
                click_pt: tuple | None,
                already: str | None) -> np.ndarray:
    img = base.copy()
    cat = entry["category"]
    color = CAT_COLORS[cat]

    if click_pt:
        cv2.circle(img, click_pt, 9, (255, 220, 0), -1)
        cv2.circle(img, click_pt, 9, (0, 0, 0), 1)
        cv2.putText(img, f"({click_pt[0]},{click_pt[1]})",
                    (click_pt[0] + 11, click_pt[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 0), 1)

    lines = [
        (CAT_SHORT[cat], color),
        (f"{entry['game']}  {entry['stem']}  [{idx+1}/{total}]  done={n_done}"
         + (f"  [prev: {already}]" if already else ""),
         (200, 200, 200)),
        (CAT_HINT[cat], (160, 160, 160)),
    ]
    if click_pt:
        lines.append((f"Click set at ({click_pt[0]},{click_pt[1]})  — A will use this position",
                      (255, 220, 0)))
    _draw_bar(img, lines)
    return img


def draw_preview(raw: np.ndarray, entry: dict,
                 action: str, cx: float, cy: float) -> np.ndarray:
    """Render what the frame will look like after the fix is applied."""
    img = raw.copy() if raw is not None else np.zeros((MODEL_H, MODEL_W, 3), np.uint8)
    cat = entry["category"]

    if action == "add":
        # Show newly added label as bright white circle
        cv2.circle(img, (int(cx), int(cy)), 10, (255, 255, 255), 2)
        cv2.circle(img, (int(cx), int(cy)), 3,  (255, 255, 255), -1)
        label = f"ADD ball at ({cx:.0f},{cy:.0f})"
        label_color = (100, 255, 100)

    elif action == "reposition":
        # Show old label (red, faded) and new position (white)
        if entry["label_x"] >= 0:
            cv2.circle(img, (int(entry["label_x"]), int(entry["label_y"])),
                       8, (60, 60, 180), 1)   # faded red = old
        cv2.circle(img, (int(cx), int(cy)), 10, (255, 255, 255), 2)
        cv2.circle(img, (int(cx), int(cy)), 3,  (255, 255, 255), -1)
        if entry["label_x"] >= 0:
            cv2.arrowedLine(img,
                            (int(entry["label_x"]), int(entry["label_y"])),
                            (int(cx), int(cy)),
                            (200, 200, 200), 1, tipLength=0.3)
        label = f"REPOSITION  ({entry['label_x']:.0f},{entry['label_y']:.0f}) → ({cx:.0f},{cy:.0f})"
        label_color = (100, 220, 255)

    elif action == "remove":
        # Show the label that will be zeroed out (struck through)
        if entry["label_x"] >= 0:
            lx, ly = int(entry["label_x"]), int(entry["label_y"])
            cv2.circle(img, (lx, ly), 10, (60, 60, 180), 2)
            cv2.line(img, (lx - 12, ly - 12), (lx + 12, ly + 12), (60, 60, 180), 2)
            cv2.line(img, (lx + 12, ly - 12), (lx - 12, ly + 12), (60, 60, 180), 2)
        label = "REMOVE label (mark as no-ball)"
        label_color = (80, 80, 255)

    elif action == "zero":
        # Show both positions struck through
        for (px, py) in [(entry["label_x"], entry["label_y"]),
                         (entry["pred_x"],  entry["pred_y"])]:
            if px >= 0:
                ix, iy = int(px), int(py)
                cv2.circle(img, (ix, iy), 10, (100, 100, 100), 2)
                cv2.line(img, (ix-12, iy-12), (ix+12, iy+12), (100, 100, 100), 2)
                cv2.line(img, (ix+12, iy-12), (ix-12, iy+12), (100, 100, 100), 2)
        label = "ZERO — neither position correct"
        label_color = (160, 220, 255)

    elif action in ("skip(no action)", "keep"):
        # Redraw original annotations
        if entry["label_x"] >= 0:
            cv2.circle(img, (int(entry["label_x"]), int(entry["label_y"])), 8, (0, 0, 255), 2)
        if entry["pred_x"] >= 0:
            cv2.circle(img, (int(entry["pred_x"]), int(entry["pred_y"])), 8, (0, 255, 0), 2)
        label = "NO CHANGE — skipping this frame"
        label_color = (160, 160, 160)
    else:
        label = action
        label_color = (200, 200, 200)

    lines = [
        ("PREVIEW — what will be saved:", (220, 220, 220)),
        (label, label_color),
        ("Y / Enter = CONFIRM     ESC / N = CANCEL", (100, 255, 100)),
    ]
    _draw_bar(img, lines)
    return img


# ---------------------------------------------------------------------------
# Mouse callback state
# ---------------------------------------------------------------------------

_click_pt: tuple | None = None


def _mouse_cb(event, x, y, flags, param):
    global _click_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_pt = (x // DISPLAY_SCALE, y // DISPLAY_SCALE)
    elif event == cv2.EVENT_RBUTTONDOWN:
        _click_pt = None


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------

def review(disagree_dir: Path, data_dir: str, start_idx: int = 0,
           category_filter: str | None = None) -> None:
    global DATA_DIR, _click_pt
    DATA_DIR = data_dir

    entries = load_all_entries(disagree_dir, category_filter)
    if not entries:
        print("No disagreement entries found. Run find_disagreements.py --visualize first.")
        return

    progress = load_progress()
    n_done = len(progress["reviewed"])

    print(f"Loaded {len(entries)} entries.")
    print(f"Already reviewed: {n_done} (resuming from first unreviewed).")

    if start_idx == 0:
        for i, e in enumerate(entries):
            if entry_key(e) not in progress["reviewed"]:
                start_idx = i
                break
        else:
            print("All entries already reviewed!")
            return

    cv2.namedWindow("TrackNet Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TrackNet Labeler", DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback("TrackNet Labeler", _mouse_cb)

    idx = start_idx
    while 0 <= idx < len(entries):
        entry  = entries[idx]
        ek     = entry_key(entry)
        _click_pt = None

        base_img = _load_vis_image(entry, disagree_dir)
        if base_img is None:
            print(f"  [skip] image not found: {ek}")
            idx += 1
            continue

        raw_img = _load_raw_frame(entry)
        already = progress["reviewed"].get(ek)

        # ---- Stage 1: Review ------------------------------------------------
        pending_action: str | None = None
        pending_cx = pending_cy = -1.0

        stage = "review"
        while True:
            if stage == "review":
                hud = draw_review(base_img, entry, idx, len(entries),
                                  n_done, _click_pt, already)
            else:  # preview
                preview_raw = raw_img if raw_img is not None else base_img
                hud = draw_preview(preview_raw, entry,
                                   pending_action, pending_cx, pending_cy)

            display = cv2.resize(hud, (DISPLAY_W, DISPLAY_H),
                                  interpolation=cv2.INTER_NEAREST)
            cv2.imshow("TrackNet Labeler", display)
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            # ---- Preview stage: only Y/Enter to confirm or ESC/N to cancel --
            if stage == "preview":
                if key in (ord('y'), 13):   # Y or Enter = confirm
                    # Log the action
                    if pending_action not in ("skip(no action)", "keep"):
                        _queue_fix(entry, pending_action, pending_cx, pending_cy)
                    if ek not in progress["reviewed"]:
                        n_done += 1
                    progress["reviewed"][ek] = pending_action
                    if len(_fix_log_queue) >= 50:
                        _flush_fix_log()
                        save_progress(progress)
                    print(f"  [{idx+1}/{len(entries)}] {ek}  → {pending_action}")
                    idx += 1
                    break
                elif key in (27, ord('n')):  # ESC or N = cancel
                    stage = "review"
                continue

            # ---- Review stage -----------------------------------------------

            # Navigation (no action)
            if key in (ord('n'), 83):   # n or →
                idx += 1; break
            if key in (ord('p'), 81):   # p or ←
                idx = max(0, idx - 1); break

            # Quit
            if key in (ord('q'), 27):
                _flush_fix_log()
                save_progress(progress)
                print(f"\nSaved. {n_done} entries reviewed total.")
                cv2.destroyAllWindows()
                _print_next_steps()
                return

            # Undo last queued fix_log entry
            if key == ord('u'):
                if _fix_log_queue:
                    removed = _fix_log_queue.pop()
                    # Remove from progress by matching game+stem
                    for k in list(progress["reviewed"]):
                        g, s, _ = k.split("/", 2)
                        if g == removed["game"] and s == removed["stem"]:
                            del progress["reviewed"][k]
                            n_done -= 1
                            already = None
                            print(f"  Undid: {removed['action']} on {g}/{s}")
                            break
                    save_progress(progress)
                else:
                    print("  Nothing to undo (already flushed to disk).")
                continue

            # Action keys — compute proposed fix, then enter preview
            cat = entry["category"]
            cx_click = float(_click_pt[0]) if _click_pt else -1.0
            cy_click = float(_click_pt[1]) if _click_pt else -1.0
            cx_model, cy_model = entry["pred_x"], entry["pred_y"]

            proposed_action = None
            proposed_cx = proposed_cy = -1.0

            if key == ord('a'):
                if cat == "false_negatives":
                    if _click_pt:
                        proposed_action, proposed_cx, proposed_cy = "add", cx_click, cy_click
                    elif cx_model >= 0:
                        proposed_action, proposed_cx, proposed_cy = "add", cx_model, cy_model
                    else:
                        print("  No position — click the ball first, then A")
                        continue
                elif cat == "false_positives":
                    proposed_action = "remove"
                elif cat == "position_mismatch":
                    if _click_pt:
                        proposed_action, proposed_cx, proposed_cy = "reposition", cx_click, cy_click
                    elif cx_model >= 0:
                        proposed_action, proposed_cx, proposed_cy = "reposition", cx_model, cy_model
                    else:
                        print("  No model prediction — click the correct position first")
                        continue

            elif key == ord('r'):
                if cat == "false_negatives":
                    proposed_action = "skip(no action)"  # model was FP, no label to write
                elif cat == "false_positives":
                    proposed_action = "remove"
                elif cat == "position_mismatch":
                    proposed_action = "zero"

            elif key == ord('k'):
                proposed_action = "keep"

            if proposed_action is not None:
                pending_action = proposed_action
                pending_cx     = proposed_cx
                pending_cy     = proposed_cy
                stage = "preview"
                continue

    # All entries processed
    _flush_fix_log()
    save_progress(progress)
    cv2.destroyAllWindows()
    print(f"\nAll {len(entries)} entries processed. {n_done} total reviewed.")
    _print_next_steps()


def _print_next_steps() -> None:
    print("\nNext steps:")
    print("  python merge_labels.py")
    print("  python tracknet.py --train --data data/tracknet_merged --epochs 40 --batch 8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive point-and-click labeler for TrackNet disagreements")
    parser.add_argument("--disagreements", default="output/disagreements")
    parser.add_argument("--data", default="data/tracknet_merged")
    parser.add_argument("--category",
                        choices=["false_negatives", "false_positives", "position_mismatch"],
                        default=None, help="Review only this category (default: all)")
    parser.add_argument("--from", dest="start_idx", type=int, default=0)
    parser.add_argument("--reset-progress", action="store_true",
                        help="Ignore saved progress and start from entry 0")
    args = parser.parse_args()

    if args.reset_progress:
        p = Path(PROGRESS_FILE)
        if p.exists():
            p.unlink()
            print("Progress reset.")

    review(
        disagree_dir=Path(args.disagreements),
        data_dir=args.data,
        start_idx=args.start_idx,
        category_filter=args.category,
    )
