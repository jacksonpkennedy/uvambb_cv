"""Append-only fix log — persists every label correction so re-merging
source data doesn't wipe out cleaning work.

Schema of data/tracknet_merged/fix_log.csv:
    timestamp, game, stem, action, visibility, x, y

Actions:
    remove      — label was a false positive; set visibility=0, x=-1, y=-1
    reposition  — label position was corrected to (x, y); visibility unchanged
    add         — previously-invisible frame got a real ball label at (x, y)
    zero        — neither-correct position_mismatch zeroed

Replayed in order at the end of merge_labels.py so a fresh re-merge always
reflects all prior corrections.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

FIELDS = ["timestamp", "game", "stem", "action", "visibility", "x", "y"]


def log_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "fix_log.csv"


def append(data_dir: str | Path, entries: list[dict]) -> None:
    """Append a batch of fix entries. Creates the file with header if needed.

    Each entry must have: game, stem, action. visibility/x/y optional
    (defaults: visibility=0, x=-1, y=-1 for removes/zeros). Timestamp is
    added automatically.
    """
    if not entries:
        return
    p = log_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_file = not p.exists()
    ts = datetime.now().isoformat(timespec="seconds")
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for e in entries:
            w.writerow({
                "timestamp": ts,
                "game":       e["game"],
                "stem":       e["stem"],
                "action":     e["action"],
                "visibility": e.get("visibility", 0),
                "x":          e.get("x", -1),
                "y":          e.get("y", -1),
            })


def read_entries(data_dir: str | Path) -> list[dict]:
    """Return all entries in chronological order (preserved by csv order)."""
    p = log_path(data_dir)
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def apply_to_rows(rows: list[dict], entries: list[dict]) -> dict:
    """Mutate `rows` (list of dicts with frame_path/visibility/x/y) by
    replaying each fix entry in order. Returns a stats dict.
    """
    stats = {"remove": 0, "reposition": 0, "add": 0, "zero": 0, "missing": 0}

    # Index rows by (game, stem) for O(1) lookup
    def _key(path: str) -> tuple[str, str]:
        pp = Path(path)
        return (pp.parent.name, pp.stem.replace("_640x360", ""))

    index: dict[tuple[str, str], dict] = {}
    for row in rows:
        index[_key(row["frame_path"])] = row

    for e in entries:
        k = (e["game"], e["stem"])
        row = index.get(k)
        if row is None:
            stats["missing"] += 1
            continue
        action = e["action"]
        if action in ("remove", "zero"):
            row["visibility"] = "0"
            row["x"] = "-1"
            row["y"] = "-1"
            stats[action] += 1
        elif action == "reposition":
            row["x"] = str(e["x"])
            row["y"] = str(e["y"])
            stats["reposition"] += 1
        elif action == "add":
            row["visibility"] = "1"
            row["x"] = str(e["x"])
            row["y"] = str(e["y"])
            stats["add"] += 1
    return stats
