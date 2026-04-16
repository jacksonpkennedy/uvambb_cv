"""Shared CSV backup helper for label-mutation scripts.

Every script that modifies train.csv / val.csv should call backup_csvs()
before the first write. Backups land in a timestamped subdirectory of
data/tracknet_merged/backups/ so history is never overwritten.
"""
import shutil
from datetime import datetime
from pathlib import Path


def backup_csvs(data_dir: str | Path, tag: str = "") -> Path:
    """Snapshot train.csv and val.csv to a timestamped backup directory.

    Returns the backup directory path.
    """
    data_dir = Path(data_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    backup_dir = data_dir / "backups" / f"{ts}{suffix}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("train.csv", "val.csv"):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
            copied.append(name)
    print(f"[backup] {', '.join(copied)} -> {backup_dir}")
    return backup_dir


def latest_backup(data_dir: str | Path) -> Path | None:
    """Return the most recent backup directory, or None if none exist."""
    backups = Path(data_dir) / "backups"
    if not backups.exists():
        return None
    subs = sorted([p for p in backups.iterdir() if p.is_dir()])
    return subs[-1] if subs else None
