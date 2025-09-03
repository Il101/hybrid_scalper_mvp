"""rotate_logs.py

Safe log rotation for CSV and log files in this project.
- Archives and gzips files when they exceed a size threshold (MB) or are older than X days.
- Keeps only the latest N archives per filename.
- After archiving a CSV, rewrites the original file with its header only (so processes can keep appending).

Usage:
  python scripts/rotate_logs.py                 # run with defaults
  python scripts/rotate_logs.py --threshold-mb 100 --keep 14

This script is conservative and safe: it compresses the existing file and preserves the header.
"""
from __future__ import annotations
import argparse
import os
import gzip
import shutil
import glob
from datetime import datetime, timedelta
from pathlib import Path


def archive_file(path: Path, archive_dir: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{path.name}.{ts}.gz"
    # Copy and gzip
    with path.open("rb") as src, gzip.open(archive_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return archive_path


def rotate_file(path: Path, archive_dir: Path):
    print(f"Archiving {path} -> {archive_dir}")
    # Read header (for CSVs). If cannot read, header will be empty.
    header = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
    except Exception:
        header = ""
    # Archive compress
    archive_path = archive_file(path, archive_dir)
    # Truncate original file and write header back (if any)
    try:
        with path.open("w", encoding="utf-8") as fh:
            if header:
                fh.write(header if header.endswith("\n") else header + "\n")
    except Exception as e:
        print(f"Warning: failed to truncate/write header to {path}: {e}")
    print(f"Archived to {archive_path}")


def cleanup_archives(archive_dir: Path, base_name: str, keep: int):
    # Find archives for this base_name sorted by mtime desc and keep 'keep' newest
    pattern = archive_dir / f"{base_name}.*.gz"
    items = sorted(glob.glob(str(pattern)), key=lambda p: os.path.getmtime(p), reverse=True)
    if len(items) <= keep:
        return
    for old in items[keep:]:
        try:
            os.remove(old)
            print(f"Removed old archive: {old}")
        except Exception as e:
            print(f"Failed to remove {old}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate and archive log files in logs/")
    parser.add_argument("--paths", nargs="*",
                        default=["logs/signals.csv", "logs/trades.csv", "logs/*.log"],
                        help="Files or glob patterns to rotate")
    parser.add_argument("--threshold-mb", type=int, default=50,
                        help="Rotate files larger than this size in MB")
    parser.add_argument("--older-than-days", type=int, default=0,
                        help="Rotate files older than this many days (0 = ignore)")
    parser.add_argument("--keep", type=int, default=7,
                        help="How many recent archives to keep per filename")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        for match in glob.glob(p):
            files.append(Path(match))

    if not files:
        print("No files matched the given paths.")
        raise SystemExit(0)

    now = datetime.utcnow()
    threshold_bytes = args.threshold_mb * 1024 * 1024

    for path in files:
        if not path.exists():
            continue
        stat = path.stat()
        size = stat.st_size
        mtime = datetime.utcfromtimestamp(stat.st_mtime)
        age_days = (now - mtime).days
        should_rotate = False
        if args.older_than_days > 0 and age_days >= args.older_than_days:
            should_rotate = True
        if size >= threshold_bytes:
            should_rotate = True

        if should_rotate:
            archive_dir = path.parent / "archive"
            try:
                rotate_file(path, archive_dir)
                cleanup_archives(archive_dir, path.name, args.keep)
            except Exception as e:
                print(f"Failed to rotate {path}: {e}")
        else:
            print(f"Skipping {path}: size={size} bytes, age_days={age_days}")

    print("Rotation complete.")
