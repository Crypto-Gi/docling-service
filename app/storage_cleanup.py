from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

MAX_STORAGE_BYTES = 100 * 1024 * 1024  # 100 MiB
RETENTION_DAYS = 30


def cleanup_directory(path: Path, max_bytes: int = MAX_STORAGE_BYTES, retention_days: int = RETENTION_DAYS) -> None:
    """Remove files older than retention or exceeding the directory size budget."""
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if retention_days < 0:
        raise ValueError("retention_days must be non-negative")
    if not path.exists():
        return

    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    entries: List[Tuple[Path, float, int]] = []
    total_size = 0

    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            stat = item.stat()
        except FileNotFoundError:
            continue
        modified = datetime.utcfromtimestamp(stat.st_mtime)
        if retention_days > 0 and modified < cutoff:
            try:
                item.unlink()
            except FileNotFoundError:
                pass
            continue
        entries.append((item, stat.st_mtime, stat.st_size))
        total_size += stat.st_size

    if total_size <= max_bytes:
        return

    entries.sort(key=lambda entry: entry[1])
    for file_path, _mtime, size in entries:
        try:
            file_path.unlink()
            total_size -= size
        except FileNotFoundError:
            continue
        if total_size <= max_bytes:
            break


def cleanup_storage(
    root: Path,
    *,
    uploads_subdir: str = "uploads",
    outputs_subdir: str = "outputs",
    max_bytes: int = MAX_STORAGE_BYTES,
    retention_days: int = RETENTION_DAYS,
) -> None:
    """Clean both uploads and outputs directories rooted at ``root``."""
    cleanup_directory(root / uploads_subdir, max_bytes=max_bytes, retention_days=retention_days)
    cleanup_directory(root / outputs_subdir, max_bytes=max_bytes, retention_days=retention_days)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Docling storage directories")
    parser.add_argument("root", type=Path, nargs="?", default=Path("/data"), help="Storage root path")
    parser.add_argument("--max-bytes", type=int, default=MAX_STORAGE_BYTES, help="Maximum size per directory in bytes")
    parser.add_argument("--retention-days", type=int, default=RETENTION_DAYS, help="Retention window in days")
    parser.add_argument("--uploads-subdir", type=str, default="uploads", help="Uploads subdirectory name")
    parser.add_argument("--outputs-subdir", type=str, default="outputs", help="Outputs subdirectory name")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cleanup_storage(
        root=args.root,
        uploads_subdir=args.uploads_subdir,
        outputs_subdir=args.outputs_subdir,
        max_bytes=args.max_bytes,
        retention_days=args.retention_days,
    )


if __name__ == "__main__":
    main()
