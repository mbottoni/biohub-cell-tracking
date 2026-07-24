#!/usr/bin/env python3
"""Extract the competition archive, with progress and resume.

Python's zipfile is used rather than `unzip` because the archive is ~87 GB (zip64) and holds
~25k small files, and we want progress reporting plus the ability to resume after an
interruption. Already-extracted entries with the right size are skipped.

Usage:
    python scripts/unzip_data.py --zip /Volumes/extSSD/kaggle/biohub/raw/*.zip --dest ...
    python scripts/unzip_data.py --zip ... --dest ... --wait   # block until the zip is complete
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

EXPECTED_BYTES = 87_393_127_165


def wait_for_complete(zip_path: Path, expected: int, poll: float = 30.0) -> None:
    """Block until the archive stops growing and the partial marker is gone."""
    marker = Path(str(zip_path) + ".kaggle-partial")
    last, stable = -1, 0
    while True:
        size = zip_path.stat().st_size if zip_path.exists() else 0
        if not marker.exists() and size >= expected:
            print(f"archive complete: {size / 1e9:.1f} GB", flush=True)
            return
        if size == last:
            stable += 1
            if stable >= 4 and not marker.exists():
                print(f"archive stopped growing at {size / 1e9:.1f} GB "
                      f"(expected {expected / 1e9:.1f}) -- proceeding anyway", flush=True)
                return
        else:
            stable = 0
        last = size
        print(f"waiting: {size / 1e9:.1f} / {expected / 1e9:.1f} GB", flush=True)
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--wait", action="store_true", help="block until the download completes")
    ap.add_argument("--expected", type=int, default=EXPECTED_BYTES)
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    zip_path, dest = Path(args.zip), Path(args.dest)
    if args.wait:
        wait_for_complete(zip_path, args.expected)
    if not zip_path.exists():
        sys.exit(f"missing {zip_path}")

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        entries = zf.infolist()
        total = sum(e.file_size for e in entries)
        print(f"{len(entries)} entries, {total / 1e9:.1f} GB uncompressed", flush=True)

        done = skipped = 0
        for i, entry in enumerate(entries, 1):
            out = dest / entry.filename
            if out.exists() and not entry.is_dir() and out.stat().st_size == entry.file_size:
                done += entry.file_size
                skipped += 1
                continue
            zf.extract(entry, dest)
            done += entry.file_size
            if i % 500 == 0 or i == len(entries):
                print(f"  [{i}/{len(entries)}] {100 * done / total:5.1f}%  "
                      f"{done / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)

    print(f"EXTRACTED {len(entries)} entries ({skipped} already present)", flush=True)
    if not args.keep_zip:
        zip_path.unlink()
        print("removed archive", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
