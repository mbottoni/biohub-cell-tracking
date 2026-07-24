#!/usr/bin/env python3
"""Download a *subset* of the competition data, by dataset prefix.

The full archive is ~87 GB, which does not fit on a laptop alongside its own unzipped
contents. Since this is a kernels-only competition the heavy runs happen on Kaggle anyway --
locally we only need a couple of datasets for EDA and pipeline development.

Usage:
    # what's there, and how big
    python scripts/fetch_subset.py --list

    # pull specific zarr stores (+ their .geff ground truth), preserving paths
    python scripts/fetch_subset.py --datasets train/abcd_1234 --out data/raw

    # pull the n smallest training datasets
    python scripts/fetch_subset.py --smallest 2 --split train

Auth: KAGGLE_API_TOKEN (bearer). Sourced from .env at the repo root if present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

COMP = "biohub-cell-tracking-during-development"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "filelist.json"


def _token() -> str:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip().removeprefix("export ")
                if line.startswith("KAGGLE_API_TOKEN="):
                    os.environ["KAGGLE_API_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
    tok = os.environ.get("KAGGLE_API_TOKEN")
    if not tok:
        sys.exit("KAGGLE_API_TOKEN unset -- see README")
    return tok


def _get(url: str, tok: str, *, raw: bool = False, retries: int = 6):
    """GET with backoff. Kaggle 429s readily when enumerating thousands of files."""
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {tok}", "User-Agent": "kaggle-api"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read() if raw else json.loads(r.read())
        except Exception as exc:  # noqa: BLE001 -- retry everything, surface on exhaustion
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}: {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def list_files(tok: str, *, refresh: bool = False) -> list[dict]:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    files, page_token, page = [], None, 0
    while True:
        url = f"https://www.kaggle.com/api/v1/competitions/data/list/{COMP}?pageSize=200"
        if page_token:
            url += "&pageToken=" + urllib.parse.quote(page_token)
        data = _get(url, tok)
        batch = data.get("files", [])
        files += batch
        page_token = data.get("nextPageTokenNullable")
        page += 1
        if page % 20 == 0:
            print(f"  page {page}: {len(files)} files", file=sys.stderr)
        if not page_token or not batch:
            break
        time.sleep(1.0)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(files))
    return files


def group_by_dataset(files: list[dict]) -> dict[str, dict]:
    """Group files by their owning .zarr/.geff store, e.g. 'test/44b6_0113de3b'."""
    groups: dict[str, dict] = defaultdict(lambda: {"files": [], "bytes": 0})
    for f in files:
        name = f["name"]
        key = name
        for marker in (".zarr/", ".geff/"):
            if marker in name:
                key = name.split(marker)[0]
                break
        else:
            key = name.rsplit(".", 1)[0] if "/" in name else name
        groups[key]["files"].append(f)
        groups[key]["bytes"] += f.get("totalBytes") or 0
    return dict(groups)


def download(files: list[dict], out: Path, tok: str) -> None:
    total = sum(f.get("totalBytes") or 0 for f in files)
    done = 0
    for i, f in enumerate(files, 1):
        name = f["name"]
        dest = out / name
        size = f.get("totalBytes") or 0
        if dest.exists() and dest.stat().st_size == size:
            done += size
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMP}/{urllib.parse.quote(name)}"
        blob = _get(url, tok, raw=True)
        dest.write_bytes(blob)
        done += size
        pct = 100 * done / total if total else 100
        print(f"\r  [{i}/{len(files)}] {pct:5.1f}%  {done/1e9:6.2f}/{total/1e9:.2f} GB  {name[-52:]:<52}",
              end="", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show datasets and sizes, download nothing")
    ap.add_argument("--datasets", nargs="*", default=[], help="dataset keys to fetch")
    ap.add_argument("--smallest", type=int, help="fetch the N smallest datasets")
    ap.add_argument("--split", default="", help="restrict to a prefix, e.g. train or test")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--refresh", action="store_true", help="re-fetch the file listing")
    args = ap.parse_args()

    tok = _token()
    files = list_files(tok, refresh=args.refresh)
    groups = group_by_dataset(files)

    if args.list:
        total = sum(g["bytes"] for g in groups.values())
        print(f"{len(groups)} datasets, {total/1e9:.1f} GB total\n")
        for key, g in sorted(groups.items(), key=lambda kv: -kv[1]["bytes"]):
            print(f"  {key:<50} {g['bytes']/1e9:8.2f} GB  {len(g['files']):>6} files")
        return

    keys = list(args.datasets)
    if args.smallest:
        candidates = {k: v for k, v in groups.items() if k.startswith(args.split)}
        keys += [k for k, _ in sorted(candidates.items(), key=lambda kv: kv[1]["bytes"])[: args.smallest]]
    if not keys:
        sys.exit("nothing selected -- pass --datasets or --smallest (or --list to explore)")

    selected: list[dict] = []
    for k in keys:
        matches = [g for name, g in groups.items() if name == k or name.startswith(k)]
        if not matches:
            sys.exit(f"no dataset matching {k!r} -- try --list")
        for g in matches:
            selected += g["files"]

    size = sum(f.get("totalBytes") or 0 for f in selected)
    print(f"fetching {len(keys)} dataset(s): {len(selected)} files, {size/1e9:.2f} GB -> {args.out}")
    download(selected, Path(args.out), tok)


if __name__ == "__main__":
    main()
