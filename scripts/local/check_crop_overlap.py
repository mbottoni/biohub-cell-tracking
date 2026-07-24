#!/usr/bin/env python3
"""Do same-embryo crops overlap spatially?

This decides whether `ctrack.data.splits.kfold` is sound. The corpus has only two source
embryos, so leave-one-embryo-out gives just two folds. Splitting *within* an embryo buys more
folds but is only valid if that embryo's crops cover disjoint regions -- otherwise near-copies
land on both sides of a fold and every CV number is inflated.

Two checks, cheap to expensive:

1. **Exact duplicates** -- md5 of each volume's first frame. Identical crops (as train/ and
   test/ turned out to be for four ids) show up immediately.
2. **Content similarity** -- correlation between heavily downsampled frames. Overlapping or
   adjacent crops of the same embryo share structure and correlate far above the baseline set
   by two unrelated crops.

Usage:
    python scripts/check_crop_overlap.py --root /Volumes/extSSD/kaggle/biohub/raw/train
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.splits import embryo_of  # noqa: E402
from ctrack.data.volume import find_volumes, open_volume  # noqa: E402


def signature(vol, t: int = 0, bins: int = 8) -> tuple[str, np.ndarray]:
    """(md5 of the raw frame, coarse block-mean descriptor)."""
    frame = vol.frame(t)
    digest = hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()

    z, y, x = frame.shape
    trimmed = frame[: z - z % bins, : y - y % bins, : x - x % bins].astype(np.float32)
    desc = trimmed.reshape(
        bins, (z - z % bins) // bins, bins, (y - y % bins) // bins, bins, (x - x % bins) // bins
    ).mean(axis=(1, 3, 5)).ravel()
    desc = (desc - desc.mean()) / (desc.std() + 1e-8)
    return digest, desc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="correlation above which crops are called suspicious")
    args = ap.parse_args()

    paths = find_volumes(args.root)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.exit(f"no .zarr stores under {args.root}")
    print(f"{len(paths)} volumes\n")

    names, digests, descs = [], [], []
    for i, p in enumerate(paths, 1):
        v = open_volume(p)
        d, s = signature(v)
        names.append(v.dataset)
        digests.append(d)
        descs.append(s)
        if i % 25 == 0:
            print(f"  ...{i}/{len(paths)}", file=sys.stderr)

    print("== exact duplicates (identical first frame) ==")
    seen: dict[str, list[str]] = {}
    for n, d in zip(names, digests):
        seen.setdefault(d, []).append(n)
    dupes = {d: ns for d, ns in seen.items() if len(ns) > 1}
    if dupes:
        for ns in dupes.values():
            print(f"  IDENTICAL: {ns}")
    else:
        print("  none")

    matrix = np.vstack(descs)
    corr = (matrix @ matrix.T) / matrix.shape[1]
    np.fill_diagonal(corr, np.nan)

    same = np.array([[embryo_of(a) == embryo_of(b) for b in names] for a in names])
    np.fill_diagonal(same, False)

    within = corr[same & ~np.isnan(corr)]
    across = corr[~same & ~np.isnan(corr)]

    print("\n== content correlation ==")
    for label, vals in (("same embryo", within), ("different embryo", across)):
        if vals.size:
            print(f"  {label:18s} n={vals.size:6d}  median {np.median(vals):+.3f}  "
                  f"p95 {np.percentile(vals, 95):+.3f}  max {vals.max():+.3f}")

    hits = [
        (corr[i, j], names[i], names[j])
        for i in range(len(names)) for j in range(i + 1, len(names))
        if same[i, j] and corr[i, j] > args.threshold
    ]
    print(f"\n== suspicious pairs (corr > {args.threshold}) ==")
    if hits:
        for c, a, b in sorted(hits, reverse=True)[:20]:
            print(f"  {c:+.3f}  {a}  <->  {b}")
        print(f"\n  {len(hits)} pair(s). Same-embryo crops are NOT clearly disjoint --")
        print("  prefer split_by_embryo; treat kfold results as optimistic.")
    else:
        print("  none")
        print("\n  No same-embryo pair exceeds the threshold, and the within/across medians")
        print("  above should be compared: if they are similar, crops look spatially")
        print("  independent and kfold within an embryo is defensible.")


if __name__ == "__main__":
    main()
