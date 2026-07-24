#!/usr/bin/env python3
"""Train the 3D U-Net point detector locally (thin wrapper over ctrack.detect.train.train_fold).

Validation is peak recall: the fraction of held-out GT points within 7 µm (the metric's match
radius) of a predicted peak — a cheap per-epoch proxy for edge Jaccard. See ctrack.detect.train
for the actual loop, shared with the Colab / colab-CLI / Vast entry points.

    python scripts/train_detector.py --raw .../train --gt .../train --fold 0 --epochs 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.detect.train import train_fold  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--val-threshold", type=float, default=0.3)
    ap.add_argument("--train-stride", type=int, default=3)
    ap.add_argument("--val-stride", type=int, default=4)
    ap.add_argument("--limit", type=int, help="cap datasets per split (fast smoke test)")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    train_fold(
        args.raw, args.gt, fold=args.fold, k=args.k, epochs=args.epochs, batch=args.batch,
        lr=args.lr, base=args.base, val_threshold=args.val_threshold,
        train_stride=args.train_stride, val_stride=args.val_stride, limit=args.limit,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
