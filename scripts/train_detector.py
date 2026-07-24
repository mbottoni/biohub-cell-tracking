#!/usr/bin/env python3
"""Train the 3D U-Net point detector.

Validation is **peak recall**: the fraction of held-out GT points within 7 µm (the metric's
matching radius) of a predicted heatmap peak. It is a cheap per-epoch proxy for the real
target (edge Jaccard) that avoids running the full link pipeline every epoch. Recall drives
the score (the sweep showed the optimum is at max recall), so it is the right thing to watch.

    python scripts/train_detector.py \
        --raw /Volumes/extSSD/kaggle/biohub/raw/train \
        --gt  /Volumes/extSSD/kaggle/biohub/gt/train \
        --fold 0 --epochs 20 --out models/unet_fold0.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ctrack.data.geff import find_geff_stores, load_geff  # noqa: E402
from ctrack.data.splits import hold_out_golden, kfold  # noqa: E402
from ctrack.data.volume import ISOTROPIC_SUBSAMPLE, isotropic_scale  # noqa: E402
from ctrack.detect.dataset import DetectionFrames  # noqa: E402


def peak_recall(pred_hm: np.ndarray, gt_centres_iso: np.ndarray, spacing: float,
                threshold: float, nms_vox: int, radius_um: float = 7.0) -> tuple[int, int]:
    """(#GT points matched, #GT points). A GT point matches if a predicted peak is within
    radius_um. Matching radius = the metric's node-match radius."""
    from scipy import ndimage

    if len(gt_centres_iso) == 0:
        return 0, 0
    pooled = ndimage.maximum_filter(pred_hm, size=nms_vox, mode="nearest")
    peaks = np.argwhere((pred_hm == pooled) & (pred_hm > threshold)).astype(np.float64)
    if len(peaks) == 0:
        return 0, len(gt_centres_iso)
    d = np.linalg.norm((gt_centres_iso[:, None, :] - peaks[None, :, :]) * spacing, axis=2)
    matched = int((d.min(axis=1) <= radius_um).sum())
    return matched, len(gt_centres_iso)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from ctrack.detect.unet3d import UNet3D, masked_bce_loss

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
    ap.add_argument("--limit", type=int, help="cap datasets per split (fast smoke test)")
    ap.add_argument("--out", default="models/unet.pt")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    all_ds = sorted(p.name.replace(".geff", "") for p in find_geff_stores(args.gt))
    remaining, golden = hold_out_golden(all_ds, n=8, seed=0)
    split = kfold(remaining, k=args.k, seed=0)[args.fold]
    train_ds, val_ds = split.train, split.val
    if args.limit:
        train_ds, val_ds = train_ds[: args.limit], val_ds[: args.limit]
    print(f"fold {args.fold}: {len(train_ds)} train, {len(val_ds)} val datasets "
          f"(golden {len(golden)} held out)")

    train = DetectionFrames(train_ds, args.raw, args.gt, augment=True)
    val = DetectionFrames(val_ds, args.raw, args.gt, augment=False)
    print(f"frames: {len(train)} train, {len(val)} val")
    train_dl = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)

    model = UNet3D(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    spacing = float(isotropic_scale(subsample=ISOTROPIC_SUBSAMPLE)[0])
    nms_vox = max(1, int(round(3.0 / spacing)) | 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y, m in train_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = masked_bce_loss(model(x), y, m)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        train_loss = tot / max(1, len(train_dl))

        # validation: peak recall against held-out GT
        model.eval()
        matched = total = 0
        with torch.no_grad():
            for j in range(len(val)):
                ds, t = val.index[j]
                x, _, _ = val[j]
                hm = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
                centres_iso = val._centres[(ds, t)] / np.array(ISOTROPIC_SUBSAMPLE)
                a, b = peak_recall(hm, centres_iso, spacing, args.val_threshold, nms_vox)
                matched += a
                total += b
        recall = matched / max(1, total)
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  val peak-recall {recall:.3f}  "
              f"({matched}/{total})  {time.time() - t0:.0f}s", flush=True)

        if recall > best:
            best = recall
            torch.save({"model": model.state_dict(), "base": args.base,
                        "fold": args.fold, "recall": recall}, args.out)
            print(f"  saved {args.out} (recall {recall:.3f})", flush=True)

    print(f"best val peak-recall: {best:.3f}")


if __name__ == "__main__":
    main()
