"""Detector training loop — the single implementation shared by every entry point.

scripts/train_detector.py (local MPS), the Colab notebook, and the colab-CLI / Vast VM job all
call `train_fold`, so training stays in sync everywhere via the ctrack-src dataset. Handles
CUDA (AMP), MPS, and CPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ctrack.data.geff import find_geff_stores
from ctrack.data.splits import hold_out_golden, kfold
from ctrack.data.volume import ISOTROPIC_SUBSAMPLE, isotropic_scale


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peak_recall(pred_hm, gt_iso, spacing, nms_vox, threshold=0.5, radius_um=7.0):
    """(#GT points matched within radius_um of a predicted peak, #GT points)."""
    from scipy import ndimage

    if len(gt_iso) == 0:
        return 0, 0
    pooled = ndimage.maximum_filter(pred_hm, size=nms_vox, mode="nearest")
    peaks = np.argwhere((pred_hm == pooled) & (pred_hm > threshold)).astype(np.float64)
    if len(peaks) == 0:
        return 0, len(gt_iso)
    d = np.linalg.norm((gt_iso[:, None, :] - peaks[None, :, :]) * spacing, axis=2)
    return int((d.min(axis=1) <= radius_um).sum()), len(gt_iso)


def train_fold(
    raw_root,
    gt_root,
    *,
    fold: int = 0,
    k: int = 5,
    epochs: int = 25,
    batch: int = 8,
    base: int = 32,
    lr: float = 1e-3,
    train_stride: int = 2,
    val_stride: int = 4,
    val_threshold: float = 0.3,
    out_dir="models",
    limit: int | None = None,
    only_downloaded: bool = True,
    num_workers: int = 4,
    cache_dir=None,
) -> float:
    """Train one CV fold. Returns best held-out peak-recall; saves best weights to out_dir.

    If `cache_dir` is given, frames are read from the compact .npz cache (ctrack.detect.cache)
    instead of zarr — the persistent-across-runs path. `only_downloaded` then filters to the
    datasets whose cache exists rather than whose zarr exists.
    """
    import torch
    from torch.utils.data import DataLoader

    from ctrack.detect.unet3d import UNet3D, masked_bce_loss

    # raw_root/gt_root are None in cache mode (frames come from cache_dir), so don't Path() them.
    raw_root = Path(raw_root) if raw_root is not None else None
    gt_root = Path(gt_root) if gt_root is not None else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    is_cuda = device.type == "cuda"
    name = torch.cuda.get_device_name(0) if is_cuda else str(device)
    print(f"device: {device} ({name})", flush=True)

    # The CV split runs over the full id universe (deterministic), then filters to what's
    # actually available — so folds are identical whether reading zarr or the cache, and
    # regardless of which volumes are present this run.
    if cache_dir is not None:
        from ctrack.detect.cache import cached_all_ids

        all_ds = cached_all_ids(cache_dir) or sorted(
            p.name.replace(".npz", "") for p in Path(cache_dir).glob("*.npz"))
    else:
        all_ds = sorted(p.name.replace(".geff", "") for p in find_geff_stores(gt_root))

    # Split over the FULL universe (deterministic), THEN filter to available volumes, so the
    # fold assignment never shifts with what's present this run.
    remaining, golden = hold_out_golden(all_ds, n=8, seed=0)
    split = kfold(remaining, k=k, seed=0)[fold]
    tr, va = list(split.train), list(split.val)
    if only_downloaded:
        if cache_dir is not None:
            have = {p.name.replace(".npz", "") for p in Path(cache_dir).glob("*.npz")}
        else:
            have = {p.name.replace(".zarr", "") for p in raw_root.glob("*.zarr")}
        tr = [d for d in tr if d in have]
        va = [d for d in va if d in have]
    if limit:
        tr, va = tr[:limit], va[:limit]
    print(f"fold {fold}: {len(tr)} train, {len(va)} val datasets "
          f"(golden {len(golden)} held out)", flush=True)

    if cache_dir is not None:
        from ctrack.detect.cache import CachedDetectionFrames

        train = CachedDetectionFrames(tr, cache_dir, augment=True, frame_stride=train_stride)
        val = CachedDetectionFrames(va, cache_dir, augment=False, frame_stride=val_stride)
    else:
        from ctrack.detect.dataset import DetectionFrames

        train = DetectionFrames(tr, raw_root, gt_root, augment=True, frame_stride=train_stride)
        val = DetectionFrames(va, raw_root, gt_root, augment=False, frame_stride=val_stride)
    print(f"frames: {len(train)} train, {len(val)} val", flush=True)
    dl = DataLoader(train, batch_size=batch, shuffle=True, num_workers=num_workers,
                    drop_last=True, pin_memory=is_cuda)

    model = UNet3D(base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
    spacing = float(isotropic_scale(subsample=ISOTROPIC_SUBSAMPLE)[0])
    nms_vox = max(1, int(round(3.0 / spacing)) | 1)

    best = -1.0
    for epoch in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y, m in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=is_cuda):
                loss = masked_bce_loss(model(x), y, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
        sched.step()

        model.eval()
        matched = total = 0
        with torch.no_grad():
            for j in range(len(val)):
                ds, t = val.index[j]
                frame = val.frame_only(j)
                xx = torch.from_numpy(np.ascontiguousarray(frame))[None, None].float().to(device)
                hm = torch.sigmoid(model(xx))[0, 0].cpu().numpy()
                centres = val._centres[(ds, t)] / np.array(ISOTROPIC_SUBSAMPLE)
                a, b = peak_recall(hm, centres, spacing, nms_vox, threshold=val_threshold)
                matched += a
                total += b
        rec = matched / max(1, total)
        print(f"epoch {epoch:2d} loss {tot/max(1,len(dl)):.4f} val peak-recall {rec:.3f} "
              f"({matched}/{total}) {time.time()-t0:.0f}s", flush=True)
        if rec > best:
            best = rec
            torch.save({"model": model.state_dict(), "base": base, "fold": fold, "recall": rec},
                       out_dir / f"unet_fold{fold}.pt")
            print(f"  saved unet_fold{fold}.pt (recall {rec:.3f})", flush=True)

    print(f"DONE fold {fold} best val peak-recall {best:.3f}", flush=True)
    return best
