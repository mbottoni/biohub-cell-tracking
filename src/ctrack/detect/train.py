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
    log_every: int = 20,
    max_minutes: float | None = None,
    calibrate_only: bool = False,
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
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
    spacing = float(isotropic_scale(subsample=ISOTROPIC_SUBSAMPLE)[0])
    nms_vox = max(1, int(round(3.0 / spacing)) | 1)
    steps_per_epoch = len(dl)
    print(f"model: UNet3D base={base}  {n_params/1e6:.2f}M params  "
          f"| batch={batch}  steps/epoch={steps_per_epoch}  val frames={len(val)}", flush=True)
    if max_minutes:
        print(f"time budget: {max_minutes:.0f} min (will stop before exceeding it)", flush=True)

    def _mem() -> str:
        if is_cuda:
            return f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
        if device.type == "mps":
            return f"{torch.mps.current_allocated_memory()/1e9:.1f}GB"
        return "-"

    # ---- calibration: time a few steps, print the ETA, and (optionally) exit ----
    if calibrate_only:
        model.train()
        # MPS compiles kernels over the first ~20 steps; time only AFTER that warmup, or the
        # estimate runs ~2x slow. warmup=20, then measure the next `measure` steps.
        warmup = min(20, max(0, steps_per_epoch - 4))
        measure = min(12, steps_per_epoch - warmup) or 1
        c0 = time.time()
        for step, (x, y, m) in enumerate(dl):
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=is_cuda):
                loss = masked_bce_loss(model(x), y, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step == warmup - 1:   # start the clock once warm
                c0 = time.time()
            if step >= warmup + measure - 1:
                break
        per_step = (time.time() - c0) / max(1, measure)
        ep_train_s = per_step * steps_per_epoch
        # val is one forward per frame (no backward) ≈ 0.7× a per-frame train step (measured)
        ep_val_s = len(val) * (per_step / max(1, batch)) * 0.7
        ep_s = ep_train_s + ep_val_s
        # The measured rate is data-starved (DataLoader prefetch not yet full) and runs ~1.5×
        # slow vs steady state; treat these as UPPER bounds. The --max-minutes guard, which uses
        # the real epoch-0 time, is the actual safety.
        print("\n=== CALIBRATION (upper bound; real ~1.5x faster once the loader warms) ===",
              flush=True)
        print(f"  {per_step:.2f} s/step  ({batch/per_step:.1f} frames/s)  mem {_mem()}", flush=True)
        print(f"  <= {ep_s/60:.1f} min/epoch  (train {ep_train_s/60:.1f} + val {ep_val_s/60:.1f})"
              f"  -> steady ~{ep_s/60/1.5:.1f} min/epoch", flush=True)
        print(f"  steady fits in 55 min: ~{int(55*60/(ep_s/1.5))} epochs  |  requested {epochs} "
              f"~{ep_s/1.5*epochs/60:.0f} min. Rely on --max-minutes to bound it.", flush=True)
        return float("nan")

    run_t0 = time.time()
    best, est_epoch_s = -1.0, None
    for epoch in range(epochs):
        # ---- budget guard: stop before starting an epoch we can't finish in time ----
        if max_minutes and est_epoch_s is not None:
            elapsed_min = (time.time() - run_t0) / 60
            if elapsed_min + est_epoch_s / 60 > max_minutes:
                print(f"STOP: {elapsed_min:.1f} min elapsed, next epoch ~{est_epoch_s/60:.1f} min "
                      f"would exceed the {max_minutes:.0f} min budget. Trained {epoch} epochs.",
                      flush=True)
                break

        model.train()
        ep_t0, tot, seen = time.time(), 0.0, 0
        for step, (x, y, m) in enumerate(dl):
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=is_cuda):
                loss = masked_bce_loss(model(x), y, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            seen += x.shape[0]

            # ---- per-step heartbeat + ETA (calibrates on the fly) ----
            if step == 0 or (step + 1) % log_every == 0 or step == steps_per_epoch - 1:
                el = time.time() - ep_t0
                sps = (step + 1) / el
                eta_ep = (steps_per_epoch - step - 1) / max(sps, 1e-9)
                print(f"  e{epoch:02d} step {step+1:4d}/{steps_per_epoch}  "
                      f"loss {tot/(step+1):.4f}  {seen/el:.1f} frames/s  {sps:.2f} it/s  "
                      f"mem {_mem()}  ep-eta {eta_ep:5.0f}s", flush=True)
                # first full epoch gives the estimate for the budget guard
                if epoch == 0 and step >= min(9, steps_per_epoch - 1) and est_epoch_s is None:
                    est_epoch_s = steps_per_epoch / max(sps, 1e-9)
                    tot_min = est_epoch_s * epochs / 60
                    print(f"  >>> ETA: ~{est_epoch_s/60:.1f} min/epoch, "
                          f"~{tot_min:.0f} min for {epochs} epochs "
                          f"{'(OVER BUDGET — will stop early)' if max_minutes and tot_min > max_minutes else ''}",
                          flush=True)
        sched.step()
        train_s = time.time() - ep_t0
        est_epoch_s = train_s  # refine estimate each epoch

        # ---- validation: held-out peak recall ----
        val_t0 = time.time()
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
        val_s = time.time() - val_t0
        run_min = (time.time() - run_t0) / 60
        remaining = epochs - epoch - 1
        tag = ""
        if rec > best:
            best = rec
            torch.save({"model": model.state_dict(), "base": base, "fold": fold, "recall": rec},
                       out_dir / f"unet_fold{fold}.pt")
            tag = "  <- saved (best)"
        print(f"EPOCH {epoch:2d}/{epochs-1}  loss {tot/max(1,steps_per_epoch):.4f}  "
              f"val-recall {rec:.3f} ({matched}/{total})  lr {sched.get_last_lr()[0]:.2e}  "
              f"train {train_s:.0f}s val {val_s:.0f}s  total {run_min:.1f}min  "
              f"~{remaining*(train_s+val_s)/60:.0f}min left{tag}", flush=True)

    print(f"DONE fold {fold} best val peak-recall {best:.3f} in {(time.time()-run_t0)/60:.1f} min",
          flush=True)
    return best
