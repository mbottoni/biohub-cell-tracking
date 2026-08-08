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
    """(#GT points matched within radius_um of a predicted peak, #GT points, #predicted peaks).

    The 3rd return value (peak count) is what lets validation see over-detection: peak-recall
    alone rewards firing everywhere, but the competition metric penalizes total node count via
    `total_node_ratio`. See train_fold's node-ratio-aware selection.
    """
    from scipy import ndimage

    pooled = ndimage.maximum_filter(pred_hm, size=nms_vox, mode="nearest")
    peaks = np.argwhere((pred_hm == pooled) & (pred_hm > threshold)).astype(np.float64)
    n_peaks = len(peaks)
    if len(gt_iso) == 0:
        return 0, 0, n_peaks
    if n_peaks == 0:
        return 0, len(gt_iso), 0
    d = np.linalg.norm((gt_iso[:, None, :] - peaks[None, :, :]) * spacing, axis=2)
    return int((d.min(axis=1) <= radius_um).sum()), len(gt_iso), n_peaks


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
    val_threshold: float = 0.5,
    pos_weight: float = 4.0,
    offset_weight: float = 0.0,
    out_dir="models",
    limit: int | None = None,
    only_downloaded: bool = True,
    num_workers: int = 4,
    cache_dir=None,
    log_every: int = 20,
    max_minutes: float | None = None,
    calibrate_only: bool = False,
    early_stop_patience: int | None = None,
    on_best=None,
    resume: str | None = "auto",
    pseudo_dir=None,
    pseudo_weight: float = 0.3,
    seed: int = 0,
    iso_mode: str = "stride",
) -> float:
    """Train one CV fold. Returns best held-out sel-score (recall x node-ratio penalty, the
    competition-metric-aligned selection); saves best weights to out_dir.

    If `cache_dir` is given, frames are read from the compact .npz cache (ctrack.detect.cache)
    instead of zarr — the persistent-across-runs path. `only_downloaded` then filters to the
    datasets whose cache exists rather than whose zarr exists.
    """
    import torch
    from torch.utils.data import DataLoader

    from ctrack.detect.unet3d import UNet3D, masked_bce_loss, masked_offset_loss

    use_offset = offset_weight > 0

    # Seeded so two concurrent T4 sessions give genuinely different ensemble members,
    # and a resumed run re-derives the same augmentation stream family.
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"seed={seed}  iso_mode={iso_mode}", flush=True)

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
        if use_offset:
            raise ValueError("offset_weight>0 needs the geff (parent links); not supported in "
                             "cache mode. Train the offset head from zarr/geff, not the cache.")
        from ctrack.detect.cache import CachedDetectionFrames

        train = CachedDetectionFrames(tr, cache_dir, augment=True, frame_stride=train_stride)
        val = CachedDetectionFrames(va, cache_dir, augment=False, frame_stride=val_stride)
    else:
        from ctrack.detect.dataset import DetectionFrames

        train = DetectionFrames(tr, raw_root, gt_root, augment=True, frame_stride=train_stride,
                                with_offset=use_offset, pseudo_dir=pseudo_dir,
                                pseudo_weight=pseudo_weight, iso_mode=iso_mode)
        val = DetectionFrames(va, raw_root, gt_root, augment=False, frame_stride=val_stride,
                              iso_mode=iso_mode)
    print(f"frames: {len(train)} train, {len(val)} val", flush=True)
    dl = DataLoader(train, batch_size=batch, shuffle=True, num_workers=num_workers,
                    drop_last=True, pin_memory=is_cuda)

    model = UNet3D(base=base, out_offset=use_offset).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if use_offset:
        print(f"offset head ON (offset_weight={offset_weight}) — motion-aware linking", flush=True)
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
        for step, batch in enumerate(dl):
            x, y, m = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=is_cuda):
                loss = masked_bce_loss(model(x), y, m, pos_weight=pos_weight)
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

    # Per-volume estimated TRUE cell count -> lets validation penalize over-detection the way the
    # competition's total_node_ratio does (predicting ~all cells, not just the sparse annotations).
    # Unavailable in cache mode (no gt_root); node-ratio selection then falls back to recall only.
    est_by_ds: dict[str, float] = {}
    if gt_root is not None:
        from ctrack.data.geff import load_geff

        for ds in va:
            try:
                g = load_geff(gt_root / f"{ds}.geff")
                e = float(getattr(g, "estimated_number_of_nodes", 0) or 0)
                if e > 0:
                    est_by_ds[ds] = e
            except Exception:  # noqa: BLE001
                pass
    T_FRAMES = 100  # every volume is 100 timepoints (dataset spec) -> target cells/frame = est/100
    node_sel = len(est_by_ds) > 0
    sel_desc = "recall x node-ratio-penalty" if node_sel else "recall only (no estimated counts)"
    print(f"selection metric: {sel_desc}  ({len(est_by_ds)}/{len(va)} val vols have est counts)",
          flush=True)

    # ---- resumable checkpointing -------------------------------------------------------
    # `checkpoint_last.pt` carries everything a killed session needs to continue exactly:
    # model/optimizer/scheduler/scaler state, epoch counter, best sel + patience counter,
    # and the torch RNG. Written ATOMICALLY (tmp -> rename) after every epoch, because a
    # Kaggle session can die mid-write at the 12 h wall. resume="auto" picks it up if
    # present; a fresh run just starts at epoch 0.
    last_path = out_dir / f"checkpoint_last_fold{fold}.pt"
    start_epoch, best, since_best = 0, -1.0, 0
    if resume:
        rp = last_path if resume == "auto" else Path(resume)
        if rp.exists():
            ck = torch.load(rp, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            assert not missing and not unexpected, \
                f"resume mismatch: missing={len(missing)} unexpected={len(unexpected)}"
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            if is_cuda and ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            torch.set_rng_state(ck["torch_rng"].cpu())
            start_epoch = int(ck["epoch"]) + 1
            best = float(ck["best"])
            since_best = int(ck["since_best"])
            print(f"RESUMED from {rp.name}: epoch {start_epoch} (best sel {best:.3f}, "
                  f"{since_best} epochs since best, {len(ck['model'])} tensors matched)",
                  flush=True)
        elif resume != "auto":
            raise FileNotFoundError(f"--resume {rp} does not exist")

    def save_last(epoch: int) -> None:
        tmp = last_path.with_suffix(".tmp")
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if is_cuda else None,
            "torch_rng": torch.get_rng_state(),
            "epoch": epoch, "best": best, "since_best": since_best,
            "base": base, "fold": fold, "out_offset": use_offset,
            "seed": seed, "iso_mode": iso_mode,
        }, tmp)
        tmp.replace(last_path)

    run_t0 = time.time()
    est_epoch_s = None
    for epoch in range(start_epoch, epochs):
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
        for step, batch in enumerate(dl):
            x, y, m = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=is_cuda):
                if use_offset:
                    off, offm = batch[3].to(device), batch[4].to(device)
                    hm, off_pred = model(x, return_offset=True)
                    loss = (masked_bce_loss(hm, y, m, pos_weight=pos_weight)
                            + offset_weight * masked_offset_loss(off_pred, off, offm))
                else:
                    loss = masked_bce_loss(model(x), y, m, pos_weight=pos_weight)
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

        # ---- validation: peak recall AND predicted count vs the estimated true count ----
        # sel_score mirrors the competition's adj term: recall x (1 - 0.1*|total_node_ratio|).
        # This stops "best" from selecting the saturated, over-detecting checkpoint that maximises
        # bare recall by firing everywhere.
        # THRESHOLD-SWEPT selection (row 42's lesson): a fixed 0.5 threshold conflates
        # CALIBRATION with quality — the s0 run early-stopped at epoch 0 because peak
        # counts at 0.5 exploded while recall climbed to 99%. The pipeline runs its
        # detector at a tuned threshold, so selection must too: sel = max over a
        # threshold grid of recall x exp(-0.1|ratio|), and the chosen threshold is
        # reported so integration starts from the right operating point.
        VAL_THRS = sorted({val_threshold, 0.5, 0.9, 0.98, 0.995})
        val_t0 = time.time()
        model.eval()
        agg = {thr: [0, 0, 0.0] for thr in VAL_THRS}   # matched, total, n_peaks
        sum_target = 0.0
        with torch.no_grad():
            for j in range(len(val)):
                ds, t = val.index[j]
                frame = val.frame_only(j)
                xx = torch.from_numpy(np.ascontiguousarray(frame))[None, None].float().to(device)
                hm = torch.sigmoid(model(xx))[0, 0].cpu().numpy()
                centres = val._centres[(ds, t)] / np.array(ISOTROPIC_SUBSAMPLE)
                counted = ds in est_by_ds
                if counted:
                    sum_target += est_by_ds[ds] / T_FRAMES
                for thr in VAL_THRS:
                    a, b, n_peaks = peak_recall(hm, centres, spacing, nms_vox, threshold=thr)
                    agg[thr][0] += a
                    agg[thr][1] += b
                    if counted:
                        agg[thr][2] += n_peaks
        sel, rec, node_ratio, sel_thr, sum_pred = -1.0, 0.0, float("nan"), VAL_THRS[0], 0.0
        for thr in VAL_THRS:
            m_, t_, p_ = agg[thr]
            r_ = m_ / max(1, t_)
            if node_sel and sum_target > 0:
                nr_ = (p_ - sum_target) / sum_target
                # Smooth, always-positive penalty (not the competition's hard clamp, which
                # zeroes every over-detecting epoch and destroys ranking / trips early-stop).
                s_ = r_ * float(np.exp(-0.1 * abs(nr_)))
            else:
                nr_, s_ = float("nan"), r_
            if s_ > sel:
                sel, rec, node_ratio, sel_thr, sum_pred = s_, r_, nr_, thr, p_
        val_s = time.time() - val_t0
        run_min = (time.time() - run_t0) / 60
        remaining = epochs - epoch - 1
        tag = ""
        ckpt_path = out_dir / f"unet_fold{fold}.pt"
        if sel > best:
            best = sel
            since_best = 0
            torch.save({"model": model.state_dict(), "base": base, "fold": fold, "iso_mode": iso_mode,
                        "out_offset": use_offset, "recall": rec, "sel_score": sel,
                        "node_ratio": node_ratio, "epoch": epoch}, ckpt_path)
            tag = "  <- saved (best)"
            if on_best is not None:
                # hook: e.g. upload the checkpoint so it's accessible mid-run and survives a kill
                try:
                    on_best(ckpt_path, epoch, sel)
                except Exception as e:  # noqa: BLE001
                    print(f"  on_best hook failed (non-fatal): {str(e)[:120]}", flush=True)
        else:
            since_best += 1
        print(f"EPOCH {epoch:2d}/{epochs-1}  loss {tot/max(1,steps_per_epoch):.4f}  "
              f"sel {sel:.3f}  recall {rec:.3f} ({matched}/{total})  "
              f"node_ratio {node_ratio:+.2f} (pred {sum_pred:.0f}/tgt {sum_target:.0f})  "
              f"lr {sched.get_last_lr()[0]:.2e}  train {train_s:.0f}s val {val_s:.0f}s  "
              f"total {run_min:.1f}min  ~{remaining*(train_s+val_s)/60:.0f}min left{tag}", flush=True)

        save_last(epoch)

        if early_stop_patience and since_best >= early_stop_patience:
            print(f"EARLY STOP: no sel-score improvement in {early_stop_patience} epochs "
                  f"(best {best:.3f} @ epoch {epoch - since_best}). Trained {epoch+1} epochs.",
                  flush=True)
            break

    print(f"DONE fold {fold} best val sel-score {best:.3f} in {(time.time()-run_t0)/60:.1f} min",
          flush=True)
    return best
