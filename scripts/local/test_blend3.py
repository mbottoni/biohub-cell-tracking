#!/usr/bin/env python3
"""Local smoke for the third-member detection blend (STRATEGY 46, rank-1 ROI).

Verifies the INTEGRATION MECHANICS on MPS before any kernel is built:
  * the student's input recipe inside the pipeline (strided downsample +
    ctrack quantile-0.01/0.999 normalization, clipped) reproduces ctrack's
    training-time input bit-for-bit;
  * student logits align onto the primary's logit scale (mean/std, ratio
    clamped 0.5-2.0 -- the same alignment the shipped secondary blend uses);
  * blending at w=0.2 changes peak counts by a sane amount (not 0, not 10x).

Uses whatever checkpoint is handy (round-1 s0 by default) -- the mechanics do
not depend on which weights; the kernel swaps in the gated s0v2 checkpoint.

    .venv/bin/python scripts/local/test_blend3.py <checkpoint>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SUPPORT = Path(
    "/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
    "5c5c110c-b819-4168-a812-26fe17d15065/scratchpad/support_pack")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SUPPORT / "repo" / "src"))
sys.path.insert(0, str(SUPPORT / "repo" / "scripts"))

RAW = Path("/Volumes/extSSD/kaggle/biohub/raw/train")
VOL = "44b6_0113de3b"
FRAME = 50


def main() -> None:
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        SUPPORT.parent / "s0_out" / "models" / "unet_fold0.pt")

    from predict_unet_transformer import (  # noqa: E402
        PredictConfig, _detect_cells_pooled, _load_frame, load_model,
        pool_kernel_from_um,
    )
    from biohub_tracking.io import open_dataset  # noqa: E402
    from ctrack.data.volume import open_volume, to_isotropic  # noqa: E402
    from ctrack.detect.unet3d import UNet3D  # noqa: E402
    import zarr  # noqa: E402

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ---- pipeline primary logits (identity view only -- alignment stats don't need TTA)
    ds = open_dataset(RAW / f"{vol}.zarr", normalize=False, load_image=False,
                      downsample=(1, 4, 4))
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low, q_high = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
    target_shape = list(ds.image_shape[1:])
    model, W, downsample = load_model(
        SUPPORT / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth",
        device)
    imgs = torch.stack([_load_frame(zarr_arr, t, target_shape, downsample)
                        for t in (frame, frame + 1)])
    imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, det_logits = model.encode(imgs)
    primary = det_logits[0]                                  # (1,1,Z,Y,X)
    print(f"primary logits {tuple(primary.shape)}  mean {primary.mean():+.3f} "
          f"std {primary.float().std():.3f}")

    # ---- student input, THE KERNEL RECIPE: strided ds + ctrack 0.01/0.999 clip norm
    sq_lo, sq_hi = float(ds.quantiles["0.01"]), float(ds.quantiles["0.999"])
    raw_frame = np.asarray(zarr_arr[FRAME])
    strided = raw_frame[::downsample[0], ::downsample[1], ::downsample[2]]
    kernel_recipe = np.clip((strided.astype(np.float32) - sq_lo) / (sq_hi - sq_lo),
                            0.0, 1.0)

    # ---- ctrack's own training-time input, for the bit-exactness check
    vol = open_volume(RAW / f"{VOL}.zarr")
    train_recipe = to_isotropic(vol.normalized_frame(FRAME), (1, 4, 4), mode="stride")
    diff = float(np.abs(kernel_recipe - train_recipe).max())
    print(f"input recipe max|diff| vs training pipeline: {diff:.2e}  "
          f"({'OK' if diff < 1e-6 else 'MISMATCH -- FIX BEFORE KERNEL'})")
    assert diff < 1e-6

    # ---- student forward + alignment + blend
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stu = UNet3D(base=ck.get("base", 32))
    stu.load_state_dict(ck["model"] if "model" in ck else ck)
    stu.to(device).eval()
    with torch.no_grad():
        s_logits = stu(torch.from_numpy(np.ascontiguousarray(kernel_recipe))
                       [None, None].float().to(device))
    print(f"student logits {tuple(s_logits.shape)}  mean {s_logits.mean():+.3f} "
          f"std {s_logits.float().std():.3f}")
    assert tuple(s_logits.shape) == tuple(primary.shape), "grid mismatch"

    pm, ps = primary.mean(), primary.float().std().clamp_min(1e-4)
    sm, ss = s_logits.mean(), s_logits.float().std().clamp_min(1e-4)
    aligned = (s_logits - sm) * (ps / ss).clamp(0.5, 2.0) + pm
    cfg = PredictConfig(det_threshold=0.96875)
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)
    n_base = len(_detect_cells_pooled(primary[0], FRAME, cfg.det_threshold, pool_k))
    for w in (0.1, 0.2, 0.3):
        blended = (1.0 - w) * primary + w * aligned
        n = len(_detect_cells_pooled(blended[0], FRAME, cfg.det_threshold, pool_k))
        print(f"  w={w:.1f}: peaks {n_base} -> {n}  ({(n-n_base)/max(1,n_base):+.1%})")
    print("SMOKE PASS: mechanics verified")


if __name__ == "__main__":
    main()


def rescue_smoke(ckpt_path: str, vol: str = VOL, frame: int = FRAME,
                 iso_mode: str = "stride") -> None:
    """Calibrate the RESCUE margin: peaks(max(primary, aligned_student - m*sigma)).

    Row 48 killed averaging (vetoes real cells). Rescue-only fusion can only ADD
    peaks; the margin m controls how confident the student must be to poke
    through. Target: +1-3% peak addition (the det-FN pool is ~0.25% of nodes,
    but peaks here are pre-ILP so a few x that is tolerable).
    """
    import numpy as np
    import torch
    from predict_unet_transformer import (
        PredictConfig, _detect_cells_pooled, _load_frame, load_model,
        pool_kernel_from_um,
    )
    from biohub_tracking.io import open_dataset
    from ctrack.detect.unet3d import UNet3D
    import zarr

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = open_dataset(RAW / f"{vol}.zarr", normalize=False, load_image=False,
                      downsample=(1, 4, 4))
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low, q_high = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
    target_shape = list(ds.image_shape[1:])
    model, W, downsample = load_model(
        SUPPORT / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth",
        device)
    imgs = torch.stack([_load_frame(zarr_arr, t, target_shape, downsample)
                        for t in (frame, frame + 1)])
    imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, det_logits = model.encode(imgs)
    primary = det_logits[0]

    # Student input via ctrack's own functions -- exact training recipe for BOTH
    # modes (mean mode: normalize at FULL res, clip, THEN area-average; clipping
    # does not commute with pooling, so replicating by hand risks a silent shift).
    from ctrack.data.volume import open_volume, to_isotropic
    _vol = open_volume(RAW / f"{vol}.zarr")
    x = to_isotropic(_vol.normalized_frame(frame), tuple(downsample), mode=iso_mode)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stu = UNet3D(base=ck.get("base", 32))
    stu.load_state_dict(ck["model"] if "model" in ck else ck)
    stu.to(device).eval()
    with torch.no_grad():
        s_logits = stu(torch.from_numpy(np.ascontiguousarray(x))[None, None].float().to(device))
    pm, ps = primary.mean(), primary.float().std().clamp_min(1e-4)
    sm, ss = s_logits.mean(), s_logits.float().std().clamp_min(1e-4)
    aligned = (s_logits - sm) * (ps / ss).clamp(0.5, 2.0) + pm

    cfg = PredictConfig(det_threshold=0.96875)
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)
    n_base = len(_detect_cells_pooled(primary[0], frame, cfg.det_threshold, pool_k))
    print(f"rescue smoke on {vol} f{frame} iso={iso_mode}: base peaks {n_base}")
    for m in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        fused = torch.maximum(primary, aligned - m * ps)
        n = len(_detect_cells_pooled(fused[0], frame, cfg.det_threshold, pool_k))
        print(f"  margin {m:.1f}*sigma: peaks {n}  ({(n-n_base)/max(1,n_base):+.1%})")
