#!/usr/bin/env python3
"""Dump the TEACHER's detections for noisy-student training (STRATEGY 45.2 route 1).

The pilkwang 2-seed x 8-view detector is the teacher; its peaks (at the shipped
threshold, purity 99.90% on the verifiable subset — experiments row 36) become
reduced-weight positives for our own detector (ctrack.detect.dataset pseudo_dir).

Per volume: data/teacher_peaks/{ds}.npz with coords (N,4) int16 [t, z, y, x] in
FULL-RESOLUTION voxels — the same space as GT centres, so the dataset can mix them.

Single-seed, 4-view flip TTA (the base repo path). The shipped kernel averages
2 seeds x 8 views; the student sees slightly noisier labels than the shipped
teacher would give. Accepted: peak sets differ by ~1% at this threshold.

    .venv/bin/python scripts/local/dump_teacher_peaks.py --volumes 60
"""

from __future__ import annotations

import argparse
import sys
import time
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
OUT = REPO / "data" / "teacher_peaks"
DET_THRESHOLD = 0.96875


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=int, default=60)
    ap.add_argument("--skip", nargs="*", default=None,
                    help="volume stems to exclude (e.g. the training holdout)")
    args = ap.parse_args()

    from predict_unet_transformer import (  # noqa: E402
        PredictConfig, _detect_cells_pooled, _load_frame, load_model,
        pool_kernel_from_um,
    )
    from biohub_tracking.io import open_dataset  # noqa: E402
    import zarr  # noqa: E402

    all_vols = sorted(p.stem for p in RAW.glob("*.zarr"))
    if args.skip:
        all_vols = [v for v in all_vols if v not in set(args.skip)]
    vols = all_vols[:args.volumes]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    weights = SUPPORT / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    model, window_size, downsample = load_model(weights, device)
    ds_arr = np.array(downsample, dtype=np.float32)
    cfg = PredictConfig(det_threshold=DET_THRESHOLD)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device={device} downsample={downsample} volumes={len(vols)}", flush=True)

    for ds_name in vols:
        out_path = OUT / f"{ds_name}.npz"
        if out_path.exists():
            continue
        t0 = time.time()
        ds = open_dataset(RAW / f"{ds_name}.zarr", normalize=False,
                          load_image=False, downsample=downsample)
        zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
        q_low, q_high = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
        T = ds.image_shape[0]
        target_shape = list(ds.image_shape[1:])
        voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
        pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)

        W = window_size
        stride = max(W - 1, 1)
        window_starts = list(range(0, T - W + 1, stride))
        if not window_starts or window_starts[-1] + W < T:
            window_starts.append(max(T - W, 0))
        seen: set[int] = set()
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for ws in window_starts:
                fi = list(range(ws, ws + W))
                imgs = torch.stack([_load_frame(zarr_arr, t, target_shape, downsample)
                                    for t in fi])
                imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
                imgs = imgs.unsqueeze(0).to(device)
                _, det_logits = model.encode(imgs)
                for dims in [(-1,), (-2,), (-2, -1)]:
                    _, det_flip = model.encode(imgs.flip(dims))
                    for f in range(W):
                        det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                for f in range(W):
                    det_logits[f] = det_logits[f] / 4
                for f_idx, t in enumerate(fi):
                    if t in seen:
                        continue
                    seen.add(t)
                    arr = _detect_cells_pooled(det_logits[f_idx][0], t,
                                               cfg.det_threshold, pool_k).astype(np.float32)
                    if len(arr):
                        arr[:, 1:] *= ds_arr          # ds voxels -> full-res voxels
                    rows.append(arr)
                del imgs, det_logits

        coords = (np.concatenate(rows) if rows else
                  np.empty((0, 4), dtype=np.float32)).astype(np.int16)
        np.savez_compressed(out_path, coords=coords)
        print(f"{ds_name}: {len(coords):,} peaks over {len(seen)} frames "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
