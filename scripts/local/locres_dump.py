#!/usr/bin/env python3
"""STRATEGY 44.4 item 2, phase A -- dump (peak features, residual-to-GT) pairs locally.

42.4 closed the CHEAP localisation programme: readout tricks cannot express a gain
in an integer submission, and what remains of the 2.186 um residual is genuine
detector error. 40.2 measured that removing it is worth +0.0259 transferable --
the largest validated lever -- and 42.5 left "detector fine-tune" as the only route.

41.1's observation makes the bet precise: the residual survives 16-way seed+view
averaging, so it is SHARED BIAS plus ambiguity. If the bias is predictable from
local image content, a small head can learn it; if it is ambiguity, nothing can.
This dump feeds that question:

  for each detected peak matched to a GT node (<=7 um):
    features = 32-ch UNet feature vector at the peak
             + 3x5x5 local detection-logit patch (ds space)
             + peak logit
    target   = GT_um - peak_um   (3-vector, microns, fine-grid GT)

Trained/evaluated by `locres_fit.py` under the PRE-REGISTERED 40.5 gate:
matched-node RMS must fall >=30% on held-out volumes before anything goes near a
kernel. The CenterNet-style offset-at-GT-voxel head is NOT buildable here: the
detection target is one-hot at the (integer) GT voxel and downsample_z = 1, so
supervised offsets at GT voxels are identically zero in z -- the head must
instead correct the error the detector actually makes, measured at ITS peaks.

Deviation from the shipped kernel, stated: detection here uses the base repo's
4-view flip TTA and the primary seed only (the kernel averages 2 seeds x 8 views).
Residual structure should transfer; the kernel run re-measures it exactly.

    .venv/bin/python scripts/local/locres_dump.py --volumes 24
"""

from __future__ import annotations

import argparse
import json
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

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes,
)

RAW = Path("/Volumes/extSSD/kaggle/biohub/raw/train")
GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
OUT = REPO / "data" / "locres"
DET_THRESHOLD = 0.96875           # the shipped operating point
PATCH = (1, 2, 2)                 # half-extents of the logit patch (ds voxels)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=int, default=24,
                    help="how many evalset volumes to process")
    ap.add_argument("--volume-list", nargs="*", default=None)
    args = ap.parse_args()

    from predict_unet_transformer import (  # noqa: E402
        PredictConfig, _detect_cells_pooled, _load_frame, load_model,
        pool_kernel_from_um,
    )
    from biohub_tracking.io import open_dataset  # noqa: E402
    import zarr  # noqa: E402

    if args.volume_list:
        vols = args.volume_list
    else:
        evalset = json.loads((REPO / "configs" / "evalset.json").read_text())
        vols = evalset["stems"][:args.volumes]

    device = pick_device()
    weights = SUPPORT / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    model, window_size, downsample = load_model(weights, device)
    ds_arr = np.array(downsample, dtype=np.float32)
    print(f"device={device} window={window_size} downsample={downsample}", flush=True)

    cfg = PredictConfig(det_threshold=DET_THRESHOLD)
    OUT.mkdir(parents=True, exist_ok=True)

    for ds_name in vols:
        out_path = OUT / f"{ds_name}.npz"
        if out_path.exists():
            print(f"{ds_name}: exists, skip", flush=True)
            continue
        t0 = time.time()
        ds = open_dataset(RAW / f"{ds_name}.zarr", normalize=False,
                          load_image=False, downsample=downsample)
        zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
        q_low = float(ds.quantiles["0.001"])
        q_high = float(ds.quantiles["0.999"])
        T = ds.image_shape[0]
        target_shape = list(ds.image_shape[1:])
        voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
        pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)

        feats_all, patches_all, resid_all, coords_all = [], [], [], []
        W = window_size
        stride = max(W - 1, 1)
        window_starts = list(range(0, T - W + 1, stride))
        if not window_starts or window_starts[-1] + W < T:
            window_starts.append(max(T - W, 0))
        seen_frames: set[int] = set()
        peaks_by_t: dict[int, np.ndarray] = {}
        featmap_rows: list[np.ndarray] = []
        with torch.no_grad():
            for ws in window_starts:
                frame_indices = list(range(ws, ws + W))
                imgs = torch.stack([
                    _load_frame(zarr_arr, t, target_shape, downsample)
                    for t in frame_indices])
                imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
                imgs = imgs.unsqueeze(0).to(device)
                unet_out, det_logits = model.encode(imgs)
                for dims in [(-1,), (-2,), (-2, -1)]:
                    _, det_flip = model.encode(imgs.flip(dims))
                    for f in range(W):
                        det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                for f in range(W):
                    det_logits[f] = det_logits[f] / 4
                for f_idx, t in enumerate(frame_indices):
                    if t in seen_frames:
                        continue
                    seen_frames.add(t)
                    arr = _detect_cells_pooled(det_logits[f_idx][0], t,
                                               cfg.det_threshold, pool_k)
                    peaks_by_t[t] = arr
                    if len(arr) == 0:
                        continue
                    lg = det_logits[f_idx][0, 0]              # (Z, Y, X) ds space
                    fm = unet_out[0, f_idx]                   # (C, Z, Y, X)
                    pz, py, px = PATCH
                    lgp = torch.nn.functional.pad(
                        lg, (px, px, py, py, pz, pz), value=0.0)
                    for (tt, z, y, x) in arr:
                        z, y, x = int(z), int(y), int(x)
                        featmap_rows.append(np.concatenate([
                            fm[:, z, y, x].float().cpu().numpy(),
                            lgp[z:z + 2 * pz + 1, y:y + 2 * py + 1,
                                x:x + 2 * px + 1].flatten().float().cpu().numpy(),
                            [float(lg[z, y, x].cpu())],
                        ]))
                        coords_all.append([tt, z, y, x])
                del imgs, unet_out, det_logits

        coords_ds = np.asarray(coords_all, dtype=np.float32)      # (N,4) ds space
        feats = np.asarray(featmap_rows, dtype=np.float32)
        # Fine-grid coords exactly as the pipeline writes them (x ds, int16 cast).
        fine = coords_ds.copy()
        fine[:, 1:] = (coords_ds[:, 1:] * ds_arr).astype(np.int16)

        gt = load_geff(GT_ROOT / f"{ds_name}.geff")
        pred_nodes = np.concatenate([
            np.arange(len(fine), dtype=np.float64)[:, None], fine.astype(np.float64),
        ], axis=1)
        pred_g = Graph(nodes=pred_nodes, edges=np.zeros((0, 2), dtype=np.int64))
        matched = match_nodes(gt.graph, pred_g,
                              max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
        gid_row = {int(g[0]): k for k, g in enumerate(gt.graph.nodes)}
        gt_um = gt.graph.nodes[:, 2:5] * SCALE
        pred_um = fine[:, 1:] * SCALE

        rows, resids = [], []
        for pid, gid in matched.items():
            rows.append(int(pid))
            resids.append(gt_um[gid_row[int(gid)]] - pred_um[int(pid)])
        rows = np.asarray(rows, dtype=np.int64)
        resids = np.asarray(resids, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.sum(resids ** 2, axis=1)))) if len(resids) else 0.0
        np.savez_compressed(
            out_path,
            feats=feats[rows], resid=resids,
            coords_ds=coords_ds[rows], n_peaks=len(fine),
        )
        print(f"{ds_name}: peaks {len(fine):,}  matched {len(rows):,}  "
              f"RMS {rms:.3f} um  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
