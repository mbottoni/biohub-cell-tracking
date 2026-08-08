#!/usr/bin/env python3
"""Golden-8 gates for the trained student detector (STRATEGY 45.2 route 1, row 42).

The T4 run early-stopped on sel@0.5 while recall climbed to 99% — which conflates
CALIBRATION with quality: the pipeline runs its detector at a tuned threshold, not
0.5. This script renders the honest verdict: a threshold sweep on the golden 8
(held out from BOTH our training and our teacher-peak supervision -- though NOT
from the teacher's own 400-epoch training, so the teacher reference is inflated).

Per (checkpoint, threshold): annotated-cell recall (<=7 um), node-count ratio vs
estimated_number_of_nodes, matched-node RMS. Teacher reference: the dumped peaks
(single-seed 4-view, threshold 0.96875) on the same volumes.

GATES (pre-registered, §44.4 item 2 analogue):
  recall  >= teacher's at a threshold where |node_ratio| <= teacher's
  RMS     not worse than teacher +0.1 um

    .venv/bin/python scripts/local/student_gate.py <models_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ctrack.data.geff import find_geff_stores, load_geff  # noqa: E402
from ctrack.data.splits import hold_out_golden  # noqa: E402
from ctrack.data.volume import (  # noqa: E402
    ISOTROPIC_SUBSAMPLE, isotropic_scale, open_volume, to_isotropic,
)

RAW = Path("/Volumes/extSSD/kaggle/biohub/raw/train")
GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
PEAKS = REPO / "data" / "teacher_peaks"
SCALE = np.array([1.625, 0.40625, 0.40625])
THRESHOLDS = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995]


def peak_stats(peaks_full_vox: np.ndarray, gt_zyx: np.ndarray, est: float):
    """(recall, ratio, matched RMS) for one volume's full peak set vs its GT points."""
    if len(peaks_full_vox) == 0:
        return 0.0, -1.0, float("nan")
    d = np.linalg.norm(
        (gt_zyx[:, None, :] - peaks_full_vox[None, :, :]) * SCALE, axis=2)
    mind = d.min(axis=1)
    matched = mind <= 7.0
    recall = float(matched.mean()) if len(gt_zyx) else 0.0
    rms = float(np.sqrt(np.mean(mind[matched] ** 2))) if matched.any() else float("nan")
    ratio = (len(peaks_full_vox) - est) / est if est > 0 else float("nan")
    return recall, ratio, rms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models_dir")
    ap.add_argument("--ckpt", nargs="*", default=["unet_fold0.pt", "checkpoint_last_fold0.pt"])
    ap.add_argument("--iso-mode", default="stride")
    ap.add_argument("--thresholds", type=float, nargs="*", default=None)
    args = ap.parse_args()

    global THRESHOLDS
    import torch
    from scipy import ndimage
    from ctrack.detect.unet3d import UNet3D

    if args.thresholds:
        THRESHOLDS = args.thresholds
    all_ds = sorted(p.name.replace(".geff", "") for p in find_geff_stores(GT_ROOT))
    _, golden = hold_out_golden(all_ds, n=8, seed=0)
    print(f"golden-8: {golden}")

    spacing = isotropic_scale(subsample=ISOTROPIC_SUBSAMPLE)
    nms_vox = max(1, int(round(3.0 / spacing[0])) | 1)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sub = np.array(ISOTROPIC_SUBSAMPLE, dtype=np.float64)

    gt_data = {}
    for ds in golden:
        g = load_geff(GT_ROOT / f"{ds}.geff")
        t = g.graph.t.astype(int)
        gt_data[ds] = {
            "by_t": {int(f): g.graph.zyx[t == f] for f in np.unique(t)},
            "est": float(getattr(g, "estimated_number_of_nodes", 0) or 0),
        }

    # ---- teacher reference (dumped peaks, thr 0.96875)
    print(f"\n{'model':>28} {'thr':>6} {'recall':>7} {'ratio':>7} {'RMS':>6}")
    rec_n = rec_d = 0
    rat_p = rat_e = 0.0
    rms_acc = []
    for ds in golden:
        pk = np.load(PEAKS / f"{ds}.npz")["coords"].astype(np.float64)
        gd = gt_data[ds]
        for f, gz in gd["by_t"].items():
            if not len(gz):
                continue
            fp = pk[pk[:, 0] == f, 1:4]
            r, _, m = peak_stats(fp, gz, 0)
            rec_n += r * len(gz)
            rec_d += len(gz)
            if np.isfinite(m):
                rms_acc.append(m)
        rat_p += len(pk)
        rat_e += gd["est"]
    print(f"{'TEACHER (dump, 0.96875)':>28} {'':>6} {rec_n/max(1,rec_d):>7.3f} "
          f"{(rat_p-rat_e)/rat_e:>+7.2f} {np.mean(rms_acc):>6.3f}")

    # ---- student checkpoints x thresholds
    for ck_name in args.ckpt:
        ck_path = Path(args.models_dir) / ck_name
        if not ck_path.exists():
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        state = ck["model"] if "model" in ck else ck
        base = ck.get("base", 32)
        model = UNet3D(base=base)
        model.load_state_dict(state)
        model.to(device).eval()
        tag = f"{ck_name} (e{ck.get('epoch', '?')})"

        # per-threshold accumulators
        acc = {thr: [0.0, 0, 0.0, 0.0, []] for thr in THRESHOLDS}  # recn, recd, np, ne, rms
        with torch.no_grad():
            for ds in golden:
                vol = open_volume(RAW / f"{ds}.zarr")
                gd = gt_data[ds]
                T = vol.n_frames
                per_frame_est = gd["est"] / T
                for f in range(T):
                    frame = to_isotropic(vol.normalized_frame(f), ISOTROPIC_SUBSAMPLE,
                                         mode=args.iso_mode)
                    x = torch.from_numpy(np.ascontiguousarray(frame))[None, None].float().to(device)
                    hm = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                    pooled = ndimage.maximum_filter(hm, size=nms_vox, mode="nearest")
                    is_peak = hm == pooled
                    gz = gd["by_t"].get(f, np.zeros((0, 3)))
                    for thr in THRESHOLDS:
                        pk_iso = np.argwhere(is_peak & (hm > thr)).astype(np.float64)
                        pk_full = pk_iso * sub
                        a = acc[thr]
                        a[2] += len(pk_full)
                        a[3] += per_frame_est
                        if len(gz):
                            r, _, m = peak_stats(pk_full, gz, 0)
                            a[0] += r * len(gz)
                            a[1] += len(gz)
                            if np.isfinite(m):
                                a[4].append(m)
        for thr in THRESHOLDS:
            recn, recd, npk, ne, rmss = acc[thr]
            print(f"{tag:>28} {thr:>6.3f} {recn/max(1,recd):>7.3f} "
                  f"{(npk-ne)/max(1,ne):>+7.2f} {np.mean(rmss) if rmss else float('nan'):>6.3f}")

    print("\nGATES: recall >= teacher at |ratio| <= teacher's, RMS <= teacher +0.1.")


if __name__ == "__main__":
    main()
