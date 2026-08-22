#!/usr/bin/env python3
"""Raw detection recall/precision of cached (coords, edges) pkls vs GT — §52.2 evidence.

Purpose: measure the MEMORISATION GAP directly. The §52.2 argument says every local
verdict inherits an instrument whose detector memorised its own volumes (row 90), so
row 87's "+0.003 detection cap" (recall 0.9958 on memorised volumes) does not bind
hidden test. This script turns that argument into a measurement: run the SAME honest
checkpoint (best_ep20) on volumes it TRAINED on vs volumes it never saw, same config,
and read the recall gap. The public detector cannot be given this test (it trained on
every train volume) — ep20 is the same recipe/architecture, so its gap is the estimate.

Matching = the metric's own rule (<=7.0 um, anisotropy-scaled, Hungarian per frame),
functions lifted verbatim from ladder_oracle.py.

    .venv/bin/python scripts/local/detection_recall_cache.py <cache_dir> [<cache_dir> ...]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from ctrack.data.geff import load_geff  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
VOXEL_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
MATCH_UM = 7.0


def match_frame(pred_zyx, gt_zyx):
    if len(pred_zyx) == 0 or len(gt_zyx) == 0:
        return []
    d = np.linalg.norm(pred_zyx[:, None, :] * VOXEL_UM - gt_zyx[None, :, :] * VOXEL_UM, axis=-1)
    big = MATCH_UM * 1e3
    cost = np.where(d <= MATCH_UM, d, big)
    ri, ci = linear_sum_assignment(cost)
    return [(int(a), int(b)) for a, b in zip(ri, ci) if cost[a, b] < big]


def main() -> None:
    for cache_dir in sys.argv[1:]:
        cache = Path(cache_dir)
        print(f"\n== {cache} ==")
        print(f"{'volume':<16}{'n_pred':>8}{'n_gt':>8}{'matched':>9}{'recall':>8}{'precision':>10}")
        tot_p = tot_g = tot_m = 0
        for f in sorted(cache.glob("*.pkl")):
            coords, _edges = pickle.loads(f.read_bytes())
            coords = np.asarray(coords)
            gt = load_geff(GT_ROOT / f"{f.stem}.geff")
            gnodes = gt.graph.nodes.astype(np.float64)
            assert gnodes.shape[1] == 5, f"geff nodes must be [id,t,z,y,x], got {gnodes.shape}"
            gt_t, gt_zyx = gnodes[:, 1].astype(int), gnodes[:, 2:].astype(np.float64)
            m = 0
            for t in np.unique(gt_t):
                gsel = np.flatnonzero(gt_t == t)
                psel = np.flatnonzero(coords[:, 0] == t)
                if len(gsel) and len(psel):
                    m += len(match_frame(coords[psel, 1:].astype(np.float64), gt_zyx[gsel]))
            n_p, n_g = len(coords), len(gnodes)
            tot_p += n_p; tot_g += n_g; tot_m += m
            print(f"{f.stem:<16}{n_p:>8,}{n_g:>8,}{m:>9,}{m/n_g:>8.4f}{m/n_p:>10.4f}")
        if tot_g:
            print(f"{'TOTAL':<16}{tot_p:>8,}{tot_g:>8,}{tot_m:>9,}{tot_m/tot_g:>8.4f}{tot_m/tot_p:>10.4f}")


if __name__ == "__main__":
    main()
