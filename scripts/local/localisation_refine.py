#!/usr/bin/env python3
"""Is the localisation error REACHABLE? Refine predicted centroids from the raw image.

`localisation_doseresponse.py` established the payoff curve: snapping matched nodes
to their GT position is worth +0.0305 headline / +0.0259 transferable end-to-end and
captures ~32% of the association oracle -- by far the largest capture of that oracle
ever measured (every previous attempt got 0-1%).
The response is strongly CONVEX (lambda 0.5 buys +0.0080, lambda 1.0 buys +0.0305),
so a SMALL localisation improvement buys almost nothing and only a LARGE one pays.

This asks the reachability half, entirely offline: how much of the ~2.2 um residual
can be removed using only the raw image, with no GT and no retraining?

Predicted coordinates are pure integer voxels (25 / row 47). The shipped pipeline
takes the argmax of a detection heatmap and never refines it. Two GT-free refiners
are tested here against the measured residual:

  com      intensity-weighted centre of mass in a local window
  com_thr  same, restricted to voxels above a fraction of the window peak
           (suppresses pull from neighbouring cells, which at 3-8 um diameters
           and ~11.7 um spacing are close enough to bias a plain CoM)

25 struck sub-voxel refinement off the list on the grounds that QUANTISATION is
only 5.1% of the error variance. That is true and is not the same claim: CoM does
not merely undo rounding, it re-estimates the centroid from the intensity profile,
which can also remove the systematic peak-vs-centroid offset. 25 tested the
narrower thing.

Readout is the residual to GT for matched nodes, before and after, plus the
end-to-end replay score -- never a proxy (38.5).

    python scripts/local/localisation_refine.py submissions/evalset_d4edge/submission.csv -n 6
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    MAX_MATCH_DISTANCE_UM, SCALE, Graph, evaluate, match_nodes,
    per_sample_metrics, summarise,
)
from relink_replay import relink_volume  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
RAW_ROOT = Path("/Volumes/extSSD/kaggle/biohub/raw/train")

# Window half-widths in VOXELS, ~ +-3.25 um isotropic given SCALE (1.625, .40625, .40625).
HALF = np.array([2, 8, 8])


def load_nodes(path: Path) -> dict[str, np.ndarray]:
    nodes: dict[str, list] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["row_type"] == "node":
                nodes[r["dataset"]].append(
                    (int(r["node_id"]), int(r["t"]),
                     float(r["z"]), float(r["y"]), float(r["x"])))
    return {ds: np.asarray(v, dtype=np.float64) for ds, v in nodes.items()}


def refine_frame(img: np.ndarray, zyx: np.ndarray, mode: str,
                 half: np.ndarray = HALF, frac: float = 0.5) -> np.ndarray:
    """Centre-of-mass refinement of integer-voxel peaks. Returns float voxel coords."""
    out = zyx.astype(np.float64).copy()
    zz, yy, xx = img.shape
    for k, p in enumerate(zyx):
        c = p.astype(int)
        lo = np.maximum(c - half, 0)
        hi = np.minimum(c + half + 1, [zz, yy, xx])
        w = img[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float64)
        if w.size == 0:
            continue
        peak = w.max()
        floor = np.percentile(w, 20.0)
        if mode == "com_thr":
            w = np.where(w >= floor + frac * (peak - floor), w - floor, 0.0)
        else:
            w = np.clip(w - floor, 0.0, None)
        s = w.sum()
        if s <= 0:
            continue
        g = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float64)
        out[k] = np.array([(g[a] * w).sum() / s for a in range(3)])
    return out


def process(job) -> dict:
    ds, nodes, mode, half, frac = job
    half = np.asarray(half)
    import zarr

    arr = zarr.open(str(RAW_ROOT / f"{ds}.zarr"), mode="r")["0"]
    ids = nodes[:, 0].astype(np.int64)
    t = nodes[:, 1].astype(int)
    zyx = nodes[:, 2:5]

    refined = zyx.copy()
    for frame in np.unique(t):
        sel = np.flatnonzero(t == frame)
        img = np.asarray(arr[int(frame)])
        refined[sel] = refine_frame(img, zyx[sel], mode, half, frac)

    gt = load_geff(GT_ROOT / f"{ds}.geff")
    base_g = Graph(nodes=nodes, edges=np.zeros((0, 2), dtype=np.int64))
    matched = match_nodes(gt.graph, base_g, max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
    row_of = {int(i): k for k, i in enumerate(ids)}
    gid_row = {int(g[0]): k for k, g in enumerate(gt.graph.nodes)}
    gt_zyx = gt.graph.nodes[:, 2:5]

    d_before, d_after = [], []
    for pid, gid in matched.items():
        r, gr = row_of.get(int(pid)), gid_row.get(int(gid))
        if r is None or gr is None:
            continue
        d_before.append(np.linalg.norm((zyx[r] - gt_zyx[gr]) * SCALE))
        d_after.append(np.linalg.norm((refined[r] - gt_zyx[gr]) * SCALE))

    # End-to-end: link on refined coords, score on ORIGINAL coords (35.3 discipline).
    pos = {int(i): refined[k] * SCALE for k, i in enumerate(ids)}
    t_of = {int(i): int(tt) for i, tt in zip(ids, t)}
    edges = relink_volume(pos, t_of)
    scored = Graph(nodes=nodes,
                   edges=(np.asarray(edges, dtype=np.int64) if edges
                          else np.zeros((0, 2), dtype=np.int64)))
    m = per_sample_metrics(evaluate(gt.graph, scored), gt.estimated_number_of_nodes)
    m["dataset"] = ds
    m["rms_before"] = float(np.sqrt(np.mean(np.square(d_before)))) if d_before else float("nan")
    m["rms_after"] = float(np.sqrt(np.mean(np.square(d_after)))) if d_after else float("nan")
    m["n_matched"] = len(d_before)

    pos0 = {int(i): zyx[k] * SCALE for k, i in enumerate(ids)}
    e0 = relink_volume(pos0, t_of)
    s0 = Graph(nodes=nodes, edges=(np.asarray(e0, dtype=np.int64) if e0
                                   else np.zeros((0, 2), dtype=np.int64)))
    m0 = per_sample_metrics(evaluate(gt.graph, s0), gt.estimated_number_of_nodes)
    m["base_adj"] = m0["adj_edge_jaccard"]
    m["base_edge_jaccard"] = m0["edge_jaccard"]
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("-n", type=int, default=6, help="volumes to test (0 = all)")
    ap.add_argument("--mode", default="com_thr", choices=["com", "com_thr"])
    ap.add_argument("--half", type=int, nargs=3, default=[2, 8, 8])
    ap.add_argument("--frac", type=float, default=0.5)
    args = ap.parse_args()

    subs = load_nodes(Path(args.submission))
    stems = sorted(subs)
    if args.n:
        stems = stems[::max(1, len(stems) // args.n)][:args.n]
    print(f"mode={args.mode}  volumes={len(stems)}  window=+-{args.half} voxels  frac={args.frac}\n")

    jobs = [(ds, subs[ds], args.mode, args.half, args.frac) for ds in stems]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(process, jobs))

    print(f"{'volume':>16} {'matched':>8} {'RMS before':>11} {'RMS after':>10} {'reduction':>10} "
          f"{'adj base':>9} {'adj ref':>8} {'delta':>8}")
    for m in rows:
        red = 1 - m["rms_after"] / m["rms_before"]
        print(f"{m['dataset']:>16} {m['n_matched']:>8} {m['rms_before']:>10.3f}u "
              f"{m['rms_after']:>9.3f}u {red:>9.1%} {m['base_adj']:>9.4f} "
              f"{m['adj_edge_jaccard']:>8.4f} {m['adj_edge_jaccard']-m['base_adj']:>+8.4f}")

    rb = float(np.sqrt(np.mean(np.square([m["rms_before"] for m in rows]))))
    ra = float(np.sqrt(np.mean(np.square([m["rms_after"] for m in rows]))))
    base = np.mean([m["base_adj"] for m in rows])
    ref = np.mean([m["adj_edge_jaccard"] for m in rows])
    print(f"\n  localisation RMS {rb:.3f} um -> {ra:.3f} um   ({1-ra/rb:+.1%})")
    print(f"  macro adj        {base:.4f} -> {ref:.4f}   ({ref-base:+.4f})")
    print("\n  Read against the dose-response: lambda 0.5 (a 50% error cut) was worth")
    print("  +0.0080 and lambda 1.0 (a 100% cut) +0.0305. A refiner has to move the")
    print("  RMS a long way before it clears the 0.005 readable floor.")


if __name__ == "__main__":
    main()
