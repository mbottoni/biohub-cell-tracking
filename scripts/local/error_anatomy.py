#!/usr/bin/env python3
"""STRATEGY 32 -- the anatomy of our association errors. Three open questions.

Everything we know says association is 85% of the remaining loss (§27) and that
every lever failed on ADMISSIBILITY rather than ranking (§21.1, §24.3, §28,
§31.2). What we still cannot answer is *which edges* we lose and *why they look
the way they do*. Three questions, all answerable from files on disk.

Q1. WHERE IS THE SPEED CLIFF, AND HOW MUCH SCORE SITS BEHIND IT?
    P(we link it correctly | GT displacement). §31.2 showed swapped edges have a
    median GT displacement of 4.82 um against a corpus median of 1.82. If recall
    collapses past some speed, the total score locked behind that cliff is
    computable exactly -- and it is the hard ceiling on what `tight9`/`tight12`
    can possibly return.

Q2. IS OUR LOCALISATION ERROR ALIGNED WITH MOTION? (a motion-blur test)
    §31.3 found predicted source->successor distance is 9.37 um where the GT
    displacement is only 4.82 -- our detections roughly DOUBLE the apparent
    motion. If a fast cell is smeared across the exposure, the detected peak
    should sit somewhere along the direction of travel, so the residual
    `pred - gt` would project systematically onto the motion direction. That
    would be a SYSTEMATIC, correctable bias rather than isotropic scatter --
    and §25 only ruled out quantisation, not this.

Q3. ARE ERRORS BURSTY OR UNIFORM?
    If swaps cluster in a few frames, something episodic is happening (a division
    wave, a fast tissue movement, an imaging artifact) and a frame-level
    intervention is possible. If they are Poisson-uniform, they are a property of
    individual cells and only a per-cell fix works. Compared against a Poisson
    null via the variance-to-mean ratio.

    python scripts/local/error_anatomy.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import SCALE, Graph, match_nodes  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--gt-root", type=Path, default=GT_ROOT)
    args = ap.parse_args()

    df = pd.read_csv(args.submission)
    nodes = df[df.row_type == "node"]
    edges = df[df.row_type == "edge"]

    disp, ok = [], []                 # Q1
    proj_c, perp_c, proj_s, perp_s = [], [], [], []   # Q2
    per_vol_times: dict[str, list[int]] = {}          # Q3
    per_vol_frames: dict[str, int] = {}

    for ds, gn in nodes.groupby("dataset"):
        ge = edges[edges.dataset == ds]
        s = ge.source_id.astype(int).to_numpy()
        t = ge.target_id.astype(int).to_numpy()
        pe = set(zip(s.tolist(), t.tolist()))
        pos = {int(r.node_id): np.array([r.z, r.y, r.x], float) * SCALE
               for r in gn.itertuples()}
        t_of = dict(zip(gn.node_id.astype(int), gn.t.astype(int)))

        gt = load_geff(args.gt_root / f"{ds}.geff")
        gpos = {int(r[0]): np.asarray(r[2:5], float) * SCALE for r in gt.graph.nodes}
        gt_t = {int(r[0]): int(r[1]) for r in gt.graph.nodes}
        pred = Graph(
            nodes=np.asarray([[r.node_id, r.t, r.z, r.y, r.x] for r in gn.itertuples()],
                             dtype=np.float64),
            edges=np.asarray(list(zip(s, t)), dtype=np.int64),
        )
        matched = match_nodes(gt.graph, pred, max_distance=7.0, scale=SCALE)
        g2p = {g: p for p, g in matched.items()}

        swap_times = []
        for ga, gb in gt.graph.edges:
            ga, gb = int(ga), int(gb)
            pa, pb = g2p.get(ga), g2p.get(gb)
            if pa is None or pb is None:
                continue                       # detection FN: not an association test
            d = float(np.linalg.norm(gpos[gb] - gpos[ga]))
            good = (pa, pb) in pe
            disp.append(d)
            ok.append(good)
            if not good:
                swap_times.append(gt_t[ga])

            # Q2: project the source's localisation residual onto motion direction
            if d > 1e-6:
                u = (gpos[gb] - gpos[ga]) / d
                r = pos[pa] - gpos[ga]
                p = float(r @ u)
                q = float(np.linalg.norm(r - p * u))
                (proj_c if good else proj_s).append(p)
                (perp_c if good else perp_s).append(q)

        per_vol_times[ds] = swap_times
        per_vol_frames[ds] = int(max(t_of.values())) + 1 if t_of else 0

    disp = np.array(disp); ok = np.array(ok, bool)
    print(f"association-testable GT edges (both endpoints detected): {len(disp):,}\n")

    # ---- Q1 --------------------------------------------------------------
    print("Q1. RECALL vs GT DISPLACEMENT  -- where is the speed cliff?")
    print(f"{'GT disp (um)':>16} {'edges':>7} {'% of all':>9} {'P(correct)':>11} {'lost edges':>11}")
    bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 1e9]
    lost_by_bin = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (disp >= lo) & (disp < hi)
        if not m.sum():
            continue
        lost = int((~ok[m]).sum())
        lost_by_bin.append((f"{lo:g}-{hi:g}" if hi < 1e9 else f">{lo:g}", lost))
        print(f"{(f'{lo:g}-{hi:g}' if hi<1e9 else f'>{lo:g}'):>16} {m.sum():>7,} "
              f"{m.mean():>8.1%} {ok[m].mean():>11.1%} {lost:>11,}")
    tot_lost = int((~ok).sum())
    fast = disp >= 4
    print(f"\n  edges we get WRONG: {tot_lost:,} of {len(disp):,}")
    print(f"  of those, {int((~ok & fast).sum()):,} = {(~ok & fast).sum()/max(tot_lost,1):.1%} "
          f"have GT displacement >= 4 um")
    print(f"  but edges with disp >= 4 um are only {fast.mean():.1%} of all testable edges")
    print(f"  => fixing every edge above 4 um recovers {(~ok & fast).sum()/max(tot_lost,1):.0%} "
          f"of all association loss")

    # ---- Q2 --------------------------------------------------------------
    print("\nQ2. IS LOCALISATION ERROR ALIGNED WITH MOTION?  (motion-blur test)")
    print(f"{'':>22} {'along motion':>14} {'perpendicular':>15}")
    for name, p, q in (("correct links", proj_c, perp_c), ("mis-linked", proj_s, perp_s)):
        p, q = np.array(p), np.array(q)
        print(f"{name:>22} {np.mean(p):>+9.3f} um {np.mean(q):>12.3f} um   "
              f"(n={len(p):,})")
    allp = np.array(proj_c + proj_s)
    print(f"\n  mean projection onto direction of travel: {allp.mean():+.3f} um "
          f"(SE {allp.std()/np.sqrt(len(allp)):.3f})")
    z = allp.mean() / (allp.std() / np.sqrt(len(allp)))
    print(f"  z = {z:+.1f}  =>  " + ("SYSTEMATIC alignment with motion"
                                     if abs(z) > 3 else "no systematic alignment"))

    # ---- Q3 --------------------------------------------------------------
    print("\nQ3. ARE SWAPS BURSTY OR UNIFORM?  (variance-to-mean of per-frame counts)")
    print(f"{'volume':>16} {'swaps':>6} {'frames':>7} {'mean/frame':>11} {'VMR':>7} {'verdict':>10}")
    vmrs = []
    for ds, times in sorted(per_vol_times.items(), key=lambda kv: -len(kv[1]))[:8]:
        n = per_vol_frames[ds]
        if not times or n < 5:
            continue
        counts = np.bincount(times, minlength=n)[:n]
        vmr = counts.var() / counts.mean() if counts.mean() > 0 else float("nan")
        vmrs.append(vmr)
        print(f"{ds:>16} {len(times):>6} {n:>7} {counts.mean():>11.3f} {vmr:>7.2f} "
              f"{'BURSTY' if vmr > 1.5 else 'uniform':>10}")
    if vmrs:
        print(f"\n  Poisson (uniform) would give VMR = 1.0.  median observed = {np.median(vmrs):.2f}")
        print("  => " + ("errors CLUSTER in time: episodic, so a frame-level intervention exists"
                         if np.median(vmrs) > 1.5 else
                         "errors are ~uniform in time: a per-cell property, no episodic fix"))


if __name__ == "__main__":
    main()
