#!/usr/bin/env python3
"""STRATEGY 31 -- can COLLECTIVE motion separate swaps where individual motion cannot?

The facts this is built on, all measured:

* §18.2: swapped cells move **5.20 um** against 1.46 um for correct links, and in
  **377/379 = 99%** of swaps the link we took was SHORTER than the correct one.
  So raw distance is actively misleading on exactly these cases.
* §18.2 also killed `MOTION_RELINK_VELOCITY_WEIGHT`: flipping the cell's OWN
  extrapolated velocity fixed 0 of 65 swaps at every setting tested.
* §28: the learned probability is inert -- BONUS 1/3/6/17 all land within +0.006,
  and 72% of swap-correct edges are not candidates at all (§21.1).

So the relink cost `motion_dist + 0.05*raw_dist - BONUS*prob` has exhausted every
term it contains. The question is whether a term it does NOT contain would work.

**Embryonic cells move as tissue, not independently.** Neighbours share a flow
field. A cell that appears to jump 8 um against the local flow is far more likely
to be a mis-link than a real fast cell. The current cost has no neighbourhood term
at all -- `VELOCITY_WEIGHT` uses the cell's own history, which is exactly the
signal a swap corrupts, whereas the flow field is estimated from many cells and is
robust to any single bad link.

This measures, on the known swaps, whether flow-compensated distance ranks the
CORRECT successor above the one we took. It is the same instrument logic as
`probdump`: ask whether the signal discriminates before building anything.

⚠ Why this is worth doing even after §30: a flow term is pure geometry. It does
not touch model confidence, so it sits in §30.2's "structural" row where the
local instrument is trustworthy -- unlike `edgethr30`, which did not survive
contact with the board.

    python scripts/local/tissue_flow.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import SCALE, Graph, match_nodes  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def flow_at(p: np.ndarray, tree: cKDTree, srcs: np.ndarray, disps: np.ndarray,
            k: int, exclude: int | None) -> np.ndarray | None:
    """Median displacement of the k nearest LINKED cells in the same frame."""
    if len(srcs) == 0:
        return None
    kk = min(k + 1, len(srcs))
    _, idx = tree.query(p, k=kk)
    idx = np.atleast_1d(idx)
    if exclude is not None:
        idx = idx[idx != exclude]
    if len(idx) == 0:
        return None
    return np.median(disps[idx], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--gt-root", type=Path, default=GT_ROOT)
    ap.add_argument("--k", type=int, default=12, help="neighbours in the flow estimate")
    args = ap.parse_args()

    df = pd.read_csv(args.submission)
    nodes = df[df.row_type == "node"]
    edges = df[df.row_type == "edge"]

    tot = defaultdict(int)
    rows = []

    for ds, gn in nodes.groupby("dataset"):
        ge = edges[edges.dataset == ds]
        s_arr = ge.source_id.astype(int).to_numpy()
        t_arr = ge.target_id.astype(int).to_numpy()
        pos = {int(r.node_id): np.array([r.z, r.y, r.x], float) * SCALE
               for r in gn.itertuples()}
        t_of = dict(zip(gn.node_id.astype(int), gn.t.astype(int)))
        child = defaultdict(list)
        for a, b in zip(s_arr, t_arr):
            child[a].append(b)
        pred_edges = set(zip(s_arr.tolist(), t_arr.tolist()))

        gt = load_geff(args.gt_root / f"{ds}.geff")
        pred = Graph(
            nodes=np.asarray([[r.node_id, r.t, r.z, r.y, r.x] for r in gn.itertuples()],
                             dtype=np.float64),
            edges=np.asarray(list(zip(s_arr, t_arr)), dtype=np.int64),
        )
        matched = match_nodes(gt.graph, pred, max_distance=7.0, scale=SCALE)
        g2p = {g: p for p, g in matched.items()}

        # Per-frame flow field from every linked cell in that frame.
        by_t_src: dict[int, list[int]] = defaultdict(list)
        for a in child:
            by_t_src[t_of[a]].append(a)
        flow_cache = {}
        for t, srcs in by_t_src.items():
            P = np.stack([pos[a] for a in srcs])
            D = np.stack([pos[child[a][0]] - pos[a] for a in srcs])
            flow_cache[t] = (cKDTree(P), np.array(srcs), P, D)

        for ga, gb in gt.graph.edges:
            pa, pb = g2p.get(int(ga)), g2p.get(int(gb))
            if pa is None or pb is None:
                continue                      # detection FN, not an association error
            if (pa, pb) in pred_edges:
                continue                      # correct link
            kids = child.get(pa)
            if not kids:
                continue                      # no link at all (assFN, not a swap)
            pw = kids[0]                      # the node we linked to instead
            if pw == pb:
                continue

            t = t_of[pa]
            if t not in flow_cache:
                continue
            tree, srcs, P, D = flow_cache[t]
            where = np.where(srcs == pa)[0]
            f = flow_at(pos[pa], tree, srcs, D, args.k,
                        int(where[0]) if len(where) else None)
            if f is None:
                continue

            pred_pos = pos[pa] + f
            d_flow_correct = float(np.linalg.norm(pos[pb] - pred_pos))
            d_flow_wrong = float(np.linalg.norm(pos[pw] - pred_pos))
            d_raw_correct = float(np.linalg.norm(pos[pb] - pos[pa]))
            d_raw_wrong = float(np.linalg.norm(pos[pw] - pos[pa]))

            tot["swaps"] += 1
            tot["raw_picks_correct"] += d_raw_correct < d_raw_wrong
            tot["flow_picks_correct"] += d_flow_correct < d_flow_wrong
            rows.append((d_raw_correct, d_raw_wrong, d_flow_correct, d_flow_wrong,
                         float(np.linalg.norm(f))))

    n = tot["swaps"]
    if not n:
        print("no swaps found")
        return
    a = np.array(rows)
    print(f"swaps analysed: {n}   (flow = median displacement of the {args.k} nearest linked cells)\n")
    print(f"{'criterion':<44} {'picks the CORRECT successor':>28}")
    print(f"{'raw distance (what the cost mostly uses)':<44} "
          f"{tot['raw_picks_correct']:>8} / {n}  = {tot['raw_picks_correct']/n:>6.1%}")
    print(f"{'FLOW-compensated distance':<44} "
          f"{tot['flow_picks_correct']:>8} / {n}  = {tot['flow_picks_correct']/n:>6.1%}")

    print(f"\nmedian distances (um):")
    print(f"  raw   -> correct {np.median(a[:,0]):6.2f}   -> taken {np.median(a[:,1]):6.2f}")
    print(f"  flow  -> correct {np.median(a[:,2]):6.2f}   -> taken {np.median(a[:,3]):6.2f}")
    print(f"  |local flow| median {np.median(a[:,4]):.2f} um")

    gain = tot["flow_picks_correct"] / n - tot["raw_picks_correct"] / n
    print(f"\nflow beats raw distance on {gain:+.1%} of swaps")
    if tot["flow_picks_correct"] / n > 0.5:
        print("  => a flow term would rank the correct successor first on a MAJORITY of")
        print("     the cases the current cost gets wrong. Worth building.")
    else:
        print("  => flow does NOT discriminate these cases either. The swaps are not")
        print("     explained by collective motion; this lever dies here, for free.")


if __name__ == "__main__":
    main()
