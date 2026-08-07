#!/usr/bin/env python3
"""STRATEGY 44.4 item 4 gate -- is pseudo-label purity >= 95% on the verifiable subset?

Dense pseudo-labels (ELEPHANT / noisy-student) would take confident predicted
tracks as training truth for the 96% of cells with no annotation. Purity can
only be verified where annotation exists: predicted edges whose BOTH endpoints
match annotated GT nodes (<=7 um). On that subset we measure edge precision
under increasingly strict confidence filters:

    all edges | displacement <= 4 um | disp <= 4 & track length >= 10

32.1 says links under 4 um are 98% correct, so the pre-registered expectation
is that conservative filters clear 95% easily -- the gate exists to catch the
opposite surprise before anything trains on pseudo-labels.

    .venv/bin/python scripts/local/pseudo_purity.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes,
)

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    args = ap.parse_args()

    import csv
    nodes_by_ds = defaultdict(list)
    edges_by_ds = defaultdict(list)
    with open(args.submission) as f:
        for r in csv.DictReader(f):
            if r["row_type"] == "node":
                nodes_by_ds[r["dataset"]].append(
                    (int(r["node_id"]), int(r["t"]),
                     float(r["z"]), float(r["y"]), float(r["x"])))
            else:
                edges_by_ds[r["dataset"]].append(
                    (int(r["source_id"]), int(r["target_id"])))

    stats = {k: [0, 0] for k in ("all", "disp4", "disp4_len10")}
    for ds in sorted(nodes_by_ds):
        nodes = np.asarray(nodes_by_ds[ds], dtype=np.float64)
        gt = load_geff(GT_ROOT / f"{ds}.geff")
        pred_g = Graph(nodes=nodes, edges=np.zeros((0, 2), dtype=np.int64))
        matched = match_nodes(gt.graph, pred_g,
                              max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
        gt_edges = {(int(s), int(t)) for s, t in gt.graph.edges}
        pos = {int(n[0]): np.asarray(n[2:5]) * SCALE for n in nodes}

        succ, pred_deg = defaultdict(list), defaultdict(int)
        for s, t in edges_by_ds[ds]:
            succ[s].append(t)
        # track length via union of chains
        length = defaultdict(int)
        indeg = defaultdict(int)
        for s, t in edges_by_ds[ds]:
            indeg[t] += 1
        roots = [s for s, _ in edges_by_ds[ds] if indeg[s] == 0]
        for r in set(roots):
            chain, cur = [r], r
            while succ.get(cur):
                cur = succ[cur][0]
                chain.append(cur)
            for n in chain:
                length[n] = max(length[n], len(chain))

        for s, t in edges_by_ds[ds]:
            gs, gtid = matched.get(s), matched.get(t)
            if gs is None or gtid is None:
                continue                       # unverifiable
            ok = (gs, gtid) in gt_edges
            disp = float(np.linalg.norm(pos[t] - pos[s]))
            stats["all"][0] += ok
            stats["all"][1] += 1
            if disp <= 4.0:
                stats["disp4"][0] += ok
                stats["disp4"][1] += 1
                if length.get(s, 0) >= 10:
                    stats["disp4_len10"][0] += ok
                    stats["disp4_len10"][1] += 1

    print(f"{'filter':>14} {'verifiable':>11} {'purity':>8}")
    for k, (tp, n) in stats.items():
        print(f"{k:>14} {n:>11,} {tp/max(1,n):>8.2%}")
    print("\nGATE: >= 95% purity on the strict filter, else pseudo-labels are unsafe.")


if __name__ == "__main__":
    main()
