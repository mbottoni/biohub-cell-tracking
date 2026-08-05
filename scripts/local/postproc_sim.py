#!/usr/bin/env python3
"""STRATEGY 31 -- optimise final-graph post-processing OFFLINE, with the real metric.

`submission.csv` is the complete predicted graph. Any post-processing step that
only REMOVES nodes or edges can therefore be applied to it directly and scored
exactly, in seconds, against real GT on all 24 volumes -- no kernel, no GPU, no
submission. A class of experiments that has been costing ~4 h each becomes a
local sweep.

⚠ What this is and is not. It does NOT simulate changing an env var, because an
env var also changes everything downstream of it inside the pipeline. It
evaluates a NEW post-filter appended at the very end, which is a thing we could
actually ship. For removal-only rules those are the same operation; for anything
that adds nodes or edges they are not, and this cannot model it.

⚠ It also inherits §30's contamination caveat, but favourably: these are pure
geometry/topology rules with no dependence on model confidence, which is §30.2's
"structural" row where the local instrument held up.

Sweeps implemented:

  * `minlen`  -- drop connected components with fewer than L nodes. Exp 9 tested
    L=20 on the board (-0.027) and nothing between 6 and 20 was ever measured.
  * `maxedge` -- drop edges longer than X um (shipped cap is 14.0; observed max
    is ~7.3, so this is expected to be inert and is included as a control).
  * `divtrim` -- ORACLE bound on §26.4's FP-trim: delete the second child of
    exactly those divisions that are false positives, using GT. No real selector
    can beat this, so it caps the whole FP-removal half of lever #2.

    python scripts/local/postproc_sim.py submissions/evalset_d4edge/submission.csv
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
from ctrack.eval.biohub import (  # noqa: E402
    SCALE, Graph, evaluate, match_nodes, per_sample_metrics, summarise,
)

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def load(sub: Path):
    df = pd.read_csv(sub)
    n = df[df.row_type == "node"]
    e = df[df.row_type == "edge"]
    out = {}
    for ds, gn in n.groupby("dataset"):
        ge = e[e.dataset == ds]
        out[ds] = (
            np.asarray([[r.node_id, r.t, r.z, r.y, r.x] for r in gn.itertuples()],
                       dtype=np.float64),
            np.asarray(list(zip(ge.source_id.astype(int), ge.target_id.astype(int))),
                       dtype=np.int64),
        )
    return out


def components(nodes: np.ndarray, edges: np.ndarray) -> dict[int, int]:
    parent = {int(i): int(i) for i in nodes[:, 0]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    size: dict[int, int] = defaultdict(int)
    for i in parent:
        size[find(i)] += 1
    return {i: size[find(i)] for i in parent}


def apply_minlen(nodes, edges, L):
    if L <= 1:
        return nodes, edges
    sz = components(nodes, edges)
    keep = {i for i, s in sz.items() if s >= L}
    nm = np.array([r[0] in keep for r in nodes])
    em = np.array([int(a) in keep and int(b) in keep for a, b in edges]) if len(edges) else np.zeros(0, bool)
    return nodes[nm], edges[em]


def apply_maxedge(nodes, edges, X):
    if not len(edges):
        return nodes, edges
    pos = {int(r[0]): r[2:5] * SCALE for r in nodes}
    d = np.array([np.linalg.norm(pos[int(b)] - pos[int(a)]) for a, b in edges])
    return nodes, edges[d <= X]


def apply_divtrim_oracle(nodes, edges, gt):
    """Delete the second child of every division that is a false positive (uses GT)."""
    if not len(edges):
        return nodes, edges
    pred = Graph(nodes=nodes, edges=edges)
    matched = match_nodes(gt.graph, pred, max_distance=7.0, scale=SCALE)
    gt_out = defaultdict(int)
    for a, _ in gt.graph.edges:
        gt_out[int(a)] += 1
    out = defaultdict(list)
    for k, (a, b) in enumerate(edges):
        out[int(a)].append(k)
    pos = {int(r[0]): r[2:5] * SCALE for r in nodes}
    drop = set()
    for s, ks in out.items():
        if len(ks) < 2:
            continue
        g = matched.get(s)
        # FP division: it is matched to an annotated node that does NOT divide.
        if g is not None and gt_out.get(int(g), 0) < 2:
            # remove the farther child, which is what the safe-division step added
            far = max(ks, key=lambda k: np.linalg.norm(pos[int(edges[k][1])] - pos[s]))
            drop.add(far)
    if not drop:
        return nodes, edges
    return nodes, edges[[k for k in range(len(edges)) if k not in drop]]


def score(graphs, gts, transform=None, **kw):
    rows = []
    for ds, (nodes, edges) in graphs.items():
        gt = gts[ds]
        n, e = (nodes, edges) if transform is None else transform(nodes, edges, gt=gt, **kw)
        m = per_sample_metrics(evaluate(gt.graph, Graph(nodes=n, edges=e)),
                               gt.estimated_number_of_nodes)
        m["dataset"] = ds
        rows.append(m)
    return summarise(rows), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--gt-root", type=Path, default=GT_ROOT)
    args = ap.parse_args()

    graphs = load(Path(args.submission))
    gts = {ds: load_geff(args.gt_root / f"{ds}.geff") for ds in graphs}
    print(f"{len(graphs)} volumes loaded\n")

    base, _ = score(graphs, gts)
    print(f"{'variant':<26} {'edge_J':>8} {'adj':>8} {'div_J':>8} {'SCORE':>8} {'delta':>9}")
    print(f"{'BASELINE (as submitted)':<26} {base['edge_jaccard']:>8.4f} "
          f"{base['adj_edge_jaccard']:>8.4f} {base['division_jaccard']:>8.4f} "
          f"{base['score']:>8.4f} {'--':>9}")

    def row(label, agg):
        print(f"{label:<26} {agg['edge_jaccard']:>8.4f} {agg['adj_edge_jaccard']:>8.4f} "
              f"{agg['division_jaccard']:>8.4f} {agg['score']:>8.4f} "
              f"{agg['score']-base['score']:>+9.4f}")

    for L in (7, 8, 10, 12, 15, 20):
        agg, _ = score(graphs, gts, lambda n, e, gt, L=L: apply_minlen(n, e, L))
        row(f"minlen >= {L}", agg)

    for X in (10.0, 8.0, 6.0):
        agg, _ = score(graphs, gts, lambda n, e, gt, X=X: apply_maxedge(n, e, X))
        row(f"maxedge <= {X} um", agg)

    agg, _ = score(graphs, gts, lambda n, e, gt: apply_divtrim_oracle(n, e, gt))
    row("divtrim ORACLE (uses GT)", agg)
    print("\n  ^ divtrim is an UPPER BOUND on the FP-removal half of lever #2 (§26.4):")
    print("    it deletes exactly the false-positive divisions using ground truth,")
    print("    so no real selector can beat it.")


if __name__ == "__main__":
    main()
