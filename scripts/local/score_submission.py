#!/usr/bin/env python3
"""Score a kernel's `submission.csv` against real GT, locally.

The four volumes in `test/` are byte-identical to train volumes (docs/data.md),
so their ground truth is already on disk. A commit run's `submission.csv` is a
complete predicted graph over exactly those four -- which means every quantity we
have been *inferring* from LB deltas (edge_J, div_J, TP/FP/FN, the node-ratio
multiplier) is directly computable here, for free, before spending a submission.

STRATEGY 16 is the reason this exists. Three division estimates in two days were
each revised by measuring something already in hand, and the last one cost a
readable 0.006 on the board. This closes that loop.

READ THE CAVEAT: four volumes carry only 3 annotated divisions between them, so
`division_jaccard` here is quantised into huge steps and is a WEAK estimate of the
hidden-set value. Use it to answer "is TP zero or not", not to tune. `edge_J` is
backed by ~2,100 GT edges and is far more stable. The hidden test is ~10x larger.

    python scripts/local/score_submission.py A/submission.csv B/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import Graph, evaluate, per_sample_metrics, summarise  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def load_submission(path: Path) -> dict[str, Graph]:
    nodes: dict[str, list] = defaultdict(list)
    edges: dict[str, list] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]
            if r["row_type"] == "node":
                nodes[ds].append((int(r["node_id"]), int(r["t"]),
                                  float(r["z"]), float(r["y"]), float(r["x"])))
            else:
                edges[ds].append((int(r["source_id"]), int(r["target_id"])))
    return {ds: Graph(nodes=np.asarray(v, dtype=np.float64),
                      edges=(np.asarray(edges[ds], dtype=np.int64)
                             if edges[ds] else np.zeros((0, 2), dtype=np.int64)))
            for ds, v in nodes.items()}


def score(path: Path) -> tuple[dict, list[dict]]:
    preds = load_submission(path)
    rows = []
    for ds, pred in sorted(preds.items()):
        gt = load_geff(GT_ROOT / f"{ds}.geff")
        m = per_sample_metrics(evaluate(gt.graph, pred), gt.estimated_number_of_nodes)
        m["dataset"] = ds
        rows.append(m)
    return summarise(rows), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", nargs="+")
    args = ap.parse_args()

    results = []
    for s in args.submissions:
        p = Path(s)
        agg, rows = score(p)
        results.append((p, agg, rows))
        print(f"\n===== {p.parent.name}/{p.name} =====")
        print(f"{'volume':>16} {'edge_J':>8} {'adj':>8} {'ratio':>8} {'nrecall':>8} "
              f"{'e_tp':>6} {'e_fp':>6} {'e_fn':>6} {'d_tp':>5} {'d_fp':>5} {'d_fn':>5}")
        for m in rows:
            print(f"{m['dataset']:>16} {m['edge_jaccard']:>8.4f} {m['adj_edge_jaccard']:>8.4f} "
                  f"{m['total_node_ratio']:>+8.3f} {m['node_recall']:>8.4f} "
                  f"{m['edge_tp']:>6} {m['edge_fp']:>6} {m['edge_fn']:>6} "
                  f"{m['division_tp']:>5} {m['division_fp']:>5} {m['division_fn']:>5}")
        print(f"  edge_jaccard      {agg['edge_jaccard']:.4f}")
        print(f"  adj_edge_jaccard  {agg['adj_edge_jaccard']:.4f}")
        print(f"  division_jaccard  {agg['division_jaccard']:.4f}")
        print(f"  SCORE             {agg['score']:.4f}")

    if len(results) == 2:
        (pa, a, _), (pb, b, _) = results
        print(f"\n===== delta: {pb.parent.name} - {pa.parent.name} =====")
        for k in ("edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score"):
            print(f"  {k:<18} {a[k]:>8.4f} -> {b[k]:>8.4f}   {b[k]-a[k]:>+8.4f}")


if __name__ == "__main__":
    main()
