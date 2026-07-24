#!/usr/bin/env python3
"""Verify ctrack.eval.biohub against the reference implementation.

The whole strategy rests on my numpy reimplementation of the metric being faithful. This runs
the *reference* implementation (biohub_tracking.metrics, which the top public notebooks
validate against) and my reimplementation on the SAME (prediction, ground-truth) pairs and
asserts they agree.

The predictions come from the real baseline on real train volumes, so this exercises the
metric on realistic graphs, not toy cases.

    python scripts/verify_metric.py \
        --root /Volumes/extSSD/kaggle/biohub/raw \
        --gt-root /Volumes/extSSD/kaggle/biohub/gt/train \
        --pack /path/to/biohub_tracking_parent \
        --datasets 44b6_0113de3b 6bba_062c8d37
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.data.volume import open_volume  # noqa: E402
from ctrack.detect.blob import detect_volume  # noqa: E402
from ctrack.eval.biohub import Graph, evaluate, per_sample_metrics  # noqa: E402
from ctrack.link.hungarian import build_graph  # noqa: E402

SCALE = (1.625, 0.40625, 0.40625)


def to_tracksdata(nodes: np.ndarray, edges: np.ndarray):
    """Build a tracksdata graph from (N,5)=[id,t,z,y,x] nodes and (E,2) edges.

    The reference matches by node attributes, so ids only need to be internally consistent;
    we map our node_id -> tracksdata index and translate edges through it.
    """
    import polars as pl
    import tracksdata as td
    from tracksdata.graph import RustWorkXGraph

    K = td.DEFAULT_ATTR_KEYS
    g = RustWorkXGraph()
    for key in (K.Z, K.Y, K.X):
        g.add_node_attr_key(key, dtype=pl.Float64, default_value=0.0)
    id_to_index: dict[int, int] = {}
    for nid, t, z, y, x in nodes:
        idx = g.add_node({K.T: int(t), K.Z: float(z), K.Y: float(y), K.X: float(x)})
        id_to_index[int(nid)] = idx
    for s, t in edges:
        g.add_edge(source_id=id_to_index[int(s)], target_id=id_to_index[int(t)], attrs={})
    return g


def reference_score(pred_nodes, pred_edges, gt_nodes, gt_edges, n_total: float) -> dict:
    """Run the reference metrics on the same graphs."""
    from biohub_tracking.metrics import evaluate as ref_evaluate
    from biohub_tracking.metrics import node_recall, per_sample_metrics as ref_psm

    pred_g = to_tracksdata(pred_nodes, pred_edges)
    gt_g = to_tracksdata(gt_nodes, gt_edges)

    er = ref_evaluate(pred_g, gt_g, scale=SCALE, max_distance=7.0)
    recall = node_recall(pred_g, gt_g)
    row = ref_psm(er=er, n_total=n_total, node_recall=recall)
    return dict(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--pack", required=True, help="dir containing the biohub_tracking package")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    sys.path.insert(0, args.pack)

    fields = ["edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
              "num_pred_nodes", "edge_jaccard", "adj_edge_jaccard", "total_node_ratio"]

    all_ok = True
    for dataset in args.datasets:
        vol = open_volume(Path(args.root) / "train" / f"{dataset}.zarr")
        det = detect_volume(vol, threshold=args.threshold)
        p_nodes, p_edges = build_graph(det, np.array(SCALE))
        gt = load_geff(Path(args.gt_root) / f"{dataset}.geff")

        mine = per_sample_metrics(
            evaluate(gt.graph, Graph(nodes=p_nodes, edges=p_edges)),
            gt.estimated_number_of_nodes,
        )
        ref = reference_score(
            p_nodes, p_edges, gt.graph.nodes, gt.graph.edges, gt.estimated_number_of_nodes
        )

        print(f"\n== {dataset} (thr {args.threshold}, {len(p_nodes)} pred nodes) ==")
        print(f"  {'field':18s} {'mine':>14s} {'reference':>14s}  match")
        for f in fields:
            m, r = mine.get(f), ref.get(f)
            if isinstance(m, float) and isinstance(r, float):
                ok = np.isclose(m, r, rtol=1e-6, atol=1e-9) or (np.isnan(m) and np.isnan(r))
            else:
                ok = m == r
            all_ok &= bool(ok)
            print(f"  {f:18s} {str(m):>14.14s} {str(r):>14.14s}  {'OK' if ok else 'MISMATCH <<<'}")

    print("\n" + ("ALL METRICS MATCH — reimplementation verified" if all_ok
                  else "MISMATCH — investigate before trusting local scores"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
