"""Verify ctrack.eval.biohub against the reference implementation on synthetic graphs.

The strategy rests on the numpy metric being faithful. scripts/verify_metric.py does this on
real predictions (and confirmed an exact match on both embryos, 2026-07-24); this test does it
on small synthetic graphs so the check runs in CI without the dataset. Both are skipped when
tracksdata / the reference package are unavailable, so the default suite does not require them.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ctrack.eval.biohub import Graph, evaluate, per_sample_metrics

pytest.importorskip("tracksdata")
pytest.importorskip("polars")

# The reference package is only present when the support pack has been downloaded; point at it
# via CTRACK_PACK. Skip cleanly otherwise.
_PACK = os.environ.get("CTRACK_PACK")
if _PACK:
    import sys

    sys.path.insert(0, _PACK)
_ref = pytest.importorskip("biohub_tracking.metrics") if _PACK else None
pytestmark = pytest.mark.skipif(_PACK is None, reason="set CTRACK_PACK to the support-pack dir")

SCALE = (1.625, 0.40625, 0.40625)


def _to_td(nodes: np.ndarray, edges: np.ndarray):
    import polars as pl
    import tracksdata as td
    from tracksdata.graph import RustWorkXGraph

    K = td.DEFAULT_ATTR_KEYS
    g = RustWorkXGraph()
    for key in (K.Z, K.Y, K.X):
        g.add_node_attr_key(key, dtype=pl.Float64, default_value=0.0)
    idx = {}
    for nid, t, z, y, x in nodes:
        idx[int(nid)] = g.add_node({K.T: int(t), K.Z: float(z), K.Y: float(y), K.X: float(x)})
    for s, t in edges:
        g.add_edge(source_id=idx[int(s)], target_id=idx[int(t)], attrs={})
    return g


def _reference(pred: Graph, gt: Graph, n_total: float) -> dict:
    from biohub_tracking.metrics import evaluate as ref_eval
    from biohub_tracking.metrics import node_recall, per_sample_metrics as ref_psm

    pg, gg = _to_td(pred.nodes, pred.edges), _to_td(gt.nodes, gt.edges)
    er = ref_eval(pg, gg, scale=SCALE, max_distance=7.0)
    return dict(ref_psm(er=er, n_total=n_total, node_recall=node_recall(pg, gg)))


def _lineage_with_division() -> Graph:
    nodes = np.array([
        [1, 0, 32, 100, 100], [2, 1, 32, 102, 100], [3, 2, 32, 104, 100],  # track, then divides
        [4, 3, 32, 106, 100], [5, 3, 32, 102, 100],
    ], dtype=np.float64)
    edges = np.array([[1, 2], [2, 3], [3, 4], [3, 5]], dtype=np.int64)
    return Graph(nodes=nodes, edges=edges)


@pytest.mark.parametrize("drop_edges,extra", [(0, 0), (1, 0), (0, 2)])
def test_matches_reference(drop_edges: int, extra: int) -> None:
    gt = _lineage_with_division()
    p_nodes = gt.nodes.copy()
    p_edges = gt.edges[: len(gt.edges) - drop_edges].copy() if drop_edges else gt.edges.copy()

    # optionally add spurious far-away detections (should not affect edge metric)
    for k in range(extra):
        p_nodes = np.vstack([p_nodes, [100 + k, k, 10, 10 + 20 * k, 10]])

    pred = Graph(nodes=p_nodes, edges=p_edges)
    n_total = 500.0

    mine = per_sample_metrics(evaluate(gt, pred), n_total)
    ref = _reference(pred, gt, n_total)

    for f in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
              "num_pred_nodes", "edge_jaccard", "adj_edge_jaccard", "total_node_ratio"):
        m, r = mine[f], ref[f]
        if isinstance(m, float):
            assert np.isclose(m, r, rtol=1e-6) or (np.isnan(m) and np.isnan(r)), f
        else:
            assert m == r, f
