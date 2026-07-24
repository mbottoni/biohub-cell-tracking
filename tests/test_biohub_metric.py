"""Tests for the reimplemented competition metric.

These pin the *structural* properties that drive strategy -- especially that edges between
unannotated cells are invisible, and that the node-count multiplier is unclamped above.
If a future refactor breaks one of these, the tuning conclusions in docs/metric.md are void.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctrack.eval.biohub import (
    ALPHA,
    Graph,
    evaluate,
    match_nodes,
    node_budget_multiplier,
    per_sample_metrics,
    summarise,
)


def _chain(start_id: int, n: int, *, y: float = 100.0, t0: int = 0) -> Graph:
    """A single cell tracked over n frames, drifting 2 voxels in y per frame."""
    nodes = np.array(
        [[start_id + i, t0 + i, 32.0, y + 2 * i, 100.0] for i in range(n)], dtype=np.float64
    )
    edges = np.array([[start_id + i, start_id + i + 1] for i in range(n - 1)], dtype=np.int64)
    return Graph(nodes=nodes, edges=edges)


def test_perfect_prediction() -> None:
    gt = _chain(1, 5)
    c = evaluate(gt, _chain(1, 5))
    assert (c.edge_tp, c.edge_fp, c.edge_fn) == (4, 0, 0)
    assert c.num_matched_gt == 5
    assert per_sample_metrics(c, n_total=1000)["edge_jaccard"] == 1.0


def test_edges_between_unannotated_cells_are_invisible() -> None:
    """The single most important property: over-detection is free at the edge level.

    GT is one 5-frame track. The prediction reproduces it and *also* tracks 20 other cells
    far away that nobody annotated. Score must be unchanged.
    """
    gt = _chain(1, 5)

    extra_nodes, extra_edges = [], []
    nid = 1000
    for k in range(20):
        chain = _chain(nid, 5, y=100.0 + 40 * (k + 1))
        extra_nodes.append(chain.nodes)
        extra_edges.append(chain.edges)
        nid += 100

    pred = Graph(
        nodes=np.vstack([_chain(1, 5).nodes, *extra_nodes]),
        edges=np.vstack([_chain(1, 5).edges, *extra_edges]),
    )
    c = evaluate(gt, pred)
    assert (c.edge_tp, c.edge_fp, c.edge_fn) == (4, 0, 0), "unannotated edges must not count"
    assert c.num_pred_nodes == 105  # they do still count against the node budget


def test_spurious_edge_on_an_annotated_node_does_count() -> None:
    """The flip side: hanging a wrong edge off a GT-matched node is penalised."""
    gt = _chain(1, 5)
    pred = _chain(1, 5)
    decoy = np.array([[900, 3, 32.0, 106.0, 130.0]], dtype=np.float64)  # >7um away in x
    pred = Graph(
        nodes=np.vstack([pred.nodes, decoy]),
        edges=np.vstack([pred.edges, np.array([[3, 900]], dtype=np.int64)]),
    )
    c = evaluate(gt, pred)
    assert c.edge_tp == 4
    assert c.edge_fp == 1, "edge from an annotated node to a decoy must be a FP"


def test_missing_edges_are_false_negatives() -> None:
    gt = _chain(1, 5)
    pred = Graph(nodes=_chain(1, 5).nodes, edges=_chain(1, 5).edges[:2])
    c = evaluate(gt, pred)
    assert (c.edge_tp, c.edge_fp, c.edge_fn) == (2, 0, 2)


def test_matching_respects_the_7um_radius() -> None:
    gt = _chain(1, 3)
    # shift 10 voxels in y = 10 * 0.40625 = 4.06 um -> inside
    near = Graph(nodes=_chain(1, 3).nodes + np.array([0, 0, 0, 10, 0]), edges=_chain(1, 3).edges)
    assert len(match_nodes(gt, near)) == 3
    # shift 4 voxels in z = 4 * 1.625 = 6.5 um -> inside; 6 voxels = 9.75 um -> outside
    far = Graph(nodes=_chain(1, 3).nodes + np.array([0, 0, 6, 0, 0]), edges=_chain(1, 3).edges)
    assert len(match_nodes(gt, far)) == 0


def test_node_budget_multiplier_is_unclamped_above() -> None:
    """Under-predicting nodes inflates the score. This asymmetry is a core strategic lever."""
    assert node_budget_multiplier(1000, 1000) == pytest.approx(1.0)
    assert node_budget_multiplier(1500, 1000) == pytest.approx(1 - ALPHA * 0.5)  # 0.95
    assert node_budget_multiplier(800, 1000) == pytest.approx(1 + ALPHA * 0.2)   # 1.02
    assert node_budget_multiplier(500, 1000) == pytest.approx(1 + ALPHA * 0.5)   # 1.05

    gt = _chain(1, 5)
    c = evaluate(gt, _chain(1, 5))
    m = per_sample_metrics(c, n_total=10.0)  # 5 predicted vs 10 estimated
    assert m["edge_jaccard"] == 1.0
    assert m["adj_edge_jaccard"] == pytest.approx(1.05), "score can exceed 1.0"


def test_division_recovered_and_missed() -> None:
    # GT: node 1 at t=0 divides into 2 and 3 at t=1
    gt = Graph(
        nodes=np.array([[1, 0, 32, 100, 100], [2, 1, 32, 104, 100], [3, 1, 32, 96, 100]], float),
        edges=np.array([[1, 2], [1, 3]], dtype=np.int64),
    )
    c = evaluate(gt, gt)
    assert (c.division_tp, c.division_fn, c.division_fp) == (1, 0, 0)

    # prediction keeps only one daughter -> division missed
    linear = Graph(nodes=gt.nodes, edges=np.array([[1, 2]], dtype=np.int64))
    c2 = evaluate(gt, linear)
    assert (c2.division_tp, c2.division_fn) == (0, 1)


def test_unmatched_predicted_divisions_are_free() -> None:
    """Why the public 'fake forks' augmentation is a no-op for divisions.

    Injects a 2-child node at coordinates no GT node can be near. It must not become a FP.
    """
    gt = _chain(1, 5)
    fork = np.array(
        [[9001, -1000, -10000, -10000, -10000],
         [9002, -999, -10000, -10000, -10000],
         [9003, -999, -10001, -10000, -10000]], dtype=np.float64
    )
    pred = Graph(
        nodes=np.vstack([_chain(1, 5).nodes, fork]),
        edges=np.vstack([_chain(1, 5).edges, np.array([[9001, 9002], [9001, 9003]], np.int64)]),
    )
    c = evaluate(gt, pred)
    assert c.division_fp == 0, "unmatched predicted divisions must be excluded"
    assert c.edge_fp == 0, "edges between unmatched nodes must be invisible"


def test_summarise_drops_division_term_when_gt_has_none() -> None:
    gt = _chain(1, 5)
    rows = [per_sample_metrics(evaluate(gt, _chain(1, 5)), n_total=5.0)]
    s = summarise(rows)
    assert s["division_jaccard"] != s["division_jaccard"]  # NaN
    assert s["score"] == pytest.approx(s["adj_edge_jaccard"])
