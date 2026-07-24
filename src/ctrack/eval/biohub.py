"""The competition metric, reimplemented in numpy/scipy.

Derived by reading `biohub_tracking/metrics.py` and `division_metrics.py` from the public
support pack that the top notebooks validate against. Reimplemented here rather than imported
so we own it, can run it without tracksdata/polars, and can sweep thousands of configs offline.

    total_node_ratio = (N_pred - N_total) / N_total     # N_total = estimated_number_of_nodes
    edge_jaccard     = tp / (tp + fp + fn)
    adj_edge_jaccard = max(0, edge_jaccard * (1 - ALPHA * total_node_ratio))
    score            = adj_edge_jaccard + DIVISION_WEIGHT * division_jaccard

Two structural facts drive everything (see docs/metric.md):

* An edge counts only if an endpoint matched a GT node that itself has an edge. Edges between
  two unannotated cells are invisible -- neither TP nor FP. The GT is ~0.2% dense, so this is a
  "reproduce a few dozen traced trajectories" problem, not a "track 25k cells" problem.
* `adj_edge_jaccard` is clamped below at 0 but NOT above, so predicting fewer nodes than
  N_total multiplies the score up.

!! UNVERIFIED against the real leaderboard. The support pack is a community reimplementation,
not the official scorer. Reproduce a known LB score before trusting this for tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

# (Z, Y, X) micrometres per voxel.
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
MAX_MATCH_DISTANCE_UM = 7.0
ALPHA = 0.1
DIVISION_WEIGHT = 0.1


@dataclass(frozen=True)
class Graph:
    """A tracking graph. `nodes` is (N, 5): node_id, t, z, y, x. `edges` is (E, 2)."""

    nodes: np.ndarray
    edges: np.ndarray

    @property
    def ids(self) -> np.ndarray:
        return self.nodes[:, 0].astype(np.int64)

    @property
    def t(self) -> np.ndarray:
        return self.nodes[:, 1].astype(np.int64)

    @property
    def zyx(self) -> np.ndarray:
        return self.nodes[:, 2:5].astype(np.float64)

    def out_degree(self) -> dict[int, int]:
        d: dict[int, int] = {}
        for s in self.edges[:, 0].astype(np.int64):
            d[int(s)] = d.get(int(s), 0) + 1
        return d

    def in_degree(self) -> dict[int, int]:
        d: dict[int, int] = {}
        for tgt in self.edges[:, 1].astype(np.int64):
            d[int(tgt)] = d.get(int(tgt), 0) + 1
        return d


@dataclass(frozen=True)
class Counts:
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int
    num_matched_gt: int
    num_gt_nodes: int


def match_nodes(
    gt: Graph,
    pred: Graph,
    *,
    max_distance: float = MAX_MATCH_DISTANCE_UM,
    scale: np.ndarray = SCALE,
) -> dict[int, int]:
    """Match predicted nodes to GT nodes by centroid distance, per timepoint.

    Optimal (Hungarian) assignment within each frame, keeping only pairs within
    `max_distance` in physical units. Returns {pred_node_id: gt_node_id}.
    """
    matches: dict[int, int] = {}
    gt_t, pred_t = gt.t, pred.t
    for frame in np.unique(gt_t):
        gi = np.flatnonzero(gt_t == frame)
        pi = np.flatnonzero(pred_t == frame)
        if gi.size == 0 or pi.size == 0:
            continue
        cost = np.linalg.norm(
            (gt.zyx[gi][:, None, :] - pred.zyx[pi][None, :, :]) * scale, axis=2
        )
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= max_distance:
                matches[int(pred.ids[pi[c]])] = int(gt.ids[gi[r]])
    return matches


def evaluate(
    gt: Graph,
    pred: Graph,
    *,
    max_distance: float = MAX_MATCH_DISTANCE_UM,
    scale: np.ndarray = SCALE,
) -> Counts:
    """Edge and division TP/FP/FN for one volume."""
    matched = match_nodes(gt, pred, max_distance=max_distance, scale=scale)

    gt_out, gt_in = gt.out_degree(), gt.in_degree()
    gt_edge_set = {(int(s), int(t)) for s, t in gt.edges}

    edge_tp = valid_pred = 0
    seen: set[tuple[int, int]] = set()
    for s, t in pred.edges:
        key = (int(s), int(t))
        if key in seen:  # the reference dedups; duplicate edges must not inflate the count
            continue
        seen.add(key)
        gs, gt_ = matched.get(key[0]), matched.get(key[1])
        # An edge is only "visible" to the metric if it touches an annotated cell that
        # actually has an edge in the GT. Everything else is neither rewarded nor punished.
        out_valid = gs is not None and gt_out.get(gs, 0) > 0
        in_valid = gt_ is not None and gt_in.get(gt_, 0) > 0
        if not (out_valid or in_valid):
            continue
        valid_pred += 1
        if gs is not None and gt_ is not None and (gs, gt_) in gt_edge_set:
            edge_tp += 1

    edge_fp = valid_pred - edge_tp
    edge_fn = len(gt.edges) - edge_tp

    div_tp, div_fp, div_fn = _evaluate_divisions(gt, pred, matched, gt_out)

    return Counts(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=div_tp,
        division_fp=div_fp,
        division_fn=div_fn,
        num_pred_nodes=len(pred.nodes),
        num_matched_gt=len(set(matched.values())),
        num_gt_nodes=len(gt.nodes),
    )


def _evaluate_divisions(
    gt: Graph, pred: Graph, matched: dict[int, int], gt_out: dict[int, int]
) -> tuple[int, int, int]:
    """Division TP/FP/FN.

    TP: a GT dividing node whose matched predicted node also divides.
    FN: a GT dividing node not recovered.
    FP: predicted divisions at *matched* nodes whose GT node isn't dividing.

    Unmatched predicted divisions are excluded entirely -- which is exactly why the public
    "fake forks" augmentation (2-child nodes at impossible coordinates) is a free no-op here.

    NOTE: the reference `score_divisions` additionally verifies daughter-lineage coverage via
    grandchildren and a bipartite matching. This is the simpler out-degree form; it will
    slightly over-count TPs where a division is topologically present but the daughters are
    wrong. Divisions are worth at most `DIVISION_WEIGHT`, so this is an acceptable
    approximation until validated -- but do not tune divisions hard against it.
    """
    pred_out = pred.out_degree()
    gt_to_pred = {g: p for p, g in matched.items()}

    gt_dividers = [n for n, d in gt_out.items() if d >= 2]
    tp = sum(
        1
        for g in gt_dividers
        if (p := gt_to_pred.get(g)) is not None and pred_out.get(p, 0) >= 2
    )
    fn = len(gt_dividers) - tp

    matched_pred_divs = sum(
        1
        for p, g in matched.items()
        if pred_out.get(p, 0) >= 2 and gt_out.get(g, 0) >= 1
    )
    fp = max(0, matched_pred_divs - tp)
    return tp, fp, fn


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


def per_sample_metrics(c: Counts, n_total: float) -> dict:
    """Per-volume metrics. `n_total` is the geff `estimated_number_of_nodes`."""
    ratio = (c.num_pred_nodes - n_total) / n_total if n_total > 0 else float("nan")
    edge_jaccard = _jaccard(c.edge_tp, c.edge_fp, c.edge_fn)
    adj = (
        max(0.0, edge_jaccard * (1 - ALPHA * ratio))
        if edge_jaccard == edge_jaccard and ratio == ratio
        else float("nan")
    )
    return {
        "edge_tp": c.edge_tp, "edge_fp": c.edge_fp, "edge_fn": c.edge_fn,
        "division_tp": c.division_tp, "division_fp": c.division_fp,
        "division_fn": c.division_fn,
        "num_pred_nodes": c.num_pred_nodes,
        "node_recall": c.num_matched_gt / c.num_gt_nodes if c.num_gt_nodes else float("nan"),
        "total_node_ratio": ratio,
        "edge_jaccard": edge_jaccard,
        "adj_edge_jaccard": adj,
    }


def summarise(rows: list[dict]) -> dict:
    """Micro-average across volumes: sum counts, then take the Jaccard.

    The division term is dropped entirely when no volume has a GT division, matching the
    reference's `has_divisions` gate.
    """
    valid = [r for r in rows if r["edge_jaccard"] == r["edge_jaccard"]]
    if not valid:
        return {"edge_jaccard": float("nan"), "division_jaccard": float("nan"),
                "score": float("nan")}

    tot = {k: sum(r[k] for r in valid) for k in
           ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")}
    edge_jaccard = _jaccard(tot["edge_tp"], tot["edge_fp"], tot["edge_fn"])

    has_divisions = (tot["division_tp"] + tot["division_fn"]) > 0
    division_jaccard = (
        _jaccard(tot["division_tp"], tot["division_fp"], tot["division_fn"])
        if has_divisions else float("nan")
    )

    adj = [r["adj_edge_jaccard"] for r in valid if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    adj_edge_jaccard = float(np.mean(adj)) if adj else edge_jaccard

    score = adj_edge_jaccard + DIVISION_WEIGHT * division_jaccard if has_divisions else adj_edge_jaccard
    return {
        **tot,
        "edge_jaccard": edge_jaccard,
        "adj_edge_jaccard": adj_edge_jaccard,
        "division_jaccard": division_jaccard,
        "score": score,
    }


def node_budget_multiplier(n_pred: int, n_total: float) -> float:
    """The unclamped-above multiplier on edge Jaccard. >1 when under-predicting nodes.

    Exposed on its own because the detection-threshold sweep trades edge recall against this
    term, and the field appears to be tuning that tradeoff only by blind LB probing.
    """
    return 1 - ALPHA * ((n_pred - n_total) / n_total)
