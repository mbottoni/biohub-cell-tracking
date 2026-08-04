#!/usr/bin/env python3
"""Split each volume's edge_J loss into DETECTION loss and ASSOCIATION loss.

The metric's edge_fn is `len(gt.edges) - edge_tp`, and a predicted edge can only
be a TP if BOTH endpoints matched a GT node. So for a fixed node set the best
achievable edge_J is

    oracle_edge_J = (# GT edges with BOTH endpoints matched) / (# GT edges)

which is a pure DETECTION quantity -- no linker can beat it, and a perfect linker
reaches it (an oracle emits exactly the matched GT edges, so it has no FPs).
The gap between `oracle_edge_J` and the measured `edge_J` is therefore exactly
what better association could still buy.

This replaces STRATEGY 13.3's `edge_J ~ recall^2` heuristic with a direct
measurement. 17.4 already showed that heuristic fails on the dominant volume.

It also classifies every error:
  det_fn    GT edge with an unmatched endpoint  -- only detection can fix it
  assoc_fn  GT edge, both endpoints matched, we did not predict it
  swap_fn   an assoc_fn whose source DID get linked, to the wrong target
  assoc_fp  predicted edge between matched nodes that is not a GT edge

    python scripts/local/link_ceiling.py <submission.csv> [...]
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
from ctrack.eval.biohub import (  # noqa: E402
    MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes,
)

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


def analyse(ds: str, pred: Graph) -> dict:
    gt = load_geff(GT_ROOT / f"{ds}.geff")
    g = gt.graph
    matched = match_nodes(g, pred, max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
    gt_to_pred = {gid: pid for pid, gid in matched.items()}

    pred_edges = {(int(s), int(t)) for s, t in pred.edges}
    pred_out = defaultdict(set)
    for s, t in pred_edges:
        pred_out[int(s)].add(int(t))

    n_gt = len(g.edges)
    both_matched = tp = det_fn = assoc_fn = swap_fn = 0
    for gs, gtg in g.edges:
        ps, pt = gt_to_pred.get(int(gs)), gt_to_pred.get(int(gtg))
        if ps is None or pt is None:
            det_fn += 1
            continue
        both_matched += 1
        if (ps, pt) in pred_edges:
            tp += 1
        else:
            assoc_fn += 1
            if pred_out.get(ps):        # source was linked, just to the wrong node
                swap_fn += 1

    # Reproduce the metric's own FP rule: an edge counts if EITHER endpoint is a
    # matched GT node that has a GT edge. So linking an annotated cell to an
    # *unannotated* node is an FP too -- which is why a both-endpoints-matched
    # definition returns 0 and does not reconcile with `evaluate`.
    gt_edge_set = {(int(a), int(b)) for a, b in g.edges}
    gt_out_deg, gt_in_deg = g.out_degree(), g.in_degree()
    fp = fp_src = fp_tgt = 0
    for s, t in pred_edges:
        gs, gt_ = matched.get(int(s)), matched.get(int(t))
        out_valid = gs is not None and gt_out_deg.get(gs, 0) > 0
        in_valid = gt_ is not None and gt_in_deg.get(gt_, 0) > 0
        if not (out_valid or in_valid):
            continue
        if gs is not None and gt_ is not None and (gs, gt_) in gt_edge_set:
            continue
        fp += 1
        fp_src += out_valid          # annotated source, wrong successor
        fp_tgt += in_valid           # annotated target, wrong predecessor

    # THE DECISIVE TEST for raising MOTION_RELINK_LEARNED_BONUS.
    # The relinker minimises a distance-dominated cost. For every swap, compare
    # the length of the CORRECT edge against the length of the edge we actually
    # took from that source. If the wrong link is SHORTER, distance is actively
    # choosing wrong and only a stronger non-geometric signal can fix it.
    pos = {int(r[0]): r[2:5] * SCALE for r in pred.nodes}
    shorter = longer = 0
    for gs, gtg in g.edges:
        ps, pt = gt_to_pred.get(int(gs)), gt_to_pred.get(int(gtg))
        if ps is None or pt is None or (ps, pt) in pred_edges:
            continue
        taken = pred_out.get(ps)
        if not taken:
            continue
        d_correct = float(np.linalg.norm(pos[ps] - pos[pt]))
        d_taken = min(float(np.linalg.norm(pos[ps] - pos[q])) for q in taken if q in pos)
        shorter += d_taken < d_correct
        longer += d_taken >= d_correct

    return {
        "dataset": ds, "gt_edges": n_gt,
        "oracle": both_matched / n_gt if n_gt else float("nan"),
        "actual": tp / (tp + fp + (n_gt - tp)) if n_gt else float("nan"),
        "tp": tp, "det_fn": det_fn, "assoc_fn": assoc_fn, "swap_fn": swap_fn,
        "fp": fp, "fp_src": fp_src, "fp_tgt": fp_tgt,
        "wrong_shorter": shorter, "wrong_longer": longer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", nargs="+")
    args = ap.parse_args()
    for s in args.submissions:
        p = Path(s)
        preds = load_submission(p)
        rows = [analyse(ds, pr) for ds, pr in sorted(preds.items())]
        print(f"\n===== {p.parent.name}/{p.name} =====")
        print(f"{'volume':>16} {'gtE':>6} {'ORACLE':>7} {'actual':>7} {'gap':>8} "
              f"{'tp':>5} {'detFN':>6} {'assFN':>6} {'swap':>5} {'FP':>5} {'FPsrc':>6} {'FPtgt':>6}")
        for r in rows:
            print(f"{r['dataset']:>16} {r['gt_edges']:>6} {r['oracle']:>7.4f} {r['actual']:>7.4f} "
                  f"{r['oracle']-r['actual']:>+8.4f} {r['tp']:>5} {r['det_fn']:>6} "
                  f"{r['assoc_fn']:>6} {r['swap_fn']:>5} {r['fp']:>5} {r['fp_src']:>6} {r['fp_tgt']:>6}")
        mo = np.mean([r['oracle'] for r in rows]); ma = np.mean([r['actual'] for r in rows])
        print(f"\n  MACRO oracle (detection ceiling) {mo:.4f}")
        print(f"  MACRO actual  edge_J             {ma:.4f}")
        print(f"  ASSOCIATION HEADROOM             {mo-ma:+.4f}   "
              f"= {100*(mo-ma)/(1-ma):.0f}% of all remaining loss")
        tot = {k: sum(r[k] for r in rows)
               for k in ("det_fn","assoc_fn","swap_fn","fp","wrong_shorter","wrong_longer")}
        print(f"  detection-FN {tot['det_fn']}   association-FN {tot['assoc_fn']} "
              f"(swaps {tot['swap_fn']})   FP {tot['fp']}")
        w = tot['wrong_shorter'] + tot['wrong_longer']
        if w:
            print(f"  swaps where the WRONG link is SHORTER than the correct one: "
                  f"{tot['wrong_shorter']}/{w} = {tot['wrong_shorter']/w:.0%}")


if __name__ == "__main__":
    main()
