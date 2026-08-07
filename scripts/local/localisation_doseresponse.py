#!/usr/bin/env python3
"""Is ASSOCIATION blocked by localisation NOISE rather than by the cost or the gate?

STRATEGY 31.3 measured, on the 379 swaps, that the source->CORRECT-successor
distance is a median 9.37 um while the GT displacement of those same edges is
only 4.82 um.  Roughly HALF the distance that pushes a correct link outside
`MOTION_RELINK_TIGHT_UM` = 6.0 is therefore OUR OWN COORDINATE ERROR, not motion.

Every association lever so far moved the GATE or the COST and lost (33: widening
6->9->12 is monotonically catastrophic, because a wider gate admits far-but-WRONG
targets as readily as far-but-right ones).  Sharpening the COORDINATES is the one
intervention that is asymmetric: it moves correct targets inward without moving
wrong ones inward.  It has never been measured.

TWO experiments, because each alone is confounded:

  (A) NOISE INJECTION -- confound-free, gives the local slope.
      Add anisotropic Gaussian noise to EVERY node (per-axis sigma from 25's
      measured residual scatter, z 1.707 / y 0.943 / x 1.039 um), replay the
      relinker, score end to end.  All nodes degrade together, exactly as a real
      localisation change would improve them together.  The slope dScore/dRMS at
      the operating point is what a detector improvement would buy per um.

  (B) GT SNAP -- and it is a LOWER bound, not the optimistic reading it looks like.
      Shrink matched nodes toward their GT position by lambda.  Only ~annotated
      cells can be snapped, so a correct target becomes exact while an unannotated
      distractor stays noisy.  That looks like an unfair advantage; it is the
      reverse.  A real detector improvement sharpens distractors too, and
      distractor noise can only ever move a WRONG target closer to the source by
      chance, manufacturing false competition.  Removing it helps.  So uniform
      perfect localisation is >= snap-matched-only.

  (C) PLACEBO -- move each matched node the SAME distance in a RANDOM direction.
      Controls for "does displacing annotated nodes help regardless of accuracy?".
      Measured: -0.0140 against snap's +0.0305 at identical displacement.

METHOD DISCIPLINE (35.3, 38.5).  Perturbed coordinates are used ONLY to make the
LINKING decision.  Scoring always uses the ORIGINAL positions, so node recall,
node matching and the node-budget ratio are held byte-fixed and the ONLY thing
that can move is which edges the assignment picks.  And the readout is
end-to-end score on real GT, never a pairwise metric -- 37.3 and 38.4 both showed
per-item accuracy does not convert here.

    python scripts/local/localisation_doseresponse.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    MAX_MATCH_DISTANCE_UM, SCALE, Graph, evaluate, match_nodes,
    per_sample_metrics, summarise,
)
from relink_replay import relink_volume  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")

# 25 / row 47: measured per-axis residual scatter of predicted vs GT centroids.
RESIDUAL_SIGMA = np.array([1.707, 0.943, 1.039])


def load_submission(path: Path) -> dict[str, Graph]:
    nodes: dict[str, list] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["row_type"] == "node":
                nodes[r["dataset"]].append(
                    (int(r["node_id"]), int(r["t"]),
                     float(r["z"]), float(r["y"]), float(r["x"])))
    return {ds: Graph(nodes=np.asarray(v, dtype=np.float64),
                      edges=np.zeros((0, 2), dtype=np.int64))
            for ds, v in nodes.items()}


def run_volume(job) -> dict:
    """Replay the relinker on perturbed coords, score on ORIGINAL coords."""
    ds, nodes, mode, level, seed = job
    gt = load_geff(GT_ROOT / f"{ds}.geff")

    ids = nodes[:, 0].astype(np.int64)
    t_of = {int(i): int(tt) for i, tt in zip(ids, nodes[:, 1])}
    pos_um = nodes[:, 2:5] * SCALE                      # (N,3) micrometres

    work = pos_um.copy()
    if mode == "noise" and level > 0:
        rng = np.random.default_rng(abs(hash((ds, seed))) % (2**32))
        work = work + rng.normal(0.0, 1.0, work.shape) * (RESIDUAL_SIGMA * level)
        shift = float(np.sqrt(np.mean(np.sum((work - pos_um) ** 2, axis=1))))
    elif mode in ("snap", "placebo") and level > 0:
        # `placebo` moves each matched node by the SAME distance as `snap` would,
        # but in a random direction. It is the control for "does merely displacing
        # annotated nodes help, regardless of whether the displacement is correct?"
        pred_g = Graph(nodes=nodes, edges=np.zeros((0, 2), dtype=np.int64))
        matched = match_nodes(gt.graph, pred_g, max_distance=MAX_MATCH_DISTANCE_UM,
                              scale=SCALE)
        gid_row = {int(g[0]): k for k, g in enumerate(gt.graph.nodes)}
        row_of = {int(i): k for k, i in enumerate(ids)}
        gt_um = gt.graph.nodes[:, 2:5] * SCALE
        rng = np.random.default_rng(abs(hash((ds, seed, mode))) % (2**32))
        for pid, gid in matched.items():
            r = row_of.get(int(pid))
            if r is None:
                continue
            delta = level * (gt_um[gid_row[int(gid)]] - pos_um[r])
            if mode == "placebo":
                u = rng.normal(size=3)
                delta = u / np.linalg.norm(u) * np.linalg.norm(delta)
            work[r] = pos_um[r] + delta
        shift = float(np.sqrt(np.mean(np.sum((work - pos_um) ** 2, axis=1))))
    else:
        shift = 0.0

    pos = {int(i): work[k] for k, i in enumerate(ids)}
    edges = relink_volume(pos, t_of)

    scored = Graph(nodes=nodes,                                  # ORIGINAL coords
                   edges=(np.asarray(edges, dtype=np.int64)
                          if edges else np.zeros((0, 2), dtype=np.int64)))
    m = per_sample_metrics(evaluate(gt.graph, scored),
                           gt.estimated_number_of_nodes)
    m["dataset"] = ds
    m["shift_rms"] = shift
    return m


def sweep(subs: dict[str, Graph], mode: str, levels, seed: int = 0):
    out = []
    with ProcessPoolExecutor() as ex:
        for level in levels:
            jobs = [(ds, g.nodes, mode, level, seed) for ds, g in sorted(subs.items())]
            rows = list(ex.map(run_volume, jobs))
            agg = summarise(rows)
            agg["level"] = level
            agg["shift_rms"] = float(np.mean([r["shift_rms"] for r in rows]))
            agg["rows"] = rows
            out.append(agg)
            print(f"  {mode:>5} {level:>5.2f}  added/removed RMS {agg['shift_rms']:>6.3f} um"
                  f"   edge_J {agg['edge_jaccard']:.4f}  adj {agg['adj_edge_jaccard']:.4f}"
                  f"  div_J {agg['division_jaccard']:.4f}  SCORE {agg['score']:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--snap", type=float, nargs="*", default=[0.5, 1.0])
    args = ap.parse_args()

    subs = load_submission(Path(args.submission))
    print(f"{len(subs)} volumes, {sum(len(g.nodes) for g in subs.values()):,} nodes\n")

    print("(A) NOISE INJECTION -- all nodes perturbed together (confound-free)")
    noise = sweep(subs, "noise", args.noise)
    base = noise[0]

    print("\n(B) GT SNAP on matched nodes -- OPTIMISTIC upper bound")
    snap = sweep(subs, "snap", args.snap)

    print("\n===== dose-response, relative to the replay baseline =====")
    print(f"{'arm':>18} {'RMS change':>12} {'edge_J':>8} {'adj':>8} {'SCORE':>8} {'delta':>9}")
    for r in noise:
        tag = "baseline" if r["level"] == 0 else f"noise x{r['level']:g}"
        print(f"{tag:>18} {r['shift_rms']:>+11.3f}u {r['edge_jaccard']:>8.4f} "
              f"{r['adj_edge_jaccard']:>8.4f} {r['score']:>8.4f} "
              f"{r['score']-base['score']:>+9.4f}")
    for r in snap:
        print(f"{'snap l=' + format(r['level'], 'g'):>18} {-r['shift_rms']:>+11.3f}u "
              f"{r['edge_jaccard']:>8.4f} {r['adj_edge_jaccard']:>8.4f} {r['score']:>8.4f} "
              f"{r['score']-base['score']:>+9.4f}")

    hurt = [r for r in noise if r["level"] > 0]
    if hurt:
        slopes = [(r["score"] - base["score"]) / r["shift_rms"] for r in hurt]
        print(f"\n  local slope dSCORE/dRMS from injection: "
              f"{np.mean(slopes):+.4f} per um  (arms: "
              f"{', '.join(f'{s:+.4f}' for s in slopes)})")
        print("  ^ if a detector change cuts localisation RMS by d um, this is the")
        print("    first-order score it is worth -- assuming the response is locally linear,")
        print("    which the multiple injection levels above are there to check.")


if __name__ == "__main__":
    main()
