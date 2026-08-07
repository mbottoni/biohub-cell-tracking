#!/usr/bin/env python3
"""Can a DIVISION-AWARE assignment recover the divisions the 1-1 Hungarian cannot?

24.6 established the structural fact: `motion_relink_edges` is a strict 1-1
matching, so over all 116,134 shipped edges **no node has in-degree >= 2 and a
source can never be given two children**. Every division we output comes from the
post-hoc scavenge (`add_safe_divisions_postlink`), none from the assignment.

26.3 then measured, at scale, where the division FNs actually go:

    FN-reachable at a 12 um gate   9/25 (36%)
    FN-consumed                    8/25 (32%)   <-- the daughter WAS detected, but
                                                    the 1-1 matching gave her to a
                                                    different parent
    FN-daughter (not detected)     5/25 (20%)

The FN-consumed bucket is unreachable by ANY gate or selector bolted onto the
existing matching -- it is caused by the 1-1 constraint itself. 34.3 priced
"division-aware assignment" at +0.0123 and filed it as "requires rewriting the
assignment". 36's replay means we now own `assign_pass`, so it costs minutes.

THE FORMULATION. Each source is duplicated: row `i` is its ordinary link, row
`i + S` is a "second child" at `cost + DIV_PENALTY_UM`. The Hungarian then solves a
degree-constrained matching in which a source may take two targets, paying a
penalty for the second. A sister sits ~10.6 um from her sibling (15) and the parent
lies between them, so where a rival source stole a daughter the parent is usually
the closer claimant and should win her back.

DISCIPLINE.
* DIV_PENALTY = inf must reproduce the replay baseline EXACTLY. That is the
  acceptance test, not a formality (23.2).
* Scored end-to-end with the real metric, never on division count (38.5).
* Watch edge FP, not just divisions: 18.3's ~40:1 asymmetry means a change that
  fixes every division and corrupts 2% of correct links nets negative.

CAVEAT, stated up front: the replay emits NO divisions at baseline (div_J = 0.0000)
because a 1-1 matching cannot, whereas the shipped pipeline scores div_J 0.0308 via
the post-hoc scavenge. So this measures whether a division-aware ASSIGNMENT creates
value over a division-free assignment. It does NOT by itself establish that it beats
the existing scavenge -- that needs the two in the same run.

    python scripts/local/division_aware_assign.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    SCALE, Graph, evaluate, per_sample_metrics, summarise,
)
from relink_replay import (  # noqa: E402
    MAX_FRAME_NODES, RELAXED_UM, TIGHT_UM, VELOCITY_WEIGHT,
)
from localisation_doseresponse import load_submission  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def relink_divaware(pos, t_of, div_penalty: float,
                    tight: float = TIGHT_UM, relaxed: float = RELAXED_UM):
    """Two-stage greedy relink in which a source may take a SECOND child.

    Identical to `relink_replay.relink_volume` when `div_penalty` is infinite --
    the duplicated rows are then all above the gate and never selected.
    """
    by_t = defaultdict(list)
    for n, t in t_of.items():
        by_t[t].append(n)
    for ids in by_t.values():
        ids.sort()
    if by_t and max(len(v) for v in by_t.values()) > MAX_FRAME_NODES:
        return []

    prev_pos: dict[int, np.ndarray] = {}
    out: list[tuple[int, int]] = []
    for t in sorted(by_t):
        srcs, tgts = by_t.get(t, []), by_t.get(t + 1, [])
        if not srcs or not tgts:
            continue
        un_t = set(tgts)
        n_child: dict[int, int] = defaultdict(int)
        finite = bool(np.isfinite(div_penalty))
        # Eligibility is the ONLY thing that differs from the shipped 1-1 pass.
        # With an infinite penalty a source is eligible only while childless, which
        # is exactly `s in unmatched_sources` -- so the inf arm must reproduce the
        # replay bit for bit. The first version of this filtered on
        # `s in un_s or n_child[s] == 1` in BOTH arms, which let the relaxed pass
        # hand a second child to an already-matched source at ordinary cost; the
        # inf arm then emitted 13,007 forks and failed its own acceptance test.
        max_children = 2 if finite else 1
        committed = []
        for gate in (tight, relaxed):
            ps = [s for s in srcs if n_child[s] < max_children]
            pt = [g for g in tgts if g in un_t]
            if not ps or not pt:
                continue
            S = np.stack([pos[s] for s in ps])
            T = np.stack([pos[g] for g in pt])
            P = np.stack([pos[s] + VELOCITY_WEIGHT * (pos[s] - prev_pos[s])
                          if s in prev_pos else pos[s] for s in ps])
            raw = np.linalg.norm(T[None, :, :] - S[:, None, :], axis=-1)
            motion = np.linalg.norm(T[None, :, :] - P[:, None, :], axis=-1)
            big = gate * 1000.0 + 1.0
            base = motion + 0.05 * raw
            base = np.where(raw > gate, big, base)

            if finite:
                second = np.where(raw > gate, big, base + div_penalty)
                for i, s in enumerate(ps):
                    if n_child[s] >= 1:
                        base[i, :] = big          # its ordinary slot is already spent
                cost = np.concatenate([base, second], axis=0)
            else:
                cost = base

            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] >= big:
                    continue
                s = ps[int(i) % len(ps)]
                g = pt[int(j)]
                if g not in un_t or n_child[s] >= max_children:
                    continue
                un_t.discard(g)
                n_child[s] += 1
                committed.append((s, g))
        for s, g in committed:
            out.append((s, g))
            prev_pos[g] = pos[s]
    return out


def run(job) -> dict:
    ds, nodes, penalty = job
    gt = load_geff(GT_ROOT / f"{ds}.geff")
    ids = nodes[:, 0].astype(np.int64)
    t_of = {int(i): int(t) for i, t in zip(ids, nodes[:, 1].astype(int))}
    pos = {int(i): nodes[k, 2:5] * SCALE for k, i in enumerate(ids)}
    edges = relink_divaware(pos, t_of, penalty)
    g = Graph(nodes=nodes,
              edges=np.asarray(edges, np.int64) if edges else np.zeros((0, 2), np.int64))
    m = per_sample_metrics(evaluate(gt.graph, g), gt.estimated_number_of_nodes)
    m["dataset"] = ds
    outdeg = defaultdict(int)
    for s, _ in edges:
        outdeg[s] += 1
    m["forks"] = sum(1 for v in outdeg.values() if v >= 2)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--penalties", type=float, nargs="*",
                    default=[float("inf"), 8.0, 6.0, 4.0, 2.0, 1.0])
    args = ap.parse_args()

    subs = load_submission(Path(args.submission))
    order = sorted(subs)
    print(f"{len(order)} volumes\n")
    print(f"{'div penalty':>12} {'forks':>7} {'edge_J':>8} {'adj':>8} {'div_J':>7} "
          f"{'SCORE':>8} {'delta':>9} {'eFP':>7} {'dTP':>5} {'dFP':>5} {'dFN':>5}")

    base = None
    for pen in args.penalties:
        with ProcessPoolExecutor() as ex:
            rows = list(ex.map(run, [(ds, subs[ds].nodes, pen) for ds in order]))
        agg = summarise(rows)
        if base is None:
            base = agg["score"]
        tag = "inf (baseline)" if not np.isfinite(pen) else f"{pen:.1f} um"
        print(f"{tag:>12} {sum(r['forks'] for r in rows):>7,} "
              f"{agg['edge_jaccard']:>8.4f} {agg['adj_edge_jaccard']:>8.4f} "
              f"{agg['division_jaccard']:>7.4f} {agg['score']:>8.4f} "
              f"{agg['score']-base:>+9.4f} {sum(r['edge_fp'] for r in rows):>7,} "
              f"{sum(r['division_tp'] for r in rows):>5} "
              f"{sum(r['division_fp'] for r in rows):>5} "
              f"{sum(r['division_fn'] for r in rows):>5}")

    print("\n  ACCEPTANCE TEST: the inf row must reproduce the replay baseline "
          "(edge_J 0.9046 / adj 0.8930).")
    print("  Watch edge FP, not fork count -- 18.3's ~40:1 asymmetry means fixing every")
    print("  division while corrupting 2% of correct links nets negative.")


if __name__ == "__main__":
    main()
