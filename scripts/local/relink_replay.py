#!/usr/bin/env python3
"""STRATEGY 35.2 phase 0 -- reimplement the relinker locally and prove it faithful.

`assign_pass` inside `motion_relink_edges` consumes only node positions, so it can
be replayed offline over the node set in a downloaded `submission.csv`. If the
replay reproduces the shipped edges, every future cost function can be evaluated
in seconds instead of ~4 GPU-h.

Faithful reimplementation of the shipped logic (cell 11):

    times = sorted frames
    for t:
      unmatched_sources = nodes at t ; unmatched_targets = nodes at t+1
      for gate in (TIGHT_UM, RELAXED_UM):          # two-stage GREEDY (31.6)
          cost[i,j] = motion + 0.05*raw - BONUS*prob   , excluded if raw > gate
          linear_sum_assignment(cost)
          commit matches, remove from unmatched
      predecessor_position_um[target] = position[source]     # chains forward
    predicted = source_pos + VELOCITY_WEIGHT * (source_pos - prev_pos)

⚠ KNOWN SOURCES OF DISAGREEMENT, stated before measuring so the gate is honest.
The submission's node set is NOT the one the relinker saw:

  * `linefit_smooth_output_graph` MOVES node positions after relinking
    (`linefit_smoothed_nodes` ~= every node), so the coordinates we replay on are
    smoothed versions of the inputs the relinker actually used;
  * gap-closing ADDED nodes and the short-track filter REMOVED them
    (`short_track_nodes_removed` = 22,773 over the 24 volumes);
  * safe divisions added second-child edges the relinker never produced;
  * `learned_edge_probs` are internal and unavailable locally, so BONUS*prob is
    dropped -- bounded by BONUS=1.0 x prob<=1 = 1 um, and §28 showed the term is
    inert across BONUS 1..17.

So perfect agreement is impossible by construction. The gate is whether agreement
is high enough that RELATIVE comparisons between cost functions are meaningful.

    python scripts/local/relink_replay.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SCALE = np.array([1.625, 0.40625, 0.40625])
TIGHT_UM, RELAXED_UM = 6.0, 10.0
VELOCITY_WEIGHT = 0.5
MAX_FRAME_NODES = 2600


def geometric_cost(src_pos, tgt_pos, pred_pos, raw, gate):
    """The shipped cost: motion + 0.05*raw, with BONUS*prob dropped (see docstring)."""
    motion = np.linalg.norm(tgt_pos[None, :, :] - pred_pos[:, None, :], axis=-1)
    return motion + 0.05 * raw


def relink_volume(pos: dict[int, np.ndarray], t_of: dict[int, int],
                  cost_fn=geometric_cost,
                  tight: float = TIGHT_UM, relaxed: float = RELAXED_UM):
    by_t: dict[int, list[int]] = defaultdict(list)
    for n, t in t_of.items():
        by_t[t].append(n)
    for ids in by_t.values():
        ids.sort()
    if by_t and max(len(v) for v in by_t.values()) > MAX_FRAME_NODES:
        return []                                   # shipped: skip whole volume

    prev_pos: dict[int, np.ndarray] = {}
    out: list[tuple[int, int]] = []
    for t in sorted(by_t):
        srcs, tgts = by_t.get(t, []), by_t.get(t + 1, [])
        if not srcs or not tgts:
            continue
        un_s, un_t = set(srcs), set(tgts)
        committed = []
        for gate in (tight, relaxed):
            ps = [s for s in srcs if s in un_s]
            pt = [g for g in tgts if g in un_t]
            if not ps or not pt:
                continue
            S = np.stack([pos[s] for s in ps])
            T = np.stack([pos[g] for g in pt])
            P = np.stack([pos[s] + VELOCITY_WEIGHT * (pos[s] - prev_pos[s])
                          if s in prev_pos else pos[s] for s in ps])
            raw = np.linalg.norm(T[None, :, :] - S[:, None, :], axis=-1)
            big = gate * 1000.0 + 1.0
            cost = cost_fn(S, T, P, raw, gate)
            cost = np.where(raw > gate, big, cost)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] >= big:
                    continue
                s, g = ps[int(i)], pt[int(j)]
                if s not in un_s or g not in un_t:
                    continue
                un_s.discard(s); un_t.discard(g)
                committed.append((s, g))
        for s, g in committed:
            out.append((s, g))
            prev_pos[g] = pos[s]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    args = ap.parse_args()

    df = pd.read_csv(args.submission)
    nodes = df[df.row_type == "node"]
    edges = df[df.row_type == "edge"]

    print(f"{'volume':>16} {'nodes':>7} {'shipped':>8} {'replay':>8} {'matched':>8} {'agree':>7}")
    tot_ship = tot_rep = tot_match = 0
    for ds, gn in sorted(nodes.groupby("dataset")):
        ge = edges[edges.dataset == ds]
        pos = {int(r.node_id): np.array([r.z, r.y, r.x], float) * SCALE
               for r in gn.itertuples()}
        t_of = dict(zip(gn.node_id.astype(int), gn.t.astype(int)))
        shipped = set(zip(ge.source_id.astype(int).tolist(),
                          ge.target_id.astype(int).tolist()))
        rep = set(relink_volume(pos, t_of))
        inter = shipped & rep
        tot_ship += len(shipped); tot_rep += len(rep); tot_match += len(inter)
        print(f"{ds:>16} {len(pos):>7,} {len(shipped):>8,} {len(rep):>8,} "
              f"{len(inter):>8,} {len(inter)/max(len(shipped),1):>7.1%}")

    print(f"\n{'TOTAL':>16} {'':>7} {tot_ship:>8,} {tot_rep:>8,} {tot_match:>8,} "
          f"{tot_match/max(tot_ship,1):>7.1%}")
    a = tot_match / max(tot_ship, 1)
    print("\nPHASE 0 GATE (§35.2):")
    if a >= 0.90:
        print(f"  PASS -- {a:.1%} of shipped edges reproduced. The replay is faithful enough")
        print("  that relative comparisons between cost functions are meaningful.")
    elif a >= 0.75:
        print(f"  MARGINAL -- {a:.1%}. Usable for coarse ranking, but the disagreement is")
        print("  large enough that small deltas cannot be trusted. Investigate before phase 2.")
    else:
        print(f"  FAIL -- only {a:.1%} reproduced. We do not understand the stage well enough")
        print("  to replace it. Per §35.4 the plan STOPS here rather than proceeding on a")
        print("  broken testbed.")


if __name__ == "__main__":
    main()
