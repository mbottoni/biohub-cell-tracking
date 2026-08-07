#!/usr/bin/env python3
"""STRATEGY 44.4 item 1 -- replay the public 0.915 line's association deltas locally.

The yusuketogashi "Biohub 159B/162" line (the 104-team 0.915 shelf) differs from our
`d4edge` in the relink cost only through two mechanisms, both reimplemented here
faithfully from the pulled notebook (scratchpad pub3rd, cells 6/11/15):

  RANKER   cost = motion + 0.05*raw - BONUS*(0.85*ranker_prob + 0.15*primary_prob)
           where ranker_prob is the public 22-feature MLP
           (pilkwang/biohub-local-association-ranker-unet300-v1, hidden 64,
           val score 0.9777 -- NOTE: numerically equal to our 37.3 distance-only
           AUC on contested pairs, so scepticism is pre-registered).
  LOOKAHEAD  subtract up to 0.20 um when the candidate edge has a smooth
           continuation into t+2 (residual < 4 um between successive velocities).

⚠ REPLAY DEGRADATIONS, stated in advance (same family 36 survived at 99.7%):
  * primary_prob / has_learned_edge / target_best_next_prob come from the
    transformer's learned probs, unavailable offline -> 0.0. The 0.15 primary
    retain term is therefore inert here; the ranker sees zeros for 3/22 features.
  * context degrees come from the SHIPPED final edges, not the ILP raw edges.

PRE-REGISTERED PREDICTION (before first run): per 28 (a <=1 um bonus is inert)
and 37 (a learned cost does not move the argmin), all arms land within +-0.003
of baseline on the 18-thick-volume robust mean. If the ranker arm exceeds
+0.005 robust, that contradicts 37 and the mechanism must be understood before
porting anything.

    .venv/bin/python scripts/local/replay_pub162.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    SCALE, Graph, evaluate, per_sample_metrics, summarise,
)

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
RANKER_DIR = Path(
    "/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
    "5c5c110c-b819-4168-a812-26fe17d15065/scratchpad/ranker")

TIGHT_UM, RELAXED_UM = 6.0, 10.0
VELOCITY_WEIGHT = 0.5
MAX_FRAME_NODES = 2600
LEARNED_BONUS = 1.0
RANKER_FULL_WEIGHT = 0.85
PRIMARY_RETAIN_WEIGHT = 0.15          # inert offline: primary_prob == 0
LOOKAHEAD_MAX_ACCEL_UM = 4.0
LOOKAHEAD_MAX_BONUS = 0.20

FEATURES = [
    "edge_prob", "has_learned_edge", "source_out_degree", "target_in_degree",
    "source_frame_count", "target_frame_count", "source_density_7um",
    "target_density_7um", "candidate_rank_dist", "candidate_count",
    "edge_dz_um", "edge_dy_um", "edge_dx_um", "edge_dist_um", "edge_xy_um",
    "edge_abs_z_um", "motion_dist_um", "motion_gain_um", "source_has_prev",
    "target_has_next", "target_best_next_prob", "t_norm",
]


class RankerMLP:
    """Loads the public checkpoint exactly as the notebook's inferred-MLP path does."""

    def __init__(self, path: Path):
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = None
        if isinstance(payload, dict):
            for key in ("model_state_dict", "state_dict", "weights"):
                v = payload.get(key)
                if isinstance(v, dict):
                    state = v
                    break
            if state is None and any(hasattr(v, "shape") for v in payload.values()):
                state = payload
        if state is None:
            raise RuntimeError("no state dict in ranker checkpoint")
        witems = sorted(
            ((k, v) for k, v in state.items() if k.endswith(".weight") and v.ndim == 2),
            key=lambda kv: [int(p) if p.isdigit() else p
                            for p in __import__("re").split(r"(\d+)", kv[0])])
        self.layers = []
        for wk, w in witems:
            b = state.get(wk[:-len(".weight")] + ".bias")
            self.layers.append((w.numpy().astype(np.float64),
                                None if b is None else b.numpy().astype(np.float64)))
        self.input_dim = self.layers[0][0].shape[1]
        # The artifact ships no scaler stats -> identity normalisation (as on Kaggle).
        self.mean = np.zeros(self.input_dim)
        self.std = np.ones(self.input_dim)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + (b if b is not None else 0.0)
            if i + 1 < len(self.layers):
                h = np.maximum(h, 0.0)
        if h.shape[1] == 1:
            v = h[:, 0]
            return 1.0 / (1.0 + np.exp(-v)) if (v.min() < 0 or v.max() > 1) else v
        e = np.exp(h - h.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, 1]


_RANKER: RankerMLP | None = None


def get_ranker() -> RankerMLP:
    global _RANKER
    if _RANKER is None:
        _RANKER = RankerMLP(RANKER_DIR / "model" / "local_association_ranker.pt")
    return _RANKER


def build_context(ids, pos, t_of, shipped_edges):
    in_deg: dict[int, int] = defaultdict(int)
    out_deg: dict[int, int] = defaultdict(int)
    for s, g in shipped_edges:
        out_deg[s] += 1
        in_deg[g] += 1
    by_t: dict[int, list[int]] = defaultdict(list)
    for n, t in t_of.items():
        by_t[t].append(n)
    for v in by_t.values():
        v.sort()
    dens: dict[int, float] = {}
    for t, nn in by_t.items():
        pts = np.stack([pos[n] for n in nn])
        tree = cKDTree(pts)
        counts = tree.query_ball_point(pts, r=7.0, return_length=True)
        for n, c in zip(nn, counts):
            dens[n] = float(max(0, int(c) - 1))
    return {
        "in_deg": in_deg, "out_deg": out_deg, "by_t": by_t, "dens": dens,
        "max_t": max(by_t) if by_t else 1,
    }


def relink_pub(pos, t_of, ctx, use_ranker: bool, use_lookahead: bool,
               ranker_w: float = RANKER_FULL_WEIGHT,
               la_bonus: float = LOOKAHEAD_MAX_BONUS):
    by_t = ctx["by_t"]
    if by_t and max(len(v) for v in by_t.values()) > MAX_FRAME_NODES:
        return []
    ranker = get_ranker() if use_ranker else None
    prev_pos: dict[int, np.ndarray] = {}
    out: list[tuple[int, int]] = []

    for t in sorted(by_t):
        srcs, tgts = by_t.get(t, []), by_t.get(t + 1, [])
        if not srcs or not tgts:
            continue
        un_s, un_t = set(srcs), set(tgts)
        committed = []
        for gate in (TIGHT_UM, RELAXED_UM):
            ps = [s for s in srcs if s in un_s]
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
            valid = raw <= gate
            cost = np.where(valid, motion + 0.05 * raw, big)

            if use_ranker and valid.any():
                rows, cols = np.nonzero(valid)
                # candidate_rank_dist: 1-based rank of raw distance within row.
                feats = np.zeros((len(rows), 22))
                for k, (i, j) in enumerate(zip(rows, cols)):
                    s, g = ps[i], pt[j]
                    vc = np.nonzero(valid[i])[0]
                    order = vc[np.lexsort((np.array([pt[c] for c in vc]), raw[i, vc]))]
                    rank = int(np.nonzero(order == j)[0][0]) + 1
                    d = T[j] - S[i]
                    feats[k] = [
                        0.0,                              # edge_prob (unavailable)
                        0.0,                              # has_learned_edge
                        ctx["out_deg"].get(s, 0), ctx["in_deg"].get(g, 0),
                        len(by_t.get(t, [])), len(by_t.get(t + 1, [])),
                        ctx["dens"].get(s, 0.0), ctx["dens"].get(g, 0.0),
                        rank, len(vc),
                        d[0], d[1], d[2],
                        raw[i, j], float(np.hypot(d[1], d[2])), abs(d[0]),
                        motion[i, j], raw[i, j] - motion[i, j],
                        float(ctx["in_deg"].get(s, 0) > 0),
                        float(ctx["out_deg"].get(g, 0) > 0),
                        0.0,                              # target_best_next_prob
                        t / max(1, ctx["max_t"]),
                    ]
                probs = ranker.predict_proba(feats)
                for k, (i, j) in enumerate(zip(rows, cols)):
                    evidence = ranker_w * probs[k]      # primary retain term == 0
                    cost[i, j] -= LEARNED_BONUS * evidence

            if use_lookahead and valid.any():
                nxt = by_t.get(t + 2, [])
                if nxt:
                    NX = np.stack([pos[n] for n in nxt])
                    rows, cols = np.nonzero(valid)
                    for i, j in zip(rows, cols):
                        vel = T[j] - S[i]
                        nv = NX - T[j]
                        keep = np.linalg.norm(nv, axis=1) <= RELAXED_UM
                        if not keep.any():
                            continue
                        res = float(np.min(np.linalg.norm(nv[keep] - vel, axis=1)))
                        support = max(0.0, 1.0 - res / LOOKAHEAD_MAX_ACCEL_UM)
                        cost[i, j] -= la_bonus * support

            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] >= big:
                    continue
                s, g = ps[int(i)], pt[int(j)]
                if s not in un_s or g not in un_t:
                    continue
                un_s.discard(s)
                un_t.discard(g)
                committed.append((s, g))
        for s, g in committed:
            out.append((s, g))
            prev_pos[g] = pos[s]
    return out


def run_volume(job):
    ds, nodes, shipped_edges, arm = job
    gt = load_geff(GT_ROOT / f"{ds}.geff")
    ids = nodes[:, 0].astype(np.int64)
    t_of = {int(i): int(tt) for i, tt in zip(ids, nodes[:, 1])}
    pos_um = nodes[:, 2:5] * SCALE
    pos = {int(i): pos_um[k] for k, i in enumerate(ids)}
    ctx = build_context(ids, pos, t_of, shipped_edges)

    kw = dict(use_ranker=False, use_lookahead=False)
    if arm == "ranker":
        kw = dict(use_ranker=True, use_lookahead=False)
    elif arm == "lookahead":
        kw = dict(use_ranker=False, use_lookahead=True)
    elif arm == "ranker+la":
        kw = dict(use_ranker=True, use_lookahead=True)
    edges = relink_pub(pos, t_of, ctx, **kw)

    scored = Graph(nodes=nodes,
                   edges=(np.asarray(edges, dtype=np.int64)
                          if edges else np.zeros((0, 2), dtype=np.int64)))
    m = per_sample_metrics(evaluate(gt.graph, scored), gt.estimated_number_of_nodes)
    m["dataset"] = ds
    m["gt_edges"] = int(len(gt.graph.edges))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--arms", nargs="*",
                    default=["base", "ranker", "lookahead", "ranker+la"])
    args = ap.parse_args()

    nodes_by_ds: dict[str, list] = defaultdict(list)
    edges_by_ds: dict[str, list] = defaultdict(list)
    with open(args.submission) as f:
        for r in csv.DictReader(f):
            if r["row_type"] == "node":
                nodes_by_ds[r["dataset"]].append(
                    (int(r["node_id"]), int(r["t"]),
                     float(r["z"]), float(r["y"]), float(r["x"])))
            else:
                edges_by_ds[r["dataset"]].append(
                    (int(r["source_id"]), int(r["target_id"])))
    subs = {ds: np.asarray(v, dtype=np.float64) for ds, v in nodes_by_ds.items()}
    print(f"{len(subs)} volumes")

    results = {}
    per_vol = {}
    for arm in args.arms:
        jobs = [(ds, subs[ds], edges_by_ds[ds], arm) for ds in sorted(subs)]
        with ProcessPoolExecutor() as ex:
            rows = list(ex.map(run_volume, jobs))
        agg = summarise(rows)
        results[arm] = agg
        per_vol[arm] = {r["dataset"]: r for r in rows}
        thick = [r for r in rows if r["gt_edges"] >= 200]
        robust = float(np.mean([r["adj_edge_jaccard"] for r in thick]))
        agg["robust_adj"] = robust
        print(f"  {arm:>10}  edge_J {agg['edge_jaccard']:.4f}  adj {agg['adj_edge_jaccard']:.4f}"
              f"  div_J {agg['division_jaccard']:.4f}  SCORE {agg['score']:.4f}"
              f"  robust18 {robust:.4f}")

    base = results.get("base")
    if base:
        print("\n===== deltas vs geometric baseline =====")
        print(f"{'arm':>10} {'d(score)':>9} {'d(robust adj)':>14}")
        for arm in args.arms:
            if arm == "base":
                continue
            r = results[arm]
            print(f"{arm:>10} {r['score']-base['score']:>+9.4f} "
                  f"{r['robust_adj']-base['robust_adj']:>+14.4f}")


if __name__ == "__main__":
    main()
