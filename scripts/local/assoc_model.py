#!/usr/bin/env python3
"""STRATEGY 35 phases 2-3 -- learn the relink COST and gate it on held-out volumes.

§33 closed association to every knob (bonus, edge threshold, velocity weight,
relink radius, neighbourhood flow: five axes, shipped value wins each time). §35.1
argued the reason a better edge *probability* cannot help is architectural — the
relinker discards 100% of learned edges and §28 showed the `BONUS` channel is
inert — so the model has to **replace the cost and own the assignment**.

§36 built and validated the testbed: replaying `assign_pass` locally reproduces
99.7% of shipped edges and predicts the real `tight9`/`tight12` runs within 13%.
So a cost function can now be evaluated end-to-end, with the real metric, in
minutes.

WHAT THE MODEL SEES THAT GEOMETRY DOES NOT (§35.5):
  * competition structure -- how many sources want this target, and the margin to
    the runner-up on both sides. The Hungarian resolves competition globally but
    the COST is blind to it.
  * density-normalised distance -- §24.1 ranked `parent_over_spacing` 2nd of 14
    features for divisions, and §29 showed fixed um thresholds fail on a corpus
    spanning 150x in coverage.
  * mutual-nearest-neighbour structure and rank, on both sides.

EVALUATION DISCIPLINE (§35.3), all three enforced here:
  1. Trained ONLY on the 16 evalset volumes on the honest-split TRAIN side;
     scored ONLY on the 8 on the TEST side.
  2. Scored END-TO-END through the Hungarian with the real metric -- never on
     pairwise AUC. §33 is the cautionary case: candidate availability improved
     and the score fell 0.021 because the assignment got worse.
  3. Reports edge FP alongside, because of the ~40:1 correct-to-swap asymmetry
     (§18.3): fixing every swap while corrupting 2% of correct links nets zero.

    python scripts/local/assoc_model.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/local"))

from relink_replay import relink_volume, geometric_cost  # noqa: E402
from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    SCALE, Graph, evaluate, match_nodes, per_sample_metrics, summarise,
)

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
SUB = REPO / "submissions/evalset_d4edge/submission.csv"
GATE = 10.0                      # widest gate the shipped relinker uses

FEATURES = ["raw", "motion", "raw_over_spacing", "motion_over_spacing", "spacing",
            "src_speed", "rank_fwd", "rank_bwd", "n_cand", "n_src",
            "margin_fwd", "margin_bwd", "mutual_nn"]


def frame_features(S, T, P, raw, gate=GATE):
    """Features for every (source, target) pair inside `gate`.

    ONE implementation, used by both training and inference, so there is no
    train/serve skew. Returns (idx_i, idx_j, X).
    """
    ii, jj = np.where(raw <= gate)
    if len(ii) == 0:
        return ii, jj, np.zeros((0, len(FEATURES)))

    motion = np.linalg.norm(T[None, :, :] - P[:, None, :], axis=-1)
    # local scale: median nearest-neighbour distance among targets in this frame
    if len(T) > 1:
        d = np.linalg.norm(T[None, :, :] - T[:, None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        spacing = float(np.median(d.min(axis=1)))
    else:
        spacing = 1.0
    spacing = max(spacing, 1e-6)

    big = np.where(raw <= gate, raw, np.inf)
    rank_fwd = np.argsort(np.argsort(big, axis=1), axis=1)          # target's rank for a source
    rank_bwd = np.argsort(np.argsort(big, axis=0), axis=0)          # source's rank for a target
    n_cand = (raw <= gate).sum(axis=1)
    n_src = (raw <= gate).sum(axis=0)

    srt_f = np.sort(big, axis=1)
    best_f = srt_f[:, 0]
    second_f = srt_f[:, 1] if srt_f.shape[1] > 1 else np.full(len(S), np.inf)
    srt_b = np.sort(big, axis=0)
    best_b = srt_b[0, :]
    second_b = srt_b[1, :] if srt_b.shape[0] > 1 else np.full(len(T), np.inf)
    fin = lambda a: np.where(np.isfinite(a), a, gate * 2)

    src_speed = 2.0 * np.linalg.norm(P - S, axis=1)                 # P = S + 0.5*(S-prev)

    X = np.column_stack([
        raw[ii, jj],
        motion[ii, jj],
        raw[ii, jj] / spacing,
        motion[ii, jj] / spacing,
        np.full(len(ii), spacing),
        src_speed[ii],
        rank_fwd[ii, jj],
        rank_bwd[ii, jj],
        n_cand[ii],
        n_src[jj],
        raw[ii, jj] - fin(best_f)[ii],          # how much worse than this source's best
        raw[ii, jj] - fin(best_b)[jj],          # how much worse than this target's best
        ((rank_fwd[ii, jj] == 0) & (rank_bwd[ii, jj] == 0)).astype(float),
    ])
    return ii, jj, X


def iter_frames(pos, t_of):
    """Yield (t, src_ids, tgt_ids, S, T, P) reproducing the replay's frame loop.

    Uses the GEOMETRIC assignment to build the predecessor chain, so training
    features match what the model will see when it is first plugged in.
    """
    by_t = defaultdict(list)
    for n, t in t_of.items():
        by_t[t].append(n)
    for ids in by_t.values():
        ids.sort()
    prev = {}
    for t in sorted(by_t):
        srcs, tgts = by_t.get(t, []), by_t.get(t + 1, [])
        if not srcs or not tgts:
            continue
        S = np.stack([pos[s] for s in srcs])
        T = np.stack([pos[g] for g in tgts])
        P = np.stack([pos[s] + 0.5 * (pos[s] - prev[s]) if s in prev else pos[s] for s in srcs])
        yield t, srcs, tgts, S, T, P
        raw = np.linalg.norm(T[None, :, :] - S[:, None, :], axis=-1)
        from scipy.optimize import linear_sum_assignment
        c = np.where(raw > 6.0, 1e6, np.linalg.norm(T[None, :, :] - P[:, None, :], axis=-1) + 0.05 * raw)
        r, cc = linear_sum_assignment(c)
        for i, j in zip(r, cc):
            if c[i, j] < 1e6:
                prev[tgts[int(j)]] = S[int(i)]


def load_volume(ds, gn):
    pos = {int(r.node_id): np.array([r.z, r.y, r.x], float) * SCALE for r in gn.itertuples()}
    t_of = dict(zip(gn.node_id.astype(int), gn.t.astype(int)))
    return pos, t_of


def build_training(vols, subs):
    X_all, y_all = [], []
    for ds in vols:
        gn = subs[ds]
        pos, t_of = load_volume(ds, gn)
        gt = load_geff(GT_ROOT / f"{ds}.geff")
        nd = np.asarray([[r.node_id, r.t, r.z, r.y, r.x] for r in gn.itertuples()], dtype=np.float64)
        m = match_nodes(gt.graph, Graph(nodes=nd, edges=np.zeros((0, 2), np.int64)),
                        max_distance=7.0, scale=SCALE)
        g2p = {g: p for p, g in m.items()}
        true_edges = set()
        for ga, gb in gt.graph.edges:
            pa, pb = g2p.get(int(ga)), g2p.get(int(gb))
            if pa is not None and pb is not None:
                true_edges.add((pa, pb))
        labelled_src = {a for a, _ in true_edges}
        for t, srcs, tgts, S, T, P in iter_frames(pos, t_of):
            raw = np.linalg.norm(T[None, :, :] - S[:, None, :], axis=-1)
            ii, jj, X = frame_features(S, T, P, raw)
            if not len(ii):
                continue
            keep, lab = [], []
            for k, (i, j) in enumerate(zip(ii, jj)):
                s = srcs[int(i)]
                if s not in labelled_src:
                    continue          # no GT for this source -> no label exists
                keep.append(k)
                lab.append(1 if (s, tgts[int(j)]) in true_edges else 0)
            if keep:
                X_all.append(X[keep]); y_all.append(np.array(lab))
    return np.vstack(X_all), np.concatenate(y_all)


def make_cost_fn(model, mode, w=1.0):
    def fn(S, T, P, raw, gate):
        ii, jj, X = frame_features(S, T, P, raw, gate=gate)
        out = np.full(raw.shape, 1e6)
        geo = np.linalg.norm(T[None, :, :] - P[:, None, :], axis=-1) + 0.05 * raw
        if len(ii):
            p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
            if mode == "logp":
                out[ii, jj] = -np.log(p)
            else:                       # blend: geometry minus a learned term
                out[ii, jj] = geo[ii, jj] - w * np.log(p / (1 - p))
        return np.where(raw <= gate, out, 1e6)
    return fn


def score(vols, subs, gts, cost_fn):
    rows = []
    for ds in vols:
        gn = subs[ds]
        pos, t_of = load_volume(ds, gn)
        e = relink_volume(pos, t_of, cost_fn=cost_fn)
        nd = np.asarray([[r.node_id, r.t, r.z, r.y, r.x] for r in gn.itertuples()], dtype=np.float64)
        ed = np.asarray(e, dtype=np.int64) if e else np.zeros((0, 2), np.int64)
        m = per_sample_metrics(evaluate(gts[ds].graph, Graph(nodes=nd, edges=ed)),
                               gts[ds].estimated_number_of_nodes)
        m["dataset"] = ds
        rows.append(m)
    agg = summarise(rows)
    agg["edge_fp"] = sum(r["edge_fp"] for r in rows)
    agg["edge_fn"] = sum(r["edge_fn"] for r in rows)
    return agg, rows


def main() -> None:
    sp = json.loads((REPO / "configs/honest_split.json").read_text())[0]
    ev = json.loads((REPO / "configs/evalset.json").read_text())["stems"]
    train_v = [v for v in ev if v in set(sp["train"])]
    test_v = [v for v in ev if v in set(sp["test"])]

    df = pd.read_csv(SUB)
    nodes = df[df.row_type == "node"]
    subs = {ds: gn for ds, gn in nodes.groupby("dataset")}
    gts = {ds: load_geff(GT_ROOT / f"{ds}.geff") for ds in ev}

    print(f"honest split within evalset: {len(train_v)} train / {len(test_v)} test volumes\n")
    print("building training pairs ...")
    Xtr, ytr = build_training(train_v, subs)
    print(f"  {len(ytr):,} labelled pairs, {int(ytr.sum()):,} positive ({ytr.mean():.1%})\n")

    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.06,
                                           random_state=0)
    model.fit(Xtr, ytr)

    print("PHASE 3 GATE -- end-to-end on HELD-OUT volumes, real metric:\n")
    print(f"{'cost function':<28} {'edge_J':>8} {'adj':>8} {'score':>8} {'e_FP':>7} {'e_FN':>7} {'delta':>9}")
    base, _ = score(test_v, subs, gts, geometric_cost)
    print(f"{'geometric (shipped)':<28} {base['edge_jaccard']:>8.4f} {base['adj_edge_jaccard']:>8.4f} "
          f"{base['score']:>8.4f} {base['edge_fp']:>7,} {base['edge_fn']:>7,} {'--':>9}")
    results = {}
    for label, mode, w in (("learned  -log(p)", "logp", 0.0),
                           ("geo - 0.5*logit(p)", "blend", 0.5),
                           ("geo - 2.0*logit(p)", "blend", 2.0),
                           ("geo - 8.0*logit(p)", "blend", 8.0)):
        agg, _ = score(test_v, subs, gts, make_cost_fn(model, mode, w))
        results[label] = agg
        print(f"{label:<28} {agg['edge_jaccard']:>8.4f} {agg['adj_edge_jaccard']:>8.4f} "
              f"{agg['score']:>8.4f} {agg['edge_fp']:>7,} {agg['edge_fn']:>7,} "
              f"{agg['score']-base['score']:>+9.4f}")

    best = max(results, key=lambda k: results[k]["score"])
    d = results[best]["score"] - base["score"]
    print(f"\nbest: {best}   delta {d:+.4f}")
    print("GATE (§35.4 phase 3): beat the geometric cost on held-out volumes.")
    print("  " + ("PASS -- proceed to a kernel run." if d > 0 else
                  "FAIL -- the learned cost does not beat geometry out of sample."))
    if d > 0 and results[best]["edge_fp"] > base["edge_fp"]:
        print("  ⚠ but edge FP INCREASED -- check the ~40:1 asymmetry (§18.3) before shipping.")


if __name__ == "__main__":
    main()
