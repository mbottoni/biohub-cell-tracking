#!/usr/bin/env python3
"""STRATEGY 23.3 -- fit the division selector on the REAL candidate distribution.

`division_separability.py` showed a fitted model beats the hand-written ranking
2.4x at a fixed budget, but on a pool built from GT nodes only: 47 candidates per
volume against the ~3,374 a real run sees (§16: 13,495 over 4 volumes). That is
70x too little competition, so its absolute precision means nothing.

This script removes that caveat WITHOUT a GPU run. A commit run's
`submission.csv` is the full predicted graph -- ~30,000 nodes per volume, real
detections, real linking errors. The safe-division candidate pool can be
reconstructed from it directly:

    sources    = predicted nodes at t with out-degree exactly 1
    candidates = predicted nodes at t+1 with in-degree 0
    admitted   = pairs inside the wide gates (§15)

which is the same rule `add_safe_divisions_postlink` applies.

⚠ It is a reconstruction, not the identical pool. The real step runs on an
intermediate graph, before short-track filtering and before the divisions it
adds are themselves in the graph. So node counts differ slightly from the true
call site. What it gets right -- and what the GT pool got catastrophically wrong
-- is the DENSITY of competing candidates, which is the thing that decides
precision.

THE LABEL IS NOT "is this the true daughter"
--------------------------------------------
From `_evaluate_divisions`: a division TP is *a GT dividing node whose matched
predicted node also divides*. It does not check which candidate was chosen. So
for a candidate `(source, cand)`:

    source matches a GT node with out-degree >= 2   ->  adding it scores a TP
    source matches a GT node with out-degree  < 2   ->  adding it scores an FP
    source matches nothing                          ->  INVISIBLE to the metric

Only the first two are labelled; the third is excluded from fitting because its
label does not exist. That third class is most of the pool -- §23.5's audit found
only 1.86% of inserted divisions land on an annotated node at all.

So the model estimates P(this source is a real dividing cell | geometry), which
is precisely the `P(division|geometry)` that §16.2 identified as missing and
never built.

    python scripts/local/division_candidates.py submissions/d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import SCALE, Graph, match_nodes  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")

WIDE_PARENT_UM, WIDE_SISTER_UM, WIDE_CHILD_UM = 12.0, 14.4, 12.0
SHIP_PARENT_UM, SHIP_SISTER_UM, SHIP_CHILD_UM = 4.66, 8.5, 7.65

FEATURES = [
    "parent_dist", "sister_dist", "child_dist", "cos_angle",
    "cand_fwd_len", "src_back_len", "src_speed",
    "spacing", "parent_over_spacing", "sister_over_spacing",
    "n_cand", "cand_rank", "cand_nn_dist", "cand_isolation",
]


def chain_len(start: int, adj: dict[int, list[int]], cap: int = 12) -> int:
    n, cur = 0, start
    while n < cap:
        nxt = adj.get(cur)
        if not nxt:
            break
        cur, n = nxt[0], n + 1
    return n


def pool_for_volume(ds: str, gn: pd.DataFrame, ge: pd.DataFrame,
                    gt_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_of = dict(zip(gn.node_id.astype(int), gn.t.astype(int)))
    pos = {int(r.node_id): np.array([r.z, r.y, r.x], dtype=float) * SCALE
           for r in gn.itertuples()}

    src_a = ge.source_id.astype(int).to_numpy()
    tgt_a = ge.target_id.astype(int).to_numpy()
    children: dict[int, list[int]] = defaultdict(list)
    parents: dict[int, list[int]] = defaultdict(list)
    indeg: dict[int, int] = defaultdict(int)
    for a, b in zip(src_a, tgt_a):
        children[a].append(b)
        parents[b].append(a)
        indeg[b] += 1

    by_t: dict[int, list[int]] = defaultdict(list)
    for nid, t in t_of.items():
        by_t[t].append(nid)

    # Per-frame spacing and per-node isolation, on the FULL predicted node set --
    # this is the quantity the GT-only pool could not represent.
    spacing: dict[int, float] = {}
    nn_dist: dict[int, float] = {}
    for t, ids in by_t.items():
        if len(ids) < 2:
            spacing[t] = np.nan
            for i in ids:
                nn_dist[i] = np.nan
            continue
        P = np.stack([pos[i] for i in ids])
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        d, _ = tree.query(P, k=2)
        spacing[t] = float(np.median(d[:, 1]))
        for i, dd in zip(ids, d[:, 1]):
            nn_dist[i] = float(dd)

    # Label source nodes by matching the predicted graph to GT.
    gt = load_geff(gt_root / f"{ds}.geff")
    pred = Graph(
        nodes=np.asarray([[r.node_id, r.t, r.z, r.y, r.x] for r in gn.itertuples()],
                         dtype=np.float64),
        edges=np.asarray(list(zip(src_a, tgt_a)), dtype=np.int64),
    )
    matched = match_nodes(gt.graph, pred, max_distance=7.0, scale=SCALE)  # pred_id -> gt_id
    gt_out: dict[int, int] = defaultdict(int)
    for a, _ in gt.graph.edges:
        gt_out[int(a)] += 1

    rows, labels, groups = [], [], []
    for t in sorted(by_t):
        nxt = by_t.get(t + 1, [])
        if not nxt:
            continue
        sources = [n for n in by_t[t] if len(children.get(n, ())) == 1]
        cands = [n for n in nxt if indeg.get(n, 0) == 0]
        if not sources or not cands:
            continue
        from scipy.spatial import cKDTree
        CP = np.stack([pos[c] for c in cands])
        ctree = cKDTree(CP)

        for s in sources:
            gid = matched.get(s)
            if gid is None:
                continue  # invisible to the metric -- no label exists
            label = 1 if gt_out.get(int(gid), 0) >= 2 else 0

            ex = children[s][0]
            if t_of.get(ex) != t + 1:
                continue
            child_dist = float(np.linalg.norm(pos[ex] - pos[s]))
            if child_dist > WIDE_CHILD_UM:
                continue

            near = ctree.query_ball_point(pos[s], WIDE_PARENT_UM)
            local = []
            for ci in near:
                c = cands[ci]
                if c == ex:
                    continue
                parent_dist = float(np.linalg.norm(pos[c] - pos[s]))
                sister_dist = float(np.linalg.norm(pos[c] - pos[ex]))
                if sister_dist > WIDE_SISTER_UM:
                    continue
                local.append((parent_dist, sister_dist, c))
            if not local:
                continue
            local.sort()

            v_child = pos[ex] - pos[s]
            nk = np.linalg.norm(v_child)
            sp = parents.get(s)
            src_speed = float(np.linalg.norm(pos[s] - pos[sp[0]])) if sp else 0.0
            back = chain_len(s, {c: [p[0]] for c, p in parents.items() if p})
            spc = spacing.get(t + 1, np.nan)

            for rank, (parent_dist, sister_dist, c) in enumerate(local):
                v_cand = pos[c] - pos[s]
                nc = np.linalg.norm(v_cand)
                cos_angle = float(v_child @ v_cand / (nc * nk)) if nc > 1e-9 and nk > 1e-9 else 0.0
                rows.append([
                    parent_dist, sister_dist, child_dist, cos_angle,
                    chain_len(c, children), back, src_speed,
                    spc,
                    parent_dist / spc if spc else np.nan,
                    sister_dist / spc if spc else np.nan,
                    len(local), rank, nn_dist.get(c, np.nan),
                    nn_dist.get(c, np.nan) / spc if spc else np.nan,
                ])
                labels.append(label)
                groups.append(ds)

    X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(FEATURES)))
    return X, np.asarray(labels, dtype=np.int64), np.asarray(groups)


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s)
    yt = y[order]
    prec = np.cumsum(yt) / np.arange(1, len(yt) + 1)
    return float((prec * yt).sum() / y.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", nargs="+")
    ap.add_argument("--gt-root", type=Path, default=GT_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="write pooled rows as .npz")
    args = ap.parse_args()

    Xs, ys, gs = [], [], []
    for sub in args.submissions:
        df = pd.read_csv(sub)
        nodes = df[df.row_type == "node"]
        edges = df[df.row_type == "edge"]
        print(f"{sub}: {nodes.dataset.nunique()} volumes, {len(nodes):,} nodes")
        for ds, gn in nodes.groupby("dataset"):
            ge = edges[edges.dataset == ds]
            X, y, g = pool_for_volume(ds, gn, ge, args.gt_root)
            if len(y):
                Xs.append(X); ys.append(y); gs.append(g)
            print(f"    {ds:>16}  labelled candidates {len(y):>7,}  positives {int(y.sum()):>5}")

    X = np.vstack(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    print(f"\npool: {len(y):,} LABELLED candidate rows "
          f"({int(y.sum())} positive, {y.mean():.4%}) over {len(set(g))} volumes")
    print(f"      density: {len(y)/len(set(g)):,.0f} labelled candidates/volume "
          f"(GT-only pool had 47)")

    if args.out:
        np.savez_compressed(args.out, X=X, y=y, g=g, features=np.array(FEATURES))
        print(f"      wrote {args.out}")

    pd_, sd_, cd_ = X[:, 0], X[:, 1], X[:, 2]
    ship_gate = (pd_ <= SHIP_PARENT_UM) & (sd_ <= SHIP_SISTER_UM) & (cd_ <= SHIP_CHILD_UM)
    print(f"\nshipped gates admit {ship_gate.sum():,} rows "
          f"({int(y[ship_gate].sum())}/{int(y.sum())} positives)")

    ship_score = -(pd_ + 0.15 * sd_)
    print(f"\n{'ranking':<34} {'AP':>8}")
    print(f"{'random (positive rate)':<34} {y.mean():>8.4f}")
    print(f"{'shipped  parent+0.15*sister':<34} {average_precision(y, ship_score):>8.4f}")

    if y.sum() < 5 or len(set(g)) < 3:
        print("\n⚠ too few positives / volumes to cross-validate a model here.")
        print("  Re-run once evalset_d4edge lands (24 volumes, 25 GT divisions).")
        return

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    n_splits = min(5, len(set(g)))
    model = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.06,
                                           class_weight="balanced", random_state=0)
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=g):
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
    print(f"{f'gradient boosting ({n_splits}-fold)':<34} {average_precision(y, pred):>8.4f}")

    print(f"\nat a fixed insertion budget (top-K of {len(y):,} labelled candidates):")
    print(f"{'K':>7} {'shipped TP':>14} {'model TP':>14}")
    for K in (25, 50, 100, 200, 400):
        a = int(y[np.argsort(-ship_score)[:K]].sum())
        b = int(y[np.argsort(-pred)[:K]].sum())
        print(f"{K:>7} {a:>10} /{int(y.sum()):<4} {b:>10} /{int(y.sum()):<4}")


if __name__ == "__main__":
    main()
