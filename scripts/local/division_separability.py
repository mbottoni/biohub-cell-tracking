#!/usr/bin/env python3
"""STRATEGY 23.3 -- can a FITTED selector beat the hand-written division ranking?

§16 widened the safe-division gates 38x and picked 119 of 13,495 candidates with

    score = |parent_dist - 5.75| + |sister_dist - 10.57|      (ascending)

and lost 0.006 on the board. §16.2 diagnosed the error precisely -- "the EDA
measured P(geometry|division) (recall), not P(division|geometry) (precision)" --
and then never built the thing that estimates a precision. §20 concluded
"divisions are closed" from three knob failures measured against 3 GT divisions.

We own 151 labelled divisions across 199 volumes. This script asks the question
§16 should have asked: on IDENTICAL candidate data, is a model fitted to those
labels more discriminative than the hand-written score?

WHAT THIS CAN AND CANNOT ESTABLISH
----------------------------------
The candidate pool here is built from GT nodes only, i.e. the ~3.6% of cells that
are annotated. Two biases run in OPPOSITE directions and neither is small:

  * PESSIMISTIC -- we admit every annotated node at t+1 inside the wide gates,
    not just the orphans the shipped code would consider. More negatives than
    reality, among annotated cells.
  * OPTIMISTIC -- a predicted graph holds ~25,000 cells per volume against ~670
    annotated ones, so the true competing pool is far denser. §26 hit exactly
    this ("GT is too sparse for candidates to compete per frame").

So absolute precision here is NOT an estimate of inference-time precision, and
is not reported as one. What IS valid is the RELATIVE comparison: shipped score
vs fitted model, same pool, same rows, grouped cross-validation. If the fitted
model does not clearly win here -- in the regime most favourable to simple
geometry -- the lever is dead for free and no Kaggle run gets spent.

Gates are the WIDE ones from §15 (parent 12 / sister 14.4 / child 12), which
admit 89.4% of real divisions, because the point is to let selection rather than
gating do the work. The shipped gates (4.66/8.5/7.65) are scored on the same pool
so the comparison includes "what the shipped config would have picked".

    python scripts/local/division_separability.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import find_geff_stores, load_geff  # noqa: E402
from ctrack.eval.biohub import SCALE  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")

# Wide gates (§15): admit 89.4% of real divisions and let the model select.
WIDE_PARENT_UM, WIDE_SISTER_UM, WIDE_CHILD_UM = 12.0, 14.4, 12.0
# Shipped gates, scored on the same pool for reference (d4edge cell 4).
SHIP_PARENT_UM, SHIP_SISTER_UM, SHIP_CHILD_UM = 4.66, 8.5, 7.65

FEATURES = [
    "parent_dist", "sister_dist", "child_dist",
    "asym", "cos_angle",
    "cand_fwd_len", "src_back_len", "src_speed",
    "spacing", "parent_over_spacing", "sister_over_spacing",
    "n_cand", "cand_rank", "cand_nn_dist",
]


def chain_len(start: int, adj: dict[int, list[int]], cap: int = 12) -> int:
    """How many frames this node's lineage continues (or extends back), capped."""
    n, cur = 0, start
    while n < cap:
        nxt = adj.get(cur)
        if not nxt:
            break
        cur, n = nxt[0], n + 1
    return n


def build_rows(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    gt = load_geff(path)
    g = gt.graph
    pos = {int(r[0]): np.asarray(r[2:5], dtype=np.float64) * SCALE for r in g.nodes}
    tof = {int(r[0]): int(r[1]) for r in g.nodes}

    children: dict[int, list[int]] = defaultdict(list)
    parent: dict[int, int] = {}
    for a, b in g.edges:
        children[int(a)].append(int(b))
        parent[int(b)] = int(a)

    by_t: dict[int, list[int]] = defaultdict(list)
    for nid, t in tof.items():
        by_t[t].append(nid)

    # Per-frame spacing: the median nearest-neighbour distance among annotated
    # cells. A scale normaliser -- gates in raw um mean different things in a
    # 3,780-node volume and a 78,600-node one.
    spacing: dict[int, float] = {}
    nn_dist: dict[int, float] = {}
    for t, ids in by_t.items():
        if len(ids) < 2:
            spacing[t] = np.nan
            for i in ids:
                nn_dist[i] = np.nan
            continue
        P = np.stack([pos[i] for i in ids])
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        spacing[t] = float(np.median(nn))
        for i, d in zip(ids, nn):
            nn_dist[i] = float(d)

    rows, labels = [], []
    for src, kids in children.items():
        t = tof[src]
        # Reproduce the inference configuration: the linker has already taken
        # ONE child (per §18.2 it takes the shorter link), and the safe-division
        # step asks whether to add a second.
        kids_next = [k for k in kids if tof[k] == t + 1]
        if not kids_next:
            continue
        dists = {k: float(np.linalg.norm(pos[k] - pos[src])) for k in kids_next}
        existing = min(dists, key=dists.get)
        child_dist = dists[existing]
        if child_dist > WIDE_CHILD_UM:
            continue
        true_second = {k for k in kids_next if k != existing}

        pool = [c for c in by_t.get(t + 1, []) if c != existing and c != src]
        feats_here, labs_here, pdists = [], [], []
        for cand in pool:
            parent_dist = float(np.linalg.norm(pos[cand] - pos[src]))
            if parent_dist > WIDE_PARENT_UM:
                continue
            sister_dist = float(np.linalg.norm(pos[cand] - pos[existing]))
            if sister_dist > WIDE_SISTER_UM:
                continue

            v_child = pos[existing] - pos[src]
            v_cand = pos[cand] - pos[src]
            nc, nk = np.linalg.norm(v_cand), np.linalg.norm(v_child)
            cos_angle = float(v_child @ v_cand / (nc * nk)) if nc > 1e-9 and nk > 1e-9 else 0.0

            src_par = parent.get(src)
            src_speed = float(np.linalg.norm(pos[src] - pos[src_par])) if src_par in pos else 0.0

            feats_here.append({
                "parent_dist": parent_dist,
                "sister_dist": sister_dist,
                "child_dist": child_dist,
                "asym": abs(parent_dist - child_dist) / (parent_dist + child_dist + 1e-9),
                "cos_angle": cos_angle,
                "cand_fwd_len": chain_len(cand, children),
                "src_back_len": chain_len(src, {c: [p] for c, p in parent.items()}),
                "src_speed": src_speed,
                "spacing": spacing.get(t + 1, np.nan),
                "parent_over_spacing": parent_dist / spacing[t + 1] if spacing.get(t + 1) else np.nan,
                "sister_over_spacing": sister_dist / spacing[t + 1] if spacing.get(t + 1) else np.nan,
                "cand_nn_dist": nn_dist.get(cand, np.nan),
            })
            labs_here.append(1 if cand in true_second else 0)
            pdists.append(parent_dist)

        # n_cand / cand_rank describe the competition this source faces, which is
        # only knowable once the whole pool for that source exists.
        order = np.argsort(pdists) if pdists else []
        rank = {int(i): int(r) for r, i in enumerate(order)}
        for i, (f, lab) in enumerate(zip(feats_here, labs_here)):
            f["n_cand"] = len(feats_here)
            f["cand_rank"] = rank[i]
            rows.append([f[k] for k in FEATURES])
            labels.append(lab)

    X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(FEATURES)))
    y = np.asarray(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64)
    return X, y, path.stem


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AP with score DESCENDING = more division-like."""
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s)
    yt = y[order]
    tp = np.cumsum(yt)
    prec = tp / np.arange(1, len(yt) + 1)
    return float((prec * yt).sum() / y.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=GT_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    stores = sorted(find_geff_stores(args.root))
    if args.limit:
        stores = stores[: args.limit]
    print(f"building candidate rows from {len(stores)} GT graphs "
          f"(wide gates {WIDE_PARENT_UM}/{WIDE_SISTER_UM}/{WIDE_CHILD_UM})...")

    Xs, ys, groups = [], [], []
    for i, p in enumerate(stores):
        X, y, stem = build_rows(p)
        if len(y) == 0:
            continue
        Xs.append(X)
        ys.append(y)
        groups.append(np.full(len(y), i))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(stores)}  rows so far {sum(len(a) for a in ys):,}")

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(groups)
    print(f"\npool: {len(y):,} candidate rows, {int(y.sum())} true second-daughters "
          f"({y.mean():.4%} positive), {len(set(g))} volumes")

    # ---- the shipped ranking, on this exact pool -------------------------------
    pd_, sd_, cd_ = X[:, 0], X[:, 1], X[:, 2]
    ship_gate = (pd_ <= SHIP_PARENT_UM) & (sd_ <= SHIP_SISTER_UM) & (cd_ <= SHIP_CHILD_UM)
    print(f"\nshipped gates ({SHIP_PARENT_UM}/{SHIP_SISTER_UM}/{SHIP_CHILD_UM}) admit "
          f"{ship_gate.sum():,} rows and {int(y[ship_gate].sum())}/{int(y.sum())} "
          f"= {y[ship_gate].sum()/y.sum():.1%} of real second-daughters")

    # ascending score -> negate for a descending ranking
    ship_score = -(pd_ + 0.15 * sd_)
    divgeom_score = -(np.abs(pd_ - 5.75) + np.abs(sd_ - 10.57))
    print(f"\n{'ranking':<34} {'AP':>8}")
    print(f"{'random (positive rate)':<34} {y.mean():>8.4f}")
    print(f"{'shipped  parent+0.15*sister':<34} {average_precision(y, ship_score):>8.4f}")
    print(f"{'divgeom  |p-5.75|+|s-10.57| (§16)':<34} {average_precision(y, divgeom_score):>8.4f}")

    # ---- fitted model, grouped CV by volume ------------------------------------
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    models = {
        "logistic regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gradient boosting": HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.06,
            class_weight="balanced", random_state=0),
    }
    oof = {}
    for name, model in models.items():
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
            model.fit(X[tr], y[tr])
            pred[te] = model.predict_proba(X[te])[:, 1]
        oof[name] = pred
        print(f"{name+' (5-fold, grouped)':<34} {average_precision(y, pred):>8.4f}")

    # ---- what it buys at a fixed budget ---------------------------------------
    # The pipeline inserts a bounded number of divisions, so the operating point
    # that matters is precision/recall at top-K, not the whole curve.
    print(f"\nat a fixed insertion budget (top-K of {len(y):,} candidates):")
    print(f"{'K':>7} {'shipped':>18} {'divgeom':>18} {'best model':>18}")
    best_name = max(oof, key=lambda k: average_precision(y, oof[k]))
    for K in (151, 300, 600, 1200):
        line = f"{K:>7}"
        for s in (ship_score, divgeom_score, oof[best_name]):
            hit = int(y[np.argsort(-s)[:K]].sum())
            line += f"   TP {hit:>3} / {int(y.sum())} ({hit/y.sum():>5.1%})"
        print(line)
    print(f"\nbest model = {best_name}")

    # ---- construction-bias check ----------------------------------------------
    # Positives are built by designating the NEARER daughter as the already-linked
    # child, so parent_dist >= child_dist holds for every positive by construction.
    # Negatives carry no such constraint. If the model is mostly exploiting that,
    # the result is an artifact of how the rows were made, not a real signal.
    pos_ge = float((pd_[y == 1] >= cd_[y == 1]).mean())
    neg_ge = float((pd_[y == 0] >= cd_[y == 0]).mean())
    print(f"\nconstruction-bias check:  P(parent_dist >= child_dist) = "
          f"{pos_ge:.3f} for positives (1.000 by construction) vs {neg_ge:.3f} for negatives")

    keep = [i for i, f in enumerate(FEATURES) if f not in ("child_dist", "asym")]
    Xa = X[:, keep]
    abl = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.06,
                                         class_weight="balanced", random_state=0)
    pred_a = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(Xa, y, groups=g):
        abl.fit(Xa[tr], y[tr])
        pred_a[te] = abl.predict_proba(Xa[te])[:, 1]
    ap_a = average_precision(y, pred_a)
    print(f"ablation (drop child_dist + asym, the biased pair): AP {ap_a:.4f} "
          f"vs full {average_precision(y, oof[best_name]):.4f} "
          f"vs shipped {average_precision(y, ship_score):.4f}")
    hit_a = int(y[np.argsort(-pred_a)[:151]].sum())
    print(f"  top-151: {hit_a}/{int(y.sum())} TP vs shipped "
          f"{int(y[np.argsort(-ship_score)[:151]].sum())}/{int(y.sum())}")

    # ---- which features carry it ----------------------------------------------
    rng = np.random.default_rng(0)
    base = average_precision(y, pred_a)
    print("\npermutation importance on the ablated model (drop in AP when shuffled):")
    imp = []
    for j, name in enumerate([FEATURES[i] for i in keep]):
        drops = []
        for _ in range(3):
            Xp = Xa.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            p = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(Xp, y, groups=g):
                abl.fit(Xa[tr], y[tr])
                p[te] = abl.predict_proba(Xp[te])[:, 1]
            drops.append(base - average_precision(y, p))
        imp.append((float(np.mean(drops)), name))
    for d, name in sorted(imp, reverse=True):
        print(f"   {name:<22} {d:+.4f}")

    print("\n⚠ Absolute precision here is NOT an inference-time estimate (see docstring):")
    print("   the pool is annotated cells only, ~3.6% of the true cell population.")
    print(f"   This pool has {len(y)/len(set(g)):.0f} candidates/volume; a real run has ~3,374")
    print("   (§16: 13,495 over 4 volumes) -- roughly 70x denser competition.")
    print("   The valid read is the RELATIVE ordering of the rankings on identical rows.")


if __name__ == "__main__":
    main()
