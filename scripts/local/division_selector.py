#!/usr/bin/env python3
"""STRATEGY 44.4 item 3 -- a PRECISION selector over the division-aware assignment's forks.

43 measured the two halves of the division lever separately:

    division-aware assignment (penalty 8um):  dTP 8 / dFP 296 / edge FP +289   net -0.0117
    shipped post-hoc scavenge:                dTP 2 / dFP 40                   div_J 0.0308

and 43.3 showed they compose to +0.0123 IF a selector can keep the ~8 real forks
out of ~13,000 candidates (base rate 0.06%). This script builds that selector and
scores it END-TO-END (38.5: never on AP/AUC).

Two arms, because the base rate makes fitting fragile:

  A. GEOMETRY GATES -- the shipped scavenge's own admission thresholds
     (parent<=4.66, sister<=8.5 um, from the public config) applied to the fork
     pool. No fitting, no overfit risk; 15.2 measured these gates' recall on real
     divisions at 0.265, so expect ~2/8 kept but near-zero FP.
  B. FITTED GBM -- 24.1's approach refit on THIS pool (43.3's requirement),
     leave-one-volume-out so every prediction is out of sample. With ~8 positives
     total this is at the edge of feasibility, which is the point of arm A.

PRE-REGISTERED READ: arm A is the floor (known-precision, low recall); arm B must
beat arm A's end-to-end score on the held-out side to justify its complexity.
Success gate for either: div term contribution > 0 with edge FP within +40 of the
1-1 baseline on the honest-test-side volumes.

    .venv/bin/python scripts/local/division_selector.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import json
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
    MAX_MATCH_DISTANCE_UM, SCALE, Graph, evaluate, match_nodes,
    per_sample_metrics, summarise,
)
from relink_replay import (  # noqa: E402
    MAX_FRAME_NODES, RELAXED_UM, TIGHT_UM, VELOCITY_WEIGHT,
)
from localisation_doseresponse import load_submission  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
DIV_PENALTY = 8.0
SCAVENGE_PARENT_UM = 4.66      # shipped SAFE_DIV_MAX_UM in the public 0.915 config
SCAVENGE_SISTER_UM = 8.5       # shipped SAFE_DIV_SISTER_MAX_UM


def relink_forks(pos, t_of):
    """Division-aware relink (43's formulation) that also returns fork details."""
    by_t = defaultdict(list)
    for n, t in t_of.items():
        by_t[t].append(n)
    for ids in by_t.values():
        ids.sort()
    if by_t and max(len(v) for v in by_t.values()) > MAX_FRAME_NODES:
        return [], []

    prev_pos: dict[int, np.ndarray] = {}
    out: list[tuple[int, int]] = []
    child_of: dict[int, list[tuple[int, bool]]] = defaultdict(list)  # s -> [(g, is_second)]
    spacing_cache: dict[int, float] = {}

    def spacing(t):
        if t not in spacing_cache:
            ids = by_t.get(t, [])
            if len(ids) < 2:
                spacing_cache[t] = 11.7
            else:
                pts = np.stack([pos[n] for n in ids])
                d, _ = cKDTree(pts).query(pts, k=2)
                spacing_cache[t] = float(np.median(d[:, 1]))
        return spacing_cache[t]

    for t in sorted(by_t):
        srcs, tgts = by_t.get(t, []), by_t.get(t + 1, [])
        if not srcs or not tgts:
            continue
        un_t = set(tgts)
        n_child: dict[int, int] = defaultdict(int)
        committed = []
        for gate in (TIGHT_UM, RELAXED_UM):
            ps = [s for s in srcs if n_child[s] < 2]
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
            second = np.where(raw > gate, big, base + DIV_PENALTY)
            for i, s in enumerate(ps):
                if n_child[s] >= 1:
                    base[i, :] = big
            cost = np.concatenate([base, second], axis=0)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] >= big:
                    continue
                s = ps[int(i) % len(ps)]
                g = pt[int(j)]
                if g not in un_t or n_child[s] >= 2:
                    continue
                un_t.discard(g)
                n_child[s] += 1
                committed.append((s, g, int(i) >= len(ps)))
        for s, g, is_second in committed:
            out.append((s, g))
            child_of[s].append((g, is_second))
            prev_pos[g] = pos[s]

    forks = []
    for s, children in child_of.items():
        if len(children) < 2:
            continue
        (g1, _), (g2, _) = children[0], children[1]
        p, c1, c2 = pos[s], pos[g1], pos[g2]
        d1, d2 = float(np.linalg.norm(c1 - p)), float(np.linalg.norm(c2 - p))
        sister = float(np.linalg.norm(c1 - c2))
        v1, v2 = c1 - p, c2 - p
        cosang = float(np.dot(v1, v2) /
                       max(1e-9, np.linalg.norm(v1) * np.linalg.norm(v2)))
        sp = spacing(t_of[s] + 1)
        forks.append({
            "s": s, "g1": g1, "g2": g2, "t": t_of[s],
            "d1": d1, "d2": d2, "dmax": max(d1, d2), "sister": sister,
            "cos_angle": cosang,
            "parent_mid": float(np.linalg.norm(p - (c1 + c2) / 2)),
            "spacing": sp, "dmax_over_sp": max(d1, d2) / max(1e-9, sp),
            "sister_over_sp": sister / max(1e-9, sp),
        })
    return out, forks


def process_volume(job):
    ds, nodes = job
    gt = load_geff(GT_ROOT / f"{ds}.geff")
    ids = nodes[:, 0].astype(np.int64)
    t_of = {int(i): int(t) for i, t in zip(ids, nodes[:, 1].astype(int))}
    pos = {int(i): nodes[k, 2:5] * SCALE for k, i in enumerate(ids)}
    edges, forks = relink_forks(pos, t_of)

    pred_g = Graph(nodes=nodes, edges=np.zeros((0, 2), dtype=np.int64))
    matched = match_nodes(gt.graph, pred_g, max_distance=MAX_MATCH_DISTANCE_UM,
                          scale=SCALE)
    gt_out = gt.graph.out_degree()
    for f in forks:
        gid = matched.get(int(f["s"]))
        f["matched"] = gid is not None
        f["label"] = bool(gid is not None and gt_out.get(gid, 0) >= 2)
        f["dataset"] = ds
    return ds, edges, forks


def score_policy(ds, nodes, edges, drop_edges, gt):
    keep = [e for e in edges if e not in drop_edges]
    g = Graph(nodes=nodes,
              edges=np.asarray(keep, np.int64) if keep else np.zeros((0, 2), np.int64))
    m = per_sample_metrics(evaluate(gt.graph, g), gt.estimated_number_of_nodes)
    m["dataset"] = ds
    return m


def evaluate_policy(name, kept_by_ds, all_forks, edges_by_ds, subs, gts, test_side):
    """Drop every fork's second edge unless the fork is in the kept set."""
    rows = []
    for ds in sorted(subs):
        drop = set()
        for f in all_forks:
            if f["dataset"] != ds:
                continue
            key = (f["s"], f["g1"], f["g2"])
            if key not in kept_by_ds.get(ds, set()):
                drop.add((f["s"], f["g2"]))          # revert to 1-1: drop 2nd child
        rows.append(score_policy(ds, subs[ds].nodes, edges_by_ds[ds], drop, gts[ds]))
    agg = summarise(rows)
    tr = summarise([r for r in rows if r["dataset"] in test_side]) \
        if test_side else None
    kept_forks = [f for f in all_forks
                  if (f["s"], f["g1"], f["g2"]) in kept_by_ds.get(f["dataset"], set())]
    ktp = sum(1 for f in kept_forks if f["label"])
    print(f"  {name:>24}  kept {len(kept_forks):>6,} (TP {ktp})  "
          f"edge_J {agg['edge_jaccard']:.4f}  adj {agg['adj_edge_jaccard']:.4f}  "
          f"div_J {agg['division_jaccard']:.4f}  SCORE {agg['score']:.4f}"
          + (f"  | test-side SCORE {tr['score']:.4f}" if tr else ""))
    return agg, tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    args = ap.parse_args()

    subs = load_submission(Path(args.submission))
    sp = json.loads((REPO / "configs/honest_split.json").read_text())[0]
    test_side = set(sp["test"]) & set(subs)
    print(f"{len(subs)} volumes, {len(test_side)} on the honest-test side\n")

    with ProcessPoolExecutor() as ex:
        results = list(ex.map(process_volume,
                              [(ds, subs[ds].nodes) for ds in sorted(subs)]))
    edges_by_ds = {ds: e for ds, e, _ in results}
    all_forks = [f for _, _, forks in results for f in forks]
    gts = {ds: load_geff(GT_ROOT / f"{ds}.geff") for ds in subs}

    pos_n = sum(1 for f in all_forks if f["label"])
    print(f"fork pool: {len(all_forks):,} candidates, {pos_n} labelled real "
          f"({pos_n/max(1,len(all_forks)):.3%} base rate), "
          f"{sum(1 for f in all_forks if not f['matched']):,} unmatched (div-invisible)")

    key = lambda f: (f["s"], f["g1"], f["g2"])  # noqa: E731

    # ---- arm 0: baselines
    evaluate_policy("keep NONE (1-1 baseline)", {}, all_forks, edges_by_ds,
                    subs, gts, test_side)
    evaluate_policy("keep ALL (43's -0.0117)",
                    defaultdict(set, {ds: {key(f) for f in all_forks if f["dataset"] == ds}
                                      for ds in subs}),
                    all_forks, edges_by_ds, subs, gts, test_side)
    oracle = defaultdict(set)
    for f in all_forks:
        if f["label"]:
            oracle[f["dataset"]].add(key(f))
    evaluate_policy("ORACLE (keep true only)", oracle, all_forks, edges_by_ds,
                    subs, gts, test_side)

    # ---- arm A: scavenge geometry gates
    armA = defaultdict(set)
    for f in all_forks:
        if f["dmax"] <= SCAVENGE_PARENT_UM and f["sister"] <= SCAVENGE_SISTER_UM:
            armA[f["dataset"]].add(key(f))
    evaluate_policy("A: scavenge gates", armA, all_forks, edges_by_ds,
                    subs, gts, test_side)

    # ---- arm B: fitted GBM, leave-one-volume-out
    from sklearn.ensemble import GradientBoostingClassifier
    feats = ["d1", "d2", "dmax", "sister", "cos_angle", "parent_mid",
             "spacing", "dmax_over_sp", "sister_over_sp"]
    X = np.array([[f[c] for c in feats] for f in all_forks])
    y = np.array([f["label"] for f in all_forks])
    ds_of = np.array([f["dataset"] for f in all_forks])
    prob = np.zeros(len(all_forks))
    for ds in sorted(subs):
        tr = ds_of != ds
        if y[tr].sum() == 0:
            continue
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                         random_state=0)
        clf.fit(X[tr], y[tr])
        te = ~tr
        prob[te] = clf.predict_proba(X[te])[:, 1]

    for thr_q in (0.999, 0.9995, 0.9999):
        thr = float(np.quantile(prob, thr_q))
        armB = defaultdict(set)
        for k, f in enumerate(all_forks):
            if prob[k] >= thr:
                armB[f["dataset"]].add(key(f))
        evaluate_policy(f"B: GBM LOVO q={thr_q}", armB, all_forks, edges_by_ds,
                        subs, gts, test_side)

    print("\nGATE (44.4): a policy passes only if SCORE beats the 1-1 baseline on")
    print("the honest-test side AND kept-FP stays ~O(40) (the scavenge's own FP budget).")


if __name__ == "__main__":
    main()


def inspect_pool(submission: str) -> None:
    """Dump the true forks' features + corpus GT division geometry for gate design."""
    subs = load_submission(Path(submission))
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(process_volume,
                              [(ds, subs[ds].nodes) for ds in sorted(subs)]))
    all_forks = [f for _, _, forks in results for f in forks]
    feats = ["d1", "d2", "dmax", "sister", "cos_angle", "parent_mid",
             "dmax_over_sp", "sister_over_sp"]
    print("TRUE forks:")
    print("  " + " ".join(f"{c:>12}" for c in ["dataset"] + feats))
    for f in all_forks:
        if f["label"]:
            print("  " + f"{f['dataset']:>12} " +
                  " ".join(f"{f[c]:>12.3f}" for c in feats))
    pool = np.array([[f[c] for c in feats] for f in all_forks])
    print("\npool quantiles (10/50/90%):")
    for k, c in enumerate(feats):
        q = np.quantile(pool[:, k], [0.1, 0.5, 0.9])
        print(f"  {c:>14}: {q[0]:>8.3f} {q[1]:>8.3f} {q[2]:>8.3f}")

    print("\ncorpus GT division geometry (all 199 volumes):")
    rows = []
    for p in sorted(GT_ROOT.glob("*.geff")):
        gt = load_geff(p)
        out_edges = defaultdict(list)
        for s, t in gt.graph.edges:
            out_edges[int(s)].append(int(t))
        node_pos = {int(n[0]): np.asarray(n[2:5]) * SCALE for n in gt.graph.nodes}
        for s, kids in out_edges.items():
            if len(kids) < 2:
                continue
            pp, c1, c2 = node_pos[s], node_pos[kids[0]], node_pos[kids[1]]
            v1, v2 = c1 - pp, c2 - pp
            rows.append([float(np.linalg.norm(v1)), float(np.linalg.norm(v2)),
                         max(float(np.linalg.norm(v1)), float(np.linalg.norm(v2))),
                         float(np.linalg.norm(c1 - c2)),
                         float(np.dot(v1, v2) / max(1e-9, np.linalg.norm(v1) * np.linalg.norm(v2)))])
    arr = np.array(rows)
    print(f"  n={len(arr)}")
    for k, c in enumerate(["d1", "d2", "dmax", "sister", "cos_angle"]):
        q = np.quantile(arr[:, k], [0.1, 0.5, 0.9])
        print(f"  {c:>14}: {q[0]:>8.3f} {q[1]:>8.3f} {q[2]:>8.3f}")


def arm_c(submission: str) -> None:
    """Corpus-derived gates: fitted on the 151 GT divisions (no pool leakage)."""
    subs = load_submission(Path(submission))
    sp = json.loads((REPO / "configs/honest_split.json").read_text())[0]
    test_side = set(sp["test"]) & set(subs)
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(process_volume,
                              [(ds, subs[ds].nodes) for ds in sorted(subs)]))
    edges_by_ds = {ds: e for ds, e, _ in results}
    all_forks = [f for _, _, forks in results for f in forks]
    gts = {ds: load_geff(GT_ROOT / f"{ds}.geff") for ds in subs}

    # Gate constants from corpus GT division quantiles (p90 cos 0.034,
    # p90 dmax 10.05, p10/p90 sister 6.36/14.36) -- NOT from the 8 pool positives.
    GATES = {
        "C1: cos<=.05 dmax<=10 sis<=14.4":
            lambda f: f["cos_angle"] <= 0.05 and f["dmax"] <= 10.05 and f["sister"] <= 14.4,
        "C2: C1 & sister>=5":
            lambda f: f["cos_angle"] <= 0.05 and f["dmax"] <= 10.05
            and 5.0 <= f["sister"] <= 14.4,
        "C3: C2 & d2>=2.6":
            lambda f: f["cos_angle"] <= 0.05 and f["dmax"] <= 10.05
            and 5.0 <= f["sister"] <= 14.4 and min(f["d1"], f["d2"]) >= 1.0,
        "C4: cos<=-0.3 & C2":
            lambda f: f["cos_angle"] <= -0.3 and f["dmax"] <= 10.05
            and 5.0 <= f["sister"] <= 14.4,
    }
    evaluate_policy("keep NONE (1-1 baseline)", {}, all_forks, edges_by_ds,
                    subs, gts, test_side)
    for name, gate in GATES.items():
        kept = defaultdict(set)
        for f in all_forks:
            if gate(f):
                kept[f["dataset"]].add((f["s"], f["g1"], f["g2"]))
        evaluate_policy(name, kept, all_forks, edges_by_ds, subs, gts, test_side)


def arm_d(submission: str) -> None:
    """Top-K per volume by corpus-typicality rank inside the C4 gate."""
    subs = load_submission(Path(submission))
    sp = json.loads((REPO / "configs/honest_split.json").read_text())[0]
    test_side = set(sp["test"]) & set(subs)
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(process_volume,
                              [(ds, subs[ds].nodes) for ds in sorted(subs)]))
    edges_by_ds = {ds: e for ds, e, _ in results}
    all_forks = [f for _, _, forks in results for f in forks]
    gts = {ds: load_geff(GT_ROOT / f"{ds}.geff") for ds in subs}

    # Typicality: distance from corpus medians (sister 10.57, dmax 7.13) plus a
    # strong preference for antipodal daughters. All constants corpus-derived.
    def typ(f):
        return abs(f["sister"] - 10.57) + abs(f["dmax"] - 7.13) + 5.0 * (f["cos_angle"] + 0.746)

    evaluate_policy("keep NONE (1-1 baseline)", {}, all_forks, edges_by_ds,
                    subs, gts, test_side)
    gated = [f for f in all_forks
             if f["cos_angle"] <= -0.3 and f["dmax"] <= 10.05
             and 5.0 <= f["sister"] <= 14.4]
    by_ds = defaultdict(list)
    for f in gated:
        by_ds[f["dataset"]].append(f)
    for K in (1, 2, 3, 5):
        kept = defaultdict(set)
        for ds, fl in by_ds.items():
            for f in sorted(fl, key=typ)[:K]:
                kept[ds].add((f["s"], f["g1"], f["g2"]))
        evaluate_policy(f"D: C4 top-{K}/volume", kept, all_forks, edges_by_ds,
                        subs, gts, test_side)
