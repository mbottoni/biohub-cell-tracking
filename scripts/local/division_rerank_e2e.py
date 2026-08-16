#!/usr/bin/env python3
"""END-TO-END test of the appearance division re-ranker on evalset-24.

Row 72 showed appearance separates GT divisions from RANDOM non-dividing nodes
(AUC 0.845). That is a proxy on an easy pool. This script answers the real
question with the real objective: build candidates from OUR OWN predicted graph,
score them by appearance, insert the top ones as divisions, and re-score the
submission with the official scorer.

    score = adj_edge_jaccard + 0.1 * division_jaccard      (row 70)

Every inserted division is a bet: it can win up to 0.1*div_J but it rewires an
edge, so it can lose adj_edge_J. Sweeping K prices that trade directly -- which
is the thing no previous division attempt (rows 24/35/43) could see.

Candidate pool: for every predicted edge p(t) -> c1(t+1), every other predicted
node c2 at t+1 within RADIUS of p is a candidate second child. Positive iff GT
says p divides and {c1,c2} match its two daughters.

Validation: GroupKFold BY VOLUME, out-of-fold scores only -- with ~1 division
per volume, any within-volume leakage would be fatal and invisible.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/Users/maruanottoni/home/kaggle/biohub-cell-tracking/src")
from ctrack.data.geff import load_geff  # noqa: E402

S = ("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
     "746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad")
SUB = f"{S}/runs/reverify_stack/submission.csv"
OUT_DIR = f"{S}/runs/divrerank"
RAW = "/Volumes/extSSD/kaggle/biohub/raw/train"
GT = "/Volumes/extSSD/kaggle/biohub/gt/train"
# Measured GT division geometry (row 24 / STRATEGY 15): parent->child ~5.75 µm,
# sister<->sister ~10.57 µm. Geometry is a GOOD proposer and a BAD selector (rows
# 35/43) -- so use it only as a high-recall prefilter and let appearance choose.
RADIUS = 11.0        # µm, candidate second child from the parent
SISTER_MAX = 18.0    # µm, between the two daughters
MAX_CAND = 3         # nearest candidates kept per (parent, first child)
MATCH = 10.0         # µm, node <-> GT match
RZ, RY, RX = 2, 12, 12
SEED = 0


def crop(arr, t, z, y, x):
    T, Z, Y, X = arr.shape
    t = int(np.clip(t, 0, T - 1))
    return np.asarray(arr[t,
                          max(0, z - RZ):min(Z, z + RZ + 1),
                          max(0, y - RY):min(Y, y + RY + 1),
                          max(0, x - RX):min(X, x + RX + 1)], dtype=np.float32)


def appearance(arr, t, zyx):
    z, y, x = (int(round(v)) for v in zyx)
    cur = crop(arr, t, z, y, x)
    if cur.size == 0:
        return None
    prv, nxt = crop(arr, t - 1, z, y, x), crop(arr, t + 1, z, y, x)
    out = []
    for c in (cur, prv, nxt):
        if c.size == 0:
            return None
        f = c.ravel()
        p = np.percentile(f, [50, 90, 99])
        out += [f.mean(), f.std(), p[0], p[1], p[2], f.max()]
    a, b, c_ = np.array(out[0:6]), np.array(out[6:12]), np.array(out[12:18])
    out += list(a - b) + list(c_ - a)
    mid = cur.max(axis=0)
    cy, cx = np.array(mid.shape) // 2
    q = [mid[:cy, :cx].mean(), mid[:cy, cx:].mean(), mid[cy:, :cx].mean(), mid[cy:, cx:].mean()]
    out += [float(np.std(q)), float(max(q) - min(q))]
    return out


def load_sub(path):
    nodes = defaultdict(dict)          # ds -> node_id -> (t,z,y,x)
    edges = defaultdict(list)          # ds -> [(src,tgt)]
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        for r in rd:
            rows.append(r)
            ds = r["dataset"]
            if r["row_type"] == "node":
                nodes[ds][int(r["node_id"])] = (
                    int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"]))
            else:
                edges[ds].append((int(r["source_id"]), int(r["target_id"])))
    return rows, fields, nodes, edges


def main():
    import zarr
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    rows, fields, nodes, edges = load_sub(SUB)
    vols = sorted(nodes)
    print(f"submission: {len(vols)} volumes", flush=True)

    X, y, meta, groups = [], [], [], []
    kept_tot, truth_tot = [0], [0]
    for gi, v in enumerate(vols):
        nd, ed = nodes[v], edges[v]
        by_t = defaultdict(list)
        for nid, (t, z, yy, xx) in nd.items():
            by_t[t].append(nid)
        trees = {t: (cKDTree(np.array([nd[n][1:] for n in ns])), ns)
                 for t, ns in by_t.items() if ns}
        # GT divisions -> matched (parent, {kids}) in predicted-node space
        g = load_geff(f"{GT}/{v}.geff"); G = g.graph
        kids = defaultdict(list)
        for s_, t_ in G.edges:
            kids[s_].append(t_)
        idx = {nid: i for i, nid in enumerate(G.ids)}

        def match(gid):
            i = idx[gid]; t = int(G.t[i])
            tr = trees.get(t)
            if tr is None:
                return None
            d, j = tr[0].query(np.array(G.zyx[i]))
            return tr[1][j] if d <= MATCH else None

        truth = set()
        for pid, ch in kids.items():
            if len(ch) < 2:
                continue
            mp, m1, m2 = match(pid), match(ch[0]), match(ch[1])
            if mp is not None and m1 is not None and m2 is not None:
                truth.add((mp, frozenset((m1, m2))))
        try:
            z = zarr.open(f"{RAW}/{v}.zarr", mode="r")
            arr = z[list(z.array_keys())[0]]
        except Exception as e:
            print(f"  skip {v}: {e}"); continue

        cache = {}
        def app(nid):
            if nid not in cache:
                t, *zyx = nd[nid]
                cache[nid] = appearance(arr, t, zyx)
            return cache[nid]

        npos = 0
        for (p, c1) in ed:
            if p not in nd or c1 not in nd:
                continue
            tp = nd[p][0]
            tr = trees.get(tp + 1)
            if tr is None:
                continue
            pz = np.array(nd[p][1:])
            cands = []
            for j in tr[0].query_ball_point(pz, RADIUS):
                c2 = tr[1][j]
                if c2 == c1:
                    continue
                if np.linalg.norm(np.array(nd[c1][1:]) - np.array(nd[c2][1:])) > SISTER_MAX:
                    continue
                cands.append((np.linalg.norm(pz - np.array(nd[c2][1:])), c2))
            for _, c2 in sorted(cands)[:MAX_CAND]:
                fp, f1, f2 = app(p), app(c1), app(c2)
                if fp is None or f1 is None or f2 is None:
                    continue
                d1 = np.linalg.norm(pz - np.array(nd[c1][1:]))
                d2 = np.linalg.norm(pz - np.array(nd[c2][1:]))
                dsis = np.linalg.norm(np.array(nd[c1][1:]) - np.array(nd[c2][1:]))
                lab = 1 if (p, frozenset((c1, c2))) in truth else 0
                X.append(fp + f1 + f2 + [d1, d2, dsis, abs(d1 - d2)])
                y.append(lab); groups.append(gi); meta.append((v, p, c1, c2)); npos += lab
        print(f"  {v}: {npos}/{len(truth)} truth kept by prefilter, pool now {len(X):,}", flush=True)
        kept_tot[0] += npos; truth_tot[0] += len(truth)

    X = np.nan_to_num(np.array(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y); groups = np.array(groups)
    print(f"\nPREFILTER RECALL: {kept_tot[0]}/{truth_tot[0]} detectable divisions survive "
          f"(ceiling on everything downstream)", flush=True)
    print(f"pool {X.shape[0]:,} candidates, {y.sum()} positives "
          f"({100*y.mean():.3f}%), {X.shape[1]} features", flush=True)
    if y.sum() < 5:
        print("too few positives to model"); return

    oof = np.zeros(len(y))
    for tr_i, te_i in GroupKFold(n_splits=6).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                           max_leaf_nodes=15, random_state=SEED)
        m.fit(X[tr_i], y[tr_i])
        oof[te_i] = m.predict_proba(X[te_i])[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f"out-of-fold ROC-AUC on the REAL pool: {roc_auc_score(y, oof):.4f}", flush=True)

    np.savez(f"{OUT_DIR}_scores.npz", oof=oof, y=y, groups=groups,
             meta=np.array([f"{a}|{b}|{c}|{d}" for a, b, c, d in meta]))

    # --- insert top-K as divisions and write submissions for the official scorer
    import os
    order = np.argsort(-oof)
    for K in (10, 20, 30, 50, 80, 120):
        take = [meta[i] for i in order[:K]]
        add = defaultdict(list); drop = defaultdict(set)
        for (v, p, c1, c2) in take:
            add[v].append((p, c2)); drop[v].add(c2)     # c2 gets a new parent
        d = f"{OUT_DIR}_k{K}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/submission.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            nid = 0
            for r in rows:
                if r["row_type"] == "edge" and int(r["target_id"]) in drop[r["dataset"]]:
                    continue                             # a node may have one parent
                r = dict(r); r["id"] = nid; nid += 1; w.writerow(r)
            for v, pairs in add.items():
                for (p, c2) in pairs:
                    w.writerow({"id": nid, "dataset": v, "row_type": "edge", "node_id": -1,
                                "t": -1, "z": -1, "y": -1, "x": -1,
                                "source_id": p, "target_id": c2}); nid += 1
        print(f"  wrote K={K}: +{sum(len(p) for p in add.values())} division edges", flush=True)
    print("\nnow score each with scripts/local/score_official.py", flush=True)


if __name__ == "__main__":
    main()
