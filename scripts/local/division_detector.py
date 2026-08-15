#!/usr/bin/env python3
"""Per-node division DETECTOR — the route row 84 reopened.

WHY PER-NODE, AND WHY NOW
-------------------------
Rows 79/81 killed the division route by pricing every wrong prediction as a
false positive. Row 84 measured the truth by injection: a fork is only FP if it
matches an ANNOTATED GT node within 7, and ~3.6% of cells are annotated, so
1,000 random forks cost ONE FP and -0.0003 of score. The budget is ~10^3
inserted forks, not ~10^1 -- four orders of magnitude easier than rows 79/81
assumed.

Carried forward from the dead attempts (still true):
  * ceiling is 16 of 25 reachable divisions          (row 71)
  * geometry cannot SELECT                           (rows 76, 80)
  * the proposer must be PER-NODE, never pair enumeration -- the pair pool is
    what blew up to 1:283,000                        (rows 79, 81)

PROTOCOL
--------
Train on the 175 corpus volumes NOT in evalset-24, deploy on the 24. No volume
appears in both, so the local score is an honest estimate rather than a fit.
Positives are GT division parents; negatives are time-matched non-dividing GT
nodes from the same volume and frame (row 72's scheme).

Features are position-based (raw intensity statistics in a crop at t-1/t/t+1),
so a model trained on GT node positions transfers directly to our predicted
node positions at deployment.

Coordinates are VOXEL indices in both GT and submission (z<64, y<256, x<256) --
earlier rows label these distances "um"; they are voxels. Internally consistent,
so prior measurements stand.

Frames are single zarr chunks (1,64,256,256), so everything is frame-cached: one
read per (volume, t) instead of one per node.

    .venv/bin/python scripts/local/division_detector.py features   # stage 1+2
    .venv/bin/python scripts/local/division_detector.py deploy     # stage 3
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import zarr
from scipy.spatial import cKDTree

sys.path.insert(0, "/Users/maruanottoni/home/kaggle/biohub-cell-tracking/src")
from ctrack.data.geff import load_geff  # noqa: E402

S = ("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
     "746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad")
N = ("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
     "2ad68fab-98b7-4dc4-b103-ce7227df708f/scratchpad")
RAW = "/Volumes/extSSD/kaggle/biohub/raw/train"
GT = "/Volumes/extSSD/kaggle/biohub/gt/train"
SUB = f"{S}/runs/reverify_stack/submission.csv"
EVALSET = f"{S}/evalset24"

RZ, RY, RX = 2, 12, 12        # crop half-extents (voxels)
NEG_PER_POS = 50              # time-matched negatives per positive
CAND_RADIUS = 26.0            # 2nd-child search radius; 26 gave 13/16 (row 81 sweep)
SEED = 0


# ---------------------------------------------------------------- features
def crop(frame, z, y, x):
    Z, Y, X = frame.shape
    return frame[max(0, z - RZ):min(Z, z + RZ + 1),
                 max(0, y - RY):min(Y, y + RY + 1),
                 max(0, x - RX):min(X, x + RX + 1)]


def feats(f_prev, f_cur, f_next, z, y, x):
    """32 appearance features — row 72's scheme (AUC 0.845 at node level)."""
    out = []
    cur = crop(f_cur, z, y, x)
    if cur.size == 0:
        return None
    for f in (cur, crop(f_prev, z, y, x), crop(f_next, z, y, x)):
        if f.size == 0:
            return None
        r = f.ravel().astype(np.float32)
        p = np.percentile(r, [50, 90, 99])
        out += [r.mean(), r.std(), p[0], p[1], p[2], r.max()]
    a, b, c = np.array(out[0:6]), np.array(out[6:12]), np.array(out[12:18])
    out += list(a - b) + list(c - a)
    mid = cur.max(axis=0).astype(np.float32)
    cy, cx = np.array(mid.shape) // 2
    q = [mid[:cy, :cx].mean(), mid[:cy, cx:].mean(), mid[cy:, :cx].mean(), mid[cy:, cx:].mean()]
    out += [float(np.std(q)), float(max(q) - min(q))]
    return out


def extract(volume, want):
    """want: {t: [(key, z, y, x), ...]} -> {key: featvec}. One read per frame."""
    arr = zarr.open(f"{RAW}/{volume}.zarr", mode="r")["0"]
    T = arr.shape[0]
    cache: dict[int, np.ndarray] = {}

    def frame(t):
        t = int(np.clip(t, 0, T - 1))
        if t not in cache:
            if len(cache) > 4:
                cache.pop(next(iter(cache)))
            cache[t] = np.asarray(arr[t])
        return cache[t]

    out = {}
    for t in sorted(want):
        fp, fc, fn = frame(t - 1), frame(t), frame(t + 1)
        for key, z, y, x in want[t]:
            v = feats(fp, fc, fn, int(round(z)), int(round(y)), int(round(x)))
            if v is not None:
                out[key] = v
    return out


# ---------------------------------------------------------------- stage 1+2
def build_and_train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    evalset = {f[:-5] for f in os.listdir(EVALSET) if f.endswith(".zarr")}
    allvol = sorted(f[:-5] for f in os.listdir(GT) if f.endswith(".geff"))
    train_vols = [v for v in allvol if v not in evalset]
    print(f"corpus {len(allvol)} volumes | evalset-24 held out | training on {len(train_vols)}",
          flush=True)

    rng = np.random.default_rng(SEED)
    X, y, groups = [], [], []
    npos_total = 0
    for gi, v in enumerate(train_vols):
        try:
            G = load_geff(f"{GT}/{v}.geff").graph
        except Exception as e:
            print(f"  skip {v}: {e}", flush=True)
            continue
        kids = defaultdict(list)
        for s_, t_ in G.edges:
            kids[s_].append(t_)
        idx = {nid: i for i, nid in enumerate(G.ids)}
        div_parents = [p for p, ch in kids.items() if len(ch) >= 2]
        if not div_parents:
            continue

        by_t = defaultdict(list)
        for nid, i in idx.items():
            by_t[int(G.t[i])].append(nid)

        want = defaultdict(list)
        lab = {}
        for p in div_parents:
            i = idx[p]
            t = int(G.t[i])
            want[t].append((("P", p), *G.zyx[i]))
            lab[("P", p)] = 1
            pool = [n for n in by_t[t] if n not in kids or len(kids[n]) < 2]
            if pool:
                for n in rng.choice(pool, size=min(NEG_PER_POS, len(pool)), replace=False):
                    j = idx[int(n)]
                    want[t].append((("N", int(n)), *G.zyx[j]))
                    lab[("N", int(n))] = 0
        got = extract(v, want)
        for k, vec in got.items():
            X.append(vec); y.append(lab[k]); groups.append(gi)
        npos_total += sum(1 for k in got if lab[k] == 1)
        if gi % 25 == 0:
            print(f"  [{gi}/{len(train_vols)}] {v}: pool {len(X):,}, positives {npos_total}",
                  flush=True)

    X = np.nan_to_num(np.array(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y); groups = np.array(groups)
    print(f"\nTRAIN SET: {X.shape[0]:,} nodes, {int(y.sum())} positives, {X.shape[1]} features",
          flush=True)
    if y.sum() < 20:
        print("too few positives"); return

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                           max_leaf_nodes=15, random_state=SEED)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"volume-grouped CV ROC-AUC: {auc:.4f}   (row 72 probe: 0.845)", flush=True)
    order = np.argsort(-oof)
    for frac in (0.001, 0.005, 0.01, 0.05):
        k = max(1, int(frac * len(y)))
        print(f"  top {frac*100:5.2f}% ({k:6,d}): recall {y[order[:k]].sum()/y.sum():.3f}",
              flush=True)

    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                           max_leaf_nodes=15, random_state=SEED)
    model.fit(X, y)
    with open(f"{N}/div_model.pkl", "wb") as f:
        pickle.dump({"model": model, "auc": float(auc)}, f)
    print(f"\nwrote {N}/div_model.pkl", flush=True)


# ---------------------------------------------------------------- stage 3
def deploy():
    with open(f"{N}/div_model.pkl", "rb") as f:
        model = pickle.load(f)["model"]

    rows = list(csv.DictReader(open(SUB)))
    fields = list(rows[0].keys())
    nodes = defaultdict(dict); outdeg = defaultdict(int); indeg = defaultdict(int)
    for r in rows:
        if r["row_type"] == "node":
            nodes[r["dataset"]][int(r["node_id"])] = (
                int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"]))
        else:
            outdeg[(r["dataset"], int(r["source_id"]))] += 1
            indeg[(r["dataset"], int(r["target_id"]))] += 1

    scored = []
    for v in sorted(nodes):
        nd = nodes[v]
        by_t = defaultdict(list)
        for nid, (t, *_) in nd.items():
            by_t[t].append(nid)
        trees = {t: (cKDTree(np.array([nd[n][1:] for n in ns])), ns)
                 for t, ns in by_t.items() if ns}

        want = defaultdict(list); pair = {}
        for p, (t, z, y, x) in nd.items():
            if outdeg[(v, p)] != 1:          # need exactly 1 child -> adding 1 makes a fork
                continue
            tr = trees.get(t + 1)
            if tr is None:
                continue
            best = None
            for j in tr[0].query_ball_point(np.array([z, y, x]), CAND_RADIUS):
                c = tr[1][j]
                if indeg[(v, c)] != 0:        # keep indegree <= 1 (audit invariant)
                    continue
                d = float(np.linalg.norm(np.array(nd[c][1:]) - np.array([z, y, x])))
                if best is None or d < best[0]:
                    best = (d, c)
            if best is None:
                continue
            want[t].append(((v, p), z, y, x)); pair[(v, p)] = best[1]

        got = extract(v, want)
        if got:
            keys = list(got)
            P = model.predict_proba(np.nan_to_num(
                np.array([got[k] for k in keys], dtype=np.float32),
                nan=0.0, posinf=0.0, neginf=0.0))[:, 1]
            scored += [(float(pr), k[0], k[1], pair[k]) for pr, k in zip(P, keys)]
        print(f"  {v}: {len(got):,} candidate parents scored (running {len(scored):,})",
              flush=True)

    scored.sort(key=lambda r: -r[0])
    np.save(f"{N}/div_scored.npy", np.array([r[0] for r in scored]))
    print(f"\nTOTAL candidates {len(scored):,}", flush=True)

    for K in (250, 500, 1000, 2000, 5000, 10000, 20000):
        take = scored[:K]
        if not take:
            continue
        add = defaultdict(list)
        for _, ds, p, c in take:
            add[ds].append((p, c))
        d = f"{N}/det_k{K}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/submission.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            i = 0
            for r in rows:
                r = dict(r); r["id"] = i; i += 1; w.writerow(r)
            for ds, prs in add.items():
                for p, c in prs:
                    w.writerow({"id": i, "dataset": ds, "row_type": "edge", "node_id": -1,
                                "t": -1, "z": -1, "y": -1, "x": -1,
                                "source_id": p, "target_id": c}); i += 1
        print(f"  wrote K={K:6,d}  (min prob {take[-1][0]:.4f})", flush=True)
    print("\nnow score each det_k*/submission.csv with scripts/local/score_official.py",
          flush=True)


if __name__ == "__main__":
    {"features": build_and_train, "deploy": deploy}[sys.argv[1]]()
