#!/usr/bin/env python3
"""Does a dividing nucleus LOOK different? The gate on the row-70/71 division route.

Row 70: score = adj_edge_J + 0.1·div_J, and we bank 0.007 of a possible 0.100.
Row 71: 16 of 25 missed divisions already have parent AND both daughters in our
node set -- the failure is assignment, not detection, and fixing it is worth
~+0.036 (past the 0.949 bar).

Rows 35/43/38.5 killed every GEOMETRIC selector on this problem ("the information
to separate 8 real forks from ~6,000 coincidences does not exist in geometry").
APPEARANCE has never been tried: mitotic nuclei round up, brighten, and split, and
that is a different information source. This probe answers -- from GT only, before
any GPU credit is spent -- whether that signal exists at all.

DESIGN (deliberately conservative, so a PASS is believable):
  * Positives: GT parent nodes at the division frame (151 corpus-wide).
  * Negatives: GT non-dividing nodes, sampled from the SAME volumes and matched
    to the positives' time distribution -- otherwise a classifier can win by
    learning "late frames" rather than "mitosis".
  * Features: intensity/shape statistics on a small 3D crop, plus TEMPORAL
    deltas (t-1 -> t -> t+1), which is where the mitotic signature lives.
  * Validation: GroupKFold by VOLUME. Never let a volume span train and test:
    with ~0.8 divisions per volume, leakage would be trivial and fatal.
  * Report: PR curve + precision at the recall we actually need. The deployment
    target is ~15 TP with ~10 FP per 24 volumes.

PRE-REGISTERED GATE (stated before looking at the output): the deployment pool is
~1 candidate per node, so the classifier must reach ROC-AUC >= 0.75 AND hold
precision >= 0.30 at recall 0.50 under volume-grouped CV. Below that, appearance
is no better than the geometry that already failed and the route is closed --
recorded as a finding, no Colab spend.
"""
from __future__ import annotations

import glob
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/maruanottoni/home/kaggle/biohub-cell-tracking/src")
from ctrack.data.geff import load_geff  # noqa: E402

RAW = "/Volumes/extSSD/kaggle/biohub/raw/train"
GT = "/Volumes/extSSD/kaggle/biohub/gt/train"
NEG_PER_POS = 20
RZ, RY, RX = 2, 12, 12          # crop half-extent in voxels
SEED = 0


def crop(arr, t, z, y, x):
    T, Z, Y, X = arr.shape
    t = int(np.clip(t, 0, T - 1))
    z0, z1 = max(0, z - RZ), min(Z, z + RZ + 1)
    y0, y1 = max(0, y - RY), min(Y, y + RY + 1)
    x0, x1 = max(0, x - RX), min(X, x + RX + 1)
    return np.asarray(arr[t, z0:z1, y0:y1, x0:x1], dtype=np.float32)


def feats(arr, t, z, y, x):
    """Appearance now, and how it CHANGED -- mitosis is a temporal event."""
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
    # temporal deltas on the summary stats (brightening / condensation)
    a, b, c_ = np.array(out[0:6]), np.array(out[6:12]), np.array(out[12:18])
    out += list(a - b) + list(c_ - a)
    # spatial structure: is the intensity mass splitting into two lobes?
    mid = cur.max(axis=0) if cur.ndim == 3 else cur
    if mid.size:
        cy, cx = np.array(mid.shape) // 2
        q = [mid[:cy, :cx].mean(), mid[:cy, cx:].mean(),
             mid[cy:, :cx].mean(), mid[cy:, cx:].mean()]
        out += [float(np.std(q)), float(max(q) - min(q))]
    else:
        out += [0.0, 0.0]
    return out


def main() -> None:
    import zarr
    rng = np.random.default_rng(SEED)
    X, y, groups = [], [], []
    vols = sorted(glob.glob(f"{GT}/*.geff"))
    n_pos_total = 0
    for gi, gf in enumerate(vols):
        name = gf.split("/")[-1].replace(".geff", "")
        try:
            g = load_geff(gf)
        except Exception:
            continue
        G = g.graph
        kids = defaultdict(list)
        for s, t in G.edges:
            kids[s].append(t)
        div_ids = [n for n, ch in kids.items() if len(ch) >= 2]
        if not div_ids:
            continue                      # 112/199 volumes have no divisions
        try:
            z = zarr.open(f"{RAW}/{name}.zarr", mode="r")
            arr = z["raw"] if "raw" in z else z[list(z.array_keys())[0]]
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}", flush=True)
            continue
        idx = {nid: i for i, nid in enumerate(G.ids)}
        div_set = set(div_ids)
        pos_t = []
        for nid in div_ids:
            i = idx[nid]
            f = feats(arr, int(G.t[i]), *[int(round(v)) for v in G.zyx[i]])
            if f:
                X.append(f); y.append(1); groups.append(gi); pos_t.append(int(G.t[i]))
                n_pos_total += 1
        # time-matched negatives from the SAME volume
        cand = [nid for nid in G.ids if nid not in div_set]
        if not cand or not pos_t:
            continue
        cand_t = np.array([int(G.t[idx[n]]) for n in cand])
        for _ in range(len(pos_t) * NEG_PER_POS):
            tt = pos_t[rng.integers(len(pos_t))]
            near = np.flatnonzero(np.abs(cand_t - tt) <= 2)
            if near.size == 0:
                near = np.arange(len(cand))
            nid = cand[int(near[rng.integers(near.size)])]
            i = idx[nid]
            f = feats(arr, int(G.t[i]), *[int(round(v)) for v in G.zyx[i]])
            if f:
                X.append(f); y.append(0); groups.append(gi)
        print(f"  {name}: +{len(pos_t)} divisions (running total {n_pos_total})", flush=True)

    X = np.array(X, dtype=np.float32); y = np.array(y); groups = np.array(groups)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"\ndataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{y.sum()} positives, {len(set(groups))} volumes")
    np.savez("/Volumes/extSSD/kaggle/biohub/replay_assets/division_probe.npz",
             X=X, y=y, groups=groups)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=15, random_state=SEED)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    auc = roc_auc_score(y, oof)
    prec, rec, _ = precision_recall_curve(y, oof)
    print(f"\n=== VOLUME-GROUPED CV ===\nROC-AUC {auc:.4f}   (chance 0.5)")
    print(f"{'recall':>8} {'precision':>10}")
    for target in (0.3, 0.5, 0.7, 0.9):
        ok = rec >= target
        p = prec[ok].max() if ok.any() else 0.0
        print(f"{target:8.2f} {p:10.4f}")
    p50 = prec[rec >= 0.5].max() if (rec >= 0.5).any() else 0.0
    print(f"\nGATE: AUC>=0.75 and precision@recall0.50>=0.30 -> "
          f"{'PASS' if (auc >= 0.75 and p50 >= 0.30) else 'FAIL'} "
          f"(auc {auc:.3f}, p@r50 {p50:.3f})")


if __name__ == "__main__":
    main()
