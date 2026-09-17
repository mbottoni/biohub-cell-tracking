#!/usr/bin/env python3
"""Discriminate the DivNet preprocessing/architecture assumptions on GT divisions.

The public checkpoint (giorgosi/biohub-divnet-v2) documents its input config but not
the norm-layer type, channel order, pool type, or normalization scope. The correct
combination should reproduce the manifest's AUC (~0.87-0.89 per-fold val, 0.845 OOF
single); a wrong one lands near 0.5. Protocol = row 72's: GT division parents as
positives, time-matched same-frame non-dividing GT nodes as negatives.

    .venv/bin/python scripts/local/divnet_local_check.py
"""
from __future__ import annotations
import os, sys
from collections import defaultdict
import numpy as np
import zarr

sys.path.insert(0, "/Users/maruanottoni/home/kaggle/biohub-cell-tracking/src")
sys.path.insert(0, "/Users/maruanottoni/home/kaggle/biohub-cell-tracking/scripts")
from ctrack.data.geff import load_geff  # noqa: E402
from divnet_infer import load_divnet, build_input, gaussian_marker  # noqa: E402

BASE = "/Volumes/extSSD/kaggle/biohub"
GT, RAW = f"{BASE}/gt/train", f"{BASE}/raw/train"
CKPT = "/tmp/divnet/best_overall.pt"
NEG_PER_POS = 5
TARGET_POS = 40
SEED = 314159


def pool_frame(frame, k=4, how='mean'):
    Z, Y, X = frame.shape
    f = frame[:, :(Y // k) * k, :(X // k) * k].reshape(Z, Y // k, k, X // k, k)
    return f.mean(axis=(2, 4)) if how == 'mean' else f.max(axis=(2, 4))


def collect_samples():
    """-> list of (volume, t, z, y, x, label) with y,x in RAW voxels."""
    rng = np.random.default_rng(SEED)
    vols = sorted(f[:-5] for f in os.listdir(GT) if f.endswith('.geff'))
    per_vol = []
    for v in vols:
        try:
            G = load_geff(f"{GT}/{v}.geff").graph
        except Exception:
            continue
        kids = defaultdict(list)
        for s_, t_ in G.edges:
            kids[s_].append(t_)
        parents = [p for p, ch in kids.items() if len(ch) >= 2]
        if parents:
            per_vol.append((len(parents), v, G, kids))
    per_vol.sort(reverse=True, key=lambda r: r[0])
    samples = []
    npos = 0
    for ndiv, v, G, kids in per_vol:
        idx = {nid: i for i, nid in enumerate(G.ids)}
        by_t = defaultdict(list)
        for nid, i in idx.items():
            by_t[int(G.t[i])].append(nid)
        for p in kids:
            if len(kids[p]) < 2:
                continue
            i = idx[p]
            t = int(G.t[i])
            z, y, x = G.zyx[i]
            samples.append((v, t, z, y, x, 1))
            npos += 1
            pool = [n for n in by_t[t] if len(kids.get(n, [])) < 2 and n != p]
            for n in rng.choice(pool, size=min(NEG_PER_POS, len(pool)), replace=False):
                j = idx[n]
                zn, yn, xn = G.zyx[j]
                samples.append((v, t, zn, yn, xn, 0))
        if npos >= TARGET_POS:
            break
    return samples


def main():
    import torch
    samples = collect_samples()
    npos = sum(s[-1] for s in samples)
    print(f"samples: {len(samples)} ({npos} positives) from "
          f"{len(set(s[0] for s in samples))} volumes", flush=True)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    models = {n: load_divnet(CKPT, device=device, norm=n)[0]
              for n in ['instance', 'group8', 'group4', 'group1']}
    _, cfg = load_divnet(CKPT, norm='instance')

    # cache pooled frames per (vol, t, how)
    arrs, fcache = {}, {}

    def pooled(v, t, how):
        key = (v, t, how)
        if key not in fcache:
            if v not in arrs:
                arrs[v] = zarr.open(f"{RAW}/{v}.zarr", mode='r')['0']
            a = arrs[v]
            t = int(np.clip(t, 0, a.shape[0] - 1))
            fcache[key] = pool_frame(np.asarray(a[t], dtype=np.float32), 4, how)
            if len(fcache) > 400:
                fcache.pop(next(iter(fcache)))
        return fcache[key]

    from sklearn.metrics import roc_auc_score
    labels = np.array([s[-1] for s in samples])
    results = []
    for how in ['mean', 'max']:
        # build inputs once per (how, marker, scope) — model norm reuses them
        for scope in ['crop', 'frame']:
            for marker_last in [True, False]:
                inputs = []
                for v, t, z, y, x, _ in samples:
                    fbl = {lag: pooled(v, t + lag, how) for lag in cfg['input']['image_lags']}
                    inputs.append(build_input(fbl, z, y / 4.0, x / 4.0, cfg,
                                              norm_scope=scope, marker_last=marker_last))
                X = torch.from_numpy(np.stack(inputs)).to(device)
                for nname, m in models.items():
                    with torch.inference_mode():
                        probs = torch.sigmoid(m(X)).squeeze(1).cpu().numpy()
                    auc = roc_auc_score(labels, probs)
                    results.append((auc, how, scope,
                                    'marker_last' if marker_last else 'marker_first', nname,
                                    float(probs[labels == 1].mean()), float(probs[labels == 0].mean())))
                    print(f"AUC {auc:.4f} | pool={how} scope={scope} "
                          f"marker={'last' if marker_last else 'first'} norm={nname} "
                          f"| p_pos {probs[labels==1].mean():.3f} p_neg {probs[labels==0].mean():.3f}",
                          flush=True)
    results.sort(reverse=True)
    print("\nBEST:", results[0])


if __name__ == '__main__':
    main()
