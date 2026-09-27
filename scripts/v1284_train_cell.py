"""Train our own V1284 coordinate head from capture-mode dumps (rows 173-174).

x138's V1284 head ("0.953 ORIGINAL") is a 224->32->3 MLP on frozen UNet features at each
fused detection, regressing a bounded (<=2 um) offset toward the true nucleus centre. Its
author trained it on 20 train movies and notes the 4-movie version was WORSE (+14.9%) while
20 movies gave -10.4% held-out residual: data volume is the lever. We have 199 train movies.

Inputs  : <cap_dir>/<stem>/<t:04d>.npz  {coords (N,4) [t,z,y,x] feature grid, features (N,224)}
          <train_dir>/<stem>.geff       GT; nodes/props/{t,z,y,x}/values in full-res voxels
Pairing : per frame, Hungarian on detection<->GT distance in um, kept if < PAIR_RADIUS_UM.
          feature grid is 1.625 um isotropic; GT um = voxel * (1.625, 0.40625, 0.40625)
          (verified 09-27 against x138's own submission vs local GT: median 2.63 um, all < 7).
Target  : gt_um - det_um, clipped to norm < 2 (the head's bounded range).
Split   : by MOVIE (group split) for model selection; the final head is refit on all movies
          for the selected number of epochs.
Output  : {'state_dict','mean','scale'} -- the exact format x138's refine() loads with
          torch.load(weights_only=True) -- plus a json report and the pairs (for local refits).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.optimize import linear_sum_assignment

SPACING_UM = 1.625
GT_SCALE_UM = np.array([1.625, 0.40625, 0.40625])
PAIR_RADIUS_UM = float(os.environ.get('V1284_PAIR_RADIUS_UM', '4.0'))
SEED = 0


def make_head():
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head, x):
    delta = head(x)
    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


def load_gt(geff_path):
    g = zarr.open(str(geff_path), mode='r')
    p = {k: np.asarray(g[f'nodes/props/{k}/values']) for k in 'tzyx'}
    um = np.stack([p['z'], p['y'], p['x']], 1).astype(np.float64) * GT_SCALE_UM
    return p['t'].astype(np.int64), um


def build_pairs(cap_dir, train_dir, stems):
    X, Y, G = [], [], []
    per_movie = {}
    for gi, stem in enumerate(stems):
        frames = sorted((Path(cap_dir) / stem).glob('*.npz'))
        geff = Path(train_dir) / f'{stem}.geff'
        if not frames or not geff.exists():
            continue
        gt_t, gt_um = load_gt(geff)
        n0 = len(X)
        for f in frames:
            d = np.load(f)
            coords, feats = d['coords'], d['features']
            t = int(coords[0, 0])
            gt = gt_um[gt_t == t]
            if not len(gt) or not len(coords):
                continue
            det = coords[:, 1:].astype(np.float64) * SPACING_UM
            cost = np.linalg.norm(det[:, None, :] - gt[None, :, :], axis=2)
            r, c = linear_sum_assignment(cost)
            keep = cost[r, c] < PAIR_RADIUS_UM
            for i, j in zip(r[keep], c[keep]):
                X.append(feats[i]); Y.append(gt[j] - det[i]); G.append(gi)
        per_movie[stem] = len(X) - n0
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32); G = np.asarray(G, np.int64)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    Y = np.where(nrm > 1.99, Y * (1.99 / np.maximum(nrm, 1e-9)), Y).astype(np.float32)
    return X, Y, G, per_movie


def fit(X, Y, epochs, lr=3e-3, wd=1e-4, Xv=None, Yv=None):
    torch.manual_seed(SEED)
    mean = X.mean(0); scale = X.std(0) + 1e-6
    xt = torch.from_numpy((X - mean) / scale); yt = torch.from_numpy(Y)
    xv = torch.from_numpy((Xv - mean) / scale) if Xv is not None else None
    yv = torch.from_numpy(Yv) if Yv is not None else None
    head = make_head(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    curve = []
    for ep in range(epochs):
        head.train(); perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 2048):
            idx = perm[b:b + 2048]
            loss = torch.nn.functional.smooth_l1_loss(bounded(head, xt[idx]), yt[idx], beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step()
        if xv is not None:
            head.eval()
            with torch.no_grad():
                res = torch.linalg.vector_norm(yv - bounded(head, xv), dim=1).mean().item()
            curve.append(res)
    return head, mean, scale, curve


def main(cap_dir, train_dir, stems, out_dir, epochs=150):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X, Y, G, per_movie = build_pairs(cap_dir, train_dir, stems)
    movies = np.unique(G)
    print(f'[v1284-train] pairs={len(X)} movies={len(movies)} radius={PAIR_RADIUS_UM}um '
          f'baseline mean residual={np.linalg.norm(Y, axis=1).mean():.4f}um', flush=True)
    assert len(movies) >= 5 and len(X) >= 1000, 'too little capture to train a head'
    rng = np.random.default_rng(SEED)
    val_m = rng.choice(movies, size=max(2, len(movies) // 5), replace=False)
    vm = np.isin(G, val_m)
    head, mean, scale, curve = fit(X[~vm], Y[~vm], epochs, Xv=X[vm], Yv=Y[vm])
    base_v = float(np.linalg.norm(Y[vm], axis=1).mean())
    best_ep = int(np.argmin(curve)) + 1
    best_v = float(min(curve))
    rel = (best_v - base_v) / base_v
    print(f'[v1284-train] HELD-OUT BY MOVIE ({len(val_m)} movies, {vm.sum()} pairs): '
          f'{base_v:.4f} -> {best_v:.4f} um ({100 * rel:+.1f}%) at epoch {best_ep}/{epochs} '
          f'(author: -10.4% on 20 movies)', flush=True)
    final, mean, scale, _ = fit(X, Y, best_ep)
    torch.save({'state_dict': final.state_dict(),
                'mean': torch.from_numpy(mean.astype(np.float32)),
                'scale': torch.from_numpy(scale.astype(np.float32))}, out_dir / 'v1284_head.pt')
    np.savez_compressed(out_dir / 'v1284_pairs.npz', X=X, Y=Y, G=G,
                        stems=np.array([s for s in per_movie]))
    report = {'pairs': int(len(X)), 'movies': int(len(movies)), 'radius_um': PAIR_RADIUS_UM,
              'heldout_movies': [str(s) for s in np.array(list(per_movie))[val_m]] if len(per_movie) == len(movies) else None,
              'heldout_baseline_um': base_v, 'heldout_best_um': best_v, 'heldout_rel': rel,
              'best_epoch': best_ep, 'curve': curve, 'per_movie_pairs': per_movie}
    (out_dir / 'v1284_train_report.json').write_text(json.dumps(report, indent=1))
    # round trip exactly as x138's refine() loads it
    s = torch.load(out_dir / 'v1284_head.pt', map_location='cpu', weights_only=True)
    h = make_head(); h.load_state_dict(s['state_dict'])
    assert tuple(s['mean'].shape) == (224,) and tuple(s['scale'].shape) == (224,)
    print(f'[v1284-train] saved {out_dir / "v1284_head.pt"} (round-trip load OK)', flush=True)
    return report
