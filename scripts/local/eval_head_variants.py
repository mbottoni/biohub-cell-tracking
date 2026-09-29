"""Held-out-by-movie eval of V1284 head variants on the 120-movie pairs (same split as
train_debiased_heads.py). Variants: author head scaled by a, split-refit zmean/amean scaled by a,
and author+zmean averages. Reports raw residual, centred residual, realised mean shift."""
import importlib.util, json, sys, time
import numpy as np, torch
spec = importlib.util.spec_from_file_location('v', 'scripts/v1284_train_cell.py'); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
d = np.load('logs/wave/cap120_pairs/v1284_out/v1284_pairs.npz'); X, Y, G = d['X'], d['Y'], d['G']
rng = np.random.default_rng(0); movies = np.unique(G)
vm = np.isin(G, rng.choice(movies, size=max(2, len(movies) // 5), replace=False))
ymean = Y[~vm].mean(0); AUTHOR_MEAN = np.array([0.194, 0.038, -0.108], np.float32)
def pred(head, mean, scale, Xs):
    head.eval()
    with torch.no_grad(): return v.bounded(head, torch.from_numpy(((Xs - mean) / scale).astype(np.float32))).numpy()
P = {}
s = torch.load(sys.argv[1], map_location='cpu', weights_only=True); h = v.make_head(); h.load_state_dict(s['state_dict'])
P['author'] = pred(h, s['mean'].numpy(), s['scale'].numpy(), X[vm])
for tag, ep, shift in (('zmean', 53, np.zeros(3, np.float32)), ('amean', 68, AUTHOR_MEAN)):
    t0 = time.time(); Yt = (Y - ymean + shift).astype(np.float32)
    nrm = np.linalg.norm(Yt, axis=1, keepdims=True); Yt = np.where(nrm > 1.99, Yt * 1.99 / nrm, Yt).astype(np.float32)
    hh, m, sc, _ = v.fit(X[~vm], Yt[~vm], ep); P[tag] = pred(hh, m, sc, X[vm]); print(tag, 'fit', round(time.time() - t0), 's', flush=True)
Yv = Y[vm]; yc = Yv - Yv.mean(0)
def rep(p):
    raw = np.linalg.norm(Yv - p, axis=1); cen = np.linalg.norm(yc - (p - p.mean(0)), axis=1)
    return dict(raw=round(float(raw.mean()), 4), within1=round(float((raw < 1.0).mean()), 4), centred=round(float(cen.mean()), 4), mean=p.mean(0).round(3).tolist())
out = {'base (no head)': rep(np.zeros_like(Yv))}
for k in ('author', 'zmean', 'amean'):
    for a in (0.75, 1.0, 1.25, 1.5): out[f'{k} x{a}'] = rep(P[k] * a)
for w in (0.5,):
    out['avg author+zmean'] = rep(w * P['author'] + (1 - w) * P['zmean'])
    out['avg author+amean'] = rep(w * P['author'] + (1 - w) * P['amean'])
for k, r in out.items(): print(f'{k:22s}', r)
print('target mean on held-out', Yv.mean(0).round(3).tolist())
