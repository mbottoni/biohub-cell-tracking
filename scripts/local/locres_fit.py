#!/usr/bin/env python3
"""STRATEGY 44.4 item 2, phase B -- can a head PREDICT the detector's residual?

Consumes `locres_dump.py`'s (peak features, residual) pairs and answers the
40.5 pre-registered gate on the honest split:

    corrected matched-node RMS on HELD-OUT volumes must fall >= 30%
    (2.186 -> <= 1.53 um) or the detector fine-tune lever DIES here, for free.

Three arms, weakest first, so the result is attributable:

  bias    -- subtract the train-side mean residual (a constant). 32.3 measured
             a +0.43 um motion-aligned bias; if most of the reducible error is
             a fixed offset, the MLP is not needed.
  ridge   -- linear map features -> residual. Catches linearly-predictable bias.
  mlp     -- 2-hidden-layer MLP (the real bet: 41.1's "shared bias" hypothesis).

Every arm is ALSO scored after re-rounding the corrected coordinate to the
integer fine grid (z lattice 1.625 um), because 42.3 proved the submission
cannot express sub-voxel values -- the post-rounding number is the honest one.

    .venv/bin/python scripts/local/locres_fit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "locres"
SCALE = np.array([1.625, 0.40625, 0.40625])


def load(split_test: set[str]):
    Xtr, Ytr, Xte, Yte, te_vols = [], [], [], [], {}
    for p in sorted(DATA.glob("*.npz")):
        d = np.load(p)
        ds = p.stem
        if len(d["resid"]) == 0:
            continue
        if ds in split_test:
            te_vols[ds] = (d["feats"], d["resid"])
            Xte.append(d["feats"])
            Yte.append(d["resid"])
        else:
            Xtr.append(d["feats"])
            Ytr.append(d["resid"])
    return (np.concatenate(Xtr), np.concatenate(Ytr),
            np.concatenate(Xte), np.concatenate(Yte), te_vols)


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(y ** 2, axis=1))))


def rounded_rms(resid_left_um: np.ndarray) -> float:
    """RMS after the corrected coordinate is re-rounded to the integer fine grid.

    resid_left = gt - corrected, with gt itself on the grid (integer fine voxels).
    corrected_r = round(corrected/SCALE)*SCALE, and since gt/SCALE is integer,
    gt - corrected_r = round(resid_left/SCALE)*SCALE: a leftover under half a
    voxel rounds away entirely; a leftover past half a voxel snaps to the next
    lattice site. z sits on a 1.625 um lattice, y/x on 0.40625 um.
    """
    frac = resid_left_um / SCALE                     # in fine voxels
    return rms(np.round(frac) * SCALE)


def evaluate(name, pred_tr, pred_te, Ytr, Yte, te_vols, predict_fn):
    left_te = Yte - pred_te
    r0, r1 = rms(Yte), rms(left_te)
    r1r = rounded_rms(left_te)
    print(f"  {name:>6}  train RMS {rms(Ytr):.3f}->{rms(Ytr - pred_tr):.3f}  "
          f"HELD-OUT {r0:.3f}->{r1:.3f} ({(r1-r0)/r0:+.1%})  "
          f"post-rounding {r1r:.3f} ({(r1r-r0)/r0:+.1%})")
    worst = []
    for ds, (X, Y) in sorted(te_vols.items()):
        p = predict_fn(X)
        worst.append((rms(Y - p) / max(1e-9, rms(Y)), ds, rms(Y), rms(Y - p)))
    worst.sort()
    for frac, ds, a, b in worst[:3]:
        print(f"          best vol {ds}: {a:.3f} -> {b:.3f}")
    return r1r, r0


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    sp = json.loads((REPO / "configs/honest_split.json").read_text())[0]
    have = {p.stem for p in DATA.glob("*.npz")}
    split_test = set(sp["test"]) & have
    Xtr, Ytr, Xte, Yte, te_vols = load(split_test)
    print(f"train {len(Xtr):,} pairs / held-out {len(Xte):,} pairs "
          f"({len(split_test)} volumes)  feat dim {Xtr.shape[1]}")
    print(f"baseline held-out RMS {rms(Yte):.3f} um  "
          f"(gate: <= {0.7*rms(Yte):.3f} for -30%)\n")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    # ---- arm 1: constant bias
    b = Ytr.mean(0)
    evaluate("bias", np.tile(b, (len(Ytr), 1)), np.tile(b, (len(Yte), 1)),
             Ytr, Yte, te_vols, lambda X: np.tile(b, (len(X), 1)))

    # ---- arm 2: ridge
    lam = 10.0
    A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
    Wr = np.linalg.solve(A, Ztr.T @ (Ytr - b))
    ridge = lambda Z: Z @ Wr + b  # noqa: E731
    evaluate("ridge", ridge(Ztr), ridge(Zte), Ytr, Yte, te_vols,
             lambda X: ridge((X - mu) / sd))

    # ---- arm 3: MLP
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(Ztr.shape[1], 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 3))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(Ztr).float()
    Yt = torch.from_numpy(Ytr).float()
    n = len(Xt)
    val_n = max(1000, n // 10)
    perm = torch.randperm(n)
    tr_idx, val_idx = perm[val_n:], perm[:val_n]
    best, best_state, patience = np.inf, None, 0
    for epoch in range(200):
        net.train()
        for i in range(0, len(tr_idx), 4096):
            idx = tr_idx[i:i + 4096]
            opt.zero_grad()
            loss = nn.functional.huber_loss(net(Xt[idx]), Yt[idx], delta=1.0)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(nn.functional.huber_loss(net(Xt[val_idx]), Yt[val_idx],
                                                delta=1.0))
        if vl < best - 1e-5:
            best, patience = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    net.load_state_dict(best_state)
    net.eval()

    def mlp_predict(X):
        with torch.no_grad():
            return net(torch.from_numpy((X - mu) / sd).float()).numpy()

    evaluate("mlp", mlp_predict(Xtr), mlp_predict(Xte),
             Ytr, Yte, te_vols, mlp_predict)

    # ---- arm 4: small 3D CNN over the 7x7x7 logit patch + point features.
    # The flattened MLP has no spatial prior; the z-profile of the heatmap is
    # the natural signal for a z error, and convolution encodes that.
    P = 7
    n_patch = P ** 3
    n_point = Ztr.shape[1] - n_patch

    class PatchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool3d(1))
            self.fc = nn.Sequential(
                nn.Linear(32 + n_point, 64), nn.ReLU(), nn.Linear(64, 3))

        def forward(self, x):
            pt, pa = x[:, :n_point], x[:, n_point:]
            h = self.conv(pa.reshape(-1, 1, P, P, P)).flatten(1)
            return self.fc(torch.cat([pt, h], dim=1))

    # NOTE: dump layout is [32 unet feats, patch(343), peak_logit] -> reorder so
    # point features (unet + peak logit) come first, patch last.
    def reorder(Z):
        return np.concatenate([Z[:, :32], Z[:, -1:], Z[:, 32:32 + n_patch]], axis=1)

    torch.manual_seed(0)
    cnn = PatchNet()
    opt = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt2 = torch.from_numpy(reorder(Ztr)).float()
    best, best_state, patience = np.inf, None, 0
    for epoch in range(120):
        cnn.train()
        for i in range(0, len(tr_idx), 2048):
            idx = tr_idx[i:i + 2048]
            opt.zero_grad()
            loss = nn.functional.huber_loss(cnn(Xt2[idx]), Yt[idx], delta=1.0)
            loss.backward()
            opt.step()
        cnn.eval()
        with torch.no_grad():
            vl = float(nn.functional.huber_loss(cnn(Xt2[val_idx]), Yt[val_idx],
                                                delta=1.0))
        if vl < best - 1e-5:
            best, patience = vl, 0
            best_state = {k: v.clone() for k, v in cnn.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    cnn.load_state_dict(best_state)
    cnn.eval()

    def cnn_predict(X):
        Z = reorder((X - mu) / sd)
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), 8192):
                out.append(cnn(torch.from_numpy(Z[i:i + 8192]).float()).numpy())
        return np.concatenate(out)

    n_point_stated = n_point  # 33 = 32 unet + 1 peak logit
    r1r, r0 = evaluate("cnn", cnn_predict(Xtr), cnn_predict(Xte),
                       Ytr, Yte, te_vols, cnn_predict)

    # Per-axis decomposition for the best arm (which axis is predictable?)
    for name, fn in (("mlp", mlp_predict), ("cnn", cnn_predict)):
        left = Yte - fn(Xte)
        pre = np.sqrt(np.mean(Yte ** 2, axis=0))
        post = np.sqrt(np.mean(left ** 2, axis=0))
        print(f"  per-axis RMS ({name}):  z {pre[0]:.3f}->{post[0]:.3f}  "
              f"y {pre[1]:.3f}->{post[1]:.3f}  x {pre[2]:.3f}->{post[2]:.3f}")

    print("\n==== 40.5 GATE ====")
    red = (r0 - r1r) / r0
    print(f"post-rounding held-out reduction (cnn): {red:+.1%}  "
          f"({'PASS >= 30%' if red >= 0.30 else 'FAIL < 30%'})")


if __name__ == "__main__":
    main()
