#!/usr/bin/env python3
"""Train the learned edge-affinity model on GT trajectories.

For every GT node, its true successor in the next frame is a positive pair; its other GT
neighbours within the linking radius are negatives. This is exactly the discrimination the ILP
needs — "which nearby candidate is the real continuation" — and it trains in seconds because
it is feature-based (ctrack.link.edge_model).

    python scripts/train_edge.py --raw .../train --gt .../train --fold 0 --out models/edge_fold0.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ctrack.data.geff import find_geff_stores, load_geff  # noqa: E402
from ctrack.data.splits import hold_out_golden, kfold  # noqa: E402
from ctrack.data.volume import ISOTROPIC_SUBSAMPLE, open_volume, to_isotropic  # noqa: E402
from ctrack.eval.biohub import SCALE  # noqa: E402
from ctrack.link.edge_model import pair_features  # noqa: E402

MAX_DISP_UM = 12.0


def _intensity_at(vol, t: int, zyx_full: np.ndarray) -> np.ndarray:
    """Normalized intensity at full-res voxel coords, sampled from the isotropic frame."""
    iso = to_isotropic(vol.normalized_frame(t), ISOTROPIC_SUBSAMPLE)
    idx = np.round(zyx_full / np.array(ISOTROPIC_SUBSAMPLE)).astype(int)
    idx = np.clip(idx, 0, np.array(iso.shape) - 1)
    return iso[idx[:, 0], idx[:, 1], idx[:, 2]]


def build_pairs(datasets: list[str], raw: Path, gt_root: Path):
    """Return (features, labels) over all positive+negative candidate pairs in `datasets`."""
    feats, labels = [], []
    for ds in datasets:
        gt = load_geff(gt_root / f"{ds}.geff")
        g = gt.graph
        vol = open_volume(raw / f"{ds}.zarr")
        ids, t, zyx = g.ids, g.t.astype(int), g.zyx
        id_to_row = {int(i): k for k, i in enumerate(ids)}
        succ = {int(s): int(d) for s, d in g.edges}  # true GT successor

        intens_cache: dict[int, np.ndarray] = {}
        for f in np.unique(t):
            cur = np.flatnonzero(t == f)
            nxt = np.flatnonzero(t == f + 1)
            if cur.size == 0 or nxt.size == 0:
                continue
            if f not in intens_cache:
                intens_cache[f] = _intensity_at(vol, int(f), zyx[cur])
            iv_cur = intens_cache[f]
            iv_nxt = _intensity_at(vol, int(f + 1), zyx[nxt])
            d = np.linalg.norm((zyx[cur][:, None, :] - zyx[nxt][None, :, :]) * SCALE, axis=2)
            for i, ci in enumerate(cur):
                in_radius = np.flatnonzero(d[i] <= MAX_DISP_UM)
                if in_radius.size == 0:
                    continue
                true_tgt = succ.get(int(ids[ci]))
                for j in in_radius:
                    is_pos = true_tgt is not None and int(ids[nxt[j]]) == true_tgt
                    ft = pair_features(
                        zyx[ci][None], zyx[nxt[j]][None],
                        np.array([iv_cur[i]]), np.array([iv_nxt[j]]),
                    )
                    feats.append(ft[0])
                    labels.append(1.0 if is_pos else 0.0)
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.float32)


def main() -> None:
    import torch

    from ctrack.link.edge_model import EdgeMLP

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="models/edge.pt")
    args = ap.parse_args()

    all_ds = sorted(p.name.replace(".geff", "") for p in find_geff_stores(args.gt))
    remaining, _golden = hold_out_golden(all_ds, n=8, seed=0)
    split = kfold(remaining, k=args.k, seed=0)[args.fold]
    train_ds, val_ds = split.train, split.val
    if args.limit:
        train_ds, val_ds = train_ds[: args.limit], val_ds[: args.limit]

    print(f"building pairs: {len(train_ds)} train, {len(val_ds)} val datasets ...")
    Xtr, ytr = build_pairs(train_ds, Path(args.raw), Path(args.gt))
    Xva, yva = build_pairs(val_ds, Path(args.raw), Path(args.gt))
    print(f"train pairs {len(ytr)} ({ytr.mean():.1%} pos), val {len(yva)} ({yva.mean():.1%} pos)")

    model = EdgeMLP()
    model.mean.copy_(torch.from_numpy(Xtr.mean(0)))
    model.std.copy_(torch.from_numpy(Xtr.std(0) + 1e-6))

    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)
    pos_weight = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(Xtr_t), ytr_t, pos_weight=pos_weight
        )
        loss.backward()
        opt.step()

        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                pva = torch.sigmoid(model(Xva_t)).numpy()
            # average precision-ish: fraction of positives ranked above 0.5, and AUC proxy
            acc = ((pva > 0.5) == (yva > 0.5)).mean()
            pos_recall = (pva[yva > 0.5] > 0.5).mean() if (yva > 0.5).any() else float("nan")
            print(f"epoch {epoch+1:3d}  loss {float(loss):.4f}  val acc {acc:.3f}  "
                  f"pos-recall {pos_recall:.3f}")
            if acc > best:
                best = acc
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict()}, args.out)

    print(f"saved {args.out} (best val acc {best:.3f})")


if __name__ == "__main__":
    main()
