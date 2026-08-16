#!/usr/bin/env python3
"""TRACKASTRA GATE — does a foreign pretrained linker carry signal on OUR candidates?

Row 87 closed the detection-side routes (recall on annotated cells is already
0.9958). Row 88 measured retention at +0.0003, saturating, budget-limited. What
row 87 also established is where the remaining loss actually is:

    recoverable GT edges 666 = 139 RETENTION (20.9%) + 527 LINKING (79.1%)

Four fifths of it is association: both endpoints survive to the final graph and
the pipeline never links them. That is what this tests.

Trackastra (Gallusser & Weigert, ECCV 2024; `ctc` checkpoint = successor of the
ISBI-2024 CTC generalizable-linking winner) is a pretrained transformer linker.
It is a genuinely FOREIGN opinion: different architecture, different training
data, different group -- so its errors are decorrelated with our transformer's.
That is exactly what §37's kill does NOT cover (§37 killed OUR OWN learned cost,
same feature family, trained on the same sparse GT, at -0.0002 out of sample).

⚠ THE KNOWN HANDICAP, stated up front so the result is read correctly.
Trackastra consumes instance MASKS; we only have points. We synthesise spheres.
The `ctc` model's features are `regionprops2`:

    equivalent_diameter_area   <- CONSTANT for spheres (dead)
    intensity_mean             <- live (sampled from the image)
    inertia_tensor             <- CONSTANT for spheres (dead)
    border_dist                <- live (position-derived)

so HALF the appearance features go dead, and the two that die are fed constants
the model never saw in training. A null here therefore does NOT falsify
Trackastra on this data -- it falsifies *Trackastra driven by sphere masks*.

WHAT THIS MEASURES (cheap, local, GT-blind at inference).
Not a score. A discrimination test: over OUR OWN candidate edges, how well does
Trackastra's weight separate GT-true from GT-false links, versus our shipped
`edge_prob` on the identical edge set? ROC-AUC on the same pairs, so the two are
directly comparable.

  * AUC ~= 0.5  -> spheres killed it; route dead for the cost of an afternoon.
  * AUC well below ours -> a weak second opinion; blending is unlikely to pay.
  * AUC comparable or better -> a genuinely independent signal. THEN build the
    blend and gate it on evalset-24 against 0.9135.

    .venv/bin/python scripts/local/trackastra_gate.py --volumes 2 --frames 30
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from sklearn.metrics import roc_auc_score  # noqa: E402
from ctrack.data.geff import load_geff  # noqa: E402
from ladder_oracle import GT_ROOT  # noqa: E402
from retention_gap import matched_map  # noqa: E402

OLD = Path("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
           "746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad")
# ⚠ row 67 / row 89: /private/tmp is swept at 00:00 and ate this cache TWICE in
# two days (24 -> 3 survivors on 08-16). The cache lives on the external SSD now.
CACHE = Path(os.environ.get("BIOHUB_PRED_CACHE",
                            "/Volumes/extSSD/kaggle/biohub/pred_cache"))
RAW = Path("/Volumes/extSSD/kaggle/biohub/raw/train")
DS = 4  # [1,4,4] subsample -> 64^3 isotropic at 1.625 um (README)


def build_masks(coords: np.ndarray, shape_zyx: tuple[int, int, int],
                n_t: int, radius: int = 1) -> np.ndarray:
    """Paint a sphere per detection on the DOWNSAMPLED isotropic grid.

    Label = detection row index + 1, so Trackastra's regionprops labels map
    straight back to our candidate-edge indices with no re-matching step.
    ⚠ Later detections overwrite earlier ones where spheres overlap; the label
    -> row map is rebuilt from the painted array afterwards, never assumed.
    """
    Z, Y, X = shape_zyx
    masks = np.zeros((n_t, Z, Y, X), dtype=np.int32)
    off = [(dz, dy, dx)
           for dz in range(-radius, radius + 1)
           for dy in range(-radius, radius + 1)
           for dx in range(-radius, radius + 1)
           if dz * dz + dy * dy + dx * dx <= radius * radius]
    for i, (t, z, y, x) in enumerate(coords):
        t, z = int(t), int(z)
        y, x = int(y) // DS, int(x) // DS
        if not (0 <= t < n_t and 0 <= z < Z and 0 <= y < Y and 0 <= x < X):
            continue
        for dz, dy, dx in off:
            zz, yy, xx = z + dz, y + dy, x + dx
            if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                masks[t, zz, yy, xx] = i + 1
    return masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma-separated volume names")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-dir", default=str(Path(
        "/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
        "f017a762-549a-440e-b9c3-104caa286e68/scratchpad/ta_models")))
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)
    import zarr
    from trackastra.model import Trackastra

    model = Trackastra.from_pretrained("ctc", device=args.device,
                                       download_dir=Path(args.model_dir))
    print(f"[gate] ctc model loaded on {args.device}; "
          f"coord_dim={model.transformer.config['coord_dim']} "
          f"window={model.transformer.config['window']}")

    vols = (args.only.split(",") if args.only else
            sorted(p.stem for p in CACHE.glob("*.pkl"))[:args.volumes])
    for v in vols:
        coords, edges = pickle.loads((CACHE / f"{v}.pkl").read_bytes())
        coords, edges = np.asarray(coords), np.asarray(edges)
        keep = coords[:, 0] < args.frames
        print(f"\n[gate] {v}: {keep.sum()} detections in first {args.frames} frames "
              f"(of {len(coords)}), {len(edges)} candidate edges total")

        z = zarr.open(str(RAW / f"{v}.zarr"), mode="r")["0"]
        imgs = np.asarray(z[:args.frames, :, ::DS, ::DS]).astype(np.float32)
        masks = build_masks(coords[keep], imgs.shape[1:], args.frames, args.radius)
        n_lab = len(np.unique(masks)) - 1
        print(f"[gate] imgs {imgs.shape} masks {masks.shape} "
              f"labels painted {n_lab} (of {keep.sum()} detections; "
              f"{keep.sum() - n_lab} lost to sphere overlap)")

        preds = model._predict(imgs, masks, edge_threshold=0.05,
                               progbar_class=lambda x, **k: x)

        # Trackastra node id -> OUR detection row index, via the painted label.
        rows = np.flatnonzero(keep)
        nid2row = {int(n["id"]): int(rows[int(n["label"]) - 1]) for n in preds["nodes"]}

        # ---- label candidate pairs with GT -----------------------------------
        gt = load_geff(GT_ROOT / f"{v}.geff")
        g = gt.graph.nodes.astype(np.float64)
        assert g.shape[1] == 5, "geff nodes are [id,t,z,y,x]"
        gt_t, gt_zyx = g[:, 1].astype(int), g[:, 2:5]
        gt_idx = {int(i): k for k, i in enumerate(g[:, 0])}
        m_gt = matched_map(coords, gt_t, gt_zyx)        # gt index -> our row
        row2gt = {r: gi for gi, r in m_gt.items()}      # our row  -> gt index
        gt_edges = {(gt_idx[int(a)], gt_idx[int(b)])
                    for a, b in gt.graph.edges.astype(np.int64)
                    if int(a) in gt_idx and int(b) in gt_idx}

        def label(r1: int, r2: int):
            """True/False if BOTH endpoints matched GT; None if unlabelable."""
            a, b = row2gt.get(r1), row2gt.get(r2)
            if a is None or b is None:
                return None
            return (a, b) in gt_edges or (b, a) in gt_edges

        ta, ours = {}, {}
        for (i_, j_), w in preds["weights"]:
            r1, r2 = nid2row.get(int(i_)), nid2row.get(int(j_))
            if r1 is None or r2 is None:
                continue
            if int(coords[r1][0]) > int(coords[r2][0]):
                r1, r2 = r2, r1
            ta[(r1, r2)] = float(w)
        for e in edges:
            r1, r2 = int(e[0]), int(e[1])
            if keep[r1] and keep[r2]:
                if int(coords[r1][0]) > int(coords[r2][0]):
                    r1, r2 = r2, r1
                ours[(r1, r2)] = float(e[2])

        # ---- RANKING TEST -------------------------------------------------
        # Requiring BOTH endpoints annotated yields almost no negatives (GT is
        # ~0.2% dense and annotated cells are tracked, so such pairs are nearly
        # all true). Instead score the decision the linker actually makes: for a
        # GT-tracked cell whose true successor we detected, can the scorer rank
        # that successor above the OTHER candidates leaving the same source?
        # Every competing candidate is a negative by construction.
        by_src = {}
        for (r1, r2) in set(ta) | set(ours):
            by_src.setdefault(r1, []).append(r2)

        def report(name, score):
            top1 = rr = n = 0
            for gi, r1 in m_gt.items():
                succ = [b for (a, b) in gt_edges if a == gi]
                if len(succ) != 1 or succ[0] not in m_gt:
                    continue                      # skip divisions / undetected
                r2 = m_gt[succ[0]]
                cands = by_src.get(r1, [])
                if r2 not in cands or len(cands) < 2:
                    continue                      # need a real choice to make
                sc = [(score.get((r1, c), 0.0), c) for c in cands]
                sc.sort(key=lambda z: -z[0])
                rank = next(k for k, (_, c) in enumerate(sc) if c == r2) + 1
                top1 += rank == 1; rr += 1.0 / rank; n += 1
            if n:
                print(f"        {name:<12} top-1 {top1}/{n} = {top1/n:.4f}   MRR {rr/n:.4f}")
            return n

        n = report("trackastra", ta)
        report("ours", ours)
        for w in (0.15, 0.30, 0.50):
            blend = {k: (1 - w) * ours.get(k, 0.0) + w * ta.get(k, 0.0)
                     for k in set(ta) | set(ours)}
            report(f"blend {w:.2f}", blend)
        print(f"[gate] decisions scored: {n} (GT cells with a detected unique successor "
              f"and >=2 candidates)")



if __name__ == "__main__":
    main()
