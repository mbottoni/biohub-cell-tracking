#!/usr/bin/env python3
"""TRACKASTRA at SCORE level — blend its edge weights into a pred_cache, then replay.

Row 89 measured that Trackastra carries INDEPENDENT signal on our own candidate
edges: it is WORSE alone on every volume (top-1 0.69-0.89 vs ours 0.96-1.00) and
yet blending it at 0.15 IMPROVED every volume, 781/801 -> 787/801 decisions. That
is the ensemble signature -- decorrelated error, not raw strength. §37's kill does
not cover it: §37 killed OUR OWN learned cost, same feature family, same sparse GT.

But row 89 is a RANKING result, not a score. The pipeline's ILP consumes these
probabilities inside a global objective with many other terms, and §37 is the
standing warning that a better-ranking cost can still lose in the objective. This
script closes that gap.

MECHANISM -- the cache is the interface. The replay driver caches
`(coords, edges)` per volume, where `edges[:,2]` is the edge probability the whole
downstream stack (ILP, motion relink, gap closing, post-proc) consumes. So we do
NOT need to touch inference: rewrite column 2 in a COPY of the cache and replay,
which is cache-hit and takes minutes instead of hours.

    new_prob = (1-w) * pipeline_prob + w * trackastra_weight     (pairs both scored)
    new_prob = pipeline_prob                                      (pairs TA did not score)

⚠ THE HANDICAP, restated so the result is read correctly (row 89). Trackastra
consumes instance MASKS; we have points, so we paint spheres. That makes 2 of the
`ctc` model's 4 `regionprops2` features (`equivalent_diameter_area`,
`inertia_tensor`) CONSTANT -- values it never saw in training. Only
`intensity_mean` and `border_dist` survive. Any gain here is a LOWER bound.

⚠ Writes a NEW cache dir; never mutates the source (row 93: a write-through cost
us the support pack and two invalid experiments).

    <tv>/bin/python scripts/local/trackastra_blend_cache.py \
        --src-cache <dir> --dst-cache <dir> --weight 0.15
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "local"))

RAW = Path("/Volumes/extSSD/kaggle/biohub/raw/train")
DS = 4  # [1,4,4] subsample -> 64^3 isotropic at 1.625 um


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-cache", required=True)
    ap.add_argument("--dst-cache", required=True)
    ap.add_argument("--weight", type=float, default=0.15)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    help="blend RAW Trackastra weights (distribution-shifting)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-dir", default="/private/tmp/claude-501/"
                    "-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
                    "f017a762-549a-440e-b9c3-104caa286e68/scratchpad/ta_models")
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)
    import zarr
    from trackastra.model import Trackastra
    from trackastra_gate import build_masks

    src, dst = Path(args.src_cache), Path(args.dst_cache)
    assert src.resolve() != dst.resolve(), "refusing to write over the source cache"
    dst.mkdir(parents=True, exist_ok=True)

    model = Trackastra.from_pretrained("ctc", device=args.device,
                                       download_dir=Path(args.model_dir))
    print(f"[blend] ctc loaded on {args.device}; weight={args.weight}")
    print(f"[blend] {src}  ->  {dst}\n")

    tot = dict(edges=0, blended=0, unmatched=0)
    for f in sorted(src.glob("*.pkl")):
        v = f.stem
        coords, edges_orig = pickle.loads(f.read_bytes())
        # ⚠ PRESERVE THE CONTAINER TYPE. The pipeline's build_graph() does
        # `if edges:` -- fine for the list the cache actually stores, but a
        # ValueError on a numpy array ("truth value of an array ... ambiguous").
        # Writing back an ndarray crashes inference on the FIRST volume.
        _was_list = isinstance(edges_orig, list)
        coords = np.asarray(coords)
        edges = np.asarray(edges_orig, dtype=np.float64)
        n_t = int(coords[:, 0].max()) + 1

        z = zarr.open(str(RAW / f"{v}.zarr"), mode="r")["0"]
        imgs = np.asarray(z[:n_t, :, ::DS, ::DS]).astype(np.float32)
        masks = build_masks(coords, imgs.shape[1:], n_t, args.radius)
        preds = model._predict(imgs, masks, edge_threshold=0.05,
                               progbar_class=lambda x, **k: x)

        nid2row = {int(n["id"]): int(n["label"]) - 1 for n in preds["nodes"]}
        ta: dict[tuple[int, int], float] = {}
        for (i, j), w in preds["weights"]:
            r1, r2 = nid2row.get(int(i)), nid2row.get(int(j))
            if r1 is None or r2 is None:
                continue
            if int(coords[r1][0]) > int(coords[r2][0]):
                r1, r2 = r2, r1
            ta[(r1, r2)] = float(w)

        out = edges.copy()
        idx = [k for k in range(len(out))
               if (int(out[k, 0]), int(out[k, 1])) in ta]
        raw = np.array([ta[(int(out[k, 0]), int(out[k, 1]))] for k in idx])
        if args.calibrate and len(raw) > 10:
            # ⚠ QUANTILE-CALIBRATE Trackastra onto the pipeline's own probability
            # distribution before blending. Its weights live on a different scale:
            # a raw blend at w=0.15 moved the MEAN edge prob 0.9486 -> 0.8702, a
            # uniform ~0.08 downshift. The pipeline gates on ABSOLUTE thresholds
            # (DUAL_SEED_EDGE_THRESHOLD 0.48, edge threshold 0.48), so that shift
            # would drop edges through the gates for reasons unrelated to ranking
            # quality -- and ranking is all row 89 actually measured (ROC/top-1 are
            # scale-invariant). Calibrating makes the blend change ORDER, not the
            # marginal distribution, which is the hypothesis under test.
            ref = np.sort(out[idx, 2])
            r = raw.argsort().argsort()                     # ranks of TA weights
            raw = ref[(r * (len(ref) - 1) // max(1, len(r) - 1))]
        for k, tv in zip(idx, raw):
            out[k, 2] = (1.0 - args.weight) * out[k, 2] + args.weight * float(tv)
        hit = len(idx)
        if _was_list:
            # rebuild element-by-element from the ORIGINAL rows, replacing only
            # column 2, so element type and any extra columns survive untouched
            out_obj = []
            for k, row in enumerate(edges_orig):
                r = list(row)
                r[2] = float(out[k, 2])
                out_obj.append(type(row)(r) if isinstance(row, tuple) else r)
        else:
            out_obj = out
        (dst / f.name).write_bytes(pickle.dumps((coords, out_obj)))
        tot["edges"] += len(out); tot["blended"] += hit
        tot["unmatched"] += len(out) - hit
        print(f"  {v:16s} edges {len(out):>7,}  blended {hit:>7,} "
              f"({hit/max(1,len(out))*100:4.1f}%)  TA scored {len(ta):,}")

    print(f"\n[blend] TOTAL edges {tot['edges']:,} | blended {tot['blended']:,} "
          f"({tot['blended']/max(1,tot['edges'])*100:.1f}%) | untouched {tot['unmatched']:,}")
    print(f"[blend] wrote {dst}")
    print("[blend] next: replay with BIOHUB_PRED_CACHE=<dst> (cache-hit), then score")


if __name__ == "__main__":
    main()
