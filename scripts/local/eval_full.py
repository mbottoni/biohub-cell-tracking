#!/usr/bin/env python3
"""Full owned pipeline eval: detector(+TTA/ensemble) -> multi-hypothesis candidates ->
greedy/ILP linker (optionally with a learned edge model) -> verified metric.

This exercises all the upgrades together so we can measure the real golden-8 score:
  --linker greedy|ilp     greedy is fast (no SCIP); ilp is global
  --multi-hyp             confidence-weighted node selection (ILP) / singleton pruning (greedy)
  --edge-model            train an EdgeMLP on --edge-train datasets, use its affinities
  --tta                   test-time flip averaging
  --ckpt a.pt b.pt ...    ensemble (average heatmaps)

Example:
  python scripts/local/eval_full.py --ckpt w/unet_fold0.pt --raw R --gt G \
    --datasets <golden8> --linker greedy --multi-hyp --sweep 0.5 0.7 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.data.volume import ISOTROPIC_SUBSAMPLE, open_volume, to_isotropic  # noqa: E402
from ctrack.detect.blob import detect_frame, rethreshold  # noqa: E402
from ctrack.detect.infer import predict_heatmap, predict_heatmap_ensemble  # noqa: E402
from ctrack.eval.biohub import SCALE, Graph, evaluate, per_sample_metrics, summarise  # noqa: E402
from ctrack.link.candidates import build_nodes, build_nodes_with_offsets, prune_singletons  # noqa: E402
from ctrack.link.edge_model import affinity_map, motion_affinity, train_edge_model  # noqa: E402
from ctrack.link.greedy import link_greedy  # noqa: E402
from ctrack.link.ilp import link_ilp  # noqa: E402


def detect(models, vol, device, thr_low, tta, motion=False, subsample=ISOTROPIC_SUBSAMPLE):
    """Detect once at the low threshold; peaks carry intensity so a sweep re-filters. When
    `motion`, also extract the predicted parent-offset at each peak (needs an offset-head model;
    single-model only) and return (coords, scores, offsets) with offsets in full-res voxels."""
    out = {}
    off_ok = motion and len(models) == 1 and getattr(models[0], "offset_head", None) is not None
    for t in range(vol.n_frames):
        iso = to_isotropic(vol.normalized_frame(t), subsample)
        if off_ok:
            hm, off = predict_heatmap(models[0], iso, device, tta=tta, return_offset=True)
        else:
            hm = (predict_heatmap_ensemble(models, iso, device, tta=tta) if len(models) > 1
                  else predict_heatmap(models[0], iso, device, tta=tta))
            off = None
        coords, scores = detect_frame(hm, np.array([1.625, 1.625, 1.625]), threshold=thr_low)
        if off is not None and len(coords):
            ci = np.round(coords).astype(int)
            ci = np.clip(ci, 0, np.array(hm.shape) - 1)
            peak_off = off[:, ci[:, 0], ci[:, 1], ci[:, 2]].T * np.array(subsample)  # iso->full
            out[t] = (coords * np.array(subsample), scores, peak_off)
        elif off_ok:
            out[t] = (coords * np.array(subsample), scores, np.zeros((len(coords), 3)))
        else:
            out[t] = (coords * np.array(subsample), scores)
    return out


def main() -> None:
    import torch

    from ctrack.detect.unet3d import UNet3D

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints (ensemble)")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--edge-train", nargs="+", default=[], help="datasets to fit the edge model on")
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.5])
    ap.add_argument("--linker", choices=["greedy", "ilp"], default="greedy")
    ap.add_argument("--multi-hyp", action="store_true")
    ap.add_argument("--edge-model", action="store_true")
    ap.add_argument("--motion", action="store_true",
                    help="use the detector's parent-offset head for label-free motion affinity")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--max-disp", type=float, default=12.0)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for c in args.ckpt:
        st = torch.load(c, map_location="cpu")
        m = UNet3D(base=st["base"], out_offset=st.get("out_offset", False)).to(device).eval()
        m.load_state_dict(st["model"])
        models.append(m)
    print(f"loaded {len(models)} model(s); linker={args.linker} multi_hyp={args.multi_hyp} "
          f"edge_model={args.edge_model} tta={args.tta}", file=sys.stderr)

    edge_model = None
    if args.edge_model:
        if not args.edge_train:
            sys.exit("--edge-model needs --edge-train datasets (disjoint from --datasets)")
        assert not (set(args.edge_train) & set(args.datasets)), "edge-train leaks into eval set"
        graphs = [load_geff(Path(args.gt) / f"{d}.geff").graph for d in args.edge_train]
        edge_model = train_edge_model(graphs, SCALE, max_disp_um=args.max_disp, verbose=True)

    thresholds = sorted(args.sweep)
    base = {}
    for ds in args.datasets:
        vol = open_volume(Path(args.raw) / f"{ds}.zarr")
        gt = load_geff(Path(args.gt) / f"{ds}.geff")
        base[ds] = (gt, detect(models, vol, device, thresholds[0], args.tta, motion=args.motion))
        print(f"  detected {ds}", file=sys.stderr)

    def _rethr(det0, thr):
        """Threshold-filter, preserving the optional offsets channel (3-tuples)."""
        out = {}
        for t, tup in det0.items():
            keep = tup[1] >= thr
            out[t] = tuple([tup[0][keep], tup[1][keep]] + ([tup[2][keep]] if len(tup) > 2 else []))
        return out

    for thr in thresholds:
        rows = []
        for ds, (gt, det0) in base.items():
            det = _rethr(det0, thr)
            has_off = len(next(iter(det.values()))) > 2 if det else False
            if has_off:
                nodes, node_scores, node_off = build_nodes_with_offsets(det)
            else:
                nodes, node_scores = build_nodes(det)
                node_off = {}
            values = node_scores  # detector confidence per node id
            if args.motion and has_off:
                aff = motion_affinity(nodes, node_off, max_disp_um=args.max_disp)
            elif edge_model is not None:
                aff = affinity_map(edge_model, nodes, values, max_disp_um=args.max_disp)
            else:
                aff = None
            if args.linker == "greedy":
                edges = link_greedy(nodes, SCALE, max_disp_um=args.max_disp,
                                    edge_affinity=aff, allow_division=True)
                if args.multi_hyp:
                    nodes, edges = prune_singletons(nodes, edges, node_scores, min_score=thr)
            else:
                edges = link_ilp(nodes, SCALE, max_disp_um=args.max_disp, topk=3,
                                 edge_affinity=aff,
                                 node_scores=node_scores if args.multi_hyp else None,
                                 appearance_weight=2.0, disappearance_weight=2.0)
            m = per_sample_metrics(evaluate(gt.graph, Graph(nodes=nodes, edges=edges)),
                                   gt.estimated_number_of_nodes)
            rows.append(m)
        s = summarise(rows)
        mean_nodes = np.mean([r["num_pred_nodes"] for r in rows])
        print(f"thr {thr:.2f}  score {s['score']:.4f}  edge_J {s['edge_jaccard']:.4f}  "
              f"adj {s['adj_edge_jaccard']:.4f}  div_J {s.get('division_jaccard', 0):.4f}  "
              f"mean_nodes {mean_nodes:.0f}")


if __name__ == "__main__":
    main()
