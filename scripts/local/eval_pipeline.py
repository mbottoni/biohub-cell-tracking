#!/usr/bin/env python3
"""Score a trained detector through the full pipeline: UNet -> NMS -> ILP -> metric.

Peak recall (what training reports) is a proxy; this gives the real edge-Jaccard score on
held-out volumes via the verified metric, so we know what a submission would score before
spending one.

    python scripts/eval_pipeline.py --ckpt models/unet_fold0.pt \
        --raw .../train --gt .../train --datasets 6bba_x 44b6_y --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.data.volume import ISOTROPIC_SUBSAMPLE, open_volume, to_isotropic  # noqa: E402
from ctrack.detect.blob import detect_frame  # noqa: E402
from ctrack.eval.biohub import SCALE, Graph, evaluate, per_sample_metrics, summarise  # noqa: E402
from ctrack.link.hungarian import build_graph, close_gaps  # noqa: E402


def detect_unet(model, vol, device, threshold, subsample=ISOTROPIC_SUBSAMPLE):
    import torch

    out = {}
    for t in range(vol.n_frames):
        iso = to_isotropic(vol.normalized_frame(t), subsample)
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(iso))[None, None].float().to(device)
            hm = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        coords, scores = detect_frame(hm, np.array([1.625, 1.625, 1.625]), threshold=threshold)
        out[t] = (coords * np.array(subsample), scores)
    return out


def main() -> None:
    import torch

    from ctrack.detect.unet3d import UNet3D

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sweep", type=float, nargs="+", help="sweep these thresholds (detect once)")
    ap.add_argument("--ilp", action="store_true", default=True)
    ap.add_argument("--hungarian", action="store_true", help="use Hungarian instead of ILP")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu")
    model = UNet3D(base=state["base"]).to(device).eval()
    model.load_state_dict(state["model"])
    print(f"loaded {args.ckpt}: base {state['base']}, train peak-recall {state['recall']:.3f}")

    from ctrack.detect.blob import rethreshold
    from ctrack.link.ilp import link_ilp

    thresholds = sorted(args.sweep or [args.threshold])
    # detect once at the lowest threshold, then re-filter (heatmap peaks are threshold-independent)
    base = {}
    for ds in args.datasets:
        vol = open_volume(Path(args.raw) / f"{ds}.zarr")
        gt = load_geff(Path(args.gt) / f"{ds}.geff")
        base[ds] = (gt, detect_unet(model, vol, device, thresholds[0]))
        print(f"  detected {ds}", file=sys.stderr)

    for thr in thresholds:
        rows = []
        for ds, (gt, det0) in base.items():
            det = rethreshold(det0, thr)
            nodes, h_edges = build_graph(det, SCALE, max_displacement_um=12)
            if args.hungarian:
                nodes, edges = close_gaps(nodes, h_edges, SCALE, max_total_um=12)
            else:
                edges = link_ilp(nodes, SCALE, max_disp_um=12, topk=3,
                                 appearance_weight=2.0, disappearance_weight=2.0)
            m = per_sample_metrics(evaluate(gt.graph, Graph(nodes=nodes, edges=edges)),
                                   gt.estimated_number_of_nodes)
            rows.append(m)
        s = summarise(rows)
        mean_nodes = np.mean([r["num_pred_nodes"] for r in rows])
        mean_recall = np.mean([r["node_recall"] for r in rows])
        print(f"thr {thr:.2f}  score {s['score']:.4f}  edge_J {s['edge_jaccard']:.4f}  "
              f"adj {s['adj_edge_jaccard']:.4f}  node_recall {mean_recall:.3f}  "
              f"mean_nodes {mean_nodes:.0f}")


if __name__ == "__main__":
    main()
