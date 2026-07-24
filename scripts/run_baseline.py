#!/usr/bin/env python3
"""End-to-end baseline: detect -> link -> close gaps -> submission.csv (+ local score).

No learning. The point is to exercise the whole path -- volume reader, detector, linker,
metric harness, submission writer -- and to establish the floor a learned detector must beat.

    # score on train volumes that have ground truth
    python scripts/run_baseline.py --root /Volumes/extSSD/kaggle/biohub/raw --split train --limit 5

    # sweep the detection threshold against the real metric
    python scripts/run_baseline.py --root ... --split train --limit 10 \
        --sweep 0.2 0.3 0.4 0.5 0.6 0.7 0.8

    # produce a submission for the test volumes
    python scripts/run_baseline.py --root ... --split test --out submission.csv

The sweep is the interesting mode. The metric's node-budget multiplier is unclamped above
(docs/metric.md), so under-detecting inflates the score -- the optimum is NOT at peak
detection F1, and the public notebooks appear to be finding it by blind LB probing at 5
submissions/day. This finds it offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.data.volume import find_volumes, open_volume  # noqa: E402
from ctrack.detect.blob import detect_volume, rethreshold  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    SCALE,
    Graph,
    evaluate,
    per_sample_metrics,
    summarise,
)
from ctrack.link.hungarian import build_graph, close_gaps  # noqa: E402
from ctrack.submit.format import edge_row, node_row, write  # noqa: E402


def track_one(det, max_disp: float, gaps: bool, use_ilp: bool = False):
    nodes, edges = build_graph(det, SCALE, max_displacement_um=max_disp)
    if use_ilp:
        from ctrack.link.ilp import link_ilp

        edges = link_ilp(nodes, SCALE, max_disp_um=max_disp, topk=3,
                         appearance_weight=2.0, disappearance_weight=2.0)
        return nodes, edges
    if gaps:
        nodes, edges = close_gaps(nodes, edges, SCALE, max_total_um=max_disp)
    return nodes, edges


def score_one(nodes, edges, geff_path: Path) -> dict:
    gt = load_geff(geff_path)
    counts = evaluate(gt.graph, Graph(nodes=nodes, edges=edges))
    row = per_sample_metrics(counts, gt.estimated_number_of_nodes)
    row["dataset"] = gt.dataset
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="data root containing train/ and test/")
    ap.add_argument("--gt-root", help="directory of .geff stores, if not alongside the .zarr")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--datasets", nargs="*", help="explicit dataset ids to use")
    ap.add_argument("--limit", type=int, help="only process the first N volumes")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sweep", type=float, nargs="+", help="try these thresholds and compare")
    ap.add_argument("--max-disp", type=float, default=10.0, help="linking cap, µm")
    ap.add_argument("--no-gaps", action="store_true", help="disable gap closing")
    ap.add_argument("--ilp", action="store_true", help="use ILP linking instead of Hungarian")
    ap.add_argument("--out", help="write a submission CSV here")
    args = ap.parse_args()

    vols = find_volumes(Path(args.root) / args.split)
    if not vols:
        sys.exit(f"no .zarr stores under {Path(args.root) / args.split}")
    if args.datasets:
        wanted = set(args.datasets)
        vols = [v for v in vols if v.name.replace(".zarr", "") in wanted]
    if args.limit:
        vols = vols[: args.limit]
    print(f"{len(vols)} volumes from {args.split}/\n")

    thresholds = sorted(args.sweep or [args.threshold])
    results = []

    # Detect once at the lowest threshold; higher ones are a cheap re-filter. Volume I/O
    # dominates, so this turns an N-point sweep into one pass over the data.
    print(f"detecting at threshold {thresholds[0]:.2f} ...", file=sys.stderr)
    base: dict[str, tuple] = {}
    for path in vols:
        vol = open_volume(path)
        base[vol.dataset] = (path, detect_volume(vol, threshold=thresholds[0]))
        print(f"  {vol.dataset}: {sum(len(c) for c, _ in base[vol.dataset][1].values())} peaks",
              file=sys.stderr)

    for threshold in thresholds:
        rows, all_nodes = [], []
        for dataset, (path, det0) in base.items():
            det = rethreshold(det0, threshold)
            nodes, edges = track_one(det, args.max_disp, not args.no_gaps, use_ilp=args.ilp)

            gt_dir = Path(args.gt_root) if args.gt_root else path.parent
            geff = gt_dir / f"{dataset}.geff"
            if geff.exists():
                rows.append(score_one(nodes, edges, geff))
            all_nodes.append((dataset, nodes, edges))
            print(f"  [{threshold:.2f}] {dataset}: {len(nodes)} nodes, {len(edges)} edges",
                  file=sys.stderr)

        if rows:
            s = summarise(rows)
            s["threshold"] = threshold
            s["mean_nodes"] = float(np.mean([r["num_pred_nodes"] for r in rows]))
            s["mean_node_ratio"] = float(np.mean([r["total_node_ratio"] for r in rows]))
            s["mean_node_recall"] = float(np.mean([r["node_recall"] for r in rows]))
            results.append(s)
            print(f"  -> score {s['score']:.4f}  edge_J {s['edge_jaccard']:.4f}  "
                  f"node_recall {s['mean_node_recall']:.3f}  nodes {s['mean_nodes']:.0f}")

        if args.out and threshold == thresholds[-1]:
            out_rows = []
            for dataset, nodes, edges in all_nodes:
                for nid, t, z, y, x in nodes:
                    out_rows.append(node_row(dataset, int(nid), int(t), int(z), int(y), int(x)))
                for s_, t_ in edges:
                    out_rows.append(edge_row(dataset, int(s_), int(t_)))
            write(out_rows, args.out)
            print(f"\nwrote {args.out} ({len(out_rows)} rows)")

    if results:
        df = pd.DataFrame(results)[
            ["threshold", "score", "edge_jaccard", "adj_edge_jaccard",
             "mean_node_recall", "mean_nodes", "mean_node_ratio",
             "edge_tp", "edge_fp", "edge_fn"]
        ]
        pd.set_option("display.width", 160)
        print("\n== threshold sweep ==")
        print(df.to_string(index=False))

        best = df.loc[df.score.idxmax()]
        print(f"\nbest threshold {best.threshold:.2f} -> score {best.score:.4f}")
        peak_recall = df.loc[df.mean_node_recall.idxmax()]
        if best.threshold != peak_recall.threshold:
            print(f"  (peak node recall is at {peak_recall.threshold:.2f}, score "
                  f"{peak_recall.score:.4f} -- the metric optimum is NOT peak detection)")


if __name__ == "__main__":
    main()
