#!/usr/bin/env python3
"""Does per-volume LOCALISATION ERROR explain the collapsed volumes?

26.6 / 27.1 found volumes where detection is fine (node recall >=96%, oracle
0.95-0.96) and the linker collapses anyway -- `44b6_87bba6c4` adj 0.5743,
`6bba_786893ac` 0.7245. 29 looked for a cause and landed on "scale, not speed"
(annotation density -0.577, disp_p90 +0.460, R^2 ~ 0.21 over n=24), which was
explicitly filed as directional, not a law, and never produced a working lever.

The snap oracle suggests a different cause. On `6bba_786893ac` the mean
predicted-to-GT residual is 2.70 um against 1.68 um on a healthy volume, and
snapping recovers +0.0877 of its adj. This measures that correlation over all
24 volumes.

Residual is computed on matched nodes only (the ones the metric can see), so it
is the localisation error of the cells that actually carry score.

    python scripts/local/localisation_vs_gap.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    MAX_MATCH_DISTANCE_UM, SCALE, Graph, evaluate, match_nodes, per_sample_metrics,
)
from localisation_doseresponse import load_submission  # noqa: E402
from relink_replay import relink_volume  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def one(job) -> dict:
    ds, nodes = job
    gt = load_geff(GT_ROOT / f"{ds}.geff")
    ids = nodes[:, 0].astype(np.int64)
    t_of = {int(i): int(t) for i, t in zip(ids, nodes[:, 1].astype(int))}
    pos_um = nodes[:, 2:5] * SCALE

    matched = match_nodes(gt.graph,
                          Graph(nodes=nodes, edges=np.zeros((0, 2), np.int64)),
                          max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
    gid_row = {int(g[0]): k for k, g in enumerate(gt.graph.nodes)}
    row_of = {int(i): k for k, i in enumerate(ids)}
    gt_um = gt.graph.nodes[:, 2:5] * SCALE
    res = [float(np.linalg.norm(gt_um[gid_row[int(g)]] - pos_um[row_of[int(p)]]))
           for p, g in matched.items() if int(p) in row_of and int(g) in gid_row]

    edges = relink_volume({int(i): pos_um[k] for k, i in enumerate(ids)}, t_of)
    m = per_sample_metrics(
        evaluate(gt.graph, Graph(nodes=nodes,
                                 edges=np.asarray(edges, np.int64) if edges
                                 else np.zeros((0, 2), np.int64))),
        gt.estimated_number_of_nodes)
    both = 0
    gt_to_pred = {g: p for p, g in matched.items()}
    for a, b in gt.graph.edges:
        both += int(a) in gt_to_pred and int(b) in gt_to_pred
    return {"dataset": ds, "adj": m["adj_edge_jaccard"],
            "edge_j": m["edge_jaccard"],
            "oracle": both / max(len(gt.graph.edges), 1),
            "resid": float(np.mean(res)) if res else float("nan"),
            "resid_rms": float(np.sqrt(np.mean(np.square(res)))) if res else float("nan"),
            "gt_edges": int(m["edge_tp"] + m["edge_fn"]), "n_matched": len(res)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    args = ap.parse_args()
    subs = load_submission(Path(args.submission))
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one, [(ds, g.nodes) for ds, g in sorted(subs.items())]))

    rows.sort(key=lambda r: -r["resid"])
    print(f"{'volume':>16} {'GTedges':>8} {'resid':>7} {'ORACLE':>7} {'edge_J':>7} "
          f"{'gap':>8}")
    for r in rows:
        print(f"{r['dataset']:>16} {r['gt_edges']:>8} {r['resid']:>6.3f}u "
              f"{r['oracle']:>7.4f} {r['edge_j']:>7.4f} "
              f"{r['oracle']-r['edge_j']:>+8.4f}")

    thick = [r for r in rows if r["gt_edges"] >= 200]
    for label, rs in (("all 24", rows), (f"{len(thick)} thick", thick)):
        x = np.array([r["resid"] for r in rs])
        gap = np.array([r["oracle"] - r["edge_j"] for r in rs])
        adj = np.array([r["adj"] for r in rs])
        print(f"\n  [{label}] corr(residual, association gap) = "
              f"{np.corrcoef(x, gap)[0,1]:+.3f}")
        print(f"  [{label}] corr(residual, adj)             = "
              f"{np.corrcoef(x, adj)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
