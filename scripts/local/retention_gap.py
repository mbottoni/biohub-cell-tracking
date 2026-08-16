#!/usr/bin/env python3
"""RETENTION LEAK — how many ANNOTATED cells does post-processing delete after
detection already found them, and what is that worth in EDGES?

Row 87 closed the detection-side routes: raw detection recall on annotated cells
is 0.9958 and the whole DET_THRESHOLD ladder hides only 9 more. But the same
measurement left a residue -- final-submission recall came out LOWER than raw
detection recall, i.e. the pipeline throws away cells it had already found
correctly. This script prices that residue properly, on a MATCHED volume set
(row 87's 0.9958-vs-0.9895 compared 21 volumes against 24 and is therefore not
a fair comparison -- fixing that is step one and can kill the route outright).

The pipeline instruments the deletion itself (`reverify_stack/run_stats.csv`,
24 volumes, the sha-87d51dda stack):

    short_track_components_removed    1,516
    short_track_nodes_removed         6,410
    short_track_edges_removed         4,894      (1.02% of final edges)
    short_track_rescue_triggered          0      <-- never fires

so the question is not whether nodes are deleted, it is what FRACTION of them
are annotated. GT is ~0.2% dense: if the deleted set were random, ~13 of the
6,410 would be annotated and the route is dead. If short tracks are enriched for
real-but-hard-to-track cells, it is worth much more.

Scoring is on EDGES, so node counts are only the intermediate. The bound that
matters is: GT edges whose endpoints we DID detect but which are absent from the
final graph. Those are recoverable in principle by retention alone.

⚠ ORACLE. Uses GT to identify what was lost. Not a submittable config -- it is
the ceiling any real (GT-blind) retention rule must be judged against.

    .venv/bin/python scripts/local/retention_gap.py
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from ctrack.data.geff import load_geff        # noqa: E402
from ladder_oracle import match_frame, GT_ROOT  # noqa: E402

OLD = Path("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
           "746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad")


def matched_map(coords: np.ndarray, gt_t: np.ndarray,
                gt_zyx: np.ndarray) -> dict[int, int]:
    """GT global index -> matching pred row index (metric's rule, per frame)."""
    out: dict[int, int] = {}
    for t in np.unique(gt_t):
        gsel = np.flatnonzero(gt_t == t)
        psel = np.flatnonzero(coords[:, 0] == t)
        if not len(gsel) or not len(psel):
            continue
        for pi, gj in match_frame(coords[psel, 1:].astype(np.float64), gt_zyx[gsel]):
            out[int(gsel[gj])] = int(psel[pi])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-cache", default=str(OLD / "pred_cache"))
    ap.add_argument("--final", default=str(OLD / "runs/reverify_stack/submission.csv"),
                    help="the sha-87d51dda stack submission (local 0.9135)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_cache)

    # ---- final submission: nodes and edges per dataset -----------------------
    fin_nodes: dict[str, list] = defaultdict(list)
    fin_edge_ids: dict[str, list] = defaultdict(list)
    node_rowid: dict[str, dict[int, int]] = defaultdict(dict)
    with open(args.final) as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]
            if r["row_type"] == "node":
                node_rowid[ds][int(r["node_id"])] = len(fin_nodes[ds])
                fin_nodes[ds].append((int(float(r["t"])), int(float(r["z"])),
                                      int(float(r["y"])), int(float(r["x"]))))
            else:
                fin_edge_ids[ds].append((int(r["source_id"]), int(r["target_id"])))

    vols = sorted(p.stem for p in raw_dir.glob("*.pkl") if p.stem in fin_nodes)
    print(f"[retention] matched volume set: {len(vols)} "
          f"(raw cache {len(list(raw_dir.glob('*.pkl')))}, final {len(fin_nodes)})")
    print(f"[retention] final = {Path(args.final).parent.name}\n")

    T = dict(gt=0, raw_hit=0, fin_hit=0, lost=0,
             gt_edges=0, matched_edges=0, recoverable=0, both_detected=0,
             by_retention=0, by_linking=0)
    rows = []
    for v in vols:
        raw_coords, _ = pickle.loads((raw_dir / f"{v}.pkl").read_bytes())
        raw_coords = np.asarray(raw_coords)
        fin_coords = np.array(fin_nodes[v])

        gt = load_geff(GT_ROOT / f"{v}.geff")
        g = gt.graph.nodes.astype(np.float64)
        assert g.shape[1] == 5, "geff nodes are [id,t,z,y,x]"
        gt_t, gt_zyx = g[:, 1].astype(int), g[:, 2:5]
        gt_id_to_idx = {int(i): k for k, i in enumerate(g[:, 0])}

        m_raw = matched_map(raw_coords, gt_t, gt_zyx)
        m_fin = matched_map(fin_coords, gt_t, gt_zyx)
        lost = set(m_raw) - set(m_fin)   # detected, then deleted

        # ---- edge-level bound -----------------------------------------------
        # A GT edge is RECOVERABLE by retention iff we detected BOTH endpoints
        # but the final graph does not carry the edge between their matches.
        fin_pair = {(node_rowid[v].get(s, -1), node_rowid[v].get(d, -1))
                    for s, d in fin_edge_ids[v]}
        fin_rev = {b: a for a, b in m_fin.items()}  # pred row -> gt idx
        del fin_rev
        ge = gt.graph.edges
        n_gt_e = len(ge)
        # Split the recoverable set by CAUSE -- this is the decisive number:
        #   RETENTION : an endpoint was detected and then deleted downstream
        #   LINKING   : both endpoints survive, the pipeline just never linked them
        both_det = matched = recoverable = 0
        by_retention = by_linking = 0
        for s_id, d_id in ge.astype(np.int64):
            si, di = gt_id_to_idx.get(int(s_id)), gt_id_to_idx.get(int(d_id))
            if si is None or di is None:
                continue
            if si in m_raw and di in m_raw:
                both_det += 1
                if si in m_fin and di in m_fin:
                    if (m_fin[si], m_fin[di]) in fin_pair:
                        matched += 1
                    else:
                        recoverable += 1
                        by_linking += 1    # both kept, edge absent
                else:
                    recoverable += 1
                    by_retention += 1      # an endpoint was DELETED

        rows.append((v, len(gt_t), len(m_raw), len(m_fin), len(lost),
                     n_gt_e, both_det, matched, recoverable, by_retention, by_linking))
        T["gt"] += len(gt_t); T["raw_hit"] += len(m_raw)
        T["fin_hit"] += len(m_fin); T["lost"] += len(lost)
        T["gt_edges"] += n_gt_e; T["both_detected"] += both_det
        T["matched_edges"] += matched; T["recoverable"] += recoverable
        T["by_retention"] += by_retention; T["by_linking"] += by_linking

    hdr = (f"{'volume':<16}{'GT':>6}{'raw':>6}{'final':>6}{'LOST':>6}"
           f"{'gtE':>7}{'bothDet':>8}{'matchE':>8}{'recov':>7}"
           f"{'RETEN':>7}{'LINK':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:<16}{r[1]:>6}{r[2]:>6}{r[3]:>6}{r[4]:>6}"
              f"{r[5]:>7}{r[6]:>8}{r[7]:>8}{r[8]:>7}{r[9]:>7}{r[10]:>7}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<16}{T['gt']:>6}{T['raw_hit']:>6}{T['fin_hit']:>6}{T['lost']:>6}"
          f"{T['gt_edges']:>7}{T['both_detected']:>8}{T['matched_edges']:>8}"
          f"{T['recoverable']:>7}")

    print(f"\n[retention] node recall  raw {T['raw_hit']/T['gt']:.4f} "
          f"-> final {T['fin_hit']/T['gt']:.4f}   "
          f"LOST {T['lost']} annotated cells (SAME volume set)")
    print(f"[retention] GT edges {T['gt_edges']}, both endpoints detected "
          f"{T['both_detected']}, currently matched {T['matched_edges']}")
    print(f"[retention] RECOVERABLE GT edges (endpoints detected, edge absent): "
          f"{T['recoverable']}")
    print(f"[retention]   ├─ by RETENTION (an endpoint was deleted): {T['by_retention']}"
          f"  ({T['by_retention']/max(1,T['recoverable'])*100:.1f}%)")
    print(f"[retention]   └─ by LINKING   (both kept, never linked):  {T['by_linking']}"
          f"  ({T['by_linking']/max(1,T['recoverable'])*100:.1f}%)")
    if T["gt_edges"]:
        print(f"[retention] ceiling on edge recall from perfect retention+linking: "
              f"{T['matched_edges']/T['gt_edges']:.4f} -> "
              f"{(T['matched_edges']+T['recoverable'])/T['gt_edges']:.4f}")


if __name__ == "__main__":
    main()
