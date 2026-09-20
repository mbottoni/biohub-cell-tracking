#!/usr/bin/env python3
"""Stage-by-stage association audit on the 947 densad base (ds_ctl), holdout6.

Three edge sets exist for every volume and none of the archive rows compares them on this line:
  CAND  : transformer candidates (prob > 0.48 after bidir+secondary fusion)  <- cache_densad/<vol>.pkl ['ce']
  ILP   : tracksdata ILP solution                                             <- tracking_repo/predictions/.../<vol>.geff
  FINAL : submission after motion-relink (REPLACES ILP edges) + gap + safe-div + short-track
Node ids are shared (geff id == cache coords index == submission node_id; synthetic gap nodes get new ids).

Reports, per volume and pooled, on GT edges whose endpoints are matched (official 7um per-frame matching
against the FINAL node set):
  * cross-tab CAND x ILP x FINAL  -> how many correct links the relinker DESTROYED (ILP yes, FINAL no)
    vs REPAIRED (ILP no, FINAL yes); how many were never a candidate (unreachable by any bonus)
  * for destroyed links: pred-pred length vs the density group's tight gate, what the relinker did instead
  * ILP-graph edge TP/FP/FN scored directly (official validity semantics) = the norelink counterfactual
    for the edge term on the ILP node set
  * relinker-created edges (FINAL not ILP): length + TP/FP among annotated
  * long ILP edges (> tight gate): survival + correctness
"""
from __future__ import annotations

import csv
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zarr

REPO = Path("/Users/maruanottoni/home/kaggle/biohub-cell-tracking")
sys.path.insert(0, str(REPO / "src"))
from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
RUN = Path("~/biohub_replay947_runs/ds_ctl").expanduser()
CACHE = Path("/Volumes/extSSD/kaggle/biohub/replay947/cache_densad")
GEFF_DIR = RUN / "tracking_repo/predictions/maruanottoni/unet_transformer/split_0"
SUB = RUN / "submission.csv"
DENS = {"low": dict(tight=6.5, relaxed=11.0, bonus=3.0), "middle": dict(tight=6.5, relaxed=9.0, bonus=8.0),
        "high": dict(tight=5.5, relaxed=10.0, bonus=1.0)}


def load_submission(path):
    nodes, edges = defaultdict(list), defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]
            if r["row_type"] == "node":
                nodes[ds].append((int(r["node_id"]), int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"])))
            else:
                edges[ds].append((int(r["source_id"]), int(r["target_id"])))
    return {ds: Graph(nodes=np.asarray(v, float), edges=np.asarray(edges[ds], np.int64).reshape(-1, 2))
            for ds, v in nodes.items()}


def edge_counts(gt_graph, pred_graph):
    """Official-semantics edge TP/FP/FN (pred_valid = out_valid(src) | in_valid(tgt))."""
    matched = match_nodes(gt_graph, pred_graph, max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
    gt_edges = {(int(a), int(b)) for a, b in gt_graph.edges}
    gt_out, gt_in = defaultdict(set), defaultdict(set)
    for a, b in gt_edges:
        gt_out[a].add(b); gt_in[b].add(a)
    tp = fp = 0
    for s, t in pred_graph.edges:
        gs, gt_ = matched.get(int(s)), matched.get(int(t))
        out_valid = gs is not None and len(gt_out.get(gs, ())) > 0
        in_valid = gt_ is not None and len(gt_in.get(gt_, ())) > 0
        if not (out_valid or in_valid):
            continue
        if gs is not None and gt_ is not None and (gs, gt_) in gt_edges:
            tp += 1
        else:
            fp += 1
    fn = len(gt_edges) - tp
    return tp, fp, fn, matched


def main():
    subs = load_submission(SUB)
    pooled = Counter()
    destroyed_rows, repaired_rows, unreachable_rows = [], [], []
    ilp_long_pool = Counter()
    relink_new_pool = Counter()
    fp_origin_pool = Counter()
    cand_margin_rows = []
    per_vol_scores = []

    for ds in sorted(subs):
        final = subs[ds]
        gt = load_geff(GT_ROOT / f"{ds}.geff")
        g = gt.graph
        d = pickle.load(open(CACHE / f"{ds}.pkl", "rb"))
        coords, cand = d["ce"]
        z = zarr.open(str(GEFF_DIR / f"{ds}.geff"), mode="r")
        ilp_ids = z["nodes/ids"][:].astype(int)
        ilp_edges = {(int(a), int(b)) for a, b in z["edges/ids"][:]}
        ilp_prob = {(int(a), int(b)): float(p) for (a, b), p in zip(z["edges/ids"][:], z["edges/props/edge_prob/values"][:])}
        pos = coords[:, 1:].astype(float) * SCALE          # um, cache index == node id
        tcoord = coords[:, 0].astype(int)
        n_frames = int(tcoord.max()) + 1
        # density group as the pipeline computes it (avg nodes/frame on the ILP node set)
        avg = len(ilp_ids) / max(len(set(tcoord[ilp_ids])), 1)
        grp = "low" if avg < 120 else ("middle" if avg < 400 else "high")
        tight, relaxed, bonus = DENS[grp]["tight"], DENS[grp]["relaxed"], DENS[grp]["bonus"]

        cand_prob = {(int(a), int(b)): float(p) for a, b, p, _ in cand}
        cand_by_tgt, cand_by_src = defaultdict(list), defaultdict(list)
        for (a, b), p in cand_prob.items():
            cand_by_tgt[b].append((a, p)); cand_by_src[a].append((b, p))

        final_edges = {(int(s), int(t)) for s, t in final.edges}
        final_out, final_in = defaultdict(set), defaultdict(set)
        for s, t in final_edges:
            final_out[s].add(t); final_in[t].add(s)
        ilp_out, ilp_in = defaultdict(set), defaultdict(set)
        for s, t in ilp_edges:
            ilp_out[s].add(t); ilp_in[t].add(s)

        # --- official matching against the FINAL node set (what the score sees) ---
        tpF, fpF, fnF, matched = edge_counts(g, final)
        g2p = {gid: pid for pid, gid in matched.items()}
        # --- ILP graph scored directly (norelink counterfactual on the ILP node set) ---
        ilp_graph = Graph(nodes=np.column_stack([ilp_ids, coords[ilp_ids].astype(float)]),
                          edges=np.asarray(sorted(ilp_edges), np.int64).reshape(-1, 2))
        tpI, fpI, fnI, matchedI = edge_counts(g, ilp_graph)
        per_vol_scores.append((ds, grp, tpF, fpF, fnF, tpI, fpI, fnI, len(final.nodes), len(ilp_ids)))

        gt_edges = {(int(a), int(b)) for a, b in g.edges}
        n_cache = len(coords)
        ct = Counter()
        for a, b in sorted(gt_edges):
            pa, pb = g2p.get(a), g2p.get(b)
            if pa is None or pb is None:
                ct["det_fn"] += 1
                continue
            if pa >= n_cache or pb >= n_cache:
                ct["synthetic_endpoint"] += 1
                continue
            c = (pa, pb) in cand_prob
            i = (pa, pb) in ilp_edges
            f = (pa, pb) in final_edges
            key = f"cand{int(c)}_ilp{int(i)}_final{int(f)}"
            ct[key] += 1
            dpp = float(np.linalg.norm(pos[pb] - pos[pa]))
            if i and not f:
                # what did the relinker do with pa and pb instead?
                alt_t = final_out.get(pa, set())
                alt_s = final_in.get(pb, set())
                d_alt_t = min((float(np.linalg.norm(pos[q] - pos[pa])) for q in alt_t if q < n_cache), default=float("nan"))
                d_alt_s = min((float(np.linalg.norm(pos[pb] - pos[q])) for q in alt_s if q < n_cache), default=float("nan"))
                destroyed_rows.append(dict(ds=ds, t=int(tcoord[pa]), d=dpp, over_tight=dpp > tight, over_relaxed=dpp > relaxed,
                                           prob=ilp_prob.get((pa, pb)), src_relinked_to=d_alt_t, tgt_relinked_from=d_alt_s,
                                           n_cand_tgt=len(cand_by_tgt[pb]), n_cand_src=len(cand_by_src[pa])))
            if f and not i:
                repaired_rows.append(dict(ds=ds, t=int(tcoord[pa]), d=dpp, cand=c, prob=cand_prob.get((pa, pb)),
                                          ilp_src_went=[(q, round(float(np.linalg.norm(pos[q] - pos[pa])), 2), round(ilp_prob[(pa, q)], 3)) for q in ilp_out.get(pa, ())],
                                          ilp_tgt_from=[(q, round(float(np.linalg.norm(pos[pb] - pos[q])), 2), round(ilp_prob[(q, pb)], 3)) for q in ilp_in.get(pb, ())]))
            if not f and not c:
                # unreachable by bonus: correct edge never a candidate
                comp_t = sorted(cand_by_tgt[pb], key=lambda x: -x[1])
                comp_s = sorted(cand_by_src[pa], key=lambda x: -x[1])
                unreachable_rows.append(dict(ds=ds, t=int(tcoord[pa]), d=dpp,
                                             top_cand_into_tgt=comp_t[0][1] if comp_t else None,
                                             top_cand_from_src=comp_s[0][1] if comp_s else None))
            if not f and c:
                # correct edge was a candidate: what was its prob vs the competitor the pipeline chose?
                taken = final_out.get(pa, set())
                p_taken = max((cand_prob.get((pa, q), 0.0) for q in taken), default=None)
                cand_margin_rows.append(dict(ds=ds, t=int(tcoord[pa]), d=dpp, p_correct=cand_prob[(pa, pb)], p_taken=p_taken,
                                             ilp=i, d_taken=min((float(np.linalg.norm(pos[q] - pos[pa])) for q in taken if q < n_cache), default=None)))
        pooled.update(ct)

        # --- FP origin: final FP edges (annotated), were they ILP edges? ---
        gt_out, gt_in = defaultdict(set), defaultdict(set)
        for a, b in gt_edges:
            gt_out[a].add(b); gt_in[b].add(a)
        for s, t in final_edges:
            gs, gt_ = matched.get(s), matched.get(t)
            out_valid = gs is not None and len(gt_out.get(gs, ())) > 0
            in_valid = gt_ is not None and len(gt_in.get(gt_, ())) > 0
            if not (out_valid or in_valid):
                continue
            is_tp = gs is not None and gt_ is not None and (gs, gt_) in gt_edges
            if s >= n_cache or t >= n_cache:
                fp_origin_pool["synthetic_" + ("tp" if is_tp else "fp")] += 1
                continue
            src = "ilp" if (s, t) in ilp_edges else ("cand" if (s, t) in cand_prob else "geom")
            fp_origin_pool[f"{src}_{'tp' if is_tp else 'fp'}"] += 1

        # --- relinker-created edges (FINAL not ILP, both real nodes) ---
        for s, t in final_edges:
            if s >= n_cache or t >= n_cache:
                continue
            if (s, t) in ilp_edges:
                continue
            dd = float(np.linalg.norm(pos[t] - pos[s]))
            relink_new_pool["n"] += 1
            relink_new_pool["over_tight"] += dd > tight
            relink_new_pool["was_cand"] += (s, t) in cand_prob

        # --- long ILP edges ---
        for (s, t) in ilp_edges:
            dd = float(np.linalg.norm(pos[t] - pos[s]))
            if dd <= tight:
                continue
            ilp_long_pool["n"] += 1
            ilp_long_pool["survived_final"] += (s, t) in final_edges
            gs, gt_ = matched.get(s), matched.get(t)
            if gs is not None and gt_ is not None:
                ilp_long_pool["annot_both"] += 1
                ilp_long_pool["annot_tp"] += (gs, gt_) in gt_edges
                ilp_long_pool["annot_tp_survived"] += ((gs, gt_) in gt_edges) and ((s, t) in final_edges)

        print(f"\n=== {ds}  group={grp} (avg {avg:.0f} nodes/frame; tight {tight} relaxed {relaxed} bonus {bonus}) ===")
        print(f"  FINAL edges TP/FP/FN {tpF}/{fpF}/{fnF} J={tpF/(tpF+fpF+fnF):.4f} | ILP-graph direct TP/FP/FN {tpI}/{fpI}/{fnI} J={tpI/(tpI+fpI+fnI):.4f}  (nodes final {len(final.nodes)} vs ILP {len(ilp_ids)})")
        print("  GT-edge crosstab (both endpoints matched to final nodes):", dict(sorted(ct.items())))

    print("\n================ POOLED ================")
    TPF = sum(r[2] for r in per_vol_scores); FPF = sum(r[3] for r in per_vol_scores); FNF = sum(r[4] for r in per_vol_scores)
    TPI = sum(r[5] for r in per_vol_scores); FPI = sum(r[6] for r in per_vol_scores); FNI = sum(r[7] for r in per_vol_scores)
    print(f"FINAL micro edge_J {TPF/(TPF+FPF+FNF):.4f} (TP {TPF} FP {FPF} FN {FNF})   ILP-graph-direct micro edge_J {TPI/(TPI+FPI+FNI):.4f} (TP {TPI} FP {FPI} FN {FNI})")
    print("crosstab pooled:", dict(sorted(pooled.items())))
    print(f"\nDESTROYED by relinker (ILP correct, FINAL wrong): n={len(destroyed_rows)}")
    if destroyed_rows:
        dd = np.array([r["d"] for r in destroyed_rows])
        print(f"  d_pred: med {np.median(dd):.2f} p90 {np.percentile(dd,90):.2f} | over tight gate {sum(r['over_tight'] for r in destroyed_rows)} | over relaxed {sum(r['over_relaxed'] for r in destroyed_rows)}")
        print(f"  ILP prob of destroyed: med {np.median([r['prob'] for r in destroyed_rows]):.3f}")
        for r in destroyed_rows:
            print("   ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nREPAIRED by relinker (ILP wrong/absent, FINAL correct): n={len(repaired_rows)}  (of which correct edge was a candidate: {sum(r['cand'] for r in repaired_rows)})")
    for r in repaired_rows:
        print("   ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nUNREACHABLE (FINAL wrong, correct edge never a candidate > 0.48): n={len(unreachable_rows)}")
    if unreachable_rows:
        dd = np.array([r["d"] for r in unreachable_rows])
        tt = [r["top_cand_into_tgt"] for r in unreachable_rows if r["top_cand_into_tgt"] is not None]
        print(f"  d_pred med {np.median(dd):.2f} p90 {np.percentile(dd,90):.2f}; competitor top-prob into the true target: n={len(tt)} med {np.median(tt) if tt else float('nan'):.3f}; targets with NO candidate at all: {sum(r['top_cand_into_tgt'] is None for r in unreachable_rows)}")
    print(f"\nCANDIDATE-BUT-LOST (FINAL wrong, correct edge WAS a candidate): n={len(cand_margin_rows)}  (ILP had it: {sum(r['ilp'] for r in cand_margin_rows)})")
    for r in cand_margin_rows:
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nFINAL annotated-edge origin (ilp / cand-not-ilp / geometry-only / synthetic) x (tp/fp):", dict(sorted(fp_origin_pool.items())))
    print("RELINKER-CREATED edges (final, not ILP, real nodes):", dict(relink_new_pool))
    print("LONG ILP edges (> tight gate):", dict(ilp_long_pool))
    print("\nper-volume:", per_vol_scores)


if __name__ == "__main__":
    main()
