#!/usr/bin/env python3
"""Fresh error anatomy on the 947 line (ds_ctl base). Scratch wrapper -- does not touch the repo.

Adds to link_ceiling.py / error_anatomy.py:
  * per-volume node_recall, node_ratio, adj (actual) and adj under a PERFECT LINKER on the
    current node set, aggregated with the OFFICIAL sample-size weights (w_i = TP+FP+FN)
  * FP sub-classification: both-matched wrong pair / annotated->unannotated / unannotated->annotated,
    and whether each FP is FORCED by a detection miss (the GT successor/predecessor was undetected)
  * assoc-FN sub-classification: swap (source linked elsewhere) vs termination, and whether the
    correct target was orphaned or stolen
  * character: GT displacement, wrong-link length, nearest-neighbour + constant-velocity
    counterfactuals, local density, division / track-end proximity, frame position, z-fraction
  * detection-FN character: distance from the unmatched GT node to the nearest predicted node
    (merge < 7um  /  displaced 7-15um  /  missing > 15um), track-end/division proximity
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path("/Users/maruanottoni/home/kaggle/biohub-cell-tracking")
sys.path.insert(0, str(REPO / "src"))
from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
SUB = Path(sys.argv[1] if len(sys.argv) > 1 else "~/biohub_replay947_runs/ds_ctl/submission.csv").expanduser()
DIV_WIN = 3      # frames: "near a division" if a GT division sits within +-DIV_WIN frames on the lineage
END_WIN = 3      # frames: "near a track end" if within END_WIN frames of a GT root/leaf
DENS_R = 10.0    # um: local density radius


def load_submission(path: Path) -> dict[str, Graph]:
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


def q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if a.size else float("nan")


def summ(a):
    a = np.asarray(a, float)
    if not a.size:
        return "n=0"
    return f"n={a.size} mean={a.mean():.2f} med={np.median(a):.2f} p90={q(a,90):.2f}"


def main():
    preds = load_submission(SUB)
    per_vol = []
    # pooled feature stores
    F = defaultdict(list)          # feature -> list per category
    det_fn_nodes_info = []
    fp_rows_all = []
    fn_rows_all = []

    for ds in sorted(preds):
        pred = preds[ds]
        gt = load_geff(GT_ROOT / f"{ds}.geff")
        g = gt.graph
        n_est = gt.estimated_number_of_nodes
        matched = match_nodes(g, pred, max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
        g2p = {gid: pid for pid, gid in matched.items()}

        # positions (um) and times
        ppos = {int(r[0]): r[2:5] * SCALE for r in pred.nodes}
        pt = {int(r[0]): int(r[1]) for r in pred.nodes}
        gpos = {int(r[0]): r[2:5] * SCALE for r in g.nodes}
        gtt = {int(r[0]): int(r[1]) for r in g.nodes}
        n_frames = int(max(pt.values())) + 1

        # per-frame KD trees of predicted nodes
        frame_ids = defaultdict(list)
        for pid, t in pt.items():
            frame_ids[t].append(pid)
        trees = {}
        for t, ids in frame_ids.items():
            arr = np.asarray([ppos[i] for i in ids])
            trees[t] = (cKDTree(arr), np.asarray(ids))

        def density(t, x, r=DENS_R):
            if t not in trees:
                return 0
            return len(trees[t][0].query_ball_point(x, r))

        def nearest_pred(t, x, exclude=None, k=3):
            """nearest predicted node in frame t to point x (um). returns (id, dist)."""
            if t not in trees:
                return None, float("inf")
            tree, ids = trees[t]
            kk = min(k, len(ids))
            d, i = tree.query(x, k=kk)
            d, i = np.atleast_1d(d), np.atleast_1d(i)
            for dd, ii in zip(d, i):
                if ids[ii] != exclude:
                    return int(ids[ii]), float(dd)
            return None, float("inf")

        pred_edges = {(int(s), int(t)) for s, t in pred.edges}
        pred_out, pred_in = defaultdict(set), defaultdict(set)
        for s, t in pred_edges:
            pred_out[s].add(t); pred_in[t].add(s)
        gt_edges = {(int(a), int(b)) for a, b in g.edges}
        gt_out, gt_in = defaultdict(set), defaultdict(set)
        for a, b in gt_edges:
            gt_out[a].add(b); gt_in[b].add(a)
        gt_parent = {b: a for a, b in gt_edges}
        div_nodes = {a for a, ch in gt_out.items() if len(ch) >= 2}

        # distance (frames) to nearest GT division along lineage (up via parents, down via children)
        def frames_to_div(n, win=DIV_WIN):
            best = None
            # up
            cur, k = n, 0
            while cur is not None and k <= win:
                if cur in div_nodes:
                    best = k if best is None else min(best, k)
                    break
                cur = gt_parent.get(cur); k += 1
            # down (BFS)
            frontier, k = [n], 0
            while frontier and k <= win:
                nxt = []
                for c in frontier:
                    if c in div_nodes and k > 0:
                        best = k if best is None else min(best, k)
                    nxt.extend(gt_out.get(c, ()))
                frontier = nxt; k += 1
            return best

        def frames_to_root(n):
            k = 0; cur = n
            while cur in gt_parent:
                cur = gt_parent[cur]; k += 1
                if k > 200: break
            return k

        def frames_to_leaf(n):
            k = 0; cur = n
            while gt_out.get(cur):
                cur = next(iter(gt_out[cur])); k += 1
                if k > 200: break
            return k

        # ---------------- GT edges ----------------
        n_gt = len(gt_edges)
        tp = det_fn = both = 0
        det_src_miss = det_tgt_miss = det_both_miss = 0
        assoc_rows = []
        for a, b in sorted(gt_edges):
            pa, pb = g2p.get(a), g2p.get(b)
            if pa is None or pb is None:
                det_fn += 1
                det_src_miss += (pa is None and pb is not None)
                det_tgt_miss += (pb is None and pa is not None)
                det_both_miss += (pa is None and pb is None)
                continue
            both += 1
            d_gt = float(np.linalg.norm(gpos[b] - gpos[a]))
            good = (pa, pb) in pred_edges
            ta = gtt[a]
            # features common
            nn_id, nn_d = nearest_pred(ta + 1, ppos[pa])
            nn_is_correct = (nn_id == pb)
            # nearest OTHER node to the correct target in t+1 (competition spacing)
            other_id, other_d = nearest_pred(ta + 1, ppos[pa], exclude=pb)
            d_pred_correct = float(np.linalg.norm(ppos[pb] - ppos[pa]))
            # constant-velocity extrapolation from GT-matched predecessor of a (predicted graph)
            prev = next(iter(pred_in[pa]), None)
            cv_pos = None
            if prev is not None and prev in ppos:
                cv_pos = ppos[pa] + (ppos[pa] - ppos[prev])
            dens_t1 = density(ta + 1, ppos[pa])
            dens_t = density(ta, ppos[pa])
            to_div = frames_to_div(a)
            to_root = frames_to_root(a); to_leaf = frames_to_leaf(b)
            loc_a = float(np.linalg.norm(ppos[pa] - gpos[a])); loc_b = float(np.linalg.norm(ppos[pb] - gpos[b]))
            dz = abs(gpos[b][0] - gpos[a][0])
            row = dict(ds=ds, a=a, b=b, pa=pa, pb=pb, t=ta, good=good, d_gt=d_gt, d_pred_correct=d_pred_correct,
                       nn_d=nn_d, nn_is_correct=nn_is_correct, other_d=other_d, dens_t=dens_t, dens_t1=dens_t1,
                       to_div=to_div, to_root=to_root, to_leaf=to_leaf, loc_a=loc_a, loc_b=loc_b,
                       zfrac=dz / d_gt if d_gt > 0 else 0.0, is_div_src=a in div_nodes,
                       frac_t=ta / max(n_frames - 1, 1))
            if good:
                tp += 1
                row["kind"] = "tp"
            else:
                taken = pred_out.get(pa, set())
                stolen = pred_in.get(pb, set())
                if taken and stolen:
                    kind = "swap_both"       # source went elsewhere AND target taken by another
                elif taken:
                    kind = "swap_src"        # source went elsewhere, correct target orphaned
                elif stolen:
                    kind = "term_tgt_stolen" # source terminated, target got a different parent
                else:
                    kind = "gap"             # source terminated, target orphan: pure missing link
                row["kind"] = kind
                if taken:
                    tk = min(taken, key=lambda qid: np.linalg.norm(ppos[qid] - ppos[pa]))
                    d_taken = float(np.linalg.norm(ppos[tk] - ppos[pa]))
                    row["d_taken"] = d_taken
                    row["taken_is_matched"] = tk in matched
                    row["taken_gt"] = matched.get(tk)
                    if cv_pos is not None:
                        row["cv_correct_closer"] = np.linalg.norm(cv_pos - ppos[pb]) < np.linalg.norm(cv_pos - ppos[tk])
                        row["cv_err_correct"] = float(np.linalg.norm(cv_pos - ppos[pb]))
                        row["cv_err_taken"] = float(np.linalg.norm(cv_pos - ppos[tk]))
                    row["taken_dens"] = density(ta + 1, ppos[tk])
                if stolen:
                    st = next(iter(stolen))
                    row["stolen_from_matched"] = st in matched
                    row["stolen_gt"] = matched.get(st)
                    row["d_stolen"] = float(np.linalg.norm(ppos[pb] - ppos[st]))
            assoc_rows.append(row)
        fn_rows_all.extend(r for r in assoc_rows if not r["good"])

        # ---------------- predicted edges: FP classification ----------------
        fp_rows = []
        for s, t in pred_edges:
            gs, gt_ = matched.get(s), matched.get(t)
            out_valid = gs is not None and len(gt_out.get(gs, ())) > 0
            in_valid = gt_ is not None and len(gt_in.get(gt_, ())) > 0
            if not (out_valid or in_valid):
                continue
            if gs is not None and gt_ is not None and (gs, gt_) in gt_edges:
                continue
            # FP
            if gs is not None and gt_ is not None:
                cls = "both_matched_wrong_pair"
                forced = False
            elif out_valid:
                cls = "annot_src->unannot_tgt"
                succ = gt_out[gs]
                forced = all(g2p.get(c) is None for c in succ)   # true successor(s) undetected
            else:
                cls = "unannot_src->annot_tgt"
                pred_ = gt_in[gt_]
                forced = all(g2p.get(c) is None for c in pred_)
            # is this FP paired with an assoc FN (same source or same target)?
            paired = any((r["pa"] == s or r["pb"] == t) for r in assoc_rows if not r["good"])
            d = float(np.linalg.norm(ppos[t] - ppos[s]))
            fp_rows.append(dict(ds=ds, s=s, t=t, cls=cls, forced=forced, paired=paired, d=d, t_frame=pt[s],
                                src_is_pred_div=len(pred_out[s]) >= 2))
        fp_rows_all.extend(fp_rows)
        fp = len(fp_rows)
        fp_forced = sum(r["forced"] for r in fp_rows)

        # ---------------- detection FN nodes ----------------
        gt_nodes_with_edge = set(gt_out) | set(gt_in)
        unmatched_gt = [n for n in gt_nodes_with_edge if n not in g2p]
        for n in unmatched_gt:
            t = gtt[n]
            nid, d = nearest_pred(t, gpos[n], k=1)
            # was the nearest pred node matched to a different GT node (merge / competition)?
            comp = nid is not None and nid in matched
            det_fn_nodes_info.append(dict(ds=ds, n=n, t=t, d_nearest=d, nearest_taken=comp,
                                          to_div=frames_to_div(n), to_root=frames_to_root(n),
                                          to_leaf=frames_to_leaf(n), is_div=n in div_nodes,
                                          is_daughter=gt_parent.get(n) in div_nodes,
                                          frac_t=t / max(n_frames - 1, 1),
                                          dens=density(t, gpos[n])))

        # ---------------- per-volume metrics ----------------
        n_pred = len(pred.nodes)
        ratio = (n_pred - n_est) / n_est
        n_gt_nodes = len(g.nodes)
        recall = len(set(matched.values())) / n_gt_nodes
        fn = n_gt - tp
        edge_j = tp / (tp + fp + fn)
        w = tp + fp + fn
        adj = max(0.0, edge_j * (1 - 0.1 * ratio))
        oracle = both / n_gt
        adj_oracle = max(0.0, oracle * (1 - 0.1 * ratio))
        # realistic oracle: perfect linker that still emits the FORCED FPs (cannot know successor undetected)
        oracle_forced = both / (n_gt + fp_forced)
        adj_oracle_forced = max(0.0, oracle_forced * (1 - 0.1 * ratio))
        assoc_fn = both - tp
        kinds = Counter(r["kind"] for r in assoc_rows if not r["good"])
        fpc = Counter(r["cls"] for r in fp_rows)
        per_vol.append(dict(ds=ds, n_gt=n_gt, n_gt_nodes=n_gt_nodes, n_est=n_est, n_pred=n_pred, ratio=ratio,
                            recall=recall, tp=tp, fp=fp, fn=fn, w=w, edge_j=edge_j, adj=adj, oracle=oracle,
                            adj_oracle=adj_oracle, oracle_forced=oracle_forced, adj_oracle_forced=adj_oracle_forced,
                            det_fn=det_fn, det_src_miss=det_src_miss, det_tgt_miss=det_tgt_miss,
                            det_both_miss=det_both_miss, assoc_fn=assoc_fn, kinds=kinds, fpc=fpc,
                            fp_forced=fp_forced, fp_paired=sum(r["paired"] for r in fp_rows),
                            n_frames=n_frames, nodes_per_frame=n_pred / n_frames,
                            n_div=len(div_nodes), n_unmatched_gt=len(unmatched_gt),
                            n_tracks=gt.n_tracks()))

    # ================= REPORT =================
    print(f"SUBMISSION {SUB}")
    print("\n== PER-VOLUME (official semantics; adj = edge_J*(1-0.1*ratio); w = TP+FP+FN) ==")
    hdr = f"{'volume':>14} {'gtE':>5} {'gtN':>5} {'trk':>4} {'div':>3} {'n_pred':>6} {'n_est':>7} {'ratio':>7} {'recall':>7} {'nodes/f':>7} | {'tp':>4} {'fp':>3} {'fn':>3} {'edgeJ':>6} {'adj':>6} | {'ORACLE':>6} {'adjOR':>6} {'ORfrc':>6} | {'detFN':>5} {'assFN':>5} {'FPfrc':>5} {'FPpair':>6}"
    print(hdr)
    for r in per_vol:
        print(f"{r['ds']:>14} {r['n_gt']:>5} {r['n_gt_nodes']:>5} {r['n_tracks']:>4} {r['n_div']:>3} {r['n_pred']:>6} {r['n_est']:>7.0f} {r['ratio']:>+7.4f} {r['recall']:>7.4f} {r['nodes_per_frame']:>7.0f} | "
              f"{r['tp']:>4} {r['fp']:>3} {r['fn']:>3} {r['edge_j']:>6.4f} {r['adj']:>6.4f} | {r['oracle']:>6.4f} {r['adj_oracle']:>6.4f} {r['oracle_forced']:>6.4f} | "
              f"{r['det_fn']:>5} {r['assoc_fn']:>5} {r['fp_forced']:>5} {r['fp_paired']:>6}")
        print(f"{'':>14}   det miss src/tgt/both={r['det_src_miss']}/{r['det_tgt_miss']}/{r['det_both_miss']}  assocFN kinds={dict(r['kinds'])}  FP classes={dict(r['fpc'])}")

    W = sum(r["w"] for r in per_vol)
    adj_w = sum(r["w"] * r["adj"] for r in per_vol) / W
    Wg = sum(r["n_gt"] for r in per_vol)
    adj_or_w = sum(r["n_gt"] * r["adj_oracle"] for r in per_vol) / Wg          # oracle weights: FP=0 -> w=n_gt
    adj_or_w_actualw = sum(r["w"] * r["adj_oracle"] for r in per_vol) / W
    Wf = sum(r["n_gt"] + r["fp_forced"] for r in per_vol)
    adj_orf_w = sum((r["n_gt"] + r["fp_forced"]) * r["adj_oracle_forced"] for r in per_vol) / Wf
    TP = sum(r["tp"] for r in per_vol); FP = sum(r["fp"] for r in per_vol); FN = sum(r["fn"] for r in per_vol)
    DET = sum(r["det_fn"] for r in per_vol); ASS = sum(r["assoc_fn"] for r in per_vol)
    FPF = sum(r["fp_forced"] for r in per_vol)
    BOTH = TP + ASS
    print("\n== TOTALS ==")
    print(f"GT edges {Wg}  TP {TP}  FP {FP}  FN {FN}  micro edge_J {TP/(TP+FP+FN):.4f}")
    print(f"weighted adj (official) {adj_w:.4f}   (+0.1*div_J 0.1818 = {adj_w+0.01818:.4f}; official printed 0.9525/0.9344)")
    print(f"micro ORACLE edge_J (perfect linker, current nodes) {BOTH/Wg:.4f}   weighted adj_oracle {adj_or_w:.4f} (w=n_gt) / {adj_or_w_actualw:.4f} (actual w)")
    print(f"  -> perfect-linker SCORE (adj only, div fixed 0.1818) = {adj_or_w + 0.01818:.4f}   association headroom on adj = {adj_or_w - adj_w:+.4f}")
    print(f"realistic oracle keeping detection-FORCED FPs: micro edge_J {BOTH/(Wg+FPF):.4f}  weighted adj {adj_orf_w:.4f}  (headroom {adj_orf_w-adj_w:+.4f})")
    loss = FP + FN
    print(f"\nLOSS SPLIT (denominator terms, total {loss}):")
    print(f"  det_fn (GT edge with undetected endpoint)        {DET:>4}  {DET/loss:5.1%}")
    print(f"  FP forced by detection (true partner undetected) {FPF:>4}  {FPF/loss:5.1%}")
    print(f"  assoc_fn (both endpoints detected, not linked)   {ASS:>4}  {ASS/loss:5.1%}")
    print(f"  FP avoidable (association)                       {FP-FPF:>4}  {(FP-FPF)/loss:5.1%}")
    print(f"  => detection-attributable {DET+FPF} ({(DET+FPF)/loss:.1%})   association-attributable {ASS+FP-FPF} ({(ASS+FP-FPF)/loss:.1%})")
    print(f"  (link_ceiling's convention: det {DET} vs assoc {ASS+FP} -> assoc share {(ASS+FP)/loss:.1%})")

    # ---------------- assoc FN character ----------------
    fn_rows = fn_rows_all
    tp_rows = [r for v in [None] for r in []]  # placeholder
    print("\n== ASSOC-FN CHARACTER (pooled over volumes) ==")
    kinds = Counter(r["kind"] for r in fn_rows)
    print(f"kinds: {dict(kinds)}   (swap_both = src linked elsewhere & tgt taken by another; swap_src = src linked elsewhere, tgt orphan; term_tgt_stolen = src track ended, tgt taken by another; gap = both dangling)")
    swaps = [r for r in fn_rows if r["kind"].startswith("swap")]
    print(f"GT displacement (um): assoc-FN {summ([r['d_gt'] for r in fn_rows])}")
    print(f"   swaps only        : {summ([r['d_gt'] for r in swaps])}")
    print(f"   non-swap (term/gap): {summ([r['d_gt'] for r in fn_rows if not r['kind'].startswith('swap')])}")
    if swaps:
        sh = sum(r["d_taken"] < r["d_gt"] for r in swaps)
        sh2 = sum(r["d_taken"] < r["d_pred_correct"] for r in swaps)
        print(f"wrong link shorter than correct: vs GT-disp {sh}/{len(swaps)}; vs PRED-distance to correct target {sh2}/{len(swaps)}")
        print(f"   d_taken {summ([r['d_taken'] for r in swaps])}   d_pred_correct {summ([r['d_pred_correct'] for r in swaps])}   ratio d_correct/d_taken med={np.median([r['d_pred_correct']/max(r['d_taken'],1e-6) for r in swaps]):.2f}")
        tm = sum(r["taken_is_matched"] for r in swaps)
        print(f"   taken target is an ANNOTATED (matched) node: {tm}/{len(swaps)}  (rest: unannotated cell or spurious detection)")
        cv = [r for r in swaps if "cv_correct_closer" in r]
        print(f"   constant-velocity extrapolation would pick CORRECT target: {sum(r['cv_correct_closer'] for r in cv)}/{len(cv)} (with predecessor available)")
        if cv:
            print(f"     cv err to correct {summ([r['cv_err_correct'] for r in cv])}  cv err to taken {summ([r['cv_err_taken'] for r in cv])}")
    nn_all = [r for r in fn_rows]
    print(f"nearest-neighbour (pred, t+1) IS the correct target: assoc-FN {sum(r['nn_is_correct'] for r in nn_all)}/{len(nn_all)}")
    print(f"'confusion ratio' d_pred_correct / d_nearest_other: assoc-FN med={np.median([r['d_pred_correct']/max(r['other_d'],1e-6) for r in fn_rows]):.2f}  (>1 means another node sits closer than the true successor)")
    print(f"local density (pred nodes within {DENS_R}um, t+1): assoc-FN {summ([r['dens_t1'] for r in fn_rows])}")
    print(f"localisation residual (um) src/tgt: assoc-FN {summ([r['loc_a'] for r in fn_rows])} / {summ([r['loc_b'] for r in fn_rows])}")
    print(f"z-fraction of GT displacement: assoc-FN med={np.median([r['zfrac'] for r in fn_rows]):.2f}")
    nd = sum(r["to_div"] is not None for r in fn_rows); nds = sum(r["is_div_src"] for r in fn_rows)
    print(f"near a GT division (within +-{DIV_WIN} frames on lineage): {nd}/{len(fn_rows)}; source IS a division node: {nds}")
    ne = sum((r["to_root"] <= END_WIN or r["to_leaf"] <= END_WIN) for r in fn_rows)
    print(f"near a GT track end (<= {END_WIN} frames from root or leaf): {ne}/{len(fn_rows)}  (to_root {summ([r['to_root'] for r in fn_rows])}; to_leaf {summ([r['to_leaf'] for r in fn_rows])})")
    print(f"frame position (fraction of movie): assoc-FN {summ([r['frac_t'] for r in fn_rows])}")
    print(f"per-frame: assoc-FN frames = {sorted(Counter((r['ds'][-4:], r['t']) for r in fn_rows).values(), reverse=True)[:10]} (max errors in any single volume-frame)")
    # per-kind stolen info
    st = [r for r in fn_rows if "stolen_from_matched" in r]
    if st:
        print(f"target stolen by an ANNOTATED source: {sum(r['stolen_from_matched'] for r in st)}/{len(st)}")

    print("\n== ASSOC-FN LIST ==")
    print(f"{'vol':>14} {'t':>3} {'kind':>15} {'d_gt':>5} {'d_tk':>5} {'d_pc':>5} {'nnOK':>4} {'oth':>5} {'dens':>4} {'div':>4} {'end':>4} {'locA':>4} {'locB':>4} {'tkMat':>5} {'cvOK':>4}")
    for r in sorted(fn_rows, key=lambda r: (r["ds"], r["t"])):
        end = min(r["to_root"], r["to_leaf"])
        print(f"{r['ds']:>14} {r['t']:>3} {r['kind']:>15} {r['d_gt']:>5.1f} {r.get('d_taken', float('nan')):>5.1f} {r['d_pred_correct']:>5.1f} {int(r['nn_is_correct']):>4} {r['other_d']:>5.1f} {r['dens_t1']:>4} "
              f"{'-' if r['to_div'] is None else r['to_div']:>4} {end:>4} {r['loc_a']:>4.1f} {r['loc_b']:>4.1f} {str(r.get('taken_is_matched','-')):>5} {str(r.get('cv_correct_closer','-')):>4}")

    # ---------------- FP character ----------------
    print("\n== FP CHARACTER ==")
    fpc = Counter(r["cls"] for r in fp_rows_all)
    print(f"classes: {dict(fpc)}")
    print(f"forced by detection (GT partner undetected): {sum(r['forced'] for r in fp_rows_all)}/{len(fp_rows_all)}")
    print(f"paired with an assoc-FN (same src or tgt): {sum(r['paired'] for r in fp_rows_all)}/{len(fp_rows_all)}")
    print(f"source is a predicted division (out-deg 2): {sum(r['src_is_pred_div'] for r in fp_rows_all)}/{len(fp_rows_all)}")
    print(f"FP edge length (um): {summ([r['d'] for r in fp_rows_all])}")
    for cls in fpc:
        rows = [r for r in fp_rows_all if r["cls"] == cls]
        print(f"  {cls:>28}: n={len(rows)} forced={sum(r['forced'] for r in rows)} paired={sum(r['paired'] for r in rows)} pred_div_src={sum(r['src_is_pred_div'] for r in rows)} len {summ([r['d'] for r in rows])}")

    # ---------------- detection FN character ----------------
    print("\n== DETECTION-FN NODE CHARACTER (GT nodes with an edge, unmatched) ==")
    D = det_fn_nodes_info
    print(f"unmatched GT nodes: {len(D)}  per volume {dict(Counter(r['ds'] for r in D))}")
    dn = np.array([r["d_nearest"] for r in D])
    print(f"distance to nearest predicted node (um): {summ(dn)}")
    print(f"  <7 (pred exists but Hungarian gave it to another GT node = merge/2-in-1): {(dn<7).sum()}   7-15 (displaced): {((dn>=7)&(dn<15)).sum()}   >15 (missing): {(dn>=15).sum()}")
    print(f"  nearest pred node already matched to another GT node: {sum(r['nearest_taken'] for r in D)}")
    print(f"is GT division node: {sum(r['is_div'] for r in D)}; is daughter (parent divides): {sum(r['is_daughter'] for r in D)}; near division +-{DIV_WIN}: {sum(r['to_div'] is not None for r in D)}")
    print(f"near track end (<= {END_WIN} from root/leaf): {sum((r['to_root']<=END_WIN or r['to_leaf']<=END_WIN) for r in D)}   frame frac {summ([r['frac_t'] for r in D])}")
    print(f"local pred density within {DENS_R}um: {summ([r['dens'] for r in D])}")
    # consecutive runs per track
    by_vol = defaultdict(list)
    for r in D:
        by_vol[r["ds"]].append(r)
    print("per-volume unmatched GT frames:")
    for ds, rows in by_vol.items():
        ts = sorted(r["t"] for r in rows)
        print(f"  {ds}: t={ts}")


if __name__ == "__main__":
    main()
