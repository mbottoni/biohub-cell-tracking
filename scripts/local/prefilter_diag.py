#!/usr/bin/env python3
"""Where does division_rerank_e2e.py's PREFILTER lose the detectable divisions?

The e2e run reports 0/2 truth kept after 6 volumes. If prefilter recall is ~0 the
downstream AUC is meaningless, so find the BINDING constraint before waiting 2.5h.

Pure geometry over submission.csv + GT (no zarr), so it runs in seconds. Mirrors
the e2e prefilter exactly: RADIUS 11.0, SISTER_MAX 18.0, MAX_CAND 3, MATCH 10.0.

Reports a funnel over the row-71 population (GT divisions whose parent AND both
daughters are already in our node set) -- each stage is a way the prefilter can
drop an event that IS reachable.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/Users/maruanottoni/home/kaggle/biohub-cell-tracking/src")
from ctrack.data.geff import load_geff  # noqa: E402

S = ("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
     "746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad")
SUB = f"{S}/runs/reverify_stack/submission.csv"
GT = "/Volumes/extSSD/kaggle/biohub/gt/train"
RADIUS, SISTER_MAX, MAX_CAND, MATCH = 11.0, 18.0, 3, 10.0


def main():
    nodes, edges = defaultdict(dict), defaultdict(list)
    with open(SUB) as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]
            if r["row_type"] == "node":
                nodes[ds][int(r["node_id"])] = (
                    int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"]))
            else:
                edges[ds].append((int(r["source_id"]), int(r["target_id"])))

    stage = defaultdict(int)
    for v in sorted(nodes):
        nd = nodes[v]
        by_t = defaultdict(list)
        for nid, (t, *_) in nd.items():
            by_t[t].append(nid)
        trees = {t: (cKDTree(np.array([nd[n][1:] for n in ns])), ns)
                 for t, ns in by_t.items() if ns}
        out_edges = defaultdict(list)
        for s_, t_ in edges[v]:
            out_edges[s_].append(t_)

        try:
            g = load_geff(f"{GT}/{v}.geff")
        except Exception as e:
            print(f"  skip {v}: {e}")
            continue
        G = g.graph
        kids = defaultdict(list)
        for s_, t_ in G.edges:
            kids[s_].append(t_)
        idx = {nid: i for i, nid in enumerate(G.ids)}

        def match(gid):
            i = idx[gid]
            tr = trees.get(int(G.t[i]))
            if tr is None:
                return None
            d, j = tr[0].query(np.array(G.zyx[i]))
            return tr[1][j] if d <= MATCH else None

        for pid, ch in kids.items():
            if len(ch) < 2:
                continue
            stage["gt_divisions"] += 1
            mp, m1, m2 = match(pid), match(ch[0]), match(ch[1])
            if not (mp is not None and m1 is not None and m2 is not None):
                continue
            if len({mp, m1, m2}) < 3:
                stage["collapsed_to_same_node"] += 1
                continue
            stage["all_three_detected"] += 1          # the row-71 population

            # (1) does our graph already link the parent to EITHER daughter?
            linked = [c for c in out_edges.get(mp, []) if c in (m1, m2)]
            if not linked:
                stage["FAIL_no_parent_edge_to_either_daughter"] += 1
                continue
            stage["has_parent_edge"] += 1

            c1 = linked[0]
            c2 = m2 if c1 == m1 else m1
            p_pos = np.array(nd[mp][1:])
            d_par = np.linalg.norm(p_pos - np.array(nd[c2][1:]))
            d_sis = np.linalg.norm(np.array(nd[c1][1:]) - np.array(nd[c2][1:]))

            # (2) geometric gates
            if d_par > RADIUS:
                stage["FAIL_radius"] += 1
                print(f"    radius: {v} parent->c2 {d_par:.1f} > {RADIUS}")
                continue
            if d_sis > SISTER_MAX:
                stage["FAIL_sister"] += 1
                print(f"    sister: {v} sis {d_sis:.1f} > {SISTER_MAX}")
                continue
            stage["passes_gates"] += 1

            # (3) MAX_CAND truncation -- is the true c2 among the 3 nearest?
            tr = trees.get(nd[mp][0] + 1)
            cands = []
            for j in tr[0].query_ball_point(p_pos, RADIUS):
                cc = tr[1][j]
                if cc == c1:
                    continue
                if np.linalg.norm(np.array(nd[c1][1:]) - np.array(nd[cc][1:])) > SISTER_MAX:
                    continue
                cands.append((np.linalg.norm(p_pos - np.array(nd[cc][1:])), cc))
            keep = [c for _, c in sorted(cands)[:MAX_CAND]]
            rank = [c for _, c in sorted(cands)].index(c2) + 1 if c2 in [c for _, c in cands] else -1
            if c2 in keep:
                stage["KEPT"] += 1
            else:
                stage["FAIL_max_cand_truncation"] += 1
                print(f"    truncated: {v} true c2 ranked {rank} of {len(cands)} (keep {MAX_CAND})")

    print("\n=== PREFILTER FUNNEL (all 24 volumes) ===")
    for k in ("gt_divisions", "collapsed_to_same_node", "all_three_detected",
              "FAIL_no_parent_edge_to_either_daughter", "has_parent_edge",
              "FAIL_radius", "FAIL_sister", "passes_gates",
              "FAIL_max_cand_truncation", "KEPT"):
        print(f"  {k:42s} {stage[k]}")
    reach = stage["all_three_detected"]
    print(f"\nPREFILTER RECALL on the reachable population: {stage['KEPT']}/{reach}")


if __name__ == "__main__":
    main()
