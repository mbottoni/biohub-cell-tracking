#!/usr/bin/env python3
"""Division anatomy on a 947-line submission (holdout6).

For every GT division: trace parent/daughters/grandchildren through the pred graph and
the safe-div gates (parent<=9, sister<=14, mutual-NN, divergence>=2.25 w/ t+2 successors,
symmetry<=0.6, orphan status). For every pred fork: classify vs GT (matched? GT outdeg?
children matched? chained?).

    .venv/bin/python divanat947.py ~/biohub_replay947_runs/ds_ctl/submission.csv
"""
from __future__ import annotations
import sys, csv
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

REPO = Path('/Users/maruanottoni/home/kaggle/biohub-cell-tracking')
sys.path.insert(0, str(REPO / 'src'))
from ctrack.data.geff import load_geff  # noqa
from ctrack.eval.biohub import SCALE, Graph, match_nodes  # noqa

GT_ROOT = Path('/Volumes/extSSD/kaggle/biohub/gt/train')
P_MAX, S_MAX, SYM, DIV, EXC = 9.0, 14.0, 0.6, 2.25, 10.0


def load_sub(path):
    nodes = defaultdict(list); edges = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            ds = r['dataset']
            if r['row_type'] == 'node':
                nodes[ds].append((int(r['node_id']), int(r['t']), float(r['z']), float(r['y']), float(r['x'])))
            else:
                edges[ds].append((int(r['source_id']), int(r['target_id'])))
    return nodes, edges


def d_um(a, b):
    return float(np.linalg.norm((a - b) * SCALE))


def main():
    sub = sys.argv[1]
    nodes, edges = load_sub(sub)
    tot = Counter()
    for ds in sorted(nodes):
        gt = load_geff(GT_ROOT / f'{ds}.geff')
        g = gt.graph
        gpos = {int(r[0]): np.asarray(r[2:5], float) for r in g.nodes}
        gt_t = {int(r[0]): int(r[1]) for r in g.nodes}
        gch = defaultdict(list); gpa = defaultdict(list)
        for a, b in g.edges:
            gch[int(a)].append(int(b)); gpa[int(b)].append(int(a))
        pn = np.asarray(nodes[ds], float)
        pe = np.asarray(edges[ds], np.int64) if edges[ds] else np.zeros((0, 2), np.int64)
        pred = Graph(nodes=pn, edges=pe)
        matched = match_nodes(g, pred, max_distance=7.0, scale=SCALE)  # pred->gt
        p2g = {int(p): int(gg) for p, gg in matched.items()}
        g2p = {gg: p for p, gg in p2g.items()}
        ppos = {int(r[0]): np.asarray(r[2:5], float) for r in pn}
        pt = {int(r[0]): int(r[1]) for r in pn}
        pch = defaultdict(list); ppa = defaultdict(list)
        for a, b in pe:
            pch[int(a)].append(int(b)); ppa[int(b)].append(int(a))
        by_t = defaultdict(list)
        for n, t in pt.items():
            by_t[t].append(n)
        # track length helper
        def tracklen(n):
            L = 1; x = n
            while len(pch[x]) == 1 and len(ppa[pch[x][0]]) == 1:
                x = pch[x][0]; L += 1
            x = n
            while len(ppa[x]) == 1 and len(pch[ppa[x][0]]) == 1:
                x = ppa[x][0]; L += 1
            return L

        gdivs = [n for n, ks in gch.items() if len(ks) >= 2]
        print(f'\n===== {ds}: {len(gdivs)} GT divisions, pred forks {sum(1 for n in pch if len(pch[n])>=2)} =====')
        for P in gdivs:
            tP = gt_t[P]; D = gch[P]
            Pp = g2p.get(P)
            print(f'  GT div parent {P} t={tP} daughters {D} (t+1 GT pos dist P->D: ' + ', '.join(f'{d_um(gpos[P], gpos[k]):.1f}' for k in D) + f'; sister {d_um(gpos[D[0]], gpos[D[1]]):.1f} um)')
            gcs = {k: gch[k] for k in D}
            gc_all = [c for k in D for c in gch[k]]
            if len(gc_all) == 2:
                print(f'    GT grandchild dist {d_um(gpos[gc_all[0]], gpos[gc_all[1]]):.1f} -> GT divergence {d_um(gpos[gc_all[0]], gpos[gc_all[1]]) - d_um(gpos[D[0]], gpos[D[1]]):+.1f}; GT daughters have succ: {[len(gch[k]) for k in D]}')
            else:
                print(f'    GT daughters have succ: {[len(gch[k]) for k in D]} (grandchildren {gc_all})')
            gpar = gpa[P]
            print(f'    GT grandparent {gpar}; frames to movie end: {max(gt_t.values()) - tP}')
            if Pp is None:
                print('    PARENT UNMATCHED -> FN-parent'); tot['fn_parent'] += 1; continue
            print(f'    pred parent {Pp} (resid {d_um(gpos[P], ppos[Pp]):.1f} um) out={pch[Pp]} in={ppa[Pp]}')
            for c in pch[Pp]:
                print(f'      child {c} t={pt[c]} matched GT {p2g.get(c)} dist {d_um(ppos[Pp], ppos[c]):.1f} out={pch[c]}')
            for k in D:
                Dp = g2p.get(k)
                if Dp is None:
                    # nearest pred at t+1
                    cand = [(d_um(gpos[k], ppos[n]), n) for n in by_t[tP + 1]]
                    cand.sort()
                    print(f'    daughter GT {k} UNMATCHED; nearest pred at t+1: {cand[:2]}  (matched to GT {[p2g.get(n) for _, n in cand[:2]]})')
                    continue
                inn = ppa[Dp]
                s = f'    daughter GT {k} -> pred {Dp} resid {d_um(gpos[k], ppos[Dp]):.1f}; in={inn}; out={pch[Dp]}; dist P_p->D_p {d_um(ppos[Pp], ppos[Dp]):.1f}; tracklen {tracklen(Dp)}'
                print(s)
                for S in inn:
                    if S == Pp:
                        continue
                    sg = p2g.get(S)
                    # what else could S link to? S's GT successor if annotated
                    alt = ''
                    if sg is not None:
                        gs = gch.get(sg, [])
                        alt = f' S matched GT {sg} (GT succ {gs} -> pred {[g2p.get(x) for x in gs]})'
                    print(f'      STEALER {S} t={pt[S]} dist S->D {d_um(ppos[S], ppos[Dp]):.1f} um; S in={ppa[S]} out={pch[S]}; S tracklen {tracklen(S)}; dist S->P_p {d_um(ppos[S], ppos[Pp]):.1f}{alt}')
                    # nearest orphan-free alternatives for S at t+1
                    others = sorted((d_um(ppos[S], ppos[n]), n, len(ppa[n])) for n in by_t[pt[S] + 1])[:3]
                    print(f'      S nearest at t+1: {[(round(d,1), n, indeg) for d, n, indeg in others]}')
            # gate check for the fork at Pp with existing child + other daughter
            D1p, D2p = g2p.get(D[0]), g2p.get(D[1])
            if D1p is not None and D2p is not None:
                d1, d2 = d_um(ppos[Pp], ppos[D1p]), d_um(ppos[Pp], ppos[D2p])
                sis = d_um(ppos[D1p], ppos[D2p])
                sym = abs(d1 - d2) / max((d1 + d2) / 2, 1e-6)
                gc1, gc2 = pch[D1p], pch[D2p]
                divg = None
                if len(gc1) == 1 and len(gc2) == 1 and pt[gc1[0]] == tP + 2 and pt[gc2[0]] == tP + 2:
                    divg = d_um(ppos[gc1[0]], ppos[gc2[0]]) - sis
                print(f'    GATES on pred: parent d1={d1:.1f} d2={d2:.1f} (<=9 {d1<=P_MAX and d2<=P_MAX}); sister {sis:.1f} (<=14 {sis<=S_MAX}); symmetry {sym:.2f} (<=0.6 {sym<=SYM}); divergence {divg} (>=2.25 {None if divg is None else divg>=DIV}); D1 in={ppa[D1p]} D2 in={ppa[D2p]}')
            fork = len(pch[Pp]) >= 2 or any(len(pch[c]) >= 2 for c in pch[Pp]) or any(len(pch[q]) >= 2 for q in ppa[Pp])
            print(f'    fork within +-1 frame on pred lineage: {fork}')
        # ---- pred forks ----
        forks = [n for n in pch if len(pch[n]) >= 2]
        chained = 0
        for f in forks:
            gm = p2g.get(f)
            kids = pch[f]
            kid_m = [p2g.get(k) for k in kids]
            gc = [p2g.get(x) for k in kids for x in pch[k]]
            # chained: child or grandchild or parent is a fork
            ch = any(len(pch[k]) >= 2 for k in kids) or any(len(pch[x]) >= 2 for k in kids for x in pch[k]) or any(len(pch[q]) >= 2 for q in ppa[f])
            chained += ch
            status = 'unmatched-free'
            if gm is not None:
                if len(gch[gm]) >= 2:
                    status = 'AT GT DIVIDER'
                elif len(gch[gm]) == 1:
                    status = 'FP (annotated non-dividing, evaluable)'
                else:
                    status = 'free (GT track end)'
            # daughter branch lengths
            blen = [tracklen(k) for k in kids]
            if status != 'unmatched-free' or any(m is not None for m in kid_m + gc):
                print(f'  FORK {f} t={pt[f]} -> {status}; GT {gm}; kids matched {kid_m}; grandkids matched {gc}; branch lens {blen}; chained {ch}; kid dists {[round(d_um(ppos[f], ppos[k]),1) for k in kids]}')
        blens = [min(tracklen(k) for k in pch[f]) for f in forks]
        print(f'  forks {len(forks)}; chained-within-2 {chained}; min-branch-len hist {Counter(min(b,10) for b in blens)}')
        tot['forks'] += len(forks); tot['chained'] += chained
    print('\nTOTALS', dict(tot))


if __name__ == '__main__':
    main()
