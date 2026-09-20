"""Offline simulation of feifanque's harmonic_tracklet_repair + endpoint_tracklet_repair
on the shipped ds_ctl submission. Post-proc only, so this is a faithful port target.

FIDELITY CAVEATS (state with any number):
 - edge_prob unknown offline -> set 0.0, which MAXIMISES gain (gain -= prob) => UPPER BOUND on firing.
 - 'motion_relinked' set 1 for every edge (true for ~99%: 20143/20326 on 0b24845f).
 - max_distance = the volume's densad RELAXED_UM (the kernel passes MOTION_RELINK_RELAXED_UM).
"""
import sys, csv, collections, numpy as np
import feif_tracklet, feif_endpoint

SUB = sys.argv[1]; OUT = sys.argv[2]; MODE = sys.argv[3]  # tracklet|endpoint|both
rows = list(csv.DictReader(open(SUB)))
byds = collections.defaultdict(lambda: {'nodes': {}, 'edges': [], 'raw': []})
for r in rows:
    d = byds[r['dataset']]; d['raw'].append(r)
    if r['row_type'] == 'node':
        d['nodes'][int(r['node_id'])] = dict(t=int(r['t']), z=float(r['z']), y=float(r['y']), x=float(r['x']))
    else:
        d['edges'].append(dict(source_id=int(r['source_id']), target_id=int(r['target_id']),
                               motion_relinked=1, edge_prob=0.0))

def group(nodes):
    nf = len(nodes) / max(1, len(set(n['t'] for n in nodes.values())))
    return (7.25, 11.0) if nf < 120 else ((6.5, 9.0) if nf < 400 else (5.5, 10.0))

tot = collections.Counter(); changed_map = {}
for ds, d in sorted(byds.items()):
    tight, relaxed = group(d['nodes'])
    edges = d['edges']; before = [(e['source_id'], e['target_id']) for e in edges]
    st = {}
    if MODE in ('tracklet', 'both'):
        edges, s = feif_tracklet.repair(d['nodes'], edges, enabled=True, max_distance=relaxed, min_gain=1.0)
        st.update(s)
    if MODE in ('endpoint', 'both'):
        edges, s, _ = feif_endpoint.repair(d['nodes'], edges, enabled=True, max_distance=relaxed,
                                           min_gain=1.0, ambiguity_margin=0.5)
        st.update(s)
    after = [(e['source_id'], e['target_id']) for e in edges]
    nch = sum(1 for a, b in zip(before, after) if a != b)
    print(f"{ds:18s} nf={len(d['nodes'])/max(1,len(set(n['t'] for n in d['nodes'].values()))):6.1f} "
          f"relaxed={relaxed:4.1f} edges={len(edges):6d} CHANGED={nch:5d} "
          + ' '.join(f"{k}={v}" for k, v in sorted(st.items()) if v))
    for k, v in st.items(): tot[k] += v
    tot['changed'] += nch
    changed_map[ds] = dict(zip(range(len(edges)), after))
    d['new'] = after

print('TOTAL', dict(tot))
# write repaired submission preserving row order/ids
w = csv.DictWriter(open(OUT, 'w', newline=''), fieldnames=rows[0].keys()); w.writeheader()
idx = collections.Counter()
for r in rows:
    if r['row_type'] != 'node':
        d = byds[r['dataset']]; i = idx[r['dataset']]; idx[r['dataset']] += 1
        s, t = d['new'][i]; r = dict(r); r['source_id'] = s; r['target_id'] = t
    w.writerow(r)
