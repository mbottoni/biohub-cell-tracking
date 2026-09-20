"""One-pass, two-sided tracklet swaps on the existing harmonic motion graph."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree


def repair(nodes, edges, *, enabled=True, max_distance=10.0, min_gain=1.0):
    stats = dict(tracklet_candidates=0, tracklet_supported=0, tracklet_swaps=0,
                 tracklet_changed_edges=0, tracklet_conflicts=0)
    if not enabled or not edges:
        return edges, stats
    if max_distance <= 0 or min_gain <= 0:
        raise ValueError("Repair thresholds must be positive")
    pos = {int(k): np.array([v['z'], v['y'], v['x']], dtype=float)
           * np.array([1.625, .40625, .40625]) for k, v in nodes.items()}
    if any(not np.isfinite(p).all() for p in pos.values()):
        raise ValueError("Non-finite node coordinates")
    incoming, outgoing = {}, {}
    by_time = {}
    for i, e in enumerate(edges):
        a, b = int(e['source_id']), int(e['target_id'])
        if a not in nodes or b not in nodes or int(nodes[b]['t']) != int(nodes[a]['t']) + 1:
            raise ValueError("Invalid motion graph endpoint or frame")
        incoming.setdefault(b, []).append(i)
        outgoing.setdefault(a, []).append(i)
        by_time.setdefault(int(nodes[a]['t']), []).append(i)

    def context(node, backwards):
        steps, support = [], set()
        for _ in range(2):
            choices = (incoming if backwards else outgoing).get(node, [])
            if len(choices) != 1:
                break
            idx = choices[0]
            e = edges[idx]
            other = int(e['source_id'] if backwards else e['target_id'])
            if len((outgoing if backwards else incoming).get(other, [])) != 1:
                break
            # Only observed ordinary links are suitable velocity evidence.
            if not e.get('motion_relinked') or e.get('gap_closed') or e.get('gap2_recovered'):
                break
            step = pos[node] - pos[other] if backwards else pos[other] - pos[node]
            if np.linalg.norm(step) > max_distance:
                break
            steps.append(step)
            support.add(idx)
            node = other
        if not steps:
            return None
        return np.mean(steps, axis=0), support

    def evidence(e):
        try:
            p = float(e.get('edge_prob') or 0)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(p):
            return 0.0
        if p < 0 or p > 1:
            p = 1 / (1 + math.exp(-max(-20., min(20., p))))
        return p

    proposals = []
    for t, indices in sorted(by_time.items()):
        usable = []
        contexts = {}
        for idx in indices:
            e = edges[idx]
            a, b = int(e['source_id']), int(e['target_id'])
            if (not e.get('motion_relinked') or len(outgoing[a]) != 1
                    or len(incoming[b]) != 1 or len(incoming.get(a, [])) > 1
                    or len(outgoing.get(b, [])) > 1):
                continue
            past, future = context(a, True), context(b, False)
            if past is not None and future is not None:
                usable.append(idx)
                contexts[idx] = past, future
        if len(usable) < 2:
            continue
        tree = cKDTree([pos[int(edges[i]['source_id'])] for i in usable])
        for u, v in sorted(tree.query_pairs(2 * max_distance)):
            i, j = usable[u], usable[v]
            ei, ej = edges[i], edges[j]
            a, b = int(ei['source_id']), int(ei['target_id'])
            c, d = int(ej['source_id']), int(ej['target_id'])
            if max(np.linalg.norm(pos[d]-pos[a]), np.linalg.norm(pos[b]-pos[c])) > max_distance:
                continue
            stats['tracklet_candidates'] += 1
            (va, sa), (vb, sb) = contexts[i]
            (vc, sc), (vd, sd) = contexts[j]
            old_f = np.array([np.linalg.norm(pos[b]-pos[a]-va), np.linalg.norm(pos[d]-pos[c]-vc)])
            new_f = np.array([np.linalg.norm(pos[d]-pos[a]-va), np.linalg.norm(pos[b]-pos[c]-vc)])
            old_b = np.array([np.linalg.norm(pos[b]-pos[a]-vb), np.linalg.norm(pos[d]-pos[c]-vd)])
            new_b = np.array([np.linalg.norm(pos[d]-pos[a]-vd), np.linalg.norm(pos[b]-pos[c]-vb)])
            # Both time directions must independently prefer the entire swap.
            if old_f.sum()-new_f.sum() < .5 or old_b.sum()-new_b.sum() < .5:
                continue
            old_error = .5 * (old_f + old_b)
            new_error = .5 * (new_f + new_b)
            if np.any(new_error > old_error + .25):
                continue
            gain = float(old_error.sum()-new_error.sum())
            # Pay the full original probability bonus before replacing edges.
            gain -= evidence(ei) + evidence(ej)
            if gain < min_gain:
                continue
            stats['tracklet_supported'] += 1
            proposals.append((-gain, i, j, sa | sb | sc | sd))

    result = list(edges)
    changed, protected = set(), set()
    for _, i, j, support in sorted(proposals):
        # No later swap may invalidate the measurements used for an accepted one.
        if {i, j} & (changed | protected) or support & changed:
            stats['tracklet_conflicts'] += 1
            continue
        for idx, target in ((i, int(edges[j]['target_id'])), (j, int(edges[i]['target_id']))):
            e = dict(edges[idx])
            source = int(e['source_id'])
            e.update(target_id=target, edge_prob=0.0,
                     distance_um=float(np.linalg.norm(pos[target]-pos[source])),
                     motion_distance_um=None, motion_pass='tracklet_swap', tracklet_repaired=1)
            result[idx] = e
        changed.update((i, j))
        protected.update(support)
        stats['tracklet_swaps'] += 1
    stats['tracklet_changed_edges'] = len(changed)
    return result, stats
