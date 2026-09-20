"""Cardinality-preserving endpoint reassignment; no detection or division edits.

Compare p->a with p->b (b unmatched), or q->a (q unmatched).
The unmatched endpoint moves; it is not interpreted as a new division.
Scores are heuristic bidirectional motion residuals, not probabilities.
"""
import math
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, .40625, .40625])


def repair(nodes, edges, *, enabled=True, max_distance=10., min_gain=1., ambiguity_margin=.5):
    stats = Counter(dict(endpoint_enabled=int(enabled), endpoint_candidates=0,
                         endpoint_eligible=0, endpoint_accepted=0))
    events = []
    if not enabled or not nodes or not edges:
        return edges, dict(stats), events
    if min(max_distance, min_gain, ambiguity_margin) <= 0:
        raise ValueError('Positive distances and margins required')
    pos = {int(k): np.array([n['z'], n['y'], n['x']], float)*SCALE for k,n in nodes.items()}
    if any(not np.isfinite(v).all() for v in pos.values()):
        raise ValueError('Nonfinite coordinates')
    incoming, outgoing, frames = {}, {}, {}
    for k,n in nodes.items(): frames.setdefault(int(n['t']), []).append(k)
    for i,e in enumerate(edges):
        a,b = int(e['source_id']),int(e['target_id'])
        if a not in nodes or b not in nodes or nodes[b]['t'] != nodes[a]['t']+1:
            raise ValueError('Invalid endpoints or time')
        incoming.setdefault(b, []).append(i);outgoing.setdefault(a, []).append(i)
    if any(len(v)>1 for v in incoming.values()) or any(len(v)>2 for v in outgoing.values()):
        raise ValueError('Invalid degrees')

    def ordinary(i):
        e=edges[i]
        return e.get('motion_relinked') and not any(e.get(k) for k in
            ('tracklet_repaired','gap_closed','gap2_recovered','safe_division','endpoint_repaired'))

    def context(n, back):
        steps, support = [],set()
        for _ in range(2):
            ids=(incoming if back else outgoing).get(n, [])
            if len(ids)!=1 or not ordinary(ids[0]):break
            i=ids[0];other=int(edges[i]['source_id' if back else 'target_id'])
            if len((outgoing if back else incoming).get(other, []))!=1:break
            step=pos[n]-pos[other] if back else pos[other]-pos[n]
            if np.linalg.norm(step)>max_distance:break
            steps.append(step);support.add(i);n=other
        return (np.mean(steps,axis=0),support) if steps else None

    past={k:context(k,True) for k in nodes}
    future={k:context(k,False) for k in nodes}
    def probability(e):
        try:p=float(e.get('edge_prob') or 0.)
        except (TypeError,ValueError):return 0.
        if not math.isfinite(p):return 0.
        return p if 0<=p<=1 else 1/(1+math.exp(-max(-20.,min(20.,p))))

    starts={};ends={}
    for t,ids in frames.items():
        for lookup, pool in ((starts,[k for k in ids if not incoming.get(k) and future[k] is not None]),
                             (ends,[k for k in ids if not outgoing.get(k) and past[k] is not None])):
            pool=sorted(pool)
            if pool:lookup[t]=(pool,cKDTree([pos[k] for k in pool]))
    proposals=[]
    for i,e in enumerate(edges):
        a,b=int(e['source_id']),int(e['target_id'])
        if not ordinary(i) or len(outgoing[a])!=1 or past[a] is None or future[b] is None:continue
        old_delta=pos[b]-pos[a]
        old_f=float(np.linalg.norm(old_delta-past[a][0]))
        old_b=float(np.linalg.norm(old_delta-future[b][0]))
        options=[]
        for kind,pools,t,anchor in [('target',starts,int(nodes[b]['t']),a),
                                   ('source',ends,int(nodes[a]['t']),b)]:
            if t not in pools:continue
            pool,tree=pools[t]
            for j in sorted(tree.query_ball_point(pos[anchor],max_distance)):
                other=pool[j];u,v=(a,other) if kind=='target' else (other,b)
                delta=pos[v]-pos[u]
                nf=float(np.linalg.norm(delta-past[u][0]));nb=float(np.linalg.norm(delta-future[v][0]))
                gain=.5*(old_f+old_b-nf-nb)-probability(e)
                row=dict(kind=kind,t=int(nodes[a]['t']),old_source=a,old_target=b,
                    new_source=u,new_target=v,old_forward=old_f,old_backward=old_b,
                    new_forward=nf,new_backward=nb,old_probability=probability(e),gain=gain,
                    decision='rejected_motion')
                events.append(row);r=len(events)-1;stats['endpoint_candidates']+=1
                if old_f-nf<.5 or old_b-nb<.5 or gain<min_gain:continue
                row['decision']='eligible';stats['endpoint_eligible']+=1
                support=past[a][1]|future[b][1]|past[u][1]|future[v][1]
                options.append((gain,i,a,b,u,v,support,r))
        # Do not resolve near-tied alternatives by node ID.
        options.sort(key=lambda x:(-x[0],x[4],x[5]))
        if len(options)>1 and options[0][0]-options[1][0]<ambiguity_margin:
            for opt in options:events[opt[-1]]['decision']='rejected_ambiguous'
            continue
        if options:
            proposals.append(options[0])
            for opt in options[1:]:events[opt[-1]]['decision']='rejected_lower_gain'
    result=list(edges);changed=set();protected=set();used=set()
    for gain,i,a,b,u,v,support,r in sorted(proposals,key=lambda x:(-x[0],x[1])):
        row=events[r]
        if {a,b,u,v}&used or i in changed|protected or support&changed:
            row['decision']='rejected_conflict';continue
        new=dict(edges[i]);new.update(source_id=u,target_id=v,edge_prob=0.,
            distance_um=float(np.linalg.norm(pos[v]-pos[u])),motion_distance_um=None,
            motion_pass='endpoint_reassignment',endpoint_repaired=1)
        result[i]=new;used.update((a,b,u,v));changed.add(i);protected.update(support)
        row['decision']='accepted';stats['endpoint_accepted']+=1
    stats.update({'endpoint_rejected_'+reason:sum(r['decision']=='rejected_'+reason for r in events)
                  for reason in ('motion','ambiguous','conflict','lower_gain')})
    return result,dict(stats),events
