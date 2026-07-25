"""Fast greedy/LAP linking — no ILP, so local iteration isn't blocked by the SCIP fallback.

Trackastra (ECCV'24) showed that greedy decoding of learned pairwise link probabilities matches
a full ILP on most datasets, at a fraction of the cost. We hit exactly the failure the ILP path
suffers from: without a Gurobi license, tracksdata falls back to SCIP and a threshold sweep took
hours. This module decodes tracks from candidate edges in milliseconds, using either a learned
affinity or plain distance, and handles divisions natively (a parent may take two children).

Decoding: rank all consecutive-frame candidate edges by score (affinity, or -distance), then add
each edge if its child still has no parent and its parent has < max_children. Divisions are
allowed only when the second child's score clears `div_min_score`, so a division is proposed only
on strong evidence (they are rare — ~151 in 199 volumes; see docs/data.md).
"""

from __future__ import annotations

import numpy as np


def _candidate_edges(nodes: np.ndarray, scale: np.ndarray, max_disp_um: float, topk: int):
    """(src_idx, tgt_idx, distance_um) for topk nearest next-frame neighbours within the cap."""
    t = nodes[:, 1].astype(int)
    pos = nodes[:, 2:5]
    out: list[tuple[int, int, float]] = []
    for f in np.unique(t):
        cur = np.flatnonzero(t == f)
        nxt = np.flatnonzero(t == f + 1)
        if cur.size == 0 or nxt.size == 0:
            continue
        d = np.linalg.norm((pos[cur][:, None, :] - pos[nxt][None, :, :]) * scale, axis=2)
        for i, ci in enumerate(cur):
            for j in np.argsort(d[i])[:topk]:
                if d[i, j] <= max_disp_um:
                    out.append((int(ci), int(nxt[j]), float(d[i, j])))
    return out


def link_greedy(
    nodes: np.ndarray,
    scale: np.ndarray,
    *,
    max_disp_um: float = 12.0,
    topk: int = 3,
    edge_affinity: dict[tuple[int, int], float] | None = None,
    allow_division: bool = True,
    div_min_score: float = 0.5,
    min_score: float = 0.0,
) -> np.ndarray:
    """Greedy directed-forest linking. `nodes` is (N,5)=[id,t,z,y,x]; returns edges (E,2) as
    node_id pairs. A node has <=1 parent and <=2 children (2 only on a strong division).

    Scores: if `edge_affinity` maps (src_id,tgt_id)->prob in (0,1], that is the score (higher =
    better). Otherwise score = 1 - distance/max_disp_um (so distance 0 -> 1, cap -> 0).
    """
    if len(nodes) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    ids = nodes[:, 0].astype(np.int64)

    scored: list[tuple[float, int, int]] = []  # (score, src_id, tgt_id)
    for si, ti, dist in _candidate_edges(nodes, scale, max_disp_um, topk):
        s_id, t_id = int(ids[si]), int(ids[ti])
        if edge_affinity is not None:
            score = float(edge_affinity.get((s_id, t_id), 0.0))
        else:
            score = 1.0 - dist / max_disp_um
        if score >= min_score:
            scored.append((score, s_id, t_id))

    scored.sort(reverse=True)  # highest score first
    children: dict[int, int] = {}   # parent_id -> #children taken
    has_parent: set[int] = set()
    edges: list[tuple[int, int]] = []
    for score, s_id, t_id in scored:
        if t_id in has_parent:
            continue
        n_child = children.get(s_id, 0)
        if n_child >= 2:
            continue
        if n_child == 1 and not (allow_division and score >= div_min_score):
            continue  # second child only on a strong division
        edges.append((s_id, t_id))
        has_parent.add(t_id)
        children[s_id] = n_child + 1

    return np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
