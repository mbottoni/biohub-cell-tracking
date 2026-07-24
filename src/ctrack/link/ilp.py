"""Global ILP linking via tracksdata (the Ultrack-style linker the top solutions use).

Replaces the greedy Hungarian linker. Instead of committing to one assignment per frame pair,
build a candidate graph over a temporal window and let an integer program pick the
globally-optimal directed forest under flow constraints — jointly modelling appearance,
disappearance, and division.

Two edge-weight sources:
  * distance (default) — weight ∝ physical distance, no training needed. A strict improvement
    over Hungarian because it optimizes globally and handles divisions.
  * learned affinity — pass edge probabilities from the edge model; weight = −logit. This is
    where the real gains are (docs/detector.md).

ILP convention (tracksdata minimizes total weight):
  * `node_weight` must be NEGATIVE — a reward for selecting a detection. With node_weight=0 the
    trivial all-zero solution wins and nothing is linked.
  * `edge_weight` NEGATIVE for attractive links (short distance / high affinity), so the ILP
    prefers connecting a node over paying its appearance+disappearance penalty.
  * `appearance/disappearance_weight` POSITIVE — the cost of a track starting/ending, which is
    what makes long continuous tracks preferable to fragmented ones.
  * `division_weight` POSITIVE penalizes divisions (rare — 151 in 199 volumes — so discourage
    unless evidence is strong).
"""

from __future__ import annotations

import numpy as np

from ctrack.eval.biohub import Graph

DEFAULT_MAX_DISPLACEMENT_UM = 12.0   # 80/199 GT steps exceed 10µm; widen the cap (docs/data.md)
DEFAULT_TOPK = 3                      # candidate parents/children per node


def _candidate_edges(
    nodes: np.ndarray,
    scale: np.ndarray,
    max_disp_um: float,
    topk: int,
) -> list[tuple[int, int, float]]:
    """Candidate (source_idx, target_idx, distance_um) between consecutive frames.

    For each node, keep its topk nearest neighbours in the next frame within the cap. Keeping a
    few candidates rather than one (as Hungarian forces) is what lets the ILP resolve
    crossings and divisions globally.
    """
    t = nodes[:, 1].astype(int)
    pos = nodes[:, 2:5]
    out: list[tuple[int, int, float]] = []
    frames = np.unique(t)
    for f in frames:
        cur = np.flatnonzero(t == f)
        nxt = np.flatnonzero(t == f + 1)
        if cur.size == 0 or nxt.size == 0:
            continue
        d = np.linalg.norm((pos[cur][:, None, :] - pos[nxt][None, :, :]) * scale, axis=2)
        for i, ci in enumerate(cur):
            order = np.argsort(d[i])[:topk]
            for j in order:
                if d[i, j] <= max_disp_um:
                    out.append((int(ci), int(nxt[j]), float(d[i, j])))
    return out


def link_ilp(
    nodes: np.ndarray,
    scale: np.ndarray,
    *,
    max_disp_um: float = DEFAULT_MAX_DISPLACEMENT_UM,
    topk: int = DEFAULT_TOPK,
    edge_affinity: dict[tuple[int, int], float] | None = None,
    node_reward: float = 1.0,
    division_weight: float = 1.0,
    appearance_weight: float = 1.0,
    disappearance_weight: float = 1.0,
) -> np.ndarray:
    """Solve the linking ILP. `nodes` is (N,5)=[id,t,z,y,x]; returns selected edges (E,2) as
    node_id pairs.

    If `edge_affinity` maps (src_id, tgt_id) -> probability in (0,1], edges are weighted by
    −logit(p) so confident links are cheap; otherwise weight = distance.
    """
    import polars as pl
    import tracksdata as td
    from tracksdata.graph import RustWorkXGraph
    from tracksdata.solvers import ILPSolver

    K = td.DEFAULT_ATTR_KEYS
    g = RustWorkXGraph()
    for key in (K.Z, K.Y, K.X):
        g.add_node_attr_key(key, dtype=pl.Float64, default_value=0.0)
    g.add_node_attr_key("score", dtype=pl.Float64, default_value=0.0)
    g.add_edge_attr_key("weight", dtype=pl.Float64, default_value=0.0)

    ids = nodes[:, 0].astype(int)
    idx_of = {}
    id_of_idx = {}
    for row_i, (nid, t, z, y, x) in enumerate(nodes):
        # negative node score = reward for selecting this detection
        gi = g.add_node({K.T: int(t), K.Z: float(z), K.Y: float(y), K.X: float(x),
                         "score": -abs(node_reward)})
        idx_of[row_i] = gi
        id_of_idx[gi] = int(nid)

    cands = _candidate_edges(nodes, scale, max_disp_um, topk)
    if not cands:
        return np.zeros((0, 2), dtype=np.int64)

    for si, ti, dist in cands:
        if edge_affinity is not None:
            p = edge_affinity.get((int(ids[si]), int(ids[ti])), 1e-4)
            p = min(max(p, 1e-4), 1 - 1e-4)
            weight = -float(np.log(p / (1 - p)))          # −logit: confident edge -> negative
        else:
            # negative reward, most attractive at distance 0, crossing zero at the cap
            weight = float(dist) - max_disp_um
        g.add_edge(source_id=idx_of[si], target_id=idx_of[ti], attrs={"weight": weight})

    solver = ILPSolver(
        edge_weight="weight",
        node_weight="score",
        division_weight=division_weight,
        appearance_weight=appearance_weight,
        disappearance_weight=disappearance_weight,
    )
    solution = solver.solve(g)
    if solution is None:
        return np.zeros((0, 2), dtype=np.int64)

    edge_df = solution.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET])
    edges = [
        (id_of_idx[int(s)], id_of_idx[int(t)])
        for s, t in zip(edge_df[K.EDGE_SOURCE].to_list(), edge_df[K.EDGE_TARGET].to_list())
    ]
    return np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def link_graph_ilp(nodes: np.ndarray, scale: np.ndarray, **kwargs) -> Graph:
    """Convenience: return a Graph with ILP-selected edges."""
    return Graph(nodes=nodes, edges=link_ilp(nodes, scale, **kwargs))
