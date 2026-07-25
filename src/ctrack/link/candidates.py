"""Candidate nodes for multi-hypothesis linking (Ultrack-style selection).

Rather than committing to one detection threshold — which we saw saturate badly (a fixed
threshold either floods 90k nodes or clips real cells) — feed a generous LOW-threshold candidate
set, each node carrying its detector confidence, and let the linker SELECT the temporally
consistent subset. The ILP already picks nodes via its reward/appearance trade-off; scaling each
node's reward by confidence turns that into confidence-aware multi-hypothesis selection. The
greedy linker approximates the same by pruning low-confidence singletons.

`build_nodes` assigns globally-unique ids in frame order (matching ctrack.link.hungarian.build_graph
so ids are interchangeable) and returns the per-node confidence keyed by id.
"""

from __future__ import annotations

import numpy as np


def build_nodes(
    detections: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    start_id: int = 1,
) -> tuple[np.ndarray, dict[int, float]]:
    """(nodes (N,5)=[id,t,z,y,x], node_scores {id: confidence}) from per-frame (coords, scores).

    Coordinates are taken as given (full-resolution voxels, as the detector emits them).
    """
    rows: list[list[float]] = []
    scores: dict[int, float] = {}
    next_id = start_id
    for t in sorted(detections):
        coords, vals = detections[t]
        if vals is None or len(vals) != len(coords):
            vals = np.ones(len(coords), dtype=np.float64)
        for (z, y, x), v in zip(coords, vals):
            rows.append([float(next_id), float(t), float(z), float(y), float(x)])
            scores[next_id] = float(v)
            next_id += 1
    nodes = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 5))
    return nodes, scores


def build_nodes_with_offsets(
    detections: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    start_id: int = 1,
) -> tuple[np.ndarray, dict[int, float], dict[int, np.ndarray]]:
    """Like `build_nodes`, but detections are (coords, scores, offsets); also returns
    {node_id: predicted parent-offset (3,)} for the motion-based linker (ctrack.link.edge_model.
    motion_affinity). Offsets are in the same (full-res voxel) frame as coords.
    """
    rows: list[list[float]] = []
    scores: dict[int, float] = {}
    node_off: dict[int, np.ndarray] = {}
    next_id = start_id
    for t in sorted(detections):
        coords, vals, offs = detections[t]
        if vals is None or len(vals) != len(coords):
            vals = np.ones(len(coords), dtype=np.float64)
        for k, ((z, y, x), v) in enumerate(zip(coords, vals)):
            rows.append([float(next_id), float(t), float(z), float(y), float(x)])
            scores[next_id] = float(v)
            if offs is not None and len(offs) == len(coords):
                node_off[next_id] = np.asarray(offs[k], dtype=np.float64)
            next_id += 1
    nodes = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 5))
    return nodes, scores, node_off


def prune_singletons(
    nodes: np.ndarray,
    edges: np.ndarray,
    node_scores: dict[int, float],
    *,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop nodes that ended up in no edge AND whose confidence is below `min_score`.

    This is the greedy path's version of multi-hypothesis selection: low-confidence candidates
    that the linker could not attach to a track are almost always spurious, and every extra node
    is taxed by total_node_ratio. Linked nodes are always kept regardless of confidence.
    """
    if len(nodes) == 0:
        return nodes, edges
    linked = set(edges[:, 0].astype(int).tolist()) | set(edges[:, 1].astype(int).tolist()) \
        if len(edges) else set()
    keep_mask = np.array([
        (int(nid) in linked) or (node_scores.get(int(nid), 1.0) >= min_score)
        for nid in nodes[:, 0]
    ])
    return nodes[keep_mask], edges
