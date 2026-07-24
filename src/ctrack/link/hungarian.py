"""Frame-to-frame linking by optimal assignment.

The baseline linker. Consecutive frames only, and deliberately so: every ground-truth edge we
have inspected connects adjacent timepoints (dt=1). A skip edge hung off an annotated node
would be a pure false positive, so gap closing must insert an intermediate *node* plus two
dt=1 edges rather than a long edge -- see `close_gaps`.

Displacement measured from ground truth: median 2.88 µm, p90 4.67 µm, max 7.81 µm. The public
baselines cap at 10 µm, which is generous; the cap is a knob worth sweeping.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_MAX_DISPLACEMENT_UM = 10.0


def link_frames(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    scale: np.ndarray,
    *,
    max_displacement_um: float = DEFAULT_MAX_DISPLACEMENT_UM,
) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment between two frames.

    Returns index pairs `(i in a, j in b)` for matches within the displacement cap.
    """
    if len(coords_a) == 0 or len(coords_b) == 0:
        return []
    cost = np.linalg.norm(
        (coords_a[:, None, :] - coords_b[None, :, :]) * scale, axis=2
    )
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= max_displacement_um]


def build_graph(
    detections: dict[int, tuple[np.ndarray, np.ndarray]],
    scale: np.ndarray,
    *,
    max_displacement_um: float = DEFAULT_MAX_DISPLACEMENT_UM,
    start_id: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn per-frame detections into a tracking graph.

    Returns `(nodes, edges)` where nodes is (N, 5) [node_id, t, z, y, x] and edges is (E, 2).
    Node ids are globally unique and assigned in frame order.
    """
    node_rows: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    ids_by_frame: dict[int, np.ndarray] = {}

    next_id = start_id
    for t in sorted(detections):
        coords, _ = detections[t]
        ids = np.arange(next_id, next_id + len(coords), dtype=np.int64)
        next_id += len(coords)
        ids_by_frame[t] = ids
        for nid, (z, y, x) in zip(ids, coords):
            node_rows.append([float(nid), float(t), float(z), float(y), float(x)])

    frames = sorted(detections)
    for t_prev, t_cur in zip(frames, frames[1:]):
        if t_cur - t_prev != 1:
            continue  # only consecutive frames; see module docstring
        a, _ = detections[t_prev]
        b, _ = detections[t_cur]
        for i, j in link_frames(a, b, scale, max_displacement_um=max_displacement_um):
            edges.append((int(ids_by_frame[t_prev][i]), int(ids_by_frame[t_cur][j])))

    nodes = np.array(node_rows, dtype=np.float64) if node_rows else np.zeros((0, 5))
    edge_arr = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
    return nodes, edge_arr


def close_gaps(
    nodes: np.ndarray,
    edges: np.ndarray,
    scale: np.ndarray,
    *,
    max_total_um: float = DEFAULT_MAX_DISPLACEMENT_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """Bridge single-frame gaps by inserting a midpoint node and two dt=1 edges.

    A track that ends at t and another that starts at t+2 are joined through a synthetic node
    at t+1, placed at the midpoint. We insert a node rather than a t -> t+2 edge because no
    ground-truth edge we have seen spans more than one frame, so a skip edge touching an
    annotated node would score as a false positive.

    Only unambiguous pairs are joined -- a tail and a head that are each other's sole candidate
    within the distance cap. Greedy many-to-one closing would invent branches.
    """
    if len(nodes) == 0:
        return nodes, edges

    ids = nodes[:, 0].astype(np.int64)
    t = nodes[:, 1].astype(np.int64)
    pos = nodes[:, 2:5]
    index = {int(i): k for k, i in enumerate(ids)}

    has_out = set(edges[:, 0].astype(np.int64).tolist()) if len(edges) else set()
    has_in = set(edges[:, 1].astype(np.int64).tolist()) if len(edges) else set()
    tails = [int(i) for i in ids if int(i) not in has_out]
    heads = [int(i) for i in ids if int(i) not in has_in]

    candidates: dict[int, list[tuple[float, int]]] = {}
    for tail in tails:
        kt = index[tail]
        for head in heads:
            kh = index[head]
            if t[kh] - t[kt] != 2:
                continue
            d = float(np.linalg.norm((pos[kh] - pos[kt]) * scale))
            if d <= max_total_um:
                candidates.setdefault(tail, []).append((d, head))

    # keep only mutually-unambiguous pairs
    head_claims: dict[int, list[int]] = {}
    for tail, cands in candidates.items():
        if len(cands) == 1:
            head_claims.setdefault(cands[0][1], []).append(tail)

    new_nodes, new_edges = [], []
    next_id = int(ids.max()) + 1 if len(ids) else 1
    for head, claiming in head_claims.items():
        if len(claiming) != 1:
            continue
        tail = claiming[0]
        kt, kh = index[tail], index[head]
        mid = (pos[kt] + pos[kh]) / 2.0
        new_nodes.append([float(next_id), float(t[kt] + 1), *mid.tolist()])
        new_edges += [(tail, next_id), (next_id, head)]
        next_id += 1

    if not new_nodes:
        return nodes, edges
    return (
        np.vstack([nodes, np.array(new_nodes, dtype=np.float64)]),
        np.vstack([edges, np.array(new_edges, dtype=np.int64)]) if len(edges) else
        np.array(new_edges, dtype=np.int64),
    )
