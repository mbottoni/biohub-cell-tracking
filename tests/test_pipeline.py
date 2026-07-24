"""Detection and linking on synthetic volumes with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from ctrack.detect.blob import detect_frame, rethreshold
from ctrack.link.hungarian import build_graph, close_gaps, link_frames

ISO = np.array([1.625, 1.625, 1.625])


def _blobs(shape: tuple[int, int, int], centres: list[tuple[int, int, int]]) -> np.ndarray:
    """A frame with a bright gaussian-ish blob at each centre."""
    vol = np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for cz, cy, cx in centres:
        d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        vol = np.maximum(vol, np.exp(-d2 / 2.0).astype(np.float32))
    return vol


def test_detects_well_separated_blobs() -> None:
    centres = [(10, 10, 10), (10, 10, 30), (30, 40, 20)]
    coords, scores = detect_frame(_blobs((48, 48, 48), centres), ISO, threshold=0.3)
    assert len(coords) == 3
    found = {tuple(c.astype(int)) for c in coords}
    assert found == set(centres)
    assert (scores > 0.3).all()


def test_threshold_controls_the_node_budget() -> None:
    """Raising the threshold must monotonically shrink the detection count.

    This is the knob the metric's unclamped node-budget multiplier is swept against.
    """
    frame = _blobs((48, 48, 48), [(10, 10, 10), (10, 10, 30), (30, 40, 20)])
    frame[30, 40, 20] *= 0.4  # one dim blob
    counts = [len(detect_frame(frame, ISO, threshold=th)[0]) for th in (0.1, 0.5, 0.95)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_rethreshold_equals_a_fresh_higher_threshold_detection() -> None:
    """The sweep optimization: re-filtering a low-threshold detection must equal detecting
    fresh at the higher threshold. NMS peaks are threshold-independent, so this must hold
    exactly -- if it ever doesn't, the sweep results are invalid."""
    frame = _blobs((48, 48, 48), [(10, 10, 10), (10, 10, 30), (30, 40, 20)])
    frame[30, 40, 20] *= 0.5

    low = {0: detect_frame(frame, ISO, threshold=0.1)}
    refiltered = rethreshold(low, 0.6)[0]
    fresh = detect_frame(frame, ISO, threshold=0.6)

    np.testing.assert_array_equal(np.sort(refiltered[0], axis=0), np.sort(fresh[0], axis=0))


def test_link_frames_respects_displacement_cap() -> None:
    a = np.array([[10.0, 10.0, 10.0]])
    b = np.array([[10.0, 12.0, 10.0]])  # 2 voxels * 1.625 = 3.25 um
    assert link_frames(a, b, ISO, max_displacement_um=10.0) == [(0, 0)]
    assert link_frames(a, b, ISO, max_displacement_um=3.0) == []


def test_link_frames_prefers_the_global_optimum() -> None:
    """Two cells that nearly swap: greedy nearest-neighbour would cross them."""
    a = np.array([[0.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    b = np.array([[0.0, 5.0, 0.0], [0.0, 1.0, 0.0]])
    assert sorted(link_frames(a, b, ISO, max_displacement_um=20.0)) == [(0, 1), (1, 0)]


def test_build_graph_produces_a_directed_forest() -> None:
    det = {t: (np.array([[10.0, 10.0 + 2 * t, 10.0], [30.0, 30.0, 30.0]]), np.ones(2))
           for t in range(4)}
    nodes, edges = build_graph(det, ISO)
    assert len(nodes) == 8
    assert len(edges) == 6  # 2 tracks x 3 transitions

    assert len(set(nodes[:, 0].astype(int))) == 8, "node ids must be unique"
    targets = edges[:, 1]
    assert len(set(targets.tolist())) == len(targets), "no node may have two parents"


def test_build_graph_skips_non_consecutive_frames() -> None:
    det = {0: (np.array([[10.0, 10.0, 10.0]]), np.ones(1)),
           2: (np.array([[10.0, 11.0, 10.0]]), np.ones(1))}
    _, edges = build_graph(det, ISO)
    assert len(edges) == 0, "must not emit a t->t+2 skip edge"


def test_close_gaps_inserts_a_node_not_a_skip_edge() -> None:
    """GT edges are all dt=1, so a bridged gap must materialise an intermediate node."""
    nodes = np.array([[1, 0, 10.0, 10.0, 10.0], [2, 2, 10.0, 12.0, 10.0]], dtype=np.float64)
    edges = np.zeros((0, 2), dtype=np.int64)

    new_nodes, new_edges = close_gaps(nodes, edges, ISO, max_total_um=10.0)
    assert len(new_nodes) == 3, "a synthetic node at t=1 should have been added"
    assert len(new_edges) == 2, "and two dt=1 edges, never one dt=2 edge"

    synth = new_nodes[new_nodes[:, 0] == 3][0]
    assert synth[1] == 1
    np.testing.assert_allclose(synth[2:5], [10.0, 11.0, 10.0])

    node_t = {int(r[0]): int(r[1]) for r in new_nodes}
    for s, t in new_edges:
        assert node_t[int(t)] - node_t[int(s)] == 1


def test_close_gaps_refuses_ambiguous_pairs() -> None:
    """Two tails competing for one head must not invent a branch."""
    nodes = np.array([
        [1, 0, 10.0, 10.0, 10.0],
        [2, 0, 10.0, 12.0, 10.0],
        [3, 2, 10.0, 11.0, 10.0],
    ], dtype=np.float64)
    new_nodes, new_edges = close_gaps(nodes, np.zeros((0, 2), np.int64), ISO, max_total_um=10.0)
    assert len(new_nodes) == 3 and len(new_edges) == 0


def test_close_gaps_respects_distance_cap() -> None:
    nodes = np.array([[1, 0, 10.0, 10.0, 10.0], [2, 2, 10.0, 40.0, 10.0]], dtype=np.float64)
    out_nodes, out_edges = close_gaps(nodes, np.zeros((0, 2), np.int64), ISO, max_total_um=10.0)
    assert len(out_nodes) == 2 and len(out_edges) == 0
