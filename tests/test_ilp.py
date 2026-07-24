"""Tests for the ILP linker. Requires tracksdata; skipped otherwise."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tracksdata")

from ctrack.link.ilp import link_ilp  # noqa: E402

ISO = np.array([1.625, 1.625, 1.625])


def test_links_two_parallel_tracks() -> None:
    nodes = np.array([
        [1, 0, 32, 100, 100], [2, 1, 32, 102, 100], [3, 2, 32, 104, 100],
        [10, 0, 32, 150, 150], [11, 1, 32, 150, 152], [12, 2, 32, 150, 154],
    ], dtype=np.float64)
    edges = {tuple(e) for e in link_ilp(nodes, ISO, max_disp_um=10).tolist()}
    assert edges == {(1, 2), (2, 3), (10, 11), (11, 12)}


def test_forest_constraint_no_two_parents() -> None:
    """Even when two frame-t nodes are near one frame-t+1 node, the ILP result is a forest."""
    nodes = np.array([
        [1, 0, 32, 100, 100], [2, 0, 32, 104, 100],  # two candidates at t=0
        [3, 1, 32, 102, 100],                          # one node at t=1, between them
    ], dtype=np.float64)
    edges = link_ilp(nodes, ISO, max_disp_um=10, topk=3)
    targets = edges[:, 1].tolist() if len(edges) else []
    assert len(targets) == len(set(targets)), "no node may have two parents"


def test_respects_distance_cap() -> None:
    nodes = np.array([[1, 0, 32, 100, 100], [2, 1, 32, 200, 100]], dtype=np.float64)  # far apart
    edges = link_ilp(nodes, ISO, max_disp_um=10)
    assert len(edges) == 0


def test_learned_affinity_overrides_distance() -> None:
    """A confident learned edge to the farther node should win over the nearer one."""
    nodes = np.array([
        [1, 0, 32, 100, 100],
        [2, 1, 32, 103, 100],   # nearer
        [3, 1, 32, 106, 100],   # farther but high affinity
    ], dtype=np.float64)
    affinity = {(1, 3): 0.99, (1, 2): 0.01}
    edges = {tuple(e) for e in link_ilp(nodes, ISO, max_disp_um=15, edge_affinity=affinity).tolist()}
    assert (1, 3) in edges and (1, 2) not in edges
