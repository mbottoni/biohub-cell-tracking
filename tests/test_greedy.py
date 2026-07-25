"""Tests for the fast greedy divisions-aware linker."""

import numpy as np

from ctrack.link.greedy import link_greedy

SCALE = np.array([1.625, 0.40625, 0.40625])


def _nodes(rows):
    return np.array(rows, dtype=np.float64)


def test_simple_chain():
    # one cell moving straight over 3 frames -> a 2-edge chain
    nodes = _nodes([[1, 0, 10, 10, 10], [2, 1, 10, 10, 12], [3, 2, 10, 10, 14]])
    edges = link_greedy(nodes, SCALE, max_disp_um=12)
    assert {tuple(e) for e in edges} == {(1, 2), (2, 3)}


def test_each_child_one_parent():
    # two parents both near one child: child takes only its best (closest) parent
    nodes = _nodes([[1, 0, 10, 10, 10], [2, 0, 10, 10, 40], [3, 1, 10, 10, 11]])
    edges = link_greedy(nodes, SCALE, max_disp_um=30)
    tgt = [e for e in edges if e[1] == 3]
    assert len(tgt) == 1 and tgt[0][0] == 1  # closer parent wins, single parent


def test_division_allowed_on_strong_evidence():
    # one parent, two close children -> division (parent gets 2 children)
    nodes = _nodes([[1, 0, 10, 10, 10], [2, 1, 10, 10, 11], [3, 1, 10, 10, 9]])
    edges = link_greedy(nodes, SCALE, max_disp_um=12, allow_division=True, div_min_score=0.1)
    kids = sorted(int(e[1]) for e in edges if e[0] == 1)
    assert kids == [2, 3]  # both children linked to the single parent


def test_division_suppressed_when_disabled():
    nodes = _nodes([[1, 0, 10, 10, 10], [2, 1, 10, 10, 11], [3, 1, 10, 10, 9]])
    edges = link_greedy(nodes, SCALE, max_disp_um=12, allow_division=False)
    kids = [e for e in edges if e[0] == 1]
    assert len(kids) == 1  # only one child without divisions


def test_affinity_overrides_distance():
    # closer child (2) but affinity prefers the farther child (3)
    nodes = _nodes([[1, 0, 10, 10, 10], [2, 1, 10, 10, 11], [3, 1, 10, 10, 14]])
    aff = {(1, 2): 0.1, (1, 3): 0.9}
    edges = link_greedy(nodes, SCALE, max_disp_um=12, edge_affinity=aff,
                        allow_division=False)
    assert [tuple(int(x) for x in e) for e in edges] == [(1, 3)]


def test_empty():
    assert link_greedy(np.zeros((0, 5)), SCALE).shape == (0, 2)
