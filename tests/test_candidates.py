"""Tests for multi-hypothesis candidate nodes + singleton pruning."""

import numpy as np

from ctrack.link.candidates import build_nodes, prune_singletons


def test_build_nodes_ids_and_scores():
    det = {
        0: (np.array([[1.0, 2, 3], [4, 5, 6]]), np.array([0.9, 0.3])),
        1: (np.array([[7.0, 8, 9]]), np.array([0.7])),
    }
    nodes, scores = build_nodes(det, start_id=1)
    assert nodes.shape == (3, 5)
    assert list(nodes[:, 0].astype(int)) == [1, 2, 3]      # frame-order ids
    assert list(nodes[:, 1].astype(int)) == [0, 0, 1]      # timepoints
    assert scores == {1: 0.9, 2: 0.3, 3: 0.7}


def test_build_nodes_missing_scores_default_one():
    det = {0: (np.array([[1.0, 2, 3]]), None)}
    _, scores = build_nodes(det)
    assert scores == {1: 1.0}


def test_prune_singletons_keeps_linked_and_confident():
    nodes = np.array([[1, 0, 0, 0, 0], [2, 1, 0, 0, 1], [3, 0, 5, 5, 5]], dtype=np.float64)
    edges = np.array([[1, 2]], dtype=np.int64)  # node 3 is an isolated singleton
    scores = {1: 0.2, 2: 0.2, 3: 0.1}           # 1,2 low-conf but LINKED -> kept; 3 dropped
    kept, _ = prune_singletons(nodes, edges, scores, min_score=0.5)
    assert set(kept[:, 0].astype(int)) == {1, 2}


def test_prune_singletons_keeps_confident_singleton():
    nodes = np.array([[3, 0, 5, 5, 5]], dtype=np.float64)
    edges = np.zeros((0, 2), dtype=np.int64)
    kept, _ = prune_singletons(nodes, edges, {3: 0.9}, min_score=0.5)
    assert set(kept[:, 0].astype(int)) == {3}
