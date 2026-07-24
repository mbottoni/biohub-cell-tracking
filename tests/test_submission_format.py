"""Guard the submission invariants. A malformed CSV burns one of only 5 daily submissions."""

from __future__ import annotations

import pandas as pd
import pytest

from ctrack.submit.format import (
    SubmissionError,
    edge_row,
    node_row,
    validate,
    write,
)


def _lineage() -> list[dict]:
    """Two frames, cell 1 divides into 2 and 3; cell 4 appears unlinked."""
    return [
        node_row("ds", 1, t=0, z=5, y=10, x=10),
        node_row("ds", 2, t=1, z=5, y=12, x=10),
        node_row("ds", 3, t=1, z=5, y=8, x=10),
        node_row("ds", 4, t=1, z=20, y=20, x=20),
        edge_row("ds", 1, 2),
        edge_row("ds", 1, 3),
    ]


def test_valid_lineage_stats() -> None:
    stats = validate(pd.DataFrame(_lineage()))["ds"]
    assert stats == {
        "nodes": 4, "edges": 2, "roots": 2, "divisions": 1,
        "over_divisions": 0, "frames": 2,
    }


def test_duplicate_node_id_rejected() -> None:
    rows = _lineage() + [node_row("ds", 1, t=1, z=0, y=0, x=0)]
    with pytest.raises(SubmissionError, match="duplicate node_id"):
        validate(pd.DataFrame(rows))


def test_two_parents_rejected() -> None:
    """The forest constraint: a cell cannot have two parents. Merges must be resolved."""
    rows = _lineage() + [edge_row("ds", 4, 2)]
    with pytest.raises(SubmissionError, match="directed forest"):
        validate(pd.DataFrame(rows))


def test_dangling_edge_endpoint_rejected() -> None:
    rows = _lineage() + [edge_row("ds", 1, 999)]
    with pytest.raises(SubmissionError, match="undeclared nodes"):
        validate(pd.DataFrame(rows))


def test_three_children_rejected() -> None:
    rows = _lineage() + [node_row("ds", 5, t=1, z=6, y=10, x=10), edge_row("ds", 1, 5)]
    with pytest.raises(SubmissionError, match=">2 children"):
        validate(pd.DataFrame(rows))


def test_synthetic_nodes_gated_by_flag() -> None:
    """Out-of-volume hub nodes (the metric augmentation) must be opted into explicitly."""
    rows = _lineage() + [node_row("ds", 99, t=-1000, z=-10000, y=-10000, x=-10000)]
    with pytest.raises(SubmissionError, match="negative node coordinates"):
        validate(pd.DataFrame(rows))
    validate(pd.DataFrame(rows), allow_synthetic=True)


def test_write_adds_id_column(tmp_path) -> None:
    path = tmp_path / "submission.csv"
    write(_lineage(), str(path))
    out = pd.read_csv(path)
    assert list(out.columns) == [
        "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
    ]
    assert out.id.tolist() == list(range(6))
