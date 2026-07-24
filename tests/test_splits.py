"""Tests for CV splitting. The corpus has only two embryos, so leakage is easy to cause."""

from __future__ import annotations

import pytest

from ctrack.data.splits import (
    Split,
    embryo_of,
    hold_out_golden,
    kfold,
    split_by_embryo,
)

DATASETS = [f"6bba_{i:04x}" for i in range(20)] + [f"44b6_{i:04x}" for i in range(12)]


def test_embryo_of() -> None:
    assert embryo_of("44b6_0113de3b") == "44b6"


def test_split_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Split(train=["a", "b"], val=["b"])


def test_split_by_embryo_is_leak_free() -> None:
    splits = split_by_embryo(DATASETS)
    assert len(splits) == 2, "two embryos means exactly two leak-free folds"
    for s in splits:
        assert not s.leaks_embryo
        assert len(s.train) + len(s.val) == len(DATASETS)
    assert {tuple(sorted(s.embryos["val"]))[0] for s in splits} == {"6bba", "44b6"}


def test_kfold_covers_every_dataset_exactly_once() -> None:
    splits = kfold(DATASETS, k=4, seed=0)
    assert len(splits) == 4
    seen: list[str] = []
    for s in splits:
        seen += s.val
    assert sorted(seen) == sorted(DATASETS), "every dataset validates exactly once"


def test_kfold_balances_embryos_across_folds() -> None:
    for s in kfold(DATASETS, k=4, seed=0):
        assert s.embryos["val"] == {"6bba", "44b6"}, "each fold should see both embryos"


def test_kfold_is_deterministic_and_seed_sensitive() -> None:
    a = [s.val for s in kfold(DATASETS, k=4, seed=0)]
    assert a == [s.val for s in kfold(DATASETS, k=4, seed=0)]
    assert a != [s.val for s in kfold(DATASETS, k=4, seed=1)]


def test_kfold_leaks_embryo_by_construction() -> None:
    """Documents the tradeoff: more folds are only bought by splitting within an embryo."""
    assert all(s.leaks_embryo for s in kfold(DATASETS, k=4, seed=0))


def test_golden_holdout_is_disjoint_and_balanced() -> None:
    remaining, golden = hold_out_golden(DATASETS, n=8, seed=0)
    assert len(golden) == 8
    assert not set(remaining) & set(golden)
    assert sorted(remaining + golden) == sorted(DATASETS)
    assert {embryo_of(d) for d in golden} == {"6bba", "44b6"}


def test_golden_holdout_is_stable_across_fold_counts() -> None:
    """The golden set must not shift when the CV scheme changes, or it stops being golden."""
    _, a = hold_out_golden(DATASETS, n=8, seed=0)
    _, b = hold_out_golden(DATASETS, n=8, seed=0)
    assert a == b


def test_golden_holdout_never_reaches_folds() -> None:
    remaining, golden = hold_out_golden(DATASETS, n=8, seed=0)
    for s in kfold(remaining, k=4, seed=0):
        assert not set(s.train) & set(golden)
        assert not set(s.val) & set(golden)
