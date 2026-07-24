"""Cross-validation splits.

The corpus is 199 train crops drawn from only **two** source embryos (`6bba`: 128, `44b6`: 71).
That is an uncomfortable constraint and it shapes everything here.

- `split_by_embryo` is the only split guaranteed free of embryo-level leakage, but it gives
  exactly two folds, each training on one embryo and validating on the other. High variance,
  and it conflates "generalizes across embryos" with "generalizes at all".
- `kfold` gives more folds by splitting *within* embryos. That is only safe if the crops from
  one embryo do not overlap spatially. **This is not yet verified** -- see
  `docs/data.md` open questions. Until it is, treat `kfold` numbers as optimistic and confirm
  any conclusion against `split_by_embryo`.

Both are deterministic given a seed. The golden holdout is carved out first so it can never
leak into a fold, and its selection does not shift when the fold count changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def embryo_of(dataset: str) -> str:
    """`44b6_0113de3b` -> `44b6`."""
    return dataset.split("_")[0]


def _stable_order(datasets: list[str], seed: int) -> list[str]:
    """Deterministic shuffle that doesn't depend on Python's hash randomization."""
    return sorted(datasets, key=lambda d: hashlib.md5(f"{seed}:{d}".encode()).hexdigest())


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.val)
        if overlap:
            raise ValueError(f"train/val overlap: {sorted(overlap)[:5]}")

    @property
    def embryos(self) -> dict[str, set[str]]:
        return {
            "train": {embryo_of(d) for d in self.train},
            "val": {embryo_of(d) for d in self.val},
        }

    @property
    def leaks_embryo(self) -> bool:
        """True if any embryo appears on both sides."""
        return bool(self.embryos["train"] & self.embryos["val"])


def hold_out_golden(datasets: list[str], n: int = 8, seed: int = 0) -> tuple[list[str], list[str]]:
    """Carve off a golden holdout, balanced across embryos.

    Returns `(remaining, golden)`. Touch the golden set only in the final weeks -- see PLAN.md.
    Selection is independent of fold count so it stays fixed as the CV scheme evolves.
    """
    by_embryo: dict[str, list[str]] = {}
    for d in _stable_order(datasets, seed):
        by_embryo.setdefault(embryo_of(d), []).append(d)

    golden: list[str] = []
    per = max(1, n // max(1, len(by_embryo)))
    for group in by_embryo.values():
        golden += group[:per]
    golden = golden[:n]

    remaining = [d for d in datasets if d not in set(golden)]
    return remaining, sorted(golden)


def split_by_embryo(datasets: list[str]) -> list[Split]:
    """Leave-one-embryo-out. The only leakage-free split available -- but only 2 folds."""
    embryos = sorted({embryo_of(d) for d in datasets})
    return [
        Split(
            train=sorted(d for d in datasets if embryo_of(d) != e),
            val=sorted(d for d in datasets if embryo_of(d) == e),
        )
        for e in embryos
    ]


def kfold(datasets: list[str], k: int = 5, seed: int = 0) -> list[Split]:
    """K folds stratified by embryo, splitting *within* embryos.

    Only valid if same-embryo crops are spatially disjoint; unverified. Prefer
    `split_by_embryo` for any decision that matters until that is settled.
    """
    if k < 2:
        raise ValueError("k must be >= 2")

    by_embryo: dict[str, list[str]] = {}
    for d in _stable_order(datasets, seed):
        by_embryo.setdefault(embryo_of(d), []).append(d)

    # round-robin within each embryo so folds stay balanced across embryos
    assignment: dict[str, int] = {}
    for group in by_embryo.values():
        for i, d in enumerate(group):
            assignment[d] = i % k

    return [
        Split(
            train=sorted(d for d in datasets if assignment[d] != f),
            val=sorted(d for d in datasets if assignment[d] == f),
        )
        for f in range(k)
    ]
