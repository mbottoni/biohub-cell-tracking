"""Read GEFF ground-truth graphs.

GEFF is a zarr-backed graph format from the Royer lab:

    <id>.geff/
      zarr.json                     attributes.geff.{axes, extra, ...}
      nodes/ids                     (N,) uint64
      nodes/props/{t,z,y,x}/values  (N,) int64
      edges/ids                     (E, 2) uint64 -- (source, target)

We read it directly with zarr rather than via the `geff` package: it's a handful of arrays,
and this keeps the dependency surface small enough to vendor into a Kaggle notebook.

`estimated_number_of_nodes` (in the geff `extra` metadata) is the denominator of the metric's
node-budget term -- see ctrack.eval.biohub and docs/metric.md. The annotated node count is a
tiny fraction of it; the ground truth is only ~0.2% dense.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ctrack.eval.biohub import Graph


@dataclass(frozen=True)
class GroundTruth:
    graph: Graph
    estimated_number_of_nodes: float
    dataset: str

    @property
    def n_nodes(self) -> int:
        return len(self.graph.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.graph.edges)

    @property
    def density(self) -> float:
        """Annotated nodes as a fraction of the estimated true cell count."""
        return self.n_nodes / self.estimated_number_of_nodes if self.estimated_number_of_nodes else float("nan")

    def n_divisions(self) -> int:
        return sum(1 for d in self.graph.out_degree().values() if d >= 2)

    def n_tracks(self) -> int:
        """Number of roots -- nodes with no parent."""
        targets = set(self.graph.edges[:, 1].astype(np.int64).tolist()) if self.n_edges else set()
        return sum(1 for i in self.graph.ids.tolist() if i not in targets)


def load_geff(path: str | Path) -> GroundTruth:
    """Load one `.geff` store into a Graph plus its metadata."""
    import zarr

    path = Path(path)
    root = zarr.open(str(path), mode="r")

    ids = np.asarray(root["nodes/ids"][:]).astype(np.int64)
    t = np.asarray(root["nodes/props/t/values"][:]).astype(np.float64)
    z = np.asarray(root["nodes/props/z/values"][:]).astype(np.float64)
    y = np.asarray(root["nodes/props/y/values"][:]).astype(np.float64)
    x = np.asarray(root["nodes/props/x/values"][:]).astype(np.float64)
    nodes = np.column_stack([ids.astype(np.float64), t, z, y, x])

    edges_arr = root["edges/ids"]
    edges = (
        np.asarray(edges_arr[:]).astype(np.int64).reshape(-1, 2)
        if edges_arr.shape[0] > 0
        else np.zeros((0, 2), dtype=np.int64)
    )

    meta = json.loads((path / "zarr.json").read_text())
    extra = meta["attributes"]["geff"].get("extra") or {}
    n_total = float(extra.get("estimated_number_of_nodes", float("nan")))

    return GroundTruth(
        graph=Graph(nodes=nodes, edges=edges),
        estimated_number_of_nodes=n_total,
        dataset=path.name.replace(".geff", ""),
    )


def find_geff_stores(root: str | Path) -> list[Path]:
    """All `*.geff` stores under `root`, sorted."""
    return sorted(Path(root).glob("**/*.geff"))
