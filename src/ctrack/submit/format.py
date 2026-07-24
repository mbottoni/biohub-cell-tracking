"""Submission format: build and validate the point-graph CSV.

Schema verified against the official starter notebook
(`inversion/cell-tracking-getting-started-w-nearest-neighbor`):

    id, dataset, row_type, node_id, t, z, y, x, source_id, target_id

  row_type == "node"  -> a detected cell centroid at (t, z, y, x) with identity `node_id`;
                         source_id/target_id are -1
  row_type == "edge"  -> a link source_id -> target_id; node_id/t/z/y/x are -1

The graph must be a **directed forest** per dataset: node_ids unique, and no node has more
than one parent (no duplicated target_id). A division is a node with two children.

Coordinates are in *voxel* units of the full-resolution array, ordered (z, y, x). Physical
scale is VOXEL_SIZE_UM -- remember to convert before any distance threshold.

Submissions are scarce (5/day) and a malformed one wastes a slot, so validate before writing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
NODE, EDGE = "node", "edge"

# (Z, Y, X) micrometres per voxel. Z is 4x coarser than Y/X -- subsampling Y/X by 4 yields
# isotropic 1.625 um voxels, which is what the public baselines do (SUBSAMPLE = [1, 4, 4]).
VOXEL_SIZE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


class SubmissionError(ValueError):
    """Raised when a submission violates the required structure."""


def node_row(dataset: str, node_id: int, t: int, z: int, y: int, x: int) -> dict:
    return {
        "dataset": dataset, "row_type": NODE, "node_id": int(node_id),
        "t": int(t), "z": int(z), "y": int(y), "x": int(x),
        "source_id": -1, "target_id": -1,
    }


def edge_row(dataset: str, source_id: int, target_id: int) -> dict:
    return {
        "dataset": dataset, "row_type": EDGE, "node_id": -1,
        "t": -1, "z": -1, "y": -1, "x": -1,
        "source_id": int(source_id), "target_id": int(target_id),
    }


def validate(df: pd.DataFrame, *, allow_synthetic: bool = False) -> dict:
    """Check the directed-forest invariants. Returns per-dataset stats, or raises.

    `allow_synthetic` permits the out-of-volume hub/fork nodes used by the metric
    augmentation (negative t and coordinates). Keep it False for the clean pipeline so real
    bugs -- negative coordinates from an off-by-one, say -- still surface. See PLAN.md s2.
    """
    missing = set(COLUMNS) - set(df.columns) - {"id"}
    if missing:
        raise SubmissionError(f"missing columns: {sorted(missing)}")

    bad_types = set(df.row_type.unique()) - {NODE, EDGE}
    if bad_types:
        raise SubmissionError(f"unexpected row_type values: {sorted(bad_types)}")

    stats = {}
    for dataset, group in df.groupby("dataset"):
        nodes = group[group.row_type == NODE]
        edges = group[group.row_type == EDGE]

        ids = nodes.node_id.astype(np.int64)
        if ids.duplicated().any():
            dupes = ids[ids.duplicated()].unique()[:5]
            raise SubmissionError(f"{dataset}: duplicate node_id, e.g. {list(dupes)}")

        targets = edges.target_id.astype(np.int64)
        if targets.duplicated().any():
            dupes = targets[targets.duplicated()].unique()[:5]
            raise SubmissionError(
                f"{dataset}: node has >1 parent (duplicated target_id), e.g. {list(dupes)} "
                "-- submission must be a directed forest"
            )

        known = set(ids)
        dangling = (set(edges.source_id.astype(np.int64)) | set(targets)) - known
        if dangling:
            raise SubmissionError(
                f"{dataset}: {len(dangling)} edge endpoint(s) reference undeclared nodes, "
                f"e.g. {sorted(dangling)[:5]}"
            )

        if not allow_synthetic and (nodes[["t", "z", "y", "x"]] < 0).any().any():
            raise SubmissionError(
                f"{dataset}: negative node coordinates with allow_synthetic=False"
            )

        n_children = edges.groupby("source_id").size() if len(edges) else pd.Series(dtype=int)
        stats[dataset] = {
            "nodes": len(nodes),
            "edges": len(edges),
            "roots": len(known - set(targets)),
            "divisions": int((n_children == 2).sum()),
            "over_divisions": int((n_children > 2).sum()),
            "frames": int(nodes.t.max()) + 1 if len(nodes) else 0,
        }

    over = {d: s["over_divisions"] for d, s in stats.items() if s["over_divisions"]}
    if over:
        raise SubmissionError(f"nodes with >2 children (not a valid division): {over}")

    return stats


def write(rows: list[dict], path: str, *, allow_synthetic: bool = False) -> pd.DataFrame:
    """Validate then write submission.csv with the required `id` index column."""
    df = pd.DataFrame(rows)
    validate(df, allow_synthetic=allow_synthetic)
    df.insert(0, "id", np.arange(len(df), dtype=np.int64))
    df[COLUMNS].to_csv(path, index=False)
    return df
