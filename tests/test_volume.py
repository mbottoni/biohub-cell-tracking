"""Tests for the volume reader, against a synthetic zarr v3 store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ctrack.data.volume import (
    ISOTROPIC_SUBSAMPLE,
    SCALE_ZYX,
    isotropic_scale,
    open_volume,
    to_isotropic,
)

zarr = pytest.importorskip("zarr")


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A (4, 8, 16, 16) store mirroring the competition layout and metadata."""
    path = tmp_path / "abcd_1234.zarr"
    root = zarr.open(str(path), mode="w")
    data = np.arange(4 * 8 * 16 * 16, dtype=np.uint16).reshape(4, 8, 16, 16) % 3000
    root.create_array("0", shape=data.shape, chunks=(1, 8, 16, 16), dtype="uint16")[:] = data

    meta = json.loads((path / "zarr.json").read_text())
    meta["attributes"] = {
        "multiscales": [{
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}]}],
        }],
        "image_statistics": {"quantiles": {"0.01": 38.0, "0.999": 2145.0}},
    }
    (path / "zarr.json").write_text(json.dumps(meta))
    return path


def test_reads_shape_scale_and_quantiles(store: Path) -> None:
    v = open_volume(store)
    assert v.dataset == "abcd_1234"
    assert v.shape == (4, 8, 16, 16)
    assert v.n_frames == 4
    np.testing.assert_allclose(v.scale, SCALE_ZYX)  # T axis dropped
    assert v.quantiles[0.01] == 38.0


def test_frame_and_normalization(store: Path) -> None:
    v = open_volume(store)
    assert v.frame(0).shape == (8, 16, 16)

    n = v.normalized_frame(0)
    assert n.dtype == np.float32
    assert 0.0 <= n.min() and n.max() <= 1.0

    # normalization uses the stored quantiles, not per-frame percentiles
    mid = v.normalize(np.array([[[38.0]]], dtype=np.uint16))
    assert mid[0, 0, 0] == pytest.approx(0.0)
    mid = v.normalize(np.array([[[2145.0]]], dtype=np.uint16))
    assert mid[0, 0, 0] == pytest.approx(1.0)


def test_scale_falls_back_when_metadata_absent(tmp_path: Path) -> None:
    path = tmp_path / "bare.zarr"
    root = zarr.open(str(path), mode="w")
    root.create_array("0", shape=(2, 4, 4, 4), chunks=(1, 4, 4, 4), dtype="uint16")
    np.testing.assert_allclose(open_volume(path).scale, SCALE_ZYX)


def test_isotropic_subsample_makes_voxels_cubic() -> None:
    arr = np.zeros((64, 256, 256), dtype=np.float32)
    iso = to_isotropic(arr)
    assert iso.shape == (64, 64, 64), "the full competition volume is 64^3 isotropic"

    spacing = isotropic_scale(SCALE_ZYX, ISOTROPIC_SUBSAMPLE)
    np.testing.assert_allclose(spacing, [1.625, 1.625, 1.625])
    np.testing.assert_allclose(np.array(iso.shape) * spacing, [104.0, 104.0, 104.0])
