"""Read the imaging volumes.

Each store is a zarr v3 group with the array at `"0"`, shape `(T, Z, Y, X)` uint16, chunked
one timepoint per chunk and blosc2-compressed. zarr reads those chunks natively -- the public
starter notebook decompresses them by hand with `blosc2.decompress`, which is unnecessary.

Voxel spacing is `[1.625, 0.40625, 0.40625]` µm (Z, Y, X): Z is exactly 4x coarser. Subsampling
Y/X by 4 gives isotropic 1.625 µm voxels, which is what `to_isotropic` does and what the public
baselines mean by `SUBSAMPLE = [1, 4, 4]`.

The group metadata carries `image_statistics.quantiles` for the whole volume -- prefer those
over recomputing per frame, both for speed and so normalization is consistent across frames
despite photobleaching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

SCALE_ZYX = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
ISOTROPIC_SUBSAMPLE = (1, 4, 4)  # (Z, Y, X) -> isotropic 1.625 um


@dataclass
class Volume:
    """A `(T, Z, Y, X)` imaging volume plus its metadata."""

    path: Path
    dataset: str

    @cached_property
    def _array(self):
        import zarr

        return zarr.open(str(self.path), mode="r")["0"]

    @cached_property
    def _meta(self) -> dict:
        return json.loads((self.path / "zarr.json").read_text())

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def scale(self) -> np.ndarray:
        """(Z, Y, X) µm per voxel, read from the multiscales transform."""
        try:
            ms = self._meta["attributes"]["multiscales"][0]
            s = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
            return np.asarray(s[1:], dtype=np.float64)  # drop the T axis
        except (KeyError, IndexError):
            return SCALE_ZYX.copy()

    @cached_property
    def quantiles(self) -> dict[float, float]:
        q = self._meta["attributes"].get("image_statistics", {}).get("quantiles", {})
        return {float(k): float(v) for k, v in q.items()}

    def frame(self, t: int) -> np.ndarray:
        """One `(Z, Y, X)` uint16 timepoint."""
        return np.asarray(self._array[t])

    def normalized_frame(self, t: int, lo: float = 0.01, hi: float = 0.999) -> np.ndarray:
        """A float32 frame clipped to stored quantiles and scaled to [0, 1].

        Uses whole-volume quantiles so intensity drift across time does not silently
        renormalize each frame differently.
        """
        return self.normalize(self.frame(t), lo=lo, hi=hi)

    def normalize(self, arr: np.ndarray, lo: float = 0.01, hi: float = 0.999) -> np.ndarray:
        q = self.quantiles
        low = q.get(lo, float(np.percentile(arr, lo * 100)))
        high = q.get(hi, float(np.percentile(arr, hi * 100)))
        if high <= low:
            high = low + 1.0
        return np.clip((arr.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def to_isotropic(
    arr: np.ndarray,
    subsample: tuple[int, int, int] = ISOTROPIC_SUBSAMPLE,
    mode: str = "stride",
) -> np.ndarray:
    """Subsample a `(Z, Y, X)` array to (near-)isotropic voxels.

    mode="stride": plain striding, matching the public baselines — keeps 1/16 of the
    xy photons and aliases small nuclei.
    mode="mean": area-average over each (sz, sy, sx) block — same output grid, ~16x
    the photon count per output voxel. The A/B this docstring has promised since
    July; a model trained under one mode must be INFERRED under the same mode.
    """
    sz, sy, sx = subsample
    if mode == "stride":
        return arr[::sz, ::sy, ::sx]
    if mode == "mean":
        z, y, x = arr.shape
        zc, yc, xc = z - z % sz, y - y % sy, x - x % sx
        a = arr[:zc, :yc, :xc].astype(np.float32)
        return a.reshape(zc // sz, sz, yc // sy, sy, xc // sx, sx).mean(axis=(1, 3, 5))
    raise ValueError(f"unknown to_isotropic mode {mode!r}")


def isotropic_scale(
    scale: np.ndarray = SCALE_ZYX, subsample: tuple[int, int, int] = ISOTROPIC_SUBSAMPLE
) -> np.ndarray:
    """Voxel spacing after `to_isotropic` -- needed to convert coordinates back to µm."""
    return np.asarray(scale, dtype=np.float64) * np.asarray(subsample, dtype=np.float64)


def open_volume(path: str | Path) -> Volume:
    path = Path(path)
    return Volume(path=path, dataset=path.name.replace(".zarr", ""))


def find_volumes(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("**/*.zarr"))
