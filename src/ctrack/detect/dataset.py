"""Torch dataset for detector training.

One item = one (volume, timepoint): the isotropic frame the detector sees, plus the heatmap
target and trusted-loss mask (ctrack.detect.targets). Only frames with >=1 GT annotation are
used — a frame with no labels tells us nothing about where cells are (the whole frame would be
'ambiguous'), so it contributes no usable signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ctrack.data.geff import load_geff
from ctrack.data.volume import ISOTROPIC_SUBSAMPLE, open_volume, to_isotropic
from ctrack.detect.targets import build_frame_target


class DetectionFrames(Dataset):
    """(frame, heatmap, mask) triples over annotated timepoints of the given datasets."""

    def __init__(
        self,
        datasets: list[str],
        raw_root: str | Path,
        gt_root: str | Path,
        *,
        subsample: tuple[int, int, int] = ISOTROPIC_SUBSAMPLE,
        sigma_um: float = 2.5,
        augment: bool = False,
    ):
        self.raw_root = Path(raw_root)
        self.gt_root = Path(gt_root)
        self.subsample = subsample
        self.sigma_um = sigma_um
        self.augment = augment

        # index (dataset, t) for every annotated frame, and cache GT centres per (dataset, t)
        self.index: list[tuple[str, int]] = []
        self._centres: dict[tuple[str, int], np.ndarray] = {}
        for ds in datasets:
            gt = load_geff(self.gt_root / f"{ds}.geff")
            t = gt.graph.t.astype(int)
            zyx = gt.graph.zyx
            for frame in np.unique(t):
                self._centres[(ds, int(frame))] = zyx[t == frame]
                self.index.append((ds, int(frame)))

    def __len__(self) -> int:
        return len(self.index)

    def frame_only(self, i: int) -> np.ndarray:
        """The isotropic frame for item i, without building the (expensive) target/mask.

        Validation only needs the input frame — skip the Gaussian splat + dilation that
        __getitem__ does for training targets.
        """
        ds, t = self.index[i]
        vol = open_volume(self.raw_root / f"{ds}.zarr")
        return to_isotropic(vol.normalized_frame(t), self.subsample)

    def __getitem__(self, i: int):
        import torch

        ds, t = self.index[i]
        vol = open_volume(self.raw_root / f"{ds}.zarr")
        frame_iso = to_isotropic(vol.normalized_frame(t), self.subsample)
        centres = self._centres[(ds, t)]

        heatmap, mask = build_frame_target(
            centres, frame_iso, sigma_um=self.sigma_um, subsample=self.subsample
        )

        if self.augment:
            frame_iso, heatmap, mask = _augment(frame_iso, heatmap, mask)

        x = torch.from_numpy(np.ascontiguousarray(frame_iso))[None].float()
        y = torch.from_numpy(np.ascontiguousarray(heatmap))[None].float()
        m = torch.from_numpy(np.ascontiguousarray(mask))[None].float()
        return x, y, m


def _augment(frame: np.ndarray, heatmap: np.ndarray, mask: np.ndarray):
    """Random axis flips. Isotropic grid, so flips are label-preserving and cheap."""
    rng = np.random.default_rng()
    for ax in range(3):
        if rng.random() < 0.5:
            frame = np.flip(frame, ax)
            heatmap = np.flip(heatmap, ax)
            mask = np.flip(mask, ax)
    return frame, heatmap, mask
