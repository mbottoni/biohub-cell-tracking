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
from ctrack.detect.targets import build_frame_target, build_offset_target


class DetectionFrames(Dataset):
    """(frame, heatmap, mask) triples over annotated timepoints of the given datasets.

    With `with_offset=True`, each item is instead (frame, heatmap, mask, offset, offset_mask):
    the extra tensors supervise the detector's parent-offset head (motion-aware linking).
    """

    def __init__(
        self,
        datasets: list[str],
        raw_root: str | Path,
        gt_root: str | Path,
        *,
        subsample: tuple[int, int, int] = ISOTROPIC_SUBSAMPLE,
        sigma_um: float = 2.5,
        augment: bool = False,
        frame_stride: int = 1,
        with_offset: bool = False,
        pseudo_dir: str | Path | None = None,
        pseudo_weight: float = 0.3,
        iso_mode: str = "stride",
    ):
        self.raw_root = Path(raw_root)
        self.gt_root = Path(gt_root)
        self.subsample = subsample
        self.sigma_um = sigma_um
        self.augment = augment
        self.with_offset = with_offset
        self.pseudo_weight = pseudo_weight
        self.iso_mode = iso_mode
        # Teacher detections (dump_teacher_peaks.py): {ds}.npz with coords (N,4) [t,z,y,x] in
        # full-resolution voxels. Merged into build_frame_target as reduced-weight positives.
        self._pseudo: dict[tuple[str, int], np.ndarray] = {}
        if pseudo_dir is not None:
            n_pts = 0
            for ds in datasets:
                p = Path(pseudo_dir) / f"{ds}.npz"
                if not p.exists():
                    continue
                coords = np.load(p)["coords"].astype(np.float64)
                for frame in np.unique(coords[:, 0]).astype(int):
                    pts = coords[coords[:, 0] == frame, 1:4]
                    self._pseudo[(ds, int(frame))] = pts
                    n_pts += len(pts)
            print(f"pseudo-labels: {n_pts:,} teacher points over "
                  f"{len(self._pseudo):,} frames (weight {pseudo_weight})", flush=True)
        # Volumes have up to ~100 annotated frames; adjacent frames are near-identical, so a
        # stride cuts epoch time with little information loss. Keep 1 for validation.
        self._vol_cache: dict[str, object] = {}

        # index (dataset, t) for annotated frames, and cache GT centres per (dataset, t)
        self.index: list[tuple[str, int]] = []
        self._centres: dict[tuple[str, int], np.ndarray] = {}
        self._parent_offsets: dict[tuple[str, int], np.ndarray] = {}
        for ds in datasets:
            gt = load_geff(self.gt_root / f"{ds}.geff")
            t = gt.graph.t.astype(int)
            zyx = gt.graph.zyx
            ids = gt.graph.ids.astype(np.int64)
            id_to_zyx = {int(i): zyx[k] for k, i in enumerate(ids)}
            id_to_t = {int(i): int(t[k]) for k, i in enumerate(ids)}
            # parent (source) of each target node, from the GT edges
            parent_of = {int(tg): int(sc) for sc, tg in gt.graph.edges.astype(np.int64)} \
                if len(gt.graph.edges) else {}
            frames = np.unique(t)[::frame_stride]
            # Pseudo-labels make GT-empty frames informative too (the teacher covers every
            # frame); without them a frame with no labels is all-ambiguous and stays excluded.
            pseudo_frames = sorted({f for (d, f) in self._pseudo if d == ds})
            if pseudo_frames:
                extra = [f for f in pseudo_frames[::frame_stride] if f not in set(frames)]
                frames = np.concatenate([frames, np.array(extra, dtype=frames.dtype)]) \
                    if extra else frames
            for frame in frames:
                sel = t == frame
                self._centres[(ds, int(frame))] = zyx[sel]
                if with_offset:
                    frame_ids = ids[sel]
                    offs = np.full((len(frame_ids), 3), np.nan)
                    for k, nid in enumerate(frame_ids):
                        p = parent_of.get(int(nid))
                        # parent must exist and be exactly one frame earlier
                        if p is not None and id_to_t.get(p) == int(frame) - 1:
                            offs[k] = id_to_zyx[p] - id_to_zyx[int(nid)]
                    self._parent_offsets[(ds, int(frame))] = offs
                self.index.append((ds, int(frame)))

    def __len__(self) -> int:
        return len(self.index)

    def _volume(self, ds: str):
        vol = self._vol_cache.get(ds)
        if vol is None:
            vol = open_volume(self.raw_root / f"{ds}.zarr")
            self._vol_cache[ds] = vol
        return vol

    def frame_only(self, i: int) -> np.ndarray:
        """The isotropic frame for item i, without building the (expensive) target/mask.

        Validation only needs the input frame — skip the Gaussian splat + dilation that
        __getitem__ does for training targets.
        """
        ds, t = self.index[i]
        return to_isotropic(self._volume(ds).normalized_frame(t), self.subsample, mode=self.iso_mode)

    def __getitem__(self, i: int):
        import torch

        ds, t = self.index[i]
        frame_iso = to_isotropic(self._volume(ds).normalized_frame(t), self.subsample, mode=self.iso_mode)
        centres = self._centres[(ds, t)]

        heatmap, mask = build_frame_target(
            centres, frame_iso, sigma_um=self.sigma_um, subsample=self.subsample,
            pseudo_centres_full_vox=self._pseudo.get((ds, t)),
            pseudo_weight=self.pseudo_weight,
        )

        if not self.with_offset:
            if self.augment:
                frame_iso, heatmap, mask = _augment(frame_iso, heatmap, mask)
            x = torch.from_numpy(np.ascontiguousarray(frame_iso))[None].float()
            y = torch.from_numpy(np.ascontiguousarray(heatmap))[None].float()
            m = torch.from_numpy(np.ascontiguousarray(mask))[None].float()
            return x, y, m

        offset, offset_mask = build_offset_target(
            centres, self._parent_offsets[(ds, t)], frame_iso.shape, subsample=self.subsample
        )
        if self.augment:
            frame_iso, heatmap, mask, offset, offset_mask = _augment(
                frame_iso, heatmap, mask, offset, offset_mask
            )
        x = torch.from_numpy(np.ascontiguousarray(frame_iso))[None].float()
        y = torch.from_numpy(np.ascontiguousarray(heatmap))[None].float()
        m = torch.from_numpy(np.ascontiguousarray(mask))[None].float()
        off = torch.from_numpy(np.ascontiguousarray(offset)).float()          # (3,D,H,W)
        offm = torch.from_numpy(np.ascontiguousarray(offset_mask))[None].float()
        return x, y, m, off, offm


def _augment(frame, heatmap, mask, offset=None, offset_mask=None):
    """Random axis flips. Isotropic grid, so flips are label-preserving and cheap. When an offset
    field is present, flipping spatial axis `ax` also negates the offset's `ax` component."""
    rng = np.random.default_rng()
    for ax in range(3):
        if rng.random() < 0.5:
            frame = np.flip(frame, ax)
            heatmap = np.flip(heatmap, ax)
            mask = np.flip(mask, ax)
            if offset is not None:
                offset = np.flip(offset, ax + 1)         # spatial axis is ax+1 in (3,D,H,W)
                offset[ax] = -offset[ax]
                offset_mask = np.flip(offset_mask, ax)
    if offset is not None:
        return frame, heatmap, mask, offset, offset_mask
    return frame, heatmap, mask
