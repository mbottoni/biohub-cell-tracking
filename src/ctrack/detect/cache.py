"""Compact, persistable training cache for the detector.

Re-downloading 87 GB of zarr every Colab session is wasteful across many training rounds. The
detector only ever needs the *annotated* isotropic frames (64³) plus each frame's GT centres —
a tiny fraction of the raw data. This packs those into one `.npz` per volume (frames as
float16), ~5 GB total for 150 volumes, which lives on Google Drive and loads in seconds.

Flow: build the cache once (needs the raw data that one time) → store on Drive → every later run
loads the cache and skips the download entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ctrack.data.geff import load_geff
from ctrack.data.volume import ISOTROPIC_SUBSAMPLE, open_volume, to_isotropic
from ctrack.detect.targets import build_frame_target


def build_volume_cache(dataset, raw_root, gt_root, cache_dir,
                       subsample=ISOTROPIC_SUBSAMPLE) -> Path:
    """Write cache_dir/{dataset}.npz with the annotated frames + centres. Returns the path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{dataset}.npz"
    if out.exists():
        return out

    gt = load_geff(Path(gt_root) / f"{dataset}.geff")
    vol = open_volume(Path(raw_root) / f"{dataset}.zarr")
    t_all = gt.graph.t.astype(int)
    zyx = gt.graph.zyx

    frames_t = sorted(set(t_all.tolist()))
    frames, offsets, centres = [], [0], []
    for t in frames_t:
        frames.append(to_isotropic(vol.normalized_frame(t), subsample).astype(np.float16))
        c = zyx[t_all == t]
        centres.append(c)
        offsets.append(offsets[-1] + len(c))

    np.savez_compressed(
        out,
        frames=np.stack(frames),                       # (N, D, H, W) float16
        t=np.array(frames_t, dtype=np.int32),          # (N,)
        centres=np.concatenate(centres).astype(np.float32) if centres else np.zeros((0, 3), np.float32),
        offsets=np.array(offsets, dtype=np.int64),     # (N+1,) into centres
        n_total=np.float64(gt.estimated_number_of_nodes),
    )
    return out


def ensure_cache(datasets, raw_root, gt_root, cache_dir, subsample=ISOTROPIC_SUBSAMPLE,
                 all_ids=None, log=print) -> None:
    """Build any missing per-volume caches. Idempotent — safe to call every run.

    Writes `_all_ids.json` (the full CV id universe) so splits reproduce identically across
    runs no matter which volumes happen to be cached — pass `all_ids` (defaults to `datasets`).
    """
    import json

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_file = cache_dir / "_all_ids.json"
    if all_ids is not None and not ids_file.exists():
        ids_file.write_text(json.dumps(sorted(all_ids)))

    missing = [d for d in datasets if not (cache_dir / f"{d}.npz").exists()]
    log(f"cache: {len(datasets) - len(missing)}/{len(datasets)} present, building {len(missing)}")
    for i, d in enumerate(missing, 1):
        build_volume_cache(d, raw_root, gt_root, cache_dir, subsample)
        if i % 20 == 0:
            log(f"  built {i}/{len(missing)}")


def cached_all_ids(cache_dir):
    """The full CV id universe stored alongside the cache, or None."""
    import json

    p = Path(cache_dir) / "_all_ids.json"
    return json.loads(p.read_text()) if p.exists() else None


class CachedDetectionFrames(Dataset):
    """Like DetectionFrames but reads the compact .npz cache instead of zarr.

    Yields (frame, heatmap, mask); heatmap+mask are built on the fly (cheap) from the cached
    frame and centres, so cache stays small and target params can still be tuned per run.
    """

    def __init__(self, datasets, cache_dir, *, subsample=ISOTROPIC_SUBSAMPLE, sigma_um=2.5,
                 augment=False, frame_stride=1):
        self.cache_dir = Path(cache_dir)
        self.subsample = subsample
        self.sigma_um = sigma_um
        self.augment = augment
        self._blobs: dict[str, dict] = {}
        self.index: list[tuple[str, int]] = []   # (dataset, t) — same interface as DetectionFrames
        self._row: list[int] = []                # parallel: row into that volume's frames array
        self._centres: dict[tuple[str, int], np.ndarray] = {}

        for ds in datasets:
            z = np.load(self.cache_dir / f"{ds}.npz")
            self._blobs[ds] = {"frames": z["frames"], "t": z["t"]}
            t, centres, offs = z["t"], z["centres"], z["offsets"]
            for k in range(0, len(t), frame_stride):
                self._centres[(ds, int(t[k]))] = centres[offs[k]:offs[k + 1]]
                self.index.append((ds, int(t[k])))
                self._row.append(k)

    def __len__(self):
        return len(self.index)

    def frame_only(self, i):
        ds, _ = self.index[i]
        return self._blobs[ds]["frames"][self._row[i]].astype(np.float32)

    def __getitem__(self, i):
        import torch

        ds, t = self.index[i]
        frame = self._blobs[ds]["frames"][self._row[i]].astype(np.float32)
        centres = self._centres[(ds, t)]
        heatmap, mask = build_frame_target(centres, frame, sigma_um=self.sigma_um,
                                           subsample=self.subsample)
        if self.augment:
            rng = np.random.default_rng()
            for ax in range(3):
                if rng.random() < 0.5:
                    frame, heatmap, mask = (np.flip(a, ax) for a in (frame, heatmap, mask))
        x = torch.from_numpy(np.ascontiguousarray(frame))[None].float()
        y = torch.from_numpy(np.ascontiguousarray(heatmap))[None].float()
        m = torch.from_numpy(np.ascontiguousarray(mask))[None].float()
        return x, y, m
