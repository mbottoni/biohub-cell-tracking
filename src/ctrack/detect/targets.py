"""Training targets for the learned detector.

Turns a GT graph into supervision on the isotropic detection grid. The central difficulty
(docs/detector.md): the GT annotates only ~6% of cells, and the labels are a representative
*sample*, so most unlabeled voxels contain real nuclei. Treating them all as background teaches
the model to suppress true detections.

So we emit two arrays per frame:

  heatmap : float32, a Gaussian-splatted target peaking at each GT centre
  loss_mask : float32 in [0, 1], where the loss is trusted

The mask is 1 near GT points (clear positives) and 1 in confidently dark regions (clear
background), and 0 in the ambiguous middle — bright voxels with no nearby GT label, which are
probably unlabeled real cells. Supervising only where we trust the label is the point.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ctrack.data.volume import ISOTROPIC_SUBSAMPLE, isotropic_scale


def gaussian_heatmap(
    centres_vox: np.ndarray,
    shape: tuple[int, int, int],
    sigma_vox: float,
) -> np.ndarray:
    """Splat unit-height isotropic gaussians at `centres_vox` (M,3) on an isotropic grid.

    Heights combine by max (overlapping cells stay ≤1), matching a peak-detection target.
    """
    hm = np.zeros(shape, dtype=np.float32)
    if len(centres_vox) == 0:
        return hm
    idx = np.round(centres_vox).astype(int)
    inb = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx = idx[inb]
    if len(idx):
        hm[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        hm = ndimage.gaussian_filter(hm, sigma=sigma_vox, mode="constant")
        if hm.max() > 0:
            hm /= hm.max()
    return hm


def build_frame_target(
    gt_centres_full_vox: np.ndarray,
    frame_iso: np.ndarray,
    *,
    sigma_um: float = 2.5,
    pos_radius_um: float = 4.0,
    bg_percentile: float = 40.0,
    subsample: tuple[int, int, int] = ISOTROPIC_SUBSAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Heatmap + trusted-loss mask for one frame.

    `gt_centres_full_vox` are (M,3) GT centres in full-resolution voxels; `frame_iso` is the
    normalized isotropic frame the detector sees. Returns (heatmap, loss_mask), both frame_iso
    shape.
    """
    spacing = isotropic_scale(subsample=subsample)  # isotropic, so scalar-ish
    iso_vox = spacing[0]
    sigma_vox = sigma_um / iso_vox

    centres_iso = gt_centres_full_vox / np.array(subsample)
    heatmap = gaussian_heatmap(centres_iso, frame_iso.shape, sigma_vox)

    # Positive region: within pos_radius of any GT centre.
    pos = np.zeros(frame_iso.shape, dtype=bool)
    idx = np.round(centres_iso).astype(int)
    inb = np.all((idx >= 0) & (idx < np.array(frame_iso.shape)), axis=1)
    for z, y, x in idx[inb]:
        pos[z, y, x] = True
    if pos.any():
        pos = ndimage.binary_dilation(pos, iterations=max(1, int(round(pos_radius_um / iso_vox))))

    # Confident background: dark voxels (at/below a low intensity percentile) that are NOT
    # positive. `<=` so the intensity floor (often the modal value after normalization) counts
    # as background rather than being excluded on a strict inequality.
    dark = frame_iso <= np.percentile(frame_iso, bg_percentile)
    confident_bg = dark & ~pos

    # Ambiguous middle (bright, unlabeled) is left unsupervised.
    loss_mask = (pos | confident_bg).astype(np.float32)
    return heatmap, loss_mask
