"""Tests for detector training-target generation."""

from __future__ import annotations

import numpy as np

from ctrack.detect.targets import build_frame_target, gaussian_heatmap


def test_heatmap_peaks_at_centres() -> None:
    hm = gaussian_heatmap(np.array([[5.0, 8.0, 8.0], [5.0, 24.0, 24.0]]), (16, 32, 32), sigma_vox=1.5)
    assert hm.max() <= 1.0 + 1e-6
    # peaks sit at the requested centres
    assert hm[5, 8, 8] > 0.9
    assert hm[5, 24, 24] > 0.9
    # midpoint between well-separated cells is near zero
    assert hm[5, 16, 16] < 0.1


def test_heatmap_empty_is_zero() -> None:
    hm = gaussian_heatmap(np.zeros((0, 3)), (8, 8, 8), sigma_vox=1.0)
    assert hm.shape == (8, 8, 8) and hm.max() == 0.0


def test_mask_trusts_positives_and_dark_background_only() -> None:
    """The core of the sparse-label handling: a bright unlabeled region must be left
    unsupervised (mask 0), so the detector isn't punished for finding an unlabeled cell."""
    frame = np.zeros((8, 32, 32), dtype=np.float32)
    frame[4, 8, 8] = 1.0                       # a labeled cell
    frame[4, 24, 24] = 1.0                     # a BRIGHT but UNLABELED cell

    gt_full_vox = np.array([[4.0, 8.0 * 4, 8.0 * 4]])  # only the first, in full-res voxels
    _, mask = build_frame_target(gt_full_vox, frame, pos_radius_um=4.0, bg_percentile=40.0)

    assert mask[4, 8, 8] == 1.0, "labeled cell must be supervised"
    assert mask[4, 24, 24] == 0.0, "bright unlabeled cell must NOT be supervised"
    # a dark corner far from everything is confident background -> supervised
    assert mask[0, 0, 0] == 1.0


def test_mask_and_heatmap_share_shape() -> None:
    frame = np.random.default_rng(0).random((8, 32, 32)).astype(np.float32)
    gt = np.array([[4.0, 40.0, 40.0]])
    hm, mask = build_frame_target(gt, frame)
    assert hm.shape == mask.shape == frame.shape
