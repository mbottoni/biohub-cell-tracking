"""Classical nucleus detection: smooth, then non-maximum suppression on local peaks.

This is the baseline detector -- no learning. Its job is to get a full pipeline running end to
end so the metric harness and submission path are exercised before any model exists, and to
give the learned detector something to beat.

Design note. The metric pays for *selective* detection twice over (docs/metric.md):

  * edges between two unannotated cells are invisible, so extra detections cost nothing
    directly -- but every extra node raises N_pred, and
  * the node-budget multiplier `1 - 0.1 * (N_pred - N_total)/N_total` is unclamped above, so
    coming in under `estimated_number_of_nodes` multiplies the score up.

So `threshold` is the single most important knob here and is meant to be swept against the
real metric, not against detection F1. `detect_frame` returns peak intensities alongside
coordinates precisely so a sweep can re-threshold without re-running the filter.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# A zebrafish nucleus is roughly 5-7 um across; the public baselines suppress within 3 um.
DEFAULT_NMS_DIAMETER_UM = 3.0
DEFAULT_SMOOTH_SIGMA_UM = 1.0


def _kernel_size(um: float, spacing: np.ndarray) -> tuple[int, ...]:
    """Odd voxel kernel spanning `um` micrometres along each axis (min 1)."""
    return tuple(max(1, int(round(um / s)) | 1) for s in spacing)


def detect_frame(
    frame: np.ndarray,
    spacing: np.ndarray,
    *,
    threshold: float = 0.5,
    nms_diameter_um: float = DEFAULT_NMS_DIAMETER_UM,
    smooth_sigma_um: float = DEFAULT_SMOOTH_SIGMA_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect nuclei in one normalized `(Z, Y, X)` frame.

    Returns `(coords, scores)` where coords is (N, 3) float voxel positions in the frame's own
    grid and scores is the smoothed intensity at each peak.
    """
    if smooth_sigma_um > 0:
        sigma = np.asarray(smooth_sigma_um, dtype=np.float64) / np.asarray(spacing, dtype=np.float64)
        smoothed = ndimage.gaussian_filter(frame.astype(np.float32), sigma=sigma)
    else:
        smoothed = frame.astype(np.float32)

    size = _kernel_size(nms_diameter_um, np.asarray(spacing, dtype=np.float64))
    pooled = ndimage.maximum_filter(smoothed, size=size, mode="nearest")

    peaks = (smoothed == pooled) & (smoothed > threshold)
    coords = np.argwhere(peaks).astype(np.float64)
    if coords.size == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    scores = smoothed[peaks]
    return coords, scores


def rethreshold(
    detections: dict[int, tuple[np.ndarray, np.ndarray]], threshold: float
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Filter an existing detection set to a higher threshold.

    NMS peaks are threshold-independent -- raising the threshold only removes peaks, never
    moves or adds them. So a sweep detects once at the lowest threshold and re-filters, instead
    of re-reading and re-filtering every frame per threshold. On an N-point sweep that is an
    N-fold saving, and volume I/O dominates here.
    """
    return {
        t: (coords[scores >= threshold], scores[scores >= threshold])
        for t, (coords, scores) in detections.items()
    }


def detect_volume(
    volume,
    *,
    threshold: float = 0.5,
    subsample: tuple[int, int, int] = (1, 4, 4),
    frames: range | None = None,
    **kwargs,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Detect across all timepoints. Coordinates are returned in FULL-resolution voxels.

    The detector runs on the isotropic subsampled grid (64^3 for this dataset -- the whole
    volume, no tiling), then coordinates are scaled back so everything downstream, including
    the submission, speaks full-resolution voxels.
    """
    from ctrack.data.volume import isotropic_scale, to_isotropic

    spacing = isotropic_scale(volume.scale, subsample)
    sub = np.asarray(subsample, dtype=np.float64)

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for t in (frames if frames is not None else range(volume.n_frames)):
        iso = to_isotropic(volume.normalized_frame(t), subsample)
        coords, scores = detect_frame(iso, spacing, threshold=threshold, **kwargs)
        out[t] = (coords * sub, scores)
    return out
