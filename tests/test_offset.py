"""Tests for the parent-offset target and the detector's offset head."""

import numpy as np
import torch

from ctrack.detect.targets import build_offset_target
from ctrack.detect.unet3d import UNet3D, masked_offset_loss


def test_offset_target_paints_vector_at_cell():
    shape = (16, 16, 16)
    centres = np.array([[4.0, 20, 20]])          # full-res voxels (subsample [1,4,4] -> iso [4,5,5])
    parent_off = np.array([[2.0, 8.0, 0.0]])     # parent 2 z-voxels, 8 y-voxels away (full-res)
    off, mask = build_offset_target(centres, parent_off, shape, subsample=(1, 4, 4))
    assert off.shape == (3, *shape) and mask.shape == shape
    # somewhere in the painted region the offset equals parent_off / subsample (iso voxels)
    zc, yc, xc = 4, 5, 5
    np.testing.assert_allclose(off[:, zc, yc, xc], [2.0, 2.0, 0.0], atol=1e-5)
    assert mask[zc, yc, xc] == 1.0


def test_offset_target_skips_missing_parent():
    shape = (16, 16, 16)
    centres = np.array([[4.0, 20, 20]])
    parent_off = np.array([[np.nan, np.nan, np.nan]])   # no parent
    off, mask = build_offset_target(centres, parent_off, shape, subsample=(1, 4, 4))
    assert mask.sum() == 0 and np.all(off == 0)


def test_model_offset_head_shapes():
    m = UNet3D(base=8, out_offset=True)
    x = torch.randn(1, 1, 16, 16, 16)
    hm, off = m(x, return_offset=True)
    assert hm.shape == (1, 1, 16, 16, 16)
    assert off.shape == (1, 3, 16, 16, 16)
    # default forward stays single-tensor (backward compatible)
    assert m(x).shape == (1, 1, 16, 16, 16)


def test_model_without_offset_head_returns_none():
    m = UNet3D(base=8, out_offset=False)
    x = torch.randn(1, 1, 16, 16, 16)
    hm, off = m(x, return_offset=True)
    assert off is None


def test_masked_offset_loss_zero_when_perfect():
    pred = torch.ones(1, 3, 8, 8, 8)
    tgt = torch.ones(1, 3, 8, 8, 8)
    mask = torch.ones(1, 1, 8, 8, 8)
    assert masked_offset_loss(pred, tgt, mask).item() == 0.0
