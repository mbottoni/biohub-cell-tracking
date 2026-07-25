"""Tests for TTA/ensemble inference and edge-model training pairs."""

import numpy as np
import torch

from ctrack.detect.infer import predict_heatmap, predict_heatmap_ensemble
from ctrack.detect.unet3d import UNet3D
from ctrack.link.edge_model import FEATURE_DIM_MOTION, pair_features


class _ConstModel(torch.nn.Module):
    """Returns a fixed heatmap regardless of input, so TTA-averaging is identity-checkable."""

    def __init__(self, out_offset=False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.offset_head = True if out_offset else None

    def forward(self, x, return_offset=False):
        d = x.shape[-3:]
        hm = torch.full((1, 1, *d), -2.0)  # constant logits
        if return_offset:
            off = torch.zeros(1, 3, *d) if self.offset_head else None
            return hm, off
        return hm


def test_tta_matches_identity_for_constant_model():
    m = _ConstModel()
    frame = np.random.rand(8, 8, 8).astype(np.float32)
    plain = predict_heatmap(m, frame, torch.device("cpu"), tta=False)
    tta = predict_heatmap(m, frame, torch.device("cpu"), tta=True)
    np.testing.assert_allclose(plain, tta, atol=1e-5)  # constant field: TTA changes nothing


def test_predict_heatmap_shape_real_model():
    m = UNet3D(base=8).eval()
    frame = np.random.rand(16, 16, 16).astype(np.float32)
    hm = predict_heatmap(m, frame, torch.device("cpu"), tta=False)
    assert hm.shape == (16, 16, 16) and hm.min() >= 0 and hm.max() <= 1


def test_ensemble_averages():
    frame = np.random.rand(8, 8, 8).astype(np.float32)
    hm = predict_heatmap_ensemble([_ConstModel(), _ConstModel()], frame, torch.device("cpu"))
    np.testing.assert_allclose(hm, 1 / (1 + np.exp(2.0)), atol=1e-5)  # sigmoid(-2)


def test_motion_feature_dim():
    f = pair_features(np.array([[10.0, 10, 10]]), np.array([[10.0, 10, 12]]),
                      np.array([1.0]), np.array([1.0]), tgt_offset=np.array([[0.0, 0, -2]]))
    assert f.shape[1] == FEATURE_DIM_MOTION
