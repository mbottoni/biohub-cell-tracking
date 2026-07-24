"""Learned edge-affinity model for linking.

Replaces the ILP's distance cost with a learned probability that two detections in consecutive
frames are the same cell. Feeding −logit(p) into the ILP (ctrack.link.ilp) is where the real
linking gains are — distance alone can't tell a real 8 µm step from a coincidental near-miss.

Deliberately lightweight and feature-based rather than the winners' UNet-feature transformer:
it trains in seconds on GT trajectories and needs no detector features, so it is a strong,
fast first learned linker. Features per candidate pair (source at t, target at t+1):

  physical Δz, Δy, Δx ; distance ; source & target heatmap/intensity ; |Δintensity|

Trained on GT graphs: a node's true successor is positive, its other in-radius neighbours in
the next frame are negatives. That teaches "which of the nearby candidates is the real one",
which is exactly the ambiguity the ILP must resolve.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ctrack.eval.biohub import SCALE

FEATURE_DIM = 8


def pair_features(
    src_zyx: np.ndarray, tgt_zyx: np.ndarray, src_val: np.ndarray, tgt_val: np.ndarray,
    scale: np.ndarray = SCALE,
) -> np.ndarray:
    """(M, FEATURE_DIM) features for M candidate pairs. Positions in voxels, values in [0,1]."""
    d = (tgt_zyx - src_zyx) * scale
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    return np.concatenate(
        [d, dist, src_val[:, None], tgt_val[:, None], np.abs(tgt_val - src_val)[:, None]],
        axis=1,
    ).astype(np.float32)


class EdgeMLP(nn.Module):
    """Small MLP: pair features -> edge logit."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # normalization buffers, filled from training data
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std).squeeze(-1)

    @torch.no_grad()
    def probs(self, feats: np.ndarray) -> np.ndarray:
        self.eval()
        x = torch.from_numpy(np.ascontiguousarray(feats)).float()
        return torch.sigmoid(self(x)).cpu().numpy()


def affinity_map(
    model: EdgeMLP,
    nodes: np.ndarray,
    values: dict[int, float],
    *,
    max_disp_um: float,
    topk: int = 5,
    scale: np.ndarray = SCALE,
) -> dict[tuple[int, int], float]:
    """Score all candidate consecutive-frame pairs -> {(src_id, tgt_id): prob}.

    `values` maps node_id -> its heatmap/intensity value (detector confidence). Only pairs
    within `max_disp_um` are scored; the rest are implicitly near-zero to the ILP.
    """
    ids = nodes[:, 0].astype(int)
    t = nodes[:, 1].astype(int)
    zyx = nodes[:, 2:5]
    out: dict[tuple[int, int], float] = {}
    for f in np.unique(t):
        cur = np.flatnonzero(t == f)
        nxt = np.flatnonzero(t == f + 1)
        if cur.size == 0 or nxt.size == 0:
            continue
        d = np.linalg.norm((zyx[cur][:, None, :] - zyx[nxt][None, :, :]) * scale, axis=2)
        for i, ci in enumerate(cur):
            order = np.argsort(d[i])[:topk]
            keep = [j for j in order if d[i, j] <= max_disp_um]
            if not keep:
                continue
            keep = np.array(keep)
            src_ids = np.full(len(keep), ids[ci])
            tgt_ids = ids[nxt[keep]]
            feats = pair_features(
                np.repeat(zyx[ci][None], len(keep), axis=0), zyx[nxt[keep]],
                np.array([values.get(int(ids[ci]), 1.0)] * len(keep)),
                np.array([values.get(int(t_), 1.0) for t_ in tgt_ids]),
                scale,
            )
            for p, s_, t_ in zip(model.probs(feats), src_ids, tgt_ids):
                out[(int(s_), int(t_))] = float(p)
    return out
