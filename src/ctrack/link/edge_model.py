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

FEATURE_DIM = 7      # Δz, Δy, Δx, distance, src intensity, tgt intensity, |Δintensity|
FEATURE_DIM_MOTION = 11  # + predicted-offset agreement: residual (3) and its norm (1)


def pair_features(
    src_zyx: np.ndarray, tgt_zyx: np.ndarray, src_val: np.ndarray, tgt_val: np.ndarray,
    scale: np.ndarray = SCALE,
    tgt_offset: np.ndarray | None = None,
) -> np.ndarray:
    """(M, FEATURE_DIM[+4]) features for M candidate pairs. Positions in voxels, values in [0,1].

    If `tgt_offset` (M,3, in the SAME voxel frame as positions) is given — the detector's predicted
    offset-to-parent at the target — we add motion-agreement features: the residual between where
    the target *thinks* its parent is (tgt_zyx + tgt_offset) and the actual source position, plus
    its norm. A true parent makes this residual ~0; this is the linajea motion cue the linker lacks.
    """
    d = (tgt_zyx - src_zyx) * scale
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    feats = [d, dist, src_val[:, None], tgt_val[:, None], np.abs(tgt_val - src_val)[:, None]]
    if tgt_offset is not None:
        predicted_parent = tgt_zyx + tgt_offset
        residual = (predicted_parent - src_zyx) * scale        # ~0 if target's parent is src
        res_norm = np.linalg.norm(residual, axis=1, keepdims=True)
        feats += [residual, res_norm]
    return np.concatenate(feats, axis=1).astype(np.float32)


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


def build_training_pairs(
    graph, scale: np.ndarray = SCALE, *, max_disp_um: float = 12.0, topk: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """From a GT graph, build (features, labels) for the edge model.

    Positive = a true parent->child edge. Negatives = the parent's OTHER in-radius candidates in
    the next frame (the near-misses the linker must reject). Teaches "which nearby candidate is
    the real successor" — exactly the ambiguity distance alone can't resolve. Intensities are
    unknown for GT nodes, so they are set to 1 (geometry-only model; motion features add the rest).
    """
    ids = graph.ids.astype(np.int64)
    t = graph.t.astype(int)
    zyx = graph.zyx
    idx_of = {int(i): k for k, i in enumerate(ids)}
    true_child = {}
    for s, d in graph.edges.astype(np.int64):
        true_child.setdefault(int(s), set()).add(int(d))

    feats, labels = [], []
    for f in np.unique(t):
        cur = np.flatnonzero(t == f)
        nxt = np.flatnonzero(t == f + 1)
        if cur.size == 0 or nxt.size == 0:
            continue
        dmat = np.linalg.norm((zyx[cur][:, None, :] - zyx[nxt][None, :, :]) * scale, axis=2)
        for i, ci in enumerate(cur):
            order = [j for j in np.argsort(dmat[i])[:topk] if dmat[i, j] <= max_disp_um]
            if not order:
                continue
            children = true_child.get(int(ids[ci]), set())
            for j in order:
                lbl = 1.0 if int(ids[nxt[j]]) in children else 0.0
                feats.append(pair_features(zyx[ci][None], zyx[nxt[j]][None],
                                           np.array([1.0]), np.array([1.0]), scale)[0])
                labels.append(lbl)
    if not feats:
        return np.zeros((0, FEATURE_DIM), np.float32), np.zeros((0,), np.float32)
    return np.stack(feats), np.array(labels, np.float32)


def train_edge_model(
    graphs: list, scale: np.ndarray = SCALE, *, epochs: int = 200, lr: float = 1e-3,
    hidden: int = 64, max_disp_um: float = 12.0, topk: int = 5, verbose: bool = True,
) -> "EdgeMLP":
    """Fit an EdgeMLP on GT graphs (trains in seconds). Returns the model with normalization set."""
    X, y = [], []
    for g in graphs:
        fx, fy = build_training_pairs(g, scale, max_disp_um=max_disp_um, topk=topk)
        if len(fx):
            X.append(fx)
            y.append(fy)
    X = np.concatenate(X) if X else np.zeros((0, FEATURE_DIM), np.float32)
    y = np.concatenate(y) if y else np.zeros((0,), np.float32)
    if len(X) == 0:
        raise ValueError("no training pairs — check GT graphs have edges")

    model = EdgeMLP(in_dim=X.shape[1], hidden=hidden)
    model.mean.copy_(torch.from_numpy(X.mean(0)))
    model.std.copy_(torch.from_numpy(X.std(0) + 1e-6))
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float()
    pos_frac = float(y.mean()) or 0.5
    pw = torch.tensor((1 - pos_frac) / max(pos_frac, 1e-3))  # balance rare positives
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt), yt, pos_weight=pw)
        loss.backward()
        opt.step()
        if verbose and (e == 0 or (e + 1) % 50 == 0):
            with torch.no_grad():
                acc = ((torch.sigmoid(model(Xt)) > 0.5).float() == yt).float().mean().item()
            print(f"  edge-model epoch {e+1}/{epochs} loss {loss.item():.4f} acc {acc:.3f} "
                  f"(n={len(X)}, pos={pos_frac:.3f})", flush=True)
    return model


def motion_affinity(
    nodes: np.ndarray,
    offsets: dict[int, np.ndarray],
    *,
    max_disp_um: float = 12.0,
    topk: int = 5,
    sigma_um: float = 3.0,
    scale: np.ndarray = SCALE,
) -> dict[tuple[int, int], float]:
    """Label-free edge affinity from the detector's predicted parent-offset (linajea edge cost).

    `offsets` maps node_id -> the detector's predicted offset-to-parent at that node (same voxel
    frame as node positions). For a candidate pair (src@t, tgt@t+1), the target's predicted parent
    position is tgt + offset[tgt]; if that lands on src, the edge is strongly favoured. Affinity =
    exp(-(residual_um/sigma_um)^2), so it needs no training — the motion signal comes straight from
    the movement-vector head. Falls back to a distance prior for nodes without a predicted offset.
    """
    ids = nodes[:, 0].astype(int)
    t = nodes[:, 1].astype(int)
    zyx = nodes[:, 2:5]
    id_to_row = {int(i): k for k, i in enumerate(ids)}
    out: dict[tuple[int, int], float] = {}
    for f in np.unique(t):
        cur = np.flatnonzero(t == f)
        nxt = np.flatnonzero(t == f + 1)
        if cur.size == 0 or nxt.size == 0:
            continue
        d = np.linalg.norm((zyx[cur][:, None, :] - zyx[nxt][None, :, :]) * scale, axis=2)
        for i, ci in enumerate(cur):
            for j in np.argsort(d[i])[:topk]:
                if d[i, j] > max_disp_um:
                    continue
                s_id, tg_id = int(ids[ci]), int(ids[nxt[j]])
                off = offsets.get(tg_id)
                if off is not None:
                    predicted_parent = zyx[nxt[j]] + off
                    residual = float(np.linalg.norm((predicted_parent - zyx[ci]) * scale))
                    aff = float(np.exp(-((residual / sigma_um) ** 2)))
                else:
                    aff = float(np.exp(-((d[i, j] / max_disp_um) ** 2)))  # distance prior
                out[(s_id, tg_id)] = aff
    return out


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
