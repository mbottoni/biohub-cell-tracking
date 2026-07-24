"""Biohub cell tracking — classical baseline submission (kernels-only).

Self-contained: blob detection → Hungarian linking → single-frame gap closing → submission.csv.
No learning; this establishes the submission path end-to-end and lets us confirm that the local
metric (verified against the reference implementation) tracks the real leaderboard.

Mirrors ctrack.{data.volume, detect.blob, link.hungarian, submit.format} in the repo, inlined so
the kernel needs no extra dataset. Keep the two in sync when the pipeline changes.
"""

import json

import blosc2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from pathlib import Path

# The Kaggle image has neither zarr nor internet, so read the zarr v3 + blosc2 store by hand
# (as the public starter notebook does): parse zarr.json for shape/dtype, then decompress each
# per-timepoint chunk at 0/c/{t}/0/0/0. Mirrors ctrack.data.volume, which uses zarr locally.


class RawVolume:
    def __init__(self, store: Path):
        self.store = store
        self.group_meta = json.loads((store / "zarr.json").read_text())
        arr_meta = json.loads((store / "0" / "zarr.json").read_text())
        self.shape = tuple(arr_meta["shape"])            # (T, Z, Y, X)
        self.dtype = np.dtype(arr_meta["data_type"])
        self.n_t = self.shape[0]

    def frame(self, t: int) -> np.ndarray:
        chunk = self.store / "0" / "c" / str(t) / "0" / "0" / "0"
        raw = blosc2.decompress(chunk.read_bytes())
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape[1:])  # (Z, Y, X)

    @property
    def quantiles(self) -> dict:
        return self.group_meta.get("attributes", {}).get("image_statistics", {}).get("quantiles", {})

def _find_test_dir() -> Path:
    """Locate the test/ directory. Kaggle mounts competition data under
    /kaggle/input/competitions/<slug>/ (not the usual /kaggle/input/<slug>/), so search rather
    than hardcode -- the public starter uses the competitions/ path. CTRACK_TEST_DIR overrides
    for local testing."""
    import os

    if os.environ.get("CTRACK_TEST_DIR"):
        return Path(os.environ["CTRACK_TEST_DIR"])
    candidates = [
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test"),
        Path("/kaggle/input/biohub-cell-tracking-during-development/test"),
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.zarr")):
            return c
    # last resort: find any test/ dir with zarr stores anywhere under /kaggle/input
    for p in Path("/kaggle/input").glob("**/test"):
        if p.is_dir() and any(p.glob("*.zarr")):
            return p
    raise FileNotFoundError("could not locate a test/ dir with .zarr stores under /kaggle/input")


TEST_DIR = _find_test_dir()
SCALE = np.array([1.625, 0.40625, 0.40625])       # (Z, Y, X) µm/voxel
SUBSAMPLE = (1, 4, 4)                               # -> isotropic 1.625 µm
ISO_SCALE = SCALE * np.array(SUBSAMPLE)
THRESHOLD = 0.10                                    # sweep-optimal on the blob baseline
NMS_DIAMETER_UM = 3.0
SMOOTH_SIGMA_UM = 1.0
MAX_DISP_UM = 10.0

COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def normalized_frame(vol, t):
    q = vol.quantiles
    frame = vol.frame(t).astype(np.float32)
    lo = float(q.get("0.01", np.percentile(frame, 1)))
    hi = float(q.get("0.999", np.percentile(frame, 99.9)))
    return np.clip((frame - lo) / max(hi - lo, 1.0), 0.0, 1.0)


def detect_frame(frame, spacing, threshold):
    sigma = SMOOTH_SIGMA_UM / spacing
    smoothed = ndimage.gaussian_filter(frame, sigma=sigma)
    size = tuple(max(1, int(round(NMS_DIAMETER_UM / s)) | 1) for s in spacing)
    pooled = ndimage.maximum_filter(smoothed, size=size, mode="nearest")
    peaks = (smoothed == pooled) & (smoothed > threshold)
    return np.argwhere(peaks).astype(np.float64)


def link(a, b, scale, cap):
    if len(a) == 0 or len(b) == 0:
        return []
    cost = np.linalg.norm((a[:, None, :] - b[None, :, :]) * scale, axis=2)
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(r, c) if cost[i, j] <= cap]


def close_gaps(nodes, edges, scale, cap):
    """Bridge single-frame gaps with a midpoint node + two dt=1 edges (mirrors the repo).

    Only mutually-unambiguous tail/head pairs are joined; every GT edge is dt=1, so we insert a
    node rather than emit a skip edge. nodes: list of [id,t,z,y,x]; edges: list of (src,tgt).
    """
    if not nodes:
        return nodes, edges
    arr = np.array(nodes, dtype=np.float64)
    ids, t, pos = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2:5]
    index = {int(i): k for k, i in enumerate(ids)}
    has_out = {s for s, _ in edges}
    has_in = {d for _, d in edges}
    tails = [int(i) for i in ids if int(i) not in has_out]
    heads = [int(i) for i in ids if int(i) not in has_in]

    cand = {}
    for tail in tails:
        kt = index[tail]
        for head in heads:
            kh = index[head]
            if t[kh] - t[kt] != 2:
                continue
            d = float(np.linalg.norm((pos[kh] - pos[kt]) * scale))
            if d <= cap:
                cand.setdefault(tail, []).append((d, head))

    head_claims = {}
    for tail, cs in cand.items():
        if len(cs) == 1:
            head_claims.setdefault(cs[0][1], []).append(tail)

    next_id = int(ids.max()) + 1
    for head, claiming in head_claims.items():
        if len(claiming) != 1:
            continue
        tail = claiming[0]
        mid = (pos[index[tail]] + pos[index[head]]) / 2.0
        nodes.append([next_id, int(t[index[tail]] + 1), int(mid[0]), int(mid[1]), int(mid[2])])
        edges.append((tail, next_id))
        edges.append((next_id, head))
        next_id += 1
    return nodes, edges


def track_volume(dataset, rows):
    vol = RawVolume(TEST_DIR / f"{dataset}.zarr")
    n_t = vol.n_t

    per_frame_coords, ids_by_frame = {}, {}
    nodes, edges = [], []
    next_id = 1
    for t in range(n_t):
        iso = normalized_frame(vol, t)[:: SUBSAMPLE[0], :: SUBSAMPLE[1], :: SUBSAMPLE[2]]
        coords = detect_frame(iso, ISO_SCALE, THRESHOLD) * np.array(SUBSAMPLE)  # -> full-res voxels
        per_frame_coords[t] = coords
        ids = np.arange(next_id, next_id + len(coords))
        ids_by_frame[t] = ids
        next_id += len(coords)
        for nid, (z, y, x) in zip(ids, coords):
            nodes.append([int(nid), t, int(z), int(y), int(x)])

    for t in range(n_t - 1):
        for i, j in link(per_frame_coords[t], per_frame_coords[t + 1], SCALE, MAX_DISP_UM):
            edges.append((int(ids_by_frame[t][i]), int(ids_by_frame[t + 1][j])))

    nodes, edges = close_gaps(nodes, edges, SCALE, MAX_DISP_UM)
    for nid, t, z, y, x in nodes:
        rows.append([-1, dataset, "node", int(nid), int(t), int(z), int(y), int(x), -1, -1])
    for s, d in edges:
        rows.append([-1, dataset, "edge", -1, -1, -1, -1, -1, int(s), int(d)])
    print(f"{dataset}: {len(nodes)} nodes, {len(edges)} edges")


def main():
    datasets = sorted(p.name.replace(".zarr", "") for p in TEST_DIR.glob("*.zarr"))
    print("test datasets:", datasets)
    rows = []
    for d in datasets:
        track_volume(d, rows)

    # Fail loudly rather than write an empty CSV -- an empty submission passes a content
    # validator but is rejected by the scorer as "incorrect format" (this bit us on v2).
    assert datasets, "no test datasets found -- check the input mount path"
    assert rows, "no nodes/edges produced -- pipeline emitted nothing"

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["id"] = np.arange(len(df), dtype=np.int64)
    df.to_csv("submission.csv", index=False)
    print(f"wrote submission.csv: {len(df)} rows across {len(datasets)} datasets")


if __name__ == "__main__":
    main()
