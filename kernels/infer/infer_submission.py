"""Owned-pipeline submission kernel (no internet, no ILP deps).

UNet3D detector (+parent-offset head) -> NMS peaks -> multi-hypothesis candidates ->
label-free MOTION affinity (linajea) -> fast greedy divisions-aware linker -> submission.csv.

The greedy linker is pure numpy/scipy, so this kernel needs NO tracksdata/pyscipopt wheels — a
big robustness win over the ILP path (which fell back to slow SCIP without a Gurobi license).

Attach (all as Kaggle datasets — competition is no-internet):
  * the competition data
  * ctrack-src                 (the ctrack package)
  * biohub-detector-weights    (trained detector; carries out_offset if it has the motion head)
GPU on, internet OFF.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBSAMPLE = (1, 4, 4)
ISO = np.array([1.625, 1.625, 1.625])
POINT_THRESHOLD = 0.6     # tuned on golden-8; sweepable
MAX_DISP_UM = 12.0
DIV_MIN_SCORE = 0.5       # greedy: second child only on strong evidence
USE_TTA = False           # flip-TTA (8x slower); enable once base pipeline is validated


def _add_ctrack():
    import shutil
    import zipfile

    for init in Path("/kaggle/input").glob("**/ctrack/__init__.py"):
        sys.path.insert(0, str(init.parent.parent))
        return
    for marker in Path("/kaggle/input").glob("**/detect/unet3d.py"):
        pkg_root = marker.parent.parent
        if (pkg_root / "__init__.py").exists():
            dest = Path("/kaggle/working/_ctrack_pkg")
            if not (dest / "ctrack").exists():
                shutil.copytree(pkg_root, dest / "ctrack")
            sys.path.insert(0, str(dest))
            return
    for z in Path("/kaggle/input").glob("**/ctrack.zip"):
        dest = Path("/kaggle/working/_ctrack_src")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        sys.path.insert(0, str(dest))
        return
    raise FileNotFoundError("ctrack not found")


_add_ctrack()

import torch  # noqa: E402

from ctrack.detect.blob import detect_frame  # noqa: E402
from ctrack.detect.infer import predict_heatmap  # noqa: E402
from ctrack.detect.unet3d import UNet3D  # noqa: E402
from ctrack.eval.biohub import SCALE  # noqa: E402
from ctrack.link.candidates import build_nodes, build_nodes_with_offsets, prune_singletons  # noqa: E402
from ctrack.link.edge_model import motion_affinity  # noqa: E402
from ctrack.link.greedy import link_greedy  # noqa: E402
from ctrack.submit.format import COLUMNS, edge_row, node_row  # noqa: E402


def find_dir(name):
    for p in Path("/kaggle/input").glob(f"**/{name}"):
        if p.is_dir() and any(p.glob("*.zarr")):
            return p
    raise FileNotFoundError(name)


def load_detector():
    ckpt = next(Path("/kaggle/input").glob("**/unet_fold*.pt"), None)
    assert ckpt is not None, "no detector weights attached (biohub-detector-weights)"
    state = torch.load(ckpt, map_location="cpu")
    has_off = bool(state.get("out_offset", False))
    model = UNet3D(base=state.get("base", 32), out_offset=has_off)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded detector {ckpt.name}: base {state.get('base')} out_offset {has_off} "
          f"epoch {state.get('epoch')} sel {state.get('sel_score')}", flush=True)
    return model, has_off


class RawVolume:
    """zarr v3 + blosc2 reader (no zarr dependency; see baseline kernel)."""

    def __init__(self, store):
        import json
        self.store = Path(store)
        self.gmeta = json.loads((self.store / "zarr.json").read_text())
        am = json.loads((self.store / "0" / "zarr.json").read_text())
        self.shape = tuple(am["shape"])
        self.dtype = np.dtype(am["data_type"])

    def frame(self, t):
        import blosc2
        raw = blosc2.decompress((self.store / "0" / "c" / str(t) / "0" / "0" / "0").read_bytes())
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape[1:])

    def norm(self, t):
        q = self.gmeta.get("attributes", {}).get("image_statistics", {}).get("quantiles", {})
        f = self.frame(t).astype(np.float32)
        lo = float(q.get("0.01", np.percentile(f, 1)))
        hi = float(q.get("0.999", np.percentile(f, 99.9)))
        return np.clip((f - lo) / max(hi - lo, 1.0), 0, 1)


def detect(model, has_off, vol, device):
    """{t: (coords_fullvox, scores[, offsets_fullvox])} via the UNet heatmap (+offset head)."""
    out = {}
    for t in range(vol.shape[0]):
        iso = vol.norm(t)[:: SUBSAMPLE[0], :: SUBSAMPLE[1], :: SUBSAMPLE[2]]
        if has_off:
            hm, off = predict_heatmap(model, iso, device, tta=USE_TTA, return_offset=True)
        else:
            hm, off = predict_heatmap(model, iso, device, tta=USE_TTA), None
        coords, scores = detect_frame(hm, ISO, threshold=POINT_THRESHOLD)
        if off is not None and len(coords):
            ci = np.clip(np.round(coords).astype(int), 0, np.array(hm.shape) - 1)
            peak_off = off[:, ci[:, 0], ci[:, 1], ci[:, 2]].T * np.array(SUBSAMPLE)
            out[t] = (coords * np.array(SUBSAMPLE), scores, peak_off)
        elif has_off:
            out[t] = (coords * np.array(SUBSAMPLE), scores, np.zeros((len(coords), 3)))
        else:
            out[t] = (coords * np.array(SUBSAMPLE), scores)
    return out


def _safe_device(model):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        m = model.to("cuda")
        with torch.no_grad():
            m(torch.zeros(1, 1, 16, 16, 16, device="cuda"))
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as e:  # noqa: BLE001
        print("CUDA unusable, CPU fallback:", str(e)[:120], flush=True)
        return torch.device("cpu")


def main():
    model, has_off = load_detector()
    device = _safe_device(model)
    model.to(device)
    print("device:", device, flush=True)

    test_dir = find_dir("test")
    datasets = sorted(p.name.replace(".zarr", "") for p in test_dir.glob("*.zarr"))
    print("test datasets:", datasets, flush=True)

    rows = []
    for ds in datasets:
        vol = RawVolume(test_dir / f"{ds}.zarr")
        det = detect(model, has_off, vol, device)
        if has_off:
            nodes, node_scores, node_off = build_nodes_with_offsets(det)
            aff = motion_affinity(nodes, node_off, max_disp_um=MAX_DISP_UM)
        else:
            nodes, node_scores = build_nodes(det)
            aff = None
        edges = link_greedy(nodes, SCALE, max_disp_um=MAX_DISP_UM, edge_affinity=aff,
                            allow_division=True, div_min_score=DIV_MIN_SCORE)
        nodes, edges = prune_singletons(nodes, edges, node_scores, min_score=POINT_THRESHOLD)
        for nid, t, z, y, x in nodes:
            rows.append(node_row(ds, int(nid), int(t), int(z), int(y), int(x)))
        for s, d in edges:
            rows.append(edge_row(ds, int(s), int(d)))
        print(f"{ds}: {len(nodes)} nodes, {len(edges)} edges", flush=True)

    assert datasets and rows, "empty submission"
    df = pd.DataFrame(rows)
    df.insert(0, "id", np.arange(len(df), dtype=np.int64))
    df[COLUMNS].to_csv("submission.csv", index=False)
    print(f"wrote submission.csv: {len(df)} rows across {len(datasets)} datasets", flush=True)


if __name__ == "__main__":
    main()
