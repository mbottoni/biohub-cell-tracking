"""Learned-pipeline submission kernel (no internet).

UNet3D detector -> NMS peaks -> learned edge affinities -> ILP -> submission.csv.

Attach (all as Kaggle datasets, since the competition is no-internet):
  * the competition data
  * ctrack-src            (the ctrack package)
  * biohub-offline-wheels (tracksdata/geff/pyscipopt + deps, cp312)
  * the trained detector weights (output of the GPU training kernel)
  * the trained edge-model weights

GPU on, internet OFF. Falls back gracefully: if ILP deps fail to install, links with the
pure-scipy Hungarian path so a submission is still produced.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- offline deps (no internet): install tracksdata + friends from bundled wheels ----
WHEELS = next(Path("/kaggle/input").glob("**/wheels"), None)
HAVE_ILP = False
if WHEELS is not None:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={WHEELS}",
         "tracksdata", "geff", "pyscipopt"],
        capture_output=True, text=True,
    )
    HAVE_ILP = r.returncode == 0
    print("ILP deps installed:", HAVE_ILP)
    if not HAVE_ILP:
        print(r.stderr[-800:])


def _add_ctrack():
    for init in Path("/kaggle/input").glob("**/ctrack/__init__.py"):
        sys.path.insert(0, str(init.parent.parent))
        return
    import zipfile
    for z in Path("/kaggle/input").glob("**/ctrack.zip"):
        dest = Path("/kaggle/working/_ctrack_src")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        sys.path.insert(0, str(dest))
        return
    raise FileNotFoundError("ctrack not found")


_add_ctrack()

import torch  # noqa: E402

from ctrack.detect.unet3d import UNet3D  # noqa: E402
from ctrack.detect.blob import detect_frame  # noqa: E402  (reused for NMS on the heatmap)
from ctrack.eval.biohub import SCALE  # noqa: E402
from ctrack.link.hungarian import build_graph, close_gaps  # noqa: E402
from ctrack.submit.format import COLUMNS, edge_row, node_row  # noqa: E402

SUBSAMPLE = (1, 4, 4)
ISO = np.array([1.625, 1.625, 1.625])
POINT_THRESHOLD = 0.5
MAX_DISP_UM = 12.0


def find_dir(name):
    for p in Path("/kaggle/input").glob(f"**/{name}"):
        if p.is_dir() and any(p.glob("*.zarr")):
            return p
    raise FileNotFoundError(name)


def load_detector():
    ckpt = next(Path("/kaggle/input").glob("**/unet_fold*.pt"), None)
    if ckpt is None:
        return None
    state = torch.load(ckpt, map_location="cpu")
    model = UNet3D(base=state.get("base", 32))
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded detector {ckpt.name} (recall {state.get('recall')})")
    return model


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


def detect(model, vol, device):
    """Per-frame detections via the UNet heatmap. Returns {t: (coords_fullvox, scores)}."""
    out = {}
    for t in range(vol.shape[0]):
        iso = vol.norm(t)[:: SUBSAMPLE[0], :: SUBSAMPLE[1], :: SUBSAMPLE[2]]
        if model is not None:
            with torch.no_grad():
                x = torch.from_numpy(np.ascontiguousarray(iso))[None, None].float().to(device)
                hm = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            coords, scores = detect_frame(hm, ISO, threshold=POINT_THRESHOLD)
        else:  # fall back to classical blobs if no weights attached
            coords, scores = detect_frame(iso, ISO, threshold=0.1)
        out[t] = (coords * np.array(SUBSAMPLE), scores)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_detector()
    if model is not None:
        model.to(device)

    test_dir = find_dir("test")
    datasets = sorted(p.name.replace(".zarr", "") for p in test_dir.glob("*.zarr"))
    print("test datasets:", datasets)

    rows = []
    for ds in datasets:
        vol = RawVolume(test_dir / f"{ds}.zarr")
        det = detect(model, vol, device)
        nodes, h_edges = build_graph(det, SCALE, max_displacement_um=MAX_DISP_UM)
        if HAVE_ILP:
            from ctrack.link.ilp import link_ilp
            edges = link_ilp(nodes, SCALE, max_disp_um=MAX_DISP_UM, topk=3,
                             appearance_weight=2.0, disappearance_weight=2.0)
        else:
            nodes, edges = close_gaps(nodes, h_edges, SCALE, max_total_um=MAX_DISP_UM)
        for nid, t, z, y, x in nodes:
            rows.append(node_row(ds, int(nid), int(t), int(z), int(y), int(x)))
        for s, d in edges:
            rows.append(edge_row(ds, int(s), int(d)))
        print(f"{ds}: {len(nodes)} nodes, {len(edges)} edges")

    assert datasets and rows, "empty submission"
    df = pd.DataFrame(rows)
    df.insert(0, "id", np.arange(len(df), dtype=np.int64))
    df[COLUMNS].to_csv("submission.csv", index=False)
    print(f"wrote submission.csv: {len(df)} rows across {len(datasets)} datasets")


if __name__ == "__main__":
    main()
