#!/usr/bin/env python
"""Build pub920_div917_tophat — confirmation-gated detection rescue (row 110 idea 4).

WHY: det955 (row 113) proved the 0.90–0.96875 detection band holds real edges ON
MEMORIZED volumes and noise elsewhere — a blanket threshold drop fails. This kernel
keeps the strict 0.96875 threshold and rescues band peaks ONLY when an independent
detector confirms them: xiaolei's tophat-trained UNet3D (different architecture,
different preprocessing, different training view; ported in scripts/m001_confirm.py,
smoke-tested: ~500 det/frame @ 0.4 s/frame). Confirmation radius 3.0 um on the
isotropic 1.625 um logits grid.

ONE mechanism vs pub920_div917 (LB 0.923): the rescue path. det_threshold itself is
untouched, so the cell01 guard still passes. Two anchor edits to the staged repo's
predict_unet_transformer.py, both hard-fail if absent (no silent no-op, row 93):
  1. globals + _m001_set_video() before _detect_cells_pooled;
  2. band-rescue OR-ed into is_peak inside _detect_cells_pooled;
  3. per-video tree build after `ds_path = data_dir / name`.

Local gate: fresh inference (detection changes), own cache. Reject-only; the
memorized-volume trap applies to any detection change — weigh the 6-volume read
against det955's failure shape (edge_J must rise WITHOUT node_recall falling).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kernels" / "pub920_div917" / "pub920_div917.ipynb"
MOD = REPO / "scripts" / "m001_confirm.py"
DST_DIR = REPO / "kernels" / "pub920_div917_tophat"
DST = DST_DIR / "pub920_div917_tophat.ipynb"

mod_b64 = base64.b64encode(MOD.read_text().encode()).decode()

CELL_BLOCK = f'''

# ===== TOPHAT CONFIRM RESCUE (row 110 idea 4): independent-model detection rescue =====
import base64 as _th_b64
import glob as _th_glob
if "BIOHUB_M001_WEIGHTS" not in os.environ:
    # this image mounts datasets under /kaggle/input/datasets/<user>/<slug>/ (v1 ERROR was
    # the bare-slug path); probe the known layouts then glob as a last resort.
    _th_cands = [
        "/kaggle/input/datasets/xiaoleilian/biohub-unet3d-weights/unet3d_traintophat.pt",
        "/kaggle/input/biohub-unet3d-weights/unet3d_traintophat.pt",
    ] + _th_glob.glob("/kaggle/input/**/unet3d_traintophat.pt", recursive=True)
    _th_hit = next((p for p in _th_cands if Path(p).exists()), None)
    assert _th_hit is not None, f"tophat weights not found; /kaggle/input: {{_th_glob.glob('/kaggle/input/*/*')}}"
    os.environ["BIOHUB_M001_WEIGHTS"] = _th_hit
os.environ["BIOHUB_DET_RESCUE_ENABLE"] = "1"
os.environ["BIOHUB_DET_RESCUE_THRESHOLD"] = "0.90"
os.environ["BIOHUB_DET_CONFIRM_UM"] = "3.0"
assert Path(os.environ["BIOHUB_M001_WEIGHTS"]).exists(), os.environ["BIOHUB_M001_WEIGHTS"]
print("m001 weights:", os.environ["BIOHUB_M001_WEIGHTS"])

_M001_MODULE_B64 = "{mod_b64}"
(REPO_DIR / "scripts" / "m001_confirm.py").write_text(_th_b64.b64decode(_M001_MODULE_B64).decode())
print("m001_confirm.py staged into repo scripts/")


def apply_tophat_rescue_patch():
    marker = "BIOHUB_M001_RESCUE_PATCH_V1"
    ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
    s = ps.read_text()
    if marker in s:
        print("tophat rescue patch already present")
        return
    a1 = "def _detect_cells_pooled("
    g = (
        "# BIOHUB_M001_RESCUE_PATCH_V1: confirmation-gated band rescue.\\n"
        "_M001_RESCUE_ENABLE = os.environ.get(\\"BIOHUB_DET_RESCUE_ENABLE\\", \\"0\\") != \\"0\\"\\n"
        "_M001_RESCUE_THR = float(os.environ.get(\\"BIOHUB_DET_RESCUE_THRESHOLD\\", \\"0.90\\"))\\n"
        "_M001_CONFIRM_UM = float(os.environ.get(\\"BIOHUB_DET_CONFIRM_UM\\", \\"3.0\\"))\\n"
        "_M001_GRID_UM = np.array([1.625, 1.625, 1.625])\\n"
        "_M001_TREES = None\\n\\n\\n"
        "def _m001_set_video(ds_path):\\n"
        "    global _M001_TREES\\n"
        "    if not _M001_RESCUE_ENABLE:\\n"
        "        return\\n"
        "    import m001_confirm as _m001c\\n"
        "    _M001_TREES = _m001c.confirm_trees(ds_path)\\n\\n\\n"
    )
    a2 = "    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)\\n"
    r2 = (
        "    _sig = torch.sigmoid(logits)\\n"
        "    is_peak = (logits == pooled) & (_sig > det_threshold)\\n"
        "    if _M001_RESCUE_ENABLE and _M001_TREES is not None:\\n"
        "        _tree = _M001_TREES.get(int(t))\\n"
        "        if _tree is not None:\\n"
        "            _band = (logits == pooled) & (_sig > _M001_RESCUE_THR) & (_sig <= det_threshold)\\n"
        "            _bidx = torch.nonzero(_band[0, 0])\\n"
        "            if _bidx.shape[0]:\\n"
        "                _pts = _bidx.float().cpu().numpy() * _M001_GRID_UM[None, :]\\n"
        "                _dist, _ = _tree.query(_pts)\\n"
        "                _keep = _bidx[torch.from_numpy(_dist <= _M001_CONFIRM_UM)]\\n"
        "                if _keep.shape[0]:\\n"
        "                    is_peak[0, 0][_keep[:, 0], _keep[:, 1], _keep[:, 2]] = True\\n"
    )
    a3 = "        ds_path = data_dir / name\\n"
    r3 = a3 + "        _m001_set_video(ds_path)\\n"
    for a in (a1, a2, a3):
        if s.count(a) != 1:
            raise RuntimeError(f"tophat rescue anchor not unique: {{a[:60]!r}} x{{s.count(a)}}")
    s = s.replace(a1, g + a1, 1)
    s = s.replace(a2, r2, 1)
    s = s.replace(a3, r3, 1)
    ps.write_text(s)
    print(f"tophat rescue patch applied (thr={{os.environ['BIOHUB_DET_RESCUE_THRESHOLD']}}, "
          f"confirm_um={{os.environ['BIOHUB_DET_CONFIRM_UM']}})")


apply_tophat_rescue_patch()
assert "BIOHUB_M001_RESCUE_PATCH_V1" in (REPO_DIR / "scripts" / "predict_unet_transformer.py").read_text()

'''

nb = json.loads(SRC.read_text())
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
s4 = "".join(cells[4]["source"])
ANCHOR = "start_time = time.time()"
assert s4.count(ANCHOR) == 1
s4 = s4.replace(ANCHOR, CELL_BLOCK + ANCHOR, 1)
cells[4]["source"] = s4.splitlines(keepends=True)

DST_DIR.mkdir(exist_ok=True)
DST.write_text(json.dumps(nb))
meta = json.loads((REPO / "kernels" / "pub920_div917" / "kernel-metadata.json").read_text())
meta["id"] = "maruanbottoni/biohub-pub920-div917-tophat"
meta["title"] = "biohub pub920 div917 tophat"
meta["code_file"] = "pub920_div917_tophat.ipynb"
meta["dataset_sources"] = sorted(set(meta["dataset_sources"]) | {"xiaoleilian/biohub-unet3d-weights"})
(DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {DST} ({DST.stat().st_size} bytes)")
