#!/usr/bin/env python3
"""Build `evalset_blend3` — the student detector joins the detection consensus (STRATEGY 46).

Rank-1 ROI move: blend our trained UNet3D's logits as a THIRD detection member,
after the shipped dual-seed blend and before peak extraction, inside the evalset
instrument — so the result scores locally on 24 volumes against the measured
0.8947 baseline before any submission is spent.

Mechanics (each verified by scripts/local/test_blend3.py on MPS, input recipe
bit-exact vs training):
  * student input  = strided (1,4,4) downsample + ctrack quantile 0.01/0.999
    normalization CLIPPED to [0,1] — NOT the pipeline's 0.001/0.999 unclipped;
  * alignment      = mean/std to the primary's logit scale, ratio clamped
    0.5–2.0 (the same treatment the shipped secondary member gets);
  * blend          = (1-w)·blended_primary + w·aligned_student,
    w from BIOHUB_STUDENT_DET_WEIGHT (default 0.2).

The patch inserts into the ALREADY-PATCHED predict_unet_transformer.py (the
d4edge merged patch runs first), with exact-count asserts on both anchors —
a patch that silently fails to apply looks exactly like success (§CLAUDE.md).

Prereqs: dataset `maruanbottoni/biohub-student-unet` carrying the gated
checkpoint as unet_fold0.pt (upload after the golden-8 gates pass), plus
`maruanbottoni/ctrack-src` for the UNet3D class.

Run: python scripts/build_blend3_kernel.py [--weight 0.2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "kernels/evalset_d4edge/evalset_d4edge.ipynb"
OUT = REPO / "kernels/evalset_blend3"

HELPER_ANCHOR = "@torch.no_grad()\ndef predict_video("
HELPER_BLOCK = r'''# STRATEGY 46: third-member student detection blend ---------------------------
import os as _s46_os
_S46_W = float(_s46_os.environ.get("BIOHUB_STUDENT_DET_WEIGHT", "0.2"))
_S46_MODEL = [None]
_S46_CACHE = {}


def _s46_model(device):
    if _S46_MODEL[0] is None:
        import sys as _s46_sys
        if "/kaggle/working/src" not in _s46_sys.path:
            _s46_sys.path.insert(0, "/kaggle/working/src")
        from ctrack.detect.unet3d import UNet3D as _S46_U
        _ck = torch.load(_s46_os.environ["BIOHUB_STUDENT_WEIGHTS"],
                         map_location="cpu", weights_only=False)
        _m = _S46_U(base=_ck.get("base", 32))
        _m.load_state_dict(_ck["model"] if "model" in _ck else _ck)
        _m.to(device).eval()
        _S46_MODEL[0] = _m
        print(f"[46] student loaded ({sum(p.numel() for p in _m.parameters())/1e6:.2f}M params) "
              f"w={_S46_W} from {_s46_os.environ['BIOHUB_STUDENT_WEIGHTS']}", flush=True)
    return _S46_MODEL[0]


def _s46_logits(zarr_arr, t, downsample, q01, q999, device, ref):
    key = (id(zarr_arr), int(t))
    if key not in _S46_CACHE:
        if len(_S46_CACHE) > 8:
            _S46_CACHE.clear()
        fr = np.asarray(zarr_arr[int(t)])
        fr = fr[::downsample[0], ::downsample[1], ::downsample[2]]
        fr = np.clip((fr.astype(np.float32) - q01) / max(1e-6, q999 - q01), 0.0, 1.0)
        with torch.no_grad():
            out = _s46_model(device)(
                torch.from_numpy(np.ascontiguousarray(fr))[None, None].float().to(device))
        _S46_CACHE[key] = out.to(ref.dtype)
    return _S46_CACHE[key]
# -----------------------------------------------------------------------------


@torch.no_grad()
def predict_video('''

BLEND_ANCHOR = """        del imgs

        # --- Detect cells in each frame (dedup across windows) ---"""
BLEND_BLOCK = """        del imgs

        # STRATEGY 46: blend the student's aligned logits as a third member.
        if _S46_W > 0:
            _s46_q01 = float(ds.quantiles["0.01"])
            _s46_q999 = float(ds.quantiles["0.999"])
            for _s46_f, _s46_t in enumerate(frame_indices):
                _s46_base = det_logits[_s46_f]
                _s46_stu = _s46_logits(zarr_arr, _s46_t, downsample,
                                       _s46_q01, _s46_q999, device, _s46_base)
                _s46_pm = _s46_base.mean()
                _s46_ps = _s46_base.float().std().clamp_min(1e-4)
                _s46_sm = _s46_stu.mean()
                _s46_ss = _s46_stu.float().std().clamp_min(1e-4)
                _s46_al = (_s46_stu - _s46_sm) * (_s46_ps / _s46_ss).clamp(0.5, 2.0) + _s46_pm
                det_logits[_s46_f] = (1.0 - _S46_W) * _s46_base + _S46_W * _s46_al

        # --- Detect cells in each frame (dedup across windows) ---"""

PATCH_CELL_TMPL = r'''# ---------------------------------------------- STRATEGY 46: student blend-in
import glob, os, shutil, sys
from pathlib import Path

os.environ.setdefault("BIOHUB_STUDENT_DET_WEIGHT", "{weight}")

# ctrack package (for UNet3D) from the ctrack-src dataset.
_hits = [h for h in glob.glob("/kaggle/input/**/detect/unet3d.py", recursive=True)
         if "tracking_repo" not in h]
assert _hits, "ctrack-src dataset not attached"
_pkg = Path("/kaggle/working/src/ctrack")
if _pkg.exists():
    shutil.rmtree(_pkg)
shutil.copytree(Path(_hits[0]).parents[1], _pkg)
(_pkg / "__init__.py").touch()

# student checkpoint from the biohub-student-unet dataset.
_wts = sorted(glob.glob("/kaggle/input/**/biohub-student-unet/**/*.pt", recursive=True)) or \
       sorted(glob.glob("/kaggle/input/**/unet_fold0*.pt", recursive=True))
assert _wts, "biohub-student-unet dataset not attached"
os.environ["BIOHUB_STUDENT_WEIGHTS"] = _wts[0]
print("student weights:", _wts[0], "| w =", os.environ["BIOHUB_STUDENT_DET_WEIGHT"])

# Patch the (already d4edge-patched) predict driver. Exact-count asserts: a
# patch that silently fails to apply looks exactly like success.
_pp = Path("/kaggle/working/tracking_repo/scripts/predict_unet_transformer.py")
_src = _pp.read_text()
_helper_anchor = {helper_anchor!r}
_blend_anchor = {blend_anchor!r}
assert _src.count(_helper_anchor) == 1, f"helper anchor x{{_src.count(_helper_anchor)}}"
assert _src.count(_blend_anchor) == 1, f"blend anchor x{{_src.count(_blend_anchor)}}"
_src = _src.replace(_helper_anchor, {helper_block!r}, 1)
_src = _src.replace(_blend_anchor, {blend_block!r}, 1)
compile(_src, str(_pp), "exec")
_pp.write_text(_src)
print("[46] blend patch applied (2/2 anchors matched, compile OK)", flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="0.2")
    args = ap.parse_args()

    nb = json.loads(BASE.read_text())
    # HARD LESSON (first run, byte-identical output): cell 9 contains BOTH the
    # d4edge patch AND the shard driver -- a separate cell inserted "after" it
    # executes after prediction has already finished (t=4019s vs shards at
    # t=266s in the run log). The only correct place is INSIDE cell 9, after
    # its own patch application and before the driver. `def list_test_stems()`
    # is the unique boundary between the two.
    cell_src = PATCH_CELL_TMPL.format(
        weight=args.weight, helper_anchor=HELPER_ANCHOR, blend_anchor=BLEND_ANCHOR,
        helper_block=HELPER_BLOCK, blend_block=BLEND_BLOCK)
    c9 = "".join(nb["cells"][9]["source"])
    driver_anchor = "\ndef list_test_stems()"
    assert c9.count(driver_anchor) == 1, f"driver anchor x{c9.count(driver_anchor)}"
    assert c9.find("patch applied") < c9.find(driver_anchor)
    c9 = c9.replace(driver_anchor, "\n\n" + cell_src + "\n" + driver_anchor, 1)
    compile(c9, "cell9.py", "exec")
    nb["cells"][9]["source"] = c9.splitlines(keepends=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evalset_blend3.ipynb").write_text(json.dumps(nb))

    meta = json.loads((BASE.parent / "kernel-metadata.json").read_text())
    meta["id"] = "maruanbottoni/biohub-evalset-blend3"
    meta["title"] = "Biohub evalset blend3"
    meta["code_file"] = "evalset_blend3.ipynb"
    meta["dataset_sources"] = sorted(set(meta["dataset_sources"]) | {
        "maruanbottoni/ctrack-src", "maruanbottoni/biohub-student-unet"})
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"built {OUT} (w={args.weight})")
    print("prereq: kaggle datasets create/version with the gated checkpoint as "
          "biohub-student-unet, then: .venv/bin/kaggle kernels push -p", OUT)


if __name__ == "__main__":
    main()
