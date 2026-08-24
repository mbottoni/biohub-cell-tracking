#!/usr/bin/env python
"""Build pub920_divilp — division-aware ILP: the model tells the GLOBAL solver where to fork.

WHY (rows 116-117): every post-hoc division mechanism pins TP at 4 because correct second
daughters are CLAIMED by continuation edges, and greedy stealing cannot price the cascade
(the old parent's fate). The ILP can — but with the shipped costs it structurally never
forks: appearance 0.0 is always cheaper than division 1.0 minus any edge gain (p <= 1).
That is why kunal's division-weight sweep (0.3-3.0, all 0.915) read inert and why 100% of
this family's divisions come from the postlink rule.

ONE mechanism, two coupled knobs, env-gated:
  1. For each frame pair, parents scored by the trained division model (AUC 0.863,
     32 crop features at raw resolution, base64-embedded); for parents with
     model_prob >= BIOHUB_DIVILP_MIN_PROB (0.85), the SECOND-best target's edge
     probability is boosted in logit space by BIOHUB_DIVILP_BONUS x model_prob (4.0) —
     lifting real fork candidates over the 0.48 candidate threshold.
  2. BIOHUB_ILP_DIVISION_WEIGHT 1.0 -> 0.45: below the candidate threshold, so any
     candidate second edge is fork-eligible IN THE GLOBAL OBJECTIVE. Unboosted organic
     double-candidates were measured harmless at weight 0.3 on LB (kunal). The ILP then
     steals a claimed daughter ONLY when the cascade resolves (old parent has a
     comparable alternative; disappearance 1.5 otherwise vetoes) — exactly the pricing
     greedy steal (row 117) lacked.

Patch anchors (all must appear exactly once; hard-fail otherwise, row 93):
  A1 'def _detect_cells_pooled('        <- globals + per-video hook defs
  A2 '            candidates = sorted(' <- boost injection (t_src/c_src/ds_arr/probs in scope)
  A3 '        ds_path = data_dir / name' <- per-video model/zarr binding
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kernels" / "pub920_div917" / "pub920_div917.ipynb"
MODEL = Path("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
             "914d4e78-261f-4ae8-9e3a-e159bcbcf840/scratchpad/divdet/div_model.pkl")
DST_DIR = REPO / "kernels" / "pub920_divilp"
DST = DST_DIR / "pub920_divilp.ipynb"

MODULE_SRC = '''"""divilp_model — division-model scoring for the division-aware ILP (row 118)."""
import json
import os
import pickle
from pathlib import Path

import numpy as np

RZ, RY, RX = 2, 12, 12
_MODEL = None
_ARR = None
_T = 0
_FRAMES = {}


def _model():
    global _MODEL
    if _MODEL is None:
        with open(os.environ["BIOHUB_DIVILP_MODEL"], "rb") as f:
            _MODEL = pickle.load(f)["model"]
        print("[divilp] model loaded", flush=True)
    return _MODEL


def set_video(ds_path):
    global _ARR, _T, _FRAMES
    zp = Path(str(ds_path))
    if not (zp / "0" / "zarr.json").exists():
        alt = zp.parent / (zp.name + ".zarr")
        if (alt / "0" / "zarr.json").exists():
            zp = alt
        else:
            raise FileNotFoundError(f"[divilp] no zarr at {zp} or {alt}")
    import zarr
    _ARR = zarr.open(str(zp), mode="r")["0"]
    _T = _ARR.shape[0]
    _FRAMES = {}
    print(f"[divilp] video bound: {zp.name} T={_T}", flush=True)


def _frame(t):
    t = int(np.clip(t, 0, _T - 1))
    if t not in _FRAMES:
        if len(_FRAMES) > 4:
            _FRAMES.pop(next(iter(_FRAMES)))
        _FRAMES[t] = np.asarray(_ARR[t])
    return _FRAMES[t]


def _crop(frame, z, y, x):
    Z, Y, X = frame.shape
    return frame[max(0, z - RZ):min(Z, z + RZ + 1),
                 max(0, y - RY):min(Y, y + RY + 1),
                 max(0, x - RX):min(X, x + RX + 1)]


def _feats(f_prev, f_cur, f_next, z, y, x):
    out = []
    cur = _crop(f_cur, z, y, x)
    if cur.size == 0:
        return None
    for f in (cur, _crop(f_prev, z, y, x), _crop(f_next, z, y, x)):
        if f.size == 0:
            return None
        r = f.ravel().astype(np.float32)
        p = np.percentile(r, [50, 90, 99])
        out += [r.mean(), r.std(), p[0], p[1], p[2], r.max()]
    a, b, c = np.array(out[0:6]), np.array(out[6:12]), np.array(out[12:18])
    out += list(a - b) + list(c - a)
    mid = cur.max(axis=0).astype(np.float32)
    cy, cx = np.array(mid.shape) // 2
    q = [mid[:cy, :cx].mean(), mid[:cy, cx:].mean(), mid[cy:, :cx].mean(), mid[cy:, cx:].mean()]
    out += [float(np.std(q)), float(max(q) - min(q))]
    return out


def score_parents(t, raw_zyx):
    """raw_zyx: (N,3) float voxel coords at original resolution -> (N,) model probs."""
    if _ARR is None or len(raw_zyx) == 0:
        return np.zeros(len(raw_zyx), dtype=np.float32)
    fp, fc, fn = _frame(t - 1), _frame(t), _frame(t + 1)
    X, ok = [], []
    for z, y, x in raw_zyx:
        v = _feats(fp, fc, fn, int(round(z)), int(round(y)), int(round(x)))
        if v is None:
            X.append([0.0] * 32); ok.append(False)
        else:
            X.append(v); ok.append(True)
    P = _model().predict_proba(np.nan_to_num(
        np.array(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0))[:, 1]
    P = P.astype(np.float32)
    P[~np.array(ok)] = 0.0
    return P
'''

mod_b64 = base64.b64encode(MODULE_SRC.encode()).decode()
model_b64 = base64.b64encode(MODEL.read_bytes()).decode()

CELL_BLOCK = f'''

# ===== DIVISION-AWARE ILP (row 118): model-gated fork candidates + fork-eligible costs =====
import base64 as _di_b64
os.environ["BIOHUB_DIVILP_ENABLE"] = "1"
os.environ.setdefault("BIOHUB_DIVILP_MIN_PROB", "0.85")
os.environ.setdefault("BIOHUB_DIVILP_BONUS", "4.0")
_DI_MODULE_B64 = "{mod_b64}"
_DI_MODEL_B64 = "{model_b64}"
(REPO_DIR / "scripts" / "divilp_model.py").write_text(_di_b64.b64decode(_DI_MODULE_B64).decode())
_di_model_path = REPO_DIR / "scripts" / "div_model.pkl"
_di_model_path.write_bytes(_di_b64.b64decode(_DI_MODEL_B64))
os.environ.setdefault("BIOHUB_DIVILP_MODEL", str(_di_model_path))
print(f"divilp: module + model staged ({{_di_model_path.stat().st_size}} bytes)")


def apply_divilp_patch():
    marker = "BIOHUB_DIVILP_PATCH_V1"
    ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
    s = ps.read_text()
    if marker in s:
        print("divilp patch already present")
        return
    a1 = "def _detect_cells_pooled("
    g = (
        "# BIOHUB_DIVILP_PATCH_V1: division-aware ILP inputs.\\n"
        "_DIVILP_ENABLE = os.environ.get(\\"BIOHUB_DIVILP_ENABLE\\", \\"0\\") != \\"0\\"\\n"
        "_DIVILP_MIN_PROB = float(os.environ.get(\\"BIOHUB_DIVILP_MIN_PROB\\", \\"0.85\\"))\\n"
        "_DIVILP_BONUS = float(os.environ.get(\\"BIOHUB_DIVILP_BONUS\\", \\"4.0\\"))\\n"
        "_DIVILP_STATS = {{\\"scored\\": 0, \\"floor\\": 0, \\"boosted\\": 0}}\\n\\n\\n"
        "def _divilp_set_video(ds_path):\\n"
        "    if not _DIVILP_ENABLE:\\n"
        "        return\\n"
        "    import divilp_model as _divilp\\n"
        "    _divilp.set_video(ds_path)\\n"
        "    if _DIVILP_STATS[\\"scored\\"]:\\n"
        "        print(f\\"[divilp] running totals: {{{{_DIVILP_STATS}}}}\\", flush=True)\\n\\n\\n"
    )
    a2 = "            candidates = sorted("
    r2 = (
        "            if _DIVILP_ENABLE and n_src and n_tgt >= 2:\\n"
        "                import divilp_model as _divilp\\n"
        "                _dv_raw = c_src[:, 1:].astype(np.float32) * ds_arr\\n"
        "                _dv_p = _divilp.score_parents(int(t_src), _dv_raw)\\n"
        "                _DIVILP_STATS[\\"scored\\"] += int(n_src)\\n"
        "                for _di in range(n_src):\\n"
        "                    _mp = float(_dv_p[_di])\\n"
        "                    if _mp < _DIVILP_MIN_PROB:\\n"
        "                        continue\\n"
        "                    _DIVILP_STATS[\\"floor\\"] += 1\\n"
        "                    _row = probs[_di]\\n"
        "                    _j2 = int(np.argsort(-_row)[1])\\n"
        "                    _p2 = float(_row[_j2])\\n"
        "                    _lo = np.log(max(_p2, 1e-6) / max(1.0 - _p2, 1e-6)) + _DIVILP_BONUS * _mp\\n"
        "                    _pb = 1.0 / (1.0 + np.exp(-_lo))\\n"
        "                    if _pb > _p2:\\n"
        "                        probs[_di, _j2] = _pb\\n"
        "                        _DIVILP_STATS[\\"boosted\\"] += 1\\n"
        "            candidates = sorted("
    )
    a3 = "        ds_path = data_dir / name\\n"
    r3 = a3 + "        _divilp_set_video(ds_path)\\n"
    for a in (a1, a2, a3):
        if s.count(a) != 1:
            raise RuntimeError(f"divilp anchor not unique: {{a[:60]!r}} x{{s.count(a)}}")
    s = s.replace(a1, g + a1, 1)
    s = s.replace(a2, r2, 1)
    s = s.replace(a3, r3, 1)
    ps.write_text(s)
    print(f"divilp patch applied (min_prob={{os.environ['BIOHUB_DIVILP_MIN_PROB']}}, "
          f"bonus={{os.environ['BIOHUB_DIVILP_BONUS']}})")


apply_divilp_patch()
assert "BIOHUB_DIVILP_PATCH_V1" in (REPO_DIR / "scripts" / "predict_unet_transformer.py").read_text()

'''

nb = json.loads(SRC.read_text())
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

# cell00: make the ILP fork-eligible (division weight below the 0.48 candidate threshold).
s0 = "".join(cells[0]["source"])
OLD_W = 'os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.0"'
assert s0.count(OLD_W) == 1
s0 = s0.replace(
    OLD_W,
    'os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "0.45"  # DIVILP: fork-eligible in the global objective (row 118)',
    1)
cells[0]["source"] = s0.splitlines(keepends=True)

s4 = "".join(cells[4]["source"])
ANCHOR = "start_time = time.time()"
assert s4.count(ANCHOR) == 1
s4 = s4.replace(ANCHOR, CELL_BLOCK + ANCHOR, 1)
cells[4]["source"] = s4.splitlines(keepends=True)

guard_src = "".join(cells[1]["source"])
assert "DIVILP" not in guard_src and "DIVISION_WEIGHT" not in guard_src

DST_DIR.mkdir(exist_ok=True)
DST.write_text(json.dumps(nb))
meta = json.loads((REPO / "kernels" / "pub920_div917" / "kernel-metadata.json").read_text())
meta["id"] = "maruanbottoni/biohub-pub920-divilp"
meta["title"] = "biohub pub920 divilp"
meta["code_file"] = "pub920_divilp.ipynb"
(DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {DST} ({DST.stat().st_size} bytes)")
