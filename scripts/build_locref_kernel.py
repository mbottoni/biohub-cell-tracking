#!/usr/bin/env python3
"""STRATEGY 41.2 -- sub-voxel readout of the detection heatmap.

40.2 measured that perfect localisation is worth +0.0259 transferable adj and
captures 31.8% of the association oracle -- the largest capture of any oracle in
this project. 41 then killed two of the three cheap routes to it for free: both
detector D4 TTA (`det=d4`) and detector-seed blending (`detection weight=0.475`)
are ALREADY in the shipped config, so the 2.186 um residual is what survives after
averaging 2 seeds x 8 views = 16 forward passes.

That is exactly what makes THIS lever credible. Sixteen-way averaging produces a
smooth heatmap whose SHAPE encodes where those views collectively place the cell,
and the shipped readout is

    peak_idx = torch.nonzero(is_peak[0, 0])        # an INTEGER ARGMAX
    coords   = peak_idx.float().cpu().numpy()

which keeps only the mode, on a lattice, and throws the rest away. The ensemble
information we already paid 16 forward passes for is present in the heatmap and
discarded at readout. A sub-voxel estimator recovers it at ZERO extra inference
cost.

PARABOLIC, NOT CENTRE-OF-MASS. 40.4 measured CoM on the raw image as actively
harmful (4 configurations, RMS 2.186 -> 2.316..2.909) because at 3-8 um cell
diameters and ~11.7 um spacing neighbouring nuclei pull the centroid. A 3-point
parabola is local to the peak and immune to that. It is also the right estimator
for a smooth unimodal peak, which a 16-way average is.

AND THE SECOND QUANTISATION MATTERS. `predict_video` does

    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    coords = coords.astype(np.int16)          # <-- destroys any sub-voxel gain

with `downsample = [1, 4, 4]`, i.e. an isotropic 1.625 um detection lattice. So
refining the peak WITHOUT also dropping that cast is a silent no-op -- the exact
failure mode CLAUDE.md warns about. Both edits are asserted to match exactly once.

`build_graph` stores z/y/x as `pl.Float64`, so float coordinates flow through the
graph, the ILP and the submission unchanged.

Run:  python scripts/build_locref_kernel.py
      python scripts/build_evalset_kernel.py locref     # -> 24-volume instrument
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"
OUT_DIR = REPO / "kernels/locref"

# --- edit 1: sub-voxel refinement inside _detect_cells_pooled -----------------
ARGMAX_OLD = "    coords = peak_idx.float().cpu().numpy()\n"

ARGMAX_NEW = '''    # STRATEGY 41.2 -- sub-voxel readout. `logits` here is already an average over
    # 2 seeds x 8 D4 views (41), so its shape encodes where those 16 views
    # collectively place the cell. The integer argmax keeps only the mode.
    # Three-point parabola per axis, vertex offset for f(-1), f(0), f(+1):
    #     delta = (f(-1) - f(+1)) / (2 * (f(-1) - 2 f(0) + f(+1)))
    # The denominator is 2a and is NEGATIVE at a maximum, so a non-negative
    # denominator means the fit is degenerate and the offset is dropped. Peaks
    # come from a 3x3x3 max-pool so neighbours are <= f(0) by construction.
    # Parabolic rather than centre-of-mass: 40.4 measured CoM on the raw image as
    # harmful because neighbouring nuclei pull the centroid; a 3-point parabola is
    # local to the peak and immune to that.
    _lg = logits[0, 0]
    _sz = _lg.shape
    coords = peak_idx.float()
    _f0 = _lg[peak_idx[:, 0], peak_idx[:, 1], peak_idx[:, 2]].float()
    _absd = 0.0
    for _ax in range(3):
        _lo = peak_idx.clone()
        _hi = peak_idx.clone()
        _lo[:, _ax] = _lo[:, _ax] - 1
        _hi[:, _ax] = _hi[:, _ax] + 1
        _ok = (_lo[:, _ax] >= 0) & (_hi[:, _ax] < _sz[_ax])
        _lo[:, _ax] = _lo[:, _ax].clamp(0, _sz[_ax] - 1)
        _hi[:, _ax] = _hi[:, _ax].clamp(0, _sz[_ax] - 1)
        _fm = _lg[_lo[:, 0], _lo[:, 1], _lo[:, 2]].float()
        _fp = _lg[_hi[:, 0], _hi[:, 1], _hi[:, 2]].float()
        _den = _fm - 2.0 * _f0 + _fp
        _delta = torch.where(
            _den < -1e-6,
            0.5 * (_fm - _fp) / _den.clamp(max=-1e-6),
            torch.zeros_like(_den),
        )
        _delta = torch.nan_to_num(_delta, nan=0.0, posinf=0.0, neginf=0.0)
        _delta = _delta.clamp(-0.5, 0.5)
        _delta = torch.where(_ok, _delta, torch.zeros_like(_delta))
        coords[:, _ax] = coords[:, _ax] + _delta
        _absd = _absd + float(_delta.abs().sum())
    coords = coords.cpu().numpy()
    # Guard against a silent no-op: if the offsets are identically zero the patch
    # applied but did nothing, which looks exactly like success in the logs.
    _st = getattr(_detect_cells_pooled, "_locref_stats", None)
    if _st is None:
        _st = _detect_cells_pooled._locref_stats = [0, 0.0]
    _prev = _st[0]
    _st[0] += int(coords.shape[0])
    _st[1] += _absd
    if _st[0] // 20000 != _prev // 20000:
        print(
            f"[41.2] locref peaks={_st[0]:,} "
            f"mean|delta|={_st[1] / max(3 * _st[0], 1):.4f} voxel/axis",
            flush=True,
        )
'''

# --- edit 2: keep sub-voxel precision through the int16 cast ------------------
RETURN_OLD = (
    "    t_col = np.full((len(coords), 1), t, dtype=np.float32)\n"
    "    return np.concatenate([t_col, coords], axis=1).astype(np.int16)\n"
)
RETURN_NEW = (
    "    t_col = np.full((len(coords), 1), t, dtype=np.float32)\n"
    "    # STRATEGY 41.2: float32, not int16 -- an int cast here silently discards\n"
    "    # every sub-voxel offset computed above.\n"
    "    return np.concatenate([t_col, coords], axis=1).astype(np.float32)\n"
)

# --- edit 3: the SECOND quantisation, inside the merged predict_video ---------
CAST_OLD = (
    "    coords = coords.astype(np.float32)\n"
    "    coords[:, 1:] *= ds_arr\n"
    "    coords = coords.astype(np.int16)\n"
)
CAST_NEW = (
    "    coords = coords.astype(np.float32)\n"
    "    coords[:, 1:] *= ds_arr\n"
    "    # STRATEGY 41.2: the shipped code re-quantised to int16 here, on a lattice\n"
    "    # that is isotropic 1.625 um for downsample=(1,4,4). Keeping float32 is what\n"
    "    # makes the sub-voxel refinement survive to the linker, which runs BEFORE\n"
    "    # linefit smoothing and therefore sees these coordinates directly.\n"
)

# --- edit 4: print the downsample so the lattice is on the record ------------
PRINT_OLD = (
    '        f"weights={weights_path} | device={device} | window_size={window_size} '
    '| pool_kernel_um={cfg.pool_kernel_um}",\n'
)
PRINT_NEW = (
    '        f"weights={weights_path} | device={device} | window_size={window_size} '
    '| pool_kernel_um={cfg.pool_kernel_um} | downsample={downsample} | locref=parabolic",\n'
)

PATCH_BLOCK = '''
# ---------------------------------------------------------------------------
# STRATEGY 41.2 -- sub-voxel readout of the detection heatmap.
# Applied AFTER the merged edge-TTA/dual-seed patch above, because it edits
# `_detect_cells_pooled` and the driver print, which that merge leaves alone.
# ---------------------------------------------------------------------------
_s = _ps.read_text()
_locref_edits = {edits!r}
for _i, (_old, _new) in enumerate(_locref_edits, start=1):
    _n = _s.count(_old)
    if _n != 1:
        raise RuntimeError(
            f"locref edit {{_i}} expected exactly one match, found {{_n}}. "
            "Refusing to run: a patch that half-applies is worse than none."
        )
    _s = _s.replace(_old, _new, 1)
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print("STRATEGY 41.2 sub-voxel readout applied (4/4 edits matched)", flush=True)
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())

    # The merged predict_video is a string literal in cell 9; edit 3 lives inside
    # it, so patch the literal at build time rather than at kernel runtime.
    cell9 = "".join(nb["cells"][9]["source"])
    tree = ast.parse(cell9)
    merged = None
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and getattr(node.targets[0], "id", None) == "_merged"):
            merged = node.value.value
    if merged is None:
        raise RuntimeError("could not find the _merged predict_video literal in cell 9")
    if merged.count(CAST_OLD) != 1:
        raise RuntimeError(
            f"merged predict_video: int16 cast block found {merged.count(CAST_OLD)} "
            "times, expected 1"
        )
    new_merged = merged.replace(CAST_OLD, CAST_NEW, 1)
    compile(new_merged, "<merged>", "exec")

    old_lit = repr(merged)
    if cell9.count(old_lit) != 1:
        raise RuntimeError("could not substitute the _merged literal unambiguously")
    cell9 = cell9.replace(old_lit, repr(new_merged), 1)

    # The remaining three edits target the repo file and are applied at runtime.
    edits = [
        (ARGMAX_OLD, ARGMAX_NEW),
        (RETURN_OLD, RETURN_NEW),
        (PRINT_OLD, PRINT_NEW),
    ]
    anchor = 'print(\n    "Merged edge-TTA + dual-seed patch applied |",'
    if cell9.count(anchor) != 1:
        raise RuntimeError("could not find the end of the merged patch block in cell 9")
    tail = cell9.index(anchor)
    end = cell9.index(")\n", cell9.index('"identity_weight="', tail)) + 2
    cell9 = cell9[:end] + PATCH_BLOCK.format(edits=edits) + cell9[end:]

    ast.parse(cell9)
    nb["cells"][9]["source"] = cell9.splitlines(keepends=True)

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                raise RuntimeError(f"cell {i} no longer parses: {e}") from e

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "locref.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((REPO / "kernels/d4edge/kernel-metadata.json").read_text())
    meta["id"] = "maruanbottoni/biohub-locref"
    meta["title"] = "Biohub locref"
    meta["code_file"] = "locref.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_DIR}/locref.ipynb  ->  {meta['id']}")


if __name__ == "__main__":
    main()
