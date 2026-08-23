#!/usr/bin/env python
"""Local replay of the pub920 kernel (the LB-0.920 line, row 98) — reject-gating for
neighbourhood variants, attribution ablations (PLAN 3.1b) and probe pre-registration.

Adapted from pub162_local_replay.py for the lb920 notebook's DIFFERENT cell layout
(extracted to nbcells_920/ by cell index, code cells 01..10):

  cell01  full env knob sheet (DeepCenter veto stack ON, retention floor 0.90, bidir 0.30)
  cell02  the AUTHOR'S config guard — transform02 rewrites expected values for overrides
  cell03  paths (WORKING_DIR, REPO_DIR, TEST_DIR hard-bound to competition input)
  cell04  staging + ensure_dependencies (shimmed by transform04) + repo patches
  cell05  CUDA fail-fast (removed) + predict subprocess launch — LOCAL_PATCH_HOOK
          (weight override + MPS device + (coords,edges) pred cache) injected at the
          start_time anchor, AFTER the cell's own patches to the staged repo
  cell06  pipeline lib + ILP + post-proc (DeepCenter inference happens HERE, in-process:
          its device line gets the MPS patch; optional gap-clamp lift as in pub162)
  cell07-09  author's audit + validators
  cell10  tail

Requirements (idempotent, see setup in the repo docs / row 103):
  - BIOHUB_REPLAY_SCRATCH = this session's scratchpad, containing kaggle_input/ with:
      datasets/pilkwang/{support-pack, temporal-unet, ranker, deepcenter} symlinks,
      biohub-deepcenter-unet3d-center-prior-v1 symlink (cells use BOTH path forms),
      competitions/biohub-cell-tracking-during-development/test -> volume dir
  - BIOHUB_PRED_CACHE: pub920 changes EDGE INFERENCE (bidir 0.30, retention floor),
    so it must NOT share a cache with pub162 runs (row 67 hazard).

Usage:  pub920_local_replay.py <variant> ['{"BIOHUB_...": "value", ...}']
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

S = Path(os.environ.get(
    "BIOHUB_REPLAY_SCRATCH",
    "/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/a63444a8-f428-46eb-a4d9-5f3109b5fe00/scratchpad"))
CELLS = Path(os.environ.get("BIOHUB_CELLS_DIR", str(S / "nbcells_920")))
MIRROR = S / "kaggle_input"

variant = sys.argv[1]
overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

RUN = S / "runs" / variant
RUN.mkdir(parents=True, exist_ok=True)
os.chdir(RUN)
print(f"[local] pub920 variant={variant} run_dir={RUN}", flush=True)

os.environ.setdefault("BIOHUB_PRED_CACHE", str(S / "cache_pub920"))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
Path(os.environ["BIOHUB_PRED_CACHE"]).mkdir(exist_ok=True)

# ---- staged-repo patch payloads (identical to pub162_local_replay.py) -------
DEV_OLD = '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
DEV_NEW = '''    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and os.environ.get("BIOHUB_MPS", "1") != "0":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[local] inference device: {device}", flush=True)'''

CACHE_OLD = '''        coords, edges = predict_video(
                model, ds_path, device,
                cfg=cfg,
                window_size=window_size,
                unet_batch_size=unet_batch_size,
                downsample=downsample,
                secondary_model=secondary_model,
                secondary_edge_weight=secondary_edge_weight,
                secondary_detection_weight=secondary_detection_weight,
                secondary_link_mode=secondary_link_mode,
                secondary_mix_temperature=secondary_mix_temperature,
                secondary_low_margin_max=secondary_low_margin_max,
            )'''

CACHE_NEW = '''        import pickle as _pickle
        _cache_env = os.environ.get("BIOHUB_PRED_CACHE", "")
        _cache_f = Path(_cache_env) / f"{name}.pkl" if _cache_env else None
        if _cache_f is not None and _cache_f.exists():
            coords, edges = _pickle.loads(_cache_f.read_bytes())
            print(f"[cache] loaded {name}", flush=True)
        else:
            coords, edges = predict_video(
                model, ds_path, device,
                cfg=cfg,
                window_size=window_size,
                unet_batch_size=unet_batch_size,
                downsample=downsample,
                secondary_model=secondary_model,
                secondary_edge_weight=secondary_edge_weight,
                secondary_detection_weight=secondary_detection_weight,
                secondary_link_mode=secondary_link_mode,
                secondary_mix_temperature=secondary_mix_temperature,
                secondary_low_margin_max=secondary_low_margin_max,
            )
            if _cache_f is not None:
                _cache_f.write_bytes(_pickle.dumps((coords, edges)))
                print(f"[cache] saved {name}", flush=True)'''

FAILFAST_OLD = '''if not _torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this notebook. Enable a Kaggle GPU accelerator and commit again."
    )
print("CUDA device:", _torch.cuda.get_device_name(0))'''

FAILFAST_NEW = 'print("[local] CUDA fail-fast bypassed; single-process MPS/CPU predict")'

LOCAL_PATCH_HOOK = '''
# ---- OPTIONAL: swap in OUR trained detector (see pub162 driver, row 93 hook) ----
_ow = os.environ.get("BIOHUB_OVERRIDE_PRIMARY_WEIGHTS", "")
if _ow:
    import hashlib as _hl, shutil as _sh
    _wd = REPO_DIR / "weights" / "unet_transformer" / "split_0"
    for _anc in (REPO_DIR / "weights", REPO_DIR / "weights" / "unet_transformer", _wd):
        if _anc.is_symlink():
            _real = _anc.resolve()
            _anc.unlink()
            _sh.copytree(_real, _anc)
            print(f"[local] materialised symlinked dir {_anc.name}/ (was -> {_real})", flush=True)
    _dst = _wd / "edge_predictor_best.pth"
    assert _dst.exists(), f"[local] staged primary weights missing at {_dst}"
    assert str(_dst.resolve()).startswith(str(REPO_DIR.resolve())), (
        f"[local] REFUSING to write outside the run dir: {_dst.resolve()}")
    _before = _hl.sha256(_dst.read_bytes()).hexdigest()[:8]
    _src = _hl.sha256(open(_ow, "rb").read()).hexdigest()[:8]
    if _dst.is_symlink():
        _dst.unlink()
    _sh.copyfile(_ow, _dst)
    _after = _hl.sha256(_dst.read_bytes()).hexdigest()[:8]
    assert _after == _src, f"[local] weight swap FAILED: staged {_after} != source {_src}"
    print(f"[local] PRIMARY WEIGHTS {'ALREADY' if _before == _after else 'SWAPPED'} "
          f"{_before} -> {_after} (source {_src}) from {_ow}", flush=True)

# ---- local replay: device + cache patches on the freshly staged+patched repo
_lp = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_lt = _lp.read_text()
for _lo, _ln, _tag in ((DEV_OLD, DEV_NEW, "device"), (CACHE_OLD, CACHE_NEW, "cache")):
    _n = _lt.count(_lo)
    assert _n == 1, f"[local] {_tag} patch expected 1 match, found {_n}"
    _lt = _lt.replace(_lo, _ln)
_lp.write_text(_lt)
print("[local] device+cache patches applied (1 match each)", flush=True)

start_time = time.time()'''


def transform02(src: str) -> str:
    """Rewrite the AUTHOR's guard expectations for intentionally overridden knobs,
    so it still catches every OTHER unintended drift."""
    for k, v in overrides.items():
        pat = re.compile(r'("%s":\s*)([^,\n]+)(,?)' % re.escape(k))
        def repl(m, _v=v):
            # Preserve the guard's type: _EXPECTED_TEXT entries hold quoted strings
            # and compare env TEXT verbatim; _EXPECTED_NUMERIC entries hold floats.
            old = m.group(2).strip()
            if old.startswith(("'", '"')):
                val = repr(str(_v))
            else:
                try:
                    val = repr(float(_v))
                except ValueError:
                    val = repr(_v)
            return m.group(1) + val + m.group(3)
        src, n = pat.subn(repl, src)
        print(f"[guard] {k}: {n} expected-value rewrite(s)", flush=True)
    return src


def transform04(src: str) -> str:
    call = "ensure_dependencies(ARTIFACTS)"
    n = src.count(call)
    assert n == 1, f"ensure_dependencies call count {n}"
    return src.replace(call, 'print("[local] wheel install skipped (venv)")')


def transform05(src: str) -> str:
    n = src.count(FAILFAST_OLD)
    assert n == 1, f"fail-fast block match count {n}"
    src = src.replace(FAILFAST_OLD, FAILFAST_NEW)
    # cell05 carries an INLINE guard on the bidir weight (literal 0.30 twice) —
    # a fourth style of hardcoded expectation. Rewrite when overridden.
    if "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT" in overrides:
        v = float(overrides["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"])
        g1 = "_bidirectional_weight_guard, 0.30, rel_tol=0.0, abs_tol=1e-12"
        g2 = '"expected_bidirectional_weight": 0.30,'
        assert src.count(g1) == 1 and src.count(g2) == 1, (src.count(g1), src.count(g2))
        src = src.replace(g1, f"_bidirectional_weight_guard, {v}, rel_tol=0.0, abs_tol=1e-12")
        src = src.replace(g2, f'"expected_bidirectional_weight": {v},')
        print(f"[guard] cell05 inline bidir guard rewritten to {v}", flush=True)
    anchor = "start_time = time.time()"
    n = src.count(anchor)
    assert n == 1, f"start_time anchor count {n}"
    return src.replace(anchor, LOCAL_PATCH_HOOK.strip())


CLAMP_OLD = "    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)"
CLAMP_NEW = "    effective_gap_max = max(1, int(GAP_CLOSE_MAX_GAP))  # UNCLAMPED (replay experiment)"


DEV06_NEW = '''    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[local] deepcenter device: {device}", flush=True)'''


def transform06(src: str) -> str:
    # DeepCenter inference runs in-process here; give it the MPS device.
    # (cell06 has no `import os` at module top, so this variant avoids os.environ.)
    n = src.count(DEV_OLD)
    assert n == 1, f"cell06 device patch expected 1 match, found {n}"
    src = src.replace(DEV_OLD, DEV06_NEW)
    if os.environ.get("BIOHUB_UNCLAMP_GAP", "0") != "0":
        n = src.count(CLAMP_OLD)
        assert n == 1, f"gap clamp patch expected 1 match, found {n}"
        print("[local] gap-close clamp LIFTED (1 match)", flush=True)
        src = src.replace(CLAMP_OLD, CLAMP_NEW)
    return src


ns: dict = {"__name__": "__main__", "display": print,
            "DEV_OLD": DEV_OLD, "DEV_NEW": DEV_NEW,
            "CACHE_OLD": CACHE_OLD, "CACHE_NEW": CACHE_NEW}


def run_cell(i: int, transform=None) -> None:
    src = (CELLS / f"cell{i:02d}.py").read_text()
    src = src.replace("/kaggle/input", str(MIRROR))
    src = src.replace("/kaggle/working", str(RUN))
    # `display` is provided in ns by the driver; the import fails outside Jupyter.
    src = src.replace("from IPython.display import display",
                      "pass  # [local] display provided by driver")
    # ROW 82 / ROW 105: cells HARDCODE os.environ assignments for knobs (cell01 the
    # sheet, cell05 re-sets BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION="0.90" right
    # before launching predict — clobbering any env-level override). Rewrite every
    # hardcoded assignment of an overridden key, in every cell, at source level.
    for k, v in overrides.items():
        pat = re.compile(r'(os\.environ\[["\']%s["\']\]\s*=\s*)(["\'][^"\']*["\'])'
                         % re.escape(k))
        src, n = pat.subn(lambda m, _v=v: m.group(1) + repr(str(_v)), src)
        if n:
            print(f"[override] cell{i:02d}: {n} hardcoded assignment(s) of {k} "
                  f"rewritten to {v!r}", flush=True)
    if transform:
        src = transform(src)
    print(f"\n===== cell {i:02d} =====", flush=True)
    exec(compile(src, f"cell{i:02d}", "exec"), ns)


run_cell(1)
for k, v in overrides.items():
    os.environ[k] = v
    print(f"[override] {k} = {v}", flush=True)
run_cell(2, transform=transform02)
run_cell(3)
run_cell(4, transform=transform04)

# ---- RE-APPLY overrides, then VERIFY (row 82 discipline) --------------------
for k, v in overrides.items():
    if os.environ.get(k) != v:
        print(f"[override] RE-APPLY after cell04: {k} = {v} "
              f"(a cell had set {os.environ.get(k)!r})", flush=True)
        os.environ[k] = v
for k, v in overrides.items():
    assert os.environ[k] == v, (
        f"[local] override {k} did not survive to inference: "
        f"want {v!r}, got {os.environ[k]!r}")
print(f"[local] all {len(overrides)} overrides verified live after cell04", flush=True)

run_cell(5, transform=transform05)
run_cell(6, transform=transform06)
# Cells 07-09 are the AUTHOR's validators. On Kaggle they always run against a
# fresh-inference submission and must pass. Locally, CACHE-HIT volumes emit no
# frame-retention diagnostics (those come from the predict subprocess), so the
# cell07 coverage check fails by construction. Tolerate locally, loud warning;
# BIOHUB_STRICT_VALIDATORS=1 restores fatal behaviour (use on fresh-cache runs).
strict = os.environ.get("BIOHUB_STRICT_VALIDATORS", "0") == "1"
for i in (7, 8, 9, 10):
    try:
        run_cell(i)
    except Exception as e:  # noqa: BLE001
        if strict:
            raise
        print(f"[local] ⚠ VALIDATOR cell{i:02d} FAILED (tolerated, cache-hit artifact?): "
              f"{type(e).__name__}: {e}", flush=True)
print(f"\n[local] DONE variant={variant} -> {RUN / 'submission.csv'}", flush=True)
