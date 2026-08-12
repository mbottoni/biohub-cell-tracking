#!/usr/bin/env python
"""Local dev-4 replay of the pub162 kernel — quota-free variant gating.

Executes the notebook's own code cells (extracted to nbcells/) in order, with
exactly three transforms:
  1. "/kaggle/input" -> local mirror tree (symlinks to support pack / ranker /
     secondary weights; secondary bytes are still SHA256-verified by cell 8).
  2. cell 13's CUDA fail-fast removed (single-process fallback then engages).
  3. after cell 13 applies its own runtime patches to the staged pristine repo,
     two extra patches: MPS-aware device selection + a (coords, edges) pickle
     cache so ILP/post-proc variants replay without re-running inference.

Usage:
    run_local.py <variant> ['{"BIOHUB_...": "value", ...}']
Overrides are applied AFTER cell 4 (which hardcodes the baseline env).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

S = Path("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad")
OLD = Path("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/5c5c110c-b819-4168-a812-26fe17d15065/scratchpad")
CELLS = Path(os.environ.get("BIOHUB_CELLS_DIR", str(S / "nbcells")))
MIRROR = S / "kaggle_input"

variant = sys.argv[1]
overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

RUN = S / "runs" / variant
RUN.mkdir(parents=True, exist_ok=True)
os.chdir(RUN)
print(f"[local] variant={variant} run_dir={RUN}", flush=True)

os.environ.setdefault("BIOHUB_TEST_DIR", "/Volumes/extSSD/kaggle/biohub/raw/test")
os.environ["BIOHUB_PRIMARY_ARTIFACT_MANIFEST"] = str(OLD / "support_pack/ARTIFACT_MANIFEST.json")
os.environ["BIOHUB_SECONDARY_ARTIFACT_MANIFEST"] = str(
    MIRROR / "datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json")
os.environ["BIOHUB_LOCAL_RANKER_ROOT"] = str(OLD / "ranker")
# ⚠ CACHE COLLISION HAZARD. The cache is keyed by VOLUME NAME only, so it is
# valid only across variants that change nothing upstream of (coords, edges) --
# i.e. ILP and post-processing knobs. Any variant touching DETECTION or EDGE
# inference (DET_THRESHOLD, SECONDARY_*, EDGE_TTA_*, BIDIRECTIONAL_*) MUST be
# given its own BIOHUB_PRED_CACHE dir, or it will silently load the baseline's
# predictions and reproduce the baseline score while looking like a real run.
os.environ.setdefault("BIOHUB_PRED_CACHE", str(S / "pred_cache"))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
Path(os.environ["BIOHUB_PRED_CACHE"]).mkdir(exist_ok=True)

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


def transform10(src: str) -> str:
    """Cell 10 freezes every knob at the pub162 baseline and raises on drift.
    A variant IS one intended drift: rewrite the guard's expected value for the
    overridden key(s) so the guard still catches any OTHER unintended change."""
    import re
    for k, v in overrides.items():
        short = k[len("BIOHUB_"):] if k.startswith("BIOHUB_") else k
        pat = re.compile(r'("(?:%s|%s)":\s*)([^,\n]+)(,?)' % (re.escape(short), re.escape(k)))
        def repl(m, _v=v):
            try:
                val = repr(float(_v))
            except ValueError:
                val = repr(_v)
            return m.group(1) + val + m.group(3)
        src, n = pat.subn(repl, src)
        print(f"[guard] {k}: {n} expected-value rewrite(s)", flush=True)
    return src


CLAMP_OLD = "    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)"
CLAMP_NEW = "    effective_gap_max = max(1, int(GAP_CLOSE_MAX_GAP))  # UNCLAMPED (replay experiment)"


def transform15(src: str) -> str:
    """Optionally lift the hardcoded 1-frame cap on gap closing.

    The shipped line is `min(GAP_CLOSE_MAX_GAP, 1)`, so BIOHUB_GAP_CLOSE_MAX_GAP
    is inert for every value >= 1 and only t->t+2 holes are ever bridged. With
    BIOHUB_UNCLAMP_GAP=1 the env var is honoured, letting the pass bridge
    t->t+3 (gap 2) and beyond -- the detection-FN recovery lever of 13/22.
    """
    if os.environ.get("BIOHUB_UNCLAMP_GAP", "0") == "0":
        return src
    n = src.count(CLAMP_OLD)
    assert n == 1, f"[local] gap clamp patch expected 1 match, found {n}"
    print("[local] gap-close clamp LIFTED (1 match)", flush=True)
    return src.replace(CLAMP_OLD, CLAMP_NEW)


def transform13(src: str) -> str:
    n = src.count(FAILFAST_OLD)
    assert n == 1, f"fail-fast block match count {n}"
    src = src.replace(FAILFAST_OLD, FAILFAST_NEW)
    anchor = "start_time = time.time()"
    n = src.count(anchor)
    assert n == 1, f"start_time anchor count {n}"
    return src.replace(anchor, LOCAL_PATCH_HOOK.strip())


ns: dict = {"__name__": "__main__", "display": print,  # IPython builtin, absent outside Jupyter
            "DEV_OLD": DEV_OLD, "DEV_NEW": DEV_NEW,
            "CACHE_OLD": CACHE_OLD, "CACHE_NEW": CACHE_NEW}


def run_cell(i: int, transform=None) -> None:
    src = (CELLS / f"cell{i:02d}.py").read_text()
    src = src.replace("/kaggle/input", str(MIRROR))
    if transform:
        src = transform(src)
    print(f"\n===== cell {i:02d} =====", flush=True)
    exec(compile(src, f"cell{i:02d}", "exec"), ns)


run_cell(4)
for k, v in overrides.items():
    os.environ[k] = v
    print(f"[override] {k} = {v}", flush=True)
run_cell(5)
ns["ensure_dependencies"] = lambda *a, **k: print("[local] wheel install skipped (venv)")
run_cell(6)
run_cell(8)
# Kaggle's WORKING_DIR is absolute; locally it is "." and the predict subprocess
# runs with cwd=REPO_DIR, so the secondary-weights path must be absolutized.
os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(Path(os.environ["BIOHUB_SECONDARY_WEIGHTS"]).resolve())
print(f"[local] BIOHUB_SECONDARY_WEIGHTS -> {os.environ['BIOHUB_SECONDARY_WEIGHTS']}", flush=True)
run_cell(10, transform=transform10)
run_cell(11)
run_cell(13, transform=transform13)
run_cell(15, transform=transform15)
print(f"\n[local] DONE variant={variant} -> {RUN / 'submission.csv'}", flush=True)
