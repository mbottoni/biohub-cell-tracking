#!/usr/bin/env python
"""Local replay of the pub947 family (the 0.947/0.948 line, rows 122-131) — the
reject-gate instrument for everything on the densad base (PLAN §0.1 item 2).

Unlike pub162/pub920 (cell-split notebooks) the pub947 family is ONE code cell that
stages the pilkwang repo itself and launches `scripts/predict_unet_transformer.py`
as a subprocess (sharded on 2 GPUs on Kaggle; a built-in single-process branch fires
automatically when torch.cuda.device_count() == 0 — locally that is us). So the
driver exec()s the single cell with source-level transforms:

  T1/T2  /kaggle/input -> mirror, /kaggle/working -> run dir
  T3     IPython display shim
  T4     CUDA fail-fast raise -> print (single-process MPS/CPU branch takes over)
  T5     ensure_dependencies(ARTIFACTS) -> skipped (venv already has the deps)
  T6     in-cell DeepCenter device line (single-quoted!) -> cuda/mps/cpu chain
  T7     LOCAL hook at the `start_time = time.time()` anchor (INDENTED - it lives
         inside `with _capture_stage_output(...)` - the hook is re-indented to
         match): patches the FRESHLY STAGED repo's predict script with the MPS
         device chain + the (coords, edges) pickle pred-cache (row 67 semantics:
         one cache dir per inference-affecting config, NEVER shared across them)
  T8     hardcoded `os.environ['K'] = 'v'` assignments of overridden knobs
         rewritten at source level (row 82: cells clobber env-level overrides)
  T9     the notebook's own guard tables: expected values rewritten for
         intentionally overridden knobs so the guard still catches everything else

Durable scratch (rows 67/67b: /private/tmp is GC'd and ATE the assets once):
everything lives under /Volumes/extSSD/kaggle/biohub/replay947/.

The notebook's own resume machinery (biohub_live_resume/ in the run dir) makes an
interrupted overnight run restartable: re-invoke the same variant, completed stages
are skipped. Validators are OFF by default locally (BIOHUB_VALIDATOR_ENABLE=0):
they would predict held-out TRAIN volumes we do not need — we score with
scripts/local/score_official.py instead. Local scores may REJECT, never CONFIRM.

Usage:
  pub947_local_replay.py <variant> ['{"BIOHUB_...": "value", ...}']
Env:
  BIOHUB_REPLAY_NB        notebook to replay (default kernels/pub947_densad/...)
  BIOHUB_REPLAY_TEST_DIR  volume dir (default /Volumes/extSSD/kaggle/biohub/holdout6)
  BIOHUB_PRED_CACHE       pickle cache dir (default <scratch>/cache_densad)
  BIOHUB_MPS              0 to force CPU inside predict (default MPS when available)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S = Path(os.environ.get("BIOHUB_REPLAY_SCRATCH", "/Volumes/extSSD/kaggle/biohub/replay947"))
NB = Path(os.environ.get("BIOHUB_REPLAY_NB", str(REPO / "kernels/pub947_densad/pub947_densad.ipynb")))
TEST_DIR = Path(os.environ.get("BIOHUB_REPLAY_TEST_DIR", "/Volumes/extSSD/kaggle/biohub/holdout6"))
ASSETS = Path("/Volumes/extSSD/kaggle/biohub/replay_assets")
MIRROR = S / "kaggle_input"

variant = sys.argv[1]
overrides: dict[str, str] = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

# ---- mirror (idempotent; both Kaggle mount layouts, matching _biohub_ds) ----
PILK = MIRROR / "datasets" / "pilkwang"
PILK.mkdir(parents=True, exist_ok=True)
(MIRROR / "competitions").mkdir(exist_ok=True)
_links = [
    (PILK / "biohub-tracking-support-pack-50ep-v1", ASSETS / "support_pack"),
    (PILK / "biohub-temporal-unet3d-seed314159-v1", ASSETS / "biohub-temporal-unet3d-seed314159-v1"),
    (PILK / "biohub-deepcenter-unet3d-center-prior-v1", ASSETS / "deepcenter_dl"),
    (MIRROR / "biohub-deepcenter-unet3d-center-prior-v1", ASSETS / "deepcenter_dl"),
    (MIRROR / "competitions" / "biohub-cell-tracking-during-development", None),  # dir, test link inside
]
for link, target in _links[:-1]:
    assert target.is_dir(), f"[local] asset missing: {target}"
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        link.unlink()
    link.symlink_to(target)
_comp = MIRROR / "competitions" / "biohub-cell-tracking-during-development"
_comp.mkdir(exist_ok=True)
_test_link = _comp / "test"
if _test_link.is_symlink() or _test_link.exists():
    _test_link.unlink()
assert TEST_DIR.is_dir() and list(TEST_DIR.glob("*.zarr")), f"[local] no volumes in {TEST_DIR}"
_test_link.symlink_to(TEST_DIR)
print(f"[local] mirror ok; test -> {TEST_DIR} ({len(list(TEST_DIR.glob('*.zarr')))} volumes)", flush=True)

RUN = S / "runs" / variant
RUN.mkdir(parents=True, exist_ok=True)
os.chdir(RUN)
print(f"[local] pub947 variant={variant} nb={NB.name} run_dir={RUN}", flush=True)

os.environ.setdefault("BIOHUB_PRED_CACHE", str(S / "cache_densad"))
os.environ.setdefault("BIOHUB_VALIDATOR_ENABLE", "0")  # score locally with the official scorer instead
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
Path(os.environ["BIOHUB_PRED_CACHE"]).mkdir(parents=True, exist_ok=True)
print(f"[local] pred cache = {os.environ['BIOHUB_PRED_CACHE']} "
      f"(row 67: never share across inference-affecting configs)", flush=True)

# ---- staged-repo patch payloads (anchors verified against support_pack/repo) ----
DEV_OLD = '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
DEV_NEW = '''    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and os.environ.get("BIOHUB_MPS", "1") != "0":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[local] inference device: {device}", flush=True)'''

# The notebook's own runtime patches (D4-TTA / dual-seed / edge-TTA) rewrite the
# predict_video CALL SITE, so the cache is a function-level wrapper instead: rename
# the (already notebook-patched) def and shim a memoizing predict_video before main.
CACHE_DEF_OLD = "def predict_video("
CACHE_DEF_NEW = "def _predict_video_uncached("
CACHE_MAIN_ANCHOR = "def main() -> None:"
CACHE_WRAPPER = '''def predict_video(model, ds_path, device, **kw):
    import os as _os, pickle as _pickle
    _c = _os.environ.get("BIOHUB_PRED_CACHE", "")
    _f = Path(_c) / (ds_path.stem + ".pkl") if _c else None
    if _f is not None and _f.exists():
        print(f"[cache] loaded {ds_path.stem}", flush=True)
        return _pickle.loads(_f.read_bytes())
    _r = _predict_video_uncached(model, ds_path, device, **kw)
    if _f is not None:
        _f.write_bytes(_pickle.dumps(tuple(_r)))
        print(f"[cache] saved {ds_path.stem}", flush=True)
    return _r


def main() -> None:'''

# Re-indented to the anchor's depth at inject time (T7).
HOOK_LINES = '''_lp = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_lt = _lp.read_text()
if "BIOHUB_PRED_CACHE" in _lt:
    print("[local] staged repo already patched (resume)", flush=True)
else:
    for _lo, _ln, _tag in ((DEV_OLD, DEV_NEW, "device"),
                           (CACHE_DEF_OLD, CACHE_DEF_NEW, "cache-def"),
                           (CACHE_MAIN_ANCHOR, CACHE_WRAPPER, "cache-wrapper")):
        _n = _lt.count(_lo)
        assert _n == 1, f"[local] {_tag} patch expected 1 match, found {_n}"
        _lt = _lt.replace(_lo, _ln)
    _lp.write_text(_lt)
    print("[local] device + cache-wrapper patches applied to staged repo", flush=True)
start_time = time.time()'''

CUDA_RAISE = ("raise RuntimeError('CUDA GPU is required for this notebook. "
              "Enable a Kaggle GPU accelerator and commit again.')")
DC_DEV_OLD = "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
DC_DEV_NEW = ("device = torch.device('cuda' if torch.cuda.is_available() else "
              "('mps' if torch.backends.mps.is_available() and "
              "os.environ.get('BIOHUB_MPS', '1') != '0' else 'cpu'))")

nb = json.loads(NB.read_text())
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
assert len(cells) == 1, f"expected the single-cell pub947 layout, got {len(cells)}"
src = "".join(cells[0]["source"])

def swap(s: str, old: str, new: str, tag: str, n: int = 1) -> str:
    found = s.count(old)
    assert found == n, f"[local] transform {tag}: expected {n} match(es), found {found}"
    return s.replace(old, new)

src = src.replace("/kaggle/input", str(MIRROR))          # T1
src = src.replace("/kaggle/working", str(RUN))           # T2
src = swap(src, "from IPython.display import display",
           "pass  # [local] display provided by driver", "T3-display")
src = swap(src, CUDA_RAISE, "print('[local] CUDA fail-fast bypassed; "
           "single-process MPS/CPU predict branch will fire')", "T4-failfast")
src = swap(src, "print('CUDA device:', _torch.cuda.get_device_name(0))",
           "print('[local] device:', 'mps' if _torch.backends.mps.is_available() else 'cpu')",
           "T4b-devname")
src = swap(src, "ensure_dependencies(ARTIFACTS)",
           "print('[local] wheel install skipped (venv)')", "T5-deps")
src = swap(src, DC_DEV_OLD, DC_DEV_NEW, "T6-dcdevice")

# T7 — inject the staged-repo patch hook at the (indented) start_time anchor
_anchor = "start_time = time.time()"
_m = [m for m in re.finditer(r"^([ \t]*)" + re.escape(_anchor) + r"$", src, re.M)]
assert len(_m) == 1, f"[local] T7: start_time anchor count {len(_m)}"
_indent = _m[0].group(1)
_hook = "\n".join(_indent + l if l else l for l in HOOK_LINES.split("\n"))
src = src[:_m[0].start()] + _hook + src[_m[0].end():]

# T8 — hardcoded env assignments of overridden knobs (row 82 discipline)
for k, v in overrides.items():
    pat = re.compile(r'(os\.environ\[["\']%s["\']\]\s*=\s*)(["\'][^"\']*["\'])' % re.escape(k))
    src, n = pat.subn(lambda m, _v=v: m.group(1) + repr(str(_v)), src)
    if n:
        print(f"[override] {n} hardcoded assignment(s) of {k} rewritten to {v!r}", flush=True)

# T9 — guard expected-value rewrites for intentional overrides
for k, v in overrides.items():
    pat = re.compile(r'("%s":\s*)([^,\n]+)(,?)' % re.escape(k))
    def repl(m, _v=v):
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
    if n:
        print(f"[guard] {k}: {n} expected-value rewrite(s)", flush=True)

for k, v in overrides.items():
    os.environ[k] = v
    print(f"[override] {k} = {v}", flush=True)

ns: dict = {"__name__": "__main__", "display": print,
            "DEV_OLD": DEV_OLD, "DEV_NEW": DEV_NEW,
            "CACHE_DEF_OLD": CACHE_DEF_OLD, "CACHE_DEF_NEW": CACHE_DEF_NEW,
            "CACHE_MAIN_ANCHOR": CACHE_MAIN_ANCHOR, "CACHE_WRAPPER": CACHE_WRAPPER}
print("[local] transforms complete; executing the cell", flush=True)
exec(compile(src, f"pub947cell_{variant}", "exec"), ns)

for k, v in overrides.items():
    assert os.environ.get(k) == v, (
        f"[local] override {k} did not survive: want {v!r}, got {os.environ.get(k)!r}")
print(f"\n[local] DONE variant={variant} -> {RUN / 'submission.csv'}", flush=True)
