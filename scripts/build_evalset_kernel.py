#!/usr/bin/env python3
"""STRATEGY 23.2 -- wrap any existing kernel so it scores a WIDE eval set.

The instrument this project has used for every local decision is the four
`test/` volumes: 2,127 of the 128,883 GT edges we own (1.65%) and 3 of the 151
divisions (2%). Worse, it is a biased 1.65% -- 1.8x denser in cells and 2x
thinner in annotation than the corpus median (23.2).

The shipped kernel already supports the fix and nobody used it. Cell 5:

    _test_dir_override = os.environ.get("BIOHUB_TEST_DIR", "").strip()
    TEST_DIR = Path(_test_dir_override) if _test_dir_override else COMP_DIR / "test"

and `list_test_stems()` simply globs `TEST_DIR/*.zarr`, while `read_test_frame`
reads `TEST_DIR/{dataset}.zarr`. Both are read-only operations, so a symlink farm
in /kaggle/working pointing into the read-only train mount works. (The Errno 30
in exp 19 was a *write* through such a symlink, not a read.)

So: link the 24 volumes chosen by `select_evalset.py` into
/kaggle/working/evalset, export BIOHUB_TEST_DIR, and the entire pipeline runs
unchanged over 7.8x the GT edges and 8.3x the divisions -- for one commit run and
zero submissions.

ONE CHANGE ONLY. Every other env var is left byte-identical to the base kernel,
so the four dev volumes inside the wide set must reproduce the numbers we already
hold for that config. If they do not, the instrument is wrong and no result built
on it counts. That reproduction is the acceptance test, not a formality.

⚠ THIS KERNEL MUST NEVER BE SUBMITTED. Its submission.csv covers train volumes,
not the real test set. The patch raises rather than falling back to `test/` if
the train mount is absent, so a misconfigured run dies instead of silently
producing a real-looking submission.

Run: python scripts/build_evalset_kernel.py d4edge [edgethr30 ...]
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVALSET = REPO / "configs/evalset.json"

ANCHOR = 'print("BIOHUB_PRESET:", BIOHUB_PRESET)'

PATCH = '''
# ---------------------------------------------------------------------------
# STRATEGY 23.2 -- widen the eval set from 4 volumes to {n}.
#
# The four `test/` volumes are 1.65% of the GT we own and are 1.8x denser / 2x
# less annotated than the corpus. Every thin-volume pathology in the log (19.2's
# "80% one edge", 20.3's 3-division statistic, 22.1's 2x robustness bug) is a
# consequence of measuring there. These {n} train volumes are stratified on
# est_n_nodes x embryo and carry {edges:,} GT edges and {divisions} divisions.
#
# The four dev volumes are included so this run reproduces the numbers already
# recorded for this config. That reproduction is the acceptance test.
# ---------------------------------------------------------------------------
_EVAL_STEMS = {stems!r}

from pathlib import Path as _Path

_COMPETITION = "biohub-cell-tracking-during-development"
_comp = next(
    (p for p in (_Path(f"/kaggle/input/competitions/{{_COMPETITION}}"),
                 _Path(f"/kaggle/input/{{_COMPETITION}}")) if p.exists()),
    None,
)
_train = _comp / "train" if _comp is not None else None
if _train is None or not _train.exists():
    # Refuse to fall back to the real `test/` dir: a silent fallback would
    # produce a genuine-looking submission.csv from an experiment that was
    # never meant to be scored.
    raise RuntimeError(f"evalset: train mount not found under {{_comp}}; refusing to run")

_dst = _Path("/kaggle/working/evalset")
_dst.mkdir(parents=True, exist_ok=True)
_missing = []
for _stem in _EVAL_STEMS:
    _src = _train / f"{{_stem}}.zarr"
    if not _src.exists():
        _missing.append(_stem)
        continue
    _link = _dst / f"{{_stem}}.zarr"
    if not _link.exists():
        _link.symlink_to(_src, target_is_directory=True)
if _missing:
    raise RuntimeError(f"evalset: {{len(_missing)}} stems absent from train mount: {{_missing[:5]}}")
_linked = sorted(p.name[:-5] for p in _dst.glob("*.zarr"))
if _linked != sorted(_EVAL_STEMS):
    raise RuntimeError(f"evalset: linked {{len(_linked)}} != expected {{len(_EVAL_STEMS)}}")

os.environ["BIOHUB_TEST_DIR"] = str(_dst)
print(f"[evalset] BIOHUB_TEST_DIR={{_dst}} with {{len(_linked)}} volumes")
print("[evalset] DO NOT SUBMIT -- this scores train volumes, not the real test set")
'''


def slug(title: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def build(base: str) -> None:
    src_nb = REPO / "kernels" / base / f"{base}.ipynb"
    if not src_nb.exists():
        raise SystemExit(f"no such base kernel: {src_nb}")
    cfg = json.loads(EVALSET.read_text())
    stems = cfg["stems"]

    nb = json.loads(src_nb.read_text())
    cell = "".join(nb["cells"][4]["source"])
    if cell.count(ANCHOR) != 1:
        raise RuntimeError(f"{base}: anchor count = {cell.count(ANCHOR)}, expected 1")
    patch = PATCH.format(n=len(stems), stems=stems,
                         edges=cfg["gt_edges"], divisions=cfg["gt_divisions"])
    # Insert BEFORE the preset print so BIOHUB_TEST_DIR is exported well ahead of
    # cell 5, which snapshots it into TEST_DIR at definition time.
    cell = cell.replace(ANCHOR, patch + "\n" + ANCHOR, 1)
    nb["cells"][4]["source"] = cell.splitlines(keepends=True)

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                raise RuntimeError(f"cell {i} no longer parses: {e}") from e

    name = f"evalset_{base}"
    title = f"Biohub evalset {base}"
    out = REPO / "kernels" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))

    meta = json.loads((REPO / "kernels" / base / "kernel-metadata.json").read_text())
    meta["id"] = f"maruanbottoni/{slug(title)}"
    meta["title"] = title
    meta["code_file"] = f"{name}.ipynb"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote kernels/{name}  ->  {meta['id']}")
    print(f"  base={base}  volumes={len(stems)}  gt_edges={cfg['gt_edges']:,}  "
          f"divisions={cfg['gt_divisions']}")


if __name__ == "__main__":
    bases = sys.argv[1:] or ["d4edge"]
    for b in bases:
        build(b)
