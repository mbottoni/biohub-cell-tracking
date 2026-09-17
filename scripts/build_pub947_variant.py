#!/usr/bin/env python3
"""Generate a hardened one-knob variant of the pub947 port notebook.

Row 123 postmortem: Kaggle mounts attached datasets at EITHER
/kaggle/input/datasets/<user>/<slug> OR /kaggle/input/<slug>, nondeterministically
per session (row 119 hit the same thing; tophat v1 died on it, and pub947 relink6 v1 /
capup v1 both did too). zhincez's 0.947 copy only ran because its session drew the
first layout: with the second layout its hardcoded BIOHUB_MODEL_ARTIFACTS path misses
AND its BIOHUB_TARGET_ARTIFACT_SLUG ('biohub-tracking-support-pack', reyhan's private
slug) matches no mounted dataset, so the ALLOW_ARTIFACT_FALLBACK glob picks the
alphabetically-first manifest — the SECONDARY seed — and the notebook's own sha256
guard kills the run (primary expected 12f6881e..., got 9bac2fa0...). The guard is why
a bad mount can never reach the LB silently.

Hardening = a _biohub_ds() probe that resolves a slug under either layout, applied to
every env-settable path: MODEL_ARTIFACTS, DEEPCENTER_CHECKPOINT (x3),
SECONDARY_ARTIFACT_MANIFEST, plus PRIMARY_ARTIFACT_MANIFEST and
DEEPCENTER_MANIFEST_DEFAULT (whose in-code defaults are layout-pinned).

Usage: build_pub947_variant.py <name> [<old_string> <new_string>]...
Reads kernels/pub947_port/pub947_port.ipynb, writes kernels/<name>/<name>.ipynb.
Every old_string must occur exactly once.
"""
import json, sys, os

PROBE = '''
def _biohub_ds(slug):
    import os as _o, glob as _g
    for _p in ('/kaggle/input/datasets/pilkwang/' + slug, '/kaggle/input/' + slug):
        if _o.path.isdir(_p):
            return _p
    _hits = sorted(_g.glob('/kaggle/input/**/' + slug, recursive=True))
    if _hits:
        return _hits[0]
    raise RuntimeError('dataset not found under /kaggle/input: ' + slug)

'''

SUPPORT = 'biohub-tracking-support-pack-50ep-v1'
DC = 'biohub-deepcenter-unet3d-center-prior-v1'
SEC = 'biohub-temporal-unet3d-seed314159-v1'

HARDEN = [
    ("from __future__ import annotations\nimport os\n",
     "from __future__ import annotations\nimport os\n" + PROBE, 1),
    (f"os.environ['BIOHUB_MODEL_ARTIFACTS'] = '/kaggle/input/datasets/pilkwang/{SUPPORT}'",
     f"os.environ['BIOHUB_MODEL_ARTIFACTS'] = _biohub_ds('{SUPPORT}')\n"
     f"os.environ['BIOHUB_PRIMARY_ARTIFACT_MANIFEST'] = _biohub_ds('{SUPPORT}') + '/ARTIFACT_MANIFEST.json'\n"
     f"os.environ['BIOHUB_DEEPCENTER_MANIFEST_DEFAULT'] = _biohub_ds('{DC}') + '/ARTIFACT_MANIFEST.json'", 1),
    (f"os.environ['BIOHUB_SECONDARY_ARTIFACT_MANIFEST'] = '/kaggle/input/datasets/pilkwang/{SEC}/ARTIFACT_MANIFEST.json'",
     f"os.environ['BIOHUB_SECONDARY_ARTIFACT_MANIFEST'] = _biohub_ds('{SEC}') + '/ARTIFACT_MANIFEST.json'", 1),
    (f"os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] = '/kaggle/input/datasets/pilkwang/{DC}/weights/full_frame_center/best.pt'",
     f"os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] = _biohub_ds('{DC}') + '/weights/full_frame_center/best.pt'", 3),
]


def main():
    name = sys.argv[1]
    knob_edits = list(zip(sys.argv[2::2], sys.argv[3::2]))
    nb = json.load(open('kernels/pub947_port/pub947_port.ipynb'))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 1
    src = ''.join(cells[0]['source'])
    for old, new, n in HARDEN:
        assert src.count(old) == n, f'harden: {old[:60]!r} x{src.count(old)} != {n}'
        src = src.replace(old, new)
    for old, new in knob_edits:
        assert src.count(old) == 1, f'knob: {old[:60]!r} x{src.count(old)}'
        src = src.replace(old, new)
    cells[0]['source'] = src.splitlines(keepends=True)
    os.makedirs(f'kernels/{name}', exist_ok=True)
    json.dump(nb, open(f'kernels/{name}/{name}.ipynb', 'w'))
    # post-conditions
    s2 = ''.join(c for cell in json.load(open(f'kernels/{name}/{name}.ipynb'))['cells']
                 if cell['cell_type'] == 'code' for c in cell['source'])
    assert s2.count('_biohub_ds(') == 1 + 7, s2.count('_biohub_ds(')  # def + 7 call sites
    for old, new in knob_edits:
        assert s2.count(new) >= 1 and (old in new or old not in s2)
    print(f'{name}: hardened base + {len(knob_edits)} knob edit(s); post-conditions pass')


if __name__ == '__main__':
    main()
