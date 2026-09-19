#!/usr/bin/env python3
"""Build pub947_densad_tuned — densad base with the two micro-grid-tuned knobs baked in.

Row 146/147: the densad-table sweep found exactly two live, DISJOINT-volume levers
(low group = 2 vols, middle = 4 vols, so additive):
  low  tight_um 7.25 -> 6.5  (+0.0007 local; peaks at 6.5, 6.0 reverses negative)
  mid  learned_bonus 6.0 -> 8.0  (+0.0006 local; plateaus 8/10/12, 8.0 = least aggressive)
Combo verified local 0.9538 vs base 0.9525 (+0.0013 = exact additive sum). Below the
0.007 LB-resolution floor (row 90) so NOT a confirmed gain — the honest read is
"best config found, zero-risk aggressive final-2 candidate". All other table knobs
were flat-to-negative (radii optimal at shipped, low_bonus flat).

Two exact-once literal edits on the shipped densad table. Local-gate before Kaggle
(row 144): the tuned kernel must reproduce 0.9538 via pub947_local_replay.py (cache-hit)
before any push.
"""
import json, os

NB_IN = 'kernels/pub947_densad/pub947_densad.ipynb'
NAME = 'pub947_densad_tuned'

EDITS = [
    ("'low': {'tight_um': 7.25, 'relaxed_um': 11.0, 'velocity_weight': 0.5, 'learned_bonus': 3.0}",
     "'low': {'tight_um': 6.5, 'relaxed_um': 11.0, 'velocity_weight': 0.5, 'learned_bonus': 3.0}"),
    ("'middle': {'tight_um': 6.5, 'relaxed_um': 9.0, 'velocity_weight': 0.0, 'learned_bonus': 6.0}",
     "'middle': {'tight_um': 6.5, 'relaxed_um': 9.0, 'velocity_weight': 0.0, 'learned_bonus': 8.0}"),
]


def main():
    nb = json.load(open(NB_IN))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 1
    src = ''.join(cells[0]['source'])
    for old, new in EDITS:
        assert src.count(old) == 1, f'edit anchor x{src.count(old)}: {old[:50]!r}'
        src = src.replace(old, new)
    compile(src, 'tuned_cell', 'exec')
    cells[0]['source'] = src.splitlines(keepends=True)
    os.makedirs(f'kernels/{NAME}', exist_ok=True)
    json.dump(nb, open(f'kernels/{NAME}/{NAME}.ipynb', 'w'))
    meta = json.load(open('kernels/pub947_port/kernel-metadata.json'))
    meta['id'] = f"maruanbottoni/biohub-{NAME.replace('_', '-')}"
    meta['title'] = f"biohub {NAME.replace('_', ' ')}"
    meta['code_file'] = f'{NAME}.ipynb'
    json.dump(meta, open(f'kernels/{NAME}/kernel-metadata.json', 'w'), indent=2)
    print(f'{NAME}: built (low tight 7.25->6.5, mid bonus 6.0->8.0; 2 edits, compiles)')


if __name__ == '__main__':
    main()
