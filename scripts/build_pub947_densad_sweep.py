#!/usr/bin/env python3
"""Build pub947_densad_sweep — the densad 0.948 base with its density table made
env-overridable, for the local reject-gate micro-grid (PLAN option 2, row 143 gate).

densad's 12 table numbers + 2 group boundaries were copied VERBATIM from haideptry
and never tuned on this base (only hidense/row 135 tested the high-group bonus, found
saturated). This variant reads each from an env var (defaulting to the exact shipped
densad value), so `pub947_local_replay.py` can sweep one knob per run — all POST-PROC
(motion-relink runs in filter_output_graph, after the cached predict_video), so every
variant is CACHE-HIT: minutes, not the 80-min inference pass.

Reject-gate only (local <0.007 uninterpretable, §50.2/row 90): a variant that beats
base 0.9525 EARNS one LB slot, it does not confirm. holdout6 exercises middle (4 vols)
+ low (2 vols); HIGH group is untestable locally AND already LB-saturated (row 135).

Byte-identity guarantee: with NO env overrides the emitted table is numerically the
densad table, so an unset sweep == the row-143 baseline (the sweep's own control arm).
"""
import json, os

NB_IN = 'kernels/pub947_densad/pub947_densad.ipynb'
NAME = 'pub947_densad_sweep'

OLD = (
    "DENSITY_GROUP_OVERRIDES = {\n"
    "    'low': {'tight_um': 7.25, 'relaxed_um': 11.0, 'velocity_weight': 0.5, 'learned_bonus': 3.0},\n"
    "    'middle': {'tight_um': 6.5, 'relaxed_um': 9.0, 'velocity_weight': 0.0, 'learned_bonus': 6.0},\n"
    "    'high': {'tight_um': 5.5, 'relaxed_um': 10.0, 'velocity_weight': 0.5, 'learned_bonus': 1.0},\n"
    "}\n"
    "\n"
    "def determine_density_group(nodes_by_id):\n"
    "    if not nodes_by_id:\n"
    "        return 'middle'\n"
    "    times = [int(n['t']) for n in nodes_by_id.values()]\n"
    "    avg = len(nodes_by_id) / max(len(set(times)), 1)\n"
    "    return 'low' if avg < 120 else ('middle' if avg < 400 else 'high')\n"
)

NEW = (
    "def _densad_env(_k, _d):\n"
    "    return float(os.environ.get('BIOHUB_DENSAD_' + _k, str(_d)))\n"
    "DENSITY_GROUP_OVERRIDES = {\n"
    "    'low': {'tight_um': _densad_env('LOW_TIGHT', 7.25), 'relaxed_um': _densad_env('LOW_RELAXED', 11.0), 'velocity_weight': _densad_env('LOW_VEL', 0.5), 'learned_bonus': _densad_env('LOW_BONUS', 3.0)},\n"
    "    'middle': {'tight_um': _densad_env('MID_TIGHT', 6.5), 'relaxed_um': _densad_env('MID_RELAXED', 9.0), 'velocity_weight': _densad_env('MID_VEL', 0.0), 'learned_bonus': _densad_env('MID_BONUS', 6.0)},\n"
    "    'high': {'tight_um': _densad_env('HIGH_TIGHT', 5.5), 'relaxed_um': _densad_env('HIGH_RELAXED', 10.0), 'velocity_weight': _densad_env('HIGH_VEL', 0.5), 'learned_bonus': _densad_env('HIGH_BONUS', 1.0)},\n"
    "}\n"
    "_DENSAD_LOW_MAX = _densad_env('LOW_MAX', 120.0)\n"
    "_DENSAD_MID_MAX = _densad_env('MID_MAX', 400.0)\n"
    "print('[densad_sweep] table = ' + json.dumps(DENSITY_GROUP_OVERRIDES, sort_keys=True) + ' | bounds ' + str(_DENSAD_LOW_MAX) + '/' + str(_DENSAD_MID_MAX), flush=True)\n"
    "\n"
    "def determine_density_group(nodes_by_id):\n"
    "    if not nodes_by_id:\n"
    "        return 'middle'\n"
    "    times = [int(n['t']) for n in nodes_by_id.values()]\n"
    "    avg = len(nodes_by_id) / max(len(set(times)), 1)\n"
    "    return 'low' if avg < _DENSAD_LOW_MAX else ('middle' if avg < _DENSAD_MID_MAX else 'high')\n"
)


def main():
    nb = json.load(open(NB_IN))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 1
    src = ''.join(cells[0]['source'])
    n = src.count(OLD)
    assert n == 1, f'densad table anchor: {n} matches (expected 1)'
    src = src.replace(OLD, NEW, 1)
    # `json` is imported later in the cell than this block; ensure it's available here.
    assert 'import json' in src
    compile(src, 'sweep_cell', 'exec')
    cells[0]['source'] = src.splitlines(keepends=True)
    os.makedirs(f'kernels/{NAME}', exist_ok=True)
    json.dump(nb, open(f'kernels/{NAME}/{NAME}.ipynb', 'w'))
    print(f'{NAME}: built (densad table env-parameterized, 14 knobs, defaults = shipped densad)')


if __name__ == '__main__':
    main()
