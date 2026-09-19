#!/usr/bin/env python3
"""Build pub947_parity — per-frame detection node-count parity on the densad 0.948 base.

Mechanism (ONE change vs 56313419 / densad): the 0.947 line's frozen frame-retention
guard is a BINARY cliff — when probability-averaged dual detection kills >10% of the
solo detector's candidates in a frame, the whole frame falls back to primary-only.
hiranorm's LB-logged series (row 110) showed the killed band is recoverable: lowering
the FUSED threshold to node-count parity with solo scored +0.0185 on their holdout,
and 60% of their recoverable edges failed on one-sided detection. det096 (row 130)
tested a UNIFORM lower threshold and came back flat — parity is per-frame calibrated,
which the flat uniform read does not falsify, and rides the densad lesson (row 131:
conditioning is where the remaining signal lives).

Implementation: replace the guard's binary det_logits assignment with a per-frame
binary search for the largest fused threshold t* in [floor=0.90, det_threshold] whose
blended candidate count reaches the solo count (parity, never past it — node-ratio is
SIGNED, row 110). A logit shift makes the downstream detection call (which uses
cfg.det_threshold) cut at t* without touching any other consumer — det_logits feeds
ONLY _detect_cells_pooled (verified: staged repo line 560 is the sole reader).
use_primary_detection is forced off on parity frames: the cliff IS what this replaces.

Fail-loud: patch application count-asserted and compile()d in-kernel, per-frame
BIOHUB_PARITY json lines + the 'parity patch applied' print are the mechanism-fired
pre-read signals (row 140b standing rule) — read them from the commit log BEFORE
submitting, and compare per-volume node counts against the consensus commit's base
variant (row 140 table: 25609/20719/6170/70290 nodes).

Env knobs: BIOHUB_DET_PARITY (default 1), BIOHUB_DET_PARITY_FLOOR (default 0.90).
"""
import json, os

NB_IN = 'kernels/pub947_densad/pub947_densad.ipynb'
NAME = 'pub947_parity'

ANCHOR = ("print('Frozen frame retention guard applied at ' + "
          "os.environ['BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION'])")

PARITY_BLOCK = '''

# ---- Detection node-count parity (hiranorm row 110, per-frame; edges lane) ----
_s = _ps.read_text()
_parity_old = "                    det_logits[f] = primary_det if use_primary_detection else blended_det"
_parity_new = _patch_text("""_parity_on = os.environ.get('BIOHUB_DET_PARITY', '1') != '0'
if _parity_on and primary_candidates > 0 and blended_candidates < primary_candidates:
    _p_thr0 = float(cfg.det_threshold)
    _p_lo = float(os.environ.get('BIOHUB_DET_PARITY_FLOOR', '0.90'))
    _p_hi = _p_thr0
    for _p_i in range(12):
        _p_mid = 0.5 * (_p_lo + _p_hi)
        if len(_detect_cells_pooled(blended_det[0], int(frame_indices[f]), _p_mid, pool_k)) >= primary_candidates:
            _p_lo = _p_mid
        else:
            _p_hi = _p_mid
    import math as _p_math
    _p_shift = _p_math.log(_p_thr0 / (1.0 - _p_thr0)) - _p_math.log(_p_lo / (1.0 - _p_lo))
    det_logits[f] = blended_det + _p_shift
    use_primary_detection = False
    _p_after = len(_detect_cells_pooled(det_logits[f][0], int(frame_indices[f]), _p_thr0, pool_k))
    print('BIOHUB_PARITY ' + json.dumps({'dataset': ds_path.stem, 'frame': int(frame_indices[f]), 'primary': int(primary_candidates), 'blended': int(blended_candidates), 'tstar': round(_p_lo, 5), 'after': int(_p_after)}, sort_keys = True), flush = True)
else:
    det_logits[f] = primary_det if use_primary_detection else blended_det""", indent = 20, trailing_newline = False)
_parity_matches = _s.count(_parity_old)

if _parity_matches != 1:
    raise RuntimeError(f'Parity patch expected one det_logits assignment, found {_parity_matches}')

_s = _s.replace(_parity_old, _parity_new, 1)
compile(_s, str(_ps), 'exec')
_ps.write_text(_s)
print('Detection node-count parity patch applied (floor=' + os.environ.get('BIOHUB_DET_PARITY_FLOOR', '0.90') + ')')
'''


def main():
    nb = json.load(open(NB_IN))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 1
    src = ''.join(cells[0]['source'])
    n = src.count(ANCHOR)
    assert n == 1, f'retention-guard print anchor: {n} matches'
    src = src.replace(ANCHOR, ANCHOR + PARITY_BLOCK, 1)
    cells[0]['source'] = src.splitlines(keepends=True)
    os.makedirs(f'kernels/{NAME}', exist_ok=True)
    json.dump(nb, open(f'kernels/{NAME}/{NAME}.ipynb', 'w'))
    meta = json.load(open('kernels/pub947_port/kernel-metadata.json'))
    meta['id'] = f"maruanbottoni/biohub-{NAME.replace('_', '-')}"
    meta['title'] = f"biohub {NAME.replace('_', ' ')}"
    meta['code_file'] = f'{NAME}.ipynb'
    json.dump(meta, open(f'kernels/{NAME}/kernel-metadata.json', 'w'), indent=2)
    print(f'{NAME}: built on densad base (anchor x1, parity block inserted)')


if __name__ == '__main__':
    main()
