#!/usr/bin/env python3
"""Build pub947_rescue — EMPTY-REGION detection rescue, the last lane with a real ceiling.

THE MEASUREMENT THAT MOTIVATES IT (row 160/163 + today). Of 3,487 GT nodes on holdout6,
38 are unmatched, and ALL 38 sit in genuinely empty space: no predicted node within the
scorer's 7 µm radius. They cost 47 GT edges. Recovering every one with zero new FP is
worth **+0.0134 edge_J** — nearly 2× the 0.007 interpretability floor, and the largest
untested ceiling left on this line.

WHY THIS IS NOT THE ROUTE THAT ALREADY DIED. Row 145's "parity" lowered the detection
threshold GLOBALLY, recovered 39,398 detections, and lost 0.0157 — node_recall actually
FELL, because the new candidates crowded existing cells, stole their matches and
destabilised the graph. The metric explains exactly why, and also how to avoid it
(metrics.md): an edge FP requires a predicted edge whose endpoint MATCHES a GT node.
A node that lands far from every GT node is not an edge FP at all — it costs only the
0.1·Δnode_ratio scaling. So the safe construction is:

    accept a rescued peak ONLY if no already-accepted peak lies within RESCUE_MIN_SEP_UM

That gate is what makes this additive rather than competitive: a rescued node can never
steal a match from a node we already have, because it is never placed near one. It can
only occupy space that is currently empty — which is precisely where all 38 misses are.

WHAT IT DOES. `_detect_cells_pooled` currently keeps peaks that are BOTH a local max
under the pool AND above `det_threshold` (0.965). This adds a second, lower band: peaks
that are still genuine local maxima but sit in [RESCUE_THR, det_threshold). They are
sorted by confidence, then greedily accepted only where they are ≥ RESCUE_MIN_SEP_UM
from every accepted peak (primary first, so primaries always win).

⚠ THE OPEN QUESTION THIS EXPERIMENT ANSWERS. Row 161 established the NMS pool is not
what hides these cells (±1.625 µm suppression vs an 8.6 µm gap). So the only way a miss
is recoverable is if the heatmap has a genuine sub-threshold local maximum there. If it
does not — if the network produces no mode at all at those locations — this rescue will
add nodes elsewhere and gain nothing. The run reports `rescue_added` per frame so a
zero/ineffective outcome is visible immediately rather than inferred from the score.

KNOBS (default OFF ⇒ an unset build is byte-identical to densad):
    BIOHUB_RESCUE_THR          lower band floor, e.g. 0.80; 0 disables (default 0)
    BIOHUB_RESCUE_MIN_SEP_UM   keep a rescued peak only this far from any accepted peak
                               (default 7.0 = the scorer's own matching radius)
    BIOHUB_RESCUE_MAX_FRAC     safety cap: max rescued nodes as a fraction of primaries
                               (default 0.05; row 145 added ~2× and blew up)

    python scripts/build_pub947_rescue.py                    # sweep build, pass OFF
    python scripts/build_pub947_rescue.py --thr 0.80 --name pub947_rescue_t80
"""
import argparse
import json
import os

NB_IN = 'kernels/pub947_densad/pub947_densad.ipynb'
NAME = 'pub947_rescue'

# The notebook hex-patches the staged repo. We add one more patch, on the peak extractor.
OLD_FN = '''    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)

    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)'''

NEW_FN = '''    import os as _os
    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    _probs = torch.sigmoid(logits)
    is_peak = (logits == pooled) & (_probs > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)

    # ---- empty-region rescue (row 164): a second, lower confidence band, accepted
    # ONLY where no already-accepted peak sits within RESCUE_MIN_SEP_UM. A rescued node
    # can therefore never steal a match from a node we already have; it can only fill
    # space that is currently empty. See scripts/build_pub947_rescue.py for the metric
    # argument that makes this safe where global threshold-lowering (row 145) was not.
    _r_thr = float(_os.environ.get('BIOHUB_RESCUE_THR', '0'))
    _r_sep = float(_os.environ.get('BIOHUB_RESCUE_MIN_SEP_UM', '7.0'))
    _r_max_frac = float(_os.environ.get('BIOHUB_RESCUE_MAX_FRAC', '0.05'))

    if _r_thr > 0.0 and _r_thr < det_threshold:
        _lo = (logits == pooled) & (_probs > _r_thr) & (_probs <= det_threshold)
        _lo_idx = torch.nonzero(_lo[0, 0])

        if _lo_idx.shape[0] > 0:
            _vox = torch.tensor([float(v) for v in _VOXEL_UM_FOR_RESCUE], device = _lo_idx.device)
            _conf = _probs[0, 0][_lo_idx[:, 0], _lo_idx[:, 1], _lo_idx[:, 2]]
            _order = torch.argsort(_conf, descending = True)
            _lo_idx = _lo_idx[_order]
            _acc = peak_idx.float() * _vox if peak_idx.shape[0] > 0 else torch.empty((0, 3), device = _lo_idx.device)
            _budget = max(1, int(_r_max_frac * max(peak_idx.shape[0], 1)))
            _keep = []

            for _i in range(_lo_idx.shape[0]):
                if len(_keep) >= _budget:
                    break
                _p = _lo_idx[_i].float() * _vox

                if _acc.shape[0] > 0 and torch.min(torch.linalg.norm(_acc - _p, dim = 1)).item() < _r_sep:
                    continue
                _keep.append(_lo_idx[_i])
                _acc = torch.cat([_acc, _p.unsqueeze(0)], dim = 0)

            if _keep:
                peak_idx = torch.cat([peak_idx, torch.stack(_keep)], dim = 0)
                print('[rescue] t=' + str(t) + ' primary=' + str(int(is_peak.sum())) + ' band=' + str(int(_lo_idx.shape[0])) + ' added=' + str(len(_keep)), flush = True)

    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)'''

# The detection lattice is isotropic 1.625 um after the (1,4,4) downsample (row 159).
VOXEL_CONST = "_VOXEL_UM_FOR_RESCUE = (1.625, 1.625, 1.625)\n\n\ndef _detect_cells_pooled("

PATCH_BLOCK = '''
# ---- empty-region detection rescue patch (row 164) ----
_rs = _ps.read_text()
_rs_old = {old!r}
_rs_new = {new!r}

if _rs_old not in _rs:
    raise RuntimeError('Could not apply the empty-region rescue patch: anchor not found')
_rs = _rs.replace(_rs_old, _rs_new, 1)
_rs = _rs.replace({vox_old!r}, {vox_new!r}, 1)
compile(_rs, str(_ps), 'exec')
_ps.write_text(_rs)
print('Empty-region rescue patch applied (thr=' + os.environ.get('BIOHUB_RESCUE_THR', '0') + ')')
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--thr', type=float, default=0.0, help='bake this rescue floor in as the default')
    ap.add_argument('--sep', type=float, default=7.0)
    ap.add_argument('--max-frac', type=float, default=0.05)
    ap.add_argument('--name', default=NAME)
    a = ap.parse_args()

    nb = json.load(open(NB_IN))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 1
    src = ''.join(cells[0]['source'])

    # Append our patch right after the notebook's own last predict-script patch, so the
    # staged file exists and every earlier patch has already been applied.
    anchor = "print('Calibrated dual-seed runtime patch applied')\n"
    assert src.count(anchor) == 1, 'anchor for the rescue patch moved'
    block = PATCH_BLOCK.format(
        old=OLD_FN, new=NEW_FN,
        vox_old="def _detect_cells_pooled(", vox_new=VOXEL_CONST)
    src = src.replace(anchor, anchor + block, 1)

    if a.thr > 0.0:
        env = ("os.environ['BIOHUB_RESCUE_THR'] = '%s'\n"
               "os.environ['BIOHUB_RESCUE_MIN_SEP_UM'] = '%s'\n"
               "os.environ['BIOHUB_RESCUE_MAX_FRAC'] = '%s'\n") % (a.thr, a.sep, a.max_frac)
        marker = "os.environ['BIOHUB_DIAGNOSTIC_ARM'] = 'harmonic_association_production'\n"
        assert src.count(marker) == 1, 'env marker moved'
        src = src.replace(marker, marker + env, 1)

    compile(src, 'rescue_cell', 'exec')
    assert src.count('BIOHUB_RESCUE_THR') >= 1
    assert src.count('[rescue] t=') == 1
    cells[0]['source'] = src.splitlines(keepends=True)

    name = a.name
    os.makedirs(f'kernels/{name}', exist_ok=True)
    json.dump(nb, open(f'kernels/{name}/{name}.ipynb', 'w'))
    meta = json.load(open('kernels/pub947_densad/kernel-metadata.json'))
    meta['id'] = f'maruanbottoni/biohub-{name.replace("_", "-")}'
    meta['title'] = f'biohub {name.replace("_", " ")}'
    meta['code_file'] = f'{name}.ipynb'
    json.dump(meta, open(f'kernels/{name}/kernel-metadata.json', 'w'), indent=2)
    state = f'ARMED thr={a.thr} sep={a.sep} max_frac={a.max_frac}' if a.thr > 0 else 'rescue OFF by default'
    print(f'{name}: built ({state}) -> {meta["id"]}')


if __name__ == '__main__':
    main()
