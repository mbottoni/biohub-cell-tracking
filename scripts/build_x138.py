#!/usr/bin/env python3
"""Build the two x138 kernels (row 167): a verbatim control and x138 + densad.

x138 (anvithpothula/biohub-x138, pulled 09-23 into census_0923/) is the only public
kernel plausibly above 0.947 (row 166): the #1 result of the competition's
scoreDescending kernel sort, author rank 52 at 0.956. Its own score cannot be read
via the API. It is NOT on our reyhan-0.947 base: it forks the harmonic-v3 / "0.939"
substrate and adds a neighbourhood-flow motion prior in the relinker, readmission of
discarded detections near open track ends, and a sub-threshold-peak gap filler.

pub947x_x138 — VERBATIM. No hardening needed: x138 resolves every checkpoint under
  both mount layouts itself and sha256-guards primary/secondary/DeepCenter, so a bad
  mount fails loud (the row-123 failure mode cannot reach the LB silently).

pub947x_x138_densad — x138 + our only LB-confirmed gain on the 947 line, the
  density-adaptive relink table (row 127, 0.947 -> 0.948), grafted verbatim at the
  relink callsite. ⚠ NOT a clean transplant: in x138's flow mode the tight pass gate
  is BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM (7.0, fixed), so the table's tight_um lands on
  the flow SEED gate (x138: 5.5) instead of the tight gate; relaxed_um and
  learned_bonus act as in densad; velocity_weight only acts on frames with no flow
  field. Hence the verbatim control: without it a flat combo is unattributable.
"""
import json, os

SRC = 'census_0923/anvithpothula_biohub-x138/biohub-x138.ipynb'

DENSAD_BLOCK = '''

# ============ Density-adaptive motion-relink (haideptry table, row 127) ===========
DENSITY_GROUP_OVERRIDES = {
    'low': {'tight_um': 7.25, 'relaxed_um': 11.0, 'velocity_weight': 0.5, 'learned_bonus': 3.0},
    'middle': {'tight_um': 6.5, 'relaxed_um': 9.0, 'velocity_weight': 0.0, 'learned_bonus': 6.0},
    'high': {'tight_um': 5.5, 'relaxed_um': 10.0, 'velocity_weight': 0.5, 'learned_bonus': 1.0},
}

def determine_density_group(nodes_by_id):
    if not nodes_by_id:
        return 'middle'
    times = [int(n['t']) for n in nodes_by_id.values()]
    avg = len(nodes_by_id) / max(len(set(times)), 1)
    return 'low' if avg < 120 else ('middle' if avg < 400 else 'high')
# ==================================================================================

'''
READ_FRAME_ANCHOR = 'def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:'
CALLSITE_OLD = ('        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)\n'
                '        if motion_edges and READMIT_RADIUS_UM > 0:')
CALLSITE_NEW = ('''        _dg = determine_density_group(nodes_by_id)
        _dg_cfg = DENSITY_GROUP_OVERRIDES[_dg]
        globals()['MOTION_RELINK_TIGHT_UM'] = _dg_cfg['tight_um']
        globals()['MOTION_RELINK_RELAXED_UM'] = _dg_cfg['relaxed_um']
        globals()['MOTION_RELINK_VELOCITY_WEIGHT'] = _dg_cfg['velocity_weight']
        globals()['MOTION_RELINK_LEARNED_BONUS'] = _dg_cfg['learned_bonus']
        print(f'[densad] {dataset}: density group {_dg} -> {_dg_cfg}', flush=True)
''' + CALLSITE_OLD)


def build(name, edits):
    nb = json.load(open(SRC))
    n_code = sum(c['cell_type'] == 'code' for c in nb['cells'])
    for old, new in edits:
        hits = [i for i, c in enumerate(nb['cells']) if old in ''.join(c['source'])]
        assert len(hits) == 1, f'{name}: {old[:60]!r} in cells {hits}'
        s = ''.join(nb['cells'][hits[0]]['source'])
        assert s.count(old) == 1, f'{name}: {old[:60]!r} x{s.count(old)}'
        nb['cells'][hits[0]]['source'] = s.replace(old, new).splitlines(keepends=True)
    for c in nb['cells']:  # outputs from the author's run must not ride along
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    for i, c in enumerate(nb['cells']):  # every cell must still parse
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'{name}[cell {i}]', 'exec')
    os.makedirs(f'kernels/{name}', exist_ok=True)
    json.dump(nb, open(f'kernels/{name}/{name}.ipynb', 'w'))
    meta = json.load(open('kernels/pub947_densad/kernel-metadata.json'))
    meta['id'] = f"maruanbottoni/biohub-{name.replace('_', '-')}"
    meta['title'] = f"biohub {name.replace('_', ' ')}"
    meta['code_file'] = f'{name}.ipynb'
    json.dump(meta, open(f'kernels/{name}/kernel-metadata.json', 'w'), indent=2)
    print(f'{name}: {n_code} code cells, {len(edits)} edit(s), compiles, datasets={meta["dataset_sources"]}')


def main():
    build('pub947x_x138', [])
    build('pub947x_x138_densad', [
        (READ_FRAME_ANCHOR, DENSAD_BLOCK + READ_FRAME_ANCHOR),
        (CALLSITE_OLD, CALLSITE_NEW),
    ])


if __name__ == '__main__':
    main()
