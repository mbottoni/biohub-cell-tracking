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


def build(name, edits, extra=()):
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
    meta['dataset_sources'] = meta['dataset_sources'] + [d for d in extra if d not in meta['dataset_sources']]
    json.dump(meta, open(f'kernels/{name}/kernel-metadata.json', 'w'), indent=2)
    print(f'{name}: {n_code} code cells, {len(edits)} edit(s), compiles, datasets={meta["dataset_sources"]}')


def main():
    # ☠ Row 169: x138 as published CANNOT RUN outside its author's account. Cell 5 loads
    # biohub-v1284-head-s075/v1284_head.pt, a coordinate-refinement MLP the author trained
    # on 20 train movies, from a PRIVATE dataset (the '' entry in its metadata). Our v1 of
    # pub947x_x138 died on the mount check at ~6 min. The two builds below are kept for
    # the record; they error the same way.
    build('pub947x_x138', [], extra=['anvithpothula/biohub-v1284-head-s075'])
    build('pub947x_x138_densad', [
        (READ_FRAME_ANCHOR, DENSAD_BLOCK + READ_FRAME_ANCHOR),
        (CALLSITE_OLD, CALLSITE_NEW),
    ], extra=['anvithpothula/biohub-v1284-head-s075'])
    # Portable form: head OFF via the module's own V1284_MODE='zero' (refine() returns the
    # detections unchanged; integer coords through the trilinear index reproduce the native
    # gather exactly, per its docstring). Everything else in x138 stays on.
    HEAD_OFF = [
        ("_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-s075/v1284_head.pt'))\n"
         "if len(_myhead) != 1:\n"
         "    raise RuntimeError(('my V1284 head mount mismatch', [str(p) for p in _myhead]))\n",
         "_myhead = ['/nonexistent/v1284_head.pt']  # head is PRIVATE (row 169): mode 'zero'\n"),
        ("os.environ['V1284_MODE']='candidate'", "os.environ['V1284_MODE']='zero'"),
    ]
    # Ablations on the portable form (row 169): scored on visible-4 from commit outputs.
    # Each switches one mechanism off at its env line; 'sub' = all three off.
    FLOW = ('os.environ["BIOHUB_MOTION_RELINK_FLOW_MODE"] = "seed"',
            'os.environ["BIOHUB_MOTION_RELINK_FLOW_MODE"] = "off"')
    READMIT = ('os.environ["BIOHUB_READMIT_RADIUS_UM"] = "4"',
               'os.environ["BIOHUB_READMIT_RADIUS_UM"] = "0"')
    GAPFILL = ('os.environ["BIOHUB_GAPFILL_MAX_GAP"] = "3"',
               'os.environ["BIOHUB_GAPFILL_MAX_GAP"] = "0"')
    # Row 173: one knob on the 0.953 ORIGINAL (head ON, public dataset attached below):
    # DeepCenter safe-div threshold 0.25 -> 0.20, the reyhan line's last LB-paid step
    # (0.15 then lost -0.001, row 124) -- the one setting where x138 differs from our 947 base.
    build('pub947x_x138_dc20', [('os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"',
                                 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"')],
          extra=['anvithpothula/biohub-v1284-head-s075'])
    build('pub947x_nh', HEAD_OFF)
    build('pub947x_nh_sub', HEAD_OFF + [FLOW, READMIT, GAPFILL])
    build('pub947x_nh_noflow', HEAD_OFF + [FLOW])
    build('pub947x_nh_noreadmit', HEAD_OFF + [READMIT])
    build('pub947x_nh_nogapfill', HEAD_OFF + [GAPFILL])


if __name__ == '__main__':
    main()


def build_capture(name='pub947x_v1284cap', n_per_type=20):
    """Row 174: capture-and-train kernel for OUR V1284 head. x138 cells 0-7 (setup, test
    inference, validator inference on 2*n_per_type TRAIN movies -- the validator already
    excludes the 4 visible test stems, so visible-4 stays an honest check for our head)
    with V1284_MODE='capture'; cells 8-11 (validator scoring, PPSWEEP, manifest) dropped;
    scripts/v1284_train_cell.py appended as the last cell. NOT a submission kernel."""
    nb = json.load(open(SRC))
    edits = [
        ('os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"', 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"'),
        ('os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "4"', f'os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "{n_per_type}"'),
        ("os.environ['V1284_MODE']='candidate'",
         "os.environ['V1284_MODE']='capture'\nos.environ['V1284_CAPTURE']='/kaggle/working/v1284cap'"),
    ]
    for old, new in edits:
        hits = [i for i, c in enumerate(nb['cells']) if old in ''.join(c['source'])]
        assert len(hits) == 1, (old, hits)
        s = ''.join(nb['cells'][hits[0]]['source'])
        assert s.count(old) == 1
        nb['cells'][hits[0]]['source'] = s.replace(old, new).splitlines(keepends=True)
    code = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert 'VALIDATOR_ENABLE = os.environ.get' in ''.join(code[7]['source'])
    assert 'def match_nodes_bipartite' in ''.join(code[8]['source'])
    train_src = open('scripts/v1284_train_cell.py').read()
    tail = train_src + '''

# ---- kernel entry (row 174) ----
assert val_stems, 'validator selected no TRAIN stems -- nothing to train on'
_cap = sorted(p.name for p in Path(os.environ['V1284_CAPTURE']).iterdir())
print(f'[v1284-train] capture dirs: {len(_cap)} ({len(val_stems)} val stems)', flush=True)
main(os.environ['V1284_CAPTURE'], TRAIN_DIR, list(val_stems), '/kaggle/working/v1284_out', epochs=150)
import shutil
shutil.rmtree(os.environ['V1284_CAPTURE'], ignore_errors=True)  # GBs of per-frame features
'''
    new_cell = {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                'source': tail.splitlines(keepends=True)}
    nb['cells'] = code[:8] + [new_cell]
    for c in nb['cells']:
        c['outputs'], c['execution_count'] = [], None
        compile(''.join(c['source']), name, 'exec')
    os.makedirs(f'kernels/{name}', exist_ok=True)
    json.dump(nb, open(f'kernels/{name}/{name}.ipynb', 'w'))
    meta = json.load(open('kernels/pub947_densad/kernel-metadata.json'))
    meta.update(id=f"maruanbottoni/biohub-{name.replace('_', '-')}", title=f"biohub {name.replace('_', ' ')}",
                code_file=f'{name}.ipynb')
    meta['dataset_sources'] = meta['dataset_sources'] + ['anvithpothula/biohub-v1284-head-s075']
    json.dump(meta, open(f'kernels/{name}/kernel-metadata.json', 'w'), indent=2)
    print(f'{name}: {len(nb["cells"])} cells, capture on {2 * n_per_type} train movies, compiles')


if __name__ == '__main__' and os.environ.get('BUILD_CAPTURE'):
    build_capture()


OURHEAD_DS = 'maruanbottoni/biohub-v1284-head-ours'


def build_ourhead(name='pub947x_x138_ourhead'):
    """Row 174: the 0.953 ORIGINAL with the author's V1284 head swapped for OURS (trained by
    pub947x_v1284cap on 40 train movies, 2x the author's 20; visible-4 excluded from training).
    ONE mechanism vs pub947x_x138: the head's training set."""
    swap = ("_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-s075/v1284_head.pt'))",
            "_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-ours/v1284_head.pt'))  # OUR head, row 174")
    build(name, [swap], extra=[OURHEAD_DS])
    meta_p = f'kernels/{name}/kernel-metadata.json'
    meta = json.load(open(meta_p))
    assert 'anvithpothula/biohub-v1284-head-s075' not in meta['dataset_sources']
    json.dump(meta, open(meta_p, 'w'), indent=2)


if __name__ == '__main__' and os.environ.get('BUILD_OURHEAD'):
    build_ourhead()


def build_densad_ourhead(name='pub947x_x138_densad_ourhead'):
    """Row 177: densad (LB 0.957, row 177) + OUR V1284 head (row 175) on the 0.953 ORIGINAL."""
    swap = ("_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-s075/v1284_head.pt'))",
            "_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-ours/v1284_head.pt'))  # OUR head, row 174")
    build(name, [(READ_FRAME_ANCHOR, DENSAD_BLOCK + READ_FRAME_ANCHOR), (CALLSITE_OLD, CALLSITE_NEW), swap],
          extra=[OURHEAD_DS])


if __name__ == '__main__' and os.environ.get('BUILD_DO'):
    build_densad_ourhead()


# Row 178: extend densad's per-density conditioning to x138's MAIN flow tight gate, which
# densad leaves at a fixed 7.0 um for every volume. Deltas mirror densad's own tight_um deltas
# vs the 6.0 default it replaced (low +1.25, middle +0.5, high -0.5): 7.0 -> 8.25 / 7.5 / 6.5.
FLOW_TIGHT_BY_GROUP = {'low': 8.25, 'middle': 7.5, 'high': 6.5}
CALLSITE_NEW_FT = CALLSITE_NEW.replace(
    "        print(f'[densad] {dataset}",
    "        globals()['MOTION_RELINK_FLOW_TIGHT_UM'] = " + repr(FLOW_TIGHT_BY_GROUP) + "[_dg]\n"
    "        print(f'[densad+ft] flow_tight={MOTION_RELINK_FLOW_TIGHT_UM}', flush=True)\n"
    "        print(f'[densad] {dataset}")


def build_densad_flowtight(name='pub947x_x138_densad_ft'):
    assert CALLSITE_NEW_FT != CALLSITE_NEW
    build(name, [(READ_FRAME_ANCHOR, DENSAD_BLOCK + READ_FRAME_ANCHOR), (CALLSITE_OLD, CALLSITE_NEW_FT)],
          extra=['anvithpothula/biohub-v1284-head-s075'])


if __name__ == '__main__' and os.environ.get('BUILD_FT'):
    build_densad_flowtight()


OURHEAD120_DS = 'maruanbottoni/biohub-v1284-head-ours120'


def build_row178():
    # 120-movie capture (60/prefix; 44b6 has ~69 non-test movies) -> head v2 (3x the 40-movie head)
    build_capture('pub947x_v1284cap120', n_per_type=60)
    # densad + threshold 0.20 (queued only if dc20 beats the x138 control on the LB)
    build('pub947x_x138_densad_dc20',
          [(READ_FRAME_ANCHOR, DENSAD_BLOCK + READ_FRAME_ANCHOR), (CALLSITE_OLD, CALLSITE_NEW),
           ('os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"',
            'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"')],
          extra=['anvithpothula/biohub-v1284-head-s075'])
    # densad + the 120-movie head (queued only if the 40-movie head holds up on the LB)
    build('pub947x_x138_densad_ourhead120',
          [(READ_FRAME_ANCHOR, DENSAD_BLOCK + READ_FRAME_ANCHOR), (CALLSITE_OLD, CALLSITE_NEW),
           ("_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-s075/v1284_head.pt'))",
            "_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-ours120/v1284_head.pt'))  # 120-movie head, row 178")],
          extra=[OURHEAD120_DS])


if __name__ == '__main__' and os.environ.get('BUILD_178'):
    build_row178()


# Row 181: three one-knob moves along densad's own axes, each vs densad (0.957). densad's
# values were tuned for the non-flow relinker; on x138 they act on different stages (tight_um
# = flow SEED gate, relaxed_um = relaxed pass, learned_bonus = cost bonus), so its optimum may
# have moved. Row 180 showed conditioning the MAIN gate hurts; these stay on densad's knobs.
DENSAD_TABLE = """DENSITY_GROUP_OVERRIDES = {
    'low': {'tight_um': 7.25, 'relaxed_um': 11.0, 'velocity_weight': 0.5, 'learned_bonus': 3.0},
    'middle': {'tight_um': 6.5, 'relaxed_um': 9.0, 'velocity_weight': 0.0, 'learned_bonus': 6.0},
    'high': {'tight_um': 5.5, 'relaxed_um': 10.0, 'velocity_weight': 0.5, 'learned_bonus': 1.0},
}"""
DENSAD_VARIANTS = {
    # seed gate +0.75 um in every group
    'pub947x_x138_densad_seed': DENSAD_TABLE.replace("'tight_um': 7.25", "'tight_um': 8.0")
        .replace("'tight_um': 6.5", "'tight_um': 7.25").replace("'tight_um': 5.5", "'tight_um': 6.25"),
    # learned bonus x1.5 where densad raised it (low 3->4.5, middle 6->9); high stays 1.0
    'pub947x_x138_densad_bonus': DENSAD_TABLE.replace("'learned_bonus': 3.0", "'learned_bonus': 4.5")
        .replace("'learned_bonus': 6.0", "'learned_bonus': 9.0"),
    # relaxed pass +1.0 um in every group
    # (per-line replacement: chained str.replace hit the wrong group in the first build)
    'pub947x_x138_densad_relax': DENSAD_TABLE
        .replace("'low': {'tight_um': 7.25, 'relaxed_um': 11.0", "'low': {'tight_um': 7.25, 'relaxed_um': 12.0")
        .replace("'middle': {'tight_um': 6.5, 'relaxed_um': 9.0", "'middle': {'tight_um': 6.5, 'relaxed_um': 10.0")
        .replace("'high': {'tight_um': 5.5, 'relaxed_um': 10.0", "'high': {'tight_um': 5.5, 'relaxed_um': 11.0"),
}


def build_row181():
    assert DENSAD_TABLE in DENSAD_BLOCK
    for name, table in DENSAD_VARIANTS.items():
        assert table != DENSAD_TABLE, name
        build(name, [(READ_FRAME_ANCHOR, DENSAD_BLOCK.replace(DENSAD_TABLE, table) + READ_FRAME_ANCHOR),
                     (CALLSITE_OLD, CALLSITE_NEW)], extra=['anvithpothula/biohub-v1284-head-s075'])


if __name__ == '__main__' and os.environ.get('BUILD_181'):
    build_row181()
