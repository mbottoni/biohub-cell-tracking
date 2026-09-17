#!/usr/bin/env python3
"""Build the two graft kernels on the hardened pub947 base (rows 127-128).

pub947_divnet — the row-126 combo: giorgosi's DivNet (correctly reconstructed —
  haideptry's public integration is a double no-op, row 128) as a RANK-based
  confirm on safe-div forks + the global cap raise. Rank-based because the local
  GT check (scripts/local/divnet_local_check.py) found AUC 0.78 with compressed
  absolute probabilities (p_pos 0.538 vs p_neg 0.530, instance-norm): ranking
  carries signal, absolute thresholds do not. Fail-LOUD: if the checkpoint
  cannot load, the kernel RAISES (BIOHUB_DIVNET_REQUIRE=1) — no silent no-op.

pub947_densad — haideptry's density-adaptive motion-relink table verbatim
  (verified generic: grouped by measured avg nodes/frame, not volume ids).
  Refines row 124's relink6 reject: uniform bonus 6.0 lost; conditioned 6.0
  rides only middle-density volumes.

Both single-mechanism vs pub947_port (0.947), built via build_pub947_variant
hardening + surgical string edits with assert-exactly-once.
"""
import json, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pub947_variant import HARDEN, PROBE  # noqa: E402

DIVNET_BLOCK = r'''

# ==================== DivNet confirm gate (giorgosi/biohub-divnet-v2, row 128) =====
# Architecture reconstructed from the checkpoint state dict (strict-load verified);
# preprocessing pinned by the embedded config + local GT discrimination (AUC 0.78,
# marker-last / mean-pool-4 / per-crop percentile norm / InstanceNorm3d(affine)).
import torch
import numpy as np

DIVNET_VETO_FRAC = float(os.environ.get('BIOHUB_DIVNET_VETO_FRAC', '0.40'))
DIVNET_REQUIRE = os.environ.get('BIOHUB_DIVNET_REQUIRE', '1') != '0'

class _DivnetBlock(torch.nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            torch.nn.InstanceNorm3d(cout, affine=True), torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            torch.nn.InstanceNorm3d(cout, affine=True), torch.nn.ReLU(inplace=True))
    def forward(self, x):
        return self.block(x)

class _DivnetUNet3DGAP(torch.nn.Module):
    def __init__(self, cin=5, base=16):
        super().__init__()
        self.enc1 = _DivnetBlock(cin, base); self.enc2 = _DivnetBlock(base, base * 2)
        self.enc3 = _DivnetBlock(base * 2, base * 4)
        self.bottleneck = _DivnetBlock(base * 4, base * 8)
        self.pool = torch.nn.MaxPool3d(2)
        self.up3 = torch.nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _DivnetBlock(base * 8, base * 4)
        self.up2 = torch.nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _DivnetBlock(base * 4, base * 2)
        self.up1 = torch.nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = _DivnetBlock(base * 2, base)
        self.head = torch.nn.Linear(base, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1.mean(dim=(2, 3, 4)))

def _divnet_load():
    ckpt = _biohub_ds('biohub-divnet-v2') + '/best_overall.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = _DivnetUNet3DGAP()
    model.load_state_dict(ck['model_state'], strict=True)  # strict: fail loud, row 82
    model.to(device).eval()
    print(f'[divnet] loaded {ckpt} epoch={ck.get("epoch")} strict=True OK')
    return {'model': model, 'device': device, 'cfg': ck['config']}

try:
    DIVNET_BUNDLE = _divnet_load()
except Exception as _e:
    if DIVNET_REQUIRE:
        raise RuntimeError(f'DivNet checkpoint required but failed to load: {_e}')
    DIVNET_BUNDLE = None

_DIVNET_MARKER = None

def _divnet_marker():
    global _DIVNET_MARKER
    if _DIVNET_MARKER is None:
        sz, sy, sx = DIVNET_BUNDLE['cfg']['input']['marker_sigma']
        zz, yy, xx = np.ogrid[:16, :32, :32]
        _DIVNET_MARKER = np.exp(-(((zz - 8) ** 2) / (2 * sz ** 2) + ((yy - 16) ** 2) / (2 * sy ** 2)
                                  + ((xx - 16) ** 2) / (2 * sx ** 2))).astype(np.float32)
    return _DIVNET_MARKER

def _divnet_pooled_frame(dataset, t, frame_cache, pooled_cache):
    key = (dataset, int(t))
    if key not in pooled_cache:
        fr = np.asarray(read_test_frame(dataset, int(t), frame_cache), dtype=np.float32)
        Z, Y, X = fr.shape
        pooled_cache[key] = fr[:, :(Y // 4) * 4, :(X // 4) * 4].reshape(
            Z, Y // 4, 4, X // 4, 4).mean(axis=(2, 4))
        if len(pooled_cache) > 64:
            pooled_cache.pop(next(iter(pooled_cache)))
    return pooled_cache[key]

def _divnet_score_parents(dataset, parents, frame_cache, pooled_cache):
    cfg = DIVNET_BUNDLE['cfg']
    ins = []
    for node in parents:
        t = int(node['t'])
        cz, cy, cx = float(node['z']), float(node['y']) / 4.0, float(node['x']) / 4.0
        chans = []
        for lag in cfg['input']['image_lags']:
            fr = _divnet_pooled_frame(dataset, max(0, t + lag), frame_cache, pooled_cache)
            Z, Y, X = fr.shape
            z0, y0, x0 = int(round(cz)) - 8, int(round(cy)) - 16, int(round(cx)) - 16
            crop = np.zeros((16, 32, 32), dtype=np.float32)
            zs, ys, xs = max(0, z0), max(0, y0), max(0, x0)
            ze, ye, xe = min(Z, z0 + 16), min(Y, y0 + 32), min(X, x0 + 32)
            if ze > zs and ye > ys and xe > xs:
                crop[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = fr[zs:ze, ys:ye, xs:xe]
            chans.append(crop)
        img = np.stack(chans, 0)
        lo = np.percentile(img, cfg['normalization']['lo_pct'])
        hi = np.percentile(img, cfg['normalization']['hi_pct'])
        img = np.clip((img - lo) / max(hi - lo, 1e-6), *cfg['normalization']['clip'])
        ins.append(np.concatenate([img, _divnet_marker()[None]], 0).astype(np.float32))
    device = DIVNET_BUNDLE['device']
    model = DIVNET_BUNDLE['model']
    probs = []
    with torch.inference_mode():
        for i in range(0, len(ins), 64):
            x = torch.from_numpy(np.stack(ins[i:i + 64])).to(device)
            probs.extend(torch.sigmoid(model(x)).squeeze(1).float().cpu().numpy().tolist())
    return probs

def _divnet_prune_safe_divisions(dataset, nodes_by_id, added, frame_cache, stats):
    if DIVNET_BUNDLE is None or len(added) < 3:
        return added
    pooled_cache = {}
    parents = [nodes_by_id[int(e['source_id'])] for e in added]
    probs = _divnet_score_parents(dataset, parents, frame_cache, pooled_cache)
    order = sorted(range(len(added)), key=lambda i: probs[i], reverse=True)
    n_keep = max(1, int(round(len(added) * (1.0 - DIVNET_VETO_FRAC))))
    keep = set(order[:n_keep])
    stats['divnet_scored'] = stats.get('divnet_scored', 0) + len(added)
    stats['divnet_vetoed'] = stats.get('divnet_vetoed', 0) + (len(added) - n_keep)
    print(f'[divnet] {dataset}: scored {len(added)} safe-div forks, '
          f'kept {n_keep} (veto_frac {DIVNET_VETO_FRAC})')
    return [e for i, e in enumerate(added) if i in keep]
# ==================== end DivNet block ============================================

'''

DENSAD_BLOCK = r'''

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

DENSAD_CALLSITE_OLD = "        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)"
DENSAD_CALLSITE_NEW = """        _dg = determine_density_group(nodes_by_id)
        _dg_cfg = DENSITY_GROUP_OVERRIDES[_dg]
        globals()['MOTION_RELINK_TIGHT_UM'] = _dg_cfg['tight_um']
        globals()['MOTION_RELINK_RELAXED_UM'] = _dg_cfg['relaxed_um']
        globals()['MOTION_RELINK_VELOCITY_WEIGHT'] = _dg_cfg['velocity_weight']
        globals()['MOTION_RELINK_LEARNED_BONUS'] = _dg_cfg['learned_bonus']
        print(f'[densad] density group {_dg} -> {_dg_cfg}')
        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)"""

DIVNET_RETURN_OLD = """    if added:
        stats['safe_divisions_added'] = len(added)
        return [*edges, *added]
    return edges"""
DIVNET_RETURN_NEW = """    if added:
        added = _divnet_prune_safe_divisions(dataset, nodes_by_id, added, frame_cache, stats)
        stats['safe_divisions_added'] = len(added)
        return [*edges, *added]
    return edges"""

READ_FRAME_ANCHOR = "def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:"


def build(name, edits, extra_datasets=()):
    nb = json.load(open('kernels/pub947_port/pub947_port.ipynb'))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 1
    src = ''.join(cells[0]['source'])
    for old, new, n in HARDEN:
        assert src.count(old) == n, f'harden {old[:50]!r}: {src.count(old)}'
        src = src.replace(old, new)
    for old, new in edits:
        assert src.count(old) == 1, f'{name}: {old[:70]!r} x{src.count(old)}'
        src = src.replace(old, new)
    cells[0]['source'] = src.splitlines(keepends=True)
    os.makedirs(f'kernels/{name}', exist_ok=True)
    json.dump(nb, open(f'kernels/{name}/{name}.ipynb', 'w'))
    meta = json.load(open('kernels/pub947_port/kernel-metadata.json'))
    meta['id'] = f"maruanbottoni/biohub-{name.replace('_', '-')}"
    meta['title'] = f"biohub {name.replace('_', ' ')}"
    meta['code_file'] = f'{name}.ipynb'
    meta['dataset_sources'] = meta['dataset_sources'] + list(extra_datasets)
    json.dump(meta, open(f'kernels/{name}/kernel-metadata.json', 'w'), indent=2)
    print(f'{name}: built ({len(edits)} edits, {len(meta["dataset_sources"])} datasets)')


def main():
    build('pub947_divnet', [
        (READ_FRAME_ANCHOR, DIVNET_BLOCK + READ_FRAME_ANCHOR),
        (DIVNET_RETURN_OLD, DIVNET_RETURN_NEW),
        ("os.environ['BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP'] = '0.00375'",
         "os.environ['BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP'] = '0.0076'"),
    ], extra_datasets=['giorgosi/biohub-divnet-v2'])

    build('pub947_densad', [
        (READ_FRAME_ANCHOR, DENSAD_BLOCK + READ_FRAME_ANCHOR),
        (DENSAD_CALLSITE_OLD, DENSAD_CALLSITE_NEW),
    ])


if __name__ == '__main__':
    main()
