#!/usr/bin/env python3
"""ORACLE CEILING on per-cell selection over the detection-threshold ladder.

This prices PLAN's "multi-hypothesis detection" route (the ultrack idea) WITHOUT
ultrack, without an ILP solver, and without spending quota. It answers the only
question that matters before any packaging:

    If selection over the threshold ladder were PERFECT, how much score is there?

Why this is the right gate. Row 77 measured that loosening DET_THRESHOLD
monotonically improves the association term (adj 0.9065 -> 0.9072 -> 0.9075 at
0.96875 / 0.96 / 0.94) while the division term collapses, so the TOTAL loses.
That is the signature of a *blanket* decision: the loose threshold surfaces real
cells AND junk, and one global cut cannot separate them. The multi-hypothesis
thesis is that a per-cell decision can.

The ceiling of any such method is: keep every loose node that is real, drop every
loose node that is not. "Real" here means it matches an annotated GT node under
the metric's own rule -- centroid distance <= 7.0 um, anisotropy-scaled, optimal
(Hungarian) per frame (docs/metric.md). No selector, learned or ILP, can beat
that on this candidate pool.

  * If the ceiling is at/below the 0.9135 control, the route is DEAD and ultrack's
    1.8 GB / 363-dep packaging tax buys nothing. Kill it here, for free.
  * If the ceiling is large, ultrack becomes worth its cost, and this number is
    the pre-registered target its real (imperfect) selection must be judged against.

⚠ This is an ORACLE. It uses GT to select, so it is NOT a submittable config and
its score is NOT achievable. It is an upper bound, nothing else.

Stage 1 (this script, default) is pure counting -- no replay, seconds:
how many annotated GT cells does the loose set recover that the baseline misses?
That alone bounds the prize.

    .venv/bin/python scripts/local/ladder_oracle.py
    .venv/bin/python scripts/local/ladder_oracle.py --emit-cache <dir>   # stage 2
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ctrack.data.geff import load_geff  # noqa: E402

# Session scratchpad of the run that produced the caches (row 67: assets outlive
# the session that made them, but /private/tmp is swept -- verify before trusting).
OLD = Path("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
           "746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad")
GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")

# docs/metric.md: node matching is centroid distance <= 7.0 um, anisotropy-scaled,
# optimal per frame. Voxel size (z, y, x) um from README / docs/data.md.
VOXEL_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
MATCH_UM = 7.0


def match_frame(pred_zyx: np.ndarray, gt_zyx: np.ndarray) -> list[tuple[int, int]]:
    """Optimal 1-1 matching within MATCH_UM. Returns (pred_idx, gt_idx) pairs."""
    if len(pred_zyx) == 0 or len(gt_zyx) == 0:
        return []
    d = np.linalg.norm(
        pred_zyx[:, None, :] * VOXEL_UM - gt_zyx[None, :, :] * VOXEL_UM, axis=-1)
    # Forbid pairs beyond the radius by making them prohibitively expensive.
    big = MATCH_UM * 1e3
    cost = np.where(d <= MATCH_UM, d, big)
    ri, ci = linear_sum_assignment(cost)
    return [(int(a), int(b)) for a, b in zip(ri, ci) if cost[a, b] < big]


def matched_gt_ids(coords: np.ndarray, gt_t: np.ndarray, gt_zyx: np.ndarray) -> set[int]:
    """Global indices of GT nodes matched by `coords` (both are [t,z,y,x] / [t]+[zyx])."""
    out: set[int] = set()
    for t in np.unique(gt_t):
        gsel = np.flatnonzero(gt_t == t)
        psel = np.flatnonzero(coords[:, 0] == t)
        if not len(gsel) or not len(psel):
            continue
        for _pi, gj in match_frame(coords[psel, 1:].astype(np.float64), gt_zyx[gsel]):
            out.add(int(gsel[gj]))
    return out


def node_key(c: np.ndarray) -> set[tuple[int, int, int, int]]:
    return {tuple(int(v) for v in row) for row in c}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cache", default=str(OLD / "pred_cache"),
                    help="cached (coords,edges) at the shipped threshold 0.96875")
    ap.add_argument("--loose-cache", default=str(OLD / "cache_det094"),
                    help="cached (coords,edges) at the loose threshold 0.94")
    ap.add_argument("--emit-cache", default=None,
                    help="stage 2: write oracle-selected (coords,edges) here")
    args = ap.parse_args()

    base_dir, loose_dir = Path(args.base_cache), Path(args.loose_cache)
    emit = Path(args.emit_cache) if args.emit_cache else None
    if emit:
        emit.mkdir(parents=True, exist_ok=True)

    vols = sorted(p.stem for p in loose_dir.glob("*.pkl"))
    print(f"[oracle] loose cache {loose_dir.name}: {len(vols)} volumes")
    print(f"[oracle] base  cache {base_dir.name}:  "
          f"{len(list(base_dir.glob('*.pkl')))} volumes")
    print(f"[oracle] match rule: <= {MATCH_UM} um, voxel {VOXEL_UM.tolist()}\n")

    tot = dict(gt=0, base_hit=0, loose_hit=0, gained=0,
               base_n=0, loose_n=0, oracle_n=0)
    rows = []
    for v in vols:
        bf, lf = base_dir / f"{v}.pkl", loose_dir / f"{v}.pkl"
        if not bf.exists():
            print(f"  {v}: SKIP (no baseline cache)")
            continue
        base_coords, _ = pickle.loads(bf.read_bytes())
        loose_coords, loose_edges = pickle.loads(lf.read_bytes())
        base_coords = np.asarray(base_coords)
        loose_coords = np.asarray(loose_coords)

        gt = load_geff(GT_ROOT / f"{v}.geff")
        g = gt.graph.nodes.astype(np.float64)
        # ⚠ geff nodes are [id, t, z, y, x] -- FIVE columns. Reading col 0 as `t`
        # silently matches nothing (node ids are ~1e6..1e8, so every frame set is
        # empty and recall comes out 0.0000). Caught by the zero, not by a crash.
        assert g.shape[1] == 5, f"expected [id,t,z,y,x], got {g.shape[1]} cols"
        gt_t, gt_zyx = g[:, 1].astype(int), g[:, 2:5]

        m_base = matched_gt_ids(base_coords, gt_t, gt_zyx)
        m_loose = matched_gt_ids(loose_coords, gt_t, gt_zyx)
        gained = m_loose - m_base

        # Oracle node set: every baseline node, plus the loose-only nodes that
        # recover a GT cell the baseline missed. Nothing else from the loose set.
        base_keys = node_key(base_coords)
        keep = np.zeros(len(loose_coords), dtype=bool)
        for i, row in enumerate(loose_coords):
            if tuple(int(x) for x in row) in base_keys:
                keep[i] = True
        if gained:
            gsel = np.array(sorted(gained))
            for t in np.unique(gt_t[gsel]):
                gidx = gsel[gt_t[gsel] == t]
                psel = np.flatnonzero(loose_coords[:, 0] == t)
                if not len(psel):
                    continue
                for pi, gj in match_frame(
                        loose_coords[psel, 1:].astype(np.float64), gt_zyx[gidx]):
                    keep[psel[pi]] = True

        rows.append((v, len(gt_t), len(m_base), len(m_loose), len(gained),
                     len(base_coords), len(loose_coords), int(keep.sum())))
        tot["gt"] += len(gt_t); tot["base_hit"] += len(m_base)
        tot["loose_hit"] += len(m_loose); tot["gained"] += len(gained)
        tot["base_n"] += len(base_coords); tot["loose_n"] += len(loose_coords)
        tot["oracle_n"] += int(keep.sum())

        if emit:
            idx = np.flatnonzero(keep)
            remap = -np.ones(len(loose_coords), dtype=np.int64)
            remap[idx] = np.arange(len(idx))
            e = np.asarray(loose_edges)
            if len(e):
                s, d = e[:, 0].astype(int), e[:, 1].astype(int)
                ok = keep[s] & keep[d]
                e = e[ok].copy()
                e[:, 0] = remap[s[ok]]
                e[:, 1] = remap[d[ok]]
            (emit / f"{v}.pkl").write_bytes(
                pickle.dumps((loose_coords[idx].astype(np.int16), e)))

    hdr = f"{'volume':<16}{'GT':>7}{'base':>7}{'loose':>7}{'gain':>7}" \
          f"{'nBase':>9}{'nLoose':>9}{'nOracle':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:<16}{r[1]:>7}{r[2]:>7}{r[3]:>7}{r[4]:>7}"
              f"{r[5]:>9}{r[6]:>9}{r[7]:>9}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<16}{tot['gt']:>7}{tot['base_hit']:>7}{tot['loose_hit']:>7}"
          f"{tot['gained']:>7}{tot['base_n']:>9}{tot['loose_n']:>9}{tot['oracle_n']:>9}")

    br = tot["base_hit"] / tot["gt"] if tot["gt"] else 0
    lr = tot["loose_hit"] / tot["gt"] if tot["gt"] else 0
    print(f"\n[oracle] GT node recall: base {br:.4f} -> loose {lr:.4f} "
          f"(+{lr - br:.4f}, {tot['gained']} cells)")
    print(f"[oracle] node budget: base {tot['base_n']} -> oracle {tot['oracle_n']} "
          f"({(tot['oracle_n'] / tot['base_n'] - 1) * 100:+.2f}%), "
          f"loose-blanket would be {tot['loose_n']} "
          f"({(tot['loose_n'] / tot['base_n'] - 1) * 100:+.2f}%)")
    if emit:
        print(f"\n[oracle] wrote oracle pred_cache -> {emit}")
        print("[oracle] stage 2: replay with BIOHUB_PRED_CACHE=<that dir>, then score_official.py")


if __name__ == "__main__":
    main()
