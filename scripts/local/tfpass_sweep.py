#!/usr/bin/env python
"""tfpass reject-gate: the pass-0 hard pre-commit micro-grid on holdout6.

Mechanism and its provenance: see scripts/build_pub947_tfpass.py. In one line, a
pass-0 commits (a,b) when prob >= P0, (a,b) is mutual-top, and dist <= GATE, then the
tight and relaxed geometric passes run unchanged on what is left.

The pass is pure POST-PROC (motion_relink_edges runs inside filter_output_graph, after
the cached predict_video), so every arm is CACHE-HIT against cache_densad: ~12 min
each, zero GPU quota, zero submission slots.

ARM 0 IS NOT OPTIONAL. `ctl` runs the tfpass notebook with the pass DISABLED and must
reproduce the row-143 baseline BYTE-FOR-BYTE against ds_ctl/submission.csv. A patch
that quietly fails to apply looks exactly like a patch that applies and does nothing
(CLAUDE.md; row 82 is the scar). The grid is only believable after ctl is sha-identical
and the armed arms print a non-zero `[tfpass] pool=`.

Reject-gate semantics (row 90): local may REJECT, never CONFIRM; a delta under 0.007
is uninterpretable. What this grid can legitimately produce is a REJECT (kill the
mechanism for free) or a not-rejected candidate that earns ONE pre-registered LB slot.

Note the two known blind spots of holdout6 for THIS mechanism (row 154): it routes 2
low + 4 middle density volumes and contains no HIGH-density volume, and the high group
ships learned_bonus 1.0 — the regime where linking is closest to pure geometry and a
prob-first pass-0 would hand over the most. Cross-check any survivor on the visible-4
(scripts/local/visible4_calibration.py) before it is allowed near a slot.

Usage:  tfpass_sweep.py [grid]      grids: main (default) | wide
        BIOHUB_SWEEP_FORCE=1 to re-run arms that already have a submission.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("/Volumes/extSSD/kaggle/biohub/replay947")
NB = REPO / "kernels/pub947_tfpass/pub947_tfpass.ipynb"
CACHE = SCRATCH / "cache_densad"
RUNS = Path.home() / "biohub_replay947_runs"
DRIVER = REPO / "scripts/local/pub947_local_replay.py"
SCORER = REPO / "scripts/local/score_official.py"
PY = REPO / ".venv/bin/python"
BASE = 0.9525          # row 143 holdout6 reference (densad)
FLOOR = 0.007          # row 90: below this a local delta is uninterpretable
CTL_REF = RUNS / "ds_ctl/submission.csv"

GRIDS = {
    # p0 x gate around the simulated optimum, plus the arm that isolates mutual-top.
    "main": [
        ("ctl",            {}),
        ("p90_g14",        {"BIOHUB_TFPASS_P0": "0.90", "BIOHUB_TFPASS_GATE_UM": "14.0"}),
        ("p95_g14",        {"BIOHUB_TFPASS_P0": "0.95", "BIOHUB_TFPASS_GATE_UM": "14.0"}),
        ("p80_g14",        {"BIOHUB_TFPASS_P0": "0.80", "BIOHUB_TFPASS_GATE_UM": "14.0"}),
        ("p90_g11",        {"BIOHUB_TFPASS_P0": "0.90", "BIOHUB_TFPASS_GATE_UM": "11.0"}),
        ("p95_g11",        {"BIOHUB_TFPASS_P0": "0.95", "BIOHUB_TFPASS_GATE_UM": "11.0"}),
        ("p90_g14_nomut",  {"BIOHUB_TFPASS_P0": "0.90", "BIOHUB_TFPASS_GATE_UM": "14.0",
                            "BIOHUB_TFPASS_MUTUAL": "0"}),
    ],
    "wide": [
        ("p99_g14",        {"BIOHUB_TFPASS_P0": "0.99", "BIOHUB_TFPASS_GATE_UM": "14.0"}),
        ("p50_g14",        {"BIOHUB_TFPASS_P0": "0.50", "BIOHUB_TFPASS_GATE_UM": "14.0"}),
        ("p90_g8",         {"BIOHUB_TFPASS_P0": "0.90", "BIOHUB_TFPASS_GATE_UM": "8.0"}),
    ],
}

SUMMARY_RE = re.compile(
    r"score=([\d.]+)\s+edge_jaccard=([\d.]+)\s+adj_edge_jaccard=([\d.]+).*?division_jaccard=([\d.]+)")
POOL_RE = re.compile(r"\[tfpass\] p0=\S+ gate=\S+ mutual=\S+ pool=(\d+) mutual_pairs=(\d+)")
PRECOMMIT_RE = re.compile(r"motion_relink_precommit_edges")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def run_variant(name: str, overrides: dict) -> dict | None:
    run_dir = RUNS / f"tf_{name}"
    sub = run_dir / "submission.csv"
    log_path = SCRATCH / f"tf_{name}.log"
    env = {**os.environ, "BIOHUB_REPLAY_NB": str(NB), "BIOHUB_PRED_CACHE": str(CACHE),
           "BIOHUB_REPLAY_RUNS": str(RUNS)}

    if not (sub.exists() and os.environ.get("BIOHUB_SWEEP_FORCE") != "1"):
        print(f"\n===== [{name}] running ({overrides or 'DISABLED = ctl'}) =====", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as logf:
            r = subprocess.run([str(PY), str(DRIVER), f"tf_{name}", json.dumps(overrides)],
                               env=env, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[{name}] DRIVER FAILED rc={r.returncode} — see {log_path}", flush=True)
            return None
    else:
        print(f"\n===== [{name}] cached, scoring only =====", flush=True)
    if not sub.exists():
        print(f"[{name}] no submission produced", flush=True)
        return None

    # --- no-op / silent-no-op guards ------------------------------------------------
    log = log_path.read_text() if log_path.exists() else ""
    pools = POOL_RE.findall(log)
    pool_total = sum(int(p) for p, _ in pools)
    if name == "ctl":
        same = CTL_REF.exists() and sha(sub) == sha(CTL_REF)
        print(f"[ctl] sha {sha(sub)} vs ds_ctl {sha(CTL_REF) if CTL_REF.exists() else 'MISSING'}"
              f" -> {'IDENTICAL (patch inert when disabled)' if same else '*** DIFFERS ***'}",
              flush=True)
        if not same:
            print("[ctl] ABORT: the disabled patch changed the output; the grid is not "
                  "interpretable until this is explained.", flush=True)
            return None
        if pools:
            print("[ctl] ABORT: pass-0 announced itself while disabled.", flush=True)
            return None
    elif pool_total == 0:
        print(f"[{name}] ABORT: pass-0 never announced a pool — the patch did not fire "
              f"(silent no-op). Check {log_path}.", flush=True)
        return None

    sc = subprocess.run([str(PY), str(SCORER), str(sub)], capture_output=True, text=True)
    m = SUMMARY_RE.search(sc.stdout)
    if not m:
        print(f"[{name}] SCORER produced no summary:\n{sc.stdout[-400:]}", flush=True)
        return None
    score, ej, adj, dj = (float(x) for x in m.groups())
    delta = score - BASE
    verdict = ("PASS(>base)" if delta > 0 else
               "REJECT" if delta <= -FLOOR else "flat/uninterp")
    print(f"[{name}] score={score:.4f} delta={delta:+.4f} pool={pool_total} -> {verdict}",
          flush=True)
    return {"variant": name, "score": round(score, 4), "delta": round(delta, 4),
            "edge_J": round(ej, 4), "adj_J": round(adj, 4), "div_J": round(dj, 4),
            "precommit_pool": pool_total, "verdict": verdict,
            "overrides": json.dumps(overrides)}


def main() -> None:
    grid = sys.argv[1] if len(sys.argv) > 1 else "main"
    variants = GRIDS.get(grid, GRIDS["main"])
    assert NB.exists(), f"build it first: python scripts/build_pub947_tfpass.py ({NB})"
    assert (CACHE / "6bba_062c8d37.pkl").exists(), f"prediction cache missing at {CACHE}"
    RUNS.mkdir(exist_ok=True)

    results = []
    for name, ov in variants:
        row = run_variant(name, ov)
        if row:
            results.append(row)
        elif name == "ctl":
            print("\n*** ctl failed its guard — stopping before the grid. ***")
            return

    results.sort(key=lambda r: r["score"], reverse=True)
    out = SCRATCH / "tfpass_sweep_results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "score", "delta", "edge_J", "adj_J",
                                          "div_J", "precommit_pool", "verdict", "overrides"])
        w.writeheader()
        w.writerows(results)
    print(f"\n===== TFPASS SWEEP DONE ({len(results)}/{len(variants)}) — base {BASE} =====")
    print(f"{'variant':16} {'score':>7} {'delta':>8} {'adj_J':>7} {'div_J':>7} {'pool':>7}  verdict")
    for r in results:
        print(f"{r['variant']:16} {r['score']:>7.4f} {r['delta']:>+8.4f} {r['adj_J']:>7.4f} "
              f"{r['div_J']:>7.4f} {r['precommit_pool']:>7}  {r['verdict']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
