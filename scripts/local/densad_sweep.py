#!/usr/bin/env python
"""densad-table micro-grid: reject-gate sweep on the pub947_densad_sweep notebook.

Each variant changes ONE densad-table knob via a BIOHUB_DENSAD_* env override, runs
the local replay CACHE-HIT (motion-relink is post-proc; predict_video is cached in
cache_densad), scores with the official scorer, and is ranked against base 0.9525.

Reject-gate semantics (§50.2/row 90): local may REJECT, never CONFIRM. A variant that
beats base earns an LB slot candidacy; a delta <0.007 is uninterpretable. holdout6 =
4 middle + 2 low volumes; HIGH knobs are untestable here (and LB-saturated, row 135) so
they are excluded. The 'ctl' arm (no overrides) MUST reproduce base 0.9525 — it proves
the harness + cache are faithful before any variant is believed.

Usage:  densad_sweep.py [grid_name]     grids: core (default) | wide
Runs sequentially (MPS); writes runs to ~/biohub_replay947_runs/ds_<variant>/ and a
ranked table to <scratch>/densad_sweep_results.csv. Safe to re-run: completed variants
with an existing scored submission are skipped unless BIOHUB_SWEEP_FORCE=1.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("/Volumes/extSSD/kaggle/biohub/replay947")
NB = REPO / "kernels/pub947_densad_sweep/pub947_densad_sweep.ipynb"
CACHE = SCRATCH / "cache_densad"          # shared: all densad variants share inference
RUNS = Path.home() / "biohub_replay947_runs"
DRIVER = REPO / "scripts/local/pub947_local_replay.py"
SCORER = REPO / "scripts/local/score_official.py"
PY = REPO / ".venv/bin/python"
BASE = 0.9525                              # row 143 local reference
FLOOR = 0.007                              # row 90: below this, uninterpretable

# (variant, {env overrides}). ctl first — it validates the harness against base.
GRIDS = {
    "core": [
        ("ctl",            {}),
        ("mid_bonus_4p5",  {"BIOHUB_DENSAD_MID_BONUS": "4.5"}),
        ("mid_bonus_8p0",  {"BIOHUB_DENSAD_MID_BONUS": "8.0"}),
        ("mid_tight_6p0",  {"BIOHUB_DENSAD_MID_TIGHT": "6.0"}),
        ("mid_tight_7p0",  {"BIOHUB_DENSAD_MID_TIGHT": "7.0"}),
        ("mid_relaxed_8p0",{"BIOHUB_DENSAD_MID_RELAXED": "8.0"}),
        ("mid_relaxed_10p0",{"BIOHUB_DENSAD_MID_RELAXED": "10.0"}),
        ("low_bonus_2p0",  {"BIOHUB_DENSAD_LOW_BONUS": "2.0"}),
        ("low_bonus_4p5",  {"BIOHUB_DENSAD_LOW_BONUS": "4.5"}),
        ("low_tight_6p5",  {"BIOHUB_DENSAD_LOW_TIGHT": "6.5"}),
        ("low_tight_8p0",  {"BIOHUB_DENSAD_LOW_TIGHT": "8.0"}),
    ],
    "wide": [
        ("mid_vel_0p5",    {"BIOHUB_DENSAD_MID_VEL": "0.5"}),
        ("low_relaxed_9p5",{"BIOHUB_DENSAD_LOW_RELAXED": "9.5"}),
        ("low_relaxed_12p5",{"BIOHUB_DENSAD_LOW_RELAXED": "12.5"}),
        ("low_vel_0p25",   {"BIOHUB_DENSAD_LOW_VEL": "0.25"}),
        ("bound_low_100",  {"BIOHUB_DENSAD_LOW_MAX": "100"}),
        ("bound_low_150",  {"BIOHUB_DENSAD_LOW_MAX": "150"}),
    ],
}

SUMMARY_RE = re.compile(
    r"score=([\d.]+)\s+edge_jaccard=([\d.]+)\s+adj_edge_jaccard=([\d.]+).*?division_jaccard=([\d.]+)")


def run_variant(name: str, overrides: dict) -> dict | None:
    run_dir = RUNS / f"ds_{name}"
    sub = run_dir / "submission.csv"
    env = {**os.environ,
           "BIOHUB_REPLAY_NB": str(NB),
           "BIOHUB_PRED_CACHE": str(CACHE),
           "BIOHUB_REPLAY_RUNS": str(RUNS)}
    if not (sub.exists() and os.environ.get("BIOHUB_SWEEP_FORCE") != "1"):
        print(f"\n===== [{name}] running (overrides={overrides or 'NONE=ctl'}) =====", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = SCRATCH / f"ds_{name}.log"
        with log_path.open("w") as logf:
            r = subprocess.run(
                [str(PY), str(DRIVER), f"ds_{name}", json.dumps(overrides)],
                env=env, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[{name}] DRIVER FAILED (rc={r.returncode}) — see {log_path}", flush=True)
            return None
    else:
        print(f"\n===== [{name}] cached submission exists, scoring only =====", flush=True)
    if not sub.exists():
        print(f"[{name}] no submission produced", flush=True)
        return None
    sc = subprocess.run([str(PY), str(SCORER), str(sub)], capture_output=True, text=True)
    m = SUMMARY_RE.search(sc.stdout)
    if not m:
        print(f"[{name}] SCORER produced no summary:\n{sc.stdout[-400:]}\n{sc.stderr[-400:]}", flush=True)
        return None
    score, ej, adj, dj = (float(x) for x in m.groups())
    delta = score - BASE
    verdict = ("PASS(>base)" if delta > 0 else
               "reject" if delta <= -FLOOR else "flat/uninterp")
    row = {"variant": name, "score": round(score, 4), "delta": round(delta, 4),
           "edge_J": round(ej, 4), "adj_J": round(adj, 4), "div_J": round(dj, 4),
           "verdict": verdict, "overrides": json.dumps(overrides)}
    print(f"[{name}] score={score:.4f} delta={delta:+.4f} -> {verdict}", flush=True)
    return row


def main():
    grid = sys.argv[1] if len(sys.argv) > 1 else "core"
    variants = GRIDS[grid] if grid in GRIDS else GRIDS["core"] + GRIDS["wide"]
    assert NB.exists(), f"build the sweep notebook first: {NB}"
    assert (CACHE / "6bba_062c8d37.pkl").exists(), (
        f"prediction cache missing at {CACHE} — run the base replay first (row 143)")
    RUNS.mkdir(exist_ok=True)
    results = []
    for name, ov in variants:
        row = run_variant(name, ov)
        if row:
            results.append(row)
    results.sort(key=lambda r: r["score"], reverse=True)
    out = SCRATCH / "densad_sweep_results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "score", "delta", "edge_J",
                                          "adj_J", "div_J", "verdict", "overrides"])
        w.writeheader(); w.writerows(results)
    print(f"\n===== SWEEP DONE ({len(results)}/{len(variants)} scored) — base {BASE} =====")
    print(f"{'variant':18} {'score':>7} {'delta':>8} {'adj_J':>7} {'div_J':>7}  verdict")
    for r in results:
        print(f"{r['variant']:18} {r['score']:>7.4f} {r['delta']:>+8.4f} "
              f"{r['adj_J']:>7.4f} {r['div_J']:>7.4f}  {r['verdict']}")
    winners = [r for r in results if r["verdict"] == "PASS(>base)" and r["variant"] != "ctl"]
    print(f"\nCandidates for an LB slot (beat base, NOT ctl): "
          f"{[r['variant'] for r in winners] or 'NONE — config lever dry'}")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
