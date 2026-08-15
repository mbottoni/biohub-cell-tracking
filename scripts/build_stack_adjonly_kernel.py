#!/usr/bin/env python3
"""pub162_stack + adjacency filter — the fix for the row 83 audit failure.

WHY THIS EXISTS
---------------
`pub162_stack` (ILP disappearance 2.0 + gap-close clamp lifted, max_gap 3) was
pushed on 08-15 when the GPU quota freed, and the commit run DIED at 00:36:

    AssertionError: Clean graph audit failed. Review the audit JSON above.

The audit (cell18) requires twelve invariants to be exactly 0. Eleven were.
The twelfth, `non_adjacent_edges`, was **1226** — edges spanning more than one
frame (dt=2: 699, dt=3: 527). Lifting the gap-close clamp is what creates them:
the pub162 port that scored LB 0.915 has **0** non-adjacent edges and its audit
passes. So the audit is correctly enforcing an invariant of the public pipeline
and our config violates it. The autosubmit gate worked exactly as designed —
it logged FATAL and refused to submit rather than ship an unvetted artifact.

THE KEY MEASUREMENT (row 83)
----------------------------
GT edges are all t -> t+1, so a dt=2 or dt=3 prediction can never match one.
Stripping all 6,588 non-adjacent edges from the measured stack submission and
re-scoring with the official scorer gives:

    before  0.9135  edge_J 0.9037  adj 0.9065  div_J 0.0698  node_recall 0.9878
    after   0.9135  edge_J 0.9037  adj 0.9065  div_J 0.0698  node_recall 0.9878

Identical to four decimals in every term. **The metric is inert to them** — it
counts them neither as true nor as false positives. Two consequences: the
stack's +0.0036 does NOT come from non-adjacent edges and survives their
removal intact, and a compliant submission is available at exactly the same
score. This filter is therefore free: it buys audit compliance for nothing.

WHERE THE FILTER GOES
---------------------
Immediately after `filter_output_graph` returns and before the writer loop, so
the dropped edges never reach submission.csv, the audit, or run_stats. Verified
against the real failed Kaggle output: 1226 non-adjacent -> 0, 243,511 rows.

    .venv/bin/python scripts/build_stack_adjonly_kernel.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kernels/pub162_stack/pub162_stack.ipynb"
OUT_DIR = REPO / "kernels/pub162_stack_adjonly"
CELL = 15

OLD = """        raw_node_count = len(nodes_by_id)
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")
"""

NEW = """        raw_node_count = len(nodes_by_id)
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")

        # --- row 83: drop edges spanning more than one frame -----------------
        # Lifting the gap-close clamp emits dt=2/dt=3 edges. cell18's audit
        # requires non_adjacent_edges == 0 and killed the 08-15 commit run over
        # 1226 of them. GT edges are all t->t+1, so these can never match; the
        # official scorer is INERT to them (0.9135 -> 0.9135, every term equal),
        # which makes this filter free. Nodes are untouched: gap-close keeps its
        # node-level recovery, which is where the +0.0036 actually lives.
        _adj_edges = []
        _nonadj = 0
        for _e in edges:
            _s = nodes_by_id.get(int(_e["source_id"]))
            _t = nodes_by_id.get(int(_e["target_id"]))
            if _s is not None and _t is not None and int(_t["t"]) - int(_s["t"]) != 1:
                _nonadj += 1
                continue
            _adj_edges.append(_e)
        edges = _adj_edges
        _NONADJ_DROPPED[dataset] = _nonadj
"""

ANCHOR_OLD = """stats_rows: list[dict[str, object]] = []
seen_datasets: set[str] = set()"""

ANCHOR_NEW = """stats_rows: list[dict[str, object]] = []
seen_datasets: set[str] = set()
_NONADJ_DROPPED: dict[str, int] = {}"""

REPORT_OLD = """print(f"Wrote {SUBMISSION_PATH} with {row_id:,} rows")"""

REPORT_NEW = """print(f"[adjonly] dropped {sum(_NONADJ_DROPPED.values()):,} non-adjacent edges "
      f"across {len(_NONADJ_DROPPED)} datasets (audit requires non_adjacent_edges == 0)")
print(f"Wrote {SUBMISSION_PATH} with {row_id:,} rows")"""


def main() -> None:
    nb = json.loads(SRC.read_text())
    src = "".join(nb["cells"][CELL]["source"])

    for tag, old, new in (("filter", OLD, NEW),
                          ("anchor", ANCHOR_OLD, ANCHOR_NEW),
                          ("report", REPORT_OLD, REPORT_NEW)):
        n = src.count(old)
        assert n == 1, f"{tag}: expected 1 match, found {n} — kernel changed, DO NOT SHIP"
        src = src.replace(old, new)
        print(f"  [patch] {tag}: 1 match applied")

    nb["cells"][CELL]["source"] = src.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pub162_stack_adjonly.ipynb").write_text(json.dumps(nb))

    meta_src = SRC.parent / "kernel-metadata.json"
    meta = json.loads(meta_src.read_text())
    meta["id"] = meta["id"].replace("biohub-pub162-stack", "biohub-pub162-stack-adjonly")
    meta["title"] = "biohub pub162 stack adjonly"
    meta["code_file"] = "pub162_stack_adjonly.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {OUT_DIR}/pub162_stack_adjonly.ipynb")
    print(f"      {OUT_DIR}/kernel-metadata.json  (id={meta['id']})")
    print("\nNOT pushed. Push consumes GPU quota and submits a DIFFERENT artifact")
    print("than the one row 66 verified — that is a decision for the operator.")


if __name__ == "__main__":
    main()
