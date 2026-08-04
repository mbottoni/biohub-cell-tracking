#!/usr/bin/env python3
"""STRATEGY 18.4 -- dump the learned edge probabilities so the bonus bet becomes a measurement.

18.2 established that the 65 swaps are geometrically unreachable: 100% took a
shorter link than the correct one, and simulating the relinker's own cost flips
0/65 at every velocity weight. So the learned edge probability is the ONLY term
in `cost = motion + 0.05*raw - BONUS*prob` that can recover them.

But `motion_relink_edges.learned_prob` returns **0.0** for any pair absent from
`learned_edge_probs`, and that dict is built only from `edges` -- the raw
transformer candidates that survived the edge threshold and the length gate. So
there are two very different worlds and we cannot currently tell them apart:

  A) the correct edge IS a candidate with a decent prob -> raising BONUS can work,
     and 18.3's margin table says it needs to reach ~10.
  B) the correct edge is NOT a candidate at all (prob = 0) -> the bonus term is
     identically zero there and NO reweighting can ever recover it. The lever is
     then a new association model, not a knob.

This kernel writes one row per candidate edge (dataset, source, target, prob) to
`/kaggle/working/edge_probs.csv`, which `scripts/local/edge_prob_audit.py` joins
against the 65 known-wrong cases from `link_ceiling.py`.

Inference is untouched -- this is a pure observation run and its submission.csv
must be byte-identical to d4edge's. That is asserted locally after the run.

Run: python scripts/build_probdump_kernel.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"

ANCHOR = """        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)"""

DUMP = '''        # STRATEGY 18.4: observation only -- write the candidate edges and their
        # transformer probabilities, then carry on unchanged. learned_prob() is
        # 0.0 for any pair missing from this dict, so "is the correct edge even a
        # candidate?" decides whether MOTION_RELINK_LEARNED_BONUS is a lever at
        # all. Appending keeps one file across all volumes.
        try:
            import csv as _csv, os as _os
            _p = "/kaggle/working/edge_probs.csv"
            _new = not _os.path.exists(_p)
            with open(_p, "a", newline="") as _f:
                _w = _csv.writer(_f)
                if _new:
                    _w.writerow(["dataset", "source_id", "target_id", "prob"])
                for (_s, _t), _pr in learned_edge_probs.items():
                    _w.writerow([dataset, _s, _t, f"{float(_pr):.6f}"])
            print(f"[18.4] wrote {len(learned_edge_probs)} edge probs for {dataset}", flush=True)
        except Exception as _e:  # never let observability break the run
            print(f"[18.4] edge-prob dump failed: {_e!r}", flush=True)

        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)'''

DIAG = ('os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"',
        'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"')


def slug(title: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def main() -> None:
    nb = json.loads(BASE_NB.read_text())

    cfg = "".join(nb["cells"][4]["source"])
    assert cfg.count(DIAG[0]) == 1
    nb["cells"][4]["source"] = cfg.replace(*DIAG, 1).splitlines(keepends=True)

    cell = "".join(nb["cells"][11]["source"])
    if cell.count(ANCHOR) != 1:
        raise RuntimeError(f"anchor count = {cell.count(ANCHOR)}, expected 1")
    cell = cell.replace(ANCHOR, DUMP, 1)
    # `dataset` must be in scope at the insertion point, or the dump silently
    # falls into the except branch and we learn nothing.
    fn = cell[cell.find("def filter_output_graph"):]
    sig = fn[:fn.find(")")]
    if "dataset" not in sig:
        raise RuntimeError("filter_output_graph does not take `dataset` -- fix the dump scope")
    ast.parse(cell)
    nb["cells"][11]["source"] = cell.splitlines(keepends=True)

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))

    name, title = "probdump", "Biohub edge prob dump"
    out = REPO / "kernels" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"maruanbottoni/{slug(title)}", "title": title,
        "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
    }, indent=2))
    print(f"wrote kernels/{name}  ->  maruanbottoni/{slug(title)}")


if __name__ == "__main__":
    main()
