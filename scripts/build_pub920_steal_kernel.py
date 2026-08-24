#!/usr/bin/env python
"""Build pub920_div917steal — enlarge the safe-div orphan pool by stealing weak links.

WHY (row 114): div917 = LB 0.923, and the remaining division gap (~0.007 vs the 917
family's 0.024) is in the POOL, not the filters — dv15/capx2/SD2 all rejected locally.
Our post-ILP graph claims nearly every detection, leaving too few orphans to fork to.
This patch admits, as second-daughter candidates, nodes whose SINGLE incoming edge has
edge_prob < STEAL_MAX_PROB; if such a candidate passes ALL existing filters (mutual-NN,
divergence 2.25, DeepCenter veto, caps) the weak edge is removed and the division takes
the node. The bet: a barely-above-threshold link is more often wrong than a filtered
division claim is.

Safety rails (all in the patch):
  - only single-parent nodes (single_parent_repair guarantees this shape);
  - never steal from a parent with out-degree 2 (would break an existing division);
  - never steal synthetic edges (edge_prob None) or safe-division edges;
  - a proposal whose own existing daughter was stolen this frame is skipped;
  - telemetry: safe_div_steal_pool / safe_div_stolen (no-op guard: pool must be > 0).

STEAL_MAX_PROB is env-tunable (BIOHUB_SAFE_DIV_STEAL_MAX_PROB); the local gate probes
0.55 and 0.65 — edges below the 0.48 dual-seed threshold never exist, so the band is
narrow by construction and the pool stat decides which operating point is meaningful.

ONE mechanism vs pub920_div917 (LB 0.923). Local gate: cache-hit replay (postproc-only);
reject if it does not clearly improve on div917's local 0.9412.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kernels" / "pub920_div917" / "pub920_div917.ipynb"
DST_DIR = REPO / "kernels" / "pub920_div917steal"
DST = DST_DIR / "pub920_div917steal.ipynb"

nb = json.loads(SRC.read_text())
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
s6 = "".join(cells[5]["source"])

EDITS = []

# 1. Build the weak-edge map right after the incoming/out_by_source scan.
OLD1 = """    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))
"""
NEW1 = OLD1 + """
    # STEAL PATCH (row 114): nodes whose single incoming edge is weak are candidates too.
    STEAL_ENABLE = os.environ.get("BIOHUB_SAFE_DIV_STEAL_ENABLE", "0") != "0"
    STEAL_MAX_PROB = float(os.environ.get("BIOHUB_SAFE_DIV_STEAL_MAX_PROB", "0.55"))
    weak_in_edge: dict[int, dict[str, object]] = {}
    if STEAL_ENABLE:
        _in_edges: dict[int, list[dict[str, object]]] = {}
        for edge in edges:
            _in_edges.setdefault(int(edge["target_id"]), []).append(edge)
        for _tid, _elist in _in_edges.items():
            if len(_elist) != 1:
                continue
            _e = _elist[0]
            _p = _e.get("edge_prob")
            if _p is None or float(_p) >= STEAL_MAX_PROB or _e.get("safe_division"):
                continue
            if len(out_by_source.get(int(_e["source_id"]), [])) != 1:
                continue  # never break an existing division on the old parent's side
            weak_in_edge[_tid] = _e
        stats["safe_div_steal_pool"] = len(weak_in_edge)
"""
EDITS.append((OLD1, NEW1))

# 2. Admit weak-linked nodes into the per-frame candidate pool.
OLD2 = """        candidate_ids = [node_id for node_id in child_frame_ids if node_id not in incoming and node_id not in used_targets]"""
NEW2 = """        candidate_ids = [node_id for node_id in child_frame_ids if (node_id not in incoming or node_id in weak_in_edge) and node_id not in used_targets]"""
EDITS.append((OLD2, NEW2))

# 3. Acceptance: allow stolen candidates through the incoming check; skip proposals whose
#    own daughter was stolen; record the theft.
OLD3 = """            if candidate_id in used_targets or candidate_id in incoming:
                continue
            if source_id in used_sources:
                continue
            candidate = nodes_by_id[candidate_id]
            added.append({
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": None,
                "distance_um": parent_dist,
                "safe_division": 1,
            })
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1
"""
NEW3 = """            if candidate_id in used_targets or (candidate_id in incoming and candidate_id not in weak_in_edge):
                continue
            if source_id in used_sources:
                continue
            if out_by_source[source_id][0].get("_stolen"):
                continue  # this proposal's existing daughter was stolen this frame
            candidate = nodes_by_id[candidate_id]
            added.append({
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": None,
                "distance_um": parent_dist,
                "safe_division": 1,
            })
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1
            _stolen_edge = weak_in_edge.pop(candidate_id, None)
            if _stolen_edge is not None:
                _stolen_edge["_stolen"] = 1
                stats["safe_div_stolen"] = stats.get("safe_div_stolen", 0) + 1
"""
EDITS.append((OLD3, NEW3))

# 4. Drop stolen edges from the returned edge list.
OLD4 = """    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges
"""
NEW4 = """    if added:
        stats["safe_divisions_added"] = len(added)
        _kept = [e for e in edges if not e.get("_stolen")]
        return [*_kept, *added]
    return edges
"""
EDITS.append((OLD4, NEW4))

for old, new in EDITS:
    n = s6.count(old)
    assert n == 1, f"expected exactly 1 occurrence, got {n}:\n{old[:120]}"
    s6 = s6.replace(old, new)
cells[5]["source"] = s6.splitlines(keepends=True)

# Enable knob in cell00 (default probe value; the gate sweeps it via env overrides).
s0 = "".join(cells[0]["source"])
anchor = 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.12"'
assert s0.count(anchor) == 1
s0 = s0.replace(anchor, anchor + '\nos.environ["BIOHUB_SAFE_DIV_STEAL_ENABLE"] = "1"\nos.environ["BIOHUB_SAFE_DIV_STEAL_MAX_PROB"] = "0.55"', 1)
cells[0]["source"] = s0.splitlines(keepends=True)

guard_src = "".join(cells[1]["source"])
assert "STEAL" not in guard_src

DST_DIR.mkdir(exist_ok=True)
DST.write_text(json.dumps(nb))
meta = json.loads((REPO / "kernels" / "pub920_div917" / "kernel-metadata.json").read_text())
meta["id"] = "maruanbottoni/biohub-pub920-div917steal"
meta["title"] = "biohub pub920 div917steal"
meta["code_file"] = "pub920_div917steal.ipynb"
(DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {DST} ({DST.stat().st_size} bytes); 4 cell06 edits + cell00 knobs applied")
