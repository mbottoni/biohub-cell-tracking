#!/usr/bin/env python3
"""End-to-end test of the patched safe-division heuristic, on real GT divisions.

Runs the ACTUAL `add_safe_divisions_postlink` lifted out of each built notebook
(plus that notebook's own `BIOHUB_SAFE_DIV_*` config), so this exercises the
shipped code path rather than a reimplementation.

Test construction, per volume: take the GT graph as the "predicted" graph, then
for every real division delete ONE of the two child edges. That leaves the
parent with exactly one child and the deleted daughter as an orphan at t+1 --
precisely the situation the heuristic is built to repair. A correct heuristic
re-inserts that edge.

  recall    = fraction of real divisions whose exact edge was re-inserted
  decoys    = insertions that are NOT the held-out division edge

CAVEAT: GT is a ~1.6-3.6% sample of cells, so the orphan population here is far
sparser than at inference. RECALL transfers; the decoy count is optimistic and
must not be read as a false-positive rate. See STRATEGY 15.

    python scripts/local/test_divgeom.py
"""
from __future__ import annotations

import ast
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ctrack.data.geff import find_geff_stores, load_geff  # noqa: E402

GT_ROOT = "/Volumes/extSSD/kaggle/biohub/gt/train"
NEEDED = ("edge_distance_um", "node_point", "add_safe_divisions_postlink")


def load_impl(kernel: str) -> tuple[callable, dict]:
    """Lift the safe-division code + its config out of a built notebook."""
    nb = json.loads((REPO / f"kernels/{kernel}/{kernel}.ipynb").read_text())
    cfg_src = "".join(nb["cells"][4]["source"])
    cell11 = "".join(nb["cells"][11]["source"])

    env: dict[str, str] = {}
    for node in ast.walk(ast.parse(cfg_src)):
        if (isinstance(node, ast.Assign)
                and isinstance(t := node.targets[0], ast.Subscript)
                and getattr(t.value, "attr", "") == "environ"):
            env[ast.literal_eval(t.slice)] = ast.literal_eval(node.value)

    def f(name, default):
        return float(env.get(f"BIOHUB_{name}", default))

    ns: dict = {
        "math": math, "np": np,
        "VOXEL_SCALE_UM": (1.625, 0.40625, 0.40625),
        "OUTPUT_SAFE_DIVISIONS": True,
        "SAFE_DIV_MAX_UM": f("SAFE_DIV_MAX_UM", 4.7),
        "SAFE_DIV_SISTER_MAX_UM": f("SAFE_DIV_SISTER_MAX_UM", 7.2),
        "SAFE_DIV_EXISTING_CHILD_MAX_UM": f("SAFE_DIV_EXISTING_CHILD_MAX_UM", 7.2),
        "SAFE_DIV_FRAME_FRAC_CAP": f("SAFE_DIV_FRAME_FRAC_CAP", 0.0076),
        "SAFE_DIV_GLOBAL_FRAC_CAP": f("SAFE_DIV_GLOBAL_FRAC_CAP", 0.00375),
        # The veto needs the DeepCenter model; it is exercised on Kaggle, not here.
        "DEEPCENTER_SAFE_DIV_VETO": False,
        "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.25,
        "deepcenter_accept_repair_point": lambda *a, **k: True,
    }
    tree = ast.parse(cell11)
    found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = [n for n in NEEDED if n not in found]
    if missing:
        raise RuntimeError(f"{kernel}: missing {missing}")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NEEDED:
            exec(compile(ast.Module([node], []), "<nb>", "exec"), ns)  # noqa: S102
    return ns["add_safe_divisions_postlink"], {
        k: ns[k] for k in
        ("SAFE_DIV_MAX_UM", "SAFE_DIV_SISTER_MAX_UM", "SAFE_DIV_EXISTING_CHILD_MAX_UM",
         "SAFE_DIV_FRAME_FRAC_CAP", "SAFE_DIV_GLOBAL_FRAC_CAP")
    }



def seed_stats(kernel: str) -> dict:
    """Reproduce filter_output_graph's PLAIN pre-seeded stats dict.

    It is not a defaultdict, so any counter the patch introduces must also be
    declared in that literal. divgeom v1 died on Kaggle with
    `KeyError: safe_division_frame_cap_hit` because this fixture used a
    defaultdict and silently absorbed the missing key. Parse the real literal
    out of the notebook so the fixture cannot be more permissive than the
    caller.
    """
    nb = json.loads((REPO / f"kernels/{kernel}/{kernel}.ipynb").read_text())
    cell11 = "".join(nb["cells"][11]["source"])
    i = cell11.find("    stats = {")
    lit = cell11[cell11.find("{", i):cell11.find("}", i) + 1]
    keys = re.findall(r'"([a-z0-9_]+)":', lit)
    return {k: 0 for k in keys}


def build_case(path: Path):
    """GT graph with one child edge removed per real division."""
    gt = load_geff(path)
    g = gt.graph
    nodes = {int(r[0]): {"t": int(r[1]), "z": float(r[2]), "y": float(r[3]), "x": float(r[4])}
             for r in g.nodes}
    kids = defaultdict(list)
    for a, b in g.edges:
        kids[int(a)].append(int(b))

    held: set[tuple[int, int]] = set()
    for par, ks in kids.items():
        if len(ks) >= 2:
            # Hold out the daughter the linker would be least likely to pick:
            # the farther one. Deterministic, no RNG.
            far = max(ks[:2], key=lambda k: math.dist(
                (nodes[k]["z"] * 1.625, nodes[k]["y"] * 0.40625, nodes[k]["x"] * 0.40625),
                (nodes[par]["z"] * 1.625, nodes[par]["y"] * 0.40625, nodes[par]["x"] * 0.40625)))
            held.add((par, far))

    edges = [{"source_id": int(a), "target_id": int(b)}
             for a, b in g.edges if (int(a), int(b)) not in held]
    return nodes, edges, held


def run(kernel: str, uncap: bool = False) -> dict:
    """`uncap` lifts the frac caps.

    The caps are fractions of the EDGE COUNT, and a GT graph has ~660 edges
    against ~120k in the real predicted graph -- so `global_cap` collapses to ~1
    per volume here purely as an artifact of the sparse test fixture. Running
    uncapped isolates what the gates and the ranking actually do, which is the
    part that transfers to inference.
    """
    fn, cfg = load_impl(kernel)
    if uncap:
        fn.__globals__["SAFE_DIV_GLOBAL_FRAC_CAP"] = 1.0
        fn.__globals__["SAFE_DIV_FRAME_FRAC_CAP"] = 1.0
    tot_held = tot_hit = tot_ins = 0
    frame_hits = glob_hits = 0
    for p in find_geff_stores(GT_ROOT):
        nodes, edges, held = build_case(p)
        if not held:
            continue
        stats = seed_stats(kernel)   # plain dict, exactly like filter_output_graph
        out = fn(nodes, edges, stats)
        added = {(int(e["source_id"]), int(e["target_id"]))
                 for e in out[len(edges):]} if len(out) > len(edges) else set()
        tot_held += len(held)
        tot_hit += len(held & added)
        tot_ins += len(added)
        frame_hits += stats.get("safe_division_frame_cap_hit", 0)
        glob_hits += stats.get("safe_division_skipped_cap", 0)
    return {"kernel": kernel + (" [uncapped]" if uncap else ""), "cfg": cfg,
            "held": tot_held, "hit": tot_hit, "inserted": tot_ins,
            "frame_cap_hits": frame_hits, "global_cap_hits": glob_hits}


def run_hybrid(budget: int) -> list[dict]:
    """Wide gates + OLD ranking vs wide gates + NEW ranking, at an equal budget.

    At inference `global_cap` = 120k edges x 0.0010 = ~120 insertions while the
    widened gates will propose far more, so the cap binds and the RANKING alone
    decides what survives. This isolates the re-rank from the gate widening by
    giving the old scorer the new gates.
    """
    out = []
    for kernel, label in (("d4edge", "wide gates + OLD rank"),
                          ("divgeom", "wide gates + NEW rank")):
        fn, _ = load_impl(kernel)
        g = fn.__globals__
        g["SAFE_DIV_MAX_UM"] = 12.0
        g["SAFE_DIV_SISTER_MAX_UM"] = 14.4
        g["SAFE_DIV_EXISTING_CHILD_MAX_UM"] = 12.0
        g["SAFE_DIV_FRAME_FRAC_CAP"] = 1.0
        held_t = hit_t = ins_t = 0
        for p in find_geff_stores(GT_ROOT):
            nodes, edges, held = build_case(p)
            if not held:
                continue
            # Equal absolute budget per volume, independent of graph size.
            g["SAFE_DIV_GLOBAL_FRAC_CAP"] = budget / max(1, len(edges))
            stats = seed_stats(kernel)
            res = fn(nodes, edges, stats)
            added = {(int(e["source_id"]), int(e["target_id"]))
                     for e in res[len(edges):]} if len(res) > len(edges) else set()
            held_t += len(held); hit_t += len(held & added); ins_t += len(added)
        out.append({"label": label, "held": held_t, "hit": hit_t, "inserted": ins_t})
    return out


def main() -> None:
    rows = [run("d4edge"), run("divgeom"),
            run("d4edge", uncap=True), run("divgeom", uncap=True)]
    for r in rows:
        c = r["cfg"]
        print(f"\n=== {r['kernel']} ===")
        print(f"  gates: cand<={c['SAFE_DIV_MAX_UM']}  sister<={c['SAFE_DIV_SISTER_MAX_UM']}  "
              f"existing<={c['SAFE_DIV_EXISTING_CHILD_MAX_UM']}")
        print(f"  real divisions held out : {r['held']}")
        print(f"  correctly re-inserted   : {r['hit']}  (recall {r['hit']/max(1,r['held']):.3f})")
        print(f"  total insertions        : {r['inserted']}   decoys {r['inserted']-r['hit']}")
        if r["inserted"]:
            print(f"  precision               : {r['hit']/r['inserted']:.3f}")
        print(f"  cap breaks: frame={r['frame_cap_hits']}  global={r['global_cap_hits']}")

    base, new, base_u, new_u = rows
    print("\n=== verdict ===")
    for lab, b, n in (("capped (fixture-limited)", base, new),
                      ("uncapped (gates+ranking)", base_u, new_u)):
        br, nr = b["hit"] / max(1, b["held"]), n["hit"] / max(1, n["held"])
        print(f"  {lab:<26} recall {br:.3f} -> {nr:.3f}  ({nr-br:+.3f})")
    assert new["held"] == base["held"] == new_u["held"], "test sets differ"
    if new_u["hit"] <= base_u["hit"]:
        print("  FAIL: the re-centred config does not admit more real divisions.")
        sys.exit(1)

    print("\n=== ranking quality at an equal budget (isolates the re-rank) ===")
    for budget in (2, 5):
        print(f"  budget {budget} insertions/volume:")
        for r in run_hybrid(budget):
            prec = r["hit"] / r["inserted"] if r["inserted"] else float("nan")
            print(f"    {r['label']:<24} recall {r['hit']/max(1,r['held']):.3f}  "
                  f"precision {prec:.3f}  ({r['hit']}/{r['inserted']})")
    print("\n  PASS")


if __name__ == "__main__":
    main()
