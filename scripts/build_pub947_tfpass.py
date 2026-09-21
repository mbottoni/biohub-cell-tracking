#!/usr/bin/env python3
"""Build pub947_tfpass — a pass-0 that hard-commits confident transformer pairs
BEFORE the geometric relink. The one survivor of the row-153 route audit.

WHY THIS MECHANISM EXISTS (row 152). The edges we submit are not the solver's.
`filter_output_graph` does `edges = motion_edges` whenever the relinker returns
anything, so the final linking is a per-frame-pair 1-1 Hungarian on

    cost = motion_dist + 0.05 * raw_dist - MOTION_RELINK_LEARNED_BONUS * prob

with 20,143 of 20,326 edges replaced on 44b6_0b24845f. The transformer supplies the
node set and a BONUS; geometry decides. Row 151 then measured what that costs: the
wrong link is shorter than the correct one in 48 of 49 swaps, which is exactly the
failure mode a distance-first assignment produces.

The obvious fix — hand the whole frame to the transformer (a global prob-Hungarian
pass-0) — was REFUTED by both skeptics: it hands over 95.0-99.5% of all final edges,
bypassing the tight gate whose own dose-response is brutal (§33/T1-r68: tight 6->9 =
-0.0209, 6->12 = -0.1255), and its extra gain over the hedged form is 100% one
149-GT-edge volume at 5.7% weight. So this builds the HEDGED form only:

    pass-0 commits (a,b) iff  prob >= P0  AND  (a,b) is MUTUAL-TOP  AND  dist <= GATE
    then tight and relaxed run COMPLETELY UNCHANGED on whatever is left.

Mutual-top means b is a's highest-prob target and a is b's highest-prob source, over
the whole volume's candidate pool. It is the cheapest available proxy for "the
transformer is not merely confident, it is unambiguous", and it is what makes this a
first-refusal rule rather than a replacement linker.

POOL VERIFICATION (done before building, `<scratch>/pool_compare.py`). The simulation
that motivated this (`scripts/local/analysis947/skep_p0.py`) drew candidates from the
PRE-ILP cache, while `filter_output_graph` builds `learned_edge_probs` from the geff.
Those could have been different pools. They are not: the geff retains ALL candidate
edges plus a `solution` mask, and `raw_edges` is built from `graph.edge_attrs()`
WITHOUT filtering on it. Measured over holdout6: 104,295 cache pairs vs 103,184 geff
edges (the difference is candidates whose endpoints the ILP node filter dropped), and
PC90-eligible pairs are 46,094 in BOTH pools. The mechanism is buildable in-pipeline
exactly as simulated.

WHAT THE LOCAL GATE MUST STILL SETTLE. The simulation ran on the FINAL node set
(20,719 on 0b24845f); the real pass-0 runs on the POST-ILP node set (22,491, +8.6%,
mostly short-track competitors that the min-track-len filter removes later). More
competitors change every Hungarian it precedes, so the simulated +0.0049 is an
estimate, not a result. That is the whole point of the gate.

KNOBS (all default to OFF, so an unset build is byte-identical to densad):
    BIOHUB_TFPASS_P0        probability floor; 0 disables the pass entirely (default 0)
    BIOHUB_TFPASS_GATE_UM   distance ceiling for a pre-commit (default 14.0 = OUTPUT_EDGE_MAX_UM)
    BIOHUB_TFPASS_MUTUAL    1 = require mutual-top (default), 0 = confidence only

Base is `pub947_densad` (LB 0.948, sub 56313419) and NOT densad_tuned, so the read is
ONE mechanism against the confirmed anchor. No guard-table edit is needed: the guards
check the value of knobs that already exist, and these are new names.

    python scripts/build_pub947_tfpass.py
"""
import json
import os

NB_IN = 'kernels/pub947_densad/pub947_densad.ipynb'
NAME = 'pub947_tfpass'

# --- anchor 1: volume-level precompute, right after the position table is built -----
OLD_A = (
    "    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}\n"
    "    predecessor_position_um: dict[int, np.ndarray] = {}\n"
    "    selected_edges: list[dict[str, object]] = []\n"
)

NEW_A = (
    "    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}\n"
    "    predecessor_position_um: dict[int, np.ndarray] = {}\n"
    "    selected_edges: list[dict[str, object]] = []\n"
    "\n"
    "    # ==== pass-0: hard pre-commit of confident mutual-top transformer pairs (row 155) ====\n"
    "    # Runs BEFORE tight/relaxed and removes what it takes from their pools; the two\n"
    "    # geometric passes are otherwise untouched. P0 <= 0 disables it entirely.\n"
    "    _tf_p0 = float(os.environ.get('BIOHUB_TFPASS_P0', '0'))\n"
    "    _tf_gate = float(os.environ.get('BIOHUB_TFPASS_GATE_UM', '14.0'))\n"
    "    _tf_mutual_required = os.environ.get('BIOHUB_TFPASS_MUTUAL', '1') != '0'\n"
    "    _tf_by_source_frame: dict[int, list[tuple[int, int, float]]] = {}\n"
    "    _tf_mutual: set[tuple[int, int]] = set()\n"
    "\n"
    "    if _tf_p0 > 0.0 and learned_edge_probs:\n"
    "        _tf_best_source: dict[int, tuple[float, int]] = {}\n"
    "        _tf_best_target: dict[int, tuple[float, int]] = {}\n"
    "\n"
    "        for _tf_a, _tf_b in learned_edge_probs:\n"
    "            if _tf_a not in nodes_by_id or _tf_b not in nodes_by_id:\n"
    "                continue\n"
    "            _tf_p = learned_prob(_tf_a, _tf_b)\n"
    "\n"
    "            if _tf_a not in _tf_best_source or (_tf_p, _tf_b) > _tf_best_source[_tf_a]:\n"
    "                _tf_best_source[_tf_a] = (_tf_p, _tf_b)\n"
    "\n"
    "            if _tf_b not in _tf_best_target or (_tf_p, _tf_a) > _tf_best_target[_tf_b]:\n"
    "                _tf_best_target[_tf_b] = (_tf_p, _tf_a)\n"
    "\n"
    "            if _tf_p >= _tf_p0:\n"
    "                _tf_by_source_frame.setdefault(int(nodes_by_id[_tf_a]['t']), []).append((_tf_a, _tf_b, _tf_p))\n"
    "        _tf_mutual = {(_tf_s, _tf_v[1]) for _tf_s, _tf_v in _tf_best_source.items() if _tf_best_target.get(_tf_v[1], (0.0, -1))[1] == _tf_s}\n"
    "\n"
    "        for _tf_list in _tf_by_source_frame.values():\n"
    "            _tf_list.sort(key = lambda _tf_row: (-_tf_row[2], _tf_row[0], _tf_row[1]))\n"
    "        stats['motion_relink_precommit_pool'] = sum((len(_tf_v) for _tf_v in _tf_by_source_frame.values()))\n"
    "        stats['motion_relink_precommit_mutual'] = len(_tf_mutual)\n"
    "        print('[tfpass] p0=' + str(_tf_p0) + ' gate=' + str(_tf_gate) + ' mutual=' + str(int(_tf_mutual_required)) + ' pool=' + str(stats['motion_relink_precommit_pool']) + ' mutual_pairs=' + str(len(_tf_mutual)), flush = True)\n"
    "    # ====================================================================================\n"
)

# --- anchor 2: per-frame commit, immediately before the tight/relaxed passes --------
OLD_B = (
    "        frame_matches: list[tuple[int, int, float, float, str, float]] = []\n"
    "\n"
    "        for pass_name, gate_um in (('tight', MOTION_RELINK_TIGHT_UM), ('relaxed', MOTION_RELINK_RELAXED_UM)):\n"
)

NEW_B = (
    "        frame_matches: list[tuple[int, int, float, float, str, float]] = []\n"
    "\n"
    "        if _tf_p0 > 0.0:\n"
    "            for _tf_a, _tf_b, _tf_p in _tf_by_source_frame.get(t, ()):\n"
    "                if _tf_a not in unmatched_sources or _tf_b not in unmatched_targets:\n"
    "                    continue\n"
    "\n"
    "                if _tf_mutual_required and (_tf_a, _tf_b) not in _tf_mutual:\n"
    "                    continue\n"
    "                _tf_raw = float(np.linalg.norm(position_um[_tf_b] - position_um[_tf_a]))\n"
    "\n"
    "                if _tf_raw > _tf_gate:\n"
    "                    continue\n"
    "                unmatched_sources.remove(_tf_a)\n"
    "                unmatched_targets.remove(_tf_b)\n"
    "                frame_matches.append((_tf_a, _tf_b, _tf_raw, _tf_raw, 'precommit', _tf_p))\n"
    "                stats['motion_relink_precommit_edges'] = stats.get('motion_relink_precommit_edges', 0) + 1\n"
    "\n"
    "        for pass_name, gate_um in (('tight', MOTION_RELINK_TIGHT_UM), ('relaxed', MOTION_RELINK_RELAXED_UM)):\n"
)

# the tight/relaxed bookkeeping branch must not miscount a pre-committed edge
OLD_C = (
    "                if pass_name == 'tight':\n"
    "                    stats['motion_relink_tight_edges'] += 1\n"
    "                else:\n"
    "                    stats['motion_relink_relaxed_edges'] += 1\n"
)


def main() -> None:
    nb = json.load(open(NB_IN))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 1, f'expected one code cell, found {len(cells)}'
    src = ''.join(cells[0]['source'])

    for label, old in (('A/precompute', OLD_A), ('B/frame-commit', OLD_B), ('C/pass-bookkeeping', OLD_C)):
        n = src.count(old)
        assert n == 1, f'anchor {label}: {n} matches (expected 1)'

    src = src.replace(OLD_A, NEW_A, 1)
    src = src.replace(OLD_B, NEW_B, 1)

    # Guard against silent no-ops (CLAUDE.md): prove both blocks are present.
    assert src.count("_tf_by_source_frame.setdefault") == 1
    assert src.count("'precommit'") == 1
    assert src.count("BIOHUB_TFPASS_P0") == 1
    # The pre-commit runs before the tight/relaxed loop, so its edges can never be
    # counted as tight/relaxed: the bookkeeping branch is inside that loop only.
    assert src.index("frame_matches.append((_tf_a") < src.index("for pass_name, gate_um in")

    compile(src, 'tfpass_cell', 'exec')
    cells[0]['source'] = src.splitlines(keepends=True)

    os.makedirs(f'kernels/{NAME}', exist_ok=True)
    json.dump(nb, open(f'kernels/{NAME}/{NAME}.ipynb', 'w'))

    meta_in = f'kernels/pub947_densad/kernel-metadata.json'
    meta = json.load(open(meta_in))
    meta['id'] = 'maruanbottoni/biohub-pub947-tfpass'
    meta['title'] = 'biohub pub947 tfpass'
    meta['code_file'] = f'{NAME}.ipynb'
    json.dump(meta, open(f'kernels/{NAME}/kernel-metadata.json', 'w'), indent=2)

    print(f'{NAME}: built on {NB_IN} (pass-0 OFF by default; set BIOHUB_TFPASS_P0 to arm)')


if __name__ == '__main__':
    main()
