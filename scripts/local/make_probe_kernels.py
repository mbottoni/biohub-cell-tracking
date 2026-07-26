#!/usr/bin/env python3
"""Generate the node-ratio probe kernels from the 0.912 two-seeds notebook.

Both probes are derived mechanically from `kernels/two_seeds_repro/` so the ONLY difference
from our best confirmed submission is the thing being measured.

Why these two (see docs/STRATEGY.md 10.5):
  score = edge_J * (1 - 0.1*ratio) + 0.1*div_J,  ratio = (N_pred - est_n_nodes)/est_n_nodes

  probe_pad    adds edgeless nodes far outside the volume. They cannot match a GT node (7um
               match radius) and create no predicted edges, so they move N_pred and NOTHING
               else. Score delta isolates the multiplier term: ~-0.033 at pad 0.5 if the term
               is live and unclamped, ~-0.011 if negative ratio is clamped at 0.
  probe_len20  raises OUTPUT_MIN_TRACK_LEN 6 -> 20 (config only). GT tracks are ~37 nodes
               (median, docs/gt_stats.csv), so this should shed mostly-unannotated short
               components -> big negative ratio at little edge_J cost. This is the real lever.

Usage: python3 scripts/local/make_probe_kernels.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_DIR = REPO / "kernels" / "two_seeds_repro"
BASE_NB = BASE_DIR / "biohub-cell-tracking-two-seeds-logit-blend.ipynb"

PAD_DECL = '''
# --- PROBE (node-ratio) -------------------------------------------------------------------
# Fraction of edgeless "far-field" nodes to append per dataset. 0.0 = identical to the 0.912 run.
NODE_PAD_FRAC = float(os.environ.get("BIOHUB_NODE_PAD_FRAC", "0.0"))
print(f"NODE_PAD_FRAC = {NODE_PAD_FRAC}")
'''

PAD_ANCHOR = '''        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")
'''

PAD_BLOCK = '''
        # --- PROBE: measure the node-ratio multiplier -------------------------------------
        # These nodes carry no edges and sit far outside the 104um volume, so they cannot
        # match a GT node and cannot produce a predicted edge. They move N_pred and nothing
        # else -> the score delta is a clean read of (1 - 0.1*total_node_ratio).
        if NODE_PAD_FRAC > 0.0:
            pad_n = int(round(NODE_PAD_FRAC * len(nodes_by_id)))
            next_pad_id = max(nodes_by_id) + 1
            for pad_k in range(pad_n):
                pad_id = next_pad_id + pad_k
                nodes_by_id[pad_id] = {
                    "node_id": pad_id,
                    "t": 0,
                    "z": 100000,
                    "y": 100000,
                    "x": 100000 + pad_k,
                }
            print(f"{dataset}: PROBE padded {pad_n:,} isolated nodes -> {len(nodes_by_id):,} nodes")
'''


def load_base() -> dict:
    return json.loads(BASE_NB.read_text())


def cell_src(nb: dict, idx: int) -> str:
    return "".join(nb["cells"][idx]["source"])


def set_cell(nb: dict, idx: int, src: str) -> None:
    nb["cells"][idx]["source"] = src.splitlines(keepends=True)


def replace_once(src: str, old: str, new: str, what: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"anchor for {what!r} matched {src.count(old)} times, expected 1")
    return src.replace(old, new)


# Kaggle creates a kernel under the slug derived from its TITLE, not the id we send, so these
# are the slugs the first push actually created. Keep them pinned or a re-push forks a new kernel.
KERNEL_IDS = {
    "probe_pad": "maruanbottoni/biohub-probe-pad-node-ratio-diagnostic",
    "probe_len20": "maruanbottoni/biohub-probe-min-track-len-20",
}


def write_kernel(slug: str, title: str, nb: dict, tag: str) -> None:
    out = REPO / "kernels" / slug
    out.mkdir(parents=True, exist_ok=True)

    cfg = cell_src(nb, 5)
    cfg = replace_once(
        cfg,
        'EXPERIMENT_TAG = "selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap"',
        f'EXPERIMENT_TAG = "{tag}"',
        "EXPERIMENT_TAG",
    )
    set_cell(nb, 5, cfg)

    nb_name = f"{slug}.ipynb"
    (out / nb_name).write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": KERNEL_IDS[slug],
                "title": title,
                "code_file": nb_name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "machine_shape": "NvidiaTeslaT4",
                "enable_internet": False,
                "competition_sources": ["biohub-cell-tracking-during-development"],
                "dataset_sources": [
                    "pilkwang/biohub-tracking-support-pack-50ep-v1",
                    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
                    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote kernels/{slug}/ ({nb_name})")


def build_probe_pad() -> None:
    nb = load_base()

    cfg = cell_src(nb, 4)
    cfg = replace_once(
        cfg,
        'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"',
        'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"\n'
        '\n# PROBE: append 0.5 * N_pred edgeless far-field nodes (diagnostic only, see 10.5).\n'
        'os.environ["BIOHUB_NODE_PAD_FRAC"] = "0.5"',
        "pad env var",
    )
    set_cell(nb, 4, cfg)

    main = cell_src(nb, 11)
    main = replace_once(
        main,
        'CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]\n',
        'CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]\n' + PAD_DECL,
        "pad declaration",
    )
    main = replace_once(main, PAD_ANCHOR, PAD_ANCHOR + PAD_BLOCK, "pad block")
    set_cell(nb, 11, main)

    write_kernel("probe_pad", "Biohub probe pad (node-ratio diagnostic)", nb, "probe_node_pad_0p5")


def build_probe_len20() -> None:
    nb = load_base()

    cfg = cell_src(nb, 4)
    cfg = replace_once(
        cfg,
        'os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"',
        '# PROBE: 6 -> 20. GT tracks are ~37 nodes; short components are mostly unannotated,\n'
        '# so dropping them should buy node-ratio bonus at little edge_J cost (see 10.5).\n'
        'os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "20"',
        "min track len",
    )
    set_cell(nb, 4, cfg)

    write_kernel("probe_len20", "Biohub probe min-track-len 20", nb, "probe_min_track_len_20")


if __name__ == "__main__":
    if not BASE_NB.exists():
        raise SystemExit(f"missing base notebook: {BASE_NB}")
    build_probe_pad()
    build_probe_len20()
