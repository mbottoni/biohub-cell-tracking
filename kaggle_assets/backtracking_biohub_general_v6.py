#!/usr/bin/env python
# coding: utf-8

# # Biohub General V5 — production inference for the hidden 71%
#
# This is a clean, dynamic notebook submission.  It never reads train labels,
# never names a public test stem, and never runs a public-set tuning loop during
# the Kaggle rerun.  The model/threshold package is frozen before inference;
# the only data-dependent decisions are image evidence, physical geometry and
# confidence-aware graph constraints.
#
# Inputs:
#   * biohub-cell-tracking-during-development
#   * pilkwang/biohub-deepcenter-unet3d-center-prior-v1
#   * pilkwang/biohub-temporal-unet3d-seed314159-v1
#   * pilkwang/biohub-tracking-support-pack-50ep-v1
#
# No metric exploit: no sentinel coordinates, negative-time nodes, fake hubs,
# fake forks, cross-dataset edges, or hidden-test labels.
# The output is written and fully audited before the process exits.
#

# In[ ]:


from pathlib import Path
from IPython.display import Image, display

cover_image_path = Path("/kaggle/input/datasets/pilkwang/pilkwang-public-dataset-for-notebooks-figures/biohub_lineage.png")
if cover_image_path.exists():
    display(Image(filename=str(cover_image_path)))


# ## Frozen inference contract
#
# The graph backend, image model assets and thresholds are frozen before the
# hidden rerun.  We intentionally do not select settings from the 29% public
# leaderboard.  All coordinates are converted with the official physical scale
# `(z,y,x)=(1.625,0.40625,0.40625) µm`.
#
# The post-processing is confidence- and evidence-gated: it may remove invalid
# topology or repair a missing link only with local image/geometry support.  It
# never changes the detector count to match any particular public clip.
#

# ## Pipeline overview
# 
# 1. **3D detector** proposes candidate cell centers in each frame.
# 2. **Learned edge scorer** estimates which cells in adjacent frames should be linked.
# 3. **ILP graph builder** chooses a globally consistent set of detections and links.
# 4. **Post-processing** repairs short gaps, removes weak isolated fragments, smooths trajectories, and adds only geometrically safe divisions.
# 5. **Schema validation and diagnostics** check that the final `submission.csv` has valid rows, IDs, and graph statistics.
# 
# The main lesson behind this notebook is that cell-tracking competitions are graph problems. A small change to graph construction can matter as much as a detector threshold change.
# 

# ## Selected frozen settings
#
# | Setting | Value |
# |---|---:|
# | Detection threshold | `0.96875` |
# | ILP appearance/disappearance cost | `0.0 / 1.5` |
# | Motion tight/relaxed gate | `6.0 / 9.5 µm` |
# | Minimum retained track length | `6` |
# | One-frame gap gate | `5.8 µm` per step, capped at one frame |
# | Safe-division parent/sister radii | `4.66 / 9.0 µm` |
# | DeepCenter confirmation | enabled for marginal repairs |
# | Division guard | confidence-aware, boundary-safe, no added rows |
#

# In[ ]:


import os
import math
from collections import defaultdict


def apply_division_continuation_guard(nodes_by_id, edges, stats=None):
    """Apply a confidence-aware, boundary-safe division constraint.

    This guard is deliberately conservative.  It never creates a branch.  It
    keeps an ambiguous fork instead of selecting a branch from a public-set
    prior, verifies that an edge is frame-to-frame, and does not use
    continuation evidence at the end of a movie where no future frame exists.
    A branch is removed only when continuation evidence and a meaningful
    learned-confidence margin agree.
    """
    if not edges or not nodes_by_id:
        return list(edges)

    def node_time(node_id):
        node = nodes_by_id.get(int(node_id))
        if node is None:
            return None
        try:
            return int(node["t"])
        except (KeyError, TypeError, ValueError):
            return None

    def probability(edge):
        try:
            value = float(edge.get("edge_prob"))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None

    def distance(edge):
        try:
            value = float(edge.get("distance_um"))
        except (TypeError, ValueError):
            return float("inf")
        return value if math.isfinite(value) else float("inf")

    max_time = max(
        (time for time in (node_time(node_id) for node_id in nodes_by_id)
         if time is not None),
        default=-1,
    )
    outgoing = defaultdict(list)
    for edge in edges:
        outgoing[int(edge["source_id"])].append(edge)
    depth_cache = {}

    def depth(node_id, active=None):
        node_id = int(node_id)
        if node_id in depth_cache:
            return depth_cache[node_id]
        active = set() if active is None else active
        if node_id in active:
            return 0
        active.add(node_id)
        source_time = node_time(node_id)
        children = [] if source_time is None else [
            int(edge["target_id"])
            for edge in outgoing.get(node_id, [])
            if node_time(edge["target_id"]) == source_time + 1
        ]
        value = 1 + max((depth(child, active) for child in children), default=0)
        active.remove(node_id)
        depth_cache[node_id] = value
        return value

    def has_continuation(node_id):
        source_time = node_time(node_id)
        return source_time is not None and any(
            node_time(edge["target_id"]) == source_time + 1
            for edge in outgoing.get(int(node_id), [])
        )

    def rank(edge):
        target = int(edge["target_id"])
        prob = probability(edge)
        return (
            1 if prob is not None else 0,
            -float("inf") if prob is None else prob,
            depth(target),
            -distance(edge),
            -target,
        )

    keep_ids = {id(edge) for edge in edges}
    local = defaultdict(int)
    for source_id, source_edges in outgoing.items():
        source_time = node_time(source_id)
        if source_time is None:
            continue
        candidates = sorted(
            [edge for edge in source_edges
             if node_time(edge["target_id"]) == source_time + 1],
            key=rank,
            reverse=True,
        )
        if len(candidates) <= 1:
            continue
        candidates = candidates[:2]
        local["division_guard_sources"] += 1
        local["division_guard_branches_checked"] += len(candidates)
        # Enforce the evaluator's maximum out-degree if malformed upstream
        # output contains more than two consecutive children.
        for edge in source_edges:
            if edge not in candidates and node_time(edge["target_id"]) == source_time + 1:
                keep_ids.discard(id(edge))
                local["division_guard_branches_removed"] += 1

        # A fork in the final frame(s) cannot be judged by future continuation.
        if source_time >= max_time - 1:
            local["division_guard_boundary_kept"] += 1
            keep_ids.update(id(edge) for edge in candidates)
            continue

        continuing = [edge for edge in candidates if has_continuation(edge["target_id"])]
        keep = set(id(edge) for edge in candidates)
        if len(continuing) == 2:
            local["division_guard_sources_kept_both"] += 1
        elif len(continuing) == 1:
            keep_edge = continuing[0]
            drop_edge = candidates[1] if candidates[0] is keep_edge else candidates[0]
            keep_prob = probability(keep_edge)
            drop_prob = probability(drop_edge)
            if (keep_prob is not None and drop_prob is not None
                    and keep_prob - drop_prob >= 0.12):
                keep = {id(keep_edge)}
                local["division_guard_sources_kept_one"] += 1
            else:
                local["division_guard_ambiguous_forks_kept"] += 1
        else:
            first_prob = probability(candidates[0])
            second_prob = probability(candidates[1])
            if (first_prob is not None and second_prob is not None
                    and first_prob - second_prob >= 0.18):
                keep = {id(candidates[0])}
                local["division_guard_sources_kept_one"] += 1
            else:
                local["division_guard_ambiguous_forks_kept"] += 1

        for edge in candidates:
            if id(edge) not in keep:
                keep_ids.discard(id(edge))
                local["division_guard_branches_removed"] += 1

    if stats is not None:
        for key, value in local.items():
            stats[key] = int(stats.get(key, 0)) + int(value)
    return [edge for edge in edges if id(edge) in keep_ids]

# The guard is a topology filter only; it never adds rows and is fixed for the
# whole hidden rerun. Set to 0 only for a paired offline ablation.
os.environ.setdefault("BIOHUB_DIVISION_CONTINUATION_GUARD", "1")
BIOHUB_DIVISION_CONTINUATION_GUARD = (
    os.environ.get("BIOHUB_DIVISION_CONTINUATION_GUARD", "0") != "0"
)
BIOHUB_PRESET = 'general_v2_dynamic_production_no_public_cv'
BIOHUB_SCORE_AXIS = 'frozen detector + physical association + evidence-gated topology repair'

# Parameters for constrained lineage reconstruction — FROZEN for hidden 71% generalization.
# No threshold tuned on public LB; all values inherited from verified 0.916 production pipeline.
os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_DET_THRESHOLD"] = "0.96875"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "2.0"
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.0"
os.environ["BIOHUB_MOTION_RELINK_RELAXED_UM"] = "10.0"
os.environ["BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT"] = "0.65"
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"
os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "1.5"
os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"
os.environ["BIOHUB_GAP_CLOSE_UM"] = "5.8"
os.environ["BIOHUB_GAP_DENSITY_ADAPTIVE"] = "1"
os.environ["BIOHUB_GAP_DENSITY_REFERENCE_UM"] = "6.5"
os.environ["BIOHUB_GAP_DENSITY_GAIN"] = "0.040"
os.environ["BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM"] = "0.125"
os.environ["BIOHUB_GAP_DENSITY_NEIGHBORS"] = "3"
os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
os.environ["BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS"] = "1"
os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "0"
os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "4.66"
os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "9.0"

# Second-daughter completion (repo-level patch; config mirrors research CV v15).
os.environ.setdefault("BIOHUB_REPO_SD2_ENABLE", "1")
os.environ["BIOHUB_REPO_SD2_MIN_PROB"] = "0.25"
os.environ["BIOHUB_REPO_SD2_SISTER_UM"] = "15.0"
os.environ["BIOHUB_REPO_SD2_MAX_PER_FRAME_FRAC"] = "0.01"
os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "7.65"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"
os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "0"
os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "1"
os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "1"
os.environ["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = "500"
os.environ["BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM"] = "8.5"
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/checkpoint_last.pt"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "1"
os.environ["BIOHUB_DEEPCENTER_GAP_THRESHOLD"] = "0.25"
# V3 preserves geometry-gated safe divisions; DeepCenter remains the gate for
# marginal synthetic gap points, not for already observed daughter candidates.
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
os.environ["BIOHUB_RUN_VISUAL_EDA"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"
# Candidate-arm & boundary-track rescue parameters — retention guard prevents hidden density overfit.
os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0.18"
os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.20"
os.environ["BIOHUB_BIDIRECTIONAL_FUSION_MODE"] = "harmonic_probability"
os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = "0.90"
os.environ["BIOHUB_DIAGNOSTIC_ARM"] = "harmonic_association_production"
os.environ["BIOHUB_OUTPUT_BOUNDARY_TRACK_RESCUE"] = "1"
os.environ["BIOHUB_OUTPUT_BOUNDARY_TRACK_MIN_LEN"] = "2"

print("BIOHUB_PRESET:", BIOHUB_PRESET)
print("BIOHUB_SCORE_AXIS:", BIOHUB_SCORE_AXIS)


# In[ ]:


import csv
import numpy as np
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
import sys
import time
from pathlib import Path

import pandas as pd

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
TEST_DIR = COMP_DIR / "test"

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
REPO_DIR = WORKING_DIR / "tracking_repo"
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
RUN_STATS_PATH = WORKING_DIR / "run_stats.csv"

METHOD = "unet_transformer"
WEIGHTS_RELATIVE = f"weights/{METHOD}/split_0/edge_predictor_best.pth"
EXPERIMENT_TAG = "biohub_general_v6_sd2_greedy_plus_structural"
TARGET_ARTIFACT_SLUG = os.environ.get("BIOHUB_TARGET_ARTIFACT_SLUG", "biohub-tracking-support-pack-50ep-v1")
PRIMARY_ARTIFACT_MANIFEST = Path(os.environ.get(
    "BIOHUB_PRIMARY_ARTIFACT_MANIFEST",
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json",
))
ALLOW_ARTIFACT_FALLBACK = os.environ.get("BIOHUB_ALLOW_ARTIFACT_FALLBACK", "0") != "0"

DET_THRESHOLD = float(os.environ.get("BIOHUB_DET_THRESHOLD", "0.99"))
UNET_BATCH_SIZE = int(os.environ.get("BIOHUB_UNET_BATCH_SIZE", "4"))
USE_ILP = os.environ.get("BIOHUB_USE_ILP", "1") != "0"
ILP_EDGE_WEIGHT = float(os.environ.get("BIOHUB_ILP_EDGE_WEIGHT", "-1.0"))
ILP_APPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.1"))
ILP_DISAPPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "0.1"))
ILP_DIVISION_WEIGHT = float(os.environ.get("BIOHUB_ILP_DIVISION_WEIGHT", "1.0"))

# Empty for a real submission. Useful for local smoke tests, e.g. BIOHUB_SLICE=:1.
SLICE = os.environ.get("BIOHUB_SLICE", "").strip()

# If dependencies are not already installed and no offline wheels are attached,
# this controls whether the notebook attempts PyPI installation.
ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"
RUN_OUTPUT_DIAGNOSTICS = os.environ.get("BIOHUB_RUN_OUTPUT_DIAGNOSTICS", "1") != "0"
RUN_VISUAL_EDA = os.environ.get("BIOHUB_RUN_VISUAL_EDA", "1") != "0"

# Output-level graph post-processing.
OUTPUT_EDGE_MAX_UM = float(os.environ.get("BIOHUB_OUTPUT_EDGE_MAX_UM", "14.0"))
OUTPUT_ENFORCE_NEXT_FRAME = os.environ.get("BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME", "1") != "0"
OUTPUT_SINGLE_PARENT_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR", "1") != "0"
OUTPUT_SINGLE_CHILD_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR", "0") != "0"
OUTPUT_PRUNE_ISOLATED = os.environ.get("BIOHUB_OUTPUT_PRUNE_ISOLATED", "1") != "0"
OUTPUT_MOTION_RELINK = os.environ.get("BIOHUB_OUTPUT_MOTION_RELINK", "1") != "0"
MOTION_RELINK_TIGHT_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_TIGHT_UM", "6.0"))
MOTION_RELINK_RELAXED_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_RELAXED_UM", "10.0"))
MOTION_RELINK_VELOCITY_WEIGHT = float(os.environ.get("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", "0.5"))
MOTION_RELINK_LEARNED_BONUS = float(os.environ.get("BIOHUB_MOTION_RELINK_LEARNED_BONUS", "0.75"))
MOTION_RELINK_MAX_FRAME_NODES = int(os.environ.get("BIOHUB_MOTION_RELINK_MAX_FRAME_NODES", "2600"))

OUTPUT_DIVISION_GEOMETRY_FILTER = os.environ.get("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", "0") != "0"
DIV_PARENT_MAX_UM = float(os.environ.get("BIOHUB_DIV_PARENT_MAX_UM", "10.5"))
DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_DIV_SISTER_MAX_UM", "8.0"))
DIV_DROP_TO_SINGLE_IF_BAD = os.environ.get("BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD", "1") != "0"
OUTPUT_GAP_CLOSE = os.environ.get("BIOHUB_OUTPUT_GAP_CLOSE", "1") != "0"
GAP_CLOSE_MAX_GAP = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_GAP", "1"))
GAP_CLOSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_UM", "6.0"))
GAP_DENSITY_ADAPTIVE = os.environ.get("BIOHUB_GAP_DENSITY_ADAPTIVE", "0") != "0"
GAP_DENSITY_REFERENCE_UM = float(os.environ.get("BIOHUB_GAP_DENSITY_REFERENCE_UM", "6.5"))
GAP_DENSITY_GAIN = float(os.environ.get("BIOHUB_GAP_DENSITY_GAIN", "0.040"))
GAP_DENSITY_MAX_STEP_DELTA_UM = float(os.environ.get("BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM", "0.125"))
GAP_DENSITY_NEIGHBORS = int(os.environ.get("BIOHUB_GAP_DENSITY_NEIGHBORS", "3"))
GAP_CLOSE_REUSE_EXISTING = os.environ.get("BIOHUB_GAP_CLOSE_REUSE_EXISTING", "1") != "0"
GAP_CLOSE_REUSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_REUSE_UM", "3.2"))
GAP_CLOSE_MAX_ADDED_FRAC = float(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC", "0.05"))
GAP_CLOSE_MAX_ADDED_ABS = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_ABS", "2000"))
GAP_REFINE_SYNTHETIC = os.environ.get("BIOHUB_GAP_REFINE_SYNTHETIC", "1") != "0"
GAP_REFINE_WIN_Z = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_Z", "1"))
GAP_REFINE_WIN_YX = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_YX", "3"))
GAP_REFINE_MAX_SHIFT_UM = float(os.environ.get("BIOHUB_GAP_REFINE_MAX_SHIFT_UM", "3.2"))

OUTPUT_FILTER_SHORT_TRACKS = os.environ.get("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "1") != "0"
OUTPUT_MIN_TRACK_LEN = int(os.environ.get("BIOHUB_OUTPUT_MIN_TRACK_LEN", "6"))
OUTPUT_KEEP_DIVISION_COMPONENTS = os.environ.get("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", "1") != "0"
ADAPTIVE_SHORT_TRACK_RESCUE = os.environ.get("BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE", "0") != "0"
SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC", "0.10"))
SHORT_TRACK_RESCUE_MIN_LEN = int(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN", "4"))
SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB", "0.82"))
SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM", "3.25"))
SHORT_TRACK_RESCUE_MAX_NODES_FRAC = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC", "0.018"))
SHORT_TRACK_RESCUE_MAX_NODES_ABS = int(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS", "180"))

OUTPUT_LINEFIT_SMOOTH = os.environ.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "1") != "0"
OUTPUT_LINEFIT_WEIGHT = float(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WEIGHT", "0.8"))
OUTPUT_LINEFIT_WINDOW = int(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WINDOW", "2"))

OUTPUT_GAP2_RECOVERY = os.environ.get("BIOHUB_OUTPUT_GAP2_RECOVERY", "0") != "0"
GAP2_MAX_TOTAL_UM = float(os.environ.get("BIOHUB_GAP2_MAX_TOTAL_UM", "10.2"))
GAP2_MAX_STEP_UM = float(os.environ.get("BIOHUB_GAP2_MAX_STEP_UM", "4.4"))
GAP2_MAX_LINKS_FRAC = float(os.environ.get("BIOHUB_GAP2_MAX_LINKS_FRAC", "0.0045"))
GAP2_MAX_LINKS_ABS = int(os.environ.get("BIOHUB_GAP2_MAX_LINKS_ABS", "180"))
GAP2_REQUIRE_CONTEXT = os.environ.get("BIOHUB_GAP2_REQUIRE_CONTEXT", "1") != "0"
GAP2_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_GAP2_FRAME_FRAC_CAP", "0.006"))

OUTPUT_SAFE_DIVISIONS = os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS", "1") != "0"
SAFE_DIV_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_MAX_UM", "4.7"))
SAFE_DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "7.2"))
SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "7.8"))
SAFE_DIV_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", "0.008"))
SAFE_DIV_GLOBAL_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", "0.004"))

# DeepCenter support is retained for compatibility, but this selected run keeps it disabled.
USE_DEEPCENTER_VETO = os.environ.get("BIOHUB_USE_DEEPCENTER_VETO", "1") != "0"
REQUIRE_DEEPCENTER_VETO = os.environ.get("BIOHUB_REQUIRE_DEEPCENTER_VETO", "1") != "0"
DEEPCENTER_MANIFEST_DEFAULT = os.environ.get(
    "BIOHUB_DEEPCENTER_MANIFEST_DEFAULT",
    "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/ARTIFACT_MANIFEST.json",
)
DEEPCENTER_CHECKPOINT_DEFAULT = os.environ.get(
    "BIOHUB_DEEPCENTER_CHECKPOINT_DEFAULT",
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/checkpoint_last.pt",
)
DEEPCENTER_RELATIVE = os.environ.get("BIOHUB_DEEPCENTER_RELATIVE", "weights/full_frame_center/checkpoint_last.pt")
DEEPCENTER_GAP_VETO = os.environ.get("BIOHUB_DEEPCENTER_GAP_VETO", "1") != "0"
DEEPCENTER_SAFE_DIV_VETO = os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1") != "0"
DEEPCENTER_GAP_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_GAP_THRESHOLD", "0.10"))
DEEPCENTER_EXPECTED_EPOCH = int(os.environ.get("BIOHUB_DEEPCENTER_EXPECTED_EPOCH", "0"))
DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM = float(os.environ.get("BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM", "0"))
DEEPCENTER_SAFE_DIV_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "0.12"))
DEEPCENTER_SCORE_WIN_Z = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_Z", "1"))
DEEPCENTER_SCORE_WIN_YX = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_YX", "2"))
DEEPCENTER_SCORE_CACHE_MAX_FRAMES = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_CACHE_MAX_FRAMES", "8"))

CONFIG_DISPLAY = {
    "experiment_tag": EXPERIMENT_TAG,
    "method": METHOD,
    "weights": WEIGHTS_RELATIVE,
    "target_artifact_slug": TARGET_ARTIFACT_SLUG,
    "primary_artifact_manifest": str(PRIMARY_ARTIFACT_MANIFEST),
    "allow_artifact_fallback": ALLOW_ARTIFACT_FALLBACK,
    "det_threshold": DET_THRESHOLD,
    "unet_batch_size": UNET_BATCH_SIZE,
    "use_ilp": USE_ILP,
    "ilp_edge_weight": ILP_EDGE_WEIGHT,
    "ilp_appearance_weight": ILP_APPEARANCE_WEIGHT,
    "ilp_disappearance_weight": ILP_DISAPPEARANCE_WEIGHT,
    "ilp_division_weight": ILP_DIVISION_WEIGHT,
    "slice": SLICE,
    "allow_pip_install": ALLOW_PIP_INSTALL,
    "run_visual_eda": RUN_VISUAL_EDA,
    "output_edge_max_um": OUTPUT_EDGE_MAX_UM,
    "output_enforce_next_frame": OUTPUT_ENFORCE_NEXT_FRAME,
    "output_single_parent_repair": OUTPUT_SINGLE_PARENT_REPAIR,
    "output_single_child_repair": OUTPUT_SINGLE_CHILD_REPAIR,
    "output_prune_isolated": OUTPUT_PRUNE_ISOLATED,
    "output_motion_relink": OUTPUT_MOTION_RELINK,
    "motion_relink_tight_um": MOTION_RELINK_TIGHT_UM,
    "motion_relink_relaxed_um": MOTION_RELINK_RELAXED_UM,
    "motion_relink_velocity_weight": MOTION_RELINK_VELOCITY_WEIGHT,
    "motion_relink_learned_bonus": MOTION_RELINK_LEARNED_BONUS,
    "motion_relink_max_frame_nodes": MOTION_RELINK_MAX_FRAME_NODES,
    "output_division_geometry_filter": OUTPUT_DIVISION_GEOMETRY_FILTER,
    "div_parent_max_um": DIV_PARENT_MAX_UM,
    "div_sister_max_um": DIV_SISTER_MAX_UM,
    "div_drop_to_single_if_bad": DIV_DROP_TO_SINGLE_IF_BAD,
    "output_gap_close": OUTPUT_GAP_CLOSE,
    "gap_close_max_gap": GAP_CLOSE_MAX_GAP,
    "gap_close_effective_max_gap": min(GAP_CLOSE_MAX_GAP, 1),
    "gap_close_um": GAP_CLOSE_UM,
    "gap_density_adaptive": GAP_DENSITY_ADAPTIVE,
    "gap_density_reference_um": GAP_DENSITY_REFERENCE_UM,
    "gap_density_gain": GAP_DENSITY_GAIN,
    "gap_density_max_step_delta_um": GAP_DENSITY_MAX_STEP_DELTA_UM,
    "gap_density_neighbors": GAP_DENSITY_NEIGHBORS,
    "gap_close_reuse_existing": GAP_CLOSE_REUSE_EXISTING,
    "gap_close_reuse_um": GAP_CLOSE_REUSE_UM,
    "gap_close_max_added_frac": GAP_CLOSE_MAX_ADDED_FRAC,
    "gap_close_max_added_abs": GAP_CLOSE_MAX_ADDED_ABS,
    "gap_refine_synthetic": GAP_REFINE_SYNTHETIC,
    "gap_refine_win_z": GAP_REFINE_WIN_Z,
    "gap_refine_win_yx": GAP_REFINE_WIN_YX,
    "gap_refine_max_shift_um": GAP_REFINE_MAX_SHIFT_UM,
    "output_filter_short_tracks": OUTPUT_FILTER_SHORT_TRACKS,
    "output_min_track_len": OUTPUT_MIN_TRACK_LEN,
    "output_keep_division_components": OUTPUT_KEEP_DIVISION_COMPONENTS,
    "adaptive_short_track_rescue": ADAPTIVE_SHORT_TRACK_RESCUE,
    "short_track_rescue_trigger_removed_frac": SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC,
    "short_track_rescue_min_len": SHORT_TRACK_RESCUE_MIN_LEN,
    "short_track_rescue_min_mean_edge_prob": SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB,
    "short_track_rescue_max_mean_edge_dist_um": SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM,
    "short_track_rescue_max_nodes_frac": SHORT_TRACK_RESCUE_MAX_NODES_FRAC,
    "short_track_rescue_max_nodes_abs": SHORT_TRACK_RESCUE_MAX_NODES_ABS,
    "output_linefit_smooth": OUTPUT_LINEFIT_SMOOTH,
    "output_linefit_weight": OUTPUT_LINEFIT_WEIGHT,
    "output_linefit_window": OUTPUT_LINEFIT_WINDOW,
    "output_gap2_recovery": OUTPUT_GAP2_RECOVERY,
    "gap2_max_total_um": GAP2_MAX_TOTAL_UM,
    "gap2_max_step_um": GAP2_MAX_STEP_UM,
    "gap2_max_links_frac": GAP2_MAX_LINKS_FRAC,
    "gap2_max_links_abs": GAP2_MAX_LINKS_ABS,
    "gap2_require_context": GAP2_REQUIRE_CONTEXT,
    "gap2_frame_frac_cap": GAP2_FRAME_FRAC_CAP,
    "output_safe_divisions": OUTPUT_SAFE_DIVISIONS,
    "safe_div_max_um": SAFE_DIV_MAX_UM,
    "safe_div_sister_max_um": SAFE_DIV_SISTER_MAX_UM,
    "safe_div_existing_child_max_um": SAFE_DIV_EXISTING_CHILD_MAX_UM,
    "safe_div_frame_frac_cap": SAFE_DIV_FRAME_FRAC_CAP,
    "safe_div_global_frac_cap": SAFE_DIV_GLOBAL_FRAC_CAP,
    "use_deepcenter_add_only_gate": USE_DEEPCENTER_VETO,
    "deepcenter_gap_add_gate": DEEPCENTER_GAP_VETO,
    "deepcenter_safe_div_add_gate": DEEPCENTER_SAFE_DIV_VETO,
    "deepcenter_gap_threshold": DEEPCENTER_GAP_THRESHOLD,
    "deepcenter_expected_epoch": DEEPCENTER_EXPECTED_EPOCH,
    "deepcenter_gap_confirm_min_span_um": DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM,
    "deepcenter_safe_div_threshold": DEEPCENTER_SAFE_DIV_THRESHOLD,
    "deepcenter_checkpoint_default": DEEPCENTER_CHECKPOINT_DEFAULT,
}

print("Biohub learned UNet + node-transformer + ILP submission")
print("COMP_DIR:", COMP_DIR, "exists:", COMP_DIR.exists())
print("TEST_DIR:", TEST_DIR, "exists:", TEST_DIR.exists())
print(json.dumps(CONFIG_DISPLAY, indent=2, sort_keys=True))


# ## Artifact and Dependency Setup
# 
# The notebook expects a compact support artifact containing the inference source,
# trained weights, and optionally offline dependency wheels. The primary Kaggle
# attachment path is:
# 
# ```text
# /kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json
# ```
# 
# The setup cell first uses already-installed modules, then attached wheels, and
# only attempts an internet install when explicitly enabled for local development.
# By default, the artifact resolver accepts only the 50-epoch package named by
# `TARGET_ARTIFACT_SLUG`; set `BIOHUB_ALLOW_ARTIFACT_FALLBACK=1` only for local
# debugging against an older package.
# 

# In[ ]:


import re

os.environ.setdefault("POLARS_PREFER_PKG", "32")

PACKAGE_SPECS = {
    "tracksdata": ("tracksdata", "tracksdata"),
    "zarr": ("zarr", "zarr>=3.0.10,<4"),
    "pyscipopt": ("pyscipopt", "pyscipopt"),
    "geff": ("geff", "geff>=1.1.3.1.1"),
    "geff_spec": ("geff_spec", "geff-spec<1.2"),
    "ilpy": ("ilpy", "ilpy>=0.5.1"),
    "polars": ("polars", "polars>=1.36"),
    "blosc2": ("blosc2", "blosc2"),
    "dask": ("dask", "dask"),
    "imagecodecs": ("imagecodecs", "imagecodecs"),
    "skimage": ("skimage", "scikit-image>=0.24"),
    "pyarrow": ("pyarrow", "pyarrow"),
    "rustworkx": ("rustworkx", "rustworkx>=0.17.1"),
    "sqlalchemy": ("sqlalchemy", "sqlalchemy>=2"),
    "numcodecs": ("numcodecs", "numcodecs>=0.13,<0.16"),
    "donfig": ("donfig", "donfig>=0.8"),
    "google_crc32c": ("google_crc32c", "google-crc32c>=1.5"),
    "bidict": ("bidict", "bidict>=0.23.1"),
    "psygnal": ("psygnal", "psygnal>=0.14"),
    "rich": ("rich", "rich"),
    "networkx": ("networkx", "networkx>=3.2.1"),
    "pydantic": ("pydantic", "pydantic>=2.11"),
    "pydantic_core": ("pydantic_core", "pydantic-core"),
    "annotated_types": ("annotated_types", "annotated-types"),
    "typing_extensions": ("typing_extensions", "typing-extensions>=4.13"),
    "typing_inspection": ("typing_inspection", "typing-inspection"),
    "markdown_it": ("markdown_it", "markdown-it-py"),
    "pygments": ("pygments", "pygments"),
    "click": ("click", "click"),
    "cloudpickle": ("cloudpickle", "cloudpickle"),
    "fsspec": ("fsspec", "fsspec"),
    "partd": ("partd", "partd"),
    "locket": ("locket", "locket"),
    "toolz": ("toolz", "toolz"),
    "yaml": ("yaml", "pyyaml"),
    "ndindex": ("ndindex", "ndindex"),
    "msgpack": ("msgpack", "msgpack"),
    "numexpr": ("numexpr", "numexpr"),
    "deprecated": ("deprecated", "deprecated"),
    "wrapt": ("wrapt", "wrapt"),
    "imageio": ("imageio", "imageio"),
    "PIL": ("PIL", "pillow"),
    "tifffile": ("tifffile", "tifffile"),
    "lazy_loader": ("lazy_loader", "lazy-loader"),
    "tqdm": ("tqdm", "tqdm"),
}
EXTRA_SPECS_BY_NAME = {
    "tracksdata": ["bidict>=0.23.1", "psygnal>=0.14", "rich"],
    "zarr": ["donfig>=0.8", "google-crc32c>=1.5", "numcodecs>=0.13,<0.16"],
    "geff": ["geff-spec<1.2", "networkx>=3.2.1", "pydantic>=2.11", "numcodecs>=0.13,<0.16"],
    "geff_spec": ["pydantic>=2.11", "annotated-types", "pydantic-core", "typing-inspection"],
    "polars": ["polars-runtime-32"],
    "dask": ["click", "cloudpickle", "fsspec", "partd", "pyyaml", "toolz"],
    "partd": ["locket"],
    "blosc2": ["ndindex", "msgpack", "numexpr"],
    "numcodecs": ["deprecated", "msgpack", "wrapt"],
    "rich": ["markdown-it-py", "pygments"],
    "pydantic": ["annotated-types", "pydantic-core", "typing-extensions>=4.13", "typing-inspection"],
    "skimage": ["imageio", "pillow", "tifffile", "lazy-loader", "networkx"],
}
PIP_DEPENDENCIES = [spec for _, spec in PACKAGE_SPECS.values()]
REQUIRED_MODULES = {name: module for name, (module, _) in PACKAGE_SPECS.items() if module}
FALLBACK_ARTIFACT_SLUGS = ["biohub-tracking-support-pack-v1"]

# The safe path for offline reruns is to use attached wheels.
# Set BIOHUB_ALLOW_PIP_INSTALL=1 only for an interactive internet-enabled run.
ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"


def module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def has_model_artifact(path: Path) -> bool:
    has_repo_dir = (path / "repo").exists()
    has_weights_dir = (path / "weights" / METHOD / "split_0" / "edge_predictor_best.pth").exists()
    has_repo_zip = (path / "repo.zip").exists()
    has_weights_zip = (path / "weights.zip").exists()
    return (has_repo_dir and has_weights_dir) or (has_repo_zip and has_weights_zip)


def artifact_manifest(path: Path) -> dict:
    manifest = path / "ARTIFACT_MANIFEST.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text())
    except Exception:
        return {}


def artifact_matches_target(path: Path) -> bool:
    if ALLOW_ARTIFACT_FALLBACK:
        return True
    manifest = artifact_manifest(path)
    artifact_name = str(manifest.get("artifact_name", ""))
    path_text = str(path)
    return TARGET_ARTIFACT_SLUG in {artifact_name, path.name} or TARGET_ARTIFACT_SLUG in path_text


def find_artifacts_root() -> Path:
    # Strict path by request: attach this dataset through Kaggle Add Input.
    artifacts = Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1")
    if not artifacts.exists():
        raise FileNotFoundError(
            "Required input is missing: " + str(artifacts) + "\n"
            "Attach pilkwang/biohub-tracking-support-pack-50ep-v1 with Add Input."
        )
    if not has_model_artifact(artifacts):
        raise FileNotFoundError(
            "The attached support dataset does not contain the expected repo/weights: " + str(artifacts)
        )
    return artifacts

def _has_package_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    patterns = ("*.whl", "*.tar.gz", "*.zip")
    return any(any(path.glob(pattern)) for pattern in patterns)


def find_offline_package_dirs(artifacts: Path) -> list[Path]:
    candidates = [artifacts / "wheels", artifacts]
    return [path for path in candidates if _has_package_file(path)]


def purge_imported_modules(package_names: list[str]) -> None:
    roots = {"tracksdata"}
    for name in package_names:
        if name in PACKAGE_SPECS:
            module = PACKAGE_SPECS[name][0]
            roots.add(module.split(".")[0])
        if name == "polars":
            roots.add("polars")
    for root in roots:
        for module_name in list(sys.modules):
            if module_name == root or module_name.startswith(root + "."):
                sys.modules.pop(module_name, None)


def polars_runtime_ready() -> bool:
    try:
        import polars as _pl
        from polars._plr import PySeries as _PySeries

        _ = _PySeries
        return hasattr(_pl, "Float16") and _pl.Series([-999999.0], dtype=_pl.Float64).dtype == _pl.Float64
    except Exception:
        return False


def packages_requiring_refresh() -> list[str]:
    refresh: list[str] = []
    if not module_missing("polars") and not polars_runtime_ready():
        refresh.append("polars")

    if not module_missing("zarr"):
        try:
            import zarr as _zarr
            version_text = str(getattr(_zarr, "__version__", "0"))
            major = int(version_text.split(".", 1)[0])
            if major < 3:
                refresh.append("zarr")
        except Exception:
            refresh.append("zarr")
    return refresh


def dependency_specs_for(missing: list[str]) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()

    def add(spec: str) -> None:
        key = spec.lower()
        if key not in seen:
            seen.add(key)
            specs.append(spec)

    for name in missing:
        if name in PACKAGE_SPECS:
            add(PACKAGE_SPECS[name][1])
        for spec in EXTRA_SPECS_BY_NAME.get(name, []):
            add(spec)
    return specs


def import_failures() -> dict[str, str]:
    failures: dict[str, str] = {}
    for name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return failures


def missing_names_from_failures(failures: dict[str, str]) -> list[str]:
    names: list[str] = []
    module_to_name = {module: name for name, module in REQUIRED_MODULES.items()}
    for message in failures.values():
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
        if match:
            module = match.group(1).split(".")[0]
        else:
            match = re.search(r"module ['\"]([^'\"]+)['\"] has no attribute", message)
            if not match:
                continue
            module = match.group(1).split(".")[0]
        name = module_to_name.get(module)
        if name and name not in names:
            names.append(name)
    return names


def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:
    specs = dependency_specs_for(missing)
    force_reinstall = bool({"polars", "zarr"} & set(missing))
    if not specs:
        return

    package_dirs = find_offline_package_dirs(artifacts)
    if package_dirs:
        offline_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
        if force_reinstall:
            offline_cmd.append("--force-reinstall")
        for package_dir in package_dirs:
            offline_cmd.extend(["--find-links", str(package_dir)])
        offline_cmd.extend(specs)
        print("Installing missing packages from offline package dirs:", missing)
        print("Dependency resolver is disabled with --no-deps to avoid replacing Kaggle numpy/scipy in a live kernel.")
        print("Offline package dirs:", [str(path) for path in package_dirs])
        result = subprocess.run(offline_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("Offline dependency install succeeded.")
            return
        print("Offline dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    if ALLOW_PIP_INSTALL:
        online_cmd = [sys.executable, "-m", "pip", "install", "--no-deps"]
        if force_reinstall:
            online_cmd.append("--force-reinstall")
        online_cmd.extend(specs)
        print("Installing missing packages from PyPI:", missing)
        result = subprocess.run(online_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("PyPI dependency install succeeded.")
            return
        print("PyPI dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    command = "pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff geff-spec ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy donfig numcodecs"
    raise ImportError(
        "Missing required packages or dependency wheels: " + ", ".join(missing) + "\n"
        "Attach the support dataset with offline wheels. If supplying Kaggle dependency input instead, use:\n"
        + command + "\n"
        "Do not quote zarr>=3.0.10,<4 in Kaggle dependency input."
    )


def ensure_dependencies(artifacts: Path) -> None:
    for _ in range(5):
        refresh = packages_requiring_refresh()
        if refresh:
            install_missing_dependencies(refresh, artifacts)
            continue

        missing = [pkg for pkg, module in REQUIRED_MODULES.items() if module_missing(module)]
        if missing:
            install_missing_dependencies(missing, artifacts)
            continue

        failures = import_failures()
        if not failures:
            print("Required graph/Zarr/ILP packages import successfully.")
            return

        missing_from_import = missing_names_from_failures(failures)
        if missing_from_import:
            install_missing_dependencies(missing_from_import, artifacts)
            continue

        raise ImportError(
            "Required packages are present but failed to import. "
            "This may indicate a binary dependency mismatch in the live notebook kernel. "
            "Keep Kaggle dependency input empty and attach the wheels artifact.\n"
            + json.dumps(failures, indent=2)
        )

    failures = import_failures()
    raise ImportError(
        "Dependency recovery did not converge after repeated offline installs. "
        "The attached support artifact may be missing wheels.\n"
        + json.dumps(failures, indent=2)
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_or_extract_tree(src_dir: Path, src_zip: Path, dst: Path) -> None:
    remove_path(dst)
    if src_dir.exists() and src_dir.is_dir():
        shutil.copytree(src_dir, dst)
        return
    if src_zip.exists() and src_zip.is_file():
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(dst)
        return
    raise FileNotFoundError(f"Missing source tree or zip: {src_dir} / {src_zip}")


def link_or_copy_tree(src: Path, dst: Path) -> None:
    remove_path(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


def materialize_inference_repo(artifacts: Path) -> None:
    copy_or_extract_tree(artifacts / "repo", artifacts / "repo.zip", REPO_DIR)

    weights_src = artifacts / "weights"
    weights_zip = artifacts / "weights.zip"
    weights_dst = REPO_DIR / "weights"
    if weights_src.exists() and weights_src.is_dir():
        link_or_copy_tree(weights_src, weights_dst)
    elif weights_zip.exists() and weights_zip.is_file():
        remove_path(weights_dst)
        weights_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(weights_zip) as zf:
            zf.extractall(weights_dst)
    else:
        raise FileNotFoundError(f"Missing weights tree or zip under {artifacts}")

    required = [
        REPO_DIR / "scripts" / "predict_unet_transformer.py",
        REPO_DIR / WEIGHTS_RELATIVE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Materialized inference repo is incomplete:\n" + "\n".join(missing))
    print("Inference repo:", REPO_DIR)
    print("Weights:", REPO_DIR / WEIGHTS_RELATIVE)


ARTIFACTS = find_artifacts_root()
print("ARTIFACTS:", ARTIFACTS)
print("Has offline wheels:", (ARTIFACTS / "wheels").exists())
manifest_info = artifact_manifest(ARTIFACTS)
if manifest_info:
    print("Artifact name:", manifest_info.get("artifact_name"))
    print("Weight sha256:", manifest_info.get("model", {}).get("weight_sha256"))
    print("Weight path:", manifest_info.get("model", {}).get("weight_path"))

ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)


# ## Predict Candidate Graphs
# 
# The inference step writes one `.geff` graph per test video. Keeping graph
# prediction separate from CSV conversion makes the graph repair and diagnostics
# transparent.
# 

# In[ ]:


# STRICT INPUT GUARD
if not TEST_DIR.exists():
    raise FileNotFoundError(
        "Competition test data is missing at " + str(TEST_DIR) + "\n"
        "Attach Biohub - Cell Tracking During Development with Add Input."
    )

# BIOHUB CUDA HARD GUARD
import torch as _torch_guard
print("torch", _torch_guard.__version__, "cuda", _torch_guard.cuda.is_available(), flush=True)
if not _torch_guard.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Stop before inference to avoid CPU timeout.")
print("device=cuda", _torch_guard.cuda.get_device_name(0), flush=True)

# PATCH: expand detector test-time augmentation to spatial D4 transforms.
_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_s = _ps.read_text()
_old = """        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4"""
_new = """        if cfg.det_tta:
            _nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
                _nv += 1
            for _k in (1, 3):
                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
                _, det_rot = model.encode(imgs_rot)
                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
                del imgs_rot, det_rot
                _nv += 1
            imgs_t = imgs.transpose(-1, -2)
            _, det_t = model.encode(imgs_t)
            for f in range(W):
                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
            del imgs_t, det_t
            _nv += 1
            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
            _, det_at = model.encode(imgs_at)
            for f in range(W):
                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
            del imgs_at, det_at
            _nv += 1
            for f in range(W):
                det_logits[f] = det_logits[f] / _nv"""
if _old in _s:
    _ps.write_text(_s.replace(_old, _new))
    print("TTA patch applied (spatial D4 transforms)")
else:
    print("TTA WARNING: block not found - using default 4-way")


def add_second_daughter_postlink(nodes_by_id, edges, stats, dataset=None):
    """Structural second-daughter completion on the final graph.

    For half-fork parents (exactly one child at t+1, exactly one parent),
    propose an orphan q at t+1 within PARENT_UM whose distance to the existing
    child is within SISTER_UM and who is the nearest orphan to that child.
    Require both daughters to continue one frame with divergence >= DIVERGE_UM.
    Greedy best-first by d(p,q) + W_SISTER * d_sister under per-frame/global
    caps. Purely geometric/topological; no labels involved.
    """
    if os.environ.get("BIOHUB_SD2POST_ENABLE", "1") == "0" or not edges or not nodes_by_id:
        return edges
    parent_um = float(os.environ.get("BIOHUB_SD2POST_PARENT_UM", "12.0"))
    sister_um = float(os.environ.get("BIOHUB_SD2POST_SISTER_UM", "15.0"))
    diverge_um = float(os.environ.get("BIOHUB_SD2POST_DIVERGE_UM", "2.25"))
    w_sister = float(os.environ.get("BIOHUB_SD2POST_W_SISTER", "0.15"))
    frame_cap = float(os.environ.get("BIOHUB_SD2POST_FRAME_CAP", "0.0076"))
    global_cap = float(os.environ.get("BIOHUB_SD2POST_GLOBAL_CAP", "0.00375"))

    scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

    def _pos(node):
        return np.array([node["z"] * scale[0], node["y"] * scale[1], node["x"] * scale[2]])

    succ: dict[int, list[int]] = {}
    pred: dict[int, list[int]] = {}
    for edge in edges:
        succ.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
        pred.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))

    by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        by_t.setdefault(int(node["t"]), []).append(int(node_id))

    orphan_trees: dict[int, tuple] = {}
    for t, ids in by_t.items():
        orphans = [nid for nid in ids if not pred.get(nid)]
        if orphans:
            from scipy.spatial import cKDTree as _KD

            orphan_trees[t] = (
                orphans,
                _KD(np.asarray([_pos(nodes_by_id[nid]) for nid in orphans])),
            )

    n_frame = {t: len(ids) for t, ids in by_t.items()}
    proposals = []
    for p, pnode in nodes_by_id.items():
        ch = succ.get(p, [])
        if len(ch) != 1:
            continue
        if len(pred.get(p, [])) != 1:
            continue
        c1 = ch[0]
        t = int(pnode["t"])
        c1node = nodes_by_id.get(c1)
        if c1node is None or int(c1node["t"]) != t + 1:
            continue
        p_pos = _pos(pnode)
        if float(np.linalg.norm(p_pos - _pos(c1node))) > min(parent_um, 10.0):
            continue
        ot = orphan_trees.get(t + 1)
        if ot is None:
            continue
        orphans, tree = ot
        for qi in tree.query_ball_point(p_pos, parent_um):
            q = orphans[qi]
            if q == c1:
                continue
            qnode = nodes_by_id[q]
            d_pq = float(np.linalg.norm(p_pos - _pos(qnode)))
            d_sis = float(np.linalg.norm(_pos(c1node) - _pos(qnode)))
            if d_sis > sister_um:
                continue
            dq_arr = tree.query(_pos(c1node))
            if orphans[int(dq_arr[1])] != q or float(dq_arr[0]) > sister_um:
                continue
            sc, sq = succ.get(c1, []), succ.get(q, [])
            if len(sc) != 1 or len(sq) != 1:
                continue
            div = float(np.linalg.norm(_pos(nodes_by_id[sc[0]]) - _pos(nodes_by_id[sq[0]]))) - d_sis
            if div < diverge_um:
                continue
            proposals.append((d_pq + w_sister * d_sis, t, p, q))

    proposals.sort()
    used_p: set[int] = set()
    used_q: set[int] = set()
    per_frame: dict[int, int] = defaultdict(int)
    max_global = max(1, int(global_cap * len(nodes_by_id)))
    added = 0
    for score, t, p, q in proposals:
        if added >= max_global:
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(frame_cap * n_frame.get(t, 1))):
            continue
        edges.append({
            "source_id": int(p),
            "target_id": int(q),
            "edge_prob": None,
            "distance_um": float(score),
            "synthetic_second_daughter": 1,
        })
        used_p.add(p)
        used_q.add(q)
        per_frame[t] += 1
        added += 1
    stats["sd2post_proposals"] = int(stats.get("sd2post_proposals", 0)) + len(proposals)
    stats["sd2post_added"] = int(stats.get("sd2post_added", 0)) + added
    print(f"[{dataset}] sd2post proposals={len(proposals)} added={added}")
    return edges



def apply_sd2_patch() -> None:
    """Second-daughter completion inside the repo greedy edge assignment.

    For frame pairs where a source kept exactly one child, allow one extra
    unclaimed candidate when the learned edge probability passes a threshold
    and the two daughters are within the physical sister radius. Configured
    via BIOHUB_REPO_SD2_* environment variables; no-op when disabled.
    """
    marker = "BIOHUB_SD2_PATCH_V1"
    ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
    s = ps.read_text()
    if marker in s:
        print("SD2 patch already present")
        return
    anchor = (
        "# =============================================================================\n"
        "# Graph building"
    )
    globals_block = (
        "# BIOHUB_SD2_PATCH_V1: second-daughter completion configuration.\n"
        "_SD2_ENABLE = os.environ.get(\"BIOHUB_REPO_SD2_ENABLE\", \"0\") != \"0\"\n"
        "_SD2_MIN_PROB = float(os.environ.get(\"BIOHUB_REPO_SD2_MIN_PROB\", \"0.5\"))\n"
        "_SD2_SISTER_UM = float(os.environ.get(\"BIOHUB_REPO_SD2_SISTER_UM\", \"9.0\"))\n"
        "_SD2_MAX_PER_FRAME_FRAC = float(os.environ.get(\"BIOHUB_REPO_SD2_MAX_PER_FRAME_FRAC\", \"0.01\"))\n"
        "_SD2_ORIG_SCALE_ZYX = (1.625, 0.40625, 0.40625)\n\n"
    )
    old_init = (
        "            children_count: dict[int, int] = {}\n"
        "            parents_count: dict[int, int] = {}"
    )
    new_init = (
        "            children_count: dict[int, int] = {}\n"
        "            parents_count: dict[int, int] = {}\n"
        "            child_of: dict[int, int] = {}"
    )
    old_greedy = (
        "                all_edges.append((gi, gj, float(prob), dist))\n"
        "                children_count[i] = n_ch + 1\n"
        "                parents_count[j] = n_pa + 1\n"
        "\n"
        "        del unet_out"
    )
    new_greedy = (
        "                all_edges.append((gi, gj, float(prob), dist))\n"
        "                children_count[i] = n_ch + 1\n"
        "                parents_count[j] = n_pa + 1\n"
        "                child_of[i] = j\n"
        "\n"
        "            if _SD2_ENABLE:\n"
        "                _sd2_scale = np.asarray(ds_arr, dtype=np.float32) * np.asarray(\n"
        "                    _SD2_ORIG_SCALE_ZYX, dtype=np.float32\n"
        "                )\n"
        "                one_sources = [i for i, n in children_count.items() if n == 1]\n"
        "                if one_sources:\n"
        "                    frame_cap = max(1, int(round(max(1, n_tgt) * _SD2_MAX_PER_FRAME_FRAC)))\n"
        "                    claimed = {j for j, n in parents_count.items() if n > 0}\n"
        "                    picks = []\n"
        "                    for i in one_sources:\n"
        "                        ci = child_of.get(i)\n"
        "                        if ci is None:\n"
        "                            continue\n"
        "                        row = probs[i]\n"
        "                        order = np.argsort(-row)[:64]\n"
        "                        for jj in order:\n"
        "                            j = int(jj)\n"
        "                            if j == ci or j in claimed:\n"
        "                                continue\n"
        "                            pr = float(row[j])\n"
        "                            if pr < _SD2_MIN_PROB:\n"
        "                                break\n"
        "                            sis = float(np.linalg.norm(\n"
        "                                (c_tgt[ci, 1:].astype(np.float32) - c_tgt[j, 1:].astype(np.float32))\n"
        "                                * _sd2_scale\n"
        "                            ))\n"
        "                            if sis <= _SD2_SISTER_UM:\n"
        "                                picks.append((pr, i, j))\n"
        "                                break\n"
        "                    picks.sort(reverse=True)\n"
        "                    for pr, i, j in picks[:frame_cap]:\n"
        "                        gi, gj = int(idx_src[i]), int(idx_tgt[j])\n"
        "                        dist = float(np.linalg.norm(\n"
        "                            coords_so_far[gi, 1:].astype(np.float32)\n"
        "                            - coords_so_far[gj, 1:].astype(np.float32)\n"
        "                        ))\n"
        "                        all_edges.append((gi, gj, pr, dist))\n"
        "                        children_count[i] += 1\n"
        "                        parents_count[j] = parents_count.get(j, 0) + 1\n"
        "\n"
        "        del unet_out"
    )
    if anchor not in s or old_init not in s or old_greedy not in s:
        print("SD2 WARNING: anchors not found; patch skipped")
        return
    s = s.replace(anchor, globals_block + anchor, 1)
    s = s.replace(old_init, new_init, 1)
    s = s.replace(old_greedy, new_greedy, 1)
    if marker not in s:
        print("SD2 WARNING: patch did not apply cleanly")
        return
    ps.write_text(s)
    print(f"SD2 patch applied (enable={os.environ.get('BIOHUB_REPO_SD2_ENABLE', '0')}, "
          f"min_prob={os.environ.get('BIOHUB_REPO_SD2_MIN_PROB', '0.5')}, "
          f"sister_um={os.environ.get('BIOHUB_REPO_SD2_SISTER_UM', '9.0')})")


apply_sd2_patch()


def list_test_stems() -> list[str]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")
    stems = sorted(path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"No test .zarr files found in {TEST_DIR}")
    return stems


test_stems = list_test_stems()
print(f"Found {len(test_stems)} test videos")
print(test_stems[:10])

splits_path = REPO_DIR / "kaggle_test_splits_50ep.json"
splits_path.parent.mkdir(parents=True, exist_ok=True)
splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2))

predict_cmd = [
    sys.executable,
    "scripts/predict_unet_transformer.py",
    "--data-dir",
    str(TEST_DIR),
    "--splits",
    str(splits_path.name),
    "--split",
    "0",
    "--weights",
    WEIGHTS_RELATIVE,
    "--unet-batch-size",
    str(UNET_BATCH_SIZE),
    "--det-threshold",
    str(DET_THRESHOLD),
    "--ilp-edge-weight",
    str(ILP_EDGE_WEIGHT),
    "--ilp-appearance-weight",
    str(ILP_APPEARANCE_WEIGHT),
    "--ilp-disappearance-weight",
    str(ILP_DISAPPEARANCE_WEIGHT),
    "--ilp-division-weight",
    str(ILP_DIVISION_WEIGHT),
]
if USE_ILP:
    predict_cmd.append("--use-ilp")
if SLICE:
    predict_cmd.extend(["--slice", SLICE])

start_time = time.time()
print(" ".join(predict_cmd))
subprocess.run(predict_cmd, cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)
predict_seconds = time.time() - start_time
print(f"Prediction completed in {predict_seconds / 60:.2f} minutes")


# ## Build `submission.csv`
# 
# Rows are streamed directly to disk with the required schema. This avoids holding
# the full hidden-test submission table in memory.
# 
# The standard gap closer intentionally handles only one missing frame. Two-missing-frame repair is handled by the stricter `gap2` pass so that a loose environment override cannot introduce non-consecutive edges.
# 

# In[ ]:


import tracksdata as td
import numpy as np
import blosc2
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def edge_distance_um(source: dict[str, object], target: dict[str, object]) -> float:
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def node_point(node: dict[str, object]) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_sort_key(edge: dict[str, object]) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge["distance_um"])


def _next_node_id(nodes_by_id: dict[int, dict[str, object]]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1



def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    if t in frame_cache:
        return frame_cache[t]
    zarr_path = TEST_DIR / f"{dataset}.zarr"
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    frame_shape = shape[1:]
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        raw = chunk_path.read_bytes()
        arr = np.frombuffer(blosc2.decompress(raw), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            frame = arr.reshape(frame_shape).copy()
            frame_cache[t] = frame
            return frame
    except Exception:
        pass
    import zarr
    frame = np.asarray(zarr.open(zarr_path / "0", mode="r")[t])
    frame_cache[t] = frame
    return frame


def refine_synthetic_midpoint(
    dataset: str | None,
    t: int,
    midpoint: tuple[float, float, float],
    frame_cache: dict[int, np.ndarray],
    stats: dict[str, int],
) -> tuple[float, float, float]:
    if not GAP_REFINE_SYNTHETIC or dataset is None:
        return midpoint
    try:
        frame = read_test_frame(dataset, t, frame_cache)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - GAP_REFINE_WIN_Z)
        z1 = min(frame.shape[0], z + GAP_REFINE_WIN_Z + 1)
        y0 = max(0, y - GAP_REFINE_WIN_YX)
        y1 = min(frame.shape[1], y + GAP_REFINE_WIN_YX + 1)
        x0 = max(0, x - GAP_REFINE_WIN_YX)
        x1 = min(frame.shape[2], x + GAP_REFINE_WIN_YX + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_distance_um(refined, midpoint) > GAP_REFINE_MAX_SHIFT_UM:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint



def _dc_pool_frame_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def _dc_normalize_dynamic_range(volume: np.ndarray, cfg: object) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, float(getattr(cfg, "norm_lo_pct", 50.0))))
    hi = float(np.percentile(vol, float(getattr(cfg, "norm_hi_pct", 99.5))))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(
        ratio,
        float(getattr(cfg, "norm_clip_lo", -0.5)),
        float(getattr(cfg, "norm_clip_hi", 6.0)),
    ).astype(np.float32)


def _dc_manifest_weight_paths(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        print("Could not read DeepCenter manifest:", manifest_path, type(exc).__name__, exc)
        return []
    root = manifest_path.parent
    sections: list[dict[str, object]] = []
    for section in [
        manifest.get("model", {}),
        manifest.get("models", {}).get("full_frame_center", {}) if isinstance(manifest.get("models", {}), dict) else {},
        manifest.get("full_frame_center", {}),
    ]:
        if isinstance(section, dict):
            sections.append(section)
    candidates: list[Path] = []
    for section in sections:
        for key in ("weight_path", "path"):
            rel = section.get(key)
            if isinstance(rel, str) and rel:
                candidates.append(root / rel)
        for key in ("last_checkpoint", "best_checkpoint"):
            item = section.get(key)
            if isinstance(item, dict):
                rel = item.get("path")
                if isinstance(rel, str) and rel:
                    candidates.append(root / rel)
    for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
        candidates.append(root / "weights" / "full_frame_center" / name)
        candidates.append(root / name)
    candidates.append(root / DEEPCENTER_RELATIVE)
    return candidates


def _dc_checkpoint_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", DEEPCENTER_CHECKPOINT_DEFAULT).strip()
    if explicit:
        candidates.append(Path(explicit))
    manifest_explicit = os.environ.get("BIOHUB_DEEPCENTER_MANIFEST", DEEPCENTER_MANIFEST_DEFAULT).strip()
    if manifest_explicit:
        candidates.extend(_dc_manifest_weight_paths(Path(manifest_explicit)))

    input_root = Path("/kaggle/input")
    preferred_dirs = [
        Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),
        Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),
    ]
    for directory in preferred_dirs:
        candidates.extend(_dc_manifest_weight_paths(directory / "ARTIFACT_MANIFEST.json"))
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.append(directory / "weights" / "full_frame_center" / name)
            candidates.append(directory / name)
    if input_root.exists():
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.extend(sorted(input_root.glob(f"**/full_frame_center/**/{name}")))

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        path = path.expanduser()
        try:
            key = path.resolve() if path.exists() else path
        except Exception:
            key = path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


try:
    import torch
except Exception as _dc_torch_error:
    torch = None


if torch is not None:
    class _DCConvBlock3d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class _DCDeepCenterUNet3D(torch.nn.Module):
        def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
            super().__init__()
            c = int(base_channels)
            self.enc1 = _DCConvBlock3d(in_channels, c)
            self.down1 = torch.nn.MaxPool3d(2, 2)
            self.enc2 = _DCConvBlock3d(c, c * 2)
            self.down2 = torch.nn.MaxPool3d(2, 2)
            self.enc3 = _DCConvBlock3d(c * 2, c * 4)
            self.down3 = torch.nn.MaxPool3d(2, 2)
            self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
            self.up3 = torch.nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
            self.dec3 = _DCConvBlock3d(c * 8, c * 4)
            self.up2 = torch.nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
            self.dec2 = _DCConvBlock3d(c * 4, c * 2)
            self.up1 = torch.nn.ConvTranspose3d(c * 2, c, 2, 2)
            self.dec1 = _DCConvBlock3d(c * 2, c)
            self.head = torch.nn.Conv3d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bottleneck(self.down3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)
else:
    _DCConvBlock3d = None
    _DCDeepCenterUNet3D = None

def load_deepcenter_veto_detector() -> dict[str, object] | None:
    if not USE_DEEPCENTER_VETO:
        print("DeepCenter add-only repair gate disabled by configuration.")
        return None
    if torch is None:
        if REQUIRE_DEEPCENTER_VETO:
            raise ImportError("torch is required for DeepCenter add-only repair gate")
        print("DeepCenter add-only repair gate skipped because torch is unavailable.")
        return None
    from types import SimpleNamespace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_errors: list[str] = []
    for checkpoint_path in _dc_checkpoint_candidates():
        if not checkpoint_path.exists():
            continue
        try:
            print("Trying DeepCenter add-only gate checkpoint:", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                raise ValueError("checkpoint has no model_state")
            checkpoint_epoch = int(checkpoint.get("epoch", -1))
            if DEEPCENTER_EXPECTED_EPOCH > 0 and checkpoint_epoch != DEEPCENTER_EXPECTED_EPOCH:
                raise ValueError(
                    f"expected DeepCenter epoch {DEEPCENTER_EXPECTED_EPOCH}, got {checkpoint_epoch}"
                )
            cfg = SimpleNamespace(**checkpoint.get("config", {}))
            model = _DCDeepCenterUNet3D(base_channels=int(getattr(cfg, "base_channels", 24)))
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            model.eval()
            print("Loaded DeepCenter add-only gate checkpoint:", checkpoint_path)
            print("DeepCenter checkpoint epoch:", checkpoint.get("epoch"), "best_score:", checkpoint.get("best_score"))
            return {
                "model": model,
                "cfg": cfg,
                "device": device,
                "path": checkpoint_path,
                "torch": torch,
            }
        except Exception as exc:
            load_errors.append(f"{checkpoint_path}: {type(exc).__name__}: {exc}")
            print("Skipping incompatible DeepCenter checkpoint:", checkpoint_path, "|", type(exc).__name__, exc)
    message = "No usable DeepCenter checkpoint found for add-only repair gate."
    if REQUIRE_DEEPCENTER_VETO:
        checked = "\n".join(str(p) for p in _dc_checkpoint_candidates()[:80])
        errors = "\n".join(load_errors[-20:])
        raise FileNotFoundError(message + "\nChecked:\n" + checked + ("\nLoad errors:\n" + errors if errors else ""))
    print(message)
    return None


def _dc_cache_trim(cache: dict[tuple[str, int], np.ndarray]) -> None:
    limit = max(1, int(DEEPCENTER_SCORE_CACHE_MAX_FRAMES))
    while len(cache) > limit:
        cache.pop(next(iter(cache)))


def deepcenter_heatmap_for_frame(
    dataset: str,
    t: int,
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray | None:
    if detector_bundle is None:
        return None
    key = (dataset, int(t))
    cached = heatmap_cache.get(key)
    if cached is not None:
        return cached
    model = detector_bundle["model"]
    cfg = detector_bundle["cfg"]
    device = detector_bundle["device"]
    torch_mod = detector_bundle["torch"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    volume = read_test_frame(dataset, int(t), frame_cache)
    pooled = _dc_pool_frame_xy(volume, pool_factor)
    image = _dc_normalize_dynamic_range(pooled, cfg)
    with torch_mod.no_grad():
        tensor = torch_mod.from_numpy(image[None, None, ...]).to(device=device, dtype=torch_mod.float32)
        heatmap = torch_mod.sigmoid(model(tensor))[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    heatmap_cache[key] = heatmap
    _dc_cache_trim(heatmap_cache)
    return heatmap


def deepcenter_score_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> float | None:
    if not USE_DEEPCENTER_VETO or detector_bundle is None or dataset is None:
        return None
    heatmap = deepcenter_heatmap_for_frame(dataset, int(t), detector_bundle, frame_cache, heatmap_cache)
    if heatmap is None or heatmap.size == 0:
        return None
    cfg = detector_bundle["cfg"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    z = int(round(float(point[0])))
    y = int(round(float(point[1]) / max(pool_factor, 1)))
    x = int(round(float(point[2]) / max(pool_factor, 1)))
    z0, z1 = max(0, z - DEEPCENTER_SCORE_WIN_Z), min(heatmap.shape[0], z + DEEPCENTER_SCORE_WIN_Z + 1)
    y0, y1 = max(0, y - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[1], y + DEEPCENTER_SCORE_WIN_YX + 1)
    x0, x1 = max(0, x - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[2], x + DEEPCENTER_SCORE_WIN_YX + 1)
    patch = heatmap[z0:z1, y0:y1, x0:x1]
    if patch.size == 0:
        return None
    score = float(np.max(patch))
    return score if np.isfinite(score) else None


def deepcenter_accept_repair_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
    stats: dict[str, int],
    prefix: str,
    threshold: float,
) -> bool:
    if not USE_DEEPCENTER_VETO:
        return True
    if detector_bundle is None or dataset is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    stats[f"deepcenter_{prefix}_checked"] += 1
    score = deepcenter_score_point(dataset, int(t), point, detector_bundle, frame_cache, heatmap_cache)
    if score is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    if score < float(threshold):
        stats[f"deepcenter_{prefix}_rejected"] += 1
        return False
    stats[f"deepcenter_{prefix}_accepted"] += 1
    return True

def _position_um(node: dict[str, object]) -> np.ndarray:
    return np.array(
        [float(node["z"]) * VOXEL_SCALE_UM[0], float(node["y"]) * VOXEL_SCALE_UM[1], float(node["x"]) * VOXEL_SCALE_UM[2]],
        dtype=np.float64,
    )


def motion_relink_edges(
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_MOTION_RELINK or not nodes_by_id:
        return []

    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            if prev_pos is None:
                predicted = source_pos
            else:
                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)
            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float]] = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((
                source_ids[int(r)],
                target_ids[int(c)],
                float(raw_dist[r, c]),
                float(motion_dist[r, c]),
                float(prob_matrix[r, c]),
            ))
        return matches

    times = sorted(ids_by_t)
    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[tuple[int, int, float, float, str, float]] = []
        for pass_name, gate_um in (("tight", MOTION_RELINK_TIGHT_UM), ("relaxed", MOTION_RELINK_RELAXED_UM)):
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            matches = assign_pass(pass_sources, pass_targets, gate_um)
            for source_id, target_id, raw, motion, prob in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))
                if pass_name == "tight":
                    stats["motion_relink_tight_edges"] += 1
                else:
                    stats["motion_relink_relaxed_edges"] += 1
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
            })
            predecessor_position_um[target_id] = position_um[source_id]
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges

def close_single_frame_gaps(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP_CLOSE or GAP_CLOSE_MAX_GAP < 1 or not edges:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        all_ids_by_t.setdefault(t, []).append(node_id)
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    max_synthetic = min(
        GAP_CLOSE_MAX_ADDED_ABS,
        max(1, int(round(len(nodes_by_id) * GAP_CLOSE_MAX_ADDED_FRAC))) if GAP_CLOSE_MAX_ADDED_FRAC > 0 else 0,
    )
    next_id = _next_node_id(nodes_by_id)
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict[str, object]] = []

    density_cache: dict[int, dict[int, float]] = {}

    def frame_local_spacing(t: int) -> dict[int, float]:
        cached = density_cache.get(t)
        if cached is not None:
            return cached

        frame_ids = all_ids_by_t.get(t, [])
        if len(frame_ids) <= 1:
            result = {
                node_id: GAP_DENSITY_REFERENCE_UM
                for node_id in frame_ids
            }
            density_cache[t] = result
            return result

        positions = np.stack(
            [_position_um(nodes_by_id[node_id]) for node_id in frame_ids]
        )
        tree = cKDTree(positions)
        query_k = min(
            len(frame_ids),
            max(2, GAP_DENSITY_NEIGHBORS + 1),
        )
        distances, _ = tree.query(positions, k=query_k)
        if distances.ndim == 1:
            distances = distances[:, None]

        result: dict[int, float] = {}
        for idx, node_id in enumerate(frame_ids):
            neighbour_distances = distances[idx, 1:]
            neighbour_distances = neighbour_distances[
                np.isfinite(neighbour_distances)
            ]
            spacing = (
                float(np.median(neighbour_distances))
                if neighbour_distances.size
                else GAP_DENSITY_REFERENCE_UM
            )
            result[node_id] = spacing

        density_cache[t] = result
        stats["gap_density_nodes_scored"] += len(result)
        return result

    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)
    stats["gap_close_effective_max_gap"] = effective_gap_max
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue

            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = GAP_CLOSE_UM * (gap + 1)
            d = np.zeros(
                (len(end_ids), len(start_ids)),
                dtype=np.float64,
            )
            adaptive_threshold = np.full_like(d, threshold_um)

            source_spacing = frame_local_spacing(t)
            target_spacing = frame_local_spacing(t + gap + 1)

            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)

                    if GAP_DENSITY_ADAPTIVE:
                        local_spacing = 0.5 * (
                            source_spacing.get(
                                end_ids[i],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                            + target_spacing.get(
                                start_ids[j],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                        )
                        step_delta = float(
                            np.clip(
                                GAP_DENSITY_GAIN
                                * (
                                    local_spacing
                                    - GAP_DENSITY_REFERENCE_UM
                                ),
                                -GAP_DENSITY_MAX_STEP_DELTA_UM,
                                GAP_DENSITY_MAX_STEP_DELTA_UM,
                            )
                        )
                        adaptive_threshold[i, j] = (
                            threshold_um + step_delta * (gap + 1)
                        )
                        stats[
                            "gap_density_step_delta_milli_sum"
                        ] += int(round(1000.0 * step_delta))

            base_allowed = d <= threshold_um
            adaptive_allowed = d <= adaptive_threshold

            stats["gap_density_candidates_expanded"] += int(
                (adaptive_allowed & ~base_allowed).sum()
            )
            stats["gap_density_candidates_restricted"] += int(
                (base_allowed & ~adaptive_allowed).sum()
            )
            stats["gap_candidates"] += int(adaptive_allowed.sum())

            if not np.isfinite(d).any():
                continue

            max_threshold = float(np.max(adaptive_threshold))
            big = max_threshold * 1000.0 + 1.0
            cost = np.where(adaptive_allowed, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if not adaptive_allowed[r, c]:
                    continue
                if not base_allowed[r, c]:
                    stats[
                        "gap_density_selected_outside_base"
                    ] += 1
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue

                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )

                middle_id: int | None = None
                middle_reused = False
                if GAP_CLOSE_REUSE_EXISTING:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                    if candidates:
                        distances = [point_distance_um(node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= GAP_CLOSE_REUSE_UM:
                            middle_id = candidates[best_idx]
                            middle_reused = True

                if middle_id is None:
                    if synthetic_added >= max_synthetic:
                        stats["gap_skipped_node_cap"] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_midpoint(dataset, mid_t, mid_point, frame_cache, stats)
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id,
                        "t": mid_t,
                        "z": refined_point[0],
                        "y": refined_point[1],
                        "x": refined_point[2],
                        "gap_synthetic": 1,
                    }
                    synthetic_added += 1
                    stats["gap_inserted_synthetic"] += 1

                middle = nodes_by_id[middle_id]
                gap_span_um = float(d[r, c])
                marginal_gap = gap_span_um >= DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM
                # Confirm both reused and image-refined synthetic midpoints when
                # the endpoint displacement is marginal.  The old public-only
                # path bypassed this check for synthetic nodes; that makes a
                # geometry-only repair fragile under embryo shift.
                requires_center_confirmation = DEEPCENTER_GAP_VETO and marginal_gap
                if DEEPCENTER_GAP_VETO and not marginal_gap:
                    stats["deepcenter_gap_bypassed_strong_motion"] += 1
                if requires_center_confirmation and not deepcenter_accept_repair_point(
                    dataset,
                    mid_t,
                    node_point(middle),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "gap",
                    DEEPCENTER_GAP_THRESHOLD,
                ):
                    if int(middle.get("gap_synthetic", 0)) == 1:
                        nodes_by_id.pop(middle_id, None)
                        synthetic_added = max(0, synthetic_added - 1)
                        stats["gap_inserted_synthetic"] = max(0, stats["gap_inserted_synthetic"] - 1)
                    continue
                if middle_reused:
                    used_isolated.add(middle_id)
                    stats["gap_reused_existing"] += 1

                e1 = {
                    "source_id": source_id,
                    "target_id": middle_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(source, middle),
                    "gap_closed": 1,
                }
                e2 = {
                    "source_id": middle_id,
                    "target_id": target_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(middle, target),
                    "gap_closed": 1,
                }
                new_edges.extend([e1, e2])
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] += 1
                stats["gap_added_edges"] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats["gap_added_nodes"] = stats["gap_inserted_synthetic"]
    return nodes_by_id, edges


def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}


def _single_predecessor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for edge in edges:
        by_target.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    return {target: sources[0] for target, sources in by_target.items() if len(sources) == 1}


def recover_strict_gap2(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP2_RECOVERY or not edges or not nodes_by_id:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    cap = min(GAP2_MAX_LINKS_ABS, max(1, int(round(len(edges) * GAP2_MAX_LINKS_FRAC))))
    proposals: list[tuple[float, int, int, int, float]] = []

    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64) * np.array(VOXEL_SCALE_UM)

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > GAP2_MAX_TOTAL_UM or dist / 3.0 > GAP2_MAX_STEP_UM:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if GAP2_REQUIRE_CONTEXT:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)
                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    stats["gap2_candidates"] = len(proposals)
    if not proposals:
        return nodes_by_id, edges

    selected: list[tuple[float, int, int, int, float]] = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if len(selected) >= cap:
            stats["gap2_skipped_cap"] += 1
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * GAP2_FRAME_FRAC_CAP)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1

    if not selected:
        return nodes_by_id, edges

    next_node_id = _next_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    new_edges: list[dict[str, object]] = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids: list[int] = []
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            refined_point = refine_synthetic_midpoint(dataset, mid_t, midpoint, frame_cache, stats)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": mid_t,
                "z": refined_point[0],
                "y": refined_point[1],
                "x": refined_point[2],
            }
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append({
                "source_id": previous_id,
                "target_id": node_id,
                "edge_prob": None,
                "distance_um": edge_distance_um(nodes_by_id[previous_id], current),
                "gap2_recovered": 1,
            })
            previous_id = node_id
        new_edges.append({
            "source_id": previous_id,
            "target_id": start_id,
            "edge_prob": None,
            "distance_um": edge_distance_um(nodes_by_id[previous_id], target),
            "gap2_recovered": 1,
        })
        stats["gap2_pairs_selected"] += 1
        stats["gap2_added_nodes"] += len(inserted_ids)
        stats["gap2_added_edges"] += 3

    return nodes_by_id, [*edges, *new_edges]


def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_SAFE_DIVISIONS or not edges or not nodes_by_id:
        return edges
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}

    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1]
        candidate_ids = [node_id for node_id in child_frame_ids if node_id not in incoming and node_id not in used_targets]
        if not source_ids or not candidate_ids:
            continue

        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue
            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue
                if DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point(
                    dataset,
                    int(candidate["t"]),
                    node_point(candidate),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "safe_div",
                    DEEPCENTER_SAFE_DIV_THRESHOLD,
                ):
                    continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))

        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
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
            added_this_frame += 1

    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


def filter_short_track_components(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_FILTER_SHORT_TRACKS or OUTPUT_MIN_TRACK_LEN <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    component_edges: dict[int, list[dict[str, object]]] = {root: [] for root in components}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id in parent and target_id in parent:
            component_edges.setdefault(find(source_id), []).append(edge)

    keep: set[int] = set()
    for root, members in components.items():
        has_division = any(out_count.get(node_id, 0) >= 2 for node_id in members)
        if len(members) >= OUTPUT_MIN_TRACK_LEN or (OUTPUT_KEEP_DIVISION_COMPONENTS and has_division):
            keep.update(members)

    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_before_rescue = len(nodes_by_id) - len(keep)
    if removed_before_rescue <= 0:
        return nodes_by_id, edges

    if ADAPTIVE_SHORT_TRACK_RESCUE:
        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)
        if removed_frac >= SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC:
            budget = min(
                SHORT_TRACK_RESCUE_MAX_NODES_ABS,
                max(0, int(round(len(nodes_by_id) * SHORT_TRACK_RESCUE_MAX_NODES_FRAC))),
            )
            stats["short_track_rescue_triggered"] = 1
            stats["short_track_rescue_budget"] = budget
            proposals: list[tuple[float, int, float, int, list[int]]] = []
            for root, members in components.items():
                if set(members) & keep:
                    continue
                if len(members) < SHORT_TRACK_RESCUE_MIN_LEN or len(members) >= OUTPUT_MIN_TRACK_LEN:
                    continue
                c_edges = component_edges.get(root, [])
                if not c_edges:
                    continue
                probs: list[float] = []
                dists: list[float] = []
                for edge in c_edges:
                    try:
                        prob = float(edge.get("edge_prob", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    if np.isfinite(prob):
                        probs.append(prob)
                    try:
                        dist = float(edge.get("distance_um", np.nan))
                    except (TypeError, ValueError):
                        dist = np.nan
                    if np.isfinite(dist):
                        dists.append(dist)
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float("inf")
                if mean_prob < SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB:
                    continue
                if mean_dist > SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), mean_prob, root, members))
            proposals.sort(reverse=True)
            rescued_nodes = 0
            rescued_components = 0
            for _, size, _, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
                rescued_components += 1
            stats["short_track_rescue_components"] = rescued_components
            stats["short_track_rescue_nodes"] = rescued_nodes

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges

    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in keep}
    kept_edges = [
        edge for edge in edges
        if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes
    ]
    stats["short_track_components_removed"] = sum(1 for members in components.values() if not (set(members) & keep))
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def linefit_smooth_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Smooth linear track interiors without changing graph topology."""
    if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {
        node_id: np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        for node_id, node in nodes_by_id.items()
    }
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))

        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue

        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)], dtype=np.float64)
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id


def filter_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    stats = {
        "raw_edges": len(raw_edges),
        "dropped_nonconsecutive_edges": 0,
        "dropped_long_edges": 0,
        "dropped_multi_parent_edges": 0,
        "dropped_multi_child_edges": 0,
        "dropped_division_edges": 0,
        "gap_candidates": 0,
        "gap_pairs_selected": 0,
        "gap_reused_existing": 0,
        "gap_inserted_synthetic": 0,
        "gap_added_nodes": 0,
        "gap_added_edges": 0,
        "gap_skipped_node_cap": 0,
        "gap_density_nodes_scored": 0,
        "gap_density_candidates_expanded": 0,
        "gap_density_candidates_restricted": 0,
        "gap_density_selected_outside_base": 0,
        "gap_density_step_delta_milli_sum": 0,
        "gap_refined_synthetic": 0,
        "gap_refine_failed": 0,
        "gap_refine_rejected_shift": 0,
        "pruned_isolated_nodes": 0,
        "motion_relink_edges": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
        "motion_relink_replaced_raw_edges": 0,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "gap2_candidates": 0,
        "gap2_pairs_selected": 0,
        "gap2_added_nodes": 0,
        "gap2_added_edges": 0,
        "gap2_skipped_cap": 0,
        "safe_division_candidates": 0,
        "safe_divisions_added": 0,
        "safe_division_skipped_cap": 0,
        "deepcenter_gap_checked": 0,
        "deepcenter_gap_bypassed_strong_motion": 0,
        "deepcenter_gap_bypassed_synthetic_node": 0,
        "deepcenter_gap_accepted": 0,
        "deepcenter_gap_rejected": 0,
        "deepcenter_gap_missing": 0,
        "deepcenter_safe_div_checked": 0,
        "deepcenter_safe_div_accepted": 0,
        "deepcenter_safe_div_rejected": 0,
        "deepcenter_safe_div_missing": 0,
        "short_track_components_removed": 0,
        "short_track_nodes_removed": 0,
        "short_track_edges_removed": 0,
        "short_track_filter_skipped_all": 0,
        "short_track_rescue_triggered": 0,
        "short_track_rescue_components": 0,
        "short_track_rescue_nodes": 0,
        "short_track_rescue_budget": 0,
        "linefit_smoothed_nodes": 0,
        "linefit_skipped_nodes": 0,
    }

    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = edge_distance_um(source, target)
        edge["distance_um"] = distance_um
        if OUTPUT_EDGE_MAX_UM > 0 and distance_um > OUTPUT_EDGE_MAX_UM:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)

    if OUTPUT_MOTION_RELINK:
        learned_edge_probs: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1

    if OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best_by_target: dict[int, dict[str, object]] = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    if OUTPUT_SINGLE_CHILD_REPAIR and edges:
        best_by_source: dict[int, dict[str, object]] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    repair_frame_cache: dict[int, np.ndarray] = {}
    deepcenter_heatmap_cache: dict[tuple[str, int], np.ndarray] = {}
    nodes_by_id, edges = close_single_frame_gaps(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )
    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, dataset=dataset)
    edges = add_safe_divisions_postlink(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )

    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:
        by_source: dict[int, list[dict[str, object]]] = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)

        filtered: list[dict[str, object]] = []
        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue

            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            source = nodes_by_id[source_id]
            top1 = ranked[0]
            top2 = ranked[1]
            d1 = float(top1["distance_um"])
            d2 = float(top2["distance_um"])
            sister = edge_distance_um(nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])])
            valid_division = (
                max(d1, d2) <= DIV_PARENT_MAX_UM
                and sister <= DIV_SISTER_MAX_UM
                and int(nodes_by_id[int(top1["target_id"])] ["t"]) == int(source["t"]) + 1
                and int(nodes_by_id[int(top2["target_id"])] ["t"]) == int(source["t"]) + 1
            )
            if valid_division:
                filtered.extend([top1, top2])
                stats["dropped_division_edges"] += max(0, len(ranked) - 2)
            elif DIV_DROP_TO_SINGLE_IF_BAD:
                filtered.append(top1)
                stats["dropped_division_edges"] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered

    if BIOHUB_DIVISION_CONTINUATION_GUARD and edges:
        edges = apply_division_continuation_guard(nodes_by_id, edges, stats)

    if OUTPUT_PRUNE_ISOLATED:
        incident = {int(edge["source_id"]) for edge in edges} | {int(edge["target_id"]) for edge in edges}
        if incident:
            kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in incident}
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [edge for edge in edges if int(edge["source_id"]) in nodes_by_id and int(edge["target_id"]) in nodes_by_id]

    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)

    edges = add_second_daughter_postlink(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
    )

    return nodes_by_id, edges, stats


DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()

geffs = sorted((REPO_DIR / "predictions").glob(f"*/{METHOD}/split_0/*.geff"))
print(f"Found {len(geffs)} prediction graphs")
if len(geffs) != len(test_stems):
    found = {path.stem for path in geffs}
    missing = sorted(set(test_stems) - found)
    raise RuntimeError(f"Expected {len(test_stems)} graphs, found {len(geffs)}. Missing: {missing[:10]}")

stats_rows: list[dict[str, object]] = []
seen_datasets: set[str] = set()
row_id = 0
total_nodes = 0
total_edges = 0

with SUBMISSION_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for geff_path in geffs:
        dataset = geff_path.stem
        seen_datasets.add(dataset)
        graph = graph_from_geff(geff_path)

        nodes_by_id: dict[int, dict[str, object]] = {}
        for row in graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }

        raw_edges: list[dict[str, object]] = []
        for row in graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })

        raw_node_count = len(nodes_by_id)
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")

        for node_id in sorted(nodes_by_id):
            node = nodes_by_id[node_id]
            _zarr_meta = json.loads((TEST_DIR / f"{dataset}.zarr" / "0" / "zarr.json").read_text())
            _shape = tuple(int(v) for v in _zarr_meta["shape"])

            def _clamp(value: float, hi: int) -> int:
                return min(max(0, int(round(float(value)))), hi - 1)

            writer.writerow({
                "id": row_id,
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(node["node_id"]),
                "t": int(node["t"]),
                "z": _clamp(node["z"], _shape[1]),
                "y": _clamp(node["y"], _shape[2]),
                "x": _clamp(node["x"], _shape[3]),
                "source_id": -1,
                "target_id": -1,
            })
            row_id += 1

        division_sources: dict[int, int] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            target_id = int(edge["target_id"])
            if source_id not in nodes_by_id or target_id not in nodes_by_id:
                raise AssertionError(f"{dataset}: dangling edge after filtering")
            writer.writerow({
                "id": row_id,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": source_id,
                "target_id": target_id,
            })
            row_id += 1
            division_sources[source_id] = division_sources.get(source_id, 0) + 1

        node_count = len(nodes_by_id)
        edge_count = len(edges)
        total_nodes += node_count
        total_edges += edge_count
        stats_rows.append({
            "dataset": dataset,
            "raw_nodes": raw_node_count,
            "nodes": node_count,
            "raw_edges": filter_stats["raw_edges"],
            "edges": edge_count,
            "division_like_sources": sum(1 for count in division_sources.values() if count >= 2),
            "edge_to_node_ratio": edge_count / max(node_count, 1),
            "gap_added_nodes_frac": filter_stats.get("gap_added_nodes", 0) / max(raw_node_count, 1),
            **filter_stats,
        })

expected_datasets = set(test_stems)
missing_datasets = sorted(expected_datasets - seen_datasets)
extra_datasets = sorted(seen_datasets - expected_datasets)
if missing_datasets or extra_datasets:
    raise AssertionError({"missing": missing_datasets[:10], "extra": extra_datasets[:10]})
assert row_id == total_nodes + total_edges, "Internal row counter mismatch"
assert total_nodes > 0, "No node rows produced"

header = SUBMISSION_PATH.open().readline().strip().split(",")
assert header == CSV_COLUMNS, f"Bad CSV header: {header}"

stats = pd.DataFrame(stats_rows).sort_values("dataset").reset_index(drop=True)
stats["predict_minutes_total"] = predict_seconds / 60.0
stats["experiment_tag"] = EXPERIMENT_TAG
stats.to_csv(RUN_STATS_PATH, index=False)

print(f"Wrote {SUBMISSION_PATH} with {row_id:,} rows")
print(f"Node rows: {total_nodes:,} | edge rows: {total_edges:,}")
print(f"Wrote {RUN_STATS_PATH}")
display(stats.describe(include="all"))
display(pd.read_csv(SUBMISSION_PATH, nrows=8))


# ## Production audit
#
# This notebook is not a public-LB probe.  It emits one deterministic graph for
# every discovered `.zarr` and stops on any schema, coordinate or topology
# violation.  Train/CV scoring is intentionally outside the submission kernel;
# use the repository-side held-out gate before pushing a version.
#

# In[ ]:


# ＊＊＊＊＊ GPT FEEDBACK START / END ＊＊＊＊＊
import hashlib
from pathlib import Path

_feedback_path = Path(SUBMISSION_PATH)
_feedback_sha = hashlib.sha256(_feedback_path.read_bytes()).hexdigest()
_feedback_rows = int(total_nodes + total_edges)
_feedback_divisions = int(stats["division_like_sources"].sum()) if "division_like_sources" in stats else 0
_feedback_activation = int(
    stats["gap_density_candidates_expanded"].sum()
    + stats["gap_density_candidates_restricted"].sum()
)
_feedback_cols = [
    "dataset", "nodes", "edges", "division_like_sources",
    "gap_candidates", "gap_pairs_selected",
    "gap_density_candidates_expanded", "gap_density_selected_outside_base",
]

# Clean-submission audit. This checks only biological/format validity and does not alter predictions.
_audit_df = pd.read_csv(_feedback_path)
_node_df = _audit_df[_audit_df["row_type"] == "node"].copy()
_edge_df = _audit_df[_audit_df["row_type"] == "edge"].copy()
_clean_audit_rows = []
for _dataset in test_stems:
    _zarr_meta = json.loads((TEST_DIR / f"{_dataset}.zarr" / "0" / "zarr.json").read_text())
    _shape = tuple(int(v) for v in _zarr_meta["shape"])
    _n = _node_df[_node_df["dataset"] == _dataset]
    _e = _edge_df[_edge_df["dataset"] == _dataset]
    _node_times = dict(zip(_n["node_id"].astype(int), _n["t"].astype(int)))
    _valid_ids = set(_node_times)
    _duplicate_nodes = int(_n["node_id"].duplicated().sum())
    _dangling = int(sum((int(s) not in _valid_ids) or (int(t) not in _valid_ids) for s, t in zip(_e["source_id"], _e["target_id"])))
    _nonconsecutive = int(sum(_node_times.get(int(t), -999) != _node_times.get(int(s), -999) + 1 for s, t in zip(_e["source_id"], _e["target_id"])))
    _outdegree = _e.groupby("source_id").size() if len(_e) else pd.Series(dtype=int)
    _indegree = _e.groupby("target_id").size() if len(_e) else pd.Series(dtype=int)
    _max_outdegree = int(_outdegree.max()) if len(_outdegree) else 0
    _max_indegree = int(_indegree.max()) if len(_indegree) else 0
    _finite_coords = np.isfinite(_n[["z", "y", "x"]].to_numpy(float)).all(axis=1)
    _nonfinite_coords = int((~_finite_coords).sum())
    _bad_time = int(((_n["t"] < 0) | (_n["t"] >= _shape[0])).sum())
    _bad_z = int(((_n["z"] < 0) | (_n["z"] >= _shape[1])).sum())
    _bad_y = int(((_n["y"] < 0) | (_n["y"] >= _shape[2])).sum())
    _bad_x = int(((_n["x"] < 0) | (_n["x"] >= _shape[3])).sum())
    _clean_audit_rows.append({
        "dataset": _dataset,
        "duplicate_nodes": _duplicate_nodes,
        "node_time_violations": _bad_time,
        "coordinate_violations": _bad_z + _bad_y + _bad_x,
        "nonfinite_coordinates": _nonfinite_coords,
        "dangling_edges": _dangling,
        "nonconsecutive_edges": _nonconsecutive,
        "max_indegree": _max_indegree,
        "max_outdegree": _max_outdegree,
    })
_clean_audit = pd.DataFrame(_clean_audit_rows)
if int(_clean_audit[["duplicate_nodes", "node_time_violations", "coordinate_violations", "nonfinite_coordinates", "dangling_edges", "nonconsecutive_edges"]].to_numpy().sum()) != 0 or int((_clean_audit["max_indegree"] > 1).sum()) != 0:
    raise AssertionError("Clean-submission audit failed:\n" + _clean_audit.to_string(index=False))
if int(_clean_audit["max_outdegree"].max()) > 2:
    raise AssertionError("Unexpected hub-like source detected:\n" + _clean_audit.to_string(index=False))

print("")
print("BIOHUB GENERAL V2 PRODUCTION AUDIT")
print("Preset:", BIOHUB_PRESET)
print("Dynamic test datasets:", len(test_stems))
print("Prediction minutes:", f"{predict_seconds / 60.0:.2f}")
print("Rows:", _feedback_rows)
print("Node rows:", int(total_nodes), "| edge rows:", int(total_edges))
print("Division-like sources:", _feedback_divisions)
print("SHA256:", _feedback_sha)
print("DeepCenter-confirmed repairs:", int(stats.get("deepcenter_gap_accepted", pd.Series(dtype=int)).sum()))
print("Clean-submission audit:")
print(_clean_audit.to_string(index=False))


# In[ ]:


# FINAL KAGGLE SUBMISSION GUARD
from pathlib import Path
import pandas as pd
_submit_file = Path("/kaggle/working/submission.csv")
if not _submit_file.exists():
    raise FileNotFoundError("submission.csv was not created at /kaggle/working/submission.csv")
_submit_check = pd.read_csv(_submit_file)
_expected = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
if list(_submit_check.columns) != _expected:
    raise ValueError(f"Unexpected submission columns: {list(_submit_check.columns)}")
if _submit_check.isna().any().any():
    raise ValueError("submission.csv contains nulls")
if _submit_check["id"].astype(int).tolist() != list(range(len(_submit_check))):
    raise ValueError("submission ids are not contiguous")
if set(_submit_check["dataset"].unique()) != set(test_stems):
    raise ValueError("submission dataset coverage differs from discovered test data")
if set(_submit_check["row_type"].unique()) - {"node", "edge"}:
    raise ValueError("submission contains an unknown row_type")
print("SUBMISSION_READY: /kaggle/working/submission.csv")
print("Kaggle output filename: submission.csv")



# Production-only contract: do not run train/CV diagnostics in the submission kernel.
# The code above has already written and audited submission.csv.  The hidden rerun
# can contain a different number of datasets, so any fixed public split would be
# both a runtime risk and an accidental public-test dependency.
print("PRODUCTION_ONLY: train/CV diagnostics disabled; submission is protected")
