"""Golden-8 tuning harness for the frontier unet_transformer pipeline (core: detect+edge+ILP).

Runs predict_unet_transformer.py --evaluate on the golden-8 TRAIN volumes across a small knob
grid, printing the reference metric (score/edge_jaccard/adj/division_jaccard) per setting.

CAVEAT: the pilkwang weights were trained on all train volumes incl. golden-8, so absolute
scores are inflated (train-on-test). Use this only to RANK inference-knob settings before
spending LB submissions; the hidden LB is the clean test.

Attach: competition data + pilkwang/biohub-tracking-support-pack-50ep-v1. GPU T4, no internet.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

GOLDEN = ["44b6_2f31fc2f", "44b6_a2bb48bb", "44b6_c96cfa10", "44b6_eb2880fc",
          "6bba_7b5d3b2c", "6bba_b329af44", "6bba_d82a4fc6", "6bba_debd7bfa"]

import shutil

pred = next(Path("/kaggle/input").glob("**/scripts/predict_unet_transformer.py"))
SRC_REPO = pred.parents[1]                  # .../repo  (read-only under /kaggle/input)
# predict writes predictions/ under the repo dir, which is read-only in /kaggle/input -> copy it
REPO = Path("/kaggle/working/repo")
if not REPO.exists():
    shutil.copytree(SRC_REPO, REPO)
SCRIPTS, SRC = REPO / "scripts", REPO / "src"
WEIGHTS = next(Path("/kaggle/input").glob("**/unet_transformer/split_0/edge_predictor_best.pth"))
WHEELS = next(Path("/kaggle/input").glob("**/wheels"), None)
TRAIN = next(p for p in [
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
    Path("/kaggle/input/biohub-cell-tracking-during-development/train")] if p.exists())
print(f"repo={REPO}\nweights={WEIGHTS}\nwheels={WHEELS}\ntrain={TRAIN}", flush=True)

# offline dependency install
if WHEELS is not None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    f"--find-links={WHEELS}", "tracksdata", "zarr", "pyscipopt", "geff", "ilpy",
                    "polars", "blosc2", "dask", "imagecodecs", "pyarrow", "rustworkx",
                    "sqlalchemy"], check=False)

# golden-8 as fold-0 test list
Path("/kaggle/working/golden.json").write_text(json.dumps([{"test": GOLDEN}]))

env = os.environ.copy()
env["PYTHONPATH"] = f"{SRC}:{SCRIPTS}"
env["BIOHUB_DATA_DIR"] = str(TRAIN)

# knob grid: detection threshold x ILP division weight (appearance/disappearance from the preset)
results = []
for det in [0.95, 0.96875, 0.98]:      # preset is 0.96875; probe either side
    for divw in [0.5, 1.0]:            # preset is 1.0; lower allows more divisions
        cmd = [sys.executable, str(SCRIPTS / "predict_unet_transformer.py"),
               "--data-dir", str(TRAIN), "--splits", "/kaggle/working/golden.json", "--split", "0",
               "--weights", str(WEIGHTS), "--det-threshold", str(det), "--use-ilp",
               "--ilp-appearance-weight", "0.0", "--ilp-disappearance-weight", "1.5",
               "--ilp-division-weight", str(divw), "--evaluate"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(SCRIPTS))
        line = next((l for l in r.stdout.splitlines() if "Evaluation" in l), None)
        tag = f"det={det} divw={divw}"
        if line:
            print(f"[{tag}] {line.strip()}", flush=True)
            results.append((tag, line.strip()))
        else:
            print(f"[{tag}] FAILED: {r.stderr[-400:]}", flush=True)

print("\n===== SUMMARY =====", flush=True)
for tag, line in results:
    print(f"{tag} :: {line}", flush=True)
