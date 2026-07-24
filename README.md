# biohub-cell-tracking

Solution repo for the Kaggle competition
[**Biohub — Cell Tracking During Development**](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development):
detect and track cells in 3D+time through zebrafish embryo development (light-sheet
microscopy, Zebrahub data).

📋 **Strategy lives in [PLAN.md](PLAN.md).** Read it first.

**$60,000 · deadline 2026-09-29 · kernels-only · 5 submissions/day**

Submission is a **point graph** CSV (node centroids + parent→child edges forming a directed
forest), not segmentation masks. Volumes are zarr v3, `(T, Z, Y, X)`, voxel size
`[1.625, 0.40625, 0.40625]` µm. Ground truth ships as `.geff` graphs.

## Setup

```bash
uv sync                 # create .venv and install deps
source .venv/bin/activate
```

Auth uses a **bearer token**, not the legacy `kaggle.json` — and it needs Kaggle CLI **≥2.x**
(1.5.13 does not support it). Put it in a gitignored `.env` at the repo root:

```bash
echo 'export KAGGLE_API_TOKEN=KGAT_...' > .env && chmod 600 .env
```

## Data

```bash
make data               # download + unzip competition files into data/raw/
```

~8 GB+. `data/` and `.env` are gitignored — neither is ever committed.

## Layout

```
src/ctrack/
  data/       loading, preprocessing, registration, CV splits
  detect/     3D nucleus detection / segmentation
  link/       frame-to-frame linking, global optimization, divisions, gap closing
  eval/       local CTC / CHOTA metric harness  ← build this first
  submit/     submission writers
configs/      experiment configs
scripts/      one-off CLI entry points
notebooks/    EDA (stripped of outputs before commit)
docs/         research notes, experiment log
tests/
```

## Experiment log

Every run gets a row in [docs/experiments.md](docs/experiments.md): config, CV, LB, runtime,
notes. If it isn't logged it didn't happen.
