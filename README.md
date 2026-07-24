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
  eval/biohub.py       the competition metric, reimplemented in numpy
  data/geff.py         ground-truth graph reader
  data/volume.py       (T,Z,Y,X) zarr reader, normalization, isotropic resampling
  data/splits.py       CV folds + golden holdout (see leakage notes in the module)
  detect/blob.py       classical nucleus detection (baseline)
  link/hungarian.py    frame-to-frame assignment + gap closing
  submit/format.py     submission schema + directed-forest validator

scripts/
  fetch_data.sh          full download (87.6 GB)
  fetch_subset.py        per-dataset download
  analyze_gt.py          GT characterisation across all volumes
  check_crop_overlap.py  do same-embryo crops overlap? (decides CV validity)
  run_baseline.py        detect → link → score / submit, with --sweep

docs/
  metric.md      the scoring formula, decoded  ← read this first
  data.md        verified dataset facts
  experiments.md experiment log
```

## Key facts

- Volumes are `(100, 64, 256, 256)` uint16 — 100 frames of a 104 µm cube. Subsampled `[1,4,4]`
  the whole volume is **64³ isotropic**, so inference is one forward pass per frame, no tiling.
- Ground truth is **~0.2% dense**: a couple of hand-traced trajectories per volume against
  ~25k real cells. The metric only sees edges touching annotated cells.
- `score = max(0, edge_jaccard · (1 − 0.1·(N_pred−N_total)/N_total)) + 0.1 · division_jaccard`,
  and that multiplier is **unclamped above** — under-detecting inflates the score. See
  [docs/metric.md](docs/metric.md).

## Experiment log

Every run gets a row in [docs/experiments.md](docs/experiments.md): config, CV, LB, runtime,
notes. If it isn't logged it didn't happen.
