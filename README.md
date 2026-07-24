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

The repo separates **the library** (runs anywhere), **where code runs** (local vs cloud), and
**how we submit** (Kaggle kernels):

```
src/ctrack/            THE LIBRARY — importable, single source of truth
  data/                geff + zarr readers, CV splits
  detect/              detection: blob baseline, 3D U-Net, targets, cache, train loop
  link/                hungarian, ILP, learned edge affinities
  eval/biohub.py       the competition metric, reimplemented in numpy
  submit/format.py     submission schema + directed-forest validator

scripts/
  local/               RUNS ON THIS MACHINE (Apple MPS / CPU): train, eval, baseline, data prep
  cloud/               RUNS ON RENTED GPUs (Colab, Vast) — placeholder, empty for now

notebooks/             notebook-based training (interactive Colab)
kernels/               KAGGLE submission kernels: baseline/, infer/, train/

docs/
  metric.md            the scoring formula, decoded  ← read this first
  data.md              verified dataset facts
  detector.md          learned-detector design notes
  experiments.md       experiment log
```

**Where things run:**
- **Local (`scripts/local/`)** — the default. Detector training on MPS reaches >98% peak recall
  in ~4 epochs; everything is reliable and directly visible.
- **Cloud (`scripts/cloud/`)** — deferred. Only worth the setup for a bigger model or many-fold
  sweeps. The training *logic* stays in `ctrack.detect.train`; cloud scripts are thin drivers.
- **Kaggle (`kernels/`)** — the submission path (kernels-only competition, no internet).

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
