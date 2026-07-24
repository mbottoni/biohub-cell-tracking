# biohub-cell-tracking

Solution repo for the Kaggle competition
[**Biohub — Cell Tracking During Development**](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development):
detect and track cells in 3D+time through zebrafish embryo development (light-sheet
microscopy, Zebrahub data).

📋 **Strategy lives in [PLAN.md](PLAN.md).** Read it first.

## Setup

```bash
uv sync                 # create .venv and install deps
source .venv/bin/activate
```

Kaggle credentials go in `~/.kaggle/kaggle.json` (kaggle.com/settings → API → Create New
Token), mode `600`.

## Data

```bash
make data               # download + unzip competition files into data/raw/
```

`data/` is gitignored — nothing from it is ever committed.

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
