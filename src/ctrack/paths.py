"""Canonical paths. Everything else imports from here so no path strings get hardcoded."""

from __future__ import annotations

import os
from pathlib import Path

# Override with CTRACK_DATA=/some/big/disk when the dataset outgrows the repo volume.
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CTRACK_DATA", ROOT / "data"))

RAW = DATA / "raw"
INTERIM = DATA / "interim"       # registered / normalized volumes
PROCESSED = DATA / "processed"   # training crops, cached detections

OUTPUTS = ROOT / "outputs"       # per-run predictions
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
CONFIGS = ROOT / "configs"


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, PROCESSED, OUTPUTS, MODELS, SUBMISSIONS):
        p.mkdir(parents=True, exist_ok=True)
