# scripts/local — runs on this machine (Apple MPS / CPU)

Development, training, evaluation, and data prep. All import `ctrack` from `src/`.

| script | what it does |
|---|---|
| `train_detector.py` | train the 3D U-Net detector (MPS) — `ctrack.detect.train.train_fold` |
| `train_edge.py` | train the learned edge-affinity model |
| `run_baseline.py` | classical blob→link→score pipeline (+ `--sweep`, `--ilp`) |
| `eval_pipeline.py` | score a trained detector end-to-end via the real metric |
| `verify_metric.py` | check `ctrack.eval.biohub` against the reference implementation |
| `analyze_gt.py` | ground-truth stats across all volumes |
| `check_crop_overlap.py` | do same-embryo crops overlap? (CV validity) |
| `fetch_data.sh` / `fetch_subset.py` | download competition data |
| `unzip_data.py` | extract the archive with progress/resume |

Run from the repo root, e.g.:
```bash
python scripts/local/train_detector.py --raw /Volumes/extSSD/kaggle/biohub/raw/train \
    --gt /Volumes/extSSD/kaggle/biohub/gt/train --fold 0 --epochs 12
```
