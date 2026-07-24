# Colab GPU training

`train_detector_colab.ipynb` trains the 3D U-Net detector on a Colab GPU. This exists because
the dev Mac's MPS is ~18–58 min/epoch (3D convs) and the Kaggle GPU pool assigned a P100 whose
compute capability (sm_60) the Kaggle image's PyTorch doesn't support. A Colab A100/L4/T4 is
both fast and supported.

## Use

1. Upload `train_detector_colab.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. **Runtime → Change runtime type → GPU** (A100 or L4 if your Pro tier offers them; T4 works).
3. Run cells top to bottom. You'll paste your Kaggle token once (cell 3).
4. It downloads `ctrack` + `N_VOLUMES` train volumes from Kaggle, trains, and **pushes weights
   back to Kaggle** as the `biohub-detector-weights` dataset.

## The loop it closes

```
Colab GPU: train UNet ──push weights──▶ Kaggle dataset: biohub-detector-weights
                                                │
                              inference kernel attaches it, runs
                              detect → ILP → submission.csv
```

So after a Colab run finishes, the submission kernel (`kernel_infer/`) just needs
`biohub-detector-weights` in its `dataset_sources` and it uses the freshly trained detector.

## Knobs (cells 5–6)

- `N_VOLUMES` (default 150 of 199): fewer = less disk/faster download. 150 ≈ 82 GB.
- `FOLD`, `EPOCHS`, `BATCH`, `BASE`: training config. `BASE=32` is the real model (the Mac ran
  `BASE=16` for speed). On an A100 you can push `BATCH` higher.
- `TRAIN_STRIDE=2`: use every 2nd annotated frame (adjacent frames are near-identical).

## Notes

- The notebook is self-contained: training logic mirrors `scripts/train_detector.py` but
  imports the `ctrack` package pulled from the `ctrack-src` Kaggle dataset, so it stays in sync
  with the repo as long as that dataset is refreshed (`kaggle datasets version` from
  `kaggle_assets/ctrack_src`).
- To train all 5 folds, set `FOLD` to 0…4 and re-run cells 6–7 (weights accumulate in the
  dataset as `unet_fold{0..4}.pt`).
