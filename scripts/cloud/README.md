# scripts/cloud — GPU training on rented/hosted compute

Cloud GPU orchestration for the full-recipe retrain lives in `colab/` (smoke-verified
08-24; launch is an operator decision gated by PLAN 3.1e). Local MPS
(`scripts/local/train_detector.py`) remains the path for the small ctrack detector.

## Layout

- `colab/` — headless [google-colab-cli](https://github.com/googlecolab/google-colab-cli)
  orchestration (A100/L4/T4) for the **PLAN 3.1e funded retrain**, smoke-verified 08-24.
  Read `colab/RETRAIN_PLAN.md` first: data path, GPU choice, cost arithmetic, the
  checkpoint-beacon stop mechanism (row 91's fix), and the exact launch commands.
  Supersedes the repo-root `colab/` directory (w3-probe / row-91 era), which is kept as
  the historical record. For interactive Colab, `notebooks/train_detector_colab.ipynb`
  still exists but trains the ctrack detector, not the full recipe.
- `vast/` — [Vast.ai](https://vast.ai) rental orchestration. Not started.

## When to fill this in

- The detector needs `base >= 32` or 3D+temporal attention (too slow on MPS), **and**
- we want all 5 CV folds and multi-seed ensembles trained in parallel.

## Contract for anything added here

A cloud training entry point should:
1. import the `ctrack` package (the single source of truth — never fork the training loop),
2. call `ctrack.detect.train.train_fold(...)`,
3. write weights out and make them retrievable (download or push to a Kaggle dataset),
4. tear the instance down on exit.

Keep the *logic* in `src/ctrack`; cloud scripts are thin drivers only.
