# scripts/cloud — GPU training on rented/hosted compute

**Placeholder — intentionally empty for now.** Cloud GPU orchestration lives here when we need
it. Right now detector training runs locally on Apple MPS (`scripts/local/train_detector.py`),
which is reliable and reaches >98% peak recall in ~4 epochs; cloud is only worth the setup once
a bigger model or many-fold sweeps outgrow the Mac.

## Layout

- `colab/` — headless [google-colab-cli](https://github.com/googlecolab/google-colab-cli)
  orchestration (A100/L4/T4). A first attempt lived here and is preserved in git history
  (commit before the refactor); it worked end to end but the integration overhead (token
  handling, dataset persistence) wasn't worth it versus local MPS for the current model size.
  For interactive Colab, use `notebooks/train_detector_colab.ipynb` instead.
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
