# Session log — 2026-07-25

Recap of the work, decisions, and results from this session. Live strategy: `docs/STRATEGY.md`.
Memories: `.claude/memory/`.

## Where we ended (state at close)

- **Best confirmed LB: 0.912** (two-seeds ensemble + DeepCenter). Target **0.932** (gap ~0.02).
- E005 repro = 0.888; baseline blob = 0.601 (local 0.644 → **~0.04 local>LB offset**).
- **unet_transformer repro** submitted (`54983506`) — **PENDING** at close (slow re-run scoring;
  two-seeds took ~6h to score). Expected ~0.94.
- **Golden-8 knob sweep** (det-threshold × ILP-weight) **RUNNING** at close (kernel
  `biohub-tune-frontier-golden8`).
- Submissions used today: 3/5.

## What we did

1. **Detector over-detection fix** (owned pipeline): selecting checkpoints by val peak-recall made
   the detector over-detect worse as it trained (golden-8 score 0.40→0.26, nodes 42k→93k). Fixed:
   select on `sel = recall·exp(−0.1·|node_ratio|)` anchored to `est_nodes/100`; `pos_weight` 20→4.
2. **Built the owned pipeline as a measurement lab** (all tested): greedy divisions linker
   (`link/greedy.py`), offset-to-parent head (linajea), multi-hypothesis candidates
   (`link/candidates.py`), label-free `motion_affinity`, TTA/ensemble (`detect/infer.py`),
   integrated `scripts/local/eval_full.py`. Measured golden-8: greedy+multi-hyp **0.489** vs
   distance-ILP 0.429. GT-trained edge model HURT (sparse-label problem) → use motion affinity.
3. **Reality check**: owned pipeline (~0.45 LB proj.) is ~half of E005 (0.888) and the target.
   Incremental tuning won't close that → **strategy pivot**.
4. **Studied 2 public reference solutions**:
   - **sdeogade / `unet_transformer`** (~0.94): temporal-UNet + transformer edge + tracksdata ILP
     + heavy post-proc (dual-seed consensus, D4 TTA, safe-division geometry, gap closing,
     short-track rescue). Driver over the **pilkwang support pack**.
   - **xiaoleilian / `mix-divaug`**: 2× UNet heatmap ensemble + Hungarian + line-fit + a division
     **metric EXPLOIT** (fake out-of-volume nodes to game `division_jaccard`; author excludes it
     from final-2). We will NOT use the exploit — target the private board with genuine quality.
5. **Pivot executed**: reproduced `unet_transformer` under our account (kernel
   `biohub-unet-transformer-repro`, ran the full ~0.94 machine), submitted it.
6. **Verified metric trust** (P3): the pack's `evaluate.py` uses `biohub_tracking.metrics` — the
   exact reference our `ctrack.eval.biohub` was checked against.
7. **Ensembling dead end (verified)**: two-seeds and unet_transformer are **99.4% identical** (same
   pilkwang weights) → not independent → ensembling them gains nothing. Our owned detector is too
   weak (~0.45) to add value. So local merging can't improve the best submission.
8. **Current lever = knob tuning** of the pilkwang pipeline (det-threshold, ILP weights,
   safe-division geometry), pre-filtered on golden-8, confirmed on the real LB.

## Key gotchas learned

- **Kernels-only comp**: CSV-via-CLI submission 400s; must submit via the notebook UI
  ("Submit to Competition"), and pick `submission.csv` (not `run_stats.csv`).
- **Contamination**: 4 visible test = 4 train volumes; pilkwang weights trained on all train incl.
  golden-8 → local golden-8 scores of the pretrained pipeline are inflated (rough pre-filter only).
- **Kaggle**: 2 concurrent GPU sessions max; default P100 breaks torch → `machine_shape=NvidiaTeslaT4`;
  no CLI cancel (`kernels_delete` frees a slot). Predictions dir must be under `/kaggle/working`.

## Open next steps (P4)

1. Read the golden-8 sweep grid → pick det/ILP that beats the preset (0.96875 / app 0.0 / disapp 1.5).
2. Build ONE variant notebook (edit the `BIOHUB_*` env vars in the config cell), submit, confirm on LB.
3. Keep only LB-confirmed winners; reserve **final-2** private picks for durable submissions.
4. (Optional) Full-notebook golden-8 harness to tune the safe-division/post-proc env vars.

## Assets / IDs

- GitHub: https://github.com/mbottoni/biohub-cell-tracking
- Kernels (maruanbottoni/): `biohub-unet-transformer-repro`, `biohub-two-seeds-repro`,
  `biohub-tune-frontier-golden8`, `biohub-train-detector-fold0` (deleted), `biohub-infer-owned-pipeline`.
- Datasets: `ctrack-src`, `biohub-detector-weights`. Support pack: `pilkwang/biohub-tracking-support-pack-50ep-v1`.
- Data staged at `/Volumes/extSSD/kaggle/biohub/{raw,gt}/train`.
