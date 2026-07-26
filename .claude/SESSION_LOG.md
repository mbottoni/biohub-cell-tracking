# Session log

Recap of the work, decisions, and results per session. Live strategy: `docs/STRATEGY.md`.
Memories: `.claude/memory/`.

---

# 2026-07-26 — the CV inverted on the real LB; pivot to LB-driven search

## Headline: §9's central finding is FALSIFIED

Both pending submissions scored:

| config | golden-8 | LB |
|---|---|---|
| FULL `selected_94` (`54983506`) | 0.862 | **0.911** (author claims ~0.94 → **0.03 repro gap**) |
| LEAN, additive post-proc off (`54985941`) | 0.882 | **0.890** |
| Δ (LEAN − FULL) | **+0.020** | **−0.021** |

The rank **inverted**. §9.1 predicted LEAN would win on unseen data; it lost by 0.021.

**Root cause — we had the contamination backwards.** We believed "LB test is contaminated, golden-8
is clean." The opposite is true: `test/` is a dev sample that Kaggle swaps for real hidden data at
rerun (kernels-only — `docs/data.md` says this explicitly), so the LB was always scored on clean
hidden crops. Meanwhile the pilkwang weights trained on all 199 train volumes including golden-8, so
**golden-8 is the leaky instrument** — and we used it to overrule the clean one.

Consequences: the LB is our best tuning signal for this pipeline; §9.2's "dead levers" were all
killed on the discredited harness and must be **re-opened**; §9.6's "generalization discipline is our
edge" no longer follows.

## New thesis: the headroom is in the node-ratio multiplier (STRATEGY §10.5)

`score = edge_J·(1 − 0.1·ratio) + 0.1·div_J`. The field optimizes `edge_J` (~0.89, near ceiling) and
ignores the multiplier (~1.03 today, max ~1.10). Two verified facts make it a free lunch:
- only edges touching **annotated** GT nodes are scored — edges between unannotated cells are invisible;
- annotation is **3.6% of the node budget** (median `nodes/est_n_nodes` over 199 vols; 659 vs 17,909).

⇒ pruning predictions from ~75% → ~10% of the budget is **+6% relative ≈ +0.055** at no `edge_J`
cost, *if* we drop only never-scored nodes. Reduces to: **predict which cells the annotators traced**
— a supervised problem with 199 labelled volumes. First experiment is free (no GPU, no submission):
measure AUC of P(annotated) from track features on train GT.

## Submission budget is the unspent resource
Deadline 2026-09-29, 5/day → **~300+ remaining, 6 used.** We were rationing submissions to protect a
CV that doesn't predict the target. Compute is the real limit (~6h/run, 2 GPU slots → 2–4/day).
Noise floor ~0.001 (two 99.4%-identical pipelines scored 0.911 vs 0.912) ⇒ effects ≥0.005 readable.

## Next steps
1. Free experiment: is annotation predictable from track features? (§10.5 gate)
2. Close the 0.03 repro gap — diff our kernel vs the published `unet_transformer` notebook (no compute).
3. LB coordinate descent from the 0.912 config: `DET_THRESHOLD` → ILP weights → dual-seed → secondary edge.
4. Re-probe the "dead" levers on the LB, one submission each (divisions first).

---

# 2026-07-25

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

## Update (2026-07-25 late) — overfit, dead levers, CV skew, playbook

- **LEAN > FULL on golden-8** (0.882 vs 0.862): the `selected_94` additive post-proc (gap-close,
  gap2, motion-relink, safe-divisions, linefit) is OVERFIT to the contaminated LB test and hurts
  held-out edge_J. LEAN submitted (`54985941`). This is the generalization/money-board edge.
- **Dead levers (verified locally):** (a) division specialist — precision hopeless (0–3 TP for
  23–119 FP), every config worse; (b) detector-ensembling-to-ADD — metric rewards under-detection
  (ratio ≈ −0.3 is a BONUS), so adding cells lowers score; (c) node pruning — tapped out.
- **Adversarial validation: golden-8 is SKEWED** — 2× denser (est 37k vs 17.6k) + embryo-balanced
  (50/50 vs train 35/65). CV not representative → built REP-VAL (12 vols, matched density/embryo);
  re-running LEAN vs FULL there (`rep_full`, `rep_lean`) to confirm the generalization finding.
- **Kaggle playbook status:** #1 CV (have, fixing skew), #4 adversarial-val (done), #5/#6 mostly
  done in-pipeline. UNTAPPED: **#2 diverse edge-ensembling** (local-association-ranker + DeepCenter/
  Spotiflow, blend edge probs) and **#3 retrain+pseudo-label** (hold out REP-VAL, densify sparse GT).
- **Next:** REP-VAL result → lock LEAN → #2 ensembling → #3 retrain. See docs/STRATEGY.md §9.

## Assets / IDs

- GitHub: https://github.com/mbottoni/biohub-cell-tracking
- Kernels (maruanbottoni/): `biohub-unet-transformer-repro`, `biohub-two-seeds-repro`,
  `biohub-tune-frontier-golden8`, `biohub-train-detector-fold0` (deleted), `biohub-infer-owned-pipeline`.
- Datasets: `ctrack-src`, `biohub-detector-weights`. Support pack: `pilkwang/biohub-tracking-support-pack-50ep-v1`.
- Data staged at `/Volumes/extSSD/kaggle/biohub/{raw,gt}/train`.
