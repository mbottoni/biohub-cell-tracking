# PLAN v3 — Colab execution of the winning-model program

**Updated:** 2026-08-10 (v3 — supersedes the v2 campaign plan, which was deleted after its
content migrated into `docs/STRATEGY.md`; read that file for how we got here, this file for
what we execute next).

| | |
|---|---|
| **Deadline** | 2026-09-29 (~7 weeks left) |
| **Our shelf** | **0.915** public LB (final-2: `pub162-port`, `pub162-thr096`) |
| **The bar** | **0.949**, genuine (exploit patched, board rescored — STRATEGY §47) |
| **Compute** | Colab Pro: **254 units** at start of campaign; ~35 spent through probe epoch 11, plus unmeasured pack/idle hours since. **Read the exact balance from the Colab UI before each stage launch.** Kaggle T4 quota (~30 GPU-h/wk) is a separate, currently idle budget. |
| **Thesis** | The 0.949 gap is detection recall on the 96% of cells nobody trains on (STRATEGY §13, §48). The full winning program (dense self-supervised pretraining → sparse fine-tune, multi-frame tracklet transformer, diverse ensemble) needs 350–500 GPU-h — beyond Colab. **Colab's job is to buy the decision tree at pilot scale**: every unit spent must produce a pre-registered verdict that either justifies the Vast scale-up or kills it honestly. |

## Stage 0 — finish the window-3 probe + matched w2 control (~35u remaining)

**Status: running.** Orchestrator relaunched 2026-08-10 evening; verified 60-volume subset
(re-packed with per-volume verify after the silent `6bba_fc83837d` omission) staged as
`biohub-w3-subset`. Checkpoint store `biohub-w3-ckpt` holds epoch 11, best sel **0.8881**
(from 0.8643 at epoch 0), target epoch 28.

1. Train w3 to epoch 28 (or patience-10 early stop). Orchestrator handles session deaths;
   checkpoints bank every epoch.
2. **w2 control under identical conditions**: same 48-volume train set, same sel-12 selection
   volumes, same epoch count, `--window-size 2`. ~2–3 h GPU (w2 epochs cost half). This is the
   only clean comparison — the public w2's 0.9574 on sel-12 is contaminated (trained on those
   volumes, row 57).

**Gate G0 (pre-registered):** w3₂₈ ≥ w2-control₂₈ + 0.005 on sel-12 → multi-frame context is
real; w3 becomes the Stage 2 base. Otherwise w2. *Either answer feeds Stage 2.*

## Stage 1 — dense pretraining pilot (~65u) — the headline

The pocket version of the learn-from-the-96% thesis.

1. **Build `scripts/pretrain_unet.py`** (M1-smoked before any Colab launch, per working
   agreement): masked-patch reconstruction on 2–3-frame windows of the subset volumes, no
   labels — mask random 3D regions of the input, reconstruct through TemporalUNet3D + a 1×1
   decode head. ~15 epochs ≈ 2.5 A100-h.
2. **Fine-tune from pretrained weights**: the trainer already supports `--unet-weights` warm
   start, so this is config, not surgery. Det+edge heads on sparse GT, 15 epochs, identical
   48/12 split.
3. **Three-way comparison at equal fine-tune epochs**: scratch-w2 (G0 control) vs scratch-w3
   (probe) vs pretrained.

**Gate G1 (pre-registered):** pretrained ≥ scratch + 0.01 on sel-12 → the thesis behind 0.949
is validated at pilot scale. This is the single most valuable bit this budget can buy: it
converts the $400–800 Vast question from a gamble into an informed decision. A null is equally
decisive — it kills the big spend before it happens.

**Infra reuse:** same orchestrator/runner/beacon pattern; the runner gains a `SCRIPT` env
parameter; pretraining banks to its own store (`biohub-mae-ckpt`).

## Stage 2 — best-combo consolidation (~40u)

Winner of G0 × winner of G1 (e.g. pretrained-w3), trained to ~30–35 epochs with EMA weights —
the best model Colab can produce. One change per experiment still applies: EMA is added only
if Stage 1 ran without it.

## Stage 3 — integration verdict (0 Colab units)

1. **Golden-8 gates locally** (free, uncontaminated — quarantined from train AND selection).
2. If passed → evalset kernel on **Kaggle T4 quota** (separate budget, idle for days).
3. Official scorer (`scripts/local/score_official.py`, calibrated: official 0.9117 ≈ ctrack
   0.8947) vs the 0.915 shelf.
4. **Gate G3:** +0.005 official-scorer improvement earns one of the 5 daily submissions.
   Anything less does not touch the final-2.

## Budget & sequencing

| Stage | Units | Wall-clock | Gate |
|---|---|---|---|
| 0 — probe + control | ~35 | 1–2 days | G0: w3 vs w2, +0.005 |
| 1 — pretrain pilot | ~65 | 2–3 days (1 day build) | G1: pretrain vs scratch, +0.01 |
| 2 — consolidation | ~40 | 1–2 days | best sel-12 model |
| 3 — integration | 0 (Kaggle) | 2 days | G3: official +0.005 → submit |
| Reserve | ~20 | | reruns, one failed session tax |

Stages 0→1 overlap: the Stage 1 build (local M1 work) happens while Stage 0's sessions burn.
**Re-budget checkpoint before Stage 1 launch**: read the real unit balance; if < 100 remain,
Stage 1 shrinks to 10 pretrain epochs + 10 fine-tune epochs (the gate still reads).

## Escalation — what moves to Vast if G1 fires

Pretraining on ALL 199 volumes at full resolution, longer schedules, the tracklet transformer,
and the 2–3 model ensemble: ~350–500 GPU-h ≈ $400–800. Estimated odds of 0.93+ conditional on
G1 passing: 40–50%. If G1 nulls and G0 nulls, the campaign closes at 0.915 with the map fully
drawn — that is a legitimate outcome, not a failure of the plan.

## Infrastructure (battle-tested, reuse as-is)

- **`colab/orchestrator.sh`** — perpetual provision→launch→poll→respawn loop on the Mac.
  Death detection = checkpoint-store/beacon staleness, never ssh silence (row 55: runtimes
  outlive their tunnels by hours). Tier-aware patience (A100 stale>40m grace 50m; L4 60/90).
  Adopt guard: never provisions while the store shows a life signal < 45 min old.
- **`colab/w3_runner.sh`** — on-VM phases: deps→auth→data→repo→ckpt-pull→train→ckpt-push;
  flock single-instance lock; stale-zip purge; live pusher banks checkpoint + epoch history +
  train-log beacon every 5 min.
- **State lives in Kaggle datasets**, never on VMs: `biohub-w3-subset` (verified 60-volume
  pack), `biohub-w3-ckpt` / `biohub-mae-ckpt` (checkpoints), `biohub-w3-beacon` (liveness +
  log tail).
- **Platform facts learned the hard way**: sessions die at 10–90 min but runtimes keep
  training for hours; `colab sessions` drops live runtimes from its listing; CPU-tier network
  is throttled (~24 MB/s) vs GPU (~300–400 MB/s) — never stage data on a CPU session; Kaggle
  auto-extracts uploaded zips.

## Discipline (unchanged, non-negotiable)

Every run gets a row in `docs/experiments.md`; verdicts and corrections go to
`docs/STRATEGY.md` as numbered sections; smoke locally on the M1 before any Colab/Kaggle
launch; one change per experiment; the LB (via the official scorer proxy) is the only
instrument that has consistently told the truth.
