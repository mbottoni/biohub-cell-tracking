# PLAN 3.1e — the funded detector retrain on Colab: execution plan

**Status gate (read PLAN.md first):** 3.1e is currently **DEPRIORITISED** (rows 106–108:
the hidden edge term shows no gap large enough to fund against; the open route is the 0.917
division graft, 3.1f). This directory makes the retrain *launch-ready to smoke level* so
that IF the gate re-opens, the full run is an env-var change — per §50.3's contract.
Launching the 400-epoch run is an operator decision, not something a session does on its own.

## What the run is

Train **TWO decorrelated seeds** of the full pilkwang recipe (TemporalUNet + transformer
edge head, `train_unet_transformer.py` from `pilkwang/biohub-tracking-support-pack-50ep-v1`)
to replace both public seeds, which memorised the train volumes (row 90: +0.062 inflation).
The 2-seed consensus veto is the pipeline's real precision mechanism; row 95 showed one
decorrelated partner grafted onto a memorising primary destroys recall — a matched PAIR is
the unmeasured configuration.

- Split: `configs/fullrecipe_splits.json` — 120 train / 12 selection / **24 holdout
  (evalset-24), quarantined**; the runner refuses to train if a holdout volume is staged.
  Built by `scripts/local/make_fullrecipe_split.py` (row 91 used exactly this split).
- Config: window 2 (§50: w3 is dead), batch 8 on A100 / 4 on L4, `--seed` per arm.
- Epochs: **cap 400** (`EPOCHS=400`), early stop `--patience 40` on sel-12. Row 91's
  patience 10 fired at e32 with `det_loss` still falling — 40 is deliberate headroom;
  0 disables. The orchestrator treats `since_best >= PATIENCE` in the *banked checkpoint*
  as terminal, so runner and orchestrator must agree on PATIENCE.
- Seeds: `SEED=1337` (arm s1) and `SEED=7331` (arm s2), fresh stores per arm. The patcher
  (`scripts/local/patch_fullrecipe_trainer.py`, now 6 anchors) adds `--seed` and seeds
  torch/np/random *before model construction* — upstream's `seed` param was never passed by
  `main()` and only seeded the DataLoader, i.e. it would NOT have decorrelated the inits.
- Optional: arm s1 may instead resume row 91's banked e32 checkpoint
  (`maruanbottoni/biohub-fullrecipe-ckpt`, best sel 0.9585 @ e20, since_best 12 < 40 so it
  resumes cleanly) — saves ~11 A100-h but inherits an unseeded init. Default is fresh.

## Data path to Colab

Per session: `kaggle competitions download` of the **81 GB competition zip** (~20–45 min,
measured), one-pass `unzip` of the 132 split volumes (~108 GB extracted), per-volume
retry + verify (row 54: bulk unzip silently skipped a volume), zip deleted before training.
Disk arithmetic (Phase-0 measured, real runtime): /content ≈ 206 GB; peak 81 + 108 = 189 GB.
Do not raise the volume count without re-doing this arithmetic. The holdout-leak gate runs
before any GPU spend. Smoke runs use `DATA_MODE=dataset` — a pre-packed 3-volume Kaggle
dataset (`maruanbottoni/biohub-frsmoke-data`, built once from the extSSD copy by
`smoke.sh`). ⚠ Per-file competition download is DEAD for this competition: the download
endpoint 404s on every file the listing returns (measured 08-24; commit 01be9b7 abandoned
the same route in July). The zip is the only competition-data channel.

## Session type

**A100 preferred, L4 fallback** (the orchestrator tries in that order):

| | s/epoch (132 vols) | 400 ep compute | units/h | compute cost/seed | wall-clock/seed |
|---|---|---|---|---|---|
| A100 batch 8 | **1,247 s measured** (row 91 history: train 1,218 + test 29) | 138.6 h | ~11.8 | ~1,630 units ≈ **$163** | ~5.8 days |
| L4 batch 4 | ~1,920 s (v1 orchestrator's observed ~32 min) | ~213 h | ~4.8 | ~1,025 units ≈ **$102** | ~9 days |

L4 is ~40% cheaper per epoch but too slow for two sequential seeds before the ~09-22
freeze (3.1h). Two A100 orchestrators in parallel (one per seed, separate stores) is the
schedule that fits — verify the account's concurrent-session allowance before relying on it.

## Cost estimate (honest version)

- Compute per seed (A100, full 400 epochs): 1,630 units ≈ **$163** at $9.99/100 units.
- Session overhead: every respawn re-stages the 81 GB zip. Row 91's arc implies ~25–30%
  overhead at observed session lifetimes → **~2,000–2,100 units ≈ $200–210 per seed**.
- **Both seeds ≈ 3,900–4,200 units ≈ $390–420 worst case.** PLAN 3.1e's "$160–200" figure
  is one seed's compute. Early stop at patience 40 can cut this a lot (row 91 plateaued by
  e20–32 at 120-vol scale); treat 400 as a cap, not an expectation.
- Smoke: T4 ≈ 1.76 units/h, ~30 min ≈ **under $0.10**.

## The checkpoint-beacon stop mechanism (row 91's fix — the 3.1e prerequisite)

v1 grepped `train.log` over ssh; ssh is unreachable while the trainer runs, so after early
stop v1 respawned sessions that re-downloaded 81 GB to train one epoch each (~100 min waste
per useless epoch). v2 (`fullrecipe_orchestrator.sh` here) opens **no channel to a running
VM** — the new colab CLI does not even have ssh. All signals ride two Kaggle datasets:

- **Checkpoint store** (`biohub-fr400-s{1,2}-ckpt`): `checkpoint_last.pth` (atomic, every
  epoch, pushed within ~5 min by the VM-side pusher) carries `epoch`, `best_score`,
  `since_best`, and the full per-epoch `history`. Orchestrator terminal conditions, read
  from the banked file at every loop top: `epoch >= EPOCHS-1`, or
  `since_best >= PATIENCE`.
- **Beacon store** (`biohub-fr400-s{1,2}-beacon`): phase marker + heartbeat while staging;
  `train.log` tail every 5 min while training; terminal `STATE=done` / `data-FATAL` beacon
  with message + log tail. A fresh `data-FATAL` makes the orchestrator **exit loudly**
  instead of crash-looping sessions against a broken config (row 58b: ~8–10 units burned).
- Staleness (no life signal beyond tier-specific STALE/GRACE) is the only respawn trigger;
  a stale `STATE=done` from a previous session is ignored via a freshness window.

## Smoke result (08-24, VERIFIED end to end)

T4 session `frsmoke1`, 3 volumes (2 train + 1 sel from the guarded split), 2 epochs,
seed 1337, `DATA_MODE=dataset`. Total ~19 min of T4 ≈ 0.6 units ≈ **$0.06**.

- session create → runner detached: **2m37s** (create 14 s; uploads + exec all clean).
- runner launch → STATE=done: **~6m20s** (deps ~2 min, dataset staging ~1 min for 1.3 GB,
  contamination guard passed "3 staged / 0 holdout", patcher **6/6 anchors**,
  `seeded: torch/np/random <- 1337` confirmed in train.log).
- 2 epochs on T4 batch 2: train **93 s / 96 s**, test ~17 s (2+1 volumes) — ~47 s/volume,
  consistent with row 91's A100 batch-8 10.2 s/volume at ~4.6×. Backward pass is 74% of
  T4 epoch time. Best score 0.8632 after 2 epochs on 2 volumes (sanity only).
- checkpoint + beacon banked to `biohub-frsmoke-{ckpt,beacon}` (stores auto-created);
  checkpoint retrieved via BOTH channels (Kaggle store and `colab download`),
  **byte-identical** (sha 3768404d…), `history` = 2 rows with per-epoch timings.
- Session stopped; `colab sessions` empty.

Two real bugs caught and fixed by the smoke: (1) the `phase done` push landed AFTER the
rich `STATE=done` beacon and masked it (latest version wins) — the terminal beacon is now
the last push and the EXIT trap is DONE-aware; (2) the dataset-ingest check parsed the
listing's separator row. Also confirmed dead: per-file competition download (404 on every
file, from the Mac and by design of the smoke's fallback to a packed dataset).

⚠ Extrapolation caveat: smoke timings are T4-on-3-volumes and structurally under-measure
the full run (no 81 GB staging, no session churn). Size the funded run ONLY from row 91's
banked full-scale measurement (1,247 s/epoch A100) — which the smoke's per-volume rate
corroborates but does not replace.

## Exact commands

Smoke (verified path; T4, 3 volumes, 2 epochs, then self-stops):

    scripts/cloud/colab/smoke.sh

Full launch, seed 1 (A100 preferred; run from the repo root, survives logout):

    SEED=1337 CKPT_DS=biohub-fr400-s1-ckpt BEACON_DS=biohub-fr400-s1-beacon \
      SPREFIX=fr400s1 EPOCHS=400 PATIENCE=40 \
      nohup scripts/cloud/colab/fullrecipe_orchestrator.sh > /tmp/fr400s1_orch.log 2>&1 &

Seed 2 (parallel if concurrent sessions are allowed, else after s1):

    SEED=7331 CKPT_DS=biohub-fr400-s2-ckpt BEACON_DS=biohub-fr400-s2-beacon \
      SPREFIX=fr400s2 EPOCHS=400 PATIENCE=40 \
      nohup scripts/cloud/colab/fullrecipe_orchestrator.sh > /tmp/fr400s2_orch.log 2>&1 &

Watch (all read-only, no VM contact):

    tail -f /tmp/fr400s1_orch.log
    .venv/bin/kaggle datasets download maruanbottoni/biohub-fr400-s1-ckpt -f history.jsonl -p /tmp/h1
    .venv/bin/kaggle datasets download maruanbottoni/biohub-fr400-s1-beacon -f beacon.txt -p /tmp/b1

Prereqs assumed present (all verified 08-24): `~/.local/bin/colab` authenticated
(`colab sessions` answers), `~/.kaggle/access_token` (KGAT), `.venv/bin/kaggle` 2.2.4,
`configs/fullrecipe_splits.json` + `configs/fullrecipe_volumes.txt` committed.

## After the run

Per seed: pull `edge_predictor_best.pth` from the ckpt store, score on the **holdout-24
with the official scorer** (the honest instrument this split was built for), then integrate
the PAIR as the dual-seed consensus in the pub920 pipeline and LB-gate one submission.
Row 91's warning stands: sel-metric parity does not imply deployed parity — the deploy gate
is the holdout replay, not sel.
