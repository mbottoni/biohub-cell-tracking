# PLAN — current position and next steps (2026-08-22, ENDGAME — §52)

**Read this first.** It is the only document you need to act. `docs/STRATEGY.md` and
`docs/experiments.md` are the archive: consult them to check whether an idea is already
dead (they are the reason this project stopped re-deriving wrong conclusions), never as a
starting point. Row numbers below point into `docs/experiments.md`.

---

## 1. Where we stand

| | value |
|---|---|
| Confirmed LB | ★ **0.920** (`55678092`) — NEW BEST, rank **149 of 2,633** |
| Previous best | 0.915 × 2 (`55325321`, `55327761`) |
| Readable bar | leader **0.958**; #2 0.957; top-10 ≈0.943; top-20 ≈0.938; top-50 0.929; **top-100 0.922** (rows 96, 99) |
| What bought it | the **public LB-0.920 line ported verbatim** (row 98) — a different BRANCH: DeepCenter veto ON, ranker/lookahead/edge-TTA OFF, bidir weight 0.30 |
| Final-2 | **0.920 + one 0.915** |
| Colab | credits available; no session running |
| **Deadline** | **2026-09-29** — ~38 days, ~190 submission slots, ~5×30 Kaggle T4-h/week + Colab |
| Fresh intel (row 99) | nothing public above 0.920; the 0.920 author is publicly testing SAFE_DIV widening on his own line and his comment logs an **LB loss** (0.914 at MAX_UM 7.0) — watch `kimi-notebook-v17`, do not copy |
| ⚠ Standing obligation | **RE-CENSUS THE BOARD WEEKLY.** Rows 87–95 spent a week on our own ideas for +0.000; one census + one port bought +0.005 (rows 96, 98) |

> ## ☠☠☠ THE STACK IS CLOSED — LB 0.912 vs 0.915 (row 90)
> `55539437` shipped clean (audit passed, `non_adjacent_edges` 0, 1,226 dropped) and **scored
> 0.912 — a LOSS of 0.003.** The pre-registration said *"~0.917 expected; flat kills the stack"*;
> it came in below flat. **Do not resubmit the stack. The final-2 stays the two genuine 0.915s**
> (`55325321`, `55327761`).
>
> ### ★★★ READ THIS BEFORE TRUSTING ANY LOCAL NUMBER
> The stack measured **+0.0036 locally** and **−0.003 on the board**: sign inverted, total error
> ≈ **0.0066**. The cause is structural, not a bug. **evalset-24 is drawn from TRAIN volumes and
> the pilkwang detector trained on EVERY train volume — so no uncontaminated local set for the
> public pipeline exists.** §50.2 measured the level inflation at +0.062; row 90 shows the
> *ranking* is unreliable too. Mechanism: the stack's gain was detection-FN recovery via a
> gap-close clamp lifted to max_gap 3. Where the detector memorised the data, the few remaining
> gaps are easy and safe to bridge; on hidden test detection is worse, so a 3-frame bridge spans
> real discontinuities and invents edges. **Gap closing is riskiest exactly where detection is
> weakest, and our local set is where it is strongest.**
>
> **Operating rule from now on:** a local delta below **~0.007** is uninterpretable. Local gating
> may be used to **REJECT** (lose locally ⇒ dead) but **never to CONFIRM**.

> **⚠ HISTORICAL (resolved above) — the stack could not be submitted as originally built (row 83).**
> Quota freed 00:09 UTC 08-15, autosubmit pushed, and the commit run died 27 min in on the
> kernel's own `clean_graph_audit`: `non_adjacent_edges = 1226`, one of twelve invariants that
> must be 0. Lifting the gap-close clamp emits dt=2/dt=3 edges; the pub162 port at LB 0.915
> has zero and passes. **The gate did its job — FATAL, no submission.**
>
> **The fix is free.** GT edges are all t→t+1, so dt≥2 edges can never match; stripping all
> 6,588 from the measured stack rescores to **0.9135 with every term identical**. The metric is
> inert to them. `scripts/build_stack_adjonly_kernel.py` builds the compliant kernel (verified
> 1226 → 0 against the real failed output). **It is NOT pushed** — it is a different artifact
> from the row-66-verified one, and spending quota plus a submission slot is an operator call.
> To ship: push `kernels/pub162_stack_adjonly`, then re-arm `scripts/autosubmit_stack.sh`
> against the new kernel id (`biohub-pub162-stack-adjonly`).

**`pub162_stack` = ILP disappearance 1.5→2.0 + gap-close clamp lifted with max_gap 3.**
Verified by replaying the built kernel: byte-identical output (sha256 `87d51dda`) to the
measured config (row 66). `scripts/autosubmit_stack.sh` will push → commit → gate on the
output → submit exactly once, unattended, when quota frees (~Sat 08-15, inferred from
last consumption on 08-08). Pre-registered read: **~0.917 expected; flat kills the stack.**

---

## 2. The reframing that changes everything (rows 70–72) — ⚠ **route falsified by rows 78–81**

> The **decomposition below is solid and still governs everything** — read it. What rows 78–81
> falsified is the *deployment* claim built on top of it: that a division classifier could
> harvest the headroom. The headroom is real; the geometric proposer cannot reach it. See §3.2.


    score = adj_edge_jaccard + 0.1 × division_jaccard
    adj_edge_jaccard = max(0, edge_J × (1 − 0.1 × node_ratio))

Read from `official_metric/src/tracking_cellmot/metrics.py:33`, not inferred.

Our stack banks `0.9065 + 0.1×0.0698 = 0.9135`. **Divisions contribute 0.007 of a possible
0.100.** Every lever this project has killed — association, localisation, distillation,
node budget, and the 08-11..13 knob sweeps — lives in the *first* term, where we fight for
+0.002 at a time. The division term holds ~13× our remaining headroom and went unexamined
for weeks because nobody read the formula.

Three facts establish the route (all measured, all free):

1. **Headroom** (row 70). On evalset-24 (25 GT divisions; we score TP 3 / FP 18 / FN 22):
   15 TP @ 10 FP → **0.9494 ≈ the bar**. Killing all 18 FPs while keeping 3 TPs is worth
   only +0.005 — **this is a recall problem, not a precision problem.**
2. **Feasibility** (row 71). Of the 25, **16 already have parent AND both daughters in our
   own node set**. We see the cells and fail to call the event: it is an *assignment*
   failure, not a detection failure. 16 is the hard ceiling for any re-ranking approach.
3. **Signal** (row 72). Appearance separates divisions: **ROC-AUC 0.845**, precision 0.314
   at recall 0.50, from 32 crude features over 151 GT divisions with volume-grouped CV
   and time-matched negatives. §35 killed *geometry* on this problem; appearance is a
   different information source and it carries.

**Deployment arithmetic** (ceiling 16, adj 0.9065 held fixed):

| recall × precision on detectable divisions | div_J | score |
|---|---|---|
| 0.5 × 0.31 (today's crude features) | 0.188 | 0.9254 (+0.012) |
| 0.7 × 0.60 | 0.345 | 0.9410 |
| **0.9 × 0.70** | 0.462 | **0.9527** |

> ☠ **THE TABLE ABOVE IS DEAD — it was never reachable (row 86).** Kept because the reasoning
> that killed it took four rows and two wrong turns, and deleting it would invite a rebuild.
>
> It prices the route on *recall × precision over detectable divisions*, which silently assumes
> the only cost of a wrong division is its own FP. **It isn't.** Built end-to-end, the detector
> hits AUC 0.863 and captures 6 of 7 attachable divisions — and **every K loses**, because a
> fork placed anywhere useful also trips the cross-component / merged-branch FP rules. Gaining
> needs `Δt > 0.0698·Δf`; we are 3× short at best and **~150× short** of a meaningful gain.
>
> **What remains true from §2:** the score decomposition, and that divisions hold 0.007 of a
> possible 0.100. **The headroom is real and unreachable.** Row 70's "13× our remaining
> headroom" is arithmetic, not an opportunity. See §3.2.

---

## 3. Next steps — THE ENDGAME PLAN (§52)

**The gap to first is +0.038 in ~38 days; there is time and budget to fund exactly ONE
structural play.** §52.2 argues (argument, not measurement) that the three biggest closures —
detection-side (row 87), the retrain (row 91), Trackastra (row 94) — were made on a local
instrument that memorised its own volumes and is therefore blind to the error class hidden
test contains. Phase 1 converts that argument into measurement before anything is spent.
Standing obligations (census, 3.1g) continue throughout.

### 3.1 Phase 1a — DECOMPOSE THE HIDDEN TEST: two probe submissions (do first)
We have never decomposed an LB number; every hidden-test read has been a single scalar.
The metric (`official_metric/.../metrics.py:33`): `score = max(0, edge_J × (1 − 0.1 ×
node_ratio)) + 0.1 × div_J`. Two designed probes on the **pub920 base** isolate the terms:

1. **`pub920_divoff`** — pub920 with safe-division insertion disabled (the `divoff`
   mechanism, e.g. `BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP=0`; verify division count → 0 in the
   graph audit). **Pre-register the expected drop from local replay first** (cache-hit,
   minutes). LB drop ≈ `0.1 × div_J_hidden` + whatever edge-side coupling replay quantifies.
2. **`pub920_pad`** — add K isolated far-field nodes (the July `probe_pad` mechanism,
   scored 0.868 on the old base) to measure the node_ratio multiplier slope → recovers
   hidden `n_gt`, hence `node_ratio_hidden`, hence **`edge_J_hidden`** from probe 1's result.

Output: `edge_J / node_ratio / div_J` of our 0.920 **on hidden test**, each compared to its
local-replay value → per-term local→hidden transfer error, replacing the single ±0.007
scalar. Cost: 2 slots scoring below 0.920 — harmless, final-2 selection protects us.
⚠ Pre-register both reads before submitting, per row 90 discipline.

### 3.1b Phase 1b — attribute the 0.920 (free, local; was §3.1c, §51.4)
Row 98 shipped the whole branch unattributed. Hypothesis: `DUAL_SEED_MIN_CANDIDATE_RETENTION
= 0.90` is the rail that lets the DeepCenter veto run without the recall collapse that killed
`divveto` (355/355 rejected) and row 95's partner (0.9838 → 0.6955). Ablate in local replay —
drop the floor, or DeepCenter — and see which returns the points. Required before BUILDING on
0.920; not needed to bank it.

### 3.1c Phase 1c — finish the HONEST INSTRUMENT (mostly already paid for, row 91)
We hold: the contamination-guarded 120/12/24 split (`configs/fullrecipe_splits.json`), an
honestly-trained checkpoint (`best_ep20_sel0.9585.pth`, sha d7c43d4d — *matches* the public
detector's inflated sel number), holdout replay, the official scorer. Missing: one clean full
pipeline run on the 24-vol holdout with **OUR checkpoints as both seeds** (e.g. ep15 + ep20),
using the **fixed** override hook (row 93) and **sha-asserting every staged weight against the
manifest before AND after the run**. That instrument's detection-error profile resembles
hidden test. Then **re-score the instrument-conditional kills on it**: Trackastra blend
(row 94, killed at −0.0007, below the floor), retention gates (row 88), gap-close (row 90's
own mechanism predicts it gains where detection is worse), the 0.917 orphan-pool division rule
(row 97). **Any sign flip is a legitimate reopening.**

### 3.1d Phase 2 — THE DECISION GATE (end of week 1)

| probe outcome | read | play |
|---|---|---|
| `edge_J_hidden` ≈ local edge_J | detection survives on hidden; row 87's cap roughly binds after all | → **3.1f** divisions / ensemble |
| `edge_J_hidden` well below local | the gap is detection/association robustness | → **3.1e** funded retrain |
| `div_J_hidden` ≫ local 0.07 | division term carries more than believed | re-price both before choosing |

### 3.1e Phase 3a — THE FUNDED RETRAIN (~$160–200 of Colab units; OPERATOR PURCHASE DECISION)
Re-priced from §50's "unfunded": 400 epochs ≈ 138 A100-h ≈ 1,630 units ≈ **$160–200
pay-as-you-go** in a $60,000 competition. Prereqs before restarting: fix the orchestrator to
read the stop signal from the **checkpoint beacon**, never ssh (row 91's ~100 min/epoch waste);
reuse the guarded split. Target: **replace BOTH seeds** so the 2-seed consensus veto (the
pipeline's real precision mechanism) runs on genuinely generalising partners — row 95 showed a
single decorrelated partner grafted onto the public primary destroys recall; a matched PAIR is
the unmeasured configuration. Gate on the honest instrument (3.1c) before spending a slot.

### 3.1f Phase 3b — DIVISIONS VIA THE 0.917 FAMILY / CROSS-FAMILY EDGE ENSEMBLE
**Not** gate-widening on the pub920 branch — row 99 is an *LB-measured* loss (0.914). Open:
1. Port the **0.917 family whole** (`reyhanksatria/graph-patches-for-cell-tracking-0-917-lb`;
   strip `id_no`/`docker_image` before push). Its value is not its 0.917 — it is a different
   family whose `CAND_THR 0.05` orphan pool feeds the mutual-NN + divergence division rule
   that starved on our graph (row 97). Measure its div_J on the honest instrument.
2. **Edge-level intersection/union across families** (pub920 / pub162 / 0917) — a different
   operation from row 95's detection-level veto, never tried. Sweep on the honest instrument;
   LB-gate one operating point.

### 3.1g STANDING — weekly board census + watch `kimi-notebook-v17` (§51.5)
`kaggle competitions leaderboard --download` + `kaggle kernels list --sort-by dateRun`.
Two specific watches: the 0.920 author's kimi-v17 score when it posts (free differential
intel on the frontier author's direction), and any new public notebook above 0.920 (port
immediately, it preempts everything).

### 3.1h Week 5 (from ~09-22) — FREEZE
Stop structural work; spend remaining slots confirming final-2 candidates. Public↔private can
invert exactly as local↔public did (row 90) — banked boring robustness beats a last-week
gamble.

---

**The sections below are carried unchanged from the pre-endgame plan (numbering preserved
for cross-references). 3.3's items are now the backlog behind the phases above.**

### 3.2 Division route — ☠ **CLOSED ON MEASUREMENT (row 86).** Do not reopen without new information
Built end-to-end and swept against the official scorer. **Every operating point loses:**

| K inserted | score | div TP | div FP |
|---|---|---|---|
| **0 (control)** | **0.9135** | 3 | 18 |
| 250 | 0.9111 | 4 | 57 |
| 2 000 | 0.9053 | 6 | 1 105 |
| 20 000 | 0.8841 | 5 | 17 851 |

**The detector is not the problem.** AUC 0.863 held out; it captures 6 of the 7 attachable
divisions by K=10 000. **The problem is that inserting a fork anywhere useful is expensive.**
FPs are charged on forks matching annotated GT nodes *and* on cross-component / locally-merged
branches. Random forks land in sparse unannotated regions and are nearly free (row 84); a
*useful* detector fires on bright, well-tracked, annotated nodes carrying matched children, so
it triggers both FP rules. **Selection and cost are positively correlated — ranking quality
cannot fix it.** Gaining needs `Δt > 0.0698·Δf`: best case is 3× short, a meaningful +0.007 is
**~150× short**. No CNN closes 150×. **Do not spend Colab credits here.**

> **The argument history, kept because it is instructive.** Rows 79/81 killed this route with
> wrong arithmetic. Row 84 corrected the arithmetic and revived it — also wrongly, by projecting
> random-fork costs onto detector-selected forks. Row 86 settled it by running the thing.
> **Three arithmetic arguments in one day, two of them wrong. Price a route by running it.**

<details><summary>Superseded re-ranker plan (kept per the corrections rule — rows 78–81 are why)</summary>

Gate passed (row 72). Build in this order:

1. **Candidate pool + honest negatives — DO THIS FIRST, it gates everything below.**
   Rows 78–79 settled the shape of this step the hard way. The pool must be the **pipeline's
   own safe-division candidate set**, never free enumeration (1:336,000 — unlearnable).
   **The one measurement to take next, cheap and decisive: how many of the 16 reachable
   divisions appear in that candidate set?** That is the true ceiling on this route.
   Row 76 is a warning it may be well under 16 — widening the generator converted *zero*
   new true divisions. If the answer is small, the bottleneck is the *generator*, and no
   amount of re-ranking or CNN capacity fixes it. **Measure this before spending Colab credit.**
2. **3D+t CNN, warm-started from the public TemporalUNet encoder** (we hold the weights;
   151 positives is too thin to train an encoder from scratch). Crops at t−1/t/t+1.
   **This is what the Colab credits are for: hours on one A100.**
3. **Integrate as a re-ranker** over safe-division candidates, replacing the geometric
   gates. Sweep its threshold on the free local instrument to pick the operating point on
   the `0.1 × div_J` vs `adj_edge_J` trade.
4. **Gate before submitting:** must beat 0.9135 on evalset-24 with the official scorer.

</details>

**Gate 4 still binds on anything that replaces this: beat 0.9135 on evalset-24, official scorer.**

### 3.2b Re-read every "flat" verdict — totals hide opposing moves (row 73)
`MOTION_RELINK_LEARNED_BONUS` 6.0 scores 0.9133 vs the stack's 0.9135 — a dead heat that a
pre-row-70 session would log as "flat". Decomposed it is **+0.0026 on the association term
(our best ever there), cancelled by −0.0027 of division term**. So: a flat total is not a
null result, and configs judged on totals before row 70 are suspect — including row 24's
`divgeom`. Re-reading them is nearly free: every run stored its `run_stats.csv`.
Relink-6 specifically is **stranded value** — once the classifier supplies divisions instead
of the geometric insertion, its association gain should stop being masked. Retest it then.

**Stranded-value ledger — what §3.2 unblocks (rows 73, 77).** Three configs now buy association
and pay for it in divisions, all blocked by the same term:

| config | Δ adj_edge_J | Δ division cost | net today |
|---|---|---|---|
| `MOTION_RELINK_LEARNED_BONUS` 6.0 | **+0.0026** | −0.0027 | −0.0002 |
| `DET_THRESHOLD` 0.94 | +0.0010 | −0.0029 | −0.0019 |
| `DET_THRESHOLD` 0.96 | +0.0007 | −0.0005 | +0.0002 |

They are **not additive** — relink and threshold both act on node/edge supply — so re-measure them
jointly once the classifier supplies divisions, never by summing. But the direction is consistent:
association is *not* exhausted, it is being taxed.

**`divgeom` re-read: the suspicion was wrong, the original verdict stands (row 76).** It scores
0.9102 — never flat — and adds 28 false divisions while converting **zero** new true ones
(TP 3→3, FN 22→22). So the bottleneck is **selection, not candidate generation**: widening the
geometric gates admits the missed events and still cannot pick them. This is the sharpest
statement yet of why 3.2 must be an appearance *selector*, not a bigger pool.

**The test that separates the two cases (row 75).** A flat total is masked value only if
`div_J` MOVED. Round 9 finished with both kinds side by side: relink-6 is flat-with-div_J-moving
(stranded, retest later), min-track-len is flat-with-div_J-*bit-identical* (a true null, never
revisit). So when re-reading a pre-row-70 "flat" verdict, read `div_J` first — if it is unchanged
the total was already the whole story and the row needs no re-litigation.

### 3.3 Cheap and still open (free, cache-hit minutes each)
- `SAFE_DIV_*` gate sweep on the stack base — queued as round 10. Row 24 re-centred these
  and lost on the LB, but was judged on total score with no view of the two terms.
- Dual-seed **edge** threshold 0.48→0.30 with motion-relink bonus 6.0: the one historical
  change that produced a confirmed LB gain for us (rows 22–23), never tried on this base.
  Needs re-inference (~7 h M1, or minutes on Colab — see 3.4).
  ⚠ **`DUAL_SEED_EDGE_THRESHOLD` is one of the six knobs `cell08` used to clobber (row 82).**
  Before row 82's fix this sweep would have silently measured the baseline and been logged as
  a null — burying our only confirmed-LB-gain association lever. The driver now asserts
  overrides survive to inference; **confirm you see `[local] all N overrides verified live
  after cell08` in the log before believing any result from this knob.**

### 3.4 Use Colab as a fast replay farm, not just a trainer
Inference-side configs cost ~7 h each on the M1 and ~20–30 min on an A100. If more
inference sweeps are wanted, ship the replay driver to a Colab session rather than
queueing days of M1 time. Machinery exists: `colab/{orchestrator,w3_runner}.sh` are
parameterised (`WINDOW`/`METHOD`/`CKPT_DS`) and proven end-to-end.

---

## 4. Closed — do not reopen without new information

> ⚠ **§52.2 caveat (08-22):** every ☠ below that was measured **locally** inherits the
> contaminated instrument (evalset = train volumes the detector memorised). Three are queued
> for honest-instrument re-test (3.1c): the detection-side bound (row 87), the full-recipe
> retrain (row 91), and the Trackastra blend (row 94, killed at −0.0007 — below the ±0.007
> floor). **LB-measured closures stand** (the stack, node-ratio/metric shaping, row 99's
> division-gate widening).

| route | verdict |
|---|---|
| Association / relink / ranker / lookahead | dead (§34, §37, §39) |
| Localisation refinement | dead, fails its gate by 3.5× (§40, row 37) |
| Detector distillation (own student) | structurally capped — a student is a candidate-subset of its teacher (§48.2, rows 48–50) |
| Window-3 temporal context | dead — +0.0021 for 1.6× cost vs a matched w2 control (§50, row 61) |
| Node-ratio / metric shaping | dead, loses points (§10.5, §10.6 ⚠ WRONG) |
| ILP appearance & division weights | inert — division weight is byte-identical at 1.5/2.0 (row 64) |
| Detection threshold — **STRICTER only** | dead; 0.98/0.99 lose on both terms (row 68) |
| ~~Detection threshold, looser~~ | **REOPENED (row 77)** — "at optimum" was read off totals. Association rises monotonically as it loosens (0.9065→0.9072→0.9075 at 0.96875/0.96/0.94); only division losses hold it back. Stranded until 3.2 lands. |
| Gap radius, min-track-len, gap2-recovery | flat or noise (rows 62, 65); **min-track-len re-confirmed decomposed — pure association knob, div_J bit-identical, 6 is optimal (row 75)** |
| ILP disappearance weight above 2.0 | dead — 2.5/3.0/4.0 all lose monotonically, and ~90% of each loss is the DIVISION term, not association (row 74) |
| **Division selection by GEOMETRY** | dead (§35, §43) — **but appearance is not; see 3.2** |
| Full-recipe retrain at public scale | ~200 A100-h (§50). ~~Unfunded~~ **re-priced §52.3: ≈1,630 Colab units ≈ $160–200 — an operator purchase decision, not a wall (3.1e)** |
| **The stack (disapp 2.0 + gap 3)** | ☠ **LB 0.912 vs 0.915 (row 90)** — measured +0.0036 locally, LOST 0.003. Local↔LB error ≈0.0066, sign inverted |
| Detection-side, ALL forms (ultrack multi-hypothesis, Spotiflow, looser thresholds) | ☠ closed on a **mechanism-independent bound** (row 87): only 68 annotated cells are undetected, so perfect detection is worth **~+0.003**— **⚠ §52.2: the bound was measured on memorised train volumes; it binds the LOCAL set, not hidden test. Re-test on the honest instrument (3.1c)** |
| Node retention / short-track rescue | ☠ +0.0003 and saturating (row 88). Selective beats blanket — but the node-budget penalty eats 80% of the gain |
| **Full-recipe retrain (own detector)** | ☠ row 91 — sel matches honestly (0.9585 vs their inflated 0.9574) but it over-detects 3×, threshold cannot fix it, and closing it needs ~400 epochs (~1,630 units) for a payoff row 87 caps at +0.003— **⚠ §52.2: the payoff cap is row 87’s (instrument-conditional) and the over-detect diagnosis vs the public primary ALONE is unmeasured (row 93). REOPENED as Phase 3a (3.1e), gated on the probes (3.1d)** |
| **Division route, ALL forms** | ☠ closed on measurement (row 86). Detector ranks well (AUC 0.863, 6/7 captured) and still loses at every K, because a useful fork is inherently expensive. ~150× short of a meaningful gain |

---

## 5. Operational essentials

- `.venv/bin/kaggle` (2.2.4). Homebrew's 1.5.13 sends Basic auth and 401s.
- Code-competition submit: `-k <user>/<kernel> -v <version> -f submission.csv`.
- 5 submissions/day (00:00 UTC); 30 GPU-h/week; ~21 min commit run; 4–6 h scoring rerun.
- **Free local instrument**: `scripts/local/pub162_local_replay.py` replays the whole
  pub162 pipeline on the M1, byte-identical to Kaggle (row 60). ILP/post-proc variants are
  cache-hit (minutes); anything touching inference needs its own `BIOHUB_PRED_CACHE`
  **or it silently reuses the baseline's predictions** (row 67).
- **Score only after the `[local] DONE` line** — `submission.csv` exists before it is
  finalised, and scoring early yields a wrong number (caught 08-13).
- Replay assets and the official scorer live on `/Volumes/extSSD/kaggle/biohub/replay_assets`.
  `/private/tmp` is garbage-collected and ate them once (rows 67, 67b); the scorer is an
  **editable** install, so it rots with the directory it points at.

---

## 6. Method notes earned the hard way

- **One change per experiment is right for attribution and wrong as a search strategy.**
  A strict one-factor-from-baseline sweep would have measured the gap fix alone, scored it
  flat, and buried the biggest lever of the week. It survived only because it was tested
  on top of an already-confirmed change (row 69).
- **Read the objective before optimising it.** Weeks went into the term worth +0.002 while
  the term worth +0.09 sat unread (row 70).
- **Verify the artifact, not the config.** The kernel we submit is a different file from
  the one measured; replay it and compare bytes (row 66).
- **A patch that reaches the code and changes nothing is a finding**, not a null result —
  it means the constraint does not bind (row 64). **But prove it reached the code first.**
  Row 82 looked exactly like row 64 — a knob whose output was byte-identical to base — and was
  a silent no-op instead: a later notebook cell overwrote the override. The two are told apart
  by evidence *outside* the score. Row 64 grepped the subprocess command line and found the
  hardcoded clamp; row 82's tell was the pipeline printing `detection weight=0.475` while the
  config said 0.30. **Any result identical to its control to 4 decimals: sha256 it against the
  control, then confirm the value actually reached inference — before calling it inert.**
