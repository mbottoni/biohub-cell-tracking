# PLAN — current position and next steps (2026-08-13)

**Read this first.** It is the only document you need to act. `docs/STRATEGY.md` and
`docs/experiments.md` are the archive: consult them to check whether an idea is already
dead (they are the reason this project stopped re-deriving wrong conclusions), never as a
starting point. Row numbers below point into `docs/experiments.md`.

---

## 1. Where we stand

| | value |
|---|---|
| Confirmed LB | **0.915 × 2** (`55325321`, `55327761`) — the current final-2 |
| Readable bar | **0.949** (§44) |
| Best local config | `kernels/pub162_stack` — official evalset-24 **0.9135** vs pub162 base 0.9099 |
| Submission status | BLOCKED on Kaggle weekly GPU quota; autosubmit watcher armed and polling |
| Colab | credits available; no session running |

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

> ⚠ **The precision column is NOT transferable — see row 79.** Those precisions come from
> row 72's probe at a **1:20** base rate (GT divisions vs sampled non-dividing nodes). Free
> enumeration of division candidates from our own graph has a **1:336,000** base rate: 5.4 M
> candidates for 16 positives. Holding the probe's implied FPR (0.055), precision there is
> **~2e-5**. Reaching 16 TP @ 10 FP needs **FPR ≤ 7e-7** — under 5 false positives in 7.5 M.
> No model closes that; **the pool is the constraint, not the classifier.**
>
> The table stays valid **only** if the classifier re-ranks the pipeline's own safe-division
> candidates (~355, or 13,495 loosened → ~1:840), which is the regime the probe describes.
> This is what §3.2 step 3 always said; `division_rerank_e2e.py` violated it by enumerating
> freely, and that is why it was stopped (rows 78–79).

---

## 3. Next steps, in order

### 3.1 Ship what is measured (no decision needed)
Autosubmit fires on quota reset. If the stack gains, `pub162_disapp20` (local 0.9119,
−2.1% nodes) is the conservative second arm. **Do not submit a gap-only variant** — row 69:
the gap fix is worth *nothing* alone (0.9098 ≈ base); the +0.0036 is an interaction with
the disappearance weight, so the stack is not decomposable.

### 3.2 Division classifier — ⚠ RE-RANKING IS CLOSED (row 81); only a learned PROPOSER is left
**Read row 81 before doing anything here.** The re-ranker plan below is superseded. Measured:
the shipped `SAFE_DIV` generator proposes 3,734 candidates containing **0 of the 16** reachable
divisions, so re-ranking upside is **exactly zero** — the events are not in the pool to rank.

The structural reason, and why no tuning fixes it: **no radius has both recall and a tractable
pool.** Predicted parent→daughter is median 18.3 µm, so recall needs ~26 µm → 3.68 M candidates
→ 1:283,000, which row 79 showed demands FPR ≤ 7e-7 (unreachable from AUC 0.845). A pool small
enough to be learnable (~1:20) contains **zero** positives. Mutually exclusive under *any*
distance-based proposal rule. This is §35/§43 turned from an empirical failure into a
quantitative impossibility.

**The appearance selector was never the weak link — the geometric proposer is.** Classifier
capacity and Colab hours cannot repair a pool that lacks the positives.

**What survives:** a *learned, non-distance-based* proposer — a per-node "this cell is dividing"
head. It operates at node level (~1:20,000) rather than pair level (1:283,000), and row 72's
AUC 0.845 was measured at node level, which is the only positive evidence we hold. Still far
from what precision demands, so it must be **trained**, not hand-featured — a detector-side
change of the kind §50 priced at ~200 A100-h and marked unfunded. Price it honestly before
committing; that price is why this is not simply "the next step".

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
| Full-recipe retrain at public scale | ~200 A100-h (§50). Unfunded, and not affordable on current credits |

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
