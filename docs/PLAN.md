# PLAN — current position and next steps (2026-09-14, THE 0.947 RESET)

> **This header supersedes everything below it** (kept per the corrections rule; the old
> §1–§3 describe the pre-0.947 world). Evidence: rows 120–126.

## 0. Where we stand (2026-09-14 night)

| | value |
|---|---|
| Confirmed LB | ★ **0.948** (`56313419`, pub947_densad, row 131) — density-adaptive motion-relink on the 0.947 base; FIRST above-shelf config. Previous: 0.947 port (`56232696`, row 125) |
| ⚠ Shelf warning | ~~450-team 0.947 tie~~ **DISSOLVED (row 139, 09-19 census): board is now a gradient — rank-100 0.949 · rank-150 0.948 · rank-200 0.947. We are 0.948 ≈ rank 138 and sinking ~3 ranks/day.** Cuts: top-50 0.953 · top-25 0.957 · top-10 0.964 · leader 0.974 |
| Final-2 | **densad 0.948 (safe) + TBD aggressive** (consensus `56345198` pending decides it) |
| Deadline | **2026-09-29**. Merger/new-entrant deadline **CONFIRMED via API: 2026-09-22 23:59 UTC** (row 139) |
| Closed 09-14 (rows 124/126) | one-knob perimeter of 0.947: DC thr 0.15 (−0.001), relink bonus 6.0 (−0.001, row-73 lever still taxed), global div cap ×2 (flat — **cap not binding on hidden**, diverges from local row 111) |
| Dead as of 09-14 | the 3.1e retrain (obsolete: 0.947 extracts +0.024 from the same detector, row 122); ALL our pre-0.947 postproc division mechanisms (rows 114–119, different substrate anyway) |
| Key structural read (row 126) | capup FLAT + dcthr NEGATIVE ⇒ if hidden holds more real divisions, they are blocked by **confirmation precision**, not the cap ⇒ an added independent confirm signal + cap raise is the funded division play |

## 0.0b Wave-2 verdicts (09-18, rows 129–132) — the plan below is REWEIGHTED by these
- **div_J_hidden(0.947 line) ≈ 0.35** (divoff 0.911): division term near-saturated; adj ≈ 0.912; **the 0.023 gap to the leader is EDGES** — division work closed on every substrate (divnet combo 0.944 confirmed).
- **densad 0.948 = new best**: density-conditioned relink is real (+0.001); uniform was −0.001. Edge mechanisms respond to per-volume conditioning — the general lesson for the edges lane.
- haideptry branch debunked (0.945); their DivNet is dead code (row 128 forensics).

## 0.1 The plan (in order)

1. **divoff probe on the 0.947 base** (1 slot) — calibrated inversion (rows 100/103/106)
   reads div_J_hidden of the 0.947 config; decides whether the 0.023 gap to the leader is
   edges or divisions BEFORE either lane is funded.
2. **pub947 local replay** on the M1 (adapt `pub162_local_replay.py`; own `BIOHUB_PRED_CACHE`)
   — reject-gating for everything below. Local may REJECT, never CONFIRM.
3. **Divisions lane (if probe says divisions):** appearance-confirm stack on THEIR candidate
   pool — div CNN (AUC 0.863, row 116 rebuild) + volume-drop signature (row 122 side-insight:
   −27% half-max volume, onset −2 frames, peak holds) as an added gate, WITH the cap raise
   (the row-126 2-knob combo). Operating point by LB probe.
4. **Edges lane (if probe says edges):** hiranorm node-count-parity DET calibration (row 110),
   relink bonus INTERMEDIATE values (2–3) only if replay-gated, D4-TTA blend-weight micro-grid.
5. **Team merge outreach to reyhanksatria** (operator decision, time-critical): they hold the
   frontier and are publicly recruiting; our dowry = probe decompositions + replay instrument +
   the two appearance division signals. Also: tell zhincez their public copy is mount-fragile
   (row 123b) — goodwill + correctness.
6. **Freeze from ~09-22** (3.1h unchanged): final-2 hygiene; the 450-tie shelf will scatter on
   the private split — banked robustness beats last-week gambles.

Standing rules unchanged: one mechanism per submission · pre-register every read ·
local gates reject-only · register → commit → push · weekly census (it bought +0.024 today).

---

# ARCHIVE — pre-0.947 plan below (2026-08-24, THREE HORIZONS — §53)

**Read this first.** It is the only document you need to act. `docs/STRATEGY.md` and
`docs/experiments.md` are the archive: consult them to check whether an idea is already
dead (they are the reason this project stopped re-deriving wrong conclusions), never as a
starting point. Row numbers below point into `docs/experiments.md`.

---

## 1. Where we stand

| | value |
|---|---|
| Confirmed LB | ★ **0.923** (`55720749`, div917 graft, row 114) — rank ~202 of 2,685; previous 0.920 (`55678092`) |
| Previous best | 0.915 × 2 (`55325321`, `55327761`) |
| Readable bar (08-23, row 110) | leader **0.962**; #2 0.959; top-10 0.944; top-20 0.939; top-50 0.932; top-100 0.923; ⚠ **0.920 ≈ rank 230** (was 149 — the public v17 compressed the board); 0.930 ≈ top-60 |
| What bought it | the **public LB-0.920 line ported verbatim** (row 98) — a different BRANCH: DeepCenter veto ON, ranker/lookahead/edge-TTA OFF, bidir weight 0.30 |
| Final-2 | **0.923 (`55720749`) + 0.920 (`55678092`)** |
| Colab | credits available; no session running |
| **Deadline** | **2026-09-29** — ~38 days, ~190 submission slots, ~5×30 Kaggle T4-h/week + Colab |
| ★★★ PROBES READ (row 106, 08-23) | **dcoff 0.920** (DeepCenter dead weight everywhere — drop it) · gapthr20 0.920 (inert) · **divoff 0.906 → div_J_hidden ≈ 0.13 > local 0.09**. Our hidden decomposition: **adj ≈ 0.906 + divisions ≈ 0.014**. Leader closes through divisions alone: 0.906 + 0.1×0.5 = 0.956 ≈ 0.958 |
| ★ Measured 08-22 (rows 100–102) | replay instrument re-established; divoff probe calibrated; scorer aggregation identified; **memorisation gap measured: ep20 recall 1.0000 on a trained-on volume vs 0.94–0.97 unseen — row 87's detection cap is instrument-conditional** |
| ★★★ PHASE-3 PROBE READ (row 108, 08-23) | **pub917_divoff = 0.893 → div_J_917_hidden ≈ 0.24, ~2× ours (0.13).** The orphan-pool architecture carries the big hidden div term — divisions-explain-the-leader has its first direct evidence. **Hybrid prices at ~0.930** (our adj 0.906 + their 0.024) ≈ top-50 bar. The hybrid is the funded priority |
| Fresh intel (rows 99, 109) | ⚠ **the 0.920 config is now FULLY PUBLIC** (`yunusgmsoy/lb-0-920-...-v17`, titled by its LB, + `kimi-notebook-v17` 106 votes — same knobs as ours, verified by pulling both) → expect rank compression at 0.920; >0.920 requires the hybrid. Author's own SAFE_DIV widening logged an **LB loss** (0.914) — do not copy. Discussions are API-unreachable (KGAT lacks `forums.get`) — census them via browser |
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

## 3. Next steps — THE THREE-HORIZON PLAN (§53, 2026-08-24)

> Supersedes the §52 route ordering below (kept as archive). Read §53 in STRATEGY for
> the evidence. One sentence of context: **we are 0.923 = adj 0.906 + div 0.017, rank
> ~200 and sinking ~0.003/day as the board compresses; stages 2–4 of the pipeline are
> measured dry; the division term is the only +0.03-class headroom and we previously
> closed it with an instrument later proven biased (row 106 vs row 86).**

### H1 — TODAY/TOMORROW: cash the last cheap picks (ceiling ≈ +0.008 combined)
1. **Tophat confirmation rescue** — kernel `pub920_div917_tophat` built; 6-volume local
   gate IN FLIGHT. If not rejected → submit (one change vs 0.923). Ceiling ~+0.003.
2. **Cross-family edge ensemble** — local 917 replay IN FLIGHT
   (`scripts/local/p917_local_run.py`, row 115 unlocked it); then
   `scripts/local/merge_edges_exp.py` scores 4 strategies (transplant_divs / union_add /
   drop_leaf / intersect). Best non-rejected variant → build kernel → submit.
   Ceiling ~+0.005.
3. **Node-count calibration probe** — the node-ratio penalty is SIGNED (hiranorm
   measured under-detection bonus +0.003, row 110). We have never tuned toward the
   calibrated count. One-knob probe AFTER 1–2 read out; costs one slot.
4. Riders for any submission: boundary-track exemption (row 110; ≤+0.0003, free).

### H2 — DAYS 2–5: THE LEARNED DIVISION PIPELINE (the main line, +0.02–0.04)
Re-opens row 86 as an LB question (the local verdict was made with a ruler that
under-prices divisions ~1.5×, row 106). Build spec:
1. **Proposer (high recall, cheap):** candidate triplets (parent, d1, d2) from our final
   graph — single-child sources × ALL nodes within 15 µm at t+1 (claimed or not; no
   filters beyond geometry). Target: ≥80% of GT divisions covered; volume is fine
   because fork FPs are ~free outside the 3.6% annotated set (row 84).
2. **Ranker (the new capability):** small 3D CNN on image patches around (parent, d1,
   d2) + the row-72 features. Train on the 151 GT divisions + hard negatives mined from
   proposer output; volume-grouped CV (the row-72 protocol). Success bar: beat the
   geometric filters' precision at 2× their recall on held-out volumes.
3. **Operating point by LB PROBE, not local score:** submit two points (conservative /
   aggressive) as pre-registered probes; use the row-103 inversion to read div_J_hidden
   directly. The local scorer may only REJECT (bias direction is known: it UNDER-prices).
4. Timeline: proposer + training data day 1, ranker day 2, first LB probe day 3.

### H3 — PARALLEL SLOW TRACKS (operator decisions needed)
1. **Funded retrain** (3.1e below, ~$160–200 Colab): the only adjacency lever
   (+~0.010); the public checkpoints memorised train (row 100). START IF BUDGET
   APPROVED — it has days of wall-clock.
2. **Team merge**: the single biggest win-probability lever. We bring 0.923 + the only
   published-quality hidden decompositions in the competition; a partner brings a
   trained model we lack. Merger deadline is the binding constraint — check it, decide
   within 2 weeks.
3. **Final-2 hygiene** (standing): one safe config (today: div917 0.923) + one
   aggressive; never two experiments. Private-LB split means probes must not chase
   public noise.

### Standing rules (unchanged, they keep paying)
- Local gates may REJECT, never CONFIRM (<0.007 uninterpretable; div term biased low).
- One mechanism per submission; pre-register every read before it scores.
- Weekly board census (rows 96/98/109/110: census beat invention twice).
- Register → commit → push before moving on.

---

## 3-ARCHIVE. The §52 endgame plan (SUPERSEDED by §53 above; kept for the sub-section
specs still referenced: 3.1e retrain spec, 3.1f graft recipe, 3.1g census)

**The gap to first is +0.038 in ~38 days; there is time and budget to fund exactly ONE
structural play.** §52.2 argues (argument, not measurement) that the three biggest closures —
detection-side (row 87), the retrain (row 91), Trackastra (row 94) — were made on a local
instrument that memorised its own volumes and is therefore blind to the error class hidden
test contains. Phase 1 converts that argument into measurement before anything is spent.
Standing obligations (census, 3.1g) continue throughout.

### 3.1 Phase 1a — the highest-probability >0.920 submissions + the ONE probe (stress-tested 08-22)

**Prerequisite (in progress):** the pub920 branch replayed locally. The instrument was
re-established 08-22 (row 100; driver now takes `BIOHUB_REPLAY_SCRATCH`); the DeepCenter
dataset is on the SSD (`replay_assets/deepcenter_dl`); pub920's cells are extracted
(`nbcells_920`) — ⚠ they are a DIFFERENT notebook layout (failfast in cell05, post-proc in
cell06, no knob-guard cell, retention-floor/`DIAGNOSTIC_ARM` code pub162 lacks), so the
driver needs a pub920 variant. Every reject-gate below runs on that replay.

1. **Neighbourhood search around the confirmed 0.920 optimum** — one knob per submission,
   local reject gate + pre-registration each: retention floor 0.85/0.95, DeepCenter gap-veto
   threshold, bidir weight 0.25/0.35, DET_THRESHOLD 0.96. The author's own public iteration
   mined the division-side neighbourhood and lost (row 99); the veto/retention neighbourhood
   is unexplored. Expected: mostly flat, one or two +0.001–0.002 → 0.921–0.923.
2. **pub920 ⊗ pub162 edge-level intersection/union** — both branches held (0.920 / 0.915),
   genuinely different mechanisms, operation never tried (row 95 killed detection-level
   vetoes, a different thing). Build post-hoc from the two submission CSVs; reject-gate; one slot.
3. **`pub920_divoff` probe — CALIBRATED (row 100):** `div_J_hidden ≈ 10 × (LB drop + 0.0005)`;
   the coupling is positive, so the naive read understates div_J. Re-register the coupling on
   the pub920 replay first (DeepCenter safe-div veto changes the division profile). **Real
   payoff:** if div_J_hidden ≈ 0, the hidden division tax blocking the stranded association
   levers (rows 73/77) is a local artifact and they unlock immediately.
   ★ **READ (row 106): div_J_hidden ≈ 0.131 — the tax is real and BIGGER on hidden. The
   stranded levers stay BLOCKED, permanently.**
4. ☠ **Pad probe CUT** (row 101): the scorer's aggregation is now exactly identified
   (per-volume J_adj weighted by edge sample size — the naive model was 16% off), and the
   July pad probe retrospectively bounds hidden n_pred/N_total ≈ 0.97: **the node term is
   small and slightly favourable; the leaders' +0.038 lives in edge_J and/or div_J.**
   Do not spend a slot re-measuring it.

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
(row 97). **Use both instruments as a BRACKET** — the public-weights instrument is too clean (memorised detections, row 102), the ep20 instrument too dirty (deploys at 0.83); hidden test sits between. A candidate must gain on the dirty instrument and not lose on the clean one to earn an LB slot; a sign flip on one instrument alone is a hypothesis, not a result.

### 3.1d Phase 2 — THE DECISION GATE — ★ RESOLVED 08-23 (row 106)

| probe outcome | read | play |
|---|---|---|
| `edge_J_hidden` ≈ local edge_J | detection survives on hidden; row 87's cap roughly binds after all | → **3.1f** divisions / ensemble |
| `edge_J_hidden` well below local | the gap is detection/association robustness | → **3.1e** funded retrain |
| `div_J_hidden` ≫ local 0.07 | division term carries more than believed | re-price both before choosing |

**Outcome: the THIRD row fired.** div_J_hidden ≈ 0.131 (> local 0.0909); hidden adj ≈ 0.906
(≈ −0.02 vs local level, the usual contamination offset — no catastrophic edge gap). The
re-pricing: **the leader's 0.958 closes through divisions alone** (our adj 0.906 + div_J 0.5),
while closing it through edges alone would need edge_J ≈ 0.94 — far above anything measured.
⇒ **Phase 3 = 3.1f (the 0917 family's division architecture), with 3.1c (honest instrument)
as its measuring device** — noting the local instrument UNDER-prices divisions by ~0.04, so
every historical local division verdict was measured with a biased ruler. Row 86 (insertion
on OUR graph) and row 99 (gate widening, LB-dead) still stand — the open route is the
0917 family's orphan-pool architecture, not our graph. **3.1e (funded retrain) is DEPRIORITISED**:
the hidden edge term shows no gap large enough to fund against. Also resolved: **DeepCenter is
dead weight on hidden (dcoff 0.920)** — §51.4's rail hypothesis is FALSE; the rail can be
dropped for free (runtime saving), and the retention floor is the last rail piece under test
(`pub920_ret00`, in flight).

### 3.1e Phase 3a — THE FUNDED RETRAIN (~$160–200 of Colab units; OPERATOR PURCHASE DECISION)
Re-priced from §50's "unfunded": 400 epochs ≈ 138 A100-h ≈ 1,630 units ≈ **$160–200
pay-as-you-go** in a $60,000 competition. Prereqs before restarting: fix the orchestrator to
read the stop signal from the **checkpoint beacon**, never ssh (row 91's ~100 min/epoch waste);
reuse the guarded split. Target: **replace BOTH seeds** so the 2-seed consensus veto (the
pipeline's real precision mechanism) runs on genuinely generalising partners — row 95 showed a
single decorrelated partner grafted onto the public primary destroys recall; a matched PAIR is
the unmeasured configuration. Gate on the honest instrument (3.1c) before spending a slot.

### 3.1f Phase 3b — DIVISIONS VIA THE 0.917 FAMILY / CROSS-FAMILY EDGE ENSEMBLE

> ★★★ **RESOLVED IN FAVOUR OF THE HYBRID (row 108, 08-23).** Step 1 below is DONE and the
> answer is loud: pub917_port = 0.917 (row 107, transfers exactly), pub917_divoff = **0.893**
> → their hidden div term ≈ **0.24 vs our 0.13**, despite a worse adjacency base (0.893 vs
> 0.906). **The build is now the graft: their CAND_THR-0.05 orphan pool + mutual-NN/divergence
> division rule producing forks ON TOP of our 0.920 pipeline's graph.** Additive price
> ≈ 0.906 + 0.024 = **0.930** ≈ top-50. Constraints that bind the graft: reproduce THEIR pool
> (row 86: insertion priced on our graph loses), no gate-widening (row 99), LB-gate the result
> (local <0.007 cannot confirm).

**Not** gate-widening on the pub920 branch — row 99 is an *LB-measured* loss (0.914). Open:
1. ✅ DONE (rows 106–108). Port the **0.917 family whole** (`reyhanksatria/...-0-917-lb`).
   Its orphan pool feeds the mutual-NN + divergence division rule
   that starved on our graph (row 97). Measured: div term ≈ 0.24 on hidden.

> ★★ **THE GRAFT RECIPE IS NOW FULLY SPECIFIED (row 110, from kunal's inline LB changelog —
> `kunaldesale2408/biohub-cell-tracking`, the 0.917-family ancestor).** Reference config:
> pool = **unclaimed t+1 orphan detections** (full-threshold; the "CAND_THR 0.05" phrasing is
> xiaolei's M001 snap pool, a separate idea); parent ≤ 12.0 µm, sister ≤ 15.0,
> existing-child ≤ 10.0; **mutual-nearest-unclaimed-orphan REQUIRED**; **forward divergence
> REQUIRED** (both daughters single-successor at t+2, grandchild − sister separation
> ≥ 2.25 µm); DeepCenter safe-div veto `best.pt` EPOCH 2, thr 0.12; one division per parent;
> caps 0.0076/0.00375. **Field-paid negative control: wide radii WITHOUT the two filters
> = 0.915→0.914** (consistent with our row 99) — the filters are the mechanism, not the
> radius. **ILP division weight is INERT (kunal: 0.3–3.0 all 0.915) — never spend there.**
> A/B variant (backtracking `biohub-general-v6`): in-linker second daughter from the
> transformer's own edge-prob row, prob ≥ 0.25, sister ≤ 15 µm, 1%/frame cap, plus a
> continuation guard (drop fork whose second branch dies at t+2 when prob margin ≥ 0.12).
> **Free LB reads to poll before spending slots: `mtoshidesu/test-notebook-v17-d17ce0`**
> (wide radii + divergence 1.5 on stock v17) **and `kimi-notebook-v17`.**

3. ★ **NEW (row 110) — fused-detection threshold recalibration (hiranorm, real-LB-logged
   series at 0.914–0.916):** probability-averaged dual-model detection kills one-sided
   detections at det 0.96875 ("averaging is not union": 0.99 vs 0.10 → 0.545); lowering the
   fused threshold to **node-count parity with solo** (their 0.90) scored holdout
   +0.0185, and 60% of their recoverable edges fail on one-sided detection. Our pub920
   blends at 0.475/thresholds 0.96875 and the retention floor only *falls back* — same
   structural loss. Experiment: calibrate `BIOHUB_DET_THRESHOLD` on the blended field to
   node-count parity locally (reject-only), then LB-gate one point. Attacks the 0.906 adj
   term; composes with the division graft. ⚠ node-ratio is signed (under-detection bonus)
   — calibrate to parity, not maximum recall.
4. (medium effort) Cross-family detector branch: xiaolei's tophat-trained UNet heatmap,
   mean/std-aligned into the pub920 detection field (M001 evidence: 3 branches 0.917 vs
   single 0.915; preprocessing diversity ≠ seed diversity).
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
  ⚠ **Redundancy caveat (08-22):** its mechanism was detection-FN recovery via node
  retention — and the pub920 branch's `DUAL_SEED_MIN_CANDIDATE_RETENTION=0.90` is itself a
  retention mechanism. Run the 3.1b attribution FIRST; if the floor bought the +0.005, this
  lever likely double-dips nothing.
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
