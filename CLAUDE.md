# Working agreement — Biohub Cell Tracking

Read `docs/STRATEGY.md` first (start at the newest numbered section and work backwards; later
sections supersede earlier ones and say so explicitly). Then `docs/experiments.md` for the run log.

## Always register findings, then commit and push

**Every finding gets written down before moving on.** A result that lives only in a chat transcript
is lost. This is not optional bookkeeping — this project has repeatedly re-derived, and repeatedly
*mis*-derived, conclusions that were already sitting in its own logs.

1. **Record it** in the right place:
   - `docs/experiments.md` — one row per run: config, local score, public LB, and what it means.
     If it isn't here, it didn't happen. Include runs that FAILED and runs not submitted, with why.
   - `docs/STRATEGY.md` — findings that change what we do next. Add a new numbered section rather
     than editing an old one; mark what it supersedes.
   - Kernel-builder docstrings in `scripts/` — the reasoning behind a specific experiment.
2. **Commit** with a message that states the evidence, not just the change. A future session should
   be able to reconstruct *why* from `git log` alone.
3. **Push.**

### Corrections are findings too
When a recorded conclusion turns out to be wrong, **do not silently edit it away**. Mark the old
section (`⚠ WRONG`, `☠ DEAD`, `SUPERSEDED by §N`), state what the evidence actually was, and keep
it. Several sections here are wrong-but-instructive; deleting them would invite repeating the
mistake. Examples now in the doc: §10.6 (concluded the metric hack pays — it loses points), §10.5
(node-ratio thesis, dead), §12.x (argued the association head was the lever — §13 shows detection
recall is).

## Measurement discipline

These are earned, not stylistic:

- **Never trust a name.** The dataset slug says `50ep`; its `ARTIFACT_MANIFEST.json` says
  `400ep-snapshot`. Read the manifest.
- **Small proxies have been reliably optimistic.** A 3-layer conv stack under-predicted real UNet
  cost by ~50×; 8-iteration timings and a misread load figure both misled. Size from full-workload
  measurements only.
- **One change per experiment.** Bundling batch-8 with `cudnn.benchmark` produced a 3× regression
  that could not be attributed to either.
- **Smoke before you spend.** A ~15 min smoke has caught a read-only-filesystem failure, a scoping
  bug, and a silent no-op warm start — each of which would have surfaced hours into a long run.
- **Guard against silent no-ops.** Assert and print: matched-tensor counts, trainable-parameter
  fractions, patch match counts. A patch that quietly fails to apply looks exactly like success.
- **The LB is the only instrument that has consistently told the truth.** Local validation has
  inverted rankings before (§10.1), and the public model's own validation split has overlap 40/40.

## Operational

- Use `.venv/bin/kaggle` (2.2.4). The homebrew CLI is 1.5.13, sends Basic auth, and 401s on
  `KGAT_*` tokens.
- Code-competition submissions need `-k <user>/<kernel> -v <version> -f submission.csv`; a plain
  file upload returns 400.
- 5 submissions/day (resets 00:00 UTC), 2 concurrent GPU sessions, ~30 GPU-h/week, 12 h/session.
- A kernel commit run is ~21 min; the scoring rerun on hidden test is ~4–6 h. Configs are cheap to
  produce and expensive to score — choose which 5 to submit deliberately.
