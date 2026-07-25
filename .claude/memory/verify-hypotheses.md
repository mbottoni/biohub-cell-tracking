---
name: verify-hypotheses
description: User expects hypotheses verified against ground truth before acting or claiming
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 04028120-dfc0-408d-9b71-734ce309387d
  modified: 2026-07-25T17:51:17.339Z
---

Always VERIFY a hypothesis before acting on it or reporting it as fact, and report outcomes
faithfully (say when something failed, was skipped, or is only a floor/early signal).

**Why**: this is a research competition where wrong assumptions waste GPU hours and submissions.
Real examples that paid off: decoding + verifying the metric field-for-field against the
reference implementation; discovering the visible-test contamination by comparing local vs LB;
catching that "best-by-recall" was selecting the WORST detector checkpoint by actually running the
pipeline eval instead of trusting the training metric. See [[metric-and-contamination]], [[detector-status]].

**How to apply**: check claims against the reference/data/a quick experiment, not intuition. When
comparing two things, hold the pipeline constant (e.g. same linker) or say the comparison isn't
apples-to-apples. Label early/undertrained results as floors, not verdicts. Prefer the fast
owned-pipeline local score as ground truth over proxy training metrics.
