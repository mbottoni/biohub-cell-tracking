# LINKER-ARCHITECTURE lens — measured findings 2026-09-20 (scripts: linkprobe.py lp2..lp8.py)

Instruments: pred cache /Volumes/extSSD/kaggle/biohub/replay947/cache_densad/<vol>.pkl holds
(coords, all_edges) = the FULL pre-ILP candidate list (src_idx, tgt_idx, prob, dist).
VERIFIED: geff/submission node_id == coords row index (t-mismatch 0-2 per 20k nodes, 6/6 vols).
So every candidate prob is joinable to the shipped graph OFFLINE, no GPU.

## F1. The learned bonus is structurally incapable of fixing any assoc-FN
Of the 22 assoc-FN whose correct pred-pred pair is inside the TIGHT gate, 18 have
candidate prob EXACTLY 0 (never proposed by the transformer -> no bonus can reach them).
For the other 4 the TAKEN (wrong) edge has BOTH higher prob AND shorter distance
(0.910 vs 0.742 @0.41um vs 4.89um; 0.716/0.501; 0.837/0.651; 0.777/0.692).
Sweep bonus = 8,12,16,20,30,50 -> correct beats taken in 1/22 at EVERY value.
=> MOTION_RELINK_LEARNED_BONUS is dead as a lever in both directions. Explains r124 (uniform 6 = -0.001)
and caps densad_tuned mid_bonus 8.0 at noise.

## F2. The geometric Hungarian overrides the transformer's unique confident parent 1,417 times
targets with a UNIQUE candidate parent at p>=0.7: 85,873. Linker chose a DIFFERENT parent: 1,417 (1.65%).
76% of those overrides (1,081) are on pairs >8um. Per-volume override rate:
0b24845f 0.6% | 5f15d135 0.5% | 87bba6c4 4.8% | 062c8d37 1.7% | 7af54fde 0.3% | d2b9fc0c 0.2%
(87bba6c4 is the worst volume, edge_J 0.528.)
For all 19 assoc-FN whose correct pair IS a candidate, rank was 1 of 1 — the transformer's ONLY
proposed parent was the correct one, and the relinker replaced it with a source it never proposed
(pr_taken_source = 0.000 in 15/19).

## F3. assoc-FN reachability census (n=69)
gate x has-candidate-prob: tight&no-prob 18 | tight&prob 4 | relaxed-only&no-prob 19 | relaxed-only&prob 6
                           outside-all&no-prob 13 | outside-all&prob 9
correct pred-pred distance: p10 4.84 p25 6.11 med 8.30 p75 10.48 p90 12.52 um
(the GT edge is short — median GT displacement 3.37um — but the two MATCHED pred nodes are 8um apart)
50/69 correct pairs are not candidates at any prob (threshold 0.48 on softmax-over-SOURCES).
source is an orphan end: 20/69 | target is an orphan start: 11/69 | both orphan: 0/69.

## F4. SIMULATED LINKER ARMS (same node set as ds_ctl submission, official matcher/scorer counting)
base ds_ctl: micro edge_J 0.9315 (TP3266 FP124 FN116), weighted(ratio-neutral) 0.9315
  pure-transformer 1-1 Hungarian (no geometry)     micro 0.9403  TP3263 FP 88 FN119
  FULL candidate probs into the EXISTING bonus cost micro 0.9309 (FLAT/neg)  <- kills "expose probs as bonus"
  same, ungated                                     micro 0.9256 (NEG)
  -log(prob)+0.15d cost                             micro 0.9264 (NEG)
  ** tf-pass(prob-Hungarian) THEN geometric fill **
     gate 6.5um   micro 0.9301  weighted dw -0.0010
     gate 8.0     0.9323  +0.0006
     gate 9.0     0.9339  +0.0021
     gate 11.0    0.9361  +0.0039
     gate 14.0    0.9411 (TP3274 FP 97 FN108)  weighted 0.9398  dw +0.0082   <== recommended
     ungated      0.9414  weighted 0.9401  dw +0.0085 (emits a 82.9um edge; 0.12% of edges >14um)
     gate14 + mutual-top-only guard: BIT-IDENTICAL to gate14 (free insurance)
     gate14 + prob floor 0.85: 0.9375 (worse — floor hurts)
     gate14, geometric fill TIGHT ONLY (drop relaxed pass): 0.9419 weighted dw +0.0089
per-volume (ship -> gate14): 0b24845f .980->.960 | 5f15d135 .880->.880 | 87bba6c4 .528->.609
   062c8d37 .985->.997 | 7af54fde .925->.926 | d2b9fc0c .988->.990
weighted-gain attribution: 87bba6c4 +0.0046 (w 5.7%), 062c8d37 +0.0031 (w 25.9%), d2b9fc0c +0.0006,
   7af54fde +0.0003, 5f15d135 -0.00003, 0b24845f -0.0003.
tf-pass edge lengths (ungated): med 1.99 p95 4.24 p99 9.93 max 82.9; 1.46% >8.64um, 0.12% >14um.

## F5. Code facts needed to build it
- filter_output_graph L2995-3011 builds learned_edge_probs from the POST-ILP geff edges only.
  The full candidate list dies inside predict_video (repo L458-486). To expose it on Kaggle, dump at
  the EXISTING patch anchor `coords = coords.astype(np.int16) / return (coords, all_edges)`
  (cell L1392-1403, already hex-patched once for detector_coordinates_*.jsonl).
- motion_relink_edges L2275: `for pass_name, gate_um in (('tight',...),('relaxed',...))` is the
  insertion point for pass-0. Costs one extra sparse Hungarian per frame pair (n<=2600) — negligible.
- NO guard blocks it: _EXPECTED_NUMERIC/_EXPECTED_TEXT (L114-115) do not list OUTPUT_MOTION_RELINK
  or any MOTION_RELINK_* key. PP_SWEEP_KEYS (L3641) does list them, so densad clobbering (L3007-3012)
  must be respected — write any new knob into DENSITY_GROUP_OVERRIDES or a separate global.
- MOTION_RELINK_MAX_FRAME_NODES 2600 (L2213): if a hidden frame exceeds it the WHOLE relinker
  (incl. any pass-0) is skipped for that volume. Local max frame ~465.
