# FEIFANQUE TRACKLET/ENDPOINT REPAIR — MEASURED DEAD 2026-09-20 (zero GPU, zero slots)

The 09-20 census named these the ★ TOP PORT CANDIDATE and "the only genuinely new
mechanism found" (only mechanism class absent from our base, post-proc only, aimed at swaps).
No ranked idea I01-I16 covered them. Measured tonight; both are structural no-ops.

## Instruments
scratchpad/extract_feif.py -> feif_tracklet.py (5890 B) + feif_endpoint.py (6116 B),
ast.literal_eval'd verbatim out of census/feifanque_*.txt (_tracklet_source / _ep_source).
scratchpad/feif_sim.py runs them on ~/biohub_replay947_runs/ds_ctl/submission.csv,
max_distance = each volume's densad RELAXED_UM (9.0 mid / 11.0 low), min_gain 1.0, ambiguity 0.5.
FIDELITY: edge_prob unknown offline -> set 0.0, which MAXIMISES gain (gain -= prob) => UPPER BOUND
on firing. motion_relinked=1 for all edges (true for ~99%: 20143/20326 on 0b24845f).

## Result: 0 interventions of 20,931 candidates, 6/6 volumes
tracklet swap : 13,836 candidates -> tracklet_supported 0, tracklet_swaps 0, changed edges 0
endpoint move :  7,095 candidates -> endpoint_eligible 0, accepted 0 (ALL rejected_motion)
per-vol tracklet cand: 1884/4127/3589/282/3090/864 ; endpoint cand: 1372/1478/2618/170/652/805

## No-op guard (feif_dose.py, 6bba_7af54fde): the harness is live, the mechanism is not
min_gain 1.0 / 0.5 / 0.25 / 0.1 / 0.01  -> supported 0, accepted 0 at EVERY value.
Two-sided direction tests (0.5 um) and the 0.25 um no-worse test DISABLED -> still 0 swaps.
=> the binding constraint is the gain itself, not any threshold.

## Why (feif_gain.py): the shipped relink is already a strict minimum of their own objective
6bba_7af54fde n=3090: gain max -1.490, p99 -8.127, median -16.323 ; gain>0 in 0 pairs.
44b6_87bba6c4 n=3589: gain max -1.865, p99 -6.222, median -12.348 ; gain>0 in 0 pairs.
fwd>=0.5 AND bwd>=0.5 satisfied in 0 of 6,679 pairs on the two hardest volumes.
The BEST available swap anywhere would WORSEN the two-sided velocity residual by 1.49 um.

## Reading
These modules score by residual against a 2-step measured velocity. That is the same signal
T1-r30 (velocity weight 0/65 swaps), T1-r60 (collective motion 1.3% vs 0.5%) and today's anatomy
(CV picks the correct target 0/47; wrong link shorter in 48/49 swaps) already priced dead.
The census's hope -- "NOT pure geometry, they use measured velocity, which is what makes the
longer link correct for a fast cell" -- is refuted: our links are already motion-consistent
(anatomy947c: "the prediction's own link is short and motion-consistent, so a linker cannot undo it").
The author shipped both DISABLED with a RuntimeError('NO INTERVENTION') guard; that guard fires.
=> DEAD. Do not spend the local cache-hit hour or the slot. Public lane confirmed dry.
