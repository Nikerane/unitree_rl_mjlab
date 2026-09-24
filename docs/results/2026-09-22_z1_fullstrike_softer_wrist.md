# Full-strike softer-wrist intervention

## Material Passport

Mode: bounded simulation experiment; status: executed and descriptively analyzed. Date2026-09-22. Inputs: two seed2,150-update E1 checkpoints; sourcecc46d99. No new learning, production edits, or hardware experiment. This result is a runtime controller intervention, not unchanged-checkpoint qualification or evidence of optimal learned impedance.

## Treatment and controls

Four evaluations: each saved dose .2/.3 policy under its original controller and under fixed J6 Kp256,Kd25.6. Other five joints retain learned variable impedance; all position commands remain policy-controlled and may change in response to altered observations. Both wrist gains change together. Three external speed goals1.5/1.85/2.2m/s,128first episodes each, native2msphysics/20mscontrol, same seed2026092101. Runtime override is explicitly recorded in `intervention.json`; saved contract metadata still describes the original strict-loaded checkpoint, not the altered effective wrist controller. No loading guard was disabled.

Matched checks pass for checkpoint, original contract, reset identities, post-reset CPU/CUDA RNG and initial proposed action. Applied wrist gains are verified in every saved treatment trace row. We did not assume subsequent actions match: this is closed-loop performance evaluation rather than frozen-action causal attribution.

## Results

All128/128episodes hit in every condition; no timeouts. Softer-wrist totals:768hits, zero wrist-speed violations versus768original;75wrist-impulse-exceeding episodes versus766original. Original/soft values below use fresh matched controls, not historical medians.

| Dose | Goal | Delivered impulse median N·s, original → soft | Tracking median mm, original → soft | J6 impulse violations /128, original → soft |
|---|---:|---:|---:|---:|
| .2 |1.5|.30060 → .30733|10.763 → 10.921|128 → 9|
| .2 |1.85|.35318 → .36899|10.025 → 10.127|128 → 21|
| .2 |2.2|.36184 → .38439|10.068 → 10.204|127 → 27|
| .3 |1.5|.29454 → .30267|12.128 → 12.036|128 → 4|
| .3 |1.85|.31856 → .30582|11.241 → 11.301|128 → 0|
| .3 |2.2|.36347 → .38565|10.136 → 9.927|127 → 14|

Wrist maximum speed .2:4.60976→.47780rad/s at every goal; .3:4.58020→.63472/.61267/.59121rad/s. Soft J6 maximum impulse/cap ratios .2:1.386/1.877/1.949; .3:1.221/.998/1.257. Thus wrist impulse compliance is not universal.

Median absolute speed errors .2 original .19568/.08023/.07923 → soft .19746/.08779/.08446; .3 original .09687/.06088/.05270 → soft .08966/.05591/.05428m/s. Neither treatment passes the unchanged all-goal .07m/s E1 criterion. Speed selection is still external, not autonomous.

## Trade-offs and incomplete safety

- Dose .2 high-goal J2 impulse violations increase5→9/128 and maximum ratio1.092→1.126. J4 velocity counts at middle/high increase1→3each; high maximum3.317→3.636rad/s.
- Dose .3 high-goal J2 impulse violations increase3→9; maximum ratio1.043→1.046. Middle-goal J4 velocity changes0→1, maximum2.550→3.381rad/s. Other subthreshold tails also shift.
- Dose .3 middle delivered impulse decreases about4%; other condition medians increase about2–6%. This is mixed, not a universal impact improvement.
- Native impulse is the existing success-censored contact-window reaction metric with provisional caps. Trials from one checkpoint per dose are not independent training-seed replication. No manufacturer safety or constrained-optimality conclusion follows.
- This full-strike observer records native substep velocity peaks and control-rate reward/contact/impulse/gains, not full-strike actuator-force occupancy or within-step signed chatter. Startup force diagnosis remains separate evidence. Contact-timestep convergence has not been established.

## Direction review

The experiment supports investigating lower-gain calibration: substantial wrist-load reduction is not merely due to missing the nail or losing most delivered impulse. It does not support declaring the problem solved, widening C indiscriminately, or claiming learned-VIC benefit from fixing one joint's gains. A defensible next proposal is a small explicitly designed wrist gain envelope/calibration study, followed by contact-resolution checks and matched-seed training only after the controller choice is justified. Preserve position action semantics, actual-impulse objective and load constraints; do not add a separate trajectory controller as a response to this finding.

The [independent literature/direction review](../research/reward-design/2026-09-22_fullstrike_direction_literature.md) identifies Bogdanovic–Khadiv–Righetti for joint position plus impedance and fair fixed-gain comparison, Tan–Liu–Turk for discrete-time PD stability, and Martín-Martín et al. for contact-task action-space evaluation. None specifies safe Z1 gain limits. Research workflow influenced this assessment by separating numerical evidence, task performance and hardware claims.

## Execution evidence

Vega job45733813 completed; all evaluation summaries report zero learning updates. [Raw artifacts](assets/2026-09-22_fullstrike_wrist/). Scratch `fullstrike_wrist_probe.py`, `fullstrike_wrist.sbatch`, `analyze_fullwrist.py` under `.superpowers/sdd/2026-09-21-vega-speed-qualification/`. Analyzer rerun after transfers completed returned exit0 and passed both matching checks. An initial local analysis attempted before download completion failed on a missing summary file; it was a partial-transfer analysis, not an experiment failure.
