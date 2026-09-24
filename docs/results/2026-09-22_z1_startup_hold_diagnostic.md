# Startup position-hold diagnostic

2026-09-22. Evaluation only; no training, production source edits, permanent action filtering, or threshold changes.

## Result

Holding the initial joint positions for one 20-ms control decision removes the J6 velocity spike during that decision, but reproduces the same spike at the next decision when policy motion starts. This is **not a fix** and does not improve J6 velocity violation frequency.

| Saved policy | Normal first-step J6 peak | Held first-step peak | Hold-run later peak | Peak step normal → hold |
|---|---:|---:|---:|---|
| impulse dose .2 | 4.609762 rad/s | <2.9e-8 rad/s | 4.609761 rad/s | 0 → 1 |
| impulse dose .3 | 4.580195 rad/s | <2.9e-8 rad/s | 4.580200 rad/s | 0 → 1 |

Each row covers all 128 first episodes at each of the three requested speeds, 1.5/1.85/2.2 m/s. All 384 episodes per treatment and policy exceed 3.1415 rad/s. Neither the original nor shifted peak occurs after accepted contact. All 1536 evaluated first episodes hit the nail. Contact impulse safety remains unresolved.

The matching checks pass: checkpoint hash, full contract, reset hashes, post-reset CPU/CUDA RNG hashes, proposed first action, actual initial arm positions/velocities, and applied first-step stiffness/damping. In the hold treatment only the first six position coordinates are replaced by `(current_q-offset)/scale`, verified within the clip range and verified against the processed target. Remaining action coordinates are identical. Later policy actions operate normally on the resulting state; they are not forcibly replayed.

## What the physics-step trace adds

Reset arm velocities are zero. J6 starts at 1.554400 rad. The normal first target is 1.499235 rad for dose .2 and 1.609565 rad for dose .3: opposite saturated setpoint changes of about .05516 rad. Kp6 is 800 in both treatments/policies.

World-zero signed velocities at successive 2-ms physics samples begin:

- dose .2 normal: −4.60976, +.72364, −4.01855, +1.22760, −3.59059, +1.58612 rad/s.
- dose .3 normal: +4.58020, −.77757, +3.94527, −1.30708, +3.50504, −1.67765 rad/s.
- hold: approximately zero for ten physics samples, then the corresponding alternating response starts.

This shows a movement-command-dependent, oscillatory startup response under the current actuator/gain/integration setup. It weakens the explanation of a reset-only kick occurring without commanded motion. It does **not** isolate wrist-only causality, numerical integration versus actuator dynamics, or the particular joint commands responsible: all six position commands were held together. No torque-channel attribution was collected in this probe. Do not infer an integrator bug or change gains from this evidence alone.

The normal controls reproduce the previous reported J6 maxima and episode counts. Whole raw traces are not bitwise identical to the older run: small floating-point differences exist. The new matched normal/hold comparisons pass their explicit identity checks; no exact full-trace replay claim is made.

## Task outcomes (median achieved speed / delivered impulse)

Goals ordered 1.5, 1.85, 2.2 m/s:

- .2 normal: speeds 1.6906/1.9222/2.2698; impulses .30061/.35317/.36182 N·s.
- .2 hold: speeds 1.6840/1.9262/2.2585; impulses .30380/.34824/.35802 N·s.
- .3 normal: speeds 1.5969/1.9109/2.1807; impulses .29456/.31856/.36347 N·s.
- .3 hold: speeds 1.5922/1.9206/2.1660; impulses .29662/.31873/.36339 N·s.

These are descriptive consequences of the intervention, not improvements or qualification.

## Provenance and verification

Vega job **45706313**, COMPLETED, exit 0, elapsed 1m29s. Source pinned at `cc46d9998a3c9cd5d8813e89b57e44248353e03e`; strict native checkpoint loading and metadata checks retained. All four evaluations report zero learning updates. Earlier attempt 45706024 was cancelled after discovering the evaluator requires three goals; 45706114 failed in the diagnostic observer on an attribute-name mismatch before yielding complete results. The observer was corrected to the installed `_processed_actions` API, not production code.

Artifacts: [raw evaluation and startup traces](assets/2026-09-22_startup_probe/). Scratch harness and analyzer: `.superpowers/sdd/2026-09-21-vega-speed-qualification/startup_probe.py`, `startup_probe.sbatch`, `analyze_startup.py`. Analyzer completed exit 0 and validated both matched pairs. No active jobs remained at the completion check.

## Next scoped diagnosis

Do not add a warmup delay as a remedy. Inspect actual applied actuator effort, damping, target application and integration timing for the alternating response, then design a one-variable test separating joint-command magnitude/coupling from actuator/integration behavior. Keep training paused and safety limits unchanged. Any lasting command shaping, gain change or timestep change is a separate treatment, not silently adopted from this diagnostic.
