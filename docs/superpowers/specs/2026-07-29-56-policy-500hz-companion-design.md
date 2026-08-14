# 56-policy fixed-reset 500 Hz companion pass — design

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer cap”
> wording below refers only to the historical registered-task boundary, not a validated
> Z1 reaction-impulse or damage limit. The design body remains frozen; see
> `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

**Date:** 2026-07-29  
**Status:** owner-approved design  
**Purpose:** obtain the exact substep evidence that the completed 50 Hz visual
library cannot provide, then freeze a defensible four-policy tracking-pilot
shortlist.

## Scope and decisions

Run all 56 accepted fixed-impedance policies once from the same already
approved reset used by the video library. Preserve the existing checkpoint
inventory, manufacturer caps, action space, gains, task rewards, and
`imp_max_p=0`. This is evaluation-only; it neither trains nor enforces.

Do not create a new evaluation framework. Reuse the existing 500 Hz substep
hook in `scripts/diag_impulse_trace.py` and add only one input:
`--fixed-reset-envelope`. Restore the realized reset with the existing
`eval_impulse.restore_reset_state` helper and refresh observations before the
policy acts.

## Inputs

- The tracked 56-row `checkpoint_inventory.tsv` and SHA sidecar.
- The same 56 local `model_499.pt` files already hash-verified by the video
  library.
- The approved reset-state digest:
  `bde511ec2adc42e5365e1e46f45ff1fb43223a93c31c6fbfb4580352e445e319`.
- The existing fixed task identity for each campaign/arm.

The historical FQ3×8 cleanliness limitation remains explicit; no missing
`clean_state` evidence is invented.

## Recorded evidence

For every 2 ms physics substep, retain only the compact arrays required for
analysis, preserving the evaluator's established phase contract rather than
pretending every simulator channel is sampled at the same instant:

- pre-integration-derived hammer-head position, live contact and contact force,
  paired with the joint velocity and nail depth cached immediately before
  integration;
- post-integration joint position/velocity and nail depth, used for the
  continuous hardware-speed legality screen;
- the shipped running per-joint **50 ms windowed constraint read**
  `Lambda_j = max(control-step latch, current rolling sum)` after its substep
  update;
- first-strike quality validity/overflow, centroid error and axiality;
- policy action, phase/reference position and reference error recorded once per
  20 ms control step, linked to substeps by control-step index rather than
  mislabeled as 500 Hz measurements.

Each policy record binds campaign/arm/seed, checkpoint SHA-256, reset digest,
task identity, rollout code/asset revisions, timing, and payload SHA-256.

## Derived analysis

At the exact accepted first-contact onset defined by the shipped tracker,
compute:

1. apex-to-onset 3-D path length/direct-distance ratio;
2. RMS and maximum lateral deviation from the direct descent chord;
3. terminal-funnel contraction and late re-expansion;
4. onset axial/lateral velocity and approach angle;
5. quality validity, centroid error and axiality;
6. productivity, first-window success and event-window depth;
7. full-horizon maximum joint velocity;
8. maximum and time history of the shipped windowed
   `Lambda_j / IMP_J_LIMIT_j` constraint read;
9. impact-versus-press/release timing diagnostics.

The final quartet is selected only inside the confirmed FQ incumbent arm.
First split eligible policies into lower- and higher-curvature halves, then
choose two matched pairs using the already frozen population covariates:
quality, useful speed, depth and first-window success. The selection algorithm
must be deterministic and must permit an honest `NO QUARTET` outcome.

## Eligibility and interpretation

A policy can enter the formal quartet only if the fixed-reset episode:

- has a valid accepted first contact;
- is productive and first-window successful;
- has no quality overflow;
- stays within the 3.1415 rad/s manufacturer joint-speed rail at every
  substep.

If fewer than four FQ policies pass, report `NO HARDWARE-LEGAL QUARTET`.
Still bank the simulation-only curvature ranking, clearly labelled as such.
Do not lower caps, reinterpret the speed rail, or replace a failed seed.

Trajectory resemblance does not prove that the actor uses the reference.
Reference-use claims require a later coherent same-state intervention; this
pass only measures geometry and actor-visible signals.

## Outputs

Write to a new dated result-asset directory:

- one compact NPZ/JSON record per policy;
- a 56-row summary CSV;
- FQ eligibility and deterministic-pairing table;
- x-z/x-y trajectory grids with the dashed observation-only reference;
- qvel and `Lambda_j/cap_j` diagnostic plots;
- a dated result note separating proven, descriptive and open conclusions.

No new videos are required: the completed fixed-reset MP4 library remains the
visual companion.

## Failure handling

Fail closed on any checkpoint/reset/task/hash mismatch, nonfinite sample,
missing substep field, wrong timing, or partial output. Resume only a complete,
hash-valid policy record. A failed policy remains in the denominator and is
reported; it is never silently skipped or replaced.

## Testing and execution

Use focused TDD for:

- exact reset restoration followed by observation refresh;
- required substep schema and 2 ms timing;
- deterministic eligibility and pairing, including `NO QUARTET`;
- qvel and `Lambda_j/cap_j` calculations;
- corrupt/partial record rejection.

Run one CPU smoke policy first. If it reproduces contact and completes without
schema failure, run all 56 sequentially on CPU. Expected simulation runtime is
approximately 9–12 minutes; implementation, tests, analysis and review should
fit within 45–75 minutes.
