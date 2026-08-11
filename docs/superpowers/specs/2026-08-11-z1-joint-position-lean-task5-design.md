# Z1 joint-position fixed-impedance: lean Task-5 design

**Status:** Approved by the owner on 2026-08-11.

## Objective

Finish the joint-position fixed-impedance infrastructure, verify that it works on
the real CPU/CUDA training path, and move quickly to preliminary experiments. The
work must implement the supervisor's requested mechanisms without building a
final-campaign provenance or evaluation system before initial results exist.

## Current state

- Tasks 1--3 are complete: the causal target tape is banked, direct six-joint
  replay passes, and the production default-offset `JointPositionActionCfg` is
  registered while the Cartesian tasks remain intact.
- Task 4's one-step command-tracking formula, negative reward sign, and exclusion
  from CaT's scaled positive return are implemented correctly.
- Task 4 is not yet accepted because runtime still loads the rejected Q90-derived
  `k_tt=1.9360715` calibration.
- Task 5 is absent: joint-policy metadata dispatch and the dedicated live smoke do
  not exist.

## Scope

### 1. Align Task 4 with the final owner decision

Keep the effective command-tracking contribution

```text
Delta R_tt(t) = -k_tt * ||q_des(t) - q(t+1)||^2
```

with reward weight `-1.0`, active through contact and classified as a negative term
outside CaT's positive-return scaling. Set `k_tt=1.0` explicitly for FIC-TT.

Remove only the RTT-specific Q90 machinery:

- delete the Q90 calibration executable and generated trackability artifact;
- remove the trackability-artifact loader and runtime dependency;
- remove Q90-specific tests and plan/spec requirements;
- retain formula, timing, six-joint indexing, sign, finite-value, and CaT-split
  tests.

Historical Q90 trajectory metrics unrelated to RTT remain untouched.

### 2. Implement a lean Task 5

Add joint-position metadata dispatch without changing the existing Cartesian
metadata. Essential joint metadata is:

- action type and action term;
- six-action dimension, target names, target IDs, and actuator names;
- default-offset semantics, per-joint scales, physical clips, and raw policy clip;
- physics step and control decimation;
- fixed gain/effort signature;
- joint-action qualification payload hash;
- whether `r_tt` is enabled and its numeric `k_tt` (`false`/not applicable for
  FIC-0, `true` and `1.0` for FIC-TT).

Do not retain a trackability-artifact hash or add elaborate ONNX/provenance
fingerprints for the rejected calibration.

Add one importable live smoke covering both FIC registrations. It must verify:

- finite reset and stepping with six actions and 47 observations;
- exact joint order and default-offset target semantics;
- unchanged fixed gains and efforts, with no gain action;
- retained P task-space guidance, D4 delivered reward, and velocity CaT;
- measured but log-only impulse CaT;
- `r_tt` absent in FIC-0 and active with `k_tt=1`, negative weight, and the correct
  CaT split in FIC-TT;
- successful metadata construction from live values.

Existing construction tests already cover detailed affine mapping, clipping,
reset, current-state independence, and gripper isolation. The smoke must not
duplicate that entire suite or replay all 16 deterministic qualification rows.

### 3. Verification boundary

Before training:

1. Run the focused joint-action/config/RTT/CaT/smoke tests.
2. Run one live CPU smoke for each FIC arm.
3. Run one genuine one-iteration CatPPO training smoke for each FIC arm.
4. Run the full CPU test suite once on the final Task-5 revision.
5. Run one live CUDA smoke on the clean deployed revision.
6. Obtain independent reviews from Opus, Gemini, and DeepSeek and resolve concrete
   correctness findings.

Do not make repeated full-suite runs, duplicate already-covered assertions, or
build the original Tasks 6--9 evaluator/manifest system before preliminary results.

## Preliminary experiment sequence

After Task 5 passes:

1. **Controlled drop first.** Run five identical centered releases of the
   `0.200 kg` flat-faced cylinder from rest at the single live `h0` height, with
   unchanged nail/block/contact physics. Use the arithmetic mean of the five
   production first-event nail-axis impulses as the new `I_ref`. No height, mass,
   orientation, or geometry sweep is part of the primary calibration.
2. **FIC integration result.** Train FIC-0 and FIC-TT (`k_tt=1`) at seed 2 with
   the same 200-iteration, 4,096-environment budget and calibrated `I_ref`. Evaluate
   the same small 64-episode population for both. Report what happens; FIC-TT is
   not required to outperform FIC-0.
3. **Active impulse-CaT diagnostic.** Use the same common diagnostic cap vector in
   both arms:

   ```text
   0.3 * [1.640, 3.280, 1.640, 1.640, 1.640, 1.640]
     = [0.492, 0.984, 0.492, 0.492, 0.492, 0.492] N m s
   ```

   Compare `imp_max_p=0` versus fixed `imp_max_p=0.25` at seed 2 with the same
   200-iteration, 4,096-environment budget. Keep RTT, velocity CaT, curriculum, and
   domain randomization out of this impulse-isolation comparison. Report cap
   crossings, per-joint reaction-impulse
   tails, task success, and first-event nail impulse. If the lowered cap is
   non-binding, report that result and stop before inventing an automatic threshold
   sweep.

Only after these results should the project decide whether additional seeds,
curriculum/domain randomization, a changed diagnostic threshold, or VIC work is
justified. VIC remains last.

## Failure handling

- Infrastructure or smoke failure: fix the specific failed contract before GPU
  training; do not tune task parameters.
- FIC pilot failure: classify the observed failure before changing action range,
  rewards, gains, or resets, and change only one cause at a time.
- Active-CaT performance reduction or no improvement: preserve and report it. No
  preferred directional result is required.
- Missing cap crossings: classify the diagnostic as non-binding. Do not claim CaT
  was tested and do not silently lower the threshold.
- Non-finite state, broken metadata, configuration drift, or CaT wiring mismatch:
  stop the affected run.

## Explicit deferrals

- Q90-based RTT calibration or RTT coefficient sweeps;
- multiple drop heights, masses, shapes, or orientations;
- ballistic-impulse estimation;
- multi-seed expansion and 512-episode final-campaign evaluation;
- new checkpoint-manifest, trace-schema, plotting, or ONNX provenance fortresses;
- curriculum/domain randomization before the baseline and active-CaT result;
- gain changes, hardware-oriented gain selection, or variable impedance;
- new jitter penalties without an observed failure.

## Acceptance

This stage is complete when the Q90 dependency is gone, FIC-TT uses exactly
`k_tt=1`, both joint tasks expose correct live metadata, focused and full CPU tests
pass, both one-iteration CatPPO smokes pass, and the clean CUDA smoke passes. At
that point implementation discussion stops and the controlled-drop experiment
begins.
