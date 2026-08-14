# Z1 Direct-Reference Fixed-Impedance Design

**Date:** 2026-08-12

**Status:** approved for implementation

**Scope:** a new straight-through-nail, joint-position fixed-impedance pair; the banked waypoint-guided pair remains immutable

## Objective

Train the Z1 hammer policy with the simplest approved task-space reference: a single spatial line from the reset hammer-head position through the nail. The policy still outputs six absolute joint-position targets. The reference is a weak, temporary reward scaffold, not an action generator, demonstration policy, clock-indexed trajectory, or policy input that replaces online RL.

This slice compares only:

- **FIC-0:** direct-reference fixed impedance without the joint-target trackability cost.
- **FIC-TT:** the identical treatment plus the existing joint-target trackability cost with exactly `k_tt=1`.

The prior waypoint-guided FIC-0/FIC-TT pilot, its task identifiers, checkpoints, evaluation JSONs, and result document remain unchanged and are not statistical comparators for this experiment.

## Reference geometry and reward

Reuse the tested `SingleStrikeReference` and `ImitationPriorTerm`; do not introduce a second reference implementation.

For each environment, let:

- `A` be the live world position of `hammer_head_site` at reset;
- `N` be the live world position of `nail_top` at reset;
- `B = N - [0, 0, 0.15] m` be the through-nail endpoint;
- `p*(phi) = (1-phi) A + phi B` be the finite reference segment.

The reference phase is spatial rather than clock indexed. It is obtained from the current hammer-head projection onto the finite segment, subject to the existing `axis_tol=0.05 m` and monotone phase latch. The reward-time path previews current phase without mutating the shared reference state.

Define

```text
d = ||p_head - p*(phi)||_2
r_ref = exp(-(d / 0.05 m)^2) * 1[before first accepted contact]
```

The `0.05 m` value is retained as the historically approved, deliberately permissive reference bandwidth. It makes the exponent dimensionless and sets the distance at which the raw reward falls to `exp(-1)`. It is not derived from the 0.15 m follow-through, hammer geometry, or a measured error distribution and must not be described as a calibrated physical constant.

The existing first-contact latch remains load bearing. Current contact or a touch-and-release within one control interval permanently sets the reference reward to zero for the rest of that episode. The 0.15 m endpoint therefore defines the incoming strike direction and scripted playback, but no post-contact pressing or follow-through receives reference credit.

The live reward key remains `r_imit` to reuse the existing manager term and tests. User-facing results call it the **direct-reference reward** and record its implementation identity.

## Reference curriculum

Retain the existing weak anneal, expressed in PPO iterations for the fixed 24-steps-per-iteration runner:

| PPO iteration | Environment step | `r_imit` weight |
|---:|---:|---:|
| 0 | 0 | 0.10 |
| 50 | 1200 | 0.08 |
| 100 | 2400 | 0.06 |
| 150 | 3600 | 0.04 |
| 200 | 4800 | 0.02 |
| 250 | 6000 | 0.00 |

The weight stays zero from iteration 250 through the planned iteration 500 endpoint. The bandwidth remains `0.05 m`; only the reward weight changes. This provides early exploration guidance followed by 250 iterations of free online optimization.

## Frozen treatment contract

Both new tasks preserve the currently qualified joint-FIC machinery:

- six absolute joint-position policy outputs with the banked affine scales, offsets, and clipping;
- fixed production joint gains and plant;
- fixed train and play reset ranges `(0.0, 0.0)`; disabling waypoint guidance must not silently restore reset randomization;
- the seven-term baseline reward;
- event-correct, event-linear impulse measurement;
- `delivered_impulse` weight `4.0` (D4);
- corrected controlled-drop normalizer `I_ref=0.2799950838088989 N s`;
- substep velocity CaT active with the existing 500 Hz peak reader, limits, probability, sign, and positive-return scaling;
- per-joint impulse CaT measured and logged only (`imp_max_p=0`), with the existing banked project-threshold vector and no active impulse pressure;
- diagnostic contact-row decomposition disabled at production scale;
- `strike_phase` and `strike_ref_error` observations retained because they own and expose the shared reference lifecycle;
- no variable impedance.

The new direct-reference treatment removes all waypoint-specific behavior:

- no `r_waypoint_progress` or `r_gate`;
- no waypoint tracker metric;
- no `next_gate_vector`, `completed_gate_fraction`, `guideline_perpendicular_error`, or `waypoint_progress_state` observations.

The resulting actor and critic observation width is exactly `40`, and the action width is exactly `6`.

FIC-TT differs from FIC-0 only by:

```text
cost_tt = k_tt * ||q_des_applied(t) - q_actual(t+1)||^2
k_tt = 1
RewardManager weight = -1
effective reward contribution = -||q_des_applied(t) - q_actual(t+1)||^2
```

The existing contact behavior is retained. `r_tt` remains in the negative-term partition and is never multiplied by the positive soft-CaT survival factor. The direct-reference reward is positive task return and is scaled through the existing CaT positive-return path.

## Additive task identity

Add immutable task registrations:

- `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-Fixed`
- `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-Fixed-TT`

Do not rename, repurpose, or remove the existing `...Guideline-CProgress...JointPosition-Fixed[-TT]` tasks. A checkpoint from the 47-observation waypoint task must fail closed rather than load as a 40-observation direct-reference policy.

## Implementation and verification

Implement test-first in bounded vertical slices:

1. Add failing configuration and registration tests for exact rewards, observations, metrics, reset, action, CaT, calibration, and FIC-0/FIC-TT isolation.
2. Add the smallest separate direct-reference joint-FIC configuration factory and registrations.
3. Extend essential joint-policy metadata so the observation width/names and direct-reference identity are explicit and incompatible checkpoints fail closed.
4. Add direct-reference support to the existing CPU/CUDA live smoke and genuine one-iteration CatPPO smoke without weakening the preserved waypoint assertions.
5. Retarget the compact FIC evaluator contract and create collision-proof reference-specific Slurm/run/result paths. Do not expand evaluator infrastructure.

Before GPU training, run:

- focused reference, joint-configuration, CatPPO/CaT, metadata, launcher, evaluator, and smoke tests;
- live CPU reset/step smokes for both tasks;
- genuine one-iteration CPU CatPPO smokes for both tasks;
- `validate_rewards.py` phases A-M, `verify_contact_sensor.py`, `verify_reward_setup.py`, and `playback_reference.py`;
- the full CPU test suite once;
- clean-tree CUDA live and genuine one-iteration CatPPO smokes on Vega.

Review the implementation with Opus, Gemini, DeepSeek, and the repository Standards/Spec review. Fix concrete correctness findings test-first and perform one bounded rereview before creating and pushing the clean training revision.

## Training and result gate

Use the clean reviewed code revision and pinned clean asset revision. Train the two arms at `4096` environments for `500` PPO iterations:

1. Run a matched seed-2 pair as the first full canary.
2. If both jobs finish with finite telemetry and preserve task/productive behavior, launch seeds 3 and 4 for both arms in parallel without changing code or parameters.

Evaluate each final checkpoint on the same compact 64-environment first-episode population. Bank only essential evidence:

- task and productive-strike success;
- first-event delivered impulse;
- peak joint velocity and legality;
- per-joint impulse and banked project-threshold utilization;
- direct-reference deviation;
- reference curriculum telemetry;
- one fixed-reset trajectory render/video per treatment;
- checkpoint, code, asset, population, and result hashes.

The result record must distinguish independent training seeds from evaluation environments. It may compare FIC-0 with FIC-TT on the intended joint-target tracking endpoint, but must not claim a Cartesian or waypoint causal comparison.

## Non-goals and stopping boundary

This slice does not add or run:

- Cartesian retraining or a Cartesian-vs-joint campaign;
- a waypoint-vs-reference statistical comparison;
- reference-weight or bandwidth sweeps;
- new reference/reward formulas, post-contact reference credit, curriculum variants, or domain randomization;
- active impulse-CaT pressure or altered registered-task impulse thresholds;
- variable impedance, gain actions, VIC training, or a VIC comparison;
- controlled-drop variants or recalibration;
- provenance/evaluator infrastructure beyond the compact existing contract.

After the direct-reference fixed-impedance results are reviewed, committed, and pushed, stop. The next separate design uses the selected fixed treatment as the unchanged basis for VIC so commanded impedance is the intended difference.
