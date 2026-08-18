# Z1 impulse-CaT thesis campaign design

## Material Passport

- Origin skill: `academic-research-suite` experiment planning
- Origin mode: `plan`
- Origin date: 2026-08-18
- Verification status: `UNVERIFIED` — this document authorizes no run
- Version label: `z1_impulse_cat_campaign_design_v1`

**Status:** user-approved design direction on 2026-08-18. This document freezes the scientific
sequence and decision gates. It does **not** authorize implementation, a Vega submission, another
training run, a cap change, or an `imp_max_p` change.

## Research question

Can per-joint impulse soft-CaT reduce the learned Z1 policy's reaction-impulse exposure without
merely moving risk into true 500 Hz joint velocity or degrading the hammering task?

The mechanism is graded PPO credit and continuation pressure, not a runtime clamp. The target claim
is therefore a reproducible learned-policy effect under explicitly project-defined thresholds. No
stage supports a manufacturer damage limit, hardware-safety guarantee, or almost-sure cap claim.

## Evidence that fixes the starting point

The completed seed-2 diagnostic-`0.9`, `imp_max_p=0` versus `0.5` pair established:

- native observed diagnostic-cap risk `1.261% -> 0.057%`;
- native observed provisional-cap risk `0.814% -> 0.008%`;
- true 500 Hz velocity-limit risk `0.618% -> 1.058%`;
- 100% task success in both roles;
- impulse pressure independently won the max soft-OR whenever it was active;
- target median native episode duration was 100 ms versus 140 ms for control;
- 97--99% of physical contacts were terminally right-censored.

The preregistered verdict is therefore redistribution/trade-off, not clean enforcement. One PPO
training seed and three evaluation replicas do not establish training-seed robustness. The next
experiment must resolve complete-contact censoring before any new dose or training treatment.

Authoritative evidence:

- `docs/results/2026-08-15_z1_impulse_cat_step1.md`;
- `docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md`;
- `docs/research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

## Approaches considered

### A. Staged calibration, one bridge, then independent-seed confirmation — selected

First obtain complete-contact evidence without learning. Select at most one admissible dose from an
offline replay rule, screen it in one matched seed, and spend the main Vega budget only after it
passes impulse, velocity, and utility gates. This is the smallest route that can distinguish dose
calibration from PPO training variability.

### B. Immediate three-arm multi-seed ladder

Training `imp_max_p in {0, p_cal, 0.5}` across several seeds would show a cleaner response curve,
but it repeats a high-dose treatment already associated with higher velocity risk and chooses a new
dose before complete-event exposure is known. Rejected as premature.

### C. Boundary-by-dose factorial

Crossing provisional and diagnostic thresholds with several doses could estimate an interaction,
but neither vector is a hardware limit and the immediate uncertainty is complete-event dose. This
is deferred unless the staged campaign exposes a genuine boundary interaction.

## Global invariants

Every stage preserves the existing VIC-TT task, policy architecture, reference observations,
trajectory, rewards, physics, 2 ms substep, 25-substep sliding Lambda window, baseline subtraction,
contact mask, velocity CaT, domain randomization, curriculum, action space, and gain range.

- Provisional project caps are `[0.820, 1.640, 0.820, 0.820, 0.820, 0.820] N.m.s`.
- Diagnostic-only caps are `[0.738, 1.476, 0.738, 0.738, 0.738, 0.738] N.m.s`.
- Provisional caps are always analyzed first; diagnostic caps are secondary engineering evidence.
- Caps must not be lowered further to manufacture activation.
- Live frozen evaluation uses `imp_max_p=0`; counterfactual doses are replayed offline.
- Controller reads and overlapping windows are never inferential units.
- Failed jobs are never retried or requeued automatically.
- Torque CaT, reward changes, stronger velocity treatment, and trajectory changes are out of scope.

## Stage 0: independent design challenge

Before implementation, send the sanitized packet
`docs/research/reward-design/Z1_IMPULSE_CAT_CAMPAIGN_CROSS_MODEL_PACKET.md` independently to each
available external model. Requested reviewers are Kimi 3, GLM 5.2, DeepSeek V4 Pro, and Qwen; these
labels and endpoints must be verified at execution time. A model is recorded as unavailable rather
than simulated when no authenticated provider exists.

Only the packet text may leave the workspace. Do not upload raw traces, checkpoint binaries, source
code, private notes, or the thesis repository. Reviews are advisory falsification hypotheses, not
votes and not evidence. Provider, exact model ID, date, prompt hash, and response hash must be
recorded before a critique influences this design.

## Stage 1: no-learning release/window-flush shadow

Implement the already planned shadow in
`docs/superpowers/plans/2026-08-17-z1-impulse-diag90-release-window-flush-shadow.md`. Reuse
`scripts/impulse_cat_activation_survey.py`; do not create another environment or evaluator.

### Frozen roles

| Role | Training dose | Checkpoint SHA-256 |
|---|---:|---|
| control | `0.0` | `f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3` |
| high-dose anchor | `0.5` | `ddd7ac4c855160bff1db2af52e642d960dab2bef34e41dd2532002185eb36d15` |

Run the 64-world deterministic population for parity only and the three already preregistered
4,096-world stochastic populations (`2`, `2026081701`, `2026081702`) for event evidence. For
each world, preserve the native trajectory through success exactly, remove only `nail_driven`
termination in the shadow, then continue until both conditions hold:

1. at least five consecutive off-contact 2 ms samples;
2. the current production 25-substep rolling Lambda is zero for all six joints.

Stop unresolved worlds 250 ms after native success and retain them as right-censored. The
post-success trajectory is a counterfactual measurement continuation, never a native task outcome.

### Stage-1 pass gates

- exact native-prefix parity for actions, observations, gains, rewards, contact, Lambda, and RNG;
- termination sets differ by exactly `{"nail_driven"}`;
- finite, shape-correct data and exact checkpoint/code/asset/manifest identities;
- at least 95% completion among control physical events associated with provisional-cap activation
  windows;
- at least 50 completed, unambiguous, provisional-cap-activating control events.

If the target has fewer than 50 activating events, report it as sparse/nonbinding target evidence;
do not pool arms. Dose calibration is based on untreated-control events. If the control gates fail,
stop: event dose remains uncalibrated, no dose is selected, and no new training starts.

## Stage 2: offline event-dose rule

On the same completed, unambiguous control events, replay
`imp_max_p in {0.05, 0.10, 0.20, 0.30, 0.50}`. For each candidate report chronological

`P_event = 1 - product_t(1 - delta_impulse_t)`,

separate impulse/velocity deltas, winner/masking shares, read count, responsible joint, and
per-joint utilization. Censored and ambiguous events remain separate.

A candidate is admissible only when:

- among reads where the replayed `delta_impulse > 0`, impulse strictly wins the max soft-OR at
  least 90% of the time (ties are not wins);
- completed-unambiguous control-event median `P_event >= 0.05`;
- completed-unambiguous control-event p95 `P_event <= 0.50`;
- fewer than 1% of completed-unambiguous events have `P_event >= 0.80`.

Select the smallest admissible candidate and name it `p_cal`. These are operational learning-dose
criteria, not physical safety criteria. `0.20` remains an analytical starting point, not the default
answer. If no candidate passes, stop and return the calibration question to the supervisor; do not
interpolate, tighten caps, or increase dose automatically.

## Stage 3: one-seed bridge

Only after Stages 1--2 pass, train one new 500-iteration seed-2 target at the provisional project
caps and `imp_max_p=p_cal`. Use 4,096 environments, 24 rollout steps, the existing VIC-TT task, and
the existing matched launcher pattern. The banked seed-2 `p=0` checkpoint is the control only after
tests reconfirm that changing a log-only cap cannot alter actions, rewards, PPO data, or policy
updates. The seed-2 diagnostic `p=0.5` policy remains a historical high-dose anchor, not a third
scientific treatment.

Evaluate native and shadow outcomes on the fixed population and all three stochastic populations.
This bridge is a screen, not an independent-seed claim.

### Bridge go/no-go gates

- provisional-cap risk reduction: upper paired one-sided 97.5% bound for target-minus-control risk
  difference
  is below `-0.50` percentage points;
- impulse tail: per-joint `rho` p95 and p99 decrease by at least 10% relative to control with no
  compensating relative p99 increase of 10% or more at another joint;
- true velocity: upper paired one-sided 97.5% bound for target-minus-control
  violation-risk difference is at
  most `+0.10` percentage points;
- success and productive-strike differences are no worse than `-1` percentage point;
- lower one-sided 97.5% bound for the first-event delivered-impulse target/control ratio is at
  least `0.90`;
- native and shadow claim boundaries remain consistent and complete-contact gates still pass.

Failure stops the campaign. Do not try the next dose automatically and do not raise `imp_max_p`.

## Stage 4: eight-seed matched confirmation

After the bridge passes, run the supervisor-approved comparison at training seeds
`{2, 3, 4, 5, 6, 7, 8, 9}`:

- control: velocity CaT active, impulse CaT log-only (`imp_max_p=0`);
- target: velocity CaT active, impulse CaT active (`imp_max_p=p_cal`);
- both: provisional project caps and otherwise identical training identity.

The seed-2 bridge/control may count only when their exact comparability and hashes pass. Otherwise
retrain seed 2 as a fresh pair. No arm may be selected or dropped after seeing outcomes.

Primary checkpoint is `model_499`. Evaluate `model_400`, `model_450`, and `model_499` without
selecting the best. Each policy/checkpoint receives the fixed 64-world regression population and
the same three paired 4,096-world stochastic populations. Run the complete-contact shadow on final
checkpoints; earlier checkpoints use native evaluation unless a preregistered anomaly requires the
same shadow for every arm.

## Statistical contract

Define `rho_L = max_t,j Lambda[t,j] / L[j]`. Primary impulse risk is the fraction of initial native
episodes with `rho_provisional > 1`. Primary true-velocity risk is the fraction of initial native
episodes with any 500 Hz joint-velocity-limit violation. Complete-event shadow outcomes calibrate
dose and diagnose mechanism; they do not replace these native-horizon primary outcomes.

The training seed is the primary unit for a learned-policy claim. Within each seed/evaluation
replica, bootstrap whole matched environment IDs. Across training seeds, resample paired seed blocks
with environments nested inside their seed. Evaluation replicas are reported separately and do not
become training replicates.

The two co-primary outcomes use one-sided 97.5% bounds:

1. provisional-cap violation-risk difference: upper bound `< -0.50` percentage points;
2. true 500 Hz velocity violation-risk difference: upper bound `< +0.10` percentage points.

Utility gates are conjunctive: success and productive-strike lower-bound differences at least
`-1` percentage point and delivered-impulse ratio lower bound at least `0.90`. At least six of
eight seed pairs must individually show provisional impulse risk difference `<= -0.50` percentage
points and nonpositive velocity-risk difference. `model_499` must pass, and at least two of the
three checkpoints must preserve the same impulse/velocity/utility directions.

Secondary descriptive outcomes are diagnostic-cap risk, Lambda/utilization p50/p95/p99/max,
positive margin, responsible joint, complete-event pressure, depth, cumulative delivered impulse,
duration, VIC gains, and actions. Do not run p-values over PPO iterations, controller reads,
overlapping windows, or pooled evaluation replicas.

## Verdicts

1. **Clean learned reduction under project caps:** all co-primary, utility, seed-consistency, and
   checkpoint gates pass. This remains simulation evidence, not hardware safety.
2. **Redistribution/trade-off:** impulse improves but velocity, another joint, or utility fails.
3. **Active but ineffective:** impulse pressure wins, yet the impulse tail does not improve.
4. **Empirically nonbinding:** provisional events are absent or too rare under the surveyed
   population; report the population and upper uncertainty bound.
5. **Uncalibrated:** complete-contact or event-count gates fail; no dose conclusion.

## Compute envelope

Historical 4,096-environment, 500-iteration VIC training takes approximately 23 minutes per A100
arm; a frozen evaluator leaf takes about two minutes.

- shadow: two role leaves, each bundling the fixed and three stochastic populations, roughly
  0.2 A100-hour;
- bridge: one new training arm plus evaluation, roughly 0.5 A100-hour;
- eight-seed confirmation: at most 15 new training arms when the seed-2 control is reusable,
  roughly 5.9 A100-hours;
- three-checkpoint frozen evaluation: up to 48 leaves, roughly 1.6 A100-hours.

The full successful route is approximately 65 scheduled leaves and 8 A100-GPU-hours, excluding
queue time. Every stage is a separate authorization and may terminate the campaign early.

## Authorization boundaries

Approval of this design permits only a detailed implementation plan. It does not authorize code
changes, local live simulation, remote writes, `sbatch --test-only`, a real submission, training,
automatic continuation to another stage, or cross-provider upload. Each stage requires reviewed
code, exact commands, estimated compute, and explicit user approval.
