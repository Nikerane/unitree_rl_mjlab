# Free-time spline following, selected impact speed, and joint impedance

Status: proposed design, informed by three independent agents and a cross-review discussion. Implementation stops after software and no-learning integration tests; training is a separate checkpoint.

## Research question and agreed intent

“Can a learned RL policy generate strong impacts while keeping the impact loads transmitted to individual robot joints within prescribed safety limits?”

The user supplies a geometric curve from the reset hammer position to the nail. One online policy chooses six joint-position targets, six joint-stiffness coordinates, and one impact-speed goal per episode. Damping remains coupled to stiffness using the existing Khadiv–Bogdanovic mapping. The policy determines acceleration, deceleration, and arrival time through its ongoing actions. No duration, clock-indexed reference, Cartesian impedance, second learned controller, or trajectory generator is introduced.

The supplied left/centre/right quintics are the initial surrogate for a future set of demonstrated human paths. Human-data import, arbitrary splines, and geometric generalization are outside this implementation milestone.

Success at this milestone means verified controller authority, reproducible configuration, and measurement tools that can answer the research question. Passing software tests does not demonstrate learned path following, useful impedance, commanded-speed compliance, or joint-load containment.

## Source and ownership

The experimental source is `/Users/nikerane/.codex/worktrees/z1-impact-speed/unitree_rl_mjlab`, at base `cc46d9998a3c9cd5d8813e89b57e44248353e03e` plus its working changes. This task's current checkout, `/Users/nikerane/.codex/worktrees/a370/unitree_rl_mjlab`, is older (`1425b15`) and contains these planning documents only.

Keep the experimental source read-only. Implementation begins by creating an isolated checkout of the experimental base and transferring the tracked diff and required untracked files with a manifest. In particular, `src/tasks/hammer/mdp/sampled_pd500.py` and `tests/test_sampled_pd500.py` are untracked and essential. Never silently omit source, tests, configuration, or referenced assets. Record excluded bulky evidence instead of treating an incomplete transfer as the complete experimental snapshot.

Jobs `45860324` and `45860325` remain owned by the original task; do not query, modify, cancel, duplicate, or take over monitoring them.

## Approaches considered

| Approach | Benefit | Cost / decision |
|---|---|---|
| Extend the existing direct policy and sampled controller | Preserves the supplied geometry, free timing, impedance action, and speed latch; smallest attributable change | **Selected.** Qualify applied gains and improve evaluation. |
| Import CNP3O's terminal-derivative and timing generator | Explicit velocity-consistent planned reference | Produces a planned timed motion and changes how motor commands are generated. Not required for the user's free-time direct-control formulation. |
| Replace guidance with a hard path tube or a new path controller | Stronger geometric restrictions | Adds a new control/reward mechanism before establishing existing guidance quality. Defer until measured failures justify it. |

CNP3O remains a useful precedent for connecting a speed choice to motion. Here the connection is the existing goal observation, event-based speed-consistency cost, and direct motor policy, verified by paired intervention. No third-party code needs copying for this milestone.

## Binding design

### 1. Keep the existing policy and speed semantics

- Action order stays `[q_target(6), stiffness_p(6), speed_selector(1)]`.
- Observations stay at the existing 42-feature schema, including phase, reference error, normalized latched speed, and pending flag.
- Initial speed bounds are `[1.5, 2.2] m/s`; this is a candidate command range, not a feasibility certificate.
- Speed is latched once at the first policy decision. Later outputs cannot revise it. The first motor command cannot condition on the newly sampled speed; conditioning starts at the next 50 Hz decision.
- Preserve masked PPO likelihood/entropy/KL, fixed-tail normalization, event speed cost, and explicit external-goal-to-policy-goal transfer.
- A constant selected speed may be a valid solution. Diversity across speeds or curves is not a success condition. Ignoring controlled goal interventions is evidence against the command-authority claim.

### 2. Give sampled joint impedance actual authority

Maintain 4 kHz physics (`dt=0.00025 s`), 50 Hz policy (`decimation=80`), and 500 Hz sampled torque (one update every 8 physics steps).

Keep `fixed` as the backward-compatible sampled-controller mode. Add explicit `variable` mode. For variable mode, map the current bounded action coordinate using the calibrated sampled nominal gains:

```text
Kp0 = [256, 384, 384, 256, 192, 128]
Kd0 = [6.4, 6.4, 6.4, 6.4, 6.4, 6.4]
p in [-1, 1], C = 1.25
Kp = Kp0 * C**p
Kd = Kd0 * sqrt(C**p)
```

Reuse the existing validated mapping and tensor-only helper. Native actuator nominal gains differ from these calibrated values: do not read native `_kp/_kd` and call that sampled VIC. On each scheduled torque update, use the current stiffness coordinate and joint target, snapshot the applied gains, compute PD torque, and clamp it to existing effort limits `[30, 60, 30, 30, 30, 30] N m`. Hold the result between updates and clear native feedback before every physics integration. Preserve gripper behavior.

Expose requested stiffness coordinates, applied `Kp/Kd`, requested/held torque, and saturation separately. Metadata validates the mapping and nominal parameters, not equality of live variable gains to the nominal vector. Full resets restart the sample clock; subset autoresets must receive fresh gains/targets at the next normal sample. Test the real ordering before adding any extra reset-time torque update.

### 3. Preserve the current treatment while enabling path comparisons

The existing pilot is LEFT-start (`-20 mm`), not centre-start. Keep it as the first comparison. It already has persistent precontact `r_imit` weight `0.2`, default width `0.05 m`, and no anneal-to-zero curriculum. The existing unsaturated first-strike delivered reward remains unchanged.

Make guidance weight and width explicit, validated experiment settings. Preserve their current defaults. Width is the scale of a soft reward, not a hard tracking tolerance. Tests may vary it; no training sweep is launched here.

Add a separately named three-start quintic task using the existing reset poses and independent equal-probability route sampling (not exact per-batch quotas). All consumers must share the same geometry settings, with `0.02 m` vertical follow-through. Keep the original left-only registration intact. Test curve identity and observation distinguishability; add no observation feature unless a concrete ambiguity is demonstrated and resolved in the design.

The path is spatially indexed. Report current-height curve error independently of the monotone observation-phase latch, which can stop progressing outside `axis_tol`. Do not silently replace the policy's phase behavior during a measurement improvement.

### 4. Preserve and expose the joint-load constraint precisely

The current task already has active velocity CaT and impulse CaT. Preserve:

```text
imp_max_p = 0.2
imp_limit_J1_to_J6 = [0.369, 0.246, 0.738, 0.369, 0.246, 0.0164] N m s
joint_velocity_limit = 3.1415 rad/s
velocity max_p = 0.5
```

Use the existing contact-masked, baseline-subtracted generalized reaction-impulse signal. It integrates `abs(qfrc_constraint - frozen_off_contact_baseline)` over a time-based 50 ms sliding window and reports each joint separately. Under sampled physics this is 200 substeps; CaT still reads at 50 Hz. Preserve the four existing 200-substep event/window/rearm settings.

These are inherited experimental boundaries. Do not replace them with the separate provisional vector or claim they are validated robot damage limits. Soft CaT applies learning pressure; it does not guarantee a hard bound. Report every joint's raw peak, cap, normalized utilization, exceedance, and configured dose. A load-compliant miss is still a failed strike.

The prior native-controller speed-dose pair reported near-universal exceedance of the inherited J6 cap; see the source's `docs/research/reward-design/2026-09-22_speed_feasibility_review.md`. This is a feasibility warning, not a result for the newer sampled controller and not proof the cap is impossible. Before a later training campaign, assess matched useful-impact versus joint-load outcomes at 1.5/1.85/2.2 m/s with complete-window status. Do not relax the cap to make the result pass.

The contact-row-only diagnostic remains a distinct optional measurement: it is currently disabled and has different event/window semantics. Do not substitute it into CaT or reuse the same thresholds without a separate validation study.

### 5. Improve evaluation without changing training rewards

Add an evaluation-only, per-physics-substep measurement collector ordered after both the shared first-strike tracker and `substep_impulse`. Reuse the accepted onset identity rather than inventing a second contact threshold. Read the authoritative impulse accumulator at each substep, not the 50 Hz CaT telemetry cache. Record the known native site-state sampling offset (one physics substep) in output metadata. Do not add extra simulator forward calls that change the control experiment.

**Path:** precontact current-height reference error, observed sample RMS/max, and 16 normalized-height-bin error/coverage summaries. Keep unvisited bins missing, and report sample counts, skipped bins, misses, and time to contact. No invented interpolation or claim of exact pacing invariance. Report postcontact follow-through separately. At 50 Hz, only a few approach samples may exist; the substep collector resolves that limitation.

**Impact:** integrate contact force for exactly 200 physics samples, including the accepted onset sample, over an onset-aligned 50 ms window. Report positive delivered axial impulse, signed axial impulse, and the integral of transverse-force magnitude as separately named quantities. Preserve the native first-strike impulse and its stop reason too.

Provide two explicitly labeled evaluation protocols: native termination and common-window continuation. For common-window evaluation only, suppress success reset until measurement finishes; preserve failure/timeout/numerical handling. Any other reset or termination before completion yields an incomplete/censored record with observed duration and partial impulse, never a completed 50 ms result. Validate the checkpoint under its original training contract before applying and recording evaluation-only overrides.

**Loads:** report per-joint peaks over the whole observed episode and over the common impact window. A common-window load peak means the maximum of the existing sliding-window Lambda observed during that interval; do not reset its ring at accepted onset. Earlier sensor contact may contribute pre-onset history and must remain included to match the actual constraint. Label this quantity separately from object-side onset-window impulse. Report sample/substep-based durations and counts with their cadence; do not describe 50 Hz read counts as independent events. Mark missing continuation as censored. Include no-contact and failed attempts in all denominators.

The primary scientific assessment is positive axial impulse delivered to the nail over the complete common window, together with joint compliance and path error; signed axial and transverse impulse remain separate diagnostics. The current training reward is a shaped, success-censored proxy. This milestone makes that distinction visible; aligning the training objective to a common-window objective would be a separate reviewed change if needed.

### 6. Configuration, checkpoints, and replay

Add `sampled_pd500_gain_mode: Literal["fixed", "variable"] = "fixed"` while preserving existing sampled-controller selection. Reject variable mode if the sampled controller is disabled. Use shared construction consistently in training, playback, and evaluation before runner/checkpoint validation.

Use a distinct sampled-VIC controller version and action type. Record nominal gains, mapping, gain mode, timing, effort limits, joint order, installed hooks, speed bounds/mode, path/guidance settings, and inherited load contract. Preserve legacy fixed/native checkpoint metadata and strict load behavior. An old fixed checkpoint must not silently become a VIC checkpoint. Exported metadata must retain the same distinction.

### 7. Milestone acceptance

Required software evidence:

1. Variable gains change physical sampled torque at a matched unsaturated state; `p=0` matches calibrated fixed behavior.
2. Endpoints, per-world reset isolation, eight-substep hold, ten updates per policy step, effort clamp, and no double PD actuation pass.
3. Native/fixed/variable checkpoint identities and replay agree; mismatched controller/path/load settings are rejected.
4. All three curves anchor correctly, reach the nail with downward tangent, and continue vertically; diagnostics neither mutate phase nor impose timing.
5. The common window includes exactly 200 samples even after early success, and partial failure/reset windows remain explicitly incomplete.
6. Per-joint exceedances cannot disappear through aggregation, route averaging, missed contacts, or early reset.
7. Speed latch/intervention tests preserve identical first motor action and reveal whether later commands change. Untrained policies are not required to demonstrate learned physical speed compliance.

After implementation, deliver test results and a compact matrix of verified software properties versus unresolved empirical questions. Stop before training.

## Agent discussion and resolutions

| Question raised | Resolution |
|---|---|
| Can sampled VIC read native action gains directly? | No. Use the sampled calibrated nominals and existing stiffness coordinates. |
| Is delivered reward saturated? | The function default can be misleading; live `event_linear=True` already makes it unsaturated. Preserve it. |
| Does the current path prior disappear? | No. The live diagonal/quintic task removes annealing and uses weight `0.2`. |
| Should first tests change to centre start? | No. Preserve the established left-start baseline; add three-start support separately. |
| Can fine path bins be reconstructed from 50 Hz traces? | No. Record substep samples; disclose coverage and sampling semantics. |
| Should an optimum require varying selected speeds? | No. A constant best speed can be valid; intervention responsiveness is a separate question. |
| Are current joint caps calibrated hardware bounds? | No. Keep their exact experimental provenance and distinguish compliance from safety validation. |
| Does evaluation need a new training reward? | No. First expose common-window outcomes and the existing reward's censoring explicitly. |
| Can the substep collector use CaT's cached telemetry? | No. Order it after the authoritative impulse accumulator and read each physics sample; preserve the sliding-window history. |

## Next research stage after the software checkpoint

Qualify external-speed sampled VIC on the left curve, compare calibrated fixed gains and nominal-gain clamp ablations, then train autonomous speed selection using the same VIC controller. After the mechanism works, train one policy on the three-start mixture. Predeclare path-guidance settings, evaluation populations, and independent training seeds before those runs. A left-only policy's successful replay cannot establish multi-curve competence.

## Review boundary

The user selected subagent-driven implementation. This document and its linked plan are the concrete review package before that execution begins. Implementation agents work sequentially with a separate reviewer after each task and a final whole-change review. No further execution-method choice is needed.
