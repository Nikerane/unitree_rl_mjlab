# Z1 impulse-CaT simplified thesis campaign design

## Material Passport

- Origin skill: `academic-research-suite` experiment planning
- Origin mode: `plan`
- Origin date: 2026-08-18
- Version label: `z1_impulse_cat_campaign_design_v2_simple`
- Verification status: `UNVERIFIED` — this document authorizes no run

**Status:** user-approved simplified direction on 2026-08-18. This replaces the earlier
contact-gated sequence in version 1. It authorizes a detailed implementation plan only, not code,
simulation, Vega submission, training, or an automatic next stage.

## Honest starting verdict

The mechanism works, but the scientific result is not yet a success.

- At diagnostic `0.9` caps and `imp_max_p=0.5`, impulse pressure activated and strictly won the
  max soft-OR whenever the impulse cap was exceeded.
- For one seed-2 trained policy, native observed provisional and diagnostic impulse exposure fell
  sharply while task success and delivered hammer impulse were retained.
- True 500 Hz joint-velocity violation risk increased in every stochastic evaluation population.
- This is redistribution/trade-off, not clean enforcement.

The evidence does not establish hard capping at `0.9`, an actuator damage limit, hardware safety,
an optimal `imp_max_p`, or robustness across PPO training seeds.

## Research question

Does one lower impulse-CaT training dose reduce native observed per-joint impulse exposure without
moving risk into true joint velocity or degrading hammering utility?

The treatment is graded PPO credit and continuation pressure, not a runtime clamp. The thresholds
remain project-defined simulation boundaries:

- provisional caps: `[0.820, 1.640, 0.820, 0.820, 0.820, 0.820] N.m.s`;
- historical diagnostic-only caps: `[0.738, 1.476, 0.738, 0.738, 0.738, 0.738] N.m.s`.

## Why contact completion is optional

Native success often terminates while contact remains active. That limits claims about a complete
physical contact event and about event pressure `P_event`, but it does not prevent measurement of
the task's native episode outcomes: Lambda observed before termination, true joint velocity,
success, productive strike, delivered hammer impulse, nail depth, gains, actions, and duration.

Therefore post-success contact/window flushing is an optional sensitivity analysis. It is not a
prerequisite, a training gate, or the center of this campaign. Native results must be described as
**native observed exposure**, not complete-contact impulse or hardware loading.

## Minimal campaign

### Step 0: existing-trace sanity check, no learning

Reuse the already banked seed-2 native control traces. At the provisional caps, replay only
`imp_max_p=0.2` and confirm:

- `delta_impulse` is finite, graded, and nonzero on cap-active reads;
- combined pressure is exactly `max(delta_velocity, delta_impulse)`;
- impulse is not completely masked by velocity;
- duration, controller-read count, censoring, and contact-aligned observed-prefix summaries remain
  reported so shorter exposure cannot be hidden.

Also reconfirm from the existing `p=0`/`p=0.5` traces that the already banked contact-aligned
observed-prefix Lambda tail has the same direction as the native episode tail. This is a
zero-training exposure sensitivity, not a complete-contact analysis. If its direction reverses,
stop and report an unresolved exposure confound rather than starting the bridge.

This check does not select or optimize a dose and cannot predict the policy learned at `p=0.2`.
The value `0.2` is a single predeclared lower-dose bridge candidate. If it is completely masked or
the trace invariants fail, stop before training.

### Step 1: one 500-iteration bridge

Train exactly one new seed-2 target:

- VIC-TT task, 4,096 environments, 24 rollout steps, 500 PPO iterations;
- velocity CaT unchanged;
- provisional project caps;
- `imp_max_p=0.2`;
- every other task, PPO, physics, reward, trajectory, observation, VIC-gain, randomization, and
  curriculum setting unchanged.

Use the banked seed-2 `imp_max_p=0` checkpoint as control only after a test reconfirms that changing
a log-only cap vector cannot alter actions, rewards, PPO data, RNG consumption, or policy updates.
In the bridge comparison, control and target are analyzed at the same provisional caps; the only
active training treatment is impulse pressure. The historical diagnostic-cap `p=0.5` policy is
context, not a causal third arm.

Do not train another `p=0` control if exact log-only identity passes. Do not train `p=0.1`, `0.3`,
or `0.5` alongside the bridge.

### Step 2: frozen native evaluation

Evaluate the control and bridge checkpoint with live `imp_max_p=0` so evaluation itself cannot
change behavior. Reuse:

- the deterministic 64-world mean-policy population as a regression check;
- stochastic populations `2`, `2026081701`, and `2026081702`, each with 4,096 matched worlds;
- the existing evaluator, telemetry, manifest, bootstrap, and comparison machinery.

The fixed population is not pooled with the stochastic populations. Evaluation populations are
reported separately and are not training replicates.

## Primary endpoints and bridge gates

For cap vector `L`, define `rho_L = max_t,j Lambda[t,j] / L[j]`. Native provisional impulse risk is
the fraction of initial episodes with `rho_provisional > 1`. Native true-velocity risk is the
fraction with any 500 Hz joint-velocity-limit violation.

The bridge passes only when all conditions hold **separately in each of the three stochastic
populations**; there is no pooled go/no-go result. The fixed-64 population remains a regression
check only.

1. upper paired one-sided 97.5% bound for target-minus-control provisional-risk difference is below
   `-0.50` percentage points;
2. global provisional `rho` p95 and p99 each decrease by at least 10% relative to control; no
   joint whose control p99 utilization is at least `0.10` has a relative p99 increase of 10% or
   more, and no previously nonviolating joint becomes cap-violating;
3. upper paired one-sided 97.5% bound for true-velocity-risk difference is at most `+0.10`
   percentage points;
4. success and productive-strike differences are no worse than `-1` percentage point;
5. lower paired one-sided 97.5% bound for the first-event delivered-impulse target/control ratio is
   at least `0.90`;
6. telemetry, population identity, finiteness, and native-horizon claim boundaries all pass.

These are strict screening criteria for a large useful effect, not safety limits and not a
training-seed confidence statement. Episode duration, contact censoring, controller-read counts,
contact-aligned observed-prefix Lambda, nail depth, cumulative delivered impulse, VIC gains, and
actions are mandatory secondary descriptors. They prevent a shorter episode from being silently
described as gentler complete contact.

## Stop rules

- Do not start the bridge if `p=0.2` is completely masked or telemetry identity fails.
- Do not retry, requeue, extend, or replace a failed job automatically.
- If the velocity gate fails, verdict is redistribution/trade-off; do not raise `imp_max_p`.
- If impulse efficacy or utility fails, stop; do not try another dose automatically.
- Do not lower caps, add torque CaT, change rewards, or change the trajectory to rescue the result.
- A zero target count is reported with an uncertainty bound, never as zero risk.

## Confirmation only after a bridge pass

Only after the bridge passes may a separate, explicitly approved plan train matched independent
PPO seeds under the same provisional-cap `p=0` versus `p=0.2` comparison. The intended thesis
confirmation set remains seeds `2--9`, with training seed as the primary unit. The bridge result
alone is a screen and cannot support a robust learned-policy claim.

## Optional later sensitivity

A release/window-flush continuation may later test whether terminal censoring changes the
complete-event interpretation. It must preserve the native prefix exactly and remain labeled
counterfactual. It does not block the bridge and cannot replace native task evaluation.

## Independent review

The simplified plan was independently reviewed through OpenCode Zen by Kimi K3, GLM-5.2,
Qwen3.6-Plus, DeepSeek V4 Pro, and Claude Opus 5. All five returned `REVISE`. Their useful common
warnings were the single training seed, the velocity trade-off, shorter target episodes, and the
absence of a hardware-valid cap. The resulting changes are the explicit native-observed claim,
mandatory duration/censoring descriptors, one predeclared dose rather than an optimization claim,
and a hard velocity stop.

Some suggestions were not adopted: frozen-policy replay cannot reveal how training dose changes
velocity behavior; repeating the known high-dose treatment does not answer the lower-dose
question; and the proposed bridge does not compare two cap vectors within an arm. Full provenance
and adjudication are in
`docs/research/reward-design/Z1_IMPULSE_CAT_SIMPLIFIED_CROSS_MODEL_REVIEWS.md`.

## Compute and authorization

The bridge is one new 500-iteration A100 training arm, historically about 23 minutes, plus frozen
evaluation. Multi-seed confirmation is a later budget and a separate approval.

Approval of this design permits only a detailed implementation plan. It does not authorize local
live simulation, remote writes, `sbatch --test-only`, submission, training, or automatic
continuation.
