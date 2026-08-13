# Z1 VIC-TT Seed-2 Engineering Canary Design

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-13
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

**Status:** approved for implementation and one Vega launch

## Objective

Run exactly one full-budget engineering canary for the qualified native
variable-impedance task:

`Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT`

The canary asks whether the 12-dimensional VIC-TT learning path can finish a
real 500-iteration run, produce a finite reloadable checkpoint, command
nontrivial bounded gains on visited training states, and achieve successful
hammer strikes on the existing frozen 64-world evaluation population.

This is an engineering go/no-go for one seed. It is not evidence that VIC is
reproducibly better than FIC.

## Frozen treatment and training protocol

- Training seed: exactly `2`.
- Environments: exactly `4096`.
- PPO iterations: exactly `500`; final checkpoint `model_499.pt`.
- Save interval: exactly `50`.
- Task observations: the qualified 40-column direct-reference schema.
- Policy action: ordered `[q_des_1..6, p_1..6]`.
- Gain map: `C=1.25`, `m=C**p`, `Kp=m*Kp0`,
  `Kd=sqrt(m)*Kd0`, with executed `p` clipped to `[-1,1]`.
- Direct-reference reward, `r_tt` with `k_tt=1`, fixed resets, velocity
  soft-CaT, and log-only impulse-CaT remain unchanged.
- One `NVIDIA A100-SXM4-40GB`; mjlab `1.4.0`, MuJoCo `3.8.1`, and
  mujoco-warp `3.8.1`.
- No resume, scientific CLI override, seed expansion, or automatic retry.

## Minimal implementation

### Checkpoint gain telemetry

For the exact VIC action signature only, `HammerOnPolicyRunner.save` attaches
one compact `infos["vic_rollout_telemetry"]` record when a populated rollout
is available. It summarizes the most recently collected 24-by-N rollout that
preceded the checkpoint's PPO update:

- schema/source identity and sample count;
- canonical six joints and gain-action indices `6..11`;
- per-joint mean, standard deviation, minimum, p05, median, p95, and maximum;
- lower- and upper-bound occupancy;
- both deterministic Gaussian means and sampled actions, before and after the
  qualified `[-1,1]` execution clip;
- per-joint exploration standard deviation.

The record is training telemetry, not a policy input, reward, metric term, or
controller behavior change. A manual VIC save before any rollout remains
valid and simply has no rollout record. FIC and Cartesian checkpoint bytes and
metadata behavior remain unchanged.

The deterministic clipped gain summary is classified without inventing a
performance threshold:

- all zeros: no measured gain authority use;
- constant nonzero: shifted fixed impedance, not state-varying scheduling;
- nonzero spread: variable gain commands on visited training states;
- any joint at one bound for the entire population: bound-pinned failure.

### Fail-closed Vega launcher

Add one no-argument, non-array launcher dedicated to this canary. It must:

1. require exact clean code and canonical clean asset revisions;
2. reject `PYTHONOPTIMIZE`, extra arguments, reused result leaves, wrong GPU,
   wrong package versions, or scientific overrides;
3. rerun the qualified CUDA VIC live/parity/authority smoke and genuine
   one-iteration CatPPO smoke at the same revision;
4. train the exact task/seed/budget above;
5. require exactly the checkpoints `0,50,...,450,499`, finite recursive
   checkpoint tensors, exact iteration identities, and VIC telemetry in
   `model_0.pt` and `model_499.pt`;
6. require finite TensorBoard scalars, all six `r_imit` curriculum plateaus,
   final curriculum weight zero, and `impossible_success` identically zero;
7. write a compact canonical gain/curriculum JSON and print SHA-256 identities.

The launcher stores artifacts only under
`$HOME/campaigns/z1-vic-prototype/runs/<code>/<assets>/<job>_victt_seed2` and
Slurm logs only under the campaign's existing `slurm/` directory.

## Deterministic behavior check

After training, run a hashed scratch wrapper around the already-tested FIC
64-world evaluator. The wrapper changes no repository file and admits only
the exact VIC-TT task/checkpoint/revisions. It retains the existing evaluation
protocol:

- evaluation seed `2026081202`;
- 64 environments;
- deterministic mean policy;
- first episode only, no auto-reset;
- four-second maximum horizon.

It reports success/productive-strike counts, first-event delivered impulse,
joint-target error, reference error, 500 Hz joint-velocity peaks, and
per-joint impulse-cap utilization. The script bytes and SHA-256 are retained
with the run so the scratch analysis is reproducible without adding another
permanent evaluator stack.

## Canary interpretation

The engineering canary passes only if:

- Slurm and training finish successfully with a finite strict-reloadable
  `model_499.pt`;
- the final checkpoint contains finite bounded gain telemetry, is not exactly
  all-zero, has nonzero cross-state deterministic gain spread, and no joint is
  pinned entirely at either bound;
- at least `58/64` evaluation worlds succeed and at least `58/64` have a
  productive first strike;
- all `64/64` evaluated first episodes remain at or below `3.1415 rad/s` on
  every joint and at or below every log-only impulse cap;
- all instrumentation is finite and no `impossible_success` occurs.

Failure is evidence about this seed, not authorization to tune, retry, or
weaken a threshold. Preserve the checkpoint and diagnostics and stop.

## Claim boundary and stopping rule

Seed 2 can support only: “the native VIC-TT training and inference path worked
for one engineering canary.” It cannot support a repeatability, superiority,
hardware-safety, or causal gain-authority claim.

RSL-RL sums entropy, log probability, and KL over action dimensions. The 12D
VIC learner is therefore not optimizer-normalized to the banked 6D FIC learner.
A formal causal comparison requires a separately approved dimension-matched
control and multiple seeds.

Stop after seed 2 regardless of outcome. Do not launch seeds 3/4, VIC-0,
active impulse-CaT, a parameter sweep, a retrained FIC arm, domain
randomization, curriculum changes, or controlled-drop work.
