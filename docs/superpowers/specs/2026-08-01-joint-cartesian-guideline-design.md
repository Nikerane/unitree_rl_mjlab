# Joint-Position and Cartesian Guideline Experiment Design

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer
> `IMP_J_LIMIT`” wording below refers only to the historical registered-task boundary,
> not a validated Z1 reaction-impulse or damage limit. The design body remains frozen;
> see `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

**Date:** 2026-08-01
**Status:** APPROVED by owner (2026-08-01) after two independent design reviews
**Branch for shared design and geometry:** `waypoint-guideline` at baseline `83b050b`

This specification supersedes the uncommitted drafts
`2026-08-01-straight-waypoint-tube-design.md` and
`2026-08-01-straight-waypoint-tube.md`. Those drafts are retained as historical
working notes but must not be implemented: they describe a two-arm hard-termination
experiment and omit the direct joint-position action, Khadiv trackability term,
waypoint gates, and matched Cartesian comparison agreed afterward.

## 1. Research objective

Determine whether a straight, spatially guided hammer strike can be learned under
fixed impedance, and separate three possible causes of improvement in stages:

1. the policy action interface: Cartesian DiffIK versus desired joint positions;
2. command trackability: the one-control-step joint-space term proposed by
   Bogdanovic, Khadiv, and Righetti;
3. spatial guidance: ordered waypoint gates, with a continuous tolerance corridor
   measured first and rewarded only if the gates leave meaningful between-gate
   wandering.

The experiment must establish whether these changes straighten the terminal hammer
descent without sacrificing nail-driving success, useful first-contact speed,
delivered impulse, per-joint impulse legality, or joint-velocity legality.

This remains a single-policy, online-RL, fixed-impedance experiment. It is a bridge
toward later variable impedance, not a VIC experiment.

## 2. Non-negotiable invariants

- The manufacturer `IMP_J_LIMIT` values remain unchanged.
- `imp_max_p` remains `0.0`; per-joint impulse is log-only.
- Fixed actuator gains remain unchanged. Do not call `set_gains`.
- Do not change the MuJoCo model, actuator model, solver, assets, physics timestep,
  control frequency, nail task, or the frozen F8 event-linear reward outside named
  treatments.
- Do not edit installed `mjlab`, `rsl_rl`, `mujoco`, or `mujoco_warp` packages.
- The existing `r_imit` implementation remains available for historical
  reproducibility but is disabled in every arm in this campaign.
- All arms receive the same physical/task observation terms and the same guideline
  observations. The `last_action` term necessarily has width 3 for Cartesian and 6
  for joint actions; no arm receives extra privileged state.
- All GPU work is CPU-qualified first, uses one GPU per run, and comes from a clean
  committed checkout. Dirty or `-dirty` evidence is invalid.
- The existing 56 Cartesian policies remain historical/descriptive evidence. They
  are not pooled statistically with the new matched campaign.

## 3. Branch and code-isolation structure

The common spatial geometry is implemented once and reused byte-for-byte:

1. `waypoint-guideline`: shared reference, gates, corridor metrics, tests,
   and this design.
2. `cartesian-guideline-fic`: branch from the reviewed shared-geometry commit;
   retains the shipped Cartesian DiffIK action.
3. `joint-position-fic`: clean branch from baseline `83b050b`; receives the same
   shared-geometry commit, then adds desired joint-position actions and `r_tt`.

Shared geometry must be cherry-picked or merged from one named commit. It must not
be independently rewritten in the two experiment branches.

## 4. Action interfaces

### 4.1 Cartesian DiffIK

The Cartesian arms retain the shipped three-dimensional position-only action:

```text
p_des = p_head + delta_pos_scale * a,    a in [-1, 1]^3
```

The existing `DifferentialIKActionCfg`, fixed PD gains, and
`delta_pos_scale=0.15` remain unchanged.

### 4.2 Desired joint positions

The joint arms use mjlab's existing `JointPositionActionCfg`; no custom low-level
controller is introduced:

```text
q_des = q_default + scale * a,           a in [-1, 1]^6
```

The configuration uses `use_default_offset=True`, an explicit actuator-name list
for Z1 joints 1--6, and a processed-target clip map equal to those joints' physical
position limits. Environment construction must assert an action dimension of six,
the expected resolved joint order, and the clip/scale vector before any rollout;
this prevents a regex or ordering change from silently permuting the policy output.
Only the six Z1 arm joints are controlled. The qualification tape is derived from
the successful Cartesian reference's actual PD targets, not its realized joint
positions: record `robot.data.joint_pos_target` at 500 Hz and take the **first
applied target** in each corresponding 20 ms control interval. That target exists at
the control-step boundary and does not depend on a future substep state. Zero-order
hold it for the complete interval in joint playback. Per-joint scales are derived
before GPU training from this causal, reduced desired-target tape:

```text
scale_j = 1.10 * (max_t |q_target_tape[t, j] - q_default[j]| + 0.05 rad)
```

The extra `0.05 rad` matches the existing reset perturbation, so every controlled
joint can correct the full randomized reset offset in addition to covering the
reference excursion. The processed target is clipped to the real joint-position
limits. The conversion and replay are a hard CPU gate: every reset seed `1000--1015`
must reproduce a genuine hammer-face contact and nail progress without NaNs,
joint-limit violations, target saturation, or peak arm velocity above
`3.1415 rad/s`. Failure stops the joint branch before reward implementation or GPU
training; gains or manufacturer limits are not changed to rescue it. Such a failure
means this 50 Hz zero-order-held joint-target parameterization failed qualification;
it is not evidence that all desired-joint-position controllers are infeasible.

This default-offset absolute target is deliberate. `RelativeJointPositionActionCfg`
is not used because it recomputes a target relative to the current state, making the
same action history-dependent and weakening the interpretation of `r_tt`. The
existing 50 Hz policy / 500 Hz fixed-PD schedule and zero-order-held target remain
unchanged; interpolation is not added unless a later, separately justified probe
finds a real target-discontinuity failure.

Existing three-output Cartesian checkpoints are incompatible with the six-output
joint policy and are not warm-started into it.

## 5. Khadiv one-step trackability term

Only the joint-position arms can implement the paper's one-step formula directly:

```text
r_tt(t) = -k_tt * ||q_des(t) - q(t+1)||^2
```

The action target is latched for the complete 20 ms control interval; reward is
computed after that interval, so the live desired target and resulting joint state
provide the required indexing without a custom controller.
`q_des` is read from the public `robot.data.joint_pos_target` field for joints
1--6; the reward does not reach into private action-term internals.

`r_tt` is evaluated throughout the episode, including contact, because the purpose
of the experiment is to measure whether trackability regularization suppresses
unexecutable impact commands. Contact-induced tracking error is therefore an
observed trade-off rather than silently gated away.

The coefficient is frozen before training using only legal scripted joint playback
over reset seeds `1000--1015` with the training-matched `+/-0.05 rad` perturbation:

```text
k_tt = 0.1 / q90_(reset,t)(||q_des(t) - q(t+1)||^2)
```

The reward-manager weight is `1.0`; `k_tt` carries the calibrated magnitude. A q90
scripted-playback error therefore contributes an **unscaled** `-0.1` rate and an
actual one-step return contribution of `-0.002` after the reward manager applies
`step_dt=0.02`. The coefficient, per-reset traces, and cumulative reward dose are
banked before any policy outcome is viewed. The q90 error must be finite and at
least `1e-8 rad^2`; every reset-bank playback must already have passed section 4.2,
and the worst reset-bank episode's cumulative **unscaled** `r_tt` cost must not
exceed `5.0`. Its actual returned cost is then at most `0.10`, or 5% of the
one-step completion term's returned value (`100 * 0.02 = 2.0`). Otherwise
calibration fails closed instead of inventing or accepting an extreme coefficient.

Log the unweighted squared error, joint RMSE, q90 error, per-joint error, desired
joint positions, actual joint positions, and processed-target saturation rate.

The Cartesian arms do not receive an `r_tt` analogue. Comparing Cartesian target
error to joint tracking error would not reproduce the paper because DiffIK
recomputes internal joint targets at every physics substep.

No directional benefit is assumed. In Bogdanovic et al., `r_tt` primarily resolves
the interpretability/identifiability problem created when a policy also commands
gains; the fixed-gain comparison changed little. In hammering, useful impact torque
may also require desired-position error. Therefore `r_tt` may reduce tracking error,
do nothing, or reduce useful impact speed; all three are valid outcomes.

## 6. Shared straight spatial guideline

### 6.1 Scope of the path

The preparation/wind-up remains unconstrained. Spatial guidance begins at the
actual hammer-head position frozen immediately after reset and covers the terminal
descent to the nail-top position frozen at the same reset. The reference script may
first lift above this entry plane; guidance starts only when the descending head
crosses the internal gates. This avoids both the kinematically infeasible
pure-vertical-from-reset probe and the desired-apex servo-lag error.

Let `p_entry` be the frozen reset head position and `p_nail` the frozen reset nail-top
position:

```text
direction = p_nail - p_entry
u = direction / ||direction||
p_ref(s) = p_entry + s * direction,       s in [0, 1]
```

The same endpoints, direction, progress coordinate, and perpendicular distance
are used by observations, gates, corridor metric, evaluation, and visualization.

This anchor is evidence-driven. On the current CPU reference, the desired apex is
`z=0.3000 m`, but the realized head reaches only `z=0.2648 m`; an apex-based gate at
`z~0.2670 m` is therefore never crossed. With the frozen reset-head anchor, the
same legal scripted strike crosses all six internal gates with `0.15--1.24 mm`
radial crossing error and stays within `1.50 mm` of the line after gate 1.

### 6.2 Ordered waypoint gates

Six virtual gates are placed strictly inside the segment at

```text
p_i = p_entry + (i / 7) * (p_nail - p_entry),   i = 1,...,6.
```

Each gate is a disk perpendicular to `u` with radius `0.015 m`. The larger gate
radius measures ordered progress robustly; it does not define the precision
corridor.

A gate is completed only when the swept hammer-head segment between consecutive
500 Hz tracker samples crosses its plane in the forward direction and the
interpolated crossing point lies inside the disk. The detector may consume
multiple consecutive gates in one step, which prevents a fast strike from being
penalized for crossing more than one gate in 20 ms.

Only the next unvisited gate can advance. Backward crossings and revisits earn
nothing. Gate progress permanently disarms after the first hammer-face/nail contact.
Raw gate payout is normalized by six, so cumulative raw payout is at
most one per episode:

```text
r_gate(t) = newly_crossed_gate_count / 6
```

Its reward weight is `8.0`, giving a maximum **unscaled** weighted episode dose of
eight. After reward-manager `step_dt=0.02`, the maximum actual returned episode
contribution is `0.16`, or 8% of the one-step completion contribution (`2.0`).
This is one-shot spatial progress, not clock-time trajectory synchronization, and
cannot be farmed by hovering.

Before any training, the production scripted strike must cross all six gates and
remain within the 5 mm corridor after gate 1 on reset seeds `1000--1015`, each with
the training-matched `+/-0.05 rad` perturbation. Gate fractions, radius, and reward
weight are not changed after viewing policy outcomes.

### 6.3 Continuous tolerance corridor metric

The corridor radius is `0.005 m`, i.e. a `0.010 m` diameter. It uses perpendicular
distance to the finite entry-to-nail segment:

```text
d_excess = max(0, d_perp - 0.005)
```

The first campaign logs corridor occupancy and excess but gives no corridor reward
and no tube termination. This keeps the urgent spatial test to one intervention:
ordered progress gates. If `C-Gate` crosses gates reliably but still has meaningful
between-gate wandering, a later preregistered `C-Gate+Tube` arm may add
`-min((d_excess / 0.005)^2, 1)` at weight `0.1`. That follow-up must bank its reward
dose before training. A hard-corridor termination remains deferred because it would
confound geometric guidance with episode survival and could hide valid
impact/impulse evidence.

### 6.4 Observations and Markov state

One `WaypointProgressTracker` is registered as an always-on
`MetricsTermCfg(per_substep=True)` in **every** treatment, including controls. It is
the sole writer of entry/nail anchors, previous head position, next-gate index,
newly-crossed pulse, and contact-disarmed state. Its `reset(env_ids)` only clears
state and marks those environments uninitialized; the first 500 Hz call after reset
freezes the post-reset head and nail positions and emits zero progress. This is
valid under the pinned mjlab lifecycle because reset finishes with `sim.forward()`
before the next decimation loop. The initial post-reset observation uses a pure
read-only preview from the current post-forward head/nail positions with zero
completed gates. Reward and observation terms are pure readers and never advance
or reset the tracker. The per-control-step newly-crossed pulse is cleared exactly
once at the first physics substep of the next control window. These rules make the
state update idempotent and prevent enabling `r_gate` from changing tracker timing
or observations. Contact disarming reads the existing per-substep
`FirstStrikeEventTracker.started` latch instead of introducing a second contact
detector with different timing. The metrics configuration orders
`FirstStrikeEventTracker` before `WaypointProgressTracker`, and a focused test
requires same-substep contact to disarm before any gate credit is emitted.

Every new arm, including controls, receives:

- vector from hammer head to the next unvisited gate;
- normalized completed-gate fraction in `[0, 1]`;
- perpendicular distance to the reference segment.

The gate tracker is reset per environment. Exposing its active progress prevents a
hidden state in the reward and ensures that treatment identity differs only in
reward activation, not available information.

The existing actor observation width is 37 with the Cartesian action. The five new
guideline values make the exact expected widths 42 for Cartesian arms and 45 for
joint arms (the latter replaces a three-value `last_action` with six values).
Environment-construction tests freeze both widths for actor and critic groups.

## 7. Staged treatment arms

The initial campaign answers two urgent questions with four arms, not six:

| Experiment | Arm | Policy action | `r_tt` | `r_gate` |
|---|---|---|---:|---:|
| Spatial guidance | `C0` | Cartesian DiffIK | no | no |
| Spatial guidance | `C-Gate` | Cartesian DiffIK | no | yes |
| VIC bridge | `J0` | desired joint position | no | no |
| VIC bridge | `J-TT` | desired joint position | yes | no |

This is two independent two-arm experiments. Two excluded smoke seeds per arm run
first. Only arms that pass the frozen continuation rules proceed to eight matched
confirmatory seeds, for at most 32 confirmatory policies.

Interaction arms are conditional rather than automatic:

- if `C-Gate` improves straightness and `J0` qualifies, compare `J-Gate` with `J0`;
- if `J-TT` and `J-Gate` both help their own primary metric without regression,
  compare `J-TT+Gate` with `J-Gate`;
- if gate completion improves but between-gate wandering remains material, compare
  `C-Gate+Tube` with `C-Gate`.

Each follow-up receives a separate preregistration. This preserves the originally
discussed `r_tt x spatial guidance` interaction without paying for or interpreting
it before both components work independently.

All initial arms share the same base task, fixed gains, reset distribution, reward
weights outside the named treatment, guideline observations, training iterations,
environment count, and evaluation RNG streams. The base task is the current F8
event-linear fixed-impedance treatment
(`Unitree-Z1-Hammer-CaT-Impulse-Event-Linear`) with the exact reward table:

| reward | weight |
|---|---:|
| `approach` | `0.1` |
| `nail_driven` | `0.5` |
| `nail_depth_delta` | `600` |
| `impact_progress` (first-strike event-linear reader) | `8` |
| `completion` | `100` |
| `action_rate` | `-0.01` |
| `joint_pos_limits` | `-10` |
| `delivered_impulse` (first-strike event-linear reader) | `2` |

`C0` is therefore a fresh matched F8 control, not a historical checkpoint. Control
arms receive the new guideline observations, so they are also not byte-identical
replicas of historical F8.

The 3-D and 6-D interfaces necessarily change policy input/output dimensions. The
existing `action_rate` weight and PPO entropy/KL settings remain frozen rather than
being retuned after outcomes. Cross-action comparisons therefore select between two
complete interfaces under the frozen training recipe; they do not identify action
dimension as the sole cause. Within-action contrasts remain the causal claims.

## 8. Qualification sequence

1. Pure CPU/TDD tests for affine joint actions, physical clipping, `r_tt` indexing,
   waypoint construction, swept gate crossings, multi-gate crossings, progress
   ratcheting, corridor-metric boundaries, contact latch, resets, and treatment
   identity.
2. Convert and replay the scripted strike through the joint-position action. Apply
   the hard action-qualification gate in section 4.2.
3. Render the straight reference, six gates, corridor, and scripted realized path
   in x-z and x-y views. The owner confirms the intended geometry.
4. Run all mandatory reward/impulse tests and `validate_rewards.py` phases A-M,
   `verify_contact_sensor.py`, `verify_reward_setup.py`, and reference playback.
5. Independently review action semantics and treatment isolation. Resolve every
   Critical/Important issue with a failing test first.
6. Extend the strict campaign evaluator with explicit, fail-closed Cartesian and
   joint contracts. The joint contract must assert the task/campaign allowlist,
   six-dimensional action, `JointPositionActionCfg`, exact resolved joint order,
   default-offset mode, frozen scale/clip maps, expected `r_tt` and gate-treatment
   identities, frozen actuator gains, 50 Hz/500 Hz timing, `imp_max_p=0.0`, and
   absence of inherited reward overrides. The Cartesian contract
   retains its existing DiffIK/three-action assertions. Variable-width action tapes
   are schema-validated rather than coerced to three columns. Mutation tests must
   show that an unlisted task, permuted joint order, altered scale, unexpected
   reward, or wrong action width is rejected before rollout; existing frozen
   Cartesian evaluator contracts must not be weakened.
7. Run one short CUDA identity/throughput smoke per unique action/treatment path.
8. Run excluded training-smoke seeds `0` and `1` per arm. These seeds test learnability and
   configuration only and are not used for confirmatory claims or coefficient tuning.
9. Freeze the confirmatory preregistration, revisions, seeds, and artifact schema.
10. Train matched confirmatory seeds `8--15` per arm, one GPU per run.

No stage automatically widens gates, widens the corridor, changes reward weights,
or swaps failed seeds. A failed gate stops and reports the evidence.

The excluded-smoke continuation rule is frozen before launch. Every row must pass
provenance and sentinel checks; at least one of the two seeds per arm must reach
sampled success `>=0.25` by the frozen final checkpoint; `C-Gate` must record gate
payout and `J-TT` must record a finite nonzero tracking penalty; processed-target
saturation must remain below 5% of joint-control samples. An arm that fails does not
receive confirmatory seeds. These thresholds qualify the pipeline and basic
learnability only; they are not scientific outcomes.

## 9. Confirmatory evaluation

The campaign contains two preregistered experiments,
each with one primary metric as required by the repository comparison protocol.

### 9.1 Trackability experiment

Primary seed-level metric:

```text
q90_joint_target_rmse_rad_sampled
```

Sole confirmatory contrast: `J-TT` versus `J0`.

For every sampled episode and control step, compute

```text
rmse_t = sqrt(mean_j((q_des(t, j) - q(t+1, j))^2)),   j = 1,...,6.
```

The episode statistic is `T_ep = q90_t(rmse_t)` over the complete episode. The
seed-level primary endpoint is `q90_ep(T_ep)`, with every sampled episode entering
exactly once. Control steps are never pooled across episodes, so a long or failed
episode cannot receive extra statistical weight merely because it contains more
samples.

### 9.2 Straightness experiment

Primary seed-level metric:

```text
q90_terminal_descent_perpendicular_error_m_sampled
```

Sole confirmatory contrast: `C-Gate` versus `C0`.

The straightness metric is unconditional over sampled episodes. It uses the
existing 500 Hz physics-substep hammer-head trace from the first valid forward
crossing of gate 1 through first hammer-face/nail contact or episode end, whether
or not the episode succeeds.
At every included sample, define the capped error

```text
e_t = min(d_perp(t), 0.050 m).
```

For a valid episode, compute `E_ep = q90_t(e_t)`. An episode that never crosses
gate 1 receives `E_ep = 0.050 m`; contact without a valid gate-1 crossing receives
the same fail-closed value. The seed-level primary endpoint is `q90_ep(E_ep)`, with
every sampled episode entering exactly once. Substeps are never pooled across
episodes. Capping valid samples at the same 50 mm failure value ensures a failed
episode can never score better than a valid but highly deviating trajectory. This
prevents treatment-dependent success censoring or episode length from making a weak
arm look straight.

Cross-action comparison `J0-C0` is descriptive/secondary because action
dimensionality and controller structure differ. Conditional interaction arms define
their own primary contrast in a later preregistration.

Every headline contrast uses eight seed rows, two-sided Mann-Whitney U, seed-level
mean, standard deviation, minimum, maximum, and one preregistered practical effect.
Trackability requires both a 20% reduction and at least `0.01 rad` absolute reduction
in q90 joint RMSE. Straightness requires both a 20% reduction and at least `0.001 m`
absolute reduction in q90 perpendicular error. Report effects below threshold or
with `p >= 0.05` as not distinguishable at this seed budget. Because each experiment
has one sole primary contrast, no within-family multiplicity correction is needed;
all additional contrasts are explicitly exploratory.

Secondary/non-regression metrics:

- terminal-descent excess path length and backward-progress count;
- fraction of descent within the 5 mm corridor and gates completed;
- sampled success and first-window success;
- sampled useful first-contact speed and nail depth;
- delivered impulse and maximum per-joint `Lambda_j / IMP_J_LIMIT_j`;
- sampled peak arm joint velocity and target saturation;
- impact-versus-press contact classification.
- sampled time to first contact and episode duration.

Required validity gates:

- `impossible_success_n == 0` for every evaluated row;
- `lambda_dead_n == 0` for every evaluated row;
- no dirty provenance;
- no completed weak seed replacement;
- no hardware-safe claim for a policy with a joint-velocity rail exceedance;
- held-out evaluation with the training-matched `+/-0.05 rad` reset perturbation.

A treatment is not adopted if sampled success falls by more than five percentage
points, useful first-contact speed falls by more than ten percent, or median time to
first contact increases by more than ten percent relative to its matched
no-treatment action-space control.

## 10. Visual evidence

Use one identical fixed reset for direct visual comparison. Produce for each new
policy:

- MP4 strike video;
- x-z and x-y hammer-head trajectory;
- black dashed reference segment;
- six gate disks/markers and translucent 10 mm-diameter corridor;
- contact marker and time colouring;
- desired-versus-actual joint traces for joint arms;
- gate progress, perpendicular error, speed, force, delivered impulse, and per-joint
  impulse traces.

Produce seed-matched arm grids and preserve compact plots, metadata, traces, hashes,
and selected MP4s outside temporary directories.

## 11. Decision rules

- Joint playback fails qualification: stop the joint branch; do not infer anything
  about `r_tt` or spatial guidance.
- `C-Gate` improves straightness without regression: ordered spatial progress is a
  validated FIC trajectory-shaping lever; proceed to the conditional joint-gate test.
- `C-Gate` crosses all gates but still wanders materially between them: the
  continuous corridor reward becomes justified for a separately preregistered test.
- `r_tt` reduces joint error but not path error: it is working as the Khadiv paper
  measures, and should not be described as a path reward.
- `r_tt` reduces speed or success beyond the non-regression margin: report the
  trackability-impact trade-off; do not silently weaken it after seeing outcomes.
- `J0` fails qualification or learns materially worse than `C0`: retain Cartesian
  DiffIK for the FIC result; this does not invalidate future VIC, but it rejects this
  joint-target parameterization under shipped gains.
- Later `J-TT+Gate` passes all gates and leads both errors: carry this fixed-gain
  interface forward as the candidate basis for the separately approved VIC phase.
- `C-Gate` does not improve straightness: curvature is more likely task-optimal,
  kinematic, or caused by a different missing reward signal than ordered progress.

## 12. Literature provenance

### Directly reproduced or closely followed

- Miroslav Bogdanovic, Majid Khadiv, and Ludovic Righetti, **Learning Variable
  Impedance Control for Contact Sensitive Tasks**, IEEE RA-L 2020,
  arXiv:1907.07500. Source for desired joint-position policy actions and the
  one-control-step trackability term. This campaign evaluates the same formula as a
  fixed-gain characterization; its original variable-gain identifiability role and
  commanded impedance are deferred.
- Seungeun Rho et al., **LineRides: Line-Guided Reinforcement Learning for Bicycle
  Robot Stunts**, IEEE RA-L 2026, arXiv:2605.05110. Source for a user-provided
  spatial guideline, ordered spatial progress, and tolerance margins without exact
  clock-time synchronization.
- Yunlong Song et al., **Autonomous Drone Racing with Deep Reinforcement Learning**,
  arXiv:2103.08624. Supports ordered gate traversal and policy observations relative
  to upcoming gates. Its gates are physical race apertures; ours are virtual disks.

### Joint-position action-space evidence

- Patrick Varin, Lev Grossman, and Scott Kuindersma, **A Comparison of Action Spaces
  for Learning Manipulation Tasks**, IROS 2019, arXiv:1908.08659. Directly compares
  joint PD, torque, inverse dynamics, and task-space impedance on hammering. Its
  joint-PD arms learned more slowly than model-based task-space interfaces, so this
  experiment retains matched Cartesian controls and makes no assumption that the
  joint interface will win.
- Xue Bin Peng and Michiel van de Panne, **Learning Locomotion Skills Using DeepRL:
  Does the Choice of Action Space Matter?**, SCA 2017, arXiv:1611.01055; and Xue Bin
  Peng et al., **DeepMimic**, ACM TOG 2018, arXiv:1804.02717. Establish the common
  multi-rate pattern in which a lower-frequency policy outputs target joint angles
  tracked by a faster fixed-gain PD loop.
- Shuxiao Chen et al., **Learning Torque Control for Quadrupedal Locomotion**,
  arXiv:2203.05194, and the official `leggedrobotics/legged_gym` implementation.
  Document the prevalent bounded residual form `q_default + scale * action`, joint
  state/velocity and previous-action observations, and torque/target clipping. These
  are implementation precedents, not hammer-task performance evidence.
- OpenAI et al., **Learning Dexterous In-Hand Manipulation**, IJRR 2020,
  arXiv:1808.00177, and Rajeswaran et al., **Learning Complex Dexterous Manipulation
  with Deep Reinforcement Learning and Demonstrations**, RSS 2018,
  arXiv:1709.10087. Show that low-rate joint-position commands can support real and
  simulated contact-rich manipulation, including hammering; they do not show that
  joint actions alone produce intuitive Cartesian paths.
- Roberto Martín-Martín et al., **Variable Impedance Control in End-Effector Space:
  An Action Space for Reinforcement Learning in Contact-Rich Tasks**, IROS 2019,
  arXiv:1906.08880. Shows that task-aligned end-effector action spaces can improve
  sample efficiency and safety. Its controller includes model-based compensation and
  interpolation, so its results are not evidence that the shipped Z1 fixed PD will
  behave identically.

### Philosophical influence and deferred extensions

- Jeonghwan Kim et al., **Flip Stunts on Bicycle Robots using Iterative Motion
  Imitation**, ICRA 2026, arXiv:2603.27944. Supports treating an imperfect reference
  as guidance that a policy can turn into dynamically feasible motion. Iterative
  replacement of the reference from policy rollouts is not implemented initially.
- Xue Bin Peng et al., **DeepMimic: Example-Guided Deep Reinforcement Learning of
  Physics-Based Character Skills**, ACM TOG 2018, arXiv:1804.02717. Historical
  context for imitation-plus-task rewards. The existing weak, Cartesian,
  ante-impact, annealed `r_imit` is not a DeepMimic reproduction and stays disabled.
- Biemond, van de Wouw, Heemels, and Nijmeijer, IEEE TAC 2013,
  DOI:10.1109/TAC.2012.2223351. Supports avoiding conventional clock-time tracking
  across state-triggered impacts whose jump times do not coincide.
- Yang and Posa, **Impact-Invariant Control**, IROS 2021, arXiv:2103.06907.
  Motivates not penalizing unavoidable contact-induced velocity jumps. No
  impact-invariant projection is implemented here.
- Acosta, Yang, and Posa, **Validating Robotics Simulators on Real-World Impacts**,
  IEEE RA-L 2022, arXiv:2110.00541. Grounds cautious interpretation of simulated
  impact and impulse results.

### Honest adaptation boundary

The 5 mm continuous corridor metric, swept control-segment/disk intersection,
six-gate layout, and 15 mm gate radius are Z1-specific design choices. They are
inspired by the cited spatial-guidance literature but are not claimed as direct
reproductions of LineRides, IMI, or drone racing.

arXiv:1709.10087 is **Learning Complex Dexterous Manipulation with Deep
Reinforcement Learning and Demonstrations**, not the drone-racing paper; it is not
used as the gate-method source.

## 13. Simplicity budget

- Use existing mjlab action classes; do not add a controller class.
- Implement one shared geometry/tracker unit, one `r_gate` term, and one joint-only
  `r_tt` term. Corridor reward code is not part of the initial campaign.
- Reuse existing training, evaluation, statistics, and video infrastructure.
- Do not add a six/eight-arm framework, hard-tube termination, iterative imitation,
  variable impedance, or a new artifact database.
- Any request exceeding approximately 250 new production-code lines across the
  shared and joint branches requires stopping to explain why the existing APIs are
  insufficient.
