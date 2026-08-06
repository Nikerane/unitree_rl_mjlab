# Paper-aligned VIC gain contract for the Z1 hammer task

**Wayfinder research ticket:** *Map paper VIC semantics onto a safe Z1 gain
contract*

**Date:** 2026-08-06

**Scope:** Desired-joint-position plus per-joint variable PD gains for the later
Z1 VIC experiment. This note does not design the controlled drop, impulse-CaT
threshold experiment, curriculum, or unrelated rewards.

## Bottom line

The closest faithful translation of Bogdanovic--Khadiv--Righetti is a
12-dimensional action

\[
  a_t = [a^q_{1:6},\;p_{1:6}], \qquad p_j\in[-1,1],
\]

where the first six entries retain the already-frozen direct joint-position
contract and the second six entries modulate each joint's nominal stiffness on
a centered logarithmic scale:

\[
  m_j=C^{p_j},\qquad
  K_{p,j}=m_jK^0_{p,j},\qquad
  K_{d,j}=\sqrt{m_j}K^0_{d,j}.
\]

Thus `p_j = 0` must reproduce the fixed controller exactly, `p_j < 0` softens
it, and `p_j > 0` stiffens it. The damping output is not independent. The final
RA-L paper states the square-root coupling; the exact normalized exponential
map appears in the authors' arXiv v1 and was omitted, rather than contradicted,
in the journal revision. [The paper's v1, Section 2 and Eqs. (3--4), PDF p. 3](https://arxiv.org/pdf/1907.07500v1#page=3),
[the final v2, Section II and Eq. (3), PDF p. 2](https://arxiv.org/pdf/1907.07500v2#page=2).

There is one important implementation correction: the live Z1 uses mjlab
1.4.0 `BuiltinPositionActuator`, which does **not** expose the
`IdealPdActuator.set_gains()` method. mjlab's supported native-actuator path is
to update the per-world `actuator_gainprm` and `actuator_biasprm` arrays
together. Retaining that native implicit actuator is the only currently known
way to make `p=0` an honest reproduction of the fixed baseline. Converting the
Z1 to `IdealPdActuator` merely to obtain a method named `set_gains` would change
the integration and effort-clipping path and would invalidate the comparison.
[mjlab 1.4.0 native gain-update implementation](https://github.com/mujocolab/mjlab/blob/v1.4.0/src/mjlab/envs/mdp/dr/actuator.py#L23-L136),
[live Z1 actuator configuration](../../../src/assets/robots/unitree_z1/z1_constants.py).

The paper does not provide a transferable numerical Z1 gain range. It says the
authors first found a stable stiffness/damping range empirically and then
bounded the learned policy inside it. Therefore the numeric envelope must be a
prototype result, not a citation claim. A useful first candidate is `C=2`
(`0.5x`--`2x` nominal stiffness), but it must be approached through the staged
`C in {1.25, 1.5, 2.0}` qualification below. It is not yet a certified safe
range.

## Evidence labels

- **[PAPER]** A claim made by the Bogdanovic--Khadiv--Righetti paper.
- **[LIVE]** A fact read from the current repository, its sibling Z1 asset, or
  installed mjlab 1.4.0 code.
- **[INFERENCE]** A proposed translation supported by those facts but not
  specified by the paper.
- **[CHOICE]** A value or design decision that still requires the owner or a
  prototype result.

This distinction matters because the paper is an architectural precedent, not
a source of Z1-specific gains or a stability certificate.

## 1. What the paper actually establishes

### 1.1 Controller and action semantics

**[PAPER] Fixed gain.** The policy emits desired joint positions and a fixed PD
law computes

\[
  \tau=K_p(q_{des}(\xi)-q)-K_d\dot q.
\]

**[PAPER] Variable gain.** The policy emits both desired joint positions and
one proportional-gain modulation per joint. Damping is derived from the same
modulation using a square-root relationship. This adds one policy output per
joint; it does not add an independent damping output. The policies are
state-based, not pre-scripted time trajectories. [Final paper, Section II,
Eqs. (2--3), PDF p. 2](https://arxiv.org/pdf/1907.07500v2#page=2).

**[PAPER] Exact centered map.** The earlier author version defines
`p_j in [-1,1]`, uses each fixed gain as the midpoint in log space, and shares
one range factor `C` across joints. Its formulas are exactly the `C**p` and
square-root laws above. Consequently the nominal-reproduction rule is not an
invention: `p=0` is the paper's fixed-gain point. [arXiv v1, Section 2,
PDF p. 3](https://arxiv.org/pdf/1907.07500v1#page=3).

**[PAPER] Why the trackability term exists.** The final paper defines

\[
  r_{tt}(t)=-k\lVert q^t_{des}-q^{t+1}\rVert^2,
\]

which penalizes the command residual after the closed loop has had one control
interval to respond. It is intentionally different from penalizing
`q_des(t)-q(t)`, which would discourage motion itself. [Final paper,
Section IV-C, PDF p. 5](https://arxiv.org/pdf/1907.07500v2#page=5).

**[PAPER] The fixed-gain controller also received `r_tt`.** Figure 5 contains
fixed-gain policies both without and with this term, and the accompanying text
says the authors repeated training with the term enabled for both fixed- and
variable-gain policies. In their hopper experiment the fixed-gain desired
trajectory changed very little, whereas the variable-gain outputs became more
interpretable. This supports the proposed Z1 FIC-0/FIC-TT ablation, but it does
not promise that `r_tt` will improve Z1 hammering. [Final paper, Fig. 5 and
Section IV-C, PDF pp. 5--6](https://arxiv.org/pdf/1907.07500v2#page=5).

### 1.2 Bounds, observations, and stability

**[PAPER] Bounds are empirical.** Before learning on the real hopper, the
authors identified a range of joint stiffness and damping that remained stable
over varied motions, then restricted the policy to that range. They report no
formal passivity certificate and no Z1-transferable number. [Final paper,
Section VI, PDF p. 7](https://arxiv.org/pdf/1907.07500v2#page=7).

**[PAPER] The numerical hopper gains are not Z1 bounds.** The hopper study
swept fixed `Kp` values from 1 to 10; a fixed value of 5 was best on average in
simulation. On that particular physical hopper a fixed value of 1 was too weak
and fixed values at or above 6 were unstable, while its variable policy could
briefly reach 10. These numbers depend on that robot's inertia, transmission,
units, and contact modes. [Final paper, Sections IV-B--C, PDF pp. 4 and 6](https://arxiv.org/pdf/1907.07500v2#page=4).

**[PAPER] Policy inputs were matched across controllers.** The hopper state
contained joint positions/velocities and base position/velocity, without an
explicit contact observation. The fixed-base KUKA state contained all joint
positions/velocities and measured end-effector force. The paper does not say
that current commanded gains were added to the observation. [Final paper,
Sections II, IV-A, and V-A, PDF pp. 2--3 and 6](https://arxiv.org/pdf/1907.07500v2#page=2).

The paper does **not** specify `C`, a gain reset value, a gain slew limit,
filtering, control frequency, torque saturation, the numerical `r_tt`
coefficient, or a Z1 actuator interface.

## 2. Live Z1 and mjlab controller facts

### 2.1 Nominal plant and controller

**[LIVE]** The current arm uses six MuJoCo native position actuators. Joints 1
and 3--6 use `Kp=1000`, `Kd=100`, and an actuator-force limit of `30 N m`.
Joint 2 uses `Kp=1500`, `Kd=150`, and `60 N m`. The gripper is a separate,
vestigial actuator and must not be gain-controlled. Armature is fixed at 0.01
kg m^2 for joints 1 and 3--6 and 0.02 kg m^2 for joint 2; commanded stiffness
must not modify it. [Z1 constants, lines 73--116](../../../src/assets/robots/unitree_z1/z1_constants.py#L73).

**[LIVE]** The sibling MJCF defines the six physical ranges as:

| Joint | Physical range (rad) | Nominal `Kp` | Nominal `Kd` | Force limit (N m) |
| --- | ---: | ---: | ---: | ---: |
| joint1 | `[-2.61799, 2.61799]` | 1000 | 100 | 30 |
| joint2 | `[0, 2.96706]` | 1500 | 150 | 60 |
| joint3 | `[-2.87979, 0]` | 1000 | 100 | 30 |
| joint4 | `[-1.51844, 1.51844]` | 1000 | 100 | 30 |
| joint5 | `[-1.3439, 1.3439]` | 1000 | 100 | 30 |
| joint6 | `[-2.79253, 2.79253]` | 1000 | 100 | 30 |

The ranges are in
`/Users/nikerane/repos/safe_impact_manipulation/hammer_z1_env/assets/z1_hammer_robot.xml`
at lines 89--124. The gains and limits are instantiated by the repository
constants above; the sibling robot XML's actuator section is deliberately not
used by mjlab.

**[LIVE]** Physics runs at 500 Hz (`dt=0.002 s`) with the `implicitfast`
integrator; policy actions are held for ten substeps, so the policy rate is
50 Hz. [Hammer environment configuration, lines 277--310](../../../src/tasks/hammer/hammer_env_cfg.py#L277).
MuJoCo's position actuator corresponds to fixed `gainprm[0]=Kp` and affine
`biasprm=(0,-Kp,-Kd)`; MuJoCo recommends an implicit integrator when actuator
damping is used. [MuJoCo position-actuator reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position).

### 2.2 Why literal `set_gains()` is the wrong compatibility test

**[LIVE]** mjlab 1.4.0 has two different gain paths:

1. `IdealPdActuator.set_gains()` changes Python-side stiffness and damping
   tensors used to compute explicit torques.
2. `BuiltinPositionActuator` has no `set_gains()` method. mjlab's own
   `mdp.dr.pd_gains` implementation changes, per environment,
   `actuator_gainprm[...,0]`, `actuator_biasprm[...,1]`, and
   `actuator_biasprm[...,2]`.

The installed source matches the official
[mjlab 1.4.0 `pd_gains` implementation](https://github.com/mujocolab/mjlab/blob/v1.4.0/src/mjlab/envs/mdp/dr/actuator.py#L23-L136).
Those fields must be expanded per world before a batched policy writes different
gains in different environments; mjlab's event-field mechanism performs that
expansion before it constructs the action manager.

**[INFERENCE]** The VIC action should therefore retain
`BuiltinPositionActuator` and use the same three-field update pattern as
`pd_gains`. A small repo-owned helper may reasonably be described as setting
native PD gains, but documentation must not claim it calls the absent
`BuiltinPositionActuator.set_gains()` API.

**[INFERENCE]** Switching to `IdealPdActuator`, `DcMotorActuator`, or the paired
native PD actuator in only the VIC arm is not matched. It changes the integration
path, control inputs, or torque-speed model. Such a controller change would
require its own FIC comparator and nominal-equivalence experiment.

### 2.3 Effort saturation is part of the fixed plant

**[LIVE]** The native position actuator clamps each actuator's force output at
the fixed 30/60 N m ranges. Gain modulation must leave those ranges unchanged.
MuJoCo documents `forcerange` as actuator-output force clamping; with the Z1's
one-to-one, unit-gear joint transmission, it is also the relevant joint input
limit. [MuJoCo force-limit documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#force-limits),
[mjlab position-actuator construction](https://github.com/mujocolab/mjlab/blob/v1.4.0/src/mjlab/utils/spec.py#L257-L340).

**[INFERENCE]** Stiffening does not increase peak actuator authority. It makes
the same limit engage at a smaller tracking error. Ignoring velocity feedback,
nominal saturation begins near 0.030 rad on the 30 N m joints and 0.040 rad on
joint 2. At `C=2`, the high corner halves these errors to 0.015 and 0.020 rad;
the low corner doubles them. Damping consumes the same force budget, so actual
saturation can occur earlier. Gain-bound qualification must therefore measure
saturation occupancy; otherwise the nominally wider action may be physically
indistinguishable over most of a strike.

## 3. Candidate executable gain contract

The following is the recommended prototype contract, not yet a frozen owner
decision.

### 3.1 Action mapping and update timing

1. **[INFERENCE]** Use action order
   `[q1,...,q6,p1,...,p6]`, with no gripper channel.
2. **[LIVE]** Preserve the Stage-1 desired-position path byte-for-byte:
   default-offset absolute joint targets, the frozen per-joint scales, physical
   clipping, zero encoder bias, and 50 Hz zero-order hold. The authoritative
   details remain in the [Stage-1 plan](../../superpowers/plans/2026-08-06-z1-joint-position-fixed-stage1.md).
3. **[INFERENCE]** Clip raw gain actions to `[-1,1]`, compute `m=C**p`, then
   derive `Kp` and `Kd` from the nominal vectors. Apply desired position and
   gain from the same policy sample before the first of the ten physics
   substeps; hold both for the full 20 ms interval.
4. **[INFERENCE]** Resolve each native actuator control ID by joint name in
   canonical order `joint1` through `joint6`. Do not assume the actuator-registry
   order: the live registry groups joints 1/3/4/5/6 separately from joint 2.
5. **[INFERENCE]** On every update, write all three coupled native fields:
   `gainprm[0]=Kp`, `biasprm[1]=-Kp`, and `biasprm[2]=-Kd`. A partial update is
   not a PD gain change and must fail a unit test.
6. **[INFERENCE]** Keep armature, passive joint damping/friction, gravity
   compensation, actuator force limits, target ranges, and solver settings
   unchanged.

### 3.2 Candidate envelope

**[CHOICE] Initial search.** Qualify a common paper-style `C` in ascending order:

1. `C=1.25` (`Kp` multiplier 0.8--1.25),
2. `C=1.5` (`Kp` multiplier 2/3--1.5),
3. `C=2.0` (`Kp` multiplier 0.5--2.0).

Stop expansion at the first failed stability gate and use the largest common
passing `C`. A common `C` preserves the paper's mapping while the nominal gains
still make the absolute bounds joint-specific. Do not infer a wider range from
the paper's hopper plots.

If `C=2` passes, the candidate bounds are:

| Joint(s) | `Kp_min` | `Kp_nom` | `Kp_max` | `Kd_min` | `Kd_nom` | `Kd_max` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1, 3, 4, 5, 6 | 500 | 1000 | 2000 | 70.711 | 100 | 141.421 |
| 2 | 750 | 1500 | 3000 | 106.066 | 150 | 212.132 |

These are **prototype corners**, not hardware-safe settings. If one joint alone
blocks a common `C`, the paper-aligned response is first to choose the largest
common passing value. Per-joint `C_j` values are a defensible later extension,
but they require an explicit owner decision because they depart from the v1
shared-`C` contract.

### 3.3 Reset

**[PAPER]** No reset rule is specified.

**[INFERENCE] Recommended rule.** At construction and at every full or partial
environment reset:

- set raw gain action `p_j=0` for the reset environments;
- restore all six native `Kp/Kd` pairs to the immutable compiled defaults;
- reset gain-action history to zero;
- leave non-reset environments untouched; and
- assert the gripper gains are unchanged.

This is required because resetting MuJoCo state does not inherently reset
per-world model parameter arrays. It also makes the first post-reset
observation/controller state deterministic and fixed-controller-equivalent.

### 3.4 Policy observations and existing action penalties

**[LIVE]** The current Z1 policy observation includes the manager's entire raw
last action. The current `action_rate` cost likewise sums squared changes over
the entire action vector. [Hammer observation configuration, lines 44--102](../../../src/tasks/hammer/hammer_env_cfg.py#L44),
[current action-rate cost](../../../src/tasks/hammer/mdp/rewards.py#L75).
Naively changing action dimension from 6 to 12 would therefore also add six
observation coordinates and a gain-rate penalty. That would be two hidden
treatments, not just variable impedance.

**[INFERENCE] Recommended first comparison.** Keep the actor and critic inputs
identical to FIC-TT by exposing only the six desired-position action values in
the existing `actions` observation, and make the existing `action_rate` cost
operate only on those six position values. Log gains separately as telemetry;
do not reward or penalize their magnitude or rate in the first integration
pilot.

This is Markov for an unsmoothed, memoryless gain action: the current gain is
chosen from the current state and is replaced at the next policy step. It is
also closest to the paper, whose compared policies had the same system-state
inputs and no stated gain-rate cost. If a gain filter, slew limiter, or delayed
gain application is introduced later, the actual normalized gain becomes
controller state and must then be observed by **both** matched arms (constant
zero in FIC, live value in VIC). That is a later design change, not something to
add silently now.

### 3.5 Telemetry contract

At every 50 Hz policy step, and at 500 Hz where marked, record:

- raw/clipped `p_j`, multiplier `m_j`, commanded `Kp_j`, and commanded `Kd_j`;
- actual native `gainprm[0]`, `-biasprm[1]`, and `-biasprm[2]` for all six joints;
- lower/upper-bound occupancy and gain-action change;
- desired joint position, actual `q(t)`, causal `q(t+1)`, per-joint
  trackability error, raw `r_tt` cost, and weighted `r_tt` return;
- actual `actuator_force` and `qfrc_actuator` at 500 Hz, plus per-joint force-limit
  occupancy and longest consecutive saturation duration;
- physical-limit clipping, joint position, and substep peak joint velocity;
- first-strike contact/onset/finalization, delivered nail impulse, per-joint
  reaction impulse, velocity-CaT delta, nail progress, and success; and
- reset IDs, gain-reset values, NaN/Inf flags, and termination reasons.

The gain writer should expose typed tensors rather than reconstructing gains
later from raw actions. An episode summary should include per-joint min/median/
max gain, fraction of steps at each bound, saturation fraction, longest
saturation run, and correlations of gain with contact phase and reaction
impulse. `actuator_force` is the direct actuation-space force; `qfrc_actuator`
is the joint-space generalized force. [MuJoCo data definitions](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html).

## 4. Qualification and stability gates

The paper's method is to prequalify a bounded range before learning. For the Z1
that should be a deterministic controller prototype with four gates.

### G0: Pure mapping and schema

- `p=0` yields exactly the immutable nominal gain tensors.
- `p=-1/+1` yields exactly `Kp0/C`, `C*Kp0`, and square-root-coupled damping.
- Random finite `p` stays positive, finite, monotone, and inside bounds.
- Joint/control-ID mapping is exactly `joint1...joint6`; the gripper is absent.
- Invalid dimensions, NaN/Inf actions, non-positive nominal gains, or `C<=1`
  fail closed.

### G1: Runtime wiring, per-environment isolation, and reset

- Run on CPU and the production CUDA path with at least two environments.
- Assign opposite gain corners to two worlds and prove each world's native
  gain/bias arrays differ correctly without cross-contamination.
- Reset only one world and prove only that world returns to nominal.
- Prove desired targets and gains are held for exactly ten 500 Hz substeps.
- Prove the native actuator force limits remain exactly 30/60 N m.
- Prove CUDA graph execution reads the expanded per-world arrays.

### G2: Nominal-controller reproduction

Run the qualified scripted direct-joint strike twice from identical state and
seed: once through FIC and once through the VIC action with `p=0` at every
step. Require:

- exact equality of processed desired targets and native gain/bias arrays;
- matching joint/head trajectories to a predeclared numerical tolerance;
- matching contact onset/finalization and success outcome; and
- matching delivered impulse, per-joint reaction impulse, actuator force,
  `r_tt`, and velocity-CaT traces to the same tolerance.

**[CHOICE] Candidate tolerance:** exact tensor equality for target/gain arrays
and `atol=rtol=1e-6` for CPU trajectory/metric traces. Establish the CUDA
tolerance from two repeated nominal runs; do not loosen it after seeing a FIC
versus VIC discrepancy. Failure means the arms are not matched and blocks
training.

### G3: Envelope and transition stress test

For each `C` in `{1.25,1.5,2.0}`, test all-low, all-nominal, all-high,
one-joint-at-a-time low/high, and alternating low/high every 20 ms in:

1. a contact-free move and final-target hold,
2. the qualified hammer strike, and
3. a post-contact target hold.

Minimum hard gates:

- no NaN/Inf, simulation guard, or uncontrolled state growth;
- no physical joint-limit violation or persistent limit saturation;
- 500 Hz joint velocity stays inside the existing 3.1415 rad/s Z1 rail;
- `abs(actuator_force)` never exceeds its fixed 30/60 N m bound beyond numeric
  tolerance;
- on the contact-free final hold, velocity and position error decay rather than
  grow; and
- high-gain operation is not continuously force-saturated outside the intended
  impact transient.

**[CHOICE] Candidate settling gate:** after 0.5 s at a constant free-space
target, all `abs(qdot)<0.1 rad/s`, no actuator remains saturated, and the final
100 ms error envelope is non-increasing. **[CHOICE] Candidate persistent-
saturation gate:** outside the first-strike contact window, reject any corner
with more than 25 consecutive saturated substeps (50 ms). These thresholds
must be frozen before running the sweep.

If a static corner passes but the alternating-corner test fails, do not
immediately add a reward. First decide whether to narrow `C` or introduce a hard
gain slew/filter. A filter makes gain a state variable and changes the
observation/reset contract; it therefore requires a new prototype decision.

Passing these gates certifies only the stated simulation/model/controller
envelope. It is not a real-Z1 stability or safety certificate.

## 5. Matched FIC/RTT/VIC experiment implications

The clean campaign is:

1. **FIC-0:** six desired-position actions, nominal fixed gains, no `r_tt`.
2. **FIC-TT:** identical to FIC-0 plus the already-frozen causal `r_tt`.
3. **VIC-TT:** identical task, desired-position path, and frozen `r_tt`, plus
   the six bounded gain actions above.

FIC-0 versus FIC-TT measures whether the trackability regularizer changes the
fixed controller on this hammer task. The paper ran precisely this kind of
fixed-gain with/without comparison and found little output change on its
hopper; the Z1 result remains empirical. FIC-TT versus VIC-TT is the primary
matched variable-impedance comparison. VIC without `r_tt` is not required for
the first campaign unless the owner later wants to isolate interpretability or
gain/position non-identifiability.

Freeze across FIC-TT and VIC-TT:

- desired-position action mapping, physical clipping, policy/simulation rates,
  reset distribution, P task-space guidance, reward terms and weights;
- the same calibrated `k_tt` and causal indexing;
- velocity and impulse CaT configuration, including whether impulse CaT is
  log-only or active;
- model, contact parameters, effort limits, armature, domain randomization,
  PPO/network settings, training iterations, seeds, checkpoint selection, and
  evaluator; and
- position-action observation/history and position action-rate semantics.

The intended treatment difference is gain authority. The VIC actor necessarily
has six additional outputs, just as in the paper; that action-dimension change
is part of the VIC representation rather than something that can be removed.
Do not additionally let generic `last_action` or `action_rate` expand without
recording it as a separate treatment.

Before multi-seed training, require a one-seed integration pilot in which both
FIC-TT and VIC-TT learn a genuine strike, remain stable, and VIC uses the gain
range without living at either bound. The existing Wayfinder map's promotion
criteria remain authoritative for experiment-level success; the gates in this
note address only controller validity.

## 6. Decisions still requiring owner judgment or a prototype

1. **Final `C`.** Recommend testing `1.25 -> 1.5 -> 2.0`; no paper-backed Z1
   number exists.
2. **Common versus per-joint range.** Recommend the paper's common `C` first.
   Use `C_j` only if one joint blocks an otherwise useful envelope and the owner
   accepts the deviation.
3. **Exact settling/saturation thresholds.** Candidate numbers are stated in
   G3, but must be preregistered before observing the sweep.
4. **Gain slew/filter.** Recommend no filter for the first bounded prototype.
   Add one only if the alternating-corner gate demonstrates a need.
5. **Gain observation.** Recommend telemetry-only for the memoryless first
   controller. If a filter/delay is added, expose actual normalized gain to both
   matched arms.
6. **Literal `set_gains` wording.** Recommend implementing the native
   gain/bias update supported for the current actuator. If a literal method call
   is mandatory, first prototype a `BuiltinPositionActuator` extension; do not
   switch only VIC to the explicit Python PD backend.
7. **Hardware envelope.** Entirely unresolved. The simulation range cannot be
   promoted to hardware without Z1-specific gain authority, firmware rate and
   limit documentation plus a separate low-energy qualification.

## 7. Blocker assessment

There is no paper-level blocker to specifying or prototyping VIC. The main
engineering blocker is narrower: the phrase "via `set_gains`" does not match
the actual native Z1 actuator API. mjlab 1.4.0 nevertheless provides the
required per-world native arrays and a first-party update pattern, so this is a
design correction rather than an inability to proceed.

The true unresolved scientific value question remains empirical: with fixed
30/60 N m limits, high gains may spend much of a hammer strike saturated, while
virtual stiffness is expected to influence post-contact active reaction more
than the instantaneous rigid-transmission ballistic impulse. The telemetry and
corner sweep above are designed to expose that before expensive training; they
must not be replaced by an assumption that a wider numerical `Kp` range implies
meaningful physical authority.

## Primary sources

- Bogdanovic, M., Khadiv, M., and Righetti, L., *Learning Variable Impedance
  Control for Contact Sensitive Tasks*, IEEE Robotics and Automation Letters
  5(4), 6129--6136 (2020), DOI
  [10.1109/LRA.2020.3011379](https://doi.org/10.1109/LRA.2020.3011379);
  [final arXiv v2](https://arxiv.org/pdf/1907.07500v2) and
  [author arXiv v1](https://arxiv.org/pdf/1907.07500v1).
- [mjlab 1.4.0 actuator gain randomization source](https://github.com/mujocolab/mjlab/blob/v1.4.0/src/mjlab/envs/mdp/dr/actuator.py),
  [native position actuator](https://github.com/mujocolab/mjlab/blob/v1.4.0/src/mjlab/actuator/builtin_actuator.py),
  [ideal PD actuator](https://github.com/mujocolab/mjlab/blob/v1.4.0/src/mjlab/actuator/pd_actuator.py),
  and [joint-position action](https://github.com/mujocolab/mjlab/blob/v1.4.0/src/mjlab/envs/mdp/actions/actions.py).
- [MuJoCo actuator XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position),
  [force-limit model documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#force-limits),
  and [runtime data definitions](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html).
- Current repository sources:
  [Z1 constants](../../../src/assets/robots/unitree_z1/z1_constants.py),
  [hammer environment](../../../src/tasks/hammer/hammer_env_cfg.py), and
  [hammer rewards](../../../src/tasks/hammer/mdp/rewards.py).
