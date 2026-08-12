# Z1 Native Variable-Impedance Prototype Design

**Date:** 2026-08-13

**Status:** approved for implementation

**Branch:** `z1-vic-prototype`

## Objective

Add the smallest scientifically matched Z1 variable-impedance controller on top of the selected direct-reference FIC-TT treatment. The policy continues to command six absolute joint-position targets and additionally commands one bounded stiffness coordinate per arm joint. It does not output torque, Cartesian impedance, damping independently, gripper gains, or a shared scalar gain.

The prototype exists to prove controller correctness, fixed-impedance parity, multi-environment isolation, CPU/CUDA execution, and one genuine CatPPO update before any overnight training campaign.

## Scientific contract

For each arm joint `j`, one policy sample contains

```text
[q_des_1, ..., q_des_6, p_1, ..., p_6]
p_j in [-1, 1]
m_j = C ** p_j
Kp_j = m_j * Kp0_j
Kd_j = sqrt(m_j) * Kd0_j
```

`C` is configurable and greater than one in the controller implementation, but this prototype freezes the treatment at the conservative value `C=1.25`. Selecting a larger envelope is a separate scientific decision and is not performed in this implementation session.

At `p=0`, `m=1`, so the commanded gains reproduce FIC exactly. The existing position affine mapping, target clipping, nominal gains, effort limits, armature, passive damping, gravity compensation, control rate, decimation, task observations, reference reward, CaT behavior, and reset distribution remain unchanged.

The joint-target trackability term remains exactly

```text
cost_tt = ||q_des_applied(t) - q_actual(t+1)||^2
k_tt = 1
RewardManager weight = -1
```

It remains in the negative-term partition and is never multiplied by the positive soft-CaT survival factor. FIC-TT and VIC-TT use the same term.

## Action architecture

Use two ordered action-manager terms that form one 12-dimensional policy vector:

1. the existing qualified `joint_position` term, unchanged, for the first six coordinates;
2. a new `joint_stiffness` term for the final six coordinates.

This preserves the banked position-target path literally. The action manager processes both terms from the same policy sample before the decimation loop and reapplies both during each physics substep, so position targets and gains are causally simultaneous and held together.

Do not replace the native actuator with `IdealPdActuator`, implement a torque-action policy, or reimplement the position action in a composite class.

## Native MuJoCo gain path

The production Z1 uses `BuiltinPositionActuator`. For unit transmissions, write all three native fields together:

```text
actuator_gainprm[..., 0] = Kp
actuator_biasprm[..., 1] = -Kp
actuator_biasprm[..., 2] = -Kd
```

This yields `Kp * (q_des - q) - Kd * qdot` before the existing force clamp.

The model gain arrays initially alias across worlds. A startup event declared with `requires_model_fields("actuator_gainprm", "actuator_biasprm")` must expand them before the action manager is constructed. Construction fails closed if the fields are not independently writable per world.

Resolve actuator control IDs by canonical joint name and require every target to be covered exactly once by a `BuiltinPositionActuator`. The live compiled order is not the policy order; the expected canonical mapping for joints 1 through 6 is `[0, 5, 1, 2, 3, 4]`. The gripper is excluded. Clone nominal `Kp0/Kd0` from `sim.get_default_field("actuator_gainprm")` and `sim.get_default_field("actuator_biasprm")` at those resolved IDs, validate the native affine signs, and preserve every non-target gain/bias slot plus `actuator_forcelimited` and `actuator_forcerange` byte-for-byte.

MuJoCo state reset does not restore model parameters. `joint_stiffness.reset(env_ids)` restores nominal `Kp0/Kd0` only for the selected environments, clears only their gain-action telemetry, and leaves every other world and the gripper unchanged.

## Matched observations and rewards

The direct-reference VIC-TT actor and critic observations remain exactly the 40-column FIC-TT schema. The `actions` observation reads only `joint_position`; stiffness history is telemetry, not a new policy input.

The existing action-rate cost also reads only `joint_position`. Gain-only changes therefore do not introduce a new regularizer. No gain-rate, gain-magnitude, jitter, acceleration, energy, or torque reward is added.

The trackability reader accepts only these exact action signatures:

```text
("joint_position",)
("joint_position", "joint_stiffness")
```

It always reads the six applied position targets and the six corresponding actual joint positions. Any other signature fails closed.

## Task and metadata

Add one immutable task:

`Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT`

Its runner metadata records the ordered 12-dimensional action contract, canonical six joint names, mapping family `author_v1_exponential`, `C`, `p` bounds, nominal `Kp0/Kd0`, native field identities, position-only observation/action-rate behavior, and exact RTT identity. Metadata resolution fails closed on an unknown action pair or mutated controller contract.

## Verification gates

- **G0 — pure mapping:** exact `p=-1,0,+1` values, monotonicity, clipping, finite outputs, invalid-`C` rejection, and coupled damping.
- **G1 — native isolation:** two worlds can hold opposite gains without aliasing; partial reset restores only selected worlds; gripper gains and all force limits remain unchanged.
- **G2 — nominal parity:** with `p=0`, VIC and FIC produce exactly equal applied position targets and native gains. CPU qpos/qvel, RTT, contact, and CaT traces use predeclared `atol=rtol=1e-6`. CUDA comparison tolerance is fixed from two repeated nominal runs before comparing FIC with VIC; it is never tuned to the FIC-VIC discrepancy. Position targets and gains must remain paired through all ten substeps.
- **G3 — bounded authority:** at the single frozen value `C=1.25`, deterministic opposite-corner and alternating-command tapes remain finite, respect force limits, preserve reset isolation, and show the expected gain-to-force ordering under a declared nonsaturated position error.
- **G4 — learning seam:** reset/step observations are finite `(N,40)`, actions are 12-wide, and a genuine one-iteration CatPPO smoke produces finite learner state and checkpoint on CPU and CUDA.

Run focused tests, repository reward/contact/reference gates, the full CPU suite once, a clean Vega A100 smoke, and independent Standards/Spec review. Fix concrete correctness findings test-first before committing and pushing.

## Training boundary

Do not launch VIC training until G0-G4 pass. Then run only a seed-2 VIC-TT engineering canary at the same direct-reference FIC protocol (`4096` environments, `500` PPO iterations). This prototype may report a finite checkpoint and existing training telemetry, but it does not add an evaluator or claim comparative task performance. Further seeds and a formal FIC-TT comparison require a separately approved evaluation plan.

## Out of scope

This prototype does not add VIC-0, active impulse-CaT, altered manufacturer caps, evaluator expansion, parameter-sweep infrastructure, gain filters, gain observations, gain rewards, domain randomization, curriculum changes, Cartesian tasks, controlled-drop changes, hardware claims, or G1 work.
