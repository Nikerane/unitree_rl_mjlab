# Joint-position action implementation review

**Date:** 2026-08-07
**Scope:** fixed-gain Z1 hammer arm (`FIC-0`) and constraints on the later variable-impedance arm
**Code inspected:** `src/tasks/hammer/config/z1/env_cfgs.py`, action/config/live tests, and the installed `mjlab==1.4.0` implementation. The fixed action itself entered at commit `e2e6ef848b87827ea98098ad8d6292c2d3a75dac`.

## Verdict

The current fixed-gain joint-position action is consistent with the canonical implementation used by mjlab, Isaac Lab, and first-party Unitree RL repositories. I found **no Critical or Important defect in the present FIC-0 action mapping**. The implementation correctly uses an absolute default-offset target, a six-joint per-axis scale, raw-policy clipping in the RSL-RL wrapper, a second absolute physical target clip, a zero-order hold across all ten physics substeps, and no gripper action. The live tests cover each of those semantics.

There are two **Important constraints for future stages**, not reasons to change FIC-0 now:

1. The Z1 currently uses mjlab's native `BuiltinPositionActuator`. In mjlab 1.4.0 that class does **not** have `set_gains`; `set_gains` exists on `IdealPdActuator`, whose explicit Python torque path is a different plant. The VIC implementation must either add a purpose-built action term that updates the native MuJoCo gain/bias model fields, or first add a tested native actuator API. Silently swapping to `IdealPdActuator` would confound FIC-versus-VIC.
2. The physical target clip is applied before mjlab subtracts `encoder_bias`. Bias is zero in the present arm and the live test proves that. If encoder-bias domain randomization is introduced later, the post-bias command can lie outside the nominal absolute clip by the bias magnitude. Preserve zero bias for the first controlled comparison, or add and test a post-bias safety rail before enabling that randomization.

## Exact current semantics

For policy output \(a\), the training path is

\[
a_c = \operatorname{clip}(a,-1,1),\qquad
\bar q_{des}=\operatorname{clip}_{q_{min},q_{max}}
\left(q_{default}+s\odot a_c\right),\qquad
q_{cmd}=\bar q_{des}-b_{encoder}.
\]

In FIC-0, \(b_{encoder}=0\). The target is therefore exactly the qualified default-offset target. `process_action()` runs once per 20 ms policy interval, while `apply_action()` writes that same processed target before each of the ten 2 ms physics steps. This is a 50 Hz policy with a 500 Hz zero-order-held position command, not a target recomputed from the current joint position.

| Concern | Current implementation | Primary-source comparison | Assessment |
|---|---|---|---|
| Absolute vs relative target | `JointPositionActionCfg(..., use_default_offset=True)` | mjlab's `JointPositionAction` replaces the offset with `default_joint_pos`; its distinct `RelativeJointPositionAction` adds the action to the *current* pose. Isaac Lab defines the same split. | Correct absolute/default-offset semantics; no drift/integration of actions. |
| Per-joint scale | Six values loaded from the banked qualification artifact: approximately `[0.0557, 0.5878, 0.1446, 0.2288, 0.0557, 0.0552]` rad | Both mjlab and Isaac Lab accept a regex-to-value scale map. First-party Unitree examples often use a uniform 0.25 rad scale, but that is a robot/task heuristic rather than an API requirement. | Correct and better justified for this strike because each scale is qualification-derived. |
| Raw policy clip | `clip_actions=1.0` in the inherited runner; `RslRlVecEnvWrapper.step()` clamps before `env.step()` | Unitree's older `unitree_rl_gym` also clips policy actions before control. | Correct. Direct unwrapped environment calls intentionally bypass this layer, and a live test covers that distinction. |
| Physical target clip | Per-joint absolute Z1 joint limits are supplied as `ActionTermCfg.clip`; mjlab clips after scale and offset | mjlab 1.4.0 introduced/ships processed-action clipping at this exact point. | Correct second rail. With normal wrapped actions, qualified `default +/- scale` remains well inside these limits. |
| Decimation/hold | One processed action is applied on every one of ten substeps | mjlab, Isaac Lab, and `unitree_rl_gym` all process/clip once and apply control inside the decimation loop. | Correct zero-order hold. Later stiffness commands should use the same policy rate and hold. |
| Joint selection/order | Exact `joint1` through `joint6`; gripper omitted | Explicit joint lists are an official Isaac Lab pattern. mjlab resolves joint transmissions in the entity's natural joint order. | Correct for the current asset; the live target-name/ID test is load-bearing. |
| Gripper exclusion | No gripper action term; a sentinel gripper target survives arm actions | First-party manipulation environments normally give arm and gripper separate action terms. | Correct. Do not add a dummy seventh arm action. |
| Reset/action history | Manager reset zeros current and two prior raw-action buffers; scene reset zeros actuator target buffers; the next action is processed before physics | mjlab and Unitree code both clear action history at episode reset. | Correct. `BaseAction.reset()` alone does not clear its private processed buffer, so future code must not read that private buffer between reset and the first action. Current code does not. |
| Fixed gains | Existing native actuator groups remain at `(kp,kd)=(1000,100)` for joints 1/3/4/5/6 and `(1500,150)` for joint 2, with the same torque limits and armatures | Khadiv et al.'s FIC parametrization is desired joint position plus pre-defined fixed PD gains. | Correct FIC scientific identity; no gain-writing code is present. |

## Primary-source implementation comparison

### mjlab 1.4.0 (the actual runtime)

The released [`actions.py`](https://github.com/mujocolab/mjlab/blob/4360096b49c1bcc478668c11ce7d458db747e49d/src/mjlab/envs/mdp/actions/actions.py) implements the exact affine map used here: raw action times `scale`, plus `offset`, followed by optional processed-action clipping. `JointPositionAction` overwrites the configured offset with the entity's default joint pose and writes the resulting position target. The same file defines relative joint action separately as current pose plus processed delta. This supports the present choice of `JointPositionActionCfg`, and rules out replacing it with `RelativeJointPositionActionCfg`.

The released [`manager_based_rl_env.py`](https://github.com/mujocolab/mjlab/blob/4360096b49c1bcc478668c11ce7d458db747e49d/src/mjlab/envs/manager_based_rl_env.py) calls `process_action` once and `apply_action` inside the decimation loop. [`vecenv_wrapper.py`](https://github.com/mujocolab/mjlab/blob/4360096b49c1bcc478668c11ce7d458db747e49d/src/mjlab/rl/vecenv_wrapper.py) performs the runner-level raw clamp, while [`config.py`](https://github.com/mujocolab/mjlab/blob/4360096b49c1bcc478668c11ce7d458db747e49d/src/mjlab/rl/config.py) documents that `clip_actions=None` means no clamp. This is why both rails in the present design are meaningful: the wrapper bounds normal policy traffic, while the action term remains safe under direct evaluation calls.

One small API subtlety is visible in [`entity.py`](https://github.com/mujocolab/mjlab/blob/4360096b49c1bcc478668c11ce7d458db747e49d/src/mjlab/entity/entity.py): joint targets selected by actuator names are returned in natural entity-joint order, and the action's `preserve_order=True` flag is not forwarded for joint transmission. This is **not a present mismatch** because the contract order and Z1 natural order are both `joint1..joint6`, and `test_live_targets_ids_bias_and_exact_affine_map` proves the live IDs and names. Treat the test, not the flag, as the order guarantee if the asset changes.

### Isaac Lab 2.3.0

Isaac Lab's pinned [`joint_actions.py`](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py) uses the same affine processing, processed-action clip, default-pose overwrite, and distinct current-pose-relative class. Its [`actions_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py) explicitly documents that `use_default_offset=True` overwrites the ordinary offset with the articulation's default joint positions. Its [`manager_based_env.py`](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/envs/manager_based_env.py) also processes once and applies throughout decimation.

This is useful corroboration because mjlab intentionally follows the Isaac Lab manager/action API, but the mjlab 1.4.0 source above remains authoritative for execution.

### First-party Unitree implementations

At commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`, Unitree's official [`unitree_rl_lab` G1-29DOF config](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py) uses `JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)`, observes the last action, and uses a decimated control loop. This directly supports the default-offset position-action pattern on Unitree hardware-oriented RL code.

At commit `276801e46c5d433564f24658bac64f254b7d2d4b`, Unitree's older official [`unitree_rl_gym` controller](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py) clips raw actions, then in position mode computes

\[
\tau=K_p\left(s a+q_{default}-q\right)-K_d\dot q
\]

on every physics substep. It also clears current and previous action buffers on reset. Although that repository uses Isaac Gym and direct torque computation, the action meaning and decimation semantics match FIC-0.

## Consequences for the later VIC implementation

Khadiv, Bogdanovic, and Righetti compare three parametrizations in their primary paper, [*Learning Variable Impedance Control for Contact Sensitive Tasks*](https://arxiv.org/abs/1907.07500): direct torque, fixed-gain PD with policy-commanded desired joint positions, and variable-gain PD with policy-commanded desired joint positions and per-joint impedance. Their FIC and VIC laws are exactly the intended scientific sequence here. They couple damping to stiffness with a fixed square-root relationship, constrain learned gains to a pre-validated stable range, and introduce the next-step desired-position tracking term for both fixed and variable controllers. They report that the term changes VIC interpretability materially while the fixed-gain solution changes little. This supports testing FIC both without and with `r_tt`, then using the same term in VIC.

The safe implementation consequences are:

1. **Keep the six qualified position dimensions unchanged.** Append six bounded per-joint stiffness dimensions; do not reintroduce Cartesian IK or alter the position scale while comparing FIC and VIC.
2. **Keep the native actuator plant if the scientific comparison is intended to isolate variable gains.** In mjlab 1.4.0, [`BuiltinPositionActuator`](https://github.com/mujocolab/mjlab/blob/4360096b49c1bcc478668c11ce7d458db747e49d/src/mjlab/actuator/builtin_actuator.py) has no `set_gains`, while [`IdealPdActuator`](https://github.com/mujocolab/mjlab/blob/4360096b49c1bcc478668c11ce7d458db747e49d/src/mjlab/actuator/pd_actuator.py) does. The latter computes torque explicitly in Python; changing to it would change integration behavior in addition to adding VIC.
3. **For native gains, update the complete MuJoCo PD encoding.** mjlab's own [`pd_gains` domain-randomization function](https://github.com/mujocolab/mjlab/blob/4360096b49c1bcc478668c11ce7d458db747e49d/src/mjlab/envs/mdp/dr/actuator.py) shows the required writes for `BuiltinPositionActuator`: `gainprm[...,0]=kp`, `biasprm[...,1]=-kp`, and `biasprm[...,2]=-kd`. Updating only `gainprm` does not implement a valid position actuator.
4. **Resolve the two actuator groups explicitly.** The Z1 has one native actuator group for joints 1/3/4/5/6 and another for joint 2. Build and test a name-to-global-control-index map; never assume the six policy columns are one contiguous actuator object.
5. **Hold stiffness over the same ten substeps as position.** Process the 12-dimensional action once per policy step and apply both the target and gain schedule before every physics step. Do not give gains an accidental 500 Hz policy bandwidth.
6. **Reset safely per environment.** Clear raw action history and restore nominal gains for reset rows, or prove that the first post-reset action overwrites every relevant native gain field before any physics step. Add a partial-reset test to prevent one environment's terminal gain from leaking into its next episode.
7. **Test the square-root damping coupling and bounds.** For every joint and extreme raw action, prove finite positive `kp`, the chosen `kd = c_j sqrt(kp)` relation, exact lower/upper gain bounds, and unchanged effort limits/armature. The proportionality constants should recover the nominal Z1 damping at nominal stiffness unless a separately justified damping ratio is selected.
8. **Revisit the physical rail before encoder-bias randomization.** The present no-bias FIC equation is exact. If bias DR is later enabled, explicitly test the final command written after bias correction against the safety limits.

## Final assessment

- **Current FIC-0:** PASS; no Critical or Important mismatch.
- **Minor robustness note:** `preserve_order=True` does not control joint-transmission ordering in mjlab 1.4.0; the existing live-order test is the actual guarantee.
- **Important before VIC:** do not call `set_gains` on the current native actuator and do not swap actuator class as an unrecorded implementation convenience. Implement/test native gain-field writes or add a native API while preserving the fixed plant.
- **Important before encoder-bias domain randomization:** the current absolute clip is pre-bias; add a post-bias safety argument or rail.

No production code was changed as part of this review.
