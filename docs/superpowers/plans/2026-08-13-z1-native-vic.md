# Z1 Native VIC Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan task-by-task. Every behavior change follows the repository `tdd` skill: public-seam RED, smallest GREEN, then refactor.

**Goal:** Implement and qualify the approved 12-dimensional Z1 native variable-impedance VIC-TT prototype without changing the selected direct-reference FIC-TT treatment.

**Architecture:** Preserve the existing six-dimensional `joint_position` action term byte-for-byte and append a six-dimensional `joint_stiffness` term. The new term maps bounded policy coordinates through the Khadiv-aligned exponential gain law and writes native MuJoCo position-actuator gain/bias fields after mandatory per-world expansion. Observations, action-rate cost, and RTT continue to use only the six position channels.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, mujoco-warp 3.8.1, RSL-RL/CatPPO, pytest, Bash/Slurm, Vega A100.

## Global constraints

- Work only in `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal` on `z1-vic-prototype`, based on `faac76a1b51b6be675a46adefe477ff574077a28`.
- Preserve all Cartesian and fixed-impedance tasks and all six owner-owned untracked research/result paths.
- Stage only explicit files belonging to this plan; never use broad staging.
- Preserve the direct-reference FIC-TT reward, reset, plant, reference, `k_tt=1`, negative-term split, velocity-CaT, log-only impulse-CaT, D4, and corrected `I_ref` exactly.
- Add only VIC-TT. No VIC-0, evaluator expansion, active impulse pressure, gain penalty/observation/filter, DR, curriculum, controlled-drop work, or training before qualification.

---

### Task 0: Bank the approved design and execution plan

**Files:**
- Create: `docs/superpowers/specs/2026-08-13-z1-native-vic-design.md`
- Create: `docs/superpowers/plans/2026-08-13-z1-native-vic.md`

- [ ] Review both documents against the current code and installed mjlab API.
- [ ] Commit exactly these two files before implementation so every later review and Vega checkout carries the frozen contract.

---

### Task 1: Implement the pure gain contract test-first

**Files:**
- Create: `tests/test_variable_impedance.py`
- Create: `src/tasks/hammer/mdp/variable_impedance.py`
- Modify: `src/tasks/hammer/mdp/__init__.py`

- [ ] RED: specify `p` clipping, `C>1`, exact `C**p`, nominal parity, square-root damping coupling, shapes/dtypes/devices, monotonicity, and finite-value failures.
- [ ] GREEN: add a small pure `variable_impedance_gains` helper and immutable typed gain contract. Do not touch simulator state in the helper.
- [ ] RED: specify startup model-field requirements, name-resolved canonical control IDs, all-three-field writes, gripper/force-range invariance, per-world independence, and selected-environment reset.
- [ ] GREEN: implement `JointStiffnessAction` and its config, typed telemetry, startup expansion event, native writer, and partial reset.
- [ ] Verify `pytest -q tests/test_variable_impedance.py` and `git diff --check`; commit only Task 1 paths.

### Task 2: Make observations, action-rate, and RTT compatible without changing semantics

**Files:**
- Modify: `tests/test_joint_trackability.py`
- Create: `tests/test_action_rate_penalty.py`
- Modify: `src/tasks/hammer/mdp/trackability.py`
- Modify: `src/tasks/hammer/mdp/rewards.py`

- [ ] RED: require RTT to accept only the one-term FIC or exact ordered two-term VIC signature while returning the unchanged six-joint sum-of-squares at `t+1`; reject reversed, renamed, or extra terms.
- [ ] GREEN: resolve only the `joint_position` target term; do not alter formula, sign, outer reward-manager `dt`, or `k_tt`.
- [ ] RED: show gain-only manager-action changes contribute zero action-rate cost when `action_name="joint_position"`; prove `None` preserves every existing task's full-action behavior and unknown names/shapes fail closed.
- [ ] GREEN: add the narrow optional action-term selector by slicing the manager's current and previous global action buffers.
- [ ] Verify focused RTT, negative split, and action-rate tests; commit only Task 2 paths.

### Task 3: Register the matched direct-reference VIC-TT task coherently

**Files:**
- Modify: `tests/test_joint_position_config.py`
- Modify: `tests/test_configs.py`
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`

- [ ] RED: require the exact new task ID, ordered action terms `joint_position` then `joint_stiffness`, action width 12, observation width 40, `C=1.25`, zero reset noise, and exact FIC-TT equality outside the gain-action, startup-event, and position-only selector seams.
- [ ] GREEN: install the startup expansion event and stiffness action onto a fresh direct-reference FIC-TT config; set both action observations and the action-rate term to `action_name="joint_position"` in the same coherent change. Leave every existing task factory unchanged.
- [ ] RED/GREEN: prove the live actuator mapping excludes the gripper and preserves nominal gains/force limits.
- [ ] Verify the focused config and live construction tests; commit only Task 3 paths.

### Task 4: Add fail-closed VIC metadata and genuine CPU smokes

**Files:**
- Modify: `tests/test_smoke_joint_position_fixed.py`
- Modify: `src/tasks/hammer/rl/runner.py`
- Modify: `scripts/smoke_joint_position_fixed.py`
- Modify: `scripts/smoke_cat_soft.py`

- [ ] RED: freeze the exact 12D action order, mapping family, `C`, bounds, nominal gains, native fields, position-only history/cost, and RTT identity; reject mutated or unknown pairs. Add a real runner save, ONNX export, and decoded metadata readback test so export failures cannot hide behind the runner warning path.
- [ ] GREEN: derive metadata from the live manager and typed stiffness action rather than duplicating mutable configuration.
- [ ] RED/GREEN: add a real CPU reset/step smoke and a genuine 24-step, one-update CatPPO smoke with finite checkpoint for the VIC task, preserving all FIC smoke assertions.
- [ ] Verify focused metadata/live tests and both CPU smokes; commit Task 4 paths.

### Task 5: Qualify isolation, nominal parity, and bounded authority

**Files:**
- Modify: `scripts/smoke_joint_position_fixed.py`
- Modify: `tests/test_smoke_joint_position_fixed.py`

- [ ] RED/GREEN: extend the existing joint-position live smoke with deterministic two-world opposite-corner, selected-reset, nominal-`p=0`, and alternating-command tapes; no separate qualification script, campaign framework, or result JSON layer.
- [ ] Compare FIC-TT with VIC-TT at `p=0`: exact equality for applied targets/native gains and CPU trace `atol=rtol=1e-6` for qpos/qvel, RTT, contact, and CaT tensors. Instrument all ten substeps to prove q targets and gains remain paired.
- [ ] Run only the frozen `C=1.25` authority tape with a declared nonsaturated position error; prove expected force ordering, finiteness, force-limit compliance, and reset isolation. Defer any larger `C` to a separate approved study.
- [ ] Verify script tests and a live CPU qualification; commit Task 5 paths.

### Task 6: Review and complete CPU/CUDA verification

**Files:**
- Create or modify only if needed: `scripts/slurm/vega_vic_smoke.sbatch`
- Modify corresponding launcher test if a launcher is added.

- [ ] Run all focused VIC/FIC, reward, reference, contact, RTT, CaT, metadata, and smoke tests plus `validate_rewards.py`, `verify_contact_sensor.py`, `verify_reward_setup.py`, and `playback_reference.py`.
- [ ] Request Opus, Gemini, DeepSeek, and repository Standards/Spec reviews plus an adversarial runtime/scientific review. Apply concrete Critical/Important fixes test-first, then bounded rereview.
- [ ] Run the full CPU suite exactly once on the reviewed final CPU candidate.
- [ ] Commit and push that reviewed candidate SHA to `z1-vic-prototype` so Vega can fetch it.
- [ ] On Vega, fetch the pushed SHA into a clean detached worktree. First run the same nominal CUDA arm twice and set each tensor's parity tolerance to `max(1e-6, 2 * max_abs_same_arm_repeat_delta)`; freeze those values before running any FIC-VIC comparison. Then run the live VIC authority smoke, FIC-VIC nominal comparison, and genuine one-iteration CatPPO smoke on an A100. Record exact code/asset revisions, derived repeatability bounds, and Slurm outcome.
- [ ] If CUDA requires a production-code fix, test and review the new SHA, rerun the full CPU suite on that new reviewed SHA, push it, and rerun the complete CUDA gate. Add only a final evidence/docs commit afterward if needed.

### Task 7: Deferred overnight engineering canary

- [ ] Do not launch training from this prototype plan. A subsequent approved plan must name and test a minimal fail-closed VIC launcher, pin clean revisions and the exact task/action/gain contract, and define the permitted telemetry boundary.
- [ ] Stop after the verified CPU/CUDA prototype and one-iteration learning smoke. Report decisions, tests, jobs, hashes, and remaining scientific questions for the separate overnight canary plan.
