# Z1 Joint-Position Lean Task 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the two six-action fixed-impedance joint tasks with fixed `k_tt=1`, essential live policy metadata, and real CPU/CUDA/CatPPO smoke gates.

**Architecture:** Keep the qualified joint-action artifact as the single source for action order, scale, clips, and its payload hash. Remove the separate RTT calibration artifact and install `r_tt` directly with `k_tt=1.0`; dispatch runner metadata by the resolved live action class; reuse the real manager and CatPPO paths for bounded smokes.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, RSL-RL/CatPPO, pytest, Vega A100.

## Global Constraints

- Work only in `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal` on `z1-joint-position-fixed`.
- Preserve every Cartesian task and the current fixed gains/efforts, P guidance, D4 reward, substep velocity CaT, and log-only impulse CaT.
- FIC-0 has no `r_tt`; FIC-TT uses `joint_trackability_cost`, weight `-1.0`, and exactly `k_tt=1.0`; `r_tt` remains outside CaT's scaled positive return.
- Keep the six absolute default-offset actions for `joint1` through `joint6`; never command the gripper or add gain actions.
- Preserve all owner-owned untracked research/results and stage only explicit Task-5 paths.
- Do not add evaluator/provenance infrastructure, sweeps, controlled-drop work, curriculum/domain randomization, gain changes, or VIC.
- Run the full CPU suite exactly once on the final revision; run CUDA only after the implementation revision is committed and deployed cleanly.

---

### Task 1: Replace RTT Q90 calibration with fixed `k_tt=1`

**Files:**
- Create: `docs/superpowers/plans/2026-08-11-z1-joint-position-lean-task5.md`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Modify: `src/tasks/hammer/config/z1/joint_position_contract.py`
- Delete: `evaluation/joint_position/calibrate_trackability.py`
- Delete: `src/tasks/hammer/config/z1/data/z1_joint_trackability_stage1.json`
- Modify: `tests/test_joint_position_config.py`
- Modify: `tests/test_joint_trackability.py`

**Interfaces:**
- Retain: `joint_trackability_cost(env, robot_cfg, k_tt) -> torch.Tensor`
- Remove: `load_joint_trackability_contract(...)`, `derive_k_tt(...)`, and all RTT-calibration payload/CLI APIs.
- Produce: FIC-TT reward params `{"robot_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True), "k_tt": 1.0}`.

- [ ] **Step 1: RED — change the config contract test first**

  Replace the artifact-derived expectation with the literal owner decision:

  ```python
  assert term.weight == -1.0
  assert term.params["k_tt"] == 1.0
  ```

  Run:

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
    tests/test_joint_position_config.py::test_fictt_installs_only_the_strict_banked_trackability_cost -q
  ```

  Expected: FAIL because runtime still supplies `1.9360715154399775`.

- [ ] **Step 2: GREEN — install the fixed gain and remove calibration-only code**

  Set `"k_tt": 1.0` directly in `_install_joint_trackability_cost`; remove the loader, contract dataclass/schema, calibration executable, generated artifact, and only the Q90/calibration tests. Retain formula, six-joint indexing, terminal/contact timing, dt separation, invalid-gain, reward-sign, and CaT-negative-split tests.

- [ ] **Step 3: Verify and commit Task 1**

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_joint_trackability.py tests/test_joint_position_config.py \
    tests/test_cat_soft_hook.py tests/test_env.py
  git diff --check
  ```

  Stage only the seven paths above and commit `fix(hammer): use fixed joint trackability gain`.

---

### Task 2: Add essential live joint-policy metadata

**Files:**
- Modify: `src/tasks/hammer/rl/runner.py`
- Create: `tests/test_smoke_joint_position_fixed.py`

**Interfaces:**
- Change: `_get_hammer_metadata(env, run_path: str, *, raw_policy_clip: float) -> dict`
- Preserve the Cartesian return dictionary exactly.
- For `JointPositionAction`, return ONNX-safe live fields for action identity/dimension, target names/IDs, actuator names, default offsets, scale, physical clips, wrapper clip, timing, fixed gain/effort signature, qualification payload hash, and `r_tt_enabled`/`r_tt_k_tt` (`not_applicable` or `1.0`).

- [ ] **Step 1: RED — add live Cartesian and joint metadata tests**

  Instantiate one live environment per action class. Assert the Cartesian dictionary is exactly:

  ```python
  {
      "run_path": "test-run",
      "action_type": "ik_delta_pos",
      "frame_name": action.cfg.frame_name,
      "delta_pos_scale": action.cfg.delta_pos_scale,
      "joint_names": list(robot.joint_names),
      "observation_names": env.observation_manager.active_terms["actor"],
  }
  ```

  For FIC-0/FIC-TT, assert all required joint fields match resolved live tensors/configs and that only FIC-TT reports `r_tt_enabled=True`, `r_tt_k_tt=1.0`. Add rejection cases for unknown/multiple actions and wrong joint order.

  Run the new metadata tests and observe the joint dispatch fail against the Cartesian-only helper.

- [ ] **Step 2: GREEN — implement explicit action-class dispatch**

  Branch on `DifferentialIKAction` and `JointPositionAction`; validate finite six-joint values and derive the fixed actuator signature from the live robot actuator configs. In `save()`, construct/validate metadata outside the ONNX serialization warning boundary and pass the wrapper-owned clip value explicitly.

- [ ] **Step 3: Verify and commit Task 2**

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_smoke_joint_position_fixed.py tests/test_joint_position_config.py tests/test_env.py
  git diff --check
  ```

  Stage only `src/tasks/hammer/rl/runner.py` and `tests/test_smoke_joint_position_fixed.py`; commit `feat(hammer): add live joint policy metadata`.

---

### Task 3: Add the lean live-manager and genuine CatPPO smokes

**Files:**
- Create: `scripts/smoke_joint_position_fixed.py`
- Modify: `scripts/smoke_cat_soft.py`
- Modify: `tests/test_smoke_joint_position_fixed.py`

**Interfaces:**
- Produce: `run_checks(task, device="cpu", num_envs=8, steps=3) -> list[tuple[str, bool, str]]` for exactly FIC-0/FIC-TT.
- Produce: a task-selectable `run_smoke(...)` in `smoke_cat_soft.py` that builds the real registered runner, asserts `CatPPO`/`CatRolloutStorage`, six actions and 47 observations, runs one update, and rejects non-finite learned/checkpoint tensors.

- [ ] **Step 1: RED — add the importable live-smoke test and exercise the missing CatPPO task option**

  Parametrize the live test over FIC-0/FIC-TT and require every returned check to pass. Before changing `smoke_cat_soft.py`, run it with `--task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed` and confirm argument parsing rejects the unsupported option.

- [ ] **Step 2: GREEN — implement only the lean live checks**

  Check finite reset/steps, `(N,47)` actor/critic observations, six actions, exact order/default offset, unchanged gain/effort signature and no gain action, P=`8`, D4=`4`, substep velocity CaT with a synthetic positive delta, measured/log-only impulse CaT, arm-specific `r_tt` sign/value/CaT split, and successful metadata construction. Do not replay the 16-row tape or duplicate affine/clipping/reset/gripper construction tests.

- [ ] **Step 3: GREEN — parameterize the existing real CatPPO smoke**

  Preserve its default historical task, add the two FIC task IDs as accepted choices, and use the same real runner/rollout/update path for `--iters 1`.

- [ ] **Step 4: Focused verification and per-arm live/training smokes**

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_joint_action_qualification.py tests/test_joint_position_config.py \
    tests/test_joint_trackability.py tests/test_cat_soft_hook.py tests/test_env.py \
    tests/test_smoke_joint_position_fixed.py
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/smoke_joint_position_fixed.py --task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed --device cpu --num-envs 8 --steps 3
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/smoke_joint_position_fixed.py --task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed-TT --device cpu --num-envs 8 --steps 3
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/smoke_cat_soft.py --task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed --device cpu --num-envs 8 --iters 1
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/smoke_cat_soft.py --task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed-TT --device cpu --num-envs 8 --iters 1
  ```

  Commit the three explicit paths as `test(hammer): add lean joint policy smoke gates`.

---

### Task 4: Final verification, independent review, clean CUDA gate, and push

**Files:**
- Modify only concrete correctness fixes identified by review, with a failing regression test first.

- [ ] **Step 1: Review the completed diff**

  Review `a26cd87...HEAD` independently with Opus, Gemini, DeepSeek, and the repository two-axis Standards/Spec review against `docs/superpowers/specs/2026-08-11-z1-joint-position-lean-task5-design.md`. Fix concrete Critical/Important correctness findings test-first; do not implement speculative scope expansions. Run focused covering tests after any fix.

- [ ] **Step 2: Apply concrete findings and run the full CPU suite once on the final tree**

  Add a failing regression test before each correctness fix, then run its focused covering tests. Once reviews and fixes are resolved, run the release gate exactly once:

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q
  git diff --check
  ```

  Confirm explicit staged paths exclude all owner-owned untracked files.

- [ ] **Step 3: Commit/push, deploy the exact revision cleanly to Vega, and run one CUDA invocation covering both arms**

  Push `z1-joint-position-fixed`, fast-forward the clean Vega checkout to the exact commit, verify clean code and asset checkouts, then run:

  ```bash
  PYTHONPATH=. .venv/bin/python scripts/smoke_joint_position_fixed.py --task all --device cuda:0 --num-envs 8 --steps 3
  ```

  Record the pushed commit and CUDA job/exit evidence. Stop before the controlled-drop experiment.
