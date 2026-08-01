# Joint-Position Trackability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a qualified six-joint desired-position action interface and compare fixed-gain J0 against J-TT with Khadiv's one-step command-trackability term.

**Architecture:** Reuse mjlab's `JointPositionActionCfg` with explicit default offsets, scales, clips, and six-joint order. A deterministic CPU tool converts the legal Cartesian reference's first 500 Hz target in each 20 ms interval into a causal zero-order-held joint tape; the joint branch exists only if all 16 reset replays qualify. `r_tt` is a pure post-decimation reader of the applied target and resulting state.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0 `JointPositionActionCfg`, MuJoCo 3.8.1, pytest, existing first-strike evaluation pipeline.

## Global Constraints

- Start from a new parent-repo worktree under `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/joint-position-fic`, not `/private/tmp`, so the sibling asset path resolves.
- Base it on the reviewed Cartesian shared-geometry commit produced by `2026-08-01-cartesian-waypoint-guideline.md`; use `superpowers:using-git-worktrees`.
- Fixed impedance only; never call `set_gains`.
- Use only arm joints `joint1` through `joint6`; do not command `jointGripper`.
- Use `JointPositionActionCfg(use_default_offset=True)`; never use `RelativeJointPositionActionCfg`.
- Keep policy/control rates at 50 Hz/500 Hz and zero-order hold each policy target for ten physics substeps.
- Keep manufacturer `IMP_J_LIMIT`, actuator gains, `imp_max_p=0.0`, physics, solver, reset distribution, and F8 task rewards unchanged.
- Do not edit installed packages and do not warm-start incompatible three-action Cartesian policies.
- Joint action scales and `k_tt` are generated from the full reset bank before training and committed as named evidence; no value is tuned after policy outcomes.
- Stop the joint branch if any reset seed 1000--1015 fails qualification.
- No spatial gate reward is active in J0 or J-TT; joint-gate interaction is a later conditional experiment.

This plan intentionally stops after a clean CUDA smoke. A separate preregistered
execution plan is written only after the 16-reset action qualification, calibrated
`k_tt`, production task identity, runtime, and trace schema are frozen. That later
plan covers excluded seeds 0/1, the continuation gate, confirmatory seeds 8--15,
fixed-reset MP4s, desired/actual joint plots, statistics, and the dated result note.

---

## File Map

- Create `evaluation/joint_position/derive_joint_target_tape.py`: record causal DiffIK targets, derive scales, replay all resets, and write immutable JSON/CSV evidence.
- Create `tests/test_joint_target_tape.py`: causal reduction, scale derivation, and qualification decisions.
- Create `src/tasks/hammer/config/z1/data/z1_joint_target_qualification.json`: generated scale/clip and qualification artifact consumed by the task config.
- Modify `src/tasks/hammer/config/z1/env_cfgs.py`: optional direct joint-position action and optional `r_tt`.
- Modify `src/tasks/hammer/config/z1/__init__.py`: register J0/J-TT.
- Modify `src/tasks/hammer/rl/runner.py`: emit typed metadata for either action interface.
- Modify `tests/test_configs.py`: exact joint config and isolation tests.
- Modify `tests/test_env.py`: 6-action/45-observation integration checks.
- Create `src/tasks/hammer/mdp/trackability.py`: pure target-error reader and calibrated reward.
- Modify `src/tasks/hammer/mdp/__init__.py`: export trackability functions.
- Create `evaluation/joint_position/calibrate_trackability.py`: derive and bank `k_tt` from all legal resets.
- Create `src/tasks/hammer/config/z1/data/z1_trackability_calibration.json`: generated coefficient/evidence consumed by the task config.
- Create `tests/test_joint_trackability.py`: indexing, dt scaling, and calibration gates.
- Modify `scripts/eval_impulse.py`: explicit J0/J-TT contract and six-action/joint-target traces.
- Modify `tests/test_eval_impulse_hook.py`: joint mutation tests.
- Modify `evaluation/analysis/guideline_campaign.py`: episode-first joint RMSE endpoint and joint plots.
- Modify `tests/test_guideline_campaign.py`: trackability aggregation tests.

---

### Task 1: Causal 500 Hz-to-50 Hz target reduction and qualification rules

**Files:**
- Create: `evaluation/joint_position/derive_joint_target_tape.py`
- Create: `tests/test_joint_target_tape.py`

**Interfaces:**
- Produces: `reduce_first_target_per_interval(targets, decimation=10) -> np.ndarray[T, 6]`.
- Produces: `derive_scale_by_joint(tape, q_default, reset_headroom=0.05, margin=1.10) -> np.ndarray[6]`.
- Produces: `qualifies_reset(row) -> bool` and `qualifies_bank(rows) -> bool` requiring exactly seeds 1000--1015.

- [ ] **Step 1: Write failing pure-function tests**

```python
def test_reduction_uses_first_not_future_target():
    targets = np.arange(20 * 6, dtype=np.float64).reshape(20, 6)
    reduced = reduce_first_target_per_interval(targets, decimation=10)
    np.testing.assert_array_equal(reduced, targets[[0, 10]])

def test_scale_includes_reset_headroom_and_margin():
    tape = np.array([[0.2, -0.1], [0.4, 0.3]])
    q_default = np.array([0.1, 0.0])
    expected = 1.10 * (np.array([0.3, 0.3]) + 0.05)
    np.testing.assert_allclose(derive_scale_by_joint(tape, q_default), expected)
```

Also require exactly six columns, finite values, no missing reset seed, contact, positive nail progress, zero target saturation, no physical limit violation, and peak qvel `<=3.1415`.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_joint_target_tape.py -q
```

- [ ] **Step 3: Implement only pure reduction/decision functions and CLI parsing**

The source tape must read `robot.data.joint_pos_target` immediately after the first `action_manager.apply_action()` of each source control interval. Never downsample the realized joint trajectory and never use the interval-final target.

- [ ] **Step 4: Run focused tests and commit**

```bash
git add evaluation/joint_position/derive_joint_target_tape.py tests/test_joint_target_tape.py
git commit -m "test(hammer): define causal joint target qualification"
```

---

### Task 2: Dynamic qualification replay before production wiring

**Files:**
- Modify: `evaluation/joint_position/derive_joint_target_tape.py`
- Create: `src/tasks/hammer/config/z1/data/z1_joint_target_qualification.json`
- Modify: `tests/test_joint_target_tape.py`

**Interfaces:**
- CLI writes a schema-versioned artifact containing target tape SHA-256, ordered joint names, defaults, physical clips, derived scales, all 16 per-reset outcomes, code/asset revisions, dt, decimation, and decision.

- [ ] **Step 1: Add a failing serialization round-trip test**

Require sorted keys, finite JSON numbers, joint order exactly `joint1`...`joint6`, exactly 16 reset rows, and recomputed content SHA equality. Reject `-dirty` revisions.

- [ ] **Step 2: Implement temporary in-memory joint action construction**

Build the production Cartesian reference env, record the causal target tape, derive scales, then create a second config whose sole action is:

```python
JointPositionActionCfg(
    entity_name="robot",
    actuator_names=ARM_ACTUATOR_NAMES,
    scale=scale_by_actuator,
    clip=physical_limit_by_actuator,
    use_default_offset=True,
)
```

Assert runtime target order and action dimension before playback. Apply normalized actions reconstructed from the tape and zero-order hold each action for one control step.

- [ ] **Step 3: Run all 16 reset seeds on CPU**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python evaluation/joint_position/derive_joint_target_tape.py --seeds 1000:1016 --out src/tasks/hammer/config/z1/data/z1_joint_target_qualification.json
```

Expected: `decision=PASS`, 16/16 contact, 16/16 positive nail progress, zero saturation/limit violations/NaNs, and every peak qvel <=3.1415. If not, stop the entire joint plan and report that this 50 Hz ZOH parameterization failed; do not change gains, limits, or reset selection.

- [ ] **Step 4: Re-run focused tests against the generated artifact**

Run Task 1 pytest. Expected: PASS.

- [ ] **Step 5: Commit qualification evidence**

```bash
git add evaluation/joint_position/derive_joint_target_tape.py src/tasks/hammer/config/z1/data/z1_joint_target_qualification.json tests/test_joint_target_tape.py
git commit -m "test(hammer): qualify direct joint target playback"
```

---

### Task 3: Production J0 action interface and metadata

**Files:**
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Modify: `src/tasks/hammer/rl/runner.py`
- Modify: `tests/test_configs.py`
- Modify: `tests/test_env.py`

**Interfaces:**
- Extends `z1_hammer_env_cfg(..., joint_position: bool=False, trackability: bool=False)`.
- Registers `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-J0`.
- Metadata records action type, dimension, ordered target names, scale, clip, default-offset mode, dt, and decimation.

- [ ] **Step 1: Write failing config and metadata tests**

Require `trackability=True` without `joint_position=True` to raise. For J0 require one `JointPositionActionCfg`, action dimension six, exact ordered joints/scales/clips loaded from the qualified artifact, `use_default_offset=True`, no DiffIK term, no `r_tt`, no `r_gate`, fixed gains, and `imp_max_p=0`. Require all old Cartesian metadata tests to remain byte-for-byte valid.

- [ ] **Step 2: Run config tests and verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_configs.py -q
```

- [ ] **Step 3: Add a small validated artifact loader**

Load only the committed schema-versioned JSON. Check its decision, revisions, joint order, finite scales/clips, target-tape digest, and 16-reset PASS before returning mappings. Do not silently regenerate it at training time.

- [ ] **Step 4: Replace only the action term for J0**

Start from the same F8+guideline config as C0, remove `ik_hammer_head`, install the built-in joint action, and retain the shared tracker/observations. Update runner metadata by explicit type dispatch over `DifferentialIKAction` and `JointPositionAction`; an unknown action type must raise rather than produce generic metadata.

- [ ] **Step 5: Add CPU environment checks**

Construct J0 and assert action `(1, 6)`, actor/critic `(1, 45)`, resolved joint order, finite reset/step observations, target clipping, and that old Cartesian checkpoints fail dimensional loading clearly.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_configs.py tests/test_env.py -q
git add src/tasks/hammer/config/z1/env_cfgs.py src/tasks/hammer/config/z1/__init__.py src/tasks/hammer/rl/runner.py tests/test_configs.py tests/test_env.py
git commit -m "feat(hammer): add qualified joint-position action"
```

---

### Task 4: Khadiv one-step trackability term and frozen calibration

**Files:**
- Create: `src/tasks/hammer/mdp/trackability.py`
- Modify: `src/tasks/hammer/mdp/__init__.py`
- Create: `evaluation/joint_position/calibrate_trackability.py`
- Create: `src/tasks/hammer/config/z1/data/z1_trackability_calibration.json`
- Create: `tests/test_joint_trackability.py`

**Interfaces:**
- Produces: `joint_target_squared_error(env, robot_cfg) -> Tensor[B]`.
- Produces: `joint_target_rmse(env, robot_cfg) -> Tensor[B]`.
- Produces: `joint_trackability_reward(env, robot_cfg, k_tt) -> Tensor[B]` equal to `-k_tt * squared_error`.

- [ ] **Step 1: Write failing indexing and formula tests**

Use a fake robot with a known applied target and post-step position:

```python
q_des = torch.tensor([[0.2, 0.0, -0.1, 0.3, 0.1, 0.0]])
q_next = torch.tensor([[0.1, 0.0, -0.1, 0.1, 0.0, 0.0]])
expected_sq = ((q_des - q_next) ** 2).sum(dim=1)
torch.testing.assert_close(joint_target_squared_error(env, ARM_CFG), expected_sq)
torch.testing.assert_close(joint_trackability_reward(env, ARM_CFG, 2.0), -2 * expected_sq)
```

Test that the term reads public `joint_pos_target`, uses only six arm joints, is evaluated after ten substeps, and that reward-manager dt scaling is reported separately from the unscaled term.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_joint_trackability.py -q
```

- [ ] **Step 3: Implement the three pure readers**

No class state and no action-manager private fields. Validate target/position shape equality and finite `k_tt`.

- [ ] **Step 4: Implement calibration from the qualified 16-reset tape**

Compute:

```python
k_tt = 0.1 / np.quantile(all_squared_errors, 0.90)
```

Fail if q90 is non-finite or below `1e-8`. For every reset, compute the unscaled cumulative cost and fail if the worst exceeds 5.0. Write q90, `k_tt`, per-reset dose, returned-dose equivalents (`*0.02`), revisions, and source-artifact digest.

- [ ] **Step 5: Run calibration and tests**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python evaluation/joint_position/calibrate_trackability.py --qualification src/tasks/hammer/config/z1/data/z1_joint_target_qualification.json --out src/tasks/hammer/config/z1/data/z1_trackability_calibration.json
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_joint_trackability.py -q
```

Expected: finite `k_tt`, q90 >=1e-8, worst unscaled episode dose <=5.0.

- [ ] **Step 6: Commit implementation and calibration**

```bash
git add src/tasks/hammer/mdp/trackability.py src/tasks/hammer/mdp/__init__.py evaluation/joint_position/calibrate_trackability.py src/tasks/hammer/config/z1/data/z1_trackability_calibration.json tests/test_joint_trackability.py
git commit -m "feat(hammer): add calibrated joint trackability term"
```

---

### Task 5: Register J-TT and prove treatment isolation

**Files:**
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Modify: `tests/test_configs.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write a failing exact-diff test**

Construct J0 and J-TT and assert that canonicalized configs differ only by one reward term:

```text
name=r_tt
func=joint_trackability_reward
weight=1.0
k_tt=<exact committed calibration value>
robot joints=joint1..joint6
```

Both arms must have identical actions, observations, metrics, resets, task rewards, gains, impulse limits, `imp_max_p`, and PPO config.

- [ ] **Step 2: Register J-TT minimally**

Add `r_tt` only when `trackability=True`, loading `k_tt` from the committed calibration artifact. Register the J-TT task beside J0.

- [ ] **Step 3: Verify reward timing in a real CPU environment**

Apply a known joint target, save the applied target before stepping, step once, and verify the logged unweighted term equals `-k_tt * ||q_des(t)-q(t+1)||^2`; ensure it remains active on a contact step.

- [ ] **Step 4: Run focused tests and commit**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_configs.py tests/test_env.py tests/test_joint_trackability.py -q
git add src/tasks/hammer/config/z1/env_cfgs.py src/tasks/hammer/config/z1/__init__.py tests/test_configs.py tests/test_env.py
git commit -m "feat(hammer): register joint trackability study"
```

---

### Task 6: Fail-closed joint evaluation and episode-first RMSE

**Files:**
- Modify: `scripts/eval_impulse.py`
- Modify: `tests/test_eval_impulse_hook.py`
- Modify: `evaluation/analysis/guideline_campaign.py`
- Modify: `tests/test_guideline_campaign.py`

- [ ] **Step 1: Write failing joint-contract mutation tests**

Reject before rollout: wrong task ID, action width not six, wrong class, permuted joints, changed scale/clip/default offset, changed gains/timing, `imp_max_p != 0`, r_tt in J0, missing/altered r_tt in J-TT, unexpected `r_gate`, inherited weight overrides, dirty provenance, or a qualification/calibration digest mismatch. Preserve the old Cartesian signature constant unchanged.

- [ ] **Step 2: Implement explicit J0/J-TT contracts and variable-width tapes**

Dispatch on the listed task, not action width alone. Persist six-action raw/processed tapes, six desired/actual joint traces, saturation, and existing 500 Hz physical/contact/impulse traces. Reject any trace whose declared and actual widths differ.

- [ ] **Step 3: Add failing episode-first aggregation tests**

For every control step use:

```python
rmse_t = np.sqrt(np.mean((q_des_t - q_next_t) ** 2, axis=-1))
T_ep = np.quantile(rmse_t, 0.90)
T_seed = np.quantile(per_episode_T, 0.90)
```

Prove with unequal episode lengths that steps are not pooled. Require each sampled episode exactly once.

- [ ] **Step 4: Implement analysis and plots**

Add `q90_joint_target_rmse_rad_sampled`, per-joint RMSE, target saturation, desired/actual time-series plots, and all shared success/speed/depth/impulse/qvel/contact-class secondary outputs. The only confirmatory joint contrast is J-TT versus J0.

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_eval_impulse_hook.py tests/test_guideline_campaign.py -q
git add scripts/eval_impulse.py tests/test_eval_impulse_hook.py evaluation/analysis/guideline_campaign.py tests/test_guideline_campaign.py
git commit -m "feat(eval): add strict joint trackability evaluation"
```

---

### Task 7: Full CPU gate, independent review, and bounded CUDA smoke

**Files:**
- Modify only named files justified by a failing test or review finding.
- Create: `docs/results/2026-08-01_joint_position_cpu_gate.md`

- [ ] **Step 1: Re-run both generated-artifact validators**

Require their digests, 16-reset membership, joint order, scales/clips/coefficient, and PASS decisions to match production config exactly. Preserve qualification, calibration-builder, and training revisions as separate provenance fields; do not require the earlier artifact-building revision to equal the later training revision.

- [ ] **Step 2: Run all mandatory CPU gates**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_joint_target_tape.py tests/test_joint_trackability.py tests/test_hammer_guideline.py tests/test_guideline_campaign.py tests/test_configs.py tests/test_env.py tests/test_impact_progress_reward.py tests/test_impulse_bound.py tests/test_impulse_constraint.py tests/test_delivered_impulse_reward.py tests/test_cat_soft_hook.py -q
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/validate_rewards.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/verify_contact_sensor.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/verify_reward_setup.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/playback_reference.py
```

- [ ] **Step 3: Request independent code and scientific-isolation reviews**

Resolve every Critical/Important issue with a focused failing test first, then re-run Step 2.

- [ ] **Step 4: Bank and commit the CPU gate**

Record exact revisions, generated-artifact hashes, all 16 playback rows, calibrated coefficient/dose, environment signatures, production line count, commands, and honest proven/open conclusions.

- [ ] **Step 5: Deploy cleanly and run J0/J-TT CUDA smokes**

Push named commits and deploy by `git fetch`/clean checkout. Run one GPU per arm. Require correct six-action/45-observation identity, nonzero finite r_tt only in J-TT, saturation <5%, `impossible_success_n==0`, `lambda_dead_n==0`, qvel legality reporting, and clean provenance.

- [ ] **Step 6: Stop at the training checkpoint**

Report smoke results and exact excluded-seed commands. Do not launch seeds 0/1 or confirmatory seeds 8--15 until the owner approves the frozen revision and smoke evidence.
