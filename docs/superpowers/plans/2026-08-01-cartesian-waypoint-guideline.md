# Cartesian Waypoint Guideline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an always-on, treatment-invariant straight-line waypoint tracker and compare fresh Cartesian F8 controls with and without one-shot ordered gate progress.

**Architecture:** A single per-substep `WaypointProgressTracker` owns all mutable geometry/progress state and is installed in both arms. Observations and reward are pure readers. The first campaign rewards six ordered virtual disk crossings only; the 5 mm-radius corridor is measured but neither rewarded nor enforced.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0 manager terms, MuJoCo 3.8.1, pytest, existing `eval_impulse.py` trace format.

## Global Constraints

- Work from a new `cartesian-guideline-fic` worktree created from reviewed commit `7a6a124` using `superpowers:using-git-worktrees`.
- Fixed impedance only; never call `set_gains`.
- `IMP_J_LIMIT` stays `[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]` N·m·s.
- `imp_max_p` stays `0.0`; the impulse constraint remains log-only.
- Keep `delta_pos_scale=0.15`, physics `dt=0.002`, decimation `10`, and all F8 weights unchanged outside `r_gate`.
- Do not edit installed `mjlab`, `rsl_rl`, `mujoco`, or `mujoco_warp`.
- Initial gate radius is `0.015 m`; corridor radius is `0.005 m`; six gates use fractions `i/7`, `i=1,...,6`.
- Add no tube reward, hard-tube termination, joint action, `r_tt`, or VIC code in this plan.
- Keep new production code for the shared geometry, readers, and wiring near the approved approximately-250-line simplicity budget; stop and explain before exceeding it.
- Every behavior change is TDD: failing focused test, minimal implementation, passing focused test, named-file commit.

This plan intentionally stops after a clean CUDA smoke. The excluded-seed and
confirmatory training/evaluation launch is a separate preregistered execution plan,
written only after the production task identity, runtime, and artifact schema are
known from this implementation. That later plan covers seeds 0/1, the frozen
continuation rule, seeds 8--15, fixed-reset MP4s, trajectory grids, statistics, and
the dated result note; none of those scientific outcomes are silently selected here.

---

## File Map

- Create `src/tasks/hammer/mdp/guideline.py`: pure line geometry plus the sole stateful waypoint tracker and pure readers.
- Modify `src/tasks/hammer/mdp/__init__.py`: export the tracker/readers.
- Modify `src/tasks/hammer/config/z1/env_cfgs.py`: add `guideline` and `gate_reward` wiring after `first_strike`.
- Modify `src/tasks/hammer/config/z1/__init__.py`: register `C0` and `C-Gate` tasks.
- Create `tests/test_hammer_guideline.py`: pure geometry, state lifecycle, and reward-reader unit tests.
- Modify `tests/test_configs.py`: treatment-isolation and fail-closed config tests.
- Modify `tests/test_env.py`: exact 42-value actor/critic observation construction check.
- Create `evaluation/guideline/qualify_reference.py`: 16-reset scripted qualification and x-z/x-y geometry plot.
- Create `tests/test_guideline_qualification.py`: qualification schema and decision-rule tests.
- Modify `scripts/eval_impulse.py`: explicit C0/C-Gate task contracts and guideline trace fields.
- Modify `tests/test_eval_impulse_hook.py`: mutation tests for the new strict contract.
- Create `evaluation/analysis/guideline_campaign.py`: episode-first straightness statistics and plots.
- Create `tests/test_guideline_campaign.py`: failure-value, episode aggregation, and seed aggregation tests.

---

### Task 1: Pure straight-line geometry and ordered swept gates

**Files:**
- Create: `src/tasks/hammer/mdp/guideline.py`
- Create: `tests/test_hammer_guideline.py`

**Interfaces:**
- Produces: `project_to_reference(points, entry, nail) -> tuple[s, d_perp]`.
- Produces: `advance_ordered_gates(prev, curr, entry, nail, next_gate, *, num_gates=6, radius_m=0.015) -> tuple[count, next_gate]`.
- Produces: constants `GUIDELINE_NUM_GATES=6`, `GUIDELINE_GATE_RADIUS_M=0.015`, `GUIDELINE_CORRIDOR_RADIUS_M=0.005`.

- [ ] **Step 1: Write failing tensor-geometry tests**

```python
def test_projection_returns_progress_and_perpendicular_distance():
    entry = torch.tensor([[0.0, 0.0, 1.0]])
    nail = torch.tensor([[0.0, 0.0, 0.0]])
    points = torch.tensor([[0.003, 0.004, 0.50]])
    s, d = project_to_reference(points, entry, nail)
    torch.testing.assert_close(s, torch.tensor([0.5]))
    torch.testing.assert_close(d, torch.tensor([0.005]))

def test_swept_crossing_consumes_multiple_ordered_gates():
    entry = torch.tensor([[0.0, 0.0, 0.7]])
    nail = torch.tensor([[0.0, 0.0, 0.0]])
    count, index = advance_ordered_gates(
        torch.tensor([[0.0, 0.0, 0.65]]),
        torch.tensor([[0.0, 0.0, 0.35]]),
        entry, nail, torch.tensor([0]),
    )
    assert count.tolist() == [3]
    assert index.tolist() == [3]
```

Also test: a 15.1 mm radial miss, backward crossing, revisit, zero-length line, and a segment that never reaches the plane.

- [ ] **Step 2: Run the tests and verify RED**

Run:
```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_hammer_guideline.py -q
```
Expected: import failure for `src.tasks.hammer.mdp.guideline`.

- [ ] **Step 3: Implement only the pure functions**

Use clamped projection for the finite segment and solve each disk crossing by linearly interpolating the swept segment at the gate plane. Iterate at most six gates so a single 20 ms control window can consume multiple gates. Return per-environment integer tensors; do not keep global state in these functions.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all geometry tests pass.

- [ ] **Step 5: Commit the pure geometry**

```bash
git add src/tasks/hammer/mdp/guideline.py tests/test_hammer_guideline.py
git commit -m "feat(hammer): add straight waypoint geometry"
```

---

### Task 2: One always-on tracker and pure readers

**Files:**
- Modify: `src/tasks/hammer/mdp/guideline.py`
- Modify: `src/tasks/hammer/mdp/__init__.py`
- Modify: `tests/test_hammer_guideline.py`

**Interfaces:**
- Produces: `_ENV_GUIDELINE_ATTR = "_hammer_waypoint_guideline"`.
- Produces: `WaypointProgressTracker(ManagerTermBase)` callable as a `MetricsTermCfg(per_substep=True)`.
- Produces pure readers: `next_gate_vector(env)`, `completed_gate_fraction(env)`, `guideline_perpendicular_error(env)`, and `ordered_gate_progress_reward(env)`.

- [ ] **Step 1: Write failing lifecycle tests using a minimal fake environment**

Cover these exact invariants:

```python
def test_reward_and_observation_reads_are_idempotent(tracker_env):
    tracker = WaypointProgressTracker(METRIC_CFG, tracker_env)
    tracker(tracker_env)
    before_index = tracker.next_gate.clone()
    before_pulse = tracker.newly_crossed.clone()
    ordered_gate_progress_reward(tracker_env)
    next_gate_vector(tracker_env)
    completed_gate_fraction(tracker_env)
    guideline_perpendicular_error(tracker_env)
    torch.testing.assert_close(tracker.next_gate, before_index)
    torch.testing.assert_close(tracker.newly_crossed, before_pulse)

def test_same_substep_contact_disarms_before_gate_credit(tracker_env):
    tracker_env._hammer_first_strike.started[:] = True
    move_head_across_next_gate(tracker_env)
    tracker(tracker_env)
    assert tracker.newly_crossed.tolist() == [0]
```

Also test partial reset isolation, lazy post-reset anchoring, first-substep zero pulse, next-control-window pulse clearing, and six-gate cumulative payout exactly one.

- [ ] **Step 2: Run the lifecycle tests and verify RED**

Run the Task 1 pytest command. Expected: missing tracker/readers.

- [ ] **Step 3: Implement the minimal tracker**

The tracker owns tensors for `initialized`, `entry`, `nail`, `previous_head`, `next_gate`, `newly_crossed`, and `disarmed`. `reset(env_ids)` clears only those rows. Its first post-reset call stores current head/nail and returns zeros. At the first substep of each control window it clears `newly_crossed`; every call then uses `advance_ordered_gates`. Read `env._hammer_first_strike.started` before credit and disarm permanently. Attach the instance to `env` under `_ENV_GUIDELINE_ATTR`; readers raise clearly if it is missing.

- [ ] **Step 4: Export names and run focused tests**

Run:
```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_hammer_guideline.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit tracker and readers**

```bash
git add src/tasks/hammer/mdp/guideline.py src/tasks/hammer/mdp/__init__.py tests/test_hammer_guideline.py
git commit -m "feat(hammer): add treatment-invariant waypoint tracker"
```

---

### Task 3: Wire C0 and C-Gate without changing legacy tasks

**Files:**
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Modify: `tests/test_configs.py`
- Modify: `tests/test_env.py`

**Interfaces:**
- Extends `z1_hammer_env_cfg(..., guideline: bool=False, gate_reward: bool=False)`.
- Registers `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0` and `...-CGate`.

- [ ] **Step 1: Write failing config-isolation tests**

Assert that `gate_reward=True` without `guideline=True` raises; both new arms contain the same three new observation terms and the same `waypoint_progress` metric; only C-Gate contains `r_gate` at weight 8; both retain exact F8 weights, DiffIK action, fixed actuator config, `imp_max_p=0`, and `r_imit` absence. Assert legacy F8 config keys and signatures remain unchanged.

- [ ] **Step 2: Run config tests and verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_configs.py -q
```

- [ ] **Step 3: Add minimal config wiring**

When `guideline=True`, require the existing `first_strike` metric, then append `waypoint_progress` after it with `per_substep=True`. Add the three pure observation readers to actor and critic. When `gate_reward=True`, add only:

```python
cfg.rewards["r_gate"] = RewardTermCfg(
    func=hammer_mdp.ordered_gate_progress_reward,
    weight=8.0,
    params={},
)
```

Register fresh C0/C-Gate tasks from `cat_impulse=True, event_correct=True, event_linear=True, guideline=True`, differing only in `gate_reward`.

- [ ] **Step 4: Add integration fixtures and exact dimension tests**

Construct one CPU environment per new arm. Assert action shape `(1, 3)`, actor/critic shape `(1, 42)`, finite observations, identical initial guideline observations under the same seed, and that repeated actor/critic computation cannot advance gates.

- [ ] **Step 5: Run focused config and construction tests**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_configs.py tests/test_env.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit treatment wiring**

```bash
git add src/tasks/hammer/config/z1/env_cfgs.py src/tasks/hammer/config/z1/__init__.py tests/test_configs.py tests/test_env.py
git commit -m "feat(hammer): register Cartesian waypoint study"
```

---

### Task 4: CPU reference qualification and owner-readable geometry

**Files:**
- Create: `evaluation/guideline/qualify_reference.py`
- Create: `tests/test_guideline_qualification.py`

**Interfaces:**
- CLI output: `--out DIR` writes `qualification.json`, `reference_xz_xy.png`, and a per-reset CSV.
- Exit 0 only when all reset seeds 1000--1015 cross six gates, contact, advance the nail, remain within the 5 mm corridor after gate 1, remain under 3.1415 rad/s, and have no invalid values.

- [ ] **Step 1: Write failing decision-rule and schema tests**

Build synthetic rows and require the aggregate function to fail if any one reset has fewer than six gates, no contact, no nail progress, corridor max above 0.005 m, qvel above 3.1415, or a non-finite value. Require exactly seeds 1000--1015 and no success-selected subset.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_guideline_qualification.py -q
```

- [ ] **Step 3: Implement the thin qualification script**

Reuse `SingleStrikeReference`, the production task config, the shared tracker, and the existing head/contact/qvel reads. Do not duplicate geometry formulas in the script. Plot the dashed black reference, six gate disks, translucent 10 mm-diameter corridor, realized x-z/x-y path, and contact marker.

- [ ] **Step 4: Run the 16-reset CPU gate**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python evaluation/guideline/qualify_reference.py --out evaluation/results/2026-08-01_guideline_qualification
```

Expected: exit 0, 16/16 qualifying rows. If any row fails, stop; do not widen gates or corridor.

- [ ] **Step 5: Inspect the PNG and commit named evidence**

Use `view_image` on `reference_xz_xy.png`. Then:

```bash
git add evaluation/guideline/qualify_reference.py tests/test_guideline_qualification.py evaluation/results/2026-08-01_guideline_qualification/qualification.json evaluation/results/2026-08-01_guideline_qualification/reference_xz_xy.png evaluation/results/2026-08-01_guideline_qualification/per_reset.csv
git commit -m "test(hammer): qualify straight guideline geometry"
```

---

### Task 5: Fail-closed evaluation and episode-first straightness analysis

**Files:**
- Modify: `scripts/eval_impulse.py`
- Modify: `tests/test_eval_impulse_hook.py`
- Create: `evaluation/analysis/guideline_campaign.py`
- Create: `tests/test_guideline_campaign.py`

**Interfaces:**
- Adds explicit evaluator contracts for C0/C-Gate; legacy task mappings and `EXPECTED_FIXED_ACTION_SIGNATURE` remain unchanged.
- Produces `q90_terminal_descent_perpendicular_error_m_sampled` from episode-first aggregation.

- [ ] **Step 1: Write failing evaluator mutation tests**

Require pre-rollout rejection for an unlisted task, altered DiffIK field, absent tracker, wrong observation width, gate reward in C0, missing gate reward in C-Gate, changed fixed gains, `imp_max_p != 0`, inherited reward overrides, or dirty provenance. Assert the legacy contract constant is byte-identical before/after the extension.

- [ ] **Step 2: Implement explicit new-task contracts**

Add a separate mapping for the two guideline tasks. Reuse the existing fixed DiffIK signature rather than generalizing it. Persist the 500 Hz head trace, reset head/nail anchors, gate index/pulse, corridor error, and contact latch in each sampled episode.

- [ ] **Step 3: Write failing episode/seed aggregation tests**

Test these exact rules:

```python
assert episode_error(trace_without_gate1) == pytest.approx(0.050)
assert episode_error(contact_before_gate1) == pytest.approx(0.050)
assert episode_error(valid_trace) == np.quantile(np.minimum(d_perp, 0.050), 0.90)
assert seed_endpoint([episode_a, episode_b]) == np.quantile([episode_a, episode_b], 0.90)
```

Include a long episode and a short episode to prove substeps are not pooled.

- [ ] **Step 4: Implement analysis and static plots**

Use the shared pure geometry on the stored 500 Hz trace. Emit per-episode CSV, seed-level CSV, x-z/x-y grids, corridor occupancy, backward-progress count, gate completion, and the existing success/speed/depth/impulse/qvel/contact-class secondary metrics.

- [ ] **Step 5: Run focused tests**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_eval_impulse_hook.py tests/test_guideline_campaign.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit evaluator and analysis**

```bash
git add scripts/eval_impulse.py tests/test_eval_impulse_hook.py evaluation/analysis/guideline_campaign.py tests/test_guideline_campaign.py
git commit -m "feat(eval): add fail-closed guideline analysis"
```

---

### Task 6: Full CPU gate, review, and bounded CUDA smoke

**Files:**
- Modify only files named by a concrete failing test or reviewer finding.
- Create: `docs/results/2026-08-01_cartesian_guideline_cpu_gate.md`

- [ ] **Step 1: Run mandatory tests and scripts**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_hammer_guideline.py tests/test_guideline_qualification.py tests/test_guideline_campaign.py tests/test_configs.py tests/test_env.py tests/test_impact_progress_reward.py tests/test_impulse_bound.py tests/test_impulse_constraint.py tests/test_delivered_impulse_reward.py tests/test_cat_soft_hook.py -q
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/validate_rewards.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/verify_contact_sensor.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/verify_reward_setup.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/playback_reference.py
```

Expected: all pytest targets pass, validation phases A-M pass, sensor/random-policy/reference gates pass.

- [ ] **Step 2: Request independent code and spec-conformance reviews**

Resolve every Critical/Important finding with a focused failing test first. Re-run Step 1 after changes.

- [ ] **Step 3: Bank the CPU-gate report and commit named files**

Record exact commands, exit codes, revision, asset revision, reward table, geometry constants, qualification table, proven/open conclusions, and production line count.

- [ ] **Step 4: Push the clean reviewed branch and deploy by git**

Use `git push`, then a clean Vega checkout via `git fetch`/`git checkout`; never `scp` tracked files.

- [ ] **Step 5: Run one short CUDA identity smoke per arm**

Require correct task/action/reward/observation signatures, `impossible_success_n==0`, `lambda_dead_n==0`, no target/geometry NaNs, and no dirty hash. Do not launch training from a failed smoke.

- [ ] **Step 6: Stop at the training checkpoint**

Report smoke evidence and the exact excluded-seed launch commands. Do not launch seeds until the owner approves the frozen continuation rule and clean revision.
