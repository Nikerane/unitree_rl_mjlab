# No-Windup Fixed-Reset Strike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active lift-then-strike reference with one fixed-reset direct Cartesian strike and qualify the matched C0/C-Gate experiment before GPU training.

**Architecture:** `SingleStrikeReference` owns one frozen reset-head-to-follow-through segment and a spatial monotone phase. The guideline arms alone use a fixed physical reset; the existing `WaypointProgressTracker` then observes the same reset-head-to-nail geometry. Historical result assets remain unchanged.

**Tech Stack:** Python 3.11, PyTorch, mjlab 1.4.0, MuJoCo/MuJoCo-Warp 3.8.1, pytest.

## Global Constraints

- Fixed impedance only; never call `set_gains`.
- Keep `IMP_J_LIMIT` manufacturer values unchanged.
- Keep `imp_max_p=0.0`.
- Do not edit installed packages.
- Use named-file staging only; never `git add -A`.
- CPU qualification precedes Vega.
- C0 and C-Gate must differ only by `r_gate`.

---

### Task 1: One-segment direct-strike reference

**Files:**
- Modify: `tests/test_strike_reference.py`
- Modify: `src/tasks/hammer/mdp/references.py`
- Modify: `src/tasks/hammer/mdp/observations.py`
- Modify: `src/tasks/hammer/mdp/rewards.py`

**Interfaces:**
- Consumes: reset head position, frozen nail-top position, episode step.
- Produces: `SingleStrikeReference.update`, `preview`, `peek`, `waypoint`, `playback_length`, and `playback_target` with one-segment geometry.

- [ ] **Step 1: Write failing behavior tests**

Add literal tests proving:

```python
def test_playback_never_rises_and_stays_on_direct_segment():
    ref = _ref(overshoot=0.02, descent_speed=0.05)
    ref.update(HEAD0, NAIL, _steps(0))
    targets = torch.stack(
        [ref.playback_target(k) for k in range(ref.playback_length() + 1)]
    )
    assert torch.all(targets[1:, :, 2] <= targets[:-1, :, 2,] + 1e-7)
    # Every target has zero perpendicular residual to HEAD0 -> target.

def test_phase_is_spatial_not_clock_driven():
    ref = _ref(overshoot=0.02)
    ref.update(HEAD0, NAIL, _steps(0))
    assert torch.equal(ref.update(HEAD0, NAIL, _steps(100)), torch.zeros(B))
```

Replace wind-up/apex assertions with literal endpoint, projection, monotonicity, batching, purity, reset, and observation-reader assertions for the direct segment.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_strike_reference.py -q
```

Expected: failures because current playback rises to `_apex` and current phase advances by the wind-up clock.

- [ ] **Step 3: Implement the minimal one-segment reference**

Use frozen `_head0` and `_target`; compute spatial projection onto `_target - _head0`, gate off-axis progress with `axis_tol`, and monotonically latch it. `waypoint(phi)` is one linear interpolation. `playback_target(k)` advances by `descent_speed` and clamps bit-exactly at `_target`. Remove `approach_height`, `min_windup_clearance`, `windup_speed`, `_apex`, `_n_windup`, and `_s0` from active code.

- [ ] **Step 4: Update active descriptions**

Describe one direct strike in `references.py`, `observations.py`, and the disabled `ImitationPriorTerm`; do not rewrite dated result records.

- [ ] **Step 5: Verify GREEN**

Run the Task 1 command and require exit 0.

---

### Task 2: Fixed reset for C0/C-Gate only

**Files:**
- Modify: `tests/test_configs.py`
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`

**Interfaces:**
- Consumes: `guideline=True` factory flag.
- Produces: identical `(0.0, 0.0)` joint reset range for C0 and C-Gate train/play configurations while leaving F8 and every legacy training arm unchanged.

- [ ] **Step 1: Write failing configuration tests**

Add assertions that both guideline training configs have fixed reset ranges, that F8 training remains `(-0.05, 0.05)`, and that normalized C0/C-Gate differences remain exactly `r_gate`.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_configs.py::TestCartesianGuidelineStudy -q
```

Expected: fixed-reset assertion fails because guideline training currently keeps `(-0.05, 0.05)`.

- [ ] **Step 3: Implement the minimal factory override**

Inside the existing `if guideline:` block set:

```python
cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
```

Do not change the base task or global reset event.

- [ ] **Step 4: Verify GREEN**

Run the Task 2 command and require exit 0.

---

### Task 3: Fixed-reset reference qualification

**Files:**
- Modify: `tests/test_guideline_qualification.py`
- Modify: `evaluation/guideline/qualify_reference.py`

**Interfaces:**
- Consumes: production C0 configuration and one-segment scripted playback.
- Produces: fail-closed fixed-reset evidence over execution seeds 1000--1015 without falsely demanding 16 distinct physical poses.

- [ ] **Step 1: Write failing qualification-contract tests**

Require `(0.0, 0.0)`, require every row's realized six-joint reset tuple to equal the first row, reject any physical reset drift, and retain exact execution-seed ordering.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_guideline_qualification.py -q
```

Expected: failures because the current evaluator requires `(-0.05, 0.05)` and 16 distinct realized reset states.

- [ ] **Step 3: Implement the fixed-reset evidence contract**

Rename the aggregate field to `identical_fixed_reset`, compare all realized reset tuples bit-for-bit, bank execution seeds separately from physical reset identity, and record that the source task is the fixed-reset production C0 arm.

- [ ] **Step 4: Verify GREEN**

Run the Task 3 command and require exit 0.

---

### Task 4A: Runtime and artifact compatibility

**Files:**
- Modify: `scripts/render_reference.py`
- Modify: `scripts/render_reference_path.py`
- Modify: `scripts/play_reference.py`
- Modify: `scripts/diag_impulse_trace.py`
- Modify: `scripts/smoke_first_strike_instrumentation.py`
- Modify: `scripts/render_policy.py`
- Modify: `evaluation/analysis/fixed_reset_video_library.py`
- Modify relevant active tests that instantiate removed wind-up parameters.

**Interfaces:**
- Consumes: the reduced `SingleStrikeReference(..., overshoot, descent_speed, axis_tol)` constructor.
- Produces: active render, smoke, and pre-training scripts that execute the same direct reference. Dated `docs/results/assets/**` probes remain untouched historical artifacts.

- [ ] **Step 1: Update tests first and verify their expected constructor failures**
- [ ] **Step 2: Remove active `approach_height`/wind-up CLI plumbing and use the direct constructor**
- [ ] **Step 3: Emit two-endpoint polylines for new renders while continuing to accept historical finite `(3,3)` traces**
- [ ] **Step 4: Run focused active-consumer tests**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_eval_impulse_hook.py \
  tests/test_contact_row_impulse.py \
  tests/test_fixed_reset_video_library.py \
  tests/test_imitation_reward.py \
  tests/test_smoke_first_strike_instrumentation.py -q
```

### Task 4B: Active scientific-gate compatibility and provenance

**Files:**
- Modify: `evaluation/whip/lambda_feasibility.py`
- Modify: `docs/research/reward-design/playback_reference.py`
- Modify: `docs/research/reward-design/reward_design_util.py`
- Modify: `docs/research/reward-design/derive_impulse_thresholds.py`
- Modify: `docs/research/reward-design/c2_enforcement_gate.py`
- Modify: `docs/research/reward-design/validate_rewards.py` only for stale active comments.

**Interfaces:**
- The current reference digest must describe only the reduced direct-segment constructor and playback rule.
- Phase-M/reference qualification must measure one nominal direct strike, not a fake approach-height sweep.
- Any repeated direct strikes are repeatability samples, not intensity settings.
- Historical calibration assets and dated results remain unchanged. No manufacturer cap changes and no enforcement run are authorized.

- [ ] **Step 1: Add or update focused source/behavior tests and observe RED**
- [ ] **Step 2: Remove false wind-up parameters and claims from active executable gates**
- [ ] **Step 3: Make stale enforcement calibration fail closed until re-qualified for the direct reference; do not run it with `imp_max_p>0` in this phase**
- [ ] **Step 4: Verify the direct-reference Phase-M gate and focused tests on CPU**

---

### Task 5: CPU qualification and regression gates

**Files:**
- Write new evidence only after all code tests pass: `evaluation/results/2026-08-02_no_windup_fixed_reset_qualification/**`
- Modify the current Task 4 report/ledger to mark the randomized-reset artifact superseded, not deleted.

- [ ] **Step 1: Run the focused reference/guideline suite**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_strike_reference.py tests/test_guideline.py \
  tests/test_guideline_qualification.py tests/test_configs.py tests/test_env.py -q
```

- [ ] **Step 2: Run mandatory pre-training gates**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python validate_rewards.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python verify_contact_sensor.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python verify_reward_setup.py
```

Also run the impulse/reward pytest set named in `AGENTS.md`.

- [ ] **Step 3: Run fixed-reset qualification**

Require 16/16 execution seeds to share the identical reset and satisfy all geometry, contact, nail-drive, qvel, and finite-value gates. Inspect the regenerated x-z/x-y plot.

- [ ] **Step 4: Audit active wind-up references**

Search `src/**`, `scripts/**`, active tests, and living docs. Classify any remaining hits as either a bug to remove or explicitly dated historical evidence.

---

### Task 6: Vega smoke and minimal matched pilot

**Files:**
- No tracked-file edits during execution.

- [ ] **Step 1: Commit named files only and push after local verification**
- [ ] **Step 2: Deploy by clean commit/push/pull; require 0-dirty code and asset revisions**
- [ ] **Step 3: Run one CUDA smoke per C0/C-Gate and require `impossible_success_n=0`, `lambda_dead_n=0`, and no failed predicates**
- [ ] **Step 4: Submit the smallest preregistered matched pilot only if both smokes pass**

Pilot outcomes must not be used to retune the gate radius or reward weight post hoc.
