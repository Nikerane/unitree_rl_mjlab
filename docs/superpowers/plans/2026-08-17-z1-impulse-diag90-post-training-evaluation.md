# Z1 diagnostic-0.9 post-training evaluation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the completed 500-iteration `imp_max_p=0` and `imp_max_p=0.5` VIC policies on
identical no-learning populations, attribute impulse versus velocity pressure, quantify task
trade-offs, and bank a reproducible one-seed verdict.

**Architecture:** Generalize the existing impulse activation survey without changing its frozen
Step-1 entry point. A checkpoint-role contract and matched RNG streams feed the existing live
recorder; the recorder gains only missing task, velocity, and VIC-gain telemetry. A small paired
analysis layer compares role summaries, while a guarded Slurm array runs one checkpoint per arm.

**Tech Stack:** Python 3.11, NumPy, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, pytest, Bash/Slurm, Vega
A100.

## Global Constraints

- Work only in `/private/tmp/unitree_rl_mjlab-z1-step1-evidence-cleanup` on branch
  `z1-step1-evidence-cleanup`; preserve the dirty main worktree.
- Do not change physics, rewards, actions, observations, VIC gains, domain randomization,
  curriculum, trajectory generation, training treatments, or caps.
- Both evaluation arms use live `imp_max_p=0`; `p=0.5` is replayed offline.
- Report provisional caps first and diagnostic caps second; neither is a hardware limit.
- Do not create another evaluator framework; reuse `impulse_cat_activation_survey.py`.
- Use RED-GREEN TDD for every behavior change.
- Before any Vega evaluation, run reward phases A-M, contact-sensor verification,
  reward-setup verification, focused tests, and a 2-environment x 8-step live smoke.
- Use clean pinned code/assets, unique output leaves, `--no-requeue`, and no automatic retry.

---

### Task 1: Freeze checkpoint roles and evaluation protocol

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Modify: `tests/test_impulse_cat_activation_survey.py`

**Interfaces:**
- Produces: `EVALUATION_CHECKPOINTS: Mapping[str, str]` and
  `validate_checkpoint_role(checkpoint: Path, role: str) -> str`.
- Preserves: `run_survey(...)` and the original `EXPECTED_CHECKPOINT_SHA256` Step-1 contract.

- [ ] **Step 1: Add failing checkpoint-role tests**

```python
def test_evaluation_checkpoint_roles_are_exact_and_fail_closed(tmp_path, monkeypatch):
  checkpoint = tmp_path / "model_499.pt"
  checkpoint.write_bytes(b"control")
  monkeypatch.setattr(survey, "_sha256", lambda _: CONTROL_SHA)
  assert survey.validate_checkpoint_role(checkpoint, "diag90_control") == CONTROL_SHA
  with pytest.raises(RuntimeError, match="role/checkpoint SHA-256 mismatch"):
    survey.validate_checkpoint_role(checkpoint, "diag90_target")
```

- [ ] **Step 2: Verify RED**

Run:
`python -m pytest -q tests/test_impulse_cat_activation_survey.py::test_evaluation_checkpoint_roles_are_exact_and_fail_closed`

Expected: FAIL because `validate_checkpoint_role` does not exist.

- [ ] **Step 3: Implement the exact immutable role mapping and validator**

```python
EVALUATION_CHECKPOINTS = {
  "diag90_control": "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3",
  "diag90_target": "ddd7ac4c855160bff1db2af52e642d960dab2bef34e41dd2532002185eb36d15",
}

def validate_checkpoint_role(checkpoint: Path, role: str) -> str:
  if checkpoint.name != "model_499.pt" or role not in EVALUATION_CHECKPOINTS:
    raise ValueError("post-training evaluation requires a known role and model_499.pt")
  actual = _sha256(checkpoint.resolve(strict=True))
  if actual != EVALUATION_CHECKPOINTS[role]:
    raise RuntimeError("role/checkpoint SHA-256 mismatch")
  return actual
```

- [ ] **Step 4: Verify GREEN and original survey compatibility**

Run: `python -m pytest -q tests/test_impulse_cat_activation_survey.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/impulse_cat_activation_survey.py tests/test_impulse_cat_activation_survey.py
git commit -m "feat(hammer): freeze impulse evaluation checkpoints"
```

### Task 2: Add segment-level compliance summaries

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Modify: `tests/test_impulse_cat_activation_survey.py`

**Interfaces:**
- Produces: `segment_compliance` in every `summarize_population` payload with any-joint and
  per-joint counts/rates, p50/p95/p99/max utilization, and conditional positive margin.
- Produces: `_finite_quantiles(values: np.ndarray) -> dict[str, float | None]`, returning
  `p50`, `p95`, `p99`, and `max` with `None` for an empty input.
- Consumes: existing `_observed_segment_peaks(...)` and native-dtype cap subtraction.

- [ ] **Step 1: Add a synthetic repeated-read test** proving one episode segment is counted once
  even when four controller reads violate.
- [ ] **Step 2: Run the test and verify RED** because `segment_compliance` is absent.
- [ ] **Step 3: Compute compliance from per-segment peak Lambda, never from read count**:

```python
segment_utilization = observed_segment_peaks / cap_array
segment_violating = segment_utilization > 1.0
segment_compliance = {
  "segments": int(segment_utilization.shape[0]),
  "any_joint_violating_segments": int(segment_violating.any(axis=1).sum()),
  "any_joint_violation_rate": float(segment_violating.any(axis=1).mean()),
  "max_joint_utilization": _finite_quantiles(segment_utilization.max(axis=1)),
}
```

The helper used above is exactly:

```python
def _finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
  flat = np.asarray(values, dtype=np.float64).reshape(-1)
  if flat.size == 0:
    return {"p50": None, "p95": None, "p99": None, "max": None}
  if not np.isfinite(flat).all():
    raise ValueError("quantile input must be finite")
  return {
    "p50": float(np.quantile(flat, 0.50)),
    "p95": float(np.quantile(flat, 0.95)),
    "p99": float(np.quantile(flat, 0.99)),
    "max": float(np.max(flat)),
  }
```

- [ ] **Step 4: Verify the focused and complete survey tests pass.**
- [ ] **Step 5: Commit** with message
  `feat(hammer): summarize impulse compliance by episode segment`.

### Task 3: Isolate matched RNG streams

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Modify: `tests/test_impulse_cat_activation_survey.py`

**Interfaces:**
- Produces: separate reset, observation, and action Torch RNG streams derived from each declared
  evaluation seed.
- `_run_population(..., rng_seeds: EvaluationRngSeeds)` records all three seeds in protocol.

- [ ] **Step 1: Add failing tests** showing reset/observation RNG consumption cannot change the
  next policy-action latent noise and that both roles derive identical streams for one base seed.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Reuse the existing `_TorchRngStream` and `_install_evaluator_rng_streams` semantics**
  with offsets `10_000_019`, `20_000_033`, and `30_000_041`; wrap each stochastic policy call in
  the action stream.
- [ ] **Step 4: Verify GREEN and fixed initial-population hash preservation.**
- [ ] **Step 5: Commit** with message `feat(hammer): isolate impulse evaluation RNG streams`.

### Task 4: Capture task, velocity, and VIC-gain telemetry before reset

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Modify: `tests/test_impulse_cat_activation_survey.py`

**Interfaces:**
- Extends `_LiveSurveyRecorder.numpy_trace()` with finite, exact-shape fields:
  `success`, `timeout`, `nail_depth_m`, `delivered_total_n_s`, `first_strike_productive`,
  `first_strike_delivered_n_s`, `substep_peak_qv_per_joint`, `vic_p`, `vic_kp`, and `vic_kd`.
- Reads existing accumulators/action telemetry only; it does not add rewards or sensors.

- [ ] **Step 1: Add failing recorder tests** using the real two-environment VIC task and assert
  pre-reset field shapes `(steps, envs)` or `(steps, envs, 6)` and finiteness.
- [ ] **Step 2: Verify RED** because the fields are absent.
- [ ] **Step 3: Capture existing tracker/action state immediately after `metrics_manager.compute`**
  and before auto-reset; fail closed if the exact VIC action pair or tracker identity is missing.
- [ ] **Step 4: Extend summaries** with fixed-population success/productive counts, terminal nail
  depth, delivered impulse, velocity-limit compliance, and first-contact gain values. Label 24-step
  stochastic task fields as censored fragments.
- [ ] **Step 5: Verify GREEN** with the recorder tests and full survey suite.
- [ ] **Step 6: Commit** with message `feat(hammer): record impulse evaluation utility telemetry`.

### Task 5: Add the paired role runner and comparison summary

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Create: `tests/test_impulse_cat_policy_comparison.py`

**Interfaces:**
- Produces:
  `run_policy_evaluation(checkpoint, role, output_dir, device, stochastic_seeds) -> dict`.
- Emits one fixed trace and three seed-keyed training-like traces plus provisional and diagnostic
  summaries; records role, checkpoint SHA, code revision, asset revision, and protocol.

- [ ] **Step 1: Add failing CLI/protocol tests** for the exact seed tuple
  `(2, 2026081701, 2026081702)`, role selection, collision refusal, both cap summaries from the same
  trace, and live `imp_max_p=0`.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement the minimal orchestration function and CLI flags**
  `--checkpoint-role`, `--checkpoint`, `--output-dir`, and `--device`; keep the banked `run_survey`
  path unchanged.
- [ ] **Step 4: Add pure paired comparison helpers** that accept two summary payloads and report
  absolute/risk-ratio differences without treating reads as independent.
- [ ] **Step 5: Verify GREEN** for both survey and comparison suites.
- [ ] **Step 6: Commit** with message `feat(hammer): compare frozen impulse CaT policies`.

### Task 6: Add a guarded two-arm Vega launcher

**Files:**
- Create: `scripts/slurm/vega_vic_impulse_diag90_500_eval.sbatch`
- Create: `tests/test_vic_impulse_diag90_500_eval_launcher.py`

**Interfaces:**
- Array arm 0 maps only to `diag90_control`; arm 1 maps only to `diag90_target`.
- Consumes explicit `RUN_ROOT`, `ASSET_REPO`, `EXPECTED_CODE_REVISION`,
  `EXPECTED_ASSET_REVISION`, `CONTROL_CHECKPOINT`, and `TARGET_CHECKPOINT`.

- [ ] **Step 1: Add a failing shell-contract test** for exact role/hash/path mapping, 20-minute
  limit, one A100, `--no-requeue`, collision refusal, clean code/assets/runtime guards, and output
  hashes.
- [ ] **Step 2: Verify RED** because the launcher is absent.
- [ ] **Step 3: Implement the zero-retry launcher** using one unique output leaf per array task and
  the paired evaluator CLI.
- [ ] **Step 4: Verify GREEN**, `bash -n`, and a fake-interpreter contract exercise.
- [ ] **Step 5: Commit** with message `feat(hammer): launch matched impulse policy evaluation`.

### Task 7: Verify locally, commit, push, and run once

**Files:**
- No production-file changes.

**Interfaces:**
- Produces one clean pushed evaluator revision and one monitored Vega array job.

- [ ] **Step 1: Run focused tests**:

```bash
python -m pytest -q \
  tests/test_cat_soft_hook.py \
  tests/test_impulse_cat_activation_survey.py \
  tests/test_impulse_cat_policy_comparison.py \
  tests/test_vic_impulse_diag90_500_eval_launcher.py
```

- [ ] **Step 2: Run mandatory pre-experiment gates**:

```bash
python scripts/validate_rewards.py
python scripts/verify_contact_sensor.py
python scripts/verify_reward_setup.py
```

- [ ] **Step 3: Run one 2-environment x 8-step live no-learning smoke per role** and verify exact
  shapes, finiteness, fixed protocol identity, live `delta_impulse == 0`, and exact max soft-OR.
- [ ] **Step 4: Run `git diff --check`, commit any final reviewed documentation, push the branch,
  and create a clean detached Vega checkout at the exact revision.**
- [ ] **Step 5: Create the Slurm log parent before submission and run `sbatch --test-only`.**
- [ ] **Step 6: Submit the array exactly once with the explicit approved command, record the job ID,
  and monitor process state, stderr growth, output growth, and the 20-minute timeout. Never retry.**

### Task 8: Validate, compare, and bank the result

**Files:**
- Create: `docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md`
- Create: compact JSON and SHA manifest under
  `docs/results/assets/2026-08-17_z1_impulse_diag90_500_evaluation/`
- Modify: `docs/README.md`
- Test: `tests/test_docs_current.py`

**Interfaces:**
- Consumes both role summaries and traces.
- Produces the frozen verdict table and reproducibility evidence.

- [ ] **Step 1: Verify both jobs completed `0:0`, stderr is empty, every expected trace/summary is
  finite, and all hashes match.**
- [ ] **Step 2: Run paired environment-cluster bootstrap analysis** with 10,000 resamples and fixed
  seed; report each stochastic replica and pooled descriptive result.
- [ ] **Step 3: Apply the five preregistered verdict branches** from the design; explicitly separate
  evidence, inference, and limitation.
- [ ] **Step 4: If terminal censoring still blocks physical event dose, stop the native-result claim
  there and write a separate test-first plan for the release/window-flush shadow. Do not increase
  `imp_max_p`.**
- [ ] **Step 5: Add the result doc and compact evidence, run `tests/test_docs_current.py` and
  `git diff --check`, then commit and push.**

## Plan self-review

- Spec coverage: checkpoint identity, matched populations/RNG, segment/event units, CaT attribution,
  task/velocity/gain trade-offs, launcher safety, analysis, and censoring gate are covered.
- Scope: the native model-499 evaluation is one testable subsystem. The release/window-flush shadow
  remains a separate follow-up if native censoring requires it.
- Type consistency: role names, hashes, seed tuple, caps, and output function names are consistent
  across tasks.
- Placeholder scan: no implementation placeholder or unbounded retry remains.
