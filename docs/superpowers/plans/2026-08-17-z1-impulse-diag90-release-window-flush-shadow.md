# Z1 diagnostic-0.9 release/window-flush shadow implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately authorized, measurement-only shadow that observes physical contact
release and the 25-substep Lambda-window flush after native success, without changing the native
task result or either learned policy.

**Architecture:** Extend the existing `scripts/impulse_cat_activation_survey.py` survey seam. The
native rollout remains authoritative through its success boundary; a shadow-only continuation
removes only `nail_driven` termination and records post-success samples until release plus window
flush. The shadow emits a distinct artifact and comparison summary so native and shadow horizons
cannot be mixed.

**Tech Stack:** Python 3.10, NumPy, PyTorch, mjlab 1.4.0, MuJoCo/MuJoCo Warp, pytest, Slurm.

## Global constraints

- This plan is not launch authorization. Obtain an explicit user authorization before any live
  smoke, Vega submission, or remote write.
- Use the frozen control and target `model_499.pt` checkpoints and exact role SHA-256 identities
  from `docs/superpowers/specs/2026-08-17-z1-impulse-diag90-post-training-evaluation-design.md`.
- Keep live `imp_max_p=0`; replay `imp_max_p=0.5` offline only. Do not change either trained cap
  vector, `tau=0.95`, `min_p=0`, impulse seed `0.001`, velocity CaT, physics, policy actions,
  observations, gains, rewards, or domain randomization.
- Remove only the `nail_driven` termination after proving byte/numeric parity through each world's
  native terminal boundary. Native task outcomes remain authoritative.
- Stop a shadow world only after five consecutive off-contact 2 ms samples and the 25-substep
  Lambda window has flushed, or at 250 ms after native success. Preserve unresolved worlds as
  right-censored.
- Complete-event dose requires at least 50 completed, unambiguous activating physical events and at
  least 95% completion. Analyze completed and right-censored events separately.
- Controller reads are not inferential units. Any paired interval resamples complete environment
  IDs across both roles with 10,000 resamples and seed `2026081703`.
- Never raise `imp_max_p` from this shadow result. Three saturated `p=0.5` reads already imply
  pressure `0.875`; four imply `0.9375`.
- Do not commit raw large traces. Bank compact sufficient statistics, immutable manifests, and
  deterministic analysis source only.

---

### Task 1: Specify and test the shadow stop state

**Files:**
- Modify: `tests/test_impulse_cat_activation_survey.py`
- Modify: `scripts/impulse_cat_activation_survey.py`

**Interfaces:**
- Consumes: one Boolean contact sample and one six-joint Lambda sample per 2 ms physics substep
  after native success.
- Produces: `ReleaseWindowFlushTracker.update(contact: np.ndarray,
  lambda_per_joint: np.ndarray) -> ShadowStopState`, where `ShadowStopState` records
  `off_contact_run`, `window_flushed`, `complete`, and `right_censored`.

- [ ] **Step 1: Write the failing stop-state tests**

```python
def test_shadow_requires_release_run_and_full_lambda_flush():
  tracker = survey.ReleaseWindowFlushTracker(num_envs=1, window_substeps=25)
  for _ in range(4):
    state = tracker.update(np.array([False]), np.zeros((1, 6)))
  assert not state.complete[0]
  state = tracker.update(np.array([False]), np.zeros((1, 6)))
  assert state.complete[0]


def test_shadow_does_not_call_contact_release_a_flush_while_lambda_is_nonzero():
  tracker = survey.ReleaseWindowFlushTracker(num_envs=1, window_substeps=25)
  for _ in range(5):
    state = tracker.update(np.array([False]), np.ones((1, 6)) * 0.01)
  assert state.off_contact_run[0] == 5
  assert not state.window_flushed[0]
  assert not state.complete[0]
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `conda run -n unitree_mjlab python -m pytest -q tests/test_impulse_cat_activation_survey.py -k release_window_flush`

Expected: FAIL because `ReleaseWindowFlushTracker` is absent.

- [ ] **Step 3: Implement the minimal vectorized state machine**

Use a fixed 25-sample Boolean ring indicating whether any joint Lambda is positive. Increment the
off-contact run only when contact is false, reset it on contact, and mark complete only when the run
is at least five and the entire ring is false. Keep timeout/right-censoring outside this class.

- [ ] **Step 4: Run the focused and survey suites and verify GREEN**

Run: `conda run -n unitree_mjlab python -m pytest -q tests/test_impulse_cat_activation_survey.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the state machine**

```bash
git add scripts/impulse_cat_activation_survey.py tests/test_impulse_cat_activation_survey.py
git commit -m "feat(hammer): track release and impulse-window flush"
```

### Task 2: Add parity-guarded native-to-shadow continuation

**Files:**
- Modify: `tests/test_impulse_cat_activation_survey.py`
- Modify: `scripts/impulse_cat_activation_survey.py`

**Interfaces:**
- Consumes: the existing `_run_population` reset/RNG streams, role checkpoint, and the native
  terminal snapshot.
- Produces: `_continue_after_native_success(wrapped, recorder, native_terminal,
  *, max_shadow_substeps=125) -> dict[str, np.ndarray]` with separate native-boundary and shadow
  telemetry.

- [ ] **Step 1: Write the failing parity and timeout tests**

Create a two-environment fake in which environment 0 releases and flushes after five samples while
environment 1 stays in contact for all 125 samples. Assert environment 0 is complete, environment 1
is right-censored, and every action, gain, observation, reward, Lambda, and contact value through
the native terminal index equals the unmodified native recorder value exactly.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `conda run -n unitree_mjlab python -m pytest -q tests/test_impulse_cat_activation_survey.py -k shadow_continuation`

Expected: FAIL because `_continue_after_native_success` is absent.

- [ ] **Step 3: Implement the continuation behind an explicit shadow flag**

Clone the existing measurement environment configuration, remove only the `nail_driven`
termination in the shadow instance, preserve the live `imp_max_p=0` recorder path, and stop each
world independently at tracker completion or 125 physics substeps. Refuse to start unless a
native-success snapshot exists and the registered termination set differs by exactly
`{"nail_driven"}`.

- [ ] **Step 4: Add fail-closed mutation tests**

Mutate one field at a time—velocity limit, action, VIC gain, reward, observation, physics timestep,
and an additional termination—and assert the parity guard raises before banking any shadow result.

- [ ] **Step 5: Run the survey suite and verify GREEN**

Run: `conda run -n unitree_mjlab python -m pytest -q tests/test_impulse_cat_activation_survey.py`

Expected: all tests pass.

- [ ] **Step 6: Commit the guarded continuation**

```bash
git add scripts/impulse_cat_activation_survey.py tests/test_impulse_cat_activation_survey.py
git commit -m "feat(hammer): add parity-guarded contact-flush shadow"
```

### Task 3: Add completed/unambiguous event-dose analysis

**Files:**
- Modify: `tests/test_impulse_cat_activation_survey.py`
- Modify: `scripts/impulse_cat_activation_survey.py`

**Interfaces:**
- Consumes: shadow physical-event labels, right-censoring flags, ambiguity flags, and chronological
  offline `delta_impulse` reads at `p=0.5`.
- Produces: `summarize_complete_event_dose(...) -> dict[str, object]` with four explicit groups:
  completed/unambiguous, completed/ambiguous, right-censored/unambiguous, and
  right-censored/ambiguous.

- [ ] **Step 1: Write the failing event-pressure test**

```python
def test_complete_event_dose_uses_chronological_soft_survival_product():
  result = survey.summarize_complete_event_dose(
    event_ids=np.array([7, 7, 7]),
    delta_impulse=np.array([0.5, 0.5, 0.5]),
    right_censored=np.array([False, False, False]),
    ambiguous=np.array([False, False, False]),
  )
  assert result["completed_unambiguous"]["events"] == 1
  assert result["completed_unambiguous"]["event_pressure"]["max"] == 0.875
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `conda run -n unitree_mjlab python -m pytest -q tests/test_impulse_cat_activation_survey.py -k complete_event_dose`

Expected: FAIL because `summarize_complete_event_dose` is absent.

- [ ] **Step 3: Implement exact grouping and calibration gates**

For each event, compute `1 - product(1 - delta_impulse_t)` in chronological order. Never merge
ambiguous events, never treat censored prefixes as completed dose, and emit
`calibrated = completed_unambiguous_events >= 50 and completion_rate >= 0.95`.

- [ ] **Step 4: Add negative fixtures**

Assert that missing event IDs, non-finite deltas, deltas outside `[0, 1]`, duplicated chronology,
and a 49-event or 94.9%-complete sample fail the calibration gate.

- [ ] **Step 5: Run the focused and survey suites and verify GREEN**

Run: `conda run -n unitree_mjlab python -m pytest -q tests/test_impulse_cat_activation_survey.py`

Expected: all tests pass.

- [ ] **Step 6: Commit event-dose analysis**

```bash
git add scripts/impulse_cat_activation_survey.py tests/test_impulse_cat_activation_survey.py
git commit -m "feat(hammer): summarize complete shadow event dose"
```

### Task 4: Add a collision-safe shadow CLI and launcher

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Create: `scripts/slurm/vega_vic_impulse_diag90_500_shadow.sbatch`
- Create: `tests/test_vic_impulse_diag90_500_shadow_launcher.py`

**Interfaces:**
- Consumes: explicit control/target checkpoint paths, frozen revisions, exact stochastic seed tuple,
  and a nonexistent output root.
- Produces: one immutable leaf per role containing a compact shadow summary, compressed trace, and
  ordered relative-path `SHA256SUMS`.

- [ ] **Step 1: Write failing launcher tests**

Copy the Task-6 launcher fixture shape and assert exact array mapping, `--no-requeue`, live
`imp_max_p=0`, explicit `--shadow-release-window-flush`, collision refusal, dirty-tree refusal,
checkpoint rehashing before and after evaluation, and exact output enumeration including dotfiles.

- [ ] **Step 2: Run the launcher test and verify RED**

Run: `conda run -n unitree_mjlab python -m pytest -q tests/test_vic_impulse_diag90_500_shadow_launcher.py`

Expected: FAIL because the shadow launcher and CLI flag are absent.

- [ ] **Step 3: Add the explicit CLI mode and launcher**

Keep the native evaluator mode unchanged. The new mode must write a distinct schema version and
must refuse any role, checkpoint, revision, cap, stochastic-seed, or output-shape drift.

- [ ] **Step 4: Run launcher, evaluator, shell-syntax, and diff checks**

```bash
conda run -n unitree_mjlab python -m pytest -q \
  tests/test_impulse_cat_activation_survey.py \
  tests/test_vic_impulse_diag90_500_shadow_launcher.py
bash -n scripts/slurm/vega_vic_impulse_diag90_500_shadow.sbatch
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit the guarded route**

```bash
git add scripts/impulse_cat_activation_survey.py \
  scripts/slurm/vega_vic_impulse_diag90_500_shadow.sbatch \
  tests/test_vic_impulse_diag90_500_shadow_launcher.py
git commit -m "feat(hammer): launch contact-flush evaluation shadow"
```

### Task 5: Review, authorize, run once, and bank separately

**Files:**
- Create after authorization: `docs/results/2026-08-17_z1_impulse_diag90_contact_flush_shadow.md`
- Modify after authorization: `docs/README.md`
- Modify after authorization: `tests/test_docs_current.py`

**Interfaces:**
- Consumes: reviewed commits from Tasks 1–4 and explicit user launch authorization.
- Produces: a separate shadow result that does not revise native task outcomes or authorize a cap
  or `imp_max_p` change.

- [ ] **Step 1: Obtain clean code/spec review and explicit launch authorization**

Do not proceed from a review alone. Record the exact approved code revision, asset revision,
checkpoint hashes, Slurm command, and output root in the task brief.

- [ ] **Step 2: Run mandatory local gates**

```bash
conda run -n unitree_mjlab python docs/research/reward-design/validate_rewards.py
conda run -n unitree_mjlab python docs/research/reward-design/verify_contact_sensor.py
conda run -n unitree_mjlab python docs/research/reward-design/verify_reward_setup.py
conda run -n unitree_mjlab python -m pytest -q \
  tests/test_impulse_cat_activation_survey.py \
  tests/test_vic_impulse_diag90_500_shadow_launcher.py
```

Expected: reward phases A–M pass, contact and reward-setup checks pass, and all focused tests pass.

- [ ] **Step 3: Run the authorized 2-environment × 8-control-step live smoke**

Require exact native parity through success and exercise at least one completed flush plus one
right-censored fixture. Stop if parity, finiteness, or output-manifest validation fails.

- [ ] **Step 4: Submit the one-shot two-role Vega array**

Use `--no-requeue`, no retry, a clean detached worktree, explicit checkpoint/revision variables,
and a collision-safe new output root. Capture the literal submission command and returned job ID.

- [ ] **Step 5: Validate and analyze without changing treatment strength**

Rehash every artifact, require both tasks `COMPLETED 0:0` with empty stderr, validate all numeric
fields as finite, and apply the 50-event/95%-completion calibration gate. If the gate fails, report
the incomplete calibration; do not extend the horizon or raise `imp_max_p` without a new design.

- [ ] **Step 6: Bank a separate result and run docs checks**

```bash
conda run -n unitree_mjlab python -m pytest -q tests/test_docs_current.py
git diff --check
```

The result must explicitly distinguish native task outcomes from shadow contact-dose evidence and
retain the one-training-seed, simulation-only, non-clamp, and non-hardware-safety limitations.

- [ ] **Step 7: Commit but do not push before review**

```bash
git add docs/results/2026-08-17_z1_impulse_diag90_contact_flush_shadow.md \
  docs/README.md tests/test_docs_current.py docs/results/assets/
git commit -m "docs(hammer): bank contact-flush shadow result"
```

## Plan self-review

- Spec coverage: exact native parity, one-termination difference, release/run and window-flush stop,
  250 ms cap, four censoring/ambiguity groups, `p=0.5` survival-product dose, 50-event/95% gate,
  collision-safe no-retry launch, immutable provenance, and separate banking are all assigned.
- Placeholder scan: every file, interface, test command, negative case, and implementation action
  is concrete; no unresolved marker or implicit delegation remains.
- Type consistency: `ReleaseWindowFlushTracker`, `ShadowStopState`,
  `_continue_after_native_success`, and `summarize_complete_event_dose` are introduced once and used
  consistently.
- Scope: this plan performs no training and grants no launch, cap, `imp_max_p`, clamp, or hardware
  authorization.
