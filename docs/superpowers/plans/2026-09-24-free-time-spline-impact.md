# Free-Time Spline Impact Implementation Plan

**Status:** Tasks 0–5 completed and independently reviewed. Final focused CPU verification: 362 passed, no failures/skips, 72 dependency warnings. [Qualification and review record](../../results/2026-09-24-free-time-spline-software-qualification.md). The optional trained-checkpoint replay was unavailable and is explicitly recorded as not exercised. No training, merge, or push was performed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make the existing free-time spline-guided policy apply sampled joint impedance and provide trustworthy speed, impact, path, and per-joint-load evidence before any new training.

**Architecture:** Preserve the 13-output/42-observation direct-control policy and once-per-episode speed latch. Add an explicit variable-gain mode to the sampled controller, preserve the fixed baseline, and add an evaluation-only measurement collector. Support all three existing quintic start poses through a separate task configuration.

**Tech Stack:** Python 3.10, PyTorch, existing mjlab/MuJoCo/Warp environment, RSL-RL, pytest; no new framework or dependencies.

**Spec:** `../specs/2026-09-24-free-time-spline-impact-design.md` (binding requirements and scientific claim boundaries).

## Global Constraints

- One online policy; joint-space impedance; no Cartesian impedance, timing polynomial, planned duration, or second learned controller.
- Speed bounds `[1.5, 2.2] m/s`; select once per episode; preserve first-action latency and masked PPO semantics.
- Keep existing 13-action/42-observation meanings and normalization.
- Physics `0.00025 s`; policy `0.02 s`; held torque update `0.002 s`; four existing event/window/rearm settings remain 200 substeps.
- Sampled nominal `Kp=[256,384,384,256,192,128]`, `Kd=[6.4]*6`; `C=1.25`; effort limits `[30,60,30,30,30,30] N m`.
- Keep LEFT-only baseline, existing unsaturated reward, persistent guidance weight `0.2`, width `0.05 m`, and `0.02 m` vertical follow-through.
- Preserve inherited CaT caps `[0.369,0.246,0.738,0.369,0.246,0.0164] N m s`, impulse dose `0.2`, velocity limit `3.1415 rad/s`, velocity max probability `0.5`.
- Experimental source and jobs `45860324/45860325` remain untouched; implement in a manifest-verified isolated snapshot.
- No training, hardware action, threshold recalibration, broad refactor, or new human-data pipeline.
- Stop after focused tests, review, and a software-versus-empirical evidence report.

## Review Focus

1. Zero stiffness coordinate must match the sampled calibrated fixed controller, not native actuator nominal gains (Task 1).
2. Asynchronous resets must not leak the previous episode's target, gains, held torque, event window, or telemetry (Tasks 1 and 4).
3. Early success or a simultaneous failure must not fabricate a complete contact window or lose a per-joint exceedance (Task 4).
4. Sparse/partial paths, off-curve recovery, and hovering must remain visible; no interpolation into unobserved bins or hidden postcontact mixing (Tasks 3 and 4).
5. Strict checkpoint loading must preserve controller, path, and load semantics; evaluation-only overrides must be recorded separately (Tasks 2 and 5).

## Workspace and execution instructions

Use the subagent-driven-development skill's plan-specific ledger, task briefs, separate task reviewers, and final reviewer. Dispatch one implementer at a time because tasks share controller/config/evaluator interfaces. The three design agents have already discussed the approach; their agreement is design review, not implementation verification.

For tests use `/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python`, with the implementation checkout as working directory and `PYTHONPATH=.`. First verify `src` resolves inside that checkout; editable installs can otherwise import the older main checkout. Record the actual installed mjlab/MuJoCo/Warp versions in the result.

Every task ends with focused verification, a scoped commit, and independent spec/quality review. Retain complete command/exit-status evidence in the ledger. Do not rerun broad unrelated suites after each small edit.

### Task 0: Create a faithful isolated experimental snapshot

**Files:** new `docs/results/2026-09-24-free-time-spline-source-manifest.json` in the implementation checkout; transfer the current experimental source and the two planning documents.

- [x] Use the git-worktrees skill to create an isolated checkout from `cc46d9998a3c9cd5d8813e89b57e44248353e03e`, on a `codex/` branch.
- [x] Inventory `git diff --binary HEAD`, tracked deletions/modes, and `git ls-files --others --exclude-standard` in the source. Hash before transfer and recheck after transfer to detect concurrent changes.
- [x] Apply tracked differences only in the destination. Transfer untracked source/tests/configuration and every required referenced asset/document. Record excluded bulky results with reasons. Preserve the two planning documents under their current repository-relative names.
- [x] Confirm the essential hashes against the freshly captured source, including the untracked sampled controller and tests. The planning-time controller hash was `c2ef31500f6d65aee2b29c6bb9a7e157ea5ebae6e80ad2a125940618065af0a5`; if it changed, inspect and update provenance rather than force the old contents.
- [x] Commit the transferred baseline separately from new implementation, preserving authorship/provenance in the commit message and manifest.
- [x] Verify imports and run the existing focused baseline once:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -c 'import src; print(src.__file__)'
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_sampled_pd500.py tests/test_impact_speed_contract.py tests/test_quintic_reference.py -q
```

**Acceptance:** copied source matches the manifest and the original remains unmodified. Record existing failures separately; do not claim a clean baseline without passing output or repair unrelated tests automatically.

### Task 1: Add physically effective sampled VIC

**Files:** modify `src/tasks/hammer/mdp/sampled_pd500.py`; reuse `src/tasks/hammer/mdp/variable_impedance.py`; extend `tests/test_sampled_pd500.py`.

**Interface:** preserve existing calls and add keyword-only `gain_mode="fixed"` to `configure_sampled_pd500` and `install_sampled_pd500`. The installed controller owns applied-gain/torque snapshots; fixed mode retains its original metadata identity.

- [x] Write failing tests for variable mode, nominal parity, mapped endpoints, and held-torque behavior. Use identical nonsaturated state/target to isolate gain effects:

```python
expected_kp = nominal_kp * torch.pow(p.new_tensor(1.25), p)
expected_kd = nominal_kd * torch.sqrt(torch.pow(p.new_tensor(1.25), p))
expected_request = expected_kp * position_error - expected_kd * joint_velocity
expected_torque = torch.maximum(torch.minimum(expected_request, upper), lower)
torch.testing.assert_close(controller.applied_kp, expected_kp)
torch.testing.assert_close(controller.applied_kd, expected_kd)
torch.testing.assert_close(controller.torque[:, controller.ids], expected_torque)
```

- [x] Run those tests and confirm failures arise from missing behavior.
- [x] Validate modes, calibrated nominals, and gain base once. Reuse the tensor-only gain map in the controller hot path; avoid per-substep host synchronization. On scheduled samples, snapshot mapped gains and compute/clamp torque; preserve the existing eight-step hold/native-feedback clearing.
- [x] Add explicit requested torque and saturation snapshots, with applied `Kp/Kd` for all six controlled joints. Do not expose native mapped gains as applied sampled gains.
- [x] Test each endpoint, distinct environments/joints, full and subset resets, effort limits, hooks, and ten updates in an 80-step interval. A changed request mid-hold must leave held torque unchanged until the next scheduled sample.
- [x] Verify `p=0` variable torque and fixed torque agree at matched states and that fixed-mode behavior is unchanged.

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_sampled_pd500.py tests/test_variable_impedance.py -q
```

**Acceptance:** actual torque authority and timing are established independently of learned behavior.

### Task 2: Make controller selection and checkpoint identity consistent

**Files:** `scripts/train.py`, `scripts/play.py`, `scripts/evaluate_impact_speed.py`, `src/tasks/hammer/rl/impact_speed_contract.py`, `src/tasks/hammer/rl/runner.py`; extend routing/contract/evaluation tests.

**Interfaces:** `sampled_pd500_gain_mode: Literal["fixed", "variable"] = "fixed"`; controller identity `sampled_vic500_v1`, action type `joint_position_variable_impedance_sampled_pd`. Add no flag that silently changes a loaded checkpoint's controller.

- [x] Add failing tests for native/fixed/variable mismatch rejection, variable mode without sampled control, and playback/evaluation construction order.
- [x] Install the selected sampled controller before wrapping the environment/constructing the runner. Preserve native and legacy fixed routes. Factor only the small shared setup/validation needed by these three entry points.
- [x] Serialize calibrated nominals, gain mapping, timing, effort limits, joint order, load settings, and path/guidance semantics for the new VIC contract. Keep old serialized contracts unchanged. Normal resume is exact; only the existing explicit external-to-policy speed transition is permitted.
- [x] Make controller metadata validation account for live variable gains without treating ordinary gain changes as drift. Changing nominal values, limits, clocks, joint mapping, or hooks must still fail validation.
- [x] Update applied-gain reads in evaluation and export metadata. Test PT save/load and deterministic export parity.

```python
with pytest.raises(ValueError, match="mismatch"):
    validate_impact_speed_metadata(fixed_metadata, variable_metadata)
with pytest.raises(ValueError, match="mismatch"):
    validate_impact_speed_metadata(native_metadata, variable_metadata)
validate_impact_speed_metadata(variable_metadata, variable_metadata)
```

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_impact_speed_contract.py tests/test_impact_speed_train_routing.py tests/test_impact_speed_evaluation.py tests/test_impact_speed_wrapper.py -q
```

**Acceptance:** a checkpoint cannot be evaluated under a physically different controller while appearing compatible.

### Task 3: Expose flexible guidance and the three-start task

**Files:** `src/tasks/hammer/config/z1/__init__.py`, `src/tasks/hammer/mdp/references.py` only for a pure shared diagnostic helper; `tests/test_quintic_reference.py`, `tests/test_imitation_reward.py`, `tests/test_configs.py`; new `tests/test_spline_impact_config.py`.

**Interfaces:** a shared private quintic configuration builder with explicit route mode and guidance weight/width; register `Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed` separately. Preserve `Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed` defaults and meaning.

- [x] Test left-only configuration equality against the snapshot baseline, persistent precontact guidance, and unchanged active CaT settings.
- [x] Validate weight as finite/nonnegative and width as finite/positive. Support controlled overrides without an automatic sweep or curriculum.
- [x] Construct the three-start task using the existing three reset poses and independent equal-probability sampling; exact per-batch route counts are not required. Apply identical quintic/follow-through parameters to reset, observations, and reward consumers.
- [x] Test each reset anchor, nail crossing, terminal tangent, vertical continuation, subset reset isolation, and observation/action schema.
- [x] Test diagnostics for paths sampled at different rates, off-curve positions, and rebound without mutating the reference's monotone latch. Keep policy phase semantics unchanged.
- [x] Check whether current observations distinguish the three supplied geometries using paired states. Record the result; a failure blocks a generalization claim and calls for a separate minimal observation change, not a silent schema expansion.

```python
assert cfg.rewards["r_imit"].weight == 0.2
assert "r_imit_anneal" not in cfg.curriculum
assert cfg.metrics["cat_soft"].params["imp_max_p"] == 0.2
assert cfg.metrics["cat_soft"].params["imp_limit"] == [0.369, 0.246, 0.738, 0.369, 0.246, 0.0164]
```

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_quintic_reference.py tests/test_imitation_reward.py tests/test_spline_impact_config.py tests/test_configs.py -q
```

**Acceptance:** geometry and guidance are configurable and testable without introducing a duration or altering the baseline.

### Task 4: Add common-window impact and honest path/load measurements

**Files:** create `src/tasks/hammer/mdp/strike_evaluation.py` for the focused evaluation collector and `tests/test_strike_evaluation.py`; integrate through `scripts/evaluate_impact_speed.py`; touch `first_strike.py` only if a minimal read-only onset signal is required.

**Interfaces:** `StrikeEvaluationCollector`, installed only for evaluation as a per-substep metric after both the existing first-strike tracker and `substep_impulse`. It owns independent counters/buffers and observes the accepted onset once. Its output explicitly separates native and common-window records.

- [x] Write tensor/synthetic tests for the exact interval convention, axial/transverse force integrals, all-joint exceedances, precontact path bins, coverage, and subset reset.
- [x] Use 200 inclusive physics samples from accepted onset for sampled control. Count wall-clock physics samples even across contact gaps; integrate force only for the intended contact pair. Freeze complete outputs so later substeps cannot add impulse.
- [x] Test a constant positive axial force analytically, with independently checked expected values:

```python
expected_impulse_ns = 10.0 * 200 * 0.00025  # 0.5 N s
assert result.window_samples == 200
assert result.window_complete
assert result.axial_positive_ns == pytest.approx(expected_impulse_ns)
```

- [x] Add protocol selection `--measurement-protocol native|common-window` with native as the compatibility default. Validate original checkpoint metadata before applying evaluation overrides. Defer only success reset until the common window completes; record success at its native time, and preserve failures/timeouts.
- [x] Test success on onset and on the final sample, contact gaps, failure before completion, timeout, simultaneous success/failure, and different onset times in a vectorized batch. Never integrate reset-state forces into the previous event.
- [x] Sample current-height path error before accepted contact, with 16 normalized-height bins. Report observed-bin RMS/max, sample RMS/max, counts, coverage, skipped bins, and path extent. Do not fill empty bins or penalize postcontact motion as approach error. A hovering synthetic trace must show low progress/coverage and a miss despite small geometric error.
- [x] Read the authoritative `SubstepImpulseAccumulator.impulse` after its update every physics substep. Test callback insertion order and a one-substep exceedance between policy reads; the collector must retain it even when CaT's cached 50 Hz telemetry has not updated.
- [x] Report each joint's whole-observed-episode and common-window peaks, configured cap, utilization/exceedance, and cadence. Common-window peak is the maximum of existing sliding-window Lambda during the interval, including any pre-onset ring history; do not reset the ring or relabel it an onset-only integral. Test an earlier sensor contact followed by accepted onset. Keep no-contact/failed attempts in summaries. Preserve native tracker impulse and stop reason.
- [x] Test that enabling diagnostic collection under native protocol leaves actions, observations, rewards, terminations, and RNG unchanged. Common-window behavior may diverge only at the explicitly recorded termination override.

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_strike_evaluation.py tests/test_impact_speed_evaluation.py tests/test_first_strike_event.py tests/test_impulse_bound.py tests/test_impulse_constraint.py -q
```

**Acceptance:** completed windows really have equal duration, censored windows are labeled, and path/load summaries cannot hide missing evidence.

### Task 5: Run focused integration checks and hand off the evidence

**Files:** extend relevant integration tests; create `docs/results/2026-09-24-free-time-spline-software-qualification.md`; update the implementation checkout's docs index with this software-only result.

- [x] Run a small CPU simulator smoke test with synthetic actions, both sampled gain modes, all three reset poses, and asynchronous resets. This is not a learning run.
- [x] Exercise the existing post-latch intervention harness with a deterministic test actor whose later motor outputs depend on the goal: identical first action, different second action, one-time speed latch. Preserve the existing masked-distribution and checkpoint-transfer tests.
- [x] If a compatible fixed checkpoint is locally available, verify fixed replay construction without relabeling it VIC. Absence of a trained VIC checkpoint is expected; record physical speed compliance as untested.
- [x] Run the consolidated focused regression:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_sampled_pd500.py tests/test_variable_impedance.py tests/test_impact_speed.py tests/test_impact_speed_policy.py tests/test_impact_speed_contract.py tests/test_impact_speed_wrapper.py tests/test_impact_speed_train_routing.py tests/test_impact_speed_evaluation.py tests/test_quintic_reference.py tests/test_imitation_reward.py tests/test_spline_impact_config.py tests/test_strike_evaluation.py tests/test_first_strike_event.py tests/test_impulse_bound.py tests/test_impulse_constraint.py tests/test_cat_soft_hook.py tests/test_cat_ppo_gae.py tests/test_cat_storage.py -q
```

- [x] Report failures, skips, simulator availability, command exit status, versions, source manifest, and test counts. No full-suite or physical-learning claim follows from a focused regression.
- [x] Obtain independent whole-change review, resolve important findings through a fix agent and scoped re-review, and publish a compact result table.
- [x] Stop. Do not launch training or modify existing jobs.

**Handoff table:** applied gain authority / hold timing / reset isolation / checkpoint separation / free-time geometry / exact window / per-joint load reporting / speed plumbing are software-testable. Learned path quality / physical speed responsiveness / impact improvement / compliance rates / three-curve competence require later experiments. Hardware safety requires a separately validated load model and limits.

## Later experiments, deliberately outside execution

1. Before learning, assess useful-impact versus per-joint-load feasibility at matched 1.5/1.85/2.2 m/s goals with complete/censored measurement status. The inherited J6 cap remains a watch item; earlier native-controller failures neither certify current sampled compliance nor prove impossibility.
2. Existing LEFT curve: qualify sampled VIC with externally varied speed goals; compare fixed nominal gains and trained-VIC nominal-gain clamp.
3. Same controller: train the autonomous speed selector and run paired post-latch goal interventions.
4. Train one policy across the three-start mixture; report each route separately.
5. Sweep a small predeclared set of guidance widths/weights only after the above mechanisms work; plot delivered impact versus path error and each joint's limit utilization.

Use seed-level replication before scientific superiority claims. Treat a constant selected speed as potentially valid. Freeze the primary assessment, sampling protocol, and acceptance criteria before examining those results.

## User review

Execution method is already selected: subagent-driven development. Review the design and plan together before implementation begins. The implementation milestone ends at Task 5; training remains outside it.
