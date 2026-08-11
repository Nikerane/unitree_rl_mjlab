# Z1 Controlled-Drop `I_ref` Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the simulation-only primary controlled-drop calibration, producing five raw production first-event nail-axis impulses and their arithmetic mean as the banked candidate `I_ref`.

**Architecture:** Add a calibration-only mjlab environment containing the unchanged production nail/block entity and a passive `0.200 kg` cylindrical impactor on one vertical slide joint. Freeze the initial lower-face-to-nail-head clearance at exactly `h0=0.150 m`, reuse the production `FirstStrikeEventTracker`, execute five fresh one-environment releases, and publish one small fail-closed JSON result. CPU tests qualify the contract and fixture; a clean pinned Vega A100 run is the authoritative calibration execution.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, mujoco-warp 3.8.1, pytest, Slurm on Vega A100.

## Global Constraints

- Work only in `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal`.
- Start execution from commit `86499b9f9e76498c403d6d5fefd8fd73c756d510` on a new branch named `z1-controlled-drop-iref`; do not rewrite `z1-joint-position-fixed`.
- Preserve every Cartesian and joint-policy task and every owner-owned untracked research/result path.
- This plan calibrates only the primary delivered-impulse reference. It does not train FIC-0/FIC-TT, modify either live `I_ref` constant, rescale D4, activate impulse CaT, add curriculum/domain randomization, change gains, or begin VIC.
- Run exactly five sequential fresh one-environment releases. Do not retry, filter, weight, trim, or exclude a trial.
- Primary dropper: mass `0.200 kg`, cylinder radius `0.012 m`, cylinder half-height `0.004 m`, centered above the nail, released from rest, constrained to the nail axis by one frictionless and undamped slide joint, with no rotational degree of freedom.
- Dropper contact side: `friction=(1.5, 0.02, 0.002)`, `contype=3`, `conaffinity=3`, and default MuJoCo `solref`/`solimp`, matching the production `hammer_head_0` side. The production `nail_head` keeps its existing radius, half-height, mass, collision bits, friction, `solref`, and `solimp` unchanged.
- The radius and half-height deliberately match the production `nail_head` primitive. This defines a centered flat contact coupon; it is not claimed to reproduce the claw-hammer face geometry or arm effective mass.
- **Owner correction (2026-08-11):** define `h0` as exactly `0.150 m` (15 cm) of axial surface-to-surface clearance from the dropper's lower face to the production nail-head top at release. Do not derive or replace it with the current hammer mesh's approximately `0.143078 m` closest-geom distance, and do not expose a height override.
- Deep-copy the complete live production FIC simulation config, whose load-bearing values are `timestep=0.002`, `iterations=10`, `ls_iterations=20`, `impratio=10`, and `cone="elliptic"`; do not reconstruct a partial solver config.
- Use the production `FirstStrikeEventTracker`: axis `(0, 0, -1)`, `window_substeps=25`, `progress_eps=5e-4`, inclusive success/window finalization, and its measured pre-contact velocity and axial delivered impulse.
- Fail the primary calibration if any release does not start at zero velocity, make contact, finalize, report a finite positive pre-contact velocity, report a finite positive axial impulse, or report a productive first event. Require all five finalized events to have one common reason (`success` or `window`); production semantics forbid pooling the two horizons. Preserve diagnostic output, but do not publish a primary result from fewer than five valid homogeneous trials.
- The arithmetic mean is `math.fsum(trial.impulse_n_s for trial in trials) / 5`. Do not round before storing it.
- CPU mujoco-warp is the development and integration-test gate. Native MuJoCo is used only to compile the production nail geometry needed to place and verify the fixed-clearance fixture; a clean, exact-revision Vega A100 `cuda:0` mujoco-warp run is the authoritative calibration result.
- Fail the authoritative run unless the live versions are exactly mjlab `1.4.0`, MuJoCo `3.8.1`, and mujoco-warp `3.8.1`; record those live values in the result.
- Prefix local commands that instantiate the new fixture with the plan-owned writable caches `MPLCONFIGDIR=/private/tmp/z1-controlled-drop-matplotlib` and `WARP_CACHE_PATH=/private/tmp/z1-controlled-drop-warp`; this avoids managed-workspace cache failures without touching repository or owner files.
- Do not add mass, height, radius, orientation, repeat-count, contact, solver, or backend sweep flags. The only runtime choices are output path, device, and the already-verified code/asset revisions recorded with the result.
- Do not add a general evaluator, manifest framework, payload hash, plotting pipeline, or automatic parameter-tuning loop.
- Implement each behavior through a public seam using one RED test followed by only enough GREEN implementation. Do not test private helpers or duplicate `FirstStrikeEventTracker`'s internal unit suite.
- Execute each task with a fresh implementer subagent, then a separate task reviewer that returns both spec-compliance and code-quality verdicts. Resolve Critical/Important findings through the bounded SDD fix loop before continuing.

---

## Approved Test Seams

These are the public boundaries under test. Tests may inspect the compiled model returned by the environment seam, but must not assert private helper calls or private tracker buffers.

1. `summarize_primary_trials(trials) -> ControlledDropSummary`: validates the five public trial records and computes the unfiltered arithmetic mean.
2. `make_controlled_drop_env_cfg() -> ManagerBasedRlEnvCfg`: produces the fixed-`0.150 m` calibration-only physical fixture with production nail/contact/tracker semantics and no height parameter.
3. `capture_clean_execution_identity(expected_code_revision, expected_asset_revision) -> tuple[str, str]`: resolves the fixed code/asset roots, requires tracked cleanliness, and returns the observed matching SHAs.
4. `run_primary_calibration(device, expected_code_revision, expected_asset_revision) -> ControlledDropResult`: binds the observed clean revisions, performs the five fresh releases, rechecks identity, and returns the complete public result.
5. `scripts/calibrate_controlled_drop.py`: writes one JSON result only after the public result passes all invariants.

## File Map

- Create `src/tasks/hammer/calibration/__init__.py`: exports the small public calibration API.
- Create `src/tasks/hammer/calibration/controlled_drop_contract.py`: immutable records, strict validation, arithmetic-mean reducer, and JSON conversion.
- Create `src/tasks/hammer/calibration/controlled_drop_env.py`: fixed-`0.150 m` calibration-only mjlab fixture.
- Create `src/tasks/hammer/calibration/controlled_drop.py`: five-release orchestration and tracker-to-trial conversion.
- Create `scripts/calibrate_controlled_drop.py`: fixed primary CLI and fail-closed result publication.
- Create `tests/test_controlled_drop_contract.py`: pure behavioral contract tests.
- Create `tests/test_controlled_drop_env.py`: compiled fixture and one-release integration tests.
- Create `tests/test_calibrate_controlled_drop.py`: five-release and CLI/result-writer tests.
- Create during authoritative execution `evaluation/results/2026-08-11_z1_controlled_drop_iref/primary.json`: five raw trials plus the unrounded mean.
- Create after result audit `docs/results/2026-08-11_z1_controlled_drop_iref.md`: concise interpretation and exact execution evidence.

## SDD Execution Setup

The controller, not an implementer, performs this setup once:

```bash
cd /Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal
test "$(git rev-parse HEAD)" = "86499b9f9e76498c403d6d5fefd8fd73c756d510"
git status --short --branch
git switch -c z1-controlled-drop-iref
git add docs/superpowers/plans/2026-08-11-z1-controlled-drop-iref.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs(hammer): plan controlled drop calibration"
git status --short --branch
```

Before the plan commit, status may contain only this plan plus the six pre-existing owner-owned untracked paths recorded in the Task-5 handoff. The staged list must contain only this plan. After the commit, status may contain only those six owner-owned paths; abort if any tracked path is dirty. This plan commit becomes the SDD execution base, while whole-branch reviews continue to compare against `86499b9f9e76498c403d6d5fefd8fd73c756d510`.

Initialize the plan-owned SDD workspace and ledger:

```bash
SDD_SKILL=/Users/nikerane/.codex/plugins/cache/openai-curated-remote/superpowers/6.2.0/skills/subagent-driven-development
PLAN=docs/superpowers/plans/2026-08-11-z1-controlled-drop-iref.md
"$SDD_SKILL/scripts/sdd-workspace" "$PLAN"
```

Create `progress.md` in the printed workspace with this exact first line:

```text
# SDD ledger — plan: docs/superpowers/plans/2026-08-11-z1-controlled-drop-iref.md
```

Before every task, record `BASE=$(git rev-parse HEAD)`, generate its brief with `scripts/task-brief`, and dispatch exactly one implementer. After its commit, generate `scripts/review-package "$PLAN" "$BASE" HEAD` and dispatch a fresh task reviewer. Never run two implementers concurrently in this shared worktree.

## Unattended Decision Rules

- Continue autonomously through implementation, focused tests, the single full CPU gate, review/fix loops, Vega submission/monitoring, result audit, and evidence banking when every frozen invariant passes.
- A scheduler wait is not a blocker; monitor the submitted job. An allocation or A100 preflight failure before the Python calibration starts may be resubmitted once with the identical SHA, assets, resources, and command, recording both job IDs.
- Once the Python calibration starts, any invalid/partial trial, mixed finalization horizon, revision drift, non-finite value, or nonzero exit is a genuine scientific blocker. Preserve logs and stop; do not retry a release, tune contact, change height/mass/geometry, or silently rerun the experiment.
- Fix concrete code/review failures test-first within the frozen design. Stop for owner input only if a correction would change the scientific protocol or another explicit scope boundary.
- After the valid candidate mean is banked and pushed, stop. Normalizer propagation and FIC training require the separate follow-on plan.

---

### Task 1: Define the five-trial calibration contract

**Files:**
- Create: `src/tasks/hammer/calibration/__init__.py`
- Create: `src/tasks/hammer/calibration/controlled_drop_contract.py`
- Create: `tests/test_controlled_drop_contract.py`

**Interfaces:**
- Produce: `ControlledDropTrial`
- Produce: `ControlledDropSummary`
- Produce: `ControlledDropExecution`
- Produce: `ControlledDropResult`
- Produce: `summarize_primary_trials(trials: Sequence[ControlledDropTrial]) -> ControlledDropSummary`
- Produce: `build_primary_result(execution: ControlledDropExecution, trials: Sequence[ControlledDropTrial]) -> ControlledDropResult`
- Produce: `ControlledDropResult.to_json_dict() -> dict[str, object]`

- [ ] **Step 1: Write the first failing public-contract test**

  Create `tests/test_controlled_drop_contract.py` with a literal worked example independent of the reducer:

  ```python
  from src.tasks.hammer.calibration.controlled_drop_contract import (
      ControlledDropTrial,
      summarize_primary_trials,
  )


  def _trial(index: int, impulse: float) -> ControlledDropTrial:
      return ControlledDropTrial(
          trial_index=index,
          h0_m=0.150,
          release_velocity_m_s=0.0,
          precontact_velocity_m_s=1.70,
          impulse_n_s=impulse,
          contacted=True,
          finalized=True,
          productive=True,
          reason="success",
          depth_at_contact_m=0.0,
          peak_depth_m=0.031,
      )


  def test_primary_summary_keeps_all_five_trials_and_uses_arithmetic_mean():
      trials = tuple(
          _trial(index, impulse)
          for index, impulse in enumerate((0.10, 0.20, 0.30, 0.40, 0.50), start=1)
      )

      summary = summarize_primary_trials(trials)

      assert summary.trials == trials
      assert summary.i_ref_mean_n_s == 0.30
  ```

- [ ] **Step 2: Run the test and witness RED**

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
    -m pytest tests/test_controlled_drop_contract.py::test_primary_summary_keeps_all_five_trials_and_uses_arithmetic_mean -q
  ```

  Expected: FAIL during import because `src.tasks.hammer.calibration` does not exist.

- [ ] **Step 3: Implement the smallest immutable trial and summary records**

  Implement these public shapes and `math.fsum(...)/5` only:

  ```python
  @dataclass(frozen=True)
  class ControlledDropTrial:
      trial_index: int
      h0_m: float
      release_velocity_m_s: float
      precontact_velocity_m_s: float
      impulse_n_s: float
      contacted: bool
      finalized: bool
      productive: bool
      reason: Literal["success", "window"]
      depth_at_contact_m: float
      peak_depth_m: float


  @dataclass(frozen=True)
  class ControlledDropSummary:
      trials: tuple[ControlledDropTrial, ...]
      i_ref_mean_n_s: float
  ```

  `summarize_primary_trials` must preserve input order and must not compute any filtered alternative.

- [ ] **Step 4: Run the first test and witness GREEN**

  Run the command from Step 2. Expected: `1 passed`.

- [ ] **Step 5: Add one failing validation slice at a time**

  Add parametrized public-seam tests for these exact failures, running each new case before implementing it:

  ```python
  @pytest.mark.parametrize(
      ("mutate", "message"),
      [
          (lambda rows: rows[:-1], "exactly five"),
          (lambda rows: rows + (rows[-1],), "exactly five"),
          (lambda rows: rows[:2] + (replace(rows[2], trial_index=4),) + rows[3:], "indices"),
          (lambda rows: (replace(rows[0], trial_index=True),) + rows[1:], "numeric type"),
          (lambda rows: rows[:2] + (replace(rows[2], h0_m=float("nan")),) + rows[3:], "h0"),
          (lambda rows: tuple(replace(row, h0_m=0.149) for row in rows), "exactly 0.150"),
          (lambda rows: rows[:2] + (replace(rows[2], release_velocity_m_s=0.01),) + rows[3:], "released from rest"),
          (lambda rows: rows[:2] + (replace(rows[2], precontact_velocity_m_s=float("inf")),) + rows[3:], "pre-contact velocity"),
          (lambda rows: rows[:2] + (replace(rows[2], impulse_n_s=float("nan")),) + rows[3:], "impulse"),
          (lambda rows: rows[:2] + (replace(rows[2], impulse_n_s=0.0),) + rows[3:], "impulse"),
          (lambda rows: rows[:2] + (replace(rows[2], precontact_velocity_m_s=0.0),) + rows[3:], "pre-contact velocity"),
          (lambda rows: rows[:2] + (replace(rows[2], contacted=False),) + rows[3:], "contacted"),
          (lambda rows: rows[:2] + (replace(rows[2], finalized=False),) + rows[3:], "finalized"),
          (lambda rows: rows[:2] + (replace(rows[2], productive=False),) + rows[3:], "productive"),
          (lambda rows: rows[:2] + (replace(rows[2], reason="other"),) + rows[3:], "reason"),
          (lambda rows: rows[:2] + (replace(rows[2], reason="window"),) + rows[3:], "one finalization reason"),
          (lambda rows: rows[:2] + (replace(rows[2], depth_at_contact_m=float("nan")),) + rows[3:], "depth"),
          (lambda rows: rows[:2] + (replace(rows[2], peak_depth_m=0.0005),) + rows[3:], "depth progress"),
      ],
  )
  def test_primary_summary_fails_closed_on_invalid_trial_sets(mutate, message):
      with pytest.raises(ValueError, match=message):
          summarize_primary_trials(mutate(_valid_trials()))
  ```

  After adding each row to the parameter list, run:

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
    -m pytest tests/test_controlled_drop_contract.py::test_primary_summary_fails_closed_on_invalid_trial_sets -q
  ```

  Witness the new row fail for the named invariant, implement only that validation, and rerun it GREEN before adding the next row. Reject Python booleans in every numeric field before converting to `float` or `int`.

- [ ] **Step 6: RED — specify the complete result envelope and exact JSON shape**

  Freeze `ControlledDropExecution` fields:

  ```text
  code_revision, asset_revision, device, backend, mujoco_version,
  mujoco_warp_version, mjlab_version, physics_dt_s, h0_m, mass_kg,
  radius_m, half_height_m, friction, slide_axis, slide_damping,
  slide_frictionloss, tracker_axis, tracker_window_substeps,
  tracker_progress_eps
  ```

  Add this exact public-schema test, using `_execution()` with literal valid values for every frozen field:

  ```python
  def test_primary_result_json_has_one_execution_and_nested_unfiltered_summary():
      result = build_primary_result(_execution(), _valid_trials())

      payload = result.to_json_dict()

      assert set(payload) == {"schema_version", "execution", "summary"}
      assert payload["schema_version"] == 1
      assert payload["execution"]["code_revision"] == "a" * 40
      assert [row["trial_index"] for row in payload["summary"]["trials"]] == [1, 2, 3, 4, 5]
      assert [row["impulse_n_s"] for row in payload["summary"]["trials"]] == [
          0.1,
          0.2,
          0.3,
          0.4,
          0.5,
      ]
      assert payload["summary"]["i_ref_mean_n_s"] == 0.3
      json.dumps(payload, sort_keys=True, allow_nan=False)
  ```

- [ ] **Step 7: Run the result-envelope test and witness RED**

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
    -m pytest tests/test_controlled_drop_contract.py::test_primary_result_json_has_one_execution_and_nested_unfiltered_summary -q
  ```

  Expected: FAIL because `ControlledDropExecution`, `ControlledDropResult`, and `build_primary_result` are absent.

- [ ] **Step 8: GREEN — implement only the frozen execution/result records**

  `ControlledDropResult` contains `schema_version=1`, one execution record, one summary, and `to_json_dict()`. Serialize tuples as JSON arrays without altering floats or dropping rows. Run the command from Step 7; expected: `1 passed`.

- [ ] **Step 9: Run Task-1 tests and commit**

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
    -m pytest tests/test_controlled_drop_contract.py -q
  git diff --check
  git add \
    src/tasks/hammer/calibration/__init__.py \
    src/tasks/hammer/calibration/controlled_drop_contract.py \
    tests/test_controlled_drop_contract.py
  git commit -m "feat(hammer): define controlled drop calibration contract"
  ```

  The implementer report must record each witnessed RED failure cause and final GREEN count. The task reviewer must verify spec compliance and test quality before the ledger receives `Task 1: complete`.

---

### Task 2: Build the fixed-`0.150 m` calibration fixture

**Files:**
- Create: `src/tasks/hammer/calibration/controlled_drop_env.py`
- Modify: `src/tasks/hammer/calibration/__init__.py`
- Create: `tests/test_controlled_drop_env.py`

**Interfaces:**
- Consume: `get_nail_block_entity_cfg()` and the production FIC-0 task registration.
- Produce: `make_controlled_drop_env_cfg() -> ManagerBasedRlEnvCfg`
- Produce constants: `PRIMARY_DROP_H0_M`, `PRIMARY_DROP_MASS_KG`, `PRIMARY_DROP_RADIUS_M`, `PRIMARY_DROP_HALF_HEIGHT_M`, `PRIMARY_DROP_FRICTION`, `PRIMARY_AXIS`, `PRIMARY_PHYSICS_DT_S`, `PRIMARY_WINDOW_SUBSTEPS`, `PRIMARY_PROGRESS_EPS`

- [ ] **Step 1: RED — specify the exact fixed-clearance fixture**

  Add `test_primary_fixture_compiles_exact_protocol`. First require `PRIMARY_DROP_H0_M == 0.150` and `inspect.signature(make_controlled_drop_env_cfg).parameters == {}`, proving there is no height override. Build `make_controlled_drop_env_cfg()`, instantiate one CPU environment, and assert through the compiled `MjModel` and public config that:

  ```text
  action dimension = 0
  exactly one dropper joint named robot/drop_axis
  drop_axis type = slide, axis = (0,0,-1), damping = 0, frictionloss = 0
  no free or hinge joint exists on the dropper
  dropper body mass = 0.200 kg
  robot/dropper_face type = cylinder, size = (0.012,0.004)
  robot/dropper_face friction = (1.5,0.02,0.002)
  robot/dropper_face contype = 3, conaffinity = 3
  initial dropper qpos = 0 and qvel = 0
  initial lower-face-to-nail-head axial surface clearance = exactly 0.150 m
  production nail joint/geom/contact fields match get_nail_block_entity_cfg()
  solver values match the Global Constraints
  rewards, commands, curriculum, and terminations are empty
  metrics contains only FirstStrikeEventTracker under "first_strike"
  tracker params match axis/window/progress constants exactly
  ```

- [ ] **Step 2: Run the fixture test and witness RED**

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-controlled-drop-matplotlib WARP_CACHE_PATH=/private/tmp/z1-controlled-drop-warp PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
    -m pytest tests/test_controlled_drop_env.py::test_primary_fixture_compiles_exact_protocol -q
  ```

  Expected: FAIL because `make_controlled_drop_env_cfg` is absent.

- [ ] **Step 3: GREEN — construct only the passive drop fixture**

  Build a fresh `mujoco.MjSpec` with this exact physical core:

  ```python
  body = spec.worldbody.add_body(
      name="dropper",
      pos=(
          nail_head_center_w[0],
          nail_head_center_w[1],
          nail_contact_surface_z_m + PRIMARY_DROP_H0_M + 0.004,
      ),
  )
  body.add_joint(
      name="drop_axis",
      type=mujoco.mjtJoint.mjJNT_SLIDE,
      axis=(0.0, 0.0, -1.0),
      damping=0.0,
      frictionloss=0.0,
  )
  body.add_geom(
      name="dropper_face",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(0.012, 0.004, 0.0),
      mass=0.200,
      friction=(1.5, 0.02, 0.002),
      contype=3,
      conaffinity=3,
  )
  body.add_site(name="dropper_face_site", pos=(0.0, 0.0, -0.004))
  ```

  Resolve `nail_head_center_w` and `nail_contact_surface_z_m` from a compiled fresh production nail spec; do not duplicate the current world-space center or `0.100 m` contact height. The compiled fixture test must require the dropper and nail-head XY centers to agree within `1e-12 m`.

  Configure two sensors matching `ContactMatch(mode="geom", pattern="dropper_face", entity="robot")` to `ContactMatch(mode="body", pattern="nail", entity="nail_block")`. Name them `hammer_nail_contact` (`reduce="maxforce"`, `fields=("found", "force")`, `track_air_time=True`) and `hammer_nail_impulse` (`reduce="netforce"`, `fields=("found", "force")`, `track_air_time=False`). The regexes are entity-local; do not prefix either pattern with a compiled entity name. Wire the tracker with `SceneEntityCfg("robot", site_names=("dropper_face_site",))` and `SceneEntityCfg("nail_block", joint_names=("nail_slide",), site_names=("nail_top",))`; these names are also entity-local and must not carry `robot/` or `nail_block/` prefixes.

  Use `decimation=1`, empty actions/observations/rewards/terminations/commands/curriculum, default deterministic reset only, and a deep copy of `load_env_cfg(FIC0_TASK, play=True).sim`. The test must compare the entire simulation config and both compiled models: require `nconmax=64`, `njmax=300`, and equality of compiled `timestep`, gravity, integrator, solver, iterations, tolerance, line-search iterations/tolerance, CCD iterations, impedance ratio, cone, Jacobian mode, disable flags, and enable flags. This makes the copied live config—not a partial list—the authority.

- [ ] **Step 4: Run the fixture test and witness GREEN**

  Run the command from Step 2. Expected: `1 passed`.

- [ ] **Step 5: Run Task-2 tests and the existing physics contracts**

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-controlled-drop-matplotlib WARP_CACHE_PATH=/private/tmp/z1-controlled-drop-warp PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_controlled_drop_env.py \
    tests/test_first_strike_event.py \
    tests/test_nail_physics.py \
    tests/test_impulse_bound.py
  git diff --check
  ```

- [ ] **Step 6: Commit Task 2**

  ```bash
  git add \
    src/tasks/hammer/calibration/__init__.py \
    src/tasks/hammer/calibration/controlled_drop_env.py \
    tests/test_controlled_drop_env.py
  git commit -m "feat(hammer): add controlled drop calibration fixture"
  ```

  The task reviewer must inspect the compiled model evidence and confirm the production task registrations and `env_cfgs.py` constants are unchanged.

---

### Task 3: Run exactly five fresh releases and publish one result

**Files:**
- Create: `src/tasks/hammer/calibration/controlled_drop.py`
- Modify: `src/tasks/hammer/calibration/__init__.py`
- Create: `scripts/calibrate_controlled_drop.py`
- Modify: `tests/test_controlled_drop_env.py`
- Create: `tests/test_calibrate_controlled_drop.py`

**Interfaces:**
- Consume: Task-1 records/reducer and Task-2 fixture.
- Produce: `run_primary_calibration(*, device: str, expected_code_revision: str, expected_asset_revision: str) -> ControlledDropResult`
- Produce: `capture_clean_execution_identity(*, expected_code_revision: str, expected_asset_revision: str) -> tuple[str, str]`
- Produce: `write_primary_result(path: Path, result: ControlledDropResult) -> None`
- Produce CLI: `scripts/calibrate_controlled_drop.py --device DEVICE --output PATH --code-revision SHA --asset-revision SHA`

- [ ] **Step 1: RED — require one real release to reach the public tracker result**

  Extend `tests/test_controlled_drop_env.py` with a CPU integration test:

  ```python
  @pytest.mark.integration
  def test_one_primary_drop_contacts_and_finalizes_with_finite_positive_measurements():
      trial = run_one_primary_drop(device="cpu", trial_index=1)

      assert trial.trial_index == 1
      assert trial.release_velocity_m_s == 0.0
      assert trial.contacted is True
      assert trial.finalized is True
      assert trial.productive is True
      assert trial.reason in {"success", "window"}
      assert math.isfinite(trial.precontact_velocity_m_s)
      assert trial.precontact_velocity_m_s > 0.0
      assert math.isfinite(trial.impulse_n_s)
      assert trial.impulse_n_s > 0.0
  ```

  `run_one_primary_drop` is public because it is the independently reviewable physical trial seam. It accepts only `device` and `trial_index`; scientific parameters are constants.

- [ ] **Step 2: Run the integration test and witness RED**

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-controlled-drop-matplotlib WARP_CACHE_PATH=/private/tmp/z1-controlled-drop-warp PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
    -m pytest tests/test_controlled_drop_env.py::test_one_primary_drop_contacts_and_finalizes_with_finite_positive_measurements -q
  ```

  Expected: FAIL because `run_one_primary_drop` does not exist.

- [ ] **Step 3: GREEN — implement one bounded physical trial**

  `run_one_primary_drop` must create a fresh environment, call `reset()`, assert the compiled initial slide velocity is exactly zero, and repeatedly call:

  ```python
  env.step(torch.empty((1, 0), dtype=torch.float32, device=env.device))
  ```

  Resolve the tracker instance through `env.metrics_manager.cfg["first_strike"].func`, assert it is a `FirstStrikeEventTracker`, and stop when its public `finalized` property is true. Fail after `500` physics substeps without finalization. Map only its public `started`, `finalized`, `productive`, `v_precontact`, `delivered`, `depth_at_contact`, `peak_depth`, and `reason` properties to one `ControlledDropTrial`, then close the environment in `finally`. Do not read the environment's `_hammer_first_strike` attribute or any tracker buffer whose name begins with `_`.

- [ ] **Step 4: RED — specify five fresh releases**

  Add a test that replaces only the public physical trial seam with five literal completed trial records and asserts orchestration behavior:

  ```python
  def test_primary_calibration_calls_five_ordered_fresh_trials_and_keeps_every_row(monkeypatch):
      calls = []

      def fake_run_one(*, device, trial_index):
          calls.append((device, trial_index))
          return _trial(trial_index, impulse=trial_index / 10)

      monkeypatch.setattr(controlled_drop, "run_one_primary_drop", fake_run_one)
      monkeypatch.setattr(
          controlled_drop,
          "capture_clean_execution_identity",
          lambda **kwargs: ("a" * 40, "b" * 40),
      )
      result = run_primary_calibration(
          device="cpu",
          expected_code_revision="a" * 40,
          expected_asset_revision="b" * 40,
      )

      assert calls == [("cpu", 1), ("cpu", 2), ("cpu", 3), ("cpu", 4), ("cpu", 5)]
      assert [row.impulse_n_s for row in result.summary.trials] == [0.1, 0.2, 0.3, 0.4, 0.5]
      assert result.summary.i_ref_mean_n_s == 0.3
  ```

  This mock is at the approved public trial seam; it does not mock tracker internals.

- [ ] **Step 5: GREEN — implement fixed orchestration**

  Loop over `range(1, 6)` with no retry branch. Read library versions from live package metadata and fail if they differ from the frozen versions in Global Constraints. Validate the two expected revisions as lowercase 40-character hexadecimal strings. Set `backend="mujoco-warp"`, preserve the requested device string, and fill every frozen execution field from Task-2 constants. Each real environment must report the requested device exactly. Run the Task-3 orchestration test by exact node ID and witness GREEN.

- [ ] **Step 6: RED then GREEN — bind recorded revisions to the live clean repositories**

  Add two public-runner tests. `test_primary_calibration_rejects_revision_mismatch_or_tracked_dirt_before_any_drop` mocks only the Git subprocess boundary and asserts each mismatch/dirty case raises before the mocked `run_one_primary_drop` is called. `test_primary_calibration_rejects_repository_change_after_fifth_drop` returns clean matching identities before trial 1, changes one observed SHA or tracked status on the post-trial read, asserts all five trial calls occurred, and asserts no `ControlledDropResult` is returned. Then implement the public `capture_clean_execution_identity` seam inside `controlled_drop.py` so it:

  - derives the code root from `Path(__file__).resolve()` and the loaded asset root from `Z1_HAMMER_XML.parents[2]`;
  - obtains both actual full SHAs with `git -C ROOT rev-parse HEAD`;
  - requires `git -C ROOT status --porcelain --untracked-files=no` to be empty, deliberately ignoring owner-owned untracked files;
  - compares actual SHAs to the expected arguments before trial 1;
  - repeats the SHA and cleanliness checks after trial 5 and fails if either repository changed;
  - stores the observed actual SHAs, never the caller strings, in `ControlledDropExecution`.

  Run this exact node after writing it (RED), implement only the checks above, and rerun it GREEN:

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_calibrate_controlled_drop.py::test_primary_calibration_rejects_revision_mismatch_or_tracked_dirt_before_any_drop \
    tests/test_calibrate_controlled_drop.py::test_primary_calibration_rejects_repository_change_after_fifth_drop
  ```

- [ ] **Step 7: RED — specify the CLI's narrow surface and fail-closed writer**

  Add these named tests:

  - `test_cli_accepts_only_output_device_and_expected_revisions`: parse the four supported options, restrict device to `cpu` or `cuda:0`, and reject `--mass`, `--height`, `--runs`, `--radius`, `--friction`, and `--window-substeps`;
  - `test_writer_publishes_nested_complete_json_without_clobber`: require `sort_keys=True`, `indent=2`, `allow_nan=False`, refuse an existing output path, reload the nested JSON, and independently verify `math.fsum(row["impulse_n_s"] for row in payload["summary"]["trials"]) / 5 == payload["summary"]["i_ref_mean_n_s"]`;
  - `test_cli_failure_leaves_no_final_or_temporary_result`: make calibration raise and assert neither a final output nor its plan-owned temporary sibling remains.

  Run them before implementation:

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_calibrate_controlled_drop.py::test_cli_accepts_only_output_device_and_expected_revisions \
    tests/test_calibrate_controlled_drop.py::test_writer_publishes_nested_complete_json_without_clobber \
    tests/test_calibrate_controlled_drop.py::test_cli_failure_leaves_no_final_or_temporary_result
  ```

  Expected: FAIL because the parser and writer are absent.

- [ ] **Step 8: GREEN — add the fixed primary CLI and publication**

  `scripts/calibrate_controlled_drop.py` must call the public orchestration once and publish only after it returns. Write to a sibling temporary file, flush and `os.fsync`, then publish without clobber by atomically hard-linking the temporary file to the non-existing destination and unlinking the temporary name. A pre-check alone followed by `os.replace` or `rename` is forbidden because it races and can overwrite. Delete only that plan-owned temporary file on failure.

  The CLI must print exactly one concise completion block containing output path, five raw impulses, unrounded mean, device, and observed code/asset revisions. It must not choose whether the mean should be installed into training. Run the command from Step 7; expected: all three tests pass.

- [ ] **Step 9: Run Task-3 focused tests**

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-controlled-drop-matplotlib WARP_CACHE_PATH=/private/tmp/z1-controlled-drop-warp PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_controlled_drop_contract.py \
    tests/test_controlled_drop_env.py \
    tests/test_calibrate_controlled_drop.py \
    tests/test_first_strike_event.py \
    tests/test_nail_physics.py \
    tests/test_impulse_bound.py

  ```

- [ ] **Step 10: Commit Task 3**

  ```bash
  git diff --check
  git add \
    src/tasks/hammer/calibration/__init__.py \
    src/tasks/hammer/calibration/controlled_drop.py \
    scripts/calibrate_controlled_drop.py \
    tests/test_controlled_drop_env.py \
    tests/test_calibrate_controlled_drop.py
  git commit -m "feat(hammer): run primary controlled drop calibration"
  ```

- [ ] **Step 11: Run the clean real CPU CLI path**

  ```bash
  CAL_CPU_DIR=$(mktemp -d /private/tmp/z1-controlled-drop-cpu.XXXXXX)
  CAL_CPU_RESULT="$CAL_CPU_DIR/primary.json"
  CAL_CODE_SHA=$(git rev-parse HEAD)
  CAL_ASSET_SHA=$(git -C /Users/nikerane/repos/safe_impact_manipulation rev-parse HEAD)
  MPLCONFIGDIR=/private/tmp/z1-controlled-drop-matplotlib WARP_CACHE_PATH=/private/tmp/z1-controlled-drop-warp PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
    scripts/calibrate_controlled_drop.py \
    --device cpu \
    --output "$CAL_CPU_RESULT" \
    --code-revision "$CAL_CODE_SHA" \
    --asset-revision "$CAL_ASSET_SHA"
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m json.tool "$CAL_CPU_RESULT" >/dev/null
  ```

  The tracked worktree must be clean before this command. The five CPU impulses may be exploratory values only; they must satisfy the contract but are not the authoritative `I_ref`.

  The task reviewer must verify both verdicts, confirm the real CPU integration evidence, and check that no live reward/config constant changed.

---

### Task 4: Verify, review, execute cleanly on Vega, and bank the result

**Files:**
- Create during execution: `evaluation/results/2026-08-11_z1_controlled_drop_iref/primary.json`
- Create after audit: `docs/results/2026-08-11_z1_controlled_drop_iref.md`
- Modify only concrete correctness fixes identified by review, each preceded by a failing regression test.

**Interfaces:**
- Consume: the complete Task-1 through Task-3 public API and CLI.
- Produce: one revision-pinned authoritative JSON result and one concise reviewed result record.
- Explicitly do not produce: a modified `I_REF_FIRST_STRIKE_SUCCESS`, modified `I_REF_DELIVERED`, training run, or CaT intervention.

- [ ] **Step 1: Run the final implementation review fan-out before the scientific execution**

  Generate one whole-branch review package from `86499b9f9e76498c403d6d5fefd8fd73c756d510` to `HEAD`, including this plan and the SDD ledger. Send the same immutable package read-only to Opus Max, Gemini, and DeepSeek, and run the repository `code-review` skill's independent Standards and Spec lanes. Ask the model lanes respectively to emphasize scientific semantics, fixture/API feasibility, and numerical/fail-closed behavior. Consolidate duplicate findings by exact file/line and require the repository review to end with both:

  ```text
  Spec compliance: PASS
  Code quality: APPROVED
  ```

  Any concrete Critical/Important correctness finding receives one failing regression test, one fix subagent, and one scoped re-review by the originating lane plus the repository Spec lane. Reject unsupported style opinions and do not convert a requested mass/height/contact sensitivity into implementation scope. If a named external reviewer is temporarily unavailable, record that operational fact and substitute a fresh high-reasoning reviewer on the identical package; do not block or broaden the experiment.

- [ ] **Step 2: Run focused tests and the full CPU suite once on the reviewed implementation revision**

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-controlled-drop-matplotlib WARP_CACHE_PATH=/private/tmp/z1-controlled-drop-warp PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_controlled_drop_contract.py \
    tests/test_controlled_drop_env.py \
    tests/test_calibrate_controlled_drop.py \
    tests/test_first_strike_event.py \
    tests/test_nail_physics.py \
    tests/test_impulse_bound.py

  MPLCONFIGDIR=/private/tmp/z1-controlled-drop-matplotlib WARP_CACHE_PATH=/private/tmp/z1-controlled-drop-warp PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q
  git diff --check
  git status --short --branch
  ```

  Run the full suite exactly once unless a production-code fix is subsequently required. If a fix is required, run the focused covering tests and record why a second full gate is necessary. If `HEAD` differs from the clean CPU CLI revision recorded in Task 3 Step 11, repeat that exact five-release CPU CLI command on the reviewed `HEAD` before pushing.

- [ ] **Step 3: Commit any reviewed fixes, push, and pin the exact clean Vega revisions**

  Push `z1-controlled-drop-iref`. On Vega, create a detached worktree next to the asset repository rather than changing the user's existing checkout:

  ```bash
  CAL_SOURCE_REPO=/ceph/hpc/home/eunikhilr/repos/unitree_rl_mjlab
  CAL_ASSET_DIR=/ceph/hpc/home/eunikhilr/repos/safe_impact_manipulation
  git -C "$CAL_SOURCE_REPO" fetch origin z1-controlled-drop-iref
  CAL_CODE_SHA=$(git -C "$CAL_SOURCE_REPO" rev-parse origin/z1-controlled-drop-iref)
  CAL_CODE_DIR="/ceph/hpc/home/eunikhilr/repos/z1-controlled-drop-$CAL_CODE_SHA"
  test ! -e "$CAL_CODE_DIR"
  git -C "$CAL_SOURCE_REPO" worktree add --detach "$CAL_CODE_DIR" "$CAL_CODE_SHA"
  test "$(git -C "$CAL_CODE_DIR" rev-parse HEAD)" = "$CAL_CODE_SHA"
  test -z "$(git -C "$CAL_CODE_DIR" status --porcelain --untracked-files=no)"
  test "$(git -C "$CAL_ASSET_DIR" rev-parse HEAD)" = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
  test -z "$(git -C "$CAL_ASSET_DIR" status --porcelain --untracked-files=no)"
  ```

  Record `CAL_CODE_SHA`, the fixed full asset SHA, and `CAL_CODE_DIR` in the ledger before submitting. Do not include untracked cache files in either identity decision.

- [ ] **Step 4: Submit one bounded authoritative Vega job**

  Use account `d2026d06-166-users`, partition `gpu`, one GPU, exclude `gn03,gn34`, and assert the allocated device name contains `A100`. Choose a revision-specific non-existing output path and submit exactly one calibration invocation:

  ```bash
  CAL_SOURCE_REPO=/ceph/hpc/home/eunikhilr/repos/unitree_rl_mjlab
  CAL_CODE_SHA=$(git -C "$CAL_SOURCE_REPO" rev-parse origin/z1-controlled-drop-iref)
  CAL_CODE_DIR="/ceph/hpc/home/eunikhilr/repos/z1-controlled-drop-$CAL_CODE_SHA"
  test "$(git -C "$CAL_CODE_DIR" rev-parse HEAD)" = "$CAL_CODE_SHA"
  CAL_RESULT_PATH="/ceph/hpc/home/eunikhilr/z1-controlled-drop-primary-$CAL_CODE_SHA.json"
  test ! -e "$CAL_RESULT_PATH"
  sbatch \
    --parsable \
    --account=d2026d06-166-users \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=8G \
    --time=00:10:00 \
    --exclude=gn03,gn34 \
    --chdir="$CAL_CODE_DIR" \
    --output=/ceph/hpc/home/eunikhilr/z1-controlled-drop-%j.out \
    --error=/ceph/hpc/home/eunikhilr/z1-controlled-drop-%j.err \
    --export=ALL,CAL_CODE_SHA="$CAL_CODE_SHA",CAL_RESULT_PATH="$CAL_RESULT_PATH" \
    --wrap='set -eu
  CAL_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/z1-controlled-drop-${SLURM_JOB_ID}"
  mkdir -p "$CAL_CACHE_DIR/matplotlib" "$CAL_CACHE_DIR/warp"
  test "$(git rev-parse HEAD)" = "$CAL_CODE_SHA"
  test -z "$(git status --porcelain --untracked-files=no)"
  test "$(git -C /ceph/hpc/home/eunikhilr/repos/safe_impact_manipulation rev-parse HEAD)" = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
  test -z "$(git -C /ceph/hpc/home/eunikhilr/repos/safe_impact_manipulation status --porcelain --untracked-files=no)"
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -q A100
  MPLCONFIGDIR="$CAL_CACHE_DIR/matplotlib" WARP_CACHE_PATH="$CAL_CACHE_DIR/warp" GIT_PYTHON_REFRESH=quiet PYTHONPATH=. /ceph/hpc/home/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python scripts/calibrate_controlled_drop.py --device cuda:0 --output "$CAL_RESULT_PATH" --code-revision "$CAL_CODE_SHA" --asset-revision b58ccd2f81fd246f27c1e8d88cf86484cd888703'
  ```

  The job must exit `0`; retain Slurm job ID, node, A100 name, elapsed time, stdout/stderr, `CAL_CODE_SHA`, and `CAL_RESULT_PATH`. The script independently rechecks and records the observed revisions before and after all five releases.

- [ ] **Step 5: Retrieve and independently audit the result before banking it**

  From the local worktree, recompute the implementation SHA and retrieve its revision-specific result without relying on a previous shell's variables:

  ```bash
  CAL_CODE_SHA=$(git rev-parse HEAD)
  CAL_LOCAL_RESULT=evaluation/results/2026-08-11_z1_controlled_drop_iref/primary.json
  test ! -e "$CAL_LOCAL_RESULT"
  mkdir -p evaluation/results/2026-08-11_z1_controlled_drop_iref
  scp "vega:/ceph/hpc/home/eunikhilr/z1-controlled-drop-primary-$CAL_CODE_SHA.json" "$CAL_LOCAL_RESULT"
  ```

  A fresh result-review subagent must verify directly from JSON:

  ```python
  import json
  import math
  import subprocess
  from pathlib import Path

  payload = json.loads(
      Path("evaluation/results/2026-08-11_z1_controlled_drop_iref/primary.json").read_text()
  )

  trials = payload["summary"]["trials"]
  assert len(trials) == 5
  assert [row["trial_index"] for row in trials] == [1, 2, 3, 4, 5]
  independent = math.fsum(row["impulse_n_s"] for row in trials) / 5
  assert independent == payload["summary"]["i_ref_mean_n_s"]
  assert all(row["contacted"] and row["finalized"] and row["productive"] for row in trials)
  assert all(row["h0_m"] == 0.150 for row in trials)
  assert all(row["release_velocity_m_s"] == 0.0 for row in trials)
  assert all(row["precontact_velocity_m_s"] > 0.0 for row in trials)
  assert all(row["impulse_n_s"] > 0.0 for row in trials)
  assert len({row["reason"] for row in trials}) == 1
  assert all(
      row["peak_depth_m"] - row["depth_at_contact_m"] > 5e-4
      for row in trials
  )
  expected_code_revision = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True
  ).strip()
  assert payload["execution"]["code_revision"] == expected_code_revision
  assert payload["execution"]["asset_revision"] == "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
  assert payload["execution"]["device"] == "cuda:0"
  assert payload["execution"]["backend"] == "mujoco-warp"
  assert payload["execution"]["mjlab_version"] == "1.4.0"
  assert payload["execution"]["mujoco_version"] == "3.8.1"
  assert payload["execution"]["mujoco_warp_version"] == "3.8.1"
  assert payload["execution"]["physics_dt_s"] == 0.002
  assert payload["execution"]["h0_m"] == 0.150
  assert payload["execution"]["mass_kg"] == 0.200
  assert payload["execution"]["radius_m"] == 0.012
  assert payload["execution"]["half_height_m"] == 0.004
  assert payload["execution"]["friction"] == [1.5, 0.02, 0.002]
  assert payload["execution"]["slide_axis"] == [0.0, 0.0, -1.0]
  assert payload["execution"]["slide_damping"] == 0.0
  assert payload["execution"]["slide_frictionloss"] == 0.0
  assert payload["execution"]["tracker_axis"] == [0.0, 0.0, -1.0]
  assert payload["execution"]["tracker_window_substeps"] == 25
  assert payload["execution"]["tracker_progress_eps"] == 5e-4
  ```

  The reviewer also checks exact code/asset/device/backend/physics/fixture/tracker identities and confirms no trial was retried or excluded. A null scientific surprise is not a reason to alter the result.

- [ ] **Step 6: Write the concise result record**

  Create `docs/results/2026-08-11_z1_controlled_drop_iref.md` with:

  ```text
  objective and approved protocol
  exact code/asset revisions and Slurm job evidence
  exact h0=0.150 m and the five raw pre-contact velocities
  the five raw first-event impulses
  arithmetic mean candidate I_ref
  finalization reasons and productive flags
  observed spread, reported descriptively without an acceptance threshold
  limitations: simulation-only, constrained cylinder, not arm effective mass
  explicit next decision: propagate the exact banked mean in a separate FIC integration plan
  ```

  Do not claim physical safety calibration or convert this object-side scalar into a per-joint CaT cap.

- [ ] **Step 7: Stage and independently review both banked artifacts**

  ```bash
  git add \
    evaluation/results/2026-08-11_z1_controlled_drop_iref/primary.json \
    docs/results/2026-08-11_z1_controlled_drop_iref.md
  git diff --cached --name-only
  git diff --cached --check
  git diff --cached
  ```

  The staged list must contain exactly the two paths shown. Dispatch a fresh artifact reviewer to check the staged JSON and Markdown together against this plan, the raw Slurm logs, and the independent arithmetic audit. Require both `Spec compliance: PASS` and `Code/data quality: APPROVED`; correct any artifact-only finding and restage before continuing.

- [ ] **Step 8: Run the final whole-branch code/spec review**

  Run the repository `code-review` skill from `86499b9f9e76498c403d6d5fefd8fd73c756d510` through the current working tree, including the two staged evidence files, covering both Standards and Spec. If any production code changes after the authoritative run, add a failing regression test, fix and commit the implementation separately, rerun the relevant focused and full-suite gates, push the new implementation revision, replace the staged result with a fresh five-release Vega result from that exact SHA, and repeat both artifact and whole-branch reviews; never bank a result produced by an older implementation SHA. Artifact-only corrections need artifact re-review but do not require a physics rerun.

- [ ] **Step 9: Commit the fully reviewed evidence**

  ```bash
  git commit -m "docs(hammer): bank controlled drop impulse reference"
  ```

- [ ] **Step 10: Push and stop at the calibration boundary**

  ```bash
  git push origin z1-controlled-drop-iref
  test "$(git rev-parse HEAD)" = "$(git rev-parse origin/z1-controlled-drop-iref)"
  git status --short --branch
  ```

  Confirm that all six pre-existing owner-owned untracked paths remain untouched.

  Report the exact candidate mean and evidence. Do not modify `I_REF_FIRST_STRIKE_SUCCESS=0.3088`, `I_REF_DELIVERED=0.6094`, D4 weight `4.0`, FIC registrations, training launchers, or CaT settings in this plan. The next plan can use the now-known literal mean to update the event normalizer and execute the matched one-seed FIC-0/FIC-TT pilot without placeholders.

---

## Acceptance Checklist

- [ ] All Task-1 through Task-3 behavior was implemented through witnessed RED then GREEN cycles at the approved public seams.
- [ ] Every task has a clean task-scoped spec and quality review in the SDD ledger.
- [ ] Final whole-branch review is clean or every residual non-load-bearing item is explicitly ruled in the ledger.
- [ ] Production Cartesian/joint tasks, rewards, gains, CaT settings, and both existing `I_ref` constants are unchanged.
- [ ] Compiled fixture has the exact mass, geometry, friction, collision bits, slide joint, solver, nail physics, and tracker contract in Global Constraints.
- [ ] `PRIMARY_DROP_H0_M` is exactly `0.150 m`, the compiled initial lower-face-to-nail-head axial clearance is exactly `0.150 m`, and no height override or hammer-mesh-derived replacement exists.
- [ ] Exactly five fresh releases ran with no retries, exclusions, sweeps, or parameter changes.
- [ ] Every trial is finalized, productive, finite, positive, and released from rest.
- [ ] All five trials share one finalization reason; success- and window-finalized horizons are never pooled.
- [ ] The stored mean equals `math.fsum(all five raw impulses)/5` exactly.
- [ ] The authoritative run used `cuda:0` on a clean exact revision and an asserted Vega A100.
- [ ] The result JSON and result record were independently audited and committed separately from implementation.
- [ ] Existing owner-owned untracked research notes and result directories remain untouched.
- [ ] Work stops before normalizer propagation, FIC training, active impulse-CaT, curriculum/domain randomization, or VIC.
