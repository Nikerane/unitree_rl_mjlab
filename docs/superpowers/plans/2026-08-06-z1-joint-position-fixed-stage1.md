# Z1 Joint-Position Fixed-Impedance Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the learned Cartesian DiffIK action with a qualified six-joint desired-position action while retaining the current task-space P trajectory guidance, fixed actuator gains, velocity CaT, D4 delivered-impulse reward, and log-only impulse CaT. Establish a stable fixed-gain joint-space baseline before the controlled-drop, active impulse-CaT, curriculum/domain-randomization, and VIC stages.

**Architecture:** Preserve every existing Cartesian task. Build two matched joint-position fixed-gain treatments from the registered P+V+D4 treatment, then replace only their action term with mjlab 1.4.0 `JointPositionActionCfg`. The policy emits six normalized actions; each maps to an absolute default-offset desired joint position, is clipped to the physical XML limit, and is held for ten 500 Hz physics substeps. Derive per-joint scales from the first causal DiffIK target in each 20 ms reference interval and prove the same scripted strike remains feasible before training. Keep the P tracker entirely in task space. FIC-0 omits command tracking; FIC-TT adds the paper-aligned one-step joint-command tracking contribution as a nonnegative cost with reward weight `-1.0`, explicitly outside soft-CaT's scaled positive return. Train and evaluate both arms with matched seeds and budgets.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, RSL-RL/CatPPO, pytest, the existing first-strike/guideline evaluation pipeline, and Slurm on Vega A100 nodes.

## Global Constraints

- Work locally only in `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal`.
- Create the implementation branch `codex/z1-joint-position-fixed-stage1` in this same worktree. Do not use the stale `joint-position-fic` branch at `83b050b` and do not create a new `/private/tmp` worktree.
- Preserve every existing Cartesian registration, signature, test, checkpoint contract, and presentation result.
- Fixed impedance only. Keep the current `BuiltinPositionActuator` gains and effort limits; do not call `set_gains`, introduce gain actions, or change actuator classes.
- Command exactly `joint1` through `joint6`; never command `jointGripper`.
- Use `JointPositionActionCfg(use_default_offset=True)`. Never use `RelativeJointPositionActionCfg`, current-state-relative deltas, or a runtime IK layer.
- Freeze the wrapper/action equation as:

  ```text
  a_policy,j = actor output before the RSL-RL wrapper
  a_env,j = clamp(a_policy,j, -1, 1)       # wrapper only; unwrapped env does not do this
  q_processed,j = clamp(q_default,j + scale_j * a_env,j, q_min,j, q_max,j)
  q_des_applied,j = q_processed,j - encoder_bias,j
  ```

- Retain the 50 Hz policy rate, 500 Hz physics rate, and ten-substep zero-order hold.
- Retain the current fixed-reset P contract: robot joint reset position and velocity ranges are both `(0.0, 0.0)`. Reset randomization belongs to the later curriculum/domain-randomization stage.
- Retain both task-space guidance channels. The base `SingleStrikeReference` contributes four observations (`strike_phase` plus the three-component `strike_ref_error`) and no `r_imit`. The P tracker contributes seven additional observations, six ordered waypoints, and `r_waypoint_progress` weight `8.0`.
- Retain velocity CaT at 500 Hz, the D4 delivered reward weight `4.0`, current impulse caps, and `imp_max_p=0.0`. This stage must not lower the impulse threshold or activate impulse CaT.
- Retain all seven existing task reward weights and the normalized raw-action rate penalty weight `-0.01`. Because the action dimension changes from three to six, report its dose; do not silently rescale or redesign it.
- During 4,096-environment training, retain the existing production performance override `substep_impulse_rows.enabled=False`; the active first-strike and impulse-CaT trackers remain enabled.
- Keep `I_ref=0.3088 N s` unchanged in this stage. The controlled-drop calibration is the next stage and must not be mixed into this branch.
- Do not warm-start a Cartesian checkpoint. The actor input changes from 44 to 47 and its output from 3 to 6, so train the joint policy from scratch.
- The scripted joint-target tape is qualification evidence only. Never expose it to PPO as an observation, imitation target, warm start, policy prior, or runtime controller.
- Generate scientific evidence only from a clean, revision-pinned code checkout and clean asset checkout. The current local untracked presentation result directories must remain untouched and must never be silently staged, moved, ignored, or deleted.

## Frozen Treatment Identity

| Item | Frozen value |
| --- | --- |
| Matched Cartesian parent | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4` |
| Joint FIC-0 task | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed` |
| Joint FIC-TT task | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed-TT` |
| Policy action | six absolute default-offset desired joint positions |
| Task-space guidance | P waypoint progress, weight `8.0` |
| Joint command tracking | absent in FIC-0; calibrated `r_tt` in FIC-TT |
| Delivered-impulse reward | D4, weight `4.0` |
| Velocity CaT | active, substep peak, limit `3.1415 rad/s` |
| Impulse CaT | measured but log-only, `imp_max_p=0.0` |
| Impedance | existing fixed gains |
| Reset | fixed `(0.0, 0.0)` position and velocity offsets |
| Action dimensions | Cartesian `3`; joint `6` |
| Observation dimensions | Cartesian P `44`; joint P `47` |
| Training | both arms at seed `2`, then both arms at seeds `3` and `4`; 200 iterations; 4,096 envs; 24 steps/env/iteration |
| Checkpoint rule | final `model_199.pt`; never select a better intermediate checkpoint |
| Sampled evaluator RNG | base `2026072900`; reset `2036072919`; observation `2046072933`; action `2056072941` |
| Sampled policy | stochastic actions; actor corruption on; critic corruption off; two completed episodes/env |
| Fixed CPU policy | one env, deterministic mean action, fixed reset, observation corruption off |

FIC-0 and FIC-TT are both behavioral treatments. Launch their seed-2 pilots as a pair; expand both to seeds 3 and 4 only if both independently pass the same absolute task/safety gates. No directional improvement from `r_tt` is required for promotion: report its effect on task success, productive impact, nail impulse, per-joint reaction impulse, command-tracking error, action saturation, joint-limit proximity, and velocity legality. Metrics that require an active reward term—positive raw `r_tt` cost and negative weighted return—apply only to FIC-TT; the corresponding expected value for FIC-0 is absence of the term. The later impedance-isolation comparison is FIC-TT versus VIC-TT with the same effective `r_tt`. Comparison to old Cartesian P+V+D4 policies remains descriptive.

## Physical and Controller Contract

The production target order is exactly:

```text
(joint1, joint2, joint3, joint4, joint5, joint6)
```

The XML position clips are:

```text
joint1: [-2.61799,  2.61799]
joint2: [ 0.00000,  2.96706]
joint3: [-2.87979,  0.00000]
joint4: [-1.51844,  1.51844]
joint5: [-1.34390,  1.34390]
joint6: [-2.79253,  2.79253]
```

The current fixed gains remain:

```text
joint1, joint3-joint6: Kp=1000, Kd=100, effort=30
joint2:                Kp=1500, Kd=150, effort=60
```

Mjlab resolves action targets in articulation order, not the listed actuator-registry order. `preserve_order=True` does not fix this path. Every config, smoke, evaluator, and metadata test must assert the live action term's `target_names`, IDs, and dimension rather than trusting configuration order.

Mjlab applies `processed_target - encoder_bias` to the articulation. The current Z1
encoder bias is zero. Assert that live six-joint bias is exactly zero in this stage;
if the asset later introduces a bias, update and requalify the action equation rather
than silently treating the processed target as the applied target.

## Scale and Trackability Decisions

For each joint, derive the action scale from the causal reference target tape:

```text
scale_j = 1.10 * (max_t |q_target(t,j) - q_default,j| + 0.05 rad)
```

The `0.05 rad` term is retained as explicit exploration authority, not as reset-noise compensation; the current P reset is fixed. The `1.10` factor supplies ten-percent margin. The resulting numeric values are generated and banked; provisional audit values must never be copied into production configuration by hand.

The effective primary command-trackability contribution is:

```text
Delta R_tt(t) = -k_tt * ||q_des(t) - q(t+1)||^2
```

- `q_des(t)` is the public applied six-joint target for the current policy interval.
- `q(t+1)` is the realized six-joint position after all ten physics substeps.
- The term remains active through contact.
- `k_tt = 0.1 / q90(||q_des-q_next||^2)` on the qualified scripted replay.
- Reject a non-finite calibration or `q90 < 1e-8`.
- Require each replay's cumulative pre-`dt` cost `sum_t k_tt ||q_des-q_next||^2 <= 5.0`.
- Implement `joint_trackability_cost = k_tt * ||q_des-q_next||^2` as a nonnegative function and configure reward term `r_tt` with weight `-1.0`.
- Add `r_tt` to `CatSoftHook._NEG_TERMS`, so velocity violations never multiply this penalty by `(1-delta)`. A unit test must prove `reward_buf = r_pos + r_neg` and that `r_tt` rides through unscaled for nonzero delta.

This joint command-tracking term is distinct from the P task-space trajectory guidance. Both are active in FIC-TT; P guidance remains active in FIC-0 as well.

## Stop/Go Gates

| Gate | Required evidence | Failure action |
| --- | --- | --- |
| G0 — source control | Current docs safely checkpointed; new branch created; existing result directories untouched | Stop before editing code |
| G1 — causal replay | 16/16 fixed-reset seeded replays succeed with exact ordering, finite state, zero clipping/saturation, legal qvel, six waypoints, and accepted strike | Stop; do not train or tune gains |
| G2 — production integration | Joint task is 6-action/47-observation; old Cartesian tasks unchanged; full CPU tests pass | Fix through TDD before any GPU work |
| G3 — live backend | CPU and CUDA smokes prove affine mapping, ten-substep hold, fixed gains, P+V+D4 identity, active velocity CaT, and log-only impulse CaT | Stop before pilot training |
| G4 — paired one-seed learnability | The seed-2 final checkpoint for both FIC-0 and FIC-TT independently passes the preregistered 64-episode pilot criteria | Diagnose one arm/cause at a time; expand neither arm |
| G5 — paired three-seed stability | Seeds 2, 3, and 4 for both arms pass the final 512-episode-per-checkpoint criteria | Report instability; do not enter the drop/VIC stages |

Seeds `1000` through `1015` in G1 are deterministic repeatability runs under the same fixed reset, not 16 different reset poses and not statistical replicates. They are retained to detect hidden RNG, ordering, and backend drift. Their causal target-tape hashes must be identical; otherwise the fixed-play contract is not deterministic and qualification fails.

## File Map

### New files

- `src/tasks/hammer/config/z1/data/` — new directory for the two frozen generated contracts.
- `evaluation/joint_position/__init__.py` — package marker.
- `evaluation/joint_position/qualify_joint_action.py` — pure tape/scale/decision helpers plus the deterministic capture-and-replay CLI.
- `evaluation/joint_position/calibrate_trackability.py` — derive and validate `k_tt` from the qualified replay.
- `evaluation/joint_position/evaluate_fixed_rollout.py` — one-environment deterministic fixed CPU companion for the joint task.
- `src/tasks/hammer/config/z1/joint_position_contract.py` — fail-closed loaders for the two generated artifacts.
- `src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json` — ordered targets, physical clips, derived scales, source tape, replay rows, digests, and PASS decision.
- `src/tasks/hammer/config/z1/data/z1_joint_trackability_stage1.json` — frozen `k_tt`, q90, replay dose, and source-artifact digest.
- `src/tasks/hammer/mdp/trackability.py` — pure six-joint error readers and nonnegative trackability cost.
- `tests/test_joint_action_qualification.py` — causal reduction, scale, artifact, and decision tests.
- `tests/test_joint_position_config.py` — additive registration and exact-isolation tests.
- `tests/test_joint_trackability.py` — reward timing, indexing, and calibration tests.
- `scripts/smoke_joint_position_fixed.py` — dedicated CPU/CUDA live-manager gate.
- `tests/test_smoke_joint_position_fixed.py` — CPU rehearsal and mutation tests for the smoke.
- `evaluation/analysis/joint_position_stage1.py` — episode-first joint/action metrics and three-seed summary.
- `tests/test_joint_position_stage1_analysis.py` — aggregation and decision-rule tests.
- `tests/test_joint_position_fixed_rollout.py` — fixed CPU rollout identity, trace, and pre-reset tests.
- `scripts/slurm/vega_joint_position_stage1.sbatch` — fail-closed seed-2 pilot and seeds-3/4 expansion launcher.
- `scripts/slurm/vega_joint_position_stage1_eval.sbatch` — fail-closed sampled-CUDA plus fixed-CPU evaluation launcher.
- `docs/results/2026-08-06_joint_position_fixed_stage1_preregistration.md` — frozen campaign identity before training.
- `docs/results/2026-08-06_joint_position_fixed_stage1_result.md` — populated only after G5.

### Existing files to modify

- `src/tasks/hammer/config/z1/env_cfgs.py` — additive helper that replaces DiffIK with the qualified joint action.
- `src/tasks/hammer/config/z1/__init__.py` — register the matched FIC-0 and FIC-TT tasks from the exact P+V+D4 parent.
- `src/tasks/hammer/mdp/__init__.py` — export trackability readers.
- `src/tasks/hammer/cat/hook.py` — classify `r_tt` as an unscaled negative term in the soft-CaT return split.
- `src/tasks/hammer/rl/runner.py` — typed Cartesian/joint metadata dispatch.
- `scripts/eval_impulse.py` — separate fail-closed Stage-1 campaign and variable-width joint traces.
- `tests/test_eval_impulse_hook.py` — joint contract mutation tests while retaining all historical Cartesian assertions.
- `tests/test_env.py` — a separate joint fixture and live 6/47 integration checks.
- `tests/test_configs.py` — add registrations without weakening the frozen existing task set or Cartesian signatures.
- `tests/test_cat_soft_hook.py` — prove nonzero CaT delta cannot discount the `r_tt` penalty.
- `tests/test_slurm_launchers.py` — exact launcher task, seeds, iterations, provenance, and override rejection.
- `docs/results/2026-08-06_PRESENTATION_HANDOFF.md` — link this executable plan; do not alter presentation evidence.

### Files deliberately unchanged

- `src/tasks/hammer/hammer_env_cfg.py` — the default action remains Cartesian DiffIK.
- `src/assets/robots/unitree_z1/z1_constants.py` and all robot/MJCF assets — no gain, mass, actuator, or geometry change.
- `src/tasks/hammer/mdp/guideline.py`, `observations.py`, and `references.py` — the realized task-space tracker is action-interface independent.
- `scripts/smoke_wave3_pv.py`, `scripts/diag_impulse_trace.py`, and the existing Cartesian reference scripts — retain their frozen historical contracts.

---

### Task 0: Safely checkpoint the meeting work and create the implementation branch

**Files:**
- Existing meeting/documentation changes already present in the worktree.
- This plan and its handoff link.

- [ ] **Step 1: Inventory without altering the worktree**

```bash
cd /Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal
git status --short --branch
git branch --list 'codex/z1-joint-position-fixed-stage1'
```

Require the two existing untracked presentation result directories to remain visible and unchanged. If the target branch name already exists, stop and inspect it; do not reuse or overwrite it automatically.

- [ ] **Step 2: Re-run the documentation guard**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_docs_current.py -q
git diff --check
```

- [ ] **Step 3: Confirm ownership, then create the feature branch before committing**

The implementation worker must show the current diff summary to the owner and
confirm that the seven named documentation changes belong in this checkpoint. Do
not infer ownership from their modified status. After confirmation:

```bash
git switch -c codex/z1-joint-position-fixed-stage1
git status --short --branch
```

Creating the branch first leaves the `overnight-impulse-minimal` branch pointer at
its existing commit.

- [ ] **Step 4: Stage only the explicitly reviewed documentation files**

Use explicit paths. Do not use `git add .`, a directory-wide add, or a glob.

```bash
git add docs/README.md
git add docs/archive/2026-06-17-z1-strike-not-press-redesign-design.md
git add docs/archive/REAL_HAMMER_PLAN.md
git add docs/research/reward-design/OPEN_QUESTIONS.md
git add docs/results/2026-08-06_PRESENTATION_HANDOFF.md
git add docs/superpowers/plans/2026-07-05-docs-audit-worklist.md
git add docs/superpowers/plans/2026-08-06-z1-joint-position-fixed-stage1.md
git diff --cached --name-only
```

The staged list must contain exactly those seven paths. Review `git diff --cached` before committing.

- [ ] **Step 5: Commit the meeting checkpoint on the feature branch**

```bash
git commit -m "docs(thesis): freeze post-meeting execution path"
git status --short --branch
```

The untracked presentation result directories may still make the local checkout provenance-dirty. That is acceptable for implementation, not for banked qualification or training evidence.

---

### Task 1: Define the causal target-tape and artifact contract with pure TDD

**Files:**
- Create: `evaluation/joint_position/__init__.py`
- Create: `evaluation/joint_position/qualify_joint_action.py`
- Create: `src/tasks/hammer/config/z1/joint_position_contract.py`
- Create: `tests/test_joint_action_qualification.py`

**Interfaces:**

```python
reduce_first_target_per_interval(targets_500hz, decimation=10) -> np.ndarray
derive_scale_by_joint(tape, q_default, exploration_floor=0.05, margin=1.10) -> np.ndarray
normalized_action_for_target(tape, q_default, scale) -> np.ndarray
qualification_passes(rows, expected_seeds=tuple(range(1000, 1016))) -> bool
load_joint_position_contract(path) -> JointPositionContract
```

- [ ] **Step 0: Create the two new package/data directories**

```bash
mkdir -p evaluation/joint_position
mkdir -p src/tasks/hammer/config/z1/data
```

- [ ] **Step 1: Write failing pure tests**

At minimum, freeze these examples:

```python
def test_reduction_uses_first_target_not_interval_future():
    targets = np.arange(20 * 6, dtype=np.float64).reshape(20, 6)
    reduced = reduce_first_target_per_interval(targets, decimation=10)
    np.testing.assert_array_equal(reduced, targets[[0, 10]])


def test_scale_has_exploration_floor_and_margin():
    tape = np.array([[0.2, -0.1], [0.4, 0.3]])
    default = np.array([0.1, 0.0])
    expected = 1.10 * (np.array([0.3, 0.3]) + 0.05)
    np.testing.assert_allclose(derive_scale_by_joint(tape, default), expected)


def test_normalized_tape_round_trips_without_saturation():
    action = normalized_action_for_target(tape, default, scale)
    assert np.max(np.abs(action)) < 1.0
    np.testing.assert_allclose(default + scale * action, tape)
```

Also reject wrong width, missing/permuted joints, missing seeds, non-finite JSON values, non-physical clips, inconsistent tape hashes, a non-PASS decision, and an artifact whose internal SHA-256 does not match its canonical sorted payload.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_joint_action_qualification.py -q
```

Expected: import or assertion failures because none of the interfaces exists yet.

- [ ] **Step 3: Implement only the pure functions and strict schema loader**

The JSON schema must include:

```text
schema_version
action_semantics
source_task_id
source_code_revision
source_asset_revision
source_task_config_sha256
joint_names
actuator_names
default_joint_pos_rad
physical_clip_rad
scale_rad
physics_dt_s
control_decimation
post_reference_hold_control_steps
seeds
source_target_tape_rad
source_target_tape_sha256
per_seed_replay_rows
decision
payload_sha256
```

Serialize with sorted keys and `allow_nan=False`. Define `payload_sha256` as the
SHA-256 of canonical JSON after removing the `payload_sha256` field itself; use the
same non-self-referential rule for the trackability artifact. Compute
`source_task_config_sha256` from an explicit JSON-compatible projection of the
action, observations, rewards, metrics, events, actuators, timing, and reset fields,
reusing the repository's reward/config digest conventions. Never serialize the full
dataclass with callables or manager objects. The loader checks structure and
scientific identity, but it must not require the artifact-builder revision to equal
a later training revision; those revisions remain separately recorded in
provenance.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_joint_action_qualification.py -q
git add evaluation/joint_position/__init__.py
git add evaluation/joint_position/qualify_joint_action.py
git add src/tasks/hammer/config/z1/joint_position_contract.py
git add tests/test_joint_action_qualification.py
git commit -m "test(hammer): define joint action qualification contract"
```

---

### Task 2: Capture the causal DiffIK tape and qualify direct-joint replay

**Files:**
- Modify: `evaluation/joint_position/qualify_joint_action.py`
- Modify: `tests/test_joint_action_qualification.py`
- Generate: `src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json`

- [ ] **Step 1: Add failing capture/replay tests**

Use a fake action term to prove the capture hook records `robot.data.joint_pos_target` immediately after the first `apply_actions()` call in each ten-substep interval. Prove it does not record realized `joint_pos` or the interval-final DiffIK target.

Freeze `POST_REFERENCE_HOLD_CONTROL_STEPS = 10`, matching the established reference feasibility script. The final target is held for ten additional 20 ms control intervals so contact and nail completion are observed without inventing new commands.

Add an integration test with `cfg.auto_reset=False` proving that the terminal
`q(t+1)`, nail depth, first-strike state, and applied target are captured before any
reset. The offline terminal tracking error must equal the live reward-manager input
on the successful contact step.

- [ ] **Step 2: Implement source capture from the exact Cartesian parent**

Load the play configuration of:

```text
Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4
```

Set `cfg.auto_reset=False`, manually reset once at the start of each seed, and break
only after recording the terminal transition. Never allow `env.step()` to replace
terminal evidence with an auto-reset state.

For every policy interval:

1. Compute the existing Cartesian playback action from `SingleStrikeReference.playback_target()`.
2. Process the 3-D action once.
3. Wrap the DiffIK term's evaluation-only `apply_actions()` call.
4. Immediately after the first substep application, copy the six public `joint_pos_target` values in resolved joint order.
5. Ignore later recomputed DiffIK targets from the same interval.

Never modify production environment stepping and never record realized joint positions as commands.

- [ ] **Step 3: Implement temporary direct-joint replay**

Construct an in-memory copy of the same parent configuration, replace its sole action with:

```python
JointPositionActionCfg(
    entity_name="robot",
    actuator_names=ARM_ACTUATOR_NAMES,
    scale=scale_map,
    clip=physical_clip_map,
    use_default_offset=True,
)
```

Read the six live model joint limits and assert they equal the physical contract
listed above before constructing `physical_clip_map`; never let a duplicated
constant silently override a changed asset. Normalize each recorded target with
`(q_target-q_default)/scale`, call the normal environment `step()` once per target,
and let the action term hold it for ten substeps. Read the public applied target
after clipping; do not reconstruct it for evidence.

Set `auto_reset=False` here as well. Record the source and replay P geometry:
entry point, frozen nail point, and all six gate centers. Require source/replay
equality within `1e-6 m`; independently passing self-defined gate sets is not enough
to prove the same task-space path was retained.

- [ ] **Step 4: Freeze all replay gates**

Apply the following gates separately to the Cartesian source and the direct-joint
replay for every seed `1000` through `1015`:

- target names exactly `joint1` through `joint6`; gripper excluded;
- action, target, state, reward, and metrics all finite;
- identical source target-tape SHA-256 across seeds;
- exactly ten applied substeps per target;
- zero normalized-action saturation;
- zero physical target clipping;
- zero joint-limit violation;
- full-rate peak arm speed `<= 3.1415 rad/s`;
- production-accepted hammer-face contact;
- `FirstStrikeEventTracker` finalizes as productive;
- accepted onset occurs no later than `playback_length + 2` control steps;
- latched productive precontact axial speed is at least `0.5 m/s`;
- finalized first-event nail-axis impulse is strictly positive;
- nail depth reaches `NAIL_SUCCESS_THRESHOLD` within that productive first event and success is true;
- all six ordered P waypoints are crossed before accepted contact;
- post-gate-1 corridor error never exceeds `0.005 m` during scripted qualification;
- no multi-gate-crossing anomaly and no waypoint credit after contact.

The bank passes only if all 16 source rows and all 16 replay rows pass. Success
during a later slow terminal hold cannot rescue a nonproductive first event. Do not
weaken the gate to eventual contact or merely positive nail progress.

- [ ] **Step 5: Run a local preliminary replay**

```bash
mkdir -p evaluation/results/2026-08-06_joint_position_fixed_stage1
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  evaluation/joint_position/qualify_joint_action.py \
  --seeds 1000:1016 \
  --out evaluation/results/2026-08-06_joint_position_fixed_stage1/preliminary_joint_action_contract.json
```

This run is diagnostic because the local checkout contains intentionally untracked presentation results. It must pass before arranging a clean evidence run, but its JSON is not committed.

- [ ] **Step 6: Commit and push the executable qualifier before banking evidence**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_joint_action_qualification.py -q
git add evaluation/joint_position/qualify_joint_action.py
git add tests/test_joint_action_qualification.py
git commit -m "feat(eval): implement causal joint action qualification"
git push -u origin codex/z1-joint-position-fixed-stage1
```

- [ ] **Step 7: Reproduce on a clean revision-pinned checkout and bank the artifact**

Deploy the exact pushed commit and clean pinned asset revision on Vega. Write two
runs outside both repositories so generation does not dirty either checkout:

```bash
STAGE1_GEN_DIR=$(mktemp -d)
PYTHONPATH=. .venv/bin/python evaluation/joint_position/qualify_joint_action.py \
  --seeds 1000:1016 --out "$STAGE1_GEN_DIR/joint_action_run1.json"
PYTHONPATH=. .venv/bin/python evaluation/joint_position/qualify_joint_action.py \
  --seeds 1000:1016 --out "$STAGE1_GEN_DIR/joint_action_run2.json"
cmp "$STAGE1_GEN_DIR/joint_action_run1.json" "$STAGE1_GEN_DIR/joint_action_run2.json"
git status --porcelain=v1 --untracked-files=all
git -C ../safe_impact_manipulation status --porcelain=v1 --untracked-files=all
```

Both status commands must remain empty. Copy `joint_action_run1.json` back as
`src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json` without editing its
bytes.

- [ ] **Step 8: Validate and commit the generated evidence**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_joint_action_qualification.py -q
git add src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json
git commit -m "test(hammer): qualify causal joint-position strike"
```

If G1 fails, stop the entire plan. Do not change gains, thresholds, reset selection, or individual scales in response to outcome data.

---

### Task 3: Add the production fixed-gain joint action and preserve Cartesian behavior

**Files:**
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Create: `tests/test_joint_position_config.py`
- Modify: `tests/test_configs.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write failing additive config tests**

Require the FIC-0 task to:

- be registered in train and play form;
- build from the exact P+V+D4 parent;
- contain one action term named `joint_position`;
- use `JointPositionActionCfg`, six targets, exact artifact scales/clips keyed by
  `joint1` through `joint6`, and default-offset mode; every mapping key must match
  exactly one live target;
- contain no DiffIK action, relative joint action, gain action, or gain-setting callback;
- retain exact rewards, P observations, metrics, events, reset, fixed gains, timing, curriculum, CatPPO, velocity-CaT settings, impulse caps, and `imp_max_p=0.0`;
- retain exactly four base reference observation values and seven P observation
  values, with `r_imit` absent and `r_waypoint_progress=8.0`;
- have action dimension 6 and actor/critic observation dimensions 47;
- leave the Cartesian P+V+D4 task at action dimension 3 and observations 44.

Canonicalize the parent and FIC-0 configs and prove the only dataclass difference is `actions`. The runtime observation-width change follows from the unchanged `last_action` observation term and must not be implemented as a second config change.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_joint_position_config.py tests/test_configs.py tests/test_env.py -q
```

- [ ] **Step 3: Add a focused action installer**

Keep `z1_hammer_env_cfg()` and its DiffIK wiring unchanged. Add a helper in `env_cfgs.py` that accepts an already-built config, loads the qualified artifact, and replaces only `cfg.actions`.

In `__init__.py`, construct FIC-0 by calling the existing `_presentation_i_off_env_cfg(velocity_cat=True, delivered_weight=4.0)`, then call the installer. This avoids changing the large factory signature and prevents any legacy registration from taking a new construction path.

- [ ] **Step 4: Add live runtime tests**

In a separate joint fixture, verify:

- reset and one step produce finite `(N,47)` actor and critic observations;
- `env.action_manager.total_action_dim == 6`;
- live `target_names == ("joint1", ..., "joint6")` and IDs match those names;
- live six-joint encoder bias is zero;
- a zero action maps to the six runtime default positions;
- `+1` and `-1` implement the exact affine map;
- oversized actions in an unwrapped environment demonstrate physical clipping;
- identical actions at two different current states produce identical desired targets;
- `last_action` is cleared at reset and the first processed zero action restores the default target;
- the gripper target is unchanged;
- an incompatible Cartesian checkpoint fails shape loading clearly.

Remember that the unwrapped environment does not apply the RSL-RL raw-action clip. Test wrapper clipping separately from processed physical target clipping.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_joint_position_config.py tests/test_configs.py tests/test_env.py -q
git add src/tasks/hammer/config/z1/env_cfgs.py
git add src/tasks/hammer/config/z1/__init__.py
git add tests/test_joint_position_config.py
git add tests/test_configs.py
git add tests/test_env.py
git commit -m "feat(hammer): add fixed-gain joint-position action"
```

---

### Task 4: Implement and freeze the one-step joint command-trackability term

**Files:**
- Create: `src/tasks/hammer/mdp/trackability.py`
- Modify: `src/tasks/hammer/mdp/__init__.py`
- Modify: `src/tasks/hammer/cat/hook.py`
- Create: `evaluation/joint_position/calibrate_trackability.py`
- Modify: `src/tasks/hammer/config/z1/joint_position_contract.py`
- Generate: `src/tasks/hammer/config/z1/data/z1_joint_trackability_stage1.json`
- Create: `tests/test_joint_trackability.py`
- Modify: `tests/test_cat_soft_hook.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Modify: `tests/test_joint_position_config.py`
- Modify: `tests/test_env.py`

**Interfaces:**

```python
joint_target_squared_error(env, robot_cfg) -> torch.Tensor
joint_target_rmse(env, robot_cfg) -> torch.Tensor
joint_trackability_cost(env, robot_cfg, k_tt) -> torch.Tensor
derive_k_tt(squared_errors, target_q90_cost=0.1) -> float
```

- [ ] **Step 1: Write failing formula and timing tests**

```python
q_des = torch.tensor([[0.2, 0.0, -0.1, 0.3, 0.1, 0.0]])
q_next = torch.tensor([[0.1, 0.0, -0.1, 0.1, 0.0, 0.0]])
expected = ((q_des - q_next) ** 2).sum(dim=1)
torch.testing.assert_close(joint_target_squared_error(env, arm_cfg), expected)
torch.testing.assert_close(joint_trackability_cost(env, arm_cfg, 2.0), 2.0 * expected)
```

Prove the reader uses only the six arm indices, reads the public applied target, observes the post-decimation position, remains active on the accepted-contact step, and reports reward-manager `dt=0.02` scaling separately from the pre-`dt` term.

In `tests/test_cat_soft_hook.py`, construct a reward table with positive task
reward, negative `action_rate`, negative `joint_pos_limits`, and negative-weighted
`r_tt`. With a nonzero velocity-CaT delta, prove `_compute_r_pos()` contains only
positive task return, `r_tt` is in `_NEG_TERMS`, and CatPPO's decomposition leaves
the full `r_tt` penalty unscaled. The existing guard must still reject any unknown
negative-weight reward term.

- [ ] **Step 2: Verify RED, then implement the pure readers**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_joint_trackability.py -q
```

Reject a non-finite/non-positive `k_tt`, target/position shape mismatch, or a
non-joint action task. Extend `_NEG_TERMS` to
`("action_rate", "joint_pos_limits", "r_tt")`; update its completeness comment and
do not change any constraint delta math.

- [ ] **Step 3: Implement calibration from the qualified replay**

With `auto_reset=False`, re-run the one canonical banked joint tape, collect the
squared error after each ten-substep interval including the terminal transition,
and compute:

```python
q90 = np.quantile(all_squared_errors, 0.90)
k_tt = 0.1 / q90
```

Compute q90 from one unique canonical trajectory only. Use the other 15 fixed-reset
runs solely to require identical tape/error hashes and metrics; never pool duplicate
samples or describe them as a calibration population. Write q90, `k_tt`, the
canonical pre-`dt` cumulative cost, corresponding returned dose multiplied by
`0.02`, repeatability hashes, artifact/code/asset revisions, and the source
qualification payload digest. Fail if q90 is below `1e-8`, any value is non-finite,
or the canonical pre-`dt` cumulative cost exceeds `5.0`.

- [ ] **Step 4: Commit and push the calibration implementation before banking evidence**

Use temporary synthetic/replay fixtures in the tests so the implementation can be verified before the production JSON exists.

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_joint_trackability.py -q
git add src/tasks/hammer/mdp/trackability.py
git add src/tasks/hammer/mdp/__init__.py
git add src/tasks/hammer/cat/hook.py
git add evaluation/joint_position/calibrate_trackability.py
git add src/tasks/hammer/config/z1/joint_position_contract.py
git add tests/test_joint_trackability.py
git add tests/test_cat_soft_hook.py
git commit -m "feat(hammer): implement joint command tracking"
git push
```

- [ ] **Step 5: Generate twice on the same clean pinned checkout and bank the exact artifact**

```bash
STAGE1_CAL_DIR=$(mktemp -d)
PYTHONPATH=. .venv/bin/python evaluation/joint_position/calibrate_trackability.py \
  --qualification src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json \
  --out "$STAGE1_CAL_DIR/trackability_run1.json"
PYTHONPATH=. .venv/bin/python evaluation/joint_position/calibrate_trackability.py \
  --qualification src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json \
  --out "$STAGE1_CAL_DIR/trackability_run2.json"
cmp "$STAGE1_CAL_DIR/trackability_run1.json" "$STAGE1_CAL_DIR/trackability_run2.json"
git status --porcelain=v1 --untracked-files=all
git -C ../safe_impact_manipulation status --porcelain=v1 --untracked-files=all
```

Both status commands must remain empty. Copy `trackability_run1.json` back as
`src/tasks/hammer/config/z1/data/z1_joint_trackability_stage1.json` without editing
its bytes.

- [ ] **Step 6: Register the primary `-TT` task and prove isolation**

Use one fresh builder invocation for each train/play registration. Never mutate an
already registered config object:

```python
def _joint_position_fixed_env_cfg(*, play: bool, trackability: bool):
    cfg = _presentation_i_off_env_cfg(
        play=play, velocity_cat=True, delivered_weight=4.0
    )
    install_z1_joint_position_action(cfg)
    if trackability:
        install_joint_trackability_cost(cfg)
    return cfg
```

The primary builder adds exactly one reward term:

```text
name: r_tt
func: joint_trackability_cost
weight: -1.0
k_tt: exact value loaded from z1_joint_trackability_stage1.json
robot joints: joint1 ... joint6
```

An exact-diff test must prove FIC-0 versus FIC-TT
differs only by `rewards.r_tt`. Assert that all four train/play config and reward
objects are independent and do not alias. Both retain P guidance.

- [ ] **Step 7: Run GREEN and commit the artifact and registration**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_joint_trackability.py tests/test_cat_soft_hook.py \
  tests/test_joint_position_config.py tests/test_env.py -q
git add src/tasks/hammer/config/z1/data/z1_joint_trackability_stage1.json
git add src/tasks/hammer/config/z1/__init__.py
git add tests/test_joint_position_config.py
git add tests/test_env.py
git commit -m "feat(hammer): register calibrated joint trackability task"
```

---

### Task 5: Add typed metadata and a dedicated live-manager smoke gate

**Files:**
- Modify: `src/tasks/hammer/rl/runner.py`
- Create: `scripts/smoke_joint_position_fixed.py`
- Create: `tests/test_smoke_joint_position_fixed.py`

- [ ] **Step 1: Write failing metadata dispatch tests**

Preserve the current Cartesian metadata dictionary byte-for-byte. For the joint task, require:

```text
action_type=joint_position_default_offset
action_term=joint_position
action_dim=6
target_names and IDs
actuator_names
scale_rad and physical_clip_rad
use_default_offset=true
effective default offsets from entity.data.default_joint_pos
raw_policy_clip=1.0
physics_dt_s=0.002
control_decimation=10
fixed_gain_signature_sha256
qualification_payload_sha256
trackability_payload_sha256
```

For FIC-0, `trackability_payload_sha256` is the explicit
string sentinel `not_applicable_no_r_tt`; ONNX metadata does not preserve Python
`None` as JSON null. For FIC-TT it is the exact frozen calibration digest. Unknown
action classes, multiple action terms, mismatched target order, or a missing
required artifact digest must raise. Do not report `cfg.offset` as the effective
default offset because mjlab replaces it at runtime.

- [ ] **Step 2: Implement explicit `DifferentialIKAction`/`JointPositionAction` dispatch**

Change the helper to accept the wrapper-owned value explicitly:

```python
_get_hammer_metadata(env, run_path, *, raw_policy_clip: float) -> dict
```

Convert tensor/ID fields to plain serializable lists. Preserve the Cartesian return
dictionary byte-for-byte. In `save()`, compute and validate metadata outside the
broad ONNX-export `try` so action-contract errors propagate; only serialization or
ONNX export failures may retain the existing warning behavior.

- [ ] **Step 3: Write the dedicated smoke with an importable `run_checks()`**

For both joint task IDs on CPU and CUDA, check:

- registration, CatPPO, 6 actions, and 47 observations;
- exact target order, affine map, physical clip, reset semantics, and ten-substep hold;
- fixed gains/efforts and no gain-setting action;
- exact base reward weights, P reward `8.0`, D4 `4.0`, and `action_rate=-0.01`;
- finite nonzero raw cost and negative weighted `r_tt` contribution only in
  FIC-TT, with that penalty excluded from soft-CaT `r_pos`;
- active substep velocity tracker and a synthetic over-limit peak producing positive CaT delta;
- `imp_max_p=0.0` while impulse measurement remains enabled;
- unchanged caps and first-strike-before-waypoint tracker registration order;
- zero clipping on the banked qualification tape, correct clipping under deliberate
  oversized input, and reported (non-gating) clipping on a reproducible random sample;
- metadata construction succeeds and matches live values.

On CUDA, additionally replay the entire banked scripted tape across the smoke envs.
Require finite state, the exact six task-space gates, productive first event,
precontact speed at least `0.5 m/s`, success, full-rate legal qvel, finite
trackability cost, live impulse measurement, and exactly zero impulse contribution
to delta because `imp_max_p=0.0`.

- [ ] **Step 4: Run the CPU rehearsal and commit**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_smoke_joint_position_fixed.py -q
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  scripts/smoke_joint_position_fixed.py \
  --task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed-TT \
  --device cpu --num-envs 8 --steps 3
git add src/tasks/hammer/rl/runner.py
git add scripts/smoke_joint_position_fixed.py
git add tests/test_smoke_joint_position_fixed.py
git commit -m "feat(hammer): validate live joint-position controller"
```

---

### Task 6: Extend evaluation without weakening historical Cartesian contracts

**Files:**
- Modify: `scripts/eval_impulse.py`
- Modify: `tests/test_eval_impulse_hook.py`
- Create: `evaluation/joint_position/evaluate_fixed_rollout.py`
- Create: `tests/test_joint_position_fixed_rollout.py`
- Create: `evaluation/analysis/joint_position_stage1.py`
- Create: `tests/test_joint_position_stage1_analysis.py`

- [ ] **Step 1: Write fail-closed joint campaign mutation tests**

Add dedicated constants and dispatch before the generic Cartesian validator:

```python
JOINT_POSITION_STAGE1_CAMPAIGN = "joint-position-fixed-stage1"
JOINT_POSITION_STAGE1_TASKS = {
    "FIC-0": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered4-JointPosition-Fixed"
    ),
    "FIC-TT": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered4-JointPosition-Fixed-TT"
    )
}
```

Dispatch this task to `_validate_joint_position_stage1_identity(...)` and explicitly
exclude it from the generic fixed-action validator. Do not add it to
`PRESENTATION3_TASK_TO_ARM` or alter the existing Cartesian action signature.

Add `--stage1-phase pilot|final` with exact populations:

```text
pilot: 32 environments x 2 completed episodes = 64
final: 256 environments x 2 completed episodes = 512
```

Parameterize the joint collector, validators, and aggregators with those expected
counts while leaving every historical `256 x 2 = 512` constant and path unchanged.

Define a Stage-1 accepted-manifest TSV loader. The CLI takes
`--accepted-manifest`, hashes the actual bytes, selects one unique row, and verifies
the row's task, seed, `model_199.pt` basename, checkpoint SHA-256, training
code/asset revisions, and both generated-artifact hashes. A digest string without
manifest contents is insufficient.

Reject before rollout, with arm-specific identity checks that require `r_tt` only
for FIC-TT and require it to be absent for FIC-0:

- wrong task ID or checkpoint task;
- action class/term/width/order drift;
- default-offset, scale, physical clip, or artifact-digest drift;
- fixed gain, effort, timing, reset, P reward, D4 dose, velocity-CaT, cap, or `imp_max_p` drift;
- missing/altered `r_tt` or `k_tt` in FIC-TT, or any `r_tt` in FIC-0;
- any unapproved reward override;
- dirty/unknown code or asset provenance;
- a declared trace width that differs from the actual arrays.

Keep every existing Cartesian signature constant and mutation test unchanged. Add
Stage-1-specific `JOINT_STAGE1_FIELDNAMES`, trace persistence, and campaign digest;
do not append joint columns to the global historical CSV header or trace schema.

- [ ] **Step 2: Persist joint/action traces explicitly**

For the new campaign, add six-column arrays for:

```text
policy_action_preclip
term_raw_action_post_wrapper
q_nominal
q_des_applied
q_actual_next
joint_target_error
physical_clip_mask
joint_limit_proximity_mask
```

Mjlab exposes processed actions only through a private field, so do not make that
private tensor the scientific contract. Record the wrapper action, independently
reconstruct `q_nominal`, and read public applied `joint_pos_target`. With the frozen
zero-bias invariant, verify expected equality after physical clipping. Capture
`q_actual_next` through a pre-reset path so terminal samples are never reset state.
Do not pad to three columns or infer action identity from width. Include the
action-interface and artifact hashes in summary provenance.

- [ ] **Step 3: Freeze episode-first metrics**

For each episode:

```python
rmse_t = np.sqrt(np.mean((q_des_applied_t - q_actual_next_t) ** 2, axis=-1))
episode_q90_rmse = np.quantile(rmse_t, 0.90)
```

Aggregate checkpoint values over episodes, never over pooled steps. Add:

- q90 joint-target RMSE and per-joint RMSE;
- raw-action saturation rate: saturated joint-elements with
  `|term_raw_action_post_wrapper| >= 0.999` divided by all valid joint-elements;
- physical-target clipping rate: joint-elements where
  `|q_des_applied-q_nominal| > 1e-7 rad` divided by all valid joint-elements;
- actual-position joint-limit proximity rate: control-rate joint-elements whose
  distance to either physical limit is at most `0.01 * (q_max-q_min)`, divided by
  all valid joint-elements;
- undiscounted episode sums of normalized `sum_j Delta a_j^2`, the same dose per
  DOF, and physical `sum_j Delta q_des,j^2`, plus the weighted action-rate reward;
- existing success, productive first event, precontact speed, nail impulse, reaction impulse/cap, qvel, P-waypoint, path, and contact-quality metrics.

Use unequal-length synthetic episodes to prove episode-first aggregation.

- [ ] **Step 4: Implement a separate fixed CPU companion**

`evaluation/joint_position/evaluate_fixed_rollout.py` accepts one admitted manifest
row and runs exactly one environment on CPU with fixed reset, observation corruption
off, deterministic mean action, `auto_reset=False`, and the same 4.0 s maximum
horizon used by the sampled evaluator. It writes a separate JSON/NPZ artifact with
pre-reset joint/action/contact/P/impulse/qvel traces and all identity hashes. It must
never reuse the sampled evaluator's device or environment count.

Define a genuine fixed-rollout strike numerically as: productive finalized first
event, precontact axial speed `>=0.5 m/s`, positive first-event nail impulse, and
nail success in that event.

- [ ] **Step 5: Preserve sampled CUDA and fixed CPU populations separately**

The summary must not pool them. The existing Cartesian P+V+D4 result is a frozen descriptive reference only; do not reinterpret it as a contemporaneous randomized control.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_eval_impulse_hook.py tests/test_joint_position_fixed_rollout.py \
  tests/test_joint_position_stage1_analysis.py -q
git add scripts/eval_impulse.py
git add tests/test_eval_impulse_hook.py
git add evaluation/joint_position/evaluate_fixed_rollout.py
git add tests/test_joint_position_fixed_rollout.py
git add evaluation/analysis/joint_position_stage1.py
git add tests/test_joint_position_stage1_analysis.py
git commit -m "feat(eval): add strict joint-position stage1 contract"
```

---

### Task 7: Freeze the launcher, preregistration, and complete CPU regression gate

**Files:**
- Create: `scripts/slurm/vega_joint_position_stage1.sbatch`
- Create: `scripts/slurm/vega_joint_position_stage1_eval.sbatch`
- Modify: `tests/test_slurm_launchers.py`
- Create: `docs/results/2026-08-06_joint_position_fixed_stage1_preregistration.md`

- [ ] **Step 1: Write launcher tests before the launcher**

The launcher accepts exactly two modes and expands the Cartesian product of arms
and seeds:

```text
pilot:  array indices 0-1; (FIC-0, seed 2), (FIC-TT, seed 2)
expand: array indices 0-3; both arms at seeds 3 and 4
```

Both training modes freeze the selected task identity, 200 iterations, 4,096 envs, 24 rollout
steps, save interval 50, final `model_199.pt`, A100, clean exact code/asset
revisions, `substep_impulse_rows.enabled=False`, and no reward/gain/reset/threshold
overrides. Reject `IMPACT_W`, `DELIVERED_W`, `NAIL_DRIVEN_W`, `IMP_MAX_P`, or any
alternative task/seed. Also test the evaluation launcher contracts described in
Step 3.

- [ ] **Step 2: Implement the dedicated fail-closed launcher**

Hardcode the two accepted 64-hex artifact payload hashes after generation. Add
mutation tests for missing, malformed, or unequal values. Launcher ordering is:

1. resolve canonical code and asset paths with shell/git;
2. reject dirty or unequal revisions before running repository Python;
3. verify artifact bytes/hashes, mode, task, seed, and array identity;
4. check for an A100;
5. run the CUDA joint smoke;
6. invoke `scripts/train.py` with
   `--env.metrics.substep-impulse-rows.params.enabled False`.

It must write the resolved task, seed, revision, artifact hashes, and exact command
to the job log.

- [ ] **Step 3: Implement the dedicated evaluation launcher**

`vega_joint_position_stage1_eval.sbatch` accepts exactly `pilot` or `final`, an
exact accepted-manifest path, a new external attempt directory, and the frozen
code/asset revisions. It validates and snapshots the manifest before rollout.

- `pilot` selects both seed-2 arm rows and runs each joint evaluator with 32 CUDA
  envs, `--stage1-phase pilot`, and the four RNG seeds frozen above.
- `final` selects both arms at seeds 2, 3, and 4 and runs each with 256 CUDA envs,
  `--stage1-phase final`, and the same RNG contract.
- After each sampled run, it invokes the separate one-env fixed CPU companion.
- Sampled and fixed outputs use separate directories and payload hashes.
- Attempt directories are claimed once and never resumed through a mutable glob.

Tests must reject an unknown phase, wrong row count/seed/task/checkpoint, missing
manifest content, output reuse, RNG override, or dirty/unequal revision.

- [ ] **Step 4: Write the preregistration before launching anything**

Record:

- both task IDs and the paper-aligned causal reason both are trained;
- exact config/action/gain/cap/artifact signatures;
- seed order `2 -> {3,4}`;
- iteration/env/minibatch/PPO settings from the handoff;
- checkpoint rule;
- evaluation RNG contract and episode counts;
- all G4/G5 thresholds below;
- the descriptive-only status of the old Cartesian comparison;
- explicit exclusions: drop calibration, lowered impulse threshold, domain randomization, VIC.

- [ ] **Step 5: Run the focused regression set**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_joint_action_qualification.py \
  tests/test_joint_position_config.py \
  tests/test_joint_trackability.py \
  tests/test_smoke_joint_position_fixed.py \
  tests/test_joint_position_fixed_rollout.py \
  tests/test_joint_position_stage1_analysis.py \
  tests/test_eval_impulse_hook.py \
  tests/test_configs.py \
  tests/test_env.py \
  tests/test_hammer_guideline.py \
  tests/test_strike_reference.py \
  tests/test_imitation_reward.py \
  tests/test_impact_progress_reward.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_cat_soft_hook.py \
  tests/test_slurm_launchers.py -q
```

- [ ] **Step 6: Run the full CPU suite**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest -q
```

- [ ] **Step 7: Run the established scientific gates as Cartesian regressions**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/validate_rewards.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_contact_sensor.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_reward_setup.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/playback_reference.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  evaluation/guideline/qualify_reference.py \
  --out evaluation/results/2026-08-06_joint_position_fixed_stage1/cartesian_reference_regression
git diff --check
```

These scripts retain their Cartesian action contract. The dedicated joint qualification and smoke are the corresponding joint gates.

- [ ] **Step 8: Run independent code/spec review and resolve findings test-first**

Use `superpowers:requesting-code-review` or the repository `code-review` skill against the branch base. Resolve every Critical/Important finding, rerun Steps 5-7, and record the exact commands and outputs in the preregistration.

- [ ] **Step 9: Commit and push the frozen training/evaluation revision**

```bash
git add scripts/slurm/vega_joint_position_stage1.sbatch
git add scripts/slurm/vega_joint_position_stage1_eval.sbatch
git add tests/test_slurm_launchers.py
git add docs/results/2026-08-06_joint_position_fixed_stage1_preregistration.md
git commit -m "docs(hammer): freeze joint-position stage1 pilot"
git status --short --branch
git push -u origin codex/z1-joint-position-fixed-stage1
```

Only the clean pushed revision may train. The local untracked presentation evidence remains untouched.

---

### Task 8: Run the bounded CPU/CUDA smoke and paired one-seed pilot

**Files:**
- Evidence only under `evaluation/results/2026-08-06_joint_position_fixed_stage1/` after returning hash-checked outputs.
- Record job IDs and immutable attempt identities in the external attempt output. Do not edit or commit tracked files between the paired seed-2 pilots and the paired seeds-3/4 expansion; all six policies must train from the same frozen revision.

- [ ] **Step 1: Run one-iteration CPU training smokes for both arms**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/train.py \
  Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed \
  --gpu-ids '[]' \
  --agent.logger tensorboard \
  --agent.run-name stage1_jointpos_fic0_cpu_smoke \
  --agent.seed 2 \
  --agent.max-iterations 1 \
  --agent.save-interval 1 \
  --env.scene.num-envs 8 \
  --env.metrics.substep-impulse-rows.params.enabled False

PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/train.py \
  Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4-JointPosition-Fixed-TT \
  --gpu-ids '[]' \
  --agent.logger tensorboard \
  --agent.run-name stage1_jointpos_fictt_cpu_smoke \
  --agent.seed 2 \
  --agent.max-iterations 1 \
  --agent.save-interval 1 \
  --env.scene.num-envs 8 \
  --env.metrics.substep-impulse-rows.params.enabled False
```

Require finite loss, reward, observation normalization, and six-action policy output
for both arms. Additionally require positive finite raw tracking cost and a negative
finite weighted `r_tt` contribution for FIC-TT, while proving the term is absent in
FIC-0. Do not add CLI reward, threshold, reset, or gain overrides.

- [ ] **Step 2: Run the live CUDA smoke on the deployed clean revision**

Run the command once for each frozen task ID. FIC-0 must prove `r_tt` absent and
FIC-TT must prove the calibrated term active.

- [ ] **Step 3: Launch both seed-2 arms and nothing else**

From the clean deployed checkout:

```bash
STAGE1_CODE_REV=$(git rev-parse HEAD)
STAGE1_ASSET_REV=$(git -C ../safe_impact_manipulation rev-parse HEAD)
MODE=pilot EXPECTED_CODE_REVISION="$STAGE1_CODE_REV" \
  EXPECTED_ASSET_REVISION="$STAGE1_ASSET_REV" \
  sbatch --array=0-1 \
  --export=ALL,MODE,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_joint_position_stage1.sbatch
```

- [ ] **Step 4: Admit only both final checkpoints by exact path and SHA-256**

Require successful process exit, clean revision match, clean asset match, exact task,
seed 2, and `model_199.pt` for both arms. Create a two-row accepted-checkpoint
manifest; never glob `latest` and never substitute an intermediate checkpoint.

- [ ] **Step 5: Evaluate 64 sampled episodes plus one fixed CPU rollout per arm**

Write the two admitted seed-2 rows to
`$HOME/unitree_rl_mjlab_eval/joint-position-fixed-stage1/accepted_training_checkpoints_pilot.tsv`,
then run the dedicated launcher:

```bash
STAGE1_CODE_REV=$(git rev-parse HEAD)
STAGE1_ASSET_REV=$(git -C ../safe_impact_manipulation rev-parse HEAD)
MODE=pilot EVAL_ATTEMPT=pilot_seed2_attempt1 \
  ACCEPTED_MANIFEST="$HOME/unitree_rl_mjlab_eval/joint-position-fixed-stage1/accepted_training_checkpoints_pilot.tsv" \
  EXPECTED_CODE_REVISION="$STAGE1_CODE_REV" \
  EXPECTED_ASSET_REVISION="$STAGE1_ASSET_REV" \
  sbatch --export=ALL,MODE,EVAL_ATTEMPT,ACCEPTED_MANIFEST,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_joint_position_stage1_eval.sbatch
```

This yields, separately for each arm, exactly 32 CUDA environments x two
completions = 64 stochastic sampled episodes plus a one-env deterministic fixed
CPU rollout. Never pool the arms.

- [ ] **Step 6: Apply the G4 promotion gate**

Each seed-2 arm must independently meet every item:

- task success at least `58/64` sampled episodes;
- productive first-event count at least `58/64`;
- `successful_first_strike_v_precontact_mean_sampled >= 0.5 m/s`;
- `first_strike_delivered_success_mean_sampled >= 0.270938 N s`. This is 80% of
  the frozen Cartesian endpoint `0.338672 N s`, which is the median across six
  checkpoint-level sampled first-event means; it is not an episode median;
- all-six-waypoints-by-contact count at least `52/64` and nonzero P return;
- `64/64` observed episodes within the `3.1415 rad/s` velocity limit;
- zero non-finite qvel/state/action/reward values;
- zero impossible-success and dead-lambda events;
- zero physical target clipping by the joint-element definition above and raw-action
  joint-element saturation `< 5%`;
- actual-position joint-limit proximity `< 1%` by the one-percent-of-range
  joint-element definition above;
- finite joint-target RMSE in both arms; in FIC-TT, finite positive raw tracking
  cost and finite negative weighted `r_tt` return; in FIC-0, verified absence of
  `r_tt`;
- fixed CPU rollout shows an accepted hammer-face impact and successful nail drive.

These are promotion criteria, not claims of hardware certification. No directional
FIC-TT advantage is required. If either arm fails, do not submit seeds 3/4 for
either arm. First classify the failure as action range, trackability dose,
exploration/learning, trajectory guidance, velocity legality, or evaluator/provenance.
Do not simultaneously tune scales, rewards, gains, and resets.

---

### Task 9: Expand to three seeds, close Stage 1, and hand off to the drop experiment

**Files:**
- Create/populate: `docs/results/2026-08-06_joint_position_fixed_stage1_result.md`
- Add accepted evidence under `evaluation/results/2026-08-06_joint_position_fixed_stage1/` only after hash validation.
- Modify: `docs/results/2026-08-06_PRESENTATION_HANDOFF.md` with the completed Stage-1 outcome and next gate.

- [ ] **Step 1: Launch both arms at seeds 3 and 4 only after both pass G4**

```bash
STAGE1_CODE_REV=$(git rev-parse HEAD)
STAGE1_ASSET_REV=$(git -C ../safe_impact_manipulation rev-parse HEAD)
MODE=expand EXPECTED_CODE_REVISION="$STAGE1_CODE_REV" \
  EXPECTED_ASSET_REVISION="$STAGE1_ASSET_REV" \
  sbatch --array=0-3 \
  --export=ALL,MODE,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_joint_position_stage1.sbatch
```

- [ ] **Step 2: Admit exact `model_199.pt` checkpoints and evaluate**

Build the six-row manifest
`$HOME/unitree_rl_mjlab_eval/joint-position-fixed-stage1/accepted_training_checkpoints_final.tsv`
for both arms at seeds 2, 3, and 4, then launch:

```bash
STAGE1_CODE_REV=$(git rev-parse HEAD)
STAGE1_ASSET_REV=$(git -C ../safe_impact_manipulation rev-parse HEAD)
MODE=final EVAL_ATTEMPT=final_three_seed_attempt1 \
  ACCEPTED_MANIFEST="$HOME/unitree_rl_mjlab_eval/joint-position-fixed-stage1/accepted_training_checkpoints_final.tsv" \
  EXPECTED_CODE_REVISION="$STAGE1_CODE_REV" \
  EXPECTED_ASSET_REVISION="$STAGE1_ASSET_REV" \
  sbatch --export=ALL,MODE,EVAL_ATTEMPT,ACCEPTED_MANIFEST,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_joint_position_stage1_eval.sbatch
```

Evaluate exactly 512 stochastic sampled CUDA episodes per checkpoint plus one
separate deterministic fixed CPU rollout per checkpoint. Persist six-joint
preclip/wrapper/nominal/applied/actual traces and their digests.

- [ ] **Step 3: Apply the G5 stability gate per checkpoint**

Each of the six checkpoints must meet:

- sampled task success at least `461/512`;
- productive first-event count at least `461/512`;
- `successful_first_strike_v_precontact_mean_sampled >=0.5 m/s`;
- `first_strike_delivered_success_mean_sampled >=0.270938 N s`;
- all-six-waypoints-by-contact count at least `410/512`;
- `512/512` observed episodes within the qvel rail and no non-finite value,
  impossible success, or dead lambda;
- zero physical target clipping, raw-action joint-element saturation `<5%`, and
  actual-position joint-limit joint-element proximity `<1%`;
- genuine impact on the fixed rollout rather than a slow press;
- finite joint-target RMSE; additionally, positive finite raw tracking cost and
  negative finite weighted `r_tt` return for FIC-TT, and verified absence of the
  term for FIC-0.

Report the paired FIC-TT minus FIC-0 effect for command tracking, success,
productive impact, nail impulse, per-joint reaction impulse, action saturation,
joint-limit proximity, and velocity legality, without imposing an unregistered
directional margin. Do not require reaction-impulse reduction at this integration
stage and do not claim the joint interface is superior to Cartesian. The scientific
replicate is the independently trained seed (`n=3` paired seeds per arm), not the
within-checkpoint episodes.

- [ ] **Step 4: Write the dated result note before changing any later-stage parameter**

Include:

- exact code/asset/artifact/checkpoint/trace hashes;
- every gate result, including failures and exceptions;
- task-space path plots and desired-versus-actual joint plots;
- action saturation, physical clip, joint-limit, qvel, nail impulse, and reaction-impulse tables;
- sampled CUDA and fixed CPU results kept separate;
- comparison to the old Cartesian P+V+D4 result explicitly labeled descriptive;
- a clear verdict: `STAGE1_PASS` or `STAGE1_NOT_READY`.

- [ ] **Step 5: Verify, review, and commit the result**

Use `superpowers:verification-before-completion`, rerun the validators against the admitted manifest and artifacts, then request independent scientific-isolation review.

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_joint_position_stage1_analysis.py \
  tests/test_eval_impulse_hook.py \
  tests/test_docs_current.py -q
git diff --check
git add docs/results/2026-08-06_joint_position_fixed_stage1_result.md
git add docs/results/2026-08-06_PRESENTATION_HANDOFF.md
git commit -m "docs(results): report joint-position fixed baseline"
```

Stage 1 is complete only with `STAGE1_PASS`. The next plan then implements the simulation-only 0.200 kg cylindrical controlled drop at `h0`. It must not begin VIC, lower impulse thresholds, or add curriculum/domain randomization in this branch.

## Later-Stage Handoff Boundaries

This plan deliberately stops after the first two canonical roadmap bullets: joint-space policy actions and retained task-space tracking.

1. **Next:** controlled-drop `I_ref` calibration, simulation only.
2. **Then:** diagnostic lowered-threshold impulse-CaT off/on comparison under fixed impedance.
3. **Then:** curriculum and domain randomization after the fixed baseline and active-CaT diagnostic are stable.
4. **Last:** branch from the verified joint baseline for VIC, adding per-joint stiffness action, bidirectional gain bounds, coupled damping, gain reset, telemetry, and a matched fixed-versus-VIC campaign.

No later-stage mechanism is authorized by executing this Stage-1 plan.
