# Wave 1 Waypoint Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a minimal matched six-policy C0/G/P screen that determines whether sparse ordered-gate payout or dense non-farmable waypoint progress improves a direct fixed-reset hammer descent.

**Architecture:** Extend the existing 500 Hz `WaypointProgressTracker`; do not create a second tracker or evaluator framework. The tracker computes identical gate and progress state in all three arms, while reward-manager registration is the only treatment difference. Reuse the existing qualification, CUDA smoke, Slurm launcher, and fixed-reset video library with only the fields required by the approved spec.

**Tech Stack:** Python 3.11, PyTorch, mjlab manager terms, pytest, MuJoCo/MuJoCo Warp, RSL-RL PPO, Vega Slurm.

## Global Constraints

- Source of truth: `docs/superpowers/specs/2026-08-02-wave1-waypoint-guidance-design.md`.
- Exact arms: C0, G, P; exact seeds: 2 and 3; 4096 environments; 200 iterations; one GPU per run.
- Six gates at `1/7..6/7`; swept 500 Hz detector; 15 mm disk radius; fixed physical reset.
- P uses center-distance new-best progress plus crossing remainder settlement; every completed target totals exactly `1/6`, every episode at most raw `1.0`.
- C0/G/P receive identical observations and compute identical tracker state; only reward registration differs.
- Fixed impedance only; no `set_gains`; manufacturer `IMP_J_LIMIT` unchanged; `imp_max_p=0.0`; velocity enforcement disabled.
- Do not modify installed packages, create a new manifest system, or broaden the legacy evaluator.
- TDD is mandatory: each production behavior must first fail a real focused test for the intended reason.
- Commit only named files; never `git add -A`; never add a co-author trailer.

---

## File map

- `src/tasks/hammer/mdp/guideline.py`: single owner of ordered gates and dense-progress state/readers.
- `src/tasks/hammer/mdp/__init__.py`: export the new observation and reward readers.
- `src/tasks/hammer/config/z1/env_cfgs.py`: register shared observations and optional P payout.
- `src/tasks/hammer/config/z1/__init__.py`: register the P training/play task.
- `tests/test_hammer_guideline.py`: pure geometry, lifecycle, reset, accumulation, and anti-farming tests.
- `tests/test_configs.py`: exact three-arm treatment isolation.
- `scripts/smoke_first_strike_instrumentation.py` and `tests/test_smoke_first_strike_instrumentation.py`: C0/P CUDA identities and live payout/tracker checks.
- `evaluation/guideline/qualify_reference.py` and `tests/test_guideline_qualification.py`: reference feasibility and nondegenerate progress-state qualification.
- `scripts/render_policy.py`, `evaluation/analysis/fixed_reset_video_library.py`, and `tests/test_fixed_reset_video_library.py`: 500 Hz trace plus fixed-stride slow-motion evidence.

---

### Task 1: Dense progress state and payout

**Files:**
- Modify: `src/tasks/hammer/mdp/guideline.py`
- Modify: `src/tasks/hammer/mdp/__init__.py`
- Modify: `tests/test_hammer_guideline.py`

**Interfaces:**
- Produces: `ordered_waypoint_progress_reward(env) -> Tensor[num_envs]`.
- Produces: `waypoint_progress_state(env) -> Tensor[num_envs, 2]`, columns `[d_start/reference_length, f_best]`.
- Extends: `WaypointProgressTracker` with `target_start_distance`, `best_target_fraction`, `window_new_credit`, `episode_credit`, and `multi_gate_crossings` tensors.

- [ ] **Step 1: Write failing lifecycle and scaling tests**

Add literal, hand-derived tests proving: initialization pays zero; a center-distance reduction from `21 mm` to `10.5 mm` earns exactly half of one target budget; hover/backtrack/revisit earns zero; a valid crossing settles only the remainder; six completed gates total raw `1.0`; two gates crossed in one substep total `2/6`; contact censors same-substep credit; partial reset changes only selected rows; all state remains finite and on-device.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_hammer_guideline.py -q
```

Expected: only tests requiring the new state/readers fail because those symbols or fields do not exist.

- [ ] **Step 3: Implement the minimal tracker extension**

At each 500 Hz call: clear only the control-window accumulator at the decimation boundary; initialize entry/nail/first-target distance without payout; disarm before credit on first accepted contact; compute new-best center-distance credit for the target active at substep start; call the existing swept detector; on `k>=1`, settle the active remainder and add `(k-1)/6`; advance and initialize the next target; cap episode credit at one. Reader functions must be idempotent and never clear or advance state.

- [ ] **Step 4: Run RED→GREEN and mutation checks**

Run the focused test command again. Then mentally/locally mutate the denominator, settlement, window clearing, and contact ordering; each must be caught by a named test.

- [ ] **Step 5: Commit only Task-1 files**

```bash
git add src/tasks/hammer/mdp/guideline.py src/tasks/hammer/mdp/__init__.py tests/test_hammer_guideline.py
git commit -m "feat(guideline): add non-farmable waypoint progress"
```

---

### Task 2: Register the matched C0/G/P arms

**Files:**
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Modify: `tests/test_configs.py`

**Interfaces:**
- Adds factory flag: `progress_reward: bool = False`; it requires `guideline=True`.
- Adds task: `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress`.
- Registers `r_waypoint_progress` only in P at weight `8.0`.

- [ ] **Step 1: Write failing arm-isolation tests**

Assert all three train/play configs share fixed reset, action/gains, base rewards, observations including two-column progress state, metrics, `imp_max_p=0.0`, and impulse caps. Normalize config dictionaries and assert C0↔G differs only by `r_gate`, C0↔P only by `r_waypoint_progress`, and G↔P only by exchanging those two readers. Assert `r_imit`, velocity termination/CaT, and variable gains are absent.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_configs.py::TestCartesianGuidelineStudy -q
```

- [ ] **Step 3: Add the minimal factory flag, shared observation, and P registration**

Do not add GP, reference tracking, hard guidance, new PPO settings, or CLI reward overrides.

- [ ] **Step 4: Run focused plus tracker tests GREEN**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_configs.py::TestCartesianGuidelineStudy tests/test_hammer_guideline.py -q
```

- [ ] **Step 5: Commit only Task-2 files**

```bash
git add src/tasks/hammer/config/z1/env_cfgs.py src/tasks/hammer/config/z1/__init__.py tests/test_configs.py
git commit -m "feat(guideline): register progress-only treatment"
```

---

### Task 3: Qualify CPU and CUDA behavior

**Files:**
- Modify: `evaluation/guideline/qualify_reference.py`
- Modify: `tests/test_guideline_qualification.py`
- Modify: `scripts/smoke_first_strike_instrumentation.py`
- Modify: `tests/test_smoke_first_strike_instrumentation.py`

**Interfaces:**
- Qualification requires finite positive `d_start`, zero scripted multi-gate substeps, six gates, success, qvel legality, and raw gate/progress total `1.0` where enabled.
- Smoke arm map includes C0 and P; C0 requires finite live tracker state with no payout reader, P requires finite positive progress payout and exact task identity.

- [ ] **Step 1: Write failing pure qualification and smoke-contract tests**

Use synthetic rows/records to prove each new sentinel fails closed independently; do not test by grepping source.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_guideline_qualification.py tests/test_smoke_first_strike_instrumentation.py -q
```

- [ ] **Step 3: Implement only the new fields and predicates**

Preserve all legacy arm contracts byte-for-byte where possible; no evaluator schema expansion is part of this task.

- [ ] **Step 4: Run the complete local pre-training gate**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_hammer_guideline.py tests/test_configs.py tests/test_guideline_qualification.py tests/test_smoke_first_strike_instrumentation.py tests/test_impact_progress_reward.py tests/test_impulse_bound.py tests/test_impulse_constraint.py tests/test_delivered_impulse_reward.py tests/test_cat_soft_hook.py -q
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python validate_rewards.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python verify_contact_sensor.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python verify_reward_setup.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python evaluation/guideline/qualify_reference.py --out /tmp/wave1_reference_qualification
```

- [ ] **Step 5: Commit, push, deploy by git, then run C0 and P CUDA smokes**

Stage only the four Task-3 files. On Vega use a clean checkout at the exact pushed commit and clean pinned asset revision; never `scp` tracked source. Require no failed predicate, `impossible_success_n=0`, `lambda_dead_n=0`, finite qvel, live Lambda, finite C0 tracker, and positive P dose.

---

### Task 4: Launch the six-policy Wave 1

**Files:**
- No production code changes.
- Create after submission: `docs/results/2026-08-02_wave1_waypoint_campaign.md`.

- [ ] **Step 1: Freeze launch identity**

Record code revision, asset revision, exact task IDs, seeds `2 3`, `ITERS=200`, `num_envs=4096`, and the three treatment config digests. Reject dirty or `-dirty` provenance and inherited `IMPACT_W`, `DELIVERED_W`, or `NAIL_DRIVEN_W`.

- [ ] **Step 2: Submit three two-seed arrays through the existing launcher**

Use `scripts/slurm/vega_train.sbatch` in `SINGLE_TASK` mode, one GPU per array element, with distinct shorts `wave1_c0`, `wave1_g`, and `wave1_p`. Do not modify the launcher merely to rename the campaign.

- [ ] **Step 3: Validate startup before leaving jobs unattended**

For all six jobs verify exact task, seed, clean code/asset hashes, 4096 envs, 200 iterations, no inherited overrides, finite first iteration, and GPU identity. Cancel only a misconfigured job; preserve logs as evidence.

- [ ] **Step 4: Bank the campaign ledger**

Record job IDs, output paths, expected checkpoint names, and the rule that all six final checkpoints—including zero-dose or weak policies—must enter analysis.

---

### Task 5: Produce fixed-reset 500 Hz evidence and stop for analysis

**Files:**
- Modify: `scripts/render_policy.py`
- Modify: `evaluation/analysis/fixed_reset_video_library.py`
- Modify: `tests/test_fixed_reset_video_library.py`
- Create: `docs/results/2026-08-02_wave1_waypoint_result.md`

**Interfaces:**
- Trace uses frozen tracker `entry -> nail`, 500 Hz site positions, control-boundary markers, contact, gate index, dense progress state, and actual manager payouts.
- MP4 samples every two physics substeps and encodes at 50 fps with identical camera/duration/reset for all six policies.

- [ ] **Step 1: Write failing artifact/plot tests**

Prove the trace contains 500 Hz samples rather than duplicated 50 Hz positions, uses the tracker geometry rather than `SingleStrikeReference`, pins stride `2`, draws dashed reference and gate disks, and binds code/asset revisions. A renderer failure must not corrupt a completed leaf.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_fixed_reset_video_library.py -q
```

- [ ] **Step 3: Implement the minimum capture/plot extension**

Reuse existing substep metric/callback machinery and the existing video-library validator. Do not add computer vision, an interactive dashboard, or a second artifact format.

- [ ] **Step 4: Render and validate all six final checkpoints**

Produce one MP4, montage, x-z/x-y plot, trace, metadata, and metrics row per policy. Validate fixed reset and identical reference geometry across all leaves.

- [ ] **Step 5: Apply the frozen exploratory decision rule and stop**

Compare both seeds per arm against fresh C0 using success, treatment dose, speed, q90 error, 5 mm occupancy, path-length ratio, backward travel, contact timing, delivered impulse, Lambda/cap, qvel, and human video inspection. Do not launch GP, constraints, or 500-iteration continuation until the owner reviews this result.

