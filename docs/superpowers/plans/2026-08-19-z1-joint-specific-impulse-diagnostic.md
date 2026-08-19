# Z1 two-boundary impulse-CaT diagnostic implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate exactly two seed-2 VIC policies at `imp_max_p=0.2`, holding velocity
CaT and impulse-maximization rewards fixed while changing only the six-joint diagnostic cap vector.

**Architecture:** Reuse the qualified two-arm training/evaluation launcher patterns, native
`CatSoftHook`, existing telemetry, frozen survey, paired RNG streams, and analysis helpers. Add
campaign-specific guarded launchers and one thin result analyzer; do not create a new algorithm,
environment, reward, evaluator framework, or telemetry layer.

**Tech Stack:** Python 3.10, NumPy, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, pytest, Bash/Slurm, Vega
NVIDIA A100-SXM4-40GB.

**Spec:** `docs/superpowers/specs/2026-08-19-z1-joint-specific-impulse-diagnostic-design.md`

## Global constraints

- Work only in `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/z1-step1-evidence-cleanup`; preserve
  the dirty main worktree.
- Train exactly two arms: `p02_uniform09` at
  `[0.738,1.476,0.738,0.738,0.738,0.738] N.m.s` and `p02_joint_stress` at
  `[0.369,0.246,0.738,0.369,0.246,0.0164] N.m.s`.
- Both arms use `imp_max_p=0.2`, seed `2`, `4096 x 24`, `500` iterations, save interval `50`, the
  native VIC-TT task, active velocity CaT, delivered-impulse reward weight `4.0`, and
  impact-progress weight `8.0`.
- Change no other reward, observation, action, physics, reference, VIC range, training distribution,
  curriculum, domain randomization, FIC/VIC treatment, or trajectory generation.
- Historical launchers, checkpoints, roles, raw traces, analyzers, and result bodies remain
  immutable.
- Every GPU job is one-shot and `--no-requeue`; never retry, extend, resume, or automatically start
  another treatment.
- All thresholds are empirical diagnostic boundaries, never manufacturer limits or hardware caps.

---

### Task 1: Add the guarded two-arm training launcher

**Files:**
- Create: `scripts/slurm/vega_vic_impulse_two_boundary_p02_train.sbatch`
- Create: `tests/test_vic_impulse_two_boundary_p02_train_launcher.py`

**Interfaces:**
- Consumes the runtime, provenance, collision, checkpoint, finiteness, and guarded-hash patterns in
  `scripts/slurm/vega_vic_impulse_dose_curve_train.sbatch`.
- Produces two immutable `model_499.pt` checkpoints and their SHA-256 values.

- [ ] Write RED shell-contract tests that fail because the launcher is absent. Exercise both array
  arms and require exact role-to-cap mapping, common `p=0.2`, native VIC-TT task, seed, rollout,
  iteration budget, reward identities, active velocity CaT, resources, `--no-requeue`, fresh leaves,
  clean provenance, exact 11-checkpoint validation, final finiteness, and guarded SHA output.
- [ ] Add negative RED cases for swapped vectors, changed dose, disabled velocity CaT, changed task,
  unexpected arguments/array state, dirty code/assets, collisions, wrong runtime/GPU, wrong
  iteration metadata, nonfinite tensors, hash command failure, empty hash, and invalid hash.
- [ ] Run
  `/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q tests/test_vic_impulse_two_boundary_p02_train_launcher.py`
  and confirm the expected missing-launcher failures.
- [ ] Implement the smallest `0-1` array launcher by adapting the qualified dose-curve launcher.
  Freeze both role/vector pairs literally and use CLI overrides only for `imp_limit` and the common
  `imp_max_p=0.2`. Add a fail-closed Python preflight that loads the registered task configuration
  and asserts velocity CaT active, delivered-impulse weight `4.0`, impact-progress weight `8.0`, and
  all non-treatment configuration identical.
- [ ] Run GREEN plus historical p=.2/dose-curve launcher tests, `bash -n`, and `git diff --check`.
- [ ] Commit and obtain independent Standards and Spec review before Task 2.

### Task 2: Pass gates and train both arms exactly once

**Files:** No production changes. Record evidence in this plan's SDD task report.

**Interfaces:**
- Consumes the reviewed Task-1 launcher and exact clean revision.
- Produces two terminal Vega records and exact final checkpoint hashes.

- [ ] Run exactly once at the launch revision: the AGENTS.md named reward/impulse tests, all relevant
  launcher tests, the real VIC one-update smoke, `validate_rewards.py` A-M,
  `verify_contact_sensor.py`, `verify_reward_setup.py`, `bash -n`, and `git diff --check`. Stop on
  any failed or unknown result.
- [ ] Push the exact clean revision once, create a fresh detached Vega worktree, verify canonical
  clean assets and exact runtime, and pre-create the exact Slurm log parent.
- [ ] Run one `sbatch --test-only`; if accepted, submit the identical two-arm array exactly once.
- [ ] Monitor without intervention. Require both arms `COMPLETED`, `0:0`, `Restarts=0`, empty
  stderr, PASS, exact checkpoints `model_{0,50,100,150,200,250,300,350,400,450,499}.pt`, final
  `iter == 499`, recursive tensor finiteness, stable code/assets/config provenance, and valid
  checkpoint SHA-256.
- [ ] Record runtime and A100 GPU-hours without inventing monetary cost. Stop before evaluation and
  obtain independent execution review.

### Task 3: Bind both hashes and add the frozen two-policy evaluation

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Modify: `tests/test_impulse_cat_activation_survey.py`
- Create: `scripts/slurm/vega_vic_impulse_two_boundary_p02_eval.sbatch`
- Create: `tests/test_vic_impulse_two_boundary_p02_eval_launcher.py`

**Interfaces:**
- Consumes the exact Task-2 uniform-0.9 and joint-stress hashes.
- Produces two role-bound five-artifact evaluation leaves with live `imp_max_p=0`.

- [ ] Write RED tests for immutable checkpoint roles/hashes and a two-arm launcher whose only
  role-specific fields are checkpoint, role, and training-cap identity. Require fixed 64 plus
  stochastic seeds `2/2026081701/2026081702`, identical population/RNG identities, finite telemetry,
  zero live impulse delta, and exact max soft-OR.
- [ ] Bind both roles and implement the guarded evaluation array by adapting the qualified dose
  evaluator. Every arm emits fixed trace, three stochastic traces, `summary.json`, and a
  deterministic guarded five-row manifest.
- [ ] Run survey/comparison/old-and-new launcher tests, `bash -n`, `git diff --check`, and one local
  real-checkpoint 2-env x 8-step smoke proving shapes, finiteness, role hashes, log-only invariants,
  and matched population/RNG identity. Commit and obtain independent review.
- [ ] Push the reviewed revision, create a fresh detached Vega evaluator worktree and log parent,
  run one dry-run, and submit the identical two-arm evaluation once. Never retry a failed arm.
- [ ] Require both arms `COMPLETED`, `0:0`, zero restarts, empty stderr, PASS, exact artifacts,
  valid manifests, finiteness, and exact checkpoint/code/asset/population/RNG provenance.

### Task 4: Analyze, bank, and stop

**Files:**
- Create: `scripts/analyze_vic_impulse_two_boundary_p02.py`
- Create: `tests/test_vic_impulse_two_boundary_p02_analysis.py`
- Create: `docs/results/2026-08-19_z1_impulse_two_boundary_p02.md`
- Create: `docs/results/assets/2026-08-19_z1_impulse_two_boundary_p02/analysis.json`
- Create: `docs/results/assets/2026-08-19_z1_impulse_two_boundary_p02/SHA256SUMS`
- Modify: `docs/results/README.md`
- Modify: `docs/README.md`
- Modify: `docs/thesis/README.md` only for bounded thesis-ready claims.
- Modify: `tests/test_docs_current.py`

**Interfaces:**
- Consumes the two exact Task-3 evaluation leaves.
- Produces one finite deterministic descriptive classification artifact.

- [ ] Write RED tests requiring exact manifests, roles, checkpoint/code/asset identity, live p=0,
  matched population/RNG identity, separate population results, cross-analysis under both training
  vectors and provisional project caps, raw-Lambda comparisons, every required endpoint from the
  spec, and bounded descriptive classifications. Fixed-64 remains descriptive and controller reads
  are never inferential units.
- [ ] Implement the thin analyzer by composing existing survey and bridge/dose analysis helpers;
  never copy telemetry, event grouping, native-dtype margin, or resampling implementations.
- [ ] Run new/historical analyzer and docs suites, deterministic byte regeneration, pycompile,
  manifest checks, `git diff --check`, and independent review.
- [ ] Bank exact hashes, outcomes, censoring/read-duration limitations, starting-condition replay,
  and the one-seed/non-hardware/non-clamp claim boundary.
- [ ] Commit, push, report, and stop. Do not automatically launch curriculum, domain randomization,
  another cap vector, dose, seed, torque constraint, contact flush, or hardware treatment.
