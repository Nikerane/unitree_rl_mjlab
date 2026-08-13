# Z1 VIC-TT Seed-2 Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimum telemetry and fail-closed Vega route needed to run
and assess one full-budget VIC-TT seed-2 engineering canary.

**Architecture:** Attach compact gain-command summaries to VIC checkpoints
from the rollout storage already populated by CatPPO. Launch one immutable
training job through a dedicated guarded Slurm script, then reuse the existing
64-world evaluator through a hashed scratch wrapper rather than creating a new
evaluation package.

**Tech Stack:** Python 3.12, PyTorch/RSL-RL, mjlab 1.4.0, MuJoCo and
mujoco-warp 3.8.1, pytest, Bash/Slurm, TensorBoard, Vega A100.

## Global constraints

- Work only in the existing `overnight-impulse-minimal` worktree and branch
  `z1-vic-prototype`.
- Preserve every existing task and all six protected user-owned untracked
  paths.
- Freeze VIC-TT at `C=1.25`, seed 2, 4096 environments, 500 iterations,
  checkpoint 499, save interval 50, and the exact existing scientific config.
- Add no evaluator package, VIC-0, active impulse pressure, sweep, reward,
  observation, filter, DR, curriculum, or controlled-drop change.
- Test public seams first: checkpoint contents and launcher behavior.
- Stop after seed 2. Do not retry or expand automatically.

---

### Task 0: Bank the approved canary design and plan

**Files:**
- Create: `docs/superpowers/specs/2026-08-13-z1-vic-canary-design.md`
- Create: `docs/superpowers/plans/2026-08-13-z1-vic-canary.md`

- [ ] Verify both records contain no placeholders or conflicting scope.
- [ ] Run `git diff --check` on the two files.
- [ ] Commit only the two records as `docs(hammer): design VIC seed-2 canary`.

### Task 1: Attach gain-use telemetry to real VIC checkpoints

**Files:**
- Modify: `src/tasks/hammer/rl/runner.py`
- Create: `tests/test_vic_rollout_telemetry.py`
- Modify: `scripts/smoke_cat_soft.py`

**Interfaces:**
- Consumes: `CatRolloutStorage.actions`,
  `CatRolloutStorage.distribution_params`, the exact ordered VIC action pair,
  and the qualified raw action clip `1.0`.
- Produces: checkpoint key
  `infos["vic_rollout_telemetry"]`, schema version 1.

- [ ] Write a failing pure test with literal 2-step, 2-world, 12D tensors.
  Require independently worked per-joint statistics, clip occupancy, indices
  `6..11`, finite validation, and JSON-safe output.
- [ ] Run the focused test and confirm RED because the telemetry seam is absent.
- [ ] Implement the smallest pure summarizer in `runner.py`.
- [ ] Run the focused test and confirm GREEN.
- [ ] Write failing tests that FIC/manual-empty saves remain unchanged, malformed
  VIC storage fails closed, and a genuine one-iteration VIC CatPPO checkpoint
  contains the telemetry record.
- [ ] Extend `smoke_cat_soft.py` to assert the exact VIC checkpoint telemetry
  only for the VIC task; keep all FIC behavior unchanged.
- [ ] Run the new tests plus the existing smoke module; confirm GREEN.
- [ ] Commit the three paths as `feat(hammer): record VIC rollout gain telemetry`.

### Task 2: Add one fail-closed VIC canary launcher

**Files:**
- Create: `scripts/slurm/vega_vic_canary.sbatch`
- Create: `tests/test_vic_canary_launcher.py`

**Interfaces:**
- Consumes: exact `EXPECTED_CODE_REVISION`, `EXPECTED_ASSET_REVISION`,
  `RUN_ROOT`, and `ASSET_REPO` environment variables.
- Produces: one collision-safe run leaf, the exact checkpoint set, canonical
  `victt_seed2_telemetry.json`, and printed artifact hashes.

- [ ] Write fake-shell RED tests covering the exact training command, CUDA
  smoke commands, task/seed/budget, consolidated artifact path, and successful
  postflight.
- [ ] Add RED mutations for arguments, array context, optimized Python, dirty or
  wrong repositories, noncanonical assets, reused leaves, wrong GPU/runtime,
  missing/extra/nonfinite checkpoints, bad checkpoint iteration/telemetry,
  missing/nonfinite TensorBoard streams, incomplete curriculum, nonzero
  `impossible_success`, and gain telemetry pinned at a bound.
- [ ] Implement the minimal no-array launcher using explicit runtime guards
  rather than Python `assert`.
- [ ] Run launcher tests, `bash -n`, and `git diff --check`; confirm GREEN.
- [ ] Commit the two paths as `feat(hammer): add guarded VIC seed-2 launcher`.

### Task 3: Review and qualify the final CPU candidate

**Files:** no new production files.

- [ ] Run focused VIC/config/RTT/CaT/launcher tests and both CPU live smokes.
- [ ] Run reward phases A-M, contact sensor, reward setup, and reference playback.
- [ ] Run repository Standards and Spec review plus independent runtime/science
  review; fix concrete Critical/Important findings test-first.
- [ ] Run the full CPU suite exactly once on the reviewed final candidate.
- [ ] Require clean tracked status and `git diff --check`.
- [ ] Push the reviewed candidate SHA to `origin/z1-vic-prototype`.

### Task 4: Launch and monitor exactly one Vega canary

**Files:** no additional repository files.

- [ ] Fetch the pushed SHA on Vega and create a fresh clean detached worktree
  beside the canonical clean asset repository.
- [ ] Create the existing campaign `slurm/` directory if absent and submit the
  exact new launcher with the four required environment variables.
- [ ] Record the exact `sbatch` command, job ID, node, GPU, revisions, and start
  time. Never auto-retry a failed job.
- [ ] Monitor Slurm state, stdout/stderr growth, checkpoint progress, and hard
  90-minute timeout. Treat stalls as advisory and preserve failed artifacts.
- [ ] On completion, independently recompute checkpoint, telemetry, config, and
  log hashes and verify the clean revisions again.

### Task 5: Run the frozen 64-world behavior check and report

**Files:**
- Create only under `/private/tmp`: `z1_vic_canary_eval.py`
- Copy the exact bytes into the canary attempt directory on Vega.

- [ ] Write the scratch wrapper around
  `evaluation.joint_position.evaluate_fic_pilot.evaluate_checkpoint`; admit only
  the VIC-TT task and exact candidate/checkpoint identities.
- [ ] Hash the script before transfer and confirm the remote hash.
- [ ] Run it once on `model_499.pt` with CUDA, evaluation seed `2026081202`,
  64 worlds, deterministic mean policy, no auto-reset, and four-second horizon.
- [ ] Validate the compact JSON independently: 64 ordered finite rows,
  recomputed summaries, at least 58 success/productive, 64/64 qvel legal, and
  64/64 below impulse caps.
- [ ] Report checkpoint gain classification from model 0 versus 499, training
  and evaluation outcomes, exact job/artifact hashes, anomalies, and the
  one-seed claim boundary.
- [ ] Stop. Do not launch another seed or tune the treatment.
