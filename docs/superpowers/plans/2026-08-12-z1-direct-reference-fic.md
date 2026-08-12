# Z1 Direct-Reference FIC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Every implementation change follows the repository TDD skill: public-seam RED, smallest GREEN, then refactor.

**Goal:** Add and qualify the approved 40-observation direct-reference FIC-0/FIC-TT pair, train matched seeds 2/3/4 for 500 PPO iterations on Vega, and bank compact fixed-impedance evidence before any VIC work.

**Architecture:** Reuse the existing `SingleStrikeReference`, `ImitationPriorTerm`, qualified six-joint action, fixed plant, soft velocity-CaT, log-only impulse-CaT, and compact FIC evaluator. Add immutable task identities rather than repurposing the banked waypoint tasks. Record literal live observation/reward identity in joint-policy metadata; strict model loading supplies the 47-versus-40 tensor-shape boundary between the waypoint and direct-reference checkpoints.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, mujoco-warp 3.8.1, RSL-RL/CatPPO, pytest, Bash/Slurm, TensorBoard event files, Vega A100.

## Global Constraints

- Work only in `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal` on `codex/z1-fic-pilot`.
- Preserve every Cartesian task and every existing `...Guideline-CProgress...JointPosition-Fixed[-TT]` task, checkpoint, JSON, and result record.
- Preserve all six owner-owned untracked research/result paths and stage only explicit paths from the task being committed.
- Direct FIC-0 has no `r_tt`; direct FIC-TT has `joint_trackability_cost`, weight `-1.0`, exactly `k_tt=1.0`, and `r_tt` remains outside CaT's positive-return partition.
- Preserve reward signs, contact latches, D4=`4.0`, `I_ref=0.2799950838088989`, fixed gains/efforts, fixed train/play reset, substep velocity-CaT, log-only impulse-CaT, disabled row attribution, and all existing negative terms.
- Keep the existing action-rate penalty `-0.01 * sum((a_t-a_{t-1})^2)`. Do not add the paper's post-task velocity-jitter term: this task terminates at success and has no post-task settling phase.
- Do not add Cartesian retraining, parameter sweeps, new reference formulas, post-contact reference credit, active impulse pressure, changed caps, DR, gain actions, VIC, or controlled-drop variants.
- Run the full CPU suite once on the reviewed implementation revision. Do not count the 64 evaluation worlds or their rows as independent training seeds.

---

### Task 1: Add immutable direct-reference joint task identities

**Files:**
- Modify: `tests/test_joint_position_config.py`
- Modify: `tests/test_configs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`

**Interfaces:**
- Add `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-Fixed`.
- Add the identical ID with suffix `-TT`.
- Add `_direct_reference_joint_position_fixed_env_cfg(*, play: bool, trackability: bool)` without modifying `_joint_position_fixed_env_cfg`.

- [ ] **Step 1: RED — freeze the registered treatment contract.**

  Add tests for train/play registration, no waypoint observations/metric/rewards, exact `r_imit` implementation and `sigma=0.05`, and the exact actor/critic term order:

  ```python
  (
      "joint_pos", "joint_vel", "ee_pos", "ee_vel", "head_pos", "head_vel",
      "nail_top_pos", "nail_depth", "strike_phase", "strike_ref_error", "actions",
  )
  ```

  Freeze this literal training curriculum:

  ```python
  [
      {"step": 0, "weight": 0.10},
      {"step": 1200, "weight": 0.08},
      {"step": 2400, "weight": 0.06},
      {"step": 3600, "weight": 0.04},
      {"step": 4800, "weight": 0.02},
      {"step": 6000, "weight": 0.00},
  ]
  ```

  Assert play retains `r_imit` but has no curriculum, both reset position/velocity ranges are `(0.0, 0.0)`, and the registry delta is exactly the two additive IDs. Run the three new selected tests and observe missing-registration failures.

- [ ] **Step 2: RED — freeze FIC identity and isolation.**

  Require the same qualified six-joint action, fixed actuator signature, D4, corrected `I_ref`, enabled production accumulator, disabled row attribution, velocity-CaT and impulse-log-only contracts as the banked pair. Prove direct FIC-TT differs from direct FIC-0 only by `r_tt` with weight `-1.0`, `k_tt=1.0`, ordered joints, and the existing negative split. Prove the direct and waypoint treatments differ only at guidance observations/metric/reward/curriculum.

- [ ] **Step 3: GREEN — install the smallest separate factory and registrations.**

  Build from:

  ```python
  z1_hammer_env_cfg(
      play=play,
      imitation=True,
      cat_impulse=True,
      event_correct=True,
      event_linear=True,
      cat_soft=True,
      vel_cat_substep=True,
  )
  ```

  Then pin D4, corrected `I_ref`, row attribution off, both reset ranges zero, the qualified joint action, and optional existing trackability installer. Do not duplicate reference or reward code.

- [ ] **Step 4: Verify Task 1.**

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-direct-config PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_joint_position_config.py tests/test_configs.py::TestCartesianGuidelineStudy \
    tests/test_configs.py::TestImitationFlag tests/test_strike_reference.py \
    tests/test_imitation_reward.py tests/test_cat_soft_hook.py
  git diff --check
  ```

---

### Task 2: Add live observation metadata and real 40-observation smokes

**Files:**
- Modify: `tests/test_smoke_joint_position_fixed.py`
- Modify: `src/tasks/hammer/rl/runner.py`
- Modify: `scripts/smoke_joint_position_fixed.py`
- Modify: `scripts/smoke_cat_soft.py`

**Interfaces:**
- Preserve the Cartesian metadata dictionary exactly.
- Add joint-only `observation_names`, `observation_widths`, `guidance_type`, `guidance_reward_key`, and `guidance_reward_impl`, derived from the live managers.
- Extend live and CatPPO smokes to the two new task IDs with exact observation width `40` and action width `6`.

- [ ] **Step 1: RED — specify the live schema and checkpoint boundary.**

  Cover all four joint tasks. Waypoint schemas remain the existing 15 names/47 columns; direct schemas are the exact 11 names/40 columns. Require live guidance identity (`waypoint_progress`/`r_waypoint_progress` versus `direct_reference`/`r_imit`) and reject mismatched actor/critic terms, widths, or reward identity. Save a real 47-column smoke checkpoint and require strict load into a 40-column runner to raise `RuntimeError` on the incompatible normalizer/first-layer tensors. This test is the width-incompatibility boundary; do not claim that ONNX metadata itself validates `.pt` loading.

- [ ] **Step 2: GREEN — implement fail-closed live metadata dispatch.**

  Read `env.observation_manager.active_terms` and `group_obs_dim`; accept only the two literal joint schemas. Direct identity requires `r_imit` and forbids the waypoint rewards, tracker, and four waypoint observations; waypoint identity requires the inverse. Leave the Cartesian branch byte-for-byte equivalent.

- [ ] **Step 3: RED/GREEN — extend both real smoke paths.**

  Give `smoke_joint_position_fixed.py` a literal four-task contract map with `{width, guidance, tt}`. Preserve every existing waypoint check and add direct checks for `(N,40)`, `r_imit=0.1`, `sigma=0.05`, no waypoint state, fixed reset/action/plant, D4/`I_ref`, positive synthetic velocity-CaT dose, log-only impulse-CaT, and exact arm-specific negative split. In `smoke_cat_soft.py`, map the old tasks to 47 and the new tasks to 40 while preserving the genuine 24-step CatPPO rollout/update, CatRolloutStorage, finite learner state, and finite saved checkpoint.

- [ ] **Step 4: Verify Task 2 and run both CPU training smokes.**

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-direct-runtime PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_smoke_joint_position_fixed.py tests/test_cat_ppo_gae.py tests/test_cat_soft_hook.py
  MPLCONFIGDIR=/private/tmp/z1-direct-live PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/smoke_joint_position_fixed.py \
    --task all --device cpu --num-envs 8 --steps 3
  MPLCONFIGDIR=/private/tmp/z1-direct-cat0 PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/smoke_cat_soft.py \
    --task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-Fixed \
    --device cpu --num-envs 8 --iters 1
  MPLCONFIGDIR=/private/tmp/z1-direct-cattt PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/smoke_cat_soft.py \
    --task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-Fixed-TT \
    --device cpu --num-envs 8 --iters 1
  git diff --check
  ```

---

### Task 3: Retarget the compact evaluator to the intended endpoints

**Files:**
- Modify: `tests/test_evaluate_fic_pilot.py`
- Modify: `evaluation/joint_position/evaluate_fic_pilot.py`

**Interfaces:**
- Retarget `TASKS` to exactly the two direct-reference IDs; banked schema-1 waypoint JSONs remain immutable at their original revision.
- Add required `--training-seed` limited to `2`, `3`, or `4`; require final checkpoint basename `model_499.pt`.
- Keep the 64-env, seed-`2026081202`, first-episode, 4-second, mean-policy, no-auto-reset protocol.

- [ ] **Step 1: RED — freeze the direct-reference evaluator contract.**

  Require exact actor/critic names and width 40, six ordered actions, fixed plant/reset, D4/`I_ref`, velocity-CaT, impulse-log-only, row attribution off, exact `r_imit`/sigma/curriculum, no waypoint behavior, and arm-specific `r_tt`. Reject the banked waypoint IDs, scientific drift, training seeds outside `{2,3,4}`, and non-`model_499.pt` checkpoints.

- [ ] **Step 2: RED — freeze the essential episode measurements and their axes.**

  Extend terminal rows with:

  ```python
  {
      "joint_velocity_peak_rad_s": six_values,
      "joint_velocity_legal": bool,
      "joint_target_rmse_rad": float,
      "joint_target_error_max_rad": float,
      "reference_error_mean_m": float,
      "reference_error_max_m": float,
      "reference_error_samples": int,
  }
  ```

  Define the row statistics exactly:

  - `joint_velocity_peak_rad_s[j] = max_t peak_qv_joint[t,j]`; legal means every column is `<=3.1415`.
  - `joint_target_rmse_rad = sqrt(sum_(t,j) (q_des_applied[t,j]-q_actual[t+1,j])^2 / (6*T))`.
  - `joint_target_error_max_rad = max_(t,j) abs(q_des_applied[t,j]-q_actual[t+1,j])`.
  - Reference samples use the current head and `ref.waypoint(ref.preview(...))` only where the live `ImitationPriorTerm._contacted` latch is false after the reward step. This is the actual reward eligibility, including `found`, `current_contact_time`, and within-control-interval `last_contact_time`; do not substitute `FirstStrikeEventTracker.started`.
  - `reference_error_mean_m` and `reference_error_max_m` are the mean/max over those eligible samples; zero samples fail closed.

  Unit-test episode-held 500 Hz joint peaks, the legality boundary, target-error axes for both arms, and reference exclusion for current contact and touch-release. Keep existing delivered impulse, six-joint impulse peak, task/productive success, first-terminal capture, RNG preservation, population hash, finite checks, ordered 64 rows, and atomic no-overwrite publication tests.

- [ ] **Step 3: GREEN — implement scalar/vector accumulators, not a new analysis framework.**

  Read the live substep peak tracker each control step. Compute target error from the same applied target/actual joint seam as `joint_target_squared_error`. Resolve the live `r_imit` manager instance once and read its contact latch after every step; compute reference distance only where that exact latch is still open. Add these unambiguous schema-2 summaries across the 64 per-env rows:

  - qvel: per-joint p95/max plus all-joints-legal count/rate;
  - target: mean/p95/max of per-episode RMSE and mean/p95/max of per-episode maximum absolute error;
  - reference: mean/p95/max of per-episode means and mean/p95/max of per-episode maxima;
  - impulse utilization: each row's six-vector `joint_impulse_peak_n_m_s / [1.64,3.28,1.64,1.64,1.64,1.64]`, per-joint p95/max, and all-joints-at-or-below-cap count/rate.

  Record the separate training seed and exact curriculum schedule. Do not add traces, saturation, or another evaluator layer.

- [ ] **Step 4: Verify Task 3.**

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-direct-eval PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_evaluate_fic_pilot.py tests/test_joint_trackability.py \
    tests/test_strike_reference.py tests/test_impulse_bound.py
  git diff --check
  ```

---

### Task 4: Add a collision-proof 500-iteration Vega launcher

**Files:**
- Create: `tests/test_fic_direct_reference_launcher.py`
- Create: `scripts/slurm/vega_fic_direct_reference.sbatch`
- Create: `scripts/slurm/vega_fic_direct_reference_smoke.sbatch`

**Interfaces:**
- Fixed map: indices `0/1` are seed-2 FIC-0/FIC-TT; `2/3` seed 3; `4/5` seed 4.
- Default `#SBATCH --array=0-1`; permit only the exact expansion invocation `--array=2-5` after the canary gate.
- Freeze 500 iterations, 4,096 environments, save interval 50, and `model_499.pt`.

- [ ] **Step 1: RED — specify the launcher and both legal array shapes.**

  Test exact task/arm/seed mapping, A100/no-requeue guards, full clean code and asset revisions, exact package versions, collision-proof attempt roots, one run directory, finite `model_499.pt`, evaluator success, and printed checkpoint/result hashes. Reject all other indices/shapes, reused paths, dirty/mismatched revisions, wrong GPU/runtime, missing/multiple runs, non-finite checkpoints, and existing evaluator output. Assert the train command contains only operational overrides and the four frozen training values; no reward, action, gain, reset, CaT, curriculum, or DR override is allowed.

- [ ] **Step 2: GREEN — implement the guarded six-index launcher.**

  Use `$HOME/z1-fic-direct-reference-500/$EXPECTED_CODE_REVISION/$EXPECTED_ASSET_REVISION/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SHORT}_seed${SEED}`. Train with:

  ```bash
  "$PY" "$RUN_ROOT/scripts/train.py" "$TASK" \
    --gpu-ids '[0]' --agent.logger tensorboard --agent.run-name "$RUN_NAME" \
    --agent.seed "$SEED" --agent.max-iterations 500 --agent.save-interval 50 \
    --env.scene.num-envs 4096
  ```

  Evaluate `model_499.pt` with the matching training seed. Inspect the TensorBoard scalar `Curriculum/r_imit_anneal/weight`; require finite and nonincreasing values. Match every raw float32 sample to exactly one of `{0.10,0.08,0.06,0.04,0.02,0.00}` with absolute tolerance `1e-6`, require all six plateaus, and require the final sample within `1e-6` of zero. Retain raw samples in `observed`; canonicalize the separately reported `final_weight` only after validation. Publish one canonical compact `${SHORT}_seed${SEED}_curriculum.json` beside the evaluation JSON:

  ```json
  {"schema_version":1,"task":"Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-Fixed","arm":"fic0","training_seed":2,"scalar":"Curriculum/r_imit_anneal/weight","expected_control_step_stages":[[0,0.1],[1200,0.08],[2400,0.06],[3600,0.04],[4800,0.02],[6000,0.0]],"observed":[{"iteration":0,"weight":0.1}],"final_weight":0.0}
  ```

  Here the displayed seed and observed list vary with the mapped run; `observed` contains the complete ordered TensorBoard scalar stream with integer event steps and finite weights. Write with sorted keys, compact separators, newline, create-new semantics, then hash both JSONs.

  Add the dedicated smoke launcher with the same clean code/asset/A100/package guards. It runs `smoke_joint_position_fixed.py --task all --device cuda:0` and one genuine CUDA CatPPO iteration for each direct-reference arm, and prints the GPU name, qualified revisions, and completion marker. It accepts no scientific overrides.

- [ ] **Step 3: Verify Task 4.**

  ```bash
  bash -n scripts/slurm/vega_fic_direct_reference.sbatch
  bash -n scripts/slurm/vega_fic_direct_reference_smoke.sbatch
  MPLCONFIGDIR=/private/tmp/z1-direct-launch PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_fic_direct_reference_launcher.py tests/test_evaluate_fic_pilot.py
  git diff --check
  ```

---

### Task 5: Add truthful fixed-reset render identity

**Files:**
- Modify: `tests/test_fixed_reset_video_library.py`
- Modify: `evaluation/analysis/fixed_reset_video_library.py`
- Modify: `scripts/render_policy.py`

- [ ] **Step 1: RED — register only the new qualitative campaign.**

  Map campaign `fic-direct-reference`, arms `FIC-0` and `FIC-TT`, to the two new tasks. Add both task IDs to `TREATMENT_BY_TASK` as direct-reference guidance, soft substep velocity-CaT, log-only impulse-CaT, and a new reference-line-only geometry identity (no gates or waypoint markers). Do not alter the frozen 56-policy inventory. Require the non-substep trajectory label to state `SingleStrikeReference (reward prior 0.10→0 by iteration 250)` for this campaign while the default label and all frozen rendering behavior remain unchanged.

- [ ] **Step 2: GREEN — parameterize the reference label with a byte-preserving default.**

  Change `_draw_trajectory(..., *, reference_legend: str)` and `write_trajectory_png(trace, path, *, reference_legend="SingleStrikeReference (observation only)", reference_footer="SingleStrikeReference (observation only) · black dashed · not rewarded")`. Use those exact defaults at both legend/footer call sites so frozen outputs do not change. `render_policy.py` passes direct-reference legend/footer strings only when `cfg.campaign == "fic-direct-reference"`. Reuse the existing fixed-reset, mean-policy renderer; do not create a second renderer or new video library.

- [ ] **Step 3: Verify Task 5.**

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-direct-render PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_fixed_reset_video_library.py
  git diff --check
  ```

---

### Task 6: Review, release-gate, push, and qualify CUDA

**Files:**
- Modify only concrete correctness fixes identified by review, with a failing regression test first.

- [ ] **Step 1: Run focused scientific gates.**

  Run Tasks 1–5 tests plus `tests/test_joint_action_qualification.py`, `tests/test_joint_trackability.py`, `tests/test_impulse_constraint.py`, `tests/test_delivered_impulse_reward.py`, and both CPU CatPPO smokes. Then run:

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-direct-gates PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/validate_rewards.py
  MPLCONFIGDIR=/private/tmp/z1-direct-gates PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/verify_contact_sensor.py
  MPLCONFIGDIR=/private/tmp/z1-direct-gates PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/verify_reward_setup.py
  MPLCONFIGDIR=/private/tmp/z1-direct-gates PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python docs/research/reward-design/playback_reference.py
  ```

- [ ] **Step 2: Review in parallel, fix concrete findings once, and rereview.**

  Review the completed diff against `docs/superpowers/specs/2026-08-12-z1-direct-reference-fic-design.md` with Opus, Gemini, DeepSeek, and repository Standards/Spec lanes. Label provider unavailability honestly. Apply only concrete Critical/Important correctness fixes, each preceded by a failing regression test, then perform one bounded rereview.

- [ ] **Step 3: Run the full CPU suite exactly once on the reviewed tree.**

  ```bash
  set -e
  set -o pipefail
  MPLCONFIGDIR=/private/tmp/z1-direct-full PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    --override-ini addopts='' --tb=short -p no:cacheprovider \
    -o faulthandler_timeout=120 --durations=25 \
    | tee /private/tmp/z1-direct-reference-full-pytest.log
  test "${pipestatus[1]}" -eq 0
  git diff --check
  ```

  Run the command through `tee /private/tmp/z1-direct-reference-full-pytest.log`, preserve pytest's exit status with zsh `pipestatus[1]`, confirm the zero-exit summary in that log, and confirm all protected untracked paths are still unmodified and unstaged.

- [ ] **Step 4: Commit/push the reviewed implementation and run clean CUDA gates.**

  Push `codex/z1-fic-pilot`. From the local controller, resolve Vega's home, fetch the branch, and create a fresh detached sibling worktree so the production asset path remains canonical:

  ```bash
  Z1_DIRECT_REMOTE_HOME="$(ssh vega 'printf %s "$HOME"')"
  Z1_DIRECT_BASE_REPO="$Z1_DIRECT_REMOTE_HOME/repos/unitree_rl_mjlab"
  ssh vega "git -C '$Z1_DIRECT_BASE_REPO' fetch origin codex/z1-fic-pilot"
  Z1_DIRECT_CODE_REV="$(ssh vega "git -C '$Z1_DIRECT_BASE_REPO' rev-parse origin/codex/z1-fic-pilot")"
  Z1_DIRECT_RUN_ROOT="$Z1_DIRECT_REMOTE_HOME/repos/unitree_rl_mjlab-direct-$Z1_DIRECT_CODE_REV"
  Z1_DIRECT_ASSET_REPO="$Z1_DIRECT_REMOTE_HOME/repos/safe_impact_manipulation"
  Z1_DIRECT_ASSET_REV="$(ssh vega "git -C '$Z1_DIRECT_ASSET_REPO' rev-parse HEAD")"
  ssh vega "test ! -e '$Z1_DIRECT_RUN_ROOT' && git -C '$Z1_DIRECT_BASE_REPO' worktree add --detach '$Z1_DIRECT_RUN_ROOT' '$Z1_DIRECT_CODE_REV'"
  Z1_DIRECT_CUDA_JOB="$(ssh vega "sbatch --parsable \
    --export=ALL,RUN_ROOT='$Z1_DIRECT_RUN_ROOT',ASSET_REPO='$Z1_DIRECT_ASSET_REPO',EXPECTED_CODE_REVISION='$Z1_DIRECT_CODE_REV',EXPECTED_ASSET_REVISION='$Z1_DIRECT_ASSET_REV' \
    '$Z1_DIRECT_RUN_ROOT/scripts/slurm/vega_fic_direct_reference_smoke.sbatch'")"
  ssh vega "sacct -j '$Z1_DIRECT_CUDA_JOB' --format=JobID,State,ExitCode,Elapsed,NodeList -P"
  ```

  Require `COMPLETED/0:0`, `Z1_DIRECT_CUDA_GPU=NVIDIA A100-SXM4-40GB` in the log, exact printed code/asset revisions, four passing live-manager task summaries, two completed real CatPPO updates, and finite checkpoints. Using `apply_patch`, create `<this plan's SDD workspace>/vega.env` with single-quoted assignments for `Z1_DIRECT_REMOTE_HOME`, `Z1_DIRECT_BASE_REPO`, `Z1_DIRECT_CODE_REV`, `Z1_DIRECT_RUN_ROOT`, `Z1_DIRECT_ASSET_REPO`, `Z1_DIRECT_ASSET_REV`, and `Z1_DIRECT_CUDA_JOB`; append the same evidence to the SDD ledger. Task 7 sources that git-ignored state file and must never re-resolve branch or asset HEAD.

---

### Task 7: Train matched seeds, render, bank, review, and stop

**Files:**
- Create: `evaluation/results/2026-08-12_z1_fic_direct_reference/fic0_seed2.json`
- Create: `evaluation/results/2026-08-12_z1_fic_direct_reference/fictt_seed2.json`
- Create: the matching four seed-3/4 JSONs and six `*_curriculum.json` files in the same directory.
- Create: `evaluation/results/2026-08-12_z1_fic_direct_reference/fic0_seed2_montage.png`
- Create: `evaluation/results/2026-08-12_z1_fic_direct_reference/fic0_seed2_trajectory.png`
- Create: the matching FIC-TT seed-2 PNGs and `SHA256SUMS`.
- Create: `docs/results/2026-08-12_z1_fic_direct_reference.md`
- Modify: `docs/results/README.md`
- Create: `tests/test_fic_direct_reference_results.py`

- [ ] **Step 1: Resolve the exact remote revisions and submit the seed-2 canary pair.**

  Reuse the exact clean revisions and detached worktree that passed Task 6. On the local controller resolve this plan's SDD workspace and source its `vega.env`. Verify the qualified worktree/assets still match, then submit with one shell-safe remote command string:

  ```bash
  Z1_DIRECT_SDD_WORKSPACE="$(/Users/nikerane/.codex/plugins/cache/openai-curated-remote/superpowers/6.2.0/skills/subagent-driven-development/scripts/sdd-workspace docs/superpowers/plans/2026-08-12-z1-direct-reference-fic.md)"
  source "$Z1_DIRECT_SDD_WORKSPACE/vega.env"
  ssh vega "test \"\$(git -C '$Z1_DIRECT_RUN_ROOT' rev-parse HEAD)\" = '$Z1_DIRECT_CODE_REV' && \
    test \"\$(git -C '$Z1_DIRECT_ASSET_REPO' rev-parse HEAD)\" = '$Z1_DIRECT_ASSET_REV' && \
    test -z \"\$(git -C '$Z1_DIRECT_RUN_ROOT' status --porcelain=v1 --untracked-files=all)\" && \
    test -z \"\$(git -C '$Z1_DIRECT_ASSET_REPO' status --porcelain=v1 --untracked-files=all)\""
  Z1_DIRECT_CANARY_JOB="$(ssh vega "sbatch --parsable \
    --export=ALL,RUN_ROOT='$Z1_DIRECT_RUN_ROOT',ASSET_REPO='$Z1_DIRECT_ASSET_REPO',EXPECTED_CODE_REVISION='$Z1_DIRECT_CODE_REV',EXPECTED_ASSET_REVISION='$Z1_DIRECT_ASSET_REV' \
    '$Z1_DIRECT_RUN_ROOT/scripts/slurm/vega_fic_direct_reference.sbatch'")"
  printf '%s\n' "$Z1_DIRECT_CANARY_JOB"
  ```

  Immediately add the single-quoted `Z1_DIRECT_CANARY_JOB` assignment to `vega.env` with `apply_patch`. Monitor from the local controller:

  ```bash
  ssh vega sacct -j "$Z1_DIRECT_CANARY_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList -P
  ssh vega squeue -j "$Z1_DIRECT_CANARY_JOB" -o '%.18i %.9P %.24j %.8T %.10M %.6D %R'
  ```

  After both array elements are terminal, copy exactly one of each canary evaluator/curriculum file to `/private/tmp/z1-direct-reference-canary/` and audit locally with tolerance-based Python checks:

  ```bash
  Z1_DIRECT_CANARY_DIR="$(mktemp -d /private/tmp/z1-direct-reference-canary.XXXXXX)"
  for Z1_DIRECT_NAME in fic0_seed2.json fictt_seed2.json fic0_seed2_curriculum.json fictt_seed2_curriculum.json
  do
    Z1_DIRECT_MATCHES="$(ssh vega "find '$Z1_DIRECT_REMOTE_HOME/z1-fic-direct-reference-500/$Z1_DIRECT_CODE_REV/$Z1_DIRECT_ASSET_REV' -type f -name '$Z1_DIRECT_NAME' -print")"
    test "$(printf '%s\n' "$Z1_DIRECT_MATCHES" | sed '/^$/d' | wc -l | tr -d ' ')" -eq 1
    rsync -av "vega:$Z1_DIRECT_MATCHES" "$Z1_DIRECT_CANARY_DIR/$Z1_DIRECT_NAME"
  done
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python - "$Z1_DIRECT_CANARY_DIR" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
payloads = {path.name: json.loads(path.read_text()) for path in root.glob("*.json")}
a, b = payloads["fic0_seed2.json"], payloads["fictt_seed2.json"]
assert len(a["episodes"]) == len(b["episodes"]) == 64
assert a["summary"]["task_success_n"] > 0 and b["summary"]["task_success_n"] > 0
assert a["summary"]["productive_first_strike_n"] > 0 and b["summary"]["productive_first_strike_n"] > 0
assert a["initial_population_sha256"] == b["initial_population_sha256"]
stages = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1)
for name in ("fic0_seed2_curriculum.json", "fictt_seed2_curriculum.json"):
    curriculum = payloads[name]
    raw = [float(row["weight"]) for row in curriculum["observed"]]
    assert raw and all(math.isfinite(value) for value in raw)
    assert all(raw[index] >= raw[index + 1] - 1e-6 for index in range(len(raw) - 1))
    assert all(any(abs(value - stage) <= 1e-6 for stage in stages) for value in raw)
    assert all(any(abs(value - stage) <= 1e-6 for value in raw) for stage in stages)
    assert abs(float(curriculum["final_weight"])) <= 1e-6
PY
  ```

  Also require exactly one filesystem match for every name, both Slurm elements `COMPLETED/0:0`, finite evaluator data, and valid printed hashes. If either arm fails, stop and diagnose; never launch only one arm.

- [ ] **Step 2: Submit seeds 3 and 4 in parallel without changing code or parameters.**

  From the local controller, source the qualified values and submit the same remote launcher with the only legal expansion shape:

  ```bash
  source "$Z1_DIRECT_SDD_WORKSPACE/vega.env"
  ssh vega "test \"\$(git -C '$Z1_DIRECT_RUN_ROOT' rev-parse HEAD)\" = '$Z1_DIRECT_CODE_REV' && test \"\$(git -C '$Z1_DIRECT_ASSET_REPO' rev-parse HEAD)\" = '$Z1_DIRECT_ASSET_REV'"
  Z1_DIRECT_EXPAND_JOB="$(ssh vega "sbatch --parsable --array=2-5 \
    --export=ALL,RUN_ROOT='$Z1_DIRECT_RUN_ROOT',ASSET_REPO='$Z1_DIRECT_ASSET_REPO',EXPECTED_CODE_REVISION='$Z1_DIRECT_CODE_REV',EXPECTED_ASSET_REVISION='$Z1_DIRECT_ASSET_REV' \
    '$Z1_DIRECT_RUN_ROOT/scripts/slurm/vega_fic_direct_reference.sbatch'")"
  printf '%s\n' "$Z1_DIRECT_EXPAND_JOB"
  ssh vega sacct -j "$Z1_DIRECT_EXPAND_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList -P
  ```

  Add `Z1_DIRECT_EXPAND_JOB` to `vega.env` and the SDD ledger with `apply_patch` immediately after submission.

  Require all four elements `COMPLETED/0:0`, the same clean revisions and contract, and six final `model_499.pt` checkpoints. On the local worktree, copy exactly the twelve uniquely named JSONs without copying their attempt-directory layout:

  ```bash
  Z1_DIRECT_LOCAL_RESULTS="$PWD/evaluation/results/2026-08-12_z1_fic_direct_reference"
  mkdir "$Z1_DIRECT_LOCAL_RESULTS"
  for Z1_DIRECT_NAME in \
    fic0_seed2.json fictt_seed2.json fic0_seed3.json fictt_seed3.json \
    fic0_seed4.json fictt_seed4.json fic0_seed2_curriculum.json \
    fictt_seed2_curriculum.json fic0_seed3_curriculum.json \
    fictt_seed3_curriculum.json fic0_seed4_curriculum.json fictt_seed4_curriculum.json
  do
    Z1_DIRECT_MATCHES="$(ssh vega "find \$HOME/z1-fic-direct-reference-500/$Z1_DIRECT_CODE_REV/$Z1_DIRECT_ASSET_REV -type f -name '$Z1_DIRECT_NAME' -print")"
    test "$(printf '%s\n' "$Z1_DIRECT_MATCHES" | sed '/^$/d' | wc -l | tr -d ' ')" -eq 1
    rsync -av "vega:$Z1_DIRECT_MATCHES" "$Z1_DIRECT_LOCAL_RESULTS/$Z1_DIRECT_NAME"
  done
  test "$(find "$Z1_DIRECT_LOCAL_RESULTS" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')" -eq 12
  ```

  Do not copy checkpoints into Git. Recompute each JSON hash and every row/summary with the result-test helper.

- [ ] **Step 3: Render the preregistered seed-2 checkpoints.**

  On the local controller, read each seed-2 checkpoint path/hash from the copied evaluation JSON, then invoke the exact detached renderer remotely. FIC-0:

  ```bash
  Z1_DIRECT_REMOTE_HOME="$(ssh vega 'printf %s "$HOME"')"
  Z1_DIRECT_FIC0_JSON="$Z1_DIRECT_LOCAL_RESULTS/fic0_seed2.json"
  Z1_DIRECT_FIC0_CKPT="$(/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["path"])' "$Z1_DIRECT_FIC0_JSON")"
  Z1_DIRECT_FIC0_SHA="$(/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["sha256"])' "$Z1_DIRECT_FIC0_JSON")"
  Z1_DIRECT_FIC0_RENDER="$Z1_DIRECT_REMOTE_HOME/z1-fic-direct-reference-renders/$Z1_DIRECT_CODE_REV/$Z1_DIRECT_ASSET_REV/fic0_seed2_$Z1_DIRECT_FIC0_SHA"
  ssh vega "test ! -e '$Z1_DIRECT_FIC0_RENDER' && \
    test \"\$(sha256sum '$Z1_DIRECT_FIC0_CKPT' | cut -d' ' -f1)\" = '$Z1_DIRECT_FIC0_SHA' && \
    srun --account=d2026d06-166-users --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=20G --time=00:15:00 \
      env MUJOCO_GL=egl PYTHONPATH='$Z1_DIRECT_RUN_ROOT' \
      '$Z1_DIRECT_REMOTE_HOME/repos/unitree_rl_mjlab/.venv/bin/python' '$Z1_DIRECT_RUN_ROOT/scripts/render_policy.py' \
      --checkpoint-file '$Z1_DIRECT_FIC0_CKPT' --campaign fic-direct-reference --arm FIC-0 \
      --training-seed 2 --checkpoint-sha256 '$Z1_DIRECT_FIC0_SHA' \
      --code-revision '$Z1_DIRECT_CODE_REV' --asset-revision '$Z1_DIRECT_ASSET_REV' \
      --task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-Fixed \
      --out-dir '$Z1_DIRECT_FIC0_RENDER' --steps 200 --device cpu"
  ```

  Run the identical local parse and GPU-allocated `srun` command for `fictt_seed2.json` with arm `FIC-TT`, the exact task suffix `-TT`, and render leaf `fictt_seed2_$Z1_DIRECT_FICTT_SHA`. Require each leaf to contain valid `metadata.json`, `trace.npz`, `trajectory.png`, `montage.png`, and `policy.mp4`. Copy the four PNGs explicitly:

  ```bash
  rsync -av "vega:$Z1_DIRECT_FIC0_RENDER/montage.png" "$Z1_DIRECT_LOCAL_RESULTS/fic0_seed2_montage.png"
  rsync -av "vega:$Z1_DIRECT_FIC0_RENDER/trajectory.png" "$Z1_DIRECT_LOCAL_RESULTS/fic0_seed2_trajectory.png"
  rsync -av "vega:$Z1_DIRECT_FICTT_RENDER/montage.png" "$Z1_DIRECT_LOCAL_RESULTS/fictt_seed2_montage.png"
  rsync -av "vega:$Z1_DIRECT_FICTT_RENDER/trajectory.png" "$Z1_DIRECT_LOCAL_RESULTS/fictt_seed2_trajectory.png"
  ssh vega sha256sum "$Z1_DIRECT_FIC0_RENDER/policy.mp4" "$Z1_DIRECT_FICTT_RENDER/policy.mp4"
  ```

  Retain both MP4s on Vega and record these absolute paths plus SHA-256 in the result document.

- [ ] **Step 4: Add result validation and the concise thesis record.**

  Test all six task/seed identities, 64 ordered finite rows, recomputed summaries, shared population hash, exact curriculum stages/final zero, checkpoint/result/render hashes, and cap/velocity arithmetic. Write sorted `SHA256SUMS` for the twelve JSONs and four PNGs. Report each independent training seed, then descriptive across-seed FIC-TT minus FIC-0 differences. State plainly that evaluation worlds are not training replicates, impulse-CaT was log-only, and no Cartesian/waypoint/hardware causal claim is made. Add the dated record to the canonical `docs/results/README.md` index.

  Run the newly created result test and the renderer contract test on the banked bytes:

  ```bash
  MPLCONFIGDIR=/private/tmp/z1-direct-results PYTHONPATH=. \
    /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
    tests/test_fic_direct_reference_results.py tests/test_fixed_reset_video_library.py
  git diff --check
  ```

- [ ] **Step 5: Review, commit, push, and stop.**

  Review result provenance/numerics with Opus, Gemini, DeepSeek, and repository Standards/Spec lanes; fix only concrete evidence errors and rereview once. Rerun the two focused tests above after any fix. Stage only this task's explicit result JSON/curriculum/PNG/hash/test/document/index paths, commit, push, and stop before active impulse-CaT, VIC, or controlled-drop work.

## Plan Self-Review

- **Spec coverage:** immutable direct-reference identities, exact 0.05 m reward, anneal-to-zero by iteration 250, fixed FIC contract, `k_tt=1`, 40×6 runtime, compact essential endpoints, three matched 500-iteration seeds, images/video, reviews, push, and the VIC stopping boundary all have executable gates.
- **Preservation:** every Cartesian and waypoint task/result remains registered and untouched; old schema-1 JSONs remain reproducible from their frozen revision.
- **Scope:** no jitter arm, evaluator framework, sweep, active impulse pressure, new normalization, DR, controlled-drop variant, or VIC implementation is included.
- **Names/types:** the two IDs, observation widths, checkpoint number, iteration/step schedule, seeds, action order, reward signs, and launcher/evaluator filenames agree across tasks.
- **Placeholder scan:** no placeholder command, unresolved scientific parameter, or user decision remains.
