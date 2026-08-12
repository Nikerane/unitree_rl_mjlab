# Z1 FIC Joint-Position Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind the corrected controlled-drop reference only to FIC-0/FIC-TT, qualify a minimal collision-proof Vega path, train the matched seed-2 pilots for 200 iterations with 4,096 environments, and evaluate both on the same 64 initial episodes.

**Architecture:** Override the delivered-impulse normalizer only inside the existing joint-position factory, leaving the historical Cartesian factory constant untouched. Keep `scripts/train.py` unchanged; add one two-element Slurm launcher that freezes the arm map and budget, plus one dedicated evaluator that accepts only the two FIC tasks and records one first episode from each of 64 identically initialized environments. Store checkpoints and logs outside the checkout and bank only compact JSON/result documentation in Git.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, mujoco-warp 3.8.1, RSL-RL/CatPPO, pytest, Bash/Slurm, Vega A100.

## Global Constraints

- Work only in `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal` on branch `codex/z1-fic-pilot`.
- Use exactly `I_ref = 0.2799950838088989 N·s`, from `evaluation/results/2026-08-12_z1_controlled_drop_poll_height_iref/primary.json`.
- Keep `src/tasks/hammer/config/z1/env_cfgs.py:I_REF_FIRST_STRIKE_SUCCESS = 0.3088` unchanged so every Cartesian task preserves its historical normalization.
- Preserve D4 weight `4.0`, P guidance, the production first-strike reader, contact behavior, substep velocity CaT, measured/log-only per-joint impulse CaT, and the fixed gains/efforts.
- FIC-0 has no `r_tt`. FIC-TT uses exactly `k_tt=1.0`, reward weight `-1.0`, remains active through contact, and remains outside CaT's positive-return partition.
- Disable only `substep_impulse_rows` in the two FIC configs. This optional row-attribution diagnostic is not read by reward or CaT and is explicitly unqualified at 4,096 environments. Keep the production substep impulse accumulator enabled.
- Train both arms at seed `2`, `200` iterations, `4,096` environments, save interval `50`, and require `model_199.pt`. Do not accept reward, gain, reset, CaT-probability, curriculum, domain-randomization, or task overrides.
- Evaluate deterministic mean policies on the same seed `2026081202`, exactly `64` environments × their first episode, a `4.0 s` horizon, and no auto-reset. Require the recorded initial-population hashes to match between arms.
- Do not add VIC, parameter sweeps, curriculum/domain randomization, controlled-drop variants, ONNX/evaluator manifests, plotting frameworks, or checkpoint copies to Git.
- Preserve and never stage the six owner-owned untracked research/result paths.

---

### Task 1: Bind the corrected reference only to FIC and expose it live

**Files:**
- Modify: `src/tasks/hammer/config/z1/__init__.py:325-354`
- Modify: `src/tasks/hammer/rl/runner.py:63-159`
- Modify: `scripts/smoke_joint_position_fixed.py:190-316`
- Modify: `tests/test_joint_position_config.py:100-345`
- Modify: `tests/test_smoke_joint_position_fixed.py:339-470`

**Interfaces:**
- Produce: private constant `_FIC_CONTROLLED_DROP_I_REF_N_S = 0.2799950838088989` beside `_joint_position_fixed_env_cfg`.
- Extend: joint metadata from `_get_hammer_metadata(...)` with `delivered_impulse_i_ref_n_s: float`.
- Preserve: Cartesian metadata dictionary byte-for-byte/equality-wise.

- [ ] **Step 1: RED — freeze the FIC-only configuration difference**

  Add train/play assertions for both FIC tasks:

  ```python
  assert fic.rewards["delivered_impulse"].params["i_ref"] == 0.2799950838088989
  assert fic.rewards["delivered_impulse"].weight == 4.0
  assert fic.metrics["substep_impulse_rows"].params["enabled"] is False
  ```

  Assert the exact Cartesian parent still uses `0.3088`, keeps D4=`4.0`, and keeps `substep_impulse_rows.enabled is True`. Update the structural-difference test so FIC-0 differs from its Cartesian parent only in `actions`, delivered `i_ref`, and the row-attribution performance flag. Retain the existing FIC-TT assertions for weight `-1.0`, `k_tt=1.0`, and the negative-term split.

  Run the selected tests and require failure on the current inherited `0.3088`/enabled row diagnostic.

- [ ] **Step 2: GREEN — implement the one factory seam**

  Inside `_joint_position_fixed_env_cfg`, immediately after `_presentation_i_off_env_cfg(...)`:

  ```python
  cfg.rewards["delivered_impulse"].params["i_ref"] = (
      _FIC_CONTROLLED_DROP_I_REF_N_S
  )
  cfg.metrics["substep_impulse_rows"].params["enabled"] = False
  ```

  Do not modify `z1_hammer_env_cfg`, `I_REF_FIRST_STRIKE_SUCCESS`, RTT, contact, or CaT code.

- [ ] **Step 3: RED/GREEN — expose and validate the live scalar**

  Extend only the joint-position metadata branch. Read `delivered_impulse` from the live reward manager, reject a non-numeric, non-finite, or non-positive `i_ref`, and emit `delivered_impulse_i_ref_n_s`. Add the exact value to the live smoke and lossless ONNX round-trip assertions. Add one rejection test that mutates the live value to `NaN` and requires `_get_hammer_metadata` to fail closed.

- [ ] **Step 4: Verify and commit Task 1**

  Run the focused configuration/metadata/smoke tests, both live CPU manager smokes, and both one-iteration CPU CatPPO smokes. Require 17-or-more live checks per arm, one real 24-step update per arm, finite learned/checkpoint tensors, and `git diff --check`. Stage only the five paths and commit `feat(hammer): bind calibrated reference to FIC`.

---

### Task 2: Add a frozen two-arm Vega launcher

**Files:**
- Create: `scripts/slurm/vega_fic_joint_pilot.sbatch`
- Create: `tests/test_fic_joint_pilot_launcher.py`

**Interfaces:**
- Consume required environment variables: `RUN_ROOT`, `EXPECTED_CODE_REVISION`, `ASSET_REPO`, `EXPECTED_ASSET_REVISION`, and Slurm's job/array identifiers.
- Produce one isolated attempt directory per array element under `$HOME/z1-fic-joint-pilot/<code-sha>/`, one `model_199.pt`, one evaluation JSON, and printed SHA-256 values.
- Fixed map: array `0 -> (FIC-0, fic0)`, `1 -> (FIC-TT, fictt)`.

- [ ] **Step 1: RED — add launcher contract tests before the file exists**

  Require the new launcher to contain `#SBATCH --array=0-1`, `#SBATCH --gres=gpu:1`, and `#SBATCH --no-requeue`; exact task/short mapping; literal seed `2`, iterations `200`, environments `4096`, save interval `50`; clean 40-hex code/asset equality guards; an empty collision-proof attempt path; exact A100 and package-version guards; no scientific override flags; and postconditions for one run directory, `model_199.pt`, finite checkpoint tensors, evaluator success, and hashes.

  Add mutation-oriented shell invocations that reject an out-of-range array index, malformed revisions, dirty code/assets, and a reused attempt directory before any training command.

- [ ] **Step 2: GREEN — implement the minimal launcher**

  Hard-code the scientific matrix in a `case "$SLURM_ARRAY_TASK_ID"` block. Run training from the fresh attempt directory with:

  ```bash
  "$PY" "$RUN_ROOT/scripts/train.py" "$TASK" \
    --gpu-ids '[0]' \
    --agent.logger tensorboard \
    --agent.run-name "$RUN_NAME" \
    --agent.seed 2 \
    --agent.max-iterations 200 \
    --agent.save-interval 50 \
    --env.scene.num-envs 4096
  ```

  Do not pass any reward, gain, reset, CaT probability, row-diagnostic, curriculum, domain-randomization, or task override. Require exactly one matching run directory and `model_199.pt`; load the checkpoint on CPU and reject missing/non-finite tensors.

- [ ] **Step 3: Verify and commit Task 2**

  Run `bash -n`, the focused launcher tests, and `git diff --check`. Stage only the launcher and its test; commit `feat(hammer): add guarded FIC pilot launcher`.

---

### Task 3: Add the dedicated matched 64-episode evaluator

**Files:**
- Create: `evaluation/joint_position/evaluate_fic_pilot.py`
- Create: `tests/test_evaluate_fic_pilot.py`

**Interfaces:**
- Produce: `validate_fic_contract(task: str, env_cfg, agent_cfg) -> dict[str, object]`.
- Produce: `evaluate_checkpoint(task: str, checkpoint: Path, output: Path, device: str = "cpu") -> dict[str, object]`.
- CLI accepts only the two exact FIC task IDs plus `--checkpoint`, `--output`, and `--device`; seed/count/horizon/policy mode are frozen constants, not override flags.

- [ ] **Step 1: RED — freeze evaluator scope and treatment identity**

  Add tests requiring: exact two-task allow-list; six canonical joint actions; D4=`4.0`; calibrated `i_ref`; velocity CaT active; impulse CaT log-only; row attribution disabled; FIC-0 lacks `r_tt`; FIC-TT has weight `-1.0` and `k_tt=1.0`. Reject a Cartesian task, missing checkpoint, existing output, normalizer drift, action-order drift, or TT drift.

- [ ] **Step 2: RED — freeze pre-reset capture and exact count**

  With a small fake environment/collector seam, prove that each environment's first done transition is recorded once, later done states are ignored, tracker values are copied before any mutation, and incomplete populations fail instead of writing partial JSON.

- [ ] **Step 3: GREEN — implement the smallest real rollout**

  Load the training configuration (`play=False`), set `scene.num_envs=64`, `episode_length_s=4.0`, `auto_reset=False`, and seed `2026081202`. Load the strict actor checkpoint through the registered runner and use `stochastic_output=False`. Hash the reset-time robot/nail joint positions and velocities into `initial_population_sha256` before stepping.

  On each environment's first done transition, record:

  ```python
  {
      "env_id": env_id,
      "success": bool(env.reset_terminated[env_id]),
      "timeout": bool(env.reset_time_outs[env_id]),
      "episode_steps": int(step_count[env_id]),
      "first_strike_started": bool(tracker.started[env_id]),
      "first_strike_finalized": bool(tracker.finalized[env_id]),
      "first_strike_productive": bool(tracker.productive[env_id]),
      "first_strike_reason": decoded_reason,
      "precontact_velocity_m_s": float(tracker.v_precontact[env_id]),
      "first_event_impulse_n_s": float(tracker.delivered[env_id]),
      "nail_depth_m": float(depth[env_id]),
      "joint_impulse_peak_n_m_s": six_finite_values,
  }
  ```

  Require exactly 64 ordered records and finite metrics. Summarize success/productive rates, impulse mean/std/min/max, and per-joint peak p95/max. Record checkpoint SHA-256, clean code/asset revisions, protocol constants, live treatment contract, and population hash. Use an atomic create-new publication boundary and never overwrite an existing result.

- [ ] **Step 4: Verify and commit Task 3**

  Run focused evaluator tests, create one-iteration CatPPO checkpoints for both arms, and execute a bounded live evaluator smoke with the real manager/checkpoint seam. Require valid compact JSON and matching population hashes. Run `git diff --check`; stage only evaluator and test; commit `feat(hammer): evaluate matched FIC pilot`.

---

### Task 4: Qualify, review, train in parallel, evaluate, and bank the result

**Files:**
- Create: `evaluation/results/2026-08-12_z1_fic_joint_pilot/fic0.json`
- Create: `evaluation/results/2026-08-12_z1_fic_joint_pilot/fictt.json`
- Create: `docs/results/2026-08-12_z1_fic_joint_pilot.md`
- Modify only files required by concrete review findings, with a failing test first.

**Interfaces:**
- Consume: reviewed Tasks 1-3, exact clean code/assets, and Vega A100.
- Produce: two `model_199.pt` checkpoint identities, two matched 64-episode JSONs, one concise result record, and a pushed branch.

- [ ] **Step 1: Run implementation verification and reviews**

  Run the focused suite and both genuine one-iteration CatPPO smokes. On the same
  final implementation revision, require all phases A--M of
  `docs/research/reward-design/validate_rewards.py`, then require
  `docs/research/reward-design/verify_contact_sensor.py` and
  `docs/research/reward-design/verify_reward_setup.py` to pass before any GPU
  training submission. Run the full CPU suite once on the final implementation
  revision. Review the immutable package with Opus, Gemini, DeepSeek, and
  repository Standards/Spec lanes; document provider unavailability truthfully
  and use explicitly labeled fresh ultra substitutes where necessary. Apply at
  most one bounded test-first code-fix wave and re-review it once.

- [ ] **Step 2: Push and deploy the exact clean implementation**

  Push `codex/z1-fic-pilot`. Create a fresh detached Vega checkout at the pushed SHA, require clean code/assets and exact mjlab/MuJoCo/mujoco-warp versions, then run one Slurm CUDA smoke invocation:

  ```bash
  python scripts/smoke_joint_position_fixed.py \
    --task all --device cuda:0 --num-envs 8 --steps 3
  ```

  Stop on any smoke/config/provenance failure; do not submit training from a different revision.

- [ ] **Step 3: Submit both matched arms in parallel**

  Submit exactly one `0-1` array from `vega_fic_joint_pilot.sbatch`. Preserve both logs. Do not retry a Python-started scientific run; classify and report any infrastructure-only pre-Python failure before deciding whether an identical resubmission is permissible.

- [ ] **Step 4: Audit and bank evaluation outputs**

  Require both elements `COMPLETED/0:0`, exact frozen identities, one finite `model_199.pt` each, exactly 64 episodes each, and equal initial-population hashes. Copy JSONs byte-for-byte, recompute hashes and aggregate metrics independently, and write the concise result document. Checkpoints remain on Vega; record their absolute paths and SHA-256 values.

- [ ] **Step 5: Final result review, commit, and push**

  Independently review every result row, aggregate, job/checkpoint identity, and limitation. State that this is a one-seed simulation pilot and that FIC-TT is not required to outperform FIC-0. Stage only the two JSONs and one result document; commit `docs(hammer): bank matched FIC pilot` and push. Stop before active impulse-CaT or VIC work.

## Plan self-review

- Spec coverage: corrected FIC-only normalization, Cartesian preservation, D4/RTT/CaT semantics, clean CUDA gate, matched 200×4096 seed-2 training, and same 64-episode evaluation all have explicit tasks.
- Scope: no controlled-drop variants, active impulse-CaT, VIC, sweep, DR/curriculum, checkpoint commit, or final-campaign evaluator infrastructure is included.
- Type/name consistency: both launcher and evaluator use the two existing registered task IDs; evaluator output fields and Task-4 banking inputs match exactly.
- Placeholder scan: no implementation placeholder or unresolved scientific choice remains.
