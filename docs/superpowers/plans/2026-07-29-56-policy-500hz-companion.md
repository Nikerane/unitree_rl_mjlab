# 56-Policy Fixed-Reset 500 Hz Companion Implementation Plan

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer impulse
> cap” wording below refers only to the historical registered-task boundary, not a
> validated Z1 reaction-impulse or damage limit. The plan body remains frozen; see
> `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-evaluate the frozen 56-policy library from its one approved reset at the native 2 ms physics rate, then bank exact-onset trajectory, joint-speed, and running per-joint impulse evidence without changing training or task semantics.

**Architecture:** Extend the existing one-policy `scripts/diag_impulse_trace.py` hook so it can restore the approved reset and emit a compact, identity-bound substep record. Add one pure analysis/validation module plus a thin batch driver that reads the already frozen 56-row inventory, runs each checkpoint, validates every leaf, and derives the FQ ranking. Reuse the existing MP4 library; do not render new videos.

**Tech Stack:** Python 3.11, PyTorch, NumPy, MuJoCo/mjlab CPU simulation, matplotlib, pytest.

## Global Constraints

- Fixed impedance only; never call `set_gains`.
- Keep manufacturer `IMP_J_LIMIT` caps unchanged.
- Keep `imp_max_p=0.0` and treat Lambda as log-only.
- Evaluation only: no training, reward changes, enforcement, or checkpoint replacement.
- CPU, one environment, one fixed reset for all 56 policies.
- Reference phase/error are 50 Hz control observations. Store them once per control step and link by control-step index; do not serialize forward-filled duplicates as 500 Hz measurements.
- Reject dirty or `-dirty` provenance, checkpoint/reset/task/hash drift, nonfinite arrays, wrong 2 ms timing, partial leaves, or silent membership loss.
- Preserve the historical FQ3 cleanliness limitation; do not invent `clean_state`.
- Do not edit installed packages.
- Stage only named files; do not use `git add -A`; do not push without owner approval.

---

### Task 1: Fixed-reset rich 500 Hz single-policy recorder

**Files:**
- Modify: `scripts/diag_impulse_trace.py`
- Create: `tests/test_diag_impulse_trace.py`

**Interfaces:**
- Consumes: `evaluation.analysis.fixed_reset_video_library.load_fixed_reset`, `scripts.eval_impulse.restore_reset_state`, one checkpoint/task identity, and the existing post-accumulator `compute_substep` hook.
- Produces: `trace.npz` plus `metadata.json` for one policy; all physical arrays have length `N`, `t_s == arange(N) * 0.002`, and payload/identity SHA-256 values are recorded.

- [ ] **Step 1: Write failing pure tests for fixed-reset restoration and observation refresh**

  Import the script as a module and exercise a small helper that performs, in this exact order:

  ```python
  env.reset()
  restored_digest = restore_reset_state(env, fixed_reset)
  env.sim.forward()
  env.sim.sense()
  env.obs_buf = env.observation_manager.compute(update_history=True)
  obs = wrapped.get_observations()
  ```

  Assert that the approved digest is returned, the refresh occurs after restoration, and a malformed/tampered envelope raises before policy inference.

- [ ] **Step 2: Run the focused reset tests and verify RED**

  Run:

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
    tests/test_diag_impulse_trace.py -k 'fixed_reset or observation_refresh' -q
  ```

  Expected: failure because the helper and CLI option do not exist.

- [ ] **Step 3: Implement the minimal reset helper and CLI contract**

  Add `--fixed-reset-envelope` to checkpoint mode. It is required for companion records but remains optional for the script's historical diagnostic use. Load through `load_fixed_reset`; restore only after the normal reset; refresh observations exactly once before the first action.

- [ ] **Step 4: Write failing schema/alignment tests**

  Define the required physical arrays and assert:

  ```python
  REQUIRED = {
      "t_s", "head_position_m", "head_velocity_m_s", "contact",
      "nail_position_m", "nail_depth_m", "joint_position_rad",
      "joint_velocity_rad_s", "qfrc_constraint_abs",
      "lambda_joint_n_m_s", "delivered_impulse_n_s", "axial_force_n",
      "action", "control_step", "strike_phase_control",
      "strike_ref_error_control_m",
  }
  ```

  Every physical/substep array must have the same leading length, be finite, and use exact `0.002 s` spacing. Store the control-rate action, phase, and reference error in separate arrays with one row per control step. Link each substep to its control row with `control_step_index`; never describe control observations as 500 Hz measurements.

- [ ] **Step 5: Run schema tests and verify RED**

  Run the whole new test file. Expected: failure because the existing tuple recorder lacks the rich fields and metadata contract.

- [ ] **Step 6: Extend the existing post-accumulator hook minimally**

  Reuse the evaluator's established pre-integration cache contract. Pair stale derived head/contact/force channels with joint velocity and nail depth cached immediately before `sim.step`; also store post-integration qpos/qvel/depth for continuous legality. Keep `orig_substep()` first so the shipped running 50 ms windowed Lambda constraint read includes the just-consumed contact sample. Store the immutable first-strike quality snapshot only when the task has quality instrumentation; zeros in F8/F0/D0 are `not_available`, not measurements. Capture action/phase/reference-error separately at control rate. Do not calculate policy-selection metrics in this script.

- [ ] **Step 7: Add fail-closed NPZ/metadata serialization**

  Write to a staging leaf, validate finite values and exact timing, compute the NPZ SHA-256 plus a canonical metadata digest, then atomically replace the final leaf. Metadata must bind campaign, arm, seed, task, checkpoint hash, reset digest, code/asset revisions, timing, and terminal reason.

- [ ] **Step 8: Run focused tests and existing impulse tests**

  Run:

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
    tests/test_diag_impulse_trace.py \
    tests/test_impulse_bound.py \
    tests/test_reset_state_replay.py -q
  ```

  Expected: all pass.

---

### Task 2: Exact 56-row batch contract and deterministic analysis

**Files:**
- Create: `evaluation/analysis/fixed_reset_500hz_companion.py`
- Create: `tests/test_fixed_reset_substep_companion.py`

**Interfaces:**
- Consumes: the frozen `checkpoint_inventory.tsv`, `/private/tmp/fq56_checkpoints/{fq4x8,fq3x8}`, the approved reset envelope, and Task 1's one-policy leaf.
- Produces: exact 56-leaf validation, `summary.csv`, FQ eligibility/ranking/pairing tables, trajectory/qvel/Lambda figures, and an honest `NO_HARDWARE_LEGAL_QUARTET` result when required.

- [ ] **Step 1: Write failing inventory/path tests**

  Require exactly the existing registered 56 identities and 56 unique hashes. Reconstruct local checkpoint files only from `campaign` plus `checkpoint_cache_path`; reject path traversal, missing files, hash mismatch, task mismatch, duplicate identity/hash, or extra/missing rows.

- [ ] **Step 2: Run inventory tests and verify RED**

  Run:

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
    tests/test_fixed_reset_substep_companion.py -k inventory -q
  ```

- [ ] **Step 3: Implement pure inventory and leaf validators**

  Reuse `validate_inventory`, `expected_task`, and `load_fixed_reset`. Keep checkpoint verification independent of outcome fields so acceptance cannot become post-outcome filtering.

- [ ] **Step 4: Write failing metric tests with synthetic traces**

  Cover exact rising-edge onset, release timing, apex-to-onset tortuosity, chord-deviation RMS/max, terminal re-expansion, axial/lateral onset velocity, maximum absolute arm qvel, per-joint `max(Lambda_j/cap_j)`, and impact-versus-press timing. Include no-contact, one-point, nonfinite, and all-zero-Lambda cases.

- [ ] **Step 5: Implement the pure metric functions**

  Use exact substep onset. Do not substitute the 50 Hz terminal boundary. Treat the unchanged `3.1415 rad/s` rail and manufacturer impulse caps as literal inputs verified against runtime metadata.

- [ ] **Step 6: Write failing eligibility and deterministic-pairing tests**

  Eligibility requires accepted onset, productivity, first-window success, no overflow, and every substep within the qvel rail. Pair only within the confirmed `fq3x8/FQ` incumbent, split eligible policies by curvature, and match on frozen population quality/speed/depth/success covariates. Assert deterministic output and an explicit `NO_HARDWARE_LEGAL_QUARTET` result for fewer than four eligible policies.

- [ ] **Step 7: Implement eligibility, ranking, and pairing**

  Always produce:

  1. simulation-only exact-onset ranking;
  2. hardware-legal ranking;
  3. either two matched lower↔higher-curvature pairs or the explicit no-quartet status.

- [ ] **Step 8: Implement the thin batch driver in the same module**

  For each row, invoke Task 1 with the exact task/checkpoint/reset identity. Reuse only a complete, digest-valid leaf. A failed row remains present with an error status and causes the final 56-row gate to fail; never skip or replace it silently.

- [ ] **Step 9: Add simple figures**

  Generate shared-axis x-z/x-y grids with the dashed observation-only reference, qvel-versus-rail plots, and Lambda/cap plots. Reuse the readable matplotlib style of the existing fixed-reset library; Plotly is not required.

- [ ] **Step 10: Run focused tests**

  Run:

  ```bash
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
    tests/test_diag_impulse_trace.py \
    tests/test_fixed_reset_substep_companion.py \
    tests/test_fixed_reset_video_library.py -q
  ```

---

### Task 3: Smoke, all-56 execution, scientific banking, and review

**Files:**
- Create: `docs/results/2026-07-29_56_policy_500hz_companion.md`
- Create under: `docs/results/assets/2026-07-29_56_policy_500hz_companion/`
- Modify: `docs/results/README.md`

**Interfaces:**
- Consumes: Tasks 1–2 and the immutable 56 checkpoints/reset.
- Produces: reviewed evidence and a tracking-pilot decision, not a new policy.

- [ ] **Step 1: Run one FQ policy CPU smoke**

  Use one confirmed FQ checkpoint. Verify checkpoint/reset/task digests, `physics_dt=0.002`, exactly ten substeps per full control step, nonempty head/qvel/Lambda arrays, and exact agreement between stored onset state and the shipped first-strike tracker.

- [ ] **Step 2: Independently review the smoke**

  Reviewer must inspect code and the actual NPZ/metadata. Stop on off-by-one onset, pre/post-accumulator misalignment, stale observations, fabricated 500 Hz reference channels, or identity drift.

- [ ] **Step 3: Run all 56 sequentially on CPU**

  Print progress and estimated completion time. Reuse only validated leaves. Require all 56 identities at completion.

- [ ] **Step 4: Validate the full artifact**

  Require 56 policies, 56 unique checkpoint hashes, one reset digest, zero missing/invalid/partial leaves, exact timing, and no nonfinite physical arrays. Bank a SHA-256 inventory.

- [ ] **Step 5: Analyze before interpreting**

  Freeze `summary.csv` and both rankings before selecting example trajectories. Report exact qvel-legal count, Lambda/cap distribution, contact-duration/release structure, missed-contact corrections relative to 50 Hz, and the selected quartet or no-quartet status.

- [ ] **Step 6: Write the result note**

  Separate:

  - proven: exact fixed-reset simulator measurements;
  - descriptive: one-reset visual/trajectory phenotypes;
  - open: cross-reset robustness, hardware transfer, and eventual enforcement.

  Do not claim reference use from geometric resemblance.

- [ ] **Step 7: Run final regression and independent code/science reviews**

  Re-run the focused tests plus the impulse/reward pre-training suite even though this pass does not train. One reviewer checks code/artifact integrity; another checks interpretation and quartet logic.

- [ ] **Step 8: Commit only named lightweight source/tests/docs**

  Keep large per-policy NPZ media local if repository conventions require it. Do not push without owner approval.
