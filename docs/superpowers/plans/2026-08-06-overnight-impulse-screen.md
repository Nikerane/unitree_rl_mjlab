# Overnight Reference-Guided Impulse Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a minimal, provenance-bound Z1 campaign that keeps straight waypoint guidance and 500 Hz velocity CaT fixed, screens the existing productive first-strike impulse reward without press/recontact farming, confirms a winner on untouched seeds, and produces presentation-ready videos and figures.

**Architecture:** Add five immutable task registrations around the existing `S8/D4` P+V task, giving a balanced `S={0,8} x D={0,4,16}` screen without adding a reward implementation. Extend the existing Slurm guard, reward-digest check, and fixed-reset renderer mappings; then train, evaluate, and summarize from existing 500 Hz traces. Active impulse CaT remains absent unless the unchanged caps pass the binding gate in the approved design.

**Tech Stack:** Python 3.11, mjlab 1.4.0, MuJoCo 3.8.1, MuJoCo-Warp 3.8.1, PyTorch/RSL-RL, pytest, Bash/Slurm, NumPy, Matplotlib, Vega A100 GPUs.

## Global Constraints

- Work only in `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube` locally and a new revision-specific detached checkout on Vega.
- Straight six-waypoint progress reward stays at weight `8.0` in every primary-screen arm.
- Velocity CaT stays `use_vel=True`, `vel_detection="substep"`, `limit=3.1415`, `max_p=0.5`, `min_p=0.0`, `tau=0.95`.
- Impulse CaT stays log-only: `imp_max_p=0.0`; caps stay `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64]`.
- Fixed Cartesian DiffIK action stays `delta_pos_scale=0.15`; no `set_gains`, action clipping, hard velocity termination, curriculum, or VIC.
- Fixed gains stay J1/J3--J6 `Kp=1000, Kd=100`; J2 `Kp=1500, Kd=150`; gripper `Kp=100, Kd=20` and is not policy-controlled.
- Seeds are exactly `2 3 4 5 6 7`, 200 iterations, 4096 environments, one A100 per policy. Completed weak seeds are never replaced.
- Do not use `IMPACT_W`, `DELIVERED_W`, or `NAIL_DRIVEN_W`; treatment weights come from registered task identities.
- Never edit installed mjlab, rsl_rl, MuJoCo, MuJoCo-Warp, or the asset checkout.
- Never `scp` tracked source. Use named Git staging only; preserve all unrelated untracked evidence.
- Keep bulk checkpoints/traces/videos on Vega while local free space is below 8 GiB.
- Primary metric is success-censored productive first-event nail-axis impulse. Episode-cumulative impulse is secondary.
- Active impulse-CaT tasks are not added in this implementation. The unchanged caps must first pass the binding gate in the approved design.

---

## File map

- `src/tasks/hammer/config/z1/__init__.py`: one screen config factory and five new immutable task registrations.
- `scripts/slurm/vega_train.sbatch`: exact `impulse6` campaign allowlist, provenance guards, and correction of the task field printed in `### TRAIN`.
- `tests/test_configs.py`: content-level isolation of all six screen cells.
- `tests/test_env.py`: live construction and 500 Hz velocity-CaT wiring for all new tasks.
- `tests/test_first_strike_campaign.py`: fail-closed launcher matrix and task-log regression.
- `evaluation/guideline/reward_config_digest.py`: prove frozen C0/G/P/P+V identities remain unchanged and emit six screen identities.
- `evaluation/analysis/fixed_reset_video_library.py`: `impulse6` task/arm mappings and truthful treatment titles.
- `tests/test_fixed_reset_video_library.py`: mappings, waypoint-only plot geometry, and title content.
- `docs/results/2026-08-06_impulse6_prereg.md`: frozen matrix, endpoints, anti-farming gates, provenance, and retry rule.
- `evaluation/results/2026-08-06_impulse6/build_impulse_screen_table.py`: result-local aggregation from validated renderer and diagnostic leaves.
- `tests/test_impulse_screen_table.py`: independent synthetic tests for success censoring, press/recontact flags, paired gates, and active-I branch closure.
- `docs/results/2026-08-06_impulse6_result.md`: reviewed result and presentation interpretation.

---

### Task 1: Register the six-cell screen with no new reward code

**Files:**
- Modify: `src/tasks/hammer/config/z1/__init__.py:235-285`
- Modify: `tests/test_configs.py:1250-1450`
- Modify: `tests/test_env.py:520-620`

**Interfaces:**
- Consumes: `z1_hammer_env_cfg(...)`, `z1_hammer_ppo_runner_cfg(cat_soft=True)`, and existing P+V/P+V+D4 registrations.
- Produces: `_impulse_screen_env_cfg(*, play: bool, impact_weight: float, delivered_weight: float)` and five task IDs.

- [ ] **Step 1: Add failing configuration-matrix tests**

Add a parametrized table with these exact task identities and weights:

```python
_IMPULSE6 = {
    "s0d0": ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0", 0.0, 0.0),
    "s0d4": ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4", 0.0, 4.0),
    "s0d16": ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16", 0.0, 16.0),
    "s8d0": ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0", 8.0, 0.0),
    "s8d4": ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4", 8.0, 4.0),
    "s8d16": ("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16", 8.0, 16.0),
}

@pytest.mark.parametrize("short,row", _IMPULSE6.items())
def test_impulse6_cell_changes_only_the_two_declared_weights(short, row):
    task, impact_w, delivered_w = row
    cfg = load_env_cfg(task)
    assert cfg.rewards["impact_progress"].weight == pytest.approx(impact_w)
    assert cfg.rewards["delivered_impulse"].weight == pytest.approx(delivered_w)
    assert cfg.rewards["r_waypoint_progress"].weight == pytest.approx(8.0)
    params = cfg.metrics["cat_soft"].params
    assert (params["use_vel"], params["vel_detection"]) == (True, "substep")
    assert params["imp_max_p"] == 0.0
    assert tuple(params["imp_limit"]) == (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
```

Also normalize every arm onto `s8d4` after replacing only the two reward weights and require canonical equality. Pin action class, scale `0.15`, absence of `set_gains`, actor/critic observation names and widths, reset ranges, fixed actuator gains, and CatPPO.

- [ ] **Step 2: Run the focused tests and observe the intended failure**

Run:

```bash
conda run -n unitree_mjlab pytest -q \
  tests/test_configs.py -k 'impulse6' \
  tests/test_env.py -k 'impulse6'
```

Expected: collection succeeds and task loading fails for the five missing task IDs.

- [ ] **Step 3: Add the minimal factory and registrations**

Insert after `_presentation_i_off_env_cfg`:

```python
def _impulse_screen_env_cfg(
    *,
    play: bool = False,
    impact_weight: float,
    delivered_weight: float,
):
    cfg = z1_hammer_env_cfg(
        play=play,
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        guideline=True,
        progress_reward=True,
        cat_soft=True,
        vel_cat_substep=True,
    )
    cfg.metrics["cat_soft"].params["imp_max_p"] = 0.0
    cfg.rewards["impact_progress"].weight = float(impact_weight)
    cfg.rewards["delivered_impulse"].weight = float(delivered_weight)
    return cfg
```

Register the five missing cells using this factory for train and play configs. Keep the existing P+V+D4 registration as `s8d4`; do not add an alias or change it.

- [ ] **Step 4: Run the focused configuration and live-env tests**

Run:

```bash
conda run -n unitree_mjlab pytest -q \
  tests/test_configs.py -k 'impulse6 or ProgressPlusVelocity' \
  tests/test_env.py -k 'impulse6 or wave3 or presentation3'
```

Expected: all selected tests pass; each new live env has `(B,6)` `substep_peak_qv` before `cat_soft` and actor/critic widths identical to P+V.

- [ ] **Step 5: Review the diff boundary**

Run:

```bash
git diff -- src/tasks/hammer/config/z1/__init__.py tests/test_configs.py tests/test_env.py
rg -n 'set_gains|vel_hard|imp_max_p.*[^0.]' src/tasks/hammer/config/z1/__init__.py
```

Expected: no reward implementation, action, termination, curriculum, or active-impulse change.

---

### Task 2: Add the fail-closed launch contract and preregistration

**Files:**
- Modify: `scripts/slurm/vega_train.sbatch:69-260`
- Modify: `tests/test_first_strike_campaign.py:3990-4100`
- Create: `docs/results/2026-08-06_impulse6_prereg.md`

**Interfaces:**
- Consumes: six task/short pairs from Task 1.
- Produces: `CAMPAIGN=impulse6` launch contract and a frozen 36-row identity matrix.

- [ ] **Step 1: Write launcher RED tests**

Test all six exact pairs with `SEEDS="2 3 4 5 6 7"`, `ITERS=200`, array indices `0..5`, and 40-hex expected revisions. Require rejection of every changed seed list, iteration count, task/short pairing, array index, inherited override, missing revision, dirty revision, and asset mismatch. Add this regression:

```python
def test_impulse6_launcher_logs_the_registered_task_not_array_index(tmp_path):
    result = run_launcher(tmp_path, CAMPAIGN="impulse6", task=_IMPULSE6["s0d0"][0], short="s0d0")
    assert f"### TRAIN task={_IMPULSE6['s0d0'][0]} arm=s0d0 seed=2" in result.stdout
    assert "### TRAIN task=0 " not in result.stdout
```

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
conda run -n unitree_mjlab pytest -q tests/test_first_strike_campaign.py -k impulse6
```

Expected: failures because `impulse6` and the corrected task log do not exist.

- [ ] **Step 3: Implement the exact campaign guard**

Add case-sensitive campaign detection, array bound `0..5`, exact seeds/iterations, unset-override checks, 40-hex revision checks, and the six task/short allowlist. Add `impulse6` to both code/asset revision equality blocks. Correct:

```bash
echo "### TRAIN task=$task arm=$short seed=$seed run=$RUN git=$GITHASH host=$(hostname) $(date '+%F %T')"
```

The final training command must continue to force `--env.metrics.cat-soft.params.imp-max-p 0`; this campaign never activates impulse CaT.

- [ ] **Step 4: Write the preregistration**

Record the 36 rows `(S,D,seed,task,short)`, the design-spec commit `ab35715`, asset revision `b58ccd2f81fd246f27c1e8d88cf86484cd888703`, fixed gains, exact endpoints, success-censored estimand, anti-press rules, D8 branch, active-I binding gate, no-replacement rule, and the fresh confirmation seeds `24..29` subject to an unused-seed audit. State that the exact training revision is the implementation commit and will be pinned by a docs-only follow-up commit before submission.

- [ ] **Step 5: Run launcher tests GREEN**

Run:

```bash
conda run -n unitree_mjlab pytest -q tests/test_first_strike_campaign.py \
  -k 'impulse6 or presentation3'
```

Expected: all selected tests pass, including the historical presentation guard.

---

### Task 3: Bind reward identities and presentation-faithful plots

**Files:**
- Modify: `evaluation/guideline/reward_config_digest.py`
- Modify: `evaluation/analysis/fixed_reset_video_library.py:115-245`
- Modify: `tests/test_fixed_reset_video_library.py:2270-2400`

**Interfaces:**
- Consumes: exact task IDs and screen labels from Task 1.
- Produces: `expected_task("impulse6", arm)`, treatment descriptors, old-digest invariance, and six new canonical digests.

- [ ] **Step 1: Add RED tests for mappings and visual truthfulness**

Require all six mappings, dashed black reference line, six numbered open diamonds, and no gate disks/tube/corridor patch. Require each title to contain `waypoint w=8`, `V-CaT 0.5 @ 500 Hz`, `I-CaT log-only`, and its exact `S`/`D` values. Require the existing frozen P+V+D4 task to resolve as `s8d4` without changing its historical `presentation3` mapping.

- [ ] **Step 2: Run the tests RED**

Run:

```bash
conda run -n unitree_mjlab pytest -q tests/test_fixed_reset_video_library.py -k impulse6
```

Expected: missing campaign/task mappings.

- [ ] **Step 3: Add only mappings and treatment descriptors**

Extend the existing dictionaries; do not change renderer rollout logic, trajectory arithmetic, or frozen-campaign behavior. All screen arms have waypoint geometry enabled and `gate=False`.

- [ ] **Step 4: Extend the reward digest command**

Keep every existing expected digest literal unchanged. For each screen cell, construct the registered config, canonically serialize the reward config, print `short sha256`, and fail if any `(S,D)` pair or any non-treatment field disagrees with the matrix. Explicitly prove `s8d4` is the existing P+V+D4 identity.

- [ ] **Step 5: Run GREEN and frozen-render regressions**

Run:

```bash
conda run -n unitree_mjlab python evaluation/guideline/reward_config_digest.py
conda run -n unitree_mjlab pytest -q tests/test_fixed_reset_video_library.py \
  -k 'impulse6 or presentation3 or frozen_render_path'
```

Expected: old digests `MATCH`; six screen rows print distinct expected treatment identities; frozen render-hash test passes byte-identically.

---

### Task 4: Verify, review, commit, and deploy

**Files:**
- Modify: only files named in Tasks 1--3 and the preregistration.

**Interfaces:**
- Consumes: complete implementation diff.
- Produces: immutable training revision `TRAIN_REV`, docs-only pin revision, and clean Vega checkout.

- [ ] **Step 1: Recover safe local test headroom**

Check for live pytest/renderer processes before touching scratch:

```bash
pgrep -af 'pytest|render_policy|diag_impulse_trace' || true
du -sh /private/var/folders/*/*/*/pytest-of-nikerane 2>/dev/null || true
df -h /System/Volumes/Data
```

Delete only a verified pytest-owned scratch directory when no test owns it. Do not delete evidence, checkpoints, or result trees. Require at least 5 GiB before local targeted tests and 8 GiB before any local full suite; otherwise run the full suite in a Vega CPU allocation.

- [ ] **Step 2: Run scientific and focused gates**

Run:

```bash
conda run -n unitree_mjlab pytest -q \
  tests/test_configs.py tests/test_env.py tests/test_cat_soft_hook.py \
  tests/test_first_strike_campaign.py tests/test_fixed_reset_video_library.py \
  tests/test_impact_progress_reward.py tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py tests/test_delivered_impulse_reward.py
conda run -n unitree_mjlab python scripts/validate_rewards.py
conda run -n unitree_mjlab python scripts/verify_contact_sensor.py
conda run -n unitree_mjlab python scripts/verify_reward_setup.py
```

Expected: zero failures and `validate_rewards.py` phases A--M pass.

- [ ] **Step 3: Run one serial full suite with a stable source fingerprint**

Record `git diff | shasum -a 256` and source mtimes before/after. Run no source-mutating reviewer concurrently.

```bash
conda run -n unitree_mjlab pytest -q > /tmp/impulse6_full_suite.txt 2>&1
tail -20 /tmp/impulse6_full_suite.txt
```

Expected: zero failures. Infrastructure errors caused by disk are invalid results and require a clean rerun after headroom is restored.

- [ ] **Step 4: Obtain two read-only reviews**

One review checks spec/config/launcher conformance; the second executes adversarial probes for fail-open overrides, changed joint identity, missing substep tracker, wrong plot geometry, and stale digest claims. Resolve findings, rerun affected tests, and require written approval with no Critical/Important findings.

- [ ] **Step 5: Commit named files only**

```bash
git add -- \
  src/tasks/hammer/config/z1/__init__.py \
  scripts/slurm/vega_train.sbatch \
  tests/test_configs.py tests/test_env.py tests/test_first_strike_campaign.py \
  evaluation/guideline/reward_config_digest.py \
  evaluation/analysis/fixed_reset_video_library.py \
  tests/test_fixed_reset_video_library.py \
  docs/results/2026-08-06_impulse6_prereg.md
git diff --cached --check
git commit -m "feat: add reference-guided impulse reward screen"
```

Set `TRAIN_REV` to that commit's 40-hex SHA. Patch only the preregistration to record `TRAIN_REV`, commit `docs: pin impulse screen training revision`, and push both commits plus `ab35715` to `origin/cartesian-guideline-fic`.

- [ ] **Step 6: Deploy a clean detached Vega checkout**

Create `~/repos/impulse6_${TRAIN_REV:0:7}/unitree_rl_mjlab` from Git at `TRAIN_REV` and a sibling asset checkout at `b58ccd2f...8703`. Use `/ceph/hpc/home/eunikhilr/repos/presentation3_ba6119c/unitree_rl_mjlab/.venv/bin/python` as the interpreter, but run with `cwd` and `PYTHONPATH` bound to the new checkout. Require both repos 0-dirty and prove imported `src` and `hammer-z1` XML resolve inside the new revision-specific tree.

- [ ] **Step 7: Run CPU/live and CUDA qualification**

Run reward digest and reference qualification in the detached checkout. Submit `scripts/smoke_wave3_pv.py --device cuda --num-envs 64`, then construct all six tasks on CUDA and assert exact weights, `(B,6)` substep peaks, velocity dose, impulse log-only state, caps, observations, fixed action, and finite step output.

Expected: every assertion passes before training submission.

---

### Task 5: Train and bank the primary screen

**Files:**
- Create on Vega: `~/evidence/working/impulse6_train-${TRAIN_REV:0:7}/`
- Update: `docs/results/2026-08-06_impulse6_prereg.md` job ledger only after completion.

**Interfaces:**
- Consumes: clean detached checkout and six task/short pairs.
- Produces: 36 operationally valid `model_199.pt` identities or preserved documented failures.

- [ ] **Step 1: Submit operational canaries without outcome inspection**

For each of the five new task arrays, submit indices `0-1` using:

```bash
env -u IMPACT_W -u DELIVERED_W -u NAIL_DRIVEN_W \
  CAMPAIGN=impulse6 SEEDS="2 3 4 5 6 7" ITERS=200 \
  SINGLE_TASK="$TASK" SINGLE_SHORT="$SHORT" \
  EXPECTED_CODE_REVISION="$TRAIN_REV" \
  EXPECTED_ASSET_REVISION="b58ccd2f81fd246f27c1e8d88cf86484cd888703" \
  sbatch --array=0-1 --export=ALL,CAMPAIGN,SEEDS,ITERS,SINGLE_TASK,SINGLE_SHORT,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_train.sbatch
```

Inspect only task, seed, revisions, 4096 envs, 200 iterations, A100 identity, finite iteration 0, and intended resolved config. Do not inspect reward curves or policy outcomes.

- [ ] **Step 2: Submit the predetermined remainder**

After operational canaries pass, submit array indices `2-5` for all five new cells regardless of policy quality. Reuse the six existing `s8d4` policies only after recording their checkpoint hashes and proving their registered config/digest, training source, seeds, iterations, env count, gains, asset revision, and velocity-CaT identity. If that proof fails, train fresh `s8d4` under `TRAIN_REV`.

- [ ] **Step 3: Verify all 36 identities operationally**

Require `COMPLETED 0:0`, `TRAIN_DONE rc=0`, model 199, five checkpoints/run, no NaN/Inf/error/OOM, and final recorded params matching the matrix. Never replace a completed weak seed. Log any infrastructure retry before resubmitting the identical identity.

- [ ] **Step 4: Bank immutable input hashes**

Copy checkpoint SHA-256, task/seed/short, code/asset revisions, final config YAML, Git records, job ID/state, and Slurm log into the Vega working evidence root. Do not copy bulk run directories to the Mac.

---

### Task 6: Collect fixed-reset 500 Hz evidence and close/open branches

**Files:**
- Create on Vega: `~/evidence/working/impulse6_train-${TRAIN_REV:0:7}/videos/<short>_seed<seed>/`
- Create on Vega: `~/evidence/working/impulse6_train-${TRAIN_REV:0:7}/impulse/<short>_seed<seed>/`

**Interfaces:**
- Consumes: all 36 checkpoint identities and canonical fixed-reset envelope.
- Produces: validated renderer/diagnostic pairs and a binding-gate decision.

- [ ] **Step 1: Run trace-only renderer for every policy**

For each row, invoke:

```bash
python scripts/render_policy.py \
  --checkpoint-file "$CKPT" --campaign impulse6 --arm "$SHORT" \
  --training-seed "$SEED" --checkpoint-sha256 "$CKPT_SHA" \
  --code-revision "$ANALYSIS_REV" --asset-revision "$ASSET_REV" \
  --task "$TASK" --out-dir "$VIDEO_DIR" --steps 80 \
  --substep-trace --trace-only \
  --training-revision "$TRAIN_REV" --analysis-revision "$ANALYSIS_REV"
```

Use CPU, auto-reset-disabled fixed-reset replay and validate every trace before continuing.

- [ ] **Step 2: Run the CPU impulse diagnostic for every policy**

```bash
python scripts/diag_impulse_trace.py \
  --ckpt "$CKPT" --task "$TASK" --device cpu --num-envs 1 --nsteps 80 --play \
  --fixed-reset-envelope "$FIXED_RESET" --campaign impulse6 --arm "$SHORT" \
  --training-seed "$SEED" --checkpoint-sha256 "$CKPT_SHA" \
  --code-revision "$ANALYSIS_REV" --asset-revision "$ASSET_REV" \
  --j-limit 1.64,3.28,1.64,1.64,1.64,1.64 --out "$IMPULSE_DIR"
```

Require hash-bound outputs and bit-identical shared head-position, contact, and post-integration qvel channels between tools.

- [ ] **Step 3: Apply the preregistered branch gates before rendering outcomes**

Compute productive success, gates by contact, qvel legality, continuous-contact duration, contact-onset count, first/cumulative impulse, transverse ratio, event duration/termination, speed, progress, cap utilization, and action saturation. Apply the exact promotion rules from the design.

Active impulse CaT remains closed unless eligible evidence has at least 5% cap crossings, nonzero lower confidence bound, p99 utilization above 1, at least 95% productive success, and at least 70% of binding-joint exposure before first release. If closed, write `ACTIVE_I_NOT_IDENTIFIABLE` with the measured utilization; do not add active-I tasks or lower caps.

---

### Task 7: Build and independently verify the screen result

**Files:**
- Create: `evaluation/results/2026-08-06_impulse6/build_impulse_screen_table.py`
- Create: `tests/test_impulse_screen_table.py`
- Create: `evaluation/results/2026-08-06_impulse6/tables/impulse6_policy_table.csv`
- Create: `evaluation/results/2026-08-06_impulse6/figures/impulse6_dose_response.png`
- Create: `evaluation/results/2026-08-06_impulse6/figures/impulse6_trajectory_grid.png`
- Create: `docs/results/2026-08-06_impulse6_result.md`

**Interfaces:**
- Consumes: validated renderer/diagnostic leaves from Task 6.
- Produces: one row/policy table, arm summaries, decision, figures, and candidate freeze.

- [ ] **Step 1: Write RED tests for the result-local builder**

Use synthetic traces to require:

- failure maps primary impulse to zero while preserving observed impulse separately;
- first-event and cumulative impulse are never interchanged;
- contact lasting 25 substeps at 500 Hz is press-like;
- only a new `False -> True` edge counts as a second onset;
- gates are counted at first contact onset, not full rollout;
- qvel uses all post-integration 500 Hz samples;
- paired seed comparisons reject missing/mismatched identities;
- the active-I gate stays closed when max utilization is below one.

- [ ] **Step 2: Implement the smallest builder**

Import `derive_episode_metrics` from `evaluation.analysis.fixed_reset_500hz_companion`; do not rederive finite-segment geometry. Join each renderer and diagnostic leaf on campaign, arm, seed, checkpoint SHA, reset digest, training revision, and asset revision. Abort on any shared-channel disagreement or unexpected row count.

- [ ] **Step 3: Generate tables and figures**

Write all 36 rows before computing summaries. The dose-response figure shows first-event impulse with individual paired seeds, success/guidance/qvel eligibility, dwell, recontact, and action saturation. The trajectory grid uses the predeclared seed for every arm and the faithful dashed line/open-diamond geometry.

- [ ] **Step 4: Render all policies on Vega**

Rerun `render_policy.py` with `--substep-trace` and without `--trace-only` for all 36 leaves. Retain all videos remotely. Copy back the predeclared matched-seed comparison, trajectory grid, dose-response plot, table, and note; do not outcome-filter the remote library.

- [ ] **Step 5: Independently verify from raw NPZ**

Use a separate arithmetic path that does not import the builder. Rebuild every numeric cell, reproduce all joins/hashes, verify the selection rule, and compare figure-source rows to the CSV. Require written approval with zero Critical/Important findings before freezing a winner.

---

### Task 8: Fresh-seed confirmation, simplification, and deep analysis

**Files:**
- Create: `docs/results/2026-08-06_impulse6_confirmation_prereg.md`
- Extend result-local builder only if the frozen screen rows reproduce byte-for-byte.
- Create: `docs/results/2026-08-07_impulse6_deep_analysis.md`

**Interfaces:**
- Consumes: frozen winner/control decision from Task 7.
- Produces: 12-policy paired confirmation, optional lean-stack result, final presentation package, and evidence-driven next plan.

- [ ] **Step 1: Audit confirmation seeds before launch**

Search all committed manifests, preregistrations, checkpoint inventories, and Vega run names for seeds 24--29. Use them only if they were not used to select or tune the reward. Freeze winner/control task identities before any confirmation outcome is inspected.

- [ ] **Step 2: Train the 12 matched confirmation policies**

Use six identical fresh seeds for winner and incumbent, 200 iterations, 4096 envs, fixed gains, P guidance, V-CaT, and log-only impulse. Apply the same operational checks and no-replacement rule.

- [ ] **Step 3: Confirm or reject the screen result**

Run the identical fixed-reset pipeline. Call the candidate replicated only if it remains eligible and improves success-weighted productive first-event impulse in at least 5/6 paired seeds without crossing anti-press margins.

- [ ] **Step 4: Run the lean-stack ablation only on the frozen winner**

If confirmation succeeds and time remains, add a separately named task that changes only `approach=0` and `nail_driven=0`, with matched seeds and the same qualification. Never fold this two-term removal into the winning treatment identity or call it harmless cleanup.

- [ ] **Step 5: Perform the six-hour deep analysis checkpoint**

Document:

- whether impulse changed and in how many paired seeds;
- whether the mechanism was speed, contact timing, dwell, force, posture/path, or recontact;
- whether downward-action saturation demonstrates an action/controller ceiling;
- whether reference guidance and velocity legality survived;
- why active impulse CaT did or did not become identifiable;
- exact controller gains and the fixed-impedance limitation;
- all negative/null results without outcome filtering.

Then produce the next implementation plan from those findings. If the dose curve is null and at least 5/6 policies have at least 90% pre-contact downward saturation, the next branch may qualify one action-scale/interface change. Otherwise prioritize the confirmed reward or lean-stack result. VIC remains after fixed-impedance evidence is complete.

- [ ] **Step 6: Freeze verified presentation evidence**

Create a recursive SHA-256 inventory, verify forward and reverse completeness, mirror it read-only to a new Vega evidence path, and keep training, analysis, and result-freeze revisions distinct. Copy only presentation-ready MP4s, plots, CSV, result note, and inventory to durable local storage. Do not modify the existing PowerPoint file.

---

## Self-review checklist

- [ ] Every approved-design requirement maps to a task above.
- [ ] No new reward class, evaluator, manifest framework, curriculum, active-I task, tube, or VIC is introduced.
- [ ] The primary screen is balanced `2 x 3`, uses all six seeds, and reports every completed policy.
- [ ] D8, active-I, action scale, lean stack, and confirmation are gated branches rather than silently mixed into the screen.
- [ ] Existing P+V+D4 reuse requires exact identity evidence or triggers fresh training.
- [ ] Plots show dashed line plus open diamonds only and contain treatment/result titles.
- [ ] The CPU/CUDA evaluation limitation and success-censored event window are explicit.
- [ ] Exact gains, caps, qvel limit, seeds, iterations, environments, and provenance rules are recorded.
- [ ] No placeholder text or unresolved task/file/interface name remains.
