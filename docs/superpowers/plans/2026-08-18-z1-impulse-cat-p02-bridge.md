# Z1 impulse-CaT `p=0.2` bridge execution plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to
> implement this plan task-by-task. Every behavior change follows RED-GREEN TDD and receives a
> separate task review before the next task starts.

**Goal:** Run one lower-dose, 500-iteration VIC seed-2 bridge at the provisional project impulse
caps, then decide whether it reduces native observed joint-impulse exposure without reproducing the
velocity-risk trade-off seen at `imp_max_p=0.5`.

**Architecture:** Reuse the existing `CatSoftHook`, frozen-policy survey, raw trace schema, matched
RNG streams, bootstrap helpers, and qualified Slurm guards. Add only (1) a pure preflight over the
banked traces, (2) a real log-only cap-identity regression, (3) one single-target training launcher,
and (4) a thin bridge-specific analysis/evaluation wrapper. The historical diagnostic launchers and
banked evidence remain unchanged.

**Tech Stack:** Python 3.10, NumPy, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, pytest, Bash/Slurm, Vega
NVIDIA A100-SXM4-40GB.

**Spec:**
`docs/superpowers/specs/2026-08-18-z1-impulse-cat-thesis-campaign-design.md`

**Execution authorization:** On 2026-08-18, after approving the simplified design, the user
explicitly authorized this exact one-target bridge, its frozen matched evaluation, and the
post-result five-model consultation. That later authorization supersedes the spec header's earlier
"implementation plan only" status; it does not authorize any experiment beyond this plan.

## Global constraints

- Work only in the linked worktree
  `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/z1-step1-evidence-cleanup` on branch
  `z1-step1-evidence-cleanup`; do not touch the dirty main worktree.
- Train exactly one new policy: VIC-TT, seed 2, 4,096 environments, 24 steps, 500 PPO iterations,
  provisional caps `[0.82,1.64,0.82,0.82,0.82,0.82]`, and `imp_max_p=0.2`.
- Keep velocity CaT, rewards, observations, actions, physics, reference, VIC gain range, domain
  randomization, curriculum, and trajectory generation unchanged.
- Reuse the banked seed-2 `p=0` checkpoint only after the exact log-only cap-identity regression
  passes. Do not train another control.
- Frozen evaluation uses live `imp_max_p=0`, fixed 64 worlds, and separate matched stochastic
  populations at seeds `2`, `2026081701`, and `2026081702`, each with 4,096 worlds.
- Provisional caps are project-defined simulation thresholds, not actuator damage limits or
  manufacturer limits. The treatment is graded PPO pressure, not a runtime clamp.
- Native episodes are the primary horizon. Contact/window completion is optional and is not part of
  this execution.
- Do not retry, requeue, extend, lower caps, increase dose, add torque CaT, or launch another seed
  automatically. Any failed job or failed scientific gate ends this plan with a report.
- Preserve every historical launcher, trace, result, and frozen checkpoint mapping.

---

### Task 1: Bank the zero-learning `p=0.2` preflight

**Files:**
- Create: `scripts/preflight_vic_impulse_p02_bridge.py`
- Create: `tests/test_vic_impulse_p02_bridge_preflight.py`
- Create: `docs/results/2026-08-18_z1_impulse_p02_bridge_preflight.md`
- Create: `docs/results/assets/2026-08-18_z1_impulse_p02_bridge_preflight/preflight.json`
- Create: `docs/results/assets/2026-08-18_z1_impulse_p02_bridge_preflight/SHA256SUMS`
- Modify: `docs/README.md`

**Reuse:**
- `scripts.impulse_cat_activation_survey.shadow_impulse_cat`
- `scripts.impulse_cat_activation_survey._native_cap_margins`
- `scripts.impulse_cat_activation_survey.summarize_population`
- `scripts.analyze_vic_impulse_diag90_500_evaluation.validate_leaf_manifest`
- `scripts.analyze_vic_impulse_diag90_500_evaluation._load_trace`
- `scripts.analyze_vic_impulse_diag90_500_evaluation._initial_episode_rho`
- `scripts.analyze_vic_impulse_diag90_500_evaluation._observed_first_contact_prefix_utilization`

- [ ] **Step 1: Write RED tests** for a pure
  `evaluate_preflight(control_leaf, historical_target_leaf, banked_analysis) -> dict` that rejects
  bad manifests, a candidate other than exactly `0.2`, nonfinite pressure, non-max combination,
  missing duration/read/censor descriptors, or an exposure-direction failure. Fixed-64 is expected
  to be nonbinding and is descriptive only. In each stochastic population, require nonzero graded
  pressure and at least one impulse-winning read; complete velocity masking fails.
- [ ] **Step 2: Run the new tests and capture RED** because the preflight module is absent.
- [ ] **Step 3: Implement the thin validator.** It must recompute the `p=0.2` impulse component from
  raw control Lambda in native cap-subtraction dtype, verify it against the stored candidate, and
  emit exact active-read, distinct-positive-dose, winner/mask/tie, read-count, duration, physical
  censoring, native-rho, and observed-first-contact-prefix-rho fields. It must not simulate or learn.
- [ ] **Step 4: Verify GREEN** for the new tests plus:

```bash
python -m pytest -q \
  tests/test_impulse_cat_activation_survey.py \
  tests/test_vic_impulse_diag90_500_analysis.py \
  tests/test_vic_impulse_p02_bridge_preflight.py
```

- [ ] **Step 5: Rehash and read the frozen Vega leaves** under the existing evaluation revision,
  run the preflight once, and bank the JSON and concise Markdown record. Freeze these expected
  control counts: fixed-64 has zero active reads; stochastic seeds `2`, `2026081701`, and
  `2026081702` have respectively `35`, `30`, and `35` active reads, with respectively `35`, `29`,
  and `35` distinct positive doses. Impulse must win respectively `35`, `30`, and `35` reads, with
  zero velocity-masked reads and zero ties in every stochastic population; all must satisfy the
  exact-max invariant.
- [ ] **Step 6: Apply the exposure-direction gate exactly at provisional caps.** For fixed-64 and
  each of the three stochastic populations, compute p0-to-historical-p=.5 target-minus-control
  differences for both p95 and p99 of (a) native initial-episode `rho` and (b) observed
  first-contact-prefix `rho`. All four differences per population must be strictly negative.
  Equality or any positive value fails and stops training. No diagnostic-cap statistic enters this
  gate.
- [ ] **Step 7: Use this exact non-overwriting transport sequence.** First commit the code/tests.
  Then create a unique `mktemp -d` directory under `/private/tmp`, and copy the two complete frozen
  leaves read-only from:

```text
/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-diag90-500/evaluation/
  142098b8af4cbd4b52777012b27f477f0e9aa358/
  b58ccd2f81fd246f27c1e8d88cf86484cd888703/
  41386821_0_diag90_control
  41386821_1_diag90_target
```

  Refuse an existing local destination, verify both remote manifests after transport, run the
  committed local preflight exactly once, and compare the copied seed-2 trace hashes to
  `a2666a6f9cde2f5607f8d7bc8cde696578f3aab65247166d20b6c530359c1547` (control) and
  `62967852edd9e3a3abaa3dbf9de5b77faae1e0cc4854e9b41ead92c010c6b54d` (historical target).
  Do not mutate or clean either Vega leaf.
- [ ] **Step 8: Bank the generated evidence, run `git diff --check` and docs tests, and commit** with
  message `docs(hammer): bank impulse p02 bridge preflight`.

### Task 2: Prove that the reused `p=0` control is cap-vector invariant

**Files:**
- Create: `tests/test_impulse_cat_log_only_cap_identity.py`
- Modify only if the test exposes a real defect: the smallest existing hook or smoke seam.

**Contract:** At `imp_max_p=0`, changing only the impulse cap vector from the historical diagnostic
vector to the provisional vector cannot change hook RNG state, aggregate delta, scaled reward,
soft-done storage, GAE returns, actions, PPO minibatch input, or a deterministic one-update policy
state.

- [ ] **Step 1: Write a RED real-path identity test.** Construct the same two-environment VIC-TT
  one-update CPU run twice from identical seeds and initial weights; mutate only `imp_limit` before
  environment construction. Capture pre-update actions/rewards/storage, RNG state, and post-update
  model/optimizer tensors. Assert exact equality, not approximate equality, while Lambda margins and
  utilization are allowed to differ.
- [ ] **Step 2: Run the isolated test and capture RED** on the missing identity harness.
- [ ] **Step 3: Add only the smallest test seam required** to inject a copied environment config;
  do not change training semantics or registered task defaults.
- [ ] **Step 4: Verify GREEN** for the new identity test, `tests/test_cat_soft_hook.py`,
  `tests/test_cat_ppo_gae.py`, and the existing real VIC one-update smoke test.
- [ ] **Step 5: Commit** with message `test(hammer): prove log-only impulse cap identity`.

### Task 3: Pre-register the bridge evaluator and six-gate analysis

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Modify: `tests/test_impulse_cat_activation_survey.py`
- Modify: `tests/test_impulse_cat_policy_comparison.py`
- Create: `scripts/analyze_vic_impulse_p02_bridge_evaluation.py`
- Create: `tests/test_vic_impulse_p02_bridge_analysis.py`

**Interfaces:**
- Extend `_LiveSurveyRecorder` with finite `policy_action` shape `(steps, envs, 12)`, captured from
  `env.action_manager.action` before auto-reset.
- Parameterize `compare_policy_evaluations(..., expected_roles=...)` while preserving its historical
  diagnostic role pair as the default.
- The bridge analyzer imports the existing manifest, trace, population, and paired-bootstrap
  helpers. It adds only per-environment/per-joint utilization and a paired bootstrap of the ratio of
  first-event delivered-impulse means.

- [ ] **Step 1: Write RED recorder/comparison tests** for the 12D action trace, exact bridge role
  pair, live `p=0`, same evaluator revision, same matched population/RNG identities, and failure on
  any drift. Re-run historical role tests unchanged.
- [ ] **Step 2: Write RED analysis tests** for all strict/inclusive gate boundaries and a synthetic
  pass/fail example. Every stochastic seed gets its own verdict; fixed-64 is descriptive only and
  no pooled verdict is allowed.
- [ ] **Step 3: Implement the recorder and role parameterization** without changing population
  execution, CaT math, or the five-artifact layout.
- [ ] **Step 4: Implement the thin analyzer** with these exact per-population gates:
  - provisional-risk paired upper 97.5% bound `< -0.005`;
  - global provisional `rho` p95 and p99 relative reductions each `>= 0.10`;
  - every joint with control p99 utilization `>= 0.10` has relative p99 increase `< 0.10`, and no
    previously nonviolating joint becomes violating;
  - true-velocity-risk paired upper 97.5% bound `<= 0.001`;
  - success and productive-strike differences each `>= -0.01`;
  - first-event delivered-impulse ratio paired lower 97.5% bound `>= 0.90`;
  - identity, finiteness, and native-observed claim boundaries pass.
- [ ] **Step 5: Freeze bootstrap semantics.** Resample whole paired environment IDs with replacement,
  use exactly 10,000 resamples and seed `2026081703`, use the 97.5th percentile for upper bounds and
  the 2.5th percentile for the delivered-impulse ratio lower bound, and never treat controller reads
  as independent. The delivered ratio is the target mean divided by the control mean within each
  paired resample; a nonpositive observed or resampled control mean makes the gate invalid/failed
  rather than silently dropping that resample.
- [ ] **Step 6: Emit mandatory secondary descriptors:** episode duration, physical-contact
  censoring, controller-read counts, observed-first-contact-prefix Lambda, nail depth, cumulative
  delivered impulse, VIC gains, and all 12 action coordinates. Do not interpret them as complete
  contact or actuator loading.
- [ ] **Step 7: Verify GREEN** for all historical survey/analyzer tests plus the new bridge tests,
  then commit with message `feat(hammer): preregister impulse p02 bridge evaluation`.

### Task 4: Add the one-target Vega training launcher

**Files:**
- Create: `scripts/slurm/vega_vic_impulse_p02_bridge.sbatch`
- Create: `tests/test_vic_impulse_p02_bridge_launcher.py`

**Contract:** One non-array job named `z1-vic-imp-p02-bridge`, one A100, 8 CPUs, 40 GB, 90-minute
limit, `--no-requeue`, and no positional arguments. Reject any Slurm-array, distributed, resume, or
optimization environment variable.

- [ ] **Step 1: Write the launcher test and capture RED** because the file is absent.
- [ ] **Step 2: Reuse the qualified diagnostic launcher guards** but set exactly:

```text
task = Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT
seed = 2
num_steps_per_env = 24
num_envs = 4096
max_iterations = 500
save_interval = 50
imp_limit = [0.82,1.64,0.82,0.82,0.82,0.82]
imp_max_p = 0.2
```

- [ ] **Step 3: Preserve clean code/assets, canonical sibling, exact runtime/A100, collision,
  checkpoint-set/finiteness, postflight revision, and SHA-256 guards.** Require exactly
  `model_{0,50,100,150,200,250,300,350,400,450,499}.pt`.
- [ ] **Step 4: Verify GREEN**, `bash -n`, both old and new launcher tests, and `git diff --check`.
- [ ] **Step 5: Commit** with message `feat(hammer): launch single impulse p02 bridge`.

### Task 5: Pass all gates and run the one training job

**Files:** No production changes. Append exact evidence to this plan's SDD task report.

- [ ] **Step 1: Run the mandatory focused suite:**

```bash
python -m pytest -q \
  tests/test_impact_progress_reward.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_cat_soft_hook.py \
  tests/test_cat_ppo_gae.py \
  tests/test_impulse_cat_log_only_cap_identity.py \
  tests/test_impulse_cat_activation_survey.py \
  tests/test_impulse_cat_policy_comparison.py \
  tests/test_vic_impulse_p02_bridge_preflight.py \
  tests/test_vic_impulse_diag90_500_analysis.py \
  tests/test_vic_impulse_p02_bridge_analysis.py \
  tests/test_vic_impulse_diag90_500_launcher.py \
  tests/test_vic_impulse_diag90_500_eval_launcher.py \
  tests/test_vic_impulse_p02_bridge_launcher.py
```

Also run the named real update smoke explicitly:

```bash
python -m pytest -q \
  tests/test_smoke_joint_position_fixed.py::test_vic_catppo_smoke_runs_one_real_update
```

- [ ] **Step 2: Run each mandatory project gate exactly once:**

```bash
python docs/research/reward-design/validate_rewards.py
python docs/research/reward-design/verify_contact_sensor.py
python docs/research/reward-design/verify_reward_setup.py
```

All reward phases A-M, the contact sensor, and the random-policy reward setup must pass. An unknown
or failed result stops submission; do not substitute or retry automatically.

- [ ] **Step 3: Run `git diff --check`, obtain a clean reviewed commit, and push the exact branch.**
  Bind `CODE_REVISION` to the validated 40-hex commit and `RUN_ROOT` to the exact Vega path
  `/ceph/hpc/home/eunikhilr/repos/unitree_rl_mjlab-z1-p02-${CODE_REVISION:0:12}`. Refuse if that path
  already exists; create it only as a fresh detached Git worktree. Never use the dirty remote base
  checkout, overwrite another worktree, remove an existing path, or clean user data.
- [ ] **Step 4: Verify the clean detached code revision and canonical asset revision
  `b58ccd2f81fd246f27c1e8d88cf86484cd888703`; create the absolute Slurm log parent before
  submission.**
- [ ] **Step 5: Run `sbatch --test-only` once.** If accepted, submit the identical command once with
  `sbatch --parsable`, explicit `RUN_ROOT`, `ASSET_REPO`, `EXPECTED_CODE_REVISION`, and
  `EXPECTED_ASSET_REVISION` exports.
- [ ] **Step 6: Monitor to terminal state** using `squeue`, `sacct`, stdout/stderr sizes, and expected
  checkpoint milestones. Do not intervene. Expected runtime is about 22 minutes; allow normal queue
  delay and the fixed 90-minute scheduler limit. Record A100 GPU-hours; do not state a currency cost
  because Vega exposes no monetary tariff.
- [ ] **Step 7: On success, verify `COMPLETED`, exit `0:0`, empty stderr, PASS marker, exact checkpoint
  set, final `iter == 499`, finiteness, and the final checkpoint SHA-256.** Any failure stops here.

### Task 6: Bind the target hash and run frozen matched evaluation

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Modify: `tests/test_impulse_cat_activation_survey.py`
- Create: `scripts/slurm/vega_vic_impulse_p02_bridge_eval.sbatch`
- Create: `tests/test_vic_impulse_p02_bridge_eval_launcher.py`

- [ ] **Step 1: Add RED tests** for immutable roles `bridge_p0_control` and
  `bridge_p02_target`, exact role/checkpoint hash mapping, two-arm evaluation array, fresh output
  leaves, five-artifact manifest, pre/post checkpoint hashes, clean code/assets, one A100 per arm,
  20-minute limit, and `--no-requeue`.
- [ ] **Step 2: Bind the exact successful `model_499.pt` SHA-256** to
  `bridge_p02_target`; bind `bridge_p0_control` to the already verified banked checkpoint SHA
  `f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3`. Do not use the historical
  p=.5 target as a causal arm.
- [ ] **Step 3: Implement the bridge launcher** by reusing the qualified evaluation guard pattern.
  Each role runs `impulse_cat_activation_survey.py` once with live `imp_max_p=0`; output exactly
  `fixed_trace.npz`, three named stochastic traces, and `summary.json`, then hash all five.
- [ ] **Step 4: Verify GREEN** for old/new survey and launcher tests, `bash -n`, and a 2-env x 8-step
  controller smoke per role with exact shapes, finiteness, zero live impulse delta, exact max soft-OR,
  and identical population/RNG identities.
- [ ] **Step 5: Commit, review, push, create a new clean detached Vega evaluator worktree, create the
  log parent, and run `sbatch --test-only` once.** Submit the two-arm array once only if accepted.
- [ ] **Step 6: Monitor both arms without retry.** Verify `COMPLETED`, `0:0`, empty stderr, role/hash
  identity, exact artifact sets, manifests, and finiteness.
- [ ] **Step 7: Run the pre-registered bridge analyzer.** A bridge PASS requires every gate to pass
  separately in each stochastic population. Fixed-64 and pooled descriptive values cannot rescue a
  failed population.

### Task 7: Bank the verdict, consult five models, and stop

**Files:**
- Create: `docs/results/2026-08-18_z1_impulse_p02_bridge.md`
- Create: `docs/results/assets/2026-08-18_z1_impulse_p02_bridge/analysis.json`
- Create: `docs/results/assets/2026-08-18_z1_impulse_p02_bridge/SHA256SUMS`
- Create: `docs/research/reward-design/Z1_IMPULSE_CAT_P02_BRIDGE_CROSS_MODEL_PACKET.md`
- Create: `docs/research/reward-design/Z1_IMPULSE_CAT_P02_BRIDGE_CROSS_MODEL_REVIEWS.md`
- Modify: `docs/README.md`
- Modify: `docs/thesis/README.md` only if the result is thesis-ready.
- Modify: `tests/test_docs_current.py`

- [ ] **Step 1: Bank a compact result** with exact code/asset/checkpoint/job/artifact hashes; separate
  per-population gates; impulse and velocity risk/tails; utility; duration; censoring; reads;
  observed-prefix Lambda; gains; actions; and explicit non-hardware/non-clamp/native-horizon claims.
- [ ] **Step 2: Write a sanitized result packet** containing only the research question, fixed
  protocol, preregistered gates, sufficient statistics, limitations, and the decision requested.
  Hash the exact packet before external review; do not send code, raw traces, checkpoint bytes,
  credentials, private paths, or unpublished personal information.
- [ ] **Step 3: Consult exactly these OpenCode Zen models independently** in empty temporary
  directories with `opencode run --pure`, no auto-approval, max variants, and the exact same packet:
  `opencode/kimi-k3`, `opencode/glm-5.2`, `opencode/qwen3.6-plus`,
  `opencode/deepseek-v4-pro`, and `opencode/claude-opus-5`. Attempt each model exactly once; record
  any failure and continue the evidence report without automatically retrying it.
- [ ] **Step 4: Bank verbatim response hashes, session IDs, provider-reported cost fields with their
  unit caveat, and an adjudication.** External votes do not override the preregistered numerical
  gates.
- [ ] **Step 5: Run docs tests, all focused tests, deterministic analysis regeneration, SHA checks,
  `git diff --check`, and a final whole-branch standards/spec review. Commit the evidence.**
- [ ] **Step 6: Stop and report.** Do not start multi-seed confirmation, another dose, contact flush,
  torque CaT, or any rescue experiment without a new explicit user approval.

## Plan self-review

- Every approved simplified-plan requirement maps to a task above.
- The sole new learning treatment is the one `p=0.2` target.
- The existing-trace gate precedes all training; the exact log-only identity test precedes control
  reuse.
- All analysis gates and their strict/inclusive boundaries are frozen before training.
- The future target hash is not guessed or left as a mutable placeholder: it is bound once from the
  successfully validated checkpoint before evaluation.
- Historical launchers, role hashes, evidence, and result bodies remain immutable.
- Contact completion is explicitly optional and excluded.
- Every job has one-shot, no-requeue, no-auto-retry behavior and an explicit hard stop.
