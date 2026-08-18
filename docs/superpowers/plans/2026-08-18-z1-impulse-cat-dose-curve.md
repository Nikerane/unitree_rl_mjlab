# Z1 impulse-CaT compact dose-curve implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate exactly two additional seed-2 VIC impulse-CaT doses, `p=0.1` and
`p=0.3`, to form a clean provisional-cap dose curve with the existing `p=0` and `p=0.2` policies.

**Architecture:** Reuse the qualified single-dose training launcher, frozen-policy survey,
four-artifact trace schema, paired RNG streams, and bridge analyzer gates. Add one two-arm training
launcher, bind the two successful final hashes, add one four-arm evaluation launcher and thin
dose-curve analyzer, then bank a bounded result. No RL or evaluator framework is added.

**Tech Stack:** Python 3.10, NumPy, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1, pytest, Bash/Slurm, Vega
NVIDIA A100-SXM4-40GB.

**Spec:** `docs/superpowers/specs/2026-08-18-z1-impulse-cat-dose-curve-design.md`

## Global constraints

- Work only in the linked worktree
  `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/z1-step1-evidence-cleanup`; preserve the dirty
  main worktree.
- Train exactly two new policies: `p=0.1` and `p=0.3`; both use seed `2`, `4096 x 24`, `500`
  iterations, save interval `50`, and provisional caps `[0.82,1.64,0.82,0.82,0.82,0.82]`.
- Change no reward, observation, action, physics, reference, VIC range, domain randomization,
  curriculum, FIC/VIC treatment, or trajectory generation.
- Historical launchers, checkpoints, role mappings, traces, and result bodies remain immutable.
- Every GPU job is one-shot, `--no-requeue`, with no automatic retry or extension.
- The result is simulation-only, native-observed, soft training pressure against project caps—not a
  clamp, manufacturer limit, hardware-safety result, optimal-dose result, or independent-seed proof.

---

### Task 1: Add the guarded two-dose training launcher

**Files:**
- Create: `scripts/slurm/vega_vic_impulse_dose_curve_train.sbatch`
- Create: `tests/test_vic_impulse_dose_curve_train_launcher.py`

**Interfaces:**
- Consumes the qualified guards and checkpoint validator from
  `scripts/slurm/vega_vic_impulse_p02_bridge.sbatch`.
- Produces two immutable final `model_499.pt` checkpoints for Task 3.

- [ ] Write RED shell-contract tests that fail because the launcher is absent. Exercise both array
  arms and require exact role-to-dose mapping (`0 -> 0.1`, `1 -> 0.3`), exact common caps, task,
  seed, rollout shape, iteration budget, resources, no-requeue, fresh leaves, clean provenance,
  exact 11-checkpoint validation, final finiteness, and guarded SHA output.
- [ ] Run the focused tests and confirm expected RED failures.
- [ ] Implement the smallest two-arm array launcher by adapting the qualified p=0.2 launcher.
  Reject non-`0-1` arrays, arguments, resume/distributed variables, dirty repositories, noncanonical
  assets, collisions, wrong runtime/GPU, incomplete checkpoints, nonfinite tensors, and invalid
  hashes.
- [ ] Run GREEN, the historical p=0.2 launcher tests, `bash -n`, and `git diff --check`.
- [ ] Commit and obtain an independent task review before Task 2.

### Task 2: Pass gates and train both doses once

**Files:** No production changes. Record evidence in this plan's SDD task report.

**Interfaces:**
- Consumes the reviewed Task-1 launcher.
- Produces terminal Slurm evidence and exact p=0.1/p=0.3 final checkpoint hashes.

- [ ] Run the AGENTS.md mandatory reward/impulse focused tests, the existing real VIC one-update
  smoke, all relevant historical/new launcher tests, `validate_rewards.py` A-M,
  `verify_contact_sensor.py`, `verify_reward_setup.py`, and `git diff --check` exactly once.
- [ ] Freeze and push the clean reviewed revision, create one fresh detached Vega worktree, verify
  canonical clean assets and exact runtime, and pre-create the Slurm log parent.
- [ ] Run one `sbatch --test-only`; if accepted, submit the identical two-arm array exactly once.
- [ ] Monitor without intervention. Require both arms `COMPLETED`, `0:0`, empty stderr, PASS, exact
  11-checkpoint sets, final `iter == 499`, recursive tensor finiteness, and stable code/asset/hash
  provenance. Record runtime and GPU-hours without inventing monetary cost.
- [ ] Stop on any arm failure; do not evaluate or rescue the other arm automatically.

### Task 3: Bind hashes and add the four-policy frozen evaluation

**Files:**
- Modify: `scripts/impulse_cat_activation_survey.py`
- Modify: `tests/test_impulse_cat_activation_survey.py`
- Create: `scripts/slurm/vega_vic_impulse_dose_curve_eval.sbatch`
- Create: `tests/test_vic_impulse_dose_curve_eval_launcher.py`

**Interfaces:**
- Consumes the exact Task-2 p=0.1/p=0.3 hashes plus banked p=0/p=0.2 hashes.
- Produces four role-bound five-artifact evaluation leaves.

- [ ] Write RED tests for immutable roles `dose_p0_control`, `dose_p01_target`, `dose_p02_target`,
  and `dose_p03_target`, exact hashes, matched RNG streams, and a four-arm launcher whose only
  arm-specific fields are role and checkpoint.
- [ ] Implement role bindings and the guarded `0-3` evaluation array by adapting the qualified
  bridge evaluator. Every arm must run with live `imp_max_p=0` and emit exactly fixed trace, three
  stochastic traces, and `summary.json`, then a five-row manifest.
- [ ] Run old/new survey, comparison, and evaluation-launcher tests; `bash -n`; `git diff --check`;
  and a 2-env x 8-step CPU smoke for all four roles proving shapes, finiteness, zero live impulse
  delta, exact max soft-OR, and matched population/RNG identity.
- [ ] Commit and obtain an independent task review before Task 4.

### Task 4: Run frozen evaluation and bank the dose curve

**Files:**
- Create: `scripts/analyze_vic_impulse_dose_curve.py`
- Create: `tests/test_vic_impulse_dose_curve_analysis.py`
- Create: `docs/results/2026-08-18_z1_impulse_cat_dose_curve.md`
- Create: `docs/results/assets/2026-08-18_z1_impulse_cat_dose_curve/analysis.json`
- Create: `docs/results/assets/2026-08-18_z1_impulse_cat_dose_curve/SHA256SUMS`
- Create: `docs/research/reward-design/Z1_IMPULSE_CAT_DOSE_CURVE_REVIEW_PACKET.md`
- Modify: `docs/README.md`
- Modify: `docs/thesis/README.md` only for bounded thesis-ready claims.
- Modify: `tests/test_docs_current.py`

**Interfaces:**
- Consumes four immutable evaluation leaves and the preregistered bridge analyzer helpers/gates.
- Produces separate per-dose/per-stochastic-population verdicts and a compact thesis record.

- [ ] Write RED analysis tests requiring exact manifests, roles, checkpoint/code/asset identity,
  live log-only evaluation, matched population/RNG identities, fixed-64 descriptive-only output,
  and separate p=0.1/p=0.2/p=0.3 versus p=0 verdicts under the existing bridge gates. Reject any
  pooled rescue or optimal-dose claim.
- [ ] Implement the thin dose analyzer by composing the existing bridge comparison; do not copy
  bootstrap or telemetry logic.
- [ ] Run old/new analyzer tests, deterministic synthetic regeneration, and obtain task review.
- [ ] Push the reviewed revision, create one clean detached Vega evaluator worktree, create logs,
  run one dry-run, then submit the four-arm evaluation once. Monitor without retry and verify exact
  terminal states, empty stderr, manifests, finiteness, and provenance.
- [ ] Run the preregistered analyzer once. Bank exact hashes, separate dose verdicts, impulse and
  velocity risks/tails, utility, durations, censoring, reads, prefix Lambda, gains/actions, and the
  explicit one-seed/non-hardware/non-clamp limitations.
- [ ] Create one sanitized common review packet and obtain concise independent interpretations from
  Kimi K3, GLM-5.2, Qwen 3.6 Plus, DeepSeek V4 Pro, and Claude Opus 5. Their opinions are advisory
  and cannot override numerical gates; do not retry a failure without explicit approval.
- [ ] Run the complete focused suite, docs tests, deterministic analysis regeneration, all manifest
  checks, `git diff --check`, and final whole-branch review. Commit, push, stop, and report; launch
  no curriculum, domain-randomization, contact-flush, torque-CaT, or further dose experiment.
