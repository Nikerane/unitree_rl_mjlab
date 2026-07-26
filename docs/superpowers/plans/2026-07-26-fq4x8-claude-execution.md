# FQ4x8 Training and Result Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the fixed-impedance four-arm reward experiment, train and
strictly evaluate 32 matched policies on Vega, and bank a defensible result
with two figures and four explanatory videos.

**Architecture:** The policy/reward experiment stays simple; the evaluator is
strict. F8/F0/D0 identify the effects of the raw speed and delivered-impulse
rewards. FQ-min tests whether replacing raw speed with a bounded
contact-quality-gated speed reward improves first contact without sacrificing
useful work. All decisions use verified schema-v3 sampled episodes and paired
training seeds.

**Tech Stack:** Python, PyTorch, mjlab 1.4.0, MuJoCo 3.8.1,
mujoco_warp 3.8.1, pytest, Slurm, Vega CUDA, NumPy/SciPy, Matplotlib.

## Division of Ownership

- **Claude executes Tasks 1–5.** Its terminal deliverable is the complete,
  hash-verified `accepted_training_checkpoints.tsv`,
  `accepted_evaluations.tsv`, and all 16,384 schema-v3 episode records.
- **Codex owns original Task 9 / analysis.** After Claude hands over those
  artifacts, Codex independently verifies them, freezes statistics, produces
  the two figures, selects/renders the four explanatory videos, and writes the
  thesis result.
- Claude must not tune a reward, replace a completed seed, select a
  representative video, or revise a statistical rule after seeing outcomes.
- If Claude chooses to run the Task 6 commands as a mechanical dry run, its
  output is advisory only; Codex recomputes the final result independently.

## Start State

- Repository: `/Users/nikerane/repos/unitree_rl_mjlab`
- Isolated worktree: `/private/tmp/unitree_rl_mjlab-first-strike-quality`
- Branch: `first-strike-quality`
- Required starting HEAD:
  `dae62eb` (`fix(analysis): freeze reward evidence and fallbacks`)
- Worktree must be clean before continuing.
- Detailed design:
  `docs/superpowers/plans/2026-07-26-first-strike-quality-conditioned.md`
- Durable execution ledger:
  `.superpowers/sdd/2026-07-26-first-strike-quality-conditioned/progress.md`
- Completed and reviewed:
  - contact-quality kernel and sensor;
  - F8/F0/D0/FQ task/config foundation;
  - schema-v3 trace/evaluation support;
  - FQ-min reward wiring;
  - fail-closed campaign analysis, statistics, and exactly two figure
    renderers.

Before editing:

```bash
cd /private/tmp/unitree_rl_mjlab-first-strike-quality
git status --short
git branch --show-current
git rev-parse HEAD
```

Expected: clean output, branch `first-strike-quality`, HEAD beginning
`dae62eb`.

## Global Constraints

- Fixed impedance only. Never call `set_gains` or command stiffness.
- Keep arm `Kp/Kd=1000/100`, except joint 2 at `1500/150`; gripper
  `Kp/Kd=100/20`.
- Keep
  `IMP_J_LIMIT=[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]`.
- Keep `imp_max_p=0.0`; the impulse constraint remains log-only.
- Do not add a superlinear excess-over-reference reward.
- Do not edit installed `mjlab`, `rsl_rl`, `mujoco`, or `mujoco_warp`.
- CPU-first. Do not submit training before Tasks 1–3 are green and reviewed.
- Deploy tracked files by commit + push + clean Vega checkout/pull; never
  `scp` tracked source.
- Reject any dirty code/asset state or `-dirty` evaluation row.
- Use one GPU per run.
- Training seeds are exactly `8 9 10 11 12 13 14 15` in every arm.
- Use only `*_sampled` decision fields.
- Any `impossible_success_n>0`, `lambda_dead_n>0`, quality overflow,
  nonfinite geometry, missing quota, hash mismatch, or incomplete manifest
  invalidates the complete campaign analysis.
- Retry only documented infrastructure failures with identical configuration.
  Retain all attempts; never replace a completed weak or unstable seed.
- Stage named files only; never `git add -A`; add no co-author trailer.

## Frozen Treatment Matrix

| Label | Short | Registered task | Impact | Delivered |
|---|---|---|---|---|
| F8 | `f8` | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear` | raw linear, weight 8 | raw linear, weight 2 |
| F0 | `f0` | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0` | raw linear, weight 0 | raw linear, weight 2 |
| D0 | `d0` | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0` | raw linear, weight 8 | disabled, weight 0 |
| FQ-min | `fq` | `Unitree-Z1-Hammer-CaT-Impulse-Event-Quality` | quality-bounded, weight 8 | disabled, weight 0 |

FQ-min pays only:

```text
8 * q_contact * clip(v_precontact / 1.4598331451416016, 0, 1)
```

Raw delivered impulse remains a diagnostic, never an FQ-min reward component
or practical-acceptance gate.

---

### Task 1: Fix the strict evaluator’s stale FQ contract

**Files:**

- Modify: `scripts/eval_impulse.py`
- Modify: `tests/test_eval_impulse_hook.py`

**Problem:** `_validate_sampled_env_contract` currently maps F0 to `(0,2)` and
D0 to `(8,0)`, then defaults every other treatment—including FQ—to `(8,2)`.
The shipped FQ-min task is `(8,0)`, so strict evaluation would reject every FQ
checkpoint before artifact production.

- [ ] **Step 1: Write RED tests**

Add focused tests that load each strict task and call
`_validate_sampled_env_contract`.

Require:

```text
F8 -> 8/2
F0 -> 0/2
D0 -> 8/0
FQ -> 8/0
```

Also mutate FQ delivered weight back to `2.0` and require rejection.

- [ ] **Step 2: Confirm RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_eval_impulse_hook.py -k sampled_env_contract -q
```

Expected before the fix: FQ fails because the evaluator expects `8/2`.

- [ ] **Step 3: Apply the minimal fix**

Replace the defaulting logic with an explicit fail-closed map containing all
four treatments. Do not change rewards, configs, tracker behavior, or schema.

- [ ] **Step 4: Confirm GREEN and review**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_eval_impulse_hook.py tests/test_configs.py \
            tests/test_first_strike_quality_reward.py -q
```

Have an independent reviewer check the exact four-arm table and mutation test.
Commit only the two named files.

---

### Task 2: Complete CPU qualification and freeze preregistration

**Files:**

- Modify: `scripts/smoke_first_strike_instrumentation.py`
- Modify: `tests/test_smoke_first_strike_instrumentation.py`
- Create: `docs/results/2026-07-26_first_strike_quality_prereg.md`

- [ ] **Step 1: Reproduce frozen FQ scale provenance**

Require exact values:

```text
raw bank SHA-256 =
ea4a82e007d95cf7ff962091ba0d6ff3e07e639f4cc1c2d0feab9fe368f47ac8
resolved manifest SHA-256 =
69d5bc66f2127d0a5b29463b35763092d388bc0e71661b06fdd81b0eea8d661f
qualification SHA-256 =
641e520c0cd35932175918d0bf48c7b6df07749ff1462b3d7da7310d76338b1b
NumPy quantile method = higher
q90 = 1.4598331451416016
calibration saturation = 26/256
group-held-out validation saturation = 33/128
```

F8/F0/D0 retain the raw `1.0` speed normalizer. Do not recalibrate from new
policy outcomes.

- [ ] **Step 2: Extend the smoke predicates**

For all four arms require finite observations/rewards, one immutable quality
snapshot, no overflow, productive first event, exact registered task/config
identity, unchanged physical channels, and:

```text
F8: raw speed payout, raw delivered payout
F0: zero speed payout, raw delivered payout
D0: raw speed payout, zero delivered payout
FQ-min: bounded quality-speed payout, zero delivered payout
```

- [ ] **Step 3: Run the full CPU gate**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest -q \
  tests/test_impact_progress_reward.py \
  tests/test_first_strike_event.py \
  tests/test_first_strike_quality_reward.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_cat_soft_hook.py \
  tests/test_eval_impulse_hook.py \
  tests/test_first_strike_campaign.py \
  tests/test_first_strike_quality_campaign.py \
  tests/test_smoke_first_strike_instrumentation.py \
  tests/test_configs.py

PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/validate_rewards.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_contact_sensor.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_reward_setup.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/playback_reference.py
```

Require pytest green, `validate_rewards.py` A–M green, verified contact sensor,
green random-policy sweep, feasible reference playback, reproduced slot-count
evidence, and no quality sentinel.

- [ ] **Step 4: Freeze preregistration before training**

The preregistration must record:

```text
fq -> FQ-min compatibility mapping
task IDs, readers, weights, normalizers, config hashes
seeds 8..15; 500 iterations; 4096 envs; model_499.pt
256 envs x 2 sampled episodes
schema-v3 invalidation and manifest contracts
primary contrasts: F0-F8, D0-F8, FQ-min-D0
MWU as sole three-test Holm family
paired sign-flip as sensitivity only
PCG64 seed 20260726; 100000 matched-seed bootstraps
FQ-min practical margins; delivered impulse excluded
exactly two figures; exactly four post-statistics videos
```

Independently review reward/config/tracker semantics and
statistics/provenance/figure rules. Commit named files only.

---

### Task 3: Make the Vega launchers fail closed for `fq4x8`

**Files:**

- Modify: `scripts/slurm/vega_train.sbatch`
- Modify: `scripts/slurm/vega_eval.sbatch`
- Add focused launcher tests in the repository’s existing Slurm test file(s).

- [ ] **Step 1: Add an explicit training branch**

The `fq4x8` branch must require:

```text
short/task mappings exactly as the frozen matrix
seeds exactly 8..15
ITERS exactly 500
one GPU per run
EXPECTED_CODE_REVISION and EXPECTED_ASSET_REVISION are clean 40-hex
IMPACT_W, DELIVERED_W, NAIL_DRIVEN_W are unset
```

Reject any inherited reward override. Generic argument acceptance is not
sufficient.

- [ ] **Step 2: Add an explicit evaluation branch**

Require the 32-row accepted-training manifest, exact campaign/task/config
mapping, and one external durable root:

```text
EVAL_ROOT=$HOME/unitree_rl_mjlab_eval
ACCEPTED_MANIFEST=$EVAL_ROOT/fq4x8/accepted_training_checkpoints.tsv
```

The script may append `fq4x8/$EVAL_ATTEMPT` exactly once. Do not place the
evaluation root inside the Git repository and do not duplicate `fq4x8` in the
path.

- [ ] **Step 3: Run shell/launcher tests and review**

Test valid four-arm invocations and rejection of wrong task, short, seed,
iteration, revision, manifest row count, duplicate evaluation path, and each
reward override. Independently review the Slurm diff.

---

### Task 4: Deploy cleanly and run Vega smoke/throughput gates

- [ ] **Step 1: Commit and push**

Commit all reviewed Task 1–3 changes. Push branch `first-strike-quality`.
Never deploy uncommitted tracked files.

- [ ] **Step 2: Create a clean Vega checkout**

On Vega, fetch/checkout the exact branch commit. Verify:

```bash
git status --short
git rev-parse HEAD
git -C ../safe_impact_manipulation status --short
git -C ../safe_impact_manipulation rev-parse HEAD
```

Both status outputs must be empty. Record both 40-hex revisions.

- [ ] **Step 3: CUDA smoke each arm at 256 environments**

Require finite observations/rewards, productive first events, no quality
overflow, `impossible_success_n==0`, `lambda_dead_n==0`, exact task/readers/
weights, and the four payout predicates from Task 2.

- [ ] **Step 4: Throughput gate at 4096 environments**

Measure at least 1000 post-warmup control steps:

```text
F8 without quality sensor
F8 with passive quality sensor
FQ-min with passive sensor and quality-speed reader
```

Record steps/s, GPU memory, overflow, and config hashes. FQ-min must be at
least 80% of uninstrumented F8 throughput with zero overflow. If not, stop and
profile; do not submit 32 jobs.

---

### Task 5: Run the matched 4×8 Vega campaign

- [ ] **Step 1: Submit four arrays**

Use campaign `fq4x8`, seeds `8..15`, 500 iterations, one GPU per job, and the
four exact task/short mappings. Before each submission:

```bash
unset IMPACT_W DELIVERED_W NAIL_DRIVEN_W
campaign_code_rev=$(git rev-parse HEAD)
campaign_asset_rev=$(git -C ../safe_impact_manipulation rev-parse HEAD)
```

Submit one `--array=0-7` job per treatment through
`scripts/slurm/vega_train.sbatch`. Do not manually override reward weights.

- [ ] **Step 2: Monitor without performance selection**

Report pending/running/completed/failed/retried counts by arm. Retry only
documented infrastructure failures with identical seed/config. Retain every
attempt.

- [ ] **Step 3: Freeze 32 accepted training rows**

Create `accepted_training_checkpoints.tsv`. Each row binds treatment/short,
task, seed, `model_499.pt` path and SHA-256, attempt history, code/asset/config
identities, fixed action/impedance/cap signatures, clean state, and
disposition. Do not include evaluation fields.

- [ ] **Step 4: Strict sampled evaluation**

For each checkpoint collect the first two completed stochastic episodes from
each of 256 environments using:

```text
reset RNG = 2036072919
observation RNG = 2046072933
action RNG = 2056072941
reset noise = +/-0.05 rad
passive quality instrumentation = on for every arm
```

Use:

```bash
campaign_eval_root="$HOME/unitree_rl_mjlab_eval"
CAMPAIGN=fq4x8 EVAL_ATTEMPT=attempt1 \
  EVAL_ROOT="$campaign_eval_root" \
  ACCEPTED_MANIFEST="$campaign_eval_root/fq4x8/accepted_training_checkpoints.tsv" \
  EXPECTED_CODE_REVISION="$campaign_code_rev" \
  EXPECTED_ASSET_REVISION="$campaign_asset_rev" \
  sbatch \
  --export=ALL,CAMPAIGN,EVAL_ATTEMPT,EVAL_ROOT,ACCEPTED_MANIFEST,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_eval.sbatch
```

Freeze exactly 32 accepted evaluation rows and 16,384 schema-v3 episodes.
Do not analyze until all hashes, quotas, identities, and sentinels verify.

---

### Task 6: Analyze, render, review, and bank the result

**Outputs:**

- `docs/results/2026-07-27_first_strike_quality_result.md`
- `docs/results/assets/2026-07-27_first_strike_quality/summary.csv`
- `docs/results/assets/2026-07-27_first_strike_quality/analysis.json`
- `docs/results/assets/2026-07-27_first_strike_quality/paired_seed_effects.png`
- `docs/results/assets/2026-07-27_first_strike_quality/aggregate_nail_plane_contact_map.png`
- `docs/results/assets/2026-07-27_first_strike_quality/explanatory_videos.tsv`

**Owner:** Codex. Claude stops after Task 5 and hands over the frozen
manifests/artifacts unless explicitly asked to perform an advisory dry run.

- [ ] **Step 1: Freeze statistics before videos**

Run `analyze_quality_campaign` on the complete accepted manifest. Require 32
seed summaries; exactly three MWU tests and one Holm family; paired sign-flip
sensitivities; all eight differences; matched bootstrap intervals/one-sided
ratio bounds; D0 mechanism endpoints; FQ-min practical rule; and all
provenance/safety/quota/liveness predicates.

- [ ] **Step 2: Render and inspect exactly two figures**

Generate only the paired-effects and aggregate nail-plane PNGs. Inspect both
at native resolution and cross-check every plotted quantity against
`analysis.json`.

- [ ] **Step 3: Independent result review**

Reviewer A recomputes statistics from CSV/schema-v3 rows. Reviewer B checks
treatment identities, contact-map geometry, and physical interpretation.
Resolve findings before prose conclusions.

- [ ] **Step 4: Select and render exactly four videos**

After statistics are frozen, choose one medoid episode per arm using
standardized Euclidean distance to that arm’s median over:

```text
contact quality
useful speed
first-contact time
raw delivered impulse
nail-plane axial coordinate
nail-plane lateral coordinate
```

Replace zero/nonfinite scale with `1.0`; break ties by
`(training_seed, env_id, episode_ordinal)`. Freeze identities before replay.
Replay stored action tapes under their bound reset/config and record trace,
action-tape, timing, MP4 hashes, and distances in `explanatory_videos.tsv`.
Large MP4 bytes remain outside Git.

- [ ] **Step 5: Write the result using four questions**

```text
What was supposed to happen?
What actually happened?
Why was there a difference?
What can we learn from this?
```

Separate proven, supported, not distinguishable, simulator-only, and open
claims. A null is not equivalence; qvel exceedance is not hardware-qualified;
log-only impulse is not enforcement.

## Stop Conditions

Stop and report rather than improvising if:

- Task 1–3 tests or independent reviews are not green.
- CPU pretraining gate fails.
- Vega code or asset checkout is dirty/mismatched.
- CUDA smoke has nonfinite values, overflow, impossible success, or dead
  Lambda.
- FQ-min throughput is below 80% of uninstrumented F8.
- Any of 32 training/evaluation identities or 16,384 quotas is incomplete.
- Any proposed retry is performance-motivated rather than infrastructure-only.
- A change would alter caps, enable `imp_max_p`, add variable impedance, or
  add a superlinear reward.

## Handoff Completion Definition

The work is complete only when:

1. 32 accepted `model_499.pt` checkpoints exist.
2. 32 accepted evaluation rows and 16,384 verified schema-v3 episodes exist.
3. `analysis.json`, `summary.csv`, and exactly two reviewed PNGs are frozen.
4. Exactly four representative MP4 identities/hashes are recorded.
5. The result document states what is proven versus still open.
6. All source/result changes have named-file commits and the final checkout is
   clean.
