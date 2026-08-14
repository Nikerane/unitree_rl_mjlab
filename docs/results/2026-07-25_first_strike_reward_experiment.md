# First-strike reward experiment preregistration

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer-derived cap”
> wording and `[1.64, 3.28, ...]` below record the historical registered-task boundary,
> not a validated Z1 reaction-impulse or damage limit. The numeric preregistration remains
> frozen; see `../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

**Date:** 2026-07-25
**Scope:** Unitree Z1, fixed impedance, log-only impulse characterization
**Status:** Local implementation and CPU integration verified; no CUDA
qualification smoke, training, or evaluation has been run under this
preregistration

## Prospective amendment A1 — joint-speed interpretation (2026-07-26)

**Timing and motivation.** This amendment is frozen before any CUDA smoke,
C/D-prime/F/E training, or 4×8 evaluation outcome. It follows a
post-qualification analysis of the already-banked, treatment-shared
384-episode legacy replay, which found that 378/384 episodes contained at
least one finite 500 Hz arm-joint-speed sample above 3.1415 rad/s,
predominantly before first contact. It is therefore a **data-informed,
prospective amendment**, not part of an untouched or blind 2026-07-25
preregistration. No 4×8 arm, seed, checkpoint, primary outcome, or treatment
comparison has been inspected or selected.

**Primary estimand.** The sole primary E-versus-D-prime estimand remains the
intention-to-treat effect of reward semantics in the frozen fixed-impedance
simulator and action configuration. A finite `|qdot| > 3.1415 rad/s`
observation does not remove an episode, seed, checkpoint, arm, or campaign
from that estimand and does not authorize retraining, replacement, or
selective exclusion. Nonfinite qvel remains a numerical/instrumentation
invalidation.

**Hardware-transport endpoint.** For every episode, report separately: any
finite exceedance, maximum absolute arm-joint speed, maximum excess above
3.1415 rad/s, samples and time above the rail, offending joint, and phase
(`precontact`, `first-event`, or `post-event`). Aggregate within checkpoint
first and report the eight training-seed summaries per arm. These endpoints
are descriptive external-validity diagnostics and cannot rescue a failed
primary.

**Interpretation rule.** If all non-qvel provenance, identity, quota,
liveness, and sentinel gates pass, the E-versus-D-prime primary may be
reported as a fixed-impedance **simulation reward comparison**. It may
additionally be labelled **sampled hardware-speed-qualified** only if both E
and D-prime have zero finite exceedances in both recorded 500 Hz phases.
Otherwise the report must state that hardware feasibility and deployment are
not established and that any reward advantage may rely on dynamics
unavailable under the manufacturer speed envelope. Apply the same
contrast-specific label to mechanism comparisons involving F. C cannot
invalidate the E-versus-D-prime primary.

**Safety scope.** Because `imp_max_p=0` and joint speed is measured rather
than physically enforced, no positive reward result establishes hard velocity
safety, impulse-constraint enforcement, or real-Z1 deployability.

## Question and design

Does the success-censored, 500 Hz first-strike event reward with saturated
delivered-impulse shaping (**E**) improve useful first-strike speed over the
equal-weight, first-event-censored legacy-readout comparator (**D-prime**)?

The experiment is one fixed **4 arms × 8 independent training seeds = 32
jobs** screen. It does not change the task physics, fixed actuator gains,
action scale, manufacturer impulse caps, or the two maximize weights. The
per-joint impulse constraint remains log-only (`imp_max_p=0`); this is not an
enforcement or variable-impedance experiment.

| Arm | Exact registered task | Frozen reward package | Seeds |
|---|---|---|---|
| C | `Unitree-Z1-Hammer-CaT-Impulse` | 50 Hz legacy readers with repeated credit | 0–7 |
| D-prime | `Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy` | 50 Hz legacy readers, first-event-censored, paid once | 0–7 |
| F | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear` | 500 Hz first-event readers, linear delivered payout | 0–7 |
| E | `Unitree-Z1-Hammer-CaT-Impulse-Event` | 500 Hz first-event readers, delivered payout saturated at the qualified event reference | 0–7 |

All 32 rows use:

- `impact_progress.weight=8` and `delivered_impulse.weight=2`;
- fixed `BuiltinPositionActuator` gains: arm joints except joint 2 at
  `Kp/Kd=1000/100`, joint 2 at `1500/150`, and gripper at `100/20`;
- unchanged manufacturer-derived `IMP_J_LIMIT =
  [1.64, 3.28, 1.64, 1.64, 1.64, 1.64] N·m·s`;
- `imp_max_p=0`, with no `set_gains` or commanded stiffness;
- 4096 training environments and exactly 500 PPO iterations;
- final checkpoint `model_499.pt`;
- deterministic run names `fsr4x8_{c|dprime|f|e}_seed{0..7}`.

The qualified event reference is the repeat-derived value banked in
`2026-07-25_first_strike_reward_qualification.md`; no conclusion here relies
on an older provisional source-code comment.

The executable local matrix is
`evaluation.analysis.first_strike_campaign.FROZEN_CAMPAIGN_MATRIX`. Its strict
validator and tests bind every row to the registered task configuration and
the launcher arguments.

## Evaluator contract

The sampled evaluator is primary. The old mean-action path is diagnostic only.
Each checkpoint is evaluated in the exact registered task that trained it,
using:

- 256 environments;
- exactly the first two completed episodes from every environment, for 512
  episodes total; later completions from faster environments are discarded;
- stochastic policy actions;
- the training configuration, not play mode: reset noise
  `[-0.05, +0.05] rad`, actor observation corruption enabled, critic
  corruption disabled;
- frozen unseen evaluator base RNG seed `2026072900`, yielding distinct
  reset/observation/action streams `2036072919`, `2046072933`, and
  `2056072941`;
- a **4.0 s per-episode first-strike failure cutoff**, equal to 200 control
  steps at 50 Hz, applied identically to all four arms; successful strikes
  typically complete in 8–15 control steps;
- a separate diagnostic mean rollout of 400 control steps;
- pre-autoreset tracker snapshots and provenance-bound raw 500 Hz traces.

The four disjoint statuses—success-finalized, window-finalized, no-contact,
and contacted-but-unfinished—must sum to 512. Never relabel an unfinished
event as window-finalized. The primary useful-speed outcome and both success
rates use all 512 episodes. No-contact and unfinished episodes contribute zero
useful speed. Contact velocity and transverse error are contact-conditioned
and report their denominator. Terminal contraction and late re-expansion are
complete-window-conditioned and report their denominator. Within-nail-radius
rate uses all 512 episodes with no-contact counted false.

`overall_success_rate_sampled` means **success by the 4.0 s cutoff**. This
campaign does not estimate recovery after 4.0 s and does not estimate success
over the full 20 s training horizon. The cutoff is an evaluation failure
boundary, not a claimed physical contact duration.

## Sole primary decision

For sampled episode \(i\),

\[
Y_i = S_{\mathrm{first},i}\,v_{\mathrm{precontact},i},
\]

where `S_first=1` only if normal task success occurs within the first event
window. Otherwise \(Y_i=0\). Average the 512 episodes within a trained
checkpoint first. The eight training-seed means per arm are the inferential
units. The CSV field is:

```text
first_strike_useful_speed_mean_sampled
```

The sole primary contrast is **E versus D-prime**, oriented as E minus
D-prime. It passes only if all of the following hold:

1. both arms form a complete valid eight-seed set;
2. a two-sided exact seed-level Mann–Whitney U test gives `p < 0.05`;
3. a two-sided exact seed-label permutation test of the difference in means
   gives `p < 0.05`;
4. `(mean_E - mean_Dprime) / mean_Dprime >= 0.10`;
5. E first-window sampled success is at least 90% and no more than five
   percentage points below D-prime;
6. E sampled success by the 4.0 s cutoff
   (`overall_success_rate_sampled`) is at least 90% and no more than five
   percentage points below D-prime;
7. every provenance, identity, quota, liveness, sentinel, and numerical
   finiteness gate below passes.

If the D-prime mean is non-positive, the relative-effect test is invalid and
the primary cannot pass.

Both exact tests enumerate all
\(\binom{16}{8}=12{,}870\) assignments of the pooled seed outcomes. The
Mann–Whitney calculation uses midranks so ties and zero outcomes remain in the
exact enumeration. The mean-difference permutation uses the same assignments.
Report:

- per-arm seed mean, sample standard deviation, minimum, and maximum;
- `U` and tie-aware `A12=U/64`, oriented as E over D-prime;
- both exact two-sided p-values;
- absolute and relative mean effects;
- a separately labelled 95% interval from 100,000 within-arm seed bootstrap
  resamples with frozen RNG seed `2026072504`.

The bootstrap interval is descriptive seed-level uncertainty, not exact
permutation inference. The word *noninferior* is allowed only if the one-sided
95% lower seed-bootstrap bounds for both E-minus-D-prime success differences
exceed `-0.05`; this is an intersection-union labeling gate, separate from the
point-estimate primary guardrails.

Eight seeds can screen for a large treatment separation. Failure to reject is
reported as “not distinguishable at n=8,” never equivalence.

## Gated mechanism and descriptive comparisons

Mechanism inference is forbidden unless the E-versus-D-prime primary passes.
If it does:

1. **F versus D-prime**, hypothesized `F > D-prime`, tests the 500 Hz event
   measurement package without saturation. F must pass the same success
   guardrails before interpretation.
2. **E versus F**, hypothesized `E > F`, tests saturation shape with otherwise
   identical event semantics.

For each mechanism contrast, compute the same exact MWU and exact
mean-difference permutation p-values, set
`p_joint=max(p_MWU,p_permutation)`, require the preregistered direction, then
apply Holm correction across the two `p_joint` values.

**C versus D-prime is descriptive only.** It changes repeated credit,
censoring, and payout timing as a package and is not described as a pure latch
effect.

Conditional successful speed, separate success/window impulse strata,
recontact, tail fraction, payout component/timing, saturation, and the
preregistered geometry diagnostics are secondary mechanism evidence. They
cannot rescue a failed primary or independently authorize a trajectory reward.
All new figures are Plotly.

## Invalidation and stop gates

The comparison is invalid, and launch or analysis stops, if any of the
following occurs. Amendment A1 prospectively removes **finite** qvel
exceedance from this list while retaining it as a mandatory hardware-transport
endpoint:

- code or sibling Z1 asset repository revision is unknown or dirty;
- a checkpoint is evaluated in a task other than its frozen arm task;
- weights differ from 8/2, gains are not fixed, impulse caps differ, or
  `imp_max_p != 0`;
- the 4×8 matrix is incomplete, has duplicate arm/seed identities, or a
  completed seed was selectively replaced;
- the sampled quota is not exactly 256×2 or the four statuses do not exhaust
  512;
- `impossible_success_n > 0` or `lambda_dead_n > 0`;
- any nonfinite qvel sample appears in either recorded 500 Hz joint-speed
  phase;
- checkpoint, accepted-manifest, raw-trace, nail-asset, campaign-config, or
  treatment-config hashes fail verification;
- the action/reset/observation RNG identities differ across comparison rows.

No partial-matrix analysis is permitted.

## Retry and attempt policy

Retry only a documented infrastructure failure, with the identical
arm/seed/task/configuration. Retain every Slurm log and partial run directory
and give the retry an explicit attempt record. A completed poor, unstable, or
scientifically inconvenient seed is data and cannot be replaced. Retrying
does not authorize deleting or overwriting the earlier attempt.

Checkpoint selection is controlled only by an **external accepted-attempt TSV
manifest**. Evaluation must never select a checkpoint by `latest`, filesystem
ordering, or a glob. The manifest has exactly 32 data rows—one accepted
attempt for every registered arm/training-seed identity—and this exact header:

```text
arm	training_seed	task	run_name	attempt_path	checkpoint_path	checkpoint_sha256	training_code_revision	training_asset_revision	disposition	reason
```

The tracked template is
`docs/results/assets/2026-07-25_first_strike_reward/accepted_attempts.template.tsv`.
Before evaluation, copy it to a campaign-state directory outside both Git
repositories, then fill explicit canonical attempt/checkpoint paths, the
checkpoint SHA-256, full training code and asset revisions,
`disposition=accepted`, and a truthful reason. All path, hash, revision,
disposition, and reason cells remain **PENDING** until training. Zero or more
than one accepted row for an arm/seed blocks evaluation. Rows with PENDING
fields block evaluation. Every attempt directory remains retained even though
only the preregistered accepted row enters this manifest.

Evaluation outputs are attempt-specific under
`$EVAL_ROOT/fsr4x8/<EVAL_ATTEMPT>/`. `$EVAL_ROOT` must be outside both the code
and sibling asset repositories, and an existing attempt directory is never
overwritten. The evaluator copies the accepted manifest into that output,
records its SHA-256, and binds every result row to that digest. The evaluation
code and asset revisions must exactly equal the common training revisions in
all 32 accepted rows.

## Prepared deployment manifest — not executed

All values labelled **PENDING** below remain genuinely unknown until their
named gate runs. No checkpoint path or hash is invented in advance.

### 1. Local CPU gate

Required before any external deployment:

```bash
PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
  tests/test_impact_progress_reward.py \
  tests/test_first_strike_event.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_cat_soft_hook.py \
  tests/test_first_strike_probe.py \
  tests/test_first_strike_campaign.py

PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/validate_rewards.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_contact_sensor.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_reward_setup.py
```

Required outcome: pytest green; validation phases A–M green; contact sensor
verified; random-policy reward sweep green.

### 2. Clean provenance boundary

This section requires explicit user authorization because no commit or push is
part of the present local task.

| Identity | Value |
|---|---|
| local code full revision | **PENDING clean named commit** |
| Vega code full revision | **PENDING fast-forward pull** |
| local/Vega code status | **PENDING; must be empty** |
| sibling asset full revision | **PENDING capture** |
| sibling asset status | **PENDING; must be empty** |
| accepted-attempt manifest SHA-256 | **PENDING post-training capture** |

Deployment must be named-file commit → push → `ssh vega git pull --ff-only`.
Never copy tracked files with `scp`. From the clean Vega code-repository root,
freeze the revisions and keep generated caches and campaign state outside
both repositories:

```bash
REPO_ROOT="$(pwd -P)"
ASSET_REPO="$(cd "$REPO_ROOT/../safe_impact_manipulation" && pwd -P)"
EXPECTED_CODE_REVISION="$(git rev-parse HEAD)"
EXPECTED_ASSET_REVISION="$(git -C "$ASSET_REPO" rev-parse HEAD)"
WARP_CACHE_PATH="$HOME/.cache/unitree_rl_mjlab/warp"
CAMPAIGN_STATE_ROOT="$HOME/unitree_rl_mjlab_campaign_state/2026-07-25_first_strike_reward"
EVAL_ROOT="$HOME/unitree_rl_mjlab_eval"
ACCEPTED_MANIFEST="$CAMPAIGN_STATE_ROOT/accepted_attempts.tsv"
export ASSET_REPO EXPECTED_CODE_REVISION EXPECTED_ASSET_REVISION
export WARP_CACHE_PATH CAMPAIGN_STATE_ROOT EVAL_ROOT ACCEPTED_MANIFEST
mkdir -p "$WARP_CACHE_PATH" "$CAMPAIGN_STATE_ROOT" "$EVAL_ROOT"
```

`ASSET_REPO` must resolve to that canonical sibling; another clean asset
checkout is not interchangeable. `WARP_CACHE_PATH`, `CAMPAIGN_STATE_ROOT`, and
`EVAL_ROOT` are external so launcher/evaluator I/O cannot dirty either source
repository.

### 3. Four-task CUDA instrumentation smoke

**OPEN PRELAUNCH GATE — local CPU integration passed; four clean 256-env CUDA
qualification JSONs are still required before any 4×8 submission.**

The reviewed task-aware entry point is
`scripts/smoke_first_strike_instrumentation.py`. It accepts exactly one
registered task per process and has no dirty-provenance bypass. Each invocation
must run from the clean revisions frozen in section 2, with exactly one GPU
visible, and write to a fresh external path:

```bash
SMOKE_ATTEMPT="$CAMPAIGN_STATE_ROOT/cuda_smoke_attempt1"
mkdir "$SMOKE_ATTEMPT"

PYTHONPATH=. .venv/bin/python scripts/smoke_first_strike_instrumentation.py \
  --task Unitree-Z1-Hammer-CaT-Impulse \
  --device cuda:0 \
  --num-envs 256 \
  --expected-code-revision "$EXPECTED_CODE_REVISION" \
  --expected-asset-revision "$EXPECTED_ASSET_REVISION" \
  --asset-repo "$ASSET_REPO" \
  --out "$SMOKE_ATTEMPT/c.json"

PYTHONPATH=. .venv/bin/python scripts/smoke_first_strike_instrumentation.py \
  --task Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy \
  --device cuda:0 \
  --num-envs 256 \
  --expected-code-revision "$EXPECTED_CODE_REVISION" \
  --expected-asset-revision "$EXPECTED_ASSET_REVISION" \
  --asset-repo "$ASSET_REPO" \
  --out "$SMOKE_ATTEMPT/dprime.json"

PYTHONPATH=. .venv/bin/python scripts/smoke_first_strike_instrumentation.py \
  --task Unitree-Z1-Hammer-CaT-Impulse-Event-Linear \
  --device cuda:0 \
  --num-envs 256 \
  --expected-code-revision "$EXPECTED_CODE_REVISION" \
  --expected-asset-revision "$EXPECTED_ASSET_REVISION" \
  --asset-repo "$ASSET_REPO" \
  --out "$SMOKE_ATTEMPT/f.json"

PYTHONPATH=. .venv/bin/python scripts/smoke_first_strike_instrumentation.py \
  --task Unitree-Z1-Hammer-CaT-Impulse-Event \
  --device cuda:0 \
  --num-envs 256 \
  --expected-code-revision "$EXPECTED_CODE_REVISION" \
  --expected-asset-revision "$EXPECTED_ASSET_REVISION" \
  --asset-repo "$ASSET_REPO" \
  --out "$SMOKE_ATTEMPT/e.json"
```

`mkdir` is intentionally non-idempotent and the writer refuses to overwrite
any output leaf. A retry therefore requires a new attempt directory while the
earlier attempt remains retained. Before any 4×8 submission, all four commands
must exit zero and each banked JSON must report
`cuda_qualification_pass=true`. The smoke verifies:

- one GPU per process and the assigned CUDA device;
- finite observations and rewards;
- exact task id, weights 8/2, fixed gains, caps, and `imp_max_p=0`;
- nonzero first-strike tracker activity for D-prime/F/E;
- `impossible_success_n==0`, `lambda_dead_n==0`;
- finite pre/post 500 Hz joint-speed channels; finite rail exceedance is
  retained as a non-gating hardware-transport predicate/label under Amendment
  A1, while any nonfinite qvel still fails the smoke;
- clean code and asset revisions equal to the deployment manifest.

Until all four qualification rows pass, training is unauthorized.

### 4. Frozen 32-job training submission

The existing `SINGLE_TASK`, `SINGLE_SHORT`, `IMPACT_W`, and `DELIVERED_W`
hooks are used. Submit four arrays of `0-7`; never one array of `0-31`, because
`SINGLE_TASK` maps the array index directly to the seed list.

```bash
CAMPAIGN=fsr4x8 SEEDS="0 1 2 3 4 5 6 7" \
SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse SINGLE_SHORT=c \
IMPACT_W=8 DELIVERED_W=2 ITERS=500 \
sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,IMPACT_W,DELIVERED_W,ITERS,ASSET_REPO,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION,WARP_CACHE_PATH \
  scripts/slurm/vega_train.sbatch

CAMPAIGN=fsr4x8 SEEDS="0 1 2 3 4 5 6 7" \
SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy SINGLE_SHORT=dprime \
IMPACT_W=8 DELIVERED_W=2 ITERS=500 \
sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,IMPACT_W,DELIVERED_W,ITERS,ASSET_REPO,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION,WARP_CACHE_PATH \
  scripts/slurm/vega_train.sbatch

CAMPAIGN=fsr4x8 SEEDS="0 1 2 3 4 5 6 7" \
SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Linear SINGLE_SHORT=f \
IMPACT_W=8 DELIVERED_W=2 ITERS=500 \
sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,IMPACT_W,DELIVERED_W,ITERS,ASSET_REPO,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION,WARP_CACHE_PATH \
  scripts/slurm/vega_train.sbatch

CAMPAIGN=fsr4x8 SEEDS="0 1 2 3 4 5 6 7" \
SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event SINGLE_SHORT=e \
IMPACT_W=8 DELIVERED_W=2 ITERS=500 \
sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,IMPACT_W,DELIVERED_W,ITERS,ASSET_REPO,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION,WARP_CACHE_PATH \
  scripts/slurm/vega_train.sbatch
```

The launcher fails closed on matrix drift, legacy reward overrides, unknown or
dirty full code provenance, unknown or dirty canonical sibling asset
provenance, or a revision that differs from either frozen
`EXPECTED_*_REVISION`.

### 5. Post-training checkpoint capture

Create the external working manifest once from the registered template:

```bash
cp -n \
  docs/results/assets/2026-07-25_first_strike_reward/accepted_attempts.template.tsv \
  "$ACCEPTED_MANIFEST"
```

After training, inspect the retained attempt directories and Slurm records,
then fill one explicit accepted row per registered arm/seed. Record canonical
`attempt_path` and `checkpoint_path` values—not patterns—the `model_499.pt`
SHA-256, the exact frozen training code and canonical asset revisions,
`disposition=accepted`, and the reason. Do not use a glob, `latest`, mtime, or
filesystem order to choose an attempt. A retry may be accepted only for the
documented infrastructure-failure rule above; every attempt directory and log
remains retained.

Before evaluation, the manifest must contain its exact header plus 32 data
rows, exactly one for each registered arm/seed, no PENDING cells, no duplicate
checkpoint, and the same training code/asset revisions in every row. The 32
resolved paths, hashes, dispositions, reasons, and manifest digest are
**PENDING training**.

### 6. Sampled evaluation

Only after all 32 checkpoint identities are complete and the stop gates pass:

```bash
REPO_ROOT="$(pwd -P)"
ASSET_REPO="$(cd "$REPO_ROOT/../safe_impact_manipulation" && pwd -P)"
EXPECTED_CODE_REVISION="$(git rev-parse HEAD)"
EXPECTED_ASSET_REVISION="$(git -C "$ASSET_REPO" rev-parse HEAD)"
export ASSET_REPO EXPECTED_CODE_REVISION EXPECTED_ASSET_REVISION

CAMPAIGN=fsr4x8 EVAL_ATTEMPT=attempt1 \
sbatch \
  --export=ALL,CAMPAIGN,EVAL_ATTEMPT,EVAL_ROOT,ACCEPTED_MANIFEST,ASSET_REPO,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION,WARP_CACHE_PATH \
  scripts/slurm/vega_eval.sbatch
```

The launcher atomically claims a new
`$EVAL_ROOT/fsr4x8/<EVAL_ATTEMPT>/` directory; an existing attempt label is a
hard failure and is never deleted or reused. Before evaluating C, it snapshots
the external manifest into that directory, makes the snapshot read-only,
validates the snapshot's 32 explicit checkpoint paths and hashes, and records
its SHA-256. The immutable snapshot—not the mutable source—is then used for
all four arms. Every checkpoint's accepted SHA-256 is passed to the evaluator
and checked both before and while creating its private byte snapshot, before
any rollout. The CSV and raw trace provenance bind both the accepted and
observed checkpoint hashes.

C, D-prime, F, and E are evaluated sequentially in their exact registered
tasks. Every arm uses `FRESH=0`: safe append-only behavior is possible because
the attempt directory was claimed atomically before the first arm. The
evaluator also rejects drift from the sole registered
`DifferentialIKActionCfg` action (`delta_pos_scale=0.15`) or the addition of a
gain-setting action, in addition to checking the fixed actuator signature. Its
four-hour allocation replaces the inherited one-hour small-campaign limit.
The action signature covers every dataclass field of the registered DiffIK
configuration (including `max_dq`, damping, position/orientation scales and
weights, joint-limit/posture settings, and clipping). Distinct paths containing
identical checkpoint bytes are rejected during manifest preflight rather than
after evaluation.

## Proven versus open

**Proven before this preregistration:** the production C/D-prime/F/E reward
packages and repeat-derived event normalizer passed the frozen CPU
qualification reported in
`2026-07-25_first_strike_reward_qualification.md`. The sampled evaluator,
analysis, trajectory diagnostics, and launcher contracts have local automated
tests.

**Not yet proven:** the reviewed local smoke implementation has not passed a
real 256-environment CUDA qualification on any of the four registered tasks.
No 4×8 policy has trained, no sampled campaign result exists, and no
E-versus-D-prime claim can be made.

**Outside this experiment:** because `imp_max_p=0`, even a positive reward
result would not prove that the per-joint impulse constraint binds or enforces
safety, and it would not justify variable impedance by itself.
