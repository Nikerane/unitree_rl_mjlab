# First-Strike Reward Semantics Implementation Plan

> **Comparator amendment:** The original dose-matched D design in Task 3 was
> falsified by its frozen validation bank. D/`alpha_D`/`alpha_F` instructions
> below are historical and must not be executed. The approved equal-weight
> D-prime repair is specified and executed by
> `docs/superpowers/plans/2026-07-25-one-shot-legacy-comparator.md`. Its frozen
> replay and full CPU pre-training gate passed on 2026-07-25 at
> `5d986539d5ef7c9503df378467e3c5bf5c6875c4`; exact evidence is banked in
> `docs/results/2026-07-25_first_strike_reward_qualification.md`. Its Vega/CUDA
> smoke has **not** begun. Resume this plan at Task 4 only after that smoke
> passes.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and validate experimental fixed-impedance reward arms that credit
the first strike using true substep pre-contact velocity and a separately
calibrated, success-censored event impulse, while isolating event semantics from
delivered-reward saturation.

**Architecture:** A shared per-substep `FirstStrikeEventTracker` owns the event
state and immutable final snapshot. Two one-shot reward terms consume that
snapshot, while all legacy diagnostic accumulators and the shipped task remain
unchanged. A separate event normalizer preserves the legacy `0.6094 N·s`
constant. Linear and saturated event tasks support the preregistered
C/D-prime/F/E comparison. Evaluation exposes sampled first-strike outcomes and
exact trajectory traces before any Vega comparison.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4 managers, MuJoCo Warp, pytest, Plotly, Slurm.

## Global Constraints

- Fixed impedance only; no `set_gains`, commanded stiffness, or variable impedance.
- `IMP_J_LIMIT` and manufacturer caps stay unchanged.
- `imp_max_p` stays `0`; no enforcement experiment.
- No superlinear excess-over-`I_ref` reward.
- Do not edit installed `mjlab`, `rsl_rl`, or `mujoco_warp` packages.
- CPU-first: no Vega training until every CPU gate passes.
- Use Plotly, not Matplotlib, for new campaign figures.
- Preserve unrelated dirty-worktree changes; edit and stage only named files.
- Before Vega, use one clean named commit deployed by commit/push/`ssh vega git pull`; no `scp` of tracked files.
- Comparison is fixed at four arms × eight training seeds, uses sampled
  evaluator columns, seed-level exact two-sided Mann-Whitney U plus an exact
  seed-label permutation test for the primary mean estimand, practical
  thresholds, and the `impossible_success_n==0` / `lambda_dead_n==0` gates.

---

### Task 1: Shared first-strike event tracker

**Files:**
- Create: `src/tasks/hammer/mdp/first_strike.py`
- Modify: `src/tasks/hammer/mdp/__init__.py`
- Create: `tests/test_first_strike_event.py`

**Interfaces:**
- Produces: `FirstStrikeEventTracker(ManagerTermBase)`, stashed on the environment under `_ENV_FIRST_STRIKE_ATTR`.
- Produces immutable tensor properties `finalized`, `productive`, `v_precontact`, `delivered`, `peak_depth`, `depth_at_contact`, and integer `reason`.
- Uses reason values `REASON_NONE=0`, `REASON_SUCCESS=1`, `REASON_WINDOW=2`.
- Uses parameters `contact_sensor_name`, `impulse_sensor_name`, `robot_cfg`,
  `nail_cfg`, `axis`, `window_substeps=25`, `progress_eps=5e-4`, and the
  existing task success/clamp constants.

- [ ] **Step 1: Write failing tracker tests**

Write real-state stub tests whose hand-derived fixtures verify:

```python
def test_first_contact_uses_onset_minus_previous_position(): ...
def test_contact_depth_is_previous_substep_depth(): ...
def test_window_age_counts_wall_time_across_contact_gaps(): ...
def test_success_substep_is_included_and_later_samples_are_frozen(): ...
def test_success_on_substep_25_precedes_window_finalization(): ...
def test_delayed_recontact_cannot_change_final_snapshot(): ...
def test_unproductive_window_finalizes_but_is_not_productive(): ...
def test_reset_mid_contact_requires_clean_free_flight_rearm(): ...
def test_subset_reset_does_not_change_other_environment(): ...
```

Use literal positions, forces, depths and `physics_dt=0.002`; do not derive expected values with tracker helpers.

- [ ] **Step 2: Verify RED**

Run:

```bash
PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q tests/test_first_strike_event.py
```

Expected: collection fails because `src.tasks.hammer.mdp.first_strike` does not exist.

- [ ] **Step 3: Implement the minimal tracker**

Implement the four-state tensor state machine from
`docs/superpowers/specs/2026-07-25-first-strike-reward-semantics.md`.
At the rising edge, compute velocity from the onset head position minus the
immediately previous off-contact position. This is not the cached off/off
velocity: under the real mjlab callback order the rising-edge sample is the
pre-force onset state, and the cached off/off value is one substep stale. Latch
the previous substep's clamped depth as `depth_at_contact`. The first-contact
substep starts wall-time age at one and includes its current force/depth.
Freeze on the inclusive success substep or after wall-time age 25, with success
precedence on age 25. Never rearm before reset.

- [ ] **Step 4: Verify GREEN and regression**

Run the Task 1 test, then:

```bash
...python -m pytest -q tests/test_first_strike_event.py tests/test_impulse_bound.py
```

Expected: all pass.

### Task 2: One-shot rewards and experimental task registration

**Files:**
- Modify: `src/tasks/hammer/mdp/rewards.py`
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Modify: `tests/test_impact_progress_reward.py`
- Modify: `tests/test_delivered_impulse_reward.py`
- Modify: `tests/test_configs.py`

**Interfaces:**
- Consumes: `_ENV_FIRST_STRIKE_ATTR` and Task 1 snapshot properties.
- Produces: `FirstStrikeImpactRewardTerm`, paying once as `v_precontact/v_expected`.
- Produces: `FirstStrikeDeliveredRewardTerm`, paying once as either
  `delivered/i_ref_event` (linear) or `min(delivered/i_ref_event, 1)`
  (saturated).
- Preserves `I_REF_DELIVERED=0.6094 N·s` for legacy behavior and adds a
  separately named `I_REF_FIRST_STRIKE_SUCCESS` derived from the exact default
  scripted reference under the success-censored event horizon.
- Produces factory flag `event_correct: bool=False`.
- Produces task ids `Unitree-Z1-Hammer-CaT-Impulse-Event` (saturated E) and
  `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear` (linear F).
- The legacy `ImpactProgressTerm`, `DeliveredImpulseTerm`, task ids and diagnostic accumulators remain byte-for-byte behavior compatible.

- [ ] **Step 1: Write failing one-shot reward tests**

Add tests showing each corrected term:

```python
def test_event_reward_waits_until_tracker_finalizes(): ...
def test_event_reward_pays_once_only(): ...
def test_event_reward_rejects_unproductive_snapshot(): ...
def test_event_impact_uses_latched_precontact_speed(): ...
def test_event_delivered_saturates_at_one_i_ref(): ...
def test_event_delivered_linear_does_not_saturate(): ...
def test_event_reward_reset_rearms_paid_latch(): ...
```

Add config tests asserting both event tasks have `first_strike` as a per-substep
metric and use the corrected reward classes and the separate event normalizer;
only F is linear. The legacy task does not contain the tracker and still uses
the legacy normalizer.

- [ ] **Step 2: Verify RED**

Run the three focused files. Expected: imports/config assertions fail because
the corrected reward classes and task flag do not exist.

- [ ] **Step 3: Implement corrected rewards and flag**

Add independent per-env `_paid` tensors. Read, but never mutate, the shared
tracker. Raise a clear `RuntimeError` when the tracker is absent. Validate
finite positive `v_expected` and `i_ref`. Validate the delivered payout mode.
Under `event_correct=True`, wire the
shared tracker after both contact sensors exist, using
`hammer_nail_contact` only for the rising edge and
`hammer_nail_impulse` for world-frame net axial force. Require
`event_correct => cat_impulse`, source nail success/clamp values from the
existing task constants, replace only the two maximize reward functions, and
register the two separate task ids. Append any new factory argument so existing
positional callers remain compatible. Reward terms resolve the tracker lazily
in `__call__`, because rewards are constructed before metrics.

- [ ] **Step 4: Verify GREEN and legacy compatibility**

Run:

```bash
...python -m pytest -q \
  tests/test_first_strike_event.py \
  tests/test_impact_progress_reward.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_configs.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_cat_soft_hook.py
```

Expected: all pass.

### Task 3: CPU phase probe and offline adversarial replay (historical; do not execute)

> **Superseded record:** Every instruction in this Task 3 section, including
> its dose-matched D calibration and transport gates, is historical and
> non-executable. The approved D-prime replacement and completed CPU
> qualification are in
> `docs/superpowers/plans/2026-07-25-one-shot-legacy-comparator.md` and
> `docs/results/2026-07-25_first_strike_reward_qualification.md`.

**Files:**
- Create: `docs/results/assets/2026-07-25_first_strike_reward/probe_first_strike.py`
- Create: `tests/test_first_strike_probe.py`
- Create after execution: `docs/results/assets/2026-07-25_first_strike_reward/probe_summary.json`

**Interfaces:**
- Consumes the registered event task, existing scripted reference, and selected
  existing checkpoints.
- Produces ten synthetic phase-shift rows for the identical physical event with
  detected `v_precontact`, independent finite-difference ground truth, both
  payout counts, event impulse, progress and finalization reason.
- Produces a frozen 2 ms force-bearing CPU event bank with contact, head
  position, clamped depth, net axial force, 500 Hz joint speed, action,
  checkpoint/training/reset provenance, trace digest and actual production
  reward-manager streams for C/E/F.
- Produces a separately provenance-bound event-horizon normalizer while
  re-verifying the unchanged legacy `I_REF_DELIVERED`.
- Produces offline rescoring rows for representative clean, chatter and
  delayed-restrike traces.

- [ ] **Step 1: Write failing pure-analysis tests**

Test the summary validator with literal good and bad rows. It must reject:
phase recall below 10/10, non-invariant matched phase values, payout count other
than one for either corrected reward on productive strikes, velocity error above
`max(0.02, 0.02*abs(v_true))`, any late recontact payout, physical trace
inequality across arms, unfrozen/missing provenance, adaptive prefix selection,
validation dose error above tolerance, or non-finite values. Test that a failed
gate still writes a summary marked invalid.

- [ ] **Step 2: Verify RED**

Run `tests/test_first_strike_probe.py`; expect import failure for the absent probe module.

- [ ] **Step 3: Implement the smallest probe**

Reuse the real environment and reference plumbing already used by
`validate_rewards.py`/`playback_reference.py` and the existing substep recorder.
Do not duplicate simulator configuration.

For phase invariance, use literal/synthetic samples representing the same
physical strike shifted across the ten decimation phases. Separately run the
real default reference and any clearance/speed sweep as coverage, not as a
matched invariance claim. Compute velocity ground truth from onset and the
immediately preceding position.

Rescore `traj_dc.json` to confirm delayed contacts beyond 50 ms cannot
contribute. Do not hard-code the expected payout; replay it algorithmically.
Because existing trajectory JSONs contain no force history, bank new
force-bearing traces before preregistration.

Freeze exact checkpoint paths and hashes plus never-inspected reset seeds before
computing payouts. Split by complete checkpoint/training-seed groups, stratified
by policy family; no policy contributes episodes to both calibration and
validation. Record one action sequence and replay it from the identical reset
through the actual C/E/F production reward managers. Assert physical trace
equality before comparing payout.

Use the complete fixed calibration bank once:

```text
alpha_D = mean_cal(discounted E saturated return)
          / mean_cal(discounted C legacy return)
alpha_F = mean_cal(discounted E saturated return)
          / mean_cal(discounted F linear return)
```

Set D/F weights to `(8*alpha, 2*alpha)`. Require held-out discounted maximize
return agreement within 5% overall and 10% for every represented major behavior
stratum. Also report undiscounted payout, component mix, saturation fraction and
payout timing.

Independently reproduce the legacy `0.6094 N·s` reference and derive the
success-censored `I_REF_FIRST_STRIKE_SUCCESS` from the exact default reference.
Never compare them as if they shared a horizon. Persist raw data and a strict
summary even when a gate fails, then exit non-zero.

- [ ] **Step 4: Run the probe and focused gates**

Run the probe with the conda interpreter and `PYTHONPATH=.`. Require 10/10 phase
recall/invariance, one payout per corrected term, normalizer reproduction,
frozen-manifest validation, physical trace equality, held-out dose tolerances,
and zero delayed payout. Then run the focused reward/impulse suite.

### Task 4: Sampled evaluation fields and Plotly comparison

**Files:**
- Modify: `scripts/eval_impulse.py`
- Create: `evaluation/analysis/first_strike_campaign.py`
- Create: `evaluation/analysis/plot_first_strike_campaign.py`
- Create: `tests/test_first_strike_campaign.py`

**Interfaces:**
- Produces sampled CSV columns:
  `first_strike_useful_speed_mean_sampled`,
  `successful_first_strike_v_precontact_mean_sampled`,
  `first_strike_success_rate_sampled`, `recontact_rate_sampled`,
  `first_strike_v_precontact_mean_sampled`,
  `first_strike_productive_rate_sampled`,
  `first_strike_delivered_success_mean_sampled`,
  `first_strike_delivered_window_mean_sampled`,
  `first_strike_success_n_sampled`, `first_strike_window_n_sampled`,
  `first_strike_no_contact_n_sampled`,
  `tail_fraction_mean_sampled`,
  `maximize_return_discounted_mean_sampled`,
  `impact_return_discounted_mean_sampled`,
  `delivered_return_discounted_mean_sampled`,
  `first_strike_saturation_rate_sampled`,
  `overall_success_rate_sampled`, `impossible_success_n`,
  `lambda_dead_n`, `qvel_violation_rate_sampled`,
  `precontact_path_length_ratio_mean_sampled`,
  `precontact_lateral_excursion_mean_sampled`, and
  `contact_approach_angle_mean_sampled`.
- Produces the E-versus-D-prime primary with seed-level exact MWU, exact
  mean-difference label permutation, the 10% relative-effect threshold and
  both sampled-success guardrails.
- Produces the primary-gated F-versus-D-prime and E-versus-F mechanism family
  with Holm correction, plus descriptive C-versus-D-prime diagnostics.
- Fails closed on dirty provenance, `impossible_success_n > 0`,
  `lambda_dead_n > 0`, or any 500 Hz hardware-speed violation.
- Produces Plotly HTML trajectory/contact figures and state-ground-truth path
  overlays for representative rendered videos. It does not introduce image
  tracking unless a later state/render disagreement or hardware-camera need is
  demonstrated.

- [ ] **Step 1: Write failing evaluator/analysis tests**

Use literal episode fixtures to verify first-window success, tail fraction,
recontact, qvel violation, useful speed, path-length ratio, lateral excursion
and contact-approach-angle aggregation. Verify reset rollouts are nested under
training seed, MWU/permutation inference is performed across seed summaries,
the sole primary is E versus D-prime, both exact tests and the 10% effect
threshold must pass, and both sampled-success guardrails are enforced. Verify
Holm is limited to F versus D-prime and E versus F after a passing primary,
C versus D-prime remains descriptive, and dirty/sentinel/hardware-speed rows
fail closed.

- [ ] **Step 2: Verify RED**

Run `tests/test_first_strike_campaign.py`; expect missing-module/column failures.

- [ ] **Step 3: Implement minimal collection and analysis**

Extend the existing evaluator snapshot path rather than creating a parallel
rollout engine. Run C, D-prime, F and E through identical event
instrumentation, record the exact registered task id and configured 8/2
weights, snapshot the tracker before autoreset, restore training-matched reset
noise and observation corruption with frozen unseen evaluator RNG streams, and
measure qvel at 500 Hz. Collect exactly the first 512 completed episodes per
trained seed, balanced as two episodes from each of 256 environments. Store the
raw substep trace that defines first-window/recontact/tail once; distinguish
counterfactual legacy and event payouts. Separate success-finalized and
window-finalized delivered impulse and their denominators; never treat their
mixed mean as a common impulse estimand. Add only the signals absent from
existing NPZ/CSV data.

Trajectory diagnostics are descriptive until an observed failure is tied to
task outcome. Use simulator state as ground truth:

- 3D, x-z side and x-y top paths, colored by normalized time;
- nail axis, contact onset and success markers;
- pre-contact path-length ratio, lateral excursion and approach angle;
- contact-aligned force, depth, cumulative event impulse and reward payout;
- paired successful/failed and C/D-prime/F/E representative traces, labelled
  with both treatment name and registered task id.

Render selected rollouts and overlay the recorded state path and event markers.
Do not add a computer-vision tracker merely to re-estimate state already
available at 500 Hz. A CV path becomes a separate validation task only for real
camera data or a demonstrated render/state timestamp mismatch.

- [ ] **Step 4: Verify GREEN**

Run the new tests plus `tests/test_rl_evidence_analysis.py`. Regenerate a test
figure/video-overlay fixture set and verify every HTML artifact contains the
expected arm labels and every overlay uses the same trace digest as its source.

### Task 5: Pre-register and prepare the Vega campaign

**Files:**
- Create: `docs/results/2026-07-25_first_strike_reward_experiment.md`
- Modify: `scripts/slurm/vega_train.sbatch`
- Modify only if needed for new sampled fields: `scripts/slurm/vega_eval.sbatch`

**Interfaces:**
- Produces four equal-weight treatment packages with seeds
  `0 1 2 3 4 5 6 7` (32 jobs):
  - **C** — `Unitree-Z1-Hammer-CaT-Impulse`, shipped repeated-credit legacy
    readers, weights 8/2;
  - **D-prime** —
    `Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy`, first-event-censored
    one-shot legacy readers, weights 8/2;
  - **F** — `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear`, 500 Hz event readers
    with linear delivered payout, weights 8/2;
  - **E** — `Unitree-Z1-Hammer-CaT-Impulse-Event`, the same 500 Hz event readers
    with saturated delivered payout, weights 8/2.
- Sole primary decision: E exceeds D-prime by at least 10% relatively in
  `first_strike_useful_speed_mean_sampled`, with two-sided exact seed-level MWU
  and exact seed-label permutation `p<0.05`. First-window and overall sampled
  success must each remain at least 90% and no more than five percentage points
  below D-prime.
- Only after the primary passes, F versus D-prime and E versus F form the
  Holm-corrected mechanism family. F must pass the same success guardrails
  before F versus D-prime is interpreted. C versus D-prime is descriptive.
- Any dirty hash, `impossible_success_n > 0`, `lambda_dead_n > 0`, or 500 Hz
  hardware-speed violation invalidates the comparison.

- [ ] **Step 1: Add configuration tests before script changes**

Extend `tests/test_first_strike_campaign.py` to validate an explicit campaign
matrix. Reject any matrix that lacks the exact C/D-prime/F/E task-id mapping
above, weights 8/2 in every arm, eight unique seeds per arm, 500 iterations,
4096 environments, final `model_499.pt`, unique run names, `imp_max_p=0`,
unchanged impulse caps, or the correct legacy/event/saturation mode for each
task.

- [ ] **Step 2: Verify RED**

Run the campaign tests; expect failure because the C/D-prime/F/E matrix does not
exist.

- [ ] **Step 3: Add the minimal task/weight hooks and pre-registration**

Reuse `SINGLE_TASK`, `IMPACT_W` and `DELIVERED_W`. Add no general campaign
framework and do not add any payout multiplier. Bank the four task ids, equal
8/2 weights, hypotheses, metrics, thresholds, invalidation gates, seed list,
500 training iterations, 4096 environments, exact balanced 512-episode
evaluator stopping rule, frozen checkpoint manifest and held-out reset-noise
evaluation. State in advance that n=8 can support a large-effect screen but
cannot establish equivalence after a null.

For the E-versus-D-prime primary, enumerate all
`C(16,8)=12,870` seed-label assignments with midranks for the two-sided exact
MWU and use the same assignments for the two-sided exact mean-difference
permutation test. Require both `p < 0.05` and the 10% relative improvement;
invalidate the relative test when D-prime's mean is non-positive. Report
`U`, tie-adjusted `A12=U/64` oriented as E over D-prime, absolute and relative
effects, and a separately labelled 100,000-resample within-arm seed-bootstrap
95% interval with a frozen RNG seed.

Gate the mechanism family on a passing primary. For F versus D-prime and E
versus F, set `p_joint=max(p_MWU,p_permutation)`, require the hypothesized
direction, then apply Holm across the two `p_joint` values. C versus D-prime
remains a descriptive repeated-credit/censoring/timing package contrast.

- [ ] **Step 4: Run the non-GPU pre-training gate**

Run:

```bash
...python -m pytest -q \
  tests/test_impact_progress_reward.py \
  tests/test_first_strike_event.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_cat_soft_hook.py \
  tests/test_first_strike_probe.py \
  tests/test_first_strike_campaign.py
PYTHONPATH=. ...python docs/research/reward-design/validate_rewards.py
PYTHONPATH=. ...python docs/research/reward-design/verify_contact_sensor.py
PYTHONPATH=. ...python docs/research/reward-design/verify_reward_setup.py
```

Require pytest green, validation A-M green, contact sensor verified, and the
random-policy reward setup sweep green.

- [ ] **Step 5: Provenance and Vega smoke**

Stage only the named implementation, test, evaluation, plan and result files
from Tasks 1–5. Create one clean commit without a co-author
trailer, push it, then `ssh vega` and fast-forward pull. Verify Vega reports the
same hash with no `-dirty`. Record the hashes and dirty states of this repository
and the sibling Z1 asset repository. Run one short CUDA instrumentation smoke
for each distinct package/task: C, D-prime, F and E. Require finite rewards and
observations, nonzero first-strike activity for D-prime/F/E, exact task ids and
8/2 weights, `impossible_success_n==0`, `lambda_dead_n==0`, and no 500 Hz
hardware-speed violation. Do not launch the campaign if any provenance,
instrumentation, safety or liveness gate fails.

- [ ] **Step 6: Launch and evaluate the full comparison**

Submit 32 one-GPU jobs: eight each for C, D-prime, F and E. Train every job for
500 iterations with 4096 environments and retain `model_499.pt`. After all
complete, run sampled evaluation on exactly the first 512 completed episodes
per checkpoint, balanced as two episodes from each of 256 environments, with
stochastic policy actions, held-out `±0.05 rad` reset noise and
training-matched observation corruption. Retry only documented infrastructure
failures with the identical arm/seed/config and retain every attempt; never
replace a completed poor or unstable seed. Never analyze an incomplete matrix.
Stop only for provenance, instrumentation, safety or liveness failure.

- [ ] **Step 7: Analyze and bank the result**

Run the campaign analysis and Plotly generator. Report seed mean ± std, min/max,
the E-versus-D-prime exact MWU and exact mean-difference permutation result,
absolute and relative effects, both success guardrails, the separately
labelled seed-bootstrap 95% interval, and tie-adjusted `A12=U/64` oriented as E
over D-prime. If the primary passes, report Holm-adjusted F-versus-D-prime and
E-versus-F mechanism results in their hypothesized directions; always report
C-versus-D-prime as descriptive. Include sentinel gates, hardware-speed
violations and trajectory diagnostics with exact C/D-prime/F/E labels. State
separately what is proven, not distinguishable, and still open.
