# One-Shot Legacy Comparator Implementation Plan

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer cap”
> wording below refers only to the historical registered-task boundary, not a validated
> Z1 reaction-impulse or damage limit. The plan body remains frozen; see
> `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status (2026-07-25):** Tasks 1 and 2 are complete. Task 3's frozen
> 384-episode replay and full CPU gate passed at
> `5d986539d5ef7c9503df378467e3c5bf5c6875c4`; evidence is banked in
> `docs/results/2026-07-25_first_strike_reward_qualification.md`. The three
> simulator paths in Task 3 were corrected to their authoritative
> `docs/research/reward-design/` locations. Task 4/Vega has not begun.

**Goal:** Replace the falsified dose-matched legacy comparator with an equal-weight, one-shot D-prime legacy-readout arm and produce a compact, provenance-bound CPU qualification for C/D-prime/F/E.

**Architecture:** D-prime reuses the existing 50 Hz legacy reward classes as inner measurements while sharing the 500 Hz `FirstStrikeEventTracker` eligibility and finalization boundary with F/E. It latches the first positive legacy impact pulse, sums incremental legacy delivered pulses through first-event finalization, and emits both once on that finalization control step. The CPU probe replays identical stochastic action tapes through all four production managers, stores one physical trace per episode, and emits an aggregate-only JSON summary.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4 managers, MuJoCo Warp, pytest, NumPy.

## Global Constraints

- Fixed impedance only; no `set_gains`, commanded stiffness, or variable impedance.
- `IMP_J_LIMIT` and manufacturer caps stay unchanged.
- `imp_max_p` stays `0`; no enforcement experiment.
- No superlinear excess-over-`I_ref` reward.
- D-prime keeps `v_expected=1.0`, `I_REF_DELIVERED=0.6094 N.s`, and weights 8/2.
- F/E keep `I_REF_FIRST_STRIKE_SUCCESS`, weights 8/2, and differ only by delivered saturation.
- Do not dose-match or tune weights from the inspected bank.
- Do not edit installed `mjlab`, `rsl_rl`, or `mujoco_warp` packages.
- CPU-first: no Vega training until every CPU gate passes.
- Preserve the failed D artifacts as negative evidence; never overwrite or delete them.
- Use Plotly, not Matplotlib, for later campaign figures.
- Preserve unrelated dirty-worktree changes; edit, stage, and commit only named files.
- Before Vega, deploy one clean named commit by push and `ssh vega git pull`; never `scp` tracked files.

---

### Task 1: D-prime production reward and task registration

**Files:**
- Modify: `src/tasks/hammer/mdp/first_strike.py`
- Modify: `src/tasks/hammer/mdp/rewards.py`
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Modify: `tests/test_first_strike_event.py`
- Modify: `tests/test_impact_progress_reward.py`
- Modify: `tests/test_delivered_impulse_reward.py`
- Modify: `tests/test_configs.py`

**Interfaces:**
- Produces: `FirstStrikeEventTracker.started: torch.Tensor`, true from first contact onset through finalization.
- Produces: `FirstStrikeLegacyImpactRewardTerm(ImpactProgressTerm)`.
- Produces: `FirstStrikeLegacyDeliveredRewardTerm(DeliveredImpulseTerm)`.
- Produces factory flag `first_strike_legacy: bool=False`, appended after existing arguments.
- Produces task id `Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy`.

- [ ] **Step 1: Write the failing tracker and D-prime tests**

Add real-state tests named:

```text
test_started_is_false_until_onset_and_stays_true_after_finalization
test_legacy_impact_waits_then_pays_first_positive_pulse_once
test_legacy_impact_ignores_second_positive_pulse_inside_window
test_legacy_delivered_sums_positive_increments_through_finalization
test_legacy_wrappers_include_finalization_boundary_values
test_legacy_wrappers_consume_unproductive_finalization_without_payout
test_legacy_wrappers_ignore_delayed_recontact
test_legacy_wrappers_reset_selected_environments_only
test_legacy_wrapper_inner_outputs_match_standalone_legacy_terms
```

Use these independent literal sequence expectations:

```text
impact raw by boundary       = [0.0, 1.25, 1.75, 0.0]
delivered raw by boundary    = [0.0, 0.20, 0.30, 0.10]
tracker finalized            = [F,   F,    T,    T]
tracker productive           = [F,   F,    T,    T]
expected D-prime impact      = [0.0, 0.0,  1.25, 0.0]
expected D-prime delivered   = [0.0, 0.0,  0.50, 0.0]
```

For the unproductive case, use the same raw values with productive false and
require all-zero outputs plus permanent consumption. For partial reset, use two
environments, reset only environment 0, and require environment 1 to remain
consumed.

Add config assertions that D-prime has the shared per-substep tracker, uses the
two D-prime wrapper classes, retains the legacy normalizer, and keeps weights
8/2. Assert C remains tracker-free and F/E remain event-linear/event-saturated
with the event normalizer.

- [ ] **Step 2: Run the focused test files and verify RED**

Run:

```bash
PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
  tests/test_first_strike_event.py \
  tests/test_impact_progress_reward.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_configs.py
```

Expected: failures because `started`, the two D-prime classes, factory flag,
and task id do not exist.

- [ ] **Step 3: Implement the minimal D-prime state**

Expose `started` from the existing tracker state without changing tracker
transitions. Each wrapper must call its legacy parent on every control step,
then apply:

```python
raw = super().__call__(env, **params)
open_mask = ~self._consumed
self._latched_or_sum = update_from_raw(raw, open_mask)
finalize = tracker.finalized & open_mask
out = torch.where(finalize & tracker.productive, self._latched_or_sum, 0.0)
self._consumed |= finalize
return out
```

For impact, `update_from_raw` writes only where no positive pulse has yet been
latched and `raw > 0`. For delivered, it adds every positive raw increment.
The update occurs before checking `finalize`, so the finalization-boundary
legacy sample is included. Accumulate raw, unweighted values; the reward
manager applies 8/2 and `step_dt` once.

Wire `first_strike_legacy=True` to the same tracker as F/E, replace only the
two maximize readers, retain the legacy normalizer/parameters, and reject
composition with `event_correct` or `event_linear`.

- [ ] **Step 4: Run focused and reward/impulse regression tests**

Run the Step 2 command, then:

```bash
PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
  tests/test_first_strike_event.py \
  tests/test_impact_progress_reward.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_configs.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_cat_soft_hook.py
```

Expected: all pass.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add \
  src/tasks/hammer/mdp/first_strike.py \
  src/tasks/hammer/mdp/rewards.py \
  src/tasks/hammer/config/z1/env_cfgs.py \
  src/tasks/hammer/config/z1/__init__.py \
  tests/test_first_strike_event.py \
  tests/test_impact_progress_reward.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_configs.py
git commit -m "feat(hammer): add one-shot legacy first-strike comparator"
```

### Task 2: Repair and extend the CPU qualification probe

**Files:**
- Modify: `docs/results/assets/2026-07-25_first_strike_reward/probe_first_strike.py`
- Modify: `tests/test_first_strike_probe.py`
- Create after execution: `docs/results/assets/2026-07-25_first_strike_reward/failed_dose_match_v2/`
- Create after execution: `docs/results/assets/2026-07-25_first_strike_reward/bank_manifest.json`
- Create after execution: `docs/results/assets/2026-07-25_first_strike_reward/probe_raw.npz`
- Create after execution: `docs/results/assets/2026-07-25_first_strike_reward/probe_summary.json`
- Create: `.superpowers/sdd/2026-07-25-one-shot-legacy-comparator/task-2-report.md`

**Interfaces:**
- Replays task ids C, D-prime, F and E from one C-generated stochastic action tape.
- Uses new reset seeds `2026072700..2026072731`.
- Uses paired action seeds `2026072800..2026072831`.
- Stores each episode's physical trace once, plus four reward-return records.
- Summary contains aggregates and SHA-256 bindings, never per-substep traces.

- [ ] **Step 1: Write failing pure-analysis tests**

Add tests named:

```text
test_legacy_reference_keeps_success_termination_and_reads_terminal_snapshot
test_collect_failure_reasons_reports_all_failed_strata_and_overall
test_raw_schema_stores_one_physical_trace_per_episode
test_summary_is_aggregate_only_and_binds_raw_sha256
test_validator_recomputes_action_and_trace_digests
test_validator_binds_checkpoint_path_to_stratum_seed_and_split
test_manifest_hashes_relevant_source_files_before_payout
test_dprime_replay_latches_first_impact_and_sums_delivered
test_dprime_f_e_share_finalization_payout_step
test_all_arms_keep_weights_eight_and_two
```

Use literal fixtures with two strata and four independent gate failures; the
failure collector must return all four messages in stable gate order. Store one
trace object with digest `sha256(canonical_json(trace))`, mutate one head
position after creating the digest, and require validation to fail. Store a
checkpoint path under the wrong seed tuple and require validation to fail.
Require `episode_rows` and `physical_traces` to be absent from the JSON summary.

Expected literals must be independent fixtures. A fake 64-character digest
must fail when its content does not hash to the stored value.

- [ ] **Step 2: Run probe tests and verify RED**

```bash
PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
  tests/test_first_strike_probe.py
```

Expected: failures for the old horizon, three-copy trace schema, first-failure
short-circuit, non-recomputed digests, missing D-prime, and per-episode JSON.

- [ ] **Step 3: Implement one root-cause fix at a time**

Implement in this order, running the covering test after each change:

1. Legacy reference: do not remove `nail_driven`; use the same reference and
   six-hold-step recipe as `derive_impulse_thresholds.py`; stop on the first
   termination/truncation and snapshot `_ENV_SUBSTEP_DELIVERED_ATTR.delivered`
   immediately. Require reproduction within 1% of `0.6094`.
2. Failure aggregation: validate independent gates into a list rather than
   raising on the first stratum.
3. Integrity: recompute checkpoint, action, trace, manifest, raw and relevant
   source-file SHA-256 values. Bind each checkpoint path to its frozen
   `(stratum, training_seed, split)` tuple.
4. Storage: write one physical trace per episode in compressed NPZ. Write only
   counts, group/stratum aggregates, gate results, normalizers, provenance and
   artifact hashes to JSON.
5. D-prime: replay the fourth production task, assert physical equality, and
   retain actual reward-manager streams. Do not reconstruct D-prime from C in
   the qualifying run.

The relevant-source manifest must include:

```text
src/tasks/hammer/mdp/first_strike.py
src/tasks/hammer/mdp/rewards.py
src/tasks/hammer/config/z1/env_cfgs.py
src/tasks/hammer/config/z1/__init__.py
docs/results/assets/2026-07-25_first_strike_reward/probe_first_strike.py
```

- [ ] **Step 4: Preserve the failed bank and commit the repaired harness**

Move the existing three artifacts into
`failed_dose_match_v2/` without changing their bytes. Record their existing
SHA-256 values in a short `README.md` explaining the legacy-horizon defect and
genuine D dose-transport failure.

Then commit only:

```bash
git add \
  docs/results/assets/2026-07-25_first_strike_reward/probe_first_strike.py \
  docs/results/assets/2026-07-25_first_strike_reward/failed_dose_match_v2/README.md \
  tests/test_first_strike_probe.py
git commit -m "fix(hammer): harden first-strike CPU qualification"
```

The large moved artifacts remain local evidence unless explicitly approved for
version control.

- [ ] **Step 5: Run the complete frozen CPU qualification**

```bash
/usr/bin/time -p env \
  PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=. \
  MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/results/assets/2026-07-25_first_strike_reward/probe_first_strike.py
```

Require exit 0, 384/384 episodes, 10/10 phase invariance, exact population,
legacy/event normalizer gates, C/D-prime/F/E physical equality, D-prime inner
reader fidelity, identical D-prime/F/E finalization payout boundary, one payout,
zero delayed payout, all failure reasons empty, and an aggregate-only summary.

### Task 3: Full CPU pre-training gate and result banking

**Files:**
- Modify: `docs/superpowers/specs/2026-07-25-first-strike-reward-semantics.md`
- Modify: `docs/superpowers/plans/2026-07-25-first-strike-reward-semantics.md`
- Create: `docs/results/2026-07-25_first_strike_reward_qualification.md`
- Modify: `.superpowers/sdd/2026-07-25-one-shot-legacy-comparator/progress.md`

- [x] **Step 1: Run the complete reward/impulse test set**

```bash
PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
  tests/test_first_strike_event.py \
  tests/test_first_strike_probe.py \
  tests/test_impact_progress_reward.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_cat_soft_hook.py \
  tests/test_configs.py
```

- [x] **Step 2: Run the three simulator validation scripts**

```bash
PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/validate_rewards.py
```

Require phases A--M pass.

```bash
PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_contact_sensor.py
```

```bash
PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_reward_setup.py
```

- [x] **Step 3: Bank the qualification result**

The result document must include:

- exact commands, exit codes, runtime, commit and source hashes;
- failed D result preserved as negative evidence;
- D-prime/F/E package definitions and causal boundaries;
- C/D-prime/F/E per-stratum discounted/undiscounted return diagnostics;
- event normalizer derivation and legacy normalizer reproduction;
- impossible-success, dead-lambda and hardware-speed gate status;
- explicit proven-versus-open statements;
- statement that reward-package qualification does not predict PPO ranking.

- [x] **Step 4: Commit named evidence files**

```bash
git add \
  docs/superpowers/specs/2026-07-25-first-strike-reward-semantics.md \
  docs/superpowers/plans/2026-07-25-first-strike-reward-semantics.md \
  docs/superpowers/plans/2026-07-25-one-shot-legacy-comparator.md \
  docs/results/2026-07-25_first_strike_reward_qualification.md
git commit -m "docs(hammer): qualify equal-weight first-strike reward packages"
```

The generated manifest, summary and raw NPZ remain local evidence. Do not add
them or unrelated worktree files unless the user explicitly requests generated
or large binary evidence in git.

### Task 4: Clean Vega smoke handoff

**Files:**
- Modify only if a smoke defect is found: files named by the failing test.
- Create after execution: `.superpowers/sdd/2026-07-25-one-shot-legacy-comparator/task-4-report.md`

- [ ] **Step 1: Push and pull the exact qualified commit**

```bash
git push origin first-strike-reward
ssh vega 'cd ~/repos/unitree_rl_mjlab && git pull --ff-only'
```

Never use `scp` for tracked files.

- [ ] **Step 2: Verify clean remote provenance and GPU visibility**

On Vega require:

```text
git status --porcelain == empty
git rev-parse HEAD == local qualified commit
nvidia-smi sees the assigned GPU
```

- [ ] **Step 3: Run one short CUDA instrumentation smoke per distinct package**

Use one GPU per process and the existing training launcher. Smoke C, D-prime,
F and E long enough to produce completed episodes and verify:

```text
impossible_success_n == 0
lambda_dead_n == 0
finite rewards and observations
nonzero first-strike event activity for D-prime/F/E
D-prime/F/E task ids and weights are exactly the qualified config
```

Do not submit the 4x8 matrix if any smoke gate fails.

- [ ] **Step 4: Return to the main first-strike plan**

After clean smokes, resume Task 4 and later tasks in
`docs/superpowers/plans/2026-07-25-first-strike-reward-semantics.md` for sampled
evaluation fields, Plotly diagnostics, the frozen 4x8 run matrix, and final
result banking.
