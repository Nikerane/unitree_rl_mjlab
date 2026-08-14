# Cartesian Guideline Pilot Evaluator Implementation Plan

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer
> `IMP_J_LIMIT`” wording below refers only to the historical registered-task boundary,
> not a validated Z1 reaction-impulse or damage limit. The plan body remains frozen;
> see `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing sampled checkpoint evaluator just enough to evaluate the excluded fixed-reset C0/C-Gate pilot with an episode-first straightness endpoint and actual gate-payout evidence.

**Architecture:** A pure NumPy reducer owns guideline episode/seed/pilot analysis. The existing sampled trace collector adds the missing tracker geometry/state and actual gate payout, binds them into the trace digest, and dispatches the two tasks only under one explicit pilot campaign. No new launcher or evaluation framework is introduced.

**Tech Stack:** Python 3.10/3.12, NumPy, PyTorch, pytest, existing mjlab sampled evaluator.

## Global Constraints

- Work only on `cartesian-guideline-fic` in the isolated worktree.
- Fixed impedance only; never call `set_gains`.
- Keep manufacturer `IMP_J_LIMIT` values unchanged.
- Keep `imp_max_p=0.0`; no enforcement or VIC.
- Fixed reset `(0.0, 0.0)`; no wind-up or reset randomization.
- C0/C-Gate differ only by literal `r_gate` treatment identity.
- Exactly four excluded-pilot identities: C0 seeds 0/1 and C-Gate seeds 0/1.
- TDD: observe RED before production changes.
- Stage named files only; never `git add -A`.

---

### Task 1: Pure episode-first guideline reducer

**Files:**
- Create: `evaluation/analysis/guideline_campaign.py`
- Create: `tests/test_guideline_campaign.py`

**Interfaces:**
- Produces `summarize_guideline_episode(trace: Mapping) -> dict`.
- Produces `aggregate_guideline_seed(episodes: Sequence[Mapping], *, expected_count: int = 512) -> dict`.
- Produces `validate_guideline_pilot_rows(rows: Sequence[Mapping]) -> dict`.

- [ ] **Step 1: Write RED tests for the episode endpoint**

Test literal behavior: no gate 1 returns `0.050`; contact before gate 1 returns
`0.050`; a valid window returns q90 of errors capped at `0.050`; no-contact uses
episode end; malformed/nonfinite geometry or misaligned trace lengths raise.

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_guideline_campaign.py -q -p no:cacheprovider
```

Require failure because the module/functions do not exist.

- [ ] **Step 3: Implement the minimal episode reducer**

Use only stored `physical`, `event_trace`, `guideline`, `first_strike`, terminal
Lambda/depth/success and frozen caps. Derive the primary endpoint, six-gate
indicator, 5 mm occupancy, backward-progress count, actual gate return, success,
speed, Lambda ratios and qvel qualification without success filtering.

- [ ] **Step 4: Add RED seed/pilot contract tests**

Prove long/short episodes are aggregated episode-first; require exactly 512
episodes for a production seed; require exact four rows `(C0,0)`, `(C0,1)`,
`(C-Gate,0)`, `(C-Gate,1)`; reject duplicate/extra/replacement identities,
non-final checkpoints, mismatched reset/geometry/RNG/provenance, sentinel or
nonfinite-qvel failures, C0 payout, or either C-Gate seed with nonpositive payout.
Require continuation only when each arm has at least one success rate `>=0.25`.

- [ ] **Step 5: Implement seed and pilot validation, then GREEN**

Run the Task 1 command and require all tests pass.

- [ ] **Step 6: Commit named files**

```bash
git add evaluation/analysis/guideline_campaign.py tests/test_guideline_campaign.py
git commit -m "feat(eval): add episode-first guideline reducer"
```

---

### Task 2: Additive guideline trace persistence and strict dispatch

**Files:**
- Modify: `scripts/eval_impulse.py`
- Modify: `tests/test_eval_impulse_hook.py`

**Interfaces:**
- Consumes the existing `GUIDELINE_ARM_TASKS`,
  `_validate_native_guideline_env_contract`, `_SampledTraceCollector`, reset
  digests and content-addressed NPZ writer.
- Produces guideline-only trace contract version 1 and the seed-row fields
  consumed by Task 1.

- [ ] **Step 1: Write RED campaign-dispatch tests**

Require C0/C-Gate only under `cartesian-guideline-pilot`; reject near-match or
unscoped tasks, campaign RNG drift, non-final checkpoint name, missing expected
checkpoint SHA, dirty/unknown code or asset provenance, fixed-reset drift, and
all existing contract mutations. Keep legacy task choices/digests unchanged.

- [ ] **Step 2: Run the focused RED selection**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_eval_impulse_hook.py -q -p no:cacheprovider -k guideline
```

Require failures because guideline checkpoints remain excluded.

- [ ] **Step 3: Implement strict dispatch with the existing native validator**

Add the pilot campaign and explicit payout semantics without changing legacy or
FQ mappings. Load `play=False`, call `_validate_native_guideline_env_contract`,
keep fixed reset, and require the frozen pilot identity/provenance inputs before
rollout.

- [ ] **Step 4: Write RED collector/digest tests**

Require episode-local frozen entry/nail, aligned 500 Hz `next_gate`,
`perpendicular_error_m`, `disarmed`, and control-rate gate payout. Prove first
completed episode data is not overwritten across autoreset. Mutating any new
field must change the trace digest. Missing, nonfinite, nonmonotone, out-of-range,
or treatment-inconsistent values must fail closed.

- [ ] **Step 5: Implement additive persistence**

For guideline tasks only, read the existing `WaypointProgressTracker` after its
per-substep update. Store the four guideline channels and actual weighted
manager payout; represent C0 as a zero payout stream with
`gate_reward_present=false`. Do not duplicate head/qvel/contact/Lambda fields.
Fold the guideline section and gate payout into the physical trace digest only
when `guideline_trace_contract_version=1` is present.

- [ ] **Step 6: Connect the pure reducer to sampled row output**

Emit the primary q90 straightness endpoint plus six-gate rate, corridor
occupancy, backward count, actual gate return, fixed-reset digest and geometry
digest. Preserve existing success/speed/Lambda/qvel sampled columns.

- [ ] **Step 7: GREEN focused suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_eval_impulse_hook.py tests/test_guideline_campaign.py \
  -q -p no:cacheprovider
```

- [ ] **Step 8: Commit named files**

```bash
git add scripts/eval_impulse.py tests/test_eval_impulse_hook.py
git commit -m "feat(eval): persist strict guideline traces"
```

---

### Task 3: Review and pre-training gate

**Files:**
- Modify only a file named by a concrete failing test or reviewer finding.
- Bank a dated evaluator qualification note after review.

- [ ] **Step 1: Independent Task 1 and Task 2 reviews**

Require SPEC PASS and QUALITY APPROVED. Resolve every Critical/Important finding
with a focused RED test before changing production code.

- [ ] **Step 2: Fresh local verification**

Run the focused evaluator suite, the existing guideline/config/environment
suite, the mandatory impulse/reward tests, `validate_rewards.py` A--M,
`verify_contact_sensor.py`, and `verify_reward_setup.py`.

- [ ] **Step 3: Direct one-environment CPU evaluator smoke**

Use a frozen local checkpoint only if one exists with clean provenance; otherwise
exercise collector/reducer fixtures without inventing a checkpoint. Do not relax
the required final-checkpoint identity.

- [ ] **Step 4: Broad final review and evidence note**

Confirm no runtime task/reward/action/gain/cap changes, no legacy evaluator
drift, and exact implementation of the four-row pilot contract.

- [ ] **Step 5: Stop before training**

Report the reviewed revision and exact four planned training identities. Ask the
owner again before pushing/deploying that revision and launching PPO.
