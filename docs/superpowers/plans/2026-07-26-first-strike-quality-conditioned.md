# Lean First-Strike Quality Reward Implementation Plan

> **Impulse-threshold provenance correction (2026-08-14):** `[1.64, 3.28, ...]`
> below records the historical registered-task boundary, not a validated Z1
> reaction-impulse or damage limit. The plan body remains frozen; see
> `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: use
> superpowers:subagent-driven-development or superpowers:executing-plans.
> Execute only the active sequence below. The deferred appendix is
> non-executable.

**Goal:** Finish and evaluate a fixed-impedance FQ-min reward in a matched
F8/F0/D0/FQ-min 4×8 campaign while preserving the completed physics-rate
contact-quality tracker and schema-v3 provenance work.

**Architecture:** A dedicated passive multi-slot contact sensor feeds a pure
Torch contact-quality kernel. The shared first-strike tracker latches one
immutable onset/finalization record. F8/F0/D0 retain raw one-shot readers.
FQ-min uses only one bounded quality-gated speed reader and disables delivered
reward. A focused offline module verifies schema-v3 artifacts, performs
seed-level inference, and generates two static figures.

**Tech stack:** Python 3.10, PyTorch, mjlab 1.4.0, MuJoCo/mujoco_warp 3.8.1,
NumPy, the repository's exact Mann–Whitney implementation, Matplotlib, pytest,
Slurm/Vega.

## Execution status and exact active sequence

Completed foundations:

1. **Task 1 — Pure contact-quality kernel:** complete at `7a55244`.
2. **Task 2 — Dedicated sensor and immutable tracker snapshot:** complete at
   `d891bbe`.
3. **Task 3 — Quality-reader/task foundation:** complete at `ccf8bde`; its
   broad delivered-reader registration is superseded by Task 3A.
4. **Task 4 — Schema-v3 sampled traces and physical-identity replay:** complete
   at `e2fbf2b`, with passive-diagnostic and raw-slot corrections at `b63ea38`
   and `ce40525`.

Execute next, in this order:

1. **Task 3A — Narrow the registered quality arm to FQ-min.**
2. **Task 5 — Implement lean schema-v3 offline analysis and two figures.**
3. **Task 6 — Complete CPU qualification and freeze the preregistration.**
4. **Task 7 — Pass clean Vega CUDA and throughput gates.**
5. **Task 8 — Run matched 4×8 training and strict sampled evaluation.**
6. **Task 9 — Analyze, bank two figures, and render four explanatory videos.**

R0/R1/R2 reference recipes are not in the active sequence.

## Global constraints

- Continue in the isolated `first-strike-quality` worktree; never overwrite the
  user's main working tree.
- Fixed impedance only. Never call `set_gains` or command stiffness.
- Keep arm `Kp/Kd=1000/100`, except joint 2 at `1500/150`, and gripper
  `Kp/Kd=100/20`.
- Keep `IMP_J_LIMIT=[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]`.
- Keep `imp_max_p=0.0`; the impulse constraint remains log-only.
- Do not add a superlinear excess-over-reference reward.
- Do not edit installed `mjlab`, `rsl_rl`, `mujoco`, or `mujoco_warp`.
- Use named-file staging only; never use `git add -A`; add no co-author trailer.
- CPU-first. Vega is authorized only after Task 6 passes on the launch commit.
- Deploy tracked files through commit, push, and clean Vega pull; never `scp`
  tracked source.
- Any dirty evaluation row, `impossible_success_n>0`, `lambda_dead_n>0`,
  quality overflow, nonfinite quality geometry, or incomplete quota invalidates
  the complete campaign analysis.
- Use one GPU per training run.
- Use training seeds 8–15 in every arm and only `*_sampled` decision fields.
- Keep evaluator RNG identities `reset=2036072919`,
  `observation=2046072933`, and `action=2056072941`.
- Keep campaign ID `fq4x8`, shorts `f8/f0/d0/fq`, and the existing registered
  task IDs. Human-facing reports map `fq` to treatment label `FQ-min`.
- Freeze all analysis code and decision rules before training outcomes exist.
- Retry only documented infrastructure failures with identical configuration.
  Retain all attempts and never replace a completed weak or unstable seed.

## Frozen active treatment map

| Label | Registered task | Short | Impact reader/weight | Delivered reader/weight |
|---|---|---|---|---|
| F8 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear` | `f8` | raw linear / 8 | raw linear / 2 |
| F0 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0` | `f0` | raw linear / 0 | raw linear / 2 |
| D0 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0` | `d0` | raw linear / 8 | disabled / 0 |
| FQ-min | `Unitree-Z1-Hammer-CaT-Impulse-Event-Quality` | `fq` | quality-bounded / 8 | disabled / 0 |

FQ-min pays only:

\[
8\,q_{\mathrm{contact}}\,
\operatorname{clip}
\left(v_{\mathrm{pre}}/1.4598331451416016,0,1\right).
\]

Raw delivered impulse remains a recorded diagnostic. It is not an FQ-min
reward component or acceptance gate.

## File map

### Completed Tasks 1–4

- `src/tasks/hammer/mdp/contact_quality.py` — pure batched contact centroid,
  error, validity, overflow, and quality.
- `src/tasks/hammer/mdp/first_strike.py` — immutable onset/finalization record
  and optional passive quality instrumentation.
- `src/tasks/hammer/mdp/rewards.py` — one-shot event readers and quality-speed
  reader foundation.
- `src/tasks/hammer/mdp/__init__.py` — exports.
- `src/tasks/hammer/config/z1/env_cfgs.py` — quality sensor/config switches.
- `src/tasks/hammer/config/z1/__init__.py` — F8/F0/D0/quality task
  registrations.
- `scripts/eval_impulse.py` — sampled artifact schema v3 and quality channels.
- `docs/results/assets/2026-07-17_fixed_impedance_diag/probes/contact_quality_sensor.py`
  — CPU slot-count probe.
- `tests/test_first_strike_quality_reward.py`,
  `tests/test_first_strike_event.py`, `tests/test_configs.py`,
  `tests/test_eval_impulse_hook.py`, and `tests/test_first_strike_campaign.py`
  — completed kernel/tracker/config/recorder tests.

### Remaining active source and test files

- `src/tasks/hammer/mdp/rewards.py` — remove the quality-delivered class only if
  safe, otherwise leave it unregistered and explicitly deferred.
- `src/tasks/hammer/mdp/__init__.py` — remove an export only if the class is
  removed.
- `src/tasks/hammer/config/z1/env_cfgs.py` — FQ-min impact-only wiring.
- `src/tasks/hammer/config/z1/__init__.py` — FQ-min delivered weight zero while
  retaining the existing task ID.
- `tests/test_first_strike_quality_reward.py` — lean reward-surface tests.
- `tests/test_configs.py` — exact FQ-min config and D0/FQ-min difference tests.
- `evaluation/analysis/first_strike_campaign.py` — schema-v3 legacy reader
  compatibility.
- `evaluation/analysis/first_strike_quality_campaign.py` — frozen 4×8 contract,
  episode/seed summaries, inference, and decisions.
- `evaluation/analysis/plot_first_strike_quality_campaign.py` — exactly two
  static figures.
- `tests/test_first_strike_quality_campaign.py` — schema, aggregation,
  inference, decision, and plotting tests.
- `scripts/smoke_first_strike_instrumentation.py` and
  `tests/test_smoke_first_strike_instrumentation.py` — FQ-min CPU/CUDA
  predicates.
- `scripts/slurm/vega_train.sbatch` and `scripts/slurm/vega_eval.sbatch` —
  mandatory fail-closed `fq4x8` matrix, treatment, manifest, and routing
  checks.

### Remaining active records and result artifacts

- `docs/results/2026-07-26_first_strike_quality_prereg.md` — frozen protocol
  created before training.
- `docs/results/2026-07-27_first_strike_quality_result.md` — final result.
- `docs/results/assets/2026-07-27_first_strike_quality/summary.csv` — 32
  seed-summary rows.
- `docs/results/assets/2026-07-27_first_strike_quality/analysis.json` —
  recomputable statistics, gates, and identities.
- `docs/results/assets/2026-07-27_first_strike_quality/paired_seed_effects.png`.
- `docs/results/assets/2026-07-27_first_strike_quality/aggregate_nail_plane_contact_map.png`.
- `docs/results/assets/2026-07-27_first_strike_quality/explanatory_videos.tsv`
  — four provenance-bound video records; large MP4 bytes remain outside Git.

There is no active HTML report, gallery builder, reference-recipe probe,
trajectory grid, medoid grid, quality/speed frontier, or force/impulse grid.

---

### Task 1: Pure contact-quality kernel — complete

**Commit:** `7a55244 feat(hammer): add batched first-contact quality kernel`

**Completed contract:**

```python
def contact_point_quality(
    *,
    found: torch.Tensor,
    force_contact: torch.Tensor,
    position_w: torch.Tensor,
    nail_top_w: torch.Tensor,
    nail_axis_w: torch.Tensor,
    nail_radius_m: float,
    num_slots: int,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return centroid_w, radial_error, quality, valid, overflow."""
```

- [x] Weight retained contacts by nonnegative contact-frame normal force.
- [x] Set `overflow=found.amax(dim=1)>num_slots`.
- [x] Return invalid zero outputs for no positive normal force, nonfinite
  inputs, or overflow.
- [x] Compute quality exactly as
  `clip(1-(radial_error/nail_radius_m)**2,0,1)`.
- [x] Cover centered/edge/outside, multiple contacts, frame invariance,
  degeneracy, nonfinite input, and overflow in CPU tests.

Do not replace this kernel with a site-center threshold.

---

### Task 2: Dedicated sensor and immutable tracker snapshot — complete

**Commit:** `d891bbe feat(hammer): latch physical first-contact quality`

- [x] Add the dedicated `hammer_nail_quality` sensor with `found`, `force`,
  `pos`, and `normal`, leaving the task contact sensor shape unchanged.
- [x] Add passive `quality_instrumentation` that does not alter policy-facing
  tensors or physics.
- [x] Latch contact point/error/quality/validity/overflow, axiality,
  precontact axial speed, and first-contact time at accepted onset.
- [x] Latch productivity/reason, axial/transverse delivered impulse,
  depth-at-contact, and peak event depth at finalization.
- [x] Preserve raw trajectory, qvel, and Lambda channels in schema v3 so
  Task 5 can derive lateral speed, approach angle, depth gain, dwell,
  recontact, and safety summaries offline.
- [x] Prove onset immutability, finalization immutability, subset reset,
  no-force invalidity, and overflow invalidity.
- [x] Add the `8 -> 16 -> 64` CPU slot-count probe with the `1e-6 m`
  adjacent-qualified-count agreement gate.

If the frozen slot evidence cannot be reproduced, stop before Task 7.

---

### Task 3: Quality-reader/task foundation — complete but superseded in one respect

**Commit:** `ccf8bde feat(hammer): add bounded quality-conditioned strike rewards`

- [x] Add the bounded quality-speed reader using the shared one-shot
  `_consume` latch.
- [x] Register explicit F0, D0, and quality task IDs on fresh config objects.
- [x] Add exact action/actuator/gain/cap/config-signature tests.
- [x] Require F0 to differ from F8 only by impact weight and D0 only by
  delivered weight.
- [x] Require quality instrumentation for the registered quality treatment.

That commit also registered a quality-conditioned delivered reader. The lean
design supersedes that registration. Preserve the completed kernel, tracker,
task IDs, tests, and config-provenance work; execute Task 3A before treating the
quality task as launchable.

---

### Task 4: Schema-v3 sampled traces and physical-identity replay — complete

**Commits:**

- `e2fbf2b feat(eval): bank first-contact quality in sampled traces`
- `b63ea38 fix(eval): separate passive replay diagnostics`
- `ce40525 fix(eval): preserve raw quality slots`

- [x] Raise new sampled artifacts to schema version 3.
- [x] Record raw slot-level found counts, normal force, contact positions, and
  normals.
- [x] Record tracker quality/error/validity/overflow, first-contact time,
  axiality, transverse impulse, and frozen first-strike snapshot fields.
- [x] Include every new physical/event channel in canonical digests.
- [x] Bank separate training/evaluation config identities for the passive
  instrumentation override.
- [x] Prove fixed-action-tape physical identity across treatments and between
  instrumented/uninstrumented F8, excluding reward payout streams.
- [x] Preserve raw slots rather than max-force-only diagnostic projections.
- [x] Keep schema-v2 legacy behavior where its fields remain available; Task 5
  completes the schema-v3 analysis-reader migration.

Do not downgrade, reshape, or omit schema-v3 physical channels in later tasks.

---

### Task 3A: Narrow the registered quality arm to FQ-min

**Files:**

- Modify: `src/tasks/hammer/config/z1/env_cfgs.py`
- Modify: `src/tasks/hammer/config/z1/__init__.py`
- Modify: `src/tasks/hammer/mdp/rewards.py` only if removing dead code
- Modify: `src/tasks/hammer/mdp/__init__.py` only if removing its export
- Modify: `tests/test_first_strike_quality_reward.py`
- Modify: `tests/test_configs.py`

**Required final contract:**

- Registered task ID remains
  `Unitree-Z1-Hammer-CaT-Impulse-Event-Quality`.
- Manifest short remains `fq`; analysis label is `FQ-min`.
- `impact_progress.weight=8.0`.
- `impact_progress.func=FirstStrikeQualityImpactRewardTerm`.
- Its speed normalizer is exactly `1.4598331451416016`.
- `delivered_impulse.weight=0.0`.
- No registered FQ-min reward term calls a quality-conditioned delivered
  reader.
- The quality-delivered class may be removed, or may remain exported as
  unregistered deferred code. Minimal safe diff wins.
- D0 and FQ-min have the same disabled delivered treatment. They differ only in
  the speed reader/normalizer and the passive quality instrumentation needed by
  FQ-min.

- [ ] **Step 1: Write RED reward/config tests**

Require, for a productive finalized event:

```python
q_contact = 0.25
v_precontact = 3.0
expected_unweighted = 0.25
```

The quality-speed reader must return `0.25`, because speed saturates at one
before multiplication by quality. Require zero for invalid quality, no
contact, nonproductive event, and a second call. Prove a larger speed cannot
exceed `q_contact`.

Config tests must fail the current broad registration by requiring:

```text
FQ-min impact weight = 8
FQ-min delivered weight = 0
FQ-min impact reader = FirstStrikeQualityImpactRewardTerm
FQ-min delivered reader is not FirstStrikeQualityDeliveredRewardTerm
FQ-min speed normalizer = 1.4598331451416016
```

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_first_strike_quality_reward.py tests/test_configs.py -q
```

- [ ] **Step 3: Apply the minimal wiring change**

Keep the existing task ID and `event_quality` instrumentation path. Replace
only the active treatment wiring needed to disable delivered reward and set
the FQ-min speed scale. Do not change tracker timing, event productivity,
actions, gains, caps, CaT, or evaluation instrumentation.

- [ ] **Step 4: Prove exact allowed differences**

Serialize reward, tracker, sensor, action, actuator, cap, and CaT signatures:

```text
F8 vs F0: impact weight only
F8 vs D0: delivered weight only
D0 vs FQ-min: impact reader, its normalizer, and required passive quality sensor
D0 and FQ-min: delivered weight 0 and no active quality-delivered reader
all arms: identical actions, actuators, gains, caps, imp_max_p, and budgets
strict evaluation: passive quality instrumentation enabled for every arm
```

- [ ] **Step 5: Run focused tests and commit named files**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_first_strike_quality_reward.py tests/test_configs.py -q
git add src/tasks/hammer/config/z1/env_cfgs.py \
        src/tasks/hammer/config/z1/__init__.py \
        src/tasks/hammer/mdp/rewards.py \
        src/tasks/hammer/mdp/__init__.py \
        tests/test_first_strike_quality_reward.py \
        tests/test_configs.py
git commit -m "fix(hammer): narrow FQ to quality-gated speed"
```

Before staging, omit any listed source file that did not actually change.

---

### Task 5: Lean schema-v3 analysis, inference, and two figures

**Files:**

- Create: `evaluation/analysis/first_strike_quality_campaign.py`
- Create: `evaluation/analysis/plot_first_strike_quality_campaign.py`
- Create: `tests/test_first_strike_quality_campaign.py`
- Modify: `evaluation/analysis/first_strike_campaign.py`
- Modify: `tests/test_first_strike_campaign.py`

**Required interfaces:**

```python
def exact_paired_sign_flip(
    treatment: np.ndarray,
    control: np.ndarray,
) -> dict: ...

def holm_adjust(
    p_values: Mapping[str, float],
) -> dict[str, float]: ...

def paired_bootstrap_ratio(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    samples: int = 100_000,
    seed: int = 20260726,
) -> dict: ...

def analyze_quality_campaign(
    rows: Sequence[Mapping],
    accepted_evaluation_manifest: Sequence[Mapping],
) -> dict: ...
```

Freeze:

- campaign `fq4x8`;
- treatments `F8/F0/D0/FQ-min`, mapped to shorts `f8/f0/d0/fq`;
- seeds 8–15;
- 500 iterations, 4096 training environments, `model_499.pt`;
- 256×2 strict sampled evaluation;
- FQ-min impact weight 8, delivered weight 0, and only the quality-speed
  reader active.

- [ ] **Step 1: Write RED matrix, manifest, and schema tests**

Reject:

- missing or duplicate treatment/seed identities;
- the wrong task ID or `fq` compatibility mapping;
- FQ-min delivered weight above zero or a registered quality-delivered reader;
- wrong weights, readers, speed normalizer, gains, caps, `imp_max_p`, training
  budget, checkpoint name, or evaluation quota;
- dirty/unknown code or asset provenance;
- missing or mismatched checkpoint, config, manifest, payload, trace, or asset
  hashes;
- a non-schema-v3 quality artifact;
- any non-`*_sampled` decision field;
- any quality, liveness, impossible-success, nonfinite, or quota sentinel;
- analysis before all 32 accepted-evaluation rows verify.

The legacy reader must accept schema v3 using the schema-v3 digest projection
while preserving schema-v2 behavior where legacy fields exist.

- [ ] **Step 2: Write RED episode/seed aggregation tests**

Aggregate, per episode and then per training seed:

```text
all-episode contact quality and radial error
nail-plane contact coordinates
onset axial speed, lateral speed, contact-normal axiality, approach angle
useful first-window speed
first-window and overall success
event-window nail-depth gain
contact dwell in milliseconds
recontact count
raw delivered impulse
peak qvel, rail exceedance, per-joint Lambda, worst Lambda/cap
```

Definitions:

```text
event_window_depth_gain = max(peak_depth_inside_event - depth_at_contact, 0)
contact_dwell_ms = raw_contact_substeps_inside_event * physics_dt * 1000
recontact_count = off_to_on_edges_after_onset_before_finalization
```

No-contact and zero-positive-normal-force physical episodes have all-episode
quality zero. Overflow, nonfinite geometry, or a missing required snapshot
invalidates the complete evaluation row.

- [ ] **Step 3: Write RED inference and decision tests**

Primary quality contrasts are:

```text
F0 - F8
D0 - F8
FQ-min - D0
```

Require the repository exact two-sided Mann–Whitney test on seed summaries and
one Holm correction across exactly those three p-values. With eight strictly
positive paired differences, require the exact two-sided sign-flip sensitivity
`p=2/256=0.0078125`, but prove that changing this sensitivity p-value cannot
pass or fail a causal decision or the FQ-min practical rule.

Test tied/zero differences, mismatched seed identities, Holm ordering,
100,000-resample PCG64 reproducibility, strict ratio boundaries, and
nonpositive/nonfinite denominator failure.

FQ-min practical acceptance versus F8 requires:

```text
mean all-episode contact-quality gain >= 0.10
one-sided useful-speed ratio lower bound > 0.95
first-window and overall success >= 0.90
first-window and overall success no more than 0.05 below F8
one-sided event-window-depth-gain ratio lower bound > 0.90
all safety/provenance/quota/liveness/numerical/sentinel gates pass
```

Raw delivered impulse must not appear in that gate.

- [ ] **Step 4: Implement the minimal analysis**

Reuse artifact verification, `exact_seed_tests`, and shared episode helpers
from `evaluation.analysis.first_strike_campaign`. Bootstrap matched seed blocks,
never arms independently. Emit all eight paired differences for every primary
contrast and every practical margin.

For F8 versus D0, emit paired intervals for depth gain, dwell, recontact, raw
delivered impulse, and both success endpoints. Permit the phrase “mainly
selected dwell/recontact” only under the four mechanism conditions frozen in
the design spec. These endpoints do not enlarge the Holm family.

- [ ] **Step 5: Implement exactly two static figures**

1. `paired_seed_effects.png`: paired seed points/lines or compact intervals for
   the three causal contrasts plus the supporting FQ-min/F8 practical margins.
2. `aggregate_nail_plane_contact_map.png`: all valid accepted contacts by arm
   in the common nail-plane basis, identical axes, compiled nail outline, and
   no site-center substitution.

Do not implement HTML, trajectory plots, representative-episode grids,
quality/speed frontiers, force/impulse grids, or video selection in Task 5.

- [ ] **Step 6: Run tests and inspect synthetic figures**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_first_strike_campaign.py \
            tests/test_first_strike_quality_campaign.py -q
```

Render synthetic fixtures and inspect both PNGs. Require unclipped labels,
identical nail-plane scales, correct arm mapping, and no hidden invalid rows.

- [ ] **Step 7: Commit Task 5**

```bash
git add evaluation/analysis/first_strike_campaign.py \
        evaluation/analysis/first_strike_quality_campaign.py \
        evaluation/analysis/plot_first_strike_quality_campaign.py \
        tests/test_first_strike_campaign.py \
        tests/test_first_strike_quality_campaign.py
git commit -m "feat(analysis): add lean first-contact quality campaign"
```

---

### Task 6: CPU qualification and preregistration

**Files:**

- Modify: `scripts/smoke_first_strike_instrumentation.py`
- Modify: `tests/test_smoke_first_strike_instrumentation.py`
- Create: `docs/results/2026-07-26_first_strike_quality_prereg.md`

- [ ] **Step 1: Reproduce the frozen FQ-min scale**

Using the locally banked bytes, require:

```text
raw bank SHA-256 =
ea4a82e007d95cf7ff962091ba0d6ff3e07e639f4cc1c2d0feab9fe368f47ac8
resolved manifest SHA-256 =
69d5bc66f2127d0a5b29463b35763092d388bc0e71661b06fdd81b0eea8d661f
qualification SHA-256 =
641e520c0cd35932175918d0bf48c7b6df07749ff1462b3d7da7310d76338b1b
NumPy quantile method = higher
q90 = 1.4598331451416016
calibration saturations = 26/256
group-held-out validation saturations = 33/128
```

Any mismatch fails closed. F8/F0/D0 keep the `1.0` raw speed normalizer.
Also verify the existing production-tracker provenance for
`I_REF_FIRST_STRIKE_SUCCESS=0.3088 N·s` and exact F8/F0 delivered-reader config
identity; do not recalibrate that normalizer.

- [ ] **Step 2: Extend smoke predicates for FQ-min**

Require finite observations/rewards, one immutable quality snapshot, no
overflow, one productive event, one quality-speed payout for FQ-min, zero
delivered payout for D0 and FQ-min, raw-reader behavior for F8/F0/D0, exact
registered task/config identities, and unchanged physical channels.

- [ ] **Step 3: Run the strict CPU suite**

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

Require pytest green, `validate_rewards.py` phases A–M green, contact sensor
verified, random-policy sweep green, reference playback feasible, the frozen
slot-count evidence reproduced, and no quality sentinel.

- [ ] **Step 4: Do not re-evaluate the archived policy zoo**

There is no archived C/D-prime/F/E GPU re-evaluation, counterfactual policy-zoo
ranking, medoid audit, or archived force-grid gate. Qualification uses the
frozen local provenance bank, pure geometry tests, schema-v3 fixtures, CPU
reference playback, and fixed-action-tape identity.

- [ ] **Step 5: Freeze the preregistration**

Record:

```text
compatibility mapping: fq -> FQ-min
task IDs, reader classes, weights, and exact config hashes
seeds 8..15
500 iterations, 4096 environments, model_499.pt
256 environments x 2 sampled episodes
schema-v3 artifact and invalidation contract
three primary causal contrasts
MWU as the sole Holm-adjusted decision family
exact paired sign-flip as sensitivity only
PCG64 seed 20260726 and 100,000 matched-seed bootstrap samples
contact-quality endpoint and zero/invalid semantics
D0 depth/dwell/recontact/raw-delivered/success mechanism definitions
FQ-min practical margins, excluding delivered impulse
V_FQ hashes, higher-q90 method, and both saturation counts
qvel/Lambda interpretation and simulator-only label
accepted-training and accepted-evaluation manifest contracts
exactly two static figures and four post-statistics explanatory videos
```

- [ ] **Step 6: Independent review and commit**

Have one reviewer inspect reward/config/tracker semantics and one inspect
statistics/provenance/figures. Resolve findings with evidence, rerun affected
tests, then:

```bash
git add scripts/smoke_first_strike_instrumentation.py \
        tests/test_smoke_first_strike_instrumentation.py \
        docs/results/2026-07-26_first_strike_quality_prereg.md
git commit -m "docs(hammer): preregister lean first-contact quality campaign"
```

---

### Task 7: Clean Vega CUDA and throughput gate

**Files:**

- Modify: `scripts/slurm/vega_train.sbatch`
- Modify: `scripts/slurm/vega_eval.sbatch`

- [ ] **Step 1: Push and deploy by provenance**

Push named commits. On Vega, use a fresh clean checkout at the exact commit and
verify both this repository and the sibling asset repository are clean at the
recorded 40-hex revisions.

- [ ] **Step 2: Run one CUDA smoke per arm**

Use 256 environments for F8, F0, D0, and FQ-min. Require finite
observations/rewards, productive first events, no quality overflow,
`impossible_success_n==0`, `lambda_dead_n==0`, exact task IDs/readers/weights,
and these payout predicates:

```text
F8: raw speed and raw delivered active
F0: speed zero, raw delivered active
D0: raw speed active, delivered zero
FQ-min: bounded quality-speed active, delivered zero
```

- [ ] **Step 3: Measure passive-sensor throughput**

Measure at least 1,000 post-warmup control steps at 4096 environments:

```text
F8 without quality sensor
F8 with passive quality sensor
FQ-min with passive quality sensor and quality-speed reader
```

Record steps/s, GPU memory, overflow, and config hashes. FQ-min must reach at
least 80% of uninstrumented F8 throughput with no overflow. If it fails, stop
and profile; do not launch 32 jobs.

- [ ] **Step 4: Freeze launcher and smoke evidence**

The launcher contract requires:

```text
campaign = fq4x8
seeds = 8 9 10 11 12 13 14 15
shorts = f8 f0 d0 fq
iterations = 500
IMPACT_W unset
DELIVERED_W unset
NAIL_DRIVEN_W unset
one GPU per run
```

Both Slurm scripts currently have campaign-specific fail-closed handling only
for the earlier `fsr4x8` campaign. Add an explicit `fq4x8` branch; generic
argument acceptance is not sufficient. The training branch must reject wrong
task/short/seed/iteration mappings and any inherited `IMPACT_W`,
`DELIVERED_W`, or `NAIL_DRIVEN_W`. The evaluation branch must require the
32-row accepted-training manifest, route attempts beneath the external
evaluation root exactly once, and reject any treatment/config mismatch.

Bank JSON evidence with code/asset/config hashes, predicate counts, throughput,
memory, and clean states. Commit only the small summary/manifest and any
necessary launcher checks.

---

### Task 8: Matched 4×8 training and strict evaluation

**Files:**

- No source changes after launch.
- Create result artifacts under
  `docs/results/assets/2026-07-27_first_strike_quality/`.

**Output:** 32 accepted training checkpoints, 32 accepted strict sampled
evaluations, and 16,384 verified sampled episodes.

- [ ] **Step 1: Submit 32 training jobs**

From the clean Vega checkout:

```bash
campaign_code_rev=$(git rev-parse HEAD)
campaign_asset_rev=$(git -C ../safe_impact_manipulation rev-parse HEAD)
unset IMPACT_W DELIVERED_W NAIL_DRIVEN_W

CAMPAIGN=fq4x8 SEEDS="8 9 10 11 12 13 14 15" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Linear \
  SINGLE_SHORT=f8 ITERS=500 \
  EXPECTED_CODE_REVISION="$campaign_code_rev" \
  EXPECTED_ASSET_REVISION="$campaign_asset_rev" \
  sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_train.sbatch

CAMPAIGN=fq4x8 SEEDS="8 9 10 11 12 13 14 15" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0 \
  SINGLE_SHORT=f0 ITERS=500 \
  EXPECTED_CODE_REVISION="$campaign_code_rev" \
  EXPECTED_ASSET_REVISION="$campaign_asset_rev" \
  sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_train.sbatch

CAMPAIGN=fq4x8 SEEDS="8 9 10 11 12 13 14 15" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0 \
  SINGLE_SHORT=d0 ITERS=500 \
  EXPECTED_CODE_REVISION="$campaign_code_rev" \
  EXPECTED_ASSET_REVISION="$campaign_asset_rev" \
  sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_train.sbatch

CAMPAIGN=fq4x8 SEEDS="8 9 10 11 12 13 14 15" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Quality \
  SINGLE_SHORT=fq ITERS=500 \
  EXPECTED_CODE_REVISION="$campaign_code_rev" \
  EXPECTED_ASSET_REVISION="$campaign_asset_rev" \
  sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_train.sbatch
```

The task ID remains `...Event-Quality`, but the frozen config hash must prove
that `fq` is FQ-min: impact weight 8, quality-speed reader at
`1.4598331451416016`, and delivered weight zero.

- [ ] **Step 2: Monitor without selecting on performance**

Report pending/running/completed/failed/retried counts by treatment. Retry only
documented infrastructure failure with identical config and seed. Retain all
attempts. Never replace a completed weak or unstable seed.

- [ ] **Step 3: Freeze `accepted_training_checkpoints.tsv`**

Freeze exactly 32 rows. Each binds:

```text
treatment label and compatibility short
registered task and training seed
model_499.pt path and SHA-256
training attempt plus full retry history
code, asset, campaign-config, and treatment-config identities
fixed action/impedance/cap signatures
clean state and disposition
```

Do not include evaluation-attempt fields in this manifest.

- [ ] **Step 4: Run strict sampled evaluation**

For each accepted checkpoint, collect exactly the first two completed
stochastic episodes from each of 256 environments under matched evaluator RNG
streams, ±0.05 rad reset noise, and the frozen observation-corruption contract.
Enable passive `quality_instrumentation` for all arms and bank its separate
evaluation config hash.

The evaluator must use:

```text
reset RNG = 2036072919
observation RNG = 2046072933
action RNG = 2056072941
```

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

- [ ] **Step 5: Freeze `accepted_evaluations.tsv`**

After infrastructure-only retries, freeze exactly 32 accepted rows. Bind each
training row to its evaluation attempt, code/asset/config identities, RNG
streams, schema-v3 payload digest and byte hash, retry history, dirty state,
512-episode quota, and every sentinel predicate. Do not run Task 9 until all 32
identities and all 16,384 episode records verify.

---

### Task 9: Analyze, bank the result, and render four explanatory videos

**Files:**

- Create: `docs/results/2026-07-27_first_strike_quality_result.md`
- Create: `docs/results/assets/2026-07-27_first_strike_quality/summary.csv`
- Create: `docs/results/assets/2026-07-27_first_strike_quality/analysis.json`
- Create:
  `docs/results/assets/2026-07-27_first_strike_quality/paired_seed_effects.png`
- Create:
  `docs/results/assets/2026-07-27_first_strike_quality/aggregate_nail_plane_contact_map.png`
- Create:
  `docs/results/assets/2026-07-27_first_strike_quality/explanatory_videos.tsv`

- [ ] **Step 1: Verify manifests and freeze the statistical result**

Recompute every artifact hash and schema-v3 episode summary from the frozen
accepted-evaluation manifest. Emit:

```text
32 seed-summary rows
three exact MWU tests and one Holm-adjusted decision family
exact paired sign-flip sensitivity for each contrast
all eight matched-seed differences
100,000-sample paired-bootstrap intervals and one-sided ratio bounds
F8/D0 depth, dwell, recontact, raw delivered, and success mechanism results
FQ-min/F8 practical acceptance margins
onset axial/lateral/angle diagnostics
qvel and Lambda legality/transport diagnostics
all quota, provenance, liveness, numerical, and sentinel predicates
```

Freeze `analysis.json` and the pass/fail decisions before rendering any video.
Delivered impulse remains descriptive for FQ-min acceptance.

- [ ] **Step 2: Generate and inspect exactly two figures**

Render `paired_seed_effects.png` and
`aggregate_nail_plane_contact_map.png`. Open both locally and verify scales,
labels, arm mapping, episode counts, and correspondence with `analysis.json`.
Do not create HTML or additional decision figures.

- [ ] **Step 3: Independent result review**

One reviewer recomputes statistics from CSV/schema-v3 artifacts. Another
reviews the contact map, treatment identities, and physical interpretation.
Neither reviewer edits the result before reporting findings.

- [ ] **Step 4: Render exactly one predeclared medoid video per arm**

Only after Step 1 decisions are frozen, select one episode per treatment over
that arm's complete verified sampled set. Use minimum standardized Euclidean
distance to the arm median over:

```text
contact quality
useful speed
first-contact time
raw delivered impulse
nail-plane axial contact coordinate
nail-plane lateral contact coordinate
```

Replace a zero/nonfinite scale by `1.0`. Break exact ties by
`(training_seed, env_id, episode_ordinal)`. Freeze the four selected identities
before rendering.

Replay each stored action tape under its bound reset/config. Record treatment,
seed, episode identity, source trace digest, action-tape digest,
frame-to-substep timing, MP4 SHA-256, and selection distance in
`explanatory_videos.tsv`. The videos explain observed behavior; they cannot
change a decision, replace an episode, or supply quantitative measurements.

Render exactly four MP4s: one F8, one F0, one D0, and one FQ-min. Large MP4
bytes stay outside Git. There is no all-seed gallery, deterministic-checkpoint
gallery, annotation UI, montage requirement, or HTML wrapper.

- [ ] **Step 5: Write the result**

Use:

```text
What was supposed to happen?
What actually happened?
Why was there a difference?
What can we learn from this?
```

Separate proven, supported, not distinguishable, simulator-only, and open
claims. A null is not equivalence. A policy with finite qvel-rail exceedance is
not hardware-speed-qualified. A log-only impulse result is not impulse
enforcement.

- [ ] **Step 6: Final verification and named-file commit**

Re-run tests appropriate to every changed source file, recompute every hash,
and verify the two figure bytes and four video-manifest rows. Stage only the
reviewed result doc, CSV, JSON, two PNGs, video manifest, and any final reviewed
source/tests:

```bash
git diff --check
git status --short
```

Never stage large MP4s, transient caches, or unrelated worktree changes.

## Deferred appendix: R0/R1/R2 reference recipes

This appendix is deliberately not a numbered task. Do not create its files,
run its 96 CPU rollouts, add it to qualification, or launch a reference-guided
GPU arm. Revisit it only after Task 9 identifies a residual terminal
credit-assignment problem that the lean reward comparison did not resolve and
a separate protocol is approved.

Preserved calibrated design:

- R0 is the current `SingleStrikeReference`.
- R1 uses the `traj_dc` transverse residual, bound to SHA-256
  `b22dabb94a10a1e7f68f3fe6a4a2f9412e14916a40a3dbafc81e2f3c3cc7f89f`, with
  R0 axial schedule and pacing.
- Source traces lack pre-apex motion, so pre-apex residual is exactly zero and
  the apex repeats at the segment boundary.
- Resample post-apex descent on 51 uniform points in the live nail frame.
  Choose the residual minimizing summed pairwise transverse L2 distance; ties
  break by trace digest.
- R2 equals R1 through 90 mm axial standoff and multiplies its transverse
  residual by
  `1-(10*u**3-15*u**4+6*u**5)`, with
  `u=clip((0.09-d)/0.07,0,1)`, reaching zero by 20 mm with continuous first and
  second derivatives.
- Generate a fresh 32-row reset manifest for exact seeds `0..31` before any
  outcome. Bind realized qpos/qvel, live head/nail poses, nail-frame basis,
  code/config/asset revisions, and canonical state digest. Every recipe must
  reproduce every row.

Corrected deadline contract:

```text
accepted productive onset no later than the final nominal script step
at most two post-script final-target control steps
normal task success no later than the end of n_script + 2
stop earlier on success
no endpoint hold
late onset, late success, press-only contact, or missing productive event fails
```

Candidate promotion gates:

```text
productive in-script first events >= 30/32
accepted onsets inside provisional 12 mm proxy >= 29/32
median v_pre >= 0.95 * R0
every accepted event v_pre > 0.5 m/s
median raw delivered >= 0.90 * R0
no prefix arm qvel > 3.1415 rad/s
no Lambda/cap > 1
no numerical or sentinel failure
```

Freeze terminal axial-speed, terminal lateral-speed, and approach-angle
thresholds in the fresh manifest before inspecting R1/R2 outcomes. Measure all
three in the same live-nail-frame window ending immediately before accepted
onset. Missing, nonfinite, wrong-window, or failed values fail promotion.

R2 additionally needs at least four more within-12-mm onsets than R1.
Replacing R0 additionally needs at least 5% higher precontact speed or at least
10% lower p95 prefix qvel. Delivered impulse alone and global straightness
cannot justify replacement.

Before a separately authorized GPU reference arm, require the generating
recipe to have the largest cumulative raw imitation score in at least 26/32
resets and median self/next-best ratio at least 1.25. Re-run R0 and any candidate
at `solref_scale=2`; reject on a 20% or larger change in speed, raw delivered
impulse, or worst-joint impulse, or on any success, centering, deadline, or
terminal-gate failure. A CPU pass would prove scripted feasibility and weak
prior separability only, not PPO learnability.
