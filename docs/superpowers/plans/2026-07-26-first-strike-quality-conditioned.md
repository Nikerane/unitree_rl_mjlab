# First-Strike Quality-Conditioned Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a fixed-impedance, physics-rate first-contact-quality reward in a matched F8/F0/D0/FQ 4×8 campaign, while separately qualifying three reference recipes on CPU.

**Architecture:** Add one dedicated multi-slot contact sensor and a small pure Torch quality kernel. The existing first-strike tracker latches the resulting contact point and quality at the accepted onset; two new bounded one-shot readers implement FQ without changing F8/F0/D0 physics. Extend the existing sampled evaluator, but keep the new campaign matrix/statistics and plotting in focused quality-campaign modules. A separate CPU probe compares R0/R1/R2 reference recipes before any reference-guided GPU arm.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0, MuJoCo/mujoco_warp 3.8.1, NumPy, SciPy-compatible exact inference implemented in-tree, Plotly plus static PNG export, pytest, Slurm/Vega.

## Global Constraints

- Work from a clean isolated branch/worktree created from commit `3286a45`; never overwrite the user's dirty working tree.
- Fixed impedance only. Never call `set_gains` or command stiffness.
- `IMP_J_LIMIT` stays `[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]`.
- `imp_max_p` stays `0.0`; the impulse constraint remains log-only.
- Do not add a superlinear excess-over-`i_ref` reward.
- Do not edit installed `mjlab`, `rsl_rl`, `mujoco`, or `mujoco_warp`.
- Use named-file staging only; never `git add -A`; no co-author trailer.
- CPU-first. Vega is allowed only after the quality-reader CPU probe and required tests pass.
- Deploy tracked code through commit, push, and clean Vega pull; never `scp` tracked files.
- Any `-dirty` evaluation row, `impossible_success_n > 0`, or `lambda_dead_n > 0` is invalid.
- Use one GPU per run.
- Use training seeds `8..15` in every arm and only `*_sampled` evaluation fields.
- Freeze campaign ID `fq4x8` and run shorts `f8`, `f0`, `d0`, `fq`.

---

## File map

### New files

- `src/tasks/hammer/mdp/contact_quality.py` — pure batched contact-centroid and quality math.
- `tests/test_first_strike_quality_reward.py` — kernel, tracker, and FQ reader unit tests.
- `docs/results/assets/2026-07-17_fixed_impedance_diag/probes/contact_quality_sensor.py` — cheap CPU sensor/slot-count probe.
- `evaluation/analysis/first_strike_quality_campaign.py` — frozen 4×8 contract, quality metrics, paired inference, and decision rules.
- `evaluation/analysis/plot_first_strike_quality_campaign.py` — fixed-scale static evidence and optional HTML.
- `evaluation/analysis/build_first_strike_review_gallery.py` — provenance-bound video/graph review library.
- `tests/test_first_strike_quality_campaign.py` — trace schema, aggregation, paired inference, and plotting tests.
- `tests/test_first_strike_review_gallery.py` — video timing, digest, medoid, and annotation isolation tests.
- `evaluation/trajectory/reference_recipe_probe.py` — matched R0/R1/R2 CPU
  replay and promotion-gate analysis.
- `tests/test_reference_recipe_probe.py` — recipe, reset-manifest, and
  promotion-contract tests.
- `docs/results/2026-07-26_reference_recipe_probe.md` — provenance-bound CPU
  recipe result.
- `docs/results/2026-07-26_first_strike_quality_prereg.md` — frozen experiment protocol before training.
- `docs/results/2026-07-27_first_strike_quality_result.md` — result record created only after the campaign.

### Existing files modified

- `src/tasks/hammer/mdp/__init__.py` — export the quality kernel/readers.
- `src/tasks/hammer/mdp/first_strike.py` — latch immutable quality/contact diagnostics.
- `src/tasks/hammer/mdp/rewards.py` — two bounded quality-conditioned one-shot readers.
- `src/tasks/hammer/config/z1/env_cfgs.py` — dedicated contact-quality sensor and F0/D0/FQ config switches.
- `src/tasks/hammer/config/z1/__init__.py` — register F0, D0, and FQ task IDs.
- `tests/test_first_strike_event.py` — tracker phase/reset/overflow tests.
- `tests/test_configs.py` — exact task/reward/sensor signatures.
- `scripts/eval_impulse.py` — sampled schema v3 contact-quality channels.
- `tests/test_eval_impulse_hook.py` — physical trace and snapshot serialization checks.
- `scripts/smoke_first_strike_instrumentation.py` — F0/D0/FQ instrumentation expectations.
- `tests/test_smoke_first_strike_instrumentation.py` — CPU/CUDA smoke predicates.
- `evaluation/analysis/first_strike_campaign.py` — schema-v3 legacy-analysis
  migration owned by Task 5.
- `evaluation/analysis/plot_first_strike_campaign.py` — only shared style helpers if reuse cannot be achieved by import.
- `scripts/slurm/vega_train.sbatch` and `scripts/slurm/vega_eval.sbatch` — add exact F0/D0/FQ task allow-list entries only if current generic arguments reject them.
- `docs/superpowers/specs/2026-07-26-first-strike-quality-conditioned-design.md` — record replay-qualified slot/overflow semantics.

---

### Task 1: Pure contact-quality kernel

**Files:**
- Create: `src/tasks/hammer/mdp/contact_quality.py`
- Create: `tests/test_first_strike_quality_reward.py`
- Modify: `src/tasks/hammer/mdp/__init__.py`

**Interfaces:**
- Produces:

```python
def contact_point_quality(
    *,
    found: torch.Tensor,           # [B, N], total-match count per retained slot
    force_contact: torch.Tensor,   # [B, N, 3], MuJoCo contact frame
    position_w: torch.Tensor,      # [B, N, 3]
    nail_top_w: torch.Tensor,      # [B, 3]
    nail_axis_w: torch.Tensor,     # [3] or [B, 3]
    nail_radius_m: float,
    num_slots: int,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return centroid_w[B,3], radial_error[B], quality[B], valid[B], overflow[B]."""
```

- `valid=False`, zero centroid/error/quality when no positive normal force,
  nonfinite input, or `overflow=True`.
- `overflow = found.amax(dim=1) > num_slots`.
- Overflow/nonfinite contact geometry fails closed to zero online payout but
  increments a sentinel; strict evaluation invalidates the complete row rather
  than treating instrumentation failure as a legitimate zero-quality episode.
- The normal-force weight is
  `where(found > 0, force_contact[..., 0].clamp_min(0), 0)`.
- Quality is exactly
  `clip(1 - (radial_error / nail_radius_m)**2, 0, 1)`.

- [ ] **Step 1: Write failing shape, value, and guard tests**

Add tests covering one centered contact, one edge contact, one outside contact,
two weighted contacts, world-frame translation, nail-axis rotation, no contact,
zero/negative normal force, NaN input, and overflow. The central assertion is:

```python
centroid, error, quality, valid, overflow = contact_point_quality(
    found=torch.tensor([[1.0, 1.0]]),
    force_contact=torch.tensor([[[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
    position_w=torch.tensor([[[0.0, 0.0, 0.1], [0.008, 0.0, 0.1]]]),
    nail_top_w=torch.tensor([[0.0, 0.0, 0.1]]),
    nail_axis_w=torch.tensor([0.0, 0.0, -1.0]),
    nail_radius_m=0.012,
    num_slots=2,
)
assert centroid[0, 0].item() == pytest.approx(0.002)
assert error[0].item() == pytest.approx(0.002)
assert quality[0].item() == pytest.approx(1.0 - (0.002 / 0.012) ** 2)
assert valid.tolist() == [True]
assert overflow.tolist() == [False]
```

- [ ] **Step 2: Run the kernel tests and confirm RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_first_strike_quality_reward.py -q
```

Expected: import failure for `contact_point_quality`.

- [ ] **Step 3: Implement the minimal vectorized kernel**

Use only Torch tensor operations. Normalize `nail_axis_w`, project
`centroid-nail_top` with
`offset - (offset*axis).sum(-1, keepdim=True)*axis`, and apply validity with
`torch.where`. Do not loop over environments or slots and do not synchronize
device tensors to the host.

- [ ] **Step 4: Run tests and confirm GREEN**

Run the Task 1 test file. Require every case to pass on CPU.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/tasks/hammer/mdp/contact_quality.py \
        src/tasks/hammer/mdp/__init__.py \
        tests/test_first_strike_quality_reward.py
git commit -m "feat(hammer): add batched first-contact quality kernel"
```

---

### Task 2: Dedicated sensor and immutable tracker snapshot

**Files:**
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py:134-147`
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py:278-294`
- Modify: `src/tasks/hammer/mdp/first_strike.py:29-206`
- Modify: `tests/test_first_strike_event.py`
- Create: `docs/results/assets/2026-07-17_fixed_impedance_diag/probes/contact_quality_sensor.py`

**Interfaces:**
- Produces a dedicated `hammer_nail_quality` `ContactSensorCfg` with:
  - the same hammer-face/nail-body match as `hammer_nail_contact`;
  - fields `("found", "force", "pos", "normal")`;
  - `reduce="maxforce"`;
  - replay-qualified `num_slots` chosen from `8`, `16`, then `64`;
  - `track_air_time=False`;
  - contact-frame force (`global_frame=False`).
- Adds an orthogonal `quality_instrumentation: bool=False` configuration mode
  that wires only this passive sensor and its diagnostic tracker fields. It
  must not change policy observations, actions, rewards, terminations, or
  physics. FQ training requires it; all F8/F0/D0/FQ action-tape audits and
  strict sampled evaluations enable it.
- Extends `FirstStrikeEventTracker` with public tensors:

```python
contact_point_w: torch.Tensor       # [B, 3]
contact_error_m: torch.Tensor       # [B]
contact_quality: torch.Tensor       # [B]
contact_quality_valid: torch.Tensor # [B] bool
contact_quality_overflow: torch.Tensor # [B] bool
first_contact_time_s: torch.Tensor  # [B]
contact_normal_axiality: torch.Tensor # [B]
delivered_transverse: torch.Tensor  # [B]
event_impulse_per_joint: torch.Tensor # [B, J]
event_peak_abs_qvel: torch.Tensor    # [B, J]
```

- Preserves a two-stage immutable record:
  - onset latches contact point/error/quality, contact normal/axiality,
    precontact speed, signed offsets, and first-contact time;
  - finalization latches productivity/reason, delivered axial/transverse
    impulse, per-joint event impulse, and peak event qvel;
  - onset fields cannot change at finalization, and no finalized field can
    change afterward.

- [ ] **Step 1: Extend the fake tracker fixture and write failing tests**

Give the fixture a separate quality sensor with mutable `found`, `force`,
`pos`, and `normal`; give the nail entity `site_pos_w`; give the fake simulator
a compiled `nail_block/nail_head` radius of `0.012`.

Test that:

```python
step(..., quality_pos=(0.003, 0.0, 0.1), quality_normal_force=10.0)
assert tracker.contact_error_m[0].item() == pytest.approx(0.003)
assert tracker.contact_quality[0].item() == pytest.approx(0.9375)
assert tracker.contact_quality_valid[0]
```

Also test onset-only latching, the two-stage onset/finalization boundary,
finalization freeze, subset reset, no-force invalidity, overflow invalidity,
first-contact time, axiality, and transverse event-impulse integration.

- [ ] **Step 2: Run tracker tests and confirm RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_first_strike_event.py \
  tests/test_first_strike_quality_reward.py -q
```

Expected: missing quality sensor/config/properties.

- [ ] **Step 3: Wire the dedicated sensor and tracker fields**

Keep the existing `hammer_nail_contact` tensor shape unchanged. Cache the
compiled nail radius once in tracker construction from
`env.sim.mj_model.geom("nail_block/nail_head").size[0]`. At `onset`, call
`contact_point_quality` and latch only the onset-phase outputs with
`torch.where`; never inspect tensor truth values in Python. Latch accumulated
delivered/transverse/per-joint impulse and peak-qvel fields only when the event
finalizes.

Increment a per-environment substep counter before onset testing and latch
`first_contact_time_s = counter * env.physics_dt`. Integrate transverse
object-side impulse only while the event is active and raw contact is present.

- [ ] **Step 4: Add and run the CPU slot-count probe**

The probe builds one `play=True` environment for each candidate slot count
`8`, `16`, `64`, executes the existing scripted reference, and emits JSON with:

```text
candidate_slots
contact_substeps
max_reported_found
overflow_n
quality_valid_n
contact_error_m
contact_quality
normal_force_sum
wall_seconds
```

Freeze the smallest candidate with `overflow_n==0` whose centroid/error agrees
with the next larger candidate to `atol=1e-6 m`. If `64` overflows or the
agreement gate fails, stop the experiment rather than approximating.

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/results/assets/2026-07-17_fixed_impedance_diag/probes/contact_quality_sensor.py \
  --out /tmp/contact_quality_sensor_cpu.json
```

- [ ] **Step 5: Run Task 2 tests and commit**

```bash
git add src/tasks/hammer/config/z1/env_cfgs.py \
        src/tasks/hammer/mdp/first_strike.py \
        tests/test_first_strike_event.py \
        tests/test_first_strike_quality_reward.py \
        docs/results/assets/2026-07-17_fixed_impedance_diag/probes/contact_quality_sensor.py
git commit -m "feat(hammer): latch physical first-contact quality"
```

---

### Task 3: FQ reward readers and explicit F0/D0/FQ tasks

**Files:**
- Modify: `src/tasks/hammer/mdp/rewards.py:357-437`
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py:68-123`
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py:408-425`
- Modify: `src/tasks/hammer/config/z1/__init__.py:118-143`
- Modify: `src/tasks/hammer/mdp/first_strike.py` — optional quality
  instrumentation wiring for the FQ task.
- Modify: `tests/test_first_strike_quality_reward.py`
- Modify: `tests/test_first_strike_event.py` — optional quality
  instrumentation tracker coverage.
- Modify: `tests/test_configs.py:433-600`

**Interfaces:**
- Produces:

```python
class FirstStrikeQualityImpactRewardTerm(_FirstStrikeRewardTerm):
    # q_contact * clip(v_precontact / V_FQ, 0, 1)

class FirstStrikeQualityDeliveredRewardTerm(_FirstStrikeRewardTerm):
    # q_contact * clip(delivered / i_ref, 0, 1)
```

- Registers:
  - F8: existing `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear`;
  - F0: `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0`;
  - D0: `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0`;
  - FQ: `Unitree-Z1-Hammer-CaT-Impulse-Event-Quality`.
- F0 differs from F8 only by `impact_progress.weight: 8.0 -> 0.0`.
- D0 differs from F8 only by `delivered_impulse.weight: 2.0 -> 0.0`.
- FQ retains weights 8/2, replaces both readers, and uses bounded components.
  It is a remedy-package test, not an isolated quality-multiplication ablation.
  `V_FQ=1.4598331451416016 m/s` is FQ-only; F8/D0 keep `1.0 m/s` and all arms
  retain `I_REF=0.3088 N·s`.
  Preregister the bounded center-blind `FB` follow-up only after this campaign;
  it is not a fifth arm in the frozen 4×8 matrix.

- [ ] **Step 1: Write failing reward-surface tests**

For a productive finalized tracker:

```python
tracker.contact_quality[:] = 0.25
tracker.contact_quality_valid[:] = True
tracker.v_precontact[:] = 3.0
tracker.delivered[:] = 10.0
assert impact(env, v_fq=1.4598331451416016).item() == pytest.approx(0.25)
assert delivered(env, i_ref=0.3088).item() == pytest.approx(0.25)
```

Require zero for invalid quality, no contact, unproductive events, and second
calls. Require finite positive normalizers and prove that arbitrarily large
speed/impulse cannot exceed `q_contact` per reader. Derive `V_FQ` only from raw
bank SHA-256 `ea4a82e007d95cf7ff962091ba0d6ff3e07e639f4cc1c2d0feab9fe368f47ac8`,
manifest SHA-256
`69d5bc66f2127d0a5b29463b35763092d388bc0e71661b06fdd81b0eea8d661f`, and
qualification SHA-256
`641e520c0cd35932175918d0bf48c7b6df07749ff1462b3d7da7310d76338b1b`. Require
NumPy q90 `method="higher"`, exact 26/256 calibration saturation, and exact
33/128 group-held-out validation saturation; missing provenance or any mismatch
fails closed.

- [ ] **Step 2: Run reward/config tests and confirm RED**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_first_strike_quality_reward.py tests/test_configs.py -q
```

- [ ] **Step 3: Implement readers and task registrations**

Reuse `_FirstStrikeRewardTerm._consume`; do not duplicate latch/reset logic.
Add explicit `event_quality: bool=False` and
`quality_instrumentation: bool=False` config switches. Enforce:

```python
if event_quality and not event_correct:
    raise ValueError("event_quality requires event_correct=True")
if event_quality and not quality_instrumentation:
    raise ValueError("event_quality requires quality_instrumentation=True")
if event_quality and event_linear:
    raise ValueError("event_quality and event_linear are separate treatments")
```

F0 is created by setting the existing F task's `impact_progress.weight=0.0` on
a fresh config object. D0 analogously sets only
`delivered_impulse.weight=0.0`. FQ replaces the two reader classes, keeps
weights 8/2, and enables `quality_instrumentation`. The evaluator has a
separate frozen override that enables `quality_instrumentation` for all four
task IDs without changing their treatment rewards. Wire the optional quality
instrumentation through `first_strike.py`; verify its tracker phase/reset
semantics in `test_first_strike_event.py`.

- [ ] **Step 4: Prove exact allowed config differences**

Serialize F8/F0/D0/FQ reward, action, actuator, tracker, sensor, caps, and CaT
signatures. Require:

```text
F8 vs F0: impact weight only
F8 vs D0: delivered-impulse weight only
F8 vs FQ: reader classes, boundedness mode, and dedicated quality sensor use
all arms: identical actions, actuators, gains, caps, imp_max_p=0, budgets
strict evaluation: quality instrumentation on for all arms; policy-facing
                   tensors and treatment rewards unchanged
```

- [ ] **Step 5: Run tests and commit**

```bash
git add src/tasks/hammer/mdp/rewards.py \
        src/tasks/hammer/mdp/first_strike.py \
        src/tasks/hammer/config/z1/env_cfgs.py \
        src/tasks/hammer/config/z1/__init__.py \
        tests/test_first_strike_event.py \
        tests/test_first_strike_quality_reward.py \
        tests/test_configs.py
git commit -m "feat(hammer): add bounded quality-conditioned strike rewards"
```

---

### Task 4: Sampled trace schema and physical-identity replay

**Files:**
- Modify: `scripts/eval_impulse.py:540-760`
- Modify: `scripts/eval_impulse.py:984-1045`
- Modify: `tests/test_eval_impulse_hook.py`
- Modify: `tests/test_first_strike_campaign.py` — raw schema-v3 slot and
  trace-digest coverage; Task 5 owns its legacy-analysis migration.

**Interfaces:**
- Raises sampled artifact schema from `2` to `3`.
- Adds raw substep channels:
  - quality-sensor found count, normal force, contact position, contact normal;
  - tracker contact point/error/quality/valid/overflow;
  - contact time, axiality, transverse impulse.
- Adds frozen `first_strike` snapshot fields with the same names.
- Builds every strict F8/F0/D0/FQ evaluation config with the passive
  `quality_instrumentation` override, banks separate training/evaluation config
  hashes, and rejects any policy-observation or treatment-reward drift caused
  by the override.
- Produces:

```python
def compare_action_tape_physics(
    traces: Mapping[str, Mapping],
) -> dict[str, object]:
    """Require byte-identical physical/event channels for F8/F0/D0/FQ replay."""
```

- [ ] **Step 1: Write failing schema and digest tests**

Require schema v3, exact new channel shapes, no NaN/Inf, snapshot/stream
agreement, and digest sensitivity to every new physical channel. Reject
overflowed quality as valid.

- [ ] **Step 2: Run evaluator tests and confirm RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_eval_impulse_hook.py tests/test_first_strike_campaign.py -q
```

- [ ] **Step 3: Extend the recorder without duplicating physics**

Capture sensor/tracker tensors in the existing post-`scene.update`,
pre-integration substep callback. Continue evaluating each reward reader exactly
once through the reward manager. Include the new channels in the canonical
trace digest and content-addressed NPZ payload.

- [ ] **Step 4: Add deterministic action-tape identity test**

Replay one fixed action tape in single-environment play configs for F8, F0, D0,
and FQ with `quality_instrumentation=True` in every arm. Compare head/nail
positions, contact, qvel, force, tracker event boundaries, delivered impulse,
and Lambda channels. Also compare instrumented versus uninstrumented F8.
Reward payout streams are expected to differ and are excluded from the physical
digest comparison.

- [ ] **Step 5: Run tests and commit**

```bash
git add scripts/eval_impulse.py \
        tests/test_eval_impulse_hook.py \
        tests/test_first_strike_campaign.py
git commit -m "feat(eval): bank first-contact quality in sampled traces"
```

---

### Task 5: Quality campaign analysis, paired inference, and figures

**Files:**
- Create: `evaluation/analysis/first_strike_quality_campaign.py`
- Create: `evaluation/analysis/plot_first_strike_quality_campaign.py`
- Create: `tests/test_first_strike_quality_campaign.py`
- Modify: `evaluation/analysis/first_strike_campaign.py` — migrate the legacy
  analysis reader to schema v3.
- Modify: `tests/test_first_strike_campaign.py` — schema-v3 legacy-analysis
  migration coverage.

**Interfaces:**
- Imports `exact_seed_tests`, artifact verification, and shared episode
  summarization from the schema-v3-migrated
  `evaluation.analysis.first_strike_campaign`.
- Produces:

```python
def exact_paired_sign_flip(treatment: np.ndarray, control: np.ndarray) -> dict: ...
def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]: ...
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

- Frozen matrix: campaign `fq4x8`, F8/F0/D0/FQ × seeds 8–15, 500 iterations, 4096 training
  environments, `model_499.pt`, 256×2 sampled evaluation.
- Static outputs:
  - `quality_xz_medoid_grid.png`;
  - `quality_xy_contact_grid.png`;
  - `quality_speed_frontier.png`;
  - `quality_force_impulse_grid.png`;
  - `quality_campaign.html`.

- [ ] **Step 1: Write failing paired-inference tests**

Use eight known positive paired differences. Require two-sided sign-flip
`p=2/256=0.0078125`. Test ties/zeros, mismatched seed identities, Holm ordering,
strict ratio boundaries, nonpositive denominator failure, and reproducibility
of the 100,000-resample PCG64 bootstrap.

- [ ] **Step 2: Write failing campaign-contract tests**

Reject missing/duplicate seeds, wrong task IDs, wrong weights/readers, dirty
hashes, missing 512-episode artifacts, any sentinel failure, changed caps,
`imp_max_p != 0`, or non-`*_sampled` decision fields.

- [ ] **Step 3: Write failing episode-quality aggregation tests**

For no physical contact or zero positive normal force, require
`first_contact_quality_sampled=0`. Overflow, nonfinite geometry, or a missing
quality snapshot is an instrumentation failure that invalidates the complete
evaluation row. Require all-episode and contact-conditional means,
central-half/full-head rates, signed onset offsets, contact time, axiality,
transverse impulse, useful speed, delivered impulse, success, and qvel
legality. For D0-versus-F8 also require:

```text
event_window_depth_gain = max(peak_depth - depth_at_contact, 0)
contact_dwell_ms = contact_substeps_inside_event * physics_dt * 1000
recontact_count = off_to_on_edges_after_onset_before_finalization
```

Aggregate all three per seed over every sampled episode, including physical
no-contact episodes as zero where defined. Instrumentation-invalid rows remain
invalid rather than becoming zero.

Also require the legacy analysis reader to accept schema v3 and preserve its
schema-v2 behavior where legacy fields are available. Task 5 owns this legacy
analysis migration; Task 4 owns only recorder/serialization and physical replay.

- [ ] **Step 4: Implement minimal analysis**

Keep the required exact Mann–Whitney tests from `exact_seed_tests`. Apply Holm
separately to the three MWU p-values and three paired sign-flip p-values.
Bootstrap matched seed blocks, never arms independently.

For the D0 mechanism analysis, emit all eight paired seed differences and
paired-bootstrap intervals for event-window depth gain, dwell, recontacts, raw
delivered impulse, and first-window/overall success. These are secondary
mechanism endpoints and do not enlarge either three-test Holm family. Permit
the phrase “mainly selected dwell/recontact” only when the one-sided 95%
paired-bootstrap upper bound for the D0-minus-F8 delivered-impulse difference
is below zero, the corresponding upper bound is below zero for either dwell or
recontact count, the lower bound for its event-window-depth-gain ratio is
greater than `0.90`, and the existing success guardrails pass. A nonpositive or
nonfinite F8 depth-gain denominator makes that mechanism claim not
distinguishable.

Implement the replacement gate verbatim from the approved design:

```text
quality gain >= 0.10
Holm MWU p < 0.05
Holm paired sign-flip p < 0.05
one-sided useful-speed ratio lower bound > 0.95
first-window and overall success >= 0.90 and no worse than F8 by >0.05
delivered-impulse ratio lower bound > 0.90
all provenance/sentinel/numerical gates pass
```

- [ ] **Step 5: Implement deterministic medoid/static plots**

Choose each seed's representative episode by minimum standardized Euclidean
distance to that seed's median vector:

```text
(quality, useful_speed, contact_time, delivered_impulse, signed_x, signed_y)
```

Break exact ties by `(env_id, episode_ordinal)`. Use identical axes across
panels; show the nail axis/outline, start, accepted contact, and time color.
Do not label examples “representative” unless selected by this rule.
Replace a zero or nonfinite within-seed standard deviation by `1.0` before
standardization.

- [ ] **Step 6: Run tests and inspect PNGs**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_first_strike_quality_campaign.py -q
```

Render synthetic fixture plots, open every PNG with the local image viewer, and
reject clipped labels, unequal spatial scales, or misleading auto-ranging.

- [ ] **Step 7: Commit Task 5**

```bash
git add evaluation/analysis/first_strike_quality_campaign.py \
        evaluation/analysis/plot_first_strike_quality_campaign.py \
        evaluation/analysis/first_strike_campaign.py \
        tests/test_first_strike_campaign.py \
        tests/test_first_strike_quality_campaign.py
git commit -m "feat(analysis): add paired first-contact quality campaign"
```

---

### Task 6: CPU qualification, archived-zoo rescoring, and preregistration

**Files:**
- Modify: `scripts/smoke_first_strike_instrumentation.py`
- Modify: `tests/test_smoke_first_strike_instrumentation.py`
- Create: `docs/results/2026-07-26_first_strike_quality_prereg.md`
- Modify: `docs/superpowers/specs/2026-07-26-first-strike-quality-conditioned-design.md`

**Interfaces:**
- Produces a frozen replay-qualified `CONTACT_QUALITY_NUM_SLOTS`.
- Requalifies `I_REF_FIRST_STRIKE_SUCCESS=0.3088 N·s` using eight production
  tracker reference derivations.
- Produces a counterfactual C/D′/F/E quality audit without fitting the quality
  function to those arms.
- Freezes exact comparison rules before new-policy results exist.

- [ ] **Step 1: Extend smoke tests for F0/D0/FQ**

Require finite observations/rewards, one productive event, one quality
snapshot, no overflow, nonzero quality payout for FQ, zero impact payout for
F0, zero delivered payout for D0, normalizer reproduction within 1%, and
unchanged physical channels.

- [ ] **Step 2: Run the full CPU qualification**

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

Require pytest green, reward phases A–M green, contact sensor verified, random
policy sweep green, and reference playback feasible.

- [ ] **Step 3: Re-evaluate the archived policy zoo with quality traces**

After the one-environment CPU probe passes, first bank the existing 32 C/D′/F/E
NPZ bytes and accepted manifest from Vega locally with SHA-256. Then use one
Vega GPU to re-evaluate those checkpoints with the new trace schema and
identical frozen evaluation streams. Call the quality results
“counterfactual” only if action-tape digests and all original physical channels
match; otherwise label them a fresh augmented re-evaluation. Bank every new
content-addressed artifact and accepted-attempt manifest locally with SHA-256.

Reject `q_contact` if counterfactual reward is nonfinite, dominated by speed,
or systematically ranks worse contact-point quality higher. Apply the frozen
falsification checks from the design: reject if any archived arm's top FQ
quartile has lower mean physical quality than its bottom quartile, or if the
absolute seed-level Spearman `corr(FQ, quality) < 0.50` while
`abs(corr(FQ, useful_speed))` exceeds the quality correlation.

- [ ] **Step 4: Produce and inspect the pretraining figures**

Generate the four fixed-scale static figures from Task 5. Inspect every image
locally and obtain one independent visual/code review.

- [ ] **Step 5: Write the preregistration**

Record:

```text
task IDs and exact config hashes
seeds 8..15
500 iterations, 4096 environments, model_499.pt
256 environments × 2 sampled episodes
quality endpoint and zero denominator rules
MWU + paired sign-flip + Holm families
PCG64 seed 20260726 and 100,000 paired bootstrap samples
D0 event-window depth-gain, dwell, and recontact definitions/claim rule
all replacement margins and invalidation gates
FQ-only V_FQ provenance, q90 method, and calibration/validation saturation checks
qvel results labelled simulator-only when illegal
```

- [ ] **Step 6: Independent review and commit**

Have one reviewer inspect sensor/phase/reward code and a second reviewer inspect
statistics/provenance/plots. Resolve findings by evidence, rerun affected tests,
then:

```bash
git add scripts/smoke_first_strike_instrumentation.py \
        tests/test_smoke_first_strike_instrumentation.py \
        docs/results/2026-07-26_first_strike_quality_prereg.md \
        docs/superpowers/specs/2026-07-26-first-strike-quality-conditioned-design.md
git commit -m "docs(hammer): preregister first-contact quality campaign"
```

---

### Task 6A: Paired CPU reference-recipe qualification

**Files:**
- Create: `evaluation/trajectory/reference_recipe_probe.py`
- Create: `tests/test_reference_recipe_probe.py`
- Create: `docs/results/2026-07-26_reference_recipe_probe.md`
- Create: static recipe grids under
  `evaluation/results/2026-07-26_reference_recipe_probe/`

**Interfaces:**
- Freezes `traj_dc` at SHA-256
  `b22dabb94a10a1e7f68f3fe6a4a2f9412e14916a40a3dbafc81e2f3c3cc7f89f`; no other
  trace may substitute. Because the source traces lack pre-apex motion, use an
  identically zero pre-apex transverse residual and repeat the apex at the
  wind-up/descent boundary rather than infer motion. Resample the observed
  post-apex descent on 51 uniform points and store the nail-frame transverse
  residual from the R0 centered path built with that source's own live
  head/nail poses. R1 is the residual template minimizing summed pairwise
  transverse L2 distance; ties break by lexicographic trace digest. Replay adds
  the residual, rotated through the signed-offset diagnostic's nail-frame
  basis, to the live R0 target at corresponding segment progress.
- Replays 32 matched randomized reset states under:
  - R0: current `SingleStrikeReference`;
  - R1: that nail-frame-relative transverse medoid template, with R0's axial
    schedule and pacing unchanged;
  - R2: R1 until 90 mm axial standoff, then multiply its transverse offset by
    `1 - (10*u**3 - 15*u**4 + 6*u**5)`, where
    `u=clip((0.09-d)/0.07, 0, 1)` and `d` is axial standoff, so it reaches zero
    with continuous first and second derivatives by 20 mm.
- Generates, before rollout outcomes are inspected, a fresh 32-row digested
  manifest using reset seeds exactly `0..31`; it must not reuse a prior
  manifest. Each row binds realized initial qpos/qvel, head/nail poses,
  nail-frame basis, code/config/asset revisions, and a canonical state digest.
  Every recipe must reproduce the row digest.
- Uses `play=False`, fixed gains/action scale/caps, `cat_impulse=True`,
  `imp_max_p=0`, and no automatic reset.
- Executes the existing live-head feedback law
  `clip((p_next-head)/0.15, -1, 1)`.
- After the nominal script, issues no more than two final-target control steps
  and stops earlier on success. There is no further endpoint hold; a rollout
  without both a productive first event and task success by `n_script + 2` is a
  press-through failure.
- Records complete 500 Hz first-event, contact, force/impulse, prefix-qvel, and
  realized-path channels.

- [ ] **Step 1: Write failing pure-template and contract tests**

Test `traj_dc` digest binding, zero pre-apex residual/repeated-apex handling,
deterministic observed-descent resampling, source-R0 residual construction,
deterministic medoid selection/tie-break, nail-frame translation/rotation
invariance, live-head and live-nail anchoring, exact R1 preservation through
R2's 90 mm gate, the stated quintic blend and zero transverse offset by 20 mm,
fresh exact reset seeds/state digests, the hard `n_script + 2` boundary with no
further endpoint hold, complete prefix-qvel inspection, and strict fail-closed
sentinel behavior.

- [ ] **Step 2: Run tests and confirm RED**

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_reference_recipe_probe.py -q
```

- [ ] **Step 3: Implement the minimal replay probe**

Reuse `SingleStrikeReference`, the existing first-strike tracker, and existing
trace/digest helpers. Do not add a production reward or change the live
reference. Compare realized head paths rather than commanded knots.

- [ ] **Step 4: Run 96 matched CPU rollouts**

First require R0 to reproduce the Phase M productive in-script contact,
`>=0.5 m/s` speed, and task-success contract, then bank its complete 32-reset
baseline. Apply the following absolute gates only to an R1 or R2 recipe
proposed for promotion; R1 may remain a diagnostic comparator:

```text
productive in-script first-event successes >= 30/32
accepted onsets inside provisional 12 mm proxy >= 29/32
median v_pre >= 0.95 * R0; every accepted event >= 0.5 m/s
median raw delivered >= 0.90 * R0
no prefix arm qvel > 3.1415 rad/s
no Lambda/cap > 1
terminal axial-speed, terminal lateral-speed, and approach-angle gates pass
no nonfinite/dead-Lambda/impossible-success/post-success/press-only failure
```

Freeze the exact terminal-gate thresholds in the fresh reset manifest before
any R1/R2 outcome is inspected; a missing, nonfinite, or failed value fails
closed.

The R2 repair contrast additionally requires at least four more within-12-mm
onsets than R1 (`>=4/32`, exactly 12.5 percentage points), whether or not R1
passes an absolute promotion gate. Replacing R0 additionally requires at least
5% higher `v_pre` or at least 10% lower p95 prefix qvel; delivered impulse alone
is not a replacement reason.
Global straightness is never a promotion target.

- [ ] **Step 5: Test reference identifiability and contact robustness**

Counterfactually score every replay under all recipes. Require the generating
recipe to have the highest cumulative raw imitation score in at least 26/32
resets and median self/next-best ratio at least 1.25. Re-run R0 and any winner
at `solref_scale=2`; reject if speed, delivered impulse, or worst-joint Lambda
changes by 20% or more, if success/centering fails, or if any terminal
axial-speed, terminal lateral-speed, or approach-angle gate is missing,
nonfinite, or fails.

State explicitly in the result that passing this probe demonstrates scripted
feasibility and separability under the current weak imitation reader only; it
does not show that PPO can learn the recipe.

- [ ] **Step 6: Plot, review, and bank**

Create identical-scale R0/R1/R2 x-z and x-y grids, contact-aligned force and
impulse plots, a metrics table, and a small video montage. Obtain independent
code and scientific review. State explicitly that the 12 mm measure is a
provisional site-center proxy until the new contact-point sensor is qualified.

- [ ] **Step 7: Commit named files**

```bash
git add evaluation/trajectory/reference_recipe_probe.py \
        tests/test_reference_recipe_probe.py \
        docs/results/2026-07-26_reference_recipe_probe.md
git commit -m "feat(hammer): qualify terminal reference recipes on CPU"
```

---

### Task 7: Clean Vega CUDA and throughput gate

**Files:**
- Modify: `scripts/slurm/vega_train.sbatch`
- Modify: `scripts/slurm/vega_eval.sbatch`

**Interfaces:**
- Produces clean CPU-vs-CUDA quality-reader agreement and a measured throughput
  ratio with/without the dedicated sensor.
- Adds a new fail-closed `fq4x8` launcher contract; it does not reuse or relax
  the old `fsr4x8` C/D-prime/F/E contract.
- Requires seeds exactly `8..15`, shorts exactly `f8/f0/d0/fq`, exact task IDs,
  500 iterations, and unset `IMPACT_W`, `DELIVERED_W`, and `NAIL_DRIVEN_W`
  because the registered tasks encode the treatments.

- [ ] **Step 1: Push and deploy by provenance**

Push the named commits. On Vega, create a fresh checkout, pull the exact commit,
and verify both main and asset repositories are clean at the recorded hashes.

- [ ] **Step 2: Run one F8/F0/D0/FQ CUDA smoke each**

Use 256 environments. Require finite rewards/observations, productive first
events, no quality overflow for FQ, `impossible_success_n==0`,
`lambda_dead_n==0`, exact task IDs/weights/readers, and the expected payout
semantics for all four arms.

- [ ] **Step 3: Run the sensor throughput comparison**

Measure at least 1,000 post-warmup control steps at 4096 environments for:

```text
F8 without dedicated quality sensor
F8 with dedicated quality sensor
FQ with dedicated quality sensor and readers
```

Record steps/s, GPU memory, and sensor overflow. The gate passes only if FQ
throughput is at least 80% of F8 without the sensor and no overflow occurs. If
the gate fails, stop and profile; do not launch 32 jobs.

- [ ] **Step 4: Bank smoke evidence**

Save JSON logs with commit hashes, asset hashes, configs, predicate counts,
throughput, memory, and dirty states. Commit only the small summary/manifest,
not transient caches.

---

### Task 8: Matched 4×8 training and strict evaluation

**Files:**
- No source changes permitted after launch.
- Create result artifacts under
  `docs/results/assets/2026-07-27_first_strike_quality/`.

**Interfaces:**
- Produces 32 accepted checkpoints and 32 strict sampled evaluations.

- [ ] **Step 1: Submit the 32 training jobs**

Submit seeds 8–15 for F8, F0, D0, and FQ. Each job uses one GPU, 4096
environments, 500 PPO iterations, a unique run name, and retains
`model_499.pt`.

From the clean Vega checkout, with exact 40-hex revisions:

```bash
CODE_REV=$(git rev-parse HEAD)
ASSET_REV=$(git -C ../safe_impact_manipulation rev-parse HEAD)

CAMPAIGN=fq4x8 SEEDS="8 9 10 11 12 13 14 15" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Linear SINGLE_SHORT=f8 ITERS=500 \
  EXPECTED_CODE_REVISION="$CODE_REV" EXPECTED_ASSET_REVISION="$ASSET_REV" \
  sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_train.sbatch

CAMPAIGN=fq4x8 SEEDS="8 9 10 11 12 13 14 15" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0 SINGLE_SHORT=f0 ITERS=500 \
  EXPECTED_CODE_REVISION="$CODE_REV" EXPECTED_ASSET_REVISION="$ASSET_REV" \
  sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_train.sbatch

CAMPAIGN=fq4x8 SEEDS="8 9 10 11 12 13 14 15" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0 SINGLE_SHORT=d0 ITERS=500 \
  EXPECTED_CODE_REVISION="$CODE_REV" EXPECTED_ASSET_REVISION="$ASSET_REV" \
  sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_train.sbatch

CAMPAIGN=fq4x8 SEEDS="8 9 10 11 12 13 14 15" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Quality SINGLE_SHORT=fq ITERS=500 \
  EXPECTED_CODE_REVISION="$CODE_REV" EXPECTED_ASSET_REVISION="$ASSET_REV" \
  sbatch --array=0-7 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_train.sbatch
```

The launcher must fail if `IMPACT_W`, `DELIVERED_W`, or `NAIL_DRIVEN_W` is
present in the environment.

- [ ] **Step 2: Monitor without selecting on performance**

Retry only documented infrastructure failures with identical config and seed.
Retain all attempts. Never replace a completed weak/unstable seed. Send periodic
status updates with counts by arm: pending/running/completed/failed/retried.

- [ ] **Step 3: Freeze the accepted-training/checkpoint manifest**

After infrastructure-only training retries are resolved, freeze exactly 32
rows in `accepted_training_checkpoints.tsv`. Each row binds arm/seed, retained
`model_499.pt`, training attempt and retry history, code/config/asset revisions,
checkpoint/artifact hash, and dirty state. Never replace a completed
weak/unstable seed, and do not include evaluation-attempt fields in this
manifest.

- [ ] **Step 4: Run strict sampled evaluation**

For every accepted checkpoint, collect exactly the first two completed episodes
from each of 256 environments under stochastic actions, matched evaluation RNG
streams, ±0.05 rad reset noise, and the existing observation-corruption
contract. Enable the passive `quality_instrumentation` evaluation override for
all four arms, bank its separate config hash, and reject
dirty/incomplete/sentinel-failing rows.

Use only the frozen accepted-training/checkpoint manifest:

```bash
CAMPAIGN=fq4x8 EVAL_ATTEMPT=attempt1 \
  EVAL_ROOT="$HOME/unitree_rl_mjlab_eval" \
  ACCEPTED_MANIFEST="$HOME/unitree_rl_mjlab_eval/fq4x8/accepted_training_checkpoints.tsv" \
  EXPECTED_CODE_REVISION="$CODE_REV" EXPECTED_ASSET_REVISION="$ASSET_REV" \
  sbatch \
  --export=ALL,CAMPAIGN,EVAL_ATTEMPT,EVAL_ROOT,ACCEPTED_MANIFEST,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
  scripts/slurm/vega_eval.sbatch
```

- [ ] **Step 5: Freeze the accepted-evaluation manifest**

After infrastructure-only evaluation retries are resolved, freeze exactly 32
rows in `accepted_evaluations.tsv`. Each row binds its accepted-training row to
the evaluation attempt, training and evaluation commits/config hashes, asset
commit, RNG stream, payload hash, retry history, dirty state, quotas, and every
sentinel predicate. Do not analyze until all 32 evaluation identities are
present and verified.

---

### Task 9: Analyze, visualize, and bank the result

**Files:**
- Create: `docs/results/2026-07-27_first_strike_quality_result.md`
- Create: `docs/results/assets/2026-07-27_first_strike_quality/summary.csv`
- Create: `docs/results/assets/2026-07-27_first_strike_quality/analysis.json`
- Create: `evaluation/analysis/build_first_strike_review_gallery.py`
- Create: `tests/test_first_strike_review_gallery.py`
- Create: static PNGs and optional HTML defined in Task 5.

**Interfaces:**
- Produces the preregistered F0-vs-F8, D0-vs-F8, and FQ-vs-F8 decisions, plus
  descriptive FQ-vs-F0 and FQ-vs-D0.

- [ ] **Step 1: Run analysis against the frozen manifest**

Require the frozen accepted-evaluation manifest, all artifact hashes, and
episode aggregates to recompute. Emit seed tables, exact MWU, paired sign-flip,
Holm-adjusted values, paired-bootstrap bounds, guardrails, all eight paired
differences, qvel legality, contact-time, axiality, transverse impulse, D0
event-window depth gain/dwell/recontact endpoints, and failure/sentinel counts.

- [ ] **Step 2: Generate and visually inspect figures**

Render the x–z medoid grid, x–y contact grid, quality-speed frontier, and
force/impulse grid. Open all PNGs locally; verify identical spatial scales and
that plotted medoids match the analysis JSON identities.

- [ ] **Step 3: Independent result review**

One reviewer recomputes statistics from CSV/NPZ; another reviews trace/figure
selection and physical interpretation. Neither reviewer edits the result
before reporting findings.

- [ ] **Step 4: Build the provenance-bound policy video library**

Render the exact medoid episode used by the trajectory grid for every accepted
arm/seed by replaying its stored `action_tape` in the same reset/config, plus
one clearly labelled deterministic mean-policy rollout per checkpoint. Extend
`scripts/render_policy.py` with an action-tape replay input only if the existing
playback path cannot accept it; keep a single renderer. Reuse
`plot_first_strike_campaign.write_video_overlay_html` and do not infer
quantitative trajectories from pixels.

Each gallery entry contains:

```text
video
x-z and x-y path
contact close-up
contact-aligned force / cumulative impulse
quality, useful speed, contact time, delivered impulse, peak qvel, legality
```

Bind each MP4 to the source trace with episode identity, trace digest, explicit
frame-to-substep timing, and SHA-256. Tests must reject a mismatched episode,
digest, frame count, nonmonotone timing map, or a non-medoid clip presented as
the representative.

Create a separate annotation export with enum
`good|questionable|bad`, reason tags, and free text. Human annotations are
qualitative diagnostics only: they cannot alter the frozen statistical
decision or trigger replacement of the medoid after results are visible.
Large MP4s stay git-ignored; bank manifests, checksums, montages, overlays, and
the blank/user-filled annotation file.

- [ ] **Step 5: Write the result**

Use the four-question structure:

```text
What was supposed to happen?
What actually happened?
Why was there a difference?
What can we learn from this?
```

Separate proven, supported, not distinguishable, simulator-only, and open.
Do not call a null equivalence or call an illegal-qvel policy hardware-ready.

- [ ] **Step 6: Run final verification and commit named files**

Re-run the targeted/full test suite appropriate to changed files, validate
every checksum, then stage only the result doc, summary, analysis, figures,
manifest, gallery code/tests/metadata, and final reviewed source/test files.
