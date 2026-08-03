# Wave-2 waypoint-guidance replication — preregistration (2026-08-03)

**Status: WRITTEN, NOT LAUNCHED.** No job has been submitted. This document is
registered *before* any Wave-2 training so the endpoints and thresholds below cannot be
chosen after seeing the data.

## Why

Wave 1 (`docs/results/2026-08-03_wave1_waypoint_pilot_result.md`) found a large
straightness effect — roughly a 5× reduction in pre-contact perpendicular error and a
path ratio near 1.0 — but on **two seeds per arm**, and one guided seed (G/2) was a
**guidance failure**: it earned zero gate payout and crossed no gates, while still
striking and driving the nail to the 32 mm stop. It was a successful *task* policy that
ignored the *guidance*. That is precisely the failure mode this study measures, and it
shows within-arm variance comparable to the between-arm difference. Wave 2 tests
**replication at fixed configuration**. It adds seeds, not machinery.

Wave 2 answers exactly one question: *does the 200-iteration guidance result hold across
six seeds per arm?* The separate question of whether C0 seed 2's collapse is an
undertraining artifact — the 500-iteration experiment — is **deliberately deferred** and
is not part of this preregistration.

## Design

| Item | Value |
| --- | --- |
| Arms | C0 (control), G (`r_gate`), P (`r_waypoint_progress`) |
| Existing seeds (Wave 1, reused as-is) | 2, 3 |
| New seeds (this wave) | 4, 5, 6, 7 — for **every** arm |
| New policies | **12** (3 arms × 4 seeds) |
| Pooled analysis population | **18** (6 per arm) |
| PPO iterations | exactly **200** |
| Environments | **4096** |
| GPUs | 1 per run |

**Nothing else changes.** Fixed reset, action space, reward terms and weights, gate
geometry (`GUIDELINE_GATE_RADIUS_M = 0.015`, 6 gates at 1/7…6/7) and treatment weights
(gate 8.0, progress 8.0) are identical to Wave 1. Reward-weight CLI overrides remain
unset (`env -u IMPACT_W DELIVERED_W NAIL_DRIVEN_W`).

Tasks: `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-{C0,CGate,CProgress}`.

### Safety configuration — unchanged and log-only

- `imp_max_p = 0.0`. Λ is measured, never acted on.
- Manufacturer impulse caps `[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]` — unchanged.
- **Velocity enforcement remains OFF for this replication.** Velocity CaT and
  deterministic velocity termination stay disabled, exactly as in Wave 1.
- qvel illegality is **measured and reported**, not enforced. Wave 1 was universally
  illegal (4.959–5.256 rad/s against the 3.1415 rad/s hardware limit) in all six
  policies. Wave 2 is expected to remain illegal; that is an observation this wave
  records, not a failure it corrects. Enforcement is a separate experiment.

### Retry rule

A distinction is drawn between a policy that *trained badly* and a run that *did not
train*:

1. **A completed weak or degenerate policy is never replaced.** Zero contact, zero Λ,
   zero treatment payout, or an obviously useless policy are all **results**, not
   errors. They are the per-arm failure rate, which is the primary endpoint. C0 seed 2
   (no contact) and G seed 2 (zero gate payout, successful strike) remain in the Wave-1
   half of the population on exactly this rule.
2. **A documented infrastructure failure may be retried**, and only with **identical
   seed, task, configuration and revision**. Qualifying causes are node failure, OOM,
   Slurm preemption or timeout, filesystem error, and a job that never reached the
   training loop — i.e. no usable policy was produced. The cause, job ID and Slurm exit
   state must be written into the launch ledger before the retry is submitted.
3. **Never substitute a different seed.** A retry re-runs the same seed. If a seed
   cannot be made to complete, it is reported as missing with its documented cause; it
   is not backfilled from the seed pool and the arm is not silently re-sized.

A retry that produces a *different* policy from a *completed* run is not a retry — it is
seed shopping, and it is forbidden.

## Endpoints

### Primary

**Proportion of policies completing all six gates, per arm, over the 6 seeds — where a
gate counts only if it is crossed no later than the first contact-onset substep.**

Gates crossed *after* contact onset do not count. Guidance is a claim about the
**approach**; a gate swept through during follow-through, after the nail is already
struck, is not guidance and must not be scored as such. This is the single frozen
definition — full-rollout gate maxima are not used as the primary endpoint anywhere.

Wave-1 baseline: C0 **0/2**, G **1/2**, P **2/2**.

**Verification against Wave 1 (run before freezing this definition).** Wave 1's published
`traj_gates_completed` column was a full-rollout maximum. Both definitions were
recomputed from the six frozen `trace.npz` files:

| policy | contact onset (substep) | gates @ onset | gates @ full rollout | published |
| --- | --- | --- | --- | --- |
| C0/2 | 799 (no contact) | 0 | 0 | 0 |
| C0/3 | 64 | 0 | 0 | 0 |
| G/2 | 58 | 0 | 0 | 0 |
| G/3 | 56 | 6 | 6 | 6 |
| P/2 | 60 | 6 | 6 | 6 |
| P/3 | 56 | 6 | 6 | 6 |

**No policy differs between the two definitions**, and both select exactly
{G/3, P/2, P/3}. The definitions are not being pooled — the pre-contact rule is frozen as
primary, and it happens to agree with the published column on all six Wave-1 policies,
so the frozen Wave-1 table remains valid under it without modification.

This agreement is **not** guaranteed for Wave 2: a policy that crosses gates only during
follow-through would score 6 full-rollout and fewer at onset. Wave-2 analysis must
compute the count at onset and report any policy where the two disagree.

### Secondary

Reported per policy and summarized per arm:

1. Pre-contact **finite-segment** perpendicular error — max and RMS (mm).
2. Pre-contact **path ratio**.
3. Task success and **physical** nail depth (clamped at the 0.032 m stop, not raw
   elastic overshoot).
4. qvel legality — peak `|q̇|` against 3.1415 rad/s.
5. Delivered impulse (N·s) and max Λ/cap ratio.

### Measurement conventions — carried over verbatim from Wave 1

These are fixed now, before data, because Wave 1 showed the definitions materially change
the numbers:

- Perpendicular error uses the **recorded `substep_perpendicular_error_m`** field only —
  the distance to the **finite** entry→nail segment, with progress clamped to [0, 1].
  It is never re-derived against the infinite line. (Doing so understated gate-completers
  by up to 2.85× in Wave 1.)
- **Pre-contact** means substeps from reset through the **first contact-onset sample,
  inclusive**. Recorded as `precontact_window_substeps`. For a policy that never makes
  contact, the pre-contact window is the entire rollout.
- Path ratio and gate completion are evaluated over the **same pre-contact window**, so
  every path-quality endpoint shares one interval. Gate completion specifically uses the
  frozen onset rule above.
- Full-rollout segment error is reported alongside but is **not** the path-quality
  result: it grows when the hammer follows through past the nail endpoint, where the
  closest point on the finite segment is pinned at the nail.
- Nail depth is the **clamped** value; raw slide-joint position includes elastic
  excursion beyond the stop.

## Predeclared descriptive straightness thresholds

A policy is descriptively **straight** if it meets all three:

1. all six gates completed;
2. pre-contact max finite-segment perpendicular error **≤ 15 mm**;
3. pre-contact path ratio **≤ 1.15**.

These are **descriptive labels, not hypothesis tests.** With 6 seeds per arm no
inferential claim about the population is licensed; the thresholds exist so the
per-arm counts are computed by a rule fixed in advance rather than chosen afterwards.

Applied to Wave 1 they select exactly {G/3, P/2, P/3} and reject {C0/2, C0/3, G/2}. The
margin is wide in both directions: straight policies ran 10.88–11.88 mm at ratio
1.035–1.097; non-straight strikers ran 52.50–54.00 mm at ratio 1.412–1.420. Nothing in
Wave 1 sits near 15 mm or 1.15.

## Analysis plan

**Reuse the frozen measurement definitions unchanged** — the finite-segment perpendicular
field, the pre-contact window, the onset gate rule, the clamped nail depth, and the
fixed-reset digest `bde511ec2adc42e5365e1e46f45ff1fb43223a93c31c6fbfb4580352e445e319`.
The definitions are frozen; the tooling that applies them is not assumed correct by
inheritance. Tools: `scripts/render_policy.py` (500 Hz trace),
`scripts/diag_impulse_trace.py` (CPU impulse),
`evaluation/results/2026-08-02_wave1_waypoint/build_comparison_table.py`.

Extending the inventory from 6 to 18 rows is a **reviewed change, not a mechanical
one**. The extension must:

1. be independently reviewed before any Wave-2 number is reported;
2. **reproduce all six frozen Wave-1 rows byte-for-byte** in every column those rows
   already have. The frozen CSV is
   `4ab58b67658ef6c899a21ffbd200d4c568327352c72da0dd1c386437dda86268`; the six Wave-1
   rows must emerge from the extended generator identical to that file's rows. Any
   diff in a Wave-1 row means the extension perturbed the measurement and must be
   fixed before proceeding — it is never accepted as a re-measurement.

Wave-1 rows are re-used from frozen artifacts and **not re-rendered**; re-running the
renderer on Wave-1 checkpoints is not a substitute for byte-for-byte reproduction.

If Wave 2 requires a column Wave 1 lacks (the at-onset gate count is the expected case,
since Wave 1's column was a full-rollout maximum that happens to coincide), the new
column is **added**, never substituted into an existing one, and clause 2 continues to
bind on all pre-existing columns. Any added column is published as a superset table with
its own hash; the Wave-1 frozen CSV and its hash are not reissued.

## Known gaps, declared up front

- **No fail-closed launcher guard for `CAMPAIGN=wave2`.** Wave 1 also ran through the
  generic `SINGLE_TASK` path with no campaign-specific guard in
  `scripts/slurm/vega_train.sbatch`. **No new launcher will be built for this wave** —
  changing the launcher would change training-relevant source and break the replication.
  The gap is closed by the frozen ledger and the automatic startup check below, which
  are external to the training path.
- **CUDA fixed-reset impulse validation remains unavailable** (`diag_impulse_trace.py`
  is CPU-only by design). Known CPU/CUDA substep differences stay unmeasured for the
  impulse channels. Training runs on CUDA; evaluation on CPU. Metrics from the two
  backends are never combined as though from one rollout.
- **Renderer metadata does not record the execution device.** Wave 1 resolved this as an
  inferred provenance bridge. Adding a `device` field and a validator requirement is a
  standing follow-up for this wave's tooling, not a blocker.

## Pinned provenance

### Revisions

| Role | Revision |
| --- | --- |
| Wave-2 training | **PENDING** — pinned in the follow-up commit, and equal to the commit that first carries this preregistration. No placeholder hash is recorded here; an unpinned field is safer than a wrong one. |
| Asset (`safe_impact_manipulation`, `hammer-z1`) | `b58ccd2f81fd246f27c1e8d88cf86484cd888703` |
| Wave-1 training (for comparison) | `a6a9c970ea9eca00b34e8e9be806d22d396a964a` |

The asset revision is **identical to Wave 1**. No asset change is permitted in this wave.

### Training-relevant source is identical to Wave 1

Everything committed on `cartesian-guideline-fic` since Wave 1 trained is documentation,
evaluation/analysis tooling, and tests. The training path is byte-identical, verified by
git tree hash rather than by inspection:

| Path | `a6a9c97` | Wave-2 training revision | |
| --- | --- | --- | --- |
| `src/` (entire tree) | `bf6596c712661009…` | `bf6596c712661009…` | **IDENTICAL** |
| `scripts/slurm/vega_train.sbatch` | `e2b8abb6727f7532…` | `e2b8abb6727f7532…` | **IDENTICAL** |
| `scripts/train.py` | `f727e2882269d01a…` | `f727e2882269d01a…` | **IDENTICAL** |

`src/` covers the tasks, rewards, guideline MDP terms, tracker, observations, robot
config and constraint machinery. Its tree hash being unchanged means **no
training-relevant source differs from Wave 1**. The only intended differences between
the two waves are **seed values and campaign identity**.

This must be re-verified at deploy time: the deployed revision is acceptable only if
these three tree hashes still match.

### Per-arm configuration digests

sha256 over the canonically serialized resolved reward configuration of each arm's
`env_cfg` (fields sorted, callables resolved to fully-qualified names):

| Arm | Task ID suffix | Reward-config sha256 |
| --- | --- | --- |
| C0 | `…-Guideline-C0` | `47d993698852dc939c753d978e41c9124e24c470e0595439f004d2b561d76fb9` |
| G | `…-Guideline-CGate` | `a5b767b22a77ecfc068eea9885ff88c4bedcd2035e5622627650a20216431e0f` |
| P | `…-Guideline-CProgress` | `dca539d938e6eb15cb0ececebf78e9d2972870045edf6b52967d16f6945ed264` |

Resolved active reward terms and weights — the eight base terms are **identical across
all three arms**, and each treatment adds exactly one term at weight 8.0:

| Term | C0 | G | P |
| --- | --- | --- | --- |
| `approach` | 0.1 | 0.1 | 0.1 |
| `nail_driven` | 0.5 | 0.5 | 0.5 |
| `nail_depth_delta` | 600.0 | 600.0 | 600.0 |
| `impact_progress` | 8.0 | 8.0 | 8.0 |
| `delivered_impulse` | 2.0 | 2.0 | 2.0 |
| `completion` | 100.0 | 100.0 | 100.0 |
| `action_rate` | −0.01 | −0.01 | −0.01 |
| `joint_pos_limits` | −10.0 | −10.0 | −10.0 |
| `r_gate` | — | **8.0** | — |
| `r_waypoint_progress` | — | — | **8.0** |

The three digests differ from one another and must be **re-derived and matched at deploy
time**. A digest mismatch is a launch abort, not a warning.

## Frozen launch matrix (12 rows) — not executed

This ledger is frozen before submission. Every row is fixed except the Slurm job ID and
final state, which are filled in as they become known. No row may be added, removed or
re-seeded.

| # | Arm | Task ID | Seed | Run identity | Iters | Envs | Job ID | State |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C0 | `…-Guideline-C0` | 4 | `wave2_wave2_c0_seed4` | 200 | 4096 | — | pending |
| 2 | C0 | `…-Guideline-C0` | 5 | `wave2_wave2_c0_seed5` | 200 | 4096 | — | pending |
| 3 | C0 | `…-Guideline-C0` | 6 | `wave2_wave2_c0_seed6` | 200 | 4096 | — | pending |
| 4 | C0 | `…-Guideline-C0` | 7 | `wave2_wave2_c0_seed7` | 200 | 4096 | — | pending |
| 5 | G | `…-Guideline-CGate` | 4 | `wave2_wave2_g_seed4` | 200 | 4096 | — | pending |
| 6 | G | `…-Guideline-CGate` | 5 | `wave2_wave2_g_seed5` | 200 | 4096 | — | pending |
| 7 | G | `…-Guideline-CGate` | 6 | `wave2_wave2_g_seed6` | 200 | 4096 | — | pending |
| 8 | G | `…-Guideline-CGate` | 7 | `wave2_wave2_g_seed7` | 200 | 4096 | — | pending |
| 9 | P | `…-Guideline-CProgress` | 4 | `wave2_wave2_p_seed4` | 200 | 4096 | — | pending |
| 10 | P | `…-Guideline-CProgress` | 5 | `wave2_wave2_p_seed5` | 200 | 4096 | — | pending |
| 11 | P | `…-Guideline-CProgress` | 6 | `wave2_wave2_p_seed6` | 200 | 4096 | — | pending |
| 12 | P | `…-Guideline-CProgress` | 7 | `wave2_wave2_p_seed7` | 200 | 4096 | — | pending |

Submission shape — one 4-job array per arm, seeds 4–7:

```
CAMPAIGN=wave2 SEEDS="4 5 6 7" ITERS=200 \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-<C0|CGate|CProgress> \
  SINGLE_SHORT=<wave2_c0|wave2_g|wave2_p> \
  EXPECTED_CODE_REVISION=<pinned training revision> \
  EXPECTED_ASSET_REVISION=b58ccd2f81fd246f27c1e8d88cf86484cd888703 \
  env -u IMPACT_W -u DELIVERED_W -u NAIL_DRIVEN_W \
  sbatch --array=0-3 --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,ITERS,EXPECTED_CODE_REVISION,EXPECTED_ASSET_REVISION \
    scripts/slurm/vega_train.sbatch
```

### Automatic startup check — every job, every field

Because the launcher has no `wave2` guard, each job's startup log is checked
**automatically against its ledger row**, not read by eye. For all 12 jobs, assert:

| Field | Requirement |
| --- | --- |
| Task ID | exactly the ledger row's task |
| Seed | exactly the ledger row's seed |
| Iterations | exactly 200 |
| Environments | exactly 4096 |
| Code revision | exactly the pinned training revision |
| Asset revision | exactly `b58ccd2f…` |
| Clean state | 0 dirty files; provenance string contains no `-dirty` |
| Reward overrides | `IMPACT_W`, `DELIVERED_W`, `NAIL_DRIVEN_W` all **absent** from the environment |
| `imp_max_p` | exactly 0.0 |
| Impulse caps | exactly `[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]` |

**On any mismatch, `scancel` the affected job immediately** and record the mismatch in
the ledger. A mismatched job is cancelled, corrected and resubmitted **at the same
seed** under the retry rule — it is never left running and never analyzed. Sibling jobs
that pass are unaffected.

Prerequisites before any submission: clean Vega checkout at the pinned revision (0
dirty, no `-dirty` provenance), the three tree hashes re-verified, the three reward-config
digests re-derived and matched, reference qualification repeated, and CUDA smokes green
for all three arms.

## Out of scope for Wave 2

500-iteration runs; constraint enforcement (`imp_max_p > 0`); velocity CaT; corridor or
tube treatments; `r_imit`, GP, hard guidance, recorded-reference tracking; variable
impedance / `set_gains`; any reward term not already in Wave 1.
