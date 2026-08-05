# Wave-3 P+V — soft velocity-CaT on the progress-guided arm — preregistration (2026-08-04)

> **STATUS: FROZEN, NOT YET LAUNCHED.** Endpoints, thresholds, decision rules, the launch
> ledger and the pinned provenance below are fixed before any Wave-3 outcome is inspected.
> Nothing in this document may be edited after training is submitted; a correction requires
> a new review and a new hash.

## Why

Wave 2 froze three findings across 18 policies (C0/G/P × seeds 2–7), independently reviewed
and hash-bound (`cc125c08…`):

1. Waypoint guidance changes trajectory behaviour substantially. P completed all six gates
   in **5/6** seeds; **3/6** met the predeclared strict straight label.
2. The per-joint impact-impulse constraint is **far from binding** — maximum Λ/cap across
   all 18 policies was **0.127732**.
3. **0/18 policies were qvel-legal.** Peaks ran 4.3751–5.2556 rad/s against the real Z1
   limit of 3.1415 rad/s.

Finding 3 is a **measurement, not a verdict on CaT**: velocity enforcement was disabled by
design in Waves 1 and 2, so no policy was ever penalised for exceeding the limit. Velocity is
the only result on the table that currently gates hardware. This wave turns velocity CaT on
and changes nothing else.

## Design

Exactly one new treatment.

**P+V** = the frozen progress-only P arm **plus** faithful soft velocity-CaT computed on the
per-joint 500 Hz substep peak.

| | |
| --- | --- |
| Task ID | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel` |
| Seeds | 2, 3, 4, 5, 6, 7 (six runs) |
| Iterations | 200 |
| Environments | 4096 |
| GPUs | one per run |
| Comparison | paired by seed against the six frozen P controls (Wave-1 seeds 2–3, Wave-2 seeds 4–7) |

### The treatment, exactly

- Velocity limit **3.1415 rad/s**, all six arm joints (`z1_description` URDF).
- **max_p = 0.5, min_p = 0.0, tau = 0.95**, frozen for the entire run. **No curriculum.**
  Curriculum is a possible follow-up only if fixed CaT causes an exploration/success collapse.
- Detection: the **per-joint 500 Hz substep peak**, `c_j = peak_500Hz(|q̇_j|) − 3.1415`,
  each joint normalised by its own EMA (CaT Eq. 6–7). Not the post-decimation control-rate
  sample, which aliases the within-window strike spike.
- Soft CaT: δ discounts the value-target bootstrap (CatPPO dual-mask GAE) and the positive
  reward only. It **never ends the episode**.

### Enforcement and measurement are the same physical quantity

`SubstepPeakJointVel` runs inside `metrics_manager.compute_substep()`, which mjlab calls
**after** `sim.step()` and `scene.update()`. `scripts/render_policy.py` records
`substep_arm_qvel_rad_s` by wrapping that **same** `compute_substep` seam.

Precisely what this claims and does not claim. It claims the **sample point and the definition
are identical**: the same post-integration state, at the same 500 Hz rate, read at the same
seam — so enforcement and measurement are one quantity, not two correlated proxies. It does
**not** claim numerical equality across backends: training runs on CUDA and evaluation on CPU,
and that gap is declared under *Known gaps* and remains unmeasured. Both statements hold
simultaneously; neither is dropped.

### Unchanged from the frozen P control

Guideline geometry and the always-on tracker; the four guideline observations and the actor/
critic observation width; the eight base reward terms and weights; `r_waypoint_progress` at
weight **8.0**; `r_gate` **absent**; the fixed nominal joint reset `(0.0, 0.0)`; actuators,
`delta_pos_scale = 0.15`, fixed impedance (**no `set_gains`**); `imp_max_p = 0.0` (impulse
log-only); manufacturer caps `IMP_J_LIMIT = [1.640, 3.280, 1.640, 1.640, 1.640, 1.640]`;
CatPPO. P+V differs from P **only** in velocity-CaT enforcement and the substep peak tracker
that enforcement requires.

### Training-relevant source DOES change this wave — declared

Wave 2 could assert a byte-identical `src/` tree hash. Wave 3 cannot, and pretending otherwise
would be dishonest. Exactly three changes:

1. `SubstepPeakJointVel` gains `peak_qv_joint` (B, J), peak-held over the same control window,
   alongside the existing worst-joint scalar `peak_qv` (B,) that `diag_policy_trace.py` reads.
2. `CatSoftHook` gains an opt-in, **fail-closed** `vel_detection="substep"`. A missing tracker
   raises at env build; it never falls back to the control-rate sample.
3. `z1_hammer_env_cfg` gains `vel_cat_substep`; one new task is registered.

Invariance of the controls is therefore proven by **configuration digest**, not by tree hash:
the three frozen Wave-2 per-arm reward-config digests must still reproduce exactly at the
Wave-3 revision. `Unitree-Z1-Hammer-CaT-Soft` and every other shipped arm keep control-rate
detection, guarded by test.

## Endpoints

### Primary

**The number of the six P+V policies that are intrinsically legal** under the identical
fixed-reset evaluation.

**Legal** ⇔ every recorded 500 Hz post-physics arm-joint velocity sample satisfies
`max_j |q̇_j| ≤ 3.1415 rad/s`.

The **exact peak and the responsible joint** are reported for **every** policy — all twelve
(six P+V and the six frozen P controls) — whether legal or not.

Reported as *"intrinsically legal on the fixed-reset evaluation"*. **Never "hardware ready"**:
one fixed reset is one initial condition, not a coverage claim.

### Secondary

1. All six ordered gates crossed no later than the first contact-onset substep.
2. Pre-contact finite-segment perpendicular error, max and RMS.
3. Pre-contact path ratio.
4. Contact, task success, and clamped 32 mm nail depth.
5. Pre-contact hammer speed.
6. Delivered impulse.
7. Maximum per-joint Λ/cap.
8. Treatment payout and CaT δ behaviour.

### Evaluation is intervention-free

No action clipping. No deterministic velocity termination. No runtime velocity brake. The
measured legality must come from **learned behaviour**, or it measures the harness instead.

## Measurement conventions — carried over verbatim from the frozen P controls

- Perpendicular error uses the recorded `substep_perpendicular_error_m` field only — distance
  to the **finite** entry→nail segment with progress clamped to [0, 1]. Never re-derived
  against the infinite line.
- **Pre-contact** = substeps from reset through the first contact-onset sample, **inclusive**.
  For a policy that never contacts, the whole rollout.
- Gate completion, path ratio and pre-contact error share that one window.
- Nail depth is the **clamped** value (raw slide position includes elastic overshoot).
- Fixed-reset digest `bde511ec2adc42e5365e1e46f45ff1fb43223a93c31c6fbfb4580352e445e319`.

## The six frozen P controls (the paired baseline, fixed before Wave 3)

| Seed | gates@onset | perp max (mm) | perp RMS (mm) | ratio | peak &#124;q̇&#124; (rad/s) | legal | depth (mm) | Λ/cap | payout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 6 | 11.88 | 7.92 | 1.0971 | 4.9661 | no | 32.0 | 0.094214 | 0.16 |
| 3 | 6 | 11.50 | 5.48 | 1.0347 | 4.9749 | no | 32.0 | 0.050454 | 0.16 |
| 4 | 6 | 16.46 | 6.41 | 1.0344 | 4.9239 | no | 32.0 | 0.068371 | 0.16 |
| 5 | 6 | 16.17 | 6.70 | 1.0413 | 4.3751 | no | 32.0 | 0.076535 | 0.16 |
| 6 | 0 | 50.41 | 39.11 | 1.6372 | 4.9779 | no | 32.0 | 0.084569 | 0.001071 |
| 7 | 6 | 13.78 | 5.85 | 1.0298 | 4.6735 | no | 32.0 | 0.096834 | 0.16 |

P baseline: **0/6 legal**, gates **5/6**, strict straight label **3/6** (seeds 2, 3, 7).

## Decision rules — fixed before data

| Outcome | Condition | Action |
| --- | --- | --- |
| **Useful success** | ≥ 4/6 P+V qvel-legal **and** ≥ 4/6 still complete all six gates | Velocity CaT works at this dose on the guided arm. |
| **Partial** | 1–3/6 legal with guidance preserved | Report as partial. Consider a separate **dose** experiment. Do not tune inside this wave. |
| **Exploration collapse** | Legality improves but gate completion or task success collapses | This — and only this — motivates a separate deterministic **curriculum** experiment. |
| **No legality improvement, striking still succeeds** | Legality unchanged | Do **not** add reward terms. Investigate velocity-aware action limiting or deterministic termination as a separate wave. |

**Impulse enforcement is not activated under any outcome.** Λ remains far from its caps.

No post-hoc thresholds. No seed replacement. No population-level significance claim at n = 6.

## Seed and retry rule

- No **completed** policy may be replaced, filtered or re-run — including a weak, failed or
  zero-payout one. A completed weak policy is a result.
- A **documented infrastructure failure** may be retried only at the identical seed, task,
  configuration and revision.
- A different seed is **never** substituted.

## Known gaps, declared up front

- **Training runs on CUDA; evaluation runs on CPU.** `scripts/diag_impulse_trace.py` is
  CPU-only by design. CPU/CUDA substep differences remain unmeasured. Trajectory metrics and
  impulse metrics are never combined as though they came from one rollout.
- **One fixed reset.** The primary endpoint is legality at a single initial condition. It is
  evidence about the learned policy, not a hardware-readiness certificate.
- **n = 6 per arm.** Counts are exact out of six; no inferential claim is licensed.
- **No fail-closed launcher guard for `CAMPAIGN=wave3`.** No new launcher will be built —
  changing the launcher would change training-relevant source beyond the declared three edits.
  The gap is closed by the frozen ledger and the automatic startup check below.

## Out of scope for Wave 3

G+P; tube/corridor rewards; impulse enforcement (`imp_max_p > 0`); deterministic velocity
termination; 500-iteration training; curriculum learning; action limiting; variable impedance
/ `set_gains`; `r_imit`, GP, hard guidance, recorded-reference tracking; any reward term not
already in the frozen P arm.

## Pinned provenance

### Revisions

| Role | Revision |
| --- | --- |
| **Wave-3 training** | the commit that first carries this preregistration — recorded in the follow-up pin commit before submission. This is the revision Vega checks out. |
| Asset (`safe_impact_manipulation`, `hammer-z1`) | `b58ccd2f81fd246f27c1e8d88cf86484cd888703` — **identical to Waves 1 and 2.** No asset change is permitted. |
| Wave-2 training (frozen P controls, seeds 4–7) | `e1a0282c9dc7b0283ae632a46a78debfd80bdf8c` |
| Wave-1 training (frozen P controls, seeds 2–3) | `a6a9c970ea9eca00b34e8e9be806d22d396a964a` |
| Wave-2 result freeze | `cc125c08bfbdd7e75c2db83204c8eda8008464c9` |

### Control invariance is proven by DIGEST, not by tree hash

Wave 2 could assert a byte-identical `src/` tree hash against Wave 1. Wave 3 changes `src/`
by design, so that argument is unavailable and is **replaced**, not quietly dropped. The
three frozen Wave-2 per-arm reward-config digests must reproduce exactly at the Wave-3
revision — re-verified at deploy time, and a mismatch is a launch abort:

| Arm | Reward-config sha256 | Re-derived at Wave-3 revision |
| --- | --- | --- |
| C0 | `47d993698852dc939c753d978e41c9124e24c470e0595439f004d2b561d76fb9` | **MATCH** |
| G | `a5b767b22a77ecfc068eea9885ff88c4bedcd2035e5622627650a20216431e0f` | **MATCH** |
| P | `dca539d938e6eb15cb0ececebf78e9d2972870045edf6b52967d16f6945ed264` | **MATCH** |
| **P+V** | `dca539d938e6eb15cb0ececebf78e9d2972870045edf6b52967d16f6945ed264` | **identical to P** |

P+V's reward digest being *identical to P's* is the point: the velocity treatment is a
**metrics** term, so it cannot appear in the reward function. If this digest ever diverges
from P's, the treatment has leaked into the reward and the paired comparison is void.

### CPU qualification — executed at this revision

| # | Check | Result |
| --- | --- | --- |
| 1 | Substep peak at or below 3.1415 ⇒ zero velocity violation | PASS (δ = 0, max margin 0.000000) |
| 2 | Controlled above-limit sample ⇒ positive δ | PASS (one joint +0.75 rad/s ⇒ δ = 0.5, control-rate sample still legal) |
| 3 | Scripted reference playback legal at 500 Hz | PASS (16/16 resets, peak 2.530755 rad/s ≤ 3.1415) |
| 4 | P reward/gate behaviour unchanged | PASS (16/16 gates 6/6, contact, 32 mm; 0 mismatches across 15 shared columns × 16 rows vs the frozen artifact) |
| 5 | Eight base reward terms unchanged | PASS |
| 6 | `r_waypoint_progress` weight exactly 8.0 | PASS |
| 7 | `r_gate` absent | PASS |
| 8 | `imp_max_p` exactly 0.0 | PASS |
| 9 | Manufacturer impulse caps unchanged | PASS (`[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]`) |
| 10 | No `IMPACT_W`/`DELIVERED_W`/`NAIL_DRIVEN_W` leakage | PASS (all absent) |
| 11 | `validate_rewards.py` phases A–M | PASS (ALL PHASES PASSED) |
| 12 | `verify_contact_sensor.py`, `verify_reward_setup.py` | PASS (exit 0; all 4 non-exempt terms fired) |

Full serial suite at this revision: **2135 passed, 1 skipped**. Invariance proofs: **29/29**.
Live-env P+V qualification: **21/21**.

The reference qualification additionally reports `"source repository has uncommitted
changes"` until this preregistration is committed; it is re-run after commit and must then
report `passed: true`.

## Frozen launch ledger (6 rows) — not executed

Frozen before submission. Every field is fixed except the Slurm job ID and final state.
No row may be added, removed or re-seeded.

| # | Arm | Task ID | Seed | Run identity | Iters | Envs | Job ID | State |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P+V | `…-Guideline-CProgress-Vel` | 2 | `wave3_wave3_pv_seed2` | 200 | 4096 | *pending* | *pending* |
| 2 | P+V | `…-Guideline-CProgress-Vel` | 3 | `wave3_wave3_pv_seed3` | 200 | 4096 | *pending* | *pending* |
| 3 | P+V | `…-Guideline-CProgress-Vel` | 4 | `wave3_wave3_pv_seed4` | 200 | 4096 | *pending* | *pending* |
| 4 | P+V | `…-Guideline-CProgress-Vel` | 5 | `wave3_wave3_pv_seed5` | 200 | 4096 | *pending* | *pending* |
| 5 | P+V | `…-Guideline-CProgress-Vel` | 6 | `wave3_wave3_pv_seed6` | 200 | 4096 | *pending* | *pending* |
| 6 | P+V | `…-Guideline-CProgress-Vel` | 7 | `wave3_wave3_pv_seed7` | 200 | 4096 | *pending* | *pending* |

### Automatic startup check — every job, every field

The launcher has no `wave3` guard (building one would change training-relevant source beyond
the three declared edits). Each job's startup log is therefore checked **automatically**
against its ledger row. For all six jobs assert:

| Field | Requirement |
| --- | --- |
| Task ID | exactly `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel` |
| Seed | exactly the ledger row's seed |
| Iterations | exactly 200 |
| Environments | exactly 4096 |
| Code revision | exactly the pinned Wave-3 training revision |
| Asset revision | exactly `b58ccd2f…` |
| Clean state | 0 dirty files; provenance contains no `-dirty` |
| Reward overrides | `IMPACT_W`, `DELIVERED_W`, `NAIL_DRIVEN_W` all **absent** |
| `imp_max_p` | exactly 0.0 |
| Impulse caps | exactly `[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]` |
| Velocity CaT | `use_vel` true, `vel_detection` `substep`, `max_p` 0.5, `min_p` 0.0, `tau` 0.95, `limit` 3.1415 |

**On any mismatch, `scancel` the affected job immediately**, record the job ID and reason,
then retry at the identical seed. Sibling jobs that pass are unaffected.

## Independent review

Two independent read-only reviewers examined the implementation diff before commit.

| Reviewer | Scope | Verdict |
| --- | --- | --- |
| Gemini (`gemini-agent`) | full diff incl. all five test files | **REQUEST-CHANGES** — 8 findings |
| DeepSeek (`deepseek-v4-pro`) | source diff only (4 files) | APPROVED — 0 findings |

The disagreement is explained by scope, not by judgement: DeepSeek never saw the test files,
where two of the four blocking findings lived. Its APPROVED is therefore **not** treated as
corroboration. Both were assessed on the merits rather than tallied.

Resolved before commit:

1. **Fail-open enforcement (ship blocker).** `use_vel`, `max_p` and `vel_detection` are all
   tyro-exposed. `--env.metrics.cat-soft.params.use-vel False` (or `--…max-p 0.0`) would have
   made δ_vel ≡ 0 while the arm still looked like the velocity arm — and with the impulse
   constraint log-only, δ ≡ 0 means P+V trains as a bit-for-bit copy of P. Six GPU runs later
   *"CaT did not bound velocity"* would have been indistinguishable from *"CaT was never on"*.
   `_validate_params` now rejects both combinations at construction, and the startup check
   records `use_vel` / `vel_detection` / `max_p` per job.
2. **Joint-set guard compared column count, not identity.** Now compares resolved joint ids.
   (A permutation turned out to be harmless — `SceneEntityCfg.resolve` normalises to model
   order — so the test pins the *reachable* case: swapping `joint6` for `jointGripper` keeps
   six columns while shifting what every margin column means.)
3. **Construction-time fail-closed had zero coverage** — every hook test builds via
   `helpers.stub`, which bypasses `__init__`. Two real-env build tests added; deleting the
   eager resolve now turns the suite red.
4. **The byte-identity test compared the implementation against itself** (a determinism test).
   It now re-renders a frozen Wave-2 leaf's trace and matches the SHA-256 recorded in that
   leaf's own manifest (`d95b9bd04cef33c6473b124b96d145286b3cac842024edc6cd014326f0d68c13`),
   so the legacy plot path cannot drift away from the hash-bound frozen figures.

Also applied: the plot report returns `reference_endpoints_m: None` when no guideline is
drawn; the canonical-equality assertion's comment no longer overstates what it checks.

### Declared presentation asymmetry (review finding 5)

The six frozen P controls live in `wave2`, which is in `FROZEN_RENDER_CAMPAIGNS`, so their
committed figures carry the **legacy** geometry (line + six 15 mm gate disks) even though P
was rewarded for waypoint *approach*, not gate crossing. P+V figures will carry the corrected
waypoint geometry. The frozen P figures are **not** re-rendered — they are hash-bound.

Resolution, fixed now rather than at analysis time: the matched 6×2 P-versus-P+V grids are
built by a dedicated grid builder that draws geometry from the treatment table, so both
columns will use identical waypoint geometry. Any regenerated per-policy P figure is a
**presentation derivative**, written to a new directory and labelled as such. The frozen
Wave-1/Wave-2 leaves are never overwritten.
