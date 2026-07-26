# Preregistration — FQ4x8 first-contact-quality campaign (fixed impedance)

**Status: FROZEN 2026-07-26, before any `fq4x8` training job was submitted.**

> **Amendment 2026-07-27 (pre-training, disclosure-only).** Amended after independent review and
> BEFORE any `fq4x8` job was submitted — no outcome data existed at the time of amendment, so no
> post-hoc adaptation is possible. Changes are confined to making §6/§7 disclosures more precise
> and more explicit; **no decision rule, endpoint, threshold, contrast, seed, count, or figure/video
> rule was altered.** Specifically: (a) the q90 reproduction is now reported as bit-exact in the
> tracker's native float32, replacing an earlier claim of agreement "within a documented
> `2.09e-7` tolerance" — that tolerance had no independent source and was written to describe an
> already-observed float64 rounding offset; (b) added an explicit statement that the calibration
> bank is untracked in Git; (c) enumerated the four D0 mechanism conditions in prose; (d) restated
> V_FQ as "not an experimental result" in the owner's language; (e) clarified the sign-flip
> arithmetic as a conditional example, not a prediction.
Nothing below may be revised after training outcomes exist. Any change to a decision rule,
endpoint, threshold, or figure/video count after outcomes are visible invalidates the campaign.

- Campaign ID: `fq4x8`
- Branch: `first-strike-quality`; code HEAD at freeze: `4a6cd50`
- Asset repo (`safe_impact_manipulation`) HEAD at freeze: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`
- Design: `docs/superpowers/plans/2026-07-26-first-strike-quality-conditioned.md`
- Execution: `docs/superpowers/plans/2026-07-26-fq4x8-claude-execution.md`

## 1. Question

Does replacing the raw ante-impact speed reward with a **bounded, contact-quality-gated** speed
reward improve where the hammer first lands on the nail, without sacrificing useful work?

## 2. Treatments (frozen 4×8 matrix)

`fq` is the manifest short for the treatment labelled **FQ-min** in all human-facing reports.

| Label | Short | Registered task | Impact reader / weight | Delivered reader / weight | Speed normalizer |
|---|---|---|---|---|---|
| F8 | `f8` | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear` | `FirstStrikeImpactRewardTerm` / 8 | `FirstStrikeDeliveredRewardTerm` / 2 | 1.0 |
| F0 | `f0` | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0` | `FirstStrikeImpactRewardTerm` / 0 | `FirstStrikeDeliveredRewardTerm` / 2 | 1.0 |
| D0 | `d0` | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0` | `FirstStrikeImpactRewardTerm` / 8 | disabled / 0 | 1.0 |
| FQ-min | `fq` | `Unitree-Z1-Hammer-CaT-Impulse-Event-Quality` | `FirstStrikeQualityImpactRewardTerm` / 8 | disabled / 0 | 1.4598331451416016 |

FQ-min pays exactly, and only:

```text
8 * q_contact * clip(v_precontact / 1.4598331451416016, 0, 1)
```

Raw delivered impulse is a recorded **diagnostic** for FQ-min. It is never an FQ-min reward
component and never a practical-acceptance gate.

Frozen per-arm reward identity hashes (bind reader classes + weights + the FQ-min normalizer):

```text
F8      47d993698852dc939c753d978e41c9124e24c470e0595439f004d2b561d76fb9
F0      a8fdd61dc521a6a4294945d94e52dde8fde6560ee963976c08251e0996d42f9c
D0      c2c8f069f5f744eb063514b0c4a20e75ed9b271b382f2397281c04e7ac12f325
FQ-min  5bdd740a32cb730e1e63392422387dff5df2c8812d52587060249fe2b54a09a7
```

Strict config identity keys carried on every accepted row: `training_config_sha256`,
`evaluation_config_sha256`, `training_policy_observation_sha256`,
`evaluation_policy_observation_sha256`, `training_treatment_reward_sha256`,
`evaluation_treatment_reward_sha256`.

**Allowed differences only.** F8 vs F0: impact weight. F8 vs D0: delivered weight. D0 vs FQ-min:
impact reader, its normalizer, and the passive quality sensor FQ-min requires. All arms share
identical actions, actuators, gains, impulse caps, `imp_max_p`, and budgets.

## 3. Fixed-impedance invariants (unchanged across all 32 runs)

```text
arm Kp/Kd = 1000/100, joint 2 = 1500/150, gripper = 100/20
IMP_J_LIMIT = [1.640, 3.280, 1.640, 1.640, 1.640, 1.640]   (manufacturer caps, unchanged)
imp_max_p = 0.0                                            (impulse constraint is LOG-ONLY)
physics dt = 0.002 s (500 Hz), control decimation = 10 (50 Hz)
no set_gains, no commanded stiffness, no variable impedance
no superlinear excess-over-reference reward
```

## 4. Training protocol

```text
seeds            = 8 9 10 11 12 13 14 15   (identical in every arm)
iterations       = 500
training envs    = 4096
checkpoint       = model_499.pt
GPUs             = one per run
reward overrides = IMPACT_W / DELIVERED_W / NAIL_DRIVEN_W must be UNSET (fail closed)
```

Retries are permitted **only** for documented infrastructure failure, re-run with identical seed
and config. Every attempt is retained. A completed weak or unstable seed is never replaced, and
seeds are never selected on performance.

## 5. Evaluation protocol

```text
256 environments x first 2 completed stochastic episodes = 512 episodes per checkpoint
32 checkpoints x 512 = 16,384 verified schema-v3 episodes
reset RNG        = 2036072919
observation RNG  = 2046072933
action RNG       = 2056072941
reset noise      = +/- 0.05 rad
actor observation corruption ON, critic corruption OFF
passive quality instrumentation ON for EVERY arm (banked under its own evaluation config hash)
```

Only `*_sampled` fields are decision-eligible:
`first_contact_quality_sampled`, `first_window_useful_speed_mean_sampled`,
`first_window_success_rate_sampled`, `overall_success_rate_sampled`,
`event_window_depth_gain_mean_sampled`.

### Invalidation contract (fail closed)

Any of the following invalidates the **complete** campaign analysis — not just the offending row:

```text
impossible_success_n > 0      lambda_dead_n > 0        quality_overflow_n > 0
quality_nonfinite_n > 0       liveness_failure_n > 0   quota_failure_n > 0
non-schema-v3 quality artifact          any non-*_sampled decision field
dirty/unknown code or asset provenance  hash/config/manifest mismatch
incomplete quota (any count != 512 per checkpoint, != 32 rows, != 16,384 episodes)
```

Derived episode definitions (frozen):

```text
event_window_depth_gain = max(peak_depth_inside_event - depth_at_contact, 0)
contact_dwell_ms        = raw_contact_substeps_inside_event * physics_dt * 1000
recontact_count         = off_to_on_edges_after_onset_before_finalization
```

No-contact and zero-positive-normal-force physical episodes have all-episode contact quality
**zero** (they are not dropped). Overflow, nonfinite geometry, or a missing required snapshot
invalidates the complete evaluation row.

## 6. Inference (frozen before outcomes)

Primary causal contrasts — exactly three:

```text
F0 - F8          (does the raw speed reward matter?)
D0 - F8          (does the delivered-impulse reward matter?)
FQ-min - D0      (does quality-gating the speed reward matter?)
```

- **Sole decision family:** the repository's exact two-sided Mann–Whitney U test on the 8 seed
  summaries per arm, with **one** Holm correction across exactly those three p-values.
- **Sensitivity only:** the exact paired sign-flip test. (Arithmetic note, not a predicted
  outcome: *if* all eight paired differences were to come out strictly the same sign, the test
  would yield `p = 2/256 = 0.0078125`, since only the all-`+` and all-`−` assignments are then
  maximally extreme.) It can never pass or fail a causal decision or the FQ-min practical rule.
- **Intervals:** matched-seed paired bootstrap, `PCG64` seed `20260726`, `100,000` resamples.
  Bootstrap resamples matched seed blocks, never arms independently.
- All eight matched-seed differences are emitted for every contrast and every practical margin.

**D0 mechanism endpoints** (paired intervals; these do NOT enlarge the Holm family):
depth gain, dwell, recontact, raw delivered impulse, and both success endpoints.

The phrase "mainly selected dwell/recontact" is permitted **only** when all four of these frozen
mechanism conditions hold simultaneously (enforced in `_d0_mechanism()`; enumerated here so the
rule is auditable without reading source):

```text
raw_delivered_upper_below_zero        paired upper bound on the raw delivered-impulse
                                      difference lies below zero
dwell_or_recontact_upper_below_zero   paired upper bound on the dwell OR recontact difference
                                      lies below zero
depth_gain_ratio_lower_gt_0_90        one-sided event-window depth-gain ratio lower bound > 0.90
success_guardrails_pass               ALL FOUR of: D0 first-window success >= 0.90;
                                      first-window no more than 0.05 below F8;
                                      D0 overall success >= 0.90;
                                      overall no more than 0.05 below F8
```

### FQ-min practical acceptance vs F8 (all must hold)

```text
mean all-episode contact-quality gain          >= 0.10
one-sided useful-speed ratio lower bound        > 0.95
first-window success                           >= 0.90
overall success                                >= 0.90
first-window success no more than 0.05 below F8
overall success no more than 0.05 below F8
one-sided event-window-depth-gain ratio lower bound > 0.90
all safety / provenance / quota / liveness / numerical / sentinel gates pass
```

Raw delivered impulse is explicitly **excluded** from this gate.

## 7. Frozen FQ-min speed scale (V_FQ) — provenance, reproduced 2026-07-26

`V_FQ = 1.4598331451416016 m/s`, applied to FQ-min only. F8/F0/D0 keep the raw `1.0` normalizer.
Derived as the **calibration-split q90** with NumPy `method="higher"`.

| Required | Reproduced | Verdict |
|---|---|---|
| raw bank SHA-256 `ea4a82e007d95cf7ff962091ba0d6ff3e07e639f4cc1c2d0feab9fe368f47ac8` | `probe_raw.npz` | MATCH |
| resolved manifest SHA-256 `69d5bc66f2127d0a5b29463b35763092d388bc0e71661b06fdd81b0eea8d661f` | `bank_manifest.json` | MATCH |
| qualification SHA-256 `641e520c0cd35932175918d0bf48c7b6df07749ff1462b3d7da7310d76338b1b` | `probe_summary.json` | MATCH |
| quantile method `higher` | only `higher`/`nearest` reproduce it | confirmed |
| q90 `1.4598331451416016` | `1.4598331451416016` in float32 (abs diff `0.0`) | **BIT-EXACT** |
| calibration saturation `26/256` | `26/256` | EXACT |
| group-held-out validation saturation `33/128` | `33/128` | EXACT |

Population: 384 episodes; `split = calibration if training_seed in (0,1) else validation`
(256/128); strata `mx_maxoff`, `dc_imponly`, `dc_delonly`, `mx_maxmax` (96 each).
`v_precontact` is not stored per episode; it is reconstructed from `physical_traces[i].physical`
as `(head_position_m[fc-1].z - head_position_m[fc].z) / 0.002` at the first index `fc` with
`contact > 0` and `fc >= 1`.

**Dtype matters and is the whole story of the reproduction.** The production tracker computed this
quantity in **float32**. Reconstructing in float32 reproduces `V_FQ` **bit-exactly** (difference
`0.0`). Promoting the same trace to float64 before the finite difference instead yields
`1.4598332345485687`, an `8.94e-8` offset. That offset is a pure floating-point-precision artifact
of the reconstruction arithmetic — **not** a discrepancy in the banked data and **not** evidence of
tracker drift. No tolerance bound is claimed or needed: in the tracker's native dtype the agreement
is exact.

`V_FQ` is an **empirical scale/knee — not a safety threshold, not a sufficiency threshold, and not
an experimental result.** No post-outcome adaptation is permitted.

Delivered-impulse reference `I_REF_FIRST_STRIKE_SUCCESS = 0.3088 N·s` confirmed against the
production tracker: mean `0.30883467197418213 N·s`, n=8, sd `0.0`, range `0.0`.
Legacy normalizer `0.6094` observed at `0.6093726754188538`.

### Disclosure — the calibration bank is NOT versioned in Git

The three files above (`probe_raw.npz`, `bank_manifest.json`, `probe_summary.json`) are
**untracked data assets**, not tracked source. They are absent from tracked source in both this
repository and the sibling asset repository, and they are absent from this isolated worktree
entirely — they live only in the main working tree at
`docs/results/assets/2026-07-25_first_strike_reward/` and were read in place.

Consequently **this campaign's FQ-min scale constant is NOT fully reproducible from Git alone.**
Reproducing `V_FQ` requires those exact bytes, which are pinned only by the SHA-256 values recorded
above. Anyone re-deriving `V_FQ` must obtain the identical bytes and verify those hashes first.

### Disclosure — dirty provenance of the historical calibration bank

`probe_summary.json.provenance` records that the bank was generated from a dirty working state
(`repo.dirty=true`, head `5d986539d5ef7c9503df378467e3c5bf5c6875c4`; `assets.dirty=true`, head
`b58ccd2f81fd246f27c1e8d88cf86484cd888703`). This is a pre-existing frozen historical artifact,
not a campaign row. The "never accept dirty provenance" rule governs the 32 training and 32
evaluation rows, which are unaffected and are independently required to be clean. Recalibration is
forbidden by design, so this is disclosed rather than remediated. It is a known limitation on the
provenance of the FQ-min scale constant only.

## 8. CPU qualification evidence (2026-07-26, code `4a6cd50`)

```text
validate_rewards.py        ALL PHASES PASSED (A-M, incl. M1-M5 impulse-CaT arm; cat_delta ≡ 0, log-only)
verify_contact_sensor.py   VERIFIED — first contact at step 5, 10.947 N on all envs
verify_reward_setup.py     VERIFIED — all 5 non-exempt reward terms fired under random policy
playback_reference.py      PHASE M GATE PASS — in-script contact at step 11, v_contact 1.37 m/s,
                           SUCCESS at step 13 for approach 0.06/0.10/0.15 m; press exploit needs
                           67 steps (the anti-press speed margin)
```

## 9. Manifest contracts

`accepted_training_checkpoints.tsv` — exactly 32 rows. Each binds treatment label + short,
registered task, seed, `model_499.pt` path and SHA-256, attempt/retry history, code/asset/
campaign-config/treatment-config identities, fixed action/impedance/cap signatures, clean state,
and disposition. It carries **no** evaluation fields.

`accepted_evaluations.tsv` — exactly 32 rows. Each binds its training row to the evaluation
attempt, code/asset/config identities, the three RNG streams, schema-v3 payload digest and byte
hash, retry history, dirty state, the 512-episode quota, and every sentinel predicate.

Evaluation data lives outside the Git repository at `$HOME/unitree_rl_mjlab_eval`, with the
accepted manifest at `$HOME/unitree_rl_mjlab_eval/fq4x8/accepted_training_checkpoints.tsv`.

## 10. Outputs (counts frozen)

**Exactly two figures**, rendered before any video is selected:
`paired_seed_effects.png` and `aggregate_nail_plane_contact_map.png`.

**Exactly four explanatory videos**, selected strictly **after** statistics are frozen — one
medoid episode per arm by minimum standardized Euclidean distance to that arm's median over
contact quality, useful speed, first-contact time, raw delivered impulse, and the nail-plane axial
and lateral contact coordinates. Zero/nonfinite scale is replaced by `1.0`; ties break by
`(training_seed, env_id, episode_ordinal)`. Videos explain behavior; they cannot change a
decision, replace an episode, or supply quantitative measurements.

No HTML report, gallery, trajectory grid, medoid grid, quality/speed frontier, or force/impulse
grid is part of this campaign.

## 11. Interpretation limits (binding on the result write-up)

```text
a null result is NOT equivalence
a policy with finite qvel-rail exceedance is NOT hardware-speed-qualified
a log-only impulse result is NOT impulse enforcement
qvel and Lambda findings are SIMULATOR-ONLY
```

Claims must be separated into proven, supported, not distinguishable, simulator-only, and open.
