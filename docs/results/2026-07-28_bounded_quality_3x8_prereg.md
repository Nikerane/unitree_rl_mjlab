# Bounded first-strike quality 3×8 preregistration

**Status:** frozen before implementation, training, or outcome inspection.

## Question

Does assigning PPO the contact-quality multiplier improve first-contact
centering when payout timing, speed snapshot, speed normalizer, saturation,
quality instrumentation, and delivered-reward weight are held fixed?

This is a simulator/training-distribution claim. It is not a per-strike causal
effect, a hardware claim, proof of globally straight motion, or impulse
enforcement.

## Frozen campaign

- Campaign ID: `fq3x8`
- Training seeds: exactly `16 17 18 19 20 21 22 23`, matched across arms.
- Training: 4096 environments, 500 iterations, fixed impedance, one GPU/run.
- Accepted checkpoint: `model_499.pt` only.
- Run names: `fq3x8_{f8,b8,fq}_seed{16..23}`; exactly 24 accepted rows.
- Retain every completed run, including weak seeds. Retry only documented
  infrastructure failures with the identical seed, task, and configuration.
- Code and asset worktrees must be clean and pinned. Any `-dirty`, unknown, or
  mismatched provenance invalidates the row.

## Arms

Let `V_FQ = 1.4598331451416016 m/s`.

| Arm | Registered behavior | Impact weight | Delivered weight |
|---|---|---:|---:|
| F8 | incumbent first-strike event-linear task | 8 | 2 |
| B8 | `clip(v_precontact / V_FQ, 0, 1)` | 8 | 0 |
| FQ | `q * clip(v_precontact / V_FQ, 0, 1)` and valid-quality gate | 8 | 0 |

B8 and FQ must be normalized whole-config identical except for
`impact_progress.func`. In particular they share:

- the same first-strike tracker and productive/finalized eligibility;
- the same one-shot payout timing and pre-contact speed snapshot;
- the same full-precision normalizer and saturation;
- the same passive eight-slot quality sensor and tracker instrumentation;
- the same observations, events, terminations, action space, fixed gains,
  reset distribution, inactive delivered reader, and all other rewards.

FQ versus B8 identifies the effect of assigning PPO the
quality-conditioned bounded reader under this contract. If any FQ episode has
invalid or overflowing quality, the treatment is reported as `q × valid`,
not “the multiplier alone.” FQ versus F8 is only a package/adoption
comparison because normalizer, saturation, and delivered reward also differ.

`V_FQ` was selected from the disclosed dirty, untracked historical calibration
bank documented in `2026-07-26_first_strike_quality_prereg.md`. Sharing it
between B8 and FQ prevents confounding the primary contrast, but it remains a
limitation of the FQ-versus-F8 package comparison.

## Hard guardrails

- `IMP_J_LIMIT` remains unchanged.
- `imp_max_p=0.0`; impulse soft-CaT remains log-only.
- No `set_gains`, variable impedance, trajectory reward, reference reward, or
  new impulse reward is part of this campaign.
- No installed `mjlab`, `rsl_rl`, `mujoco`, or `mujoco_warp` package is edited.
- Inherited `IMPACT_W`, `DELIVERED_W`, and `NAIL_DRIVEN_W` are unset and
  rejected.

## Strict evaluation

- 256 environments and exactly the first two completed sampled episodes per
  environment: 512 episodes/checkpoint, 12,288 total.
- Training configuration, stochastic sampled actions, reset noise
  `[-0.05,+0.05]` rad, actor corruption on, critic corruption off.
- Episode cutoff 4.0 s; sufficient control steps to satisfy the quota.
- New unseen RNG streams:
  - reset: `2036073019`
  - observation: `2046073033`
  - action: `2056073041`
- Decision fields must use the `*_sampled` columns.
- `impossible_success_n==0`, `lambda_dead_n==0`, finite numerical fields,
  complete quotas, clean provenance, and exact checkpoint/config identities
  are mandatory validity gates.
- B8 and FQ additionally require a live, finite, non-overflowing quality
  snapshot path. F8 has no quality-sensor requirement.

## Confirmatory decision rule

The unit of inference is the training seed. No episode is treated as an
independent replicate. All-episode quality includes no-contact episodes as
zero.

Primary endpoint:
`first_contact_quality_sampled`, averaged within checkpoint to one value per
training seed.

The mechanism criterion passes only if all three conditions hold:

1. `mean(FQ) - mean(B8) >= 0.10`;
2. exact two-sided paired sign-flip `p <= 0.05`;
3. repository-protocol two-sided Mann–Whitney across the eight seed summaries
   per arm has `p <= 0.05`.

The paired test matches the blocked design. Mann–Whitney is retained as the
required independent-sample sensitivity gate and is not described as the
exact blocked-design p-value. There is one mechanism contrast, so there is no
Holm family. Failure of any condition is “criterion not met/inconclusive,”
never equivalence or evidence of no effect.

Practical noninferiority is tested for FQ against each control
`C in {B8, F8}`. Every one-sided 95% paired-bootstrap lower bound for `FQ-C`
must be strictly greater than:

| Seed-level endpoint | Margin |
|---|---:|
| `first_window_useful_speed_mean_sampled` | `-0.20 m/s` |
| `event_window_depth_gain_mean_sampled` | `-0.001 m` |
| `first_window_success_rate_sampled` | `-0.05` |
| `overall_success_rate_sampled` | `-0.05` |

Bootstrap details: 100,000 matched-seed block resamples; NumPy `PCG64` seed
`20260728`; `np.quantile(..., method="linear")`; the 5th percentile is the
one-sided lower bound. Equality to a margin fails. These are AND gates
(intersection-union adoption rule), so no multiplicity correction is applied
to them.

The margins were chosen after observing the previous 4×8 campaign and are
therefore prior-result-informed for this fresh campaign. Their practical
rationales are: about 14% of the frozen speed knee, 1 mm versus the 30 mm
success threshold, and a conventional five-percentage-point success loss.

## Descriptive diagnostics

Report but do not test, gate, select arms, or change same-campaign rewards
using:

- full pre-contact trajectory and straightness/path-efficiency index;
- onset lateral speed, approach angle, axiality, and radial error;
- transverse and axial object-side impulse, contact dwell, and recontacts;
- per-joint `Lambda`, cap ratios/crossings, and peak joint velocity.

Any impulse crossing remains simulator-only and log-only. These diagnostics
choose the next experiment only after this campaign is frozen and reported.

## Pre-training CPU qualification (2026-07-28)

Implementation was qualified from a working tree based on
`0fba76bca7a10e46a618f0715db15da3f520a7b9`; the clean training revision is
recorded separately after the named-file commit.

- Changed-area regression:
  `pytest -q tests/test_first_strike_quality_reward.py tests/test_configs.py
  tests/test_smoke_first_strike_instrumentation.py
  tests/test_eval_impulse_hook.py tests/test_first_strike_campaign.py
  tests/test_slurm_launchers.py tests/test_impact_progress_reward.py`
  — **698 passed, 1 skipped**.
- Mandatory impulse/reward unit suite from `AGENTS.md` — **122 passed**.
- `docs/research/reward-design/validate_rewards.py` — **all phases A–M
  passed**.
- `docs/research/reward-design/verify_contact_sensor.py` — **passed**;
  the expected hammer-head primary resolved and all four environments detected
  first contact.
- `docs/research/reward-design/verify_reward_setup.py` — **passed** after
  correcting one pre-existing stochastic predicate: a uniform random policy
  often makes no hammer–nail contact, so `nail_depth_delta` can correctly stay
  zero below its 4 mm dead zone. Its liveness/reset/clamp behavior remains
  deterministically covered by validation phases C/E/F/L. Two pre-correction
  unseeded failures and the diagnosis are retained rather than hidden by
  rerunning until a lucky contact.
- `docs/research/reward-design/playback_reference.py` — **passed**; contact at
  step 11, `1.37 m/s`, and `30 mm` success, while the slow-press probe required
  67 steps.
- Explicit B8/FQ isolation smoke — **10 passed**: whole-config equality except
  the impact reader; below/at/above-knee bounded payouts; center-blind B8; live
  non-overflowing quality instrumentation.
- Final independent review found and then verified the fix for one Important
  issue: `--campaign fq3x8` now restricts evaluation to F8/B8/FQ and rejects
  any reset/observation/action RNG drift before checkpoint access or rollout.
  Final verdict: **SPEC passed; QUALITY approved**.
