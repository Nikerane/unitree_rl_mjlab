# First-strike reward semantics: 4×8 fixed-impedance result

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer-derived cap”
> wording below refers only to the historical registered-task boundary, not a validated
> Z1 reaction-impulse or damage limit. The result body remains frozen; see
> `../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

**Date:** 2026-07-26
**Scope:** Unitree Z1 simulation, fixed impedance, log-only impulse characterization
**Status:** Complete — training, strict sampled evaluation, corrected
interactive report, and fixed-reset 4×8 trajectory grid; reporting code
independently reviewed and locally verified (154/154 tests)

## Executive result

The preregistered primary comparison passed. The success-censored
physics-rate first-event snapshot with one-shot control-boundary payout
(**E**) improved useful first-strike speed over the one-shot legacy-readout
comparator (**D-prime**) by **0.436 m/s (+38.1%)**: 1.582 versus 1.146 m/s
across eight independent training seeds per arm. The exact two-sided
Mann–Whitney and seed-label permutation tests both gave `p=0.0001554`;
`U=64`, `A12=1.0`. All preregistered success, provenance, quota, liveness,
sentinel, and numerical-finiteness gates passed.

This establishes a reward-**measurement/readout-package** result in the fixed
simulator: snapshotting the event at physics rate, then emitting its latched
one-shot payout at the control/reward boundary, is materially better than
retaining the one-shot, first-event-censored legacy 50 Hz readers. It does not
isolate censoring or one-shot payment, because D-prime already has both. It
also does **not** establish hardware safety, constraint enforcement, or a need
for a larger impulse-reward weight.

## Frozen design and provenance

- Arms: C, D-prime, F, E; eight training seeds per arm; 32 accepted jobs.
- Training: 4096 environments, 500 PPO iterations, final `model_499.pt`.
- Strict evaluation: 256 environments, first two completed episodes per
  environment, 512 sampled episodes per checkpoint, stochastic actions.
- Inferential unit: the eight checkpoint/training-seed means per arm.
- Code revision: `4a05ab762f21b3c4dc1036da0ee0db283a6bf357`.
- Z1 asset revision: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Accepted-manifest SHA-256:
  `4519528ca0c3b2d0e537730758344c312400a1872b80ea950b541a819875687f`.
- Summary CSV SHA-256:
  `233fec357c3aa2c91c75a84ed0ad3b7c30f150f01317b5d24bf1eeee6a77cd8d`.
- Analysis JSON SHA-256:
  `d1e1d3e8abc62ebba114f68153b9629b46567f228a07c09176f556016aaa8856`.
- Vega training jobs: C `40169311`, D-prime `40169312`, F `40169313`,
  E `40169314`; all 32 array elements completed with return code 0.
- Strict evaluation job: `40169367`; completed with return code 0.
- Reporting revision:
  `33c6606351d58eb86d033fd4baa98161dbbaf3cc` (clean fresh Vega checkout).
- Fixed-reset x-z grid SHA-256:
  `bacd2191236976458f3b1a30c051b3b27603abcfaca1423daf1405c745130570`.

The machine-readable evidence is banked in
`docs/results/assets/2026-07-26_first_strike_4x8/`.
The corrected attempt-3 reports and static grid are under
`attempt3_33c6606/`; the earlier attempt-2 outputs are retained for audit
history rather than overwritten.
This Git evidence package is deliberately compact. The 32 raw sampled NPZ
archives remain on Vega at the per-row `summary.csv::sampled_trace_path`
locations; every row binds its NPZ payload digest and artifact SHA-256. They
are not duplicated in Git because of their size. Claims that require
episode-level reconstruction (for example D-prime seed 5's 2/512 cap
crossings) depend on those bound raw archives.

## Preregistered inference

| Contrast | Useful speed (m/s) | Absolute effect | Relative effect | Exact MWU p | Exact mean-permutation p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| E vs D-prime (primary) | 1.582 vs 1.146 | +0.436 | +38.1% | 0.000155 | 0.000155 | Pass |
| F vs D-prime (mechanism) | 1.590 vs 1.146 | +0.444 | +38.8% | 0.000155 | 0.000155 | Interpretable; Holm-adjusted p=0.000311 |
| E vs F (saturation) | 1.582 vs 1.590 | −0.008 | −0.5% | 0.959 | 0.606 | Not distinguishable at n=8; preregistered direction failed |
| C vs D-prime | 1.347 vs 1.146 | +0.201 | +17.5% | — | — | Descriptive only |

For the primary, E's eight seed means were
`1.582 ± 0.038 m/s` (range `[1.525, 1.619]`) and D-prime's were
`1.146 ± 0.544 m/s` (range `[0, 1.459]`); the exact Mann–Whitney statistic was
`U=64`.

For E versus D-prime, first-window success was 98.68% versus 80.35% and
success by the 4 s evaluation cutoff was 100% versus 83.52%. The frozen
100,000-resample seed bootstrap gave a descriptive 95% interval of
`[0.143, 0.821] m/s` for the absolute effect. The separate success
noninferiority labeling gate passed.

The +38.1% endpoint is intentionally **success-weighted**: failed sampled
episodes contribute zero. It is therefore not a pure conditional velocity
effect. Among successful sampled episodes, the pooled precontact speeds were
1.603 m/s for E and 1.426 m/s for D-prime, a descriptive +12.4%. The
preregistered useful-speed endpoint remains the valid comparison because it
rewards both acquiring a valid first strike and arriving quickly.

The mechanism comparison localizes the gain to the physics-rate first-event
snapshot plus one-shot control-boundary reader package: linear F and saturated
E both strongly beat D-prime, while E did not beat F. Saturation was active in
44.46% of E episodes, so the unresolved E–F result is not explained by a dead
saturation branch. This was not an equivalence test: the correct conclusion
is **no detectable saturation benefit at eight seeds**, not that saturation
never matters.

## What happened to trajectory geometry

Arm-level summaries from the preregistered sampled diagnostics are:

| Arm | Useful speed (m/s) | First-window success | Within 12 mm | Transverse error (mm) | Lateral excursion (cm) | Path ratio | Terminal >1.25-radius predicate | Recontact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 1.347 | 99.46% | 68.41% | 9.90 | 5.17 | 1.368 | 33.45% | 86.30% |
| D-prime | 1.146 | 80.35% | 20.95% | 16.57 | 5.29 | 1.630 | 76.25% | 66.26% |
| F | 1.590 | 98.90% | 30.71% | 14.90 | 4.49 | 1.337 | 70.53% | 31.30% |
| E | 1.582 | 98.68% | 25.98% | 15.20 | 4.28 | 1.334 | 74.71% | 33.79% |

The corrected event measurement/readout package improves speed, reliability,
and path ratio and reduces recontact relative to D-prime, but E and F retain
substantial curvature/centering indicators. E still averages 4.28 cm of
lateral precontact excursion, a 1.334 path-length ratio, 15.2 mm transverse
contact error, only 26.0% of episodes entering within the 12 mm nail radius,
and 74.7% on the terminal large-error predicate. C is substantially more
accurate at first contact (68.4% within radius) despite lower useful speed.
These are secondary/descriptive diagnostics with no preregistered straightness
or physical-harm threshold; the fixed-reset audit shows a repeatable S-shaped
family but does not establish harm or population causality.

The code and earlier figures called the final diagnostic “late lateral
re-expansion,” but that name overstates what was measured. The frozen metric
only asks whether the maximum transverse error in the last 10 ms exceeds
1.25 nail radii; it does **not** require error to increase. It is therefore
reported here as a terminal large-error predicate. A genuinely
increase-based re-expansion metric must be frozen before the next campaign.

The contact-conditional entries above are episode-pooled using their recorded
eligibility counts. This matters for D-prime: one seed had zero eligible
contacts, and treating its stored conditional zero as a real seed mean would
artificially improve the arm's error, excursion, path-ratio, and
terminal-large-error summaries. Within-12-mm rate is denominator-safe by
design: all 512 episodes per seed enter and no-contact is false.
In particular, D-prime's contact-conditional geometry uses 3,566 eligible
episodes; the other arm-level entries are seed means unless explicitly
identified as eligibility-weighted.

As post-hoc hypothesis generation only, within each of F and E the seed-level
impact return was strongly associated with **lower** within-12-mm rate
(`rho=-0.929`, unadjusted `p=0.00086` in each arm) and **higher** transverse
error and terminal large-error rate. These eight-seed correlations are
multiplicity-unadjusted and behavior determines the payout, so they are not
causal evidence. Their replicated direction is a reason to test
`impact_progress` directly rather than immediately adding a new path reward.

These geometry metrics are secondary diagnostics. They do not by themselves
authorize adding a trajectory-reference reward. The corrected report contains
an interactive audit plus a fixed-scale 4×8 x-z grid. The static grid uses the
same stochastic evaluation coordinate (`env 0`, `episode 0`) and the same
reset/action RNG streams for every policy; it is a visual cross-check, not the
inferential unit. Its `v_pre` label is actual accepted-onset speed, offset is
radial x-y error, and Viridis is normalized episode time. Only the
provenance-bound nail x-axis is drawn because nail z was not stored in the
frozen trace schema. Red points mark every raw contact sample; the accepted
onset that defines the numeric label is not separately marked in this version,
so individual recontacts cannot be identified from the static panel alone.

The first coordinate-selected representative audit already shows why one
trajectory cannot stand in for a population: the chosen F episode is inside
the nail radius although only 30.7% of F episodes are, while the chosen C
episode is just outside although 68.4% of C episodes are inside. Population
metrics determine the conclusion; paths explain possible mechanisms.

The all-seed fixed-reset grid adds one robust qualitative observation: the F
and E policies reproduce a similar S-shaped approach across almost every
training seed. The curve is therefore not a single-policy accident, and
saturation does not visibly change its family. C is also curved while being
far more accurate in the population, so global straightness is not yet the
right optimization target; terminal contact alignment is the separable
failure. D-prime visibly contains both ordinary inaccurate strikes and
qualitatively collapsed behavior (including one no-contact panel and one
large path that exits the task view), consistent with its lower sampled
success. These statements concern one common reset coordinate and remain
mechanism evidence, not estimates of arm-level rates. Future morphology grids
should mark the accepted onset separately and freeze signed x/y terminal
offsets; neither change is needed for the current population inference.

## Impulse and hardware boundaries

The manufacturer-derived `IMP_J_LIMIT` caps were unchanged and
`imp_max_p=0`, so the impulse constraint remained log-only.

- E's mean per-seed worst-joint ratio was 0.610 and its maximum seed value was
  0.737; F and C also stayed below one.
- D-prime had one seed with a sampled cap crossing (maximum ratio 1.055):
  joint 1 crossed in only 2/512 episodes. In the separate mean-action
  diagnostic, the same seed's joint-4 maximum was 1.191× and joint-4 p95 was
  0.203×; the seed-wide worst p95 was joint 1 at 0.382×. This is a sparse tail
  crossing, not meaningful controlled binding.
- The campaign therefore does **not** show a useful policy–constraint
  trade-off and gives no basis for enabling enforcement.

Every arm violated the finite 3.1415 rad/s manufacturer joint-speed envelope
in sampled simulation. The E-versus-D-prime comparison contained violations
in all 16 seed rows, with a campaign maximum of 7.808 rad/s. The fixed-simulator
reward comparison remains valid under prospective amendment A1, but no result
may be called hardware-speed-qualified or ready for Z1 deployment.

## Proven versus open

**Proven by this campaign**

1. The physics-rate first-event snapshot plus one-shot control-boundary reader
   package is load-bearing relative to one-shot legacy readers; the design
   does not isolate one-shot/censoring semantics by themselves.
2. Both corrected physics-rate-snapshot arms strongly outperform the one-shot
   legacy comparator on the preregistered useful-speed endpoint.
3. Saturating the delivered payout at the qualified reference provides no
   detectable benefit over the linear payout at eight seeds; equivalence was
   not established.
4. E and F retain secondary curvature/centering indicators; the fixed-reset
   audit shows a repeatable path family but does not establish harm or cause.

**Still open**

1. Whether—and if so which—existing reward term causes the remaining terminal
   mis-centering and curved approach.
2. Whether a phase-indexed position-only reference is necessary after the
   responsible existing term is isolated.
3. How to respect the hardware joint-speed envelope without destroying task
   success.
4. Whether a physically justified task/contact regime can reach and bind the
   unchanged per-joint impulse caps meaningfully.
5. Whether the modeled contact is genuinely impulsive or predominantly a
   contact-model-dependent press.

## Current decision and next experiment

Retain the linear physics-rate event-snapshot package (**F**) as the diagnostic
baseline. Do not increase the delivered-impulse weight, enable impulse
enforcement, or add a trajectory-reference reward from this result alone. The
highest-information next experiment is a fresh, concurrent
**2 arms × 8 training seeds** ablation:

- **F8 control:** `impact_progress=8`, `delivered_impulse=2`.
- **F0 treatment:** `impact_progress=0`, `delivered_impulse=2`.
- Everything else identical: tracker/readers, gains, action space, caps,
  `imp_max_p=0`, training budget, checkpoint choice, and strict 256×2
  stochastic evaluation.

The primary endpoint is all-episode
`first_contact_within_nail_radius_rate_sampled` with no-contact=false. F0 wins
only if both exact two-sided seed-level Mann–Whitney and exact seed-label
mean-permutation tests give `p<0.05`, the direction is F0>F8, and the absolute
gain is at least 10 percentage points.

The preregistered guardrails should require:

1. one-sided 95% seed-bootstrap lower bound for the useful-speed ratio
   `F0/F8 > 0.95`;
2. F0 first-window and overall success at least 90% and no more than five
   percentage points below F8;
3. one-sided 95% lower bound for the all-episode, success-weighted delivered
   first-event impulse ratio `F0/F8 > 0.90`;
4. the existing provenance, quota, liveness, sentinel, and numerical
   finiteness gates.

Before any GPU training, freeze and test the all-episode endpoint plus direct
signed x/y accepted-onset offset diagnostics on the existing raw traces;
replay one F action tape under F8/F0 to prove physical identity and that only
the weighted impact payout changes; validate the exact config-hash difference;
then repeat the standard reward/contact/random-policy gates and reference
playback.

The dedicated preregistration must additionally freeze the exact all-episode
success-weighted impulse estimator/field and episode definition; the bootstrap
ratio estimator, within-arm resampling scheme, RNG seed, resample count, and
strict decision boundary; signed x **and** y accepted-onset offsets; and a
genuinely increase-based re-expansion metric with tests. Until those items are
frozen, F8/F0 is designed but **not cleared for launch**. Eight seeds is a
large-effect screen; a null result will not establish equivalence.

This order separates diagnosis from remedy. If removing `impact_progress`
improves centering without materially sacrificing useful speed, success, or
delivered impulse, the existing speed bonus caused the trade-off. If it does
not improve centering, the `impact_progress` hypothesis is falsified; the next
diagnostic must separate remaining center-blind task rewards from
kinematic/controller limitations. A late, phase-indexed position-only
corridor/reference remains a lower-ranked separate experiment, not the
automatic response to F0.

## Literature-constrained remedy ladder

The independent literature review agrees with the experimental order above:
do **not** add a dense reference/path term before F8/F0. The closest task
precedents anchor task performance to contact or terminal validity:

- Büchler et al. multiply landing accuracy by post-impact speed so either weak
  component collapses the table-tennis smash reward
  ([arXiv:2006.05935](https://arxiv.org/pdf/2006.05935)).
- GIFT pays goal-direction peg acceleration only at the tool’s first contact;
  its hammer task also zeros the task reward after dropping the tool or
  gripper–peg contact and includes a separate bounded distance penalty
  ([RSS 2021](https://roboticsproceedings.org/rss17/p060.pdf)).
- A real-world table-tennis system uses terminal validity tiers and applies
  detailed skill reward only inside the valid-return tier
  ([Nature, 2026](https://www.nature.com/articles/s41586-026-10338-5)).
- Potential-based shaping can accelerate discovery but cannot repair a base
  objective whose optimum prefers fast off-center contact
  ([Ng, Harada, and Russell, 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)).

Collectively, these precedents support event-anchored payout and, in Büchler
and the Nature table-tennis system, coupling performance to validity or
accuracy. GIFT supports first-contact timing but not the multiplicative
anti-compensation claim.

Therefore, if F0 restores centering but loses unacceptable speed, the next
candidate is a **physics-rate-latched, contact-quality-conditioned
first-strike utility** with one-shot control-boundary payout, not trajectory
imitation:

\[
e_\perp =
\left\|(I-aa^\top)
\left(p_{\mathrm{head}}(t_c^-)-p_{\mathrm{nail}}(t_c^-)\right)\right\|,
\qquad
r_{\mathrm{strike}} =
\mathbf{1}[\text{accepted productive first event}]
\,q(e_\perp)\,s(I_{\mathrm{delivered}}).
\]

Here \(q\) is a smooth approximation to the frozen 12 mm acceptance radius and
\(s\) is bounded (`min(I/I_ref, 1)` or `tanh`), and \(a\) is the unit nail
axis. Multiplication/hierarchical eligibility prevents unbounded impact
strength from buying its way out of a bad contact. The tracker would need one
new immutable onset-position snapshot. If F0 does **not** improve centering,
the next diagnostic must distinguish remaining center-blind reward semantics
from kinematic/controller limitations; a late ante-impact axis corridor is a
lower-ranked fallback. Full trajectory imitation, velocity tracking near
contact, residual/two-level control, and variable impedance remain out of
scope for this fixed-impedance diagnosis.
