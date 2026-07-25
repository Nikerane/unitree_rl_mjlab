# First-strike reward semantics

**Date:** 2026-07-25
**Scope:** Z1 fixed-impedance, log-only impulse-CaT (`imp_max_p=0`)

## Problem

The current 50 Hz `ImpactProgressTerm` estimates velocity across a complete
20 ms control interval and can pay again when a later productive contact starts.
The 500 Hz delivered-impulse accumulator is read at 50 Hz and its episode total
can reward later re-strikes. Controlled reward decomposition shows the resulting
failure:

- impact-only reward creates swing, dwell and taps without increasing true
  contact speed;
- delivered-only reward creates a straight, higher-force press;
- the combined reward blends both strategies.

This is an event-credit problem. It is not caused by the log-only Λ constraint,
and it should not be hidden with a lateral-path or generic smoothness penalty.

## Approved behavior

Add one shared 500 Hz `FirstStrikeEventTracker`. It owns the event definition
consumed by both corrected rewards. Existing episode-total delivered impulse,
per-joint Λ, caps, gains and action scale remain unchanged.

Per environment, the tracker has four states:

```text
UNARMED -> ARMED -> ACTIVE -> FINALIZED
```

- `UNARMED`: after reset, observe two consecutive off-contact substeps so the
  previous substep's head position and clamped nail depth are valid.
- `ARMED`: retain the immediately previous substep's head position and clamped
  nail depth.
- `ACTIVE`: on the first contact rising edge, latch the incoming axial velocity
  from `(p_onset - p_previous) / physics_dt` and latch the *previous*
  substep's clamped nail depth. In mjlab 1.4's callback order the rising-edge
  site/contact sample is the pre-force onset state; using the last off/off
  finite difference would be one physics substep stale. For exactly 25 physics substeps
  (50 ms wall time from onset), accumulate positive downward object-side
  impulse only on contact substeps and track maximum nail depth on every
  substep. Short raw-contact gaps remain inside the same strike window.
- `FINALIZED`: finalize early, inclusively, on the substep that crosses the
  normal success threshold; otherwise finalize after the 25th wall-time
  substep. Success takes precedence when it occurs on substep 25. Ignore every
  later contact and depth change.

The resulting delivered quantity is deliberately success-censored:

- successful event: impulse through the inclusive success-crossing substep;
- event that does not succeed: impulse through at most 25 wall substeps.

It is a bounded shaping quantity, not a common full-contact impulse estimand.
Every report must separate success-finalized and window-finalized events.

The first strike is productive when:

```text
peak_depth - depth_at_first_contact > 0.0005 m
```

Corrected reward terms each have an independent `paid` latch:

```text
impact = v_precontact / v_expected
event_linear = I_first_strike / I_ref_event
event_saturated = min(event_linear, 1)
```

Each pays once, after finalization, only for a productive first strike. The
delivered saturation is deliberate while enforcement is log-only: it is
reference-sufficient impact shaping, not unlimited constrained maximization.

The legacy `I_REF_DELIVERED=0.6094 N·s` remains unchanged and continues to
belong to the shipped full-control-step accumulator. Add a separately named
`I_REF_FIRST_STRIKE_SUCCESS` for the success-censored event horizon. Derive it
from the exact default scripted reference (`min_windup_clearance=0.05`) with
the new tracker, using the same repeated-reference provenance discipline as the
legacy constant. The current development observation is approximately
`0.308835 N·s`; it is not frozen until the derivation gate reproduces it.
Record both values and their ratio, but do not require them to agree.

Later contacts may still earn ordinary geometric depth progress and completion.
The experimental arm does not add first-attempt failure termination.

## Experimental isolation

Do not silently replace the shipped task. Register separate event-correct
fixed-impedance tasks. The event package contains:

1. shared first-strike tracker;
2. corrected one-shot impact reward;
3. corrected one-shot first-strike delivered reward, selectable as linear or
   saturated at `I_REF_FIRST_STRIKE_SUCCESS`.

Keep the current task as the control. Keep `imp_max_p=0`; do not change caps,
gains, `delta_pos_scale`, reference tracking, or installed packages.

### 2026-07-25 comparator amendment (approved after the frozen-bank falsification)

The first frozen 384-episode bank falsified the proposed dose-matched legacy
arm. A single multiplier calibrated on complete checkpoint groups produced
held-out discounted-return error of 9.17% overall and 11.03--43.67% in the four
behavior strata. That failed D remains archived as exploratory negative
evidence. Do not retune it on the validation bank, relax its thresholds, or
train it as the confirmatory comparator.

Replace it with **D-prime**, a one-shot, first-event-censored legacy-readout
package. D-prime shares the production `FirstStrikeEventTracker` with F and E,
but evaluates the existing 50 Hz legacy readers unchanged:

- evaluate `ImpactProgressTerm` and `DeliveredImpulseTerm` on every control
  boundary so their internal state remains production-faithful;
- before first-strike finalization, latch only the **first positive** legacy
  impact pulse (a later positive impact pulse is another complete velocity
  estimate, not an increment);
- sum positive legacy delivered pulses through and including the control
  boundary on which the tracker is first observed finalized (those pulses are
  disjoint increments of the cumulative legacy measurement);
- on that same boundary, pay the two latched components once iff the shared
  event is productive, then consume the arm permanently even when
  unproductive; delayed recontact never changes or repays it;
- retain the legacy `v_expected=1.0` and
  `I_REF_DELIVERED=0.6094 N.s`.

D-prime, F and E therefore share the same first-event eligibility,
finalization boundary, productive gate and one payout opportunity. All four
arms use the same nominal maximize weights 8/2. No arm is post-hoc
dose-matched. The adjacent comparisons are treatment **packages**, not
perfectly atomic mechanisms:

- C versus D-prime: repeated-credit/censoring/payout-timing package;
- D-prime versus F: 50 Hz legacy measurement/gating/normalizer package versus
  the 500 Hz first-event measurement package;
- F versus E: clean linear-versus-saturated event-payout contrast.

## CPU gates

Before Vega:

- synthetic phase-shift replay of the identical physical event at all 10
  physics phases inside a control interval;
- 10/10 detection, invariant event values, and exactly one payout from each
  corrected reward;
- pre-contact velocity error no larger than `max(0.02 m/s, 2%)`;
- discriminating tests for onset-vs-stale velocity, previous-substep contact
  depth, and success exactly on window substep 25;
- contact wholly inside one control interval and contact spanning a boundary;
- short contact gaps remain in one 50 ms strike window;
- delayed recontact after the window earns no corrected reward;
- scrape/no-progress, timeout, reset-mid-contact and partial-env reset cases;
- success-crossing substep is included; later same-step force is excluded;
- offline existing-trace rescore rejects `maxmax_s2`'s +148 ms recovery credit;
- existing impulse/reward pytest set, `validate_rewards.py` A-M and
  `verify_contact_sensor.py` all pass;
- `verify_reward_setup.py` passes;
- the legacy scripted-reference path still reproduces
  `I_REF_DELIVERED=0.6094 N·s`;
- the event scripted-reference path reproducibly derives
  `I_REF_FIRST_STRIKE_SUCCESS` within 1% across its frozen repeated runs;
- a frozen force-bearing CPU event bank records 2 ms contact, head position,
  clamped depth, net axial force, 500 Hz joint speed, actions, checkpoint hash,
  training seed, reset seed, trace digest, finalization reason and actual
  production reward-manager payout streams for C, D-prime, E and F;
- the bank manifest is frozen before payout analysis. Calibration and validation
  split by complete checkpoint/training-seed group, never episode. All
  previously inspected rows are development-only. The new validation bank uses
  preregistered reset seeds that have never been inspected;
- one fixed complete calibration bank is used. There is no prefix selection,
  retry-until-pass, subset search or tolerance adjustment;
- identical recorded actions replayed from identical resets produce equal
  physical traces across reward arms before payout comparisons;
- D-prime's inner 50 Hz samples match the standalone C readers
  boundary-for-boundary before censoring; its first positive impact pulse,
  summed delivered increments, finalization-boundary inclusion, one-shot
  payout and permanent consumption match the definition above;
- D-prime, F and E pay on the same finalization control boundary, and all four
  arms retain weights 8/2. Per-stratum reward-return differences are reported
  as treatment diagnostics, never forced into a dose-equivalence tolerance;
- raw traces are stored once per episode in the compressed artifact. The JSON
  summary is aggregate-only and binds the raw artifact, manifest, checkpoints
  and relevant source/config files by recomputed SHA-256;
- every independent validation failure is retained in the summary. Validation
  must not stop reporting after the first failed stratum.

### CPU qualification status (2026-07-25)

The frozen 384-episode production replay and full CPU pre-training gate now
pass at commit `5d986539d5ef7c9503df378467e3c5bf5c6875c4`: 237 reward/impulse
tests, `validate_rewards.py` phases A--M, `verify_contact_sensor.py`, and
`verify_reward_setup.py`. The replay passed 14/14 ordered gates, 12/12
stochastic proofs, 10/10 phase detections, production-stream and cross-arm
physical-equality checks, D-prime reader fidelity, and the legacy/event
normalizer gates. Exact commands, runtimes, hashes, diagnostics and the
preserved failed D result are banked in
`docs/results/2026-07-25_first_strike_reward_qualification.md`.

This proves the CPU reward-package implementation and semantics; it does not
predict PPO ranking. The sampled-evaluator `impossible_success_n==0` and
`lambda_dead_n==0` sentinels, the 500 Hz hardware-speed acceptance gate, CUDA
instrumentation, primary E-versus-D-prime comparison and success guardrails
remain open. Do not start the 4×8 matrix before a clean Task-4 CUDA smoke.

## Training comparison

After a clean CUDA instrumentation smoke, train eight fixed seeds per arm:

- **C:** current semantics, weights 8/2;
- **D-prime:** first-event-censored one-shot legacy readout defined above,
  weights 8/2;
- **F:** event semantics with linear delivered shaping, weights 8/2;
- **E:** event semantics with delivered shaping saturated at
  `I_REF_FIRST_STRIKE_SUCCESS`, weights 8/2.

The original first-window success-rate primary was rejected before
implementation: 11/12 existing fixed-reset traces already succeed inside
50 ms, so a +10 percentage-point target is ceiling-limited.

Primary metric:

```text
first_strike_useful_speed_mean_sampled
```

For every sampled episode define `Y = S_first * v_precontact`, where
`S_first=1` only when the normal success threshold is crossed in the first
event window and `Y=0` for failure or no contact. Average episodes within a
trained policy; the eight independent training seeds are the inferential units.
Conditional successful speed remains a secondary mechanism metric.

The sole primary comparison is **E versus D-prime**. E must improve the seed-level
mean of `Y` by at least 10% relatively, with both:

- two-sided exact seed-level Mann-Whitney `p < 0.05` (repository protocol),
  computed by enumerating all `C(16,8)=12,870` seed-label assignments with
  midranks so zero/tied outcomes remain exact;
- two-sided exact seed-label permutation `p < 0.05` for the difference in
  seed-level means, using the same exhaustive assignments.

Report `U`, tie-adjusted `A12=U/64` oriented as E over D-prime, absolute and relative
mean effects, per-arm seed mean/SD/range, and a separately labelled seed-level
95% interval from 100,000 within-arm bootstrap resamples with a frozen RNG
seed. The bootstrap interval is not described as exact permutation inference.
If the D-prime mean is non-positive, the relative-effect comparison is invalid rather
than automatically passing.

First-window and overall sampled success must each remain at least 90% and no
more than five percentage points below D-prime. These are point-estimate guardrails,
not formal noninferiority. The word *noninferior* is allowed only if the
one-sided 95% lower seed-bootstrap bound for both E-minus-D-prime success differences
exceeds `-0.05`; this is an intersection-union gate. Any dirty hash,
`impossible_success_n > 0`, `lambda_dead_n > 0`, or 500 Hz hardware-speed
violation invalidates the comparison.

Only if E-versus-D-prime passes may the mechanism family be interpreted:

1. F versus D-prime: event-measurement package without saturation;
2. E versus F: saturation shape with identical event semantics.

For each mechanism contrast take
`p_joint=max(p_MWU,p_permutation)` and apply Holm correction across the two
`p_joint` values. Require the hypothesized direction before interpretation.
Apply the same success guardrails to F before interpreting F versus D-prime.
C versus D-prime is a descriptive repeated-credit/censoring/timing package
contrast, not a pure latch effect. Eight
seeds per arm make this a strong screening/thesis result for large separation;
a null is “not distinguishable at n=8,” never equivalence.

All arms are evaluated through identical event instrumentation with stochastic
policy actions, training-matched `±0.05 rad` reset noise and training-matched
observation corruption using frozen unseen evaluator RNG streams. Fixed-reset
or mean-action runs are diagnostic only. Each trained seed contributes exactly
the first 512 completed sampled episodes, balanced as two episodes from each of
256 environments; a policy-dependent fixed control-step budget is forbidden.
The evaluator snapshots the tracker before autoreset and stores raw substep
traces so first-window, recontact and tail definitions have one source of
truth. Failure, no contact, and an unfinished event at the frozen evaluation
horizon contribute `Y=0`.

Success-censored and window-censored impulse have different horizons and are
never pooled as a common impulse estimand. Report separate sampled columns and
denominators for success-finalized, window-finalized, and no-contact episodes.
The mixed overall delivered mean is shaping diagnostics only. Likewise, report
discounted maximize, impact and delivered returns separately, plus event
saturation rate; an ambiguous undifferentiated “payout” column is forbidden.

All arms use weights 8/2. The reward-manager iterable already returns
`raw × configured_weight`; do not multiply captured streams by 8/2 a second
time. Average episodes within checkpoint/training-seed group first, then
average groups equally. Report discounted and undiscounted payout,
impact/delivered component mix, saturation fraction, payout timing and every
stratum separately. These are treatment diagnostics; do not describe them as
proof that reward dose is equal. Post-hoc common or component-specific
multipliers are forbidden.

Train every arm for 500 iterations with 4096 environments and retain the final
`model_499.pt` for seeds `0…7`. A completed poor or unstable seed is data and
cannot be replaced. Retry only a documented infrastructure failure with the
identical arm/seed/config, retain every attempt, and block analysis until the
complete valid 4×8 matrix exists.

Plots are Plotly only: seed-level primary outcome, contact structure,
reward-payout mechanism, and representative contact-aligned force/depth/impulse
traces.
