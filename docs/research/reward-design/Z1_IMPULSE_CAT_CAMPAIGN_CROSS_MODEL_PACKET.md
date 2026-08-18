# Z1 impulse-CaT next-campaign cross-model review packet

## Material Passport

- Origin: Z1 impulse-CaT thesis campaign design
- Content class: sanitized aggregate experimental design
- Origin date: 2026-08-18
- Verification status: NOT_SENT
- Raw traces included: no
- Source code included: no
- Checkpoint binaries included: no
- Private notes or thesis manuscript included: no

## Use boundary

This file is the complete content that may be sent independently to external reviewers. Do not add
repository files, raw traces, checkpoint data, credentials, private notes, or another model's
response. Record the provider, exact model ID, date, prompt SHA-256, and response SHA-256.

Requested reviewer labels are Kimi 3, GLM 5.2, DeepSeek V4 Pro, and Qwen. The exact provider model
ID must be verified rather than inferred from the label. If authenticated access is absent, record
UNAVAILABLE; never simulate that review with another model.

External responses are advisory hypotheses. Agreement is not evidence, disagreement is not a veto,
and no majority vote selects a treatment.

## Scientific context

The system is a simulated Unitree Z1 hammering task. A variable-impedance PPO policy controls six
joints. Velocity soft-CaT is active. The new constraint measures per-joint reaction impulse:

- baseline-subtracted and contact-masked;
- accumulated at 500 Hz;
- a rolling 25-substep/50 ms Lambda window;
- compared independently at all six joints;
- combined with velocity pressure by exact max soft-OR.

Soft CaT changes PPO continuation/credit through a graded probability. It is not a runtime clamp.
Neither threshold vector below is a manufacturer damage limit:

- provisional project caps: [0.820, 1.640, 0.820, 0.820, 0.820, 0.820] N.m.s;
- diagnostic-only 0.9 caps: [0.738, 1.476, 0.738, 0.738, 0.738, 0.738] N.m.s.

## Existing experiment

One matched training seed compared:

- control: velocity CaT active, impulse CaT log-only (imp_max_p=0);
- target: velocity CaT active, impulse CaT active (imp_max_p=0.5);
- diagnostic-only 0.9 caps;
- 4,096 environments, 24 rollout steps, 500 PPO iterations.

Frozen no-learning evaluation used one deterministic 64-world population and three paired
4,096-world stochastic populations.

Observed aggregate results:

| Endpoint | Control | Target |
|---|---:|---:|
| Diagnostic-cap violation risk | 1.261% | 0.057% |
| Provisional-cap violation risk | 0.814% | 0.008% |
| True 500 Hz velocity violation risk | 0.618% | 1.058% |
| Task success | 100% | 100% |
| Median native episode duration | 140 ms | 100 ms |

Impulse pressure won every active max soft-OR read at offline imp_max_p=0.5; velocity did not mask
it. All impulse violations were caused by J3. However, 97--99% of physical contacts were cut by
native task termination. The supported verdict is lower native observed impulse with higher
velocity risk: redistribution/trade-off, not clean enforcement.

Limits:

- one independent PPO training seed;
- three evaluation replicas are not training replicates;
- unequal native horizons;
- overwhelmingly right-censored contacts;
- simulation only;
- no hardware-safety or hard-cap claim.

## Proposed staged campaign

### Stage 1 — complete-contact shadow, no learning

Run the frozen control and target policies with live imp_max_p=0. Preserve the native trajectory
exactly through success. In a separate shadow, remove only success termination and continue until
five consecutive off-contact 2 ms samples and the existing 50 ms rolling Lambda is zero, with a
250 ms maximum continuation.

Use the fixed 64-world parity population and all three paired 4,096-world stochastic populations.
Require exact native-prefix parity, at least 95% completion among control physical events associated
with provisional-cap activation windows, and at least 50 completed, unambiguous,
provisional-cap-activating control events. If the gate fails, stop without choosing a dose.

### Stage 2 — offline dose selection

Replay imp_max_p in {0.05, 0.10, 0.20, 0.30, 0.50} on the same completed control events. For each
event compute:

P_event = 1 - product_t(1 - delta_impulse_t).

Choose the smallest candidate satisfying:

- among reads where replayed delta_impulse is positive, impulse strictly wins the max soft-OR at
  least 90% of the time; ties are not wins;
- median completed-event pressure at least 0.05;
- p95 completed-event pressure at most 0.50;
- fewer than 1% of completed events at pressure 0.80 or higher.

These are learning-dose criteria, not physical-safety criteria. If no candidate passes, stop.

### Stage 3 — one-seed bridge

Train one new 500-iteration seed-2 target at the provisional caps and selected dose. Compare against
the banked log-only control only after reconfirming that cap values cannot affect a p=0 PPO
trajectory.

Here, impulse risk is the fraction of initial native episodes whose maximum provisional-cap
utilization exceeds one; true-velocity risk is the fraction with any 500 Hz joint-velocity-limit
violation. Shadow outcomes calibrate dose and mechanism but do not replace these native endpoints.

Pass only if:

- upper paired one-sided 97.5% bound for provisional-risk difference is below -0.50 percentage
  points;
- impulse p95/p99 decrease by at least 10% relative to control without another-joint
  redistribution;
- upper paired one-sided 97.5% bound for velocity-risk difference is at most +0.10 percentage
  points;
- success/productive-strike differences are no worse than -1 percentage point;
- lower one-sided 97.5% bound for delivered-impulse ratio is at least 0.90.

Failure stops the campaign; no automatic next dose is tried.

### Stage 4 — independent-seed confirmation

If the bridge passes, train matched p=0 and selected-dose pairs at seeds 2--9 under provisional
caps. Primary checkpoint is iteration 499; iterations 400 and 450 are fixed stability checks. The
training seed is the primary unit. Require both co-primary bounds, all utility gates, at least six
of eight seed pairs with the desired impulse direction and nonpositive velocity-risk difference,
and stable direction in at least two of three checkpoints.

Estimated successful-route compute is approximately 8 A100-GPU-hours. Stages are separately
authorized and may stop early.

## Review task

Independently stress-test the proposed campaign. Do not assume the plan is correct and do not invent
new hardware specifications.

Answer these questions:

1. What is the strongest remaining confound after the release/window-flush shadow?
2. Are the complete-event gate and dose-admissibility rule logically tied to the research question?
3. Are the bridge thresholds too weak, too strong, or unjustified? Give a concrete alternative.
4. Does eight paired PPO seeds support the intended learned-policy claim without pseudoreplication?
5. Can shorter target episodes or post-success counterfactual actions still create a false impulse
   reduction?
6. Is any proposed job redundant or any decisive test missing?
7. Under what exact observation should the campaign stop rather than increase imp_max_p?
8. Does the design ever blur provisional project caps, diagnostic thresholds, and hardware limits?

## Required response format

    PROVIDER:
    MODEL_ID:
    OVERALL: APPROVE | REVISE | REJECT

    FATAL_FLAWS:
    1.
    2.

    THRESHOLD_CRITIQUE:
    - complete-event gate:
    - dose rule:
    - impulse efficacy:
    - velocity noninferiority:
    - utility:

    PSEUDOREPLICATION_OR_CAUSALITY_RISKS:
    1.
    2.

    REDUNDANT_JOBS:
    -

    MISSING_DECISIVE_TESTS:
    -

    HARD_STOP_RULES:
    1.
    2.

    MINIMAL_REVISION:

The response should be under 1,200 words. It must distinguish evidence, inference, and
recommendation.

## Review ledger

| Requested label | Provider | Exact model ID | Status | Prompt SHA-256 | Response SHA-256 |
|---|---|---|---|---|---|
| Kimi 3 | — | — | NOT_SENT | — | — |
| GLM 5.2 | — | — | NOT_SENT | — | — |
| DeepSeek V4 Pro | — | — | NOT_SENT | — | — |
| Qwen | — | — | NOT_SENT | — | — |
