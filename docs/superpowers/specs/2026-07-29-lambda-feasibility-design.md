# Fixed-impedance Lambda feasibility sweep — frozen design

> **Impulse-threshold provenance correction (2026-08-14):** This frozen design's `[1.64, 3.28, ...]`, `kappa=2`, `27.3 ms`, and “manufacturer” interpretations are historical inputs, not a validated Z1 reaction-impulse or damage limit. Preserve the body and outcomes; see `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md` for the current banked/provisional/diagnostic distinction.

**Date:** 2026-07-29  
**Status:** design frozen before new trajectory outcomes  
**Scope:** evaluation-only characterization in the current Z1 hammer simulator  
**Enforcement:** off (`imp_max_p=0.0`)  
**Plant:** fixed impedance; no gain commands or `set_gains`

## Amendment A1 — contact-signal liveness (post-attempt-1, pre-attempt-2)

Authoritative attempt 1 was frozen read-only at SHA-256
`6755cdefb74aebfb704f2271af4b020dcfc1d83bba045c5b1a997922d7b21414`
and then failed closed. Five lateral/upward rim contacts had live, aligned
joint contact-row and object total-force signals but zero downward axial
projection. The original Stage-0 sentinel incorrectly called the reverse
direction `lambda_dead`, conflicting with the established campaign invariant
in `scripts/eval_impulse.py`.

Before any attempt-2 outcome is opened, freeze this narrow correction:

- instrument liveness means overlapping exact contact-row and object
  **total-force magnitude** on the accepted first face-contact event;
- object axial-positive overlap remains required for an eligible candidate,
  and zero axial delivery is recorded as `non_axial_contact`;
- true `lambda_dead` retains the established direction: positive object-side
  axial delivery with zero robot-side Lambda;
- missing or non-overlapping exact/total signals set the separate fatal
  `contact_signal_alignment_failure` sentinel;
- the scripted-reference calibration still requires axial-positive alignment.

No action, reset, cap, plant, reward, contact model, timeout, or numeric
eligibility threshold changes. The axial-positive criterion moves from the
erroneous global failure sentinel to per-candidate eligibility. Attempt 1
remains immutable failed evidence. Attempt 2 must use a new clean code revision
and output directory, and its separately frozen raw-physics equivalence report
must show the identical 154-row identity set and zero raw mismatches before
aggregate interpretation.

## 1. Question

Can a productive fixed-impedance strike, using the legal DiffIK action path,
reach an unchanged manufacturer-derived per-joint impulse cap while remaining
inside the Unitree Z1 joint-speed limits and exhibiting a completed impact
rather than a sustained press?

This experiment separates two questions that the current implementation
conflates:

1. **Would the shipped 50 ms Lambda signal bind?**
2. **Would an isolated hammer-nail impact bind on the cap's approximately
   27.3 ms time basis?**

A positive answer to (1) alone is not evidence of impact safety. It may be a
window-mismatch, friction-residual, or press result.

## 2. Immutable guardrails

- Keep `IMP_J_LIMIT = [1.640, 3.280, 1.640, 1.640, 1.640, 1.640] N m s`.
- Keep `imp_max_p=0.0`; this is log-only characterization.
- Keep the shipped gains, effort limits, armature, decimation, and timestep.
- The only characterized action-interface parameter is
  `delta_pos_scale in {0.15, 0.30, 0.45}`; analyze scales separately.
- Never inject joint state during a rollout and never call `set_gains`. The
  reviewed D2 reset contract may restore a frozen realized reset state after
  `env.reset()` and before action 1; no state write is permitted after rollout
  execution begins.
- Do not modify the nail/target in the primary experiment.
- Use the CPU backend first. A GPU run requires CPU evidence and a separate
  backend-parity check.

The caps are not empirical thresholds on the dirty shipped signal. They are
the frozen manufacturer formula `tau_rated * 2 * 27.3 ms`. Therefore the
contact-row diagnostic can be compared with the same unchanged caps. No cap
is re-derived or lowered.

## 3. Why the prior whip result must be re-audited

`evaluation/whip/whip_search.py::fitness()` used `Q[first_contact]` as its
hardware gate. It did not use the maximum joint speed over the wind-up and
strike. The previously reported `best_actions_hw` tapes are therefore
contact-step-legal candidates, not strict full-trajectory legality
certificates.

The first stage replays every stored `best_actions` and `best_actions_hw`
candidate before any new optimizer is trusted.

## 4. Quantities recorded on every replay

The evaluator records all candidates, including ineligible presses and
illegal trajectories:

- maximum absolute arm qvel at every 2 ms pre- and post-integration sample;
- joint-position legality and actuator-saturation occupancy;
- face contact, all other robot contacts, head pose/velocity, nail depth;
- 7-D head pose and 6-D linear/angular head twist;
- the complete first-contact quality snapshot: contact point, radial error,
  quality, validity, overflow, first-contact time, and normal axiality;
- raw 500 Hz face-contact and non-face EFC-force traces;
- object-side axial and total delivered impulse;
- raw 500 Hz object axial force plus axial/total impulse contributions;
- shipped baseline-subtracted 25-substep/50 ms per-joint Lambda;
- raw 500 Hz shipped rolling and per-substep contribution arrays;
- exact hammer-nail contact-row per-joint impulse:
  - onset-anchored 13 substeps / 26 ms;
  - onset-anchored 14 substeps / 28 ms;
  - onset-anchored 25 substeps / 50 ms;
  - maximum sliding 25-substep / 50 ms window over the full replay;
  - onset to end of the first deceleration lobe;
- the raw 500 Hz exact per-joint contact-row contribution array and tracker
  accepted-onset trace, so every derived classification can be recomputed;
- contact duration, release, recontact count, force peak/mean, lobe fraction;
- numerical failures, process crashes, and timeouts.

Every saved 30-command tape is followed by exactly ten zero relative commands.
At decimation 10 this fixes the replay horizon at 40 control steps / 400
physics substeps. Qvel and qpos legality are evaluated over the entire horizon,
including the post-tape observation tail.

The 26 and 28 ms values bracket the unrepresentable 27.3 ms cap basis.
No interpolated value decides the result.

## 5. Frozen event definitions

- **Accepted onset:** the production first-strike tracker's accepted
  hammer-face/nail onset after its two-off-substep arming rule.
- **First impact lobe:** onset through the first in-contact sample where
  downward axial head velocity becomes non-positive.
- **Release:** first of five consecutive off-contact 2 ms samples after onset,
  backdated to the first off sample. Two- and ten-sample debounces are reported
  as sensitivity checks.
- **Completed short impact:** first-lobe end exists, release begins by the
  fourteenth post-onset substep (28 ms), and no recontact occurs through
  50 ms after onset.
- **Productive:** the first event reaches the task's 30 mm success threshold
  within 50 ms of accepted onset.
- **Quality-valid:** production quality snapshot is valid, finite, and has no
  overflow. Quality magnitude is reported continuously; `q>=0.5` is a
  sensitivity analysis, not a primary eligibility gate.
- **Face-only:** no simultaneous non-face robot/environment constraint contact
  with maximum active disallowed EFC-row force above `1e-6` during onset
  through release. The detector ignores sensor-only contacts and the allowed
  hammer-face/nail pair; the tolerance is unit-tested and printed in
  provenance.
- **Strict qvel legality:** every recorded arm-joint sample is finite and
  `abs(qvel)<=3.1415 rad/s`. A `0.95*rail` margin is reported only as
  sensitivity; it does not replace the manufacturer limit.

The shadow evaluator is built from the ordinary task configuration, after
which only `terminations["nail_driven"]` is removed and `auto_reset` is
disabled so release can be observed. It must not use the `no_terminate=True`
variant because that also retunes completion from 100 to 1. The evaluator
must assert state/action parity with the ordinary task through the original
30 mm termination instant.

## 6. Binding classifications

Let `rho = max_j Lambda_j / IMP_J_LIMIT_j`, with the maximizing joint reported.
The maximum over joints is not a multiple hypothesis test: the deployed
constraint is violated if any named arm joint exceeds its own cap.

### 6.1 Shipped-mechanism binding

`rho_ship50 >= 1` using the production 25-substep accumulator.

This proves only that the current log-only mechanism would activate. It does
not by itself prove an impulsive collision.

### 6.2 Exact impact binding

All eligibility gates pass, and:

- `rho_row26 >= 1`: **definite exact impact bind**;
- `rho_row26 < 1 <= rho_row28`: **time-discretization boundary case**;
- `rho_row28 < 1`: **exact impact non-bind for that event**.

Additionally report:

- first-lobe share of the **onset-anchored** exact contact-row 50 ms impulse.
  Its numerator is clipped to that same onset-to-50-ms interval, so the share
  is bounded by one; the untruncated full-lobe impulse is reported separately;
- require that share to be at least 70% for the primary meaningful-impact
  label, with 60% and 80% sensitivity;
- change in 26/28 ms and full-lobe impulse under `solref*2`.

The 70% lobe share and 20% solref-change bands are diagnostic labels, not
standalone physical laws. A result outside either band is labeled
press/model-sensitive and cannot support a clean impact-safety claim.
Conversely, a change smaller than 20% is only a necessary simulator
model-robustness check. It is not evidence of ballistic fidelity or hardware
validity. The comparison is retained for every joint and for the baseline
binding joint, and a switch in the maximizing joint is reported rather than
hidden by a max-over-joints scalar.

### 6.3 Interpretation matrix

| Shipped 50 ms | Exact impact | Interpretation |
|---|---|---|
| no | no | no binding observed |
| yes | no | shipped press/window/contamination binding |
| no | yes | shipped baseline subtraction misses a physical contact-row bind |
| yes | yes | meaningful candidate operating regime for later enforcement study |

## 7. Staged execution

### Stage 0 — instrument qualification and saved-tape replay

- Replay the scripted reference and every distinct saved speed-search
  `best_actions` / `best_actions_hw` tape.
- The saved-tape allowlist is exactly `whip_delta015.json`,
  `whip_delta015_cma.json`, `whip_delta030.json`, and
  `whip_delta045.json`; every file must exist. Deduplicate only identical
  `(delta_pos_scale, action_digest)` pairs while retaining every source alias.
- Use fresh CPU processes and `auto_reset=False`.
- First use the original deterministic reset to reproduce the old candidate.
- Then use 16 frozen training-distribution reset states (`+/-0.05 rad`) as a
  shared-tape robustness characterization; these are not independent rows,
  a confirmatory held-out bank, or a veto on a fixed-reset existence result.
- Cross-check production rolling Lambda against its episode-peak latch.
- Require the exact contact-row and object-side **total-force magnitude**
  signals to both be live on at least one shared face-contact substep. This is
  an alignment/liveness check only: the joint-space and object-side quantities
  have different units and must never be tested for numerical equality.
  Bank axial-positive overlap separately; a zero value is a physical
  non-axial contact and cannot be eligible.
- Re-run **every Stage-0 eligible tape** under `solref*2`, not only the
  highest-ratio tape.
- Re-run the fixed-reset scripted reference under `solref*2` regardless of
  eligibility. It is the preregistered press/model-sensitivity calibration,
  remains excluded from candidate rates, and uses no post-smoke numerical
  tolerance band. It must be status-ok, accepted/live/productive, qvel/qpos
  legal, ordinary/shadow-parity clean, and remain an ineligible broad-contact
  calibration; an unexpected eligible short-impact classification is
  ambiguous and requires manual review.
- Re-run every fixed-reset baseline with the identical action/reset request
  and require an identical deterministic trace digest.

Stage 0 is descriptive. Legacy tapes were selected for speed on a fixed reset,
so they are seeds/diagnostics, not independent statistical samples.
The scripted reference is a separate calibration control and is excluded from
all legacy-tape binding rates, scale aggregates, and continuation decisions.
Stage 0 evaluates the physical/candidate lead supplied by the reference and
legacy open-loop tapes. It does not test whether the new FQ-trained policies
improved Lambda. If Stage 0 supplies no lead, any later direct 56-policy test
must be a fresh clean closed-loop checkpoint evaluation on a newly frozen
reset bank. The archived sampled action/reset replay is not admissible because
its binding is documented off-by-one/non-replayable.

### Stage-0 continuation labels

These labels are frozen before opening the authoritative Stage-0 output:

- one strict, eligible fixed-reset `dual_bind` proves **simulator existence**;
  Stage 1 is not required to establish existence;
- otherwise, an eligible fixed-reset legacy tape with `rho_row28 >= 0.70`
  is a **near lead** justifying one bounded Stage-1 scale;
- if all stable, qvel/qpos-legal fixed-reset legacy tapes have
  `rho_row28 < 0.70`, Stage 0 supplies **no Stage-1 lead**;
- all other patterns are **ambiguous** and require review.

The last two labels are budget-allocation decisions, not claims that binding
is physically impossible.

### Stage 1 — bounded pilot search

Run only after Stage 0 instrument and replay invariants pass.

- scales: 0.15, 0.30, 0.45, never pooled;
- four independent non-inferential search blocks per scale;
- population 48, 12 generations, two common search resets per block;
- constrained lexicographic ordering:
  1. finite/stable rollout;
  2. strict qvel and qpos legality;
  3. face-only, quality-valid, productive strike;
  4. completed short impact;
  5. maximize exact contact-row 26 ms `rho`;
- subprocess isolation at high delta; crash/timeout rate is an outcome;
- freeze one candidate per block before any validation-bank read.

The pilot cannot support a negative conclusion. It decides whether the
optimizer has a nonzero feasible population and whether a final search is
scientifically and computationally justified. The Stage-0 `0.70` near-lead
threshold is only a preregistered compute-triage rule, not a physical boundary
or a confirmatory scientific endpoint.

### Stage 2 — confirmatory search and untouched resets

If the pilot finds eligible candidates or a still-improving feasible frontier:

- freeze the selected scale using search-only data;
- 12 independent final search blocks at that scale;
- population 48, 24 generations, two frozen search resets per block;
- one selected candidate per block;
- exactly one evaluation on 128 untouched reset states per block;
- every block receives a disjoint held-out bank;
- no candidate or scale changes after held-out outcomes are opened.

The unconditional primary endpoint is

`p_EB = dual shipped-plus-exact, eligible, lobe-dominated binding held-out
episodes / all held-out episodes`.

Episodes remain nested in their search block. Bootstrap whole blocks, never
adaptive proposals or individual episodes.

Decision labels:

- **robustly bindable:** one-sided 95% block-bootstrap lower bound for `p_EB`
  is at least 5%;
- **existence but rare/fragile:** at least one independently replayed held-out
  exact bind, but the lower bound is below 5%;
- **not found under budget:** no eligible exact bind and every block's held-out
  99th percentile `rho_row28` is below 1;
- **ambiguous:** all other outcomes.

Any cross-scale comparison is secondary and, if tested, uses at least five
independent blocks per scale with Mann-Whitney tests and multiplicity control.

### Frozen reset-stream seeds

Reset states are materialized and hashed before the corresponding stage:

- Stage-0 robustness bank: RNG seed `2026072901`, first 16 states;
- pilot block `b`: RNG seed `2026073000 + b`;
- final search block `b`: RNG seed `2026074000 + b`;
- final selection block `b`: RNG seed `2026075000 + b`;
- final held-out block `b`: RNG seed `2026076000 + b`.

The scale arms within a pilot block use common reset states. Final held-out
banks are disjoint across blocks. Each generated bank is rejected if any
realized state digest duplicates another bank or a legacy tape's generating
fixed reset.

## 8. Fail-closed gates

Before accepting any result:

- exact caps, gains, effort limits, dt, decimation, and `imp_max_p=0` asserted;
- no dirty or `-dirty` provenance;
- exact code HEAD plus a clean loaded-asset scope
  `hammer_z1_env/assets/**`. Unrelated asset-repository metadata is recorded
  separately and disclosed, never deleted or hidden; any change inside the
  loaded asset scope fails authority;
- `impossible_success_n==0`, where an impossible success is 30 mm depth
  without an accepted face-contact event in the same replay;
- `lambda_dead_n==0`, where Lambda-dead retains the campaign definition:
  positive object-side axial delivery with zero robot-side Lambda;
- `contact_signal_alignment_failure_n==0`, where alignment failure means the
  exact contact-row and object total-force signals are absent or never live on
  the same accepted first-event face-contact sample;
- zero non-finite qvel/Lambda and zero contact-row overflow;
- deterministic replay of an identical action/reset pair;
- no post-success auto-reset contamination;
- every winner replays in a fresh process without crash;
- held-out reset digest and action-tape digests banked before outcomes.
- zero replay error/crash/timeout rows, zero failed/missing fixed-baseline
  determinism qualifications, and zero failed/missing mandatory `solref*2`
  comparisons for an accepted Stage-0 artifact. Failures remain frozen
  evidence but force an ambiguous/no-go interpretation.

For Stage 0, the full reset/action/config/provenance input envelope is written
and SHA-256 sidecar-frozen before any replay outcome is generated. The final
record-all output (including errors, crashes, and timeouts) receives its own
SHA-256 sidecar and both files are made read-only. Existing artifacts are
never overwritten.
Authoritative generation must use an output directory outside both code and
asset Git repositories, so the input artifact cannot self-dirty provenance.
After successful verification, copy the immutable artifact and sidecars into
a named `docs/results/assets/` bank as a distinct step and record both source
and banked hashes; never rerun merely to create an in-repo copy.

## 9. Claim boundary

A positive qualifying open-loop candidate establishes existence in this
simulator benchmark. It does not prove a trained closed-loop policy will
remain legal.

A negative search says only:

> No eligible fixed-impedance regime was found over the preregistered action
> scales, reset banks, optimizer, and compute budget.

It is not a proof that no closed-loop fixed-impedance policy can exist, and it
is not a real-hardware certificate without target/contact validation.

## 10. Review record

Three independent internal reviews, Gemini Pro, and Claude Opus all supported
the staged replay-first structure and identified the old contact-only qvel
gate and press/window mismatch as load-bearing threats.

Claude proposed re-deriving caps on the contact-row instrument. That proposal
is rejected here because the caps are the immutable manufacturer formula, not
an empirical dirty-signal calibration, and changing them violates the
experiment guardrail. Its valid 26-vs-28 ms objection is addressed by
bracketing the cap basis and requiring the conservative 26 ms crossing for a
definite bind.
