# Minimal overnight binding-first experiment

**Objective:** Within one six-hour window, turn the already-trained Presentation3 policies into defensible evidence about productive first-strike impulse under mandatory velocity CaT, and train new policies only when the sampled evidence makes the treatment identifiable.

## Current evidence and labels

- `M` is the existing progress-guided `P+D4` arm: delivered-impulse reward weight 4, no velocity CaT, impulse CaT log-only.
- `V` is the existing progress-guided `P+V` arm: delivered-impulse reward weight 2 plus 500 Hz velocity CaT, impulse CaT log-only.
- `V+M` is the existing progress-guided `P+V+D4` arm: delivered-impulse reward weight 4 plus 500 Hz velocity CaT, impulse CaT log-only.
- `V+I` and `V+I+M` do not yet exist. They are scientifically identifiable only if the unchanged per-joint impulse caps bind in sampled evaluation.
- The slide label for `M` is **higher impulse-maximization dose**, not impulse maximization on/off; the baseline progress arm already uses delivered-impulse weight 2.

## Global constraints

- Straight ordered waypoint-progress guidance remains present in every new arm.
- Velocity CaT uses the existing 500 Hz post-integration substep peak and limit `3.1415 rad/s`.
- Do not lower or retune impulse caps to manufacture binding. Keep `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64] Nms`.
- `imp_max_p=0.0` unless and until the binding gate passes and fresh C2 validation authorizes the existing active setting.
- Productive impulse is the first-event nail-axis delivered impulse, not cumulative press/recontact impulse.
- Preserve fixed impedance: no `set_gains`; document the existing PD gains and Cartesian DiffIK action scale.
- Reuse `scripts/eval_impulse.py`, existing renderer, diagnostics, launchers, and evidence schemas. Do not build a new framework, manifest builder, optimizer, curriculum, or presentation editor.
- Do not edit `Desktop/_Munich_hammering/Thesis_updates_ppts/Julty_updates.pptx`.
- Never replace completed weak seeds. Retry only a documented infrastructure failure with the identical identity.
- Every result records code, asset, checkpoint, task/config, reset/RNG, and analysis identities. Reject dirty provenance.

## Decision gate

Run sampled evaluation before any active impulse-CaT training. Apply the gate separately to `V` and `V+M`; `M` cannot authorize an active constraint under the mandatory-velocity regime.

Let `B` contain every complete, count-valid, provenance-valid sampled episode from one parent arm before filtering on outcome. Missing, digest-invalid, incomplete, or auto-reset-contaminated records invalidate the campaign rather than becoming exclusions.

Use a staged gate so fine-grained instrumentation is added only if it can affect the decision:

1. On `B`, require at least 95% successful, productive first events and require every post-integration 500 Hz arm-joint velocity sample to satisfy `abs(qvel) <= 3.1415 rad/s`.
2. For every one of the arm's six checkpoints, require at least 5% of its 512 episodes to have a strict unchanged-cap crossing, `max_j Lambda_j/cap_j > 1`. Report p99 descriptively; do not treat it as independent evidence.
3. Only if clauses 1–2 pass, run an attribution follow-up. A qualifying crossing must observe first release within 50 ms of first onset, and at least 70% of the exact binding-peak 50 ms accumulator exposure must come from that first contact bout. Censored, sustained-press, or later-recontact crossings count as non-qualifying.
4. In that follow-up, require a frozen cluster-aware one-sided 95% lower confidence bound above 5% for every checkpoint, keeping each environment stream's two episodes in one resampling block.

Authorize `V+I` only if `V` passes all clauses, and `V+I+M` only if `V+M` passes all clauses. The paired active-I branch requires both parents to pass, followed by fresh C2. If either parent fails an earlier clause, record `ACTIVE_I_NOT_IDENTIFIABLE`, do not add attribution machinery or train either active-I arm, and run the smaller reward-dose experiment `V+D0` so the mandatory-velocity comparison becomes delivered-impulse dose 0/2/4.

## Task 1: Extend the existing sampled evaluator, test first

Support the three already-trained Presentation3 task identities (`M`, `V`, `V+M`) without weakening any existing campaign contract.

- Add focused tests first and observe the intended failures.
- Extend only the existing campaign/task maps and fail-closed task-specific contract checks.
- Require the exact guidance reward, delivered-impulse weight, velocity-CaT setting, `imp_max_p`, unchanged caps, observation contract, and clean provenance for each arm.
- Require the exact Presentation3 campaign identity and frozen reset/observation/action RNG tuple; unscoped or arbitrary-RNG Presentation3 evaluation must fail.
- Keep the strict sampled design at 256 environments and exactly two completed episodes per environment/checkpoint.
- Run focused tests, mutation probes for wrong weight/wrong CaT/wrong caps, and an independent task review.

## Task 2: Sample the existing 18 policies on Vega

- Deploy a clean detached checkout of the reviewed evaluator revision; never copy tracked source.
- Evaluate all six checkpoints in each of `M`, `V`, and `V+M` using frozen RNG streams.
- Preserve raw schema-v3 episodes and fail closed on any identity, count, completion, or digest error.
- Compute the gate inputs from raw episodes before inspecting arm aggregates.
- Produce a signed binding-decision record containing every clause, numerator/denominator, interval method, and verdict.

## Task 3: Execute exactly one conditional training branch

### Nonbinding branch (expected)

- Register only `V+D0`: progress guidance plus velocity CaT, delivered-impulse reward disabled, impulse CaT log-only.
- TDD the exact treatment isolation and live environment construction.
- Qualify on CPU, run the existing reward/contact checks and one CUDA smoke.
- Train seeds 2–7, 200 iterations, 4096 environments, with no curriculum.

### Binding branch (only if Task 2 passes every clause)

- Revalidate C2 for the current event/window semantics and unchanged caps.
- Register only `V+I` and `V+I+M` using the already-qualified active impulse-CaT setting.
- TDD treatment isolation, run CPU qualification and CUDA smokes, then train seeds 2–7 for both arms at 200 iterations and 4096 environments.

## Task 4: Fixed-reset evidence and presentation assets

- Render every newly trained policy at the identical digest-bound reset; no outcome-based selection.
- Plot the desired guideline only for guided arms and put a plain-language treatment title above every panel.
- Reuse the existing trajectory and impulse diagnostic tools; do not add a new simulator path.
- Report first-event delivered impulse, velocity legality, six gates by contact onset, pre-contact finite-segment max/RMS error, path ratio, success/depth, contact speed, delivered impulse, and `Lambda/cap`.
- Produce matched trajectory grids, arm-level tables with all seeds visible, and representative videos selected by a predeclared rule rather than best outcome.
- State CPU/CUDA measurement provenance and that the impulse caps are project-defined engineering caps, not manufacturer-certified safety limits.

## Task 5: Independent verification and freeze

- Rebuild all reported cells from raw evidence using an independent path.
- Verify checkpoint/config/reset/revision joins and fixed baseline projection.
- Review prose separately for claims stronger than the sample supports.
- Freeze only after zero unresolved Critical/Important findings; mirror the hash inventory read-only on Vega.
