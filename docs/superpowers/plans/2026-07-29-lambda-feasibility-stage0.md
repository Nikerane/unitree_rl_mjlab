# Lambda feasibility Stage 0 implementation plan

> Execute only Stage 0 from
> `docs/superpowers/specs/2026-07-29-lambda-feasibility-design.md`.
> Do not add the CEM search until replay evidence and a separate review justify it.

## Goal

Build the smallest standalone CPU evaluator needed to replay the existing
reference and saved whip tapes with strict full-trajectory legality, the
production 50 ms Lambda signal, exact contact-row 26/28 ms, onset-anchored and
sliding 50 ms signals, and an offline impact-versus-press classification.

No core environment, reward, constraint, asset, or existing whip-search file
is changed.

## Files

- Create `evaluation/whip/lambda_feasibility.py`.
- Create `tests/test_lambda_feasibility.py`.
- Generate the authoritative artifact once under
  `/private/tmp/lambda_feasibility_stage0_authoritative_<codehash>/`, outside
  both Git repositories. After verification, bank an immutable copy and
  sidecars under a named `docs/results/assets/` directory.
- Bank the eventual result in a separate dated `docs/results/` document.

## Task 1 — Pure trace classifier (TDD)

Write failing synthetic tests for:

1. two-substep debounced release, backdated to the first off sample;
2. no release and late release classified as press/jam;
3. first-lobe endpoint from downward axial velocity crossing zero;
4. 13-, 14-, and 25-substep onset-anchored per-joint integrals;
5. earliest-time / lowest-joint deterministic tie breaking;
6. full-trajectory qvel legality, including pre-contact violations;
7. definite bind / boundary / non-bind classification;
8. shipped-only and exact-only interpretation matrix;
9. no-recontact-through-50 ms rule;
10. all denominators use the imported unchanged `IMP_J_LIMIT`.
11. the first-lobe share is bounded to the onset-anchored 50 ms denominator;
12. Stage-0 continuation labels distinguish existence, near lead, no lead,
    and ambiguity without claiming physical impossibility.

Implement only the pure NumPy/Torch-free functions needed for those tests.

## Task 2 — Saved-tape inventory and provenance (TDD)

Write failing tests that:

- load exactly `whip_delta015.json`, `whip_delta015_cma.json`,
  `whip_delta030.json`, and `whip_delta045.json`, failing if any is missing;
- validate action shape `[T,3]`, finite values, and `[-1,1]` bounds;
- deduplicate byte-identical `best_actions` / `best_actions_hw`;
- preserve every source alias, delta, and action digest;
- fail closed on missing/invalid fields;
- never use legacy `hardware_legal` as a new legality decision.

Implement the loader and deterministic SHA-256 identities.

## Task 3 — Shadow environment and recorder

Implement a one-environment CPU builder from the ordinary
`z1_hammer_env_cfg(cat_impulse=True, event_correct=True,
quality_instrumentation=True)` configuration, then remove only
`terminations["nail_driven"]`. Do not use `no_terminate=True`, because it also
retunes the completion reward:

- `cfg.auto_reset=False`;
- exact physics dt/decimation assertions;
- imported cap, gains, effort, and `imp_max_p=0` assertions;
- only `delta_pos_scale` may differ;
- no joint-state writes and no `set_gains`;
- exact hammer-face contact sensor and production first-strike tracker;
- production `SubstepImpulseAccumulator`;
- direct per-substep `contact_row_qfrc(env)` for the exact diagnostic.

At every physics callback, record:

- pre/post arm qvel and qpos;
- head position/velocity, nail depth, face contact;
- maximum active disallowed non-face EFC-row force, with frozen threshold
  `1e-6`;
- quality snapshot fields;
- production accumulator rolling/contribution/episode peak;
- exact contact-row contribution;
- object-side force/impulse diagnostics.
- the complete first-contact quality snapshot and accepted-onset trace;
- compiled baseline/`solref*2` values and the 6-D linear/angular head twist.

Use a uniform record-all schema for successful, no-contact, error, timeout,
and native-crash outcomes. A completed trace that later fails classification
must retain its trace; a partially completed rollout must retain the samples
recorded before failure.
The immutable result must retain lossless 500 Hz arrays for exact contact-row
contribution, shipped rolling/contribution, object axial force and
axial/total impulse contribution, raw face contact, non-face EFC force, and
tracker onset. Derived totals alone are insufficient.

Reset-state writes are limited to the reviewed D2 restore immediately after
`env.reset()` and before action 1. There are no state writes after execution
starts. Keep the ordinary `substep_impulse_rows` diagnostic enabled; the
shadow configuration changes only success termination/auto-reset and the
preregistered delta/solref characterization factors.

Every 30-command saved tape is followed by exactly ten zero relative commands:
40 control steps / 400 physics substeps total. The recorder and qvel/qpos
legality checks cover that whole horizon. It must not execute or classify
samples from an auto-reset episode.

## Task 4 — Runtime invariants

Fail closed unless:

- all signals are finite;
- production rolling maximum equals `_episode_peak_perjoint`;
- exact contact-row is zero before hammer-nail contact;
- contact-row and object-side axial delivered-impulse signals are both live
  on at least one shared face-contact substep (liveness/alignment only; never
  assert numerical equality between the different physical quantities);
- identical action/reset replay gives identical CPU metrics within the frozen
  tolerance;
- ordinary-task and shadow-task state/action histories match through the
  original 30 mm success instant.

The exact contact-row zero-before-contact invariant is keyed to the raw
hammer-face/nail contact mask, not the tracker onset. Report both the
onset-anchored row-50 integral and the full-replay sliding row-50 peak.

Run the focused tests after each invariant.

## Task 5 — CPU smoke

Run one reference tape and one saved delta-0.15 tape:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  evaluation/whip/lambda_feasibility.py \
  --smoke --out /tmp/lambda_feasibility_smoke.json
```

The smoke prints a compact table with:

- source/delta;
- qvel maximum and legality;
- quality/productiveness/release;
- shipped 50 ms ratio;
- exact 26/28 ms ratios;
- classification.

Inspect the JSON and run the focused tests.

## Task 6 — Full Stage 0 replay

Replay:

- scripted reference;
- every distinct saved `best_actions`;
- every distinct saved `best_actions_hw`;
- original deterministic reset;
- 16 frozen training-distribution reset states.

Run every replay in a fresh spawn subprocess. Record crashes/timeouts rather
than silently dropping them. Re-run every fixed-reset baseline identically
for determinism. Re-run **every eligible tape** with `solref*2`, retain
per-joint changes, and flag a maximizing-joint switch.

The 16 reset replays are shared-tape robustness diagnostics only: never count
them as independent candidates or use their failure to veto a fixed-reset
existence result. Keep the scripted reference as a calibration control and
exclude it from tape binding rates and scale aggregates.
Always run fixed-reset reference `solref*2` as a calibration even though the
known reference is ineligible; do not invent numerical Phase-0 tolerance
bands after observing the smoke.

Write one immutable JSON result containing:

- code/asset/tape/reset digests;
- full frozen config;
- per-replay metrics and classification;
- aggregate counts by delta/source;
- all failures.

Before outcomes exist, freeze a separate immutable input envelope containing
the clean code/asset/tape/config/reset identities. Refuse dirty provenance and
refuse to overwrite either input or output artifacts. Bank SHA-256 sidecars
and make both artifacts and sidecars read-only.
Reject an authoritative `--out` inside either Git repository. Require exact
clean code provenance and zero changes under the actually loaded
`hammer_z1_env/assets/**` scope; record unrelated asset-repository status as
disclosed metadata without deleting user files.

Apply the frozen Stage-0 continuation rule:

- eligible fixed-reset dual bind -> simulator existence proven;
- else eligible fixed-reset `rho_row28 >= 0.70` -> one-scale bounded pilot;
- else all stable legal fixed-reset rows below 0.70 -> no Stage-1 lead, not
  physical impossibility;
- otherwise -> ambiguous and review.

## Task 7 — Verification and review

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_lambda_feasibility.py \
  tests/test_contact_row_impulse.py \
  tests/test_impulse_constraint.py \
  tests/test_first_strike_event.py -q
```

Then run the standard impulse/reward preflight:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/validate_rewards.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_contact_sensor.py
```

Have an independent reviewer check:

- source for forbidden state injection/gain changes;
- pre-contact qvel violations are caught;
- success auto-reset cannot contaminate the trace;
- contact-row and shipped quantities are not swapped;
- no outcome-adaptive thresholds entered the classifier.

The reviewer must also verify that `solref*2` is described only as a necessary
model-sensitivity check, not as proof of hardware/ballistic fidelity, and that
the reference and 16-reset diagnostics cannot inflate the fixed-reset
existence denominator.

Only after this review decide whether Stage 1 search is justified.

Authoritative command template (fill the clean code hash exactly):

```bash
OUT=/private/tmp/lambda_feasibility_stage0_authoritative_<codehash>
mkdir -p "$OUT"
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  evaluation/whip/lambda_feasibility.py \
  --timeout 180 \
  --out "$OUT/lambda_feasibility_stage0.json"
```

Do not run that command until the evaluator is committed, the code and loaded
asset scope are clean, and the exact-hash independent review is approved.
