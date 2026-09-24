# Task 4 implementer report

Status: DONE. Base: `bdde187`. Scoped commit recorded in the final task response.

## Changes and integration APIs

- `src/tasks/hammer/mdp/strike_evaluation.py`: focused tensor `StrikeEvaluationCollector`, final per-substep `StrikeEvaluationMetric`, config installer, reversible `MeasurementProtocol`, pre-reset JSON snapshot, and all-attempt count summary.
- `scripts/evaluate_impact_speed.py`: `--measurement-protocol native|common-window` (native default), evaluation metric config, protocol context after runner/checkpoint validation, immutable `PreResetSnapshot.latest_evaluation`, episode record attachment, and protocol/cadence/count metadata.
- `tests/test_strike_evaluation.py`: 16 tensor, native manager, and native environment regressions.
- `tests/test_impact_speed_evaluation.py`: CLI/serialized evidence assertions and saved-checkpoint evaluation under both protocols.
- This report. No original source worktree/assets, training reward, controller, CaT cap/dose, event tracker, or reset distributions changed. No training, cluster query, or hardware work.

`configure_strike_evaluation(cfg)` appends the metric after existing `first_strike` and `substep_impulse`; only the evaluator calls it. The adapter reuses `tracker.started`, the tracker-configured intended contact sensors/axis, and the pure current-height reference diagnostic. It reads `env._hammer_substep_impulse.impulse` every physics sample, after that accumulator update. This property is precisely `max(current rolling Lambda, current control-step peak)`, not the CaT 50 Hz telemetry cache. Neither collector nor protocol resets the authoritative impulse ring.

`MeasurementProtocol` wraps only `nail_driven`, latches its native success check/sample, and under common-window mode withholds only that success term until completion. Failure/timeouts remain native, including simultaneous success/failure. A rebound cannot clear a previously observed success. The original function is restored before final contract validation. A success without any accepted onset remains immediately terminal and produces censored/no-contact evidence. Termination remains on policy boundaries: it may occur after the exact 200-sample window has frozen.

`evaluation_snapshot(env)` executes after metrics and before native autoreset. It creates independent Python records, retaining native delivered/transverse impulses and numeric stop reason separately from common-window quantities. `EpisodeAccumulator.update(..., evaluation=...)` retains the latest immutable evidence for each world and episode. All observed attempts (including misses, partial budgets, failures and timeouts) enter condition summary denominators.

## Measurement decisions

- At sampled `dt=.00025`, exactly 200 inclusive samples; native `.002` uses 25, still 50 ms. Contact gaps advance wall-time counters; only intended-pair contact contributes force. Completed window force/load records freeze.
- Positive axial, signed axial, and transverse-magnitude integrals are distinct. The hand-checked constant-force result is `10 * 200 * .00025 = .5 N s`.
- Six separate whole-episode and common-window load peaks/caps/utilizations/exceedance flags/counts/durations. The common-window peak includes pre-onset sliding-ring history by definition.
- Sixteen current-height bins, missing unvisited bins, sample counts/RMS/max, equal-observed-bin RMS/max, coverage/skipped bins and normalized phase extent. Hovering cannot claim path coverage merely from low geometric error. Postcontact error is separate.
- Metadata records the native site-state offset of minus one physics sample, physics cadence, policy termination cadence, configured impulse dose, and explicit protocol override. No extra simulator forward calls.

## TDD evidence

All commands used `PYTHONPATH=.` and `/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python`.

1. RED: `-m pytest tests/test_strike_evaluation.py -q`
   - Exit 1: `7 failed, 35 warnings in 5.95s`.
   - Expected missing implementation: `ModuleNotFoundError: No module named 'src.tasks.hammer.mdp.strike_evaluation'` inside each test.
   - GREEN, same command after collector implementation: exit 0, `7 passed, 35 warnings in 5.92s`.
2. RED: `-m pytest tests/test_strike_evaluation.py::test_pre_reset_capture_carries_immutable_collector_record tests/test_impact_speed_evaluation.py::test_cli_requires_checkpoint_except_smoke_and_checks_goals -q`
   - Exit 1: `2 failed, 35 warnings in 5.80s`.
   - Expected absent integration fields: `PreResetSnapshot` has no `latest_evaluation`; argparse namespace has no `measurement_protocol`.
   - GREEN integration run: `-m pytest tests/test_strike_evaluation.py tests/test_impact_speed_evaluation.py -q`; exit 0, `25 passed, 46 warnings in 35.23s`.
3. RED observed-bin aggregate: `-m pytest tests/test_strike_evaluation.py -q`
   - Exit 1: `1 failed, 14 passed, 35 warnings in 6.23s`; expected `KeyError: 'observed_bin_rms_m'`.
   - GREEN focused five-file run below: exit 0, `92 passed, 48 warnings in 38.61s`.
4. RED all-attempt summary: `-m pytest tests/test_strike_evaluation.py::test_summary_denominator_includes_no_contact_and_failed_attempts -q`
   - Exit 1: `1 failed, 35 warnings in 6.69s`; expected absent `summarize_strike_evaluations`.
   - GREEN focused five-file run: exit 0, `93 passed, 48 warnings in 35.81s`.

Final verification command (repeated after adding policy-cadence metadata):

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_strike_evaluation.py tests/test_impact_speed_evaluation.py tests/test_first_strike_event.py tests/test_impulse_bound.py tests/test_impulse_constraint.py -q
```

Final output/status: exit 0, `93 passed, 48 warnings in 39.25s`.

Warnings are upstream TorchScript script/load and legacy ONNX-export deprecations, not new collector failures. No pre-existing failure observed in the authorized focused scope. Repository-wide suite intentionally not run under the parent's focused-suite ruling.

## Self-review and limitations

Read the entire new module and evaluator diff; `git diff --check` clean. Native environment tests compare bitwise observations (both groups), actions, rewards, terminations/timeouts and RNG with/without diagnostics. Real MetricsManager/FirstStrikeEventTracker/SubstepImpulseAccumulator regression captures a load spike before any policy/CaT read, then accepts an onset with its prior ring history intact. Failure/timeout tests exercise real TerminationManager with simultaneous success. Saved-checkpoint native/common-window evaluation verifies original contract load and restored final metadata comparison.

Software semantics are verified, not trained performance or hardware safety. Native untrained traces do not establish useful impact, learned path following, speed compliance, or feasible joint-load containment. The collector intentionally retains the repository's native stale site-state convention and control-step peak component of the live Lambda property. Per-substep diagnostics add evaluation overhead; they are never installed by training. No correctness concern remains known at implementation handoff; independent task review is still required.
