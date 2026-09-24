# Wrist calibration: softer wrist helps, numerical qualification fails

## Decision

Do not promote a new production gain range or start long training **on the basis of this calibration**. The bounded diagnostics are complete, but **no numerically qualified wrist-gain envelope was established**. The selected fixed wrist point, Kp=128 / Kd=25.6, retains strike utility at the native timestep and strongly reduces wrist chatter. However, joint-reaction measurements remain timestep-sensitive, including before source termination. Same-timestep-validated CPU telemetry now makes the raw-force/baseline/gate bookkeeping interpretable, but its swap decomposition is algebraic rather than causal. This is a supported no-go for using this measurement to justify training gains, not a successful safety qualification.

The next action is a separately scoped, offline shadow comparison of the current every-off-contact baseline with a contact-onset-frozen baseline, using only the retained telemetry, after confirming with the supervisor what physical quantity the constraint is meant to represent. This is a diagnostic proposal, not an approved production metric change or another simulation campaign. Do not rewrite the reward, expand speed selection, or introduce a trajectory architecture to compensate for an unresolved measurement.

## Bounded finer follow-up: startup passes, reaction impulse does not

Existing traces localized candidate J5 disagreement to 1 ms and J4 to 6–31 ms, before nail contact. One candidate-only 0.25/0.125 ms replay pair then ran under a prospectively frozen first-60-ms startup gate. No additional levels are authorized in this extension.

Job **45738687** has **FAILED / exit 1** for both arms: the driver attempted to exclusively create its combined comparison file a second time. Crucially, both physics levels and identity/command/health checks had completed before that write. Raw summaries, complete substep arrays, commands, logs and failures are retained. Offline recovery independently verified full sample counts, all finite trace fields, identity against accepted 2 ms, exact command arrays versus 0.5 ms, and target/ctrl minus encoder-bias agreement at every substep. The independent analysis stores artifact hashes. No simulation was rerun, and the failed scheduler status is not relabeled successful. A future naming fix is local only and was not used to generate these results.

| Candidate policy | Startup worst-joint p95 / max peak-speed difference, fraction of cap | Censored / passive impulse p95 relative difference | Full-contact verdict |
|---|---:|---:|---|
| Dose 0.2 | 0.00331 / 0.00337 | 0.86% / 0.90% | Fail: reaction proxy |
| Dose 0.3 | 0.00801 / 0.00813 | 0.73% / 2.37% | Fail: reaction proxy |

Both startup rows pass the frozen 0.05 p95 / 0.10 max limits across all six joints and 128 worlds, with no contact in the first 60 ms. Full-horizon velocity, delivered impulse, onset and completion criteria also pass on this finest pair. Reaction does not: J6 p95 paired differences are **0.128 / 0.122 cap** (limit 0.05); dose-0.3 J5 p95 is 0.074 cap. Maximum reaction differences also fail: dose-0.2 J4/J5/J6 and dose-0.3 J1/J5/J6. All 128 worlds per level succeed and complete passive windows; no monitored overflow/nonfinite values.

Reaction sensitivity remains before each recorded source termination: J6 prefix p95 differences are **0.122 / 0.093 cap**. Therefore it cannot be dismissed as only post-terminal holding. At the finest level, native-prefix reaction exceedances are J2=10/J6=4 for dose 0.2, and J6=38 for dose 0.3; all other joint counts are zero. These remain diagnostic proxy/cap counts, not hardware damage estimates.

This narrows the problem considerably: finer integration supports bounded startup agreement for this candidate and improves delivered-impulse agreement, but **does not establish a trustworthy full-strike safety-load comparison**. Investigate the contact-gated, last-off-contact baseline subtraction and contributing constraint forces before changing that metric or safety thresholds. No causal attribution to those components is established yet. No more timestep levels were added after this check.

### Existing-trace localization

Independent inspection selected dose-0.2 worlds 66/J5 and 110/J6, and dose-0.3 worlds 75/J5 and 18/J6. All source-terminal intervals end within 36 ms of first contact, shorter than the 50 ms rolling window: eviction of old contact samples cannot explain these prefix discrepancies. Both selected J5 cases have uninterrupted contact; their paired differences are not just a shifted contact edge. Dose-0.3 world18 has a 0.25 ms coarse-only contact gap just before the 0.180 s command update, a specific baseline-history difference to inspect rather than a proven cause.

Saved reaction is a rectified, baseline-subtracted window statistic; it cannot recover signed raw generalized force and the subtracted baseline separately. A read-only telemetry repeat was designed at the **same** 0.25/0.125 ms levels, with a prospectively fixed same-dt non-interference check. It adds neither a finer timestep nor a new controller. Any permitted offline decomposition must be labeled algebraic sensitivity, not isolated contact force or causal attribution.

Telemetry update: job45739875 executed the 0.25 ms level for both candidates, then stopped on the fixed non-interference gate; neither 0.125 ms run proceeded. A few contact/finalization events differ from the original same-dt runs, with tiny state differences beginning before contact. No cap classifications flipped, but that does not override the failed gate. Both old and new jobs ran on nodegn57 with matched source/state/model/commands. Recorder-induced execution timing and ordinary run-to-run variation remain competing explanations. Internal contribution/baseline/window reconstruction agrees (maximum rolling/pulse difference about1.6e-7 N·m·s), validating bookkeeping only—not attribution to the old timestep mismatch. One unchanged, uninstrumented 0.25 ms repeat per candidate is the bounded control being prepared; no tolerances or timestep levels are added.

Data: [startup/refinement recovery and paired analysis](assets/2026-09-22_wrist_calibration/startup-refinement-analysis.json). Protocol and independent reviews are retained in `.superpowers/sdd/2026-09-22-wrist-calibration/` as `startup-refinement-protocol.md`, `startup-results-review.md`, and `startup-recovery-review.md`. Raw-result and scoped recovery reviewers independently approved this split verdict; the overall gain-calibration goal remains active because contact-reaction credibility is unresolved.

## What actually ran

- Native five-point gain screen: job 45735252, 10 evaluations, both existing seed-2 policies (dose 0.2/0.3), three speed goals, 128 first episodes per condition; **3,840/3,840 hits/completions**. All five points passed the frozen native utility gates. These are evaluations, not trained policies.
- Candidate selection: 128/25.6, with 44/768 episodes exceeding any existing joint impulse/speed cap, versus 108/768 at the SDK-derived 256/25.6 reference. Candidate high-goal J2 impulse exceedances increased from 5 to 10 and 3 to 7 relative to the original controllers. A calmer wrist is not automatically a safer whole arm.
- Contact refinement: job **45738100**, four array elements, all COMPLETED 0:0. Each reused a frozen original or candidate command/gain tape for all 128 worlds at **1 ms and 0.5 ms**, compared with the accepted native 2 ms replay. High goal 2.2 m/s only; no policy execution at modified timesteps.
- All levels preserved 20 ms control, 50 ms measurement windows, full initial state/model, and the shared 0.28 s horizon. All worlds made contact and completed their passive windows. No monitored capacity overflow or nonfinite values occurred. Learning updates: zero.

Production source, gains, safety caps, checkpoints, and assets were not changed. Source is pinned to cc46d9998a3c9cd5d8813e89b57e44248353e03e; assets to b58ccd2f81fd246f27c1e8d88cf86484cd888703.

## Finest-pair result

The frozen engineering screen requires paired delivered-impulse median/p95 differences within 5%, all-joint normalized reaction/speed median/p95 differences within 0.05 cap and maxima within 0.10 cap, onset differences within 2 ms, and no new misses/incomplete windows. These are diagnostic tolerances, not hardware limits.

Paired relative errors use the 0.5 ms value as denominator, with a 0.001 N·s floor. Each row retains the same 128 world IDs.

| Existing policy / controller | Success-censored impulse p95 difference | Passive 50 ms impulse p95 difference | Largest onset difference | Verdict |
|---|---:|---:|---:|---|
| Dose 0.2 / original | 13.44% | 23.27% | 1.5 ms | Fail |
| Dose 0.2 / candidate | 24.04% | 29.61% | 1.5 ms | Fail |
| Dose 0.3 / original | 8.05% | 11.51% | 1.0 ms | Fail |
| Dose 0.3 / candidate | 16.19% | 19.10% | 1.0 ms | Fail |

All four also fail the all-joint reaction and velocity criteria. Small median impulse differences do not rescue large paired tail errors. Candidate J6 peak-speed differences are small (max about 0.0011/0.0013 cap), while J4/J5 p95 speed differences remain about 0.149/0.227 cap and 0.172/0.209 cap for the two doses.

The issue is not solely contact or the artificial held tail: precontact median peak J4 speed for the candidate changes from 0.835 to 0.686 cap (dose 0.2) and 0.844 to 0.675 cap (dose 0.3) between 1 and 0.5 ms. Candidate J5 changes from 0.429 to 0.293 and 0.414 to 0.304 cap. These are observed sensitivities, not proof of their cause.

## Separate the real episode prefix from the diagnostic tail

After each recorded first episode terminates, the replay holds its final processed target and gains. It does not continue the learned policy or include a reset episode. This held tail can create substantial additional loading and must not be reported as native policy performance.

Example at 0.5 ms, candidate dose 0.3:

- Through its source terminal interval: J2/J3/J4 reaction exceedance counts are **0/0/0**; J6 is 35/128.
- Across the full 0.28 s replay: J2/J3/J4 counts are **82/14/55**; J6 is 58/128.

The full-horizon reaction is a contact-gated, baseline-subtracted generalized-force proxy integrated over sliding 50 ms windows. It is not isolated contact-row impulse or literal gearbox damage. Passive 50 ms delivered impulse includes post-success behavior; the native first-strike metric freezes at success or window expiry. Both are reported separately. The remaining numerical failures also exist within source-terminal prefixes, so discarding the tail does not establish qualification.

## Replay-fidelity qualification

Original strict native replay checks failed and remain recorded as failures; their small mismatch has no proven cause. Before fresh captures, a separate engineering-equivalence profile was frozen: exact commands/events, no cap-classification flips, and at most 0.5%-of-cap reaction/speed error plus unit-specific state budgets. Fresh validation job 45737812 passed that separate profile in all four arms while retaining strict failure/exit 2.

Refinement therefore has an explicit engineering-equivalence basis, **not exact native reproduction**. Its much larger timestep sensitivity is not concealed by retroactively relabeling strict failures. Optional geometry point-slot instrumentation was absent and is reported unavailable; actual contact-match/global capacity checks remained active.

## Interpretation and next gate

### CPU timestep result (45784499)

Both .125 ms CPU runs completed physics, preserving identities, command tapes and monitored health, but exited2 for scientific failure. Compared with .25 ms CPU repeat1, both startup screens, speed, delivered impulse, contact onset and complete-window checks passed; only the joint-reaction comparison failed. J6 full-horizon p95 differences were 0.12819 and 0.12181 of the existing wrist cap for doses .2/.3. These are differences normalized by a provisional cap, not percentages of physical injury risk.

Independent source-terminal-prefix analysis also fails: J6 median/p95/max differences were .02463/.12162/.16206 for dose .2 and .02135/.09284/.55474 for dose .3. Dose .3 J1/J5 maxima were .11264/.14641. Worst wrist worlds110/18 match the earlier GPU pattern closely. Thus postterminal holding and ordinary GPU repeat variation cannot individually explain away the timestep discrepancy. This similarity is not causal identification or proof of CPU–GPU equivalence.

Raw evidence: `assets/2026-09-22_wrist_calibration/cpu-timestep-20260923/`. No further timestep reduction or training followed. The original comparison JSON's note that source-prefix analysis was still required is stale; that analysis was independently recomputed from the completed telemetry and is superseded below.

### Bounded CPU telemetry decision (2026-09-23)

The authorized four instrumented CPU replays are complete: corrected coarse job **45806952** (arms 1 and 3 at 0.25 ms), then conditionally allowed fine jobs **45807169** (arm 3) and **45807170** (arm 1) at 0.125 ms. Every one of the four same-timestep observation gates passed. The earlier malformed-launch job **45794614** failed before Python/simulation and is retained as an orchestration error, not physics evidence. No additional runs, training, or production changes followed.

| Arm / dose | dt | Observation | Engineering | Original traces | Commands |
|---|---:|---|---|---|---|
| 1 / 0.2 | 0.25 ms | pass | pass | exact, 19/19 channels | exact, 3/3 arrays |
| 1 / 0.2 | 0.125 ms | pass | pass | exact, 19/19 channels | exact, 3/3 arrays |
| 3 / 0.3 | 0.25 ms | pass | pass | exact, 19/19 channels | exact, 3/3 arrays |
| 3 / 0.3 | 0.125 ms | pass | pass | exact, 19/19 channels | exact, 3/3 arrays |

Independent raw-data review confirmed the four saved trace/command pairs and their hashes. The accumulator contribution, baseline-before/after identities, 50 ms rolling sum, and 20 ms pulse latch reconstruct within the frozen `1e-7 N·m·s + 1e-5 relative` summation allowance; the largest rolling error is **1.97e-7 N·m·s**. All 128 worlds and six joints remain present. Each real source-terminal prefix is at most **36 ms after first contact**, with no counter reset, so it is distinct from the held-command tail and shorter than the 50 ms buffer.

The unchanged cross-timestep reaction gate still fails within those source-terminal prefixes. Its paired absolute differences, normalized by the existing joint caps, are:

| Dose | Statistic | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.2 | p95 | 0.01087 | 0.00614 | 0.00426 | 0.00585 | 0.03213 | **0.12162** |
| 0.2 | max | 0.01266 | 0.00927 | 0.00643 | 0.05638 | 0.08710 | **0.16206** |
| 0.3 | p95 | 0.00848 | 0.00648 | 0.00377 | 0.00487 | 0.03867 | **0.09284** |
| 0.3 | max | **0.11264** | 0.03897 | 0.01084 | 0.01126 | **0.14641** | **0.55474** |

Frozen limits remain p95 `0.05` and max `0.10`; J6 fails both doses, and dose 0.3 also exceeds max on J1 and J5. The independent recomputation confirms this is not a held-tail-only result or a telemetry mismatch. These are simulation-proxy comparisons, not physical damage probabilities.

For the four preselected worst-world cases, the symmetric algebraic allocation (coarse minus fine, N·m·s) is:

| Dose / world / joint | Net difference | Raw-force allocation | Baseline allocation | Gate residual |
|---|---:|---:|---:|---:|
| 0.2 / 66 / J5 | +0.02142560 | −0.00008093 | +0.02150652 | 0 |
| 0.2 / 110 / J6 | +0.00265777 | −0.00038657 | +0.00315991 | −0.00011557 |
| 0.3 / 75 / J5 | +0.03601685 | −0.00035926 | +0.03637612 | 0 |
| 0.3 / 18 / J6 | +0.00909773 | −0.00070415 | +0.00847336 | +0.00132852 |

Those selected J5/J6 cases are strongly baseline-history-sensitive **within the algebra**, while raw allocation partly offsets the net difference. That is not proof that the baseline implementation caused the original timestep sensitivity: raw force and baseline share a different simulated history, and gate differences can indirectly change later forces and baselines. Across all worlds, raw allocations are also material (e.g. J5 absolute p95 `0.007724 / 0.006886` for doses 0.2/0.3), so do not generalize that “the baseline explains everything.” The retained signed 128×6 per-world matrices and separate absolute per-joint distributions are in [the decomposition artifact](assets/2026-09-22_wrist_calibration/cpu-reaction-telemetry-20260923/reaction-decomposition-20260923.json).

**Decision:** no-go for using Kp=128/Kd=25.6 as a training gain justified by the current per-joint reaction measurement, or for claiming this candidate is load-qualified. This does not establish that the robot cannot strike successfully; it means a training run using this metric as its safety/load basis would not answer that question reliably. Gain calibration remains unresolved. **Next step:** propose the offline baseline-history shadow comparison and ask the supervisor whether the intended bound is all-constraint generalized joint reaction or a contact-specific quantity; do not change production code, caps, or training until that definition and a prospective validation are agreed.

Independent reviews: the raw-data reviewer and a separate Astra review both reproduced the telemetry gates/decomposition and agreed with this scoped no-go. Their review notes and limitations are retained in `.superpowers/sdd/2026-09-22-wrist-calibration/cpu-reaction-telemetry-review.md`.

### CPU repeatability result (45776710)

Both candidate arms completed two fresh-environment CPU replays at 0.25 ms (exit0, about 2 minutes 50 seconds per arm). Both the unchanged engineering comparison and exact numeric equality checks passed. On resumption after the user's pause, root independently loaded all saved substep and command arrays and confirmed exact equality for both pairs. Raw results are retained under `assets/2026-09-22_wrist_calibration/cpu-repeat-20260923/`.

This establishes repeatability for these within-process CPU pairs only. It does not establish cross-process/host determinism, CPU–GPU equivalence, physical accuracy, or a qualified wrist-gain range. The next approved bounded stage is one 0.125 ms CPU replay per candidate compared with its saved 0.25 ms CPU reference, using unchanged startup/full-contact criteria. No training or production gain/safety-limit changes occurred.

### CPU compatibility result (45747119, checked 2026-09-23)

Both candidate arms completed CPU construction, reset, sensor preflight and exact saved-state restoration successfully (exit0, 37/35 seconds). Their `probe.json` reports status passed, stage complete, no error, and zero physics rollout steps. Reset includes forward computations. This establishes compatibility of the existing CPU replay setup only, not numerical repeatability or gain qualification. Raw reports/scripts/logs are retained under `assets/2026-09-22_wrist_calibration/cpu-restore-20260923/`.

The user subsequently approved two fixed-command CPU replays per candidate. The next gate requires both exact saved-array equality and the existing unchanged consistency profile; no timestep comparison proceeds from a failed repeatability gate. No learning, production gain/cap changes, or dependency changes are part of this diagnostic.

### Uninstrumented repeat control (45740548)

Both candidate arms completed their 0.25 ms physics replay but exited 1 because the unchanged same-timestep consistency gate failed. This repeat used the original replay function without the additional reaction recorder. It therefore shows that the recorder is **not necessary** for a consistency failure; it does not establish the cause or a statistical noise floor.

Both arms exceeded state, velocity and reaction error budgets and differed in contact flags. Dose 0.2 also differed in finalization/reason flags. Delivered-impulse errors remained below the 0.001 N·s budget (0.000692 and 0.0000236 N·s respectively). Neither arm changed any joint's peak cap classification, in either the source-terminal prefix or full horizon. These narrower passes do not override the full gate failures.

The largest per-joint reaction differences were 0.01182 N·s for dose 0.2 and 0.00804 N·s for dose 0.3. These are maximum differences between corresponding trace samples, not confidence intervals or changes in physical safety limits. Reports and complete arrays are preserved under `assets/2026-09-22_wrist_calibration/reaction-control/`. No additional finer level, repeat, training, gain change, or tolerance relaxation followed this control.

The next decision is how to obtain a reproducible diagnostic reference on the installed simulator stack. CPU replay feasibility is being inspected before proposing a bounded experiment; no backend change is yet validated. See the [official-source repeatability review](../research/reward-design/2026-09-22_mjwarp_repeatability_sources.md).

1. Keep the native screen as evidence that softer wrist settings merit investigation—not evidence of an optimal or safe continuous range.
2. The observed J1/J4/J5 startup sensitivity is substantially reduced by the bounded 0.25/0.125 ms comparison above. This does not retroactively qualify the native 2 ms evaluations or establish that training at a changed timestep is equivalent.
3. Diagnose the residual contact-reaction measurement and physical constraint contributions with a separately scoped check. Preserve force limits and physical measurement durations; do not silently loosen safety caps or substitute a different quantity. Do not continue an open-ended finer-timestep search.
4. Only after precontact and contact comparisons are credible, choose a provisional gain mapping and run a small training pilot with the normal reward/contact preflight. Longer training then tests learning benefit, not whether the simulator's load measurements are stable.

Neither these results nor the existing externally conditioned speed policies establish autonomous optimal terminal-speed choice or optimal variable impedance. No trained seed replication or hardware qualification is claimed. The three-goal native screen and high-goal numerical test answer different questions.

## Reproduction and independent checks

- [Native screen](2026-09-22_z1_wrist_gain_screen.md)
- [Strict replay audit](2026-09-22_z1_native_replay_audit.md)
- [Paired numerical data, six-joint quantiles and prefix/tail cuts](assets/2026-09-22_wrist_calibration/refinement-analysis.json)
- Raw summaries, substep arrays, command tapes, and job logs: `assets/2026-09-22_wrist_calibration/refinement/`; accepted 2 ms evidence: `engineering-validation/` in the same asset root.
- Offline calculation: `.superpowers/sdd/2026-09-22-wrist-calibration/analyze_refinement.py`.
- Independent raw-data review: `.superpowers/sdd/2026-09-22-wrist-calibration/refinement-results-review.md`; separate final analyzer/synthesis review retained alongside it.

Workflow rulings retained in that workspace's `progress.md`: preserve scratch evidence/no automatic commits; read-only reuse of prior qualified observer with hashes; reuse an implementer after the agent limit with independent review; permit disjoint diagnostic implementations with integration review; freeze a prospective engineering profile without erasing strict failures. Costs: extra artifacts/dependency/context risk and possible subtle replay mismatch, mitigated by hashes, separate review, fresh validation, and explicitly exploratory conclusions.
