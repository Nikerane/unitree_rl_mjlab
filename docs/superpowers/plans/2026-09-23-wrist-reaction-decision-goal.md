# Z1 wrist reaction reliability — bounded decision goal

**Status:** User explicitly approved execution with “go for it” after goal registration. Execution preparation resumed; previous approval-wait text below is historical, not a current blocker.

## Goal

Determine whether the existing wrist candidate (fixed diagnostic Kp=128, Kd=25.6) has a numerically defensible joint-load measurement basis for a training recommendation. Explain as far as the bounded evidence permits why reaction impulse changes with timestep, distinguish raw generalized-force sensitivity from baseline/contact-history contributions, and deliver an independently reviewed go/no-go decision with one concrete next action. Do not promise a safe gain envelope or redefine a failed comparison as success.

This goal closes a specific scientific decision, not the full thesis or hardware calibration. A supported no-go can complete this investigation; it does not complete gain qualification. A mere list of finished jobs cannot complete it.

## Established evidence — do not rerun

- Native screen supports utility/chatter improvements, not a calibrated range or safety claim.
- CPU restore45747119 passed both candidates.
- CPU repeat45776710: two fresh environments per candidate, exact equality of all19trace/3command arrays at .25ms. Within-process only.
- CPU timestep45784499: .25/.125ms startup, velocity and delivered impulse comparisons pass; reaction fails. Both physics runs finished with scientific-failure exit2.
- Independent prefix analysis: J6 p95 cap-normalized difference .12162/.09284 for doses .2/.3; maxima .16206/.55474. Worst worlds110/18. Held tails do not explain the failure alone.
- Production accumulator arithmetic was internally reconstructed in prior telemetry; its GPU noninterference gate failed. Existing cumulative traces cannot recover raw force and baseline separately.

## Ordered work and gates

1. **Freeze protocol and inspect reusable code.** Reuse existing `reaction_telemetry.py`, `reaction_decomposition.py` and `reaction-analysis-protocol.md` from `.superpowers/sdd/2026-09-22-wrist-calibration/`. No general recording framework or contact-row system. Review only the minimal CPU orchestration needed.
2. **At most four instrumented CPU replays.** Candidate arms1/3, existing .25/.125ms, original128world cohorts, fixed .28s horizon,20ms controls,50ms windows, same pinned source/assets/state/command tapes. Per arm, run .25ms first and stop that arm if fidelity fails; only then .125ms. Ten-minute scheduler bound per replay, no automatic retries or third timestep. Reuse matching saved uninstrumented CPU references.
3. **Noninterference before interpretation.** Require unchanged existing engineering/health/identity/command gates AND exact numeric equality across every previously saved trace and command array versus the matching CPU reference. This stronger CPU gate is prospective. A failure ends attribution for that pair; retain all failures and do not relax thresholds or select only matching worlds.
4. **Reconstruct, then decompose.** Check actual contribution, baseline-before/after, gate,50ms rolling buffer and20ms peak latch. Use the existing reconstruction allowance (1e-7 N·m·s absolute +1e-5 relative for summed float32 arithmetic only); baseline/counter identities exact. On verified source-terminal prefixes use the existing symmetric raw/baseline/gate algebra. Report every joint/world, signed and absolute distributions and preselected J5/J6 cases. Never sum marginal percentiles, invent percentages with near-zero denominators, or claim the algebra proves physical causation.
5. **Independent decision review.** Publish a concise decision table: native task utility, CPU repeatability, timestep agreement, measurement interpretation, candidate eligibility and training eligibility. Preserve prefix versus held-tail distinction. Name one smallest evidence-supported intervention or external decision if blocked. No automatic follow-on campaign.

## Acceptance and completion

- Original reaction criteria remain unchanged: per-joint paired median/p95≤.05 cap and maximum≤.10 cap; retain other frozen startup/full-contact criteria and actual physical windows.
- Telemetry/reconstruction validity must be demonstrated before component attribution. If a validity gate fails, conclusion is **attribution unresolved, no-go**—not that an implicated mechanism was proved.
- If decomposition succeeds, conclusion states supported numerical contributions and remaining ambiguities. Raw qfrc_constraint is generalized force including multiple constraint types, not isolated contact load.
- No favorable production/training recommendation from decomposition alone. The current timestep failure remains authoritative. A favorable decision would require separate prospective validation of the intended measurement and training execution configuration; this goal does not silently authorize it.
- Completion requires retained raw artifacts/provenance, quantified all-joint findings, independent review and a concrete next-action recommendation. The report must explicitly say whether gain calibration itself remains unresolved.

## Out of scope and stop conditions

No learning, gain-map changes, safety-cap changes, reward/controller redesign, simulator upgrade, backend migration of training, smaller cohort/horizon, unbounded timestep search, hardware-safe claims, automatic commits/pushes, or deletion of evidence. Do not expand to contact-row/friction decomposition unless separately justified and approved. On timeout, invalid telemetry, nonfinite/overflow, or identity failure, preserve evidence and stop the affected work; do not count a timeout as numerical evidence.

Production measurement changes require an evidence-backed proposal, focused failing test and independent review before implementation. Changing the intended physical safety quantity or deriving hardware limits additionally needs supervisor/manufacturer input.

## Consultation and adjudication

- **GPT-6 Sol:** reuse the existing recorder/decomposition; baseline updates after every off-contact sample, including tiny gaps. Distinguish raw generalized force from contact-only force; preserve gates. Adopted.
- **GPT-6 Astra:** cap the investigation at four replays and end with a supported go/no-go; do not make positive gain qualification a promised outcome. Adopted.
- **GPT-6 Luna:** avoid replacing numerical reliability with claims about gain/hardware safety; no free-form solver investigation or threshold relaxation. Adopted. Its proposed prefix-only next audit was already completed independently, so it will not be repeated. Its instrumentation recommendation becomes the bounded stage above, without adding contact-row infrastructure.

**Execution method:** subagent-driven minimal implementation and independent review, with root coordinating result analysis. Reuse existing experiment workspace; preserve earlier failures and reports. No new goal on autonomous velocity selection or policy architecture is folded into this one.

## Pre-execution compatibility review

Read-only inspection confirms `observe_replay` wraps the configured accumulator and existing sample callback, restores both functions on exit, and captures all required raw/baseline/gate/buffer channels without adding simulator steps. Its default command-line runner assumes one reference root, whereas the CPU references are split between `cpu-repeat-20260923/arm-N/repeat-1` (.25ms) and `cpu-timestep-20260923/arm-N/dt-0p125ms` (.125ms). Therefore execution needs a small experiment-local orchestration wrapper, not a production recorder rewrite. Exact equality must compare the original channel inventory only, excluding the additional telemetry channels; those new channels are validated through reconstruction. The existing engineering comparator already handles that inventory distinction.

The compatibility review was completed before user confirmation. User confirmation has now arrived; the automatic continuation is not being used as substitute approval. Launch still requires independent review of the minimal wrapper. No production change is authorized.
