# Impact-speed implementation: independent reviews and adjudication

Date: 2026-09-21. Scope: approved option A, one actor selecting speed once and retaining online joint-target/impedance control. Read-only reviews; no production implementation or training.

## Provenance

- Three native **gpt-6-astra / ultra** reviewers independently inspected the pilot checkout at `3ca89a297a58d8e59d74c4797caeb8629499f1aa` and installed mjlab/rsl_rl: PPO/action semantics; event/reward semantics; compatibility/evaluation scope.
- **opencode-go/glm-5.3**, session `ses_f3bae0112ffe49ijapl4O3TAh5`, completed successfully.
- **opencode-go/qwen3.8-max**, session `ses_f3badfe3cffe1H16bXZp9MkpY9`, completed successfully. This is the requested Max model, not the available Flash alternative.
- External reviewers received a bounded architecture packet with no credentials, private paths, checkpoint files or raw simulation data. Tools were denied; they did not inspect the local repository. Packet at `/tmp/impact-speed-review.jvNw2t/packet.md` is temporary, not durable experiment evidence. Their review summaries and the decisions are preserved here; do not represent the summaries as verbatim responses.

## Adopted findings

| Finding | Evidence / decision |
|---|---|
| Qualification's always-zero pending flag becomes 100 under empirical normalization when first set to one | Independently reproduced by two Astra reviewers; also flagged by GLM. Fixed scaling for appended goal/pending, preserve empirical normalization only for original40. Zero-initialize otherwise untrained pending-input columns in actor and critic. |
| Clipping can hide nonfinite output | Installed wrapper maps +inf to +1 before action processing. Add opt-in raw-action finite guard before clipping, plus direct action-term checks. |
| Reset events run before action reset | Action term exclusively owns goal sampling/reset, avoiding external-goal erasure. Latch in process_actions, not physics-substep apply_actions. |
| Event reward is multiplied by control dt | Return cost/step_dt with negative coefficient to charge the declared event amount. Use existing explicit negative-term classification, not a second reward pathway. |
| Existing reward-consumption helper gates on productive contact | New cost has its own exactly-once latch consuming all accepted finalized events; finalized beats simultaneous timeout. Miss/incomplete are labeled separately, never fake measured zero. |
| Existing checkpoint telemetry accepts two Gaussian tensors and12actions only | Separate experimental three-tensor/13D schema and PT infos metadata. Legacy guards remain strict. |
| Historical evaluators force log-only safety or pinned identities | One bounded experimental evaluator; do not repurpose frozen historical campaign scripts. |
| Late cost can be discounted away | Preserve current task/CaT for initial plumbing, report timing and discounted costs, stop if waiting dominates qualification. A genuine deadline redesign needs a separate decision. |
| Goal selection receives sparse learning signal and qualification does not train selector | Explicit phase initialization, fresh optimizer and post-latch interventions; no inference of causal command from correlation. |
| Exported actor does not own goal state | Test reset/latch/observation feedback around PT/ONNX inference, not just stateless network output. |

## Advice not accepted verbatim

- **GLM: bypass reward dt and CaT mechanisms entirely.** Rejected. The current hook explicitly separates negative reward terms. A correctly rate-scaled event cost registered in that mechanism is smaller and more consistent than a parallel reward channel. Physics runs at500Hz; reward weighting is at the50Hz control step, not1/500s.
- **GLM: masks might be derived from normalized storage.** Useful guard, but not an observed local defect. Installed PPO stores raw observations and delegates normalization to the model. Recompute from raw recorded observations and assert equality with saved masks; never derive masks from normalized inputs or live environment state.
- **GLM: selector advantage enters only through the first TD residual.** Not accurate for full GAE: later residuals also contribute through the recurrence. Sparse action frequency and temporal discounting remain genuine optimization concerns; no automatic critic reweighting is justified.
- **Qwen: external/policy pending contracts are inconsistent.** The packet explicitly distinguished the modes, but its introductory first-step description could be clearer. The plan states external pending0/mask12 from the first decision; self-selected pending1/mask13 initially. This is clarification, not an architecture correction.
- **Qwen: recompute old likelihoods and store dummy ignored actions.** Unnecessary. Preserve behavior log-probabilities and raw sampled actions already stored by PPO; masked unused coordinates do not contribute. Recomputing old likelihood with updated weights would be wrong.
- **Qwen: onset finite-difference denominator may span several samples.** Checked against tracker code and callback ordering: while unarmed/armed, the latest off-contact sample is refreshed each substep. The two-sample arming requirement does not itself stale that cache. Preserve the convention and regression/instrumentation tests rather than replacing it with a one-substep-stale estimate.
- **Qwen: freeze original40 normalizer statistics.** No basis to freeze them throughout new training. Preserve normalizer behavior for those coordinates; only the two new bounded features bypass empirical normalization. Version the widened schema and do not silently load old12D checkpoints.
- **Qwen: normalize entropy/KL by active-dimension count.** This would change the objective/scheduler scaling. Keep summed active-coordinate statistics initially; monitor them before considering a separate ablation.
- **Qwen: capped squared error has zero gradient beyond the cap.** Reward saturation reduces behavioral discrimination, but PPO does not backpropagate through physical speed error. Treat it as a reward-design tradeoff, not a missing differentiable path.

## Outcome and claim boundary

No reviewer identified a reason to replace direct joint/impedance control with a planner. Several concrete integration defects were found before implementation and incorporated into the [design](../superpowers/specs/2026-09-21-policy-selected-impact-speed-design.md) and [implementation plan](../superpowers/plans/2026-09-21-policy-selected-impact-speed.md).

This is not unanimous proof, a zero-error guarantee, successful learning or a safety result. The first implementation delivers tested plumbing and tools; physical bounds/cost/run identities remain evidence-gated before training. User review of the executable plan precedes subagent-driven implementation.
