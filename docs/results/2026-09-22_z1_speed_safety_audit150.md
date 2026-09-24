# Current150-update frozen-policy safety/reward audit

## Scope and verification

User approved diagnostic checking only. Evaluation-only Vega job **45691466** completed exit0:0 in1m02s, no retries or learning. Source/controller/reward unchanged. Replayed both saved150-update policies, three goals,128 first episodes per goal. CPU64-transition observer smoke and reward-decomposition assertions passed before launch.

Observer reads after native reward/metric computation and before autoreset. It adds existing separate velocity/impulse CaT telemetry and per-term weighted,dt-scaled rewards; it does not recompute constraints, rewards, observations or actions. Immediate learner reward reconstructed as`r_total − delta*r_positive`; this is not a full PPO-return/gradient audit. All768firstepisodes hit and completed without timeout.

Checkpoint hashes, physical reset hashes, CPU/CUDA RNG hashes and first-action hashes match the original150-update evaluation. Middle-cost max speed discrepancy0.0000894m/s. High-cost has two replay deviations0.005599m/s(low/world2) and0.004087m/s(mid/world7), exceeding the diagnostic0.001m/s replay tolerance; all other differences are small. That failed strict replay check is explicitly retained in`audit.json`, not silently reclassified as exact reproduction. Aggregate errors/J6medians are essentially unchanged, but original qualification results remain authoritative. Cause of these replay differences is not isolated.

## Findings

| Policy | Goal m/s | J6-over-cap control reads | Reads where velocity masks/ties impulse | Reads at impulse probability ceiling0.2 | Episodes with terminal J6violation |
|---|---:|---:|---:|---:|---:|
| Middle |1.50|148|0|120|122/128|
| Middle |1.85|135|0|111|128/128|
| Middle |2.20|133|0|108|128/128|
| High |1.50|174|6|141|128/128|
| High |1.85|161|16|135|127/128|
| High |2.20|145|13|103|121/128|

**Velocity masking is not the main explanation in this replay.** Impulse dominates every middle-cost J6violating read, and most high-cost reads. Its safety signal is active, not absent/log-only.

**The old50-update wasted-follow-through example is not representative of most current episodes.** Middle-cost median delivered fraction at the first contact-containing read is only26.7–28.3%; substantial useful impulse is delivered afterward. A subsequent rise in J6window peak after a previous read already delivered≥95%of the final impulse appears in only2/384middle-cost and21/384high-cost episodes. The95%cutoff is an exploratory diagnostic, not a new qualification threshold.20ms sampling cannot resolve within-bin wasted motion or individual force peaks.

**Most reward arrives at the violating success transition.** Across conditions,90.2–98.4%of summed positive reward is paid on terminal steps with J6overcap. Median scaled terminal reward remains positive:2.124/2.181/2.236(middle),2.025/2.082/2.146(high). Sample worst-world terminal reward includes completion2.0 beforeCaT, with onset-speed and delivered-impulse rewards also paid there. At impulse-dominant delta0.2,80%of positive immediate reward survives; negative costs are unscaled. True termination leaves no future bootstrap to additionally discount, although earlier CaT attenuation can affect preceding returns.

This supports testing the constraint-strength/payoff trade-off; it does **not** prove that0.2caused the learned behavior, that larger probability will improve it, that clipping is a bug, or that the provisional cap is physically achievable at the requested speeds.

## Important interpretation limits

- CaT probabilities and winner attribution here use fresh128-world evaluation normalizers. The evaluator loads actor weights, not the historical1024-world stochastic rollout and its entire CaT normalizer history. These are not reconstructed training-time probabilities or policy gradients.
- Structural impulse probability ceiling0.2 and the scale-positive reward rule are shared with training. This provides a mechanism hypothesis, not retrospective causal proof.
- `marginal_impulse_reward_discount` is incremental immediate positive-reward attenuation compared with velocity-only delta on the same frozen transition. It excludes changes to future continuation, timeout bootstrap and counterfactual policy adaptation.
- Joint impulse is a contact-gated, baseline-subtracted50ms rolling-window reaction measure, latched over20ms control intervals. It is not an instantaneous force peak or certified hardware damage threshold.
- One training seed, fixed physical start and observation noise; no independent-seed/generalization claim. Both candidates still fail original E1qualification.

## Next proposal — not launched

Prefer a matched test of **one stronger impulse-CaT setting versus the unchanged0.2control**, starting with the middle speed-cost condition, rather than changing the spline, controller mapping or follow-through now. Keep provisional caps, velocity constraint, speed cost, geometry, training budget and evaluation fixed. Prespecify hit rate, actual delivered impulse, speed accuracy, and violation severity **and frequency** together; do not select using maximum load alone. Numeric dose, initialization/transfer provenance and budget need an explicit next-run plan/approval. If caps cannot be met without destroying task utility, investigate feasibility/metric meaning; do not silently raise caps.

No new training, autonomous-speed phase, safety retuning, or production code change occurred in this audit.

## Evidence

Local`docs/results/assets/2026-09-22_speed_extension150/trace150-diagnostic/`: per-arm rawtrace/episodes/summary/logs and`audit.json` including replay-check failures. Scripts:`.superpowers/sdd/2026-09-21-vega-speed-qualification/{trace150.py,trace150.sbatch,analyze150audit.py}`, layered on existing`trace_screen.py`.

Independent reviewer recomputed all six masking/ceiling/terminal counts, reward decomposition, reward-mass fractions and late-growth counts from raw traces and agreed. Reviewer emphasizes fresh-evaluation-normalizer caveat, unproven cap feasibility and historically nonmonotonic dose response: stronger safety pressure is a testable intervention, not an established fix. Future comparison must also report per-joint and velocity tails to detect redistributed loads.
