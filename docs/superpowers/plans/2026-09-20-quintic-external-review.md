# Quintic pilot external review and launch gate

Scope: uncommitted implementation relative to `30bde8eabb4ddffdb4039e7719dd1e2e1a974afa`, tracked patch SHA256 `d8f82b4f0bfb9c24030a36e09d0162ab1cc9854670c49a2af0aa2a0dbb621e82`, plus `tests/test_quintic_reference.py`. User explicitly requested GLM and Qwen reviews. Only scoped code and experiment notes were supplied through OpenCode; no credentials or datasets.

## GLM 5.3 — correctness / compatibility

Provider/model: `opencode-go/glm-5.3`; completed verdict in session `ses_f41db5589ffe3rX2C3xdIl6CUv`. No provable blocker. Geometry and shared-reference plumbing judged consistent. Qualifications and disposition:

- Base detour must be zero and both observation groups must contain the expected terms: confirmed in the live pilot/config test.
- Curved rendering requires an anchored reference: normal reset precedes rollout/rendering.
- Scripted playback is chord-paced, not constant curved arclength speed: do not transfer old scripted speed claims to this task; measure actual velocity.
- The bend is small relative to the 50 mm tolerance: do not infer tracking from phase or task success. Report geometric error separately.
- Strictly-above-nail reset is required: intentional downward-only family, fixed realized start about 148 mm above nail.
- Cached nail state is reset lazily with the other anchors: follows the existing reset contract.

## Qwen 3.7 Max — spec / failure cases

Provider/model: `opencode-go/qwen3.7-max`; initial verdict in session `ses_f41dccf34ffe2HUVy6Bqfr0OMg`. Conditional concerns about absent `overshoot` and renderer defaults were resolved by inspecting the full source: both already exist. The proposed relaxation to allow starts at nail height is declined because this pilot explicitly requires downward descent. The warning that preview becomes unconditionally zero off-curve is inaccurate: preview takes the maximum with the prior monotone latch. An off-curve head can still have large position error; there is no claim of a differentiable reward gradient through simulation.

A follow-up with full reference source was requested to make these dispositions visible to Qwen. Neither reviewer output alone qualifies training; local tests and runtime checks remain the authority.

Qwen's completed follow-up explicitly reports no blockers and retracts the constructor/default/preview concerns. Its remaining statement that chord-paced playback stops 10–15% early is declined: `playback_length` takes the ceiling of chord/speed and `playback_target` computes/clamps phase from that same chord, so the terminal parameter reaches one. Nonconstant arc speed remains a valid qualification; premature endpoint termination does not follow. No source change was made on that unsupported claim.

## Run scope

After regression/review gates: one new left-start policy, seed 2, 100 iterations, 4096 environments, 24 steps per rollout, save every 25 iterations. Keep the shipped PPO recipe, 12 actions, nominal mapping, tracking weight 0.2/sigma 0.05 m, and impulse CaT p=0.2. Use a distinct Vega checkout/run directory and the verified asset revision `b58ccd2f81fd246f27c1e8d88cf86484cd888703`. No old checkpoints are resumed. This is an early pilot, not optimal-impedance or arbitrary-trajectory generalization evidence.

Operational notes: concurrent OpenCode starts encountered a database lock; reviewers were run sequentially. One GLM attempt was stopped after five minutes without a verdict. A completed later GLM request initially emitted no final text; a follow-up obtained the explicit verdict. These are tooling issues, not approvals.
