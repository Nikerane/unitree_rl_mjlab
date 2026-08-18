# Z1 impulse-CaT compact dose-curve review packet

## Question

Does this one-training-seed screen support moving to independent training-seed confirmation of
`p=0.2` rather than increasing dose? Please answer `CONFIRM`, `STOP`, or `REVISE`, then give the
single strongest reason and the smallest necessary follow-up.

## Protocol

Four frozen VIC-TT seed-2 policies were compared: log-only control (`p=0`) and targets trained at
`p=0.1`, `0.2`, and `0.3`. Active targets used 4,096 environments x 24 rollout steps, 500 PPO
iterations, velocity CaT active, and the same project-defined caps
`[0.82,1.64,0.82,0.82,0.82,0.82] N.m.s`. Rewards, actions, observations, physics, reference,
variable-impedance range, domain randomization, and curriculum were fixed.

Frozen evaluation used live `p=0` for every role, so impulse CaT did not alter evaluation. A fixed
64-world mean-action population was descriptive only. Three matched stochastic 4,096-world
populations—seeds `2`, `2026081701`, `2026081702`—were judged separately. Each paired interval used
10,000 whole-environment bootstrap resamples with seed `2026081703`. A dose passed only if every
population passed every preregistered impulse-risk, global-tail, velocity, joint-tail, utility,
delivered-impulse, identity, and finiteness gate; no pooling could rescue failure.

## Exact result

| Dose | seed 2 | seed 2026081701 | seed 2026081702 | Overall |
|---:|---|---|---|---|
| .1 | FAIL | FAIL | FAIL | FAIL |
| .2 | PASS | FAIL | PASS | FAIL |
| .3 | FAIL | FAIL | FAIL | FAIL |

`p=.1` was too weak: impulse-tail and velocity gates failed in all populations. `p=.3` was not
monotonically better: global-tail and velocity gates failed in all populations, with censored
utility fragments in two. `p=.2` was the strongest tested policy instance. It reduced target
impulse violations to zero in all three populations, lowered global p95/p99 utilization and true
velocity risk, retained success/productive-strike rates, and retained about 99.5% delivered
impulse. It nevertheless is not a formal PASS: one population's impulse-risk CI-high was
`-0.0048828125`, while the strict rule requires `< -0.005`, a miss of `0.0001171875`.

## Repeat sensitivity

The earlier and current `p=.2` evaluations declared identical initial-population hashes and RNG
stream IDs; they were not different samples. Separate A100 executions can differ through GPU
atomic ordering. Roughly `1e-6` preterminal action differences were amplified by contact and
termination. Exactly control environment 3253 changed: the old evaluation reached read 7 with J3
Lambda `0.8731314`, margin `+0.0531314`; the new evaluation succeeded at read 6 with Lambda
`0.54809135`, margin `-0.27190864`. CI-high moved from `-0.005126953125` (PASS) to
`-0.0048828125` (FAIL). Native-float classification is correct; this is neither rounding nor an
analyzer defect.

## Limits

This is one PPO training seed in simulation and native observed horizons only. Evaluation replicas
are not training seeds. Caps are provisional project thresholds, not manufacturer damage limits.
Soft-CaT supplies pressure, not a hard clamp. Terminal censoring prevents a complete-contact
claim. There is no hardware-safety, actuator-damage, causal complete-event, or optimal-dose claim.
