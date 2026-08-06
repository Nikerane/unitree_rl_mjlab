# Presentation3 V+D0 evidence record

Status: strict sampled evaluation accepted; fixed-reset rendering and diagnostics pending.

This record establishes a post-training analysis revision for the six-policy
delivered-reward-dose-zero arm. It adds no training, environment, reward,
controller, evaluator, or renderer code.

## Treatment identity

- Presentation label: `P+V+D0`
- Strict-evaluator label: `V+D0`
- Task: `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered0`
- Ordered-waypoint progress reward: weight 8.0
- Delivered-impulse reward: weight 0.0
- Velocity CaT: enabled from the 500 Hz substep peak, limit 3.1415 rad/s,
  `max_p=0.5`, `min_p=0.0`, `tau=0.95`
- Impulse CaT: measured but log-only, `imp_max_p=0.0`
- Per-joint project caps: `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64]` N m s
- Controller: fixed-impedance Cartesian relative DiffIK; no `set_gains`
- Curriculum: none

## Provenance

- Training/evaluator revision: `db48032449cb77184579f8cf62b9a5a228cfa77f`
- Asset revision: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`
- Training job: `40752437`, seeds 2--7, all six `COMPLETED 0:0`
- Training manifest SHA-256:
  `f55f20c80774916950f56252efa5fddebecf475de2408b221eb44273a5b7efd7`
- External evaluation launcher SHA-256:
  `aac8e39cd82f8fb9f46402138f7050d9d6d686334f75c09059c4124cc3822a0d`
- Sampled-evaluation jobs: pilot `40756866_0`; remaining rows
  `40756945_1`--`40756945_5`; all `COMPLETED 0:0`
- Remote evidence root:
  `/cephhome/eunikhilr/unitree_rl_mjlab_eval/presentation3/pvd0_sampled_20260806_r1/rows`

## Accepted sampled evaluation

Each checkpoint was evaluated with 256 environments and exactly the first two
completed stochastic episodes per environment. The frozen RNG streams are
reset `2036072919`, observation `2046072933`, and action `2056072941`.
All six rows emitted `P3_ROW_ACCEPTED` after schema-v3, hash, provenance,
episode-count, impossible-success, and live-impulse checks.

- Episodes: 3,072
- Productive first-strike successes: 3,072/3,072
- Observed 500 Hz joint-velocity legal: 3,072/3,072
- All six ordered waypoints completed: 2,982/3,072
- First-event delivered impulse: mean 0.317624 N s, median 0.312937 N s
- Episode cumulative delivered impulse: mean 0.427261 N s
- Mean post-first-event tail: 0.109637 N s
- Maximum observed impulse-cap utilization: 0.467645

The reward payout for `delivered_impulse` is exactly zero in every audited
episode while the physical delivered-impulse channels remain nonzero. Active
impulse CaT remains scientifically unidentifiable at the unchanged caps and is
not introduced by this arm.

## Next analysis step

Render every seed and record the CPU fixed-reset 500 Hz diagnostic at a clean
post-training analysis revision. Compare the fixed-reset and sampled-reset
D0/D2/D4 results explicitly; do not substitute cumulative episode impulse for
the finalized productive first-event impulse.
