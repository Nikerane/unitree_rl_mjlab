# Impulse6 reference-guided impulse screen: preregistration

**Date:** 2026-08-06
**Status:** frozen before launch; exploratory fixed-impedance Z1 screen

This preregistration implements the owner-approved design at commit `ab35715`.
The required asset revision is
`b58ccd2f81fd246f27c1e8d88cf86484cd888703`. The exact training revision is
the implementation commit containing this campaign, which is intentionally not
invented in this commit. A docs-only follow-up commit will pin its 40-hex SHA
before any submission.

## Frozen controller and launch contract

- Cartesian relative DiffIK hammer-head action only, position scale `0.15`;
  no `set_gains` action and no variable impedance.
- Fixed actuator gains: joints 1, 3, 4, 5, 6 have Kp/Kd/effort limit
  `1000/100/30 N m`; joint 2 has `1500/150/60 N m`; the non-policy-controlled
  gripper has `100/20/30 N m`. MJCF passive damping and DiffIK numerical
  damping (`0.05`) are not actuator derivative gains.
- Straight six-waypoint guidance stays at weight 8. The 500 Hz substep
  joint-velocity CaT remains at limit `3.1415 rad/s`, `max_p=0.5`,
  `min_p=0`, `tau=0.95`.
- The project-defined generalized constraint-reaction exposure caps stay
  `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64]`. Impulse CaT is log-only:
  `imp_max_p=0` is forced in the task and training command.
- Every array has `CAMPAIGN=impulse6`, `SEEDS="2 3 4 5 6 7"`, `ITERS=200`,
  `--array=0-5`, 4096 environments, and clean 40-hex code and asset revisions.
  `IMPACT_W`, `DELIVERED_W`, and `NAIL_DRIVEN_W` must be unset, including not
  exported as empty strings. The registered task is the treatment identity.

## Frozen primary-screen identities

The 36 rows below are the only initial identities. `S` is the baked
impact-progress weight and `D` the baked event-linear delivered-impulse weight.
No completed weak seed is replaced.

| S | D | seed | task | short |
|---:|---:|---:|---|---|
| 0 | 0 | 2 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0` | `s0d0` |
| 0 | 0 | 3 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0` | `s0d0` |
| 0 | 0 | 4 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0` | `s0d0` |
| 0 | 0 | 5 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0` | `s0d0` |
| 0 | 0 | 6 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0` | `s0d0` |
| 0 | 0 | 7 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0` | `s0d0` |
| 0 | 4 | 2 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4` | `s0d4` |
| 0 | 4 | 3 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4` | `s0d4` |
| 0 | 4 | 4 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4` | `s0d4` |
| 0 | 4 | 5 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4` | `s0d4` |
| 0 | 4 | 6 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4` | `s0d4` |
| 0 | 4 | 7 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4` | `s0d4` |
| 0 | 16 | 2 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16` | `s0d16` |
| 0 | 16 | 3 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16` | `s0d16` |
| 0 | 16 | 4 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16` | `s0d16` |
| 0 | 16 | 5 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16` | `s0d16` |
| 0 | 16 | 6 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16` | `s0d16` |
| 0 | 16 | 7 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16` | `s0d16` |
| 8 | 0 | 2 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0` | `s8d0` |
| 8 | 0 | 3 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0` | `s8d0` |
| 8 | 0 | 4 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0` | `s8d0` |
| 8 | 0 | 5 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0` | `s8d0` |
| 8 | 0 | 6 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0` | `s8d0` |
| 8 | 0 | 7 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0` | `s8d0` |
| 8 | 4 | 2 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4` | `s8d4` |
| 8 | 4 | 3 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4` | `s8d4` |
| 8 | 4 | 4 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4` | `s8d4` |
| 8 | 4 | 5 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4` | `s8d4` |
| 8 | 4 | 6 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4` | `s8d4` |
| 8 | 4 | 7 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4` | `s8d4` |
| 8 | 16 | 2 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16` | `s8d16` |
| 8 | 16 | 3 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16` | `s8d16` |
| 8 | 16 | 4 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16` | `s8d16` |
| 8 | 16 | 5 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16` | `s8d16` |
| 8 | 16 | 6 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16` | `s8d16` |
| 8 | 16 | 7 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16` | `s8d16` |

## Endpoints, estimand, and anti-farming gates

The primary endpoint is productive first-strike nail-axis delivered impulse from
the single immutable first-contact event, with failures assigned zero. It is
reported as **success-censored productive first-event impulse**: success can end
the event at the nail-depth threshold before the nominal 50 ms window. Event
duration and termination reason are mandatory companion fields. Episode-cumulative
impulse is secondary only.

The remaining fixed endpoints are 500 Hz peak joint velocity; first-event contact
dwell; first-event peak and mean axial force; transverse delivered impulse; contact
onset count and post-event contact; pre-contact axial hammer speed; productive
nail-depth advance and clamped final depth; six-waypoint completion by contact
onset; pre-contact perpendicular max/RMS and path ratio; all six constraint-reaction
exposures and maximum cap utilization; and downward-action saturation frequency.

A candidate must have 6/6 productive successes, at least 5/6 six-waypoint
completion by first contact, 6/6 qvel-legal policies, no contact interval at or
above 50 ms, no second contact onset, and finite hash-bound evidence with intended
payout. Relative to its paired `D0` control, reject it if median dwell increases
over 25%, transverse-to-axial impulse ratio worsens over 10%, or useful
speed/progress falls over 10%. Among eligible cells, select the lowest dose with
at least 10% median primary improvement and paired improvement in at least 5/6
seeds. These exploratory margins are not significance tests; report every cell.

## Conditional branches and confirmation

`D8` is not an initial seventh cell. It can be added only when `D4` credibly
improves useful impact and `D16` degrades or is press-like, making the intermediate
dose mechanistically informative.

Active impulse CaT (`V+I` / `V+I+M`) is closed unless eligible evidence has at
least 5% cap crossings, a lower confidence bound above zero, p99 maximum cap
utilization above 1, at least 95% productive success, and at least 70% of
binding-joint exposure before first release. If it remains closed, record
`ACTIVE_I_NOT_IDENTIFIABLE` with measured utilization; do not add active-I tasks
or lower caps to manufacture an effect. If open, re-qualify the multi-read signal
and calibrate a new CaT dose; never reuse historical `imp_max_p=0.5` by default.

After screening, freeze one winner and one exact incumbent/control before looking
at confirmation outcomes. Train both on matched fresh seeds `24 25 26 27 28 29`
only after an unused-seed audit confirms none were used in development. There is no
checkpoint selection, seed substitution, or replacement of a completed weak seed.
Retry only a documented infrastructure failure at the identical
task/seed/config/revision identity. Replication requires eligibility plus paired
primary improvement in at least 5/6 seeds without crossing the anti-press margins.
