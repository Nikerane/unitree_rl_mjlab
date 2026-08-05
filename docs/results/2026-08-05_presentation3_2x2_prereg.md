# Presentation Phase 1 — waypoint, velocity-CaT, and delivered-impulse 2×2

> **STATUS: FROZEN, NOT LAUNCHED.** This document supersedes
> `2026-08-04_wave3_pv_velocity_cat_prereg.md` for operational execution. The older
> file remains unchanged as historical evidence. Outcomes, thresholds, seeds, and arms
> below may not be changed after submission.

## Question and design

The frozen progress-guided P policies suggest that waypoint progress can straighten the
approach, but all six P controls violate the 3.1415 rad/s joint-velocity limit. This phase
asks two independent questions and their interaction:

1. Does faithful soft velocity-CaT improve intrinsic 500 Hz qvel legality?
2. Does increasing the existing delivered-impulse reward from weight 2 to 4 increase useful
   delivered impulse?
3. Does velocity-CaT change the effect of the higher delivered-impulse dose?

| Cell | Guidance | Velocity CaT | Delivered reward | Source |
| --- | --- | --- | --- | --- |
| **P** | waypoint progress, weight 8 | off | 2 | frozen controls, seeds 2–7 |
| **P+V** | waypoint progress, weight 8 | on | 2 | new training |
| **P+D4** | waypoint progress, weight 8 | off | 4 | new training |
| **P+V+D4** | waypoint progress, weight 8 | on | 4 | new training |

The required contrasts are P→P+V, P→P+D4, P+D4→P+V+D4, and P+V→P+V+D4.
`D4` means a higher dose of the already-active delivered-impulse reward; it is not the first
activation of impulse maximization.

## Frozen training contract

- New arms: P+V, P+D4, P+V+D4 only.
- Seeds: exactly 2, 3, 4, 5, 6, 7 for every new arm; no replacement or filtering.
- 200 iterations, 4096 environments, one A100 per run.
- Fixed Cartesian impedance and `ik_hammer_head` action only; never `set_gains`.
- Fixed nominal reset; no initial-position randomization.
- Velocity CaT: 500 Hz per-joint peak, limit 3.1415 rad/s, `max_p=0.5`,
  `min_p=0.0`, `tau=0.95`, no curriculum.
- Impulse CaT: **log-only in every cell**, `imp_max_p=0.0`; project caps unchanged at
  `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64]` N·m·s.
- Base rewards unchanged except the named delivered weight 2→4 treatment.
- `IMPACT_W`, `DELIVERED_W`, and `NAIL_DRIVEN_W` must be absent from the submitting
  environment. Registered task identity, not a CLI override, carries D4.
- No tube, hard termination, action clipping, gate reward, VIC, or additional reward term.

Active impulse-CaT tasks are not registered and are not launcher-allowed. The historical
`imp_max_p=0.5` is stale under the current 50 ms sliding-window multi-read semantics.

## Endpoints

The checkpoint/training seed is the descriptive unit; n=6 does not license a population-level
significance claim.

Primary per contrast:

- Velocity factor: count intrinsically qvel-legal at every recorded 500 Hz sample.
- Delivered-dose factor: delivered hammer–nail axial impulse.
- Interaction: whether D4's delivered-impulse change differs with velocity CaT on versus off.

Reported for every policy, without selection:

- all six gates crossed by first contact onset;
- pre-contact finite-segment perpendicular max/RMS and path ratio;
- contact, success, and clamped 32 mm nail depth;
- pre-contact axial speed and delivered impulse;
- peak per-joint qvel, responsible joint, and legality;
- per-joint Λ/cap and maximum utilization;
- contact duration, treatment payout, and velocity-CaT δ.

Measurement uses the existing digest-bound fixed reset and 500 Hz trace definitions. Training
is CUDA; high-rate fixed-reset evaluation is CPU, and no numerical cross-backend equality is
claimed. Every new video records its execution device and shows waypoint geometry only—never
gate disks—with a title naming P+V, P+D4, or P+V+D4.

## Gate before active impulse-CaT

First inspect P+V+D4 at the unchanged real caps. Do not lower caps to manufacture activation.
Active impulse-CaT remains blocked unless a separately frozen sampled evaluation establishes
meaningful eligible binding, including task success, full-horizon qvel legality, episode-level
Λ/cap tail statistics, and impact-versus-sustained-press classification. If binding is absent,
report the constraint as non-binding at this operating point and do not spend a full active-I
campaign. If binding is present, run a new stochastic C2 calibration and freeze its dose before
registering V+I or V+I+D4.

## Provenance

| Role | Revision |
| --- | --- |
| Presentation3 training | **to be pinned by the immediate follow-up commit before submission** |
| Asset | `b58ccd2f81fd246f27c1e8d88cf86484cd888703` |
| Frozen P seeds 2–3 training | `a6a9c970ea9eca00b34e8e9be806d22d396a964a` |
| Frozen P seeds 4–7 training | `e1a0282c9dc7b0283ae632a46a78debfd80bdf8c` |
| Frozen Wave-2 result | `cc125c08bfbdd7e75c2db83204c8eda8008464c9` |

The training revision is the first commit containing this preregistration and implementation.
The follow-up commit records that already-fixed revision; Vega checks out the training revision,
not the follow-up documentation commit.

## Frozen launch ledger

Only job ID and final state may be filled after submission.

| # | Arm | Seed | Run identity | Iters | Envs | Job ID | State |
| ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | P+V | 2 | `presentation3_pv_seed2` | 200 | 4096 | pending | pending |
| 2 | P+V | 3 | `presentation3_pv_seed3` | 200 | 4096 | pending | pending |
| 3 | P+V | 4 | `presentation3_pv_seed4` | 200 | 4096 | pending | pending |
| 4 | P+V | 5 | `presentation3_pv_seed5` | 200 | 4096 | pending | pending |
| 5 | P+V | 6 | `presentation3_pv_seed6` | 200 | 4096 | pending | pending |
| 6 | P+V | 7 | `presentation3_pv_seed7` | 200 | 4096 | pending | pending |
| 7 | P+D4 | 2 | `presentation3_pd4_seed2` | 200 | 4096 | pending | pending |
| 8 | P+D4 | 3 | `presentation3_pd4_seed3` | 200 | 4096 | pending | pending |
| 9 | P+D4 | 4 | `presentation3_pd4_seed4` | 200 | 4096 | pending | pending |
| 10 | P+D4 | 5 | `presentation3_pd4_seed5` | 200 | 4096 | pending | pending |
| 11 | P+D4 | 6 | `presentation3_pd4_seed6` | 200 | 4096 | pending | pending |
| 12 | P+D4 | 7 | `presentation3_pd4_seed7` | 200 | 4096 | pending | pending |
| 13 | P+V+D4 | 2 | `presentation3_pvd4_seed2` | 200 | 4096 | pending | pending |
| 14 | P+V+D4 | 3 | `presentation3_pvd4_seed3` | 200 | 4096 | pending | pending |
| 15 | P+V+D4 | 4 | `presentation3_pvd4_seed4` | 200 | 4096 | pending | pending |
| 16 | P+V+D4 | 5 | `presentation3_pvd4_seed5` | 200 | 4096 | pending | pending |
| 17 | P+V+D4 | 6 | `presentation3_pvd4_seed6` | 200 | 4096 | pending | pending |
| 18 | P+V+D4 | 7 | `presentation3_pvd4_seed7` | 200 | 4096 | pending | pending |

Retry only a documented infrastructure failure at identical seed, task, configuration, and
revision. A completed weak or failed policy is an outcome and is never replaced.
