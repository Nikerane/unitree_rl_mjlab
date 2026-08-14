# Wave 1: Waypoint-Guidance Isolation

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer
> `IMP_J_LIMIT`” wording below refers only to the historical registered-task boundary,
> not a validated Z1 reaction-impulse or damage limit. The design body remains frozen;
> see `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

**Date:** 2026-08-02  
**Status:** owner-approved for implementation; final external review fixes incorporated
**Scope:** fixed-impedance Cartesian DiffIK trajectory shaping only

## Objective

Determine whether the hammer's direct fixed-reset descent is improved by a sparse
ordered-gate reward or by a dense, non-farmable waypoint-progress reward. This wave
does not tune impact reward and does not enforce velocity or impulse constraints.

## Experimental arms

Train fresh policies at one clean code and asset revision:

| Arm | Fixed gate-only pulse | New-best waypoint progress plus remainder settlement |
|---|---:|---:|
| C0 | absent | absent |
| G | weight 8 | absent |
| P | absent | weight 8 |

Each arm uses training seeds 2 and 3, 4096 environments, one GPU per run, and 200
iterations. All arms use the same Cartesian DiffIK action, fixed PD gains, fixed
reset, base reward terms, observations, six-waypoint geometry, timing, impulse caps,
and PPO configuration.

The existing C0/G seeds 0 and 1 remain historical diagnostic evidence. They are not
substituted for the fresh matched controls.

## Reference geometry

At reset, freeze the finite straight segment from the hammer-head entry position to
the nail target. Six intermediate waypoint gates lie at fractions 1/7 through 6/7.
Their existing swept-plane crossing detector runs at 500 Hz and keeps its 15 mm disk
radius for this wave. The six intermediate gates are also the six progress targets;
the base task rewards own the final gate-to-contact segment.

The radius is not changed in Wave 1; reward structure must be the only treatment
variable. A later experiment may tighten the winning recipe to 5 mm.

## Dense waypoint-progress reward

The P arm uses the existing ordered waypoint state but receives no gate-crossing
pulse. For the active target, the tracker stores:

- the distance when that target becomes active, `d_start`;
- the largest normalized approach fraction credited so far, `f_best`.

Both values are exposed to the actor and critic in C0, G, and P as `f_best` and
`d_start / reference_length`, each finite and bounded in `[0, 1]`. Before tracker
initialization, the observation previews the first target from the live head/nail
geometry with `f_best=0`; after all targets complete, both values are zero. This keeps
the history-dependent reward state observable and observation structure identical
across arms.

At each 500 Hz physics substep before first accepted contact:

1. Compute the current distance `d` from the hammer head to the active target.
2. Compute `f = clip((d_start - d) / max(d_start, epsilon), 0, 1)`.
3. Credit only `max(f - f_best, 0)` and update `f_best = max(f_best, f)`.
4. Hovering, moving backward, or re-covering old ground pays zero.
5. When the active intermediate gate is validly crossed, credit only its remaining
   uncredited potential `1 - f_best`, then activate the next target and reset that
   target's `d_start` and `f_best`.

Each target receives at most `1/6` raw progress credit:

```text
target_credit = (1/6) * max(f - f_best, 0)
```

The center-distance denominator spreads dense credit across the full waypoint
interval instead of exhausting it upon entering a gate-sized sphere. The remainder
settlement means a valid swept crossing earns exactly the still-unpaid part rather
than a second gate bonus. Six completed targets therefore have the same maximum raw
total of one as the existing gate reward. Initialization earns zero. The tracker owns separate
`window_new_credit` and `episode_credit` state: the former is cleared at every control
window and is the only value exposed to the 50 Hz reward reader; the latter is used
only for diagnostics and the episode cap. Per-environment reset clears both. This
prevents old progress from being paid repeatedly.

If one 2 ms swept segment crosses `k > 1` gates, P settles the uncredited remainder
for the target active at substep start and credits exactly `1/6` for each additional
crossed target. This matches G's `k/6` completion budget and advances to the same next
target; multi-cross count is logged. Reference qualification must still observe zero
multi-gate crossings. The term is zero after first accepted contact. It does not
terminate the episode and has no negative penalty outside a corridor.

This definition gives G and P the same maximum raw shaping budget: one. Both use
weight 8, so reward-manager dt scaling caps either treatment's actual episode return
at `8 * 0.02 = 0.16`. The progress state is computed in C0, G, and P; only P registers
the reward reader. This keeps treatment differences confined to reward payout.

The existing G implementation is part of this contract: `newly_crossed` accumulates
every crossing across all ten 500 Hz substeps in the current control window, the
50 Hz reader returns `newly_crossed / 6`, and the accumulator clears only at the next
window. Thus two crossings in one control window cannot be dropped. A regression test
must show that all six crossings pay raw total `1.0` (actual return `0.16` at weight 8)
regardless of whether two crossings occur in one window; the analogous P test must
establish the same cap. Reference qualification must also prove that every target's
`d_start > epsilon`. It must not require non-overlapping 15 mm gate disks: the frozen
production geometry has approximately 21.1 mm center spacing, and disk overlap does
not alter ordered active-target logic.

## Constraint state

All arms retain the existing per-joint impulse measurement and qvel logging:

- manufacturer `IMP_J_LIMIT` values stay unchanged;
- `imp_max_p=0.0`, so impulse CaT is log-only;
- velocity CaT and deterministic velocity termination are disabled;
- `set_gains` and variable impedance are forbidden in this wave.

This isolation is necessary: changing trajectory shaping and constraint enforcement
at the same time would make failure attribution impossible.

## Qualification before Vega

Use focused TDD for progress state, reset behavior, target switching, normalization,
contact censoring, and arm isolation. Then run the repository's mandatory reward and
impulse tests, `validate_rewards.py` A-M, `verify_contact_sensor.py`,
`verify_reward_setup.py`, and the reference-playback gate. Run one short CUDA smoke
for P and require finite tensors, positive progress dose, live Lambda measurement,
and the exact registered task identity. The same smoke must exercise C0's always-on
tracker and require finite tracker state even though C0 has no progress reward reader.

No new manifest framework or broad evaluator rewrite is part of Wave 1.

## Evidence after training

For every iteration-200 checkpoint produce:

- one identical-reset slow-motion MP4 with the same camera, duration, and playback
  rate; capture every two 500 Hz substeps (250 rendered frames/s of simulation) and
  encode at 50 fps;
- one 500 Hz x-z/x-y trajectory plot with the dashed straight reference, waypoint
  disks, control boundaries, contact, and normalized time;
- one compact metrics row containing code revision, asset revision, success, total
  episode return, actual gate/progress return, completed
  waypoints, q90 pre-contact reference error, 5 mm occupancy, path-length ratio,
  backward travel, time to contact, useful contact speed, delivered impulse,
  worst per-joint Lambda/cap, and peak qvel.

The dashed line and waypoint centers must come from the frozen guideline tracker's
`entry -> nail` geometry, not `SingleStrikeReference`, whose endpoint is different.
Simulator site positions are ground truth; do not build a computer-vision path
estimator. The deterministic fixed-reset figure reports its q90 as a within-episode
500 Hz statistic. Any seed-level q90 uses episode-first reduction over a separate
small sampled rollout and must not be conflated with the single video trace.

## Wave-1 decision rule

This is an exploratory two-seed screen, not a treatment-effect claim. An arm is
behaviorally promising only if both seeds:

- make genuine face contact and complete the nail under the identical fixed reset;
- have finite state and nonzero intended treatment dose;
- retain useful pre-contact speed of at least 1.50 m/s;
- avoid material hovering or time-to-contact regression; and
- improve geometry relative to fresh C0, judged jointly from q90 error, path-length
  ratio, backward travel, and the videos.

Do not select a recipe from one attractive seed. After all six 200-iteration policies
finish, stop for human and quantitative analysis. This checkpoint can promote an arm
to continued training, but cannot establish a final recipe; a promoted arm must be
continued or retrained to 500 iterations before that decision. Zero learned treatment
dose is preserved as a valid negative result, never filtered or replaced. GP,
recorded-reference tracking, hard waypoint guidance, the paper's joint-position
trackability term, impulse-reward
tuning, and every constraint-enforcement arm require a separate owner decision.

## Explicitly deferred

- GP (gate plus progress) Wave 2;
- recorded Cartesian reference-path tracking;
- hard missed-waypoint termination;
- the exact Bogdanovic-Khadiv-Righetti `r_tt`, which belongs to a held direct
  joint-position command and later VIC action space;
- velocity/impulse CaT enforcement;
- variable impedance and `set_gains`.
