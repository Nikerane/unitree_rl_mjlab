# Wave 1: Waypoint-Guidance Isolation

**Date:** 2026-08-02  
**Status:** written design reviewed; owner implementation approval pending  
**Scope:** fixed-impedance Cartesian DiffIK trajectory shaping only

## Objective

Determine whether the hammer's direct fixed-reset descent is improved by a sparse
ordered-gate reward or by a dense, non-farmable waypoint-progress reward. This wave
does not tune impact reward and does not enforce velocity or impulse constraints.

## Experimental arms

Train fresh policies at one clean code and asset revision:

| Arm | Ordered gate pulse | New-best waypoint progress |
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

At each 500 Hz physics substep before first accepted contact:

1. Compute the current distance `d` from the hammer head to the active target.
2. Compute `f = clip((d_start - d) / max(d_start - gate_radius, epsilon), 0, 1)`.
3. Credit only `max(f - f_best, 0)` and update `f_best = max(f_best, f)`.
4. Hovering, moving backward, or re-covering old ground pays zero.
5. When an intermediate gate is crossed, activate the next target and reset that
   target's `d_start` and `f_best`.

Each target receives at most `1/6` raw progress credit:

```text
target_credit = (1/6) * max(f - f_best, 0)
```

Because a swept gate crossing occurs at radial distance at most `gate_radius`, a
successfully reached target can earn its full `1/6` even without passing through its
exact center. Six completed targets therefore have the same maximum raw total of one
as the existing gate reward. Initialization earns zero. The tracker owns separate
`window_new_credit` and `episode_credit` state: the former is cleared at every control
window and is the only value exposed to the 50 Hz reward reader; the latter is used
only for diagnostics and the episode cap. Per-environment reset clears both. This
prevents old progress from being paid repeatedly.

If one 2 ms swept segment crosses multiple gates, only the target that was active at
the beginning of that substep receives dense progress credit; skipped-target count is
logged. Reference qualification must observe zero multi-gate crossings. The term is
zero after first accepted contact. It does not terminate the episode and has no
negative penalty outside a corridor.

This definition gives G and P the same maximum raw shaping budget: one. Both use
weight 8, so reward-manager dt scaling caps either treatment's actual episode return
at `8 * 0.02 = 0.16`. The progress state is computed in C0, G, and P; only P registers
the reward reader. This keeps treatment differences confined to reward payout.

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
and the exact registered task identity.

No new manifest framework or broad evaluator rewrite is part of Wave 1.

## Evidence after training

For every iteration-200 checkpoint produce:

- one identical-reset slow-motion MP4 with the same camera, duration, and playback
  rate; capture at every one or two 500 Hz substeps and encode at 50 fps;
- one 500 Hz x-z/x-y trajectory plot with the dashed straight reference, waypoint
  disks, control boundaries, contact, and normalized time;
- one compact metrics row containing success, actual gate/progress return, completed
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
