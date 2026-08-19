# Z1 two-boundary impulse-CaT behavior diagnostic design

**Status:** Approved for execution by the user on 2026-08-19.

## Question

With delivered-impulse maximization and velocity CaT unchanged, how does the implemented
impulse-CaT learner behave at `imp_max_p=0.2` under:

1. a uniformly `0.9`-scaled diagnostic boundary, where J3 is expected to be the main limiting
   joint; and
2. deliberately tight joint-specific boundaries, where several joint channels are expected to be
   active?

This is an exploratory mechanism and behavior diagnostic. It observes whether the trained policies
remain below, approach, exceed, or redistribute load around their assigned boundaries. It does not
predeclare a numerical success verdict, estimate actuator breakage, establish manufacturer or
hardware limits, prove hard enforcement, or establish reliability across training seeds.

## Treatments

Train exactly two new seed-2 VIC-TT policies:

| Role | `imp_max_p` | Caps (`N.m.s`) | Purpose |
|---|---:|---|---|
| `p02_uniform09` | `0.2` | `[0.738,1.476,0.738,0.738,0.738,0.738]` | Same dose at the historical uniform-0.9 boundary |
| `p02_joint_stress` | `0.2` | `[0.369,0.246,0.738,0.369,0.246,0.0164]` | Same dose with deliberately tight joint-specific boundaries |

Both arms use the exact same native VIC-TT task, seed `2`, `4096` environments, `24` rollout
steps, `500` PPO iterations, and save interval `50`.

The only treatment difference is the impulse cap vector. Both arms retain:

- velocity CaT active with its existing configuration;
- impulse CaT active with `imp_max_p=0.2`;
- delivered-impulse maximization at weight `4.0`;
- impact-progress reward at weight `8.0`;
- the same observations, actions, physics, reference trajectory, VIC gain range, rewards,
  trajectory generation, and existing training distribution.

No curriculum or domain-randomization change belongs to this experiment. Those are deferred until
after the result is analyzed.

## Existing zero-learning context

Offline replay of the banked log-only control predicts that the joint-specific stress vector is
dense before learning: `98.75–99.12%` of stochastic initial episodes violate at least one boundary,
J2 supplies `75.7–79.7%` of responsible reads, J6 supplies `13.7–14.5%`, and impulse wins the outer
max on `99.68–99.85%` of active reads.

These are starting-condition descriptors, not stop gates. Dense activation is deliberate here: the
experiment asks how the learner responds to it.

## Technical pre-training gates

Before launch, require only technical and identity gates:

1. Exact arm-to-vector mapping and exact common `imp_max_p=0.2`.
2. Velocity CaT, delivered-impulse reward weight `4.0`, impact-progress weight `8.0`, and the native
   VIC-TT task are identical across arms.
3. `delta_impulse` is finite and combined delta is exactly
   `max(delta_velocity, delta_impulse)` in a real-path smoke.
4. The strict joint-specific vector activates multiple joint channels in offline replay; its high
   activation rate or J2 dominance is reported but does not block training.
5. All AGENTS.md reward, contact-sensor, reward-setup, impulse, CatSoftHook, launcher, and real VIC
   smoke gates pass at the exact launch revision.
6. Vega code/assets/runtime/GPU provenance is exact and clean.

## Evaluation

Evaluate both new policies frozen with live `imp_max_p=0`, so evaluation itself cannot change
actions. Use the fixed 64-world population plus the matched stochastic populations at seeds `2`,
`2026081701`, and `2026081702`.

For each arm, analyze raw Lambda against:

- its own training cap vector;
- the other arm's cap vector; and
- the provisional project caps `[0.82,1.64,0.82,0.82,0.82,0.82] N.m.s`.

Compare raw per-joint Lambda—not utilization alone—across arms because their denominators differ.
The older full-cap `p=0.2` and uniform-0.9 `p=0.5` policies may be cited as descriptive historical
context, but they are not new control arms and cannot rescue or override the two-policy result.

Report:

- per-joint Lambda p50/p95/p99/max, utilization, positive margin, violating reads, segments, and
  activation windows;
- responsible-joint, all-active-joint, co-violation, winner-switching, and masked-active shares;
- `delta_velocity`, counterfactual `delta_impulse` at `p=0.2`, combined delta, outer winner, dose
  ceiling hits, reads per activation window, and window pressure;
- true 500 Hz joint-velocity risk/tails;
- delivered first-event impulse, impact progress, nail depth, productive strikes, success, episode
  duration, VIC gains, and actions.

Classify the observed behavior descriptively as one or more of: boundary compliance, partial
compliance, bottleneck transfer, distributed load, global suppression, task-output trade-off,
velocity-risk transfer, or no meaningful response. Every finite, provenance-correct outcome is an
experiment result.

## Stop boundary

Submit the two-arm training array once and the frozen evaluation once. Never retry, requeue,
extend, resume, or automatically tune another vector or dose. After banking the result, stop before
curriculum learning, domain-randomization changes, additional training seeds, torque constraints,
contact flushing, or hardware transfer unless the user explicitly approves the next experiment.

## Claim boundary

The result may describe how these two one-seed simulated policies respond to two diagnostic
boundary geometries at the same CaT dose. It cannot establish hardware safety, actuator damage
limits, hard enforcement, optimal thresholds, optimal `p`, individual joint causality, or
cross-training-seed reliability.
