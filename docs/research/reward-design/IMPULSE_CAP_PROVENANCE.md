# Z1 per-joint impulse-threshold provenance

**Status (2026-08-14): current threshold interpretation for the Step-1 impulse-CaT qualification.**
This note supersedes earlier claims that the project vector
`[1.64, 3.28, 1.64, 1.64, 1.64, 1.64] N·m·s` is a manufacturer-certified,
hardware, breaking, or externally applied impact-impulse limit. It is none of
those things.

## What Unitree actually publishes

- Unitree's official [Z1 product page](https://www.unitree.com/z1/) lists a
  `33 N·m` **Maximum Torque** for the displayed actuator module, a harmonic
  reducer, and a `60+` reduction ratio. It also lists collision protection for
  the arm. It does **not** publish a per-joint allowable external reaction
  impulse, shock/damage threshold, breaking impulse, reducer repeated-peak
  rating, or permissible contact duration.
- Unitree's official [`z1_description` URDF](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/z1_description/xacro/z1.urdf)
  assigns model effort limits of `30 N·m` to J1 and J3--J6 and `60 N·m` to J2.
  Those values specify actuator effort limits in the robot model. They do not
  state how much reaction torque or impulse the physical actuator/reducer can
  absorb before damage.

Consequently, neither the product-page maximum torque nor the URDF effort
limits can be presented as a Z1 breaking-strength or impact-impulse limit.
Unitree would need to provide the exact actuator/reducer duty-cycle and shock
ratings, or the project would need a validated hardware test protocol, before
making that claim.

## How the old project vector was produced

The historical project calculation was

```text
tau_model = [30, 60, 30, 30, 30, 30] N·m
kappa     = 2
duration  = 0.027333... s
L_old     = tau_model * kappa * duration
          = [1.64, 3.28, 1.64, 1.64, 1.64, 1.64] N·m·s
```

This joined three quantities that did not support the resulting hardware
claim:

1. `tau_model` came from the URDF/simulator effort limits, not a published
   per-joint reducer shock rating.
2. `kappa=2` was imported as a generic Harmonic Drive repeated-peak heuristic
   without identifying the exact Z1 reducer model or showing that its catalog
   rating applies to these Unitree actuator limits. It also *raises* the
   threshold, which is the opposite of a conservative removal when that link is
   unsupported.
3. `27.333 ms` came from an earlier reference-strike protocol whose contact was
   still closed when recording ended. It is a right-censored observed contact
   prefix, not a completed event duration and not a measured mean contact
   duration. The implemented Lambda quantity is accumulated at `500 Hz` over a
   `25`-substep (`50 ms`) sliding window, so the old duration also does not
   describe the measurement window.

The `kappa=2` interpretation is therefore retired. Historical records keep the
old arithmetic and numbers so their results remain reproducible, but their
threshold-language is superseded by this note.

## Threshold identities during Step 1

| Identity | Per-joint vector (J1--J6), N·m·s | Meaning |
|---|---:|---|
| Banked registered-task boundary | `[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]` | Retained by existing task identities and banked evaluators for exact historical compatibility. Not a manufacturer or damage limit. |
| Step-1 provisional no-`2x` threshold | `[0.820, 1.640, 0.820, 0.820, 0.820, 0.820]` | User-directed, opt-in survey threshold formed by removing the unsupported factor from the old arithmetic. A provisional experiment boundary, not a hardware limit. |
| Uniform `0.9` diagnostic threshold | `[0.738, 1.476, 0.738, 0.738, 0.738, 0.738]` | Exactly `0.9` times the provisional vector for every joint. Used only to prove activation plumbing and attribution. Not a hardware limit or scientific treatment. |

The provisional and diagnostic vectors must be explicit survey/run overrides;
they must not silently replace the threshold in existing registered tasks or
reinterpret banked results. Lowering the provisional vector by `0.9` does not
discover a material limit: it deliberately creates a nearby diagnostic boundary
against recorded behavior.

The user later authorized one **short matched diagnostic canary** at the uniform
`0.9` boundary: control `imp_max_p=0.0` versus target `imp_max_p=0.5`, with
velocity CaT active and every other treatment held fixed. This authorization is
for engineering activation evidence only. It does not select `0.9` as a
scientific threshold, `0.5` as a calibrated event dose, or either value as a
hardware-safety setting. Because one `50 ms` violating window can be read more
than once by the `50 Hz` policy, the report must retain separate per-read deltas
and event pressure rather than treating `imp_max_p` as an event-level
probability.

## What remains open

- Measure completed contact-duration distributions with release observed; do
  not relabel right-censored prefixes as completed events or means.
- Ask the supervisor which quantity should define the thesis treatment while
  exact Unitree reducer/actuator shock data are unavailable.
- Obtain exact Unitree or component-supplier permissible external reaction,
  repeated-impact duty-cycle, and damage data before making a hardware-safety
  claim.
- Keep instantaneous actuator torque/effort and windowed joint-reaction impulse
  conceptually separate. A future torque constraint may complement the impulse
  constraint, but the published actuation limit still is not an external shock
  limit.
