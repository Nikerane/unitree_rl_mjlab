# Z1 VIC J1/J6 nominal-clamp diagnostic

## Material Passport

- Origin: `academic-research-suite / experiment-agent / validate`
- Verification status: `ANALYZED`
- Evidence level: one deterministic checkpoint and one fixed reset
- Status: **preliminary and parked**
- Decision: no controller, reward, gain-range, or training change follows from
  this diagnostic

## Why this diagnostic was run

The [seed-2 VIC canary](2026-08-14_z1_vic_seed2_canary.md) commanded lower-than-nominal
gains for J1 and J6 at first contact. An offline Jacobian calculation indicated
that neither joint contributes materially to small-signal stiffness along the
vertical nail axis at that posture. The unresolved question was therefore:

> Does restoring J1 or J6 to nominal impedance change the strike behavior at
> all, even if those joints do not directly drive the hammer downward?

## Minimal intervention

Four deterministic CPU rollouts used the same checkpoint, fixed reset, initial
observation, and first raw actor action:

1. native VIC;
2. J1 stiffness command clamped to nominal;
3. J6 stiffness command clamped to nominal;
4. both J1 and J6 clamped to nominal.

The clamp was applied for the **whole rollout**, after policy inference and
before the environment step. For J1/J6, this changed `Kp/Kd` from the native
first-contact value `800/89.44` to the nominal value `1000/100`. It did not
change the position command or any other stiffness coordinate at the moment of
intervention. Because the policy continued to run closed loop, later actions
could change in response to the altered trajectory.

## Observed result

All four conditions produced a productive strike and terminated successfully
at control step 8.

| Condition | Axial impulse (N s) | Change | Final depth (mm) | Change | In-plane x offset at contact (mm) | Out-of-plane y offset at contact (mm) |
|---|---:|---:|---:|---:|---:|---:|
| Native VIC | 0.454269 | — | 51.072 | — | +9.910 | -6.284 |
| J1 nominal | 0.445082 | -2.02% | 50.172 | -0.900 | +9.881 | -10.665 |
| J6 nominal | 0.454513 | +0.05% | 51.086 | +0.014 | +9.916 | -6.326 |
| J1 + J6 nominal | 0.445433 | -1.95% | 50.209 | -0.862 | +9.887 | -10.705 |

The intended motion is in the world XZ plane. The table therefore reports x
and y separately. The earlier combined horizontal norm must **not** be called a
“lateral strike”: its increase under the J1 clamp came almost entirely from a
more negative out-of-plane y displacement, while the in-plane x displacement
was effectively unchanged.

J1 also spent 100% of the native rollout at its actuator-force rail, compared
with 53.75% when clamped to nominal. J6 never reached its force rail. All four
conditions remained below the configured joint-speed and joint-impulse limits.

## What this establishes

- Restoring J6 to nominal impedance was practically neutral in this one
  replay. The combined intervention closely followed the J1-only result.
- Restoring J1 to nominal impedance changed the closed-loop rollout: axial
  impulse fell by about 2%, final nail depth fell by 0.9 mm, and the hammer-site
  proxy moved farther out of the intended XZ plane.
- J1 is therefore not a perfectly inert policy coordinate in this replay.

## What remains unresolved

This diagnostic does **not** explain why the policy selected low J1 gain, and
it does not justify calling the behavior purposeful impact compliance:

- the intervention lasted for the whole rollout, so it mixes the direct gain
  change with later closed-loop compensation;
- J1 actuator saturation makes a simple “higher gain means a harder strike”
  interpretation invalid;
- the x/y values are the hammer-site-to-nail-top displacement, not a validated
  physical contact-point error;
- the task's contact-quality channel was invalid in every arm and its zero
  placeholders were not used;
- one checkpoint and one reset cannot establish a repeatable learned strategy;
- J6 may be weakly identified or reward-insensitive rather than deliberately
  softened.

The defensible conclusion is therefore narrow:

> J1 gain affects the complete closed-loop strike in this fixed replay, but the
> mechanism and learned purpose remain unresolved. J6 gain was locally
> insensitive. Neither finding is yet a thesis-level behavioral claim.

## Parking decision

This side investigation stops here. No gain, controller, reward, or training
configuration will be changed from this result. Revisit it only when broader
VIC evidence exists—preferably multiple resets and training seeds, a valid
contact-point measurement, and, if still useful, a phase-local or frozen-state
J1 intervention that separates the direct mechanical effect from policy
feedback.

## Provenance

- Code revision: `ca5e83bc84bc979bac857088a9330f2a2bd8034b`
- Asset revision: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`
- Checkpoint SHA-256:
  `c1544b779e78e7323bf02ce7b0f165745ee63b5aad9f930f9eea64eb6ea4ea77`
- Fixed-reset SHA-256:
  `7a8093ab5142d5887d9dc86ba7504e53ed600e9d67063f7acac06750724a4948`
- Result JSON SHA-256:
  `5782c5219545c3608dfd124bfbd2486e3518e64ca4ce3e63076d1aa67978b8cc`
- Successful run log SHA-256:
  `0445b396dc77cc42f1f1208ebc8d8634acb2b519852b958c8cc951cb18f60ed5`
- Diagnostic script SHA-256:
  `f0ac6be24ee507dc0cf8a93823213a7f61aa392947d4217f86660e06519d0dee`

The full result, script, and logs remain outside the repository in
`/private/tmp/z1-vic-j1j6-ablation/`. This note is the intentionally compact
parked record; the temporary bundle should be banked separately only if this
question is reopened.
