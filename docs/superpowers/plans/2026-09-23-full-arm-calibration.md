# Full-arm controller calibration and measurement check

Approved in conversation on 2026-09-23. Preserve production code, checkpoint
semantics, effort limits and the 3.1415 rad/s six-joint criterion.

## Task 1: Small fixed-gain transfer screen

Use the existing two seed-2 dose checkpoints, 128 first episodes at each of
1.5/1.85/2.2 m/s, native 2 ms physics and 20 ms control. Override all six
postprocessed gains, not position targets. Candidates:

- SDK-reference: Kp [512,768,768,512,384,256], Kd [25.6]*6.
- Same vector with J6 Kp 128, Kd unchanged.

These are experimental candidates, not hardware-qualified settings. Test
validation, joint order, all-world override and preservation of native
processing. Keep effective actuator-gain checks in the runtime wrapper.
Report utility, all-joint speed and current load-proxy statistics separately.
No learning in this screen; transfer failure is not proof fresh RL cannot learn.

## Task 2: Numerical and load-source comparisons

Reuse frozen-command replay tools. Compare selected candidate at native and
finer steps with unchanged 20 ms control and 50 ms physical integration window.
Measure startup chatter/saturation and physical trace sensitivity separately
from safety-limit compliance. Reuse raw contact-row helper, validate geometry
and reconstruction, and compare its absolute projected generalized force in
the same 50 ms sliding window as the baseline-subtracted quantity. Do not use
the existing uncapped contact-event scalar as though it were the same measure.
Do not call either quantity a measured gearbox load or manufacturer damage cap.

## Task 3: Conditional fresh fixed-gain pilot

After a numerically credible setup is identified, run required reward/contact
and software gates, then a short fresh fixed-gain pilot with speed CaT active,
existing reference and impact objective. Keep unresolved joint-load quantities
log-only. Fixed-gain pilot needs its own explicit controller/checkpoint identity.
Select the exact pilot budget before launch; do not silently reuse a VIC
checkpoint as fixed-gain training. Matched impulse off/on and VIC follow later.

## Review focus

Native gains must really be overridden each apply; no joint-order swaps,
altered joint targets or checkpoint metadata. Preserve effort caps and source
identity. Separate posttraining transfer from learning and numerical stability
from compliance. No new broad infrastructure or production reward changes.
