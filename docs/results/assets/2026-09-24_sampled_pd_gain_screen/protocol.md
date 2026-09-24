# Approved 500 Hz gain screen — 2026-09-24

Scope: diagnostic only, no production changes or learning. Seven fixed gain
settings: full stiffness [512,768,768,512,384,256] with damping25.6 (reference),
then full and half stiffness crossed with damping12.8,6.4,3.2.
These are simulation candidates, not manufacturer-certified gains.

Both prior128-world command tapes, first60ms only; targets50Hz, sampled/held
torque500Hz, physics0.25/0.125ms. Gripper gains and all effort/speed limits
unchanged. Exactly30 torque updates per replay. Any hammer/nail contact
invalidates the no-contact interpretation for that replay.

Record joint speed, saturation, canonical joint target error, and longest
consecutive saturated sign-reversal run per world/joint. Define sustained
chatter as at least3 consecutive saturated reversals (four torque ticks).
Unsaturated reversals and isolated target-change reversals do not qualify.

Compare paired peak-speed differences at the two physics resolutions:
report per-joint p95 and max; screening budgets0.05 and0.10 times3.1415rad/s.
Also report physical position/speed RMS differences at matched poststep times.
Numerical agreement and absence of sustained chatter define a startup candidate,
not impact capability or policy/hardware safety. Record speed breaches separately.
Do not rank gains solely by suppressing all motion; examine target error too.

Plan: test helpers RED→GREEN; adapt preserved replay in this new directory;
run seven settings on Vega; retain raw traces and summarize all28 replays.
Old experiment files remain unchanged. Full-suite baseline has known stale
docs-date assertion; report rather than changing unrelated docs/tests.
