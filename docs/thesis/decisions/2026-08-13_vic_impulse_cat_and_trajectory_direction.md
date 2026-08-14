# Supervisor direction — active joint-impulse CaT for VIC, then trajectory comparison

**Date:** 2026-08-13

**Source:** User's record of discussion with Prof. Khadiv

**Status:** Direction recorded; it does not by itself authorize a new implementation or GPU campaign.

## Decision now

The next variable-impedance experiments should use **active per-joint impulse soft-CaT** together
with the already active **velocity soft-CaT**. Joint impulse should no longer remain merely
**log-only** in the intended constrained VIC treatment.

This defines a paired experiment, not a rule that every run must enforce both constraints:

- **target treatment:** velocity soft-CaT active + per-joint impulse soft-CaT active;
- **control treatment:** velocity soft-CaT active + per-joint impulse measurement log-only.

All other task, controller, reset, learning, and evaluation settings should be matched. Keeping the
control treatment is necessary both for calibration and for attributing any observed change to
impulse enforcement.

This is a coupled experiment-policy default for future VIC work, not a reinterpretation of the
completed seed-2 canary. That canary remains the matched diagnostic baseline:

- velocity soft-CaT was active;
- per-joint impulse CaT was log-only;
- its results therefore do not demonstrate impulse-CaT enforcement.

Use the precise term *active per-joint impulse soft-CaT*, not *impulse saturation*. The method
changes the CaT continuation/learning signal when the measured joint-impulse margin is violated;
it does not physically clip or saturate the impact impulse.

## Immediate scientific question

Under the same VIC controller and task, does simultaneous velocity and joint-impulse CaT:

1. retain productive hammering and task success;
2. reduce per-joint impulse-cap violations or utilization tails;
3. preserve acceptable joint-velocity behavior; and
4. change delivered hammer-to-nail impulse or the learned joint-stiffness schedule?

The comparison must preserve the current velocity-active/impulse-log-only VIC result as the
historical control evidence. A formal paired campaign may still require a contemporaneous control
run at the same final code revision. It must change only the impulse-CaT enforcement treatment,
subject to a separate approved experiment design and launch plan.

## Later direction — trajectory sources

After the combined-constraint VIC treatment is understood, consider multiple reference
**trajectory source** families. A learned generator such as **diffusion** is one candidate,
alongside designed or otherwise generated trajectories. The purpose is to **compare trajectory families**
under the same controller, constraints, task, reset population, training budget, and
evaluation protocol.

This note records a future source-of-reference study only. It is **not authorized** implementation
work, and it does not revive the previously rejected two-level **generate-then-track** architecture:
the thesis policy remains single-policy, variable-impedance, online RL. Any diffusion component
must first be specified as a trajectory-generation treatment and reviewed with the supervisor.

## Sequencing

1. Design and qualify active joint-impulse soft-CaT for VIC while velocity soft-CaT remains active.
2. Run a controlled comparison against the existing impulse-log-only VIC baseline.
3. Only after that mechanism is understood, define candidate trajectory families and a matched
   trajectory-comparison protocol.
4. Treat diffusion-generated references as one candidate treatment, not the default architecture.

## Still open before launch

- Exact active impulse-CaT probability/calibration and the bound/window contract.
- Stop/go gates for success, delivered impulse, joint-velocity tails, and per-joint impulse tails.
- Whether the comparison needs more VIC training seeds before any general claim.
- Which trajectory families to compare and how to match feasibility, duration, geometry, and
  training exposure.
