# Z1 controlled-drop hammer-poll-height correction

**Status:** Approved by the owner on 2026-08-12.

## Purpose

Correct the controlled-drop striker before its impulse is used to normalize the
FIC delivered-impulse reward. The original `0.200 kg` cylinder was only `8 mm`
tall because it copied the production nail-head primitive. It was a contact
coupon, not a hammer-head-sized striker.

A read-only CPU comparison found that this short coupon loses force-bearing
contact after the initial penetration and passes through the soft-contact nail
geometry. The existing banked value, `0.10635668784379959 N s`, therefore must
not be propagated into FIC.

## Approved geometry

The corrected cylinder matches the axial thickness of the production
`hammer_head_0` striking-poll collision mesh (`claw_hammer_c4`):

- total axial height: `0.01756416 m`;
- half-height used by MuJoCo: `0.00878208 m`;
- radius: `0.012 m` (`0.024 m` diameter), unchanged;
- mass: `0.200 kg`, unchanged and set explicitly;
- lower-face-to-nail-head clearance: exactly `0.150 m`, unchanged;
- release velocity: zero;
- motion: one frictionless, undamped vertical slide with no rotation;
- contact properties, production nail, solver, tracker, timestep, event window,
  and five-release protocol: unchanged.

The axial dimension is the poll mesh's physical thickness along its face normal,
not its pose-dependent world-axis bounding-box projection. The radius remains
nail-matched so this correction does not also change the contact footprint.

## Alternatives rejected

1. **Match the full lateral hammer-poll scale.** A roughly `30 mm` diameter
   cylinder would also alter the contact footprint. That is a separate
   experiment and is not needed to satisfy the approved height correction.
2. **Add a taller visual shell around the original collision coupon.** This
   would preserve the old number but hide the contact-loss problem behind a
   visual/physics mismatch.
3. **Keep effective density fixed.** Increasing height at the old effective
   density would raise mass to about `0.439 kg`, changing the intended detached
   `0.200 kg` hammer surrogate and materially increasing incident momentum.

## Normalization and provenance

The live delivered-impulse reward is linear in `I_first / I_ref`; its numeric D4
weight remains `4.0`. Consequently, the corrected authoritative mean defines a
new treatment identity and must be propagated only after it is banked.

The original JSON and result document remain immutable historical evidence.
After the corrected run succeeds, the old result document receives a concise
supersession notice pointing to the new result. No historical Cartesian task or
result is rewritten.

## Public test seams

1. `make_controlled_drop_env_cfg()` must compile a `0.01756416 m`-tall,
   `0.012 m`-radius, `0.200 kg` cylinder while preserving exactly `0.150 m`
   lower-face clearance and all frozen production contact/solver settings.
2. `run_one_primary_drop(device="cpu", trial_index=1)` must produce a finite,
   productive finalized event that reaches the public `success` horizon. This
   public outcome guards against recurrence of the short-striker pass-through.
3. `run_primary_calibration(...)` and the CLI retain their existing fail-closed
   five-fresh-release and clean-revision contracts.

Tests will be added one vertical slice at a time: observe RED with the old
`8 mm` fixture, make the minimum geometry change, then observe GREEN. No exact
CPU impulse is frozen as a unit-test literal.

## Verification and authoritative execution

After focused CPU tests pass:

1. render release, first-contact, and event-finalization frames plus a montage;
2. visually confirm scale, centering, clearance, contact, and nail progress;
3. run the controlled-drop focused suite and then the full CPU suite once;
4. execute exactly five fresh releases from a clean pinned revision on Vega A100;
5. audit the result, arithmetic mean, versions, revisions, Slurm state, and log;
6. bank a new JSON/result document without overwriting the old evidence;
7. obtain independent code/spec/scientific reviews, fix concrete findings,
   commit, and push.

Only the new audited A100 arithmetic mean may become the FIC-only `I_ref`.

## Scope boundary

This correction does not change D4, FIC registrations, `k_tt`, RTT semantics,
CaT handling, joint-policy metadata, actuator gains, curriculum, domain
randomization, VIC, Cartesian tasks, or any owner-owned untracked file. FIC
training remains blocked until the corrected mean is banked and separately
propagated through the approved FIC integration seam.
