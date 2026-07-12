# One-pager for Prof. Khadiv — is the impulse constraint doing anything on the fixed-impedance Z1?

> Branch `soft-cat`, HEAD `481a26f`, CPU-only investigation (2026-07-12). Full record:
> [`2026-07-12_impulse_vacuity.md`](2026-07-12_impulse_vacuity.md).

## Finding

Before spending a GPU training run to demonstrate the per-joint impact-impulse constraint, we
tested — on CPU, with an adversarial forensics workflow (4 independent checks converging on the
same answer) — whether it can ever actually bind on the current fixed-impedance Z1 hammer task.
**It cannot, for genuine impacts.** Every strike the arm can actually reach — from the natural
reference trajectory up to its fastest observed joint velocity — stays at most 36% of the
tightest per-joint cap. A more extreme kinematic check (coordinating all six joints simultaneously
at that same velocity, beyond what the arm's action space can actually command) reaches 39%, still
under cap; the most generous analytic ceiling (every load joint at the arm's rail velocity, a full
inelastic stop) reaches 90% and is shown only for context, since it too is kinematically
unreachable. The physical reason: the target nail is soft, light (7 g), and yields under the
strike rather than reflecting the hammer's momentum back into the arm's gearboxes, and the arm's
own effort limit caps how fast it can be moving at the moment of contact (~1.3 m/s at the head,
far below the rail velocities the ceiling checks assume). The machinery itself — the substep
accumulators, contact-anchoring, friction-immunity, and the CaT δ-enforcement chain — is validated
and confirmed to enforce correctly (it fires exactly when an artificially over-cap event is
injected, and stays silent under a log-only control). There is one narrow, reward-gated exception
(below) that only a learned-policy GPU run can fully close.

## Proposed reframe (pending your review — not yet adopted in the thesis)

Treat the fixed-impedance Z1 result as a **compliance / negative-control baseline**: proof the
constraint machinery is correctly built and satisfied with headroom, and evidence that the
fixed-impedance controller is impulse-safe against genuine impacts on this yielding nail at its
effort-limited reachable velocity (item (d) below is the one open exception). Move the **binding,
protective** story — where the constraint actually shapes the policy — to **variable impedance**
(the thesis's headline contribution), where raising effective stiffness/mass at controlled moments
raises deliverable momentum against the *same* caps. Arc: fixed impedance cannot reach a dangerous
impulse (against genuine impacts) → variable impedance can → the constraint becomes load-bearing
exactly when VIC makes it necessary.

## Open decisions for you

- **(a) Bless the reframe above**, or redirect if you see it differently — this changes what the
  fixed-impedance GPU run (C3) is claimed to demonstrate (headroom/negative-control, not active
  protection).
- **(b) Confirm keeping the per-joint caps fixed** at their principled hardware values
  (rated torque × 2 for harmonic-drive repeated-peak × the measured contact window) rather than
  tightening them to force a binding result on this task. The team has already decided this
  (tightening would turn a hardware-derived safety limit into an arbitrary tuning knob); flagging
  for your sign-off.
- **(c) Optional: a rigid/heavier-target sanity control?** Our sweep predicts a sufficiently heavy
  or rigid nail *would* bind (~1.2–1.8× cap via elastic reflection), but only if struck at the
  rail joint velocities used in our analytic ceiling — velocities the fixed-impedance controller
  cannot reach while actually driving into contact (its effort-limited approach speed tops out at
  ~1.3 m/s at the head). We could add this as a small illustrative control (via direct velocity
  injection, not a scripted strike) if you'd like one demonstrated fixed-impedance binding case in
  the thesis; it is not required for the reframe to hold.
- **(d) Accept the one open residual as reward-gated and C3-settled?** The accumulator the
  constraint reads has no cap on how long it integrates a sustained press (unlike the reward-side
  accumulator, which does). A policy that presses the nail rather than striking it could in
  principle cross the cap without any high velocity. This is gated by the reward (pressing past
  task completion doesn't pay), untested outside scripted probes, and will be settled by the C3
  learned-policy run; we'll also close the underlying code gap (a small, CPU-only fix) regardless
  of the answer.

Full physics, the four supporting figures, and the artifact-by-artifact breakdown of every
apparent "binding" result (each traced to a sustained press or a super-hardware joint velocity)
are in the linked record.
