# Addendum to the one-pager — what should the impulse constraint bound, and what can VIC do about it?

> Companion to `2026-07-12_khadiv_onepager.md`. **Revised 2026-07-13** after a Codex adversarial
> review + an empirical verification pass falsified two claims in the first version (git history
> has v1; corrections marked inline). Adds the pivotal **decision (e)** — the Λ-quantity choice —
> with a decision plan. CPU/analytic, HEAD `soft-cat`. Full detail:
> `2026-07-12_state_of_everything.md` §9–§10.

## What we did

Before committing to build the variable-impedance (VIC) action space, we ran an analytic
**feasibility ceiling** (simulator mass matrix + head Jacobian at the strike pose:
`Λ_j = |Jᵀu|·m_eff·v·(1+e)`, `m_eff = 1/(uᵀ J M⁻¹ Jᵀ u)`, Khatib operational space), then an
adversarial review forced us to distinguish two quantities we had been conflating — and to
re-measure the second one:

1. the **ballistic impact impulse** (the momentum exchanged in the collision itself), and
2. the **enforced Λ** — what the shipped constraint actually reads: contact-masked
   `Σ|qfrc_constraint|·dt` over a sliding ~50 ms window, which includes **actively-driven
   reaction** (pressing) during contact, not just the collision.

## The two verified findings

**Finding 1 — VIC has ~zero lever on the *ballistic* impulse (v1 finding, survives verification).**
Reflected mass lives in `M(q)` + joint `armature` (physical rotor inertia); the commanded PD
stiffness `kp` never enters `M(q)`. The arm's reflected-mass range is only **1.33×** as modeled
(0.40–0.54 kg), and impact velocity is effort-clamped (~1.35 m/s at the head; the ballistic
crossing velocity of ~3.6–7 m/s is unreachable while driving into contact). For the brief
collision itself, impulse, energy (½·m_eff·v²) and peak force are all fixed by reflected mass ×
velocity — commanded stiffness shapes none of them.

**Finding 2 — the *enforced* Λ is NOT vacuous, and it binds today (new, corrects v1).**
Re-measured on the shipped accumulator: a fixed-impedance drive-through strike against a rigid
(bottomed-out) target reads **1.12–1.17× cap at every stiffness from 0.5× to 20×** — the ~50 ms
window of clamp-level press reaction (≈0.91× cap by itself) plus the impact spike crosses the
cap. v1's "a rigid target reaches ~0.6× and does not bind" described the *impact-gated
diagnostic*, not the deployed metric. Two corollaries:
- The "constraint is vacuous" headline is **scoped to ballistic impacts only**. Against
  press-through on a non-yielding target, the shipped constraint is live and binding — arguably
  *correct* gearbox semantics (50 ms of clamp-level reaction **is** repeated-peak load, the same
  τ_rated × window formula that produced the caps).
- **VIC's honest authority over the enforced Λ is downward**: stiffening saturates at the effort
  clamp (measured: Λ flat across 40× kp), but *below* saturation lower stiffness → lower contact
  force → lower windowed Λ. The defensible VIC story on the Z1 is *"comply to keep the enforced
  windowed Λ under cap"* — not "stiffen to unlock binding" (refuted by Finding 1), and not "VIC
  has zero lever on the constraint" (refuted by Finding 2).

*(Also fixed in the same pass: the shipped accumulator had a masking blind spot — a gentle 50 ms
touch could disarm it before a force spike; found by adversarial review, verified (the spike
registered exactly zero), and closed with a time-based sliding window that bounds any 50 ms
interval. All gates re-green.)*

## One caveat we cannot resolve without you / hardware data

The simulator's arm `armature` values (0.01–0.02 kg·m²) look like **nominal placeholders**, not
rotor inertia × gear² (harmonic drive → plausibly ~0.05–2). At 10–30× larger, reflected mass
rises to 0.89–1.39 kg and the ballistic reachable-speed Λ/cap from 0.38 toward ~0.6–1.0 — so the
ballistic-vacuity *margin* is model-sensitive (the structural Finding 1 is not).
**Please help us verify the Z1 rotor inertia before either number is treated as final.**

## The pivotal decision (e): what should Λ bound?

This is now the load-bearing choice — it decides what the constraint means, whether it can bind,
and what VIC optimizes against. Three coherent options (full trade-offs in the decision plan we
will walk you through):

1. **Ballistic impact impulse** — the collision's momentum exchange only (press excluded by a
   force-gated impact window). Cleanest match to "impact-safe" and to the cap's Δt derivation;
   but *provably vacuous on this platform* (Finding 1) and needs a watertight impact/press
   separator (the naive ones were exploitable).
2. **Windowed reaction impulse** (what now ships, log-only): any-50 ms `Σ|qfrc|·dt`, press
   included. Hardware-faithful repeated-peak reading, non-vacuous (binds on press-through), gives
   VIC a real downward role (comply) — but it binds on *pressing*, not *striking* (the thesis
   headline says "impact"), may fire on the successful nail-bottoming strike (a training-shaping
   risk to probe before C3), and for presses it reduces to average-torque × window ("why not just
   bound torque?" must be answerable).
3. **Split constraints** — a ballistic impact-impulse bound *plus* a separate sustained
   reaction-torque bound. Cleanest engineering semantics, each cap on its native quantity; costs
   a second constraint (tuning, normalizers) and needs the same impact/press separator as (1).

**Our lean:** (2) as shipped default — it is the only option that is simultaneously enforceable,
non-vacuous, and exploit-closed today — with (3) as the principled upgrade if you want the thesis
to keep a *ballistic* claim distinct from the press bound. We have deliberately NOT rewritten the
thesis framing around any option: the machinery ships log-only (`imp_max_p = 0`) until you choose.

## Decisions for you (consolidated)

- **(a)** Bless the reframe: fixed-impedance = compliance/negative-control **for ballistic
  impacts**; the binding/protective story = VIC-compliance against the enforced windowed Λ
  (per Finding 2), pending (e).
- **(b)** Keep the per-joint caps fixed at hardware values — unchanged, still our position.
- **(c)** *Superseded (corrects v1):* the rigid-target control is **done and it binds the shipped
  metric** (1.12–1.17×); no further control needed.
- **(d)** The sustained-press residual is no longer a residual — it is the binding pathway of
  option (2); subsumed by decision (e).
- **(e)** **Choose the bounded quantity: ballistic / windowed reaction / split** (above).
- **(f)** Verify the Z1 rotor inertia (armature) so the ballistic margins are trustworthy.
