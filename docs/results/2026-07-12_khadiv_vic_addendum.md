# Addendum to the one-pager — can variable impedance make the impulse constraint bind on the Z1?

> Companion to `2026-07-12_khadiv_onepager.md`. Adds one **new decision (e)** and one **new
> finding** that changes what the fixed-impedance → variable-impedance arc can claim. CPU/analytic,
> HEAD `soft-cat`. Full detail: `2026-07-12_state_of_everything.md` §9.

## What we did

Before committing to build the variable-impedance (VIC) action space, we ran a ½-day analytic
**feasibility ceiling**: using the simulator's own mass matrix and the head Jacobian at the strike
pose, we bounded the *maximum* per-joint reaction impulse Λ_j any strike could produce, and asked
whether VIC's stiffness command can move it. (`Λ_j = |Jᵀu|·m_eff·v·(1+e)`; effective mass
`m_eff = 1/(uᵀ J M⁻¹ Jᵀ u)`, Khatib operational space.)

## The finding (this is the load-bearing part)

**Variable impedance, as a per-joint *commanded stiffness* (`set_gains`), has essentially no lever
on the ballistic impact impulse on the Z1** — for a structural reason, not a tuning one:

- The impulse is set by **reflected mass × velocity**. Reflected mass lives in `M(q)` + joint
  `armature` (a *physical* rotor-inertia property); the commanded PD stiffness `kp` does **not**
  enter `M(q)`. So raising `kp` only adds an *active press* force during contact — it does not
  raise the passive reflected mass that governs the momentum exchange.
- Empirically consistent: across a 40× stiffness sweep against a rigid target, the genuine impact
  Λ/cap barely moved (~0.47→0.64), and the arm's reflected mass ranges only **1.33×** between
  fully-compliant and rigidly-coupled.
- Velocity — the *other* lever — is **effort-clamped** (~1.35 m/s at the head); `set_gains` cannot
  lift the motor torque limit. The cap is only reachable at ~3.6–7 m/s (a coordinated whip the
  effort-limited controller cannot produce).

**Consequence:** the arc we had been assuming — *"fixed impedance can't reach a binding impulse →
variable impedance unlocks it"* — **most likely does not hold on the Z1**, because VIC's stiffness
command does not change the quantity the impulse constraint bounds. This also answers a genuine
open question in the literature (does control-level impedance shape the *impulsive* peak, or only
quasi-static force?): on a rigid-transmission arm, **it shapes only the sustained/quasi-static
contact force — not the ballistic impact, whose impulse, energy (½·m_eff·v²) and peak force are all
fixed by the reflected mass and the effort-clamped velocity.**

## One important caveat we cannot resolve without you / hardware data

The simulator's arm `armature` values (0.01–0.02 kg·m²) look like **nominal placeholders**, not
values derived from the real harmonic-drive rotor inertia × gear² (which should be ~0.05–0.3). If
the true value is 10–30× larger, the reflected mass — and therefore **both** the "constraint is
slack" margin **and** how close a strike comes to the cap — would rise substantially (a strike
could be *near-binding* at fixed impedance on real hardware). This does not change the VIC verdict
(stiffness still can't command it), but it means our headline "vacuous" number is model-sensitive.
**We should verify the Z1 rotor inertia before finalizing either claim.**

## Decisions for you

Carrying over from the one-pager: **(a)** bless the compliance/negative-control reframe, **(b)**
keep the caps fixed at hardware values, **(c)** — *now answered*: a rigid target reaches ~0.6× cap
and still does not bind, so no separate rigid-target control is needed, **(d)** accept the
sustained-press residual as reward-gated + C3-settled. **New:**

- **(e) What quantity should the constraint bound — impulse, or peak-force / energy?** This is now
  the pivotal choice, because it decides whether VIC can be load-bearing at all:
  - **Impulse** (what we built): the right quantity for **gearbox / harmonic-drive reaction-torque
    protection**, but — per the finding above — **not controllable by VIC** on this arm. Its
    provenance is our own hardware derivation (τ_rated × repeated-peak × window); the safety
    literature does not ground it.
  - **Peak-force / energy** (v∝1/√μ, energy ≈ ½·m_eff·v²): what the **human-injury safety
    literature** (ISO/TS 15066, Haddadin) actually bounds, and — crucially — **what control-level
    variable impedance *does* shape.** Re-pointing the constraint here would make VIC genuinely
    load-bearing (the policy trades stiffness against a force/energy cap), at the cost of changing
    the thesis's headline quantity.

- **(f) Which way do we take the thesis given (e)?**
  - **(A)** Re-point the bounded quantity to **peak-force/energy** and keep the VIC arc *(our
    recommendation — it rescues the VIC contribution and aligns with the safety literature)*.
  - **(B)** Keep **impulse** but change the **lever** — a heavier striker or a harder/rigid task
    that raises physical m_eff·v to the cap (VIC still doesn't command it, but the constraint at
    least becomes active).
  - **(C)** Keep impulse and **reframe the thesis around the negative finding + the machinery**:
    a rigorous demonstration that control-VIC cannot shape ballistic impulse on a rigid-transmission
    arm — a real, literature-relevant result, but a negative headline.

## Our recommendation

Pursue **(A)** — bound peak-force/energy — after verifying the rotor inertia. It is the option
under which "variable impedance is the safety mechanism" is both *true on the Z1* and *supported by
the safety literature*, and it reuses essentially all of the shipped machinery (the accumulators,
soft-CaT, CatPPO) with a different measured quantity. We will not begin the VIC build until you
weigh in on (e)/(f), since it determines what VIC optimizes against.
