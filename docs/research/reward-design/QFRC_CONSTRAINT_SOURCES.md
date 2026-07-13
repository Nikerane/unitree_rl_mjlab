# `qfrc_constraint` from primary sources — verified study guide

> **Purpose.** The per-joint impulse metric Λ_j = Σ|qfrc_constraint_j|·dt is the measurement
> primitive of the thesis. This doc grounds it in the **primary sources** (MuJoCo docs, headers,
> C source, MJWarp docs, the Todorov papers) so understanding does not route through an
> assistant's paraphrase — read the sources directly; this is the map.
>
> **Provenance.** Every claim verified against the **live page text downloaded 2026-07-13**
> (MuJoCo "stable" docs; `mjdata.h` + `engine_core_constraint.c` from `main`; MJWarp docs +
> README). Quotes verbatim. Verification score over the 20 extracted claims: **18 CONFIRMED,
> 2 PARTIAL** (both are the impulse-semantics *inferences*, flagged below), **0 refuted**.
> The docs are a rolling "stable" build — **pin the MuJoCo version (repo uses mujoco 3.8.1)**
> when citing in the thesis.
>
> Companion: [`QFRC_MEASUREMENT_AUDIT.md`](QFRC_MEASUREMENT_AUDIT.md) answers *"is the
> measurement defensible"* (failure modes, solver sensitivity, field practice); this doc answers
> *"what does the primitive actually mean, per the official sources."*

---

## Part 0 — the code chain this grounds (read the code alongside the docs)

```
mujoco_warp constraint solver
  └─ writes qfrc_constraint (generalized-coordinate Jᵀf, ALL constraint rows)     [sources: A1, A2]
      └─ SubstepImpulseAccumulator — sliding-window Σ|qfrc − baseline|·dt @500 Hz
         src/tasks/hammer/mdp/impulse_bound.py  (class SubstepImpulseAccumulator)
          ├─ diagnostics: ContactRowImpulseAccumulator (contact-row-only ground truth)
          │   src/tasks/hammer/mdp/contact_row_impulse.py   [decomposition route: C-iii]
          │   + SubstepDeliveredImpulse (object-side ∫F·dt, netforce sensor)
          └─ joint_impulse_excess = Λ_j − IMP_J_LIMIT_j
             src/tasks/hammer/cat/constraints.py
              └─ CatSoftHook → δ = f(margin / cmax)  (log-only until decision (e))
                 src/tasks/hammer/cat/hook.py
                  └─ CatPPO — soft γ(1−δ) return discount + dual-mask GAE
                     src/tasks/hammer/rl/cat_ppo.py
Validation: docs/research/reward-design/derive_impulse_thresholds.py (C0 quantity gate),
            validate_rewards.py Phase M, scripts/eval_impulse.py (C3 eval).
```

---

## Part A — reading list (in reading order)

**A1. Computation chapter — the constraint model and where `qfrc_constraint` lives in the math**
<https://mujoco.readthedocs.io/en/stable/computation/index.html>
Key anchors: `#equation-eq-motion` (Eq. 1), `#constraint-model`, `#friction-loss`,
`#constraint-solver`, `#parameters`, `#piforward` (pipeline stage table).
Read FOR: the equations of motion `M v̇ + c = τ + Jᵀf` and the sentence *"The transpose of the
Jacobian maps force vectors from constraint to joint coordinates: the constraint force f maps to
force Jᵀf in joint coordinates"* — the mathematical definition of what Λ_j integrates. Also the
constraint-row ordering (*"The types are: equality, friction loss, limit, contact"*), the
friction-loss-as-constraint model (the documented root of the ~45% contamination), and the
soft-solver parameterization `R_ii = (1−d_i)/d_i·Â_ii`, `a_ref,i = −b_i(Jv)_i − k_i r_i` with
d, b, k from solimp/solref — the documented root of solref sensitivity. Caveat: this chapter is
math-notation only; the string `qfrc_constraint` never appears in it.

**A2. API reference, Types — the authoritative field definitions**
<https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html>
Key anchors: `#mjdata`, `#mjtconstraint`, `#mjcontact`.
Read FOR: `qfrc_constraint // constraint force (nv x 1)` under *"computed by
mj_fwdConstraint/mj_inverse"* in the POSITION/VELOCITY/CONTROL-dependent block; the per-row
arrays `efc_type/efc_id/efc_J/efc_state/efc_force`; the 8-value `mjtConstraint` enum including
`mjCNSTR_FRICTION_DOF // dof friction`; and `mjContact.efc_address // address in efc; -1: not
included`. Mirrors [`mjdata.h`](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h)
— cite the docs, use the header for line-level reference.

**A3. Programming/Simulation — step semantics, contact-force readout, convergence caveats**
<https://mujoco.readthedocs.io/en/stable/programming/simulation.html>
Sections: "Simulation loop", "Forward dynamics", "Inverse dynamics", "Contacts", "Diagnostics".
Read FOR: *"mj_step does two things: compute the forward dynamics in continuous time, and then
integrate over a time period specified by mjModel.opt.timestep"* and the staleness statement
(intermediate results *"available but outdated by one time step"*) — together these justify
per-substep sampling. Also the official contact-row isolation recipe
(`d->efc_force + d->contact[i].efc_address`, pyramidal-cone caveat, `mj_contactForce`) and the
fwdinv identity: constraint forces are exact only *"by running the iterative solver to full
convergence"*.

**A4. API reference, Functions — what produces and decomposes the quantity**
<https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html>
Key anchors: `#mj-constraintupdate`, `#mj-contactforce`, `#mj-fwdconstraint`, `#mj-forward`,
`#mj-muljactvec`, `#mj-rnepostconstraint`.
Read FOR: `mj_constraintUpdate` — *"Compute efc_state, efc_force, qfrc_constraint … cost =
s(jar) where jar = Jac*qacc − aref"* (the one doc line tying `qfrc_constraint` to the solver
optimization); `mj_mulJacTVec` — *"maps forces from constraint space to joint space"*;
`mj_contactForce` and `mj_rnePostConstraint` (`cfrc_int`/`cfrc_ext`) as the two alternative
official contact-reaction readouts (cross-validation candidates).

**A5. Modeling chapter, Solver parameters — what solref/solimp mean**
<https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters> (`#impedance`, `#reference`)
Read FOR: solref = (timeconst, dampratio), the direct (−stiffness, −damping) format, and:
*"the timeconst parameter controls constraint softness … Larger values correspond to softer
constraints"*, plus the `refsafe` rule (timeconst ≥ 2·timestep). Primary grounding for the
solref-sensitivity result: contact-force profiles are functions of these parameters by design.

**A6. MJWarp docs — parity and divergence for the training backend**
<https://mujoco.readthedocs.io/en/stable/mjwarp/index.html> (`#feature-parity`,
`#differences-from-mujoco`, determinism FAQ)
Read FOR: float32 vs C MuJoCo's float64 (*"Solver settings, including iterations, collision
detection, and small friction values may be sensitive to differences in floating point
representation"*); GPU non-determinism from atomics; `efc.J`/`efc.D` dense + padded; and
**`mjENBL_FWDINV` is not available** — the official solver-quality diagnostic for the Λ pipeline
can only be run by replaying trajectories in C MuJoCo. Contact override `o_solref/o_solimp`
also unavailable.

**A7. mujoco_warp README — the one-line parity statement**
<https://github.com/google-deepmind/mujoco_warp#mujoco-api-compatibility>
Verbatim: *"MuJoCo Warp supports the same features as MuJoCo with the following exceptions:
Integrator: IMPLICITFAST midpoint …; Solver: PGS and noslip …; Actuator/Sensors: PLUGIN …; Flex:
experimental."* Feature-level only — it does **not** promise value-level parity of
`qfrc_constraint`.

**A8. The two papers (canonical citations)**
- Todorov, Erez, Tassa, *"MuJoCo: A physics engine for model-based control,"* IROS 2012,
  pp. 5026–5033. DOI: [10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109) —
  the citable origin; velocity-stepping/convex contact stance.
- Todorov, *"Convex and analytically-invertible dynamics with contacts and constraints,"*
  ICRA 2014. DOI: [10.1109/ICRA.2014.6907751](https://doi.org/10.1109/ICRA.2014.6907751) — the
  theory behind the regularized convex solver whose output is `efc_force`. Note the docs add:
  *"some of the technical ideas in this chapter are new and have not been described elsewhere"*
  — the docs, not the 2014 paper, are current truth for the exact solver.

**A9. Overview chapter — "Units are unspecified"**
<https://mujoco.readthedocs.io/en/stable/overview.html>
Verbatim: *"MuJoCo does not specify basic physical units."* So "Λ_j is in N·m·s" is a
consequence of the Z1 model being authored in SI, **not a MuJoCo guarantee** — phrase it that
way in the thesis.

---

## Part B — claim verdicts (20 claims, abridged; quotes verbatim)

Sources: **T**=APItypes, **S**=simulation, **F**=APIfunctions, **H**=mjdata.h, **C**=computation, **W**=MJWarp.

| # | Claim | Verdict |
|---|---|---|
| 1 | `qfrc_constraint` def, (nv×1), computed by `mj_fwdConstraint/mj_inverse`, stage placement; page states no units / no Jᵀ identity [T] | **CONFIRMED** |
| 2 | `efc_force/efc_type/efc_id/efc_J/efc_state` per-row arrays exist [T] | **CONFIRMED** |
| 3 | `mjtConstraint` = exactly 8 types incl. `mjCNSTR_FRICTION_DOF`; active list rebuilt *"at each simulation time step"* [T] | **CONFIRMED** |
| 4 | `mjContact.efc_address` + `dim` are the contact→efc bridge [T] | **CONFIRMED** |
| 5 | `mj_fwdConstraint` is the final dynamics stage before integration [S] | **CONFIRMED** |
| 6 | mj_step = forward dynamics then integrate ⇒ f·dt is per-step impulse [S] | **PARTIAL** — first half verbatim; the impulse reading is an *inference*, exact under Euler-family integrators (which hold the force constant over dt), breaks under RK4; the docs never use "impulse" for any qfrc quantity |
| 7 | Post-step reads are one step stale = the force that produced the step [S] | **PARTIAL** — staleness verbatim; "documented valid sampling point" overstates: memory-validity is documented, the metric semantics are ours |
| 8 | Header: `qfrc_constraint (nv x 1)` generalized coords; no Jᵀ identity/units in header [H] | **CONFIRMED** |
| 9 | Header: efc row arrays sized by `nefc`; `efc_J` is `(nJ x 1)` sparse [H] | **CONFIRMED** |
| 10 | `mjContact.efc_address`, −1 if excluded [H] | **CONFIRMED** |
| 11 | `efc_frictionloss` row array; frictionloss forces are solver outputs folding into qfrc [H+C] | **CONFIRMED** |
| 12 | `mj_constraintUpdate` computes efc_state, efc_force, qfrc_constraint; cost = s(jar) [F] | **CONFIRMED** |
| 13 | `mj_contactForce` returns 6D force:torque in the contact frame [F] | **CONFIRMED** |
| 14 | qfrc_* decomposition comments + per-contact solref/solreffriction/solimp [T/H] | **CONFIRMED** |
| 15 | Per-row attribution official: efc_address arithmetic, exclude flag, pyramidal caveat [S] | **CONFIRMED** |
| 16 | Solver generally terminated early; fwdinv diagnostic monitors convergence [S] | **CONFIRMED** (balance: *"the default Newton solver converges so quickly (usually 2-3 iterations)"*; **fwdinv unavailable in MJWarp** [W]) |
| 17 | Per-row `efc_KBIP` (stiffness, damping, impedance) ⇒ forces solver-parameterized at data level [H] | **CONFIRMED** |
| 18 | `mj_fwdConstraint` *"Run selected constraint solver"* [F] | **CONFIRMED** |
| 19 | `mj_forward` = mj_step without integration [F] | **CONFIRMED** |
| 20 | `mj_mulJacTVec` maps constraint→joint space; `mj_rnePostConstraint` computes cacc/cfrc_int/cfrc_ext [F] | **CONFIRMED** (footnotes: apparent doc typo "nv x 6" vs `(nbody x 6)`; cfrc bug w/ spatial tendons, issue #832 — irrelevant to the tendon-free Z1) |

---

## Part C — usage cross-check (our six assumptions vs the primary sources)

**(i) `qfrc_constraint` is the generalized-coordinate Jᵀf of ALL constraint rows — SUPPORTED
(composite + source-code literal).** No single doc sentence prints the identity; it follows from
Eq. (1) (`M v̇ + c = τ + Jᵀf`; *"the constraint force f maps to force Jᵀf in joint
coordinates"*, [A1 `#equation-eq-motion`]), all constraint types stacking into J's rows
(`#constraint-model`), and `mj_constraintUpdate` computing qfrc_constraint [A4]. The **literal
identity is one line of C source**:
`mj_mulJacTVec(m, d, d->qfrc_constraint, d->efc_force);` inside `mj_constraintUpdate` in
[`src/engine/engine_core_constraint.c`](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_core_constraint.c)
(~line 3358 on main) — cite this when the thesis needs the exact identity. **Consequence:** rows
sum **per DOF before** any |·|, so opposite-signed simultaneous rows (e.g. friction opposing
contact on joint3) cancel inside `qfrc_constraint_j` — exactly the sign-cancellation the
sign-aware rows-bound fix (commit 704f30b) encountered.

**(ii) dof-friction (frictionloss) rows fold in — SUPPORTED.** `mjCNSTR_FRICTION_DOF` is a
first-class efc row type [A2]; friction loss *"is modeled as a constraint, namely an upper limit
on the absolute value of the force that friction can generate"*, Jacobian *"a vector of 0s with
a 1 at the joint address"* [A1 `#friction-loss`]. The ~45% contamination is therefore
**documented-by-design behavior**, not a bug or weld artifact — raw `|qfrc_constraint_j|`
measures *all* constraint reactions including dry friction; isolating contact requires row
decomposition.

**(iii) per-row decomposition via efc_type/efc_J/efc_force is the documented isolation route —
SUPPORTED.** Row arrays documented [A2]; contact-row isolation spelled out with exact pointer
arithmetic [A3 "Contacts"]; constraint→joint mapping direction documented (`mj_mulJacTVec`,
[A4]). `contact_row_impulse.py` assembles only documented pieces; the pyramidal-cone redundancy
is the documented gotcha (`mj_contactForce` exists for that reason). `mj_contactForce` and
`mj_rnePostConstraint`/`cfrc_ext` are the official alternatives to cross-validate against.

**(iv) per-substep sampling then Σ|·|·dt as impulse — PARTIALLY SUPPORTED; the impulse
semantics are the thesis's own construction.** Supported: forces recomputed every step (active
set rebuilt each step; `mj_fwdConstraint` runs before integration), post-step reads are the
force that produced the step. Under Euler-family integrators the state update applies exactly
that force over `opt.timestep`, so qfrc·dt per step is the impulse the *integrator* imparted —
a sound inference, but the docs never define an impulse readout, never bless Σ·dt, and say
nothing about the rectified |·| (a project choice that upper-bounds rather than measures net
impulse, applied *after* the per-DOF row summation of (i)). Helpful doc grounding: MuJoCo's
contact model is **soft and continuous-time** — impacts produce large finite forces over
several steps rather than instantaneous impulses, which is precisely why window accumulation at
the physics rate is the right estimator shape. **Thesis phrasing:** sampling validity is
documented; the impulse metric is defined by us on top of documented quantities, validated by
`derive_impulse_thresholds.py` + the C0 three-way quantity validation.

**(v) constraint forces are soft/solver-parameter-dependent — STRONGLY SUPPORTED.** Three
independent groundings: the model itself (impedance d sets `R_ii=(1−d_i)/d_i·Â_ii`; `a_ref`
from solref via b, k — [A1 `#parameters`] Eqs. 11–12); [A5]: *"the timeconst parameter controls
constraint softness"* + refsafe; convergence (forces exact only at full convergence, solver
*"usually terminated early"* — [A3, A4]). The −38% solref-fragility finding is an **expected
property of the model class**: the same scenario with different solref is a *different
constraint law*, by documented design. (Also documented: contact solref/solimp mix from both
geoms via `solmix` — relevant when perturbing nail vs hammer params.)

**(vi) mujoco_warp parity — FEATURE-LEVEL DOCUMENTED, VALUE-LEVEL EXPLICITLY NOT GUARANTEED.**
Feature parity statement [A7]; documented divergences touching this metric: **float32** (solver
sensitivity quote above), GPU **non-determinism** (atomics), dense/padded `efc.J`, different
warmstart init, and **`mjENBL_FWDINV` unavailable** ⇒ the official solver-quality check must be
run by replaying trajectories in C MuJoCo. Nowhere do the sources state that
`qfrc_constraint`/`efc_force` values numerically match C MuJoCo — warp-vs-C agreement of Λ_j is
something the project must (and, via the C0 gates, does) establish empirically.

---

## Where official grounding ends (the honesty paragraph for the thesis)

The docs fully ground **what** `qfrc_constraint` is: Jᵀf over all rows, all eight row types,
per-row decomposable, recomputed per step, soft and solver-parameterized. They do **not**
ground: the word "impulse" for any integral of it; the rectified |·| choice; the 50 ms window;
per-DOF sign cancellation as feature or bug; SI units (explicitly disclaimed — *"units are
unspecified"*); or numerical equality between the warp and C backends. Those are the project's
own constructions/validations and should be presented as such — backed by
`derive_impulse_thresholds.py` and the C0 three-way quantity validation, not by citation.

*Downloaded page snapshots for quote re-checking were kept under the session scratchpad
(`scratchpad/sources/`, ephemeral); re-fetch the URLs above to re-verify.*
