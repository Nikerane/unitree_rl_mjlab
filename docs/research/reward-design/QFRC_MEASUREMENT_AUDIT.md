# qfrc_constraint measurement — verification, critique, and the field's alternatives

**Date:** 2026-07-10 · **Method:** deep-research harness (5 search angles → primary-source fetch →
3-vote adversarial verification; the verify phase was twice interrupted by session limits, so each
claim below carries an explicit status tag) + code checks run directly on this repo.
**Status tags:** `[3-0]`/`[2-0]` = adversarially verified this run; `[doc]` = direct quote fetched
from the primary source, votes did not run; `[code]` = verified by running code in this repo;
`[session]` = full-text read verified earlier (2026-07-06 notation sweep); `[background]` =
standard literature cited from knowledge — run `/ars-citation-check` before thesis use.

**Bottom line:** the deep-dive's mechanics description is **correct** on every load-bearing claim,
with three corrections (the weld does not exist in our training scene; the solver works in
impulses internally with forces reported as impulse/h — which makes our `Σ|qfrc|·dt` exactly
right; and the "how it can fail" list must be extended with MJWarp-specific pitfalls). The biggest
*scientific* critique — solver-parameter dependence — is real, confirmed from the primary sources,
and has a standard defense (bound the impulse, not the force, and cross-check against the
object-side ∫F·dt), plus a cheap sensitivity experiment worth running.

---

## 1. Claim-by-claim verdict on the deep-dive text

| Deep-dive claim | Verdict |
|---|---|
| MuJoCo builds constraint rows tagged by `efc_type`: contact (frictionless/pyramidal/elliptic), joint limits, dof friction, equality | **CORRECT** `[3-0]` — the `mjtConstraint` enum has exactly these plus two tendon types we don't use ([API types](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html)) |
| Solves a convex optimization (Gauss's principle / dual), per Todorov 2012/2014 | **CORRECT** `[3-0]` — `f = argmin ½λᵀ(A+R)λ + λᵀ(a_unc − a_ref) s.t. λ∈Ω` ([Computation docs](https://mujoco.readthedocs.io/en/stable/computation/index.html)); Todorov ICRA 2014 confirms the regularized-Gauss formulation `[doc]` |
| `qfrc_constraint = efc_Jᵀ·efc_force`, summing ALL constraint types per joint | **CORRECT** `[3-0]` — `M v̇ + c = τ + Jᵀf` per the docs; `qfrc_constraint` is the aggregate `(nv,)` vector with **no per-type decomposition** `[doc]` — the contamination story is well-founded |
| Contact row counts per cone | **CORRECT** `[3-0]` — frictionless 1; elliptic `condim` rows; pyramidal `2(condim−1)`; e.g. condim 3 → 3 vs 4. MJWarp implements the same counts (`max(1, 2·(dim−1))` pyramidal, `dim` elliptic) `[doc]` |
| "the IK **weld** equality" contaminates the training signal | **WRONG for our training scene** `[code]` — the built mjlab env has `neq = 0`: no weld, no equality constraints at all (the mocap weld lives only in the standalone viewer scene; `z1_hammer_robot.xml` line 3 says so explicitly). The real contaminants are **dof friction (dominant) + joint limits**. The old memory note attributing a small share to a weld measured something else and is corrected. |
| (implicit) integrating force×dt recovers impulse | **CORRECT, with a nice subtlety** `[doc]` — Todorov 2014 solves for **impulses** in a velocity-stepping scheme; modern MuJoCo reports `efc_force`/`qfrc_constraint` in force units (impulse ÷ h). So `Σ|qfrc_j|·dt` reconstructs the per-joint impulse exactly — the quantity the solver natively works in. |

Corollary for Track 2: `efc_type`/`efc_force` per row are exactly the exposed data needed for the
contact-rows-only isolation `[doc]` — the approach is API-blessed, not a hack.

## 2. Failure modes of Λ_j = Σ|qfrc_constraint_j|·dt, ranked

1. **Solver-parameter dependence (fundamental, confirmed `[2-0]`).** The regularizer
   `R = (1−d)/d·A_diag` comes from `solimp`, the reference acceleration is a virtual spring-damper
   from `solref`, and the docs demand `timeconst ≥ 2×timestep` — so the contact *force profile* is
   a modeling choice, not physics. Todorov 2014 says the same of its ε/κ ancestors `[doc]`.
   **Why the impulse survives this:** the *integral* is pinned by momentum transfer — stopping the
   hammer requires `∫F·dt ≈ m_eff·Δv` regardless of how the force is shaped in time, provided the
   accumulation window covers the whole event (ours is contact-anchored, so it does). Peak force
   is parameter-fragile; its integral is not. This is the core thesis defense for bounding impulse
   and must be stated in the writeup. **Mitigations shipped/planned:** the object-side ∫F·dt
   cross-check (weld/friction-free by construction); the solref/solimp/timestep values recorded in
   the results provenance. **Recommended addition (cheap, strong defense artifact):** a one-off
   sensitivity run — re-run `derive_impulse_thresholds.py` with timestep halved and `solref`
   doubled and report the Λ_j delta; if the impulse moves by percent while peak force moves by
   tens of percent, the "impulse is the robust quantity" claim is demonstrated, not asserted.
2. **MJWarp-specific pitfalls (directly affect the Track-2 validator; all `[doc]`).**
   (a) `efc.J` on GPU is **dense and padded** — sparse Jacobians are "not implemented" per the
   MJWarp docs, even though the sparse-index fields exist in the dataclass; the Task-8 probe must
   resolve which path is live and never trust padding rows. (b) **Row ordering is not C MuJoCo's**
   — host readback explicitly re-sorts rows; on-device code must index through
   `contact.efc_address`/`contact.dim` and classify by `efc.type`, never by address arithmetic
   over the full array (our design already does this). (c) **Silent truncation**: `nefc` clamps to
   `njmax` and `nacon` to `naconmax` — an under-sized allocation silently drops rows; the
   validator needs explicit `nefc < njmax` / `nacon < naconmax` asserts. (d) MJWarp computes in
   **float32** and is **non-deterministic run-to-run on GPU** (atomic ordering) — Λ_j is
   backend-dependent and not bit-reproducible; tolerances and statistics, never bit-comparison.
3. **Sensor-gate false-positive caveat (`[doc]`, mitigated).** MJWarp's contact sensing can report
   a contact entry for a sensed geom pair even when not in collision. Our gate reads the sensor's
   `found` field, and the C0 gate's leak check (`shipped accumulator ≡ 0 before any contact`)
   empirically proves the gate does not leak on this scene — that leak check must stay a permanent
   gate invariant, because it is the guard against exactly this caveat.
4. **Baseline-subtraction sign flip (known).** `subtract_baseline` subtracts the frozen
   pre-contact reaction; when dof friction reverses direction at impact the subtraction
   over-corrects (the known joint-4 artifact). Quantified — not fixed — by the three-way
   comparison; the contact-rows metric is the clean answer.
5. **Rectification (|·|).** The rectified integral ≥ |signed integral|; friction-row sign flips
   within a window inflate Λ_j slightly. Deliberate (we want total shock magnitude, and the HD
   rating is direction-agnostic), but the gate should print signed and rectified side by side once
   so the gap is known.
6. **Micro-impacts / restitution.** Multiple bounces inside one sensor window accumulate into one
   Λ — intended (total shock per event); separate windows are separate pulses by design. No
   change needed; state it.

## 3. How the field measures/bounds per-joint impact load — and why

| Path | Representative refs | Needs | Why that community uses it |
|---|---|---|---|
| **Integrate solver constraint forces** (ours) | MuJoCo Computation docs; Todorov IROS 2012, ICRA 2014 | sim internals | Exact per-joint attribution of the *solved* contact reaction, including all chain coupling — free in simulation, impossible on hardware. `[3-0]`/`[doc]` |
| **Simulator contact impulse, Jᵀλ per substep** | Kang et al. 2025, arXiv:2505.12222 — `τ_inst = Σ Jᵀλ/Δt_sim` ("transmission load"), penalty + 50%-probability termination on overload `[session]` | sim internals (PhysX/Isaac convention) | Same family as our Track-2 contact-rows isolation — direct published RL precedent for bounding joint transmission load from simulator impulses. |
| **Momentum-based external-torque observer** (generalized-momentum residual) | De Luca & Mattone (ICRA 2003/2005); Haddadin, De Luca, Albu-Schäffer, *Robot Collisions: A Survey on Detection, Isolation, and Identification*, IEEE T-RO 33(6) 2017 `[background]` | dynamics model + proprioception only | **The hardware standard** — estimates the per-joint external torque τ_ext without F/T sensors or sim internals. Key defense point: our Λ_j integrand (joint-projected contact reaction) is exactly what this observer estimates on real robots, so the sim quantity has a standard sim-to-real measurement counterpart for the real Z1. |
| **Impulse dynamics / post-impact velocity jump** `M(q̇⁺−q̇⁻) = J_NᵀΛ_N` | Brogliato, *Nonsmooth Mechanics*; Aouaj, Padois, Saccon, arXiv:2010.08220 `[session]`; Pinocchio `impulseDynamics` | model + pre/post velocities | Predicts/validates the impulse from state jumps without integrating any force — our deferred Pinocchio cross-check is this family; the impact-aware (I.AM.) community uses it because velocity jumps are measurable where force transients are not. |
| **Direct joint-torque / wrist F-T sensing** | DLR-lineage torque-controlled arms (KUKA iiwa); wrist F/T + JᵀF | hardware sensors | Gold standard where the sensors exist; the Z1 has no joint torque sensors (current-based estimates only), which is precisely why the observer path above matters for us. `[background]` |
| **Pre-impact proxies** (control what you're *about* to transfer) | Vu et al. 2026, hal-05516105 — maximize `P = m_eff·ẋᵀn`, impulse `i = ΔP` `[session]`; Khurana & Billard 2024/2025 — hitting flux `[session]` | kinematics + inertia model only | Avoids measuring the impact entirely — robust, sensor-free, but blind to chain rebound and multi-joint distribution; complements rather than replaces a reaction bound (Vu et al. explicitly leave the joint-recoil bound as future work). |
| **Constraint-RL enforcement of joint load** | Ma et al. 2025, arXiv:2505.22974 — N-P3O on arm current `\|I_total\| < 8 A` `[session]`; Kang et al. 2025 termination-on-τ_load `[session]`; Chane-Sane et al. CaT, arXiv:2403.18765 — foot-force limit `[session]` | RL framework | Direct precedents that per-joint/actuator load is *enforced* (not just penalized) in current RL practice — the constraint-mechanism company our soft-CaT keeps. |

**The defense sentence this buys:** *in simulation*, integrating the solver's per-joint constraint
impulse is the only path that gives exact per-joint attribution with chain coupling; its known
solver-dependence is disarmed by bounding the momentum-pinned integral (not the parameter-fragile
peak) and validating against the solver-independent object-side ∫F·dt; and *on hardware*, the same
quantity is recoverable with the standard momentum-observer machinery — so the sim constraint has
a measurable real-robot counterpart.

## 4. Actions taken from this audit

- Plan Task 8 (`docs/superpowers/plans/2026-07-06-impulse-cat-c2c3-two-track.md`) updated: expect
  DENSE padded `efc.J` on MJWarp (probe resolves), overflow asserts (`nefc < njmax`,
  `nacon < naconmax`), never assume C row ordering, float32 tolerance note.
- Memory `qfrc-constraint-weld-pollution` corrected: the training scene has `neq = 0` — no weld;
  contamination = dof friction + joint limits.
- The old visual explainer (plan-c00a0e0144274fe4) still names the weld as a contaminant and its
  Trap-2 text carries a weld share — flagged to the user for a plan-side correction.
- Recommended (pending user OK): the solver-sensitivity run in §2.1 as a one-off defense artifact.
