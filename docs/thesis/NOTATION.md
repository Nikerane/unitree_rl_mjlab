# Thesis notation — the impulse symbol Λ_j (audit + writing rules)

**Status: CONFIRMED** (web-verified against the papers' full texts, 2026-07-06 — five-source
fetch sweep: arXiv HTML / HAL PDF; exact quotes below).

**Decision: keep `Λ_j` for the per-joint contact impulse.** It is not ad-hoc — it is the
nonsmooth-mechanics capitalization convention (lowercase λ = contact force / Lagrange multiplier,
capital Λ = its impulsive integral), taken from **Brogliato, *Nonsmooth Mechanics*** and used
verbatim by the Saccon impact-aware line. Our quantity is literally the time-integral of the
joint-space projection of MuJoCo's constraint force (`qfrc_constraint = JᵀλΔ`), so "accumulated λ"
is the most faithful possible reading of what the code computes.

---

## 1. Who uses what (per-paper audit)

| Paper | Impulse symbol | Defining usage (verbatim) |
|---|---|---|
| Aouaj, Padois & Saccon 2020 (arXiv:2010.08220) | **Λ_N** | `M(q)(q̇⁺ − q̇⁻) = J_Nᵀ(q) Λ_N` — "Λ_N represents the impulsive force magnitude"; λ_N is the normal contact force in the smooth EoM. Convention credited to Brogliato. |
| Wang & Kheddar line, IJRR 2023 (arXiv:2006.01987) | **ι** (iota) | "The impulse ι ∈ R³ fulfills Coulomb's friction cone"; `Δv = W·ι` with W the *inverse* inertia matrix. The clash-free alternative if Λ ever becomes untenable. |
| Kang et al. 2025 (arXiv:2505.12222) | **λ** (bold lowercase) | "contact impulse vector"; `τ_inst = Σ Jᵀλ / Δt_sim` — per-substep simulator impulse (Isaac convention); no capital Λ anywhere in the paper. |
| Vu et al. 2026 (hal-05516105) | **i** (lowercase italic) | `i ≜ ∫ nᵀFʰ dt` (Eq. 9), the normal impulse; maximized via projected momentum `P(t) = m_{e,n}(q)·ẋ_νᵀn` (Eq. 10). **Reserves Λ(q) ≜ J_ν M⁻¹ J_νᵀ for the *mobility matrix*** (Eq. 6 — inverse task-space inertia). |
| Chane-Sane et al. CaT, IROS 2024 (arXiv:2403.18765) | none | Constraints `c_i(s,a)`; foot contact **force** limit `‖f^foot‖₂ < f^lim = 50 N`. No impulse quantity. |
| Ma et al. 2025 (arXiv:2505.22974) | none | Arm **current** constraint `|I_total| < 8 A` via N-P3O. No impulse quantity. |
| Khurana & Billard 2024/2025 (hitting flux) | — | Use **Λ = (J M⁻¹ Jᵀ)⁻¹** for directional effective *inertia* (note: the inverse-of-Vu convention) — a second Λ-as-inertia precedent among our citations. |

The clash is therefore real and lives inside our own citation set: **Λ means impulse in the
Brogliato/Saccon school and (inverse) task-space inertia in Vu et al. / Khurana & Billard /
van Steen–Saccon**. Both readings are legitimate; the manuscript must disambiguate, not pick a
"neutral" symbol that matches nobody.

## 2. Writing rules for the manuscript

1. **Define Λ_j at first use** as the impulsive integral of the joint-projected contact force,
   citing Brogliato and Aouaj–Padois–Saccon (arXiv:2010.08220 Eq. 3) as notation precedent, and
   state explicitly that it is *not* the operational-space inertia.
2. **Rename the inertia when citing Vu et al.** In our prose their mobility matrix is written
   `W ≜ J M⁻¹ Jᵀ` (Wang–Kheddar's symbol for exactly that object) or `Λ_mob` — never bare Λ on the
   same page as Λ_j. Effective mass stays `m_eff = (nᵀWn)⁻¹`.
3. **Unify the cap's symbol.** The code name `J_limit` becomes **Λ̄_j** (or `Λ_j^max`) in prose, so
   the constraint reads `Λ_j ≤ Λ̄_j` — one symbol family, and no collision with J-for-Jacobian.
   (Code identifiers are unaffected; this is prose-only.)
4. Physics-textbook `J = ∫F dt` and plain `I` are ruled out: J collides with the Jacobian inside
   the very defining equation (`Jᵀλ`), I with inertia/identity/current (Ma et al.'s constraint is
   literally on `I_total`).

## 3. Two citable nuggets the audit surfaced

- **Kang et al. 2025 terminate the episode with 50% probability when the transmission load
  `τ_load` exceeds a critical threshold** ("probabilistic termination preserves exploration …
  while imposing a realistic cost for severe overloads") — published precedent for
  termination-based enforcement of a joint transmission-load constraint. Ours generalizes it:
  graded δ ∝ violation (soft CaT) instead of a fixed 50% coin, and impulse instead of an averaged
  load. Strong defense citation next to Chane-Sane.
- **Vu et al.'s Eq. (11), `i = ΔP`** (impulse = change of projected momentum), is the clean formal
  bridge between their maximize-momentum QP objective and our `DeliveredImpulseTerm` reward — cite
  it when arguing the reward maximizes the same physical quantity their model-based method does.

---

Detail trail: the per-paper quote sheets live in the 2026-07-06 notation-sweep records (session
workflow output); the citation anchor set is `docs/thesis/README.md` §5 and
`docs/research/reward-design/LITERATURE.md`.
