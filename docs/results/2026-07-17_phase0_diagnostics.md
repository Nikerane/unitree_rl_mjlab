# Phase 0 — fixed-impedance local diagnostic battery (2026-07-17)

> **Impulse-threshold provenance correction (2026-08-14):** The `27.3 ms` basis and
> `[1.64, 3.28, ...]` vector below are historical project inputs, not a validated Z1
> reaction-impulse or damage limit; the unsupported `kappa=2` interpretation is retired.
> The numeric body remains frozen; see `../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

**Result: the enforced Λ and the deliverable impulse are decisively PRESS integrals, not ballistic
momentum; the ballistic quantity is structurally vacuous across the entire reachable envelope; and the
enforced Λ "binds" only at the intersection of three artifacts (rigid target + sustained press + the
shipped cap/window mismatch).** Five diagnostics (A1,A2,A3 empirical; A5,A6 analytic), all consistent.
Companion to `2026-07-17_fixed_impedance_deep_dive.md` (model) + `_execution_plan.md` (plan). Probes +
raw JSON under `docs/results/assets/2026-07-17_fixed_impedance_diag/`. CPU-only; no GPU used.

## A2 — solref sensitivity (THE press discriminator) ✅ PRESS
Reference strike, baseline vs contact-solref ×2 (`sensitivity_run.py --config baseline|solref2x`):

| quantity | baseline | solref×2 | Δ |
|---|---|---|---|
| worst-joint Λ (j2, full-event, baseline-subtracted) | 0.3029 | 0.0718 | **−76.3%** |
| object-side delivered ∫F·dt | 0.6094 (= i_ref) | 0.1429 | **−76.6%** |
| peak axial force | 21.6 N | 12.4 N | −42.6% |
| contact window | 44 ms | 38 ms | −13.6% |

Λ tracks delivered impulse in **near-perfect lockstep (−76.3% vs −76.6%)** and both are maximally
solref-fragile. A momentum-pinned ballistic integral would be solref-**robust**; this is the opposite ⇒
**press**. (Stronger than the older gripper-era −38%; the current reference is a harder, cleaner strike —
delivered = i_ref = 0.6094 exactly, a harness sanity check. Real contact window 44 ms sits between the
27.3 ms cap-derivation and the 50 ms enforcement window.)

## A1 — approach-velocity sweep 0.5×–8× (`probe_battery.py` T1) ✅ PRESS + effort-clamped
| cmd factor | v_touch achieved | window | peak/mean F | rebound | Λmax |
|---|---|---|---|---|---|
| 0.5× | 0.77 m/s | 48 ms | 1.37 | no | 0.175 |
| 1.0× | 1.33 | 44 ms | 1.56 | no | 0.347 |
| 2.0× | 1.41 | 38 ms | 1.68 | no | 0.291 |
| 4.0× | 1.38 | 28 ms | 1.51 | no | 0.182 |
| 8.0× | 1.31 | 42 ms | 1.61 | no | 0.362 |

Commanding 8× harder does **not** raise contact speed past ~1.4 m/s (effort clamp); Λ is **non-monotonic**
and never exceeds **0.36** (a ballistic Λ would climb with contact speed); peak/mean force **1.4–1.7×**
(vs 5–20× for a collision); **no rebound** anywhere. Drive-through press across the whole reachable envelope.
(T3: a sustained press integrates Λ smoothly to ~0.29 over 38 ms.)

## A3 — window/cap pairing, rigid target (`impedance_one.py`, fixed impedance, 1.38 m/s) ✅ decision-(e) table
Lock verified (nail disp 0.017 mm). Formula-consistent 50 ms caps `[3,6,3,3,3,3]` = `1.829×` the shipped
`[1.64,3.28,…]` uniformly, so the 2nd column is `×0.547`.

| quantity | worst Λ/cap — shipped caps (27.3 ms) | — formula caps (50 ms) | binds? |
|---|---|---|---|
| **IMPACT-only** (9 substeps, genuine ballistic) | **0.609** | 0.333 | NO |
| **ACCUM** (shipped 50 ms sliding window = *enforced*) | **1.184** | 0.647 | shipped YES / formula NO |
| FULL (588 ms sustained press, uncapped diagnostic) | 10.87 | 5.94 | n/a |

"The constraint binds" is true **only** at the intersection of: rigid target (not the yielding nail) +
press-through (not impact — impact-only is 0.61× even here) + the shipped cap/window mismatch. Change any
one → it does not bind. On the real yielding nail, ACCUM ≈ 0.19–0.36 (A1/A2).

## A5 — rigid + velocity-injection ceiling (`ceiling.py`, analytic) ✅ ballistic vacuous
m_eff (armature-coupled) 0.544 kg; kinematic head-speed ceiling 3.97 m/s; **effort-reachable 1.35 m/s**.
Ballistic worst Λ/cap: **e=0 (real yielding nail) 0.194; e=1 (perfectly elastic) 0.388** at 1.35 m/s. The
cap is crossed only under a *doubly*-unreachable combo: e=1 **and** 3.48 m/s (2.6× the effort limit; joint3).
Ballistic load-bearing joints = j2/j3/j4 (moment arms 0.50/0.43/0.33) — note this differs from the
press-regime loaded joint (j1), a clean press-vs-ballistic tell.

## A6 — armature sensitivity / model bound (`armature_check.py` + linear ballistic map) ✅ vacuity survives
m_eff by armature scale: ×1 = 0.544, ×3 = 0.671, ×10 = 0.901, ×30 = 1.403 kg. Ballistic Λ ∝ m_eff, so
reachable (1.35 m/s) worst Λ/cap:

| armature | m_eff | Λ/cap e=0 (real) | Λ/cap e=1 (elastic) |
|---|---|---|---|
| ×1 (modeled) | 0.544 | 0.194 | 0.388 |
| ×10 | 0.901 | 0.321 | 0.643 |
| ×30 (gear²-plausible) | 1.403 | 0.500 | 1.001 |

Even at 30× armature the **realistic (e=0) ballistic impulse is 0.50× cap** — vacuity survives the model
uncertainty. Only the doubly-unrealistic ×30 **and** e=1 combination reaches cap. (Khadiv decision (f):
verify real Z1 rotor inertia, but the structural finding holds.) NB: `kp` does **not** enter M(q), so static
stiffness has zero lever here — the VIC-deferral physics.

---

## Synthesis
1. **Both spine symptoms are one physics fact.** The enforced Λ (A2 −76% solref-fragile; A1 non-monotonic,
   effort-clamped; A3 impact-only 0.61 vs press-through 1.18/10.87) is a press integral. The ballistic
   impulse is 0.19–0.39× cap at reachable velocity (A5), ≤0.50× even at 30× armature (A6). Delivered impulse
   sits at i_ref because the reachable strike *is* the reference (A2 baseline = 0.6094).
2. **Bindability is a pairing choice, not a training result.** Enforced ACCUM crosses cap (1.184×) only for
   rigid + press-through + shipped 27.3 ms caps; the formula-consistent 50 ms caps give 0.647× (no bind), and
   the yielding nail gives 0.19–0.36×.
3. **Khadiv decision (e), now with numbers:** bound the *ballistic* impulse (clean "impact-safe" but provably
   vacuous — ≤0.61× even rigid), the *windowed-press* reaction (binds only under the mismatched pairing;
   reduces to avg-torque×window ⇒ "why not bound torque?"), or *split*. Decision (f): armature ×1→×30 keeps
   realistic ballistic ≤0.50× — vacuity is model-robust.
4. **Implication for Phase 2 maximization:** on fixed impedance, contact speed is hard-clamped at ~1.4 m/s, so
   neither the press nor the ballistic quantity can be pushed materially past the reference by reward shaping
   alone — reward levers move the policy *to* the ceiling, not past it. Exceeding it needs variable impedance
   (deferred). Phase 2 will quantify how close reward reshaping gets.

**Guardrails honored:** `imp_max_p=0`, `IMP_J_LIMIT` unchanged, no VIC, no repo/asset files mutated (all probes
spec-wrap at runtime); originals under `2026-07-12_impulse_vacuity/` untouched (extended copies only).
