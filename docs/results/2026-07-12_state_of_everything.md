# State of Everything — impact-safe manipulation thesis (snapshot 2026-07-12)

> Consolidation snapshot fusing (a) a full internal repo survey (code / docs / tests / findings),
> (b) this session's new CPU results, and (c) an external scientific-literature deep-research pass
> (24 sources adversarially verified, 1 refuted). Purpose: know exactly where the project stands
> before pivoting to variable impedance. Not committed / not yet indexed in docs/README — a dated
> record, same status class as the other docs/results/ files. Authority still = code > docs/README
> index > living docs; this is evidence, not an override.

---

## 0. Bottom line (read this first)

The **machinery is built and the central finding is proven** — what's missing is not more analysis, it's *banking* what we have and *advancing* to variable impedance.

1. The full fixed-impedance impulse-CaT pipeline is **shipped end-to-end but runs LOG-ONLY** (`imp_max_p=0` ⇒ δ≡0). The "impulse constraint" has, to date, applied **zero** constraint pressure — it is instrumentation that logs Λ_j. Turning δ on/off is bit-identical; the arm still differs from the 7-term task by the `DeliveredImpulseTerm` maximize reward (weight 2.0), so it is identical to the *constraint-removed maximize arm* (`c3_imp0`), not to the plain task baseline.
2. On the fixed-impedance Z1 the constraint is **vacuous for genuine impacts** (worst reachable ≈36–39% of cap; high confidence, 4 methods + cross-check). Today we closed the last hedge: even a **genuinely rigid target at 20× stiffness tops out at ~0.6× cap** — because impact velocity is **effort-clamped at ~1.3 m/s** and stiffness can't lift it. The binding lever is *velocity*, and fixed impedance can't reach it.
3. The **external literature strongly supports the architecture**: CaT's γ(1−δ) is the right mechanism; a per-joint impulse cap is correctly an *instantaneous hard constraint* (outside CMDP); and the physics that "impedance modulates effective mass, and effective mass × velocity governs impact" is textbook (Khatib / Haddadin / Kirschner / ISO 15066). The conclusion "the protective role belongs in variable impedance" is well-grounded.
4. **Two honest challenges the literature raises** (see §4): the human-safety caps are *energy/force*-based (v∝1/√μ), not strictly *impulse* (v∝1/μ) — so the thesis must be crisp about *which* quantity it bounds and *why* (gearbox reaction-torque, not human injury); and every VIC-for-safety result is *emergent compliance on quasi-static contact*, never an *explicit impulse constraint on impulsive striking* — which is exactly the thesis's novelty, and exactly what remains untested.
5. The **debt behind the "loop" feeling is real**: the 2026-07-12 vacuity deliverables are uncommitted, `docs/README` is 7 days stale, the thesis digest doesn't reflect the vacuity reframe, the one code fix that matters (robot-side press cap) is now implemented (§5), a batch of deferred adversarial-review fixes remain, and Khadiv sign-off + the C3 GPU run are both pending. Nothing is *lost*; it's just not *banked*.

6. **⚠️ The existential risk this doc must not soft-pedal (added after adversarial review).** There is a coherent scenario in which the thesis's central mechanism — an explicit, VIC-shaped, solver-trustworthy impulse constraint that binds — **cannot be demonstrated in this simulator at all.** Two of its load-bearing assumptions are already leaning pessimistic in our own data: (a) MuJoCo may not represent the sharp impulse peak VIC is meant to shape (the reference is a smeared drive-through press; Λ is −38% solref-fragile; Acosta et al. impeach MuJoCo on impacts), and (b) VIC's two levers on Λ look *weak* here — 40× stiffness moved impact-Λ only 0.47→0.64, and velocity is torque-effort-clamped at ~1.3 m/s that `set_gains` cannot lift. If both hold, VIC is dead-on-arrival on this platform. §7 now gates on this instead of assuming VIC succeeds.

**The way out of the loop (revised, §7): run a ½-day VIC-feasibility ceiling + fire Khadiv in parallel → bank while waiting → gate VIC on the ceiling + a solver-convergence study.**

7. **Update — the gate is now RUN (§9), and it returned a strong negative.** VIC-as-commanded-stiffness has **~zero lever on the ballistic impulse**: reflected mass is fixed by `M(q)`+`dof_armature` (physical), not by kp (control), so `set_gains` only adds active-press force, never passive reflected mass. The naive "VIC unlocks binding" arc **likely fails on the Z1**, and the fork (bound energy/force instead of impulse · change the lever · reframe) is now a Khadiv decision. Separately flagged: the arm `dof_armature` looks **under-modeled** (placeholder 0.01/0.02), and both the vacuity margin *and* the VIC verdict depend on it — verify it against real Z1 rotor specs.

---

## 1. The thesis and the current direction

Impact-safe contact-rich manipulation (TU Munich / ATARI Lab / Prof. Khadiv), **Z1-primary** (G1 = optional future replication). Single-policy online RL (the two-level SURE+RL and generate-then-track designs were walked back by the supervisor). The program, built + validated on the Z1 first:

1. **Faithful soft γ(1−δ) CaT** — the constraint-incentive mechanism.
2. **Substep per-joint impact-impulse constraint** Λ_j = Σ|qfrc_constraint_j|·dt — the headline quantity.
3. **Variable impedance** (policy commands per-joint stiffness via `set_gains`) — the endgame, added **after** fixed-impedance results are complete.

Sequencing is fixed: finish fixed impedance, *then* VIC.

---

## 2. What is built (the machinery)

Shipped and wired end-to-end (`src/tasks/hammer/`):

| Component | File | Status |
|---|---|---|
| `SubstepImpulseAccumulator` (robot-side Λ_j, per-event pulse, baseline-subtracted) | `mdp/impulse_bound.py` | shipped |
| `SubstepDeliveredImpulse` (object-side ∫F·dt, **has** a 25-substep press cap) | `mdp/impulse_bound.py` | shipped |
| `ContactRowImpulseAccumulator` (efc-row isolation — friction-immune ground truth, log-only) | `mdp/contact_row_impulse.py` | shipped |
| CaT δ-math core (Chane-Sane port) + `CatSoftHook` (writes δ, r_pos) | `cat/constraint_manager.py`, `cat/hook.py` | shipped |
| `CatPPO` (scale-positives discount + dual-mask GAE) + `CatRolloutStorage` (float soft-dones) | `rl/cat_ppo.py`, `rl/cat_storage.py` | shipped |
| Per-joint caps `IMP_J_LIMIT=[1.64,3.28,1.64,1.64,1.64,1.64]` + `cat_impulse` wiring | `config/z1/env_cfgs.py` | shipped |
| `DeliveredImpulseTerm` (maximize objective) + 7-term baseline + impact_progress + imitation prior | `mdp/rewards.py` | shipped |
| C0 quantity gate (`derive_impulse_thresholds.py`), C3 eval driver (`scripts/eval_impulse.py`) | — | shipped |
| ~297 tests / 20 modules; `validate_rewards.py` phases A–M | `tests/` | shipped |
| **Enforcement path** (`imp_max_p>0`) — fully written + guarded | `cat/hook.py` | **partial (never enabled)** |
| **Variable impedance** (`set_gains` stiffness action) | — | **planned (no code)** |

**The load-bearing caveat:** `imp_max_p=0.0` everywhere ⇒ the impulse constraint is **log-only instrumentation**. Turning it on is a one-flag change gated behind (C0 gate green ✓, Khadiv confirming soft-CaT vs CMDP, local gate green).

Known code gaps: the robot-side accumulator has **no press-length cap** (the object side does); `Z1_JOINT_IMPULSE_LIMIT=0.1` is a stale-but-guarded placeholder; the Pinocchio impulse cross-check is scaffolded-not-wired; caps are non-binding for gentle strikes (per the C0 findings they'd bind at ~21× more violent, needs a trained policy to characterize).

---

## 3. What is proven (findings, high confidence)

- **Vacuity.** The impulse constraint never binds for genuine impacts on fixed impedance — worst reachable ≈36% (single-mode @4.65 rad/s), coordinated whip 39% (beyond DiffIK reach), analytic full-stop ceiling 90% (unreachable). 4 convergent CPU methods + independent re-run cross-check. **Framing caveat (defense-critical):** this is vacuity *under our specific choices* — DiffIK position-only action space, torque limits, the near-floor L6 grasp, the soft 7 g nail, and hardware-derived caps calibrated for a ~21× more violent tail the current task can't reach. It is a cap-vs-task mismatch of the project's own making, **not a law of nature**; a torque-action / heavier-striker / harder-target variant might not be vacuous. Pre-empt the examiner's "you showed *your* setup can't reach *your* caps" critique, not just "it's not a defect."
- **NEW today — the rigid-target / impedance sweep (fixes the review's C1 + I4 + I7).** Against a *genuinely rigid* target (huge-armature lock, nail moved 0.017 mm) at the reachable ~1.3 m/s strike, the **press-gated impact-only Λ/cap sits at ~0.47–0.64 across kp 0.5×→20×** and never crosses the cap. Achieved velocity stays pinned ~1.3–1.4 m/s regardless of stiffness (effort clamp). Two corollaries:
  - A rigid target roughly **triples** the impact vs the yielding nail (~0.55 vs ~0.2) — target compliance was masking ~⅔ of the impulse. The vacuity doc's last hedge ("a rigid target would bind") is now **closed: it gets much closer but still stays under cap at fixed impedance + reachable speed.**
  - The shipped accumulator reads **10.8× cap** on the *same* rigid strike (full window incl. drive-through press) vs **0.585×** for the genuine impact — an **~18× press inflation**, the sharpest possible evidence for the robot-side press-cap fix (§5).
- **δ-enforcement machinery works** (fires at cap under enforce-ON, silent under log-only control).
- **Soft-CaT velocity arm (first GPU result):** learns + complies in the mean (peak |q̇| 2.5 vs unconstrained 3.54), 100% success, fastest strike preserved — but global worst-case still exceeds the π limit (chain-coupled residual). Motivates VIC + substep impulse.
- **qfrc contaminant is dof-friction (~45%), not a weld** (training scene has neq=0). Object-side ∫F·dt ≈0.107 N·s is the clean ground truth.
- **Caps are principled hardware values** (τ_rated × 2 HD-repeated-peak × Δt_window); kept fixed by user decision.
- **The reference strike is a drive-through press, not a ballistic impact** (window 24–30 ms, flat force, no rebound); solver-sensitivity is split (timestep-robust ~5%, solref-fragile −38%) because it's a truncated press, not a released impact.
- Every "binding" result was traced to a specific artifact (sustained press or super-URDF velocity) — reported honestly.

---

## 4. Where the science lands (external literature, verified)

**Strongly CONFIRMS the architecture:**

- **CaT is the right mechanism.** Chane-Sane et al. (IROS 2024, arXiv:2403.18765) directly reformulate constraints as stochastic terminations with a graded δ and a γ(1−δ) effective discount — which the project ports. The project's positive-terms-only refinement is an application detail, not a departure.
- **A per-joint impulse cap is correctly an *instantaneous hard constraint*, outside CMDP.** CPO (Achiam 2017, arXiv:1705.10528) and the CMDP family bound *expected cumulative* cost and can violate instantaneous constraints (Shi et al. 2023, arXiv:2302.04375); a per-event impulse cap belongs to the state-wise/instantaneous-constraint class, which is precisely what CaT-via-termination encodes. Validates the `CONSTRAINED_RL_LANDSCAPE` framing. *(One over-claim was refuted: CPO does **not** guarantee per-iteration constraint satisfaction — don't cite that.)*
- **The reflected-mass physics is textbook; impedance modulates it.** Khatib operational-space reflected mass m_r=(uᵀΛ⁻¹u)⁻¹ and Kirschner/Haddadin (ISR 2021) m_r(K_J)=m_link+[K_J/(K_J+γ)]m_mot ranging from link-only (compliant) to link+motor (rigid) — **joint stiffness literally sets reflected mass** — are the textbook, load-bearing results. That peak impact force rises monotonically with m_eff is single-source (Mavrakis, IROS 2017, MEDIUM). Ghanbarzadeh & Najafi (2024) use impedance to lower m_eff and move faster at the same force-safety limit. Together these ground "the protective role belongs in variable impedance."
- **Reflected inertia is an independent protective lever alongside velocity.** Haddadin (IJRR 2012): safe v_max is a *decreasing* function of reflected mass. ISO/TS 15066 PFL: v_rel,max=F_max/√(μk). Rossi et al. (2015): an energy injury index = dissipated inelastic KE, embeddable as a constraint that *minimizes reflected mass in the impact direction*.
- **Rigid > deformable impact is directionally consistent** (single-source, qualitative, sim-vs-real confound — Mavrakis measured rigid ~100–130 N in sim vs deformable ~13–17 N on real hardware, ~7–8×). Matches the *direction* of today's rigid-target result, **not** the specific numbers.
- **MuJoCo under-resolves impulsive impacts — single-source (MEDIUM) external support.** Acosta et al. (RA-L 2022, arXiv:2110.00541) found MuJoCo the *least* accurate simulator on real impacts, with *longer contact events and lower peak forces*, tunable per-log via solref/solimp. **This is consistent with the frontier review's C2 and the project's own solver-sensitivity record** — a caution that the high-velocity impulse branch in MuJoCo needs solref/solimp convergence + hardware validation before it is trusted, not a proof.

**Two honest CHALLENGES (these are the thesis's real defense questions):**

1. **Impulse vs energy/force.** The human-safety caps are *force/energy* relations: ISO scales v∝1/√μ, injury energy ~½·m_eff·v² (velocity-*squared*). A true momentum/impulse bound scales v∝1/μ. So the "reflected-inertia × velocity" gloss reads as momentum, but the load-bearing safety literature bounds **energy/peak-force, not impulse.** The thesis bounds *impulse* (Σ|qfrc|·dt) — which is the right quantity for **gearbox / harmonic-drive reaction-torque protection** (the actual cap provenance), *not* for human injury. Be crisp: this is a **hardware-protection** impulse constraint, and its provenance (τ_rated × repeated-peak × window) is project-internal — the literature here does **not** ground harmonic-drive impact-torque caps.
2. **Emergent compliance vs explicit constraint.** Every verified VIC-for-safety result (VICES, Martín-Martín IROS 2019; Yang et al. 2021 Cartesian variable stiffness — *zero* excessive-force events on a real Franka; per-joint/per-leg stiffness, Spoljaric et al. 2025 — policy stiffens *against* a push beyond a fixed controller's max) achieves safety through **emergent compliance on quasi-static / continuous-contact tasks, with no explicit constraint.** **None** is a deliberate *impulsive strike*; **none** imposes an *explicit impulse constraint*. So the thesis's contribution is exactly the triple gap — (i) explicit impulse constraint, (ii) impulsive striking, (iii) co-trained with VIC — and its **marginal value over pure compliance is untested.** The thesis must show the explicit cap *earns its keep* over "just be compliant."

**Genuine literature GAPS (⇒ thesis novelty, but less external cover):**

- **Dynamic striking/hammering + phase-indexed reference tracking: zero verified claims.** No benchmark validates the ante-impact, phase-indexed (not clock-time) reference design — it's a project contribution. (A hammer-AMP paper exists, arXiv:2510.24257, but nothing survived verification.)
- **"When is a constraint slack/vacuous": not answered by the literature** — the vacuity observation is project-empirical (the closest formal analogue is Lagrangian complementary slackness λ=0 ⇒ non-binding).
- **Open science question that directly bears on the thesis:** does control-level (*virtual*) impedance reduction actually attenuate the sharp *initial* impulse peak, or only the post-impact quasi-static regime? Kirschner's transmission-stiffness model + ISO caveats suggest controller bandwidth and *physical* (vs virtual) compliance may bound how much of the peak VIC can shape. **If virtual VIC can't shape the impulse peak, the whole "VIC unlocks binding" premise is at risk** — this needs on-task evidence and is the single most important thing to check early in the VIC phase.

---

## 5. What is stale / owed (the debt)

**Docs lag reality:**
- `docs/README.md` dated **2026-07-05** — misses Track-2 (`contact_row_impulse.py`), the QFRC audit, the C2 gate, and the 2026-07-10 / 2026-07-12 records; still lists the retired `test_single_strike.py`; `:78`/`:82` stale/false.
- `docs/thesis/README.md` — no vacuity reframe; still frames C1/C2 as "binding, protective."
- `IMPULSE_CAT_IMPL_PLAN.md` — status "C2 enable max_p next" (superseded); §4 gate "vacuous ⇒ fail" now contradicts "vacuity is an expected task property."
- `CLAUDE.md` — calls the constraint "the thesis headline" with no vacuity caveat (pending Khadiv, so a missing caveat not a contradiction).

**Uncommitted / undone:**
- The 2026-07-12 vacuity deliverable is **uncommitted**: the record, one-pager, and assets are untracked; the `docs/results/README.md` row is an uncommitted edit to a tracked file. (This doc itself is also untracked.)
- **THE code fix that matters — ✅ DONE this session:** gave `SubstepImpulseAccumulator` the object-side `event_window` press cap. It turned out to be a coordinated **3-file** change (accumulator + the C0 derive gate's `sub_capped` mirror so the hard `ximp_err` equality compares like-with-like + a Track-2 divergence note) — *not* the "one small CPU fix" first assumed; the C0 gate would have spuriously failed on any >25-substep window otherwise (caught by the Fable-5 verification pass). Full suite (300) + C0 derive gate both green.
- C1 delivered-side flicker-press re-arm (`_event_age` resets every rising edge) — still present.
- Adversarial-review fixes **I3–I12 + minors** — deferred (metric-mean bias I3, derive-doc falsified headroom I4, tautology gate I5, non-durable C2 provenance I9, `test_contact_row_impulse.py:329` falsified invariant I10, no joint-order test I11, no eval-snapshot test I12, …).

**Blocked / pending:**
- **C3 GPU campaign** (4 arms × 3 seeds) — data-blocked on Vega SSH 2FA (copy-paste commands handed to you). Frame as compliance/headroom + negative-control + press-residual arbiter.
- **Khadiv sign-off** on the reframe (one-pager, 4 decisions a–d) — pending; not yet in the thesis digest.
- **VIC** — no code; the endgame.

**Test coverage gaps:** Cartesian obs transforms, plain reward helpers, runner/rl_cfg construction, all analysis/eval scripts (incl. the threshold-provenance `derive_impulse_thresholds.py`), and enforcement-mode (δ actually terminating) — none unit/gate-covered.

---

## 6. The honest assessment (the loop, named)

The "stuck in a loop" feeling is accurate and it maps to a specific pattern: **the machinery and the central finding are done, but they've been re-verified rather than banked and advanced.** Every recent CPU probe (whip, nail-sweep, velocity frontier, impedance) re-confirmed the same physics — the fixed-impedance Z1 on this task cannot produce a binding impact — each time with a new caveat or bug. That is genuinely informative (the vacuity finding is now airtight and the rigid-target hedge is closed), but it is not *progress toward the thesis*; it is progress toward *certainty about a negative result we already had*.

The literature makes the strategic picture crisp: the architecture is sound and the protective story provably belongs in VIC — but VIC is unbuilt, and the one deep open question (can virtual impedance shape the *impulse* peak, or only quasi-static force?) can only be answered by building it. Continuing to probe fixed impedance cannot answer it.

---

## 7. The way out — concrete next steps (revised after adversarial review)

The original "bank → Khadiv → VIC" understated the §0.6 risk and idled Khadiv's async latency. Revised, with a hard **feasibility gate before committing to VIC** and the longest-lead item fired first:

**Day 0 — fire the two longest-lever items in parallel:**

1. **VIC-feasibility ceiling (½ day, analytic — the go/no-go gate).** Before *any* `set_gains` engineering, bound the *maximum* Λ any VIC policy could reach on this platform: plug the Z1's real link + rotor inertias into the Kirschner reflected-mass model `m_r(K_J)=m_link+[K_J/(K_J+γ)]·m_mot` (§4) at the reachable ~1.3 m/s, combined with the measured effort clamp (which `set_gains` **cannot** lift). If that ceiling stays under cap, **VIC is dead-on-arrival on this platform** and the thesis needs a different lever (heavier striker / torque-action space / energy bound / hardware). A day's calc vs a quarter of engineering. Today's rigid sweep (40× stiffness → Λ 0.47→0.64) warns the ceiling may be low.
2. **Send the amended Khadiv one-pager** (longest async lead time — start now, not after banking). Add **decision (e): is the bounded quantity impulse (v∝1/μ) or energy/peak-force (v∝1/√μ)?** — settle it *before* VIC, since it changes what VIC optimizes (energy is velocity-squared-sensitive, reweighting the velocity vs reflected-mass levers). Decision (c) is now *answered*: rigid target → ~0.6× cap, still doesn't bind.

**While waiting on Khadiv — bank the debt (CPU/docs):**

3. Commit the 2026-07-12 vacuity deliverable (record + one-pager + assets; the README row is an uncommitted edit to a tracked file) + a short rigid-target/impedance addendum — local, `soft-cat`, Fable trailer.
4. Refresh `docs/README.md` to current truth (fix `:78`/`:82`; add the post-07-05 records).
5. ✅ **DONE** — robot-side accumulator **press cap** implemented (the instrument-calibration prerequisite for a clean impact-only Λ). It was a 3-file coordinated change (accumulator + C0 gate `sub_capped` mirror + Track-2 divergence note) + tests; full suite + C0 gate green.

**Before declaring vacuity settled — one falsification run (machinery already shipped):**

6. Promote a **fixed-impedance, delivered-impulse-MAXIMIZING** training run (C3's `c3_imp0` arm, already built) as the vacuity *falsification* test. Every vacuity probe so far is non-learned or reachability-bounded; the velocity soft-CaT showed a *learned* policy finds chain-coupled worst-case residuals hand-probes miss. If a policy *trained to maximize impulse* reaches the cap at fixed impedance, the "must go to VIC" motivation weakens — worth knowing before the pivot.

**Gates before ANY VIC binding claim:**

7. **Solver-convergence study** (solref/solimp/timestep): establish that impact-only Λ is a stable, converged number for an impulsive event. Λ is already −38% solref-fragile (§3) and MuJoCo under-resolves impacts (§4) — without this, a "VIC raises Λ across the cap" result is indistinguishable from solver noise.
8. **Decide hardware scope.** The cap's real provenance is hardware (harmonic-drive reaction torque); the sim's impact fidelity is impeached. State explicitly whether real-Z1 validation is in thesis scope; if not, defend the sim-only central claim (conservative caps / direction-not-magnitude).

**Then, only if the feasibility gate (1) and convergence gate (7) pass — VIC:**

9. Scope the `set_gains` action space (grouped stiffness dims per VICES/Yang/Spoljaric).
10. First VIC experiment: does dynamic impedance **modulate the impact-only Λ peak** — stiffen to raise it toward the cap, soften to attenuate it — at converged solver settings? The core thesis claim, meaningful *only* after gates 1 and 7.

**Optional backlog:** the deferred adversarial-review fixes and the branch merge/PR (`finishing-a-development-branch`) batch after banking.

---

## 8. Open questions (consolidated)

**Internal (need GPU/hardware):** does the constraint ever bind for a *learned* policy (the reward-gated press residual)? does VIC actually unlock the impulsive/binding regime? Q3/Q6/Q7/Q9/Q11/Q12 (training/hardware-gated). Is the C1 flicker-press farm reachable by a trained policy?

**External (from the literature, thesis-defense-relevant):**
- Does virtual impedance shape the *initial impulse peak* or only post-impact quasi-static force? *(the critical one — answer it first in the VIC phase)*
- Does an explicit impulse constraint beat emergent compliance (VICES/Yang got safety with none)? — the marginal-value question the thesis must win.
- Is *impulse* the right bounded quantity, or energy/peak-force? — be crisp: gearbox protection justifies impulse; human-injury literature uses energy. State the provenance.

---

## 9. VIC-feasibility ceiling — RESULT (the go/no-go gate, run 2026-07-12)

The §7 gate is now executed (`docs/results/assets/2026-07-12_impulse_vacuity/probes/ceiling.py` + `armature_check.py`; independently re-derived + reproduced by an Opus verification pass). Reflected mass from the sim's mass matrix + head Jacobian at the strike pose; per-joint ceiling Λ_j = |(Jᵀu)_j|·m_eff·v·(1+e).

**Numbers.** m_eff = **0.536 kg** (armature-coupled) / 0.401 kg (link-only) → reflected-mass lever only **1.33× as modeled**. Load joints: j2 (moment arm 0.50 m), j3 (0.43 m). Reachable head speed ~1.35 m/s; kinematic ceiling (all 6 joints at 3.14 rad/s, aligned) 3.97 m/s. Worst Λ/cap **at the reachable speed: 0.19 (inelastic) / 0.38 (elastic e=1)**. Crossing velocity: 7.13 m/s inelastic (*above* the kinematic ceiling → unreachable by any strike) / 3.57 m/s elastic (only in the unreachable coordinated-whip regime). Cross-check: m_eff = 0.54 matches the vacuity doc's independent analytic 0.49–0.55. The impedance sweep's apparent m_eff 2.9 kg / Λ 0.585 was **press-contaminated even in the impact-only window** (active kp force during deceleration): 2.9 kg exceeds even the ×30-armature reflected mass (1.39 kg), so it cannot be a physical reflected mass at any plausible armature — it must contain sustained press.

**Verdict — VIC-as-commanded-stiffness is near-dead for making the *impulse* constraint bind, for a structural reason.** Reflected mass lives in `M(q)`+`dof_armature` (a physical model property); `set_gains` changes actuator kp (control), which does **not** enter `M(q)`. So control-stiffness adds an *active-press* force during contact (the contaminant we exclude), never *passive reflected mass*. VIC therefore has ~zero authority over the ballistic impulse; the only binding lever is *velocity*, which is effort-clamped and is not an impedance function. This answers the literature's open question (§8) for a rigid-transmission arm: **virtual (control-level) impedance does not shape the ballistic impulse peak — only the quasi-static/press force.** The Kirschner "stiffness sets reflected mass" lever is a *physical*-transmission property the Z1 has fixed and the policy cannot command.

**Armature-fidelity caveat (important, bears on BOTH vacuity and VIC).** The modeled `dof_armature` = [0.01, 0.02, 0.01, 0.01, 0.01, 0.01] kg·m² look like **nominal placeholders**, not values derived from rotor inertia × gear² (a harmonic drive → realistic ~0.05–2). At the load joints, armature is only ~4–6% of joint inertia. If the real value is 10–30× larger, m_eff rises to **0.89–1.39 kg** and the reflected-mass lever to **~2.2–3.5×** — and the reachable-velocity elastic Λ/cap rises from 0.38 to **~0.6–1.0 (near-binding at the ×30 end)**. Consequences: (1) the **vacuity margin (36% of cap) is armature-sensitive** and may be an under-estimate on real hardware; (2) the VIC verdict's *magnitude* shifts, though (3) `set_gains` still cannot command it. **Verify/fix the arm armature against real Z1 rotor specs before treating either the vacuity margin or the VIC verdict as final** — now a high-priority model-fidelity item.

**A sharper corollary (from verification — the important one).** For an *impulsive* strike, commanded-VIC shapes **none** of the impact metrics: impulse, energy (½·m_eff·v²), *and* peak force are all governed by the fixed reflected mass and the effort-clamped velocity. VIC's only authority is over the **sustained / quasi-static contact force** — the active-press phase *after* the ballistic impact (and the contact-time that sets peak force is a solver property, not an arm one). So "re-point to energy/force and VIC becomes load-bearing" does **not** hold for the brief impulsive strike.

**The fork (Khadiv decision, see `2026-07-12_khadiv_vic_addendum.md`):** (A) re-point the bound to the **sustained/quasi-static contact force** VIC *does* shape — but this rescues VIC only for a contact-dominated task emphasis, not the impulsive strike, and it is a *different* physical quantity than the gearbox impact-torque spike the impulse cap targets. (B) change the **physical lever** — heavier striker / a torque action space reaching higher velocity / physical variable-stiffness hardware (all raise physical m_eff·v, which the policy still doesn't command via kp, but at least make the constraint active). (C) **reframe** as a characterization result: the machinery + the rigorous demonstration that commanded-VIC cannot shape an impulsive impact on a rigid-transmission arm — a real finding mapping to the literature's open question. **No option is a clean "VIC rescues the impulsive-strike thesis"; that specific arc likely does not hold on the Z1.**

---

## Appendix — verified citations

- **CaT:** Chane-Sane et al., *Constraints as Terminations for Legged Locomotion RL*, IROS 2024 — arXiv:2403.18765.
- **CMDP/CPO:** Achiam et al., *Constrained Policy Optimization*, ICML 2017 — arXiv:1705.10528. Shi et al. 2023 (instantaneous hard constraints) — arXiv:2302.04375. Cumulative-cost CMDP — arXiv:1802.06480 (verified as a cumulative-constraint source; canonical CMDP text is Altman 1999, a book, *not* this preprint).
- **VIC-RL:** Martín-Martín et al., *VICES*, IROS 2019 — arXiv:1906.08880. Yang et al. 2021, Cartesian variable stiffness (Örebro/Lund) — diva2:1620121. Spoljaric/Yan/Lee 2025, per-joint/per-leg stiffness — arXiv:2502.09436.
- **Effective/reflected mass + safety:** Kirschner/Haddadin, reflected mass vs joint stiffness, ISR 2021 — iliad-project.eu "Notion on the correct use…". Mavrakis et al., IROS 2017 — arXiv:1707.08150. Ghanbarzadeh & Najafi 2024 — arXiv:2311.13814. Haddadin et al., IJRR 2012 — DOI 10.1177/0278364912462256. Rossi et al. 2015, energy injury index (ResearchGate 282643108).
- **MuJoCo contact fidelity:** Acosta, Yang & Posa, *Validating Robotics Simulators on Real-World Impacts*, RA-L 2022 — arXiv:2110.00541.
- **Supporting context (found but outside the 24 verified claims — cite with care):** Ding & Thomas, *Improving Safety and Accuracy of Impedance-Controlled Manipulators…*, ICRA 2021 (ResearchGate 349711874).
- **Refuted (do not cite):** "CPO guarantees near-constraint satisfaction at each iteration" — killed 0-3.

*Scope caveats from the research pass:* VIC/impedance-safety papers are all quasi-static/continuous-contact or accidental-collision, none impulsive striking with an explicit constraint (three 2-1 verifier splits were about this analogical reach). Single-source findings (rigid-vs-deformable, MuJoCo fidelity, impedance tradeoff) rated medium. Project findings (velocity-effort-clamp; the specific ~3×/~0.6×-cap numbers) are project-empirical — the literature supports mechanism and direction, not the numbers.
