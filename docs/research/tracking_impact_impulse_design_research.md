> ⚠️ **DATED RESEARCH RECORD** (2026-06-10 design research) — kept for provenance.
> Facts below reflect their date and may contradict the current code; check `docs/README.md`.
>
> **Impulse-threshold provenance correction (2026-08-14):** The generic Harmonic Drive `2x` repeated-peak argument below is dated research rationale, not a validated Z1 limit. Unitree's published actuator maximum/URDF effort values do not establish allowable external reaction impulse or damage. The historical body is preserved; see `reward-design/IMPULSE_CAP_PROVENANCE.md`.

# Design Research — Track a Reference, Maximize Impact, Bound Joint Impulse

**Pipeline:** ARS `deep-research` (`full` mode, design-research variant) — 6 parallel literature-search agents + lead verification + inline DA/editorial checkpoints.
**Date:** 2026-06-10 · **Branch:** `hammer-z1` · **Requested by:** user directive "make our environment such that with the RL policy we can follow the given trajectory and at the same time have max force and minimize the impulse on the joints."
**Companion implementation plan:** `../archive/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md`.

> ⚠️ **EVIDENCE STATUS & SCOPE BANNER** (lesson from `../archive/PEER_REVIEW_v2.md` Tier-1 #3). This is a *design* document grounded in verified external literature and the local corpus. **No RL training was run for it.** Every design recommendation below is *hypothesized-untested* until the ablations in the companion plan are executed. Claims about what the literature reports are verified; claims about what will work in *our* env are predictions.

> **Pipeline deviations (disclosed):** (1) The user pre-authorized end-to-end execution ("go all in"), so the Phase-1→2 user-confirmation gate was skipped; the RQ brief is in §1 for retroactive review. (2) Two session-limit interruptions occurred mid-investigation; all six agents ultimately completed. (3) Source verification: every agent was required to open every source it reported (WebFetch); the lead re-verified the two most load-bearing net-new citations (Varin 2019, Freitag 2026). Tags: `[E1]`/`[E2]` = venue-confirmed / arXiv-only, opened by the reporting agent this session; `[E1†]` = additionally re-fetched by the lead; `[E3]` = inherited from the verified local corpus (`reward-design/LITERATURE.md`, audited 2026-06-10 — see session audit); `[DOCS]` = official documentation.

---

## 1. Research question (Phase 1 — Scoping)

**RQ.** How should an mjlab RL environment and training stack be designed so the policy simultaneously (a) **follows a given reference strike trajectory**, (b) **maximizes delivered impact** — operationalized as axial momentum `m_eff·v` / hitting flux, *not* commanded or simulated contact force — and (c) **bounds per-joint impulse**, for two instantiations: **A** = the current position-only DiffIK Z1 hammer env (fixed PD), **B** = the thesis G1 variable-impedance action space?

**Sub-questions.** SQ1: Where does the reference come from and how is it parameterized? SQ2: How strongly should tracking bind when the policy must *exceed* the reference? SQ3: How is per-joint impulse measured in MuJoCo/mjlab and turned into a penalty → constraint → filter? SQ4: How is the impact objective formulated and gated per action space? SQ5: How is `R = w_I·r_imit + w_G·(r_impact − λ·r_recoil) + r_safety` balanced, annealed, and validated without training?

**FINER:** Feasible (sim-only, existing stack) ✓ · Interesting ✓ · Novel (two explicit gaps found — §4.6) ✓ · Ethical (sim robotics; dual-use n/a) ✓ · Relevant (bridges Phase 0 → thesis) ✓.

**In scope:** reward/constraint/action-space design, reference generation, measurement, validation-without-training, ablation design. **Out of scope:** PPO hyperparameters, CBO-vs-PPO optimizer question (thesis ablation), sim-to-real calibration (Q7), hardware deployment.

**Constraint from the supervisor record** (`thesis_direction_update.md`): strong generate-then-track was walked back for the thesis; tracking strength is treated here as an **explicit design axis with evidence**, not assumed at either extreme.

**DA Checkpoint 1 (PASS, with bounds):** RQ is answerable at design level without GPU; all learnability claims must be routed to the ablation table, not asserted. Scope risk on SQ3/SQ4 bounded to "implementable in mjlab/rsl-rl now".

---

## 2. Evidence by axis (Phase 2 — Investigation; 6 agents, ~49 sources, all opened)

### 2.1 Tracking *through* an impact (the part the existing spec missed)

- **Biemond, van de Wouw, Heemels & Nijmeijer (2013), IEEE TAC 58(4), DOI 10.1109/TAC.2012.2223351 `[E1]`** — foundational negative result: when plant and reference jump at different instants, the conventional tracking error **cannot converge**; it peaks for arbitrarily small initial errors. A time-indexed velocity-tracking reward near contact is structurally noisy for *any* policy.
- **Rijnen, Saccon & Nijmeijer (2020), IEEE TCST 28(3) `[E1]`** — *reference spreading* (RS): ante- and post-impact references extended past nominal contact, switched on impact detection; hardware-validated, robust to delayed detection and corrupted velocity estimates.
- **van Steen, van de Wouw & Saccon (2022), ACC, arXiv:2111.05211 `[E1]`** — RS with an **interim mode**: between first and last impact, velocity feedback is *disabled entirely* (velocity error is meaningless mid-impact).
- **van Steen et al. (2023), IFAC WC, arXiv:2212.00877 `[E1]`** and **van Steen et al. (2024), arXiv:2411.09870 `[E2]`** — **time-invariant RS**: references become velocity vector fields parameterized by state, not clock time, so contact-timing mismatch no longer corrupts the reference; 600 real dual-arm experiments.
- **Yang & Posa (2021), IROS, arXiv:2103.06907 `[E1]`; (2023/24), arXiv:2303.00817 `[E2]`** — **impact-invariant projection**: track only the velocity components invariant to the impulsive contact force; necessary for Cassie hardware deployment; robust to impact timing by construction.
- **Mészáros, Franzese & Kober (2022), RA-L, arXiv:2110.04534 `[E1]`** — time-invariant (position-conditioned) LfD for non-zero-velocity contact.
- **Negative finding `[agent-verified]`:** no published **RL** version of reference spreading was found in any search. Nearest prior art is Mészáros (LfD). Unverified lead: Westervelt/Grizzle/Koditschek HZD 2003 (phase-variable references; metadata-only).

### 2.2 How strongly to bind to the reference (strong ↔ weak continuum)

- **Silver et al. (2018), arXiv:1812.06298 `[E2]`; Johannink et al. (2019), arXiv:1812.03201 `[E2]`** — *residual policy learning*: the reference lives in the **action space** (policy adds a bounded correction to a base controller), reward stays task-side. Exceeding the reference is structural; degenerate tracking cannot occur.
- **Ranjbar et al. (2021), arXiv:2106.04306 `[E2]`** — caution: naive additive residuals fight the base controller's internal feedback loops; inject the residual at the *reference/target* level.
- **ResMimic (Zhao et al. 2025), arXiv:2510.05070 `[E2]`** — documents exactly the feared failure: a tracking-only policy "lacks the precision and object awareness required" for loco-manipulation; fixed by a residual *task* policy, not a heavier prior.
- **Freitag, Åkesson & Chehreghani (2026), arXiv:2603.05113 `[E1†]`** — two-stage reward curriculum with explicit step/linear/cosine annealing ablation: schedule **shape barely matters** (longer slightly better; linear/200k chosen); **continuous sample reuse across the weight change is critical**; curriculum beats training on the full reward from scratch and is more robust to weightings.
- **ExBody (Cheng et al. 2024), arXiv:2402.16796 `[E2]`** — *selective* tracking: strictly imitate only the DOFs the task does not own; relax the rest. A per-DOF prior instead of one global imitation weight.
- **PGDM (Dasari, Gupta & Kumar 2023), ICRA, arXiv:2209.11221 `[E1]`** — the manipulation analogue of RSI: reference used for *initialization* (pre-grasp) plus an object-side objective; 50 dexterous tasks with no per-task reward engineering and **no joint-space tracking term at all**.
- **UniBYD (Yuan et al. 2025), arXiv:2512.11609 `[E2]`** — anneals imitation→task explicitly so policies *surpass* demonstrations.
- **Continuum verdict (agent + lead concur):** *no verified source supports a strong fixed ~0.7 imitation weight when the task requires exceeding the reference.* Evidence clusters at (i) structural/weak priors — residual action spaces, RSI-only, selective per-DOF tracking — or (ii) strong-to-weak **annealed** schedules with continuous data reuse.

### 2.3 Measuring impulse; deriving safe bounds (instantiation-critical)

- **MuJoCo docs (Computation; Modeling) `[DOCS]`** — `efc_force` are **forces (N), not impulses**; joint-space mapping is `qfrc_constraint = Jᵀ·efc_force`. Per-strike impulse must be accumulated as Σ F·h **across 500 Hz substeps** — sampling at the 50 Hz control rate aliases short impacts. Restitution is not a parameter; bounce emerges from `solref/solimp`.
- **Acosta, Yang & Posa (2022), RA-L, arXiv:2110.00541 `[E1]`** — MuJoCo/Drake/Bullet vs real impact data: **inelastic impacts are captured well; elastic impacts are not**. **SimBenchmark (ETH RSL) `[DOCS]`** concurs: MuJoCo "cannot simulate elastic collision"; no solref↔restitution correspondence.
- **Aouaj, Padois & Saccon (2021), ICRA, arXiv:2010.08220 `[E1]`** — hardware validation of rigid impact maps for arm impacts; real post-impact joint velocity is a **damped oscillation**, so Δq̇ must be extracted over a defined **window**, not at one sample.
- **Pinocchio `impulseDynamics` API `[DOCS]`** — rigid impact map with restitution `r_coeff` (default 0 = inelastic); the cross-validation target for sim impulse numbers (match assumptions: inelastic on both sides).
- **Harmonic Drive CSF/CSG catalog `[DOCS]`** — the engineering basis for per-joint impulse limits: rating ladder **Rated < Repeated-Peak (~2×) < Momentary-Peak (~4×, "must not occur during normal operation") < Ratcheting**, plus a **lifetime impact-event budget** (N = 10⁴ class events, catalog Eq. 7). **Unitree Z1 specs `[DOCS]`:** 33 N·m peak, harmonic reducers (~60:1).
- **Implication:** a *hammering* policy performs impacts as its **normal operating cycle**, so nominal strikes must sit under the **Repeated-Peak** rating (≈2× rated torque), not the collision rating — the constraint threshold is ~2× stricter than a naive "collision limit" reading, and repetition makes it a fatigue *budget*, arguing for per-strike impulse costs.

### 2.4 Constraint machinery in practice (penalty → CMDP → filter)

- **Ray, Achiam & Amodei (2019), OpenAI `[E1]`** — episodic-sum cost convention; **Lagrangian methods beat CPO** empirically; 3-metric protocol (return, satisfaction, cost regret).
- **Spoor et al. (2025/26), arXiv:2510.17564 `[E1]`** — practice paper: PID multipliers can overshoot cost limits, gradient-ascent multipliers oscillate, high seed variance everywhere. Recipes: **sweep fixed λ first** to map the return–cost Pareto; **train with stricter limits than the deployment target**; pick limits per regime.
- **OmniSafe on-policy benchmark `[DOCS]`** — tuned PPOLag/CPPOPID defaults and replication scripts; the starting hyperparameters before re-implementing in rsl-rl.
- **Saute RL (Sootla et al. 2022), ICML, arXiv:2202.06558 `[E1]`** — safety budget in the observation; almost-sure (not in-expectation) satisfaction; env-wrapper only — cheap to try in mjlab. **TRC (Kim & Oh 2022), RA-L, arXiv:2312.00344 `[E1]`** — CVaR-on-cost for heavy tails.
- **CBF-RL (Yang et al. 2025), arXiv:2510.14959 `[E1]`** — safety-filter the rollouts **during training**, deploy **filter-free on a Unitree G1**; no reported task-performance sacrifice. **Oh et al. (2025), arXiv:2510.18082 `[E1]`** — proof that a sufficiently permissive filter does not degrade asymptotic return. **ContactRL (Mulkana et al. 2025), arXiv:2512.03707 `[E1]`** — kinetic-energy CBF shield keeping real contact forces <10 N (UR3e); closest template for an impulse-aware shield.
- **Kim et al. (2024), IEEE T-RO, arXiv:2308.12517 `[E1]`** — constraints replace penalty stacks for legged robots; one reward coefficient left to tune; strongest practice evidence for the constraint formulation per se.

### 2.5 Impact objectives in learned striking (2024–2026)

- **Khurana & Billard (2024), ICRA-WS Agile Robotics, EPFL Infoscience `[E1]`** — hitting flux generalized to any link: `φ_h = λ_h/(λ_h+m_o) · ẋ⁻`, with directional effective inertia `λ_h = (ĥᵀΛ⁻¹ĥ)⁻¹`, `Λ = (J M⁻¹ Jᵀ)⁻¹`; flux is **restitution-independent** and posture-selectable. **Khurana, Hermus, Gautier & Billard (2025), RA-L 10(5) `[E1]`** — flux as *the* scalar hitting-control parameter; learned stochastic flux→displacement map, inverted for planning sequential hits.
- **Ma, Cramariuc, Farshidian & Hutter (2025), Science Robotics 10:eadu3922, arXiv:2505.22974 `[E1]`** — badminton on ANYmal: strike rewards **gated by a Dirac window at the predicted interception time** (single step); arm motor current bounded by an **N-P3O constraint, not a penalty**.
- **HITTER (Su et al. 2025), arXiv:2508.21043 `[E1]`** — humanoid table tennis: racket position/velocity tracking rewards are **sparse, high-weight, active only in a short window around hit time**.
- **Liu et al. (2025), arXiv:2511.11218 `[E1]`** — humanoid badminton, 19.1 m/s shuttle: **backswing/wind-up emerges from staging alone**, no explicit wind-up reward.
- **Yuan et al. (2025), Frontiers in Neurorobotics `[E1]`** — imitation→relaxation: unimodal imitation reward first, then sparse event-gated strike terms (250·r_hit) — a tested escape from local optima for sparse striking.
- **RoboStriker (Yin et al. 2026), arXiv:2601.22517 `[E1]`** — hit reward double-gated on geometric preconditions (range + facing), the field-standard anti-farming pattern (validates our depth+contact double gate).
- **Wang et al. (2025), arXiv:2510.09543 `[E1]`** — configuration-dependent **Impact Mitigation Factor as an RL reward term** — proof the `J,M` inertia machinery is practical inside RL rewards.

### 2.6 Variable-impedance action spaces (instantiation B)

- **VICES (Martín-Martín et al. 2019), IROS, arXiv:1906.08880 `[E1]`** — per-step stiffness in the action space beats joint/torque spaces on contact tasks; transfers sim-to-real.
- **Varin, Grossman & Kuindersma (2019), IROS, arXiv:1908.08659 `[E1†]`** — action-space comparison **including a hammering task**: impedance-controller references learned fastest **across all tasks and algorithms**. *The single most direct external justification for the thesis action-space switch.*
- **Aljalbout et al. (2024), RA-L, arXiv:2312.03673 `[E1]`** — 13 action spaces, 250+ agents: action-space characteristics dominate sim-to-real transfer.
- **Beltran-Hernandez et al. (2020), RA-L, arXiv:2003.00628 `[E1]`** — RL outputs time-varying PD/admittance gains **on a stiff position-controlled robot** — the gain-modulation recipe for Z1/G1-class hardware.
- **Khader et al. (2021), RA-L, arXiv:2004.10886 `[E1]`** — all-the-time-stability-certified variable impedance. **Zhang et al. (2025), arXiv:2503.00287 `[E1]`** — learned VIC policies routinely violate passivity at deployment; passivity-aware training + deployment filter fixes it.
- **Spoljaric, Yan & Lee (2025), arXiv:2502.09436 `[E1]`** — stiffness in the action space under **repeated foot impacts**; per-leg grouping beat position-only control; nearest published relative of our striking case.
- **robosuite controller docs; Isaac Lab `JointImpedanceController` docs; MuJoCo integrator docs `[DOCS]`** — implementation pattern: `variable_kp` mode with **kd slaved to critical damping (kd = 2√kp)** halves the added action dims; compute PD torque **in the env loop** (not by mutating native actuator gains); prefer `implicitfast` integration when kd is high.
- **Negative finding `[agent-verified]`:** no published **learned low–high–low stiffness schedule for striking**. Nearest is Spoljaric (locomotion).

---

## 3. Synthesis (Phase 3) — answers to the sub-questions

**SQ1 — Reference.** Start hardcoded (lift → pause → max-descent along the nail axis), generated through the existing IK; parameterize the descent by a **monotone task variable** (hammer–nail distance), not clock time (time-invariant RS; HZD precedent), with the wind-up segment time-indexed (no contact risk there). Upgrade path: Vu-style QP / Ti-style iLQR co-optimizing `m_eff·v` under joint-impulse bounds `[E3]` — an *easier* TO than SURE since it carries no timing robustness. Q1 (`test_single_strike.py`) decides single-strike vs cyclic reference; the generator must be parameterized by `n_strikes`.

**SQ2 — Tracking strength.** The evidence is one-directional: for exceed-the-reference tasks, bind **weakly or structurally**. Three graded mechanisms, in increasing coupling: (i) **RSI + object-side objective only** (PGDM) — reference anchors exploration via initialization, zero tracking term; (ii) **residual action space** (Silver/Johannink/ResMimic) — base action *is* the reference, policy adds bounded deltas; degenerate tracking structurally impossible; inject at the DiffIK-target level (Ranjbar); (iii) **weak reward-level prior** — position-only near contact (Biemond), per-DOF selective (ExBody), annealed (Freitag: shape doesn't matter, keep data continuous). This maps cleanly onto the supervisor's "weak motion prior": (ii) for the Z1 instantiation (cleanest on position-only DiffIK), (iii) for the thesis G1 phrasing, and the (ii)-vs-(iii) comparison is itself a legitimate ablation.

**SQ3 — Impulse.** Measure per-joint impulse as Σ|qfrc_constraint_j|·h **accumulated at the 500 Hz inner loop** over a **contact-anchored window** (Aouaj windowing; MuJoCo docs); keep contacts inelastic (Acosta; SimBenchmark) and cross-validate against Pinocchio `impulseDynamics` at `r_coeff=0` (thesis step). Thresholds from the Harmonic-Drive ratings ladder instantiated with Unitree specs — **Repeated-Peak (~2× rated) for nominal strikes**, with a 10⁴-event fatigue budget. Enforcement escalation: log-only → **excess-over-threshold penalty** → episodic-sum **PID-Lagrangian** (sweep fixed λ first; train stricter than deployment; OmniSafe defaults) → tails: CVaR/Saute → hardware: training-time CBF filtering (deploys filter-free on G1) or runtime QP shield.

**SQ4 — Impact objective.** Event-anchored and double-gated is the field standard (Ma; HITTER; RoboStriker — validates the existing `impact_progress` design). Upgrade the *quantity* from instantaneous momentum proxy to **windowed axial impulse** `I = ∫_W F_axial dt`, anchored at `first_contact`, **one payout per contact event**, gated on depth advance. On the Z1, posture at contact is nearly fixed by task geometry → `m_eff ≈ const`, so the **velocity/impulse proxy suffices; do not pay the J/M cost**. On the G1, use **flux** `φ_h = λ_h/(λ_h+m_o)·v` (Khurana 2024/25) — restitution-independent, posture-aware, and cheap from `J, M`. Do **not** add a backswing/wind-up reward preemptively (it emerges; Liu 2025) — consistent with the repo's augment-not-replace rule and Q1 gating.

**SQ5 — Balance & validation.** The deepest correction to the existing spec: **`r_impact` and `r_recoil = ‖Δq̇‖²` fight head-on** — both scale with `m_eff·v` (corpus litreview Q4), and raw ‖Δq̇‖² penalizes the *physically unavoidable* jump (Biemond; Yang & Posa). A weighted sum just trades them at a λ-dependent rate. The principled formulation is **bounded maximization**: maximize impact **subject to** per-joint impulse ≤ threshold (CMDP), with the penalty stage only as scaffolding — and if a penalty is used, penalize the **excess** `Σ_j max(0, Λ_j − Λ_safe,j)²`, which is silent until limits are approached. If a velocity-jump penalty is ever used, make it impact-map-aware: penalize deviation from the *predicted* post-impact velocity, or project onto the impact-invariant subspace (Yang & Posa). Anneal any imitation weight linearly (shape immaterial — Freitag). Every term gets a state-injected `validate_rewards.py` phase before any GPU run (house discipline, reaffirmed).

### 3.1 Contradiction table

| Tension | Sources | Resolution |
|---|---|---|
| Strong imitation (DeepMimic 0.7) vs weak prior | DeepMimic/HMAMP `[E3]` vs §2.2 cluster | Strong weights come from *style-fidelity* settings; for exceed-the-reference, no source supports them. Use structural/weak/annealed. Supervisor's walk-back is independently supported. |
| Penalize ‖Δq̇‖² vs maximize impact | litreview Q4 `[E3]`; Biemond; Yang & Posa | Bounded maximization (constraint), excess-form penalty as scaffold; never raw ‖Δq̇‖² at meaningful weight. |
| Windowed ∫F dt (measured outcome) vs force-is-a-solver-artifact | v2 deep dive `[E3]` vs Acosta | Both hold: *instantaneous* force is artifact-prone; *windowed impulse* in the inelastic regime is the quantity sims get right. Keep the depth gate as the geometric cross-check. |
| PID-Lagrangian as default vs its failure modes | Stooke `[E3]` vs Spoor | Keep the escalation, but enter it via a fixed-λ sweep and stricter-than-target training limits. |
| Filter costs performance? | folk concern vs Oh 2025 / CBF-RL | Evidence says a permissive filter costs ~nothing and can be applied during training only. |

### 3.2 DA Checkpoint 2 (PASS with flags)

No cherry-picking detected across agents (each axis includes negative findings and cautions). Flags carried into §5: window length W is a new magic constant (derive from logged strike data — the 4 mm-dead-zone lesson, R1-W7); residual policies can *undo* the swing if task rewards still prefer pressing (press-watchdog metric mandatory; Path A physics fixes are a prerequisite, not an option); Freitag's sample-reuse warning applies to any mid-run weight change.

---

## 4. Recommended architecture (Phase 4 — the design)

### 4.1 Decision summary

| # | Decision | Choice (Z1 / Phase-0) | Thesis (G1) variant | Evidence |
|---|---|---|---|---|
| D1 | How tracking enters | ~~Residual action space~~ → **superseded 2026-06-10 (user decision): weak annealed reward-level prior (A-PRIOR) is primary on the Z1**; residual demoted to optional ablation — see plan changelog | **Weak annealed reward-level prior** + RSI (supervisor's phrasing); residual-vs-reward is an ablation | §2.2 |
| D2 | Reference source | Hardcoded lift→strike via IK, phase = hammer–nail distance on descent | TO upgrade (QP/iLQR on `m_eff·v` s.t. impulse) | §2.1, `[E3]` |
| D3 | Impact term | **Windowed axial impulse**, contact-anchored, one payout/event, depth-gated (momentum proxy kept as ablation arm) | **Hitting flux** `φ_h` target/reward from `J,M` | §2.5, §2.3 |
| D4 | Recoil/impulse | Log per-joint `Λ_j` from day 1 → **excess-over-threshold penalty** (thresholds from ratings ladder) | **CMDP**: episodic PID-Lagrangian; CVaR if heavy tails; CBF-filtered training pre-hardware | §2.3, §2.4 |
| D5 | Tracking near contact | Position-only imitation in a contact-anchored interim window; never clock-time velocity tracking | Same + reference spreading (ante/post) — *first RL-RS instantiation* | §2.1 |
| D6 | G1 action space | n/a | `q_des + kp` per joint (**kd = 2√kp**), bounds + rate limit, env-loop PD torque, `implicitfast`, passivity filter pre-hardware | §2.6 |

### 4.2 The reward/constraint stack (target form)

```
R = w_I(t)·r_imit              # Z1: ≈0 (tracking is structural, residual); G1: weak, annealed, position-only near contact
  + w_G·r_impact               # windowed axial impulse / flux, event-anchored, depth-gated, one payout per contact
  + r_safety                   # existing action_rate, joint_pos_limits, completion (+ optional −0.1/step time penalty)
  − λ·Σ_j max(0, Λ_j − Λ_safe,j)²   # scaffold only…
s.t. E[Σ_episode Λ_j] ≤ d_j         # …target form: per-joint episodic impulse constraint (PID-Lagrangian)
```

`Λ_j` = per-joint impulse over the contact-anchored window, accumulated at substep rate; `Λ_safe,j` from Repeated-Peak torque × impact duration with the 10⁴-event fatigue budget noted.

### 4.3 What this dissolves

- **The press**: Path-A physics fixes make pressing dynamically unprofitable; the bounded one-payout-per-event impulse window caps what a press can earn; the residual's base action is a swing. Three independent guards, each measurable (press-watchdog: fraction of successes with exactly one contact event).
- **Degenerate tracking**: structurally impossible under D1 (Z1); bounded by annealing + per-DOF selectivity (G1).
- **The impact↔recoil standoff**: bounded maximization replaces scalarized trade-off; the policy is free to maximize flux until a *named, hardware-derived* limit binds.

---

## 5. Adversarial review (Phase 5 — inline panel)

**DA findings (all carried into the plan as gates):**
1. **[MAJOR, addressed]** Window length W and `Λ_safe,j` are new magic constants → both get derivation recipes (logged strike histograms; ratings-ladder arithmetic) before any training conclusion. No hardcoding without a logged justification.
2. **[MAJOR, addressed]** A residual policy with task-dominant rewards can learn `a ≈ −a_ref` (undo the swing, press anyway) if pressing remains profitable → press-watchdog metric is a mandatory logged metric in every run; Path A is a hard prerequisite.
3. **[MINOR]** Windowed ∫F dt remains solver-derived; the depth gate is the geometric cross-check, and the momentum-proxy arm (current `impact_progress`) stays in the ablation as the artifact-immune fallback.
4. **[MINOR]** `J,M` extraction cost in mujoco_warp is unbenchmarked (carried over from v2 §7) — irrelevant for Z1 (D3 avoids it), must be benchmarked before G1 flux rewards.
5. **[NOTE]** Freitag/UniBYD annealing evidence is off-policy/mixed-stack; with on-policy PPO the sample-reuse caveat is weaker but weight changes mid-run still shift the value target — prefer fresh runs per ablation arm.

**Editorial verdict (EIC-style): ACCEPT as pre-registration.** Claims are tagged, contradictions surfaced, every recommendation has a falsifying experiment in the companion plan. Nothing herein is claimable as a result until ablations run.
**Ethics: CLEARED** (sim robotics; AI-assisted research disclosed below; no dual-use concern beyond standard manipulator safety, which the work *improves*).

---

## 6. Open questions created (training- or hardware-gated)

- **Q13:** Does the residual policy beat the weak-reward-prior policy on (success, delivered impulse, press-watchdog)? [D1 ablation]
- **Q14:** Does the windowed-impulse term outperform the momentum proxy, or do they co-move? [D3 ablation]
- **Q15:** At what λ does the excess-penalty start trading impact for impulse, and does PID-Lagrangian find a better operating point than the best fixed λ? [D4; run the fixed-λ sweep first]
- **Q16:** Window length W: distribution of strike force-pulse durations from logged rollouts → set W; re-check after physics fixes (depends on Path A + Q1).
- **Q17 (G1):** `J,M`/flux computation cost per step in mujoco_warp; critic-only fallback threshold.
- **Q18 (G1):** Does low–high–low stiffness emerge with flux + impulse-constraint alone (no schedule shaping)? — the thesis's headline empirical question; §2.6 confirms it is unpublished.

---

## 7. Limitations & AI disclosure

No training was run; all "will work" statements are predictions with named falsifiers. Paywalled items excluded or tagged (Dehio & Kheddar ICRA-2021 lead dropped; Westervelt HZD metadata-only; Khurana IROS-2025 link blocked). Subagent-opened sources not re-fetched by the lead are tagged `[E1]`/`[E2]` without dagger; two were lead-re-verified `[E1†]`. Compiled with AI assistance (Claude Fable 5 orchestration; six Fable search subagents; every reported source opened by its reporting agent; unverifiable items dropped or quarantined per IRON RULE #4).

## 8. Reference index (net-new this report)

§2.1: TAC 10.1109/TAC.2012.2223351 · TCST 10.1109/TCST.2019.2898953 · arXiv 2111.05211, 2212.00877, 2411.09870, 2103.06907, 2303.00817, 2110.04534. §2.2: 1812.06298, 1812.03201, 2106.04306, 2510.05070, 2603.05113, 2402.16796, 2209.11221, 2512.11609. §2.3: 2110.00541, 2010.08220 + MuJoCo/Pinocchio/SimBenchmark/HarmonicDrive/Unitree docs. §2.4: 2510.17564, 2202.06558, 2312.00344, 2510.14959, 2510.18082, 2512.03707, 2308.12517 + OmniSafe docs + Ray/Achiam/Amodei 2019. §2.5: Khurana ICRA-WS-2024 (EPFL Infoscience), RA-L 10.1109/LRA.2025.3548496, 2505.22974, 2508.21043, 2511.11218, Frontiers 10.3389/fnbot.2025.1649870, 2601.22517, 2510.09543. §2.6: 1906.08880, 1908.08659, 2312.03673, 2003.00628, 2004.10886, 2503.00287, 2502.09436 + robosuite/IsaacLab/MuJoCo docs.
