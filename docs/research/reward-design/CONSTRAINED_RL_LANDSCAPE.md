# Constrained-RL landscape — which constraint mechanism, and do we need the heavy math?

**Status:** literature synthesis (web-grounded), 2026-06-17. Decision-oriented, thesis-facing.
**Question:** not "what is the best constrained-RL method in general," but "what is the right
mechanism for *our* constraint — a per-joint, per-substep (500 Hz) worst-case **impulse** limit plus
a per-joint **velocity box** (|q̇| ≤ 3.1415 rad/s) — and does it justify Lagrangian/CMDP machinery?"
**Verdict (up front):** **No** for the incentive layer (keep CaT); **qualified yes** for a *capability*
layer (VIC / action-projection / CBF) — but **never** Lagrangian, because Lagrangian solves a
different problem (a cumulative-cost budget), not our per-step worst case.

> ✅ **Citations verified (2026-06-17, web-grounded arXiv fetch).** Every arXiv id below resolves to a
> real paper matching the claimed authors/subject — **no fabricated ids.** Four entries had a wrong
> title/year string (now corrected inline in §5): **1907.07500** = "Learning Variable Impedance Control
> for Contact *Sensitive* Tasks" (no hyphen), v1 **2019** (RA-L 2020 journal); **2404.09080** =
> "Safe Reinforcement Learning on the Constraint Manifold: Theory and Applications" (Liu, Bou-Ammar,
> Peters, Tateo — *ATACOM* is the method, not the title); **2411.09870** = "Impact-Aware Control using
> *Time-Invariant* Reference Spreading" (van Steen, van de Wouw, Saccon); **2511.16330** = "Safe and
> Optimal Variable Impedance Control via Certified Reinforcement Learning" (Kumar, Prakash; **no
> "C-GMS" acronym**, v1 2025 / ICRA 2026). Still confirm the exact published venue per house style.

Companion docs: `FAITHFUL_SOFT_CAT_IMPL_PLAN.md`
(our faithful soft-CaT port + the CaT-as-incentive-vs-VIC deep-dive appendix), `../tracking_impact_impulse_design_research.md` (impulse-constraint design).

---

## 1. The landscape — three camps, divided by *what quantity each bounds*

This division is load-bearing: our constraint lives in only one of the three classes.

### (a) CMDP / Lagrangian — bounds EXPECTED DISCOUNTED CUMULATIVE cost
**Guarantees** feasibility of `J_C(π) = E[Σ_t γ^t c_t] ≤ d` — an expectation of a discounted *sum*.
Foundation: Altman's CMDP (1999), an LP over occupation measures with strong Lagrangian duality;
everything downstream attacks the saddle point `min_{λ≥0} max_θ [J_R − λ(J_C − d)]`.
- **PPO/TRPO-Lagrangian** (Safety Gym, Ray et al. 2019): one cost critic + scalar dual ascent — the
  de-facto baseline.
- **RCPO** (Tessler et al. 2019): three-timescale actor-critic; needs a multiplier LR + λ_max;
  almost-sure feasibility only under an often-unverifiable assumption.
- **CPO** (Achiam et al. 2017): second-order trust-region QP (Fisher matrix, conjugate gradient,
  analytic dual, infeasibility-recovery branch, line search) — the heaviest.
- **PID-Lagrangian** (Stooke 2020), **FOCOPS** (2020), **CUP** (2022), **P3O** (2022): the camp's own
  escape attempts from dual-ascent oscillation.

**Native constraint type:** cumulative budgets (total energy/wear/contacts per episode).
**A bound on `E[Σγ^t c_t]` places no bound on `max_t c_t`** — the camp acknowledges this.

### (b) Termination / CaT — simple, scale-free *incentive*; no formal guarantee
**Guarantees** a *chance* constraint `P[c_i > 0] ≤ ε` over the discounted state-action distribution —
**probability of violation, not magnitude of the worst violation**. Soft/empirical: CaT's own paper
reports residual real-robot violation even at p_max=1.0. "Hard" means hard-*in-incentive*, not
hard-*in-physics*.
**Cost:** lowest. No multiplier, no dual ascent, no cost critic, no two-timescale stability
conditions, no QP — a per-constraint EMA normalizer `c_max` + a ~3-line rollout edit
(`reward ← reward·(1−δ); done ← δ`, effective discount `γ(1−δ)`).
**Lineage:** ET-MDP (Sun et al. 2021, deterministic terminate-on-violation, proves CMDP-equivalence)
→ NaR/IPO (Kim/Hwangbo et al. T-RO 2024) → **CaT** (Chane-Sane et al. IROS 2024, the smooth
EMA-normalized probabilistic form with a soft p_max curriculum).
**Native constraint type:** per-state / instantaneous — δ from the *instantaneous* violation `c_i⁺`
at each transition. **Structural fit** to our per-step impulse/velocity constraint.

### (c) State-wise / instantaneous + safety-layer / CBF / projection — per-step HARD guarantee (capability)
- **(c-i) Soft state-wise PO:** **SCPO** (Zhao 2023) / **ASCPO** (Zhao 2024) — augment the state with
  a running-max-cost tracker ("Maximum MDP") so a per-step constraint becomes one cumulative
  surrogate; SCPO bounds the *expected* max state-wise cost, ASCPO its *tail/variance*. **High math.**
  SCPO concedes: *hard state-wise safety during training is impossible without a dynamics model* — so
  these stay **soft**, not a physical brake.
- **(c-ii) Hard per-step capability layers** (physically override the action):
  - **Dalal safety layer** (2018): learned linear cost model + closed-form QP projection; degrades
    with multiple simultaneously-active limits (plausible across 6 coupled joints at impact).
  - **OptLayer** (Pham et al. ICRA 2018): differentiable QP projecting the action onto a linear
    feasible set — **essentially the Jacobian-projection layer we planned for the velocity box.**
  - **CBF-QP** (Cheng AAAI 2019; robust RCBF Emam 2021; **kinetic-energy CBF** Califano 2024):
    forward-invariance via a per-step QP on a barrier; *hard per-state* but needs a control-affine
    model + valid CBF.
  - **ATACOM** (Liu/Tateo et al. 2024): tangent-space projection onto a constraint manifold; per-state
    hard, validated on **7-DoF dynamic air-hockey striking** — closest dynamic-manipulation analog.
  - **Shielding** (Alshiekh 2018): formally certified but exponential, discrete-model-only.

**Native constraint type:** per-step worst-case / forward-invariance. **The price of hardness is
always a model** (Jacobian, mass matrix, valid barrier, automaton).

---

## 2. Why CaT is popular — the simplicity intuition, validated with evidence

**The intuition is correct.** CaT's adoption (≈39 citations in <2 yr; ≥3 independent groups across
quadruped, humanoid loco-manip, lunar mobile-manip) is substantially driven by simplicity: scale-free
EMA normalization + ~3-line PPO edit + no multiplier + manager-based + RL-library-agnostic.

The honest sharpening: **PPO-Lagrangian is *also* chosen for simplicity, not a stronger guarantee.**
So the real incentive-layer contest is **CaT vs PPO-Lagrangian** — both first-order, both
expectation-level. CaT wins on being additionally **scale-free** (no κ / λ-LR tuned against the
reward scale) and needing **no cost critic**.

**The pain CaT avoids is documented, not hypothetical:**
- A 2025 empirical study finds the optimal multiplier λ\* is "highly task-dependent" and "can be as
  hard to find as solving the RL problem itself" (echoing Paternain 2019); dual ascent oscillates; no
  update rule wins across tasks.
- **PID-Lagrangian exists *only because* plain dual ascent is a forced oscillator** that overshoots
  the limit *during* training — an entire ICML paper band-aiding an instability CaT sidesteps by
  construction (δ is injected per-step; there is no slow global multiplier to overshoot).

**Honest correction (for the committee):** CaT's tuning is *not* zero. Its ET-MDP baseline and the
"all p_max=1.0" ablation both fail to learn with >60 constraints (exploration dies first); the soft
p_max curriculum + EMA τ are a real, if mild, tuning surface. And because `c_max` is an EMA of the
*batch-max* violation, termination pressure on the impulse term is *weak early in training* — an
argument for substep-rate detection and possibly a floor on `c_max` for the safety-critical term.
Claim: "materially less tuning than Lagrangian dual-ascent," not "tuning-free."

---

## 3. The decisive point — it is the constraint TYPE

Our constraint is **per-step worst-case**, not a cumulative-expected budget: we need
`Λ_j` (per-joint reaction impulse, per contact EVENT) `≤ limit` and `|q̇_j(t)| ≤ 3.1415` at *every substep on every joint* — velocity is the per-substep worst-case; the impulse is accumulated at 500 Hz over a contact-anchored window and read with per-event pulse semantics (`impulse_bound.py`). This is the **state-wise / almost-sure** class — formally distinct from the
standard CMDP class.

**Therefore the classic Lagrangian/CMDP machinery is the *wrong* heavy tool, not merely a heavier
one.** Every camp-(a) member bounds `E[Σγ^t c_t] ≤ d`; a policy can sit *exactly at budget d* while a
single substep spike on one joint is arbitrarily large (averaged away over time and episodes). Paying
the Lagrangian math tax buys a guarantee **about the wrong object.** The right heavier tool, *if*
needed, is camp (c) — SCPO/ASCPO (right shape, soft) or hard capability layers (CBF/ATACOM/OptLayer,
right shape, hard). **Lagrangian is not on our escalation path at all.**

**Corroborated by our own ablation (structurally, not as a tuning bug).** A1–A4 found CaT (A3) the
best impact-preserving reducer but it left worst-case joint velocity at **3.61–4.00 rad/s** control-rate
(4.12–4.95 true substep across the two CaT variants; unconstrained baseline 4.29–4.65 control-rate),
over the 3.1415 limit. The overshoot is **chain-coupled momentum** — delivered through the linkage by *other*
joints + contact rebound — that no per-joint scalar *incentive* can physically brake. A Lagrangian on
a cumulative impulse-sum would reproduce that exact gap; SCPO/ASCPO (bounding only the *expected* /
*high-probability* max) would likely leak it too. **No incentive-layer method, however heavy, brakes
the tail — that is a capability problem.**

Hence the two-layer architecture is forced by physics, not chosen for convenience:
- **CaT = incentive layer** (native per-state fit; cheap, scale-free disincentive).
- **VIC / Jacobian-projection / CBF = capability layer** (physically *prevents* the spike: variable
  impedance so the linkage cannot transmit it; or an action-level QP that hard-clips the velocity box).

Caution for the writeup: **CUP's "projection" is policy-space dual reconciliation, NOT the
action-space Jacobian/QP projection we mean.** Do not conflate them.

---

## 4. Verdict — do we need the heavy math?

**No — not the Lagrangian heavy math. Keep CaT as the incentive layer. The only principled escalation
is the capability family (VIC / action-projection / CBF), added for the *physical* worst-case, not as
a "better optimizer."** Three legs:

1. **Wrong-tool, not heavier-tool.** Lagrangian/CMDP bounds an expectation of a sum; our constraint is
   a per-step max. The tax buys a guarantee on the wrong quantity.
2. **The tax is real and the instinct is data-backed.** λ\* "as hard as the RL problem itself"; dual
   ascent oscillates; PID merely shifts the instability. CaT has no multiplier to oscillate. And the
   one thing Lagrangian *would* buy — safety *during training* (Safety Gym's purpose) — is **moot for
   us because we train in sim.**
3. **Neither CaT nor any incentive method fixes the tail; that is the capability layer's job.** Across
   all CaT adopters surveyed, **none analyzes the worst-case tail or pairs CaT with an action-level
   hard filter.** Our CaT-incentive + VIC/projection-capability split is the genuine open contribution.

**When each tool is warranted:**

| Situation | Right tool | Why |
|---|---|---|
| Per-step *incentive*, both constraints, any RL lib | **CaT** | Native per-state fit, scale-free, no multiplier; sim training makes during-training guarantees moot |
| Hard worst-case on the **velocity box** \|q̇\| ≤ π | **Action-level QP projection / OptLayer / rel-degree-1 CBF** | Clean linear/box constraint → cheap exact per-step clip; ATACOM is the model-based fallback |
| Hard worst-case on the **impact impulse** | **Variable impedance (VIC)** | Impact is discontinuous, non-control-affine, near-instantaneous, partly delivered by other joints+rebound → breaks CBF relative-degree/linearity assumptions; FACET shows compliance cuts impulse ~80% physically |
| If safety is ever recast as a true **cumulative budget** | **P3O / FOCOPS** (not CPO) | Only then does the cumulative-Lagrangian camp apply; pick first-order/fixed-penalty members |

**Hostile-committee Q&A:**
- *"CaT gives no formal guarantee."* Correct, and we never claim it does. CaT is the *incentive* layer;
  the **capability layer supplies the hard per-step guarantee** (forward-invariance for the velocity
  box; physical impulse attenuation for impact). The architecture is honest about where the guarantee
  comes from.
- *"Why not SCPO/ASCPO?"* Right shape; we cite ASCPO as the principled tail-aware incentive. But they
  still bound only the *expected/high-probability* max (not the realized worst case), so they wouldn't
  obviously close our chain-coupled tail any better than CaT — and they re-import trust-region + dual +
  line-search tuning for the *same class of soft guarantee*. SCPO concedes hard state-wise safety needs
  a dynamics model — which is our capability layer's job.
- *"Why not a Lagrangian multiplier on the impulse cost?"* It bounds the expected discounted
  impulse-*sum*; a single substep spike is averaged away. Category error for a per-step max, and it
  re-introduces the oscillation/tuning pain for *no* tail improvement.
- *"Isn't a CBF the rigorous choice for everything?"* For the velocity box, yes (clean
  relative-degree-1 control-affine). For impact, no: impact violates CBF continuity/relative-degree
  assumptions (the textbook hard case), and certified-VIC is explicitly free-space-only and breaks at
  contact — which is why the impact capability layer is **physics-based (VIC)**, not a smooth filter.

**Bottom line.** Use CaT. Spend engineering complexity on the physical capability layer
(fixed-impedance results first, then VIC), not on a more elaborate optimizer. The heavier math we
*will* eventually defend is the capability layer's (CBF/ATACOM for the velocity box; VIC physics for
impact), never the cumulative-Lagrangian camp's.

---

## 5. Canonical citations per camp (verified 2026-06-17 — see caveat at top)

**Constraint-class framing (cite first — grounds the whole argument):**
- **State-wise Safe RL: A Survey** — Zhao, He, Liu et al., IJCAI 2023 (arXiv:2302.03122). *Authoritative
  statement that Lagrangian/CMDP bound only expected-cumulative cost and cannot certify per-step worst
  case — so for our instantaneous limit the cumulative camp is the wrong tool, not a heavier one.*

**Camp (a) — CMDP / Lagrangian:**
- **Altman, Constrained MDPs**, 1999. *The foundational object; fixes what the camp can/can't promise.*
- **Stooke, Achiam, Abbeel — PID-Lagrangian**, ICML 2020 (arXiv:2007.03964). *Strongest "the simple
  multiplier oscillates" citation.*
- **Towards a Practical Understanding of Lagrangian Methods in Safe RL**, 2025 (arXiv:2510.17564).
  *Empirical tuning-tax proof.*
- (Cumulative-budget contingency only:) **P3O** IJCAI 2022 (arXiv:2205.11814), **FOCOPS**
  NeurIPS 2020 (arXiv:2002.06506).

**Camp (b) — Termination / CaT (our incentive layer):**
- **Chane-Sane et al. — CaT**, IROS 2024 (arXiv:2403.18765). *Our chosen mechanism.*
- **Sun et al. — ET-MDP**, 2021 (arXiv:2107.04200). *Deterministic precursor; proves
  CMDP-equivalence (defense ammunition); shows why naive binary terminate-on-violation fails to learn.*
- **Kim/Hwangbo et al. — Not Only Rewards But Also Constraints**, IEEE T-RO 2024 (arXiv:2308.12517).

**Camp (c) — State-wise / capability (our escalation path):**
- **Zhao et al. — SCPO** 2023 (arXiv:2306.12594) and **ASCPO** 2024 (arXiv:2410.01212). *Formal
  home of our problem; ASCPO is the tail-aware incentive to cite against "why not bound the worst case."*
- **Pham et al. — OptLayer**, ICRA 2018 (arXiv:1709.07643). *Direct precedent for action-level QP
  projection (hard per-step velocity-box clip).*
- **Liu, Bou-Ammar, Peters, Tateo — Safe RL on the Constraint Manifold: Theory and Applications** (the
  *ATACOM* method), 2024 (arXiv:2404.09080). *Hard per-state via tangent-space projection, validated on
  7-DoF dynamic air-hockey striking — model-based capability fallback.*
- **Cheng et al. — RL-CBF**, AAAI 2019 (arXiv:1903.08792) and **Califano et al. — Kinetic-Energy CBF**,
  2024 (arXiv:2411.02186). *Certified-hard option for the velocity box / energy budget.*

**Capability layer — variable impedance (the physical brake for impact):**
- **Bogdanovic, Khadiv, Righetti — Learning Variable Impedance Control for Contact Sensitive Tasks**,
  v1 2019 / RA-L 2020 (arXiv:1907.07500). *Same lab (Khadiv); direct precursor — per-joint policy-commanded
  impedance with gain regularization. Establishes our lineage; prior work used soft regularization, so
  our per-step impulse *constraint* is the novel enforcement contribution.*
- **Martin-Martin et al. — VICES**, IROS 2019 (arXiv:1906.08880). *Canonical justification for the VIC
  action space.*
- **Xu et al. — FACET**, 2025 (arXiv:2505.06883). *Empirical proof that lowering stiffness cuts
  collision impulse ~80% — the keystone for "CaT = incentive, VIC = capability."*
- **Kumar, Prakash — Safe and Optimal Variable Impedance Control via Certified RL**, v1 2025 / ICRA 2026
  (arXiv:2511.16330). *Cautionary contrast: certified VIC works in free space but breaks at contact —
  justifies our pragmatic split. (No "C-GMS" acronym in the paper.)*

**Impact-physics grounding (optional):**
- **van Steen, van de Wouw, Saccon — Impact-Aware Control using Time-Invariant Reference Spreading**,
  2024 (arXiv:2411.09870). *Validates impact safety is set ante-impact (preparation, not reaction).*

**Net-new external evidence (2026-06-18, user-supplied; digested for the impulse arm):**
- **Ma, Cramariuc, Farshidian, Hutter — Learning coordinated badminton skills for legged manipulators**,
  Science Robotics 10, eadu3922 (2025) (arXiv:2505.22974). *The nearest RL neighbour to our robot-side
  bound: a learned policy enforcing a REAL actuator-load limit (arm current `I_total < 8 A`) via the
  constrained-RL **N-P3O**, with a clean ablation — a SOFT over-current penalty was violated, the
  constraint algorithm "never violated the constraint." Third-party empirical support for "constraint
  mechanism over `‖Δq̇‖²` penalty," reinforcing [[Spoor et al. 2025]] (λ-fragility) for our soft-CaT
  choice. **N-P3O is the constrained-RL baseline to ablate against** the impulse soft-CaT. Their
  strike-velocity reward is Dirac-gated to the interception step — the phase-gated ante-impact pattern.*
- **Vu, Erens, Stefanelli, Cisneros-Limon, Benallegue — QP-based impact momentum maximization for a
  hammering task by a humanoid robot**, IEEE 2026 (hal-05516105). *The model-based EMMT QP that maximizes
  impact momentum (`m_eff·v_axial`) — the anchor for our maximize OBJECTIVE — and EXPLICITLY leaves
  joint-recoil/damage bounding as future work. Our robot-side per-joint impulse bound IS their named open
  problem; frame the contribution as "recast their objective as an RL reward + add the deferred bound."*
- **Humanoid Whole-Body Badminton via Multi-Stage RL**, 2025 (arXiv:2511.11218) and **Ti, Gao, Zhao,
  Calinon — Optimal-Control Tool Affordance for Impact Tasks** (iLQR+ADMM, nail+pilot-hole), 2024
  (arXiv:2402.05502). *Side-(1) precedents: phase-gated strike-velocity reward with NO motion prior
  (badminton) — evidence to ablate/down-weight our `r_imit`; and a second model-based "posture for max
  strike velocity" via manipulability (tool affordance).*
- **Ma, Tian, Gao — Manipulate as Human (AMP)**, Robotica 2025 (DOI 10.1017/S0263574725001444;
  github.com/ZiqiLoveSunshine/Manipulate_as_Human-AMP). *Considered-but-deferred alternative to the
  Gaussian prior: AMP grounds style from seconds of data but does not reward the strike apex, adds
  adversarial-training instability, and its mixed-sign/unbounded discriminator reward is exactly what the
  scale-positives soft-CaT decision is wary of. Non-impact; cite as the alternative we declined.*

---

## Appendix: joint-velocity-bound research [from JOINT_VELOCITY_BOUND_RESEARCH, verbatim]

> 2026-06-17 research synthesis behind the velocity-bound / CaT decision, merged verbatim. Still current: velocity is the Z1 hardware-honesty proxy; the per-joint impulse bound (`impulse_bound.py`) is the thesis target.

**Trigger:** `b_strike` produced a real single strike (~1.2 m/s, 100% success, press excluded) but the trained policy drives joints to **4.3–4.65 rad/s worst-case**, exceeding the real Z1 limit of **3.1415 rad/s**. See `docs/results/2026-06-17_b_strike.md`. Question: how do we make the strike *honest at the hardware velocity limit* without killing it?

**Method:** multi-lens research workflow `wf_aa165e9a-cb9` (5 lenses — internal corpus, constrained-RL, velocity-limited-IK, MuJoCo-physics, reward-shaping — → synthesis → adversarial critique) + the user-supplied CaT paper.

---

## TL;DR recommendation

1. **Run the cheap, decisive experiment first.** Retrain with an *honest* velocity bound — simplest: `delta_pos_scale` 0.15→0.10 (one constant) and/or a CaT velocity constraint at π — purely to answer **"does the strike survive an honest velocity limit?"** That single outcome decides everything:
   - **Survives** → we have an honest baseline; *then* decide whether a tighter bound is worth the layered plant+filter work.
   - **Dies** → that IS the Phase-0 finding ("position-only fixed-PD striking is velocity-limited"), which directly motivates the thesis's variable-impedance move. Record and move to G1 — no elaborate plant swap.
2. **When we add a soft constraint, deliver it via CaT, not a hand-weighted reward penalty** — scale-free, low-infra, manager-based (ports to mjlab), and it is the *same machinery* we reuse for the G1 impulse constraint.
3. **Bound velocity as a hardware-honesty guard, but log per-joint IMPULSE `Λ_j = Σ|qfrc_constraint_j|·h` in parallel** — because velocity is the *Z1 proxy*, not the *thesis target* (see "Right target" below).
4. **Do not chase the last 0.5 rad/s of worst-case transient on the Z1.** That is the rabbit hole. Make the strike hardware-honest, exercise the constraint machinery once on the correct quantity, stop.

---

## The failure mechanism (why no single simple fix is a hard bound)

The overshoot is **not** a single joint over-spinning. Open-loop straight-down peaks at 2.41 rad/s; the trained policy reaches 4.3–4.65 (~1.9×) via **off-axis / chain-coupled windup** — momentum delivered to a joint *through the kinematic chain* by other links' motion, plus possible contact rebound. Consequence: a mechanism that only limits one joint's own torque or its own commanded increment **cannot brake momentum it did not inject**. This is the key reason the "self-limiting" assumption failed and why soft, single-joint, or steady-state fixes are partial.

## Options (ranked, with honest bound-strength)

| # | Option | Bound strength | Effort | Thesis value | Note |
|---|---|---|---|---|---|
| A | **DcMotorActuator torque-speed envelope** (`velocity_limit=π`, train inside it) | steady-state/envelope | med | high | Honest plant ("constrain in physics, not a clip"); but a motor curve can't brake chain-coupled momentum → can still trip a worst-case flag. **Set `effort_limit < saturation_effort`** or the continuous-torque clamp is a no-op. |
| B | **Jacobian action-projection** (scale the Cartesian delta so induced joint vel ≤ π) | hard *on command* | med | strong | Velocity-CBF `h=π−max\|q̇\|`. Bounds the *commanded* velocity, not the realized PD transient — shares A's blind spot for stiff-PD overshoot. |
| C | **Substep velocity-excess penalty → CaT/PID-Lagrangian** | soft / in-expectation | med | highest (reward route) | `−w·Σ max(0,peak\|q̇_j\|−π·β)²`, substep-peak, annealed in after the strike. CaT is the preferred low-infra delivery. Hackable as a fixed penalty; CMDP auto-tunes λ. |
| D | **Lower `delta_pos_scale` → 0.10** | soft (≈ at edge) | low | low | Cheapest. Control arm. May falsify decisively (see TL;DR). |
| E | **Raw joint damping / back-EMF constant** | soft | low | med | Rejected standalone: steady-state cure for a transient; dissipates the impact impulse the task needs. If used, put the slope in A's torque-speed curve, not raw damping. |

**Forbidden (repo rule, reconfirmed):** raw `−w·‖q̇‖²` (the Unitree `dof_vel` form) — taxes every honest fast motion and the post-impact velocity jump; kills the strike. Use excess-over-threshold, which is silent below the limit.

## CaT — Constraints as Terminations (the user's find; corroborated by the constrained-RL lens)

Chane-Sane et al., IROS 2024 (`arxiv.org/abs/2403.18765`, `github.com/gepetto/constraints-as-terminations`). For each constraint `c_i(s,a)≤0`: violation `c_i⁺=max(0,c_i)`, termination probability `δ = max_i p_i^max·clip(c_i⁺/c_i^max, 0,1)` (`c_i^max` = EMA of batch-max violation). Rollout: `rewards←rewards·(1−δ)`, `done←δ` ⇒ discount becomes `γ(1−δ)`. Hard constraints `p^max=1`, soft ramp `p^max` 0.05→0.25. They enforce joint velocity, torque, accel, contact force on Solo-12 + PPO. **Why it fits us:** scale-free (no penalty-weight tuning), dominant deterrent (loses the big `completion=100`), **manager-based (`ConstraintsManager`/`ConstraintTerm`/`max_p`) → ports to mjlab's manager pattern**, and it is the exact machinery we'd reuse for the **G1 impulse constraint** (`c = Λ_j − Λ_lim`). Caveat: still **soft/in-expectation** — strong adherence, not an almost-sure worst-case bound. See memory `[[cat-constraints-as-terminations]]`.

## Right target — velocity is the Z1 proxy, NOT the thesis target

`docs/research/tracking_impact_impulse_design_research.md` (SQ3/SQ4/D4) is explicit: the thesis constrains **per-joint IMPULSE `Λ_j = Σ|qfrc_constraint_j|·h`** over a contact-anchored window — not `|q̇|`. Velocity is sanctioned as a Z1 stand-in *only* because contact posture is ~fixed so `m_eff≈const`. So:
- Bounding `q̇ ≤ π` is a legitimate, independently-needed **hardware-feasibility guard** ("don't fake the hardware").
- The **impulse** bound is a different concern ("don't damage the joint") and is the actual thesis signal.
- A velocity-excess term rehearses the constraint *shape* (excess-over-threshold + substep accumulation + escalation ladder) but on the *wrong signal*. **Fix:** log `Λ_j` (from `qfrc_constraint`, substep-accumulated, contact-anchored) in parallel from day one, so the Phase-0 rehearsal matches the G1 CMDP signal, not just its form.

## Critic's load-bearing corrections

- **Rank-1 (DcMotor) is internally inconsistent**: a steady-state/envelope bound judged against a *worst-case global-max* flag (`diag_policy_trace` `running_max_qv > 3.1415`). The motor curve makes a joint's own free-swing asymptote to π without overshoot, but cannot brake chain-coupled momentum or contact rebound — so it can pass its narrative yet fail its own pass criterion. Demote from "the answer" to "the honest plant."
- **Numeric bug**: with `effort_limit = saturation_effort` (30=30, 60=60), `_vel_at_effort_lim = 2·velocity_limit = 6.28` → the continuous-torque clamp never binds. Choose `effort_limit < saturation_effort` (e.g. ~24/48) for it to do the claimed work.
- **Plant swap ≠ tweak**: DcMotorActuator moves the arm off the native `<position>` affine PD onto mjlab's Python torque-level `IdealPdActuator`, interacts with `gravcomp=1.0`, and invalidates every prior tuned result (NEAR_NAIL pose, press-exclusion, kp/kd settling). Full re-validation required.
- **Honest hard bound = A + B together** (plant model + action projection), not either alone — and even then the chain-coupling/contact-rebound tail can leak, exactly as on real hardware.

## Sources

- CaT: `arxiv.org/abs/2403.18765` · `github.com/gepetto/constraints-as-terminations`
- CBF-RL (per-step kinematic safety filter, G1): `arxiv.org/abs/2510.14959`
- KAIST Hound (train inside the motor operating region for honest transfer): `arxiv.org/abs/2312.17507`
- Jerk-limited online trajectory generation (Ruckig-style): `arxiv.org/abs/2410.20907`
- Internal: `../tracking_impact_impulse_design_research.md` (D1–D6, SQ3/SQ4), `../../archive/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` (T0–T5), `../hammering_reward_design_deep_dive_v2.md`; memories `[[diffik-maxdq-velocity]]`, `[[z1-hardware-limits]]`, `[[cat-constraints-as-terminations]]`.
