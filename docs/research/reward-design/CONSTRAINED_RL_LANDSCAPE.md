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

Companion docs: `CAT_DEEP_DIVE.md` (CaT-as-incentive vs VIC-as-capability), `FAITHFUL_SOFT_CAT_IMPL_PLAN.md`
(our faithful soft-CaT port), `tracking_impact_impulse_design_research.md` (impulse-constraint design).

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
`impulse_per_joint(t) ≤ limit` and `|q̇_j(t)| ≤ 3.1415` at *every substep on every joint* (impulse
accumulated at 500 Hz). This is the **state-wise / almost-sure** class — formally distinct from the
standard CMDP class.

**Therefore the classic Lagrangian/CMDP machinery is the *wrong* heavy tool, not merely a heavier
one.** Every camp-(a) member bounds `E[Σγ^t c_t] ≤ d`; a policy can sit *exactly at budget d* while a
single substep spike on one joint is arbitrarily large (averaged away over time and episodes). Paying
the Lagrangian math tax buys a guarantee **about the wrong object.** The right heavier tool, *if*
needed, is camp (c) — SCPO/ASCPO (right shape, soft) or hard capability layers (CBF/ATACOM/OptLayer,
right shape, hard). **Lagrangian is not on our escalation path at all.**

**Corroborated by our own ablation (structurally, not as a tuning bug).** A1–A4 found CaT (A3) the
best impact-preserving reducer but it kept worst-case joint velocity at **4.3–4.65 rad/s**, over the
3.1415 limit. The overshoot is **chain-coupled momentum** — delivered through the linkage by *other*
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
