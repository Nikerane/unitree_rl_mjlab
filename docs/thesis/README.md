# Thesis material — confirmed inclusions

Curated, defensible material destined for the Master's thesis (*impact-safe contact-rich
manipulation*, TU Munich / ATARI Lab, Prof. Khadiv). Each entry is stated in thesis voice, with a
pointer to the authoritative detail and a **status**:
- **CONFIRMED** — established (conceptual / decision / literature); safe to write up now.
- **RESULTS-PENDING** — the framing is set but the empirical claim needs a training run before it
  goes in as a result.
- **VERIFY** — needs a fact/citation check before final use.

The full detail lives in `docs/research/reward-design/`; this folder is the distilled layer. Add to
it as material becomes thesis-ready — don't duplicate the exhaustive versions here.

---

## 1. Contributions (the novel claims)

**C1 — A two-layer constraint architecture: CaT as the *incentive*, variable impedance as the
*capability*.** Constraints-as-Terminations (soft `γ(1−δ)`) disincentivizes per-step violations
cheaply and scale-freely, but it cannot *physically* brake a violation; variable impedance (and/or
action-level Jacobian projection) physically prevents it. The split is forced by the physics, not
chosen for convenience.
- Status: **CONFIRMED** (conceptual); the empirical demonstration is **RESULTS-PENDING**.
- Open-gap framing (the originality): across surveyed CaT adopters (SoloParkour, Humanoid-CaT, lunar
  mobile-manip), **none analyzes the worst-case tail or pairs CaT with an action-level hard filter.**
- Detail: `CONSTRAINED_RL_LANDSCAPE.md` §3–4, `CAT_DEEP_DIVE.md`.

**C2 — A per-step (substep-accumulated) per-joint *impulse* constraint as the enforcement step on top
of soft-gain-regularized variable impedance.** Prior VIC work (incl. the advisor's own line) used
*soft gain regularization*; enforcing a per-step impulse limit is the new contribution.
- Status: **CONFIRMED** (positioning); enforcement results **RESULTS-PENDING**.
- Lineage anchor: Bogdanovic, Khadiv & Righetti, *Learning Variable Impedance Control for
  Contact-Sensitive Tasks*, RA-L/IROS 2020 (arXiv:1907.07500) — **same lab**.
- Detail: `tracking_impact_impulse_design_research.md`, `CONSTRAINED_RL_LANDSCAPE.md` §5.

**C3 — The chain-coupled-momentum finding: no per-joint *incentive* (reward penalty, soft or hard
CaT, even a cumulative-Lagrangian) bounds the worst-case velocity tail.** Our A1–A4 ablation: CaT was
the best impact-preserving reducer yet kept worst-case `|q̇|` at 4.3–4.65 rad/s (over the 3.1415
limit) because the overshoot is momentum delivered through the linkage by other joints + contact
rebound. This *motivates* the capability layer (C1).
- Status: **CONFIRMED** (we have the ablation data).
- Detail: `docs/results/2026-06-17_velocity_bound_ablation.md`, memory [[z1-velocity-bound-finding]].

---

## 2. Method decisions (defensible, with rationale)

**D1 — Faithful soft `γ(1−δ)` CaT, ported as a thin adapter (no edits to installed rsl_rl/mjlab).**
We replace the naive sampled-hard termination with the paper's true mechanism: env-side
`reward·(1−δ)` discount + a float soft-done carried into a `γ(1−δ)` GAE bootstrap. Mirrors the CaT
authors' own per-library adapter pattern (they never forked CleanRL/rl_games/skrl).
- Status: **CONFIRMED** (design); implementation **RESULTS-PENDING**.
- Detail: `FAITHFUL_SOFT_CAT_IMPL_PLAN.md` §1–2.

**D2 — Reward discount is *scale-positives-only*: `reward = r_total − δ·r_pos`** (apply `(1−δ)` to the
positive task terms only; leave penalties unscaled). Reached by overturning an earlier "drop the
clip / scale the whole reward" draft via adversarial review, which found a derivable
**penalty-evasion exploit**: full-reward `(1−δ)` makes the incentive-to-violate `−δ·[r_t + γV]`,
positive whenever the bracket is negative — i.e. the policy is *paid* to violate on cold-critic or
penalty-heavy steps, and high velocity (large δ) physically co-occurs with the −10 joint-limit
penalty. Scale-positives removes the coupling and is *more* faithful to CaT (whose `(1−δ)` is a
forfeit-future-*positive*-return weight, coherent only on `r ≥ 0`).
- Status: **CONFIRMED** (decision + reasoning); it is also a clean thesis narrative ("we stress-tested
  the choice and found a concrete failure mode").
- Detail: `FAITHFUL_SOFT_CAT_IMPL_PLAN.md` Decision 1; memory [[soft-cat-scale-positives-decision]].

**D3 — Honest framing of the one hosting deviation.** The reference applies the discount inside the
env `step()`; mjlab pins the env class, so we apply it in the learner (`CatPPO.process_env_step`,
where the reward is cloned before storage). **The math is identical; only the host method differs.**
State this precisely in the writeup so the "I applied CaT" claim is exact.
- Status: **CONFIRMED**. Detail: `FAITHFUL_SOFT_CAT_IMPL_PLAN.md` §2 + Decision 2.

**D4 — A CaT violation is a *true termination*, not a timeout** (must not be bootstrapped). Standard
CaT semantics. Status: **CONFIRMED**. Detail: `FAITHFUL_SOFT_CAT_IMPL_PLAN.md` Decision 4.

---

## 3. Literature framing (Related Work / method justification)

**L1 — The field divides by *what quantity each method bounds*, and our constraint lives in only one
class.** (a) CMDP/Lagrangian bounds expected *cumulative* discounted cost `E[Σγ^t c_t] ≤ d`;
(b) termination/CaT gives a per-step *chance* incentive `P[violation] ≤ ε`; (c) state-wise /
CBF / projection gives a per-step *hard* guarantee.
- Status: **CONFIRMED**. Detail: `CONSTRAINED_RL_LANDSCAPE.md` §1.

**L2 — For a per-step worst-case constraint, Lagrangian/CMDP is the *wrong* tool, not merely a heavier
one.** A bound on `E[Σγ^t c_t]` places no bound on `max_t c_t` — a policy can sit exactly at the cost
budget while one substep spike is arbitrarily large. So choosing CaT over Lagrangian is a
*correctly-typed* choice, not "easy over rigorous." The principled escalation for a hard guarantee is
camp (c) — VIC / projection / CBF — never Lagrangian.
- Status: **CONFIRMED** (this is the strongest defense framing). Detail: `CONSTRAINED_RL_LANDSCAPE.md` §3.

**L3 — CaT's popularity is driven by simplicity, and the Lagrangian tuning pain it avoids is
documented.** λ\* is reported as "as hard to find as solving the RL problem itself"; dual ascent
oscillates; PID-Lagrangian is an entire paper that exists to tame that oscillation — which CaT
sidesteps by construction. Honest caveat: CaT's tuning is *less*, not *zero* (p_max curriculum + EMA).
- Status: **CONFIRMED**. Detail: `CONSTRAINED_RL_LANDSCAPE.md` §2.

---

## 4. Defense points (anticipated committee questions → answers)

- *"CaT gives no formal guarantee."* Correct — CaT is the *incentive* layer; the **capability layer
  (VIC / projection / CBF) supplies the hard per-step guarantee.** The architecture is honest about
  where the guarantee comes from.
- *"Why not a Lagrangian multiplier on the impulse?"* It bounds the expected discounted impulse-*sum*;
  a single substep spike averages away. Category error for a per-step max, plus oscillation/tuning
  pain for no tail improvement.
- *"Why not SCPO/ASCPO (purpose-built for state-wise)?"* Right shape; we cite ASCPO as the tail-aware
  incentive. But they still bound only the *expected/high-probability* max (not the realized worst
  case) and re-import trust-region + dual + line-search tuning — and SCPO concedes hard state-wise
  safety needs a dynamics model, which is our capability layer's job.
- *"Isn't a CBF rigorous for everything?"* For the velocity box, yes (clean relative-degree-1
  control-affine). For impact, no — impact breaks CBF continuity/relative-degree assumptions, which is
  why the impact capability layer is physics-based (VIC), not a smooth filter.
- Detail: `CONSTRAINED_RL_LANDSCAPE.md` §4.

---

## 5. Canonical citations (anchor set — see `CONSTRAINED_RL_LANDSCAPE.md` §5 for the full list + VERIFY flags)

- Framing: **State-wise Safe RL: A Survey**, Zhao et al., IJCAI 2023 (arXiv:2302.03122).
- Lagrangian: **Altman, CMDPs** 1999; **PID-Lagrangian**, Stooke et al. ICML 2020 (arXiv:2007.03964).
- CaT: **Chane-Sane et al., CaT**, IROS 2024 (arXiv:2403.18765).
- Capability: **OptLayer**, Pham et al. ICRA 2018 (arXiv:1709.07643); **Safe RL on the Constraint
  Manifold** (the ATACOM method), Liu/Bou-Ammar/Peters/Tateo 2024 (arXiv:2404.09080).
- VIC: **Bogdanovic, Khadiv, Righetti — Learning Variable Impedance Control for Contact Sensitive
  Tasks**, v1 2019 / RA-L 2020 (arXiv:1907.07500); **VICES**, Martin-Martin et al. IROS 2019
  (arXiv:1906.08880); **FACET** 2025 (arXiv:2505.06883 — the "stiffness ↓ ⇒ impulse −80%" keystone).

*Citations verified 2026-06-17 (web-grounded arXiv fetch): no fabricated ids; minor title/year fixes
applied above and in `CONSTRAINED_RL_LANDSCAPE.md` §5.*

---

## 6. Still-open / not-yet-thesis-ready

- soft-CaT training **results** (the implementation itself is **DONE** on branch `soft-cat`: C0–C3
  built, adversarially reviewed + fixed, 37 CPU unit tests + `verify_cat_soft.py` (real env) +
  `smoke_cat_soft.py` (3-iter CPU train loop runs, no NaN) all green. Only the GPU *results* run —
  does the policy learn a limit-respecting strike — remains, on Vega. `train.py`'s launcher is
  GPU-gated so the full run cannot execute on the mac).
- worst-case tail closure via VIC (capability layer — after fixed-impedance results).
- ~~citation verification pass~~ **DONE** (2026-06-17): no fabricated ids; 4 minor title/year fixes
  applied. Still confirm exact published venues per the thesis house citation style.

---

## 7. Draft prose — Related Work (constraint mechanism)

> Draft paragraph, ready to adapt. Citations are verified-real (year/venue per house style).

Safe and constrained reinforcement learning offers three broadly distinct mechanisms for enforcing a
behavioural limit, distinguished by *the quantity each one bounds*. The first, rooted in the
Constrained-MDP formalism (Altman, 1999), augments the objective with a cost signal and enforces an
expected discounted **cumulative** budget `E[Σ γ^t c_t] ≤ d` via a Lagrangian multiplier; representative
algorithms include CPO (Achiam et al., 2017), RCPO (Tessler et al., 2019), and the PPO-Lagrangian
baselines of Safety Gym (Ray et al., 2019). While principled, these methods are sensitive to multiplier
tuning and prone to oscillation during training, motivating remedies such as PID-Lagrangian control
(Stooke et al., 2020). Crucially, a bound on the expected cumulative cost places no bound on the
**worst single-step violation** (Zhao et al., 2023), making this family ill-suited to an *instantaneous*
safety limit such as a per-joint impulse or velocity bound. A second mechanism,
Constraints-as-Terminations (Chane-Sane et al., 2024), dispenses with the multiplier entirely: a
per-step violation triggers a probabilistic episode termination, so the policy forfeits future return in
proportion to the violation. This is scale-free, simple to implement, and acts directly on the per-step
quantity, but it provides only a probabilistic (incentive-level) guarantee. A third mechanism enforces
per-step satisfaction directly — by augmenting the state to render a state-wise constraint tractable
(SCPO, Zhao et al., 2023) or by projecting the policy's action onto a feasible set at each step via a
safety layer (Dalal et al., 2018; OptLayer, Pham et al., 2018), a control-barrier-function QP (Cheng et
al., 2019), or a constraint-manifold projection (Liu et al., 2024) — yielding hard per-step guarantees at
the cost of requiring a model. Our work adopts the second mechanism as a lightweight **incentive** layer
and the third as a physically-grounded **capability** layer, using variable-impedance control
(Martín-Martín et al., 2019; Bogdanović, Khadiv & Righetti, 2020) to attenuate impact momentum that no
incentive term can physically brake.
