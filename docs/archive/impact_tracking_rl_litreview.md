> ⚠️ **ARCHIVED 2026-07-05** — superseded by `docs/research/reward-design/LITERATURE.md`.
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Literature Review — Generate-then-Track RL for Impact-Explicit Robotic Hammering

**Mode:** ARS `deep-research` lit-review (bibliography → source-verification → synthesis)
**Date:** 2026-06-02 · **Project:** Unitree-Z1 hammer (`hammer-z1` branch)
**Status of citations:** every source below independently verified to exist (two parallel verification agents + direct WebFetch cross-check). Citation corrections caught during verification are noted in §4.

## 1. Research question & scope

**RQ.** What is the most defensible learning-and-control architecture and reward formulation for impact-explicit robotic nail-driving on a *position-controlled* manipulator, where the objective is to **maximize delivered impact momentum** (effective mass × axial velocity, `m_eff·v`) while **minimizing joint reaction impulse** — combining an upstream-generated strike trajectory with RL tracking, with a force sensor for both objective signals and observations?

**Sub-questions:** (Q1) optimal-strike trajectory generation; (Q2) tracking-RL reward structure; (Q3) offline vs online learning; (Q4) bi-objective (impact vs joint-impulse) formulation + recoil computation.

**In scope:** reward/architecture design, impact mechanics (effective mass, recoil), motion-imitation RL, constrained RL. **Out of scope:** hardware build, PPO hyperparameter tuning, the broad manipulation-RL literature.

## 2. Annotated bibliography (verified)

### A. Motion-imitation / reference-tracking RL
- **Peng, X. B., Abbeel, P., Levine, S., & van de Panne, M. (2018). DeepMimic: Example-guided deep RL of physics-based character skills.** *ACM TOG, 37*(4). arXiv:1804.02717. — Weighted **imitation + task** reward (`r = 0.7·r^I + 0.3·r^G`); imitation = joint pose + joint velocity + end-effector position (+ CoM). Reference State Initialization + Early Termination are critical for dynamic skills. *Striking/throwing tasks need the task reward — pure imitation is insufficient.*
- **Peng, X. B., Ma, Z., Abbeel, P., Levine, S., & Kanazawa, A. (2021). AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control.** *ACM TOG, 40*(4). arXiv:2104.02180. — Replaces per-frame tracking with a learned **adversarial style reward** (discriminator on a motion dataset) **+ a task reward**; scales to unstructured motion data.
- **Ma, Z., Tian, C., & Gao, Y. (2025). Manipulate as Human: Learning Task-Oriented Manipulation Skills by Adversarial Motion Priors (HMAMP).** *Robotica, 43*(6), 2320–2332. arXiv:2510.24257. — AMP applied to **real-arm hammering**. Reward `r = 0.6·r^g + 0.4·r^s` (task = contact force + nail alignment `1−tanh‖x_f−x_c‖`; style = LSGAN). Learns the **energy-storing back-swing**, yielding ~+70–80% delivered impulse over baseline RL.

### B. Offline RL
- **Kostrikov, I., Nair, A., & Levine, S. (2022). Offline RL with Implicit Q-Learning (IQL).** *ICLR 2022.* arXiv:2110.06169. — Never queries out-of-dataset actions (expectile value + advantage-weighted BC); strong at **stitching** sub-optimal trajectories; cheap to implement.
- **Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020). D4RL: Datasets for Deep Data-Driven RL.** arXiv:2004.07219 (**preprint only**). — Establishes the Adroit **`hammer`** task as a standard offline-RL benchmark built on human mocap demos, deliberately including sub-optimal data.

### C. Impact mechanics & effective mass / recoil
- **Stronge, W. J. (2000/2018). Impact Mechanics (1st/2nd ed.).** Cambridge University Press. — Foundational rigid-body impulse-momentum and effective-mass theory underlying `m_eff = 1/(n̂ᵀJM⁻¹Jᵀn̂)`.
- **Wang, Y., Dehio, N., & Kheddar, A. (2022). Predicting Impact-Induced Joint Velocity Jumps on Kinematic-Controlled Manipulator.** *IEEE RA-L, 7*(3), 6226–6233. arXiv:2202.12646. — Joint-velocity jump `Δq̇ = M⁻¹Jᵀι`; introduces a **composite-rigid-body (CRB) impact inertia** because the naive operational-space inertia is **inaccurate for stiff/high-gain joints** (≈82% error reduction; validated on a Franka with an ATI mini45 F/T sensor). *Directly relevant: position-controlled arms are the stiff-joint regime — use the CRB form.*
- **Wang, Y., & Kheddar, A. (2019). Impact-Friendly Robust Control Design with Task-Space Quadratic Optimization.** *RSS XV*, p. 32. DOI:10.15607/RSS.2019.XV.032. — Embeds the impact-induced state-jump as a **QP constraint** so pre-impact motion stays feasible through the recoil.
- **Wang, Y., Dehio, N., Tanguy, A., & Kheddar, A. (2023). Impact-Aware Task-Space Quadratic-Programming Control.** *IJRR, 42*(14), 1265–1282. arXiv:2006.01987. — Journal extension: feasible-set polyhedra constraining post-impact critical states; recoil enforced at the controller level.

### D. Optimal-control / trajectory generation for impact
- **Vu, et al. (2026). QP-based Impact Momentum Maximization for a Hammering Task by a Humanoid Robot.** *IEEE AMC 2026* (Daegu), Xplore doc 11435814. — QP that **co-optimizes effective mass and velocity** (`m_eff·v`) on HRP-5P. *The most direct source for "maximize both mass and velocity."* (Authors beyond "Vu" paywalled — cite as "Vu et al." pending full list.)
- **Ti, B., Gao, Y., Zhao, J., & Calinon, S. (2024). An Optimal Control Formulation of Tool Affordance Applied to Impact Tasks.** *IEEE T-RO, 40*, 1966–1982. arXiv:2402.05502. — iLQR+ADMM maximizing **directional velocity manipulability** `α=√(uᵀJJᵀu)` at impact, with tool-affordance grasp constraints; real nail-hammering on a 7-DoF arm. Treats pre-impact velocity as dominant in a pilot-hole regime.

### E. Reward shaping / constrained RL / smoothness
- **Stooke, A., Achiam, J., & Abbeel, P. (2020). Responsive Safety in RL by PID Lagrangian Methods.** *ICML 2020.* arXiv:2007.03964. — CMDP constraint enforcement; the **PID multiplier** damps the cost-overshoot/oscillation of naive Lagrangian — recipe for bounding joint-impulse on top of PPO.
- **Mysore, S., Mabsout, B., Mancuso, R., & Saenko, K. (2021). Regularizing Action Policies for Smooth Control with RL (CAPS).** *ICRA 2021*, 1810–1816. arXiv:2012.06644. — Temporal + spatial smoothness regularizers for sim-to-real-friendly control.
- **Wu, Z., Lian, W., Unhelkar, V., Tomizuka, M., & Schaal, S. (2021). Learning Dense Rewards for Contact-Rich Manipulation Tasks (DREM).** *ICRA 2021*, 6214–6221. arXiv:2011.08458. — Progress-not-position dense reward for contact-rich tasks (basis for the geometric `nail_depth_delta`).

## 3. Thematic synthesis

**Q1 — Trajectory generation.** Three optimization-based works converge on **maximizing `m_eff·v`, not velocity alone**: Vu (2026) co-optimizes both in a QP; Ti (2024) maximizes directional velocity manipulability (velocity-dominant in low-resistance regimes); Wang & Kheddar (2019/2023) align the strike axis with the principal axis of `Λ=(JM⁻¹Jᵀ)⁻¹` to raise `m_eff` while keeping recoil feasible. HMAMP (2025) reaches a similar high-`m_eff`, axially-aligned posture *incidentally* via imitation of the human back-swing — its ~+70–80% impulse gain is attributed mostly to the back-swing energy-storage phase, **not** the terminal posture. **Divergence:** explicit `m_eff` optimization (Vu, Wang-Kheddar) vs velocity-manipulability surrogate (Ti) vs learned style prior (HMAMP). **Gap:** no single work unifies an effective-mass/velocity QP with a tool-affordance manipulability cost, nor makes joint-impulse a hard *trajectory-generation* constraint (Wang-Kheddar enforce it only at the controller level).

**Q2 — Tracking + task (resolves "RL = tracking only?").** Unanimous: **striking is never learned by pure tracking.** DeepMimic mixes `0.7` imitation + `0.3` task and shows imitation-only fails on strike/throw tasks; AMP and HMAMP both pair a style/imitation prior with an explicit task reward (HMAMP `0.6` task + `0.4` style, on real hammering). → For this project, plan **tracking + a thin impact/impulse task term**; "RL = tracking only" holds only if Stage-1 generation is already optimal.

**Q3 — Offline vs online.** For generate-then-track in simulation, **online PPO tracking is the natural fit** (PPO is retained as the tracker). Offline RL (IQL — OOD-safe, good at stitching) on the D4RL `hammer` precedent is the alternative when learning from a fixed demonstration corpus rather than fresh sim rollouts.

**Q4 — Bi-objective & recoil.** Recoil is analytic: `Δq̇ = M⁻¹Jᵀι`, minimize `‖Δq̇‖` (Wang/Dehio/Kheddar 2022), and **the CRB inertia must replace the naive `m_eff`** on stiff position-controlled joints. Enforcement options: a soft penalty, or a **CMDP + PID-Lagrangian** constraint `‖Δq̇‖ ≤ d` (Stooke 2020). The impact-aware-QP line (Wang-Kheddar) suggests posture is a partial *shared lever* (aligning `n̂` with `Λ`'s principal axis raises `m_eff` and limits recoil propagation), but a genuine tension remains since both `ι` and `Δq̇` scale with `m_eff·v` — the trade-off is real and is the natural RL/optimization target.

**The opening (defensible contribution).** No reviewed work combines (a) upstream trajectory optimization that *co-optimizes `m_eff·v` impact AND joint-impulse*, (b) RL tracking + a thin bi-objective task reward, and (c) a **force-sensor-measured impulse** feeding both the impact reward and the CRB-based recoil prediction, on a **position-controlled** arm. That intersection is the project's novelty.

## 4. Citation-integrity notes (corrections caught in verification)
- **AMP:** add co-author **Kanazawa**; exact title is "…for Stylized Physics-Based Character Control."
- **IQL:** publication year is **2022** (ICLR 2022); arXiv posted Oct 2021.
- **D4RL:** **preprint only** — never appeared at a peer-reviewed venue; cite as arXiv.
- **CAPS:** actual title is "Regularizing Action Policies for Smooth Control with Reinforcement Learning" (not the acronym expansion).
- **Wang/Dehio/Kheddar 2022:** exact title "…on Kinematic-Controlled Manipulator" (no "position-controlled" in title).
- **Vu et al. 2026:** existence + venue confirmed; **full author list is paywalled** — cite "Vu et al." until retrieved.
- **HMAMP impulse figure:** the paper's `I = 4238 kg·m/s` (Table I) is **physically implausible for a single strike** (≈400–800× too large) — likely cumulative/episode or a unit issue in the paper; use only the **relative +70–80%** comparison, not the absolute value.
- **Stronge:** specify edition (1st 2000 / 2nd 2018) when citing.

## 5. Limitations & AI disclosure
- **Limitations:** Vu (2026) author list and internal formulas are paywalled (existence + venue confirmed, not full-text re-read). Some internal numbers (DeepMimic strike-ablation percentages, HMAMP table values) are single-source reads worth a glance against the PDFs before manuscript use. No new empirical results here — this is a synthesis of existing literature.
- **AI disclosure:** Compiled with AI assistance (Claude Opus 4.8 orchestration; Sonnet verification subagents performing web search/fetch). Every source was independently checked for existence and correct citation; unconfirmable items would have been marked FAIL (none were). Findings from an earlier automated harness whose verification phase failed were **not** trusted on that basis — all citations here were re-verified.
