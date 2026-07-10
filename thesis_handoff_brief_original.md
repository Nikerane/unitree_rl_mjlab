> ⚠️ **DATED RESEARCH RECORD** (2026-05 original handoff brief; architecture sections superseded by `thesis_direction_update.md`) — kept for provenance.
> Facts below reflect their date and may contradict the current code; check `docs/README.md`.

# Master's Thesis — Handoff Brief

This document summarizes a long planning conversation so a new agent can continue seamlessly. It covers the student, the thesis topic, the technical understanding reached, the open decisions, and where things stand.

> **NOTE (added):** This is the *original* brief. The thesis direction shifted after a later conversation with the supervisor — see the companion document **`thesis_direction_update.md`**, which supersedes parts of the architecture described below (two-level SURE+RL → single policy; reference demoted to a weak motion prior; CBO relocated from reference-generator to candidate policy-optimizer). Read this brief first for full context, then the update for what changed.

---

## The student & context

- Master's student in Robotic Systems Engineering, doing a 6-month thesis at **TU Munich, ATARI Lab, under Prof. Majid Khadiv**.
- Background: humanoid mechatronics internship (Neura Robotics, Isaac Lab, VLA models), mobile robotics + MPC (Bosch), quadruped RL (RWTH, Unitree A1), Spot integration, MPC bachelor thesis. Strong **physical-AI / hardware** profile. Wants to stay in physical AI / robotics.
- German level ~B2; currently relocating to Munich for the thesis.
- Currently doing **online RL (PPO) in mjlab** (MuJoCo-based) and learning the trajectory-optimization side.
- Has basic familiarity with MPC and CasADi.

---

## The thesis topic (assigned by Prof. Khadiv)

**Impact-safe contact-rich manipulation on the Unitree G1 humanoid.**

Core problem: design a policy that **MAXIMIZES end-effector impulse** (e.g. for hammering/striking) while **CONSTRAINING per-joint impulse** to stay within hardware-safe actuator limits, even under **contact-timing uncertainty**.

It lives at the **intersection of two papers**, both from Khadiv's lineage:
1. **SURE** (Zhang, Zhao, Sun, Johnson, Khadiv, 2026, arXiv:2602.06864) — branching-and-rejoining trajectory optimization robust to contact-timing uncertainty. NOTE: SURE *minimizes* impact (egg-catching). The thesis *inverts* this to maximize impact.
2. **Learning Variable Impedance Control** (Bogdanovic, Khadiv, Righetti, 2020, arXiv:1907.07500) — RL policy outputs time-varying joint stiffness + desired position. This is the RL half.

Concrete demonstration task settled on: **hammering a nail** (clean, measurable, intuitive; impact effectiveness = nail penetration, safety = joint torques). Likely **fixed-base arm first**, base/locomotion as stretch goal.

---

## The proposed thesis arc (a progression of increasingly capable strikers)

1. **Fixed-gain RL striker** — RL in mjlab, policy outputs desired joint positions, fixed PD gains. Establishes RL can strike. Baseline 1.
2. **Variable-impedance RL striker** — policy ALSO outputs time-varying stiffness gains (per Bogdanovic 2020). Reward = maximize impact (flux) + penalize per-joint impulse over limits. Should learn a **low-high-low** stiffness pattern. Beats baseline 1. This is the core RL result.
3. **SURE-robust variable-impedance striker** — bring in SURE: invert its objective to maximize flux subject to joint-impulse constraints, use its branching structure for timing robustness. Two-level architecture: SURE = "what trajectory (robust to timing)"; RL = "what compliance (adapting to reality)." SURE reference enters as observation/reward term (residual RL), policy still trained ONLINE.
4. **Ablations + hardware** — fixed vs variable gain; with vs without SURE; vs pure domain randomization; varying timing uncertainty; transfer to real G1.

**Critical sequencing rule:** validate each RL phase as a standalone result BEFORE integrating SURE. If SURE integration runs out of time, the RL halves alone are still a complete thesis.

---

## Key technical understanding reached (the conceptual pillars)

### A. The two sides of one collision
One contact impulse Λ at impact does two things simultaneously:
- **Object side** (drives nail in) — governed by **hitting flux**. This is the OBJECTIVE to maximize.
- **Robot side** (reaction JᵀΛ travels back into joints) — this is the CONSTRAINT to bound per joint.
Thesis claim: with the right posture + time-varying impedance, these can be **partially decoupled** — push flux high while keeping per-joint impulse bounded. Frame everything in **impulse**, not energy.

### B. Hitting flux ≠ "mass × velocity"
Flux = **directional effective inertia × velocity in hit direction**. Effective inertia = operational-space inertia Λ = (J M⁻¹ Jᵀ)⁻¹, which is **configuration-dependent** (same arm, different posture → different effective mass). Flux is the *invariant* predicting post-impact object behavior. Source: Khurana & Billard, IEEE T-RO 2023.

### C. What SURE has vs. what the thesis adds (grounded in the paper)
- Effective mass IS in SURE (Delassus matrix G = Jc M⁻¹ Jcᵀ; named in egg-catching) BUT SURE **assumes it away** — assumes ball mass ≪ effective mass, reduces impact-minimization to relative-velocity minimization (Eqs. 14, 20). Never computes/optimizes it.
- SURE gaps the thesis fills: (1) SURE minimizes, thesis maximizes; (2) SURE assumes effective mass constant, thesis actively shapes it; (3) SURE has no per-joint impulse constraint and no learned policy.
- **SURE's own future-work section** explicitly names "learn uncertainty-conditioned policies" + "extend to floating-base systems for loco-manipulation" — i.e. the thesis executes the authors' stated next steps.
- SURE is implemented in **CasADi + Opti + IPOPT** → can likely modify their structure rather than rebuild.

### D. The deep physical insight: preparation, not reaction
- Impact is ~milliseconds; control loop 500 Hz–1 kHz → **the collision is over before feedback arrives.** No policy can sense impact and soften it in real time.
- Safety comes from being in the right **configuration + impedance state BEFORE** impact. The low-high-low impedance schedule is **pre-planned**, timed to expected contact, NOT a real-time reaction. Compliance dissipates momentum **passively** via pre-set damping.
- Without a contact sensor → SURE's **"robust nominal trajectory"** regime: one pre-planned motion safe across the ENTIRE timing window.
- Honest ceiling: no feedback → wider timing uncertainty forces giving up peak impact for worst-case safety. **This tradeoff curve is itself a publishable result.**

### E. Online RL is correct (not offline)
- Online RL (PPO) collects fresh experience each iteration. SURE reference is NOT a dataset and NOT behavior cloning — it's a reference (observation / reward-shaping term). Stays online even with SURE.
- Two orthogonal axes: (online vs offline data) × (from-scratch vs reference-guided). Thesis = **online + reference-guided**.

### F. "Won't the policy just copy the reference?" — no, if tuned right
- Reference is in the REWARD, not a copying loss. Policy optimizes the whole reward. Open-loop reference fails under perturbation, so policy is forced to deviate (the residual = where learning lives).
- Real failure mode = **degenerate tracking** (over-weighting tracking reward). Prevent via: (1) reward weighting (task dominates, anneal tracking down); (2) residual formulation; (3) domain randomization.
- Diagnostic: plot achieved impact vs reference impact. Match everywhere → copying; beat under perturbation → healthy.

### G. SURE gives a FAMILY of trajectories, not one
- Branches per solve (10 egg, 5 cart-pole) span timing uncertainty. Many solves across scenarios span task variation.
- Plan: **pre-compute a SURE library offline**, indexed by scenario; RL reads from it. Match each episode's conditions to the right reference. Keep randomization within library coverage.

### H. Domain randomization vs SURE — THE central debate (and a research question)
- DR can replicate the uncertainty INPUTS (randomize nail height = timing, etc.) but NOT the KIND OF GUARANTEE.
- DR + reward penalty = **soft, statistical, in-expectation** safety (can still violate on tails).
- SURE hard constraint = **hard, worst-case, certifiable** safety.
- **Important correction:** putting SURE in the REWARD does NOT transfer its hard guarantee — the policy can deviate from a reward term. The guarantee lived in SURE's optimization constraints.
- Safety-constraint placement, 3 options (increasing strength): (1) soft reward penalty; (2) SURE-in-reward (soft, but adds structure/interpretability); (3) SURE as **hard runtime safety filter / bounded-correction architecture** (preserves guarantee, hardest).
- The comparison (pure-DR RL vs SURE-guided RL; soft vs hard) is one of the thesis's **core experiments**, not a settled decision.

---

## Tooling / stack

- **Simulation/RL:** mjlab / MuJoCo (+ MJX for GPU parallelism). PPO. ATARI lab uses MuJoCo + Pinocchio (`mj_pin_utils` bridge).
- **Dynamics:** Pinocchio (impact map, effective inertia, impulse computations: `impulseDynamics`).
- **Trajectory optimization:** SURE uses CasADi + Opti + IPOPT. (Crocoddyl/HPIPM are alternatives.)
- **Hardware:** Unitree G1, low-level control via DDS `rt/lowcmd` at ~500 Hz, accepts (q_des, kp, kd, tau_ff) per joint — a natural match for the variable-impedance action space.
- **Robot model:** G1 in MuJoCo Menagerie (29-DoF MJCF, MJX-ready).

---

## Must-read papers (read in this order)

1. **Bogdanovic, Khadiv, Righetti (2020)** — Learning Variable Impedance Control, RA-L, arXiv:1907.07500 (RL half).
2. **SURE (2026)** arXiv:2602.06864 + predecessor **Zhao & Khadiv (2024)** arXiv:2407.11478 (TO half; the method inverted).
3. **Konno et al. (2011)** — impact dynamics + sequential optimization, IJRR 30(13) (impulse maximization w/ stability on humanoid).
4. **Wang & Kheddar et al. (2023)** — Impact-Aware Task-Space QP Control, IJRR 42(14), arXiv:2006.01987 (joint-impulse constraint + chain propagation).

Mental map: Konno = maximize impact; Wang & Kheddar = constrain joint impulse; SURE = timing uncertainty; Bogdanovic = learn impedance; thesis = synthesis.

Other valuable refs:
- **Khurana & Billard (EPFL/LASA), IEEE T-RO 2023** — "hitting flux" metric; and working paper "Hitting with Different Joints of a Robotic Manipulator" (exploits higher effective inertia at non-EE joints; near-verbatim inverse of the thesis concept). Part of EU **I.AM. project**.
- **Saccon group (TU Eindhoven)** — reference spreading (impact-aware control under timing mismatch); sister effort to EPFL under I.AM.
- **Yang & Posa** — impact-invariant control (projection-based robustness to impact timing).
- **Force-capable G1 RL:** arXiv:2511.21169 (kinematics-aware multi-policy), arXiv:2511.07407 (fall safety), FALCON (LeCAR-Lab, MIT-licensed, G1 force curriculum respecting torque limits).
- **Robotic hammering anchors:** HRP-5P QP impact-momentum maximization (IEEE 2025); iLQR+ADMM tool-affordance hammering (arXiv:2402.05502, Calinon); VSA "hammer/kick" optimal control (Garabini/Haddadin/Braun).
- **Safe/constrained RL:** omnisafe (PID-Lagrangian PPO), CPO, Reward-Constrained Policy Optimization, Dalal safety layer.
- A second opportunity was considered and set aside: **M-HOF-RL** (Xudong Sun, automatic multi-objective reward-weight tuning via PID-like control). Recommendation was to do the impact thesis; M-HOF could optionally be used as a *tool* (reward-weight tuner) inside it, not as the contribution.

NOTE on dates: SURE (2602.xxxxx) and several G1 papers (2511.xxxxx) are very recent (late-2025/2026); verify final venues. SURE code is NOT public yet — ask the authors / monitor github.com/Atarilab.

---

## Open questions to settle with Prof. Khadiv early

1. Will Khadiv be primary supervisor, or is day-to-day supervision delegated to a PhD student?
2. SURE integration: Route A (full two-level, ambitious) vs Route B (timing-randomized RL informed by SURE) acceptable? Determines time split.
3. Can the SURE CasADi/Opti code be shared as scaffold?
4. Deliverable: sim + hardware, or sim-only with thorough analysis acceptable?
5. Where should the safety constraint live — soft reward, SURE-in-reward, or hard runtime layer?

---

## Where things stand / next steps for the continuing agent

The student understands the problem deeply (concepts, gap vs SURE, physical constraints, RL design). Likely next concrete actions:
- Confirm scope with Khadiv (the 5 questions above).
- Build/validate the **fixed-gain RL striking task in mjlab** first (Phase 1), then variable-gain (Phase 2).
- Set up Pinocchio impact-map computation on the G1 and verify against MuJoCo.
- Read the four must-read papers; pull the Khurana/Billard flux paper for the objective formulation.
- Decide pre-computed SURE library design and the episode-to-reference matching pipeline.
- Plan ablations including the pure-DR-vs-SURE comparison and the safety-constraint-placement comparison.

A suggested first deliverable to the student: concrete reward-function and inverted-SURE-cost formulations for the striking task (was offered but not yet written).
