> ⚠️ **DATED RESEARCH RECORD** (supervisor-conversation update; pre-2026-06-10) — kept for provenance.
> The single-policy / variable-impedance / online-RL architecture it records is still current. The
> build-on-the-G1 platform framing is SUPERSEDED by the 2026-06-17 Z1-primary direction
> (`CLAUDE.md` DIRECTION UPDATE block). Check `docs/README.md` for current truth.

# Thesis Direction — Update / Addendum

**Read this alongside the original "Master's Thesis — Handoff Brief."** That brief is still 90% correct: the task, the robot, the physics, the tooling, and the reading list are unchanged. What changed is the *architecture* — how SURE, the reference trajectory, the RL policy, and CBO relate to each other. This document records that shift so anyone (or any agent) picking up the work knows what is now different and why.

The change came out of a planning conversation with the supervisor (Prof. Khadiv). The short version: **the design collapsed from a two-level (trajectory-optimizer + RL) system into a single policy that owns the physics objectives directly.**

---

## 1. The one-sentence summary of what changed

**Before:** A trajectory optimizer (SURE) solved "maximize impact subject to joint-impulse limits," and an RL policy *tracked* that optimized trajectory and adapted around it. The physics intelligence lived in the optimizer.

**After:** A **single RL policy** owns the physics objectives directly — it maximizes end-effector impact (hitting flux) and respects per-joint impulse limits *itself*, through its reward and constraints. A reference trajectory still exists, but it is demoted to a **weak motion prior** and no longer carries the impact objective or the safety limits.

---

## 2. What did NOT change (so the agent does not over-correct)

- **The task is the same:** impact-safe, contact-rich manipulation on the Unitree G1. Demonstration task still **hammering a nail**. Still **fixed-base arm first**, locomotion as a stretch goal.
- **Still online RL.** (Important — do not read this update as a switch to offline RL or behavior cloning. The policy is still trained online, e.g. PPO, collecting fresh experience. Pillar E of the original brief stands.)
- **Still variable impedance** (Bogdanovic 2020): the policy outputs time-varying joint stiffness + desired position. Expect it to learn a low–high–low stiffness schedule.
- **The physics insight is unchanged:** impact is faster than the control loop, so safety comes from *preparation* (pre-impact posture + impedance), not reaction. Frame everything in **impulse**, and use **hitting flux** (directional effective inertia × velocity) as the impact objective.
- **Tooling unchanged:** mjlab / MuJoCo / MJX for RL; Pinocchio for the impact map and impulse computations; G1 from MuJoCo Menagerie.
- **Sequencing philosophy unchanged:** build and validate the standalone RL striker first; add any extra machinery (references, alternative optimizers) only after that works.

---

## 3. The change, axis by axis (before → after)

### A. Architecture: two levels → one policy
- **Before:** Level 1 = SURE produces a robust, impact-maximizing, impulse-constrained reference trajectory. Level 2 = RL tracks it (residual RL) and adapts to reality.
- **After:** One policy. The reference is optional scaffolding, not a second optimization of the same problem.
- **Why:** Trajectory optimization (TO) and RL minimize the *same* objective (Σ cost subject to dynamics). Having SURE solve "max impact s.t. impulse limits" and then having RL effectively re-solve the same problem is solving it **twice**. Collapsing to one policy removes that redundancy.

### B. Role of the reference trajectory: optimized solution → weak motion prior
- **Before:** The reference *was* the SURE-optimized trajectory — it encoded the impact maximization and the joint-impulse limits, and RL tracked it.
- **After:** The reference is a **motion prior only**. It gives the policy gross structure / a sensible swing to anchor exploration around. It explicitly does **not** carry the impact objective and does **not** impose the joint-impulse limits.
- **Consequence:** the tracking term must stay **weak** (low weight, annealed, or residual formulation), otherwise the policy degenerates into copying the reference and never discovers the hard-hitting, safe-impedance behavior (the "degenerate tracking" failure mode from the original brief, pillar F). The reference anchors the *approach*; the RL owns the *strike*.

### C. Where the physics objectives live: optimizer → RL reward + constraint
- **Before:** "maximize impact" and "respect joint-impulse limits" were properties of the optimizer's solution.
- **After:** both live entirely in the RL:
  - **Maximize end-effector impulse / hitting flux** → the RL **reward**.
  - **Bound per-joint impulse** → an explicit RL **constraint** (a Constrained MDP / Lagrangian formulation, e.g. PPO-Lagrangian / PID-Lagrangian via omnisafe), or at minimum a penalty.
- **Note:** the per-joint impulse number must be *computed* to be used in either case — via the **Pinocchio impact map** (`impulseDynamics`), evaluated *before* impact from the predicted configuration/impedance. This computation is a prerequisite regardless of the rest of the design.

### D. The safety guarantee: was implicitly "hard," is now explicitly chosen
- **Clarification reached:** in the old plan, SURE's joint-impulse limits were a *hard constraint inside its optimizer*, but once that trajectory entered RL only as a tracking term, the guarantee was **lost** — a reward term is soft, and the policy can deviate from it. So the old design was, in practice, already soft.
- **After:** the safety constraint's hardness is now an **explicit design choice**, decoupled from the reference:
  - **Soft / in-expectation:** reward penalty, or a CMDP/Lagrangian constraint (safe on average; can still violate on tails).
  - **Hard / certifiable:** a **runtime safety filter** that projects the policy's action onto the joint-impulse-feasible set every control step (e.g. an impact-aware task-space QP, cf. Wang & Kheddar 2023). This is the only thing that gives a guarantee on the *deployed* policy, and it is **independent of the optimizer**.
- Useful framing: **optimizer choice** and **safety-enforcement mechanism** are two *separate* axes. The reference/CBO/PPO choice does not determine the guarantee; the constraint mechanism does.

### E. Role of CBO: candidate *reference generator* → candidate *policy optimizer*
- **Background:** CBO = Consensus-Based Optimization, a derivative-free, particle-based, *global* zero-order optimizer with convergence theory. Reference: *"Consensus-based optimization (CBO): Towards Global Optimality in Robotics,"* Sun, Jordana, Fornasier, Etesami et al., arXiv:2602.06868 (Feb 2026). Its pitch: MPPI/CEM/CMA-ES are *local*; CBO is *global*.
- **The professor's framing was always "combine CBO with RL into one policy"** — i.e. CBO as the **optimizer that trains the single policy** (evolution-strategies-style, over the policy weights), *not* as a separate planner feeding a reference into the RL.
- **So:** now that the RL owns the physics objectives, CBO does not become pointless — it relocates. It is a candidate **training engine for the one policy**, compared against PPO.
  - **Motivation:** the impact/impedance reward landscape is highly nonconvex and nonsmooth (contact discontinuities, the low–high–low schedule sits in a non-obvious basin, multiple local optima). A *global*, derivative-free optimizer is well-motivated for escaping the local optima PPO can get stuck in, and it parallelizes well on MJX.
  - **Status:** **not load-bearing.** The PPO version is a complete thesis on its own. CBO is the *methodological angle* — "is global derivative-free policy optimization better than local gradient RL on this problem?" — and a natural ablation (PPO vs CBO as the optimizer). It may be the intended novelty, or it may be dropped.
  - **Coupling:** if CBO is the optimizer, the impulse constraint cannot ride a PPO-Lagrangian (a gradient primal-dual method); it shifts to a penalty, a projection/filter, or an augmented-Lagrangian over the particle population.

### F. Timing robustness: was SURE's job → now the open question
- SURE's genuine specialty is robustness to **contact-timing uncertainty** (its branch-and-rejoin structure). With SURE no longer carrying impact/impulse, the open question is **where timing robustness now lives**:
  - **Option 1 — keep a trajectory optimizer for it:** the reference is a *timing-robust nominal motion* (SURE or CBO generated). Note this is now an *easier* TO problem (no impact-max, no impulse-constraint baked in), so it is cheaper and more reliable than full SURE.
  - **Option 2 — move it into RL:** handle timing uncertainty purely through **domain randomization** (randomize contact timing / nail height / etc.). Then the reference can be trivially cheap (a heuristic swing, a kinematic trajectory, a demo, a prior policy rollout) and SURE may not be needed at all.
- This single choice largely decides whether a real trajectory optimizer is needed in the pipeline.

---

## 4. The revised architecture in one place

> **One policy** π(state) → (desired joint positions + time-varying stiffness/impedance), trained **online**.
> - **Reward:** maximize hitting flux (directional effective inertia × velocity at the end-effector), + a **weak** tracking term toward a motion-prior reference.
> - **Constraint:** per-joint impulse ≤ actuator-safe limit, entering explicitly (CMDP / Lagrangian) or as a penalty; computed via the Pinocchio impact map.
> - **Timing uncertainty:** via domain randomization (Option 2) and/or a timing-robust reference (Option 1) — *to be decided*.
> - **Optimizer:** PPO (baseline) and/or **CBO** (global, derivative-free) as a comparison — *to be decided*.
> - **Hard safety (optional):** a runtime impulse-projection filter (e.g. impact-aware QP) layered on top, independent of the optimizer — *only if certifiable safety is required (e.g. for hardware)*.

---

## 5. Updated open questions for the supervisor

1. **Is CBO meant to be the policy optimizer** (the methodological contribution / a PPO-vs-CBO comparison), with the reference reduced to a weak motion prior? (This is the reading that reconciles the whole conversation.)
2. **Where should timing robustness live** — in a (timing-robust) reference from a trajectory optimizer, or entirely in RL via domain randomization?
3. **What level of safety guarantee is required** — in-expectation (CMDP/penalty) acceptable, or is a hard runtime filter needed? (Tied to whether the deliverable is sim-only or includes hardware.)
4. (Carried over) Primary vs delegated supervision; can the SURE/CBO code be shared as scaffold; sim-only vs hardware deliverable.

---

## 6. Implications for immediate next steps

The first build target is now sharper and *simpler* than the original two-level plan:

1. **Build the standalone constrained-RL striker first** in mjlab: one online policy, reward = hitting flux, **explicit** per-joint impulse constraint (start with a penalty, then upgrade to CMDP/Lagrangian), with a **weak** motion-prior reference if any. This is the protagonist of the thesis, not a baseline.
2. **Stand up the Pinocchio impact-map computation** (`impulseDynamics`) on the G1 and verify against MuJoCo — needed by every variant.
3. **Add timing uncertainty via domain randomization** and measure the impact-vs-safety tradeoff curve (this curve is itself a target result).
4. **Then, and only then,** treat the bigger structural choices as *experiments / ablations*: PPO vs CBO as the optimizer; soft (CMDP) vs hard (runtime filter) safety; cheap reference vs timing-robust reference.

**Net effect:** the thesis is now lower-risk and more clearly RL-centric. The trajectory-optimization side (SURE/CBO) is no longer a required scaffold the RL depends on — it is reframed as an optimizer choice and an optional source of structure, to be justified by whether it measurably helps.
