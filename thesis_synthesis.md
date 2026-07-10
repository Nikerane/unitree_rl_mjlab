> ⚠️ **DATED RESEARCH RECORD** (2026-06-10 synthesis) — kept for provenance.
> The "Z1 lacks the impulse constraint / start fresh on the G1" framing is SUPERSEDED by the 2026-06-17 Z1-primary direction (`CLAUDE.md`); the substep impulse accumulator + soft-CaT now ship on the Z1. Conceptual analysis remains useful; check `docs/README.md` for current truth.

# Thesis Direction — Deep-Research Synthesis

**Method:** ARS deep-research, local-corpus mode. Sources: `thesis_handoff_brief_original.md` (original), `thesis_direction_update.md` (post-supervisor update). Cross-referenced against the current Z1 hammer work in `docs/HANDOVER.md` and `docs/research/`. No new external literature pulled — the existing bibliographies inside both docs are taken as inherited.
**Date:** 2026-06-10.
**Status:** the two source docs are *sequential, not contradictory*. The update supersedes the original on architecture. Everything else (task, robot, physics, tooling, reading list, sequencing philosophy) is unchanged.

---

## 1. Executive summary

The Master's thesis is **impact-safe contact-rich manipulation on the Unitree G1**, demonstrated by **hammering a nail with a fixed-base arm first**, and conducted at TU Munich (ATARI Lab, Prof. Khadiv). The intellectual core has two halves: **maximize end-effector impact (hitting flux)** while **bounding per-joint impulse** to stay within actuator-safe limits, under contact-timing uncertainty.

The architecture was originally a two-level system (SURE-based trajectory optimizer feeds an RL tracker), but after a supervisor conversation it **collapsed to one policy**. A single online RL policy now owns both physics objectives — it maximizes flux in the reward and bounds joint impulse via either a CMDP/Lagrangian constraint or a runtime safety filter. The reference trajectory still exists but is demoted to a weak motion prior. CBO (consensus-based optimization) was reframed: not a separate reference generator, but a candidate optimizer for the *one* policy, compared head-to-head against PPO.

The thesis is now **lower-risk and more clearly RL-centric**. The trajectory-optimization machinery (SURE/CBO-as-planner) is no longer a required scaffold; it is reframed as an optimizer choice and an optional source of structure, to be justified by whether it measurably helps.

The current Z1 hammer work in this repo is not directly the thesis — it lacks the variable-impedance action space and the explicit impulse constraint that define the thesis architecture. But it is **directly transferable preparation**: same task (drive a nail), same physics simulator (mjlab/MuJoCo), and most of the open issues identified in `HANDOVER.md` (the press exploit, the press-vs-strike framing, the missing F/T sensing) map onto known gaps the thesis architecture is built to close.

---

## 2. The thesis as it now stands

### 2.1 The architecture (post-update)

> One policy π(state) → (desired joint positions + time-varying joint stiffness), trained online.
> **Reward:** maximize hitting flux (directional effective inertia × velocity at the end-effector), plus a *weak* tracking term toward a motion-prior reference.
> **Constraint:** per-joint impulse ≤ actuator-safe limit, entering either as an explicit CMDP/Lagrangian constraint or as a soft penalty. Computed via the Pinocchio impact map (`impulseDynamics`) on the predicted pre-impact configuration.
> **Timing uncertainty:** via domain randomization, and/or a timing-robust reference trajectory — not yet decided.
> **Optimizer:** PPO baseline and (potentially) CBO as a global derivative-free comparator.
> **Hard safety (optional):** a runtime impulse-projection filter (e.g. an impact-aware task-space QP, à la Wang & Kheddar 2023) layered on top, independent of the optimizer, only if a certifiable safety guarantee is required for hardware deployment.

### 2.2 The non-negotiable conceptual pillars

The following survive the architectural change unchanged. They are *what* the thesis is about; only the *how* shifted.

- **Impulse, not energy.** The right physical currency for impact tasks is impulse. Flux = directional effective inertia × velocity, evaluated at the end-effector, in the hit direction (Khurana & Billard, T-RO 2023). Effective inertia is `Λ(q) = (J M⁻¹ Jᵀ)⁻¹` — configuration-dependent, the lever the policy can shape via posture.
- **Preparation, not reaction.** A contact is over in milliseconds; the control loop runs at 500 Hz–1 kHz; *the policy cannot sense the impact and react during it.* Safety must come from being in the right configuration + impedance state *before* impact. The expected low–high–low impedance schedule is pre-planned, timed to expected contact, not a feedback response.
- **Variable impedance is load-bearing.** The policy outputs joint stiffness AND desired position (Bogdanovic, Khadiv, Righetti 2020). Without this, there is no impedance lever and the thesis collapses to a generic striking baseline. This is the single biggest difference from the current Z1 hammer task, which uses fixed PD gains.
- **Online RL with reference as observation/reward, not behavior cloning.** Even when a reference is used, training is online (PPO collects fresh experience). The reference enters the reward as a tracking term, not as a supervised loss. This holds in both the original and updated architectures.

### 2.3 What changed (axis by axis)

| Axis | Original (handoff brief) | Updated (post-supervisor) |
|------|--------------------------|---------------------------|
| Top-level architecture | Two levels: SURE generates trajectory, RL tracks | **One policy**, RL owns objectives directly |
| Role of reference | Optimized SURE trajectory carrying impact objective + impulse limit | **Weak motion prior** only — gross structure, no objective, no constraint |
| Impact maximization | Property of SURE's solution | **RL reward term** (hitting flux) |
| Per-joint impulse limit | Hard constraint inside SURE's QP | Explicit RL **constraint** (CMDP/Lagrangian) or penalty |
| Safety guarantee | Implicitly assumed "hard" (but actually wasn't, because once trajectory entered RL as a tracking term, the guarantee was lost) | **Explicit design choice**: soft (CMDP) vs hard (runtime QP filter), decoupled from optimizer |
| CBO role | Candidate *trajectory reference generator* | Candidate **optimizer for the one policy** (PPO vs CBO comparison) |
| Timing robustness | SURE's job (branch-and-rejoin) | **Open question**: domain randomization, or a timing-robust reference, or both |

### 2.4 Why the collapse is defensible

The reasoning, as recorded in the update: TO and RL minimize the same objective (Σ cost subject to dynamics). Having SURE solve "max impact s.t. impulse limits" and then having RL re-solve the same problem on top is solving it twice. Collapsing the redundancy is principled, not just a simplification of convenience. It also clarifies a real confusion in the original brief: SURE's "hard" guarantee evaporates the moment its output enters RL as a tracking *reward term* (a soft signal). The original design was already soft in practice; the update just admits this and makes the hardness an explicit decision.

The cost of the collapse is that the policy must now discover the impedance schedule, the posture for high `m_eff`, and the timing all on its own, with only a weak reference for structure. The bet is that variable impedance + a flux-based reward + explicit impulse constraint is a sharp enough learning signal to make this tractable. Whether this bet pays off is essentially the empirical question the thesis answers.

---

## 3. Mapping the current Z1 hammer work onto the thesis

The Z1 work in `unitree_rl_mjlab` (this repo, `hammer-z1` branch) is **directly transferable preparation**, not a parallel project. The mapping is:

| Z1 hammer current (per `HANDOVER.md`) | Thesis target (per update) |
|---------------------------------------|----------------------------|
| Unitree Z1 6-DOF arm, fixed base | Unitree G1 humanoid; **fixed-base arm first** — same starting point |
| mjlab / MuJoCo-Warp | mjlab / MuJoCo / MJX — identical |
| Action: 3-D DiffIK position delta of hammer head | Action: desired joint positions + **time-varying joint stiffness** — different |
| Action space: position-only, fixed PD gains | Variable impedance (per Bogdanovic 2020) — **the key upgrade** |
| Reward: 7-term mix (approach, depth, completion, smoothness, double-gated momentum) | Reward: hitting flux + weak tracking + impulse penalty/constraint |
| Safety: soft `joint_pos_limits` penalty | Explicit per-joint impulse constraint (CMDP / Lagrangian / runtime filter) |
| No force sensing | Force sensing / impulse computation via Pinocchio impact map |
| Episode-deterministic; one fixed reset pose | Domain randomization for timing/posture/parameters |

What the Z1 work already *gets right* for the thesis:
- **mjlab familiarity.** The thesis runs on the same stack. PPO, scene XMLs, reward managers, contact sensors, validation script discipline — all directly carry over.
- **The press exploit finding.** The peer review on the Z1 work (`PEER_REVIEW_v2.md`) discovered empirically that a position-only policy with stiff PD gains solves the hammering task by sustained pressing, not impulsive striking. *This is exactly the failure mode variable impedance is designed to solve.* The Z1 work has therefore done the diagnostic work that justifies the thesis architecture, even if it cannot itself be a complete thesis result on this action space.
- **The literature scan.** The v2 deep dive (`hammering_reward_design_deep_dive_v2.md`) and the impact-tracking lit review independently surfaced the same core sources the thesis identifies (Wang/Dehio/Kheddar 2022, Ti et al. 2024, Khurana & Billard 2023, Vu et al. 2026). The bibliography overlap is substantial; this saves real reading time on the thesis side.
- **The validation discipline.** `validate_rewards.py` + `verify_contact_sensor.py` + the unit-test pattern (`test_impact_progress_reward.py`) is exactly the pre-training gate the thesis will need before any G1 training run. The pattern is reusable.

What the Z1 work cannot solve (and why the thesis exists):
- **No variable-impedance action space.** The Z1 task uses fixed PD gains baked into the actuator config. Adding variable impedance is non-trivial: it requires a different action interface (e.g., emitting `(q_des, kp_t, kd_t)` per joint per step), changes to the actuator wrapper, and observations that expose stiffness state. None of this is in the current Z1 task.
- **No impulse computation.** The Z1 task has no `J`/`M` computation, no Pinocchio integration, no impact-map. The current `impact_progress` term finite-differences hammer-head position to get axial velocity, which is a poor man's stand-in for `m_eff · v`. The thesis needs the real thing.
- **No CMDP machinery.** The Z1 task uses soft reward penalties; the thesis (at least one branch) needs PID-Lagrangian PPO or equivalent.

In short: the Z1 work is the **right Phase 1 of the thesis arc** (per the original brief: "Fixed-gain RL striker — establishes RL can strike. Baseline 1"). The press-exploit finding adds empirical weight to *why* the thesis's Phase 2 (variable impedance) is necessary, not just a methodological choice.

---

## 4. The four still-unresolved design decisions

These appear in the update's §5 "Open questions for the supervisor." Each is genuinely open and each has downstream consequences.

### 4.1 Is CBO the optimizer, or just a comparison?

The supervisor's framing was "combine CBO with RL into one policy." Two readings:

- **Strong reading:** CBO is *the* optimizer; PPO is the baseline; the thesis's methodological novelty is "global derivative-free policy optimization for impact-RL." Coupling consequence: the impulse constraint cannot use a PPO-Lagrangian (gradient primal-dual); it must use a penalty, projection, or augmented Lagrangian over CBO's particle population.
- **Weak reading:** PPO is the workhorse; CBO is a comparison ablation (does global > local for this nonconvex landscape?). The impulse constraint defaults to PID-Lagrangian PPO and CBO inherits a different constraint mechanism for the ablation only.

The weak reading is lower risk and still produces a publishable PPO-vs-CBO ablation. The strong reading is higher risk but is potentially the thesis's distinctive methodological contribution.

### 4.2 Where does timing robustness live?

Two options spelled out in the update:

- **Option 1 — timing-robust reference from a TO.** The reference is a robust nominal motion (SURE-style branch-and-rejoin, or a CBO-generated robust trajectory). This is now a *much easier* TO problem because the reference no longer carries impact maximization or impulse limits — just "be a sensible motion across a window of contact times."
- **Option 2 — RL handles it via domain randomization.** Randomize contact timing / nail height / etc. Reference can then be trivially cheap (heuristic swing, kinematic motion, demo, prior policy rollout).

Option 2 is cleaner and removes the TO dependency entirely. Option 1 is theoretically tighter and retains a connection to SURE's intellectual lineage, which the supervisor's group cares about. The right answer probably depends on how wide the timing uncertainty actually is in the deployed scenario — narrow uncertainty makes Option 2 dominant; wide uncertainty pushes back toward Option 1.

### 4.3 What level of safety guarantee is required?

Three increasing levels:

- **Soft / in-expectation:** reward penalty, or CMDP/Lagrangian constraint. Safe on average, can still violate on tails.
- **Hard / certifiable on the trained policy:** PID-Lagrangian with convergence to constraint satisfaction in the limit.
- **Hard / runtime-enforced:** a safety filter (e.g. impact-aware task-space QP) projects every action onto the impulse-feasible set at deploy time. This is the only level that survives policy distribution shift on hardware.

Tied to whether the deliverable is sim-only or includes hardware. Sim-only → soft is acceptable. Hardware → runtime filter is the responsible choice.

### 4.4 Supervisor & code-sharing logistics (carried over from original)

Primary vs delegated supervision; whether the SURE/CBO CasADi code can be shared as scaffold; sim-only vs hardware deliverable.

---

## 5. Devil's-advocate pass

Three places where the new architecture could fail or be weaker than it looks:

**A. The "weak reference" knob is harder to tune than it sounds.** The original brief flagged "degenerate tracking" as the central failure mode of reference-guided RL (pillar F). The update makes the reference *weaker* and explicitly tells the reader the tracking term must be low-weight or annealed. But the more you weaken the reference, the less it does — at the limit of zero weight it's not a reference at all, and the policy has to discover the swing from scratch under a sparse-ish flux reward. There's a real Goldilocks problem here. The thesis will spend non-trivial empirical effort finding a tracking-weight schedule that anchors exploration without dominating.

**B. CBO's "global" promise on policy weights is unproven for this scale.** CBO has convergence theory and works well on lower-dimensional optimization, but RL policies for a 6+ DoF manipulator have thousands to millions of parameters. The literature on CBO at policy-network scale is thin. The strong-reading "CBO is the optimizer" path could spend a lot of time discovering that the particle population can't escape the same local optima PPO can't escape, just slower. The fallback (weak reading: comparison ablation) is safer.

**C. The constrained-MDP for joint impulse needs a clean signal.** PID-Lagrangian requires that the cost (here, per-joint impulse) be measurable per episode and roughly stationary in distribution given the policy. Joint impulse on a contact event is a sparse, high-variance signal. It may need careful normalization, windowing, or filtering before the Lagrangian dual update is stable. The original brief noted this; the update inherits the problem. Practical mitigation: compute impulse over a fixed time window around the contact event (already required by the spec) and use a robust statistic (median or 95th percentile) rather than the per-step value.

These are all addressable; none is fatal. But each is a place where engineering time will be burned, and they should be planned for explicitly.

---

## 6. Recommended sequencing

Based on the update's §6 ("Implications for immediate next steps") and the Z1 work that already exists, the cleanest sequence is:

**Phase 0 (immediate, ~1 week) — finish the diagnostic value of the Z1 work.**
Apply the cheap fixes from `docs/FUTURE_UPDATES.md` (`clip_actions=1.0`, raise nail `frictionloss` to ~30 N, reset randomization, nail geometry fix). If the press exploit dissolves under fixed PD gains + tuned physics, that's a useful negative result; if it persists, that's even more empirical justification for variable impedance. Either way, write up the finding because it's a publishable side result for the thesis introduction ("on a position-only stiff-PD action space, the sim allows quasi-static solutions; variable impedance is necessary").

**Phase 1 (~3–4 weeks) — fixed-gain RL striker on the G1 in mjlab.**
Port the working pieces from the Z1 task to a G1 arm-only fixed-base setup. Same reward design discipline (validate_rewards pattern). Same press-exploit watchdog. This is "Baseline 1" from the original brief; it survives the architectural collapse unchanged. Confirms the G1 model is wired correctly, the impact_progress-style diagnostics work, and provides a known-good comparison point.

**Phase 2 (~6–8 weeks) — variable-impedance RL striker.**
This is the protagonist. Add the variable-stiffness action interface (`(q_des, kp_t, kd_t)` per joint), add observations for stiffness state, switch the reward to hitting flux (requires Pinocchio impact-map computation), add explicit per-joint impulse term (start as a penalty, upgrade to CMDP/PID-Lagrangian). Domain-randomize contact timing. Verify the policy discovers the low–high–low schedule via plotting stiffness over time at convergence. This is the thesis's core RL result.

**Phase 3 (~4 weeks, optional) — CBO comparison or timing-robust reference.**
Pick one of the open questions (§4.1 or §4.2) and run the ablation. If CBO, plug it in as the optimizer over the variable-impedance policy weights; compare to PPO-Lagrangian. If timing-robust reference, generate a SURE-lite or CBO-generated robust nominal motion and compare to pure-DR. This is where the thesis's distinctive contribution lands.

**Phase 4 (stretch, ~2–3 weeks) — hardware on the G1.**
Only with the runtime safety filter (impact-aware QP) layered on top. Real-robot deployment is a separate axis from algorithmic choice; both can be addressed independently.

Total: 14–22 weeks, fitting inside a 26-week thesis with buffer. Phases 0, 1, 2 are the minimum viable thesis; 3 and 4 are upside.

---

## 7. Open questions for the supervisor (consolidated)

1. **Is CBO meant to be the policy optimizer** (the methodological contribution, the PPO-vs-CBO comparison being the thesis's distinctive angle), or a comparison ablation only? The "strong reading" vs "weak reading" of §4.1 directly affects which constraint mechanism is feasible.
2. **Where does timing robustness live?** Domain randomization (Option 2, cleaner) or a timing-robust reference (Option 1, retains SURE lineage)?
3. **What level of safety guarantee is required?** Soft (CMDP/penalty, in-expectation) vs hard (runtime impulse-projection filter). Tied to whether hardware is in scope.
4. Primary vs delegated supervision; SURE/CBO code-sharing as scaffold; sim-only vs hardware deliverable (carried over from original brief).
5. *(Project-internal, added by this synthesis.)* The Z1 hammer work in this repo currently uses fixed PD gains and a position-only DiffIK action space. Is it acceptable to (a) finish the cheap physics fixes on it as a documented baseline / negative-result side-paper, and then (b) start fresh on the G1 with variable impedance, rather than retrofitting the Z1 task to variable impedance? Doing (b) preserves the diagnostic value of the Z1 work without committing engineering time to a code base that the thesis architecture obsoletes.

---

## 8. What this synthesis does *not* cover

- **No external literature search.** Both source docs already contain extensive bibliographies; I trust them as inherited evidence. The H3-style citation-fabrication risk applies — anyone using these citations for a thesis defense should re-verify each one against arXiv directly.
- **No concrete reward formulation.** The thesis architecture is specified at the level of *what the reward measures* (hitting flux), not the specific functional form, weights, or annealing schedule. The original brief flagged this gap ("suggested first deliverable: concrete reward-function and inverted-SURE-cost formulations — was offered but not yet written"). It remains a gap.
- **No analysis of the G1 model itself.** The thesis assumes the MuJoCo Menagerie G1 MJCF is sufficient. Worth verifying that the actuator model in the published MJCF actually supports variable joint stiffness commands (it may need a Bogdanovic-style override of the actuator class), and that the URDF used by Pinocchio matches the MJCF used by mjlab.
- **No timeline alignment with TU Munich academic dates.** The 14–22 week sequencing assumes uninterrupted work; thesis submission deadlines, supervisor availability, and ATARI-lab meeting cadences are not factored in.

---

## 9. Reading list for the next agent

In priority order:

1. `thesis_direction_update.md` — *the* current architectural truth.
2. `thesis_handoff_brief_original.md` — the conceptual pillars (impulse, preparation, flux) survive intact.
3. `docs/HANDOVER.md` — the Z1 hammer state, including the press-exploit diagnostic.
4. **Bogdanovic, Khadiv, Righetti (2020), arXiv:1907.07500** — the variable-impedance RL paper that defines the action space.
5. **Khurana & Billard (T-RO 2023)** — the hitting-flux objective.
6. **Wang, Dehio & Kheddar (RA-L 2022), arXiv:2202.12646** — joint-velocity jump / CRB inertia, the physics that makes pre-impact posture and impedance the controllable levers.
7. **SURE (Zhang et al. 2026, arXiv:2602.06864)** — the predecessor whose objective is being inverted; useful even though it is no longer in the architectural critical path.
8. **Wang & Kheddar et al. (IJRR 2023), arXiv:2006.01987** — the impact-aware task-space QP, candidate for the optional hard-safety runtime filter.

Everything else (Konno 2011, FALCON, Sun et al. CBO 2026, HRP-5P hammering, Ti et al. 2024) is supporting material to consult as the relevant axis becomes active.
