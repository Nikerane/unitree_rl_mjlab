> ⚠️ **ARCHIVED 2026-07-05** — superseded by the single-policy direction — the generate-then-track architecture was walked back; see `docs/thesis/README.md`.
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Reward-Function Design Spec — Stage-2 Tracking + Bi-Objective Impact

> ⚠️ **HISTORICAL (2026-06-10).** This spec describes the generate-then-track / DeepMimic-style architecture that was considered for the Z1 hammer task in early June. **It is *not* the current thesis direction.** Per `thesis_direction_update.md`, the supervisor walked back from two-level (TO + RL) and tracking-style architectures to a **single-policy variable-impedance online RL** design. The reasoning: TO and RL minimise the same objective, so having one solve "max impact s.t. impulse limits" and the other re-solve via tracking is solving it twice.
>
> Keep this spec as a candidate architecture for the standalone Z1 task only. Do not implement as the thesis. Read `thesis_synthesis.md` (repo root) and `docs/HANDOVER.md` §0 before starting work in this direction.

**Date:** 2026-06-02 · **Branch:** `hammer-z1` · **Status:** ~~design approved, pre-implementation~~ **HISTORICAL — superseded by thesis direction (2026-06-10).**
**Architecture:** generate-then-track (Stage-1 generates the strike; **this spec = the Stage-2 RL reward**).
**Grounding:** `docs/research/impact_tracking_rl_litreview.md` (verified citations).

## 1. Goal & scope

Design the **Stage-2 RL reward** for impact-explicit hammering on the position-controlled Z1, where the RL **tracks an upstream-generated strike** and refines it to **maximize delivered impact momentum** (`m_eff·v`) while **minimizing joint reaction impulse** (recoil). PPO stays as the online tracker.

**In scope:** the reward function (terms, formulas, signals, weights, training recipe).
**Prerequisites (NOT this spec — sequenced in the impl plan):** §6.

## 2. The reward function

$$R_t \;=\; w_I\,r_{\text{imit}} \;+\; w_G\,\big(r_{\text{impact}} - \lambda\,r_{\text{recoil}}\big) \;+\; r_{\text{safety}}$$

Mirrors the **imitation + task** structure that the literature shows is required for striking (never pure tracking — DeepMimic strike ablation; HMAMP `0.6` task + `0.4` style).

### Terms

| Term | Formula (starting form) | Signal | Paper |
|---|---|---|---|
| **`r_imit`** (track) | `0.7·r^p + 0.1·r^v + 0.2·r^e`; `r^p=exp(−2·Σ(q−q*(φ))²)`, `r^v=exp(−0.1·Σ(q̇−q̇*(φ))²)`, `r^e=exp(−40·‖p_head−p*_head(φ)‖²)` (CoM term dropped — fixed base) | reference `q*(φ),q̇*(φ),p*(φ)` + phase `φ` | **DeepMimic** (Peng et al. 2018, arXiv:1804.02717) |
| **`r_impact`** (max) | `(I_axial / I_ref)·𝟙[Δdepth>ε]`, `I_axial=∫_window F_axial dt` | **force sensor** (axial impulse over contact window) + nail qpos | **HMAMP** (Ma et al. 2025, arXiv:2510.24257); geom gate from **DREM** (Wu et al. 2021, arXiv:2011.08458) |
| **`r_recoil`** (min) | `‖Δq̇‖²` = squared **joint-velocity jump across the contact step**, *measured* from `entity.data.joint_vel` (qvel) — no Jacobian. Analytic `Δq̇=M⁻¹Jᵀι` (CRB) reserved for an optional **predictive critic** | measured `joint_vel` jump (or `qfrc_constraint`); `qM` available if the analytic critic form is added | **Wang/Dehio/Kheddar 2022** (arXiv:2202.12646); **Impact-Aware QP** (Wang et al. 2023, arXiv:2006.01987; RSS'19 DOI:10.15607/RSS.2019.XV.032) |
| **`r_safety`** | `−w_a‖a_t−a_{t−1}‖² − w_l·`jointlimits `+ w_d·`completion (+ optional CAPS) | existing | existing; **CAPS** (Mysore et al. 2021, arXiv:2012.06644) |
| *(later)* | escalate recoil → CMDP constraint `‖Δq̇‖≤d` via **PID-Lagrangian** | — | **Stooke et al. 2020** (arXiv:2007.03964) |

### Weights & training
- **Starting weights** (ablation knobs, not gospel): `w_I≈0.6, w_G≈0.4` (HMAMP) or `0.7/0.3` (DeepMimic); `λ` starts small, ramps; `ε=5e-4 m`, `I_ref` ≈ expected single-strike impulse.
- **Training recipe:** **Reference State Initialization + Early Termination** (DeepMimic — critical for dynamic skills): start at a random phase of the reference; terminate on large tracking error or failure.

## 3. How it dissolves the press
Two independent guards: a press has `v_axial≈0 → I_axial≈0 → r_impact≈0`, **and** `r_imit` structurally forbids departing from the back-swing→strike reference.

## 4. Relationship to the current reward
- **Replace** (subsumed): `approach`, `nail_driven`, online `impact_progress`.
- **Keep**: `nail_depth_delta` (now the geometric anti-artifact cross-check in `r_impact`'s gate), `completion`, `action_rate`, `joint_pos_limits`.

## 5. Validation & ablation
- **Unit tests** (stub env, à la `test_impact_progress_reward.py`): `r_imit`=1 at perfect track and decays with error; `r_impact` fires only on windowed impulse **and** depth-advance; `r_recoil` = `‖M⁻¹Jᵀι‖²`; all reset cleanly.
- **Ablations** (the research result): tracking-only → +impact → +impact−recoil → +PID-Lagrangian constraint. **Metrics:** success rate, delivered axial impulse, recoil `‖Δq̇‖`, tracking error. (Tracking-only-vs-+impact answers the "RL = tracking only?" question empirically.)
- **Gate:** the existing `validate_rewards.py` style state-injection test per term before any GPU run.

## 6. Prerequisites (must exist first — separate sub-tasks)
1. **Stage-1 reference trajectory** generator → `q*(φ),q̇*(φ),p*(φ)` (start hardcoded back-swing→strike; later trajectory-opt that co-optimizes `m_eff·v` and joint-impulse — Vu 2026 / Ti 2024 / Wang-Kheddar). *Without it, `r_imit` is undefined.*
2. **Force sensor** wiring: axial contact force/impulse over a window (extend the existing `ContactSensor(found/force)` or add a wrist F/T; integrate over `history_length`).
3. **Recoil computation**: `J` (head Jacobian), `M` (joint-space inertia) from `mjData`/mujoco_warp + **CRB** inertia for ι→Δq̇. ⚠️ v2 doc flagged J/M access cost — may need **critic-only or every-k-steps**.
4. **Phase variable `φ`** added to the observation (tracking needs to know where in the swing it is).

## 7. Open questions / risks
- J/M (and CRB) access cost in mujoco_warp — benchmark before putting on the actor path.
- Force-sensor sim-to-real fidelity for impulsive contacts (window-integral impulse is more robust than instantaneous force — the reason for `∫F dt`).
- Whether **pure tracking** suffices (small `w_G,λ`) if Stage-1 is near-optimal, vs needing the task term — the central ablation.
- Reference-trajectory quality (hardcoded vs optimized) gates the whole pipeline.

## 8. Reading list
See `docs/research/impact_tracking_rl_litreview.md` §2 for verified citations. Core: DeepMimic (tracking), HMAMP (hammering imitation+task), Wang/Dehio/Kheddar 2022 (recoil + CRB), Vu 2026 / Ti 2024 (Stage-1 trajectory), Stooke 2020 (constraint).

## 9. Acceptance criteria
Per-term unit tests green; state-injection validation green; a CPU smoke run shows all terms logging and non-degenerate; the tracking-only-vs-+impact ablation is runnable. (Prerequisites §6 must be built first — this spec defines what they must provide.)
