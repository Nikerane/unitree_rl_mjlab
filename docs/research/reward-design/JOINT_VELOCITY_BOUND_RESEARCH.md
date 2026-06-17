# Bounding joint velocity for the Z1 strike policy — research synthesis (2026-06-17)

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
| B | **Jacobian action-projection** (scale the Cartesian delta so induced joint vel ≤ π) | hard *on command* | med | strong | Velocity-CBF `h=π−max|q̇|`. Bounds the *commanded* velocity, not the realized PD transient — shares A's blind spot for stiff-PD overshoot. |
| C | **Substep velocity-excess penalty → CaT/PID-Lagrangian** | soft / in-expectation | med | highest (reward route) | `−w·Σ max(0,peak|q̇_j|−π·β)²`, substep-peak, annealed in after the strike. CaT is the preferred low-infra delivery. Hackable as a fixed penalty; CMDP auto-tunes λ. |
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
- Internal: `tracking_impact_impulse_design_research.md` (D1–D6, SQ3/SQ4), `TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` (T0–T5), `hammering_reward_design_deep_dive_v2.md`; memories `[[diffik-maxdq-velocity]]`, `[[z1-hardware-limits]]`, `[[cat-constraints-as-terminations]]`.
