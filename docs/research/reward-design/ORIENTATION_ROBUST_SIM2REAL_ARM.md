# Orientation-robust striking + Vicon state estimation — future "robustness" arm

**Status:** design note / idea capture (2026-06-18). NOT yet implemented. Build this **after** the
fixed-impedance impulse-CaT result (CLAUDE.md sequencing: fixed-impedance + impulse first, then
richer control). This is the sim-to-real-robustness phase.

> Written down so it's not lost in 3 months. The short version: a real board/nail is never exactly
> perpendicular, so train the policy to **align the strike to the actual nail normal** (domain-randomized
> tilt), give it a **6-DoF DiffIK action** so it can orient, and feed it **only Vicon-obtainable state**
> (board-tracked normal + a nail-shaft tracking ball for depth) so it transfers.

---

## 1. Why (the motivation)
- A real nail/board is **never exactly 90°** to the robot's vertical. A policy trained only on a
  perfectly-vertical nail keeps striking straight down → on a real tilted nail that's an **oblique** hit →
  wasted impulse + lateral shear into the nail and the joints (the failure the impulse constraint targets).
- **Domain-randomize the board/nail tilt** in sim → the policy must align the strike to the actual normal →
  robust, sim-to-real-transferable. Standard DR argument.
- Physics: max impulse transfer needs a **collinear (perpendicular) impact** — strike-face normal aligned
  with the nail axis (effective mass `m_eff = (nᵀΛn)⁻¹`, `Λ = J M⁻¹ Jᵀ`, impulse ∝ `m_eff·v_axial` along `n`).

## 2. Decisions
- **Action: 6-DoF DiffIK** (decision 2026-06-18). Use mjlab's built-in `orientation_weight > 0`
  (`hammer_env_cfg.py:121` currently `0.0` = 3-DoF position-only) → action becomes `[Δpos(3), Δori(3)]`.
  - The irrelevant **spin DoF** (rotation about the strike axis; a don't-care for the ~axisymmetric face)
    is parked by the `action_rate` penalty (it stays ≈ still). The one real cost: a full-pose task uses all
    6 arm joints, so we **lose the arm's 1 redundant DoF** (less joint-limit/singularity slack).
  - **Alternative if 6-DoF misbehaves:** a VOT-style **5-DoF** action (3 pos + 2 normal-alignment, spin
    free) — keeps the redundant DoF, matches the JRL paper's Vector Orientation Task. Custom action term
    (align-axis IK: `rot_err = a_world × n_target`). Only worth it if 6-DoF shows exploration/limit trouble.
- **Randomize the board tilt** at reset within a realistic range; the nail is ⊥ the board.

## 3. State estimation — Vicon, NO perception (the load-bearing part)
No vision. Vicon mocap with reflective-marker clusters. **Train on exactly what Vicon can give**, never
sim oracle state. Sources:

| quantity | source |
|---|---|
| robot joints | encoders |
| hammer pose (strike point) | marker cluster on the hammer (or FK — rigid grasp) |
| **nail normal** | **3-marker cluster on the flat BOARD** → board pose → nail normal = board normal (nail ⊥ board) |
| **nail tip position + drive depth** | **a Vicon ball pierced onto the nail SHAFT** (below the head): rides down as the nail is driven → real-time depth; hammer still hits the head, not the ball |

- **Why the board, not the nail, for orientation:** Vicon needs **≥3 non-collinear markers** on a rigid
  body for a full 6-DoF pose. A thin nail can't carry 3 → a single pierced ball gives **position only, not
  orientation**. The board is flat + big → easy 3-marker cluster → robust normal.
- **The pierced-ball idea is kept for depth** (position/penetration), which is exactly our `nail_slide`
  success signal — not for orientation. Mount on the shaft so the strike face hits the head.

## 4. Sim implications (obs/reward must mirror Vicon)
- Randomize **board tilt** at reset.
- Obs = `{joint encoders, hammer pose, board normal, nail-tip position}` — the Vicon set, nothing more.
- Alignment target = **board normal**; strike target = **nail tip**; depth/success = `nail_slide`
  (= the pierced-ball reading on hardware).
- The 6-DoF action lets the policy orient the strike to the *observed* board normal → because it trained on
  the same quantities Vicon provides, it transfers.

## 5. Reference / guidance
- Extend the strike reference + `strike_ref_error` obs to carry a **target orientation** (face-down along
  the board normal), or add a **collinearity reward**, so the orientation DoF are *guided*, not flailed.
- Re-solve the per-episode start pose to be roughly aligned to the sampled board tilt (seed for the policy).

## 6. Interaction / sequencing
- +3 action dims → harder exploration; interacts with the velocity/impulse constraint (more joints moving).
- Shares "action-space budget" with **variable impedance** — don't stack both at once.
- **Order:** vertical-pose fixed-impedance impulse-CaT result FIRST → then this orientation-robust arm →
  variable impedance.

## 7. Reference / prior art
**JRL (CNRS-AIST), Vu, Erens, Stefanelli, Cisneros-Limón, M. Benallegue, A. Benallegue (2026)** —
"QP-based impact momentum maximization for a hammering task by a humanoid robot", hal-05516105.
Repo: github.com/RuudErens/hammering_task_controller (mc_rtc, HRP-5P).
- They **maximize** impact momentum via an Effective-Mass-Maximization QP task (EMMT) + a 5th-order
  jerk-bounded Bézier trajectory + a **Vector Orientation Task (VOT)** that constrains **2 DoF** (roll+pitch
  to align the hammer normal to the nail normal, yaw free) for a collinear impact.
- Their nail is placed at an **arbitrary orientation** (config `nail … rotation: [0,0,1.57]`); the VOT
  aligns to it — the direct analogue of our tilted-board randomization.
- **They explicitly leave "damage minimization on the robot's joints" and post-impact recoil as FUTURE
  WORK** — which is precisely the thesis's **per-joint impulse constraint** contribution. Strong
  citation + gap validation. Their approach is the **QP/model-based** counterpart (hard joint vel/torque
  limits in the QP = the "hardware CBF/QP filter" top rung of our enforcement ladder) to our RL+soft-CaT.
