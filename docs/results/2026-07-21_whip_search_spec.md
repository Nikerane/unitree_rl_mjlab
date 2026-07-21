# Closed-loop whip search — experiment spec (2026-07-21)

**Status:** SPEC / design — nothing has run. Awaiting sign-off on the decision thresholds + Stage-1 go-ahead.
**Formalizes:** open-experiment **#1** of `docs/results/2026-07-20_maximization_ablation_plan.md` ("Closed-loop
whip search (fixed impedance) — B2's specified-but-unrun decisive test"). That plan names the test and sketches a
rough decision rule; this doc is the runnable design.
**Scope:** pure **fixed-impedance control (FIC)**. Nothing here touches `set_gains` / per-joint stiffness — this is
the prerequisite FIC experiment *before* any VIC discussion, per the standing priority.

---

## 1. The one question

Trained RL policies top out at a hammer-head contact speed of **1.81 m/s** (verified max over 1,140 rollouts;
see §3). A kinematic probe (`coord_whip2.py`) suggests **~4.22 m/s** is reachable *at the same fixed PD gains* — but
only by **injecting joint velocities** (bypassing the controller). So:

> **Is ~1.8 m/s a genuine ceiling of the legal fixed-impedance action space, or an RL-exploration artifact —
> a whip channel that exists but RL never found?**

This is the **prerequisite for any "VIC is necessary" claim**. If a legal search can't beat ~1.8, fixed impedance
genuinely caps contact speed → VIC (or a higher-bandwidth action interface) is earned. If it can, VIC is *not* yet
justified on speed grounds — the right FIC next step is to make RL find the channel.

**Read §8 first if you only read one section** — the whip search resolves the *speed*-ceiling question, which is
**distinct from** the *delivered-impulse* question the shipped constraint actually enforces. That distinction is the
load-bearing scoping caveat.

---

## 2. Why this is decidable cheaply — open-loop shooting is a rigorous upper bound

The scene is **deterministic given a fixed reset** (MuJoCo-warp, fixed seed, deterministic contact solver). From a
fixed deterministic initial state, any *closed-loop* policy produces one unique trajectory, whose action sequence is
a specific *open-loop* sequence. Therefore:

```
{ trajectories any closed-loop policy can reach from the fixed reset }  ⊆  { open-loop action sequences }
⇒   max over open-loop sequences  ≥  max over any policy (RL or otherwise)
```

So **maximizing over open-loop action sequences from the nominal reset is an exact upper bound on what any
fixed-impedance policy can do from that state.** A null result (search can't beat 1.8) is thus strong: *nothing*
closed-loop can beat it either. This is why the decisive Stage 1 is cheap and does not require training.

*Honest caveat:* this is the ceiling **from the nominal deterministic reset** — exactly how every ceiling in this
project is quoted (Phase-0, B2), so it is apples-to-apples. Under domain randomization the per-episode achievable
speed varies; the nominal ceiling is an upper bound, not a per-episode guarantee.

---

## 3. The plant under test — code-grounded facts (trust these over memory)

Every value traced to `file:line` by the grounding pass. **Do not modify any of these** during the search — the
whole point is to measure the *shipped* plant.

| quantity | value | source |
|---|---|---|
| action | 3-D task-space **position delta** (orientation_weight=0 → 3-D) | `hammer_env_cfg.py:122`; `differential_ik.py:127-130` |
| clip_actions (→ `[-1,1]` before scale) | **1.0** | `config/z1/rl_cfg.py:57` |
| delta_pos_scale (Z1 hammer override) | **0.15 m** | `z1_constants.py:229` → `config/z1/env_cfgs.py:121` |
| max_dq (per-substep IK joint clamp) | **0.5 rad/step** — *non-binding at δ=0.15* | `differential_ik.py:70`, clamp at `:249` |
| DLS IK damping λ | 0.05 | `differential_ik.py:67` |
| arm PD gains (j1,3,4,5,6) | **kp 1000 / kd 100** | `z1_constants.py:84-85` |
| arm PD gains (j2, shoulder-lift) | **kp 1500 / kd 150** | `z1_constants.py:92-93` |
| kp/kd ratio (uniform) | **10** | derived |
| physics dt | 0.002 s | `hammer_env_cfg.py:301` |
| decimation (substeps / control step) | **10** | `hammer_env_cfg.py:309` |
| control period (the "dt" in the speed relation) | **0.02 s (50 Hz)** | `manager_based_rl_env.py:268` |
| Cartesian velocity limit | **NONE** | `differential_ik.py:25-102` |
| joint-velocity limit (baseline) | **NONE** (only max_dq; ablation-only `Z1_JOINT_VEL_LIMIT=3.1415`) | `env_cfgs.py:122-128`; `velocity_bound.py:41` |
| torque effort_limit (j1,3-6 / j2) | 30 / 60 N·m | `z1_constants.py:86,94` |
| armature (j1,3-6 / j2) | 0.01 / 0.02 kg·m² | `z1_constants.py:87,95` |
| gravity compensation | gravcomp=1.0 (PD fights inertia/contact only) | `z1_constants.py:66-68` |

**Action flow per control step** (`manager_based_rl_env.py:419-423`): policy delta → clip `[-1,1]` → ×0.15 → set as
**absolute Cartesian target = current head-site + delta**, held fixed → 10 physics substeps, each re-solving DLS-IK
toward the held target, clamping `dq≤0.5`, commanding `q_target=q+dq` to the fixed-PD actuators.

**Head-speed relation (verified):** `head_speed ≈ 0.18 · delta_pos_scale / dt_control`, where **dt is the control
period 0.02 s** (not physics_dt). Two code calibration points confirm it: `0.18·0.05/0.02 = 0.45` and
`0.18·0.15/0.02 = 1.35` m/s (`z1_constants.py:226-227`). The 0.18 is an **empirical open-loop efficiency** (finite
PD/IK bandwidth + target re-anchored to the moving head each step), *not* a physical law. The efficiency axis the
search lives on:

| regime | efficiency `v·dt/δ` | head speed @ δ=0.15 |
|---|---|---|
| open-loop scripted descent | 0.18 | 1.35 m/s |
| **trained RL (verified max)** | **0.24** | **1.81 m/s** |
| kinematic box optimum (injected, **illegal**) | 0.56 | 4.22 m/s (hw rail) / 5.44 (default) |

**RL-ceiling provenance:** max v_touch over all 1,140 rollouts = **1.812 m/s** at `lg_maxoff1500_s2` rollout #7
(`evaluation/data/lg_multireset.json`) — note this is the *mislabeled* 8/2 arm; next-fastest `mx_maxofftrk_s1`
1.71, `dc_maxmax_s0` 1.67. Typical ~1.4. v_touch = substep head speed at first face contact.

---

## 4. Why `coord_whip2.py`'s 4.22 m/s is not the answer

`coord_whip2.py` descends **legally** to standoff, then **overwrites joint velocities** via
`robot.write_joint_velocity_to_sim` with a box-optimal `qd[j] = −sign(Jz[j])·vlim` (all 6 joints saturated), and
reports `vhead_pred = Jz·qd` — a **Jacobian kinematic prediction**, not a controller-tracked velocity
(`coord_whip2.py:70,75,80-81`). Default `vlim=4.65` (> the 3.1415 hardware rail) → 5.44 m/s; at the rail → ~4.22.
The `impulse_vacuity` doc calls this configuration *"beyond DiffIK's reachable action space"* (`:97`). B2's
**corrected** verdict: the ~1.4 m/s straight-down strike exploits only ~45% of the box optimum, so it's a
*trajectory/controller-shaping* limit, not a hardware wall — and it **explicitly flags** the ~4.2 may "collapse
back toward ~1.6-1.8 m/s" through the real controller (`B2_effort_ceiling.md:78-85,104-110`). **That collapse-or-not
is exactly what this experiment settles.** The velocity injection (`write_joint_velocity_to_sim`) is the one thing
this spec forbids everywhere.

---

## 5. The measurement instrument (frozen yardstick)

Build this **first**; it is the yardstick all stages share.

- **v_ante** = downward (nail-axis, `(0,0,−1)`) head-site speed at substep **k\*−1**, where k\* is the first
  substep the **face** contact sensor flips (`hammer_nail_contact`, primary = `hammer_head_0` → **face only**,
  excludes the neck-drive exploit). k\*−1 = the last *ante-impact* substep (honest carried momentum, before the
  collision's own deceleration). Substep rate (500 Hz), read from `site_lin_vel_w` **and** cross-checked against a
  finite-difference of head_z (agree within a few %, else reject).
- **Parity re-baseline (mandatory):** re-measure the current best trained checkpoint with **this exact estimator**
  before any comparison, so v* and the ~1.81 anchor are apples-to-apples (not two velocity definitions).
- **Authenticity gates** (carry the hard-won reward-hack lessons — neck-drive, claw-drive, miss-past):
  1. **face-only** contact (not `hammer_head_1` neck),
  2. **axial-dominant** (`|v_axial| > |v_lateral|` — reject grazes/tangential),
  3. **valid strike** (contact actually occurs + nail depth increases — reject whip-past-the-nail).
  A candidate failing any gate scores as a miss (penalty ∝ closest head-to-nail approach), so the optimizer is
  pulled *toward* a real strike, never rewarded for a fast miss.
- **Read-only:** the instrument patches `metrics_manager.compute_substep` to *record* only (as `coord_whip2.py:48-56`
  does) — it never writes state.

---

## 6. The staged experiment (lazy-in-the-right-order)

The three backbones the design pass produced are **not competitors** — they answer different questions and nest.
Run the cheap dispositive one first; only pay for training if it says there's something to find.

### Stage 1 — Trajectory optimization (CMA-ES open-loop shooting) · **DECISIVE · ~1–4 h CPU**
The true action-space ceiling, RL-independent (§2).
- **Variables:** `a₁…a_T ∈ ℝ^{3T}`, T≈30 control steps (≈0.6 s; reference contacts ~step 11, so generous whip
  headroom), 90 dims. Each `aₜ` fed through `env.step(clip(aₜ,−1,1))` — **byte-identical to a policy's output**.
- **Objective:** maximize v_ante (§5). Miss → penalty gate.
- **RL-seeded:** CMA-ES mean μ₀ = the best checkpoint's own realized action rollout from the fixed reset ⇒ **v\* ≥ 1.81
  by construction**; any gain is real headroom over RL. Also seed generation 0 with the scripted reference sequence,
  a constant `[0,0,−1]` full-down drive, and μ₀±perturbations for basin diversity.
- **Batched:** one CMA sample per env, one batched T-step rollout per generation (num_envs = population 128–256).
  `env.reset()` between generations. 3–5 restarts; keep the global best. **5-generation smoke test first** to measure
  warp-CPU wall-clock (the one real unknown).
- **δ-sweep add-on (cheap, run in the same harness):** repeat at `delta_pos_scale ∈ {0.15, 0.30, 0.45}` (PD
  **unchanged** — still fixed impedance, just a different action-space parameterization). This **localizes** a ceiling:
  if raising δ breaks 1.8 → the wall is the **δ-rail** (fixable within FIC by raising δ); if even δ=0.45 caps near
  1.8 → the wall is the **PD-tracking bandwidth** (only VIC / higher-bandwidth helps). Primary answer stays δ=0.15.
- **Hardware-legal pass:** report **v\*** (action-space legal, no joint-vel cap) *and* **v\*_hw** (arm joints
  ≤3.1415 rad/s). v\*_hw is the number that bears on the real robot.

### Stage 2 — Parametrized whip (windup→dwell→strike waypoints) · **corroboration + mechanism · ~1–4 h CPU**
Closed-loop, interpretable, cheap. Tells you **why** (which joints unfurl; is there a windup transient).
- 7 knobs: windup dist / elevation / azimuth, dwell, **strike-start standoff** (the timing knob), follow-through,
  strike-axis tilt. Each step `a = clamp((waypoint − head_now)/0.15, −1, 1)` (closed-loop on live head).
- Coarse random/grid (map basins) → CMA-ES refine (3–5 restarts).
- **Role:** (a) a hand-designed lower bound Stage 1 should beat (cross-check); (b) its winning knobs + qvel
  decomposition explain the channel; (c) the winning trajectory **seeds Stage 3** (imitation prior / warm start).

### Stage 3 — Targeted RL-shaping (only if Stage 1 shows headroom) · **~1.5–2.5 days CPU / faster on GPU**
Answers the *actually useful* follow-up: **can a trainable policy realize the headroom Stage 1 proved exists?** This
is the maximization plan's proposed "retrain `imponly` with a wind-up curriculum." Do **not** run it unless Stage 1
lands "artifact" — a null here is confounded by "did RL just underconverge?"
- New task variant `Unitree-Z1-Hammer-Whip`: `impact_progress` raised to terminal weight, **terminate-on-first-face-
  contact** (single strike → return monotone in one v_touch, nothing to farm), `action_rate→0` (the whip needs the
  wind-up/snap the penalty suppresses), `approach` annealed 0.1→0 (the shipped approach term biases to the greedy
  straight descent), entropy 0.02→0.05, init_std 1.5, reset-standoff runway curriculum, 4–6 seeds.
- **Positive control** = the δ-sweep as a training arm: proves the machinery *finds* speed when it's there, so a flat
  v\*≈1.8 is a real ceiling and not a dead search. A CEILING verdict here requires (i) a ≥1000-iter flat learning
  curve across all seeds **and** (ii) the δ-sweep positive control.

---

## 7. Decision rule

Primary number: **v\*** from Stage 1 at the shipped δ=0.15 (Stage 2 corroborates; Stage 3 gates on this).

| v\* (δ=0.15, thorough search) | efficiency | verdict | consequence |
|---|---|---|---|
| **≥ 2.5–3.0 m/s** | ≥0.33 | box optimum substantially reachable | **VIC is a refinement, not a necessity** (matches plan) |
| **≥ 2.1 m/s** (RL×1.15, clears noise) | ≥0.28 | **RL-exploration ARTIFACT** | 1.8 is not the ceiling; VIC not justified on speed. → **Stage 3**: make RL find it (curriculum / warm-start from the Stage-1/2 whip) |
| **≈ 1.8–1.9 m/s** (multi-restart converged) | ~0.24 | **ceiling is REAL** | open-loop exhausted the trajectory space → fixed impedance caps contact speed. **VIC / higher-bandwidth interface / raising δ is earned.** δ-sweep says which: δ-rail (raise δ, still FIC) vs PD-bandwidth (VIC). |

Asymmetry (state it in the writeup): v\* is a **lower bound** on the true ceiling (CMA-ES may miss a better
sequence), so a *positive* result (≥2.1) is dispositive, while a *null* is "strong-but-not-formal" and must be
hardened by multi-restart + diverse seeds + (if it stalls) a smoother spline reparameterization of the delta path —
a stalled optimizer must never be misread as a physical wall.

---

## 8. What this resolves — and what it does NOT (the load-bearing caveat)

The whip search answers the **contact-speed** question. It does **not**, by itself, settle the VIC-necessity
argument, because of the project's essential fact:

> Under fixed impedance, delivered impulse is `∫F·dt` against a **still-present** target, so the reward selects
> **contact duration**, not speed. Over 1,140 rollouts, `delivered ≈ 0.027·dwell + 0.014·peak_force − 0.28`
> (R²=0.842); adding v_touch → **+0.0002**; `corr(v_touch, delivered) = −0.26` (negative).
> (`maximization_ablation_plan.md:283-290`)

So a faster whip **need not raise delivered impulse** against the current still target. The whip search tells you
about **ballistic momentum `m_eff·v`** — which matters *iff* the thesis moves to a genuinely **impulsive** strike
(fast contact against a target that can separate), i.e. the ballistic framing. It does **not** tell you about the
**press-integral Λ** the shipped accumulator currently enforces (that's a dwell/force integral where speed is
irrelevant). The VIC-necessity case therefore has **two independent halves**:

1. **(this experiment)** Is fixed impedance velocity-limited? → whip search.
2. **(a separate decision, Khadiv (e))** *Should* the constraint be on an impulsive momentum transfer (where speed is
   the lever) or a sustained press integral (where it isn't)? → the impulsive-vs-press question in the impulse-CaT
   plan.

Both must land on the "speed matters" side for "VIC is necessary for safe *impact*" to hold. The whip search closes
half of that; the spec flags the other half so we don't over-claim from a speed result alone.

---

## 9. Guardrails, legality, deliverables

**Legality assertions (fail the run if any trips):** no `write_joint_velocity_to_sim` / `write_joint_position_to_sim`
/ qvel/qpos write anywhere in the driver (grep + runtime sentinel that raises); every action ∈ `[-1,1]³`; PD gains,
kp/kd ratio, max_dq, DLS λ, armature, effort_limits, and (primary-arm) delta_pos_scale unchanged from §3 (re-assert
at build). The δ-sweep arm changes **only** delta_pos_scale and is fenced as an explicitly labeled localization
diagnostic — never the shipped-action-space answer.

**Standing guardrails honored:** `imp_max_p=0` (log-only), `IMP_J_LIMIT` unchanged, caps untouched, **no VIC /
`set_gains`**. This is entirely FIC.

**Compute:** Stage 1 & 2 — single 8–16-core box, ~1–4 h each, only new dep is `pycma` (or a ~40-line NumPy CEM).
Stage 3 (conditional) — ~1.5–2.5 days CPU or a single GPU; gated on a Stage-1 "artifact" verdict.

**Deliverables:** (1) the frozen-yardstick instrument + parity-rebaselined v_RL; (2) v\*, v\*_hw, δ-sweep curve,
per-joint qvel-at-contact decomposition, efficiency-axis placement; (3) the §7 verdict; (4) if artifact → a Stage-3
plan seeded by the winning whip; (5) update `maximization_ablation_plan.md` open-experiment #1 with the result.

**Pre-run gate (per CLAUDE.md):** full `pytest` + `validate_rewards.py` (A–M) + `verify_contact_sensor.py` before
any GPU stage.
