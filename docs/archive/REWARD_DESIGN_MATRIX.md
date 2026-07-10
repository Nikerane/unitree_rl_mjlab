> ⚠️ **ARCHIVED 2026-07-05** — superseded by `src/tasks/hammer/hammer_env_cfg.py` (live reward terms + weights).
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Reward Design Matrix — Z1 Hammer Task

Columns: **Term** | **Formula (batch tensor, shape [B])** | **When active** | **Typical weight range** | **Source** | **mjlab feasibility**

Status tags: ✅ in current mjlab config | 🔄 replace/modify | ➕ add new | ❌ drop

---

## Phase 1 — Approach

| Term | Formula | When active | Weight range | Source | mjlab feasibility |
|---|---|---|---|---|---|
| ➕ **approach_contact_gated** | `exp(−dist²/std²) × (1 − in_contact)` where `in_contact = (sensor.data.found > 0).any(dim=-1)` | Every step; zeroed when hammer touches nail | +0.3 – +1.0 | Ng et al. 1999 (gate prevents hovering); D'Ambrosio 2023 (event-gated pattern) | **Stateless**. Requires `ContactSensor` registered for hammer-head geom vs. nail geom. `sensor.data.found.any()` gives binary contact flag per env. |
| 🔄 **approach_depth_fade** *(alternative to above)* | `exp(−dist²/std²) × exp(−k × nail_depth)` where k=200 | Every step; exponentially fades as nail is driven | +0.3 – +1.0 | Legacy Gym design; Ng et al. 1999 (PBRS approximation) | **Stateless**. Reads `nail_entity.data.joint_pos`. No extra sensor needed. Simpler than ContactSensor gate; less physically precise. |
| ✅ **hammer_approach_reward** *(current — replace)* | `exp(−dist²/std²)`, std=0.15, weight=+0.5 | Always on, no phase awareness | +0.5 | Current mjlab `rewards.py` | **Stateless. Has hovering problem — replace with gated version above.** |

---

## Phase 2 — Impact

| Term | Formula | When active | Weight range | Source | mjlab feasibility |
|---|---|---|---|---|---|
| ➕ **impact_velocity_bonus** | `‖head_vel‖ × first_contact` where `first_contact = sensor.compute_first_contact(dt)` | Fires once per strike at air→contact transition | +5.0 – +20.0 (absolute, not per-step) | ARMADA (Kim et al. 2025); D'Ambrosio 2023 (event-gated) | **Stateless**. `ContactSensor.compute_first_contact()` returns [B, P] bool; `.any(dim=-1)` gives [B]. Head velocity from `robot.data.site_vel_w[:, site_ids]`. Fires on every re-strike — correct for repeated impacts. |
| ➕ **air_time_bonus** | `(last_air_time − t_min).clamp_min(0) × first_contact` | Fires at each contact transition | +2.0 – +10.0 | RSL-RL `feet_air_time` (Rudin et al. 2022) | **Stateless**. Requires `ContactSensor` with `track_air_time=True`. `sensor.data.last_air_time` gives [B, P]; squeeze to [B]. Rewards pulling higher before each strike. |
| ➕ **contact_force_magnitude** | `‖sensor.data.force‖.squeeze() × first_contact` | Fires at contact transition | +0.5 – +2.0 | ARMADA; D'Ambrosio 2023 | **Stateless**. Use with caution — contact force magnitude is least faithful quantity for sim-to-real (van Steen et al. 2024). Prefer velocity-at-contact instead. Mark as optional/ablation. |

---

## Phase 3 — Nail Depth Progress

| Term | Formula | When active | Weight range | Source | mjlab feasibility |
|---|---|---|---|---|---|
| ➕ **nail_depth_delta** | `(current_depth − max_depth_so_far).clamp_min(0)` | Every step where nail advances | +200 – +1000 | Wu et al. 2021 (DREM progress reward); Meta-World hammer | **Stateful**. Requires `ManagerTermBase` subclass with `_max_depth` tensor and `reset(env_ids)` hook. See pseudocode in RECOMMENDED_REWARD_SPEC.md. |
| ✅ **nail_driven_reward** *(current — replace)* | `exp(−(goal_depth − current_depth)² / std²)`, std=0.03, weight=+2.0 | Always on | +2.0 | Current mjlab `rewards.py` | **Stateless. Problem: near-zero gradient when nail is at 0mm (exp(−6.25) ≈ 0.002). Replace with delta-based term; optionally keep at low weight for near-goal fine gradient.** |
| ➕ **nail_driven_gaussian_neargoal** *(optional keep)* | Same as `nail_driven_reward` above | Always on | +0.5 – +1.0 | Current mjlab; Eureka pattern (Ma et al. 2023) | Keep at reduced weight alongside `nail_depth_delta`. Provides dense near-goal gradient that delta reward loses when nail stops advancing. |

---

## Phase 4 — Completion

| Term | Formula | When active | Weight range | Source | mjlab feasibility |
|---|---|---|---|---|---|
| ➕ **completion_bonus** | `(nail_depth >= success_threshold).float() × bonus` | Once when threshold crossed | +50.0 – +200.0 | Meta-World v3 hammer (+10 flat bonus); D'Ambrosio 2023 (event bonus) | **Stateless**. Read `nail_entity.data.joint_pos`. <!-- opus-audit H2: earlier text claimed this could fire repeatedly without a stateful guard — that was incorrect. mjlab's `nail_fully_driven` termination fires on the same step, ending the episode before another reward step. No guard needed. Validated by Phase H of `validate_rewards.py`. --> Fires exactly once per episode because the `nail_driven` termination triggers on the same step. |

---

## Smoothness / Sim-to-Real

| Term | Formula | When active | Weight range | Source | mjlab feasibility |
|---|---|---|---|---|---|
| ✅ **action_rate** | `Σ(a_t − a_{t−1})²` | Every step | −0.005 – −0.05 | RSL-RL (Rudin et al. 2022); current mjlab | **Stateless**. `env.action_manager.action − env.action_manager.prev_action`. Already in config at −0.01. |
| ➕ **joint_vel_penalty** | `Σ|q̇|` or `Σq̇²` on arm joints | Every step | −0.001 – −0.01 | RSL-RL; D'Ambrosio 2023 (velocity penalty) | **Stateless**. `robot.data.joint_vel[:, joint_ids]`. L1 norm (`abs().sum()`) or L2 (`square().sum()`). Apply only to arm joints, not gripper. |
| ✅ **joint_pos_limits** | Soft penalty for joint limit violation | Every step | −0.5 – −2.0 | HPRS hierarchy (Berducci et al. 2024); current mjlab | **Stateless**. Already in config at −1.0. Consider increasing during strike phase. |
| ➕ **torque_peak_penalty** | `(‖τ‖ − τ_max).clamp_min(0)²` on critical joints | Every step | −0.01 – −0.1 | DM Soccer (impact joint protection); Kim et al. 2023 (constraint RL) | **Stateless**. `robot.data.applied_torque[:, joint_ids]`. Optional: apply only to elbow/shoulder joints most stressed during hammer blows. |

---

## Conflict Register

| Conflict | Description | Resolution |
|---|---|---|
| **Approach always-on vs. contact-gated** | `hammer_approach_reward` (current, always-on) incentivises hovering near nail_top during and after contact. | Replace with contact-gated or depth-fade version. Do not run both simultaneously. |
| **nail_driven_reward Gaussian vs. nail_depth_delta** | Gaussian gives good near-goal gradient but near-zero far-from-goal signal. Delta gives uniform gradient but goes to zero once nail stops advancing. | Run both: `nail_depth_delta` at high weight (primary signal), `nail_driven_gaussian_neargoal` at low weight (fine-tuning). |
| **impact_velocity_bonus vs. smoothness penalties** | Velocity bonus encourages fast swing; joint_vel_penalty discourages fast motion. | Apply joint_vel_penalty to all steps; impact_velocity_bonus fires only at contact transition. The policies are temporally separated: policy learns to be smooth except during the strike phase. Weight `impact_velocity_bonus` at ~10× the max per-step joint_vel penalty so the bonus dominates at impact. |
| **air_time_bonus vs. efficiency** | Long retraction is rewarded, but excessive height wastes time and energy. | Clip retraction height or use `(air_time − t_min).clamp(0, t_max)` to cap the bonus. |
| **completion_bonus repeated firing** | <!-- opus-audit H2: this row's "conflict" was based on the earlier incorrect claim. There is no conflict — termination handles it. Kept for traceability. --> If the policy holds the nail past threshold, would `completion_bonus` keep firing? | No: mjlab's `nail_fully_driven` TerminationTermCfg ends the episode on the same step. Verified by `validate_rewards.py` Phase H. |

---

## lift-cube Comparison

mjlab's `staged_position_reward` (lift-cube):
```
reaching = exp(−‖ee − obj‖² / r_std²)
bringing = exp(−‖obj − target‖² / b_std²)
reward = reaching × (1 + bringing)
```

**Key difference from hammer:** lift-cube uses *multiplicative* gating — approach (reaching) gates task progress (bringing). Hammer uses *additive* terms with *event-based* gating (contact sensor). Lift-cube reward is continuous and always positive; hammer reward has discrete impact events.

| Dimension | lift-cube | hammer (recommended) |
|---|---|---|
| Approach signal | Gaussian, always-on | Gaussian, contact-gated |
| Task progress signal | Gaussian on absolute position | Delta on nail depth (progress) |
| Event bonus | None | Impact velocity at contact transition |
| Rhythmic cycle reward | N/A | Air-time bonus (retraction encouragement) |
| Termination | Time-out or success | Success (nail_fully_driven) or time-out |
| Stateful terms | None | nail_depth_delta, air_time (ContactSensor) |
