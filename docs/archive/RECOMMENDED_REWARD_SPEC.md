> ⚠️ **ARCHIVED 2026-07-05** — superseded by `src/tasks/hammer/hammer_env_cfg.py` — the implemented 7-term reward is the source of truth (this 9-term spec is aspirational).
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Recommended Reward Specification — Z1 Hammer Task
## Engineering spec for mjlab ManagerBasedRlEnv

**Goal:** Train a Z1 arm to repeatedly swing and strike a nail, driving it ~7.5 cm without learning slow-press hacks.

---

<!-- opus-audit: H1 — original "Replace" recommendations conflicted with the
     minimum-viable-change strategy actually executed (see feedback_minimum_viable_reward.md).
     Rewritten to reflect the augment-not-replace baseline. The original recommendations
     are preserved at the bottom as "Potential future replacements". -->

## 1. Current mjlab Config — Baseline (implemented) vs. Future Changes

**Implemented baseline (6 terms, all in `hammer_env_cfg.py`, validated by `validate_rewards.py`):**

| Term | Weight | Status | Rationale |
|---|---|---|---|
| `approach` (Gaussian on dist) | +0.5 | Kept | Provides early-stage signal toward the nail. Hovering risk acknowledged but deferred until observed. |
| `nail_driven` (Gaussian on absolute depth) | +2.0 | Kept | Near-zero at 0mm but provides smooth gradient near goal. Kept alongside delta. |
| `nail_depth_delta` (progress, stateful) | +500 | **Added** | Resolves the zero-gradient-at-0mm problem the Gaussian has. Stateful via `ManagerTermBase`. |
| `completion` (sparse +1 at threshold) | +100 | **Added** | Actual task reward. Sparse signal complementing dense shaping. |
| `action_rate` (L2 action delta) | −0.01 | Kept | Sim-to-real smoothness. Weight unchanged from original. |
| `joint_pos_limits` | −1.0 | Kept | Hard safety boundary. |

**Potential future replacements** (deferred until a specific failure mode appears during training):

| Original recommendation | Trigger to revisit |
|---|---|
| Replace `nail_driven` (Gaussian) → use only `nail_depth_delta` | If Gaussian on absolute depth is observed to encourage "hold-at-goal" hovering |
| Replace `approach` (always-on Gaussian) → `approach_contact_gated` (§3.2) | If the policy is observed to hover near nail_top instead of striking |
| Increase `action_rate` weight to −0.02 | If trained policy shows visible jerk in rollouts |

### Considered but deferred terms (M2)

These are standard in impact-RL baselines but not yet in the implementation. Trigger conditions for adding them:

| Term | Trigger | Source |
|---|---|---|
| `time_penalty` (−0.01/step) | If policy reaches goal but takes most of the episode | Standard RL pattern |
| `torque_sum_sq_penalty` (`sum(τ²)`) | If sim-to-real transfer fails due to actuator saturation | RSL-RL legged locomotion (Rudin 2022) |
| `orientation_alignment` (quat gate) | If policy approaches nail from wrong angle and strikes fail | Meta-World hammer-v3 (Yu et al. 2020) |

---

## 2. Terms to Port from Legacy Gym Design

| Gym term | Port as | Changes |
|---|---|---|
| `approach_delta × 50 × exp(−200·depth)` | `approach_contact_gated` (see §3) | Replace depth-based fade with ContactSensor gate — more physically correct |
| `depth_delta × 500` | `nail_depth_delta` (see §3) | Port logic, rewrite as `ManagerTermBase` for mjlab vectorised env |
| `impact_bonus × 10` (fires once) | `impact_velocity_bonus` (see §3) | Fires on EVERY strike (not just first) using `ContactSensor.compute_first_contact()` |
| `completion_bonus +100` | `completion_bonus` | Keep concept, tune weight |
| `joint_vel_penalty −0.005` | `joint_vel_penalty` | Add as new `RewardTermCfg` |
| `action_rate_penalty −0.5` | Already in mjlab — increase weight | Merge with existing `action_rate` term |

---

## 3. Full Term Pseudocode

### 3.1 `nail_depth_delta` — Stateful, ManagerTermBase

**Purpose:** Reward every improvement in nail depth. Fires on each strike that advances the nail. Equal gradient for every mm of nail travel (0–75mm).

```python
# rewards.py
from mjlab.managers.manager_term_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
import torch

_DEFAULT_NAIL_CFG = SceneEntityCfg("nail_block", joint_names=("nail_slide",))

class NailDepthDeltaTerm(ManagerTermBase):
    """Progress reward: reward improvements in nail depth each step.

    Tracks max_depth_so_far per environment across the episode.
    Only positive deltas are rewarded — nail bounce does not penalise.
    """

    def __init__(self, cfg, env):
        # opus-audit C1: ManagerTermBase.__init__ takes only `env`, not `(cfg, env)`.
        # See mjlab/managers/manager_base.py:65. Auto-instantiation passes (cfg, env)
        # to the SUBCLASS but the super() call drops cfg.
        super().__init__(env)
        self._max_depth: torch.Tensor = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._max_depth[env_ids] = 0.0

    def __call__(
        self,
        env,
        nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
    ) -> torch.Tensor:
        """Returns shape (B,)."""
        nail = env.scene[nail_cfg.name]
        depth = nail.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)  # (B,)
        delta = (depth - self._max_depth).clamp_min(0.0)
        self._max_depth = torch.maximum(self._max_depth, depth)
        return delta
```

**RewardTermCfg:**
```python
"nail_depth_delta": RewardTermCfg(
    func=hammer_mdp.NailDepthDeltaTerm,
    weight=500.0,
    params={
        "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
    },
),
```

**Weight rationale:** 500 × 0.075m (full nail travel) = 37.5 max cumulative depth reward per episode. Completion bonus (§3.5) at +100 should be ~2.5× this maximum.

---

### 3.2 `approach_contact_gated` — Stateless

**Purpose:** Guide hammer head toward nail during the swing phase. Zero during contact (prevents hovering incentive).

**Requires:** `ContactSensorCfg` added to scene for hammer head vs. nail body.

```python
# rewards.py
from mjlab.sensor import ContactSensor
from mjlab.entity import Entity

def approach_contact_gated(
    env,
    std: float,
    sensor_name: str,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
) -> torch.Tensor:
    """Gaussian approach reward, zeroed while hammer is in contact with nail.

    Shape: (B,).
    """
    robot: Entity = env.scene[robot_cfg.name]
    nail: Entity = env.scene[nail_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    head_pos = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)   # (B, 3)
    nail_pos = nail.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)     # (B, 3)

    dist_sq = torch.sum((head_pos - nail_pos) ** 2, dim=-1)              # (B,)
    approach = torch.exp(-dist_sq / std**2)                              # (B,)

    in_contact = (contact_sensor.data.found > 0).any(dim=-1).float()     # (B,)
    return approach * (1.0 - in_contact)
```

**ContactSensorCfg to add to scene:**
```python
# hammer_env_cfg.py — inside make_hammer_env_cfg(), scene section
from mjlab.sensor import ContactMatch, ContactSensorCfg

scene_sensors = {
    "hammer_nail_contact": ContactSensorCfg(
        primary=ContactMatch(
            mode="geom",
            pattern="hammer_head",     # EXACT name — single geom (verified in z1_mocap_hammer.xml)
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="nail",
            entity="nail_block",
        ),
        fields=("found", "force"),
        reduce="maxforce",
        track_air_time=True,           # needed for air_time_bonus (§3.4)
        history_length=0,
    ),
}
```

**Geom name reference** (from `hammer_z1_env/assets/z1_mocap_hammer.xml`):
- `hammer_head` — the geom that strikes the nail (use as ContactSensor primary)
- `hammer_handle` — the handle (do NOT include — contact with handle is not a strike)
- `hammer_head_site` — site for position queries (already used by `head_pos_w` in observations)

**RewardTermCfg:**
```python
"approach": RewardTermCfg(
    func=hammer_mdp.approach_contact_gated,
    weight=0.5,
    params={
        "std": 0.10,                # tighter than current 0.15 — more precise guidance
        "sensor_name": "hammer_nail_contact",
        "robot_cfg": SceneEntityCfg("robot", site_names=("hammer_head_site",)),
        "nail_cfg": SceneEntityCfg("nail_block", site_names=("nail_top",)),
    },
),
```

---

### 3.3 `impact_velocity_bonus` — Stateful (finite-differenced velocity)

**Purpose:** Fire a bonus at every strike proportional to impact speed. Teaches the policy to swing, not press. Fires on EVERY re-strike (not just first), enabling the repeated-impact training objective.

**Why stateful, not stateless:** A naive implementation would read `robot.data.site_vel_w` directly. This works if mjlab populates site velocities by default — but if `site_vel_w` is lazy-evaluated or disabled in the entity `DataCfg`, the term silently returns zero and the policy never learns to swing (Q10 in OPEN_QUESTIONS.md). The finite-difference version is robust to this because `site_pos_w` is fundamental and always populated (the approach reward depends on it).

```python
class ImpactVelocityBonusTerm(ManagerTermBase):
    """Bonus at first-contact, using finite-differenced head velocity.

    Velocity = (head_pos_t - head_pos_{t-1}) / dt. Independent of any
    lazy-evaluated velocity fields in robot.data.
    """

    def __init__(self, cfg, env):
        # opus-audit C1: same fix as NailDepthDeltaTerm — super takes only env.
        super().__init__(env)
        self._prev_head_pos: torch.Tensor = torch.zeros(
            env.num_envs, 3, dtype=torch.float32, device=env.device
        )
        self._initialized: torch.Tensor = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._initialized[env_ids] = False

    def __call__(
        self,
        env,
        sensor_name: str,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ) -> torch.Tensor:
        """Returns shape (B,)."""
        robot: Entity = env.scene[robot_cfg.name]
        sensor: ContactSensor = env.scene[sensor_name]
        dt = env.step_dt

        head_pos = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)  # (B, 3)

        # First step after reset: velocity is undefined; emit 0 to avoid garbage delta.
        head_vel = torch.where(
            self._initialized.unsqueeze(-1),
            (head_pos - self._prev_head_pos) / dt,
            torch.zeros_like(head_pos),
        )
        self._prev_head_pos = head_pos.clone()
        self._initialized.fill_(True)

        first_contact = sensor.compute_first_contact(dt=dt).any(dim=-1).float()
        impact_speed = torch.norm(head_vel, dim=-1)
        return impact_speed * first_contact
```

**RewardTermCfg:**
```python
"impact_velocity": RewardTermCfg(
    func=hammer_mdp.ImpactVelocityBonusTerm,
    weight=10.0,
    params={
        "sensor_name": "hammer_nail_contact",
        "robot_cfg": SceneEntityCfg("robot", site_names=("hammer_head_site",)),
    },
),
```

**Weight rationale:** Head speed at impact ≈ 0.5–1.0 m/s. Bonus ≈ 5–10 per strike. Per-step depth reward ≈ 0.05 per step. Impact bonus should be ~2–5× the depth reward per step to prioritise striking over pressing. Tune here.

**Alternative (if you trust lazy eval):** A simpler stateless version reads `robot.data.site_vel_w[:, robot_cfg.site_ids, :3]` directly. Use only after verifying with `verify_reward_setup.py` that `site_vel_w` is populated at reward-compute time.

---

### 3.4 `air_time_bonus` — Stateless (uses ContactSensor state)

**Purpose:** Reward pulling the hammer higher before each strike. Longer retraction → more potential energy → deeper nail drive. Direct analog of RSL-RL `feet_air_time` reward for locomotion gait.

```python
def air_time_bonus(
    env,
    sensor_name: str,
    min_air_time: float = 0.05,     # seconds; below this = no bonus
    max_air_time: float = 0.5,      # seconds; cap to prevent slow hover
) -> torch.Tensor:
    """Reward duration of retraction phase, measured at each contact transition.

    Fires only at the moment of impact (air→contact). Zero otherwise.
    Shape: (B,).
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]

    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
    any_first_contact = first_contact.any(dim=-1).float()               # (B,)

    # last_air_time: duration of the air phase that just ended, shape (B, P)
    last_air = contact_sensor.data.last_air_time                        # (B, P)
    max_air = last_air.max(dim=-1).values                               # (B,)

    bonus = (max_air - min_air_time).clamp(0.0, max_air_time - min_air_time)
    return bonus * any_first_contact
```

**RewardTermCfg:**
```python
"air_time": RewardTermCfg(
    func=hammer_mdp.air_time_bonus,
    weight=3.0,
    params={
        "sensor_name": "hammer_nail_contact",
        "min_air_time": 0.05,
        "max_air_time": 0.40,
    },
),
```

**Note:** `track_air_time=True` must be set in `ContactSensorCfg` (already included in §3.2). Start with low weight (1.0–3.0); increase if policy learns fast shallow tapping instead of retract-and-swing.

---

### 3.5 `completion_bonus` — Stateless

```python
def completion_bonus(
    env,
    success_depth: float,
    nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
) -> torch.Tensor:
    """One-time bonus when nail reaches success depth.

    mjlab terminates the episode on success, so this fires at most once.
    Shape: (B,).
    """
    nail: Entity = env.scene[nail_cfg.name]
    depth = nail.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
    return (depth >= success_depth).float()
```

**RewardTermCfg:**
```python
"completion": RewardTermCfg(
    func=hammer_mdp.completion_bonus,
    weight=100.0,
    params={
        "success_depth": NAIL_SUCCESS_THRESHOLD,
        "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
    },
),
```

---

### 3.6 `joint_vel_penalty` — Stateless (new)

```python
def joint_vel_penalty(
    env,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """L1 penalty on arm joint velocities. Shape: (B,)."""
    robot: Entity = env.scene[robot_cfg.name]
    return robot.data.joint_vel[:, robot_cfg.joint_ids].abs().sum(dim=-1)
```

**RewardTermCfg:**
```python
"joint_vel": RewardTermCfg(
    func=hammer_mdp.joint_vel_penalty,
    weight=-0.005,
    params={
        "robot_cfg": SceneEntityCfg("robot", joint_names=("joint[1-6]",)),
    },
),
```

---

## 4. Complete Recommended Config

```python
rewards = {
    # --- Phase 1: approach (gated off during contact) ---
    "approach": RewardTermCfg(
        func=hammer_mdp.approach_contact_gated,
        weight=0.5,
        params={
            "std": 0.10,
            "sensor_name": "hammer_nail_contact",
            "robot_cfg": SceneEntityCfg("robot", site_names=("hammer_head_site",)),
            "nail_cfg": SceneEntityCfg("nail_block", site_names=("nail_top",)),
        },
    ),

    # --- Phase 2: impact events ---
    "impact_velocity": RewardTermCfg(
        func=hammer_mdp.ImpactVelocityBonusTerm,    # stateful: finite-diff velocity
        weight=10.0,
        params={
            "sensor_name": "hammer_nail_contact",
            "robot_cfg": SceneEntityCfg("robot", site_names=("hammer_head_site",)),
        },
    ),
    "air_time": RewardTermCfg(
        func=hammer_mdp.air_time_bonus,
        weight=3.0,
        params={
            "sensor_name": "hammer_nail_contact",
            "min_air_time": 0.05,
            "max_air_time": 0.40,
        },
    ),

    # --- Phase 3: nail depth progress ---
    "nail_depth_delta": RewardTermCfg(
        func=hammer_mdp.NailDepthDeltaTerm,
        weight=500.0,
        params={
            "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
        },
    ),
    # Optional: keep Gaussian at low weight for near-goal fine gradient
    "nail_driven_neargoal": RewardTermCfg(
        func=hammer_mdp.nail_driven_reward,
        weight=0.5,
        params={
            "goal_depth": NAIL_GOAL_DEPTH,
            "std": 0.03,
            "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
        },
    ),

    # --- Phase 4: completion ---
    "completion": RewardTermCfg(
        func=hammer_mdp.completion_bonus,
        weight=100.0,
        params={
            "success_depth": NAIL_SUCCESS_THRESHOLD,
            "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
        },
    ),

    # --- Smoothness / sim-to-real ---
    "action_rate": RewardTermCfg(
        func=hammer_mdp.action_rate_penalty,
        weight=-0.02,               # increased from -0.01
    ),
    "joint_vel": RewardTermCfg(
        func=hammer_mdp.joint_vel_penalty,
        weight=-0.005,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=("joint[1-6]",)),
        },
    ),
    "joint_pos_limits": RewardTermCfg(
        func=envs_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
}
```

---

## 5. Ablation Order

Train one term at a time. Confirm learning signal exists before adding the next. Use reward component logging (log each term separately).

| Step | Terms active | What to verify |
|---|---|---|
| **0. Baseline** | `nail_driven_reward` (Gaussian, current) + `approach` (current, always-on) + `action_rate` | Can the policy drive the nail at all? Expect slow-press hack. Confirms sim works. |
| **1. Progress signal** | Replace `nail_driven_reward` → `nail_depth_delta` (×500) | Does the policy now get learning signal at 0mm depth? Check: `nail_depth_delta` reward should be nonzero in early rollouts. |
| **2. Contact gating** | Replace `approach` → `approach_contact_gated` | Does the policy stop hovering at nail_top? Check: `approach` reward should drop to ~0 after contact is made. |
| **3. Impact bonus** | Add `impact_velocity_bonus` (×10) | Does the policy start swinging instead of pressing? Check: look for velocity spike in `head_vel` at contact transitions. |
| **4. Air-time bonus** | Add `air_time_bonus` (×3) | Does the policy start pulling higher before striking? Check: `last_air_time` distribution should shift right. |
| **5. Smoothness** | Add `joint_vel` (−0.005) | Minimal effect on task performance; check reward curve doesn't drop. |
| **6. Completion** | Add `completion_bonus` (×100) | Episode length should decrease (faster success). |
| **7. Near-goal Gaussian** | Add `nail_driven_neargoal` (×0.5) | Improves final nail depth precision. May not matter if termination is at threshold. |

---

## 6. Pre-Training Verification (Mandatory)

<!-- opus-audit M5: updated to point at the actually-used scripts. -->

Before any training run, execute the two verification scripts:

```bash
# 1. Confirm reward terms produce expected values for known states (8 phases)
python docs/research/reward-design/validate_rewards.py

# 2. Confirm ContactSensor resolves the right geom and produces sensor data
python docs/research/reward-design/verify_contact_sensor.py

# 3. (Optional, more exhaustive) Random-policy reward-fire sweep
python docs/research/reward-design/verify_reward_setup.py
```

This catches three classes of silent failure that would otherwise waste hours of training:

1. **ContactSensor geom pattern miss** — if `"hammer_head"` doesn't resolve, `impact_velocity_bonus` returns zeros forever
2. **`site_vel_w` lazy evaluation** — only relevant if you use the stateless velocity variant (default `ImpactVelocityBonusTerm` finite-diffs `site_pos_w`, so this is no longer an issue)
3. **Dead reward terms** — any term that doesn't produce a single nonzero value in 200 random-policy steps is broken

The script must print `=== Reward setup verified. Safe to start training. ===` before launching `train.py`.

Also enable **per-term reward logging** in the trainer config (rsl_rl logs `Episode_Reward/<term_name>` automatically via `RewardManager.reset()` extras at `mjlab/managers/reward_manager.py:107`). Watch the wandb run during the first 100 iterations:
- `Episode_Reward/nail_depth_delta` must become > 0 by iteration 5
- `Episode_Reward/impact_velocity` must become > 0 by iteration 50
- `Episode_Reward/air_time` must become > 0 by iteration 50

Any of these flat at zero past these checkpoints → kill the run, debug the term.

---

## 7. Implementation Notes

### Registering ContactSensor in scene config

The `ContactSensorCfg` must be added to `SceneCfg` in `make_hammer_env_cfg()`. See §3.2 for the config. Geom pattern `"hammer_head.*"` must match the actual geom name in the Z1 XML — verify with `mj_printSchema` or scene_info.py.

### Stateful terms and vectorised envs

`NailDepthDeltaTerm` inherits `ManagerTermBase`. mjlab calls `reset(env_ids)` automatically on episode termination. The `_max_depth` tensor is shape `[num_envs]` on the correct device — no manual device management needed after `__init__`.

### ContactSensor and `site_vel_w`

`robot.data.site_vel_w` shape is `[B, num_sites, 6]` (linear + angular). Slice `[:, :, :3]` for linear velocity. Verify site index with `SceneEntityCfg("robot", site_names=("hammer_head_site",))`.

### Logging per-term rewards

In rsl_rl / PPO training, log each reward term separately to diagnose which phase is driving learning. Add to wandb/tensorboard: `reward/approach`, `reward/impact_velocity`, `reward/nail_depth_delta`, etc. Essential for ablation debugging.

---

## 8. Weight Scale Reference

<!-- opus-audit M4: per-term cumulative estimates are order-of-magnitude
     approximations, not exact totals. The "Total max per episode" arithmetic
     does not balance precisely with the row entries. Use these as sanity-check
     targets when logging per-term episode sums, not as ground truth. -->

| Term | Weight | Expected range per step | Approx. max cumulative |
|---|---|---|---|
| `approach_contact_gated` | +0.5 | 0 – 0.5 | ~50 (approach phase, before contact) |
| `impact_velocity_bonus` | +10.0 | 0 or 5–10 (at impact only) | ~50–100 (5–10 strikes) |
| `air_time_bonus` | +3.0 | 0 or 0–1.5 (at impact only) | ~15–30 |
| `nail_depth_delta` | +500.0 | 0 or 0.001–0.01 | 37.5 (full 75 mm) |
| `nail_driven_neargoal` | +0.5 | 0 – 0.5 | ~50 (near goal only) |
| `completion_bonus` | +100.0 | 0 or 100 (once) | 100 |
| `action_rate` | −0.02 | −0.0 to −0.05 | ~−10 to ~−50 (depends on episode length) |
| `joint_vel` | −0.005 | −0.005 to −0.03 | ~−10 to ~−30 |
| `joint_pos_limits` | −1.0 | 0 (safe) | 0 (safe) |

**Ballpark successful episode:** dense shaping cumulative (~50 approach + ~100 impact + ~25 air_time + 37.5 depth_delta) + 100 completion ≈ **+300 positive contributions**, minus ~−20 to ~−80 smoothness penalties. **Verify by logging per-term episode sums during the first training run** — the spec gives orders of magnitude, not balanced arithmetic.
