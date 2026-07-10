> ⚠️ **ARCHIVED 2026-07-05** — superseded by `docs/results/2026-06-17_b_strike.md` (applied items) + `docs/archive/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` (deferred reward work).
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Future updates for the Z1 hammer task

Things worth changing once the current baseline is validated. Each item lists the observation that motivated it, the concrete fix, and the file to change.

> **Status (2026-06-10, plan stage T0):** items **1** (`clip_actions=1.0`), **2a** (±0.05 rad reset noise, train-only), **5** (frictionloss 30 N, damping 0.5 — applied to BOTH scene XMLs, fixing the viz-scene desync), and **6a** (goal 0.032 / threshold 0.030, range "0 0.032") are **APPLIED**. Items 2b (nail xy randomisation), 3 (curriculum), 4 (impact_velocity_bonus — superseded by `impact_progress`), 6b, and 7 remain open. See `docs/research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md`.

---

## 1. Action range — clip policy outputs to `[-1, 1]`

**Observation:** The trained policy at iter 500 outputs actions like `[-3.06, -2.07, -7.83]` — values far outside the expected `[-1, 1]` Gaussian action range. PPO's Gaussian policy outputs unbounded values; nothing was clipping them before they reached `DifferentialIKActionCfg`. The IK then multiplies by `delta_pos_scale=0.05` and asks for ~40 cm displacement per step. Joint limits and IK saturation cap the actual motion, but the policy is essentially commanding "go as hard as possible" with no concept of bounded actions.

**Why this matters:** Brittle for sim-to-real (real hardware has hard torque limits that aren't a smooth saturation), and confuses the action-rate penalty term whose magnitudes assume bounded actions.

**Fix:** add `clip_actions=1.0` to the PPO runner config.

File: `src/tasks/hammer/config/z1/rl_cfg.py`

```python
return RslRlOnPolicyRunnerCfg(
    ...
    clip_actions=1.0,   # NEW
    ...
)
```

---

## 2. Reset randomization — policy is deterministic and memorised one trajectory

**Observation:** After training, action std = 0.12 and the policy produces the *exact same 5-step pattern* every episode. The reset always places arm and nail at identical positions, so the policy memorised one fixed sequence rather than learning a closed-loop strike.

**Why this matters:** The policy will fail if the nail moves, the arm starts slightly differently, or anything in the scene varies. It's not really a "policy", it's a recorded trajectory.

**Fix:** randomize the reset state across episodes.

### 2a. Arm initial joint noise

File: `src/tasks/hammer/hammer_env_cfg.py` — the `reset_robot_joints` event term.

```python
"reset_robot_joints": EventTermCfg(
    func=envs_mdp.reset_joints_by_offset,
    mode="reset",
    params={
        "position_range": (-0.05, 0.05),  # was (0.0, 0.0)
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
),
```

Adds ±0.05 rad (~3°) noise to each joint at reset. Small enough to keep the arm in striking range, large enough to force the policy to actually look at `joint_pos` observations.

### 2b. Nail position noise

Requires writing a new event term that perturbs the nail body's world position by a few cm in x and y. The `nail_top_pos` observation already exposes this to the policy, so the policy can learn to aim — but only if the nail actually moves.

Probably needs a new `reset_nail_xy` event func added to `src/tasks/hammer/mdp/events.py` (file would need to be created) that overwrites the nail body's qpos at reset. Reference how `envs_mdp.reset_joints_by_offset` works — same pattern but for a freejoint or a small randomization.

---

## 3. Curriculum — progressively harder reset distance

Once 1 and 2 are in place and stable, gradually move the arm reset pose further from the nail across training. The current near-nail pose makes the task trivially solvable from random actions (we measured: random downward actions reach the nail). A curriculum that starts close and ends at the neutral pose (26 cm away) would teach the full approach + strike sequence.

Easiest implementation: weighted mixture of two reset poses, with the "far" weight increasing across iterations. Reference any of mjlab's tracking-task curricula in `mjlab/envs/mdp/curriculums/` for patterns.

---

## 4. impact_velocity_bonus (deferred reward term)

Already specced in `docs/research/reward-design/RECOMMENDED_REWARD_SPEC.md` as a deferred term. Only add **if** observation: policy presses the nail slowly instead of swinging. With reset randomization (item 2) we expect a learned closed-loop swing, but it's worth measuring head_vel at contact in the trained policy before deciding.

Implementation outline (already discussed):

```python
def impact_velocity_bonus(env, contact_sensor_cfg, robot_cfg):
    sensor = env.scene.sensors[contact_sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)  # event gate, not continuous
    head_vel_w = robot.data.site_vel_w[:, robot_cfg.site_ids].squeeze(1)[:, :3]
    downward_vel = (-head_vel_w[:, 2]).clamp_min(0.0)
    return first_contact.any(dim=-1).float() * downward_vel
```

Key: gate by `compute_first_contact()` (one-shot per strike), not contact force magnitude (continuous and sim-to-real unsafe).

---

## 5. Increase nail `frictionloss` — task is too easy

**Observation:** trained policy drives the nail 9 mm → 60 mm in a single 20 ms control step (51 mm of nail travel from one strike). Real nail-in-wood resistance would require multiple strikes; here a single saturated downward IK command solves the task.

Current values in `nail_block_scene.xml`:
```xml
damping="1" frictionloss="10.0"
```

`frictionloss=10 N` was chosen so a Z1 arm (~30–60 N effort limit per joint) could overcome it. But because the hammer head has 0.25 kg mass and the IK can move it ~5 cm/step, contact transfers enough impulse to drive 51 mm in one step against 10 N of friction (work = 10 × 0.051 = 0.51 J).

**Important: this item depends on item 1.** The right value for `frictionloss` is coupled to whether `clip_actions` is enforced, because the strike force the policy can deliver depends on the IK command magnitude.

**Energy budget calculation (with item 1 applied, `clip_actions=1.0`):**
- Hammer mass: 0.25 kg
- Max commanded IK delta: 5 cm/step at 50 Hz → 2.5 m/s commanded velocity
- Realistic hammer velocity at contact (with PD damping): ~1.5 m/s
- Kinetic energy at contact: `0.5 × 0.25 × 1.5² ≈ 0.28 J`

With friction `N` newtons, work-energy gives nail travel `d = KE / N`:

| frictionloss | Travel per strike | Strikes needed for 30 mm |
|---|---|---|
| 10 N (current) | very large (sustained-push regime) | 1 |
| 30 N | ~10 mm | 3 |
| 50 N | ~6 mm | 5 |
| 100 N | ~3 mm | 10 |
| 150 N | ~2 mm | 15 (likely too hard) |

**Without item 1**, the policy commands huge deltas (up to 40 cm/step), so contact force is sustained at the IK saturation limit rather than just impulsive. Friction would need to be ~3× higher to restrict travel similarly. The values above assume item 1 is in place.

**Recommended starting point** (apply item 1 first, then):
```xml
<joint name="nail_slide"
       type="slide"
       axis="0 0 -1"
       range="0 0.075"
       damping="0.5"
       frictionloss="30.0"/>
```

If the policy stalls at zero progress, drop to 20 N. If it still completes in one swing, raise to 50 N. Damping `0.5` (down from `1.0`) reduces viscous resistance which would otherwise fight the swing we want to encourage.

Remember to update `hammer_nail_scene.xml` to match (the file headers now warn about the sync requirement).

---

## 6. Realistic nail geometry — head should end flush with block surface

**Observation:** In the current scene the nail visually passes through the block during driving. The geometry is set up so the joint range (7.5 cm) is the constraint, not contact with the block.

```
Block top z       = 0.060 m
Nail head at qpos=0       = 0.092 m  → starts 3.2 cm above surface
Nail head at qpos=0.075   = 0.017 m  → ends 4.3 cm below surface (passes through)
```

In real life, the nail head should stop when it hits the block surface — the realistic travel distance equals "distance from initial head position to block surface" ≈ 3.2 cm with the current geometry, NOT 7.5 cm.

**Two ways to fix:**

### 6a. Quick fix: shrink the goal depth to match current geometry

In `src/tasks/hammer/nail_block.py`:
```python
NAIL_GOAL_DEPTH: float = 0.032          # was 0.075
NAIL_SUCCESS_THRESHOLD: float = 0.030   # was 0.07
```

And update the joint range in `nail_block_scene.xml` and `hammer_nail_scene.xml`:
```xml
<joint name="nail_slide" ... range="0 0.032"/>
```

This matches the current physical setup: nail driven until head reaches block surface. Block_geom collision can then be re-enabled because the nail_head naturally stops at the surface.

### 6b. Proper fix: redesign geometry for a full 7.5 cm strike

If we want a 7.5 cm goal depth (more nail travel = more interesting RL task), redesign so the geometry supports it.

**Math** — derive nail body initial z:
- Block top z = 0.060 m
- Nail head bottom in world frame = `body_z - qpos + 0.006 - 0.004 = body_z - qpos + 0.002`
- At qpos = 0.075 (fully driven), want head bottom flush with block top:
  - `body_z - 0.075 + 0.002 = 0.060`
  - `body_z = 0.060 + 0.075 - 0.002 = 0.133 m`

So set `nail body pos="0.50 0 0.133"` (up from current 0.090, a shift of +4.3 cm).

**Shaft length**: current shaft is 8 cm (half-length 0.04). With block 6 cm thick, the shaft will protrude through the bottom by 2 cm when fully driven. To make the shaft fully buried (more realistic), shorten to match block thickness:
- shaft half-length 0.030 (= 6 cm full length)
- shaft local pos `0 0 -0.030` (so top of shaft sits at nail body origin level, base 6 cm below)

Concretely in `nail_block_scene.xml`:
```xml
<body name="block" pos="0.50 0 0.03">     <!-- unchanged: top at z=0.06 -->
  <geom name="block_geom" type="box" size="0.125 0.09 0.03" .../>  <!-- can re-enable collision -->
</body>

<body name="nail" pos="0.50 0 0.133">     <!-- raised from 0.09 -->
  <joint name="nail_slide" range="0 0.075" .../>
  <geom name="nail_shaft" pos="0 0 -0.030" size="0.003 0.030" .../>  <!-- 6 cm long, matches block -->
  <geom name="nail_head"  pos="0 0 0.006" size="0.012 0.004" .../>
  <site name="nail_top"   pos="0 0 0.012" .../>
</body>
```

**Verification:**
- At qpos=0: head bottom = 0.133 + 0.002 = 0.135 m (7.5 cm above block top ✓)
- At qpos=0.075: head bottom = 0.133 - 0.075 + 0.002 = 0.060 m (flush with block top ✓)
- At qpos=0.075: shaft bottom = 0.133 - 0.075 - 0.030 - 0.030 = -0.002 m (poking 0.2 cm through bottom — close enough)

**Side effect**: `nail_top` site now starts at z=0.145 instead of 0.102 — that's 4.3 cm higher. The arm's `NEAR_NAIL_JOINT_POS` will need to be retuned via `solve_ik.py` to put the hammer above this new nail position. Re-run `python hammer_z1_env/solve_ik.py --z 0.18` or similar to get new joint angles.

### Which to do

Quick fix (6a) is cheaper and is realistic for a real ~3 cm driving task (think a brad or finishing nail in a thin board). Proper fix (6b) gives a larger action range but adds geometry-tuning work. Either way, **the current 7.5 cm goal with the nail passing through the block is not physically meaningful** and should be replaced.

---

## 7. Sim-to-real readiness items (longer-term)

Not urgent but worth noting:

- **Torque mode / impedance control:** DifferentialIK + stiff PD position servos are not the right hardware paradigm for impact tasks. See the "Future direction" section in `hammer_z1_env/README.md`. Real Z1 deployment likely needs torque-mode control with task-space impedance.
- **Observation noise:** current noise is generic (±0.005 m on positions). Calibrate to the actual Z1 encoder noise / mocap noise once hardware testing begins.
- **Contact force observation:** the `hammer_nail_contact` sensor exposes `data.force` — could be added as an observation, but van Steen et al. (2024) shows contact force is the *least* faithful quantity for sim-to-real. Skip unless specifically needed.
