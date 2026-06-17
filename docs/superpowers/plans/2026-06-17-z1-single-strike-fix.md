# Z1 Single-Strike Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Z1 deliver a genuine, hardware-honest hammer strike (instead of a ~0.45 m/s servo push) by fixing the task — a nail-depth clamp, a joint-velocity rail, and a larger action step — without adding any new reward terms.

**Architecture:** Three code changes + one GPU verification. (1) Clamp the nail slide depth to its physical range `[0, 0.032]` at every reward/termination read site (the soft limit lets a hard strike overshoot to ~63 mm, which would inflate progress rewards and any future impulse integral). (2) Add the missing joint-velocity rail via the DiffIK `max_dq` clamp set to the real Z1 limit (3.1415 rad/s). (3) Raise `delta_pos_scale` so the policy can command a ~1.35 m/s strike (the terminal head speed is linear in `delta_pos_scale`). Then retrain and verify a real strike emerges, the press stays excluded, and joint speeds stay within the real limit.

**Tech Stack:** Python, PyTorch, mjlab 1.4.0, mujoco/_warp 3.8.1, rsl_rl, Slurm (Vega A100s).

## Global Constraints

- **No new reward terms** — augment-not-replace. This fix changes *task config*, not the reward stack. (Design record: `docs/superpowers/specs/2026-06-17-z1-strike-not-press-redesign-design.md`.)
- **One GPU per training run**, fan out seeds with a Slurm array. Never one multi-GPU job.
- **Run the full local gate before any GPU submit:** `validate_rewards.py` (all phases), `verify_contact_sensor.py`, `verify_reward_setup.py`, `pytest tests/`.
- **Velocity rail value:** `max_dq = 3.1415 rad/s × physics_dt`, `physics_dt = cfg.sim.mujoco.timestep = 0.002 s` ⇒ `0.006283 rad/substep`. Real Z1 joint-velocity limit = **3.1415 rad/s** (official `z1_description` URDF), torque 30/60 N·m (already matched in sim).
- **Action step target:** `delta_pos_scale = 0.15` ⇒ terminal head speed ≈ `0.18 × 0.15 / 0.02 ≈ 1.35 m/s` (max joint speed ≈ 2.2 rad/s < 3.1415 ✓; head KE ≈ 0.45 J). Conservative within-limit value; fall back to 0.10 if control degrades.
- **Scope: single-strike.** Multi-strike deferred (design §6). Do not change nail friction in this plan (that would force multi-strike).
- **This plan touches only `unitree_rl_mjlab` Python config** — no asset-XML edits, so no `safe_impact_manipulation` change.

---

## File Structure

- `src/tasks/hammer/mdp/rewards.py` — add a shared `clamped_nail_depth(env, nail_cfg)` helper; route the four depth-reading reward terms through it.
- `src/tasks/hammer/mdp/terminations.py` — route `nail_fully_driven` through the same helper.
- `docs/research/reward-design/validate_rewards.py` — add Phase K (overshoot-clamp assertion).
- `src/tasks/hammer/config/z1/env_cfgs.py` — set `ik_action.max_dq` (velocity rail).
- `src/assets/robots/unitree_z1/z1_constants.py` — raise `Z1_HAMMER_DELTA_POS_SCALE`.
- `scripts/diag_strike_probe.py` — used for verification (already exists).

---

## Task 1: Clamp nail depth at the reward/termination source

**Files:**
- Modify: `src/tasks/hammer/mdp/rewards.py` (add helper near top; edit `nail_driven_reward`, `completion_bonus`, `NailDepthDeltaTerm.__call__`, `ImpactProgressTerm.__call__`)
- Modify: `src/tasks/hammer/mdp/terminations.py:18-29` (`nail_fully_driven`)
- Test: `docs/research/reward-design/validate_rewards.py` (new Phase K)

**Interfaces:**
- Produces: `clamped_nail_depth(env: ManagerBasedRlEnv, nail_cfg: SceneEntityCfg) -> torch.Tensor` (shape `(B,)`), importable from `src.tasks.hammer.mdp.rewards`.

- [ ] **Step 1: Write the failing test (validate_rewards Phase K)**

In `docs/research/reward-design/validate_rewards.py`, (a) extend the nail_block import:

```python
from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD, NAIL_GOAL_DEPTH
```

(b) Insert this phase immediately before the final summary print / `env.close()` in `main()`:

```python
  # --- Phase K: Soft-limit overshoot is clamped at the reward source ---
  # A hard strike transiently drives nail_slide past its 0.032 m stop (~63 mm).
  # Every depth-reading reward term must see the CLAMPED physical depth, not the
  # elastic excursion (else a harder strike inflates nail_depth_delta and any
  # future impulse integral). Previously only the OBSERVATION was clamped, and
  # only on the low side (observations.py:129).
  print("--- Phase K: Soft-limit overshoot clamp ---")
  expected_delta = (NAIL_GOAL_DEPTH - SETTLE) * W_DELTA
  r = {}
  for overshoot in (0.063, 0.10):
    env.reset()                       # NailDepthDeltaTerm max-depth -> SETTLE
    force_nail_depth(env, overshoot)
    r = recompute_rewards(env)
    assert_close(
      r["nail_depth_delta"], expected_delta,
      f"K@{overshoot}m: nail_depth_delta must clamp to GOAL ({NAIL_GOAL_DEPTH} m), "
      f"not pay the {overshoot} m overshoot",
    )
  summary.append(("K", r))
```

- [ ] **Step 2: Run the test to verify it FAILS on current code**

Run: `python docs/research/reward-design/validate_rewards.py`
Expected: FAIL at Phase K — current code reads raw qpos, so at 0.063 m it pays `(0.063 − 0.004) × 600 = 35.4`, not the clamped `(0.032 − 0.004) × 600 = 16.8`.

- [ ] **Step 3: Add the shared helper in `rewards.py`**

Add the import near the other `src.tasks.hammer` imports:

```python
from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH
```

Add the helper just after the `_DEFAULT_NAIL_CFG` definition (before `nail_driven_reward`):

```python
def clamped_nail_depth(env: ManagerBasedRlEnv, nail_cfg: SceneEntityCfg) -> torch.Tensor:
  """Nail slide depth clamped to the physical joint range [0, NAIL_GOAL_DEPTH]. Shape (B,).

  The slide's soft limits let a hard strike transiently overshoot the 0.032 m stop
  to ~63 mm (and a hooking claw pull it below 0). Reward/termination terms must read
  the PHYSICAL depth, not the elastic excursion -- otherwise a harder strike inflates
  the progress reward and any windowed impulse. (Only the observation was clamped
  before, and only on the low side; observations.py:129.)
  """
  nail: Entity = env.scene[nail_cfg.name]
  depth = nail.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
  return depth.clamp(0.0, NAIL_GOAL_DEPTH)
```

- [ ] **Step 4: Route the four reward terms through the helper**

In `nail_driven_reward`, replace the two lines that fetch the entity + depth with:

```python
  current_depth = clamped_nail_depth(env, nail_cfg)
```

In `completion_bonus`, replace the entity-fetch + depth lines with:

```python
  depth = clamped_nail_depth(env, nail_cfg)
  return (depth >= success_depth).float()
```

In `NailDepthDeltaTerm.__call__`, replace the entity-fetch + `depth = ...` lines with:

```python
    depth = clamped_nail_depth(env, nail_cfg)
```

In `ImpactProgressTerm.__call__`, replace the progress-gate depth read (`depth = nail.data.joint_pos[...]`) with:

```python
    depth = clamped_nail_depth(env, nail_cfg)
```

(Leave the head-velocity / contact / sensor logic in `ImpactProgressTerm` untouched.)

- [ ] **Step 5: Route the termination through the helper**

In `src/tasks/hammer/mdp/terminations.py`, add the import:

```python
from src.tasks.hammer.mdp.rewards import clamped_nail_depth
```

and replace the body of `nail_fully_driven` (lines 27-29) with:

```python
  current_depth = clamped_nail_depth(env, nail_cfg)
  return current_depth >= success_depth
```

- [ ] **Step 6: Run validate_rewards — all phases must pass**

Run: `python docs/research/reward-design/validate_rewards.py`
Expected: PASS for every phase A–K. (Phases C/E/F/G/H inject depths ≤ 0.020 m, all below the 0.032 clamp, so their expected values are unchanged; Phase K now reads 16.8.)

- [ ] **Step 7: Run the contact + reward-setup + unit gate (no regression)**

Run:
```bash
python docs/research/reward-design/verify_contact_sensor.py
python docs/research/reward-design/verify_reward_setup.py
pytest -q tests/
```
Expected: all PASS (the clamp is a no-op below 0.032 m, where every existing test operates).

- [ ] **Step 8: Commit**

```bash
git add src/tasks/hammer/mdp/rewards.py src/tasks/hammer/mdp/terminations.py docs/research/reward-design/validate_rewards.py
git commit -m "fix(reward): clamp nail depth to physical range at reward/termination source

The slide soft-limit lets a hard strike overshoot 0.032m to ~63mm; only the
observation was clamped (low side). Route nail_driven, completion, depth-delta,
impact-progress gate, and the nail_driven termination through a shared
clamped_nail_depth() helper so a harder strike cannot inflate the progress
reward or a future impulse integral. validate_rewards Phase K added.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Add the joint-velocity rail (`max_dq`)

**Files:**
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py:48-53` (after the IK wiring block)

**Interfaces:**
- Consumes: `cfg.actions["ik_hammer_head"]` (a `DifferentialIKActionCfg` with a `max_dq` field, default 0.5).

- [ ] **Step 1: Set `max_dq` to the real velocity limit**

In `z1_hammer_env_cfg`, immediately after `ik_action.delta_pos_scale = Z1_HAMMER_DELTA_POS_SCALE` (line 53), add:

```python
  # Velocity rail: clamp the IK joint step to the REAL Z1 joint-velocity limit
  # (3.1415 rad/s, official z1_description URDF). Without this the sim has NO
  # velocity limit (default max_dq=0.5 rad/substep ~= 250 rad/s, ~80x the real
  # limit), so a raised delta_pos_scale would train strikes the real arm cannot
  # reproduce. physics_dt = cfg.sim.mujoco.timestep = 0.002 s -> 0.006283 rad/substep.
  ik_action.max_dq = 3.1415 * cfg.sim.mujoco.timestep
```

- [ ] **Step 2: Verify the rail is non-binding at the CURRENT speed (no regression)**

Run: `python scripts/diag_strike_probe.py --mode max_vel --task Unitree-Z1-Hammer --device cpu --lift-steps 0 --nsteps 30`
Expected: peak axial speed still ≈ 0.45 m/s (the rail does not throttle a 0.45 m/s strike — that needs only ~0.8 rad/s ≪ 3.1415).

- [ ] **Step 3: Verify the rail CAPS an over-fast command**

Run: `python scripts/diag_strike_probe.py --mode max_vel --task Unitree-Z1-Hammer --device cpu --lift-steps 12 --nsteps 45 --delta-scale 0.40`
Expected: peak axial speed is now **capped well below the un-railed 3.55 m/s** (the `delta=0.40` value from the design sweep), demonstrating the rail bounds joint speed. (Before this task it scaled linearly to ~3.55 m/s.)

- [ ] **Step 4: Commit**

```bash
git add src/tasks/hammer/config/z1/env_cfgs.py
git commit -m "feat(env): add joint-velocity rail (max_dq = 3.1415 rad/s * dt)

The sim enforced no joint-velocity limit (max_dq=0.5 ~= 250 rad/s vs the real Z1
3.1415 rad/s). Clamp the IK step to the real limit so any raised delta_pos_scale
stays hardware-faithful. Non-binding at the current 0.45 m/s strike.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Raise `delta_pos_scale` to enable a real strike

**Files:**
- Modify: `src/assets/robots/unitree_z1/z1_constants.py:173-174` (`Z1_HAMMER_DELTA_POS_SCALE`)

**Interfaces:**
- Consumes: `Z1_HAMMER_DELTA_POS_SCALE` (used by `env_cfgs.py:53` for the action scale and by the diag scripts as the playback `scale`).

- [ ] **Step 1: Raise the action scale**

In `src/assets/robots/unitree_z1/z1_constants.py`, replace the constant and its comment:

```python
# DifferentialIK scale: max position delta the policy commands per control step.
# 0.05 -> 0.15 (2026-06-17): terminal head speed is linear in this scale
# (~0.18 * delta_pos_scale / dt), so 0.05 capped the strike at ~0.45 m/s -- a gentle
# servo push, not a momentum blow. 0.15 -> ~1.35 m/s (max joint ~2.2 rad/s < the
# 3.1415 rad/s rail; head KE ~0.45 J), enabling a genuine single strike. See
# docs/superpowers/specs/2026-06-17-z1-strike-not-press-redesign-design.md.
Z1_HAMMER_DELTA_POS_SCALE: float = 0.15
```

- [ ] **Step 2: Verify the new strike speed (probe)**

Run: `python scripts/diag_strike_probe.py --mode max_vel --task Unitree-Z1-Hammer --device cpu --lift-steps 12 --nsteps 45`
Expected: peak axial speed ≈ **1.3–1.4 m/s** (up from 0.45), and the run reaches contact and drives the nail. (This is with the Task 2 rail active — confirms 1.35 m/s is within the limit.)

- [ ] **Step 3: Re-run the full local gate (scale change touches the scripted reference/playback)**

Run:
```bash
python docs/research/reward-design/validate_rewards.py
python docs/research/reward-design/verify_contact_sensor.py
python docs/research/reward-design/verify_reward_setup.py
pytest -q tests/
```
Expected: all PASS. **If Phase J (phase machinery) or the scripted-playback / Phase M gate fails** because the playback scaling changed with `delta_pos_scale`, the bounded fix is to bump the playback step budget (`NSTEPS`) or set `references.py` `descent_speed` to the new achievable per-step motion (~0.027 m/step); re-run the gate. Do NOT change reward weights.

- [ ] **Step 4: Commit**

```bash
git add src/assets/robots/unitree_z1/z1_constants.py
# include references.py ONLY if Step 3 required recalibrating the playback
git commit -m "feat(env): raise delta_pos_scale 0.05->0.15 for a real ~1.35 m/s strike

Terminal head speed is linear in delta_pos_scale; 0.05 capped the hammer at a
gentle 0.45 m/s servo push. 0.15 -> ~1.35 m/s (within the 3.1415 rad/s rail),
enough KE for a momentum-driven strike. Verified by diag_strike_probe.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Retrain and verify a real strike emerges (GPU)

**Files:** none (verification + training run). Uses `scripts/slurm/train_array.sbatch`, `scripts/diag_strike_probe.py`, `scripts/diag_policy_trace.py`.

**Interfaces:**
- Consumes: the Task 1–3 config (clamp + rail + `delta_pos_scale=0.15`).

- [ ] **Step 1: CPU smoke train (sanity, no GPU)**

Run a short local train to confirm the env constructs and learns with the new config:
```bash
python scripts/train.py Unitree-Z1-Hammer --agent.max-iterations 20 --env.scene.num-envs 16 --agent.logger tensorboard --device cpu
```
Expected: 20 iterations complete, no NaN, success rate climbing. (Smoke only — not a result.)

- [ ] **Step 2: Sync the branch to Vega**

```bash
git push origin hammer-z1
ssh vega 'cd ~/repos/unitree_rl_mjlab && git fetch && git checkout hammer-z1 && git reset --hard origin/hammer-z1'
```
Expected: Vega `HEAD` matches local `hammer-z1`.

- [ ] **Step 3: Submit the retrain (3 seeds, one GPU each)**

```bash
ssh vega 'cd ~/repos/unitree_rl_mjlab && ITERS=500 RUN=b_strike TASK=Unitree-Z1-Hammer sbatch --array=0-2 scripts/slurm/train_array.sbatch'
ssh vega 'squeue -u $USER -o "%.12i %.14j %.8T %.10M %R"'
```
Expected: 3 tasks RUNNING on one A100 each. (If 0.15 destabilizes — value loss spikes, success collapses — resubmit with `delta_pos_scale=0.10`, the design's fallback sweep point.)

- [ ] **Step 4: Pull final metrics + checkpoints**

After completion, pull TB scalars (success, mean ep length, per-term rewards) and the `model_499.pt` for each seed (pattern from the V1 analysis in `docs/VEGA_TRAINING_PLAN.md`). Acceptance: success rate high (≥ ~90%) and stable across seeds.

- [ ] **Step 5: Verify the strike is real and the press is still excluded**

On a pulled checkpoint:
```bash
python scripts/diag_policy_trace.py --task Unitree-Z1-Hammer --ckpt <model_499.pt> --num-envs 64 --nsteps 80 --device cpu
python scripts/diag_strike_probe.py --mode press_basin --task Unitree-Z1-Hammer --ckpt <model_499.pt> --device cpu --nsteps 30 --no-term
```
Acceptance, all of:
- peak v_axial @ contact **markedly higher than the 0.447 m/s V1 baseline** (toward ~1 m/s+) — a genuine strike;
- success rate ~100%, episodes terminate on `nail_driven` (≈0 timeout);
- press-basin: the policy still **lifts/re-strikes**, does not settle into a static push;
- implied max joint speed ≤ 3.1415 rad/s (cross-check via the Jacobian: ~1.35 m/s ⇒ ~2.2 rad/s; the rail guarantees it).

- [ ] **Step 6: Record results**

Append a "B-STRIKE" results section to `docs/VEGA_TRAINING_PLAN.md` (run IDs, strike speed before/after, success, press-basin verdict) and update the design record status to DONE. Commit the docs.

---

## Self-Review

- **Spec coverage:** §5 of the design — clamp fix (Task 1), `max_dq` rail (Task 2), `delta_pos_scale` (Task 3), retrain+verify (Task 4). `v_expected` recalibration is **deliberately deferred** (not preemptive): with the faster strike, `impact_progress` now spans a healthy 0–1.35 range and rewards speed; re-tune only if Task 4 shows it distorting (augment-not-replace). §6 multi-strike is out of scope by decision. ✓
- **Placeholder scan:** none — every code step shows the exact edit; the one conditional (Phase M recalibration in Task 3 Step 3) gives the concrete bounded fix.
- **Type consistency:** `clamped_nail_depth(env, nail_cfg) -> (B,)` defined in Task 1, consumed identically in `rewards.py` and `terminations.py`; `max_dq` and `delta_pos_scale` are existing `DifferentialIKActionCfg` / constant fields.
