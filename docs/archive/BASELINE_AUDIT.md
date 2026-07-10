> ⚠️ **ARCHIVED 2026-07-05** — superseded by `docs/VEGA_TRAINING_PLAN.md` + `docs/results/` run records.
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Z1 Hammer — Baseline Audit

**Date:** 2026-05-22  
**Task ID:** `Unitree-Z1-Hammer`  
**Repos:** `unitree_rl_mjlab` + `safe_impact_manipulation/hammer_z1_env/assets`

This document records structure comparison vs open-source mjlab/Unitree patterns and **validation results** from today's smoke test.

---

## Validation results (executed today)

| Step | Command / artifact | Result |
|------|-------------------|--------|
| Offscreen render | `./scripts/retest_video.sh` (with `PYTHON=conda env python`) | **PASS** — frame mean ~72.6, 93% non-dark pixels |
| pytest fast | `pytest tests/ -m "not integration"` | **PASS** — 98 passed, 21 deselected |
| Play zero/random | `scripts/_play_sanity.py` | **PASS** — 10 steps each agent |
| Smoke train | `train.py` 50 iters, tensorboard, 64 envs, `--gpu-ids '[]'` | **PASS** — run `2026-05-22_14-02-19`, ~6.5 min |
| Checkpoint load | `scripts/_play_checkpoint_sanity.py` → `model_49.pt` | **PASS** — `action_std≈0.90` (policy non-zero) |
| Train + video | 10 iters, 8 envs, `--video True` | **PASS** — run `2026-05-22_14-09-50`, mp4 under `videos/train/` |
| Train video pixels | `rl-video-step-100.mp4` | **PASS** — not blank (verify locally) |
| **Play + video** | `play.py ... --video True --video-length 200 --viewer viser --num-envs 1` (CPU, checkpoint `model_49.pt`) | **PASS** — `videos/play/rl-video-step-0.mp4` (88 KB, 93% non-dark pixels), viser served on `http://localhost:8080` |

### Machine-specific notes

- **GPU train failed:** PyTorch CUDA init error (driver 12.6 vs PyTorch CUDA build). Use `--gpu-ids '[]'` until driver/PyTorch aligned.
- **ONNX export warning:** `'joint_pos'` — training continues; fix in `HammerOnPolicyRunner` later.
- **Interactive viewer:** Run manually:
  ```bash
  python scripts/play.py Unitree-Z1-Hammer \
    --checkpoint-file logs/rsl_rl/z1_hammer/2026-05-22_14-02-19/model_49.pt \
    --viewer native
  ```
  Viser: elevation **+20°**, **Track camera** on.

### Recommended train command (this machine)

```bash
conda activate unitree_mjlab
cd ~/repos/unitree_rl_mjlab
export MUJOCO_GL=egl

python scripts/train.py Unitree-Z1-Hammer \
  --agent.logger tensorboard \
  --agent.max-iterations 50 \
  --agent.save-interval 25 \
  --env.scene.num-envs 64 \
  --gpu-ids '[]'
```

---

## Reference map

| Source | URL |
|--------|-----|
| mjlab architecture | https://mujocolab.github.io/mjlab/main/source/architecture_overview.html |
| mjlab env config | https://mujocolab.github.io/mjlab/main/source/environment_config.html |
| mjlab training | https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html |
| mjlab viewers | https://mujocolab.github.io/mjlab/main/source/viewers.html |
| mujocolab/mjlab | https://github.com/mujocolab/mjlab |
| unitree_rl_mjlab | https://github.com/unitreerobotics/unitree_rl_mjlab |
| MuJoCo XML | https://mujoco.readthedocs.io/en/latest/XMLreference.html |

**Closest mjlab analogue:** `manipulation/lift-cube` (fixed-base arm + object), not velocity.

---

## A. Registration and entrypoints

| Item | Status | Path / notes |
|------|--------|----------------|
| Task registered | **OK** | `src/tasks/hammer/config/z1/__init__.py` |
| `register_mjlab_task` | **OK** | `task_id`, `env_cfg`, `play_env_cfg`, `rl_cfg`, `runner_cls` |
| Package import | **OK** | `src/tasks/__init__.py`; `train.py` / `play.py` import `src.tasks` |
| Task ID naming | **Different** | `Unitree-Z1-Hammer` (Unitree prefix) vs `Mjlab-*` in upstream mjlab |

---

## B. Config layers

| Layer | Status | Path |
|-------|--------|------|
| Base env MDP factory | **OK** | `src/tasks/hammer/hammer_env_cfg.py` — `make_hammer_env_cfg()` |
| Robot wiring | **OK** | `src/tasks/hammer/config/z1/env_cfgs.py` |
| PPO config | **OK** | `src/tasks/hammer/config/z1/rl_cfg.py` — smaller nets than locomotion |
| Custom runner | **OK** | `src/tasks/hammer/rl/runner.py` — ONNX export (warn on failure) |
| Play mode overrides | **OK** | `episode_length_s=int(1e9)`, corruption off — tested in `test_configs.py` |

---

## C. MDP terms

| Manager | Status | Notes |
|---------|--------|-------|
| Actions | **Different (intentional)** | `DifferentialIKActionCfg` 3D hammer-head Δ vs lift-cube `JointPositionActionCfg` |
| Commands | **OK for now** | `{}` — goal implicit via nail slide joint |
| Observations | **OK** | actor + critic, 33-dim; sites wired in `env_cfgs.py` |
| Rewards | **OK** | `approach`, `nail_driven`, `action_rate`, `joint_pos_limits` — Gaussian shaping vs Gym progress rewards (see backlog) |
| Terminations | **OK** | `time_out`, `nail_driven` |
| Events / DR | **Gap** | Reset joints only; no friction/mass DR |
| Curriculum | **Gap** | `{}` |
| Sensors | **Gap** | No contact sensor (lift-cube has EE-ground contact) |

### Gym vs mjlab control (important)

| | Gym `hammer_z1_env` | mjlab `Unitree-Z1-Hammer` |
|--|---------------------|---------------------------|
| EE control | Mocap + weld | Differential IK + joint PD |
| Zero action | Hold mocap target | Hold *current* head pose (arm can sag) |
| Gripper | In neutral pose | Actuated but excluded from IK |

---

## D. Assets and simulation

| Item | Status | Notes |
|------|--------|-------|
| Robot MJCF | **OK** (risk) | External path: `safe_impact_manipulation/hammer_z1_env/assets/` via `z1_constants.py` |
| Nail/block scene | **OK** | `nail_block.py` |
| Actuators in Python | **OK** | `BuiltinPositionActuatorCfg`; XML has no `<actuator>` — `test_mjcf_spec.py` |
| Gripper | **OK** | Actuated; not in `ARM_ACTUATOR_NAMES` |
| Sim timestep | **OK** | 0.002 s, decimation 10 → 50 Hz control |
| Contact cone | **Different** | Pyramidal vs lift-cube elliptic — compare before sim2real |
| `num_envs` default | **Gap (ops)** | Base cfg `num_envs=1`; override at CLI for training |
| `nconmax` / `njmax` | **OK** | 64 / 300 — raise if multi-env contact issues |

---

## E. Tests

| Test | Status | File |
|------|--------|------|
| Assets | **OK** | `tests/test_assets.py` |
| MJCF spec | **OK** | `tests/test_mjcf_spec.py` |
| Config wiring | **OK** | `tests/test_configs.py` (includes play-mode invariants) |
| Env step / obs | **OK** | `tests/test_env.py` (integration marker) |
| mjlab `test_task_configs` | **Gap** | Not ported for all tasks; hammer play checks exist locally |
| mjlab `test_differential_ik_action` | **Gap** | Run in upstream mjlab repo optionally |
| Reward unit tests | **Gap** | No nail-depth / approach regression tests yet |

---

## F. Tooling and ops

| Item | Status | Notes |
|------|--------|-------|
| `retest_video.sh` | **OK** | Use `PYTHON=$(conda run -n unitree_mjlab which python) ./scripts/retest_video.sh` |
| Logger | **OK** | Use `--agent.logger tensorboard` without wandb login |
| Viser camera | **Gap (UI)** | Negate elevation (+20°) vs native (-20°) |
| Tensorboard | **OK** | Logs under `logs/rsl_rl/z1_hammer/<run>/` |
| Helper scripts | **OK** | `_play_sanity.py`, `_play_checkpoint_sanity.py` for headless checks |
| Sim2real / deploy | **Out of scope** | unitree `deploy/` — after policy quality |

---

## Research backlog (prioritized)

1. Fix GPU path: update NVIDIA driver or install PyTorch build matching driver 12.6; then retry `--env.scene.num-envs 256` without `--gpu-ids '[]'`.
2. Fix ONNX export in `HammerOnPolicyRunner` (`'joint_pos'` observation key).
3. Reward alignment study: map `hammer/mdp/rewards.py` to Gym `compute_reward` progress staging.
4. Port mjlab `test_task_configs` patterns for any new tasks.
5. Contact sensor + EE-table termination (lift-cube pattern).
6. Domain randomization events (friction, mass).
7. Viser camera elevation sign in code.
8. Vendor assets into `unitree_rl_mjlab` or git submodule `safe_impact_manipulation`.
9. Long training run: `max_iterations=5000`, tune `num_envs`, optional wandb.
10. Impedance / torque control (README future direction).

---

## Baseline gate checklist

- [x] Structure matches mjlab factory + `config/z1/` pattern
- [x] pytest fast suite passes
- [x] Offscreen rendering works (`MUJOCO_GL=egl`)
- [x] Smoke PPO train completes with checkpoint
- [x] Checkpoint loads and policy outputs non-zero actions
- [x] Training videos recorded and non-blank
- [ ] Interactive `play --viewer native` (user confirmation)
- [ ] Full research train (5000 iters) — follow-up

---

## Key log paths

| Run | Purpose |
|-----|---------|
| `logs/rsl_rl/z1_hammer/2026-05-22_14-02-19/` | Smoke train 50 iters — `model_25.pt`, `model_49.pt` |
| `logs/rsl_rl/z1_hammer/2026-05-22_14-09-50/` | Video test — `videos/train/*.mp4` |
