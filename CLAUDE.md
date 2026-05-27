# unitree_rl_mjlab — agent context

## Reward design (Z1 hammer task)

If working on rewards for the `hammer-z1` branch, **start by reading**:

- `docs/research/reward-design/DEEP_RESEARCH_REPORT.md` — synthesised entry point; indexes the 6 source files in that folder.

Load-bearing facts that override the spec:

- **Trust the code, not the spec.** The implemented 6-term baseline in `src/tasks/hammer/hammer_env_cfg.py` is the source of truth. `RECOMMENDED_REWARD_SPEC.md`'s 9-term "complete config" is aspirational; the user's explicit strategy is **augment-not-replace** (add a term only when a specific failure mode is observed during training).
- **Live weights** (read these from the code, not the spec): `approach=0.1`, `nail_driven=2.0`, `nail_depth_delta=2000`, `completion=100`, `action_rate=−0.01`, `joint_pos_limits=−10`. `validate_rewards.py` reads them dynamically via `reward_manager.get_term_cfg(name).weight` — no need to keep its constants in sync.
- **Pre-training gate:** before any GPU training, run `validate_rewards.py` (8 phases must pass) and `verify_contact_sensor.py`. Optional: `verify_reward_setup.py` for a random-policy sweep.
- **Before deciding `air_time_bonus` is essential:** run `docs/research/reward-design/test_single_strike.py` to resolve Q1 (single-strike feasibility).
- Open questions (Q2, Q3, Q6, Q7, Q9, Q11, Q12) require either GPU training or hardware — see `OPEN_QUESTIONS.md`.

## Environment

- Conda env: `unitree_mjlab` — **mjlab 1.4.0**, mujoco 3.8.1, mujoco_warp 3.8.1.
- Z1 + hammer assets live in a sibling repo: `~/repos/safe_impact_manipulation/hammer_z1_env/assets/`.
- View scene: `python hammer_z1_env/view.py` (normal mode, weld active) or `mjpython hammer_z1_env/view.py --no-weld` (sliders mode, gravity disabled, requires `mjpython` on macOS).
