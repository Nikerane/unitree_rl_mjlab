# unitree_rl_mjlab — agent context

## Reward design (Z1 hammer task)

If working on rewards for the `hammer-z1` branch, **start by reading**:

- `docs/research/reward-design/DEEP_RESEARCH_REPORT.md` — synthesised entry point; indexes the 6 source files in that folder.
- `docs/research/hammering_reward_design_deep_dive_v2.md` — **v2 deep dive (2026-05-29)**: external-literature pass (29 net-new citations). Covers repetitive hammering (reward machine), the position-only DiffIK action-space constraint (impact lever is momentum `m_eff·v_axial`, not force), safety-constrained RL, and a reward-hacking checklist. Read after the entry point above; flags several v1 recommendations (orientation gate, torque/impedance terms) as not actionable on the current action space.

Load-bearing facts that override the spec:

- **Trust the code, not the spec.** The implemented **7-term** baseline in `src/tasks/hammer/hammer_env_cfg.py` is the source of truth. `RECOMMENDED_REWARD_SPEC.md`'s 9-term "complete config" is aspirational; the user's strategy is **augment-not-replace** (add a term only when a specific failure mode is observed) — with one deliberate, literature-backed exception: changes **#1 + #2** (2026-06-02) rebalanced `nail_depth_delta` 2000→600 and added the double-gated `impact_progress` term up front. See `docs/research/reward-design/IMPACT_PROGRESS_IMPL_SPEC.md`.
- **Live weights** (read these from the code, not the spec): `approach=0.1`, `nail_driven=2.0`, `nail_depth_delta=600`, `impact_progress=8`, `completion=100`, `action_rate=−0.01`, `joint_pos_limits=−10`. `validate_rewards.py` reads them dynamically via `reward_manager.get_term_cfg(name).weight` — no need to keep its constants in sync.
- **Pre-training gate:** before any GPU training, run `validate_rewards.py` (**9 phases** must pass — Phase I covers `impact_progress`) and `verify_contact_sensor.py`. Also `verify_reward_setup.py` (random-policy sweep; refreshed to current API 2026-06-02). Unit tests: `pytest tests/test_impact_progress_reward.py`.
- **Before deciding `air_time_bonus` is essential:** run `docs/research/reward-design/test_single_strike.py` to resolve Q1 (single-strike feasibility).
- Open questions (Q2, Q3, Q6, Q7, Q9, Q11, Q12) require either GPU training or hardware — see `OPEN_QUESTIONS.md`.

## Environment

- Conda env: `unitree_mjlab` — **mjlab 1.4.0**, mujoco 3.8.1, mujoco_warp 3.8.1.
- Z1 + hammer assets live in a sibling repo: `~/repos/safe_impact_manipulation/hammer_z1_env/assets/`.
- View scene: `python hammer_z1_env/view.py` (normal mode, weld active) or `mjpython hammer_z1_env/view.py --no-weld` (sliders mode, gravity disabled, requires `mjpython` on macOS).
