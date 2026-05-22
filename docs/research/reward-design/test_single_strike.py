"""Q1 — Does a single maximum-velocity strike fully drive the nail?

Scripts a maximum-velocity downward swing from several approach heights
and logs the resulting nail depth. Answers whether repeated-strike
reward design is mandatory or just useful.

Run:
    python docs/research/reward-design/test_single_strike.py

Interpretation:
    - If max_depth ≥ 0.07 m for any approach height: single-strike solution
      is achievable. Repeated-strike reward is a "nice to have."
    - If max_depth < 0.03 m even at highest approach: repeated striking
      is mandatory. air_time_bonus and impact_velocity_bonus are critical.
    - Intermediate: design depends on the policy's preferred strategy.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg


APPROACH_HEIGHTS = [0.05, 0.10, 0.15, 0.20]  # metres above nail_top
N_TRIALS = 5
N_ENVS = len(APPROACH_HEIGHTS) * N_TRIALS


def main() -> None:
    cfg = z1_hammer_env_cfg(play=True)
    cfg.scene.num_envs = N_ENVS
    cfg.scene.env_spacing = 2.0
    cfg.episode_length_s = 5.0
    env = ManagerBasedRlEnv(cfg)
    env.reset()

    results: dict[float, list[float]] = {h: [] for h in APPROACH_HEIGHTS}

    # Action is 3D Δhammer_head. To swing down hard, command max-negative-z repeatedly.
    # First, lift up to the approach height; then drive down.
    # Each env gets a different approach height assigned round-robin.

    action_dim = env.action_manager.action_dim
    lift_action = torch.zeros(N_ENVS, action_dim, device=env.device)
    lift_action[:, 2] = +1.0   # up
    strike_action = torch.zeros(N_ENVS, action_dim, device=env.device)
    strike_action[:, 2] = -1.0  # down

    # Phase 1: lift to approach height (proportional to height — rough heuristic)
    lift_steps_per_height = {h: int(h / 0.005) for h in APPROACH_HEIGHTS}
    max_lift = max(lift_steps_per_height.values())
    for step in range(max_lift):
        env.step(lift_action)

    # Phase 2: max-velocity strike
    nail = env.scene["nail_block"]
    initial_depth = nail.data.joint_pos[:, 0].clone()

    for step in range(40):  # 40 steps = 0.8 s at 50 Hz
        env.step(strike_action)

    final_depth = nail.data.joint_pos[:, 0]
    depth_advance = (final_depth - initial_depth).cpu().tolist()

    # Group results by approach height (round-robin assignment)
    for i, depth in enumerate(depth_advance):
        h = APPROACH_HEIGHTS[i % len(APPROACH_HEIGHTS)]
        results[h].append(depth)

    print("\n=== Single-strike test results ===\n")
    print(f"{'Approach height (m)':<25} {'Mean depth advance (m)':<25} {'Max depth advance (m)'}")
    for h in APPROACH_HEIGHTS:
        depths = results[h]
        mean_d = sum(depths) / len(depths)
        max_d = max(depths)
        print(f"{h:<25.3f} {mean_d:<25.4f} {max_d:.4f}")

    overall_max = max(max(v) for v in results.values())
    threshold = 0.07
    print(f"\nNail success threshold: {threshold} m")
    print(f"Overall max single-strike depth: {overall_max:.4f} m")
    if overall_max >= threshold:
        print(">>> Single-strike solution IS achievable. Repeated-strike rewards are optional.")
    elif overall_max < 0.03:
        print(">>> Single-strike solution is NOT achievable. Repeated-strike rewards are MANDATORY.")
    else:
        print(">>> Marginal. Repeated-strike rewards likely help but may not be strictly required.")


if __name__ == "__main__":
    main()
