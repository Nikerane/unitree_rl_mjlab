"""Minimal verification: does the hammer_nail_contact ContactSensor resolve correctly?

Run:
    python docs/research/reward-design/verify_contact_sensor.py

Pass condition: ContactSensor resolves to >=1 primary geom (hammer_head),
and produces a non-zero `found` signal when the hammer is driven into the nail.
"""

from __future__ import annotations

import sys

import torch

from mjlab.envs import ManagerBasedRlEnv
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg


N_ENVS = 4


def main() -> None:
    cfg = z1_hammer_env_cfg(play=True)
    cfg.scene.num_envs = N_ENVS
    cfg.scene.env_spacing = 2.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg, device=device)

    # --- Check 1: sensor exists and resolves to >=1 primary ---
    sensor = env.scene["hammer_nail_contact"]
    primaries = sensor.primary_names
    print(f"ContactSensor primaries: {primaries}")
    assert len(primaries) > 0, (
        "FAIL: ContactSensor 'hammer_nail_contact' resolved to ZERO primary geoms."
    )
    print(f"[OK] {len(primaries)} primary geom(s) resolved.")

    # --- Check 2: sensor produces `found` data shape ---
    env.reset()
    found = sensor.data.found
    print(f"sensor.data.found shape: {tuple(found.shape)} (expected [{N_ENVS}, P])")
    print(f"sensor.data.found dtype: {found.dtype}")

    # --- Check 3: print starting geometry to diagnose any gap ---
    robot = env.scene["robot"]
    nail_block = env.scene["nail_block"]

    head_site_id = robot.find_sites("hammer_head_site")[0]
    nail_site_id = nail_block.find_sites("nail_top")[0]

    # site_pos_w is [B, num_sites, 3] when indexed with a list; squeeze to [B, 3]
    head_pos_0 = robot.data.site_pos_w[:, head_site_id].reshape(N_ENVS, 3).clone()
    nail_pos_0 = nail_block.data.site_pos_w[:, nail_site_id].reshape(N_ENVS, 3).clone()
    gap_0 = nail_pos_0 - head_pos_0   # [B, 3] vector from head to nail

    print(f"\nStarting geometry (env 0):")
    print(f"  hammer_head_site world pos: {head_pos_0[0].tolist()}")
    print(f"  nail_top         world pos: {nail_pos_0[0].tolist()}")
    print(f"  gap (nail - head):          {gap_0[0].tolist()}")
    print(f"  Euclidean distance:         {torch.norm(gap_0[0]).item():.4f} m")

    # --- Check 4: drive hammer toward nail using a direction-of-gap action ---
    # The action is Δ(hammer_head) — a unit vector toward the nail will close the gap.
    print("\nDriving hammer toward nail (direction-of-gap action) for 200 steps...")
    action_dim = env.action_manager.total_action_dim
    direction = gap_0 / torch.norm(gap_0, dim=-1, keepdim=True).clamp_min(1e-6)
    move_action = torch.zeros(N_ENVS, action_dim, device=env.device)
    move_action[:, :3] = direction

    any_contact_seen = False
    contact_step = -1
    for step in range(200):
        env.step(move_action)
        if (sensor.data.found > 0).any():
            any_contact_seen = True
            contact_step = step
            print(f"[OK] First contact detected at step {step}.")
            print(f"     found per env: {sensor.data.found.squeeze(-1).tolist()}")
            if sensor.data.force is not None:
                f_norm = torch.norm(sensor.data.force.squeeze(1), dim=-1)
                print(f"     force magnitude per env: {f_norm.tolist()}")
            break

    if not any_contact_seen:
        head_now = robot.data.site_pos_w[:, head_site_id].reshape(N_ENVS, 3)
        nail_now = nail_block.data.site_pos_w[:, nail_site_id].reshape(N_ENVS, 3)
        final_dist = torch.norm(nail_now - head_now, dim=-1)
        print(f"[FAIL] No contact detected in 200 steps.")
        print(f"       Final hammer→nail distance per env: {final_dist.tolist()}")
        print(f"       The hammer cannot reach the nail with the current neutral pose / IK reach.")
        sys.exit(1)

    print(f"\n=== ContactSensor verified. First contact at step {contact_step}. ===")


if __name__ == "__main__":
    main()
