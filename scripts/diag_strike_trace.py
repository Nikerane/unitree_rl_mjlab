"""Trace the actual scripted strike and measure closest hammer->nail approach.

Runs the real warp env exactly like render_reference, but every control step
copies qpos into the compiled CPU model and computes mj_geomDistance(head, nail).
Shows the CLOSEST the hammer collision surface ever gets to the nail during the
real motion, and where (so we can see if it stops short vs misses laterally).
"""

from __future__ import annotations

import numpy as np
import torch
import mujoco

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
    HAMMER_HEAD_SITE_NAME,
    Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.mdp.references import SingleStrikeReference

TASK_ID = "Unitree-Z1-Hammer"


def main():
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
    env.reset()

    m = env.sim.mj_model
    d_cpu = mujoco.MjData(m)
    geoms = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    head_id = next(i for i, g in enumerate(geoms) if g and g.endswith("hammer_head_0"))
    nail_id = next(i for i, g in enumerate(geoms) if g and g.endswith("nail_head"))

    rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
    ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
    robot, nail = env.scene["robot"], env.scene["nail_block"]
    head = lambda: robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
    ntop = lambda: nail.data.site_pos_w[:, ncfg.site_ids].squeeze(1)
    depth = lambda: float(nail.data.joint_pos[0, 0])

    import os
    overshoot = float(os.environ.get("OVERSHOOT", "0.015"))
    nsteps = int(os.environ.get("NSTEPS", "30"))
    ref = SingleStrikeReference(1, env.device, approach_height=0.15, overshoot=overshoot)
    print(f"[cfg] overshoot={overshoot} nsteps={nsteps}")
    ref.update(head(), ntop(), env.episode_length_buf)
    scale = Z1_HAMMER_DELTA_POS_SCALE
    sensor = env.scene["hammer_nail_contact"]

    # qpos addresses to copy warp->cpu
    def sync_cpu():
        qpos = env.sim.data.qpos
        qpos_np = qpos.detach().cpu().numpy()[0] if qpos.ndim == 2 else qpos.detach().cpu().numpy()
        d_cpu.qpos[:] = qpos_np[: m.nq]
        mujoco.mj_forward(m, d_cpu)

    print(f"{'k':>3} {'site_x':>7} {'site_z':>7} {'nail_mm':>7} {'gdist_mm':>8} "
          f"{'hullpt(x,z)':>16} {'contact':>7}")
    best = (1e9, None, None)
    for k in range(1, nsteps + 1):
        step_buf = env.episode_length_buf
        ref.update(head(), ntop(), step_buf)
        n = ref.playback_length()
        target = ref.playback_target(min(int(step_buf.max().item()) + 1, n))
        action = ((target - head()) / scale).clamp(-1.0, 1.0)
        env.step(action)

        sync_cpu()
        ft = np.zeros(6)
        gd = mujoco.mj_geomDistance(m, d_cpu, head_id, nail_id, 1.0, ft)
        sx = float(head()[0, 0]); sz = float(head()[0, 2])
        contact = bool((sensor.data.found > 0).any())
        print(f"{k:>3} {sx:>7.4f} {sz:>7.4f} {depth()*1000:>7.2f} {gd*1000:>8.2f} "
              f"({ft[0]:.3f},{ft[2]:.3f}) {str(contact):>7}")
        if gd < best[0]:
            best = (gd, (ft[0], ft[1], ft[2]), (ft[3], ft[4], ft[5]))
        if int(env.episode_length_buf[0]) == 0:
            print(f"  -- reset after step {k} --")
            break

    print(f"\nCLOSEST approach over strike: {best[0]*1000:.2f} mm")
    print(f"  hull point {np.round(best[1],4)}  nail point {np.round(best[2],4)}")


if __name__ == "__main__":
    main()
