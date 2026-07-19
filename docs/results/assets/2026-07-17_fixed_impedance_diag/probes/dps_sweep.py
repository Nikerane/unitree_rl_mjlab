"""delta_pos_scale sweep: is the DiffIK controller-bandwidth the lever that raises the fixed-impedance
impact ceiling? B.2 showed effort (torque) is a no-op; the head speed ~ delta_pos_scale/dt, so this is
the actual knob. Sweep delta_pos_scale (0.15 baseline -> 0.3/0.5/1.0), drive a SATURATED straight-down
descent (action clamps at 1 so the controller runs at max bandwidth for that scale), and measure:
  - achieved contact speed v_touch
  - delivered impulse (x i_ref)
  - MAX |joint velocity| at contact vs the real Z1 limit 3.1415 rad/s  <- the sim2real reachability gate

A scale whose strike needs joint velocity > 3.1415 is NOT reachable on the real Z1 (report it, don't
promote it). NOT VIC: gains fixed; this is an action-space bandwidth knob with a real accuracy tradeoff.
Run: PYTHONPATH=. python docs/results/assets/2026-07-17_fixed_impedance_diag/probes/dps_sweep.py
"""
from __future__ import annotations
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR, _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.velocity_bound import Z1_JOINT_VEL_LIMIT

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
I_REF = 0.6094


def run(dps: float):
    cfg = z1_hammer_env_cfg(play=True, cat_impulse=True); cfg.scene.num_envs = 1
    cfg.actions["ik_hammer_head"].delta_pos_scale = float(dps)  # the controller-bandwidth knob
    env = ManagerBasedRlEnv(cfg, device="cpu")
    robot = env.scene["robot"]; nail = env.scene["nail_block"]
    contact = env.scene["hammer_nail_contact"]; netf = env.scene["hammer_nail_impulse"]
    rc = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rc.resolve(env.scene)
    nc = SceneEntityCfg("nail_block", site_names=("nail_top",)); nc.resolve(env.scene)
    arm = SceneEntityCfg("robot", joint_names=ARM); arm.resolve(env.scene)
    dacc = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR); iacc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)
    axis = torch.tensor([0., 0., -1.]); dt = float(env.physics_dt); cap = torch.tensor(IMP_J_LIMIT)
    def head(): return robot.data.site_pos_w[:, rc.site_ids].squeeze(1)
    def ntop(): return nail.data.site_pos_w[:, nc.site_ids].squeeze(1)

    rec = []; orig = env.metrics_manager.compute_substep
    def patched():
        orig()
        f = float((netf.data.force * axis).sum(-1).sum(-1).clamp_min(0.)[0])
        qv = float(robot.data._joint_dof_field("qvel")[0, arm.joint_ids].abs().max())
        rec.append((bool((contact.data.found > 0).any()), float(head()[0, 2]), f,
                    float(dacc.delivered[0]), float((iacc._episode_peak_perjoint[0] / cap).max()), qv))
    env.metrics_manager.compute_substep = patched

    env.reset()
    ref = SingleStrikeReference(1, env.device, approach_height=0.10)
    ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length(); rec.clear()
    for k in range(1, n + 6):
        tgt = ref.playback_target(min(k, n))
        # SATURATE the descent so the controller runs at max bandwidth for this delta_pos_scale
        env.step(((tgt - head()) / 0.05).clamp(-1., 1.))
        if int(env.episode_length_buf[0]) == 0: break
    env.metrics_manager.compute_substep = orig

    fc = next((i for i, r in enumerate(rec) if r[0]), None)
    v_touch = (rec[fc - 1][1] - rec[fc][1]) / dt if fc and fc >= 1 else float("nan")
    peak_f = max((r[2] for r in rec if r[0]), default=0.0)
    delivered = max((r[3] for r in rec), default=0.0)
    wr = max((r[4] for r in rec), default=0.0)
    qv_max = max((r[5] for r in rec), default=0.0)  # max arm joint speed over the strike
    return dict(dps=dps, v_touch=v_touch, delivered=delivered, deliv_x=delivered / I_REF,
                peak_F=peak_f, wr=wr, qv_max=qv_max)


if __name__ == "__main__":
    print(f"{'dps':>6} | {'v_touch(m/s)':>12} | {'deliv x i_ref':>13} | {'peakF(N)':>8} | {'wrΛ/cap':>8} | "
          f"{'max|q̇|(rad/s)':>14} | reachable?")
    for dps in (0.15, 0.16, 0.17, 0.18, 0.20, 0.22, 0.25):
        r = run(dps)
        ok = "YES" if r["qv_max"] <= Z1_JOINT_VEL_LIMIT else f"NO (>{Z1_JOINT_VEL_LIMIT})"
        print(f"{dps:>6.2f} | {r['v_touch']:>12.3f} | {r['deliv_x']:>13.3f} | {r['peak_F']:>8.1f} | "
              f"{r['wr']:>8.3f} | {r['qv_max']:>14.3f} | {ok}")
    print(f"\n>>> If v_touch + delivered RISE with delta_pos_scale => the controller bandwidth IS the lever")
    print(f">>> that raises the fixed-impedance impact ceiling. BUT any scale needing max|q̇| > "
          f"{Z1_JOINT_VEL_LIMIT} rad/s is NOT reachable on the real Z1 -- that is the sim2real ceiling.")
