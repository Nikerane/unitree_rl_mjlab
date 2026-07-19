"""B.2 CPU diagnostic: is the ~0.87x i_ref / ~1.4 m/s delivered-impulse ceiling ACTUATOR-limited or
REWARD-limited? Scale ONLY the arm effort_limit (30/60 -> x{1,1.25,1.5}), gains/armature/action/nail
FIXED, drive the SAME scripted reference strike, and measure achieved contact speed + delivered impulse.

- If contact speed + delivered impulse RISE monotonically with effort -> the ceiling is the actuator
  torque clamp (higher torque / VIC is the real lever; reward reshaping never could move it).
- If FLAT -> the ceiling is reward/trajectory/posture, not the actuator.

NOT VIC: gains (kp/kd) are unchanged. Runtime spec-copy only; no repo/asset file edited. CPU, 1 env.
Run: PYTHONPATH=. python docs/results/assets/2026-07-17_fixed_impedance_diag/probes/effort_sweep.py
"""
from __future__ import annotations
import copy, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR, _ENV_SUBSTEP_IMPULSE_ATTR

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def build(effort_scale: float):
    cfg = z1_hammer_env_cfg(play=True, cat_impulse=True); cfg.scene.num_envs = 1
    if effort_scale != 1.0:
        art = cfg.scene.entities["robot"].articulation
        newacts = []
        for a in art.actuators:
            tn = getattr(a, "target_names_expr", ())
            if any(j in tn for j in ARM):  # scale ARM effort only (gripper untouched)
                b = copy.deepcopy(a); b.effort_limit = float(a.effort_limit) * effort_scale
                newacts.append(b)
            else:
                newacts.append(a)
        art.actuators = tuple(newacts)
    return ManagerBasedRlEnv(cfg, device="cpu")


def run(effort_scale: float, speed_factor: float = 1.0):
    env = build(effort_scale)
    robot = env.scene["robot"]; nail = env.scene["nail_block"]
    contact = env.scene["hammer_nail_contact"]; netf = env.scene["hammer_nail_impulse"]
    rc = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rc.resolve(env.scene)
    nc = SceneEntityCfg("nail_block", site_names=("nail_top",)); nc.resolve(env.scene)
    dacc = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR); iacc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)
    axis = torch.tensor([0., 0., -1.]); dt = float(env.physics_dt)
    def head(): return robot.data.site_pos_w[:, rc.site_ids].squeeze(1)
    def ntop(): return nail.data.site_pos_w[:, nc.site_ids].squeeze(1)

    # substep recorder (PRE auto-reset): contact, head_z, peak force, AND the accumulators (the strike
    # auto-resets on success, zeroing them — so snapshot here, not after the loop).
    rec = []; orig = env.metrics_manager.compute_substep
    cap = torch.tensor(IMP_J_LIMIT)
    def patched():
        orig()
        f = float((netf.data.force * axis).sum(-1).sum(-1).clamp_min(0.)[0])
        deliv = float(dacc.delivered[0])
        wr = float((iacc._episode_peak_perjoint[0] / cap).max())
        rec.append((bool((contact.data.found > 0).any()), float(head()[0, 2]), f, deliv, wr))
    env.metrics_manager.compute_substep = patched

    env.reset()
    ref = SingleStrikeReference(1, env.device, approach_height=0.10)
    ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length(); rec.clear()
    for k in range(1, n + 6):
        tgt = ref.playback_target(min(k, n))
        env.step(((tgt - head()) / Z1_HAMMER_DELTA_POS_SCALE * speed_factor).clamp(-1., 1.))
        if int(env.episode_length_buf[0]) == 0: break
    env.metrics_manager.compute_substep = orig

    # achieved contact speed = head down-speed at the substep BEFORE first contact
    fc = next((i for i, r in enumerate(rec) if r[0]), None)
    v_touch = (rec[fc - 1][1] - rec[fc][1]) / dt if fc and fc >= 1 else float("nan")
    peak_f = max((r[2] for r in rec if r[0]), default=0.0)
    delivered = max((r[3] for r in rec), default=0.0)  # peak pre-reset delivered
    wr = max((r[4] for r in rec), default=0.0)
    return dict(effort_scale=effort_scale, v_touch=v_touch, delivered=delivered,
                deliv_x_iref=delivered / 0.6094, peak_F=peak_f, worst_lambda_over_cap=wr)


if __name__ == "__main__":
    # 2D sweep: nominal (speed 1.0) AND aggressive (speed 6.0, where the ~1.4 m/s clamp bit in A1),
    # each at effort x1.0 vs x1.5. The decisive cell: does the AGGRESSIVE v_touch clamp MOVE with effort?
    print(f"{'speed':>6} {'effort':>7} | {'v_touch(m/s)':>12} | {'delivered(N·s)':>14} | {'x i_ref':>8} | {'peakF(N)':>8} | {'wrΛ/cap':>8}")
    for sp in (1.0, 6.0):
        for ef in (1.0, 1.5, 2.0, 3.0):
            r = run(ef, speed_factor=sp)
            print(f"{sp:>6.1f} {ef:>7.2f} | {r['v_touch']:>12.3f} | {r['delivered']:>14.4f} | "
                  f"{r['deliv_x_iref']:>8.3f} | {r['peak_F']:>8.1f} | {r['worst_lambda_over_cap']:>8.3f}")
    print("\n>>> If AGGRESSIVE (speed 6) v_touch RISES from x1.0 to x1.5 effort => the ~1.4 m/s ceiling is")
    print(">>> the EFFORT/torque clamp (actuator-limited; higher torque / VIC is the lever).")
    print(">>> If v_touch stays clamped regardless of effort => the ceiling is DiffIK/kp-kd/action-scale,")
    print(">>> NOT the actuator torque -- relaxing effort would not help, and neither would VIC-via-torque.")
