"""WHIP probe: arrive at contact at high joint velocity (bypasses 'can the policy accelerate?'),
measure the resulting per-joint Λ vs cap. Sweeps whip velocity magnitude to find where (if ever)
the constraint binds. Also computes an analytic momentum bound. One env/process."""
from __future__ import annotations
import argparse, json, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1","joint2","joint3","joint4","joint5","joint6")
ap = argparse.ArgumentParser()
ap.add_argument("--whip_vel", type=float, default=4.65)  # rad/s magnitude of fastest joint at contact
ap.add_argument("--height", type=float, default=0.10)
args = ap.parse_args()

cfg = z1_hammer_env_cfg(play=True, cat_impulse=True); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg, device="cpu")
robot = env.scene["robot"]; contact = env.scene["hammer_nail_contact"]; netf = env.scene["hammer_nail_impulse"]
rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm = SceneEntityCfg("robot", joint_names=ARM); arm.resolve(env.scene)
ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
jid = arm.joint_ids; axis = torch.tensor([0.,0.,-1.]); dt = float(env.physics_dt)
def head(): return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def ntop(): return env.scene["nail_block"].data.site_pos_w[:, ncfg.site_ids].squeeze(1)

rec = []; orig = env.metrics_manager.compute_substep
def patched():
    orig()
    rec.append({"in_c": bool((contact.data.found>0).any()),
                "qfrc": robot.data._joint_dof_field("qfrc_constraint")[0,jid].clone(),
                "qvel": robot.data._joint_dof_field("qvel")[0,jid].clone(),
                "f_ax": float((netf.data.force*axis).sum(-1).sum(-1).clamp_min(0.)[0]),
                "head_z": float(head()[0,2])})
env.metrics_manager.compute_substep = patched

env.reset()
ref = SingleStrikeReference(1, env.device, approach_height=args.height)
ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()
# playback to a couple steps before the natural contact onset (arm poised above the nail, descending)
pre = max(1, n - 2)
for k in range(1, pre + 1):
    target = ref.playback_target(min(k, n))
    env.step(((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1., 1.))
# natural descent direction in joint space, then override to whip magnitude
qd = robot.data._joint_dof_field("qvel")[0, jid].clone()
if float(qd.abs().max()) < 1e-6:
    qd = torch.ones(6)  # fallback direction
qd_dir = qd / qd.abs().max()                  # normalize so fastest joint = 1
whip = (qd_dir * args.whip_vel)               # (6,) target joint velocities
full = robot.data.joint_vel.clone()
full[0, jid] = whip
robot.write_joint_velocity_to_sim(full)
head_v_before = None
rec.clear()
# free-fly into the nail: keep commanding the final descent target so the arm keeps going down
for k in range(1, 20):
    target = ref.playback_target(n)
    env.step(((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1., 1.))
    if int(env.episode_length_buf[0]) == 0: break

wins, cur = [], []
for i, r in enumerate(rec):
    if r["in_c"]: cur.append(i)
    elif cur: wins.append(cur); cur = []
if cur: wins.append(cur)
if not wins:
    print(json.dumps({"whip_vel": args.whip_vel, "n_windows": 0, "note": "no contact"})); raise SystemExit
rect_all = torch.zeros(6)
for w in wins:
    for i in w: rect_all += rec[i]["qfrc"].abs() * dt
# head velocity at contact onset
i0 = wins[0][0]
v_touch = (rec[i0-1]["head_z"] - rec[i0]["head_z"]) / dt if i0 >= 1 else float("nan")
binds = [float(rect_all[j]) / IMP_J_LIMIT[j] for j in range(6)]
print(json.dumps({
    "whip_vel_radps": args.whip_vel,
    "actual_qvel_max_at_override": float(whip.abs().max()),
    "head_v_at_contact": v_touch,
    "n_windows": len(wins), "lens": [len(w) for w in wins],
    "Lambda_perjoint": [round(float(x), 3) for x in rect_all],
    "Lambda_max": round(float(rect_all.max()), 3),
    "worst_Lambda_over_cap": round(max(binds), 3),
    "caps": list(IMP_J_LIMIT),
}))
