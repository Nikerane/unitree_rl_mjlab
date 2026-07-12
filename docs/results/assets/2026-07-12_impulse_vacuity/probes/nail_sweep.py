"""BINDABLE-THRESHOLD sweep: at what nail mass / friction does a task-completing strike push the
robot-side Λ over the cap? Finds a REALISTIC task variant (heavier/harder target) that makes the
impulse constraint BIND, without touching the caps. One env/process. Read-only (spec wrapped
pre-compile, no repo/asset edit)."""
from __future__ import annotations
import argparse, json, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1","joint2","joint3","joint4","joint5","joint6")
NAIL_GEOMS = ("nail_shaft", "nail_head")
ap = argparse.ArgumentParser()
ap.add_argument("--mass_scale", type=float, default=1.0)   # x current nail mass (~7 g)
ap.add_argument("--friction", type=float, default=30.0)    # nail_slide frictionloss (N); default 30
ap.add_argument("--speed", type=float, default=4.0)
ap.add_argument("--hold", type=int, default=25)            # follow-through so the whole event is seen
args = ap.parse_args()

cfg = z1_hammer_env_cfg(play=True, cat_impulse=True); cfg.scene.num_envs = 1

# wrap nail_block spec pre-compile: scale nail geom masses + set nail_slide frictionloss
nb = cfg.scene.entities["nail_block"]
orig_fn = nb.spec_fn
def patched_spec():
    spec = orig_fn()
    for g in spec.geoms:
        if g.name in NAIL_GEOMS:
            try: g.mass = float(g.mass) * args.mass_scale
            except Exception: pass
    for j in spec.joints:
        if j.name == "nail_slide":
            try: j.frictionloss = float(args.friction)
            except Exception: pass
    return spec
nb.spec_fn = patched_spec

env = ManagerBasedRlEnv(cfg, device="cpu")
robot = env.scene["robot"]; nail_e = env.scene["nail_block"]
contact = env.scene["hammer_nail_contact"]; netf = env.scene["hammer_nail_impulse"]
rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm = SceneEntityCfg("robot", joint_names=ARM); arm.resolve(env.scene)
ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
nailj = SceneEntityCfg("nail_block", joint_names=("nail_slide",)); nailj.resolve(env.scene)
jid = arm.joint_ids; njid = nailj.joint_ids; axis = torch.tensor([0.,0.,-1.]); dt = float(env.physics_dt)
def head(): return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def ntop(): return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)
def nail_q(): return float(nail_e.data.joint_pos[0, njid].reshape(-1)[0])

rec = []; orig = env.metrics_manager.compute_substep
def patched():
    orig()
    rec.append({"in_c": bool((contact.data.found>0).any()),
                "qfrc": robot.data._joint_dof_field("qfrc_constraint")[0,jid].clone(),
                "f_ax": float((netf.data.force*axis).sum(-1).sum(-1).clamp_min(0.)[0]),
                "head_z": float(head()[0,2]), "nail_q": nail_q()})
env.metrics_manager.compute_substep = patched

env.reset()
q0 = nail_q()
ref = SingleStrikeReference(1, env.device, approach_height=0.10)
ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()
for k in range(1, n + args.hold + 1):
    target = ref.playback_target(min(k, n))
    env.step(((target - head()) / Z1_HAMMER_DELTA_POS_SCALE * args.speed).clamp(-1., 1.))
    if int(env.episode_length_buf[0]) == 0: break

# contact windows + per-event Λ (worst joint, per-event max like the shipped accumulator)
wins, cur = [], []
for i, r in enumerate(rec):
    if r["in_c"]: cur.append(i)
    elif cur: wins.append(cur); cur = []
if cur: wins.append(cur)
per_event_max = torch.zeros(6)   # max over windows of that window's rectified Λ (per-event semantics)
for w in wins:
    wl = torch.zeros(6)
    for i in w: wl += rec[i]["qfrc"].abs() * dt
    per_event_max = torch.maximum(per_event_max, wl)
whole = torch.zeros(6)
for r in rec:
    if r["in_c"]: whole += r["qfrc"].abs() * dt
binds_event = [float(per_event_max[j]) / IMP_J_LIMIT[j] for j in range(6)]
binds_whole = [float(whole[j]) / IMP_J_LIMIT[j] for j in range(6)]
nail_driven = nail_q() - q0
peak_f = max((r["f_ax"] for r in rec if r["in_c"]), default=0.0)
print(json.dumps({
    "mass_scale": args.mass_scale, "friction_N": args.friction, "speed": args.speed,
    "n_windows": len(wins), "win_lens": [len(w) for w in wins],
    "peak_contact_F": round(peak_f, 1),
    "nail_driven_mm": round(nail_driven * 1e3, 2), "success_>=27mm": bool(nail_driven >= 0.027),
    "worst_Lambda_over_cap_PEREVENT": round(max(binds_event), 3),
    "worst_Lambda_over_cap_WHOLE": round(max(binds_whole), 3),
    "Lambda_perevent_j2j3j4": [round(float(per_event_max[1]),3), round(float(per_event_max[2]),3), round(float(per_event_max[3]),3)],
    "BINDS_perevent": bool(max(binds_event) >= 1.0),
}))
