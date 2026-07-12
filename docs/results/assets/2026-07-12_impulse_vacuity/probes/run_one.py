"""ONE env, ONE strike -> discriminators + Λ/cap. Parametrized; call once per config (fresh process
to avoid multi-env warp-state corruption). Args: --kp --kd --speed --height."""
from __future__ import annotations
import argparse, json, copy, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1","joint2","joint3","joint4","joint5","joint6")
ap = argparse.ArgumentParser()
ap.add_argument("--kp", type=float, default=0)   # 0 = default gains
ap.add_argument("--kd", type=float, default=0)
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--height", type=float, default=0.10)
ap.add_argument("--hold", type=int, default=10)
args = ap.parse_args()

cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
cfg.scene.num_envs = 1
if args.kp > 0:
    art = cfg.scene.entities["robot"].articulation
    na = []
    for a in art.actuators:
        tn = getattr(a, "target_names_expr", ())
        if any(j in tn for j in ARM):
            b = copy.deepcopy(a); b.stiffness = args.kp; b.damping = args.kd; na.append(b)
        else:
            na.append(a)
    art.actuators = tuple(na)
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
for k in range(1, n+args.hold+1):
    target = ref.playback_target(min(k,n))
    env.step(((target-head())/Z1_HAMMER_DELTA_POS_SCALE*args.speed).clamp(-1.,1.))
    if int(env.episode_length_buf[0])==0: break

wins,cur=[],[]
for i,r in enumerate(rec):
    if r["in_c"]: cur.append(i)
    elif cur: wins.append(cur); cur=[]
if cur: wins.append(cur)
tag=f"kp={args.kp or 1000:.0f}/kd={args.kd or 100:.0f} sf={args.speed}"
if not wins:
    print(json.dumps({"tag":tag,"n_windows":0,"v_touch":None})); raise SystemExit
w0=wins[0]; i0,i1=w0[0],w0[-1]
fs=[rec[i]["f_ax"] for i in w0]; peak=max(fs); mean=sum(fs)/len(fs)
rect_all=torch.zeros(6)
for w in wins:
    for i in w: rect_all+=rec[i]["qfrc"].abs()*dt
vt=(rec[i0-1]["head_z"]-rec[i0]["head_z"])/dt if i0>=1 else float("nan")
hz=[rec[i]["head_z"] for i in ([i0-1] if i0>=1 else [])+w0]
hv=[-(hz[k]-hz[k-1])/dt for k in range(1,len(hz))]
vdecay=(hv[0]-hv[-1])/hv[0] if hv and abs(hv[0])>1e-9 else float("nan")
vmin=min(hv) if hv else float("nan")
binds=[float(rect_all[j])/IMP_J_LIMIT[j] for j in range(6)]
o={"tag":tag,"n_windows":len(wins),"lens":[len(w) for w in wins],"win0_ms":len(w0)*dt*1e3,
   "v_touch":vt,"peak_over_mean":peak/max(mean,1e-9),"vdecay":vdecay,"v_min":vmin,"rebound":bool(vmin<-0.02),
   "Lambda_all_max":float(rect_all.max()),"worst_Lambda_over_cap":max(binds)}
print(json.dumps(o))
