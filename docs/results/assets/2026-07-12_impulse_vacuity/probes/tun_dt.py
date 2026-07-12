"""TIMESTEP CONVERGENCE test: is the high-velocity impulse collapse genuine numerical
under-resolution (tunneling) or real light-nail physics? Re-run a whip grid at a chosen physics
timestep (keeping control rate 50 Hz: step_dt = timestep*decimation = 0.02). If a FINER timestep
recovers a larger delivered impulse / Lambda at high head velocity, the collapse is tunneling.

One config per process (avoids warp multi-env corruption). CLI: --timestep --decimation."""
from __future__ import annotations
import argparse, json, torch, numpy as np, mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1","joint2","joint3","joint4","joint5","joint6"); NAIL_HEAD_GID = 25
ap = argparse.ArgumentParser()
ap.add_argument("--timestep", type=float, default=0.002)
ap.add_argument("--decimation", type=int, default=10)
ap.add_argument("--whips", type=str, default="6.0,7.5,8.0,9.5,11.0,15.0,24.0")
args = ap.parse_args()
GRID=[float(x) for x in args.whips.split(",")]
HEIGHT=0.10; Z_TOUCH0=0.10944; BAND_TOP=Z_TOUCH0; BAND_BOT=Z_TOUCH0-0.008; V_STOP=0.25
TAG=f"dt{args.timestep}_dec{args.decimation}"
OUT=f"/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/c51273df-43b1-4386-842f-086eab5e35db/scratchpad/tun_dt_{TAG}.json"

cfg=z1_hammer_env_cfg(play=True, cat_impulse=True); cfg.scene.num_envs=1
cfg.sim.mujoco.timestep=args.timestep; cfg.decimation=args.decimation
env=ManagerBasedRlEnv(cfg, device="cpu")
robot=env.scene["robot"]; contact=env.scene["hammer_nail_contact"]; netf=env.scene["hammer_nail_impulse"]
rcfg=SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm=SceneEntityCfg("robot", joint_names=ARM); arm.resolve(env.scene)
ncfg=SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
nsl=SceneEntityCfg("nail_block", joint_names=("nail_slide",)); nsl.resolve(env.scene)
jid=arm.joint_ids; nsid=nsl.joint_ids; axis=torch.tensor([0.,0.,-1.]); dt=float(env.physics_dt)
wpd=env.sim.wp_data; mjm=env.sim.mj_model
nail_bid=mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "nail_block/nail")
NAIL_MASS=float(mjm.body_mass[nail_bid])
print(f"TAG={TAG} physics_dt={dt} step_dt={dt*args.decimation} nail_mass={NAIL_MASS}")

def head(): return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def ntop(): return env.scene["nail_block"].data.site_pos_w[:, ncfg.site_ids].squeeze(1)
def nail_qvel(): return float(env.scene["nail_block"].data.joint_vel[:, nsid].reshape(-1)[0])
def nail_qpos(): return float(env.scene["nail_block"].data.joint_pos[:, nsid].reshape(-1)[0])
def nail_pen_mm():
    na=int(wpd.nacon.numpy().reshape(-1)[0])
    if na==0: return 0.0,0
    dist=wpd.contact.dist.numpy(); geom=wpd.contact.geom.numpy(); best=0.0; cnt=0
    for i in range(min(na,dist.shape[0])):
        g0,g1=int(geom[i,0]),int(geom[i,1])
        if g0==NAIL_HEAD_GID or g1==NAIL_HEAD_GID:
            cnt+=1; pen=-float(dist[i])*1000.0
            if pen>best: best=pen
    return best,cnt

rec=[]; orig=env.metrics_manager.compute_substep
def patched():
    orig()
    pen,nc=nail_pen_mm()
    rec.append({"in_c":bool((contact.data.found>0).any()),
                "qfrc":robot.data._joint_dof_field("qfrc_constraint")[0,jid].clone(),
                "f_ax":float((netf.data.force*axis).sum(-1).sum(-1).clamp_min(0.)[0]),
                "head_z":float(head()[0,2]),"nail_v":nail_qvel(),"nail_q":nail_qpos(),"pen":pen,"nc":nc})
env.metrics_manager.compute_substep=patched

def run(wv):
    env.reset()
    ref=SingleStrikeReference(1, env.device, approach_height=HEIGHT)
    ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
    n=ref.playback_length(); pre=max(1,n-2)
    for k in range(1,pre+1):
        env.step(((ref.playback_target(min(k,n))-head())/Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.,1.))
    qd=robot.data._joint_dof_field("qvel")[0,jid].clone()
    if float(qd.abs().max())<1e-6: qd=torch.ones(6)
    whip=(qd/qd.abs().max())*wv; full=robot.data.joint_vel.clone(); full[0,jid]=whip
    robot.write_joint_velocity_to_sim(full)
    rec.clear()
    for k in range(1,24):
        env.step(((ref.playback_target(n)-head())/Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.,1.))
        if int(env.episode_length_buf[0])==0: break
    return analyze(list(rec),wv)

def analyze(rec,wv):
    hz=[r["head_z"] for r in rec]
    hv=[-(hz[i]-hz[i-1])/dt for i in range(1,len(hz))]
    def ven(i): return hv[i-1] if 1<=i<=len(hv) else float("nan")
    inband=sum(1 for z in hz if BAND_BOT<z<BAND_TOP)
    wins,cur=[],[]
    for i,r in enumerate(rec):
        if r["in_c"]: cur.append(i)
        elif cur: wins.append(cur); cur=[]
    if cur: wins.append(cur)
    lam_fd=torch.zeros(6); J_fd=0.0; v_touch=float("nan"); fd_len=0
    J_all=sum(r["f_ax"]*dt for w in wins for r in [rec[i] for i in w])
    if wins:
        i0=wins[0][0]; v_touch=ven(i0); i=i0
        while i<len(rec) and rec[i]["in_c"]:
            lam_fd+=rec[i]["qfrc"].abs()*dt; J_fd+=rec[i]["f_ax"]*dt; fd_len+=1
            if i+1<len(rec) and ven(i+1)<V_STOP: break
            i+=1
    binds_fd=[float(lam_fd[j])/IMP_J_LIMIT[j] for j in range(6)]
    peak_pen=max((r["pen"] for r in rec),default=0.0)
    peak_nail_v=max((r["nail_v"] for r in rec),default=0.0)
    net_drive=1000.0*(max((r["nail_q"] for r in rec),default=0.0)-rec[0]["nail_q"])
    return {"whip":wv,"v_touch":round(v_touch,3),"inband":inband,"fd_len":fd_len,
            "Lam_fd_max":round(float(lam_fd.max()),4),"worst_fd_cap":round(max(binds_fd),4),
            "J_fd":round(J_fd,5),"J_all":round(J_all,5),"peak_pen_mm":round(peak_pen,2),
            "peak_nail_v":round(peak_nail_v,3),"net_drive_mm":round(net_drive,2),"ep_substeps":len(rec)}

res=[]
print(f"{'whip':>4} {'vtch':>5} {'inband':>6} {'fdlen':>5} {'Lfd':>6} {'fd/cap':>6} {'Jfd':>7} {'Jall':>7} {'pkPen':>6} {'nailV':>6} {'drive':>6} {'eps':>4}")
for wv in GRID:
    r=run(wv); res.append(r)
    print(f"{r['whip']:>4.1f} {r['v_touch']:>5.2f} {r['inband']:>6d} {r['fd_len']:>5d} {r['Lam_fd_max']:>6.3f} "
          f"{r['worst_fd_cap']:>6.2f} {r['J_fd']:>7.4f} {r['J_all']:>7.4f} {r['peak_pen_mm']:>6.2f} "
          f"{r['peak_nail_v']:>6.2f} {r['net_drive_mm']:>6.1f} {r['ep_substeps']:>4d}")
env.metrics_manager.compute_substep=orig
with open(OUT,"w") as fh:
    json.dump({"tag":TAG,"physics_dt":dt,"nail_mass":NAIL_MASS,"results":res}, fh, indent=1)
print("JSON:",OUT)
