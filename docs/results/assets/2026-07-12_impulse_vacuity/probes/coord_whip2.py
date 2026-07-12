"""COORDINATED velocity-limited whip probe v2 — robust head-velocity capture + Jz reporting.
See coord_whip.py header. Improvements: (1) don't clear rec (keep full trajectory + inject index),
(2) report head velocity from pre-contact head_z samples AND predicted from Jz@qvel_at_contact,
(3) report Λ raw + baseline-subtracted (shipped semantics), delivered P, window length."""
from __future__ import annotations
import argparse, json, copy, numpy as np, torch, mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM=("joint1","joint2","joint3","joint4","joint5","joint6")
ap=argparse.ArgumentParser()
ap.add_argument("--vlim",type=float,default=4.65)
ap.add_argument("--scale",type=float,default=1.0)
ap.add_argument("--strategy",choices=["box","natural"],default="box")
ap.add_argument("--height",type=float,default=0.10)
ap.add_argument("--standoff",type=float,default=0.014)
ap.add_argument("--drive",choices=["through","ballistic","hold"],default="through")
ap.add_argument("--kp",type=float,default=0.0); ap.add_argument("--kd",type=float,default=0.0)
args=ap.parse_args()

cfg=z1_hammer_env_cfg(play=True,cat_impulse=True); cfg.scene.num_envs=1
if args.kp>0:
    art=cfg.scene.entities["robot"].articulation; na=[]
    for a in art.actuators:
        tn=getattr(a,"target_names_expr",())
        if any(j in tn for j in ARM):
            b=copy.deepcopy(a); b.stiffness=args.kp; b.damping=args.kd; na.append(b)
        else: na.append(a)
    art.actuators=tuple(na)
env=ManagerBasedRlEnv(cfg,device="cpu")
robot=env.scene["robot"]; contact=env.scene["hammer_nail_contact"]; netf=env.scene["hammer_nail_impulse"]
rcfg=SceneEntityCfg("robot",site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm=SceneEntityCfg("robot",joint_names=ARM); arm.resolve(env.scene)
ncfg=SceneEntityCfg("nail_block",site_names=("nail_top",)); ncfg.resolve(env.scene)
jid=arm.joint_ids; axis=torch.tensor([0.,0.,-1.]); dt=float(env.physics_dt)
def head(): return robot.data.site_pos_w[:,rcfg.site_ids].squeeze(1)
def ntop(): return env.scene["nail_block"].data.site_pos_w[:,ncfg.site_ids].squeeze(1)
mjm=env.sim.mj_model; hsid=mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_SITE,"robot/"+HAMMER_HEAD_SITE_NAME); nv=mjm.nv
def jac_and_M():
    qp=np.asarray(env.sim.data.qpos.numpy())[0].astype(np.float64)
    md=mujoco.MjData(mjm); md.qpos[:]=qp; mujoco.mj_forward(mjm,md)
    jp=np.zeros((3,nv)); jr=np.zeros((3,nv)); mujoco.mj_jacSite(mjm,md,jp,jr,hsid)
    Md=np.zeros((nv,nv)); mujoco.mj_fullM(mjm,Md,md.qM); return jp,Md

rec=[]; orig=env.metrics_manager.compute_substep
def patched():
    orig()
    rec.append({"in_c":bool((contact.data.found>0).any()),
                "qfrc":robot.data._joint_dof_field("qfrc_constraint")[0,jid].clone(),
                "qvel":robot.data._joint_dof_field("qvel")[0,jid].clone(),
                "f_ax":float((netf.data.force*axis).sum(-1).sum(-1).clamp_min(0.)[0]),
                "head_z":float(head()[0,2])})
env.metrics_manager.compute_substep=patched

env.reset()
ref=SingleStrikeReference(1,env.device,approach_height=args.height)
ref.update(head(),ntop(),torch.zeros(1,dtype=torch.long,device=env.device))
n=ref.playback_length(); contact_z=float(ntop()[0,2])
inject_done=False; inj_idx=None; Jz=None; qd=None; vhead_pred=None; m_arm=m_full=float('nan'); inj_head=None
k=0
while k<n+30:
    k+=1
    if (not inject_done) and (float(head()[0,2])-contact_z)<=args.standoff:
        jp,Md=jac_and_M(); Jz=jp[2,:].copy()
        if args.strategy=="box":
            qd=np.zeros(nv)
            for j in range(6): qd[j]=-np.sign(Jz[j])*args.vlim
        else:
            cur=robot.data._joint_dof_field("qvel")[0,jid].clone().numpy()
            if np.abs(cur).max()<1e-6: cur=-np.sign(Jz[:6])
            d=np.zeros(nv); d[:6]=cur; qd=d/np.abs(d[:6]).max()*args.vlim
        qd=qd*args.scale; vhead_pred=float(Jz@qd)
        try:
            m_full=1.0/float(Jz@np.linalg.inv(Md)@Jz)
            Ja=jp[2,:6]; m_arm=1.0/float(Ja@np.linalg.inv(Md[:6,:6])@Ja)
        except Exception: pass
        full=robot.data.joint_vel.clone(); full[0,jid]=torch.tensor(qd[:6],dtype=full.dtype)
        robot.write_joint_velocity_to_sim(full)
        inj_head=head().clone(); inject_done=True; inj_idx=len(rec); break
    env.step(((ref.playback_target(min(k,n))-head())/Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.,1.))
    if int(env.episode_length_buf[0])==0: break
if not inject_done:
    print(json.dumps({"error":"never reached standoff"})); raise SystemExit
for step in range(1,18):
    if args.drive=="through": target=ref.playback_target(n)
    elif args.drive=="hold": target=inj_head
    else: target=head().clone()
    env.step(((target-head())/Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.,1.))
    if int(env.episode_length_buf[0])==0: break

# find first contact window at/after injection
wins,cur=[],[]
for i in range(len(rec)):
    if rec[i]["in_c"]: cur.append(i)
    elif cur: wins.append(cur); cur=[]
if cur: wins.append(cur)
wins=[w for w in wins if w[-1]>=inj_idx-1]  # windows after injection
out={"strategy":args.strategy,"scale":args.scale,"drive":args.drive,"kp":args.kp or 1000,
     "vhead_pred_ceiling_down":round(-vhead_pred,3),"m_eff_arm":round(m_arm,3),"m_eff_full":round(m_full,3),
     "qd_box":[round(float(x),3) for x in qd[:6]]}
if not wins:
    out.update({"n_windows":0,"note":"no contact after injection"}); print(json.dumps(out)); raise SystemExit
i0=wins[0][0]
# head velocity from pre-contact head_z samples (up to 3)
pre=[rec[i]["head_z"] for i in range(max(0,i0-3),i0+1)]
hv=[-(pre[t]-pre[t-1])/dt for t in range(1,len(pre))]
v_touch=hv[-1] if hv else float('nan')
qv_c=rec[i0]["qvel"].abs().tolist()
vhead_from_qvel=float(Jz@np.array(list(rec[i0]["qvel"].numpy())+[0.0]*(nv-6)))  # actual head vz at contact
L_raw=torch.zeros(6); L_sub=torch.zeros(6)
for w in wins:
    b=rec[w[0]-1]["qfrc"].clone() if w[0]>=1 else torch.zeros(6)
    for i in w:
        L_raw+=rec[i]["qfrc"].abs()*dt; L_sub+=(rec[i]["qfrc"]-b).abs()*dt
P=0.0
for w in wins:
    for cnt,i in enumerate(w):
        if cnt<25: P+=rec[i]["f_ax"]*dt
binds_raw=[float(L_raw[j])/IMP_J_LIMIT[j] for j in range(6)]
binds_sub=[float(L_sub[j])/IMP_J_LIMIT[j] for j in range(6)]
out.update({"n_windows":len(wins),"win_lens":[len(w) for w in wins],"win0_ms":round(len(wins[0])*dt*1e3,1),
 "head_v_at_contact_measured":round(v_touch,3),"head_v_at_contact_fromJq":round(-vhead_from_qvel,3),
 "qvel_at_contact":[round(x,3) for x in qv_c],"qvel_max_at_contact":round(max(qv_c),3),
 "delivered_P_Ns":round(P,4),
 "Lambda_raw_max":round(float(L_raw.max()),3),"Lambda_sub_max":round(float(L_sub.max()),3),
 "Lambda_sub_perjoint":[round(float(x),3) for x in L_sub],
 "worst_over_cap_RAW":round(max(binds_raw),3),"worst_over_cap_SUB":round(max(binds_sub),3)})
print(json.dumps(out))
