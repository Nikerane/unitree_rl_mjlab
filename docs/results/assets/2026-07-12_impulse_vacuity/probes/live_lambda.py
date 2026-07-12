"""Poll the SHIPPED accumulators live during the box-optimal coordinated strike:
 - SubstepImpulseAccumulator (baseline-subtracted qfrc Λ, the ENFORCED quantity)
 - ContactRowImpulseAccumulator (JᵀF contact-row-only Λ, friction-immune ground truth)
captured at their in-episode peak (before completion reset zeroes them)."""
from __future__ import annotations
import argparse, numpy as np, torch, mujoco, json
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference
ap=argparse.ArgumentParser(); ap.add_argument("--scale",type=float,default=1.0); ap.add_argument("--vlim",type=float,default=4.65)
ap.add_argument("--standoff",type=float,default=0.014); args=ap.parse_args()
ARM=("joint1","joint2","joint3","joint4","joint5","joint6")
cfg=z1_hammer_env_cfg(play=True,cat_impulse=True); cfg.scene.num_envs=1
env=ManagerBasedRlEnv(cfg,device="cpu")
robot=env.scene["robot"]; contact=env.scene["hammer_nail_contact"]; netf=env.scene["hammer_nail_impulse"]
rcfg=SceneEntityCfg("robot",site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm=SceneEntityCfg("robot",joint_names=ARM); arm.resolve(env.scene)
ncfg=SceneEntityCfg("nail_block",site_names=("nail_top",)); ncfg.resolve(env.scene)
jid=arm.joint_ids; dt=float(env.physics_dt)
def head(): return robot.data.site_pos_w[:,rcfg.site_ids].squeeze(1)
def ntop(): return env.scene["nail_block"].data.site_pos_w[:,ncfg.site_ids].squeeze(1)
mjm=env.sim.mj_model; hsid=mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_SITE,"robot/"+HAMMER_HEAD_SITE_NAME); nv=mjm.nv
acc=None; rowacc=None
peak_sub=torch.zeros(6); peak_row=torch.zeros(1)
orig=env.metrics_manager.compute_substep
def patched():
    orig()
    global acc,rowacc,peak_sub,peak_row
    acc=getattr(env,"_hammer_substep_impulse",None); rowacc=getattr(env,"_hammer_substep_impulse_rows",None)
    if acc is not None:
        cur=torch.maximum(acc._pulse,acc._running)[0]  # live per-joint Λ (baseline-subtracted)
        peak_sub=torch.maximum(peak_sub,cur)
    if rowacc is not None:
        peak_row=torch.maximum(peak_row,torch.maximum(rowacc._pulse,rowacc._running)[0].amax().reshape(1))
env.metrics_manager.compute_substep=patched
env.reset()
ref=SingleStrikeReference(1,env.device,approach_height=0.10)
ref.update(head(),ntop(),torch.zeros(1,dtype=torch.long,device=env.device))
n=ref.playback_length(); contact_z=float(ntop()[0,2]); inj=False; k=0
while k<n+30:
    k+=1
    if (not inj) and (float(head()[0,2])-contact_z)<=args.standoff:
        qp=np.asarray(env.sim.data.qpos.numpy())[0].astype(np.float64)
        md=mujoco.MjData(mjm); md.qpos[:]=qp; mujoco.mj_forward(mjm,md)
        jp=np.zeros((3,nv)); jr=np.zeros((3,nv)); mujoco.mj_jacSite(mjm,md,jp,jr,hsid); Jz=jp[2,:]
        qd=np.zeros(nv)
        for j in range(6): qd[j]=-np.sign(Jz[j])*args.vlim*args.scale
        full=robot.data.joint_vel.clone(); full[0,jid]=torch.tensor(qd[:6],dtype=full.dtype)
        robot.write_joint_velocity_to_sim(full); inj=True; break
    env.step(((ref.playback_target(min(k,n))-head())/Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.,1.))
    if int(env.episode_length_buf[0])==0: break
for step in range(1,18):
    env.step(((ref.playback_target(n)-head())/Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.,1.))
    if int(env.episode_length_buf[0])==0: break
row_over=float(peak_row[0])/1.64
print(json.dumps({"scale":args.scale,
  "ENFORCED_Lambda_sub_perjoint":[round(float(x),3) for x in peak_sub],
  "ENFORCED_worst_over_cap":round(max(float(peak_sub[j])/IMP_J_LIMIT[j] for j in range(6)),3),
  "contact_ROW_Lambda_worstjoint":round(float(peak_row[0]),3),
  "contact_ROW_over_cap(vs1.64)":round(row_over,3),
  "caps":list(IMP_J_LIMIT)}))
