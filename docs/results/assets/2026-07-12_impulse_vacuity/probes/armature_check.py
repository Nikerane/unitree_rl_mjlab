"""Verify the Z1 arm dof_armature (rotor inertia) in the model, and test how sensitive the
reflected-mass VIC lever is to it. A harmonic-drive joint's reflected rotor inertia is
I_rotor*N^2 (gear ratio N~100-160), typically 0.05-2 kg*m^2. If the model's dof_armature is
that order, the 1.33x lever is realistic; if it's ~1e-4 (raw rotor only), the model under-models
the transmission coupling and the physical lever could be larger (though still not set_gains-
accessible). Also recompute m_eff with armature scaled x{1,3,10,30} to bound the lever."""
from __future__ import annotations
import json, numpy as np, torch, mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
cfg = z1_hammer_env_cfg(play=True, cat_impulse=True); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg, device="cpu")
robot = env.scene["robot"]; nail_e = env.scene["nail_block"]
rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm = SceneEntityCfg("robot", joint_names=ARM); arm.resolve(env.scene)
ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
jid = arm.joint_ids
def head(): return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def ntop(): return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

env.reset()
ref = SingleStrikeReference(1, env.device, approach_height=0.10)
ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()
for k in range(1, n + 1):
    if head()[0, 2].item() - ntop()[0, 2].item() < 0.02:
        break
    env.step(((ref.playback_target(min(k, n)) - head()) / Z1_HAMMER_DELTA_POS_SCALE * 4.0).clamp(-1., 1.))
q_arm = robot.data.joint_pos[0, jid].detach().cpu().numpy()

m = env.sim.mj_model
jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
snames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
arm_jnt = [next(nm for nm in jnames if nm and nm.endswith(f"joint{k}")) for k in range(1, 7)]
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in arm_jnt]
dadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in arm_jnt]
hs = next(s for s in snames if s and s.endswith(HAMMER_HEAD_SITE_NAME))
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, hs)
u = np.array([0., 0., -1.])

armature = [float(m.dof_armature[da]) for da in dadr]
# diagonal of M at this pose (per-joint effective inertia incl armature) for context
d0 = mujoco.MjData(m)
for a, q in zip(qadr, q_arm): d0.qpos[a] = float(q)
mujoco.mj_forward(m, d0)
M0 = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M0, d0.qM)
Mdiag = [float(M0[da, da]) for da in dadr]

def meff_with(scale):
    saved = m.dof_armature.copy()
    for da in dadr: m.dof_armature[da] = armature[dadr.index(da)] * scale if False else float(m.dof_armature[da])
    # apply scale relative to ORIGINAL
    for i, da in enumerate(dadr): m.dof_armature[da] = armature[i] * scale
    d = mujoco.MjData(m)
    for a, q in zip(qadr, q_arm): d.qpos[a] = float(q)
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, d.qM)
    Jp = np.zeros((3, m.nv)); Jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, Jp, Jr, sid)
    meff = 1.0 / float(u @ (Jp @ np.linalg.inv(M) @ Jp.T) @ u)
    m.dof_armature[:] = saved
    return round(meff, 3)

print(json.dumps({
    "arm_joint_names": arm_jnt,
    "dof_armature_kg_m2": [round(a, 5) for a in armature],
    "M_diag_effective_inertia_kg_m2": [round(x, 4) for x in Mdiag],
    "armature_fraction_of_diag": [round(armature[i] / Mdiag[i], 3) if Mdiag[i] > 0 else None for i in range(6)],
    "m_eff_vs_armature_scale": {
        "x0 (link only)": meff_with(0.0),
        "x1 (as modeled)": meff_with(1.0),
        "x3": meff_with(3.0),
        "x10": meff_with(10.0),
        "x30": meff_with(30.0),
        "x100": meff_with(100.0),
    },
    "note": "If armature is ~1e-4 (raw rotor, no N^2) the model under-models transmission coupling; "
            "a realistic gear-amplified value (0.05-2) would raise m_eff and the lever. But set_gains "
            "changes kp, NOT dof_armature, so VIC cannot access ANY of these columns in sim.",
}, indent=1))
