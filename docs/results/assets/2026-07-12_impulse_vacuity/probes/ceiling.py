"""VIC-FEASIBILITY CEILING (analytic go/no-go for the whole VIC pivot).

Bounds the MAXIMUM worst-joint reaction impulse Λ_j any strike could produce on this platform,
and asks whether VIC's stiffness lever can move it. Physics:

  A Cartesian impact impulse P = m_eff·v·(1+e) along u=-ẑ maps to joint torque-impulse τ = Jᵀ P,
  so  Λ_j = |(Jᵀu)_j| · m_eff · v · (1+e),  with
    m_eff = 1 / (uᵀ J M(q)⁻¹ Jᵀ u)        (Khatib operational-space effective mass)
    e = restitution (e=1 elastic = generous upper bound; e=0 inelastic)

Two levers a VIC policy could pull, tested here:
  (1) reflected mass m_eff — in MuJoCo this is set by M(q) which INCLUDES joint armature (the
      rotor inertia the harmonic drive couples). We compute m_eff WITH armature (≈ a rigidly-
      coupled / high-command-stiffness arm) and WITHOUT (≈ link-only / fully compliant). The
      Kirschner m_r(K_J) range. KEY: commanded PD kp (set_gains) does NOT change dof_armature, so
      VIC's stiffness command cannot move m_eff between these in sim — it only adds active press.
  (2) velocity v — effort-clamped; set_gains cannot lift τ_rated. We report the crossing velocity
      v* = cap / (|Jᵀu|·m_eff·(1+e)) and compare to the reachable (~1.3 m/s) and the KINEMATIC
      ceiling (all arm joints at the URDF velocity limit).

If v* (elastic, armature-coupled m_eff) exceeds the kinematic ceiling, the cap is unreachable by
ANY controller including VIC → VIC is dead-on-arrival for making the impulse constraint bind."""
from __future__ import annotations
import json, numpy as np, torch, mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.mdp.velocity_bound import Z1_JOINT_VEL_LIMIT

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

# --- drive to the pre-contact strike pose, snapshot arm angles ---
env.reset()
ref = SingleStrikeReference(1, env.device, approach_height=0.10)
ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()
for k in range(1, n + 1):
    if head()[0, 2].item() - ntop()[0, 2].item() < 0.02:
        break
    tgt = ref.playback_target(min(k, n))
    env.step(((tgt - head()) / Z1_HAMMER_DELTA_POS_SCALE * 4.0).clamp(-1., 1.))
q_arm = robot.data.joint_pos[0, jid].detach().cpu().numpy()

# --- CPU mujoco: mass matrix + head-site Jacobian at that pose ---
m = env.sim.mj_model
jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
snames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
arm_jnt = [next(nm for nm in jnames if nm and nm.endswith(f"joint{k}")) for k in range(1, 7)]
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in arm_jnt]
dadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in arm_jnt]
hs = next(s for s in snames if s and s.endswith(HAMMER_HEAD_SITE_NAME))
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, hs)
u = np.array([0., 0., -1.])
VELLIM = float(Z1_JOINT_VEL_LIMIT)

def compute(zero_arm_armature: bool):
    mm = env.sim.mj_model  # fresh handle; restore after
    saved = mm.dof_armature.copy()
    if zero_arm_armature:
        for da in dadr:
            mm.dof_armature[da] = 0.0
    d = mujoco.MjData(mm)
    for a, q in zip(qadr, q_arm):
        d.qpos[a] = float(q)
    mujoco.mj_forward(mm, d)
    M = np.zeros((mm.nv, mm.nv)); mujoco.mj_fullM(mm, M, d.qM)
    Jp = np.zeros((3, mm.nv)); Jr = np.zeros((3, mm.nv))
    mujoco.mj_jacSite(mm, d, Jp, Jr, sid)
    Minv = np.linalg.inv(M)
    m_eff = 1.0 / float(u @ (Jp @ Minv @ Jp.T) @ u)
    JTu = Jp.T @ u                      # (nv,)  joint torque per unit EE force along u
    jtu_arm = np.array([JTu[da] for da in dadr])
    jz_arm = np.array([Jp[2, da] for da in dadr])   # head z-velocity per unit joint velocity
    mm.dof_armature[:] = saved
    return m_eff, jtu_arm, jz_arm

m_eff_rigid, jtu, jz = compute(zero_arm_armature=False)   # armature-coupled (≈ VIC-max stiffness)
m_eff_link, _, _ = compute(zero_arm_armature=True)         # link-only (≈ fully compliant)

cap = np.array(IMP_J_LIMIT)
v_kin = float(np.sum(np.abs(jz)) * VELLIM)          # kinematic ceiling head speed (all joints max, aligned)
V_REACH = 1.35                                       # measured effort-limited reachable head speed

def over_cap(v, e, meff):
    lam = np.abs(jtu) * meff * v * (1 + e)
    return lam / cap

def crossing_v(e, meff):
    # per-joint velocity to reach cap; min over joints = easiest crossing
    vstar = cap / (np.abs(jtu) * meff * (1 + e))
    j = int(np.argmin(vstar))
    return float(vstar[j]), j + 1

out = {
    "strike_pose_arm_deg": [round(float(np.degrees(q)), 1) for q in q_arm],
    "m_eff_rigid_kg_armature_coupled": round(m_eff_rigid, 3),
    "m_eff_link_only_kg_zero_armature": round(m_eff_link, 3),
    "armature_lever_ratio": round(m_eff_rigid / max(m_eff_link, 1e-9), 2),
    "|JT.u|_perjoint_moment_arm_m": [round(float(x), 3) for x in jtu],
    "kinematic_ceiling_headspeed_mps": round(v_kin, 2),
    "reachable_headspeed_mps": V_REACH,
    "CAP": list(IMP_J_LIMIT),
}
for e in (0.0, 1.0):
    tag = "inelastic_e0" if e == 0 else "elastic_e1"
    oc_reach = over_cap(V_REACH, e, m_eff_rigid)
    oc_kin = over_cap(v_kin, e, m_eff_rigid)
    vstar, jstar = crossing_v(e, m_eff_rigid)
    out[tag] = {
        "worst_over_cap_at_reachable_1.35mps": round(float(oc_reach.max()), 3),
        "worst_over_cap_at_kinematic_ceiling": round(float(oc_kin.max()), 3),
        "crossing_velocity_mps": round(vstar, 2),
        "crossing_joint": jstar,
        "crossing_v_ABOVE_kinematic_ceiling": bool(vstar > v_kin),
    }
out["VERDICT_cap_reachable_by_any_strike"] = bool(
    over_cap(v_kin, 1.0, m_eff_rigid).max() >= 1.0)
out["VIC_stiffness_moves_m_eff_in_sim"] = "NO — dof_armature is fixed; set_gains changes actuator kp, not M(q)"
print(json.dumps(out, indent=1))
