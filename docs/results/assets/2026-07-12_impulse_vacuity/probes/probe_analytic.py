"""ANALYTIC upper bound on per-joint impact-impulse Lambda_j.

First-principles: a single-point frictionless normal contact that inelastically stops the
descending head delivers a Cartesian impulse P = m_eff * v_n along the nail normal n, where
m_eff = 1/(n^T J M^{-1} J^T n) is the operational-space (reflected) inertia along n. The
joint-space impulse is g*P with g = J^T n (6-vec). Per-joint |Lambda_j| = |g_j| * P.

We evaluate M (mj_fullM) and J (mj_jacSite) at the ACTUAL contact configuration (captured from a
natural scripted strike) AND at the NEAR_NAIL reset pose, on the exact compiled mjlab model
(env.sim.mj_model, so armature/gravcomp/etc. are the real ones). We then report:
  - m_eff along n=[0,0,1]
  - g_j = (J^T n)_j  and ||g||_1
  - v_n at the joint velocity limits: URDF 3.1415 and observed 4.65 rad/s
      * v_n_max  = v_lim * ||g||_1              (all joints aligned to maximize head speed)
      * also the natural observed v_n at contact (cross-check)
  - Lambda_j = |g_j| * m_eff * v_n   and Lambda_j / cap
One env / one process (warp-safe).
"""
from __future__ import annotations
import json, numpy as np, torch, mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1","joint2","joint3","joint4","joint5","joint6")
CAPS = np.array(IMP_J_LIMIT)
np.set_printoptions(precision=4, suppress=True)

cfg = z1_hammer_env_cfg(play=True, cat_impulse=True); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg, device="cpu")
robot = env.scene["robot"]; contact = env.scene["hammer_nail_contact"]
mjm = env.sim.mj_model                       # compiled MjModel (the real mjlab model)
nv = mjm.nv

# --- resolve arm dof indices, head site id, in the FULL model ---
# mjlab prefixes entity names: joints are "robot/joint1", the head site "robot/<HEAD>".
def _find_joint(short):
    for cand in (short, f"robot/{short}"):
        try: return mjm.joint(cand).id
        except KeyError: pass
    raise KeyError(short)
def _find_site(short):
    for cand in (short, f"robot/{short}"):
        try: return mjm.site(cand).id
        except KeyError: pass
    raise KeyError(short)
arm_jid_model = [_find_joint(j) for j in ARM]
arm_dofadr = [int(mjm.jnt_dofadr[j]) for j in arm_jid_model]     # 1 dof each (hinge)
head_sid = _find_site(HAMMER_HEAD_SITE_NAME)
print("nv=", nv, "arm_dofadr=", arm_dofadr, "head_sid=", head_sid)
print("dof_armature(all)=", np.array(mjm.dof_armature))
print("arm dof_armature=", np.array(mjm.dof_armature)[arm_dofadr])
print("arm dof_damping =", np.array(mjm.dof_damping)[arm_dofadr])

# scene-entity handles for capturing the natural strike
rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm_sc = SceneEntityCfg("robot", joint_names=ARM); arm_sc.resolve(env.scene)
ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
jid = arm_sc.joint_ids; dt = float(env.physics_dt)
def head(): return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def ntop(): return env.scene["nail_block"].data.site_pos_w[:, ncfg.site_ids].squeeze(1)

# --- record per-substep: contact flag, full qpos, head_z ---
rec = []; orig = env.metrics_manager.compute_substep
def patched():
    orig()
    rec.append({
        "in_c": bool((contact.data.found > 0).any()),
        "qpos_full": np.array(env.sim.wp_data.qpos.numpy()[0]).copy(),
        "qvel_full": np.array(env.sim.wp_data.qvel.numpy()[0]).copy(),
        "qfrc": robot.data._joint_dof_field("qfrc_constraint")[0, jid].clone().numpy().copy(),
        "head_z": float(head()[0, 2]),
    })
env.metrics_manager.compute_substep = patched

env.reset()
# capture the reset (NEAR_NAIL) full qpos before any motion
qpos_reset = np.array(env.sim.wp_data.qpos.numpy()[0]).copy()
ref = SingleStrikeReference(1, env.device, approach_height=0.10)
ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()
for k in range(1, n + 12):
    target = ref.playback_target(min(k, n))
    env.step(((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1., 1.))
    if int(env.episode_length_buf[0]) == 0: break

# find first contact-onset substep
onset = next((i for i, r in enumerate(rec) if r["in_c"]), None)
if onset is None:
    print(json.dumps({"error": "no contact onset in natural strike"})); raise SystemExit
qpos_contact = rec[onset]["qpos_full"]
# natural head velocity at contact onset (downward, along -z), magnitude:
v_n_natural = (rec[onset-1]["head_z"] - rec[onset]["head_z"]) / dt if onset >= 1 else float("nan")
qvel_contact = rec[onset]["qvel_full"]

# ------------------------------------------------------------------
def analytic(qpos_full, label):
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = qpos_full
    mjd.qvel[:] = 0.0
    mujoco.mj_forward(mjm, mjd)
    # full mass matrix (nv x nv), incl armature
    M = np.zeros((nv, nv)); mujoco.mj_fullM(mjm, M, mjd.qM)
    # head translational Jacobian (3 x nv)
    Jp = np.zeros((3, nv)); Jr = np.zeros((3, nv))
    mujoco.mj_jacSite(mjm, mjd, Jp, Jr, head_sid)
    # reduce to arm dofs
    idx = np.array(arm_dofadr)
    Marm = M[np.ix_(idx, idx)]         # 6x6 arm mass sub-block (for context/diag only)
    Jarm = Jp[:, idx]                  # 3x6 head Jacobian on arm dofs
    n_axis = np.array([0.0, 0.0, 1.0]) # nail normal (reaction on hammer is +z)
    g = Jarm.T @ n_axis                # (6,) = J^T n on arm dofs
    # RIGOROUS operational-space inertia: use the FULL nv x nv inverse. J is nonzero only on
    # arm dofs, so n^T J M^{-1} J^T n = g_full^T M^{-1} g_full with g_full nonzero on arm dofs.
    # (M is block-diagonal arm+gripper vs nail, but the gripper dof couples to the arm, so
    # (M_arm)^{-1} != (M^{-1})_arm,arm in general -- use full inverse to be exact.)
    g_full = Jp.T @ n_axis             # (nv,)
    Minv_full = np.linalg.inv(M)
    m_eff = 1.0 / float(g_full @ Minv_full @ g_full)   # operational-space inertia along n
    # cross-check with the naive arm-block inverse (to quantify the coupling error)
    m_eff_naive = 1.0 / float(g @ np.linalg.inv(Marm) @ g)
    g1 = float(np.abs(g).sum())        # ||g||_1
    # head-site world position (for context)
    site_pos = mjd.site_xpos[head_sid].copy()
    return {
        "label": label, "M_arm": Marm, "Jarm": Jarm, "g": g, "g_l1": g1,
        "m_eff": m_eff, "m_eff_naive": m_eff_naive, "site_pos": site_pos,
    }

def report(a, v_list):
    g = a["g"]; g1 = a["g_l1"]; m_eff = a["m_eff"]
    print(f"\n=== {a['label']} ===")
    print(f"head site world pos = {a['site_pos']}")
    print(f"m_eff along n=[0,0,1] = {m_eff:.4f} kg  (naive arm-block: {a['m_eff_naive']:.4f} kg)")
    print(f"g = J^T n (per-joint, N per N)  = {g}")
    print(f"||g||_1 = {g1:.4f}")
    diagM = np.diag(a["M_arm"])
    print(f"diag(M_arm) = {diagM}")
    out = {}
    for tag, v_n in v_list:
        P = m_eff * v_n                       # Cartesian normal impulse (inelastic stop)
        lam = np.abs(g) * P                    # per-joint clean impulse
        ratio = lam / CAPS
        out[tag] = {
            "v_n": round(v_n, 4), "P_cart_Ns": round(float(P), 4),
            "Lambda_perjoint": [round(float(x), 4) for x in lam],
            "Lambda_over_cap": [round(float(x), 4) for x in ratio],
            "worst_Lambda_over_cap": round(float(ratio.max()), 4),
            "worst_joint": int(ratio.argmax()) + 1,
        }
        print(f"  v_n={v_n:.3f} m/s -> P={P:.4f} N.s | Lambda={lam} | /cap={ratio} | worst {ratio.max():.3f} (j{ratio.argmax()+1})")
    return out

V_URDF = 3.1415
V_OBS = 4.65

results = {}
for qp, lab in [(qpos_contact, "CONTACT config (natural onset)"),
                (qpos_reset, "NEAR_NAIL reset pose")]:
    a = analytic(qp, lab)
    # v_n cases: natural observed, then max at each velocity limit (all joints aligned)
    v_list = [
        ("v_natural", v_n_natural),
        ("vmax_urdf_3.14",  V_URDF * a["g_l1"]),
        ("vmax_obs_4.65",   V_OBS  * a["g_l1"]),
    ]
    results[lab] = report(a, v_list)

# Also: independent cross-check of v_n_max via the actual max head speed the natural qvel direction
# would give if scaled to the velocity limit (a physically-plausible single-mode strike, not the
# multi-joint kinematic max). q̇_dir normalized so fastest joint=1, then *v_lim.
a_c = analytic(qpos_contact, "contact")
g = a_c["g"]
qd = qvel_contact[np.array(arm_dofadr)]
if np.abs(qd).max() > 1e-9:
    qd_dir = qd / np.abs(qd).max()
    for tag, vlim in [("single-mode@3.14", V_URDF), ("single-mode@4.65", V_OBS)]:
        v_n = float(g @ (qd_dir * vlim))
        P = a_c["m_eff"] * abs(v_n)
        lam = np.abs(g) * P; ratio = lam / CAPS
        print(f"\n[single-mode strike dir, {tag}] v_n={v_n:.3f} -> worst Lambda/cap={ratio.max():.3f} (j{ratio.argmax()+1})")

# --- MEASURED rectified Lambda over the natural-strike contact window (raw, NOT baseline-subtracted) ---
wins, cur = [], []
for i, r in enumerate(rec):
    if r["in_c"]: cur.append(i)
    elif cur: wins.append(cur); cur = []
if cur: wins.append(cur)
meas = np.zeros(6)
for w in wins:
    for i in w: meas += np.abs(rec[i]["qfrc"]) * dt
w0 = wins[0]; win_ms = len(w0) * dt * 1e3
meas_cap = meas / CAPS
# analytic clean bound at the SAME natural v_n, contact config
a_c2 = analytic(qpos_contact, "x"); Pnat = a_c2["m_eff"] * v_n_natural
clean_nat = np.abs(a_c2["g"]) * Pnat
print("\n=== MEASURED vs ANALYTIC-CLEAN at natural strike (v_n={:.3f}) ===".format(v_n_natural))
print(f"  n_windows={len(wins)} first_window_ms={win_ms:.1f}")
print(f"  MEASURED Lambda (raw rectified) = {meas}  worst/cap={meas_cap.max():.3f} (j{meas_cap.argmax()+1})")
print(f"  ANALYTIC clean (|g|*m_eff*v_n)  = {clean_nat}  worst/cap={(clean_nat/CAPS).max():.3f}")
infl = meas / np.maximum(clean_nat, 1e-9)
print(f"  inflation (measured/clean) per joint = {infl}")

# --- config sweep over the ACTUAL descent trajectory: max clean Lambda/cap at the rail velocity ---
def worst_over_cap_at_vlim(qp, vlim):
    a = analytic(qp, "")
    v_n = vlim * a["g_l1"]
    lam = np.abs(a["g"]) * a["m_eff"] * v_n
    r = lam / CAPS
    return float(r.max()), int(r.argmax())+1, a["m_eff"], a["g_l1"], v_n
# sample every few substeps across the whole recorded descent (kinematic reachable configs)
sweep = []
for i in range(0, len(rec), 5):
    r, j, me, g1, vn = worst_over_cap_at_vlim(rec[i]["qpos_full"], V_OBS)
    sweep.append((i, r, j, me, g1, vn))
best = max(sweep, key=lambda t: t[1])
print("\n=== CONFIG SWEEP over descent trajectory (kinematic-max v_n at 4.65 rad/s) ===")
print(f"  MAX worst-Lambda/cap over all descent configs = {best[1]:.3f} at substep {best[0]} "
      f"(j{best[2]}, m_eff={best[3]:.3f}, ||g||1={best[4]:.3f}, v_n_max={best[5]:.2f} m/s)")
r314 = max(worst_over_cap_at_vlim(rec[i]["qpos_full"], V_URDF)[0] for i in range(0, len(rec), 5))
print(f"  MAX worst-Lambda/cap over all descent configs at 3.14 rad/s = {r314:.3f}")

print("\n\nJSON=" + json.dumps({
    "sweep_max_worst_over_cap_4.65": round(float(best[1]),4),
    "sweep_max_worst_over_cap_3.14": round(float(r314),4),
    "measured_natural_Lambda": [round(float(x),4) for x in meas],
    "measured_natural_worst_over_cap": round(float(meas_cap.max()),4),
    "measured_window_ms": round(float(win_ms),1),
    "clean_natural_Lambda": [round(float(x),4) for x in clean_nat],
    "inflation_measured_over_clean": [round(float(x),3) for x in infl],
    "v_n_natural": round(float(v_n_natural), 4),
    "onset_substep": onset,
    "caps": list(IMP_J_LIMIT),
    "results": results,
}))
