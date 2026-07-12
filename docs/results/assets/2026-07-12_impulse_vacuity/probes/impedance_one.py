"""IMPEDANCE-FRONTIER probe (one kp per process). Answers: at the reachable ~1.3 m/s strike
against a GENUINELY RIGID target, does raising commanded stiffness (VIC's lever) push the
IMPACT-only worst-joint Λ/cap across the cap -- at a velocity the sim can resolve?

Folds in the frontier review's fixes:
  * RIGID target (C1): lock nail_slide to range [0,0] (a hard limit constraint, unlike the inert
    frictionloss) and ASSERT the nail stays put (<1 mm) -- catches a silent lock failure. Locking
    also removes the nail_driven truncation (I7): a rigid nail never completes, so nothing resets.
  * PRESS-GATE (I4): integrate Λ only over the IMPACT interval [first contact -> head vel <= 0],
    excluding the PD drive-through press that follows. Report full-window Λ too so press share shows.
  * IMPEDANCE sweep (C3): scale the arm PD gains by --kp_scale for the whole scripted strike; report
    the ACHIEVED impact velocity, since stiffer tracking raises both m_eff and speed (realistic VIC).

Baseline-subtracted per-joint reaction, one env/process, read-only spec-wrapping."""
from __future__ import annotations
import argparse, json, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
ap = argparse.ArgumentParser()
ap.add_argument("--kp_scale", type=float, default=1.0, help="arm PD gain multiplier (1=fixed impedance)")
ap.add_argument("--play_speed", type=float, default=4.0, help="reference tracking aggressiveness")
ap.add_argument("--hold", type=int, default=30, help="follow-through control steps after the descent")
args = ap.parse_args()

cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
cfg.scene.num_envs = 1

# --- lock the nail rigid: range [0,0] hard limit on nail_slide (frictionloss is an inert no-op here) ---
nb = cfg.scene.entities["nail_block"]
orig_fn = nb.spec_fn
def patched_spec():
    spec = orig_fn()
    for j in spec.joints:
        if j.name == "nail_slide":
            try:
                # Huge armature = effectively infinite joint inertia -> the nail cannot be accelerated
                # by the impact, so it reflects. A dof inertia property (not a violable constraint),
                # so it can't be silently blown through like frictionloss/range were.
                j.armature = 1.0e6
            except Exception as e:
                print("lock-warn", e)
    return spec
nb.spec_fn = patched_spec

env = ManagerBasedRlEnv(cfg, device="cpu")
robot = env.scene["robot"]; nail_e = env.scene["nail_block"]
contact = env.scene["hammer_nail_contact"]; netf = env.scene["hammer_nail_impulse"]
rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm = SceneEntityCfg("robot", joint_names=ARM); arm.resolve(env.scene)
ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
nailj = SceneEntityCfg("nail_block", joint_names=("nail_slide",)); nailj.resolve(env.scene)
jid = arm.joint_ids; njid = nailj.joint_ids
axis = torch.tensor([0., 0., -1.]); dt = float(env.physics_dt)
def head(): return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def ntop(): return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)
def nail_q(): return float(nail_e.data.joint_pos[0, njid].reshape(-1)[0])
def in_contact(): return bool((contact.data.found > 0).any())
acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)

# --- scale arm PD gains for the whole strike (BuiltinPositionActuator: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd) ---
if abs(args.kp_scale - 1.0) > 1e-9:
    m = env.sim.model
    eidx = torch.arange(1, device=env.device)[:, None]
    for act in robot.actuators:
        cids = act.global_ctrl_ids
        m.actuator_gainprm[eidx, cids, 0] = m.actuator_gainprm[eidx, cids, 0] * args.kp_scale
        m.actuator_biasprm[eidx, cids, 1] = m.actuator_biasprm[eidx, cids, 1] * args.kp_scale
        m.actuator_biasprm[eidx, cids, 2] = m.actuator_biasprm[eidx, cids, 2] * args.kp_scale

rec = []
orig_sub = env.metrics_manager.compute_substep
def patched_sub():
    orig_sub()
    rec.append({"in_c": in_contact(),
                "qfrc": robot.data._joint_dof_field("qfrc_constraint")[0, jid].clone(),
                "lam_acc": acc.impulse[0].detach().clone(),
                "f_ax": float((netf.data.force * axis).sum(-1).sum(-1).clamp_min(0.)[0]),
                "head_z": float(head()[0, 2]), "nail_q": nail_q()})
env.metrics_manager.compute_substep = patched_sub

env.reset()
q0 = nail_q()
ref = SingleStrikeReference(1, env.device, approach_height=0.10)
ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()
reset_during = False
for k in range(1, n + args.hold + 1):
    tgt = ref.playback_target(min(k, n))
    env.step(((tgt - head()) / Z1_HAMMER_DELTA_POS_SCALE * args.play_speed).clamp(-1., 1.))
    if int(env.episode_length_buf[0]) == 0:
        reset_during = True
        break

# --- rigid-target assertion: nail must not have moved ---
nail_disp_mm = max((abs(r["nail_q"] - q0) for r in rec), default=0.0) * 1e3
lock_ok = nail_disp_mm < 1.0

# --- windows ---
first_c = next((i for i, r in enumerate(rec) if r["in_c"]), None)
def vdown(i): return -(rec[i]["head_z"] - rec[i - 1]["head_z"]) / dt if i >= 1 else float("nan")

impact_win, full_win = [], []
achieved_v = float("nan")
if first_c is not None:
    achieved_v = vdown(first_c)
    # full contact-anchored window (first contact -> last contiguous contact)
    i = first_c
    while i < len(rec) and rec[i]["in_c"]:
        full_win.append(i); i += 1
    # impact-only window: first contact -> head velocity first <= 0 (stop/rebound), within contact
    for j in full_win:
        impact_win.append(j)
        if j >= 1 and vdown(j) <= 0.0:
            break

baseline = rec[first_c - 1]["qfrc"] if (first_c is not None and first_c >= 1) else torch.zeros(6)

def lam_over_window(win):
    L = torch.zeros(6)
    for i in win:
        L += (rec[i]["qfrc"] - baseline).abs() * dt
    return L

lam_impact = lam_over_window(impact_win)              # press-gated, baseline-subtracted
lam_full_raw = lam_over_window(full_win)              # whole contact window (impact + press)
lam_acc_peak = torch.zeros(6)                          # shipped accumulator peak (what the constraint reads)
for r in rec:
    lam_acc_peak = torch.maximum(lam_acc_peak, r["lam_acc"])

over_impact = [float(lam_impact[j]) / IMP_J_LIMIT[j] for j in range(6)]
over_full = [float(lam_full_raw[j]) / IMP_J_LIMIT[j] for j in range(6)]
over_acc = [float(lam_acc_peak[j]) / IMP_J_LIMIT[j] for j in range(6)]
wj = max(range(6), key=lambda j: over_impact[j])

delivered_impact = sum(rec[i]["f_ax"] for i in impact_win) * dt
m_eff = (delivered_impact / achieved_v) if (achieved_v == achieved_v and achieved_v > 1e-6) else float("nan")
fpeak = max((rec[i]["f_ax"] for i in full_win), default=0.0)

print(json.dumps({
    "kp_scale": args.kp_scale, "lock_ok": bool(lock_ok), "nail_disp_mm": round(nail_disp_mm, 3),
    "reset_during": reset_during, "achieved_impact_v": round(achieved_v, 3) if achieved_v == achieved_v else None,
    "impact_win_substeps": len(impact_win), "full_win_substeps": len(full_win),
    "peak_F": round(fpeak, 1), "delivered_impact_Ns": round(delivered_impact, 4),
    "m_eff_kg": round(m_eff, 3) if m_eff == m_eff else None,
    "worst_joint": wj + 1,
    "worst_over_cap_IMPACT": round(over_impact[wj], 3),
    "worst_over_cap_FULL": round(max(over_full), 3),
    "worst_over_cap_ACCUM": round(max(over_acc), 3),
    "impact_over_cap_perjoint": [round(x, 3) for x in over_impact],
    "binds_impact": bool(max(over_impact) >= 1.0 and lock_ok),
}))
