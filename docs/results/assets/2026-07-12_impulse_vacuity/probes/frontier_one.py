"""BINDING-FRONTIER probe (one point per process; warp needs one env/process).

Measures worst-joint Λ/cap as a function of *pre-impact velocity* for a BALLISTIC strike,
isolating the impulse from the velocity (the fixed PD controller can only reach ~1.3 m/s into
contact; we exceed that by direct velocity injection). Mechanism:

  1. Playback the SingleStrikeReference to just above the nail (normal gains) to place the arm on
     the strike axis, capturing the natural strike joint-velocity direction q_dot_pre.
  2. ZERO the arm PD gains (set_gains kp=kd=0). With gravcomp on, the arm now coasts ballistically
     -- no PD to bleed the injected velocity, no active pressing to contaminate the reaction.
  3. Inject arm joint velocity = q_dot_pre * (V / v_play) so the head approaches at ~V m/s, then
     step through the contact window while the SHIPPED SubstepImpulseAccumulator integrates Λ at
     500 Hz (baseline-subtracted, per-event pulse -- exactly what the constraint reads).
  4. Report Λ/cap (accumulator + a raw no-baseline cross-check), the ACHIEVED impact velocity
     (measured, so the x-axis is truthful), plus impact-vs-press discriminators (window length,
     peak/mean force) and the object-side delivered impulse for a momentum cross-check.

target=reflecting locks nail_slide (frictionloss 1e6) so the nail reflects instead of driving
home; target=yielding is the real 7 g nail. Read-only spec-wrapping (no repo/asset edits)."""
from __future__ import annotations
import argparse, json, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
NAIL_GEOMS = ("nail_shaft", "nail_head")

ap = argparse.ArgumentParser()
ap.add_argument("--v", type=float, default=3.0, help="target pre-impact head speed (m/s)")
ap.add_argument("--target", choices=["yielding", "reflecting"], default="reflecting")
ap.add_argument("--solref_scale", type=float, default=1.0, help="<1 stiffens nail contact")
ap.add_argument("--kp_scale", type=float, default=0.0,
                help="arm PD gain multiplier at impact: 0=clean ballistic, 1=fixed impedance, >1=VIC-like")
ap.add_argument("--hold", type=int, default=45, help="control steps to follow the impact through")
ap.add_argument("--gap", type=float, default=0.035, help="inject when head is this far above nail_top")
args = ap.parse_args()

cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
cfg.scene.num_envs = 1
cap = torch.tensor(IMP_J_LIMIT)

nb = cfg.scene.entities["nail_block"]
orig_fn = nb.spec_fn
def patched_spec():
    spec = orig_fn()
    for j in spec.joints:
        if j.name == "nail_slide" and args.target == "reflecting":
            try: j.frictionloss = 1.0e6
            except Exception: pass
    if args.solref_scale != 1.0:
        for g in spec.geoms:
            if g.name in NAIL_GEOMS:
                try: g.solref[0] = float(g.solref[0]) * args.solref_scale
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
jid = arm.joint_ids; njid = nailj.joint_ids
axis = torch.tensor([0., 0., -1.]); dt = float(env.physics_dt)
def head(): return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def ntop(): return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)
def nail_q(): return float(nail_e.data.joint_pos[0, njid].reshape(-1)[0])
def in_contact(): return bool((contact.data.found > 0).any())

acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)

rec = []
orig_sub = env.metrics_manager.compute_substep
def patched_sub():
    orig_sub()
    rec.append({"in_c": in_contact(),
                "qfrc": robot.data._joint_dof_field("qfrc_constraint")[0, jid].clone(),
                "lam_acc": acc.impulse[0].detach().clone(),  # shipped Λ (baseline-sub, per-event) THIS substep
                "f_ax": float((netf.data.force * axis).sum(-1).sum(-1).clamp_min(0.)[0]),
                "head_z": float(head()[0, 2]), "nail_q": nail_q()})
env.metrics_manager.compute_substep = patched_sub

env.reset()
q0 = nail_q()
ref = SingleStrikeReference(1, env.device, approach_height=0.10)
ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()

# ---- phase 1: playback to just above contact (normal gains) ----
reached = False
for k in range(1, n + 1):
    if head()[0, 2].item() - ntop()[0, 2].item() < args.gap:
        reached = True; break
    tgt = ref.playback_target(min(k, n))
    env.step(((tgt - head()) / Z1_HAMMER_DELTA_POS_SCALE * 4.0).clamp(-1., 1.))
    if in_contact():  # overshot into contact before reaching the gap
        reached = True; break

# natural strike direction + achieved descent speed from the recorded substeps
qdot_pre = robot.data.joint_vel[0, jid].clone()  # (6,)
desc = [-(rec[i]["head_z"] - rec[i-1]["head_z"]) / dt for i in range(1, len(rec))
        if not rec[i]["in_c"] and rec[i]["head_z"] < rec[i-1]["head_z"]]
v_play = float(torch.tensor(desc).median()) if desc else 1.0
v_play = max(v_play, 0.05)

# ---- phase 2: scale the arm PD gains, then inject velocity for target head speed V ----
# BuiltinPositionActuator keeps kp/kd in the model: gainprm[.,0]=kp, biasprm[.,1]=-kp, biasprm[.,2]=-kd
# (mjlab/envs/mdp/dr/actuator.py). The model gains are still the untouched defaults here, so we scale
# them in place: kp_scale=0 zeroes them (clean ballistic impact, gravcomp still holds the arm),
# kp_scale=1 keeps fixed impedance, kp_scale>1 previews a stiffer VIC command.
eids = torch.arange(1, device=env.device)
m = env.sim.model
eidx = eids[:, None]
for act in robot.actuators:
    cids = act.global_ctrl_ids
    m.actuator_gainprm[eidx, cids, 0] = m.actuator_gainprm[eidx, cids, 0] * args.kp_scale
    m.actuator_biasprm[eidx, cids, 1] = m.actuator_biasprm[eidx, cids, 1] * args.kp_scale
    m.actuator_biasprm[eidx, cids, 2] = m.actuator_biasprm[eidx, cids, 2] * args.kp_scale
qinj = (qdot_pre * (args.v / v_play)).unsqueeze(0)  # (1,6)
robot.write_joint_velocity_to_sim(qinj, joint_ids=jid, env_ids=eids)

rec.clear()
zero_act = torch.zeros(1, 3, device=env.device)
reset_during_hold = False
for _ in range(args.hold):
    env.step(zero_act)
    if int(env.episode_length_buf[0]) == 0:  # episode terminated/reset
        reset_during_hold = True
        break

# ---- metrics ----
# achieved impact velocity: head speed at the substep just before first contact
first_c = next((i for i, r in enumerate(rec) if r["in_c"]), None)
if first_c is not None and first_c >= 1:
    achieved_v = -(rec[first_c]["head_z"] - rec[first_c - 1]["head_z"]) / dt
else:
    achieved_v = float("nan")

# contact windows + per-event raw Λ (no baseline; discriminator only)
wins, cur = [], []
for i, r in enumerate(rec):
    if r["in_c"]: cur.append(i)
    elif cur: wins.append(cur); cur = []
if cur: wins.append(cur)
raw_perevent = torch.zeros(6)
for w in wins:
    wl = torch.zeros(6)
    for i in w: wl += rec[i]["qfrc"].abs() * dt
    raw_perevent = torch.maximum(raw_perevent, wl)

# shipped accumulator (baseline-subtracted, per-event pulse) = authoritative Λ. Read from the
# per-substep captures, not acc._episode_peak_perjoint, so a mid-hold episode reset (a ballistic
# bounce can trip a termination) cannot zero the monotone buffer before we read it.
acc_perjoint = torch.zeros(6)
for r in rec:
    acc_perjoint = torch.maximum(acc_perjoint, r["lam_acc"])
over_acc = [float(acc_perjoint[j]) / IMP_J_LIMIT[j] for j in range(6)]
over_raw = [float(raw_perevent[j]) / IMP_J_LIMIT[j] for j in range(6)]
worst_acc = max(range(6), key=lambda j: over_acc[j])

fwin = wins[0] if wins else []
win_len = len(fwin)
fpeak = max((rec[i]["f_ax"] for i in fwin), default=0.0)
fmean = (sum(rec[i]["f_ax"] for i in fwin) / win_len) if win_len else 0.0
peak_over_mean = (fpeak / fmean) if fmean > 1e-9 else 0.0
delivered = sum(rec[i]["f_ax"] for i in fwin) * dt  # object-side first-window impulse (N*s)
nail_driven = nail_q() - q0
m_eff_est = (delivered / achieved_v) if (achieved_v and achieved_v == achieved_v and achieved_v > 1e-6) else float("nan")

# impact vs press: a genuine impact has a brief contact window; a sustained press keeps contact
# for many substeps. Window length is the primary discriminator (peak/mean reported for context).
press_flag = win_len > 35
binds_genuine = bool(max(over_acc) >= 1.0 and not press_flag)

print(json.dumps({
    "target": args.target, "solref_scale": args.solref_scale, "kp_scale": args.kp_scale,
    "subtract_baseline": bool(getattr(acc, "_subtract_baseline", None)),
    "reset_during_hold": reset_during_hold, "v_cmd": args.v,
    "v_play_ref": round(v_play, 3), "achieved_impact_v": round(achieved_v, 3) if achieved_v == achieved_v else None,
    "n_windows": len(wins), "win_len_substeps": win_len,
    "peak_F": round(fpeak, 1), "mean_F": round(fmean, 1), "peak_over_mean": round(peak_over_mean, 2),
    "delivered_impulse_Ns": round(delivered, 4), "m_eff_est_kg": round(m_eff_est, 3) if m_eff_est == m_eff_est else None,
    "nail_driven_mm": round(nail_driven * 1e3, 2),
    "worst_joint_accum": worst_acc + 1,
    "worst_over_cap_ACCUM": round(over_acc[worst_acc], 3),
    "worst_over_cap_RAW": round(max(over_raw), 3),
    "Lambda_accum_perjoint": [round(float(x), 3) for x in acc_perjoint],
    "over_cap_accum_perjoint": [round(x, 3) for x in over_acc],
    "press_flag": bool(press_flag), "binds_genuine": binds_genuine,
}))
