"""WHIP TIME-SERIES probe: dump the FULL per-substep series for a whip injection and DECOMPOSE
the contact window into impact (first ~few ms) vs sustained-press tail.

Answers the sub-question: is the whip_vel=8 "binding" Λ_j3=2.67 (176ms/88-substep window) a GENUINE
impulsive impact or a probe artifact?

Per substep during free-fly we log: in_contact, axial contact force f_ax, per-joint qfrc_constraint,
per-joint qvel, head_z, nail penetration (nail_slide qpos). Then we decompose window 0:
 - cumulative Λ_j(t) vs time -> fraction of Λ in first 5/8/16/24 ms (impact) vs the long tail (press)
 - when the nail reaches its 0.032 hard limit and stops moving (after that, all Λ is press vs a pinned object)
 - head axial velocity trajectory: rebound/chatter (impact) vs monotone press
 - qvel_j3 trajectory: does the PD immediately fight the injected velocity? (injection-artifact check)
 - contact chatter: n windows / gaps

One env / process (mjlab warp constraint). CPU, deterministic. Read-only on repo.
"""
from __future__ import annotations
import argparse, json, math, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1","joint2","joint3","joint4","joint5","joint6")
ap = argparse.ArgumentParser()
ap.add_argument("--whip_vel", type=float, default=8.0)
ap.add_argument("--height", type=float, default=0.10)
ap.add_argument("--nfree", type=int, default=30)   # env steps of free-fly after injection
ap.add_argument("--disable_contact", action="store_true")  # remove nail_head collision -> free-space control
ap.add_argument("--out", type=str, default="/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/c51273df-43b1-4386-842f-086eab5e35db/scratchpad/whip_timeseries.json")
args = ap.parse_args()

cfg = z1_hammer_env_cfg(play=True, cat_impulse=True); cfg.scene.num_envs = 1
if args.disable_contact:
    nb = cfg.scene.entities["nail_block"]; _orig_spec = nb.spec_fn
    def _no_contact_spec():
        sp = _orig_spec()
        for g in sp.geoms:
            if g.name in ("nail_head", "nail_shaft"):
                g.contype = 0; g.conaffinity = 0
        return sp
    nb.spec_fn = _no_contact_spec
env = ManagerBasedRlEnv(cfg, device="cpu")
robot = env.scene["robot"]; contact = env.scene["hammer_nail_contact"]; netf = env.scene["hammer_nail_impulse"]
nail_e = env.scene["nail_block"]
rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm = SceneEntityCfg("robot", joint_names=ARM); arm.resolve(env.scene)
ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
njcfg = SceneEntityCfg("nail_block", joint_names=("nail_slide",)); njcfg.resolve(env.scene)
jid = arm.joint_ids; njid = njcfg.joint_ids
if isinstance(njid, slice): njid = list(range(*njid.indices(nail_e.data.joint_pos.shape[1])))
axis = torch.tensor([0.,0.,-1.]); dt = float(env.physics_dt)
def head(): return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def ntop(): return env.scene["nail_block"].data.site_pos_w[:, ncfg.site_ids].squeeze(1)
def ndepth(): return float(nail_e.data.joint_pos[0, njid[0]])

ARM_RANGES = [(-2.61799,2.61799),(0.,2.96706),(-2.87979,0.),(-1.51844,1.51844),(-1.3439,1.3439),(-2.79253,2.79253)]
import numpy as np
efc_snaps = {}
EFC_AT = {13: "impact_~2ms", 52: "tail_~80ms"}   # rec indices during free-fly
def snap_efc(tag):
    st = robot.data.data._struct
    nefc = int(np.asarray(st.nefc.numpy())[0])
    ef = st.efc
    force = np.asarray(ef.force.numpy())[0, :nefc]
    typ = np.asarray(ef.type.numpy())[0, :nefc]
    J = np.asarray(ef.J.numpy())[0, :nefc, :]   # (nefc, 8)
    efc_snaps[tag] = {"nefc": nefc, "force": force.tolist(), "type": typ.tolist(),
                      "contrib_j3": (J[:, 2] * force).tolist(), "J_j3": J[:, 2].tolist()}

rec = []; orig = env.metrics_manager.compute_substep
def patched():
    orig()
    if len(rec) in EFC_AT and not args.disable_contact:
        snap_efc(EFC_AT[len(rec)])
    rec.append({"in_c": bool((contact.data.found>0).any()),
                "qfrc": robot.data._joint_dof_field("qfrc_constraint")[0,jid].clone(),
                "qvel": robot.data._joint_dof_field("qvel")[0,jid].clone(),
                "qpos": robot.data._joint_dof_field("qpos")[0,jid].clone(),
                "f_ax": float((netf.data.force*axis).sum(-1).sum(-1).clamp_min(0.)[0]),
                "head_z": float(head()[0,2]),
                "nail_qfrc": float(nail_e.data._joint_dof_field("qfrc_constraint")[0, njid[0]]),
                "qfrc_act": robot.data._joint_dof_field("qfrc_actuator")[0,jid].clone(),
                "qfrc_bias": robot.data._joint_dof_field("qfrc_bias")[0,jid].clone(),
                "qfrc_pass": robot.data._joint_dof_field("qfrc_passive")[0,jid].clone(),
                "qacc": robot.data._joint_dof_field("qacc")[0,jid].clone(),
                "nefc": int(robot.data.data.nefc[0]),
                "ndepth": ndepth()})
env.metrics_manager.compute_substep = patched

env.reset()
ref = SingleStrikeReference(1, env.device, approach_height=args.height)
ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()
pre = max(1, n - 2)
for k in range(1, pre + 1):
    target = ref.playback_target(min(k, n))
    env.step(((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1., 1.))

qd = robot.data._joint_dof_field("qvel")[0, jid].clone()
if float(qd.abs().max()) < 1e-6:
    qd = torch.ones(6)
qd_dir = qd / qd.abs().max()
whip = (qd_dir * args.whip_vel)
qvel_pre_inject = robot.data._joint_dof_field("qvel")[0, jid].clone()
full = robot.data.joint_vel.clone(); full[0, jid] = whip
robot.write_joint_velocity_to_sim(full)
head_z_at_inject = float(head()[0,2])
rec.clear()
for k in range(1, args.nfree + 1):
    target = ref.playback_target(n)
    env.step(((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1., 1.))
    if int(env.episode_length_buf[0]) == 0: break

# windows
J3 = 2  # joint3 index
wins, cur = [], []
for i, r in enumerate(rec):
    if r["in_c"]: cur.append(i)
    elif cur: wins.append(cur); cur = []
if cur: wins.append(cur)

result = {"whip_vel": args.whip_vel, "height": args.height, "dt": dt,
          "caps": list(IMP_J_LIMIT), "nail_limit_m": 0.032,
          "qvel_j_dir_normalized": [round(float(x),3) for x in qd_dir],
          "injected_qvel": [round(float(x),3) for x in whip],
          "qvel_pre_inject": [round(float(x),4) for x in qvel_pre_inject],
          "n_windows": len(wins), "window_lens": [len(w) for w in wins],
          "window_gaps": [wins[j+1][0]-wins[j][-1]-1 for j in range(len(wins)-1)]}

if not wins:
    result["note"] = "no contact window"
    # free-space reference: cumulative rectified |qfrc_j3|*dt over the SAME free-fly period
    cumL = 0.0; raw = []
    for k, r in enumerate(rec):
        cumL += abs(float(r["qfrc"][J3])) * dt
        if k < 90:
            raw.append({"t_ms": round((k+1)*dt*1e3,1), "qfrc_j3": round(float(r["qfrc"][J3]),2),
                        "qvel_j3": round(float(r["qvel"][J3]),3), "cumL_j3": round(cumL,4),
                        "qfrc_abs": [round(float(x),2) for x in r["qfrc"]]})
    result["freespace_cumLambda_j3_90steps"] = round(cumL,4)
    result["freespace_raw"] = raw
    with open(args.out,"w") as fh: json.dump(result, fh, indent=1)
    print(f"NO-CONTACT free-space run: cumΛ_j3 over 90 substeps (180ms) = {round(cumL,4)}  (cap={IMP_J_LIMIT[J3]})")
    print("  qfrc_j3 samples (t_ms, qfrc_j3, qvel_j3, cumL):")
    for r in raw[:3]+raw[6:14:2]+raw[20:90:12]:
        print(f"   {r['t_ms']:6.1f}  qfrc_j3={r['qfrc_j3']:7.2f}  qvel_j3={r['qvel_j3']:7.3f}  cumL={r['cumL_j3']:.4f}   |qfrc|all={r['qfrc_abs']}")
    raise SystemExit

# Use the UNION of all windows treated as one event span [first onset .. last release]
i_first = wins[0][0]; i_last = wins[-1][-1]
span = list(range(i_first, i_last+1))

# cumulative Lambda per joint over the ENTIRE recorded contact (all windows), and time marks
cum_L = torch.zeros(6); cum_Lsigned = torch.zeros(6)
series = []
head_z_prev = rec[i_first-1]["head_z"] if i_first>=1 else rec[i_first]["head_z"]
marks_ms = [2,4,6,8,10,16,24,40,80,176]
cumL_at_ms = {m: None for m in marks_ms}
nail_at_limit_idx = None
nail_stop_idx = None
for pos, i in enumerate(span):
    r = rec[i]
    if r["in_c"]:
        cum_L += r["qfrc"].abs() * dt
        cum_Lsigned += r["qfrc"] * dt
    t_ms = (pos+1) * dt * 1e3
    hz = r["head_z"]; hv = -(hz - head_z_prev)/dt; head_z_prev = hz
    # distance to nearest joint limit (j3,j4)
    dlim = [min(abs(float(r["qpos"][k])-ARM_RANGES[k][0]), abs(float(r["qpos"][k])-ARM_RANGES[k][1])) for k in range(6)]
    series.append({"i": i, "t_ms": round(t_ms,1), "in_c": r["in_c"],
                   "f_ax": round(r["f_ax"],2),
                   "nail_qfrc": round(r["nail_qfrc"],2),
                   "nefc": r["nefc"],
                   "qfrc_act_j3": round(float(r["qfrc_act"][J3]),2),
                   "qfrc_bias_j3": round(float(r["qfrc_bias"][J3]),2),
                   "qfrc_pass_j3": round(float(r["qfrc_pass"][J3]),2),
                   "qacc_j3": round(float(r["qacc"][J3]),3),
                   "qfrc_j3": round(float(r["qfrc"][J3]),2),
                   "qfrc_abs": [round(float(x),2) for x in r["qfrc"]],
                   "qvel": [round(float(x),3) for x in r["qvel"]],
                   "qpos": [round(float(x),4) for x in r["qpos"]],
                   "dist_to_limit": [round(x,4) for x in dlim],
                   "head_v": round(hv,4), "ndepth": round(r["ndepth"],5),
                   "cumL_j3": round(float(cum_L[J3]),4),
                   "cumLsigned_j3": round(float(cum_Lsigned[J3]),4)})
    for m in marks_ms:
        if cumL_at_ms[m] is None and t_ms >= m - 1e-9:
            cumL_at_ms[m] = round(float(cum_L[J3]),4)
    if nail_at_limit_idx is None and r["ndepth"] >= 0.032 - 1e-4:
        nail_at_limit_idx = pos
# when nail effectively stops moving (depth change < 1e-5 m/substep for the rest)
depths = [rec[i]["ndepth"] for i in span]
for pos in range(1, len(depths)):
    remaining = depths[pos:]
    if max(remaining) - min(remaining) < 1e-4:
        nail_stop_idx = pos; break

total_L_j3 = float(cum_L[J3])
# fill any None marks with the final value (window shorter than the mark)
for m in marks_ms:
    if cumL_at_ms[m] is None: cumL_at_ms[m] = round(total_L_j3,4)

# Λ accumulated AFTER the nail stops moving == press against pinned nail
L_after_nailstop = None
if nail_stop_idx is not None:
    cL = torch.zeros(6)
    for pos in range(nail_stop_idx, len(span)):
        r = rec[span[pos]]
        if r["in_c"]: cL += r["qfrc"].abs()*dt
    L_after_nailstop = round(float(cL[J3]),4)

# head velocity min (rebound detection) and how many sign flips (chatter)
hvs = [s["head_v"] for s in series]
flips = sum(1 for k in range(1,len(hvs)) if (hvs[k]<-0.01) != (hvs[k-1]<-0.01))
result.update({
    "span_substeps": len(span), "span_ms": round(len(span)*dt*1e3,1),
    "total_Lambda_j3": round(total_L_j3,4),
    "total_Lambda_j3_SIGNED": round(float(cum_Lsigned[J3]),4),
    "signed_over_rect_j3": round(abs(float(cum_Lsigned[J3]))/max(total_L_j3,1e-9),3),
    "Lambda_j3_over_cap": round(total_L_j3/IMP_J_LIMIT[J3],3),
    "cumLambda_j3_at_ms": cumL_at_ms,
    "frac_L_in_first_8ms": round(cumL_at_ms[8]/max(total_L_j3,1e-9),3),
    "frac_L_in_first_24ms": round(cumL_at_ms[24]/max(total_L_j3,1e-9),3),
    "frac_L_after_first_24ms": round(1 - cumL_at_ms[24]/max(total_L_j3,1e-9),3),
    "nail_at_limit_substep": nail_at_limit_idx,
    "nail_at_limit_ms": None if nail_at_limit_idx is None else round((nail_at_limit_idx+1)*dt*1e3,1),
    "nail_stop_substep": nail_stop_idx,
    "nail_stop_ms": None if nail_stop_idx is None else round((nail_stop_idx+1)*dt*1e3,1),
    "Lambda_j3_after_nail_stops": L_after_nailstop,
    "frac_L_after_nail_stops": None if L_after_nailstop is None else round(L_after_nailstop/max(total_L_j3,1e-9),3),
    "head_v_at_onset": hvs[0] if hvs else None,
    "head_v_min": min(hvs) if hvs else None,
    "head_v_rebound(neg)": bool(min(hvs) < -0.02) if hvs else None,
    "head_v_sign_flips": flips,
    "f_ax_peak": round(max(s["f_ax"] for s in series),2),
    "f_ax_mean_incontact": round(sum(s["f_ax"] for s in series if s["in_c"])/max(1,sum(1 for s in series if s["in_c"])),2),
    "qvel_j3_pre_inject": round(float(qvel_pre_inject[J3]),4),
    "qvel_j3_injected": round(float(whip[J3]),4),
    "qvel_j3_first5_after_inject": [round(float(rec[k]["qvel"][J3]),3) for k in range(min(5,len(rec)))],
})
result["series"] = series
# efc decomposition of joint3 constraint force, grouped by constraint type
CTYPE = {0:"EQUALITY",1:"FRICTION_DOF",2:"FRICTION_TENDON",3:"LIMIT_JOINT",4:"LIMIT_TENDON",
         5:"CONTACT_FRICTIONLESS",6:"CONTACT_PYRAMIDAL",7:"CONTACT_ELLIPTIC"}
efc_decomp = {}
for tag, sn in efc_snaps.items():
    by = {}
    for t, c in zip(sn["type"], sn["contrib_j3"]):
        nm = CTYPE.get(int(t), f"type{t}")
        by[nm] = by.get(nm, 0.0) + c
    efc_decomp[tag] = {"nefc": sn["nefc"], "j3_by_ctype": {k: round(v,3) for k,v in by.items()},
                       "j3_total": round(sum(sn["contrib_j3"]),3)}
result["efc_decomp_j3"] = efc_decomp
with open(args.out,"w") as fh: json.dump(result, fh, indent=1)
if efc_decomp:
    print("EFC decomposition of joint3 qfrc_constraint by constraint type:")
    for tag, dd in efc_decomp.items():
        print(f"  [{tag}] nefc={dd['nefc']}  j3_total={dd['j3_total']}  by_type={dd['j3_by_ctype']}")

# console summary
print(f"whip_vel={args.whip_vel}  head_v@onset={result['head_v_at_onset']:.2f} m/s")
print(f"n_windows={result['n_windows']} lens={result['window_lens']} gaps={result['window_gaps']}  span={result['span_ms']}ms")
print(f"Λ_j3 = {result['total_Lambda_j3']:.3f}  = {result['Lambda_j3_over_cap']:.0%} of cap")
print(f"cumΛ_j3 at ms: {result['cumLambda_j3_at_ms']}")
print(f"  frac in first 8ms  = {result['frac_L_in_first_8ms']:.0%}")
print(f"  frac in first 24ms = {result['frac_L_in_first_24ms']:.0%}   (=> frac AFTER 24ms = {result['frac_L_after_first_24ms']:.0%})")
print(f"nail hits 0.032 limit at {result['nail_at_limit_ms']} ms ; nail STOPS moving at {result['nail_stop_ms']} ms")
print(f"  Λ_j3 accumulated AFTER nail stops = {result['Lambda_j3_after_nail_stops']}  = {result['frac_L_after_nail_stops']} of total")
print(f"head_v: onset={result['head_v_at_onset']:.2f}  min={result['head_v_min']:.3f}  rebound={result['head_v_rebound(neg)']}  sign_flips={result['head_v_sign_flips']}")
print(f"f_ax: peak={result['f_ax_peak']}  mean(in-contact)={result['f_ax_mean_incontact']}  peak/mean={result['f_ax_peak']/max(result['f_ax_mean_incontact'],1e-9):.2f}")
print(f"qvel_j3: pre_inject={result['qvel_j3_pre_inject']}  injected={result['qvel_j3_injected']}  first5_after={result['qvel_j3_first5_after_inject']}")
print("JSON:", args.out)
