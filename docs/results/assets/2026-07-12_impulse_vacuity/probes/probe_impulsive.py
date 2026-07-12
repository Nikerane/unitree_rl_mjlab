"""DEFINITIVE impulsive-vs-press + contact-chatter probe (reference strike, training cfg).

Extends the validated diag_event.py setup (same env, same reference driving) to additionally log
per-substep JOINT VELOCITY and head axial velocity, so we can settle two questions with DATA:

  Q1 (chatter / EXPLOIT-2 reachability): how many contact windows per strike? gaps? -> is the
      metric fragmented by numerical contact chatter on the reference strike?
  Q2 (impulsive vs press / scope-B thesis framing): is Λ = Σ|qfrc|·dt a ballistic momentum transfer
      or a sustained press integral? Discriminators (all computed on the FIRST contact window):
        (a) force profile: peak/mean axial force  (impulsive: peak>>mean spike; press: peak≈mean plateau)
        (b) window duration in ms                  (impulsive: ~1-5 ms; press: 10s of ms)
        (c) signed vs rectified constraint impulse per joint: |Σ qfrc·dt| / Σ|qfrc|·dt
            (low ratio => sustained/oscillating reaction that does NOT net-accelerate the joint = press;
             ~1 => a coherent one-directional momentum kick = impulsive)
        (d) joint-velocity change through contact: is Δq̇ abrupt (concentrated in the first 1-2
            substeps = impulsive) or gradual/tracking-through (spread over the window = press)?
        (e) head axial velocity trajectory: abrupt stop/rebound vs gradual decay.

No repo file edited; read-only diagnostic. CPU, 1 env, baseline (current training) config.
"""
from __future__ import annotations
import json, sys
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
HEIGHTS = [0.06, 0.10, 0.15]
HOLD = 6
OUT = "/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/c51273df-43b1-4386-842f-086eab5e35db/scratchpad/probe_impulsive.json"

cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
cfg.scene.num_envs = 1
assert cfg.sim.mujoco.timestep == 0.002 and cfg.decimation == 10
env = ManagerBasedRlEnv(cfg, device="cpu")
robot = env.scene["robot"]
nail_e = env.scene["nail_block"]
contact = env.scene["hammer_nail_contact"]
netf = env.scene["hammer_nail_impulse"]
rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
arm = SceneEntityCfg("robot", joint_names=ARM); arm.resolve(env.scene)
ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
jid = arm.joint_ids
axis = torch.tensor([0.0, 0.0, -1.0])
dt = float(env.physics_dt)

def head():
    return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def nail_top():
    return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

rec = []
orig = env.metrics_manager.compute_substep
def patched():
    orig()
    qfrc = robot.data._joint_dof_field("qfrc_constraint")[0, jid].clone()
    qvel = robot.data._joint_dof_field("qvel")[0, jid].clone()
    f = netf.data.force
    f_ax = float((f * axis).sum(-1).sum(-1).clamp_min(0.0)[0])
    rec.append({"in_c": bool((contact.data.found > 0).any()),
                "qfrc": qfrc, "qvel": qvel, "f_ax": f_ax,
                "head_z": float(head()[0, 2])})
env.metrics_manager.compute_substep = patched  # type: ignore

def analyze(rec, dt):
    # first contact window
    wins, cur = [], []
    for i, r in enumerate(rec):
        if r["in_c"]: cur.append(i)
        elif cur: wins.append(cur); cur = []
    if cur: wins.append(cur)
    if not wins:
        return {"n_windows": 0}
    w0 = wins[0]
    i0, i1 = w0[0], w0[-1]
    fs = [rec[i]["f_ax"] for i in w0]
    peak_f, mean_f = max(fs), (sum(fs) / len(fs))
    # per-joint signed vs rectified constraint impulse over window 0
    rect = torch.zeros(6); signed = torch.zeros(6)
    for i in w0:
        rect += rec[i]["qfrc"].abs() * dt
        signed += rec[i]["qfrc"] * dt
    signed_ratio = (signed.abs() / rect.clamp_min(1e-9)).tolist()
    # joint velocity change through window: pre = last pre-contact substep, trajectory over window
    qv_pre = rec[i0 - 1]["qvel"] if i0 >= 1 else rec[i0]["qvel"]
    dqv_total = (rec[i1]["qvel"] - qv_pre)                      # net Δq̇ over the whole event
    # per-substep |Δq̇| through the window; concentration in first 2 substeps
    steps = [rec[i0 - 1]] + [rec[i] for i in w0] if i0 >= 1 else [rec[i] for i in w0]
    per = [ (steps[k]["qvel"] - steps[k-1]["qvel"]).abs() for k in range(1, len(steps)) ]
    if per:
        cum = torch.stack(per).sum(0)                          # total variation of q̇ per joint
        first2 = torch.stack(per[:2]).sum(0) if len(per) >= 2 else per[0]
        concentration = (first2 / cum.clamp_min(1e-9)).tolist()  # frac of Δq̇ in first 2 substeps
        maxstep = torch.stack(per).amax(0).tolist()
    else:
        concentration = maxstep = [float("nan")] * 6
    # head axial (downward +) velocity trajectory through the window
    hz = [rec[i]["head_z"] for i in ([i0 - 1] if i0 >= 1 else []) + w0]
    hv = [ -(hz[k] - hz[k-1]) / dt for k in range(1, len(hz)) ]  # +v = descending
    return {
        "n_windows": len(wins),
        "window_lengths_substeps": [len(w) for w in wins],
        "window_gaps_substeps": [wins[j+1][0] - wins[j][-1] - 1 for j in range(len(wins)-1)],
        "window0_len_substeps": len(w0), "window0_len_ms": len(w0) * dt * 1e3,
        "peak_f": peak_f, "mean_f": mean_f, "peak_over_mean_f": peak_f / max(mean_f, 1e-9),
        "rect_Lambda_perjoint": rect.tolist(),
        "signed_Lambda_perjoint": signed.tolist(),
        "signed_over_rect_ratio_perjoint": signed_ratio,
        "dqv_total_perjoint": dqv_total.tolist(),
        "dqv_concentration_first2_perjoint": concentration,
        "dqv_maxstep_perjoint": maxstep,
        "head_v_trajectory_through_window": [round(v, 4) for v in hv],
    }

out = []
for h in HEIGHTS:
    env.reset()
    ref = SingleStrikeReference(1, env.device, approach_height=h)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length()
    rec.clear()
    for k in range(1, n + HOLD + 1):
        target = ref.playback_target(min(k, n))
        action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
        env.step(action)
        if int(env.episode_length_buf[0]) == 0:
            break
    a = analyze(list(rec), dt)
    a["height"] = h
    out.append(a)
    if a["n_windows"] > 0:
        print(f"h={h}: n_windows={a['n_windows']} lens={a['window_lengths_substeps']} "
              f"gaps={a['window_gaps_substeps']} win0={a['window0_len_ms']:.1f}ms "
              f"peak/mean_F={a['peak_over_mean_f']:.2f} "
              f"signed/rect(j2,j3,j4)={a['signed_over_rect_ratio_perjoint'][1]:.2f},"
              f"{a['signed_over_rect_ratio_perjoint'][2]:.2f},{a['signed_over_rect_ratio_perjoint'][3]:.2f} "
              f"dqv_conc(j2,j4)={a['dqv_concentration_first2_perjoint'][1]:.2f},"
              f"{a['dqv_concentration_first2_perjoint'][3]:.2f}")
        print(f"    head_v through window (m/s, +=descending): {a['head_v_trajectory_through_window']}")
    else:
        print(f"h={h}: NO CONTACT WINDOW")

env.metrics_manager.compute_substep = orig  # type: ignore
with open(OUT, "w") as fh:
    json.dump({"physics_dt": dt, "strikes": out}, fh, indent=1)
print("JSON:", OUT)
