"""Empirical battery: impulsive-vs-press across speed & stiffness + exploit-severity probes.

T1 speed sweep, T2 stiffness sweep, T3 sustained press (EXPLOIT-3 severity), T4 lift-restrike
(C1/I1 reachability). Read-only diagnostic; extends the validated diag_event / probe_impulsive setup.
Run on the training cfg (play, cat_impulse), 1 env, CPU. Prints one legible table.
"""
from __future__ import annotations
import json, sys, copy
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
OUT = "/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/c51273df-43b1-4386-842f-086eab5e35db/scratchpad/probe_battery.json"


def build_env(actuator=None):
    """actuator: None=default (kp=1000/kd=100); ('soft',kp,kd)=override arm PD gains."""
    cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
    cfg.scene.num_envs = 1
    assert cfg.sim.mujoco.timestep == 0.002 and cfg.decimation == 10
    if actuator is not None:
        _, kp, kd = actuator
        robot_cfg = cfg.scene.entities["robot"]
        art = robot_cfg.articulation
        newacts = []
        for a in art.actuators:
            tn = getattr(a, "target_names_expr", ())
            if any(j in tn for j in ARM):  # an ARM actuator -> override gains
                b = copy.deepcopy(a)
                b.stiffness = float(kp)
                b.damping = float(kd)
                newacts.append(b)
            else:
                newacts.append(a)  # gripper untouched
        art.actuators = tuple(newacts)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    return env, cfg


def make_reader(env):
    robot = env.scene["robot"]
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
        return env.scene["nail_block"].data.site_pos_w[:, ncfg.site_ids].squeeze(1)

    rec = []
    orig = env.metrics_manager.compute_substep
    def patched():
        orig()
        qfrc = robot.data._joint_dof_field("qfrc_constraint")[0, jid].clone()
        qvel = robot.data._joint_dof_field("qvel")[0, jid].clone()
        f = netf.data.force
        f_ax = float((f * axis).sum(-1).sum(-1).clamp_min(0.0)[0])
        rec.append({"in_c": bool((contact.data.found > 0).any()),
                    "qfrc": qfrc, "qvel": qvel, "f_ax": f_ax, "head_z": float(head()[0, 2])})
    env.metrics_manager.compute_substep = patched  # type: ignore
    return robot, head, nail_top, rec, dt, orig


def drive_strike(env, head, nail_top, rec, speed=1.0, height=0.10, hold=6, mode="strike", lift_at=None, lift_len=4):
    env.reset()
    ref = SingleStrikeReference(1, env.device, approach_height=height)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length()
    rec.clear()
    lifted_done = False
    for k in range(1, n + hold + 1):
        target = ref.playback_target(min(k, n))
        action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE * speed).clamp(-1.0, 1.0)
        if mode == "press" and k >= n:  # keep pressing the last target (hold down hard)
            action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE * speed).clamp(-1.0, 1.0)
        if mode == "lift_restrike" and lift_at is not None and k >= lift_at and not lifted_done:
            action = torch.tensor([[0.0, 0.0, 1.0]])  # command straight up (lift off)
            if k >= lift_at + lift_len:
                lifted_done = True
        env.step(action)
        if int(env.episode_length_buf[0]) == 0:
            break
    if mode == "press":  # extra hold steps pressing the bottom target
        for _ in range(80):
            target = ref.playback_target(n)
            action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE * speed).clamp(-1.0, 1.0)
            env.step(action)
            if int(env.episode_length_buf[0]) == 0:
                break
    return list(rec)


def windows_of(rec):
    wins, cur = [], []
    for i, r in enumerate(rec):
        if r["in_c"]: cur.append(i)
        elif cur: wins.append(cur); cur = []
    if cur: wins.append(cur)
    return wins


def analyze(rec, dt):
    wins = windows_of(rec)
    if not wins:
        return {"n_windows": 0}
    w0 = wins[0]; i0, i1 = w0[0], w0[-1]
    fs = [rec[i]["f_ax"] for i in w0]
    peak_f = max(fs); mean_f = sum(fs) / len(fs)
    rect = torch.zeros(6)
    for i in w0: rect += rec[i]["qfrc"].abs() * dt
    qv_pre = rec[i0 - 1]["qvel"] if i0 >= 1 else rec[i0]["qvel"]
    # approach speed at contact (head, +down)
    v_touch = (rec[i0 - 1]["head_z"] - rec[i0]["head_z"]) / dt if i0 >= 1 else float("nan")
    # head velocity trajectory through window
    hz = [rec[i]["head_z"] for i in ([i0 - 1] if i0 >= 1 else []) + w0]
    hv = [-(hz[k] - hz[k - 1]) / dt for k in range(1, len(hz))]
    v_start = hv[0] if hv else float("nan"); v_end = hv[-1] if hv else float("nan")
    v_min = min(hv) if hv else float("nan")  # most-decelerated (or negative=rebound)
    decay_frac = (v_start - v_end) / v_start if hv and abs(v_start) > 1e-9 else float("nan")
    rebounded = bool(v_min < -0.02)  # head reversed = elastic bounce
    # Δq̇ concentration in first 2 substeps
    steps = ([rec[i0 - 1]] if i0 >= 1 else []) + [rec[i] for i in w0]
    per = [(steps[k]["qvel"] - steps[k - 1]["qvel"]).abs() for k in range(1, len(steps))]
    if per:
        cum = torch.stack(per).sum(0); first2 = torch.stack(per[:2]).sum(0) if len(per) >= 2 else per[0]
        conc = (first2 / cum.clamp_min(1e-9))
        conc_j24 = [float(conc[1]), float(conc[3])]
    else:
        conc_j24 = [float("nan"), float("nan")]
    return {
        "n_windows": len(wins),
        "window_lengths": [len(w) for w in wins],
        "gaps": [wins[j + 1][0] - wins[j][-1] - 1 for j in range(len(wins) - 1)],
        "win0_ms": len(w0) * dt * 1e3,
        "approach_speed_at_contact": v_touch,
        "peak_over_mean_F": peak_f / max(mean_f, 1e-9),
        "peak_F": peak_f,
        "head_v_decay_frac": decay_frac,
        "head_v_min": v_min,
        "rebounded": rebounded,
        "dqv_conc_first2_j2j4": conc_j24,
        "rect_Lambda_j2j3j4": [float(rect[1]), float(rect[2]), float(rect[3])],
        "rect_Lambda_max": float(rect.max()),
        "head_v_traj": [round(v, 3) for v in hv],
    }


def press_severity(rec, dt):
    """T3: how large does the running per-joint Λ grow under a sustained press (single unbroken
    window)? This is the raw magnitude that would feed the normalizer margin."""
    wins = windows_of(rec)
    if not wins: return {"n_windows": 0}
    # longest window = the sustained press
    w = max(wins, key=len)
    run = torch.zeros(6); traj = []
    for i in w:
        run = run + rec[i]["qfrc"].abs() * dt
        traj.append(float(run.max()))
    return {"n_windows": len(wins), "press_window_substeps": len(w),
            "press_window_ms": len(w) * dt * 1e3,
            "Lambda_running_max_final": float(run.max()),
            "Lambda_perjoint_final": run.tolist(),
            "Lambda_growth_traj_max": [round(t, 4) for t in traj[::max(1, len(traj)//12)]]}


results = {}

# ---- T1 speed sweep (default actuator) ----
print("=== T1: APPROACH-SPEED SWEEP (default PD kp=1000/kd=100) ===")
env, cfg = build_env()
robot, head, nail_top, rec, dt, orig = make_reader(env)
t1 = []
for sp in [0.5, 1.0, 2.0, 4.0, 8.0]:
    r = drive_strike(env, head, nail_top, rec, speed=sp, height=0.10, mode="strike")
    a = analyze(r, dt); a["speed_factor"] = sp; t1.append(a)
    if a["n_windows"]:
        print(f"  sf={sp}: v_touch={a['approach_speed_at_contact']:.2f}m/s win={a['win0_ms']:.0f}ms "
              f"peak/mean_F={a['peak_over_mean_F']:.2f} vdecay={a['head_v_decay_frac']:.2f} "
              f"rebound={a['rebounded']} conc(j2,j4)={a['dqv_conc_first2_j2j4'][0]:.2f},{a['dqv_conc_first2_j2j4'][1]:.2f} "
              f"Λmax={a['rect_Lambda_max']:.3f} nwin={a['n_windows']}")
    else:
        print(f"  sf={sp}: NO CONTACT")
env.metrics_manager.compute_substep = orig  # type: ignore
results["T1_speed"] = t1

# ---- T3 sustained press severity (default) ----
print("=== T3: SUSTAINED-PRESS Λ GROWTH (EXPLOIT-3 severity, default PD) ===")
env, cfg = build_env()
robot, head, nail_top, rec, dt, orig = make_reader(env)
r = drive_strike(env, head, nail_top, rec, speed=2.0, height=0.10, mode="press")
p = press_severity(r, dt)
print(f"  press_window={p.get('press_window_ms',0):.0f}ms Λ_running_max_final={p.get('Lambda_running_max_final',0):.3f} "
      f"growth={p.get('Lambda_growth_traj_max')}")
env.metrics_manager.compute_substep = orig  # type: ignore
results["T3_press"] = p

# ---- T4 lift-restrike reachability (default) ----
print("=== T4: LIFT-RESTRIKE (C1/I1 reachability, default PD) ===")
env, cfg = build_env()
robot, head, nail_top, rec, dt, orig = make_reader(env)
# find contact onset step count first via a normal strike, then lift mid-contact
r = drive_strike(env, head, nail_top, rec, speed=2.0, height=0.10, mode="lift_restrike", lift_at=None)
env.metrics_manager.compute_substep = orig  # type: ignore
# lift_at picked from T1 window; rerun with a lift a few control steps in
results["T4_note"] = "lift-restrike run (see n_windows); lift_at heuristic"
a4 = analyze(r, dt)
print(f"  (baseline, no lift): n_windows={a4.get('n_windows')} lens={a4.get('window_lengths')}")

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=1)
print("JSON:", OUT)
