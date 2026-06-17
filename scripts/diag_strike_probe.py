"""Strike-vs-press probes for the Z1 hammer task (Path-A follow-up).

Two modes:

  --mode max_vel       Open-loop ceiling probe (NO policy). Optionally lift the
                       hammer first (more runway), then command max downward
                       action every step and measure the head's achievable
                       axial speed at contact. Answers: can this position-only
                       action space generate a HARD impact at all, or is it
                       fundamentally a gentle pusher?

  --mode press_basin   Load a trained checkpoint, SCRIPT the hammer slowly down
                       onto the nail and hold (v~0, in contact = the "press
                       basin"), then hand control to the trained policy. Answers:
                       does the policy LIFT and strike (striking is a genuine
                       global preference) or KEEP PUSHING (the strike only won
                       because the easy reset never tempted a press)?

Usage:
  python scripts/diag_strike_probe.py --mode max_vel --task Unitree-Z1-Hammer --device cpu --lift-steps 0
  python scripts/diag_strike_probe.py --mode max_vel --task Unitree-Z1-Hammer --device cpu --lift-steps 10
  python scripts/diag_strike_probe.py --mode press_basin --task Unitree-Z1-Hammer \
      --ckpt /tmp/ckpt/a_base/model_499.pt --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.managers.scene_entity_config import SceneEntityCfg

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME
from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD, NAIL_GOAL_DEPTH


def build(task, num_envs, device, play=True, no_term=False, delta_scale=None):
    env_cfg = load_env_cfg(task, play=play)
    if no_term:
        env_cfg.terminations = {}
    if delta_scale is not None:
        env_cfg.actions["ik_hammer_head"].delta_pos_scale = float(delta_scale)
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return env, agent_cfg


def load_policy(env, agent_cfg, task, ckpt, device):
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


def accessors(env):
    u = env.unwrapped
    rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(u.scene)
    robot = u.scene["robot"]; nail = u.scene["nail_block"]; sensor = u.scene["hammer_nail_contact"]
    head = lambda: robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
    depth = lambda: nail.data.joint_pos[:, 0]
    found = lambda: (sensor.data.found > 0).any(-1)
    return u, head, depth, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["max_vel", "press_basin"], required=True)
    ap.add_argument("--task", default="Unitree-Z1-Hammer")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--nsteps", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lift-steps", type=int, default=0, help="[max_vel] steps of max-up before max-down")
    ap.add_argument("--delta-scale", type=float, default=None, help="override action delta_pos_scale (default 0.05)")
    ap.add_argument("--no-term", action="store_true", help="[press_basin] disable terminations to watch post-contact behavior")
    args = ap.parse_args()

    if args.mode == "max_vel":
        run_max_vel(args)
    else:
        run_press_basin(args)


def run_max_vel(args):
    # No policy. Lift (optional) then command full downward action; measure achievable axial speed.
    env, _ = build(args.task, args.num_envs, args.device, play=True, delta_scale=args.delta_scale)
    u, head, depth, found = accessors(env)
    dt = float(u.step_dt)
    env.reset()
    A = env.unwrapped.action_space.shape[-1]
    up = torch.zeros(args.num_envs, A, device=u.device); up[:, 2] = +1.0
    down = torch.zeros(args.num_envs, A, device=u.device); down[:, 2] = -1.0

    print(f"[max_vel] task={args.task} lift_steps={args.lift_steps} dt={dt:.4f} "
          f"delta_pos_scale*clip/dt ceiling ~ {0.05/dt:.2f} m/s (nominal)")
    prev = head().clone(); have_prev = False
    peak_v = 0.0; v_at_contact = None; contact_step = None; apex_z = None
    print(f"{'k':>3} {'phase':>5} {'head_z':>7} {'v_ax':>6} {'depth_mm':>8} {'contact':>7}")
    k = 0
    # lift
    for _ in range(args.lift_steps):
        k += 1
        env.step(up)
        h = head()
        v = float((-(h[0,2]-prev[0,2]))/dt) if have_prev else 0.0
        prev = h.clone(); have_prev = True
        print(f"{k:>3} {'up':>5} {float(h[0,2]):>7.4f} {v:>6.3f} {float(depth()[0])*1000:>8.2f} {str(bool(found()[0])):>7}")
    apex_z = float(head()[0,2])
    have_prev = False  # reset velocity baseline at the turn
    prev = head().clone()
    # descend at max
    for _ in range(args.nsteps):
        k += 1
        env.step(down)
        h = head(); d = depth(); c = bool(found()[0])
        v = float((-(h[0,2]-prev[0,2]))/dt) if have_prev else 0.0
        prev = h.clone(); have_prev = True
        peak_v = max(peak_v, v)
        if c and contact_step is None:
            contact_step = k; v_at_contact = v
        print(f"{k:>3} {'down':>5} {float(h[0,2]):>7.4f} {v:>6.3f} {float(d[0])*1000:>8.2f} {str(c):>7}")
        if float(d[0]) >= NAIL_SUCCESS_THRESHOLD or (contact_step is not None and k > contact_step + 4):
            break
    print(f"\n[max_vel] apex_z={apex_z:.4f}  peak axial speed (control-rate, lower bound) = {peak_v:.3f} m/s")
    print(f"[max_vel] axial speed AT first contact = {v_at_contact}  (contact step {contact_step})")
    print(f"[max_vel] NOTE control-rate finite-diff underestimates the substep contact-instant speed.")
    env.close()


def run_press_basin(args):
    assert args.ckpt, "--ckpt required for press_basin"
    n = max(args.num_envs, 1)
    env, agent_cfg = build(args.task, n, args.device, play=True, no_term=args.no_term)
    policy = load_policy(env, agent_cfg, args.task, args.ckpt, args.device)
    u, head, depth, found = accessors(env)
    dt = float(u.step_dt)
    A = env.unwrapped.action_space.shape[-1]
    obs, _ = env.reset()

    slow = torch.zeros(n, A, device=u.device); slow[:, 2] = -0.1   # ~0.25 m/s descent
    hold = torch.zeros(n, A, device=u.device)

    # --- warmup: drive slowly to contact, then hold to settle into the press basin (v~0) ---
    print(f"[press_basin] task={args.task} ckpt={args.ckpt}")
    print("[warmup] slow descent to contact, then hold...")
    for k in range(60):
        obs, *_ = env.step(slow)
        if bool(found()[0]):
            break
    # drive a few mm in (stay well below the 27mm success threshold), then hold to damp velocity
    for _ in range(2):
        obs, *_ = env.step(slow)
    for _ in range(4):
        obs, *_ = env.step(hold)
    basin_z = float(head()[0, 2]); basin_depth = float(depth()[0]) * 1000
    print(f"[warmup] basin: head_z={basin_z:.4f}  nail_depth={basin_depth:.2f}mm  "
          f"contact={bool(found()[0])}  (success thr {NAIL_SUCCESS_THRESHOLD*1000:.0f}mm)")

    # --- hand over to the trained policy; watch lift-vs-press ---
    print("\n[policy] handing control to trained policy:")
    print(f"{'k':>3} {'head_z':>7} {'lift_mm':>7} {'v_ax':>6} {'depth_mm':>8} {'contact':>7}")
    prev = head().clone(); have_prev = False
    max_lift = 0.0; fresh_contacts = 0; prev_found = bool(found()[0]); success_step = None
    broke_contact = False
    for k in range(1, args.nsteps + 1):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)
        h = head(); d = depth(); c = bool(found()[0])
        v = float((-(h[0,2]-prev[0,2]))/dt) if have_prev else 0.0
        prev = h.clone(); have_prev = True
        lift_mm = (float(h[0,2]) - basin_z) * 1000
        max_lift = max(max_lift, lift_mm)
        if not c:
            broke_contact = True
        if c and (not prev_found):
            fresh_contacts += 1
        prev_found = c
        if success_step is None and float(d[0]) >= NAIL_SUCCESS_THRESHOLD:
            success_step = k
        print(f"{k:>3} {float(h[0,2]):>7.4f} {lift_mm:>7.1f} {v:>6.3f} {float(d[0])*1000:>8.2f} {str(c):>7}")
        if bool(dones[0]):
            print(f"  -- env reset (done) at step {k} --")
            break

    print(f"\n[press_basin] max head-lift above basin = {max_lift:.1f} mm")
    print(f"[press_basin] broke contact after handover = {broke_contact}; fresh contacts = {fresh_contacts}")
    print(f"[press_basin] success_step = {success_step}")
    verdict = ("LIFT-AND-STRIKE (striking is a global preference)"
               if (max_lift > 30.0 and fresh_contacts >= 1)
               else ("PRESS (stays in contact, pushes nail in)"
                     if max_lift < 10.0 else "AMBIGUOUS (partial lift)"))
    print(f"[press_basin] VERDICT: {verdict}")
    env.close()


if __name__ == "__main__":
    main()
