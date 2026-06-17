"""Trace a TRAINED policy rollout to classify strike vs slam vs press (Path-A).

Loads a checkpoint, rolls out the env for a number of control steps, and logs:
  - a step-by-step trace of env 0 for the first episode (head z, axial impact
    speed, nail depth, contact, phase phi, deviation from the strike reference);
  - per-episode aggregates over ALL envs (length, max nail depth, # distinct
    contact events, peak axial impact speed at first contact, terminated-by,
    mean ante-impact deviation from the reference).

Usage:
  python scripts/diag_policy_trace.py --task Unitree-Z1-Hammer \
      --ckpt /tmp/ckpt/a_base/model_499.pt --num-envs 64 --nsteps 80 --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.managers.scene_entity_config import SceneEntityCfg

import mjlab.tasks  # noqa: F401  (register builtin tasks)
import src.tasks  # noqa: F401  (register hammer tasks)
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME
from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD, NAIL_GOAL_DEPTH
from src.tasks.hammer.mdp.references import get_strike_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Unitree-Z1-Hammer")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--play", action="store_true",
                    help="play mode (zeroed reset/obs noise -> deterministic)")
    ap.add_argument("--no-term", action="store_true",
                    help="disable terminations -> watch full depth trajectory + post-strike behavior")
    args = ap.parse_args()

    env_cfg = load_env_cfg(args.task, play=args.play)
    if args.no_term:
        env_cfg.terminations = {}
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = load_rl_cfg(args.task)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    u = env.unwrapped
    dt = float(u.step_dt)
    N = u.num_envs
    dev = u.device

    rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(u.scene)
    robot = u.scene["robot"]; nail = u.scene["nail_block"]; sensor = u.scene["hammer_nail_contact"]
    head = lambda: robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
    depth = lambda: nail.data.joint_pos[:, 0]

    thr = float(NAIL_SUCCESS_THRESHOLD)
    print(f"[cfg] task={args.task} ckpt={args.ckpt}")
    print(f"[cfg] num_envs={N} nsteps={args.nsteps} device={args.device} play={args.play} "
          f"dt={dt:.4f} success_thr={thr*1000:.1f}mm goal={NAIL_GOAL_DEPTH*1000:.1f}mm")

    obs, _ = env.reset()

    # per-env episode accumulators
    prev_head = head().clone()
    have_prev = torch.zeros(N, dtype=torch.bool, device=dev)
    ep_len = torch.zeros(N, device=dev)
    max_depth = torch.zeros(N, device=dev)
    n_contacts = torch.zeros(N, device=dev)
    prev_found = torch.zeros(N, dtype=torch.bool, device=dev)
    first_contact_done = torch.zeros(N, dtype=torch.bool, device=dev)
    peak_v_impact = torch.zeros(N, device=dev)   # axial speed on the FIRST contact step
    peak_v_any = torch.zeros(N, device=dev)       # max downward axial speed any step
    dev_sum = torch.zeros(N, device=dev)          # ante-impact deviation accumulator
    dev_cnt = torch.zeros(N, device=dev)

    # collected per-episode records
    rec_len, rec_depth, rec_nc, rec_vimp, rec_vany, rec_dev, rec_succ, rec_to = ([] for _ in range(8))
    running_max_depth = 0.0  # global max depth seen (useful in --no-term mode)

    trace_rows = []  # env-0 first-episode trace
    env0_done = False

    for k in range(1, args.nsteps + 1):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)
        # Exact success signal: read BEFORE the next compute() overwrites it.
        # .terminated = non-timeout terminations (= nail_driven success); .time_outs = truncation.
        succ_mask = u.termination_manager.terminated.clone()
        to_mask = u.termination_manager.time_outs.clone()

        h = head()
        v = torch.where(have_prev[:, None], (h - prev_head) / dt, torch.zeros_like(h))
        prev_head = h.clone(); have_prev.fill_(True)
        v_axial = (-v[:, 2]).clamp_min(0.0)   # downward (z up) -> positive

        d = depth()
        max_depth = torch.maximum(max_depth, d)
        running_max_depth = max(running_max_depth, float(d.max()) * 1000)
        found = (sensor.data.found > 0).any(-1)
        rising = found & (~prev_found)
        n_contacts += rising.float()
        prev_found = found.clone()

        # impact speed on the first contact step of the episode
        first_now = rising & (~first_contact_done)
        peak_v_impact = torch.where(first_now, v_axial, peak_v_impact)
        first_contact_done = first_contact_done | rising
        peak_v_any = torch.maximum(peak_v_any, v_axial)

        # deviation from reference (ante-impact only), via the shared reference
        ref = get_strike_reference(u)
        phi = ref._phi.clone()
        p_star = ref.waypoint(phi)
        dist = (h - p_star).norm(dim=-1)
        pre = ~first_contact_done
        dev_sum += torch.where(pre, dist, torch.zeros_like(dist))
        dev_cnt += pre.float()

        ep_len += 1.0

        # env-0 trace (first episode only)
        if not env0_done:
            trace_rows.append((k, float(h[0, 2]), float(v_axial[0]), float(d[0] * 1000),
                               bool(found[0]), float(phi[0]), float(dist[0] * 1000)))

        # handle episode completions
        done_idx = torch.nonzero(dones, as_tuple=False).flatten()
        if done_idx.numel() > 0:
            for i in done_idx.tolist():
                rec_len.append(float(ep_len[i]))
                rec_depth.append(float(max_depth[i] * 1000))  # pre-terminal max (lower bound)
                rec_nc.append(int(n_contacts[i].item()))
                rec_vimp.append(float(peak_v_impact[i]))
                rec_vany.append(float(peak_v_any[i]))
                rec_dev.append(float((dev_sum[i] / dev_cnt[i].clamp(min=1)) * 1000))
                rec_succ.append(bool(succ_mask[i]))  # exact: nail_driven termination
                rec_to.append(bool(to_mask[i]))
            # reset accumulators for completed envs
            ep_len[done_idx] = 0
            max_depth[done_idx] = 0
            n_contacts[done_idx] = 0
            prev_found[done_idx] = False
            first_contact_done[done_idx] = False
            peak_v_impact[done_idx] = 0
            peak_v_any[done_idx] = 0
            dev_sum[done_idx] = 0
            dev_cnt[done_idx] = 0
            have_prev[done_idx] = False
            if bool(dones[0]):
                env0_done = True

    # --- env-0 trace ---
    print("\n--- env 0 trace (first episode) ---")
    print(f"{'k':>3} {'head_z':>7} {'v_ax':>6} {'depth_mm':>8} {'contact':>7} {'phi':>5} {'dev_mm':>7}")
    for (k, hz, va, dm, c, ph, dv) in trace_rows:
        print(f"{k:>3} {hz:>7.4f} {va:>6.3f} {dm:>8.2f} {str(c):>7} {ph:>5.2f} {dv:>7.1f}")

    # --- aggregate ---
    def stat(x):
        t = torch.tensor(x, dtype=torch.float32)
        return float(t.mean()), float(t.std())
    n = len(rec_len)
    print(f"\n--- aggregate over {n} completed episodes ---")
    if n:
        ml, sl = stat(rec_len); md, sd = stat(rec_depth)
        mc, sc = stat([float(x) for x in rec_nc]); mvi, svi = stat(rec_vimp)
        mva, sva = stat(rec_vany); mdev, sdev = stat(rec_dev)
        succ = 100.0 * sum(rec_succ) / n
        to = 100.0 * sum(rec_to) / n
        # contact-count histogram
        from collections import Counter
        hist = Counter(rec_nc)
        hist_str = "  ".join(f"{kk}:{vv}" for kk, vv in sorted(hist.items()))
        print(f"  success rate (nail_driven): {succ:.1f}%   timeout: {to:.1f}%")
        print(f"  episode length (steps)  : {ml:.2f} +/- {sl:.2f}")
        print(f"  pre-terminal max depth  : {md:.2f} +/- {sd:.2f} mm  (lower bound; terminal >= {thr*1000:.0f})")
        print(f"  # distinct contacts/ep  : {mc:.2f} +/- {sc:.2f}")
        print(f"  contact-count histogram : {hist_str}")
        print(f"  peak v_axial @ contact  : {mvi:.3f} +/- {svi:.3f} m/s")
        print(f"  peak v_axial (any step) : {mva:.3f} +/- {sva:.3f} m/s")
        print(f"  ante-impact deviation   : {mdev:.1f} +/- {sdev:.1f} mm  (r_imit sigma=50mm)")
    print(f"\n  global max nail depth observed: {running_max_depth:.2f} mm"
          + ("   (no-term: this is the true depth ceiling)" if args.no_term else ""))
    env.close()


if __name__ == "__main__":
    main()
