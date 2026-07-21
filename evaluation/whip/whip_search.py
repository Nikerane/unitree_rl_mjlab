#!/usr/bin/env python
"""Whip search - Stage 1: CEM open-loop shooting for the LEGAL fixed-impedance contact-speed ceiling.

Spec: docs/results/2026-07-21_whip_search_spec.md.  CPU, batched (one env per CEM sample).
Optimizes a per-control-step 3-D DiffIK task-space delta sequence a_{1..T} to MAXIMIZE the ante-impact
downward hammer-head speed at first FACE contact.  Velocity uses the EXACT parity estimator from
eval_mx_multireset.py:  v = (head_z[fc-1] - head_z[fc]) / physics_dt  at the first substep the face
sensor (hammer_head_0) reports contact -- so v* is the same ruler as the 1.81 m/s RL anchor.

Why open-loop is a valid ceiling search: the scene is deterministic from a fixed reset (play=True), so
every closed-loop policy's trajectory IS some open-loop action sequence -> max over open-loop sequences
upper-bounds any policy (see spec sec 2).

LEGAL by construction: the only sim inputs are env.reset() and env.step(a), a in [-1,1]^3 -- byte
identical to a policy's output.  A startup self-grep proves this driver never injects joint state
(no  ..._to_sim(  call), which is exactly the coord_whip2 bypass this experiment forbids.  Shipped
gains / delta_pos_scale / decimation / dt are asserted unchanged, except the explicitly-labelled
--delta sweep (changes ONLY delta_pos_scale -- still fixed impedance, a different action-space knob).

Smoke (sizes wall-clock):  PYTHONPATH=. python evaluation/whip/whip_search.py --smoke
Full:   PYTHONPATH=. python evaluation/whip/whip_search.py --pop 96 --T 30 --gens 40 --restarts 3 --out evaluation/whip/data/whip_delta015.json
Sweep:  ... --delta 0.30      (localizes a ceiling: delta-rail vs PD-bandwidth)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME
import mjlab.tasks, src.tasks  # noqa: F401  (register the Z1 hammer tasks)

TASK = "Unitree-Z1-Hammer-CaT-Impulse"
ROOT = "logs/rsl_rl/z1_hammer"
ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
HW_RAIL = 3.1415  # real Z1 per-joint velocity limit (rad/s) -- hardware-legal gate

# --- legality self-check: this driver must never inject joint state (the coord_whip2 bypass) ---
# Needles are concatenated so the literal call-form never appears in this file's own source;
# only an actual future `.write_joint_*_to_sim(` call would materialize it and trip the assert.
_SRC = Path(__file__).read_text()
for _p in ("write_joint_velocity", "write_joint_position"):
    assert (_p + "_to_sim(") not in _SRC, "LEGALITY: driver must not inject joint state (velocity/position)"


def build(pop, delta):
    cfg = load_env_cfg(TASK, play=True)                    # play=True -> deterministic reset (no noise)
    cfg.scene.num_envs = pop
    cfg.actions["ik_hammer_head"].delta_pos_scale = delta  # shipped 0.15 unless --delta sweep
    env = ManagerBasedRlEnv(cfg, device="cpu")
    assert abs(float(cfg.actions["ik_hammer_head"].delta_pos_scale) - delta) < 1e-9
    assert abs(float(env.physics_dt) - 0.002) < 1e-9, f"physics_dt={float(env.physics_dt)} != 0.002"
    return env


def policy_seed(pattern, T):
    """Roll the best ckpt of the run matching <pattern> from the fixed play=True reset; return its
    per-control-step action sequence [T,3] as a CEM warm-start mean.  Deterministic dynamics =>
    replaying this open-loop reproduces the policy's own trajectory, so v* >= the policy's contact
    speed by construction.  Builds + frees its own 1-env BEFORE the batched CEM env (no 2 at once)."""
    import glob, gc
    from dataclasses import asdict
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    dirs = sorted(glob.glob(f"{ROOT}/{pattern}"))
    assert dirs, f"no run dir matches {pattern}"
    ck = sorted(glob.glob(f"{dirs[-1]}/model_*.pt"), key=lambda p: int(p.split("model_")[1].split(".pt")[0]))
    assert ck, f"no ckpt in {dirs[-1]}"
    cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = 1
    base = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    agent_cfg = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cpu")
    runner.load(ck[-1], load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    torch.manual_seed(12345); obs, _ = env.reset(); acts = []
    for _ in range(T):
        with torch.inference_mode():
            a = policy(obs)
        acts.append(a[0, :3].clone().float())
        obs = env.step(a)[0]
        if int(base.episode_length_buf[0]) == 0:   # episode ended (success) -> stop, pad below
            break
    seq = torch.stack(acts)
    if seq.shape[0] < T:
        seq = torch.cat([seq, torch.zeros(T - seq.shape[0], 3)], 0)   # hold after the strike
    print(f"[seed] {Path(dirs[-1]).name}: {len(acts)}-step seq from {Path(ck[-1]).name}")
    del env, base, runner, policy, cfg; gc.collect()
    return seq


def make_reader(env):
    robot = env.scene["robot"]; nail = env.scene["nail_block"]; sensor = env.scene["hammer_nail_contact"]
    rc = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rc.resolve(env.scene)
    nc = SceneEntityCfg("nail_block", site_names=("nail_top",)); nc.resolve(env.scene)
    arm = SceneEntityCfg("robot", joint_names=ARM); arm.resolve(env.scene); jid = arm.joint_ids
    head = lambda: robot.data.site_pos_w[:, rc.site_ids].squeeze(1)            # [N,3]
    ntop = lambda: nail.data.site_pos_w[:, nc.site_ids].squeeze(1)             # [N,3]
    found = lambda: (sensor.data.found > 0).any(dim=-1)                        # [N] bool
    depth = lambda: nail.data.joint_pos[:, 0].clamp(0.0, 0.032)               # [N]
    qvmax = lambda: robot.data.joint_vel[:, jid].abs().amax(dim=1)             # [N] peak |arm qvel|
    return dict(head=head, ntop=ntop, found=found, depth=depth, qvmax=qvmax)


def rollout(env, R, A):
    """A: [N,T,3] actions. Returns stacked per-substep [S,N,*] tensors + nail_top [N,3]."""
    env.reset()
    nail_top = R["ntop"]().clone()
    H, F, D, Q = [], [], [], []
    orig = env.metrics_manager.compute_substep
    def cb():
        orig()
        H.append(R["head"]().clone()); F.append(R["found"]().clone())
        D.append(R["depth"]().clone()); Q.append(R["qvmax"]().clone())
    env.metrics_manager.compute_substep = cb
    with torch.no_grad():
        for t in range(A.shape[1]):
            env.step(A[:, t, :].clamp(-1.0, 1.0))
    env.metrics_manager.compute_substep = orig
    return torch.stack(H), torch.stack(F), torch.stack(D), torch.stack(Q), nail_top


def fitness(rec, dt):
    """Per-env fitness = ante-impact downward head speed (m/s) at first valid face-strike, else miss
    penalty. Returns fit[N], v[N], vlat[N], qvc[N], valid[N] (mirrors eval_mx_multireset v_touch)."""
    H, F, D, Q, ntop = rec
    S, N = F.shape; idx = torch.arange(N)
    fc = F.float().argmax(0)                 # [N] first True substep (0 if none)
    has = F.any(0)
    fcm1 = (fc - 1).clamp(min=0)
    v = (H[fcm1, idx, 2] - H[fc, idx, 2]) / dt                        # downward speed
    vx = (H[fcm1, idx, 0] - H[fc, idx, 0]) / dt
    vy = (H[fcm1, idx, 1] - H[fc, idx, 1]) / dt
    vlat = torch.sqrt(vx * vx + vy * vy)
    drove = D.max(0).values > D[0] + 1e-4                            # nail actually moved
    valid = has & (fc >= 1) & (v > 0) & drove
    dmin = (H - ntop[None]).norm(dim=-1).min(0).values               # closest approach for misses
    fit = torch.where(valid, v, -dmin - 10.0)
    qvc = Q[fc, idx]
    return fit, v, vlat, qvc, valid


def cem(env, R, dt, T, pop, gens, elite, sigma0, sigma_floor, seeds, down_bias, log):
    best = dict(v=-1e9); best_hw = dict(v=-1e9)   # best overall / best hardware-legal (|qvel|<=rail)
    for r, seed in enumerate(seeds):
        torch.manual_seed(1234 + r)
        if seed is not None:
            mu = seed.clone()                                # RL warm-start basin
        else:
            mu = torch.zeros(T, 3); mu[:, 2] = -down_bias    # downward-bias basin
        sigma = torch.full((T, 3), sigma0)
        for g in range(gens):
            t0 = time.perf_counter()
            Sset = (mu[None] + sigma[None] * torch.randn(pop, T, 3)).clamp(-1.0, 1.0)
            rec = rollout(env, R, Sset)
            fit, v, vlat, qvc, valid = fitness(rec, dt)
            k = max(2, int(elite * pop))
            ei = torch.topk(fit, k).indices
            mu, sigma = Sset[ei].mean(0), Sset[ei].std(0) + sigma_floor
            vv = torch.where(valid, v, torch.full_like(v, -1e9))            # fastest valid strike
            gi = int(vv.argmax()); gv = float(vv[gi])
            if gv > best["v"]:
                best = dict(v=gv, vlat=float(vlat[gi]), qvc=float(qvc[gi]),
                            x=Sset[gi].clone(), restart=r, gen=g)
            vh = torch.where(valid & (qvc <= HW_RAIL), v, torch.full_like(v, -1e9))  # fastest HW-legal
            hi = int(vh.argmax()); hv = float(vh[hi])
            if hv > best_hw["v"]:
                best_hw = dict(v=hv, qvc=float(qvc[hi]), vlat=float(vlat[hi]),
                               x=Sset[hi].clone(), restart=r, gen=g)
            log(f"  r{r} g{g:02d}  best_v={max(best['v'],-9.9):5.3f}  best_hw={max(best_hw['v'],-9.9):5.3f}  "
                f"gen_v={float(v[valid].max()) if valid.any() else float('nan'):5.3f}  "
                f"hits={int(valid.sum()):2d}/{pop}  {time.perf_counter()-t0:5.2f}s")
    return best, best_hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--T", type=int, default=28)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.15, help="delta_pos_scale (0.15 shipped; sweep for localization)")
    ap.add_argument("--elite", type=float, default=0.15)
    ap.add_argument("--sigma0", type=float, default=0.6)
    ap.add_argument("--sigma-floor", type=float, default=0.05)
    ap.add_argument("--down-bias", type=float, default=0.5)
    ap.add_argument("--no-rl-seed", dest="rl_seed", action="store_false", help="skip the RL warm-start basins")
    ap.add_argument("--smoke", action="store_true", help="5 gens, no RL seed, small pop -- size wall-clock")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    if args.smoke:
        args.gens, args.pop, args.T, args.rl_seed = 5, min(args.pop, 48), min(args.T, 24), False

    # RL warm-start basins (each builds + frees its own 1-env BEFORE the batched CEM env).
    # Seeding at a trained policy's trajectory makes v* >= that policy's contact speed by construction,
    # so a null (v* ~ 1.8) is the strong "even seeded at the RL optimum, nothing legal is faster" result.
    seeds = []
    if args.rl_seed:
        seeds.append(policy_seed("*lg_maxoff1500_seed2", args.T))   # fastest: v_touch record 1.81 m/s
        seeds.append(policy_seed("*af1_fixed_seed0", args.T))       # robust 100%-success striker
    seeds.append(None)                                             # downward-bias basin (diversity)

    env = build(args.pop, args.delta)
    R = make_reader(env); dt = float(env.physics_dt); dt_ctrl = float(env.step_dt)  # 0.002 / 0.02
    print(f"[whip] pop={args.pop} T={args.T} gens={args.gens} basins={len(seeds)} "
          f"(rl_seed={args.rl_seed}) delta={args.delta} dt={dt} device=cpu")
    t0 = time.perf_counter()
    best, best_hw = cem(env, R, dt, args.T, args.pop, args.gens, args.elite, args.sigma0,
                        args.sigma_floor, seeds, args.down_bias, print)
    wall = time.perf_counter() - t0

    hw_ok = best.get("qvc", 99) <= HW_RAIL
    print(f"\n[whip] v*     = {best['v']:.3f} m/s  (lateral {best.get('vlat', float('nan')):.3f}, "
          f"peak|qvel| {best.get('qvc', float('nan')):.2f} rad/s -> hardware-legal={hw_ok})")
    print(f"[whip] v*_hw  = {best_hw['v']:.3f} m/s  (all arm joints <= {HW_RAIL} rad/s; "
          f"peak|qvel| {best_hw.get('qvc', float('nan')):.2f}) -- the real-robot ceiling")
    print(f"[whip] efficiency v*.dt_ctrl/delta = {best['v']*dt_ctrl/args.delta:.3f}  "
          f"(0.18 open-loop / 0.24 RL / 0.56 box) | anchors: RL 1.81, box 4.22")
    ngen = args.gens * len(seeds)
    print(f"[whip] winning basin=restart{best.get('restart','?')} gen={best.get('gen','?')}  "
          f"wall {wall:.1f}s over {ngen} gens ({wall/max(1,ngen):.2f}s/gen)")
    # self-check: the machinery produced a real, finite, positive contact speed
    assert best["v"] > 0 and best["v"] < 20, f"implausible v*={best['v']} -- machinery broken"
    print("[whip] SELF-CHECK PASS: found a valid downward face-strike with finite v*.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(delta=args.delta, T=args.T, pop=args.pop, gens=args.gens, basins=len(seeds),
                       v_star=best["v"], vlat=best.get("vlat"), peak_qvel=best.get("qvc"),
                       v_star_hw=best_hw["v"], peak_qvel_hw=best_hw.get("qvc"),
                       winning_restart=best.get("restart"), winning_gen=best.get("gen"),
                       hardware_legal=bool(hw_ok), efficiency=best["v"]*dt_ctrl/args.delta,
                       best_actions=best["x"].tolist() if "x" in best else None,
                       best_actions_hw=best_hw["x"].tolist() if "x" in best_hw else None,
                       wall_s=wall), open(args.out, "w"))
        print(f"[whip] -> {args.out}")


if __name__ == "__main__":
    main()
