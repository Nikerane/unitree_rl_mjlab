"""Held-out first-episode evaluation; no learning, no load-safety claim."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np


def first_episode_peaks(peaks, done):
    alive = np.ones(done.shape[1], dtype=bool)
    maximum = np.zeros_like(peaks[0])
    for value, terminal in zip(peaks, done, strict=True):
        maximum = np.maximum(maximum, np.where(alive[:,None],value,0))
        alive &= ~terminal
    return maximum


def control_steps(episode_length, cap=None):
    if cap is not None and (isinstance(cap,bool) or not isinstance(cap,int) or cap <= 0):
        raise ValueError("smoke step cap must be a positive integer")
    return episode_length+1 if cap is None else min(episode_length+1,cap)


def condition(checkpoint, seed, goal, num_envs, device, max_steps=None):
    import torch
    from tensordict import TensorDict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_runner_cls
    from pilot import config, TASK
    from src.tasks.hammer.mdp.sampled_pd500 import install_sampled_pd500
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from scripts.evaluate_impact_speed import (
        EpisodeAccumulator, PreResetSnapshot, physical_snapshot,
        override_latched_goal, _reset_evidence)
    cfg = config(num_envs)
    cfg.env.seed = seed
    env = ManagerBasedRlEnv(cfg.env,device=device)
    try:
        install_sampled_pd500(env)
        wrapper = wrap_impact_speed_env(env,1.)
        runner = load_runner_cls(TASK)(wrapper,asdict(cfg.agent),device=device)
        runner.load(str(checkpoint),load_cfg={"actor":True},strict=True,map_location=device)
        contract = build_impact_speed_metadata(env,1.)
        env.seed(seed)
        obs,_ = wrapper.reset()
        hashes = _reset_evidence(env,obs)
        obs = TensorDict(override_latched_goal(env,goal),batch_size=[num_envs])
        policy = runner.get_inference_policy(device=device)
        accumulator = EpisodeAccumulator(num_envs,gamma=cfg.agent.algorithm.gamma,coefficient=.8493676)
        def snapshot(e):
            return {**physical_snapshot(e), "joint_speed_peak":e._hammer_substep_peak_qv.peak_qv_joint}
        peaks, terminals = [], []
        with torch.inference_mode(), PreResetSnapshot(env,reader=snapshot) as capture:
            for _ in range(control_steps(env.max_episode_length,max_steps)):
                pending = env.action_manager.get_term("strike_speed").state.pending.clone()
                raw = policy(obs)
                obs,reward,done,extras = wrapper.step(raw)
                assert torch.isfinite(raw).all() and torch.isfinite(reward).all()
                assert all(torch.isfinite(v).all() for v in capture.latest.values())
                accumulator.update(extras["impact_speed_outcome"],capture.latest,raw,pending)
                peaks.append(capture.latest["joint_speed_peak"].cpu().numpy())
                terminals.append(done.bool().cpu().numpy())
                if all(n >= 1 for n in accumulator.episodes):
                    break
        rows = sorted((r for r in accumulator.finish() if r["episode"] == 0),key=lambda r:r["world"])
        assert len(rows) == num_envs
        speeds = first_episode_peaks(np.stack(peaks),np.stack(terminals))
        for row, peak, reset in zip(rows,speeds,hashes,strict=True):
            row.update(seed=seed,goal_mps=goal,reset_hash=reset,joint_speed_peak_rad_s=peak.tolist())
        assert build_impact_speed_metadata(env,1.) == contract
        return dict(seed=seed,goal=goal,episodes=num_envs,hits=sum(r["hit"] for r in rows),
                    productive=sum(r["productive"] for r in rows),episode_censored=sum(not r["complete"] for r in rows),
                    native_finalized_first_strikes=sum(r["finalized"] for r in rows),
                    incomplete_first_strikes=sum(r["incomplete"] for r in rows),
                    all_joint_speed_safe=int((speeds <= 3.1415).all(axis=1).sum()),
                    joint_speed_violations=(speeds > 3.1415).sum(axis=0).tolist(),
                    joint_speed_peak_rad_s=speeds.max(axis=0).tolist(),
                    delivered_impulse_all_first_episodes_median_ns=float(np.median([r["delivered_impulse_ns"] for r in rows])),
                    tracking_rms_median_mm=float(np.median([r["tracking_rms_mm"] for r in rows])),
                    contract=contract,rows=rows)
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--num-envs",type=int,default=128)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-steps",type=int,help="software-smoke cap only; omit for complete first episodes")
    args = p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    summaries = []
    for seed in (2026092401,2026092402):
        initial_hashes = None
        for goal in (1.5,1.85,2.2):
            result = condition(args.checkpoint,seed,goal,args.num_envs,args.device,args.max_steps)
            hashes = [r["reset_hash"] for r in result["rows"]]
            if initial_hashes is None:
                initial_hashes = hashes
            assert hashes == initial_hashes
            (args.output / f"seed{seed}-goal{goal}.json").write_text(json.dumps(result,indent=2,allow_nan=False))
            summaries.append({k:v for k,v in result.items() if k not in ("rows","contract")})
            print(json.dumps(summaries[-1]),flush=True)
    (args.output / "summary.json").write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        conditions=summaries,smoke_step_cap=args.max_steps,
        claim="one training seed; load safety unqualified; no learning in evaluation"),indent=2))
