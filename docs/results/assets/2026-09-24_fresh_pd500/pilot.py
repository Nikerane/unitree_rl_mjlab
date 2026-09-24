"""One approved fixed-controller training pilot; smoke and full runs stay separate."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch
import mjlab.tasks
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_runner_cls
from scripts.train import TrainConfig, run_train
from src.tasks.hammer.mdp.sampled_pd500 import configure_sampled_pd500, install_sampled_pd500
from src.tasks.hammer.rl.impact_speed_wrapper import configure_impact_speed_experiment, wrap_impact_speed_env
from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata

TASK = "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed"


def config(num_envs):
    cfg = replace(TrainConfig.from_task(TASK), sampled_pd500=True,
                  speed_bounds=(1.5,2.2), speed_cost=.8493676, speed_mode="external")
    cfg.env.scene.num_envs = num_envs
    cfg.env.metrics["cat_soft"].params["imp_max_p"] = .2
    cfg.agent.seed = 2
    cfg.agent.save_interval = 25
    cfg.agent.logger = "tensorboard"
    configure_impact_speed_experiment(cfg.env,bounds=cfg.speed_bounds,cost=cfg.speed_cost,mode=cfg.speed_mode)
    configure_sampled_pd500(cfg.env)
    return cfg


def preflight():
    cfg = config(8)
    env = ManagerBasedRlEnv(cfg.env,device="cuda:0")
    try:
        hold = install_sampled_pd500(env)
        wrapper = wrap_impact_speed_env(env,1.)
        wrapper.reset()
        assert env.scene["hammer_nail_contact"].primary_names
        original = env.metrics_manager.compute_substep
        previous = None
        def verify():
            nonlocal previous
            original()
            actual = env.sim.data.actuator_force[:]
            torch.testing.assert_close(actual,hold.torque,rtol=0,atol=1e-5)
            if (hold.count-1) % 8:
                assert torch.equal(actual,previous)
            previous = actual.clone()
        env.metrics_manager.compute_substep = verify
        with torch.inference_mode():
            for _ in range(4):
                obs,reward,done,extras = wrapper.step(torch.rand(8,13,device="cuda:0")*2-1)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(v).all() for v in obs.values())
            wrapper.reset()
            assert hold.count == 0 and hold.torque is None
            wrapper.step(torch.zeros(8,13,device="cuda:0"))
        assert hold.count == 80 and hold.updates == 10
        print("ACTUAL_PROFILE_GPU_PREFLIGHT_PASS", json.dumps(build_impact_speed_metadata(env,1.)),flush=True)
    finally:
        env.close()


def reload_checkpoint(path):
    cfg = config(8)
    env = ManagerBasedRlEnv(cfg.env,device="cuda:0")
    try:
        install_sampled_pd500(env)
        wrapper = wrap_impact_speed_env(env,1.)
        runner = load_runner_cls(TASK)(wrapper,asdict(cfg.agent),device="cuda:0")
        runner.load(str(path),strict=True,map_location="cuda:0")
        obs,_ = wrapper.reset()
        with torch.inference_mode():
            action = runner.get_inference_policy(device="cuda:0")(obs)
            assert action.shape == (8,13) and torch.isfinite(action).all()
            wrapper.step(action)
        return build_impact_speed_metadata(env,1.)
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase",choices=("smoke","train"),required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    if args.phase == "smoke":
        preflight()
    cfg = config(32 if args.phase == "smoke" else 1024)
    cfg.agent.max_iterations = 2 if args.phase == "smoke" else 250
    run_train(TASK,cfg,args.output)
    checkpoint = args.output / f"model_{cfg.agent.max_iterations-1}.pt"
    contract = reload_checkpoint(checkpoint)
    payload = torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert payload["iter"] == cfg.agent.max_iterations-1
    result = dict(phase=args.phase,iterations=cfg.agent.max_iterations,seed=2,
                  checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  reload="passed",contract=contract)
    (args.output / "completion.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("PILOT_COMPLETED",json.dumps(result),flush=True)
