"""Bounded paired fixed/VIC campaign; no retries or automatic sweep expansion."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

TASK = "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed"


def run_spec(index):
    if index not in range(6):
        raise ValueError("array index must be 0..5")
    return dict(gain_mode=("fixed", "variable")[index % 2], seed=10 + index // 2,
                iterations=50, worlds=1024)


def training_config(spec):
    import mjlab.tasks
    import src.tasks
    from scripts.train import TrainConfig
    cfg = replace(TrainConfig.from_task(TASK), sampled_pd500=True,
                  sampled_pd500_gain_mode=spec["gain_mode"], speed_bounds=(1.5, 2.2),
                  speed_cost=.8493676, speed_mode="external")
    cfg.env.scene.num_envs = spec["worlds"]
    cfg.agent.seed = spec["seed"]
    cfg.agent.max_iterations = spec["iterations"]
    cfg.agent.save_interval = 25
    cfg.agent.logger = "tensorboard"
    return cfg


def evaluation_command(checkpoint, output, gain_mode, protocol, smoke):
    command = [sys.executable, "scripts/evaluate_impact_speed.py", "--task", TASK,
        "--checkpoint", str(checkpoint), "--speed-bounds", "1.5", "2.2",
        "--speed-cost", ".8493676", "--seed", "20260924", "--num-envs", "8" if smoke else "128",
        "--mode", "smoke" if smoke else "external",
        "--output", str(output), "--device", "cuda:0",
        "--sampled-pd500", "--sampled-pd500-gain-mode", gain_mode,
        "--measurement-protocol", protocol]
    if not smoke:
        command.extend(["--goals", "1.5", "1.85", "2.2"])
    return command


def configured_env(spec):
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.rl.impact_speed_wrapper import (
        configure_impact_speed_experiment, configure_impact_speed_controller,
        install_impact_speed_controller, wrap_impact_speed_env)
    cfg = training_config(dict(spec, worlds=8))
    cfg.env.seed = spec["seed"]
    configure_impact_speed_experiment(cfg.env, bounds=cfg.speed_bounds,
                                    cost=cfg.speed_cost, mode=cfg.speed_mode)
    configure_impact_speed_controller(cfg.env, sampled_pd500=True, gain_mode=spec["gain_mode"])
    env = ManagerBasedRlEnv(cfg.env, device="cuda:0")
    install_impact_speed_controller(env, sampled_pd500=True, gain_mode=spec["gain_mode"])
    return cfg, env, wrap_impact_speed_env(env, cfg.agent.clip_actions)


def preflight(spec):
    import torch
    cfg, env, wrapper = configured_env(spec)
    try:
        hold = env._sampled_pd500
        wrapper.reset()
        assert env.scene["hammer_nail_contact"].primary_names
        original = env.metrics_manager.compute_substep
        previous = None

        def verify():
            nonlocal previous
            original()
            actual = env.sim.data.actuator_force[:]
            torch.testing.assert_close(actual, hold.torque, rtol=0, atol=1e-5)
            if (hold.count - 1) % 8:
                assert torch.equal(actual, previous)
            previous = actual.clone()

        env.metrics_manager.compute_substep = verify
        with torch.inference_mode():
            for _ in range(4):
                obs, reward, _, _ = wrapper.step(torch.rand(8, 13, device="cuda:0") * 2 - 1)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(v).all() for v in obs.values())
            wrapper.reset()
            assert hold.count == 0 and hold.torque is None
            wrapper.step(torch.zeros(8, 13, device="cuda:0"))
        assert hold.count == 80 and hold.updates == 10
        print("GPU_CONTROLLER_PREFLIGHT_PASS", spec, flush=True)
    finally:
        env.close()


def train_and_evaluate(spec, output, smoke=False):
    import torch
    from mjlab.tasks.registry import load_runner_cls
    from scripts.train import run_train
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    output.mkdir(parents=True, exist_ok=False)
    cfg = training_config(spec)
    run_train(TASK, cfg, output)
    checkpoint = output / f"model_{spec['iterations'] - 1}.pt"
    cfg, env, wrapper = configured_env(spec)
    try:
        runner = load_runner_cls(TASK)(wrapper, asdict(cfg.agent), device="cuda:0")
        runner.load(str(checkpoint), strict=True, map_location="cuda:0")
        obs, _ = wrapper.reset()
        with torch.inference_mode():
            action = runner.get_inference_policy(device="cuda:0")(obs)
            assert action.shape == (8, 13) and torch.isfinite(action).all()
            wrapper.step(action)
        contract = build_impact_speed_metadata(env, cfg.agent.clip_actions)
    finally:
        env.close()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["iter"] == spec["iterations"] - 1
    for protocol in ("native", "common-window"):
        subprocess.run(evaluation_command(checkpoint, output / f"eval-{protocol}",
                                          spec["gain_mode"], protocol, smoke), check=True)
    result = dict(spec, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  reload="passed", evaluation="completed", smoke=smoke, contract=contract)
    (output / "completion.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("CAMPAIGN_RUN_COMPLETED", json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("preflight", "train"), required=True)
    p.add_argument("--index", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.phase == "preflight":
        args.output.mkdir(parents=True, exist_ok=False)
        for index in (0, 1):
            spec = dict(run_spec(index), iterations=2, worlds=32)
            preflight(spec)
            train_and_evaluate(spec, args.output / spec["gain_mode"], smoke=True)
        (args.output / "PASS").touch(exist_ok=False)
    else:
        train_and_evaluate(run_spec(args.index), args.output)


if __name__ == "__main__":
    main()
