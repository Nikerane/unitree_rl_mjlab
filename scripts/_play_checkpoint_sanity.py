"""Load a trained checkpoint and step without opening a viewer."""
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

task = "Unitree-Z1-Hammer"
ckpt = Path(sys.argv[1])
device = "cpu"
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(task)
env = ManagerBasedRlEnv(cfg=cfg, device=device)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)
obs, _ = env.reset()
for _ in range(20):
    with torch.inference_mode():
        actions = policy(obs)
    step_out = env.step(actions)
    obs = step_out[0]
    rew = step_out[1]
print(f"OK checkpoint play ckpt={ckpt.name} rew_mean={rew.mean().item():.4f} action_std={actions.std().item():.4f}")
