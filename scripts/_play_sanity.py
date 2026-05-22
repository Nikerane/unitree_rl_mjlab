"""Non-interactive play sanity: zero and random agents step without viewer."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

task = "Unitree-Z1-Hammer"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
for agent in ("zero", "random"):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent_cfg = load_rl_cfg(task)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.reset()
    shape = env.unwrapped.action_space.shape
    for _ in range(10):
        if agent == "zero":
            act = torch.zeros(shape, device=device)
        else:
            act = 2 * torch.rand(shape, device=device) - 1
        step_out = env.step(act)
        rew = step_out[1]
    print(f"OK play sanity agent={agent} rew_mean={rew.mean().item():.4f}")
    env.close()
