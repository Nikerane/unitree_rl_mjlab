"""Z1 hammer-nail task configurations."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.config.z1.rl_cfg import z1_hammer_ppo_runner_cfg
from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

register_mjlab_task(
    task_id="Unitree-Z1-Hammer",
    env_cfg=z1_hammer_env_cfg(),
    play_env_cfg=z1_hammer_env_cfg(play=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)
