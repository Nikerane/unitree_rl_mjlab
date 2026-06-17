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

# A-TRACK arm: A-BASE + the weak-annealed tracking prior r_imit (plan T2).
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-Track",
    env_cfg=z1_hammer_env_cfg(imitation=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, imitation=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)

# Velocity-bound ablation arms (keep delta_pos_scale=0.15; bound joint velocity to the
# real 3.1415 rad/s Z1 limit as a constraint). See docs/research/reward-design/
# JOINT_VELOCITY_BOUND_RESEARCH.md. A1 (delta->0.10) needs no task -- it is a CLI override.
# A2: annealed squared joint-velocity-excess reward penalty.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-VPenalty",
    env_cfg=z1_hammer_env_cfg(vel_penalty=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, vel_penalty=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)

# A3: Constraints-as-Terminations on joint velocity (CaT, Chane-Sane et al. IROS 2024).
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT",
    env_cfg=z1_hammer_env_cfg(cat_vel=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, cat_vel=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)
