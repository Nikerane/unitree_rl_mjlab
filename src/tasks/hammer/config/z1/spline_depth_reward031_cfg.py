"""Four explicit opt-in depth031 profiles; historical task defaults untouched."""
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.hammer.config.z1.spline_split_credit_cfg import split_env_cfg, split_rl_cfg
from src.tasks.hammer.mdp.depth_reward031 import (
    ARM_FLAGS, arm_flags, NativeDepthCreditObserver031, depth_collect_metric031,
    depth_progress_reward031, depth_completion_reward031,
)


def depth_task_id(arm):
    arm_flags(arm)
    return 'Unitree-Z1-Hammer-Spline-DepthReward031-' + arm


def depth_env_cfg(*, play, arm):
    progress, completion = arm_flags(arm)
    cfg = split_env_cfg(play=play, setting='vic_baseline')
    cfg.metrics['native_depth_owner031'] = MetricsTermCfg(
        func=NativeDepthCreditObserver031, per_substep=True, reduce='last',
        params={'arm': arm, 'nail_cfg': SceneEntityCfg('nail_block', joint_names=('nail_slide',))})
    cfg.metrics['native_depth_collect031'] = MetricsTermCfg(
        func=depth_collect_metric031, per_substep=False, reduce='last')
    cfg.rewards['native_depth_progress031'] = RewardTermCfg(func=depth_progress_reward031, weight=float(progress))
    cfg.rewards['native_depth_completion031'] = RewardTermCfg(func=depth_completion_reward031, weight=float(completion))
    return cfg


def register_depth_reward031_tasks():
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    for arm in ARM_FLAGS:
        register_mjlab_task(task_id=depth_task_id(arm), env_cfg=depth_env_cfg(play=False,arm=arm),
            play_env_cfg=depth_env_cfg(play=True,arm=arm), rl_cfg=split_rl_cfg(setting='vic_baseline'),
            runner_cls=HammerOnPolicyRunner)
