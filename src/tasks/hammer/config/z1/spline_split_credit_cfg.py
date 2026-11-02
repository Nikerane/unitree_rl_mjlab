"""Opt-in two-tranche tasks; all plant, observation and PPO settings unchanged."""
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_env_cfg,pilot_rl_cfg
from src.tasks.hammer.mdp.spline_split_credit import SplitSplineStrikeReward,validate_split_parameters

SETTINGS=('vic_baseline','vic_position_low')


def split_task_id(*,setting):
    if type(setting) is not str or setting not in SETTINGS:raise ValueError('split setting must be baseline or position_low')
    return 'Unitree-Z1-Hammer-Spline-SplitCredit50-200-Current-'+setting


def split_env_cfg(*,play,setting,early_fraction=.5):
    split_task_id(setting=setting);validate_split_parameters(early_fraction,50,200)
    cfg=pilot_env_cfg(play=play,setting=setting,credit_deadline_ms=50,critic_mode='current')
    term=cfg.rewards['accepted_spline_strike'];term.func=SplitSplineStrikeReward
    del term.params['credit_deadline_ms']
    term.params.update(early_fraction=early_fraction,early_deadline_ms=50,reserve_deadline_ms=200)
    return cfg


def split_rl_cfg(*,setting):
    split_task_id(setting=setting)
    return pilot_rl_cfg(setting=setting,critic_mode='current')


def register_split_tasks():
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    for setting in SETTINGS:
        register_mjlab_task(task_id=split_task_id(setting=setting),
            env_cfg=split_env_cfg(play=False,setting=setting),play_env_cfg=split_env_cfg(play=True,setting=setting),
            rl_cfg=split_rl_cfg(setting=setting),runner_cls=HammerOnPolicyRunner)
