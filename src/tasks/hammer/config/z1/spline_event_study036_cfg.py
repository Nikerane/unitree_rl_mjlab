"""The four approved event-critic recipes; existing single-deadline schemas stay intact."""
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_env_cfg, pilot_rl_cfg
from src.tasks.hammer.config.z1.spline_split_credit_cfg import split_env_cfg
from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
from src.tasks.hammer.mdp.spline_split_event036 import split_event_observation

RECIPES = (('split','vic_baseline'), ('200','vic_baseline'),
           ('split','vic_position_low'), ('50','vic_baseline'))
_REGISTERED = False


def event_task_id(*, payout, setting):
    if type(payout) is not str or type(setting) is not str or (payout,setting) not in RECIPES:
        raise ValueError('event036 requires one of the four approved payout/setting recipes')
    return 'Unitree-Z1-Hammer-Spline-EventStudy036-'+payout+'-'+setting


def event_env_cfg(*, play, payout, setting):
    event_task_id(payout=payout, setting=setting)
    if payout != 'split':
        return pilot_env_cfg(play=play, setting=setting, credit_deadline_ms=int(payout), critic_mode='event')
    cfg = split_env_cfg(play=play, setting=setting)
    for field in CREDIT_EVENT_FIELDS:
        cfg.observations['critic'].terms[field] = ObservationTermCfg(
            func=split_event_observation, params={'field':field})
    return cfg


def register_event_study_tasks():
    global _REGISTERED
    if _REGISTERED:
        return
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    for payout,setting in RECIPES:
        register_mjlab_task(
            task_id=event_task_id(payout=payout, setting=setting),
            env_cfg=event_env_cfg(play=False, payout=payout, setting=setting),
            play_env_cfg=event_env_cfg(play=True, payout=payout, setting=setting),
            rl_cfg=pilot_rl_cfg(setting=setting, critic_mode='event'),
            runner_cls=HammerOnPolicyRunner)
    _REGISTERED = True
