"""Four opt-in exploration settings at one explicit credit/critic anchor."""
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hammer.config.z1.spline_credit_critic_cfg import (
    TASKS,credit_critic_env_cfg,credit_critic_rl_cfg,DEADLINES_MS,CRITIC_MODES,
)
from src.tasks.hammer.mdp.spline_exploration_pilot import pilot_startup_age,_require_setting
from src.tasks.hammer.rl.spline_exploration_pilot import PILOT_SETTINGS,PILOT_ACTOR_CLASSES


def _require_anchor(credit_deadline_ms,critic_mode):
    if type(credit_deadline_ms) is not int or credit_deadline_ms not in DEADLINES_MS:
        raise ValueError('pilot anchor requires an admitted integer credit deadline')
    if type(critic_mode) is not str or critic_mode not in CRITIC_MODES:
        raise ValueError('pilot anchor requires current or event critic')


def pilot_task_id(*,setting,credit_deadline_ms=200,critic_mode='current'):
    _require_setting(setting);_require_anchor(credit_deadline_ms,critic_mode)
    return TASKS[credit_deadline_ms,critic_mode]+'-Pilot-'+setting


def pilot_env_cfg(*,play,setting,credit_deadline_ms=200,critic_mode='current'):
    _require_setting(setting);_require_anchor(credit_deadline_ms,critic_mode)
    cfg=credit_critic_env_cfg(play=play,credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode)
    for group in ('actor','critic'):
        term=cfg.observations[group].terms['startup_age']
        term.func=pilot_startup_age;term.params={'setting':setting}
    return cfg


def pilot_rl_cfg(*,setting,critic_mode='current'):
    _require_setting(setting);_require_anchor(200,critic_mode)
    cfg=credit_critic_rl_cfg(critic_mode=critic_mode)
    prefix='src.tasks.hammer.rl.spline_exploration_pilot:'
    cfg.actor.class_name=prefix+PILOT_ACTOR_CLASSES[setting].__name__
    cfg.actor.distribution_cfg['class_name']=prefix+'PilotGaussian'
    return cfg


def register_pilot_tasks(*,credit_deadline_ms=200,critic_mode='current'):
    """Register only the four settings for this chosen anchor, never all32."""
    _require_anchor(credit_deadline_ms,critic_mode)
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    for setting in PILOT_SETTINGS:
        register_mjlab_task(
            task_id=pilot_task_id(setting=setting,credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode),
            env_cfg=pilot_env_cfg(play=False,setting=setting,credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode),
            play_env_cfg=pilot_env_cfg(play=True,setting=setting,credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode),
            rl_cfg=pilot_rl_cfg(setting=setting,critic_mode=critic_mode),runner_cls=HammerOnPolicyRunner,
        )
