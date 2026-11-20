"""Opt-in038 exploration laws on the unchanged reward037 split/event recipe."""
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hammer.config.z1.spline_reward037_cfg import reward_env_cfg as _reward037_env_cfg
from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_rl_cfg
from src.tasks.hammer.mdp.depth_reward031 import NativeDepthCreditObserver031

SETTINGS = ('independent', 'dependent', 'gsde', 'fixed_nominal')
_REGISTERED = False


def _require_setting(setting):
    if type(setting) is not str or setting not in SETTINGS:
        raise ValueError('exploration038 setting must be independent, dependent, gsde or fixed_nominal')


def _require_impulse_reward_weight(weight):
    if type(weight) not in (int, float) or weight not in (0., 4., 8.):
        raise ValueError('exploration038 impulse reward weight must be zero, four or eight')


def task_id(setting):
    _require_setting(setting)
    return 'Unitree-Z1-Hammer-Exploration038-' + setting


class ExplorationDepthObserver038(NativeDepthCreditObserver031):
    """Capture the038 setting while inheriting every native depth operation."""
    def __init__(self, cfg, env):
        self.exploration_setting = cfg.params['exploration_setting']
        _require_setting(self.exploration_setting)
        self.impulse_reward_weight = cfg.params.get('impulse_reward_weight', 4.)
        _require_impulse_reward_weight(self.impulse_reward_weight)
        if self.impulse_reward_weight == 0. and self.exploration_setting != 'independent':
            raise ValueError('impulse reward ablation requires independent exploration038')
        super().__init__(cfg, env)

    def __call__(self, env, *, arm, nail_cfg, exploration_setting, impulse_reward_weight=4.):
        _require_setting(exploration_setting)
        _require_impulse_reward_weight(impulse_reward_weight)
        if (exploration_setting != self.exploration_setting
                or impulse_reward_weight != self.impulse_reward_weight):
            raise ValueError('live exploration038 setting differs from configured recipe')
        return super().__call__(env, arm=arm, nail_cfg=nail_cfg)


def set_impulse_reward_weight(cfg, weight):
    """Select an explicit independent038 impulse treatment."""
    _require_impulse_reward_weight(weight)
    owner = cfg.metrics['native_depth_owner031']
    if (owner.func is not ExplorationDepthObserver038
            or owner.params.get('exploration_setting') != 'independent'
            or 'impulse_reward_weight' in owner.params
            or cfg.rewards['delivered_impulse'].weight != 4.):
        raise ValueError('impulse reward ablation requires an unchanged independent038 config')
    if weight != 4.:
        cfg.rewards['delivered_impulse'].weight = float(weight)
        owner.params['impulse_reward_weight'] = float(weight)
    return cfg


def disable_impulse_reward(cfg):
    """Explicit independent038 ablation; retain native impulse constraints."""
    return set_impulse_reward_weight(cfg, 0.)


def exploration_env_cfg(*, play, setting):
    _require_setting(setting)
    cfg = _reward037_env_cfg(play=play, setting='vic_baseline')
    owner = cfg.metrics['native_depth_owner031']
    owner.func = ExplorationDepthObserver038
    owner.params.pop('reward_setting')
    owner.params['exploration_setting'] = setting
    if setting == 'fixed_nominal':
        for group in ('actor', 'critic'):
            cfg.observations[group].terms['startup_age'].params = {'setting':'fixed_nominal'}
    return cfg


def exploration_rl_cfg(*, setting):
    _require_setting(setting)
    cfg = pilot_rl_cfg(setting='vic_baseline', critic_mode='event')
    prefix = 'src.tasks.hammer.rl.reward_exploration038:'
    cfg.actor.class_name = prefix + 'ExplorationActor038'
    cfg.actor.distribution_cfg = dict(class_name=prefix+'ExplorationGaussian038',
        mode='independent' if setting == 'fixed_nominal' else setting,
        feature_dim=cfg.actor.hidden_dims[-1], fixed_nominal=setting == 'fixed_nominal')
    cfg.algorithm.class_name = prefix + 'ExplorationPPO038'
    return cfg


def register_exploration038_tasks():
    global _REGISTERED
    if _REGISTERED:
        return
    from src.tasks.hammer.rl.reward_exploration038 import ExplorationRunner038
    for setting in SETTINGS:
        register_mjlab_task(task_id=task_id(setting),
            env_cfg=exploration_env_cfg(play=False, setting=setting),
            play_env_cfg=exploration_env_cfg(play=True, setting=setting),
            rl_cfg=exploration_rl_cfg(setting=setting), runner_cls=ExplorationRunner038)
    _REGISTERED = True


# Keep the small entry-point vocabulary shared with the earlier opt-in recipes.
reward_env_cfg = exploration_env_cfg
register_tasks = register_exploration038_tasks
