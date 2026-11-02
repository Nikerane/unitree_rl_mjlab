"""Nine-run all-on recipe; no historical defaults or constraint changes."""
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.hammer.config.z1.spline_depth_reward031_cfg import depth_env_cfg
from src.tasks.hammer.config.z1.spline_split_credit_cfg import split_rl_cfg
from src.tasks.hammer.mdp.depth_reward031 import NativeDepthCreditObserver031
from src.tasks.hammer.rl.reward_exploration032 import MODES

_REGISTERED = False


class ExplorationDepthObserver032(NativeDepthCreditObserver031):
    """Reuse the exact depth ledger; identify the new recipe without claiming A3."""
    def __init__(self, cfg, env):
        self.exploration_mode = cfg.params['exploration_mode']
        if self.exploration_mode not in MODES:
            raise ValueError('unknown exploration mode')
        super().__init__(cfg, env)

    def __call__(self, env, *, arm, nail_cfg, exploration_mode):
        if exploration_mode != self.exploration_mode:
            raise ValueError('live exploration mode differs from the configured recipe')
        return super().__call__(env, arm=arm, nail_cfg=nail_cfg)


def reward_env_cfg(*, play, mode):
    if mode not in MODES:
        raise ValueError('unknown exploration mode')
    cfg = depth_env_cfg(play=play, arm='A3')
    # Restore the pre-spline normalized linear delivered-impulse coefficient.
    # Do not divide by dt or change its existing event window/normalizer.
    cfg.rewards['delivered_impulse'].weight = 4.
    owner = cfg.metrics['native_depth_owner031']
    owner.func = ExplorationDepthObserver032
    owner.params['exploration_mode'] = mode
    return cfg


def task_id(mode):
    if mode not in MODES:
        raise ValueError('unknown exploration mode')
    return 'Unitree-Z1-Hammer-RewardExploration032-' + mode


def register_tasks():
    global _REGISTERED
    if _REGISTERED:
        return
    from src.tasks.hammer.rl.reward_exploration032_runner import RewardExplorationRunner
    for mode in MODES:
        agent = split_rl_cfg(setting='vic_baseline')
        prefix = 'src.tasks.hammer.rl.reward_exploration032:'
        agent.actor.class_name = prefix + 'RewardExplorationActor'
        agent.actor.distribution_cfg = {'class_name': prefix + 'RewardExplorationGaussian', 'mode': mode}
        agent.algorithm.class_name = prefix + 'RewardExplorationPPO'
        register_mjlab_task(task_id=task_id(mode), env_cfg=reward_env_cfg(play=False, mode=mode),
            play_env_cfg=reward_env_cfg(play=True, mode=mode), rl_cfg=agent, runner_cls=RewardExplorationRunner)

    _REGISTERED = True
