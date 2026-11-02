"""Opt-in all-rewards-ON split/event recipes using the existing 031 depth ledger."""
from mjlab.tasks.registry import register_mjlab_task
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.hammer.config.z1.spline_depth_reward031_cfg import depth_env_cfg
from src.tasks.hammer.config.z1.spline_event_study036_cfg import event_env_cfg
from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_rl_cfg
from src.tasks.hammer.config.z1.spline_split_credit_cfg import SETTINGS
from src.tasks.hammer.mdp.depth_reward031 import NativeDepthCreditObserver031

_REGISTERED = False


def reward_task_id(setting, *, height_dr=False):
    if type(setting) is not str or setting not in SETTINGS:
        raise ValueError('reward037 setting must be vic_baseline or vic_position_low')
    if type(height_dr) is not bool:
        raise ValueError('reward037 height_dr must be a bool')
    return 'Unitree-Z1-Hammer-Spline-Reward037-' + setting + ('-HeightDR5mm' if height_dr else '')


class RewardDepthObserver037(NativeDepthCreditObserver031):
    """Identify this recipe while inheriting every depth sampling/payment operation."""
    def __init__(self, cfg, env):
        self.reward_setting = cfg.params['reward_setting']
        self.height_dr = cfg.params.get('height_dr',False)
        reward_task_id(self.reward_setting,height_dr=self.height_dr)
        super().__init__(cfg, env)

    def __call__(self, env, *, arm, nail_cfg, reward_setting, height_dr=False):
        if reward_setting != self.reward_setting or type(height_dr) is not bool or height_dr != self.height_dr:
            raise ValueError('live reward037 setting differs from configured recipe')
        return super().__call__(env, arm=arm, nail_cfg=nail_cfg)


def reward_env_cfg(*, play, setting, height_dr=False):
    reward_task_id(setting,height_dr=height_dr)
    cfg = event_env_cfg(play=play, payout='split', setting=setting)
    depth = depth_env_cfg(play=play, arm='A3')
    for name in ('native_depth_owner031','native_depth_collect031'):
        cfg.metrics[name] = depth.metrics[name]
    for name in ('native_depth_progress031','native_depth_completion031'):
        cfg.rewards[name] = depth.rewards[name]
    owner = cfg.metrics['native_depth_owner031']
    owner.func = RewardDepthObserver037
    owner.params['reward_setting'] = setting
    # Keep the historical normalized first50ms impulse and RewardManager dt.
    cfg.rewards['delivered_impulse'].weight = 4.
    if height_dr:
        owner.params['height_dr'] = True
        cfg.events['randomize_board_nail_height037'] = EventTermCfg(
            func=envs_mdp.dr.body_pos,mode='reset',params=dict(
                asset_cfg=SceneEntityCfg('nail_block',body_names=('block','nail')),
                ranges=(-.005,.005),axes=[2],shared_random=True,
                operation='add',distribution='uniform'))
    return cfg


def register_reward037_tasks():
    global _REGISTERED
    if _REGISTERED:
        return
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    for setting in SETTINGS:
        for height_dr in (False,True):
            register_mjlab_task(task_id=reward_task_id(setting,height_dr=height_dr),
                env_cfg=reward_env_cfg(play=False,setting=setting,height_dr=height_dr),
                play_env_cfg=reward_env_cfg(play=True,setting=setting,height_dr=height_dr),
                rl_cfg=pilot_rl_cfg(setting=setting,critic_mode='event'),
                runner_cls=HammerOnPolicyRunner)
    _REGISTERED = True
