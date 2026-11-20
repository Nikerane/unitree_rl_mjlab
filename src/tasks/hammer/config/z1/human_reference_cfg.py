"""Opt-in demo22/demo20 pairs on the unchanged independent038 recipe."""
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hammer.config.z1.spline_exploration038_cfg import exploration_env_cfg, exploration_rl_cfg
from src.tasks.hammer.mdp.human_reference import (
    MIXED_REFERENCE_IDS, REFERENCE_IDS, path_ids, reset_human_reference, reset_human_joints,
    human_phase, human_error,
)

_REGISTERED = False


def task_id(reference_id):
    path_ids(reference_id)
    return 'Unitree-Z1-Hammer-HumanReference-' + reference_id


def human_env_cfg(*, play, reference_id):
    task_id(reference_id)
    cfg = exploration_env_cfg(play=play,setting='independent')
    event = cfg.events['sample_strike_route_signs']
    event.func, event.params = reset_human_reference, {'reference_id':reference_id}
    joint = cfg.events['reset_strike_route_joints']
    joint.func = reset_human_joints
    joint.params = dict(reference_id=reference_id,asset_cfg=joint.params['asset_cfg'])
    for group in ('actor','critic'):
        for name, func in (('strike_phase',human_phase), ('strike_ref_error',human_error)):
            term = cfg.observations[group].terms[name]
            term.func = func
            term.params = {key:term.params[key] for key in ('robot_cfg','nail_cfg')}
            term.params['reference_id'] = reference_id
    return cfg


def register_human_reference_tasks():
    global _REGISTERED
    if _REGISTERED:
        return
    from src.tasks.hammer.rl.reward_exploration038 import ExplorationRunner038
    for reference_id in REFERENCE_IDS + MIXED_REFERENCE_IDS:
        register_mjlab_task(task_id=task_id(reference_id),
            env_cfg=human_env_cfg(play=False,reference_id=reference_id),
            play_env_cfg=human_env_cfg(play=True,reference_id=reference_id),
            rl_cfg=exploration_rl_cfg(setting='independent'),runner_cls=ExplorationRunner038)
    _REGISTERED = True
