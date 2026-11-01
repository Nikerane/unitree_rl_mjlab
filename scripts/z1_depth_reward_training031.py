"""Fresh twelve-cell031 adapter. No submission/allocation authority."""
from dataclasses import replace
from scripts import z1_exploration_pilot_training as _pilot
from scripts.z1_critic_noise_training021 import _initial_gate, _checkpoint_saves

SEEDS = (2026110101, 2026110102, 2026110103)
UPDATES, WORLDS, STEPS = 500, 1024, 24
SAVE_INDICES = (0, 24, 49, 99, 249, 499)
COMMON_INITIAL_HASHES = _pilot.COMMON_INITIAL_HASHES
_REGISTERED = False


def cell_for_index(index):
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError('depth031 index must be integer0..11')
    return dict(index=index, arm=('A0','A1','A2','A3')[index % 4], seed=SEEDS[index // 4],
        setting='vic_baseline', critic_mode='current', updates=UPDATES, worlds=WORLDS,
        rollout_steps=STEPS, early_fraction=.5, early_deadline_ms=50, reserve_deadline_ms=200)


def build_train_config(index):
    global _REGISTERED
    cell = cell_for_index(index)
    from scripts.train import TrainConfig
    from src.tasks.hammer.config.z1.spline_depth_reward031_cfg import register_depth_reward031_tasks, depth_task_id
    if not _REGISTERED:
        register_depth_reward031_tasks()
        _REGISTERED = True
    task = depth_task_id(cell['arm'])
    cfg = replace(TrainConfig.from_task(task), sampled_pd500=True, sampled_pd500_gain_mode='variable',
        speed_bounds=(1.,1.5), speed_cost=.8493676, speed_mode='policy',
        qualification_checkpoint=None, enable_nan_guard=True)
    cfg.env.scene.num_envs = WORLDS
    cfg.env.seed = cfg.agent.seed = cell['seed']
    cfg.agent.max_iterations = UPDATES
    cfg.agent.resume = False
    cfg.agent.save_interval = 1
    cfg.agent.logger = 'tensorboard'
    return task, cfg


def _live_recipe(runner, cell):
    from src.tasks.hammer.cat.hook import CatSoftHook
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.depth_reward031_contract import depth_identity031, validate_depth_models031
    env = runner.env.unwrapped
    hook = env.metrics_manager.cfg['cat_soft'].func
    if (type(hook) is not CatSoftHook or hook._max_p != .5 or hook._imp_max_p != .2
            or hook._velocity_pressure is not None or not hook._use_vel or not hook._use_impulse):
        raise ValueError('depth031 requires unchanged constant velocity/impulse recipe')
    metadata = build_impact_speed_metadata(env, runner.env.clip_actions)
    if metadata.get('spline_depth_reward031') != depth_identity031(cell['arm']):
        raise ValueError('live depth031 identity differs from selected cell')
    if (type(env.num_envs) is not int or env.num_envs != WORLDS
            or type(runner.cfg['num_steps_per_env']) is not int or runner.cfg['num_steps_per_env'] != STEPS
            or type(runner.cfg['seed']) is not int or runner.cfg['seed'] != cell['seed']
            or type(env.cfg.seed) is not int or env.cfg.seed != cell['seed']):
        raise ValueError('depth031 live worlds, rollout or matched seed differs')
    validate_depth_models031(runner.alg.actor, runner.alg.critic, metadata)
    return metadata


def matched_initial_state_gate(index, output_dir, *, enable_trace=True, reference_initial_state=None):
    cell = cell_for_index(index)
    if cell['arm'] != 'A0' and reference_initial_state is None:
        raise ValueError('depth031 treatment requires its durable block A0 initial-state reference')
    return _initial_gate(cell, output_dir, enable_trace=enable_trace,
        reference_initial_state=reference_initial_state, reference_cell=cell_for_index(index // 4 * 4),
        recipe=_live_recipe, marker_key='spline_depth_reward031',
        trace_module='src.tasks.hammer.rl.depth_reward031_trace', full_reference=True,
        trace_max_bytes=160 * 1024**2)


def selected_checkpoint_saves(index, output_dir):
    return _checkpoint_saves(cell_for_index(index), output_dir, recipe=_live_recipe)
