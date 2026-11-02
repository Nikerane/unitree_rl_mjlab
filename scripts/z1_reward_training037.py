"""Fresh all-task-reward split/event policies through the existing PPO trainer."""
import argparse
from dataclasses import replace
from pathlib import Path

from scripts import z1_event_critic036 as _event036
from scripts.z1_event_critic036 import _pilot, _initial_gate, _checkpoint_saves

SEEDS = _event036.SEEDS
UPDATES, WORLDS, STEPS = _event036.UPDATES, _event036.WORLDS, _event036.STEPS
SAVE_INDICES = _event036.SAVE_INDICES
COMMON_INITIAL_HASHES = _event036.COMMON_INITIAL_HASHES


def cell_for_index(index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError('reward037 index must be integer 0 through 7')
    cell = dict(index=index, payout='split', setting=('vic_baseline','vic_position_low')[index//3 if index < 6 else index-6],
                seed=SEEDS[index%3] if index < 6 else SEEDS[0], critic_mode='event',
                updates=UPDATES, worlds=WORLDS, rollout_steps=STEPS,
                early_fraction=.5, early_deadline_ms=50, reserve_deadline_ms=200)
    if index >= 6:
        cell['height_dr'] = True
    return cell


def build_train_config(index):
    cell = cell_for_index(index)
    from scripts.train import TrainConfig
    from src.tasks.hammer.config.z1.spline_reward037_cfg import reward_task_id, register_reward037_tasks
    register_reward037_tasks()
    task = reward_task_id(cell['setting'], height_dr=cell.get('height_dr',False))
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
    from src.tasks.hammer.rl.spline_reward037_contract import reward_identity, validate_reward_models
    env = runner.env.unwrapped
    hook = env.metrics_manager.cfg['cat_soft'].func
    if (type(hook) is not CatSoftHook or hook._max_p != .5 or hook._imp_max_p != .2
            or hook._velocity_pressure is not None or not hook._use_vel or not hook._use_impulse):
        raise ValueError('live constant velocity/impulse recipe differs')
    metadata = build_impact_speed_metadata(env, runner.env.clip_actions)
    if metadata.get('spline_reward037') != reward_identity(cell['setting'], height_dr=cell.get('height_dr',False)):
        raise ValueError('live all-task-reward split/event treatment differs from selected cell')
    if (type(env.num_envs) is not int or env.num_envs != WORLDS
            or type(runner.cfg['num_steps_per_env']) is not int or runner.cfg['num_steps_per_env'] != STEPS
            or type(runner.cfg['seed']) is not int or runner.cfg['seed'] != cell['seed']
            or type(env.cfg.seed) is not int or env.cfg.seed != cell['seed']):
        raise ValueError('live worlds, rollout clock or seed differs')
    validate_reward_models(runner.alg.actor, runner.alg.critic, metadata)
    return metadata


def initial_state_gate(index, output_dir):
    cell = cell_for_index(index)
    return _initial_gate(cell, output_dir, enable_trace=False, reference_initial_state=None,
                         reference_cell=cell, recipe=_live_recipe, marker_key='spline_reward037', trace_module=None)


def selected_checkpoint_saves(index, output_dir):
    return _checkpoint_saves(cell_for_index(index), output_dir, recipe=_live_recipe)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    task, cfg = build_train_config(args.index)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    from scripts.train import run_train
    with initial_state_gate(args.index, output), selected_checkpoint_saves(args.index, output):
        run_train(task, cfg, output/'train')


if __name__ == '__main__':
    main()
