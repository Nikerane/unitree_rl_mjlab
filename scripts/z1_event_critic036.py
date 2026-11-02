"""One fresh event-critic policy through the existing PPO trainer."""
import argparse
from dataclasses import replace
from pathlib import Path

from scripts import z1_exploration_pilot_training as _pilot
from scripts import z1_critic_noise_training021 as _critic021
from scripts.z1_critic_noise_training021 import _initial_gate, _checkpoint_saves

SEEDS = (2026110201, 2026110202, 2026110203)
RECIPES = (('split', 'vic_baseline'), ('200', 'vic_baseline'),
           ('split', 'vic_position_low'), ('50', 'vic_baseline'))
UPDATES, WORLDS, STEPS = 500, 1024, 24
SAVE_INDICES = (0, 24, 49, 99, 249, 499)
COMMON_INITIAL_HASHES = _pilot.COMMON_INITIAL_HASHES
if (UPDATES,WORLDS,STEPS,SAVE_INDICES) != (_critic021.UPDATES,_critic021.WORLDS,_critic021.STEPS,_critic021.SAVE_INDICES):
    raise ValueError('inherited 021 learning/checkpoint clocks differ from approved 036 recipe')


def cell_for_index(index):
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError('event-critic index must be integer 0 through 11')
    payout, setting = RECIPES[index // 3]
    cell = dict(index=index, payout=payout, setting=setting, seed=SEEDS[index % 3],
                critic_mode='event', updates=UPDATES, worlds=WORLDS, rollout_steps=STEPS)
    if payout == 'split':
        cell.update(early_fraction=.5, early_deadline_ms=50, reserve_deadline_ms=200)
    else:
        cell['credit_deadline_ms'] = int(payout)
    return cell


def build_train_config(index):
    cell = cell_for_index(index)
    from scripts.train import TrainConfig
    from src.tasks.hammer.config.z1.spline_event_study036_cfg import event_task_id, register_event_study_tasks
    register_event_study_tasks()
    task = event_task_id(payout=cell['payout'], setting=cell['setting'])
    cfg = replace(TrainConfig.from_task(task), sampled_pd500=True, sampled_pd500_gain_mode='variable',
                  speed_bounds=(1., 1.5), speed_cost=.8493676, speed_mode='policy',
                  qualification_checkpoint=None, enable_nan_guard=True)
    cfg.env.scene.num_envs = WORLDS
    cfg.env.seed = cfg.agent.seed = cell['seed']
    cfg.agent.max_iterations = UPDATES
    cfg.agent.resume = False
    cfg.agent.save_interval = 1
    cfg.agent.logger = 'tensorboard'
    return task, cfg


def _live_recipe(runner, cell):
    if cell['payout'] != 'split':
        return _pilot._live_recipe(runner, cell)
    from src.tasks.hammer.cat.hook import CatSoftHook
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.spline_split_event036_contract import split_event_identity, validate_split_event_models
    env = runner.env.unwrapped
    hook = env.metrics_manager.cfg['cat_soft'].func
    if (type(hook) is not CatSoftHook or hook._max_p != .5 or hook._imp_max_p != .2
            or hook._velocity_pressure is not None or not hook._use_vel or not hook._use_impulse):
        raise ValueError('live constant velocity/impulse recipe differs')
    metadata = build_impact_speed_metadata(env, runner.env.clip_actions)
    if metadata.get('spline_split_event036') != split_event_identity(cell['setting'], cell['early_fraction']):
        raise ValueError('live split/event treatment differs from selected cell')
    if (type(env.num_envs) is not int or env.num_envs != WORLDS
            or type(runner.cfg['num_steps_per_env']) is not int or runner.cfg['num_steps_per_env'] != STEPS
            or type(runner.cfg['seed']) is not int or runner.cfg['seed'] != cell['seed']
            or type(env.cfg.seed) is not int or env.cfg.seed != cell['seed']):
        raise ValueError('live worlds, rollout clock or seed differs')
    validate_split_event_models(runner.alg.actor, runner.alg.critic, metadata)
    return metadata


def initial_state_gate(index, output_dir):
    cell = cell_for_index(index)
    return _initial_gate(cell, output_dir, enable_trace=False, reference_initial_state=None,
                         reference_cell=cell, recipe=_live_recipe,
                         marker_key='spline_split_event036' if cell['payout'] == 'split' else 'spline_exploration_pilot',
                         trace_module=None)


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
        run_train(task, cfg, output / 'train')


if __name__ == '__main__':
    main()
