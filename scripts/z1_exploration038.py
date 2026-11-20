"""One opt-in exploration038 policy through the existing PPO trainer."""
import argparse
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from scripts import z1_reward_training037 as _reward037

SEEDS = _reward037.SEEDS
UPDATES, WORLDS, STEPS = _reward037.UPDATES, _reward037.WORLDS, _reward037.STEPS
SAVE_INDICES = _reward037.SAVE_INDICES
COMMON_INITIAL_HASHES = _reward037.COMMON_INITIAL_HASHES
SETTINGS = ('independent', 'dependent', 'gsde', 'fixed_nominal')
_INITIAL_GATE_ACTIVE = False


def cell_for_index(index):
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError('exploration038 index must be integer 0 through 11')
    return dict(index=index, payout='split', setting=SETTINGS[index//3],
                seed=SEEDS[index%3], critic_mode='event', updates=UPDATES,
                worlds=WORLDS, rollout_steps=STEPS, early_fraction=.5,
                early_deadline_ms=50, reserve_deadline_ms=200)


def build_train_config(index, *, reference_id=None):
    cell = cell_for_index(index)
    if reference_id is not None and cell['setting'] != 'independent':
        raise ValueError('human references use independent exploration and seed indices 0 through 2')
    from scripts.train import TrainConfig
    from src.tasks.hammer.config.z1.spline_exploration038_cfg import register_tasks, task_id
    # Inherit the complete approved 037 training recipe, changing only the new
    # task's environment/agent and the fourth arm's existing fixed controller.
    _, baseline = _reward037.build_train_config(index%3)
    register_tasks()
    task = task_id(cell['setting'])
    selected = TrainConfig.from_task(task)
    cfg = replace(baseline, env=selected.env, agent=selected.agent,
                  sampled_pd500_gain_mode='fixed' if cell['setting'] == 'fixed_nominal' else 'variable')
    if reference_id is not None:
        from src.tasks.hammer.config.z1.human_reference_cfg import task_id as human_task_id
        task = human_task_id(reference_id)
        cfg = replace(cfg,env=TrainConfig.from_task(task).env)
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
    from src.tasks.hammer.rl.exploration038_contract import MARKER, identity, validate_models
    env = runner.env.unwrapped
    hook = env.metrics_manager.cfg['cat_soft'].func
    if (type(hook) is not CatSoftHook or hook._max_p != .5 or hook._imp_max_p != .2
            or hook._velocity_pressure is not None or not hook._use_vel or not hook._use_impulse):
        raise ValueError('live constant velocity/impulse recipe differs')
    metadata = build_impact_speed_metadata(env, runner.env.clip_actions)
    if metadata.get(MARKER) != identity(cell['setting']):
        raise ValueError('live exploration038 treatment differs from selected cell')
    if (type(env.num_envs) is not int or env.num_envs != WORLDS
            or type(runner.cfg['num_steps_per_env']) is not int or runner.cfg['num_steps_per_env'] != STEPS
            or type(runner.cfg['seed']) is not int or runner.cfg['seed'] != cell['seed']
            or type(env.cfg.seed) is not int or env.cfg.seed != cell['seed']):
        raise ValueError('live worlds, rollout clock or seed differs')
    validate_models(runner.alg.actor, runner.alg.critic, metadata)
    return metadata


@contextmanager
def initial_state_gate(index, output_dir, *, reference_id=None):
    """Bind the core runner's actual-first-observation record to this one cell."""
    global _INITIAL_GATE_ACTIVE
    from src.tasks.hammer.rl.reward_exploration038 import ExplorationRunner038
    cell = cell_for_index(index)
    if reference_id is not None:
        from src.tasks.hammer.mdp.human_reference import reference_identity
        expected_reference = reference_identity(reference_id)
        cell['reference_id'] = reference_id
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise ValueError('existing output directory required')
    if (output/'initial-state.json').exists() or (output/'initial-state.json').is_symlink():
        raise FileExistsError('initial-state evidence already exists')
    if _INITIAL_GATE_ACTIVE:
        raise ValueError('exploration038 initial gate already active')
    original = ExplorationRunner038.learn
    evidence = []
    called = []
    def learn(runner, num_learning_iterations, init_at_random_ep_len=False):
        if (called or type(num_learning_iterations) is not int or num_learning_iterations != UPDATES
                or type(runner.current_learning_iteration) is not int or runner.current_learning_iteration != 0
                or type(runner.env.unwrapped.common_step_counter) is not int
                or runner.env.unwrapped.common_step_counter != 0
                or runner.cfg.get('resume') is not False or runner.alg.optimizer.state):
            raise ValueError('one fresh exploration038 learning call required')
        metadata = _live_recipe(runner, cell)
        if reference_id is not None and metadata.get('human_reference') != expected_reference:
            raise ValueError('live human reference differs from selected study cell')
        called.append(True)
        runner.study_output_dir = output
        runner.study_cell = dict(cell)
        result = original(runner, num_learning_iterations, init_at_random_ep_len)
        record = getattr(runner, 'initial_state_record', None)
        if not isinstance(record, dict) or record.get('cell') != cell or not (output/'initial-state.json').is_file():
            raise ValueError('core exploration038 pre-rollout initialization evidence missing')
        evidence.append(record)
        return result
    _INITIAL_GATE_ACTIVE = True
    ExplorationRunner038.learn = learn
    try:
        yield evidence
    finally:
        ExplorationRunner038.learn = original
        _INITIAL_GATE_ACTIVE = False


def selected_checkpoint_saves(index, output_dir):
    return _reward037._checkpoint_saves(cell_for_index(index), output_dir, recipe=_live_recipe)


def main(argv=None):
    from src.tasks.hammer.mdp.human_reference import REFERENCE_IDS, reference_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=int)
    parser.add_argument('--reference', choices=REFERENCE_IDS)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    reference = {'reference_id':args.reference} if args.reference is not None else {}
    task, cfg = build_train_config(args.index,**reference)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.reference is not None:
        (output/'human-reference.json').write_text(json.dumps(reference_identity(args.reference),indent=2))
    from scripts.train import run_train
    with initial_state_gate(args.index, output,**reference), selected_checkpoint_saves(args.index, output):
        run_train(task, cfg, output/'train')


if __name__ == '__main__':
    main()
