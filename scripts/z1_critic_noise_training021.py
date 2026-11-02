"""Fresh C12 critic/noise500 adapter; no execution or admission authority."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import re

from scripts import z1_exploration_pilot_training as _pilot
from scripts.z1_credit_critic_training import _directory, _write_once, _file_sha, tensor_state_digest


SEEDS = (2026103101, 2026103102, 2026103103)
UPDATES, WORLDS, STEPS = 500, 1024, 24
SAVE_INDICES = (0, 24, 49, 99, 249, 499)
COMMON_INITIAL_HASHES = _pilot.COMMON_INITIAL_HASHES
_PATCHED = set()
_live_recipe = _pilot._live_recipe


def cell_for_index(index: int) -> dict:
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError('timing/noise index must be integer0 through11')
    return dict(index=index, setting=('vic_baseline', 'vic_position_low')[index % 2],
                seed=SEEDS[index // 4], credit_deadline_ms=100,
                critic_mode='current' if index % 4 < 2 else 'event', updates=UPDATES, worlds=WORLDS, rollout_steps=STEPS)


def build_train_config(index):
    cell = cell_for_index(index)
    task, cfg = _pilot.build_pilot_train_config(index % 2,
        credit_deadline_ms=cell['credit_deadline_ms'], critic_mode=cell['critic_mode'])
    cfg.env.seed = cfg.agent.seed = cell['seed']
    cfg.agent.max_iterations = UPDATES
    return task, cfg


def matched_initial_state_gate(index, output_dir, *, enable_trace=True, reference_initial_state=None):
    return _initial_gate(cell_for_index(index), output_dir, enable_trace=enable_trace,
        reference_initial_state=reference_initial_state, reference_cell=cell_for_index(index//4*4),
        recipe=_live_recipe, marker_key='spline_exploration_pilot',
        trace_module='scripts.z1_timing_noise020.long_credit_trace')


@contextmanager
def _initial_gate(cell, output_dir, *, enable_trace, reference_initial_state, reference_cell,
                  recipe, marker_key, trace_module, full_reference=False, trace_max_bytes=96 * 1024**2):
    """Match only existing state and intercept one fresh500 learning call."""
    output = _directory(output_dir)
    if type(enable_trace) is not bool or (enable_trace and output is None):
        raise ValueError('both traces require explicit boolean and output directory')
    if output is not None and any((output / name).exists() or (output / name).is_symlink()
            for name in ('initial-state.json', 'initial-state-mismatch.json')):
        raise FileExistsError('initial-state evidence already consumed')
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    if 'learn' in _PATCHED:
        raise ValueError('timing/noise initial gate already active')
    original = HammerOnPolicyRunner.learn
    evidence = []

    def learn(runner, num_learning_iterations, init_at_random_ep_len=False):
        if (evidence or type(num_learning_iterations) is not int or num_learning_iterations != UPDATES
                or type(runner.current_learning_iteration) is not int or runner.current_learning_iteration != 0
                or type(runner.env.unwrapped.common_step_counter) is not int or runner.env.unwrapped.common_step_counter != 0
                or runner.cfg.get('resume') is not False or runner.alg.optimizer.state):
            raise ValueError('fresh once-only finite500 learning, resumeFalse and empty optimizer required')
        metadata = recipe(runner, cell)
        marker = metadata[marker_key]
        observed = _pilot._initial_observation_hashes(runner.env.unwrapped, marker)
        hashes, std = _pilot._initialize(runner, cell)
        row = dict(cell=cell, phase='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT', actual_counter=0,
            critic_width=marker['critic_width'], contract=metadata, hashes=hashes,
            initial_std=std, initial_observations=observed, model_initialization_seed=cell['seed'],
            collection_seed=cell['seed'], live_cat=dict(velocity_pmax=.5, impulse_pmax=.2, scheduler=False))
        if reference_initial_state is not None:
            ref = reference_initial_state
            expected = reference_cell
            if (not isinstance(ref, dict) or ref.get('cell') != expected or ref.get('phase') != row['phase']
                    or type(ref.get('actual_counter')) is not int or ref['actual_counter'] != 0):
                _pilot._mismatch(output, row, ref, 'common initial baseline reference identity differs')
            ref_hashes = ref.get('hashes')
            if not isinstance(ref_hashes, dict) or any(hashes[key] != ref_hashes.get(key) for key in ((*COMMON_INITIAL_HASHES, 'actor', 'critic') if full_reference else COMMON_INITIAL_HASHES)):
                _pilot._mismatch(output, row, ref, 'common initial non-treatment model/RNG fingerprints differ')
            if full_reference and ref.get('initial_std') != std:
                _pilot._mismatch(output, row, ref, 'same-setting control std differs')
            ref_observed = ref.get('initial_observations')
            if not isinstance(ref_observed, dict) or any(observed[key] != ref_observed.get(key)
                    for key in ('actor_sha256', 'critic_shared43_sha256')):
                _pilot._mismatch(output, row, ref, 'common initial observed state differs')
        if output is not None:
            _write_once(output / 'initial-state.json', row)
        evidence.append(row)
        with ExitStack() as stack:
            if enable_trace:
                from importlib import import_module
                attach_credit_training_trace = import_module(trace_module).attach_credit_training_trace
                from src.tasks.hammer.rl.velocity_population_trace import attach_velocity_population_trace
                stack.enter_context(attach_credit_training_trace(runner, output / 'training-trace.jsonl',
                    max_updates=UPDATES, max_bytes=trace_max_bytes))
                stack.enter_context(attach_velocity_population_trace(runner, output / 'velocity-population.jsonl',
                    max_updates=UPDATES, max_bytes=16 * 1024**2))
            return original(runner, num_learning_iterations, init_at_random_ep_len)

    _PATCHED.add('learn')
    HammerOnPolicyRunner.learn = learn
    try:
        yield evidence
        if len(evidence) != 1:
            raise ValueError('training must pass timing/noise gate exactly once')
    finally:
        HammerOnPolicyRunner.learn = original
        _PATCHED.remove('learn')


def selected_checkpoint_saves(index, output_dir):
    return _checkpoint_saves(cell_for_index(index), output_dir, recipe=_live_recipe)


@contextmanager
def _checkpoint_saves(cell, output_dir, *, recipe):
    """Retain six after-update labels and deduplicate one identical final save."""
    output = _directory(output_dir)
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    if 'save' in _PATCHED:
        raise ValueError('timing/noise checkpoint filter already active')
    original = HammerOnPolicyRunner.save
    evidence, seen, owner = [], {}, []

    def save(runner, path, infos=None):
        path = Path(path)
        match = re.fullmatch(r'model_(\d+)\.pt', path.name)
        if match is None:
            raise ValueError('unexpected checkpoint filename')
        label = int(match.group(1))
        if label not in range(UPDATES):
            raise ValueError('checkpoint outside finite allocation')
        if label not in SAVE_INDICES:
            return None
        if output is not None and path != output / 'train' / path.name:
            raise ValueError('checkpoint outside exact timing/noise train directory')
        counter = runner.env.unwrapped.common_step_counter
        if (type(counter) is not int or counter != (label + 1) * STEPS
                or type(runner.current_learning_iteration) is not int or runner.current_learning_iteration != label):
            raise ValueError('checkpoint transition clock differs')
        if owner and owner[0] is not runner:
            raise ValueError('checkpoint context belongs to another runner')
        metadata = recipe(runner, cell)
        saved_infos = dict(infos or {})
        if 'impact_speed_contract' in saved_infos and saved_infos['impact_speed_contract'] != metadata:
            raise ValueError('checkpoint full pilot metadata differs')
        saved_infos['impact_speed_contract'] = metadata
        hashes = {key: tensor_state_digest(getattr(runner.alg, key).state_dict()) for key in ('actor', 'critic')}
        if label in seen:
            prior = seen[label]
            if (label != UPDATES - 1 or prior.get('deduplicated') or prior['hashes'] != hashes
                    or prior['checkpoint_sha256'] != _file_sha(path)):
                raise ValueError('unexpected repeated checkpoint save')
            prior['deduplicated'] = True
            row = dict(iteration=label, common_step_counter=counter, hashes=hashes,
                checkpoint_sha256=prior['checkpoint_sha256'], deduplicated_final_save=True)
            if output is not None:
                _write_once(output / f'checkpoint-{label}-final-save-deduplicated.json', row)
            evidence.append(row)
            return None
        if path.exists() or path.is_symlink():
            raise FileExistsError('checkpoint output is consumed')
        result = original(runner, str(path), saved_infos)
        row = dict(iteration=label, common_step_counter=counter, hashes=hashes,
            checkpoint_sha256=_file_sha(path), deduplicated_final_save=False)
        if output is not None:
            _write_once(output / f'checkpoint-{label}-live.json', row)
        seen[label] = dict(row)
        owner[:] = [runner]
        evidence.append(row)
        return result

    _PATCHED.add('save')
    HammerOnPolicyRunner.save = save
    try:
        yield evidence
    finally:
        HammerOnPolicyRunner.save = original
        _PATCHED.remove('save')
