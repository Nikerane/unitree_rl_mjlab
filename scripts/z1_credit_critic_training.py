"""Finite eight-cell training configuration/contexts; no launch entrypoint.

This adapter never submits jobs or grants admission. A caller must validate its
fresh package, runtime, output mapping and approved allocation before run_train.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import stat

SEED = 2026102901
UPDATES = 100
WORLDS = 1024
STEPS = 24
SAVE_INDICES = (0, 24, 49, 99)
COMMON_INITIAL_HASHES = ('actor', 'shared_critic43', 'actor_normalizer', 'critic_normalizer',
                        'collection_torch_rng', 'collection_python_rng',
                        'collection_numpy_rng', 'collection_cuda_rng')
_PATCHED = set()


def cell_for_index(index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError('cell index must be integer0 through7')
    return dict(index=index, credit_deadline_ms=(50, 100, 150, 200)[index // 2],
                critic_mode=('current', 'event')[index % 2], seed=SEED,
                updates=UPDATES, worlds=WORLDS, rollout_steps=STEPS)


def build_train_config(index):
    cell = cell_for_index(index)
    from scripts.train import TrainConfig
    from src.tasks.hammer.config.z1.spline_credit_critic_cfg import TASKS
    task = TASKS[cell['credit_deadline_ms'], cell['critic_mode']]
    cfg = replace(TrainConfig.from_task(task), sampled_pd500=True,
                  sampled_pd500_gain_mode='variable', speed_bounds=(1., 1.5),
                  speed_cost=.8493676, speed_mode='policy',
                  qualification_checkpoint=None, enable_nan_guard=True)
    cfg.env.scene.num_envs = WORLDS
    cfg.env.seed = cfg.agent.seed = SEED
    cfg.agent.max_iterations = UPDATES
    cfg.agent.resume = False
    cfg.agent.save_interval = 1
    cfg.agent.logger = 'tensorboard'
    return task, cfg


def _directory(value):
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute() or p != p.resolve() or not p.is_dir():
        raise ValueError('output must be an existing canonical absolute directory')
    return p


def _write_once(path, record):
    body = json.dumps(record, sort_keys=True, allow_nan=False).encode()
    if len(body) > 65536:
        raise ValueError('bounded evidence exceeds64KiB')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(body); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _initial_mismatch(output, row, reference, reason):
    """Preserve actual pre-rollout fingerprints without labeling a failed gate passed."""
    if output is not None:
        def fingerprints(value, keys):
            value = value if isinstance(value, dict) else {}
            return {key: item if item is None or (type(item) is str and len(item) <= 128)
                    else {'invalid_reference_type': type(item).__name__}
                    for key in keys for item in [value.get(key)]}
        reference = reference if isinstance(reference, dict) else {}
        _write_once(output / 'initial-state-mismatch.json', dict(
            status='INITIAL_STATE_REFERENCE_MISMATCH_NO_TRAINING', reason=reason,
            actual_initial_state=row,
            expected_hashes=fingerprints(reference.get('hashes'), COMMON_INITIAL_HASHES),
            expected_initial_observations=fingerprints(reference.get('initial_observations'),
                ('actor_sha256', 'critic_shared43_sha256'))))
    raise ValueError(reason)


def tensor_state_digest(state):
    import torch
    h = hashlib.sha256()
    if not state:
        raise ValueError('nonempty tensor state required')
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError('finite tensor state required')
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _file_sha(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 32 * 1024**2:
            raise ValueError('bounded regular checkpoint required')
        h = hashlib.sha256(); count = 0
        while True:
            body = f.read(1024**2)
            if not body: break
            count += len(body)
            if count > before.st_size: raise ValueError('checkpoint grew during read')
            h.update(body)
        after = os.fstat(f.fileno())
        if (count != before.st_size or before.st_mtime_ns != after.st_mtime_ns
                or before.st_size != after.st_size):
            raise ValueError('checkpoint changed during read')
    return h.hexdigest()


def _live_recipe(runner, cell):
    from src.tasks.hammer.cat.hook import CatSoftHook
    from src.tasks.hammer.rl.spline_credit_critic_contract import (
        credit_critic_metadata, experiment_identity,
    )
    from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models
    env = runner.env.unwrapped
    hook = env.metrics_manager.cfg['cat_soft'].func
    if (type(hook) is not CatSoftHook or hook._max_p != .5 or hook._imp_max_p != .2
            or hook._velocity_pressure is not None or not hook._use_vel or not hook._use_impulse):
        raise ValueError('live constant velocity/impulse CaT recipe differs')
    marker = credit_critic_metadata(env)
    if marker != experiment_identity(cell['credit_deadline_ms'], cell['critic_mode']):
        raise ValueError('live credit/critic treatment differs from selected cell')
    if (type(env.num_envs) is not int or env.num_envs != WORLDS
            or type(runner.cfg['num_steps_per_env']) is not int
            or runner.cfg['num_steps_per_env'] != STEPS
            or type(runner.cfg['seed']) is not int or runner.cfg['seed'] != SEED
            or type(env.cfg.seed) is not int or env.cfg.seed != SEED):
        raise ValueError('live worlds, rollout clock or seed differs')
    validate_impact_speed_models(runner.alg.actor, runner.alg.critic,
        {'spline_learning_variant': {'variant': 'N'}, 'credit_critic_experiment': marker})
    return marker


def _initial_observation_hashes(env, marker):
    import torch
    obs = getattr(env, 'obs_buf', None)
    if obs is None or set(obs.keys()) != {'actor', 'critic'}:
        raise ValueError('existing initial observation buffer must contain actor and critic')
    for group, width in [('actor', 43), ('critic', marker['critic_width'])]:
        value = obs[group]
        if (not isinstance(value, torch.Tensor) or value.shape != (WORLDS, width)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError('existing initial observations require finite correctly shaped tensors')
    return dict(source='existing_env_obs_buf_no_recompute',
                actor_shape=list(obs['actor'].shape), critic_shape=list(obs['critic'].shape),
                actor_sha256=tensor_state_digest({'observation': obs['actor']}),
                critic_shared43_sha256=tensor_state_digest({'observation': obs['critic'][:, :43]}),
                scope='saved observations only; not full physical-state identity')


def _normalizer_hash(model):
    return tensor_state_digest({k: v for k, v in model.state_dict().items()
                                if k.startswith('obs_normalizer.')})


def _initialize(runner):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel, ImpactSpeedLowNoiseModel
    from src.tasks.hammer.rl.spline_credit_critic import initialize_matched_credit_models
    device = torch.device(runner.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('initialization supports onlycpu orallocatedCUDA')
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # Construct the same43-input reference independent of destination width.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED)
        x = torch.zeros(1, 43, device=device)
        obs = TensorDict({'actor': x, 'critic': x.clone()}, batch_size=[1])
        groups = {'actor': ['actor'], 'critic': ['critic']}
        keys = ('hidden_dims', 'activation', 'obs_normalization', 'distribution_cfg')
        actor_cfg = deepcopy({k: runner.cfg['actor'][k] for k in keys})
        critic_cfg = deepcopy({k: runner.cfg['critic'][k] for k in keys if k != 'distribution_cfg'})
        # RSL may omit/pop this optional key for a deterministic critic.
        critic_cfg['distribution_cfg'] = deepcopy(runner.cfg['critic'].get('distribution_cfg'))
        # RSL pops class_name while constructing the already-validated actor.
        expected = 'src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedGaussian'
        distribution = actor_cfg['distribution_cfg']
        if not isinstance(distribution, dict) or distribution.get('class_name', expected) != expected:
            raise ValueError('canonical reference requires the exact impact-speed distribution')
        distribution['class_name'] = expected
        a = ImpactSpeedLowNoiseModel(obs, groups, 'actor', 13, **actor_cfg).to(device)
        c = ImpactSpeedAgeModel(obs, groups, 'critic', 1, **critic_cfg).to(device)
    initialize_matched_credit_models(runner.alg.actor, runner.alg.critic, a, c)
    actor_hash = tensor_state_digest(runner.alg.actor.state_dict())
    critic_state = runner.alg.critic.state_dict()
    name = next(n for n, layer in runner.alg.critic.mlp.named_modules() if isinstance(layer, torch.nn.Linear))
    key = f'mlp.{name}.weight'
    if runner.alg.critic.obs_dim == 47 and torch.count_nonzero(critic_state[key][:, 43:]).item():
        raise ValueError('initial critic context columns differ from zero reference')
    shared = dict(critic_state); shared[key] = shared[key][:, :43]
    shared_hash = tensor_state_digest(shared)
    if actor_hash != tensor_state_digest(a.state_dict()) or shared_hash != tensor_state_digest(c.state_dict()):
        raise ValueError('actual initial actor/critic differ from common reference')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device.type == 'cuda': torch.cuda.manual_seed_all(SEED)
    np_state = np.random.get_state()
    np_serial = [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])]
    cuda_rng = torch.cuda.get_rng_state(device).cpu().numpy().tobytes() if device.type == 'cuda' else None
    return dict(actor=actor_hash,
                shared_critic43=shared_hash,
                critic=tensor_state_digest(runner.alg.critic.state_dict()),
                actor_normalizer=_normalizer_hash(runner.alg.actor), critic_normalizer=_normalizer_hash(runner.alg.critic),
                collection_torch_rng=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
                collection_python_rng=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                collection_numpy_rng=hashlib.sha256(json.dumps(np_serial, allow_nan=False).encode()).hexdigest(),
                collection_cuda_rng=hashlib.sha256(cuda_rng).hexdigest() if cuda_rng is not None else None)


@contextmanager
def matched_initial_state_gate(index, output_dir=None, *, enable_trace=False, reference_initial_state=None):
    cell = cell_for_index(index); output = _directory(output_dir)
    if type(enable_trace) is not bool or (enable_trace and output is None):
        raise ValueError('explicit trace requires boolean enable_trace and an output directory')
    if output is not None and any((output / name).exists() or (output / name).is_symlink()
            for name in ('initial-state.json', 'initial-state-mismatch.json')):
        raise FileExistsError('initial-state evidence already exists')
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    if 'learn' in _PATCHED: raise ValueError('initial gate already active')
    original = HammerOnPolicyRunner.learn; evidence = []
    def learn(runner, num_learning_iterations, init_at_random_ep_len=False):
        if (evidence or type(num_learning_iterations) is not int or num_learning_iterations != UPDATES
                or type(runner.current_learning_iteration) is not int or runner.current_learning_iteration != 0
                or type(runner.env.unwrapped.common_step_counter) is not int
                or runner.env.unwrapped.common_step_counter != 0 or runner.alg.optimizer.state):
            raise ValueError('fresh once-only100-update learning and empty optimizer required')
        marker = _live_recipe(runner, cell)
        observations = _initial_observation_hashes(runner.env.unwrapped, marker)
        hashes = _initialize(runner)
        row = dict(cell=cell, phase='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT', actual_counter=0,
                   critic_width=marker['critic_width'], contract=marker, hashes=hashes, initial_observations=observations,
                   model_initialization_seed=SEED, collection_seed=SEED,
                   live_cat=dict(velocity_pmax=.5, impulse_pmax=.2, scheduler=False))
        if reference_initial_state is not None:
            reference = reference_initial_state
            if (not isinstance(reference, dict) or reference.get('cell') != cell_for_index(0)
                    or reference.get('phase') != 'BEFORE_UPDATE_0_AND_FIRST_ROLLOUT'
                    or type(reference.get('actual_counter')) is not int or reference['actual_counter'] != 0):
                _initial_mismatch(output, row, reference, 'common initial reference has invalid identity')
            if any(hashes.get(k) != reference.get('hashes', {}).get(k) for k in COMMON_INITIAL_HASHES):
                _initial_mismatch(output, row, reference, 'common initial model or collection RNG fingerprints differ')
            observed = reference.get('initial_observations', {})
            if any(observations[k] != observed.get(k) for k in ('actor_sha256', 'critic_shared43_sha256')):
                _initial_mismatch(output, row, reference, 'common initial observed state differs')
        if output is not None: _write_once(output / 'initial-state.json', row)
        evidence.append(row)
        if enable_trace:
            from src.tasks.hammer.rl.credit_training_trace import attach_credit_training_trace
            with attach_credit_training_trace(runner, output / 'training-trace.jsonl', max_updates=UPDATES):
                return original(runner, num_learning_iterations, init_at_random_ep_len)
        return original(runner, num_learning_iterations, init_at_random_ep_len)
    _PATCHED.add('learn'); HammerOnPolicyRunner.learn = learn
    try:
        yield evidence
        if len(evidence) != 1: raise ValueError('training must pass initial gate exactly once')
    finally:
        HammerOnPolicyRunner.learn = original; _PATCHED.remove('learn')


@contextmanager
def selected_checkpoint_saves(index, output_dir=None):
    cell = cell_for_index(index); output = _directory(output_dir)
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    if 'save' in _PATCHED: raise ValueError('checkpoint filter already active')
    original = HammerOnPolicyRunner.save; evidence = []; seen = {}; owner = []
    def save(runner, path, infos=None):
        path = Path(path); match = re.fullmatch(r'model_(\d+)\.pt', path.name)
        if match is None: raise ValueError('unexpected checkpoint filename')
        label = int(match.group(1))
        if label not in range(UPDATES): raise ValueError('checkpoint outside finite update allocation')
        if label not in SAVE_INDICES: return None
        if output is not None and path != output / 'train' / path.name:
            raise ValueError('checkpoint outside exactcell train directory')
        counter = runner.env.unwrapped.common_step_counter
        if (type(counter) is not int or counter != (label + 1) * STEPS
                or type(runner.current_learning_iteration) is not int or runner.current_learning_iteration != label):
            raise ValueError('checkpoint update/transition clock differs')
        if owner and owner[0] is not runner: raise ValueError('checkpoint context belongs to another runner')
        _live_recipe(runner, cell)
        hashes = {k: tensor_state_digest(getattr(runner.alg, k).state_dict()) for k in ('actor', 'critic')}
        if label in seen:
            prior = seen[label]
            if (label != UPDATES - 1 or prior.get('deduplicated') or prior['hashes'] != hashes
                    or prior['checkpoint_sha256'] != _file_sha(path)):
                raise ValueError('unexpected repeated checkpoint save')
            prior['deduplicated'] = True
            row = dict(iteration=label, common_step_counter=counter, hashes=hashes,
                       checkpoint_sha256=prior['checkpoint_sha256'], deduplicated_final_save=True)
            if output is not None: _write_once(output / f'checkpoint-{label}-final-save-deduplicated.json', row)
            evidence.append(row); return None
        if path.exists() or path.is_symlink(): raise FileExistsError('checkpoint output is consumed')
        result = original(runner, str(path), infos)
        row = dict(iteration=label, common_step_counter=counter, hashes=hashes,
                   checkpoint_sha256=_file_sha(path), deduplicated_final_save=False)
        if output is not None: _write_once(output / f'checkpoint-{label}-live.json', row)
        seen[label] = dict(row); owner[:] = [runner]; evidence.append(row)
        return result
    _PATCHED.add('save'); HammerOnPolicyRunner.save = save
    try: yield evidence
    finally:
        HammerOnPolicyRunner.save = original; _PATCHED.remove('save')
