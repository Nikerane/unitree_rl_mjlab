"""Production qualification postconditions with CPU tensors, no learner execution."""

from copy import deepcopy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.z1_depth_campaign031 import probe
from tests.test_depth_probe_trace031 import assert_closed, callbacks
from tests.test_depth_reward031 import collect_depth_trace, trace_setup
from tests.test_velocity_population_trace import fixture as population_fixture


def adam_state(**changes):
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    values = dict(step=torch.tensor(20.), exp_avg=torch.tensor([.1, .2]),
                  exp_avg_sq=torch.tensor([.01, .04]))
    values.update(changes)
    return {parameter: values}


@pytest.fixture
def qualification_learn(monkeypatch, tmp_path):
    from scripts import z1_exploration_pilot_training
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

    runner, params, depth_manager = trace_setup()
    env = runner.env.unwrapped
    # Only the vector facade is production-sized; all observers use the
    # existing three-world CPU fixture, never a native environment.
    runner.env.num_envs = 1024
    runner.alg.optimizer = SimpleNamespace(state={})
    actor_weight = torch.tensor([1., 2.])
    runner.alg.actor = SimpleNamespace(state_dict=lambda: {'weight': actor_weight})
    runner.alg.critic.register_buffer('finite_weight', torch.tensor([.5, .6, .7]))
    runner.logger = SimpleNamespace(log_dir=str(tmp_path / 'production'))
    runner.current_learning_iteration = 0
    _, population_hook, _ = population_fixture(3)
    for name, value in vars(population_hook).items():
        setattr(env.hook, name, value)
    env._hammer_first_strike.productive = torch.zeros(3, dtype=torch.bool)
    env.metrics_manager.reset = lambda ids=None: None
    original_reset = env._spline_split_credit.reset

    def reset(ids):
        # The collection helper owns resets; route its pre-reset seam through
        # the actual attached population observer before clearing the buffers.
        env.metrics_manager.reset(ids)
        original_reset(ids)

    env._spline_split_credit.reset = reset
    previous = callbacks(runner)
    previous_reset = env.metrics_manager.reset
    observed = []
    result = object()
    boundary = {}

    def learner_boundary(actual_runner, updates, init_at_random_ep_len=False):
        assert actual_runner is runner and updates == 1 and init_at_random_ep_len is True
        population = env.metrics_manager.compute.__self__
        depth = population.saved[0][2].__self__
        observed.append((depth, depth.stream, population, population.stream))
        collect_depth_trace(runner, params, depth_manager)
        runner.alg.compute_returns(None)
        runner.alg.storage.step = 0
        runner.alg.optimizer.state = boundary['state']
        boundary['storage'] = deepcopy(vars(runner.alg.storage))
        boundary['models'] = [deepcopy(model.state_dict())
                              for model in (runner.alg.actor, runner.alg.critic)]
        boundary['torch_rng'] = torch.random.get_rng_state().clone()
        boundary['python_rng'] = random.getstate()
        boundary['numpy_rng'] = deepcopy(np.random.get_state())
        return result

    # Canonical initialization alone constructs real models. Keep attachment,
    # collection, joins, actual JSON writes and postconditions unpatched.
    monkeypatch.setattr(z1_exploration_pilot_training, '_initialize', lambda *a: {'cpu_fixture': True})
    monkeypatch.setattr(HammerOnPolicyRunner, 'learn', learner_boundary)

    def run(state, *, error=None):
        boundary['state'] = state
        prior = [(parameter, values, [(name, value, value.clone() if isinstance(value, torch.Tensor)
                                      else value, value._version if isinstance(value, torch.Tensor) else None)
                                     for name, value in values.items()])
                 for parameter, values in state.items()] if isinstance(state, dict) and all(
                     isinstance(values, dict) for values in state.values()) else None
        counts = {name: 0 for name in probe.COUNTS}
        counts.update(native_autoreset_worlds=0, physical_depth_observer_calls=0, synthetic_active=False)
        with probe.counters(counts):
            if error is None:
                assert HammerOnPolicyRunner.learn(runner, 1, init_at_random_ep_len=True) is result
            else:
                with pytest.raises(ValueError, match=error):
                    HammerOnPolicyRunner.learn(runner, 1, init_at_random_ep_len=True)
        assert len(observed) == counts['production_learn_calls'] == 1
        depth, depth_stream, population, population_stream = observed[0]
        assert_closed(runner, previous, depth, depth_stream)
        assert env.metrics_manager.reset is previous_reset
        assert population.stream is None and population_stream.closed
        assert runner.alg.optimizer.state is state
        if prior is not None:
            assert list(state) == [parameter for parameter, _, _ in prior]
            for parameter, values, fields in prior:
                assert state[parameter] is values and list(values) == [name for name, _, _, _ in fields]
                for name, value, saved, version in fields:
                    assert values[name] is value
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, saved, rtol=0, atol=0, equal_nan=True)
                        assert value._version == version
        for model, saved in zip((runner.alg.actor, runner.alg.critic), boundary['models']):
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        for name, saved in boundary['storage'].items():
            value = getattr(runner.alg.storage, name)
            if isinstance(saved, torch.Tensor):
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
            else:
                assert value == saved
        assert torch.equal(boundary['torch_rng'], torch.random.get_rng_state())
        assert boundary['python_rng'] == random.getstate()
        current_numpy = np.random.get_state()
        assert boundary['numpy_rng'][0] == current_numpy[0]
        np.testing.assert_array_equal(boundary['numpy_rng'][1], current_numpy[1])
        assert boundary['numpy_rng'][2:] == current_numpy[2:]
        assert runner.alg.critic.calls == 1 and env.metrics_manager.calls == 24
        assert all(counts[name] == 0 for name in (
            'constructor_worlds', 'vector_control_steps', 'actor_forwards', 'critic_forwards',
            'model_constructors', 'algorithm_updates', 'optimizer_steps', 'save_calls', 'load_calls'))
        rows = [json.loads(line) for line in (tmp_path / 'qualification-depth-trace.jsonl').read_text().splitlines()]
        assert len(rows) == 1 and rows[0]['schema'] == 'z1_depth_reward031_training_trace_v1'
        assert rows[0]['max_updates'] == 500 and len(rows[0]['steps']) == 24
        population_rows = [json.loads(line) for line in (tmp_path / 'qualification-population.jsonl').read_text().splitlines()]
        assert len(population_rows) == 2 and population_rows[0]['kind'] == 'update'
        assert population_rows[1]['updates'] == 1

    return run


@pytest.mark.parametrize('field', ['step', 'exp_avg', 'exp_avg_sq'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', '-inf'])
def test_qualification_rejects_nonfinite_adam_moment_with_finite_models(qualification_learn, field, bad):
    value = torch.tensor(bad) if field == 'step' else torch.tensor([.1, bad])
    qualification_learn(adam_state(**{field: value}),
                        error=r'nonfinite production optimizer state.*' + field)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', '-inf'])
def test_qualification_rejects_nonfinite_numeric_state_in_later_parameter(qualification_learn, bad):
    state = adam_state()
    state.update(adam_state(step=bad))
    qualification_learn(state, error=r'nonfinite production optimizer state\[1\].step')


@pytest.mark.parametrize('step', [torch.tensor(20.), 20, 20.], ids=['tensor', 'int', 'float'])
def test_qualification_accepts_populated_finite_state_without_mutation(qualification_learn, step):
    state = adam_state(step=step, exp_avg=torch.tensor([.1, .2], requires_grad=True))
    state.update(adam_state())
    qualification_learn(state)


def test_qualification_still_rejects_empty_optimizer_after_learn(qualification_learn):
    qualification_learn({}, error='production PPO completion differs')


@pytest.mark.parametrize('state', [
    [1],
    {torch.tensor([1.]): [1]},
    {torch.tensor([1.]): {}},
    {0: {'step': 20}},
    {torch.tensor([1.]): {0: 20}},
    {torch.tensor([1.]): {'': 20}},
], ids=['outer-sequence', 'entry-sequence', 'empty-entry', 'serialized-key', 'numeric-field', 'empty-field'])
def test_qualification_rejects_malformed_optimizer_state(qualification_learn, state):
    qualification_learn(state, error='malformed production optimizer state')


@pytest.mark.parametrize('value', [None, '20', True, object(), [20.], {'step': 20.}, torch.empty(0)],
                         ids=['none', 'string', 'bool', 'object', 'sequence', 'nested', 'empty-tensor'])
def test_qualification_rejects_unsupported_or_empty_state_value(qualification_learn, value):
    qualification_learn(adam_state(exp_avg=value),
                        error=r'(unsupported|malformed) production optimizer state.*exp_avg')


@pytest.fixture(autouse=True)
def forbid_native_model_learner_or_optimizer(monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    from rsl_rl.models import MLPModel
    from src.tasks.hammer.rl.cat_ppo import CatPPO

    def forbidden(*args, **kwargs):
        raise AssertionError('native/model/learner/optimizer/checkpoint execution forbidden in CPU optimizer regression')

    monkeypatch.setattr(ManagerBasedRlEnv, '__init__', forbidden)
    monkeypatch.setattr(MLPModel, '__init__', forbidden)
    monkeypatch.setattr(CatPPO, 'update', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
