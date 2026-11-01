"""Qualification capture limits on actual031 traces; tensor fixtures, no physics."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.z1_depth_campaign031 import probe
from src.tasks.hammer.rl.depth_reward031_trace import (
    _DepthTrace031, attach_credit_training_trace, validate_depth_trace_row031,
)
from tests.test_depth_reward031 import collect_depth_trace, trace_setup


def callbacks(runner):
    manager = runner.env.unwrapped.metrics_manager
    return (manager.compute, 'compute' in manager.__dict__,
            runner.alg.compute_returns, 'compute_returns' in runner.alg.__dict__)


def assert_closed(runner, previous, trace, stream):
    assert callbacks(runner) == previous
    assert not runner.alg.critic._forward_hooks
    assert trace.stream is None and stream.closed and not trace.attached


def test_qualification_learn_reaches_caller_and_restores_real_traces_on_error(monkeypatch, tmp_path):
    """The old max1 attachment must fail here at the real hard500 constructor."""
    from scripts import z1_exploration_pilot_training
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from src.tasks.hammer.rl.velocity_population_trace import _Population
    from tests.test_velocity_population_trace import fixture

    runner, _, _ = trace_setup()
    env = runner.env.unwrapped
    # The outer learn admission sees its production-sized vector facade. The
    # observers deliberately retain the existing three-world CPU tensor fixture.
    runner.env.num_envs = 1024
    runner.alg.optimizer = SimpleNamespace(state={})
    runner.logger = SimpleNamespace(log_dir=str(tmp_path / 'production'))
    _, population_hook, _ = fixture(3)
    for name, value in vars(population_hook).items():
        setattr(env.hook, name, value)
    env.metrics_manager.reset = lambda ids=None: None
    previous = callbacks(runner)
    previous_reset = env.metrics_manager.reset
    caller_error = RuntimeError('CPU caller sentinel; no learner executed')
    observed = []

    def learner_boundary(runner, updates, init_at_random_ep_len=False):
        population = env.metrics_manager.compute.__self__
        depth = population.saved[0][2].__self__
        assert type(population) is _Population and population.max_updates == 1
        assert type(depth) is _DepthTrace031 and depth.max_updates == 1
        assert updates == 1 and init_at_random_ep_len is True
        observed.append((depth, depth.stream, population, population.stream))
        raise caller_error

    # Canonical initialization constructs real models, which this CPU fixture
    # forbids. Leave all trace construction, writes and cleanup unpatched.
    monkeypatch.setattr(z1_exploration_pilot_training, '_initialize', lambda *a: {'cpu_fixture': True})
    monkeypatch.setattr(HammerOnPolicyRunner, 'learn', learner_boundary)
    counts = {name: 0 for name in probe.COUNTS}
    counts.update(native_autoreset_worlds=0, physical_depth_observer_calls=0, synthetic_active=False)
    with pytest.raises(RuntimeError, match='CPU caller sentinel') as caught:
        with probe.counters(counts):
            HammerOnPolicyRunner.learn(runner, 1, init_at_random_ep_len=True)
    assert caught.value is caller_error
    assert len(observed) == 1 and counts['production_learn_calls'] == 1
    depth, depth_stream, population, population_stream = observed[0]
    assert_closed(runner, previous, depth, depth_stream)
    assert env.metrics_manager.reset is previous_reset
    assert population.stream is None and population_stream.closed
    assert (tmp_path / 'qualification-depth-trace.jsonl').read_bytes() == b''
    final = json.loads((tmp_path / 'qualification-population.jsonl').read_text())
    assert final['status'] == 'FAILED_PARTIAL_EVIDENCE' and final['updates'] == 0
    assert all(counts[name] == 0 for name in (
        'constructor_worlds', 'vector_control_steps', 'actor_forwards', 'critic_forwards',
        'model_constructors', 'algorithm_updates', 'optimizer_steps', 'save_calls', 'load_calls'))


def test_qualification_captures_one_joined_row_under_unchanged500_schema(tmp_path):
    runner, params, depth_manager = trace_setup()
    previous = callbacks(runner)
    rng = torch.random.get_rng_state().clone()
    path = tmp_path / 'one.jsonl'
    initial_storage = deepcopy(runner.alg.storage)
    with probe.qualification_depth_trace031(runner, path) as trace:
        stream = trace.stream
        assert type(trace) is _DepthTrace031 and trace.max_updates == 1
        assert trace.cap == 160 * 1024**2 and trace.updates == 0
        for field in ('values', 'rewards', 'returns', 'actions'):
            assert torch.equal(getattr(runner.alg.storage, field), getattr(initial_storage, field))
        collect_depth_trace(runner, params, depth_manager)
        runner.alg.compute_returns(None)
        assert trace.updates == 1 and trace.buffer == []
    assert_closed(runner, previous, trace, stream)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row['schema'] == 'z1_depth_reward031_training_trace_v1'
    assert row['max_updates'] == 500 and row['update'] == 0 and len(row['steps']) == 24
    assert row['steps'][0]['pre_reset']['depth_credit031']['progress_paid'] == [1., 1.]
    assert row['steps'][9]['pre_reset']['depth_credit031']['completion_paid'] == [1., 0.]
    assert row['depth_population031']['native_episodes'] == 3
    validate_depth_trace_row031(row, 0)
    wrong_schema_cap = deepcopy(row)
    wrong_schema_cap['max_updates'] = 1
    with pytest.raises(ValueError):
        validate_depth_trace_row031(wrong_schema_cap, 0)
    assert runner.alg.critic.calls == 1 and runner.env.unwrapped.metrics_manager.calls == 24
    assert torch.equal(rng, torch.random.get_rng_state())
    metadata = probe.recipe()['production_trace']
    assert metadata['depth_rows'] == 1 and metadata['population_rows'] == 2
    assert metadata['max_updates'] == 1 and metadata['schema_max_updates'] == 500


def test_qualification_rejects_second_capture_without_writing_second_row(tmp_path):
    runner, params, depth_manager = trace_setup()
    previous = callbacks(runner)
    path = tmp_path / 'cap.jsonl'
    with probe.qualification_depth_trace031(runner, path) as trace:
        stream = trace.stream
        collect_depth_trace(runner, params, depth_manager)
        runner.alg.compute_returns(None)
        with pytest.raises(ValueError, match='cap exceeded'):
            runner.env.unwrapped.metrics_manager.compute()
        assert trace.updates == 1 and trace.buffer == []
    assert_closed(runner, previous, trace, stream)
    assert len(path.read_text().splitlines()) == 1 and runner.alg.critic.calls == 1


@pytest.mark.parametrize('collect_unjoined', [False, True])
def test_qualification_requires_one_completed_join_on_normal_exit(tmp_path, collect_unjoined):
    runner, params, depth_manager = trace_setup()
    previous = callbacks(runner)
    path = tmp_path / 'incomplete.jsonl'
    with pytest.raises(ValueError, match='one completed joined update'):
        with probe.qualification_depth_trace031(runner, path) as trace:
            stream = trace.stream
            if collect_unjoined:
                collect_depth_trace(runner, params, depth_manager)
    assert_closed(runner, previous, trace, stream)
    assert path.read_bytes() == b'' and runner.alg.critic.calls == 0


@pytest.mark.parametrize('after_join', [False, True])
def test_qualification_propagates_caller_error_and_closes_pending_or_joined_trace(tmp_path, after_join):
    runner, params, depth_manager = trace_setup()
    previous = callbacks(runner)
    path = tmp_path / 'caller-error.jsonl'
    error = RuntimeError('original caller failure')
    with pytest.raises(RuntimeError) as caught:
        with probe.qualification_depth_trace031(runner, path) as trace:
            stream = trace.stream
            collect_depth_trace(runner, params, depth_manager)
            if after_join:
                runner.alg.compute_returns(None)
            raise error
    assert caught.value is error
    assert_closed(runner, previous, trace, stream)
    assert len(path.read_text().splitlines()) == int(after_join)


@pytest.mark.parametrize('trace_kind', ['depth', 'split'])
def test_production_factories_still_reject_one_update_before_creating_output(tmp_path, trace_kind):
    from src.tasks.hammer.rl.split_credit_training_trace import attach_credit_training_trace as attach_split
    runner, _, _ = trace_setup()
    previous = callbacks(runner)
    path = tmp_path / 'production.jsonl'
    factory = attach_credit_training_trace if trace_kind == 'depth' else attach_split
    with pytest.raises(ValueError, match='requires exactly500 updates'):
        with factory(runner, path, max_updates=1):
            pytest.fail('production admitted qualification cap')
    assert callbacks(runner) == previous and not path.exists()


@pytest.fixture(autouse=True)
def forbid_native_construction_or_checkpoint(monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    from rsl_rl.models import MLPModel

    def forbidden(*args, **kwargs):
        raise AssertionError('native environment/model/checkpoint execution forbidden in CPU trace regression')

    monkeypatch.setattr(ManagerBasedRlEnv, '__init__', forbidden)
    monkeypatch.setattr(MLPModel, '__init__', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
