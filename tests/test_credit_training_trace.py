"""Pre-reset observational training traces joined to the real CaT GAE calculation."""
from copy import deepcopy
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.tasks.hammer.rl.cat_ppo import CatPPO


def trace_module():
    name = 'src.tasks.hammer.rl.credit_training_trace'
    assert importlib.util.find_spec(name) is not None, 'bounded training trace module is missing'
    return importlib.import_module(name)


class Critic(nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0
    def forward(self, obs):
        self.calls += 1
        return torch.tensor([[.5], [.6], [.7]])


class Metrics:
    def __init__(self, env): self.env = env; self.calls = 0; self.fail = False
    def compute(self):
        self.calls += 1
        if self.fail: raise RuntimeError('native metrics failure')
        self.env.extras = {'cat_delta': self.env.delta, 'cat_r_pos': self.env.r_pos}
    @property
    def cfg(self):
        return {'cat_soft': SimpleNamespace(func=self.env.hook)}
    def get_term_cfg(self, name):
        assert name == 'cat_soft'
        return SimpleNamespace(func=self.env.hook)


class Hook:
    def __init__(self, env): self.env = env
    def constraint_telemetry(self):
        d = self.env.delta
        return {'delta_velocity': d.clone(), 'delta_impulse': torch.zeros_like(d),
                'active_limit_per_joint': torch.tensor([.2] * 6),
                'lambda_per_joint': torch.ones(3, 6) * .1}


class Algorithm:
    def __init__(self, storage):
        self.storage = storage; self.critic = Critic(); self.gamma = .99; self.lam = .95
        self.normalize_advantage_per_mini_batch = False
        self.transition = SimpleNamespace(values=None, actions=None, distribution_params=None)
        self.fail = False
    def compute_returns(self, obs):
        if self.fail: raise RuntimeError('native return failure')
        return CatPPO.compute_returns(self, obs)


def setup():
    t, n = 24, 3
    values = torch.arange(t * n).reshape(t, n, 1).float() / 100
    soft = torch.zeros(t, n, 1); soft[-1, 0] = .5
    done = torch.zeros(t, n, 1, dtype=torch.uint8); done[-1, 0] = 1; done[3, 1] = 1
    raw = torch.full((t, n), -.1); raw[-1] = .9
    pos = torch.zeros(t, n); pos[-1] = 1.
    rewards = raw.unsqueeze(-1) - soft * pos.unsqueeze(-1)
    rewards[3, 1] += .99 * values[3, 1]  # Native timeout bootstrap is ALREADY stored.
    st = SimpleNamespace(num_transitions_per_env=t, step=0, values=values,
        soft_dones=soft, dones=done, rewards=rewards, returns=torch.zeros_like(values),
        advantages=torch.zeros_like(values), actions=torch.arange(t * n * 13).reshape(t, n, 13).float() / 100)
    env = SimpleNamespace(num_envs=n, extras={}, reward_buf=torch.zeros(n), reset_buf=torch.zeros(n, dtype=torch.bool),
        reset_time_outs=torch.zeros(n, dtype=torch.bool), action_manager=SimpleNamespace(action=None),
        _spline_deadline_credit=SimpleNamespace(unpaid_credit=torch.zeros(n)),
        _hammer_first_strike=SimpleNamespace(started=torch.zeros(n, dtype=torch.bool), elapsed_since_contact_s=torch.zeros(n)))
    env.step_dt = .02
    env.reward_manager = SimpleNamespace(active_terms=['cost', 'accepted_spline_strike'], _scale_by_dt=True, _step_reward=torch.zeros(3, 2))
    env.hook = Hook(env); env.metrics_manager = Metrics(env)
    alg = Algorithm(st)
    return SimpleNamespace(env=SimpleNamespace(unwrapped=env), alg=alg), raw, pos


def collect(runner, raw, pos, count=24, *, clear_after=True, optional=True):
    env, alg, st = runner.env.unwrapped, runner.alg, runner.alg.storage
    for step in range(count):
        st.step = step
        env.delta = st.soft_dones[step, :, 0].clone(); env.r_pos = pos[step].clone()
        env.reward_buf = raw[step].clone()
        if hasattr(env, 'reward_manager'):
            env.reward_manager._step_reward = torch.stack((raw[step] - pos[step], pos[step]), dim=-1) / env.step_dt
            if getattr(env, 'bad_ledger', False): env.reward_manager._step_reward[0, 0] = float('nan')
        env.reset_buf = st.dones[step, :, 0].bool()
        env.reset_time_outs = torch.tensor([False, step == 3, False])
        env._hammer_first_strike.started = torch.ones(3, dtype=torch.bool)
        env._hammer_first_strike.elapsed_since_contact_s = torch.ones(3) * step * .02
        env._spline_deadline_credit.unpaid_credit = torch.ones(3) if step < 23 else torch.zeros(3)
        alg.transition.values = st.values[step]
        alg.transition.actions = st.actions[step]
        alg.transition.distribution_params = (st.actions[step], torch.ones(3, 13), torch.ones(3, 13)) if optional else None
        env.action_manager.action = st.actions[step].clamp(-1, 1)
        env.metrics_manager.compute()
        if clear_after:
            env.reward_buf.zero_(); env.delta.zero_(); env.r_pos.zero_()
            if hasattr(env, 'reward_manager'): env.reward_manager._step_reward.zero_()
            env.reset_buf = torch.zeros(3, dtype=torch.bool)
            env._hammer_first_strike.started.zero_(); env._hammer_first_strike.elapsed_since_contact_s.zero_()
            env._spline_deadline_credit.unpaid_credit.zero_()
    st.step = count


def restored(runner):
    assert 'compute' not in runner.env.unwrapped.metrics_manager.__dict__
    assert 'compute_returns' not in runner.alg.__dict__
    assert not runner.alg.critic._forward_hooks


def test_trace_preserves_terminal_data_joins_actual_storage_and_does_not_change_rng_or_returns(tmp_path):
    runner, raw, pos = setup(); path = tmp_path / 'trace.jsonl'
    control = deepcopy(runner.alg.storage); control.step = 24
    control_alg = Algorithm(control); control_alg.compute_returns(None)
    rng = torch.random.get_rng_state().clone()
    with trace_module().attach_credit_training_trace(runner, path):
        collect(runner, raw, pos)
        runner.alg.compute_returns(None)
    restored(runner)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(runner.alg.storage.returns, control.returns, rtol=0, atol=0)
    torch.testing.assert_close(runner.alg.storage.advantages, control.advantages, rtol=0, atol=0)
    assert runner.alg.critic.calls == 1 and runner.env.unwrapped.metrics_manager.calls == 24
    doc = json.loads(path.read_text()); assert doc['world_ids'] == [0, 1] and len(doc['steps']) == 24
    last = doc['steps'][-1]
    assert last['pre_reset']['raw_reward'][0] == pytest.approx(.9)
    assert last['pre_reset']['delta'] == [.5, 0.]
    assert last['pre_reset']['native_reset'] == [True, False]
    assert last['pre_reset']['started'] == [True, True]
    assert last['pre_reset']['accepted_credit_paid'] == [1., 1.]
    assert last['pre_reset']['reward_components']['accepted_spline_strike'] == [1., 1.]
    assert last['pre_reset']['reward_components']['cost'] == pytest.approx([-.1, -.1], abs=1e-6)
    assert last['pre_reset']['r_neg_raw_minus_positive'] == (raw[-1, :2] - pos[-1, :2]).tolist()
    assert last['pre_reset']['r_neg_semantics'] == 'RAW_MINUS_POSITIVE_NOT_UNROUNDED_PRODUCER_LEDGER'
    assert len(last['pre_reset']['latent_actions']) == 2
    assert last['pre_reset']['constraints']['active_limit_per_joint'] == pytest.approx([.2] * 6)
    assert last['stored']['td_residual'][0] == pytest.approx((.9 - .5) - float(control.values[-1, 0]), abs=1e-6)
    assert last['stored']['raw_advantage'][0] == pytest.approx(float(control.returns[-1, 0] - control.values[-1, 0]))
    expected = float(control.rewards[3, 1] - control.values[3, 1])
    assert doc['steps'][3]['stored']['td_residual'][1] == pytest.approx(expected)
    assert doc['steps'][3]['pre_reset']['timeout'] == [False, True]
    for step, row in enumerate(doc['steps']):
        nxt = torch.tensor([[.5], [.6], [.7]]) if step == 23 else control.values[step + 1]
        td = control.rewards[step] + (1 - control.soft_dones[step]) * (1 - control.dones[step].float()) * .99 * nxt - control.values[step]
        assert row['stored']['td_residual'] == td[:2, 0].tolist()
        assert row['stored']['raw_advantage'] == (control.returns[step] - control.values[step])[:2, 0].tolist()


def test_unavailable_optional_distribution_and_constraint_telemetry_stay_unknown(tmp_path):
    runner, raw, pos = setup(); runner.env.unwrapped.hook = object(); path = tmp_path / 'trace.jsonl'
    with trace_module().attach_credit_training_trace(runner, path):
        collect(runner, raw, pos, optional=False); runner.alg.compute_returns(None)
    snapshot = json.loads(path.read_text())['steps'][0]['pre_reset']
    assert snapshot['distribution_mean'] is None and snapshot['distribution_std'] is None
    assert snapshot['constraints'] is None and 'UNKNOWN' in snapshot['constraint_status']


@pytest.mark.parametrize('bad', [{'world_ids': (1, 2)}, {'world_ids': (False, 1)}, {'max_updates': True}, {'max_updates': 101}, {'max_bytes': 0}, {'max_bytes': 32 * 1024**2 + 1}])
def test_invalid_scope_or_bounds_never_attach_or_create_output(tmp_path, bad):
    runner, _, _ = setup(); path = tmp_path / 'trace.jsonl'
    with pytest.raises(ValueError):
        with trace_module().attach_credit_training_trace(runner, path, **bad): pass
    assert not path.exists(); restored(runner)


def test_existing_path_never_overwrites(tmp_path):
    runner, _, _ = setup(); path = tmp_path / 'trace.jsonl'; path.write_text('preserve')
    with pytest.raises(FileExistsError):
        with trace_module().attach_credit_training_trace(runner, path): pass
    assert path.read_text() == 'preserve'; restored(runner)


@pytest.mark.parametrize('failure', ['metrics', 'returns', 'partial', 'join', 'cap', 'updates'])
def test_failures_restore_all_methods_hooks_and_never_write_a_partial_envelope(tmp_path, failure):
    runner, raw, pos = setup(); path = tmp_path / 'trace.jsonl'
    kwargs = {'max_bytes': 64} if failure == 'cap' else {'max_updates': 1}
    with pytest.raises((ValueError, RuntimeError)):
        with trace_module().attach_credit_training_trace(runner, path, **kwargs):
            if failure == 'metrics': runner.env.unwrapped.metrics_manager.fail = True
            collect(runner, raw, pos, count=23 if failure == 'partial' else 24)
            if failure == 'returns': runner.alg.fail = True
            if failure == 'join': runner.alg.storage.values[0, 0] += 1
            runner.alg.compute_returns(None)
            if failure == 'updates': collect(runner, raw, pos)
    restored(runner)
    if failure == 'updates': assert len(path.read_text().splitlines()) == 1
    else: assert path.read_bytes() == b''


def test_partial_normal_context_exit_is_not_silently_accepted(tmp_path):
    runner, raw, pos = setup(); path = tmp_path / 'trace.jsonl'
    with pytest.raises(ValueError):
        with trace_module().attach_credit_training_trace(runner, path): collect(runner, raw, pos, count=1)
    restored(runner)


@pytest.mark.parametrize('bad', ['missing', 'duplicates', 'too_many', 'shape', 'nonfinite', 'scale', 'dt'])
def test_reward_ledger_must_be_real_finite_aligned_and_dt_scaled(tmp_path, bad):
    runner, raw, pos = setup(); env = runner.env.unwrapped; path = tmp_path / 'trace.jsonl'
    if bad == 'missing': del env.reward_manager
    elif bad == 'duplicates': env.reward_manager.active_terms = ['cost', 'cost']
    elif bad == 'too_many': env.reward_manager.active_terms = [f'term{i}' for i in range(65)]
    elif bad == 'shape': env.reward_manager.active_terms = ['cost']
    elif bad == 'scale': env.reward_manager._scale_by_dt = False
    elif bad == 'dt': env.step_dt = float('nan')
    if bad == 'nonfinite': env.bad_ledger = True
    with pytest.raises((ValueError, AttributeError)):
        with trace_module().attach_credit_training_trace(runner, path):
            collect(runner, raw, pos); runner.alg.compute_returns(None)
    restored(runner); assert path.read_bytes() == b''


def test_missing_accepted_credit_component_is_unknown_not_inferred_from_balance(tmp_path):
    runner, raw, pos = setup(); runner.env.unwrapped.reward_manager.active_terms = ['cost', 'other_positive']
    path = tmp_path / 'trace.jsonl'
    with trace_module().attach_credit_training_trace(runner, path):
        collect(runner, raw, pos); runner.alg.compute_returns(None)
    doc = json.loads(path.read_text())
    assert doc['steps'][-1]['pre_reset']['accepted_credit_paid'] is None
    assert doc['steps'][-1]['pre_reset']['reward_components']['other_positive'] == [1., 1.]


def test_live_hook_uses_real_metrics_cfg_without_a_get_term_cfg_method(monkeypatch,tmp_path):
    runner, raw, pos = setup(); path = tmp_path / 'trace.jsonl'
    monkeypatch.delattr(Metrics,'get_term_cfg')
    assert not hasattr(runner.env.unwrapped.metrics_manager,'get_term_cfg')
    with trace_module().attach_credit_training_trace(runner,path):
        collect(runner,raw,pos);runner.alg.compute_returns(None)
    doc=json.loads(path.read_text())
    for row in doc['steps']:
        assert row['pre_reset']['constraint_status']=='OBSERVED_LIVE_PRE_RESET'
        assert row['pre_reset']['constraints']is not None
        assert row['pre_reset']['constraints']['delta_velocity']==row['pre_reset']['delta']
    restored(runner)
