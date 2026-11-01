"""Bounded031 two-world depth/split/CaT joins plus all-world depth totals.

Extends the existing split observer; observes the critic forward already used
by compute_returns, never adds a forward, RNG draw or physical sample.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import torch
from src.tasks.hammer.rl.split_credit_training_trace import _SplitCreditTrace, _need, _LIMIT
from src.tasks.hammer.rl.depth_reward031_contract import depth_identity031, validate_depth_identity031
from src.tasks.hammer.mdp.depth_reward031 import NativeDepthCreditObserver031


class _DepthTrace031(_SplitCreditTrace):
    def __init__(self, runner, path, world_ids, max_bytes, max_updates):
        super().__init__(runner, path, world_ids, max_bytes, max_updates)
        owner = getattr(self.env, '_z1_depth_credit031', None)
        _need(type(owner) is NativeDepthCreditObserver031 and owner.call_index == 0,
              'depth031 trace requires fresh exact observer')
        self.depth_marker = depth_identity031(owner.arm)
        self.last_depth_call = 0
        self.population = dict.fromkeys(('native_episodes', 'attained_native_episodes',
            'progress_paid_units', 'completion_paid_units', 'completion_forfeited_units',
            'external_completion_forfeited_units', 'budget_violations'), 0)

    def metrics(self, *args, **kwargs):
        try:
            result = super().metrics(*args, **kwargs)
            owner = self.env._z1_depth_credit031
            snap = owner.last_snapshot  # Never collect from tracing after reset.
            _need(owner._counter == self.env.common_step_counter and owner.call_index == self.last_depth_call + 1
                  and isinstance(snap, dict), 'depth031 trace requires one fresh pre-reset receipt')
            self.last_depth_call = owner.call_index
            pre = self.buffer[-1]['pre_reset']
            pre['depth_credit031'] = {k: self.take(v, vector=True) for k,v in snap.items()}
            for field, component in (('progress_paid','native_depth_progress031'),
                                      ('completion_paid','native_depth_completion031')):
                actual = pre['reward_components'].get(component, [0.,0.])
                _need(all(math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-5)
                    for a,b in zip(actual, pre['depth_credit031'][field])), 'depth031 actual reward component differs')
            p = self.population
            p['native_episodes'] += int(snap['native_reset'].sum())
            p['attained_native_episodes'] += int((snap['native_reset'] & snap['attained']).sum())
            for key, field in (('progress_paid_units','progress_paid'),('completion_paid_units','completion_paid'),
                    ('completion_forfeited_units','completion_forfeited'),
                    ('external_completion_forfeited_units','externally_forfeited_completion')):
                p[key] += float(snap[field].sum())
            p['budget_violations'] += int(((snap['progress_paid_total'] > 1.00002)
                | (snap['completion_paid_total'] > 1.)).sum())
            return result
        except BaseException:
            self.close()
            raise

    def returns(self, *args, **kwargs):
        handle = None
        try:
            st = self.alg.storage
            _need(self.updates < self.max_updates and len(self.buffer) == 24 and st.step == 24,
                  'trace requires one full24-step rollout before returns')
            captured = []
            def observe(_model, _inputs, output):
                self.take(output, vector=True)  # Validate the exact output already used by CaT.
                captured.append(output.detach().clone())
            handle = self.alg.critic.register_forward_hook(observe)
            try:
                result = self.original_returns(*args, **kwargs)
            finally:
                handle.remove()
                handle = None
            _need(len(captured) == 1, 'trace expected exactly one existing final critic forward')
            for step, row in enumerate(self.buffer):
                take = lambda value: self.take(value[step], vector=True)
                values, rewards = take(st.values), take(st.rewards)
                soft, done = take(st.soft_dones), take(st.dones)
                returns = take(st.returns)
                before = row['pre_reset']
                _need(values == before['value_at_action'] and soft == before['delta']
                      and done == before['native_reset'], 'trace join does not match collected transition')
                if hasattr(st, 'actions'):
                    _need(self.take(st.actions[step]) == before['latent_actions'], 'trace action join mismatch')
                next_tensor = captured[0] if step == 23 else st.values[step + 1]
                next_values = self.take(next_tensor, vector=True)
                continuation = (1. - st.soft_dones[step]) * (1. - st.dones[step].float())
                td_tensor = st.rewards[step] + continuation * self.alg.gamma * next_tensor - st.values[step]
                td = self.take(td_tensor, vector=True)
                advantage = self.take(st.returns[step] - st.values[step], vector=True)
                row['stored'] = dict(values=values, rewards_after_cat_and_bootstrap=rewards,
                    soft_dones=soft, hard_dones=done, returns=returns, next_values=next_values,
                    raw_advantage=advantage, td_residual=td)
            envelope = dict(schema='z1_depth_reward031_training_trace_v1', update=self.updates,
                split_credit=self.split_marker, max_updates=500, physics_dt_s=self.env.physics_dt, policy_dt_s=self.env.step_dt,
                world_ids=list(self.ids), scope='FIXED_TWO_WORLDS_NOT_POPULATION_STATISTICS',
                gamma=self.alg.gamma, lam=self.alg.lam, steps=self.buffer,
                final_values=self.take(captured[0], vector=True),
                executed_action_semantics='ACTION_MANAGER_INPUT_AFTER_WRAPPER_CLIP_NOT_MOTOR_TORQUE',
                storage_reward_semantics='ACTUAL_ALREADY_CAT_SCALED_AND_TIMEOUT_BOOTSTRAPPED',
                raw_advantage_semantics='RETURNS_MINUS_VALUES_BEFORE_ADVANTAGE_NORMALIZATION')
            envelope['spline_depth_reward031'] = self.depth_marker
            envelope['depth_population031'] = dict(self.population)
            body = (json.dumps(envelope, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
            _need(self.written + len(body) <= self.cap, 'trace byte cap would be exceeded')
            offset = 0
            while offset < len(body):
                count = self.stream.write(body[offset:])
                _need(type(count) is int and count > 0, 'trace write made no progress')
                offset += count
                self.written += count
            self.population = dict.fromkeys(self.population, 0)
            self.buffer = []
            self.updates += 1
            return result
        except BaseException:
            if handle is not None:
                handle.remove()
            self.close()
            raise


@contextmanager
def attach_credit_training_trace(runner, output_path, world_ids=(0,1), max_bytes=_LIMIT, max_updates=500):
    """Same bounded entry point consumed by the generic021 training gate."""
    trace = _DepthTrace031(runner, output_path, world_ids, max_bytes, max_updates)
    try:
        trace.start()
        yield trace
        _need(not trace.buffer, 'depth031 trace ended with unjoined partial rollout')
    finally:
        trace.close()


def validate_depth_trace_row031(row, update, *, previous=None):
    """Validate031 additions and reuse unchanged split/CaT chronology checks.

    The projection is in memory only; persisted rows retain031 identity.
    Return the detached full pre-reset terminal state for the next update.
    """
    from scripts.z1_timing_noise020.split_contract import validate_trace_row
    _need(type(row) is dict and row.get('schema') == 'z1_depth_reward031_training_trace_v1',
          'depth031 trace schema differs')
    marker = validate_depth_identity031(row.get('spline_depth_reward031'))
    view = deepcopy(row)
    view['schema'] = 'z1_split_credit_training_trace_v1'
    del view['spline_depth_reward031']
    population = view.pop('depth_population031')
    _need(type(population) is dict and set(population) == {'native_episodes','attained_native_episodes',
        'progress_paid_units','completion_paid_units','completion_forfeited_units',
        'external_completion_forfeited_units','budget_violations'}, 'depth031 population fields differ')
    _need(all(type(x) in (int,float) and math.isfinite(x) and x >= 0 for x in population.values())
          and population['budget_violations'] == 0
          and population['attained_native_episodes'] <= population['native_episodes'], 'depth031 population invalid')
    prior = previous
    for step, record in enumerate(row['steps']):
        pre = record['pre_reset']; snap = pre['depth_credit031']
        del view['steps'][step]['pre_reset']['depth_credit031']
        _need(type(snap) is dict and snap['call_index'] == [update*24+step+1]*2,
              'depth031 receipt clock differs')
        for key, values in snap.items():
            _need(type(values) is list and len(values) == 2
                  and all(type(x) in (int,float,bool) and math.isfinite(x) for x in values),
                  'depth031 receipt requires finite two-world vectors')
        for w in range(2):
            at = lambda key: snap[key][w]
            _need(snap['native_reset'] == pre['native_reset'], 'depth031 native flags differ')
            _need(0 <= at('current_depth_m') <= at('peak_depth_m') <= .03200001,
                  'depth031 depth bounds differ')
            _need(at('attained') == (at('peak_depth_m') >= .030 - 1e-8), 'depth031 attainment differs')
            _need(0 <= at('progress_paid_total') <= 1.00002 and 0 <= at('completion_paid_total') <= 1,
                  'depth031 component budget exceeded')
            potential = min(1.,max(0.,(at('peak_depth_m')-.004)/.026))
            _need(math.isclose(at('progress_potential'),potential,abs_tol=2e-6), 'depth031 potential differs')
            horizon = pre['started'][w] and at('elapsed_contact_ticks') >= 800
            _need(at('horizon_complete') == horizon and (not horizon or pre['native_reset'][w]),
                  'depth031 genuine horizon differs')
            for field, component in (('progress_paid','native_depth_progress031'),('completion_paid','native_depth_completion031')):
                actual = pre['reward_components'].get(component,[0.,0.])[w]
                _need(math.isclose(actual,at(field),rel_tol=2e-5,abs_tol=2e-5), 'depth031 reward component join differs')
            previous_potential = at('previous_progress_potential')
            _need(0 <= previous_potential <= at('progress_potential') + 2e-6,
                  'depth031 prior potential differs')
            expected_progress = max(0.,at('progress_potential')-previous_potential) if marker['progress']['enabled'] else 0.
            _need(math.isclose(at('progress_paid'),expected_progress,abs_tol=2e-5),
                  'depth031 paid progress differs from prior potential')
            pending = at('attained') and marker['completion']['enabled'] and not at('completion_paid_before')
            _need(at('completion_paid') == float(pending and horizon), 'depth031 completion timing differs')
            _need(at('completion_due') == at('completion_paid')
                  and at('completion_paid_after') == (at('completion_paid_before') or at('completion_paid') > 0),
                  'depth031 completion before/after accounting differs')
            _need(at('completion_forfeited') == float(pending and pre['native_reset'][w] and not horizon),
                  'depth031 completion forfeiture differs')
            if prior is not None:
                old = prior['depth_credit031']; resets = at('resets_since_collect')
                _need(at('generation') == old['generation'][w] + resets, 'depth031 generation differs')
                _need(not prior['native_reset'][w] or resets >= 1, 'depth031 native reset missing')
                if not resets:
                    _need(at('physics_sample_count') == old['physics_sample_count'][w]+80,
                          'depth031 physics cadence differs')
                    delta = max(0.,at('progress_potential')-old['progress_potential'][w])
                    expected = delta if marker['progress']['enabled'] else 0.
                    _need(math.isclose(at('progress_paid'),expected,abs_tol=2e-5), 'depth031 progress conservation differs')
            if not marker['progress']['enabled']:
                _need(at('progress_paid') == 0., 'depth031 disabled progress paid')
        prior = pre
    split_previous = None if previous is None else {k:v for k,v in previous.items() if k != 'depth_credit031'}
    validate_trace_row(view, update, previous=split_previous)
    return deepcopy(row['steps'][-1]['pre_reset'])
