"""Bounded two-world observations of existing collection and CaT return computation.

No policy calls, random draws, reward edits or storage writes are introduced.
The final-value forward hook observes the call compute_returns already makes.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path

import torch

from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY

_LIMIT = 96 * 1024**2
_MISSING = object()


def _need(value, message):
    if not value:
        raise ValueError(message)


class _SplitCreditTrace:
    def __init__(self, runner, path, world_ids, max_bytes, max_updates):
        _need(type(world_ids) is tuple and world_ids == (0, 1)
              and all(type(x) is int for x in world_ids), 'trace scope is exactly worlds0,1')
        _need(type(max_bytes) is int and 0 < max_bytes <= _LIMIT, 'split trace byte cap exceeds96MiB or is invalid')
        _need(type(max_updates) is int and max_updates == 500, 'split trace requires exactly500 updates')
        self.env, self.alg = runner.env.unwrapped, runner.alg
        _need(type(self.env.num_envs) is int and self.env.num_envs >= 2, 'trace requires at least two worlds')
        _need(self.alg.storage.num_transitions_per_env == 24, 'trace requires24-step rollouts')
        from src.tasks.hammer.mdp.spline_split_credit import SplitSplineStrikeReward
        from scripts.z1_timing_noise020.split_contract import credit_timing
        owner = getattr(self.env, '_spline_split_credit', None)
        _need(type(owner) is SplitSplineStrikeReward and owner.early_fraction == .5,
              'training trace requires actual alpha.5 split owner')
        _need(owner.call_index == 0, 'split trace requires fresh reward call counter')
        self.last_reward_call = 0
        self.split_marker = credit_timing(owner.early_fraction)
        self.path = Path(path)
        _need(self.path.is_absolute(), 'trace output must be absolute')
        self.ids, self.cap, self.max_updates = world_ids, max_bytes, max_updates
        self.buffer, self.updates, self.written = [], 0, 0
        self.stream = None
        self.original_metrics = self.env.metrics_manager.compute
        self.original_returns = self.alg.compute_returns
        self.previous_metrics = self.env.metrics_manager.__dict__.get('compute', _MISSING)
        self.previous_returns = self.alg.__dict__.get('compute_returns', _MISSING)
        self.attached = False

    def start(self):
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(fd, 'wb', buffering=0)
        try:
            self.env.metrics_manager.compute = self.metrics
            self.alg.compute_returns = self.returns
            self.attached = True
        except BaseException:
            self.close()
            raise

    def take(self, value, *, vector=False):
        _need(isinstance(value, torch.Tensor) and value.ndim > 0
              and value.shape[0] == self.env.num_envs and bool(torch.isfinite(value).all()),
              'trace requires a finite world-aligned tensor')
        if vector:
            _need(value.numel() == self.env.num_envs, 'trace scalar field has unexpected shape')
            value = value.reshape(self.env.num_envs)
        return value[list(self.ids)].detach().clone().cpu().tolist()

    def constraints(self):
        # MetricsManager exposes live term instances through cfg, not get_term_cfg.
        cfg = getattr(self.env.metrics_manager, 'cfg', None)
        if not isinstance(cfg, dict):
            return None, 'UNKNOWN_NO_HOOK_CFG'
        try:
            hook = cfg['cat_soft'].func
        except KeyError:
            return None, 'UNKNOWN_NO_CAT_HOOK'
        read = getattr(hook, 'constraint_telemetry', None)
        if not callable(read):
            return None, 'UNKNOWN_NO_TELEMETRY_METHOD'
        snapshot = read()
        _need(isinstance(snapshot, dict) and snapshot, 'trace constraint snapshot is invalid')
        result = {}
        for name, value in snapshot.items():
            if name == 'active_limit_per_joint':
                _need(isinstance(value, torch.Tensor) and value.ndim == 1
                      and bool(torch.isfinite(value).all()), 'trace joint limit vector is invalid')
                result[name] = value.detach().clone().cpu().tolist()
            else:
                result[name] = self.take(value)
        return result, 'OBSERVED_LIVE_PRE_RESET'

    def reward_components(self):
        manager = getattr(self.env, 'reward_manager', None)
        names = getattr(manager, 'active_terms', None)
        ledger = getattr(manager, '_step_reward', None)
        dt = getattr(self.env, 'step_dt', None)
        _need(isinstance(names, (list, tuple)) and 0 < len(names) <= 64
              and all(type(name) is str and name for name in names)
              and len(set(names)) == len(names), 'trace requires bounded unique active reward terms')
        _need(getattr(manager, '_scale_by_dt', None) is True
              and type(dt) in (int, float) and math.isfinite(dt) and dt > 0,
              'trace reward ledger requires actual finite positive dt-scaled rates')
        _need(isinstance(ledger, torch.Tensor) and torch.is_floating_point(ledger)
              and ledger.shape == (self.env.num_envs, len(names))
              and bool(torch.isfinite(ledger).all()), 'trace reward ledger is missing or invalid')
        # RewardManager stores weighted rates; multiply by the same native dt.
        return {name: self.take(ledger[:, index] * dt, vector=True)
                for index, name in enumerate(names)}

    def metrics(self, *args, **kwargs):
        try:
            result = self.original_metrics(*args, **kwargs)
            _need(self.updates < self.max_updates and len(self.buffer) < 24,
                  'trace update or step cap exceeded')
            _need(self.alg.storage.step == len(self.buffer), 'trace collection/storage step mismatch')
            env, tr = self.env, self.alg.transition
            constraint, status = self.constraints()
            components = self.reward_components()
            owner = env._spline_split_credit
            _need(isinstance(owner.last_snapshot, dict) and owner.call_index == self.last_reward_call + 1,
                  'split trace requires exactly one new reward call snapshot')
            self.last_reward_call = owner.call_index
            split = {key: self.take(value, vector=True) for key, value in owner.last_snapshot.items()}
            native = env.termination_manager
            _need(isinstance(native.active_terms, (list, tuple)) and 0 < len(native.active_terms) <= 64,
                  'split trace requires native termination causes')
            causes = {name: dict(time_out=native.get_term_cfg(name).time_out,
                                flags=self.take(native.get_term(name), vector=True))
                      for name in native.active_terms}
            params = getattr(tr, 'distribution_params', None)
            mean = std = None
            if params is not None:
                _need(isinstance(params, (tuple, list)) and len(params) >= 2,
                      'trace distribution snapshot is malformed')
                mean, std = self.take(params[0]), self.take(params[1])
            self.buffer.append(dict(step=len(self.buffer), pre_reset=dict(
                raw_reward=self.take(env.reward_buf, vector=True),
                reward_components=components,
                accepted_credit_paid=components.get('accepted_spline_strike'),
                r_neg_raw_minus_positive=self.take(env.reward_buf.reshape(-1) - env.extras[CAT_R_POS_KEY].reshape(-1), vector=True),
                r_neg_semantics='RAW_MINUS_POSITIVE_NOT_UNROUNDED_PRODUCER_LEDGER',
                r_pos=self.take(env.extras[CAT_R_POS_KEY], vector=True),
                delta=self.take(env.extras[CAT_DELTA_KEY], vector=True),
                native_reset=self.take(env.reset_buf, vector=True),
                timeout=self.take(env.reset_time_outs, vector=True),
                started=self.take(env._hammer_first_strike.started, vector=True),
                contact_age_s=self.take(env._hammer_first_strike.elapsed_since_contact_s, vector=True),
                unpaid_credit=self.take(owner.early_credit + owner.reserve_credit, vector=True),
                split_credit=split, native_causes=causes,
                terminated=self.take(env.reset_terminated, vector=True),
                value_at_action=self.take(tr.values, vector=True),
                latent_actions=self.take(tr.actions),
                executed_actions=self.take(env.action_manager.action),
                distribution_mean=mean, distribution_std=std,
                constraints=constraint, constraint_status=status)))
            return result
        except BaseException:
            self.close()
            raise

    def _customize_envelope(self, envelope):
        """Extend an observation envelope before the shared bounded write."""
        return envelope

    def _after_trace_write(self):
        """Advance extension state only after the complete row was written."""
        pass

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
            envelope = dict(schema='z1_split_credit_training_trace_v1', update=self.updates,
                split_credit=self.split_marker, max_updates=500, physics_dt_s=self.env.physics_dt, policy_dt_s=self.env.step_dt,
                world_ids=list(self.ids), scope='FIXED_TWO_WORLDS_NOT_POPULATION_STATISTICS',
                gamma=self.alg.gamma, lam=self.alg.lam, steps=self.buffer,
                final_values=self.take(captured[0], vector=True),
                executed_action_semantics='ACTION_MANAGER_INPUT_AFTER_WRAPPER_CLIP_NOT_MOTOR_TORQUE',
                storage_reward_semantics='ACTUAL_ALREADY_CAT_SCALED_AND_TIMEOUT_BOOTSTRAPPED',
                raw_advantage_semantics='RETURNS_MINUS_VALUES_BEFORE_ADVANTAGE_NORMALIZATION')
            envelope = self._customize_envelope(envelope)
            body = (json.dumps(envelope, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
            _need(self.written + len(body) <= self.cap, 'trace byte cap would be exceeded')
            offset = 0
            while offset < len(body):
                count = self.stream.write(body[offset:])
                _need(type(count) is int and count > 0, 'trace write made no progress')
                offset += count
                self.written += count
            self._after_trace_write()
            self.buffer = []
            self.updates += 1
            return result
        except BaseException:
            if handle is not None:
                handle.remove()
            self.close()
            raise

    def close(self):
        # Restore original instance/class lookup semantics, not just bound methods.
        for obj, name, previous in ((self.env.metrics_manager, 'compute', self.previous_metrics),
                                    (self.alg, 'compute_returns', self.previous_returns)):
            if previous is _MISSING:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, previous)
        self.attached = False
        if self.stream is not None:
            stream, self.stream = self.stream, None
            try:
                os.fsync(stream.fileno())
            finally:
                stream.close()
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)


@contextmanager
def attach_credit_training_trace(runner, output_path, world_ids=(0, 1),
                                 max_bytes=_LIMIT, max_updates=500):
    """Attach once to a fresh runner; preserve partial evidence on any exception."""
    trace = _SplitCreditTrace(runner, output_path, world_ids, max_bytes, max_updates)
    try:
        trace.start()
        yield trace
        _need(not trace.buffer, 'trace context ended with an unjoined partial rollout')
    finally:
        trace.close()

# Explicit new-schema collector, not a legacy owner/schema alias.
attach_split_credit_training_trace = attach_credit_training_trace
