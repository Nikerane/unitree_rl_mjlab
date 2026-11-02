"""Opt-in all-world policy-step envelopes; never exact 200 ms evaluation.

No forwards, RNG draws, reward/constraint calls or storage writes. Per-world
state consists only of episode flags; only aggregate tensors reach JSON.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path

import torch

_LIMIT = 16 * 1024**2
_MISSING = object()
_ATTACHED = set()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


class _Population:
    def __init__(self, runner, path, max_updates, max_bytes):
        _need(type(max_updates) is int and 1 <= max_updates <= 500, 'update cap must be1..500')
        _need(type(max_bytes) is int and 0 < max_bytes <= _LIMIT, 'byte cap must be1..16MiB')
        self.env, self.alg = runner.env.unwrapped, runner.alg
        e = self.env
        _need(type(e.num_envs) is int and 1 <= e.num_envs <= 1024, 'world count must be1..1024')
        _need(self.alg.storage.num_transitions_per_env == 24, 'population requires24-step rollouts')
        _need(type(e.common_step_counter) is int and e.common_step_counter == 0, 'fresh population counter0 required')
        _need(id(e) not in _ATTACHED, 'population observer already attached')
        self.hook = e.metrics_manager.cfg['cat_soft'].func
        _need(self.hook._use_vel is True and self.hook._vel_detection == 'substep', 'live substep velocity required')
        names = getattr(getattr(self.hook, '_robot_cfg', None), 'joint_names', None)
        self.joint_names = tuple(f'joint{i}' for i in range(1, 7))
        _need(isinstance(names, (tuple, list)) and tuple(names) == self.joint_names, 'population joint identity/order differs')
        self.limit = self.hook._limit
        _need(type(self.limit) in (int, float) and math.isfinite(self.limit) and self.limit > 0, 'finite positive velocity threshold required')
        self.n, self.device = e.num_envs, e.device
        self.dt = e.physics_dt
        _need(self.dt in (.00025, .002), 'admitted physics timestep required')
        self.path = Path(path)
        _need(self.path.is_absolute() and self.path.parent.is_dir(), 'existing absolute output parent required')
        self.cap, self.max_updates = max_bytes, max_updates
        self.stream = None; self.written = self.updates = self.steps = 0
        self.counter = e.common_step_counter
        self.contact = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.pending_auto = torch.zeros_like(self.contact)
        self.touched = torch.zeros_like(self.contact)
        self.productive_seen = torch.zeros_like(self.contact)
        self.counts = dict(all_episode_starts=self.n, observed_episode_starts=0, observed_contact_onsets=0,
            native_contact_episodes=0, native_no_contact_episodes=0,
            native_completed_productive_episodes=0, observed_productive_contact_episodes=0,
            external_reset_observed_productive_episodes=0, native_timeout_episodes=0,
            contact_resets_before_200ms=0, contact_policy_envelopes_reached_200ms=0,
            external_reset_episodes=0, external_contact_censored_episodes=0)
        self._fresh_stats()
        self.saved = [(e.metrics_manager, 'compute', e.metrics_manager.compute),
                      (self.alg, 'compute_returns', self.alg.compute_returns),
                      (e.metrics_manager, 'reset', e.metrics_manager.reset)]
        self.previous = [o.__dict__.get(name, _MISSING) for o, name, _ in self.saved]
        _need(not bool(e._hammer_first_strike.started.any()), 'initial contact history would be left-censored')

    def _fresh_stats(self):
        z = lambda: torch.zeros(6, device=self.device)
        self.winners = torch.zeros(4, dtype=torch.long, device=self.device)
        self.stats = dict(peak_max=z(), peak_sum=z(), margin_max=torch.full((6,), -torch.inf, device=self.device),
            exceed=torch.zeros(6,dtype=torch.long,device=self.device), probability_sum=z(), probability_max=z(),
            ema_min=torch.full((6,),torch.inf,device=self.device), ema_max=z())

    def _tensor(self, value, shape, *, boolean=False):
        _need(isinstance(value, torch.Tensor) and tuple(value.shape) == shape, 'population tensor shape differs')
        _need(value.device == torch.device(self.device), 'population tensor device differs')
        _need(value.dtype == torch.bool if boolean else value.is_floating_point(), 'population tensor dtype differs')
        _need(bool(torch.isfinite(value).all()), 'population tensor must be finite')
        return value.detach()

    def start(self):
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(fd, 'wb', buffering=0)
        _ATTACHED.add(id(self.env))
        for (obj, name, _), fn in zip(self.saved, (self.metrics, self.returns, self.reset)):
            setattr(obj, name, fn)

    def metrics(self, *args, **kwargs):
        _need(self.updates < self.max_updates, 'population update cap exceeded')
        _need(self.steps < 24 and self.alg.storage.step == self.steps, 'population collection/storage step mismatch')
        result = self.saved[0][2](*args, **kwargs)
        e, cat = self.env, self.hook._cat
        _need(type(e.common_step_counter) is int and e.common_step_counter == self.counter + 1,
              'population transition clock skipped or repeated')
        _need(not bool(self.pending_auto.any()), 'automatic reset was not observed at reset seam')
        margin = self._tensor(cat.raw_constraints['joint_velocity_excess'], (self.n, 6))
        prob = self._tensor(cat.probs['joint_velocity_excess'], (self.n, 6))
        imp = self._tensor(cat.probs['joint_impulse_excess'], (self.n, 6))
        _need(bool(((imp >= 0) & (imp <= 1)).all()), 'impulse probability domain differs')
        dv, di = prob.amax(1), imp.amax(1)
        active = torch.maximum(dv, di) > 0
        self.winners += torch.stack(((~active).sum(), (active & (dv > di)).sum(),
            (active & (di > dv)).sum(), (active & (dv == di)).sum()))
        ema = self._tensor(cat.running_maxes['joint_velocity_excess'], (1, 6)).squeeze(0)
        _need(bool(((prob >= 0) & (prob <= 1)).all()) and bool((ema > 0).all()), 'velocity probability/EMA domain differs')
        _need(self.hook._limit == self.limit, 'velocity threshold changed during observation')
        peak = margin + self.limit
        _need(bool((peak >= -1e-6).all()), 'negative reconstructed velocity peak')
        s = self.stats
        s['peak_max'] = torch.maximum(s['peak_max'], peak.amax(0))
        s['peak_sum'] += peak.sum(0); s['margin_max'] = torch.maximum(s['margin_max'], margin.amax(0))
        s['exceed'] += (margin > 0).sum(0)
        s['probability_sum'] += prob.sum(0); s['probability_max'] = torch.maximum(s['probability_max'],prob.amax(0))
        s['ema_min'] = torch.minimum(s['ema_min'],ema); s['ema_max'] = torch.maximum(s['ema_max'],ema)
        self.counts['observed_episode_starts'] += int((~self.touched).sum())
        self.touched.fill_(True)
        tr = e._hammer_first_strike
        started = self._tensor(tr.started, (self.n,), boolean=True)
        _need(not bool((self.contact & ~started).any()), 'unobserved reset cleared native contact history')
        self.counts['observed_contact_onsets'] += int((started & ~self.contact).sum())
        self.contact |= started
        self._observe_productivity()
        reset = self._tensor(e.reset_buf,(self.n,),boolean=True)
        self.pending_auto.copy_(reset)
        self.counter = e.common_step_counter; self.steps += 1
        return result

    def _observe_productivity(self, ids=None):
        productive = self._tensor(self.env._hammer_first_strike.productive, (self.n,), boolean=True)
        known = productive & self.contact & self.touched
        if ids is not None:
            selected = torch.zeros_like(known); selected[ids] = True
            known = known & selected
        self.counts['observed_productive_contact_episodes'] += int((known & ~self.productive_seen).sum())
        self.productive_seen |= known

    def _ids(self, ids):
        if ids is None: ids = slice(None)
        if isinstance(ids, slice):
            selected = torch.arange(self.n,device=self.device)[ids]
        else:
            _need(isinstance(ids,torch.Tensor) and ids.ndim == 1 and ids.dtype in (torch.int32,torch.int64), 'reset IDs must be integer vector or slice')
            selected = ids.to(self.device)
        _need(bool(((selected >= 0) & (selected < self.n)).all()) and selected.unique().numel() == selected.numel(), 'reset IDs duplicated or out of range')
        return selected

    def reset(self, env_ids=None, *args, **kwargs):
        ids = self._ids(env_ids)
        native = self.pending_auto[ids]
        _need(not bool(native.any()) or bool(native.all()), 'mixed unmatched external/automatic reset IDs')
        n = ids.numel(); contact = self.contact[ids]
        self._observe_productivity(ids)
        if bool(native.all()) and n:
            _need(n == int(self.pending_auto.sum()), 'automatic reset subset differs from native reset flags')
            tr = self.env._hammer_first_strike
            age = self._tensor(tr.elapsed_since_contact_s,(self.n,))[ids]
            reached = torch.round(age.to(torch.float64) / self.dt) >= round(.2 / self.dt)
            timeout = self._tensor(self.env.reset_time_outs,(self.n,),boolean=True)[ids]
            self.counts['native_contact_episodes'] += int(contact.sum())
            self.counts['native_no_contact_episodes'] += int((~contact).sum())
            self.counts['native_completed_productive_episodes'] += int((contact & self.productive_seen[ids]).sum())
            self.counts['native_timeout_episodes'] += int(timeout.sum())
            self.counts['contact_resets_before_200ms'] += int((contact & ~reached).sum())
            self.counts['contact_policy_envelopes_reached_200ms'] += int((contact & reached).sum())
        else:
            self.counts['external_reset_episodes'] += n
            self.counts['external_contact_censored_episodes'] += int(contact.sum())
            self.counts['external_reset_observed_productive_episodes'] += int(self.productive_seen[ids].sum())
        result = self.saved[2][2](env_ids, *args, **kwargs)
        self.counts['all_episode_starts'] += n
        self.contact[ids] = False; self.pending_auto[ids] = False; self.touched[ids] = False
        self.productive_seen[ids] = False
        return result

    def write(self, row):
        body = (json.dumps(row,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
        _need(len(body) <= 65536 and self.written + len(body) <= self.cap, 'population byte cap exceeded')
        offset = 0
        while offset < len(body):
            count = self.stream.write(body[offset:])
            _need(type(count) is int and count > 0, 'population write made no progress')
            offset += count; self.written += count

    def returns(self, *args, **kwargs):
        _need(self.steps == 24 and self.alg.storage.step == 24, 'population requires full24-step rollout')
        _need(not bool(self.pending_auto.any()), 'population native reset not joined')
        result = self.saved[1][2](*args, **kwargs)
        cpu = lambda tensor: tensor.detach().cpu().tolist()
        s = self.stats; denom = self.n * 24
        velocity = dict(threshold_rad_s=self.limit, joint_names=list(self.joint_names),
            peak_rad_s_max_per_joint=cpu(s['peak_max']),peak_rad_s_mean_per_joint=cpu(s['peak_sum']/denom),
            raw_margin_max_per_joint=cpu(s['margin_max']),exceedance_policy_steps_per_joint=cpu(s['exceed']),
            exceedance_policy_step_fraction_per_joint=cpu(s['exceed']/denom),
            probability_mean_per_joint=cpu(s['probability_sum']/denom),probability_max_per_joint=cpu(s['probability_max']),
            ema_min_per_joint=cpu(s['ema_min']),ema_max_per_joint=cpu(s['ema_max']))
        self.write(dict(schema='z1_velocity_population_v1',kind='update',update=self.updates,
            scope='ALL_WORLDS_POLICY_STEP_ENVELOPE',worlds=self.n,world_transitions=denom,
            common_step_counter=self.counter,velocity=velocity,episodes_cumulative=dict(self.counts),
            pressure_winner_policy_steps=dict(zip(('none','velocity','impulse','tie'),cpu(self.winners))),
            envelope_caveat='ONSET_STEP_MAY_INCLUDE_PRECONTACT_FINAL_STEP_MAY_EXCEED_200MS',
            exact_200ms_contact_window_safety=None))
        self.updates += 1; self.steps = 0; self._fresh_stats()
        return result

    def finish(self, status):
        self.write(dict(schema='z1_velocity_population_v1',kind='final',status=status,
            scope='ALL_WORLDS_POLICY_STEP_ENVELOPE',updates=self.updates,partial_policy_steps=self.steps,
            active_at_stop_right_censored=int(self.touched.sum()),active_contact_at_stop=int(self.contact.sum()),
            unobserved_episode_starts_at_stop=int((~self.touched).sum()),
            active_observed_productive_at_stop=int((self.productive_seen & self.touched).sum()),
            exact_200ms_contact_window_safety=None,**self.counts))

    def close(self):
        for (obj,name,_), previous in zip(self.saved,self.previous):
            if previous is _MISSING: obj.__dict__.pop(name,None)
            else: setattr(obj,name,previous)
        _ATTACHED.discard(id(self.env))
        if self.stream is not None:
            try: os.fsync(self.stream.fileno())
            finally: self.stream.close(); self.stream = None


@contextmanager
def attach_velocity_population_trace(runner, output_path, *, max_updates=100, max_bytes=_LIMIT):
    """Observe fresh training through metrics compute/reset and existing returns.

    Never replace controller-owned env reset methods. Metrics reset is the
    shared automatic/external seam before tracker buffers are cleared.
    """
    trace = _Population(runner,output_path,max_updates,max_bytes)
    try:
        trace.start()
        try:
            yield trace
            _need(trace.steps == 0, 'population context ended with partial rollout')
        except BaseException:
            # Preserve already-written evidence without obscuring the original error.
            try: trace.finish('FAILED_PARTIAL_EVIDENCE')
            except BaseException: pass
            raise
        else:
            trace.finish('COMPLETE_OBSERVATION_NOT_SAFETY_CERTIFICATION')
    finally:
        trace.close()
