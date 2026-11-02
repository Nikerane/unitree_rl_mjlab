"""Evaluation-only, first-attempt continuation. No learning or source mutations.

mjlab 1.4 order: substep metrics; termination; reward; metrics (CaT); autoreset.
The two compute contexts below expose the same native boundary to reward and CaT,
then restore the physical continuation state before mjlab considers autoreset.
"""
from contextlib import contextmanager
from copy import deepcopy

import torch


LOAD_CAPS = (.369, .246, .738, .369, .246, .0164)
VELOCITY_LIMIT = 3.1415
HORIZONS = (200, 400, 600, 800)


class ContinuationAccumulator:
    """Compact post-integration reader; exact windows include the onset sample."""

    def __init__(self, num_envs, device, physics_dt=.00025):
        if physics_dt != .00025 or num_envs <= 0:
            raise ValueError('requires positive worlds and frozen 0.00025 s physics')
        self.num_envs, self.device, self.dt = num_envs, device, physics_dt
        self.native_window_samples = 200
        self.active = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.observed = self._zero(dtype=torch.long)
        self.onset = self._zero(dtype=torch.long)
        self.count = self._zero(dtype=torch.long)
        self.overshoot = self._zero(dtype=torch.long)
        self.caps = torch.tensor(LOAD_CAPS, dtype=torch.float64, device=device)
        self.whole = self._stats()
        self.window = self._stats()
        self.snapshots = {n: self._stats() for n in HORIZONS}
        self.whole_snapshots = {n: self._stats() for n in HORIZONS}
        self.complete = {n: self._zero(dtype=torch.bool) for n in HORIZONS}
        self.native_stats, self.native_whole = self._stats(), self._stats()
        self.native_closed = self._zero(dtype=torch.bool)
        self.native_complete = self._zero(dtype=torch.bool)
        self.invalid = self._zero(dtype=torch.bool)
        self.end_reasons = [None] * num_envs
        self._installed = False

    def _zero(self, *shape, dtype=torch.float64):
        return torch.zeros(self.num_envs, *shape, dtype=dtype, device=self.device)

    def _stats(self):
        result = {k: self._zero() for k in ('samples', 'current_depth_m', 'peak_depth_m',
                  'positive_axial_impulse_ns', 'signed_axial_impulse_ns', 'contact_samples',
                  'first_separation_sample', 'first_recontact_sample')}
        for name in ('joint_velocity_peak_rad_s', 'joint_load_peak_nms',
                     'joint_position_violation_samples', 'joint_velocity_exceedance_samples',
                     'joint_load_exceedance_samples', 'saturation_samples'):
            result[name] = self._zero(6)
        return result

    def sample(self, *, started, contact, depth, force_world, axis,
               joint_position_violation, joint_velocity, load, saturated):
        """Read already computed native signals once; never update their sources."""
        n = self.num_envs
        expected = ((started,(n,)),(contact,(n,)),(depth,(n,)),(force_world,(n,3)),
                    (axis,(n,3)),(joint_position_violation,(n,6)),
                    (joint_velocity,(n,6)),(load,(n,6)),(saturated,(n,6)))
        for value, shape in expected:
            if value.shape != shape:
                raise ValueError('continuation inputs must have declared shapes')
            self.invalid |= self.active & ~torch.isfinite(value).reshape(n,-1).all(-1)
        self.invalid |= self.active & (load < 0).any(-1)
        self.invalid |= self.active & (torch.linalg.vector_norm(axis.double(),dim=-1)-1).abs().gt(1e-6)
        # Avoid a CPU/GPU synchronization at every 4 kHz measurement callback.
        # GPU numerical faults fail at the next policy boundary before autoreset.
        if self.invalid.device.type == 'cpu':
            self.check_valid()
        started, contact = started.bool(), contact.bool()
        self.observed += self.active.long()
        new = self.active & started & self.onset.eq(0)
        self.onset = torch.where(new, self.observed, self.onset)
        accepted = self.active & self.onset.gt(0)
        exact = accepted & self.count.lt(800)
        self.overshoot += (accepted & ~exact).long()
        self.count += accepted.long()
        axial = (force_world.double() * axis.double()).sum(-1)
        inputs = (depth.double().clamp(0.,.032), axial, contact,
                  joint_position_violation.bool(), joint_velocity.double().abs(),
                  load.double(), saturated.bool())
        self._update(self.whole, self.active, *inputs)
        self._update(self.window, exact, *inputs)
        for horizon in HORIZONS:
            close = exact & self.count.eq(horizon)
            self._copy(self.snapshots[horizon],self.window,close)
            self._copy(self.whole_snapshots[horizon],self.whole,close)
            self.complete[horizon] |= close
        prefix = exact & self.count.eq(self.native_window_samples) & ~self.native_closed
        self._copy(self.native_stats,self.window,prefix)
        self._copy(self.native_whole,self.whole,prefix)
        self.native_complete |= prefix
        self.native_closed |= prefix

    @staticmethod
    def _copy(target, source, worlds):
        for key,value in source.items():
            mask=worlds.reshape((-1,)+(1,)*(value.ndim-1))
            target[key].copy_(torch.where(mask,value,target[key]))

    def check_valid(self):
        if self.invalid.any():
            raise ValueError('continuation input must be finite; load nonnegative and axis unit length')

    def freeze_native(self, done_mask):
        """Keep an early native timeout/failure censored even if observation continues."""
        close=done_mask.bool() & self.active & ~self.native_closed
        self._copy(self.native_stats,self.window,close)
        self._copy(self.native_whole,self.whole,close)
        self.native_closed |= close

    def _update(self, s, active, depth, axial, contact, pos_bad, velocity, load, saturated):
        m = active[:,None]
        s['samples'] += active
        s['current_depth_m'].copy_(torch.where(active,depth,s['current_depth_m']))
        s['peak_depth_m'].copy_(torch.where(active,torch.maximum(depth,s['peak_depth_m']),s['peak_depth_m']))
        gated_axial=torch.where(contact & active,axial,torch.zeros_like(axial))
        s['positive_axial_impulse_ns'] += gated_axial.clamp_min(0)*self.dt
        s['signed_axial_impulse_ns'] += gated_axial*self.dt
        s['contact_samples'] += contact & active
        separate = active & ~contact & s['contact_samples'].gt(0) & s['first_separation_sample'].eq(0)
        s['first_separation_sample'].copy_(torch.where(separate,s['samples'],s['first_separation_sample']))
        recontact = active & contact & s['first_separation_sample'].gt(0) & s['first_recontact_sample'].eq(0)
        s['first_recontact_sample'].copy_(torch.where(recontact,s['samples'],s['first_recontact_sample']))
        s['joint_velocity_peak_rad_s'].copy_(torch.where(m,torch.maximum(velocity,s['joint_velocity_peak_rad_s']),s['joint_velocity_peak_rad_s']))
        s['joint_load_peak_nms'].copy_(torch.where(m,torch.maximum(load,s['joint_load_peak_nms']),s['joint_load_peak_nms']))
        s['joint_position_violation_samples'] += pos_bad & m
        s['joint_velocity_exceedance_samples'] += velocity.gt(VELOCITY_LIMIT) & m
        s['joint_load_exceedance_samples'] += load.gt(self.caps) & m
        s['saturation_samples'] += saturated & m

    def finish_step(self, done_mask, reasons=None):
        self.check_valid()
        newly_done = done_mask.bool() & self.active
        if reasons is not None:
            for i in newly_done.nonzero().flatten().tolist():
                self.end_reasons[i] = reasons[i]
        self.active &= ~done_mask.bool()

    def _row(self, stats, index, complete):
        row = {k: v[index].detach().cpu().tolist() for k,v in stats.items()}
        row['samples'] = int(row['samples'])
        row['complete'] = bool(complete)
        row['depth_30mm_attained'] = row['peak_depth_m'] >= .030
        row['contact_occupancy'] = row['contact_samples']/row['samples'] if row['samples'] else None
        for name in ('joint_position_violation','joint_velocity_exceedance','joint_load_exceedance','saturation'):
            row[name+'_samples'] = [int(x) for x in row[name+'_samples']]
            row[name+'_duration_s'] = [x*self.dt for x in row[name+'_samples']]
        for name in ('first_separation_sample','first_recontact_sample'):
            row[name] = int(row[name]) if row[name] else None
        return row

    def records(self):
        self.check_valid()
        rows=[]
        for i in range(self.num_envs):
            horizons={str(n):self._row(self.snapshots[n] if self.complete[n][i] else self.window,i,self.complete[n][i]) for n in HORIZONS}
            for n in HORIZONS:
                horizons[str(n)]['whole_attempt']=self._row(self.whole_snapshots[n] if self.complete[n][i] else self.whole,i,self.complete[n][i])
            native=self._row(self.native_stats if self.native_closed[i] else self.window,i,self.native_complete[i])
            native['whole_attempt']=self._row(self.native_whole if self.native_closed[i] else self.whole,i,self.native_complete[i])
            rows.append(dict(world_id=i,accepted_contact_sample=int(self.onset[i]) or None,
                             accepted_contact_time_s=int(self.onset[i])*self.dt if self.onset[i] else None,
                             observed_samples=int(self.observed[i]),overshoot_samples=int(self.overshoot[i]),
                             active=bool(self.active[i]),end_reason=self.end_reasons[i],
                             native_50ms=native,horizons=horizons,
                             extended=deepcopy(horizons['800']),
                             whole_attempt=self._row(self.whole,i,not self.active[i]),
                             native_50ms_semantics='exact onset-inclusive first 200 samples or earlier native censor; whole_attempt ends at that exact horizon',
                             horizon_whole_attempt_semantics='reset through exact horizon; original policy-boundary safety is separately latched by protocol',
                             depth_semantics='clamped current depth at horizon, not settled depth'))
        return rows

    @contextmanager
    def install(self, env):
        """Install before reset; sample after original authoritative substep metrics."""
        if self._installed or env.num_envs != self.num_envs or env.physics_dt != self.dt:
            raise ValueError('continuation reader already installed or environment mismatch')
        controller = env._sampled_pd500
        tracker = env._hammer_first_strike
        if env._hammer_substep_impulse._window != 200:
            raise ValueError('rolling load window must remain 200 samples')
        caps = env.cfg.metrics['cat_soft'].params
        if float(caps['limit']) != VELOCITY_LIMIT or tuple(caps['imp_limit']) != LOAD_CAPS:
            raise ValueError('frozen velocity/load caps changed')
        joint_id=env.sim.mj_model.joint('nail_block/nail_slide').id
        nail_qpos_id=int(env.sim.mj_model.jnt_qposadr[joint_id])
        limits=env.scene['robot'].data.joint_pos_limits[:,controller.ids]
        original=env.metrics_manager.compute_substep
        def sampled():
            result=original()
            ids=controller.ids
            q=env.sim.data.qpos[:,controller.qids][:,ids]
            self.sample(started=tracker.started,contact=(tracker._contact_sensor.data.found>0).any(-1),
                        depth=env.sim.data.qpos[:,nail_qpos_id],
                        force_world=tracker._impulse_sensor.data.force.sum(-2),axis=tracker._axis.expand(env.num_envs,3),
                        joint_position_violation=(q<limits[...,0]) | (q>limits[...,1]),
                        joint_velocity=env.sim.data.qvel[:,controller.vids][:,ids],
                        load=env._hammer_substep_impulse.impulse,saturated=controller.saturated[:,ids])
            return result
        self._installed=True
        env.metrics_manager.compute_substep=sampled
        try:
            yield self
        finally:
            env.metrics_manager.compute_substep=original
            self._installed=False


class NativeBoundaryContinuation:
    """Present one native event to reward/CaT while extending physical contact.

    Install around rollout only, after initial reset. All original functions run
    once. Manager term classes/params, load history and actor state are untouched.
    """
    def __init__(self, env, horizon_s=.2, post_contact_cap_s=.22, *, accumulator=None,
                 native_snapshot=None, native_window_s=.05):
        if horizon_s != .2 or post_contact_cap_s != .22 or env.step_dt != .02:
            raise ValueError('requires frozen 200 ms horizon and <=20 ms policy overshoot')
        if env.physics_dt != .00025 or env._hammer_substep_impulse._window != 200:
            raise ValueError('requires 0.25 ms physics and 200-sample rolling load window')
        if native_window_s not in (.05, .2):
            raise ValueError('native window must be explicitly .05 or .2')
        native_cfg = env.termination_manager.get_term_cfg('nail_driven')
        if native_cfg.params.get('window_s', .05) != native_window_s:
            raise ValueError('declared native window differs from original training config')
        self.native_window_s = native_window_s
        self.env=env
        self._owns_reader=accumulator is None
        self.accumulator=accumulator or ContinuationAccumulator(env.num_envs,env.device)
        if self.accumulator.num_envs != env.num_envs:
            raise ValueError('accumulator world count mismatch')
        self.accumulator.native_window_samples = round(native_window_s / env.physics_dt)
        self.snapshot=native_snapshot
        self.manager=env.termination_manager
        self.timeouts={n for n,c in zip(self.manager._term_names,self.manager._term_cfgs) if c.time_out}
        if 'nail_driven' not in self.manager._term_names or self.timeouts != {'time_out'}:
            raise ValueError('continuation requires native nail_driven and unique time_out terms')
        n=env.num_envs
        self.latched=torch.zeros(n,dtype=torch.bool,device=env.device)
        self.native_return=torch.zeros(n,dtype=torch.float64,device=env.device)
        self.diagnostic_return=torch.zeros_like(self.native_return)
        self.native_delta=torch.zeros(n,device=env.device)
        self.native_records=[None]*n
        self._pending=None
        self._reward_done=None
        self._physical_done=None
        self._reward_computed=False
        self._entered=False

    def reward_done(self, physical_done):
        """Consume the reward-time mask captured before metrics advances lifecycle."""
        if not self._entered or self._reward_done is None:
            raise RuntimeError('no unconsumed reward-time terminal evidence for this step')
        if (not isinstance(physical_done, torch.Tensor)
                or physical_done.shape != (self.env.num_envs,)
                or physical_done.dtype not in (torch.bool, torch.int32, torch.int64)
                or not bool(((physical_done == 0) | (physical_done == 1)).all())
                or not torch.equal(physical_done.bool(), self._physical_done)):
            raise ValueError('physical completion differs from captured termination evidence')
        result=self._reward_done.clone()
        self._reward_done=None
        return result

    def _state(self):
        return {name:value.clone() for name,value in self.manager._term_dones.items()}

    def _write(self, state):
        m=self.manager
        m._terminated_buf.zero_();m._truncated_buf.zero_()
        for name,value in state.items():
            m._term_dones[name].copy_(value)
            (m._truncated_buf if name in self.timeouts else m._terminated_buf).logical_or_(value)

    @contextmanager
    def _native_view(self):
        native,physical,_=self._pending
        use_native=self.accumulator.active & ~self.latched
        exposed={n:torch.where(use_native,v,physical[n]) for n,v in native.items()}
        saved=(self.env.reset_buf,self.env.reset_terminated,self.env.reset_time_outs)
        self._write(exposed)
        self.env.reset_buf=self.manager.dones
        self.env.reset_terminated=self.manager.terminated
        self.env.reset_time_outs=self.manager.time_outs
        try:
            yield
        finally:
            self._write(physical)
            self.env.reset_buf,self.env.reset_terminated,self.env.reset_time_outs=saved

    def __enter__(self):
        if self._entered:
            raise RuntimeError('continuation context cannot be entered twice')
        self._entered=True
        env=self.env
        self._reader_context=self.accumulator.install(env) if self._owns_reader else None
        if self._reader_context is not None:
            self._reader_context.__enter__()
        self.original=(self.manager.compute,env.reward_manager.compute,env.metrics_manager.compute)
        def terminate():
            self.accumulator.check_valid()
            self.original[0]()
            native=self._state()
            physical={n:v.clone() for n,v in native.items()}
            accepted=self.accumulator.onset.gt(0) & self.accumulator.active
            physical['nail_driven'] &= ~accepted
            physical['time_out'] &= ~accepted
            # Literal contact+200ms is onset-inclusive sample801. Native200
            # may end later at its next normal policy boundary. No reset frame
            # is ever sampled: finish_step closes worlds before autoreset.
            native_event = torch.zeros_like(accepted)
            for value in native.values():
                native_event |= value
            cap=accepted & self.accumulator.count.ge(801) & (self.latched | native_event)
            if (accepted & self.accumulator.count.ge(881)).any():
                raise RuntimeError('continuation exceeded its contact-aligned physical cap')
            physical['time_out'] |= cap
            event=self.manager.dones & self.accumulator.active & ~self.latched
            self._pending=(native,physical,event)
            self._write(physical)
            self._physical_done=self.manager.dones.clone()
            self._reward_done=None
            self._reward_computed=False
            return self.manager.dones
        def reward(*args,**kwargs):
            if self._pending is None or self._reward_computed:
                raise RuntimeError('reward requires one fresh native termination compute')
            self._reward_computed=True
            with self._native_view():
                reward_done=env.reset_buf.clone()
                value=self.original[1](*args,**kwargs)
                if not torch.isfinite(value).all():
                    raise ValueError('nonfinite continuation reward')
                self._reward_done=reward_done
                self.native_return += value.double()*self.accumulator.active*~self.latched
                self.diagnostic_return += value.double()*self.accumulator.active*self.latched
                return value
        def metrics(*args,**kwargs):
            with self._native_view():
                value=self.original[2](*args,**kwargs)
                from src.tasks.hammer.cat.keys import CAT_DELTA_KEY
                delta=env.extras[CAT_DELTA_KEY]
                if not torch.isfinite(delta).all():
                    raise ValueError('nonfinite continuation CaT delta')
                native,physical,event=self._pending
                snapshots=self.snapshot(env) if self.snapshot and event.any() else None
                for i in event.nonzero().flatten().tolist():
                    self.native_records[i]=dict(latched=True,termination_terms={n:bool(v[i]) for n,v in native.items()},
                         reward_total=float(self.native_return[i]),boundary_reward=float(env.reward_buf[i]),
                         boundary_cat_delta=float(delta[i]),observed_samples=int(self.accumulator.observed[i]),
                         contact_samples=int(self.accumulator.count[i]),
                         boundary_measurement=self.accumulator._row(self.accumulator.whole,i,True),
                         boundary_measurement_semantics='reset through original native policy termination; declared native window plus normal policy-step overshoot',
                         snapshot=deepcopy(snapshots[i]) if snapshots else None)
                self.latched |= event
                self.accumulator.freeze_native(event)
                if hasattr(self.accumulator, 'freeze_native_boundary'):
                    self.accumulator.freeze_native_boundary(event)
            reasons=None
            if (self.manager.dones & self.accumulator.active).any():
                terms={n:v.detach().cpu().tolist() for n,v in physical.items()}
                caps=(self.accumulator.onset.gt(0)&self.accumulator.count.ge(800)).detach().cpu().tolist()
                reasons=[','.join(n for n,v in terms.items() if v[i]) or None for i in range(env.num_envs)]
                for i,cap in enumerate(caps):
                    if cap:
                        reasons[i]='continuation_horizon' + (','+reasons[i] if reasons[i] else '')
            self.accumulator.finish_step(self.manager.dones,reasons)
            return value
        self.manager.compute=terminate
        env.reward_manager.compute=reward
        env.metrics_manager.compute=metrics
        return self

    def __exit__(self,*exc):
        self.manager.compute,self.env.reward_manager.compute,self.env.metrics_manager.compute=self.original
        if self._reader_context is not None:
            self._reader_context.__exit__(*exc)
        self._entered=False

    def records(self):
        return [dict(world_id=i,native=deepcopy(self.native_records[i]) if self.native_records[i] else
                     dict(latched=False,reward_total=float(self.native_return[i])),
                     post_native_diagnostic=dict(reward_total=float(self.diagnostic_return[i]),
                                                 label='excluded from native first-attempt return'))
                for i in range(self.env.num_envs)]
