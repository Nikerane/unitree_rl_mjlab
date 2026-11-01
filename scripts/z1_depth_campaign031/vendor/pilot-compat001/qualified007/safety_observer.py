"""Compact read-only measurements; frozen continuation and native load unchanged."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import torch
import continuation_tools as frozen

FROZEN_CONTINUATION_SHA='0b6484499258ee0e1b03feae81a533b0bfad35d0d2810dc8e78031a858f7c902'
if hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest()!=FROZEN_CONTINUATION_SHA:
    raise ValueError('frozen continuation helper differs')
NAMES=tuple(f'joint{i}' for i in range(1,7))


def canonical_mapping(env):
    """Resolve entity IDs separately from native actuator/qpos/dof addresses."""
    c=env._sampled_pd500;robot=env.scene['robot'];model=env.sim.mj_model
    entity_ids,names=robot.find_joints(NAMES,preserve_order=True)
    if tuple(names)!=NAMES or tuple(c.gains.joint_names)!=NAMES or tuple(c.joint_ids)!=tuple(entity_ids):
        raise ValueError('canonical entity/controller joint order differs')
    native=[int(model.joint('robot/'+name).id) for name in NAMES]
    controls=c.ids.detach().cpu().tolist()
    if len(set(native))!=6 or len(controls)!=6 or len(set(controls))!=6 or len(set(entity_ids))!=6:
        raise ValueError('six unique canonical joints and actuators required')
    if [int(model.actuator_trnid[i,0]) for i in controls]!=native or any(int(model.jnt_type[i])!=3 for i in native):
        raise ValueError('native actuator joint identity/type differs')
    qpos=torch.tensor([int(model.jnt_qposadr[i]) for i in native],device=env.device)
    dofs=torch.tensor([int(model.jnt_dofadr[i]) for i in native],device=env.device)
    if not torch.equal(c.qids[c.ids],qpos) or not torch.equal(c.vids[c.ids],dofs):
        raise ValueError('native qpos/dof mapping differs')
    limits=robot.data.joint_pos_limits[:,entity_ids]
    native_limits=torch.as_tensor(model.jnt_range[native].copy(),device=env.device,dtype=limits.dtype)
    if (limits.shape[-2:]!=(6,2) or not torch.isfinite(limits).all()
        or not bool((limits[...,0]<limits[...,1]).all()) or not torch.equal(limits,native_limits[None].expand_as(limits))):
        raise ValueError('canonical physical joint limits differ')
    return dict(joint_names=NAMES,entity_ids=entity_ids,native_joint_ids=native,
                actuator_ids=controls,qpos_ids=qpos,dof_ids=dofs,limits=limits)


class CanonicalContinuationAccumulator(frozen.ContinuationAccumulator):
    """Identical continuation statistics with corrected name-resolved read mapping."""
    def _sample_mapped(self,*,joint_position,limits,kp,kd,raw_load=None,held_load=None,**signals):
        return self.sample(**signals)

    @contextmanager
    def install(self,env):
        if self._installed or env.num_envs!=self.num_envs or env.physics_dt!=self.dt:raise ValueError('observer lifecycle/cadence differs')
        c=env._sampled_pd500;tracker=env._hammer_first_strike;m=canonical_mapping(env)
        load_ids=env._hammer_substep_impulse._joint_ids
        if isinstance(load_ids,slice):load_ids=list(range(len(env.scene['robot'].joint_names)))[load_ids]
        if list(load_ids)!=list(m['entity_ids']):raise ValueError('native load joint order differs')
        params=env.cfg.metrics['cat_soft'].params
        if env._hammer_substep_impulse._window!=200 or float(params['limit'])!=frozen.VELOCITY_LIMIT or tuple(params['imp_limit'])!=frozen.LOAD_CAPS:
            raise ValueError('native safety definition differs')
        self.mapping_record={k:(v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v) for k,v in m.items()}
        import inspect
        impulse=env._hammer_substep_impulse
        self.mapping_record['impulse_provenance']={
            'raw_load_field':'_hammer_substep_impulse._rolling',
            'raw_load_semantics':'instantaneous contact-masked rolling reaction-impulse window sum, before control-step hold; not raw qfrc',
            'held_load_field':'_hammer_substep_impulse._pulse',
            'held_load_semantics':'maximum rolling-window impulse latched since current control-step first substep',
            'enforced_load_field':'_hammer_substep_impulse.impulse',
            'enforced_relation':'maximum(raw_load,held_load)',
            'window_substeps':impulse._window,'baseline_subtracted':bool(impulse._subtract_baseline),
            'source_sha256':hashlib.sha256(Path(inspect.getfile(type(impulse))).read_bytes()).hexdigest(),
            'thresholds_nms':list(frozen.LOAD_CAPS)}
        j=int(env.sim.mj_model.joint('nail_block/nail_slide').id);nail=int(env.sim.mj_model.jnt_qposadr[j])
        original=env.metrics_manager.compute_substep
        def read():
            result=original();q=env.sim.data.qpos[:,m['qpos_ids']]
            self._sample_mapped(started=tracker.started,contact=(tracker._contact_sensor.data.found>0).any(-1),depth=env.sim.data.qpos[:,nail],
                force_world=tracker._impulse_sensor.data.force.sum(-2),axis=tracker._axis.expand(env.num_envs,3),
                joint_position_violation=(q<m['limits'][...,0]) | (q>m['limits'][...,1]),joint_position=q,limits=m['limits'],
                joint_velocity=env.sim.data.qvel[:,m['dof_ids']],load=env._hammer_substep_impulse.impulse,raw_load=env._hammer_substep_impulse._rolling,held_load=env._hammer_substep_impulse._pulse,
                saturated=c.saturated[:,c.ids],kp=c.applied_kp,kd=c.applied_kd)
            return result
        self._installed=True;env.metrics_manager.compute_substep=read
        try:yield self
        finally:env.metrics_manager.compute_substep=original;self._installed=False


class SafetyAccumulator(CanonicalContinuationAccumulator):
    """Original window stats plus mapped q/gains and exact literal-time depth."""
    def __init__(self,num_envs,device,physics_dt=.00025,*,native_window_s):
        if native_window_s not in (.05,.2):raise ValueError('declared native window required')
        super().__init__(num_envs,device,physics_dt)
        self.native_window_samples=round(native_window_s/physics_dt)
        self.native_window_s=native_window_s
        self.extra_whole=self._extra();self.extra_window=self._extra()
        self.post50_stats=self._stats()
        self.extra_post50=self._extra()
        self.extra_horizons={n:self._extra() for n in (200,800)}
        self.extra_native=self._extra()
        self.native_boundary_extra=self._extra()
        self.native_boundary_closed=self._zero(dtype=torch.bool)
        self.native_boundary_sample=self._zero(dtype=torch.long)
        self.instant_gains={name:[None]*num_envs for name in ('first_physics_sample','onset_sample_200','onset_sample_800')}
        self.literal={n:[None]*num_envs for n in (201,801)}
        self.raw_onset=self._zero(dtype=torch.long)
        self.last_depth=[None]*num_envs
        self.accepted_gains=[None]*num_envs
        self.mapping_record=None
        from onset_trace import OnsetTrace
        self.onset_trace=OnsetTrace(num_envs,dt=physics_dt)

    def _sample_mapped(self,**signals):
        return self.sample(**signals)

    def _extra(self):
        result={k:self._zero(6) for k in ('q_min_rad','q_max_rad','minimum_position_margin_rad','position_first_violation_sample',
                                         'kp_min','kp_max','kp_sum','kd_min','kd_max','kd_sum')}
        for k in ('q_min_rad','minimum_position_margin_rad','kp_min','kd_min'):result[k].fill_(float('inf'))
        for k in ('q_max_rad','kp_max','kd_max'):result[k].fill_(float('-inf'))
        for name in ('velocity','load','raw_load','held_load'):
            result[name+'_minimum_margin']=self._zero(6).fill_(float('inf'))
            result[name+'_first_crossing_sample']=self._zero(6)
            result[name+'_crossing_samples']=self._zero(6)
            result[name+'_peak']=self._zero(6)
        result['samples']=self._zero(dtype=torch.long)
        return result

    def _extra_update(self,s,mask,q,limits,kp,kd):
        m=mask[:,None];position_margin=torch.minimum(q-limits[...,0],limits[...,1]-q)
        s['samples']+=mask
        for name,margin in (('velocity',frozen.VELOCITY_LIMIT-self.current_velocity.abs()),('load',self.caps-self.current_load),('raw_load',self.caps-self.current_raw_load),('held_load',self.caps-self.current_held_load)):
            s[name+'_minimum_margin']=torch.where(m,torch.minimum(s[name+'_minimum_margin'],margin),s[name+'_minimum_margin'])
            s[name+'_crossing_samples']+=(m & margin.lt(0))
            magnitude=self.current_velocity.abs() if name=='velocity' else getattr(self,'current_'+name)
            s[name+'_peak']=torch.where(m,torch.maximum(s[name+'_peak'],magnitude),s[name+'_peak'])
            key=name+'_first_crossing_sample'
            first=m & margin.lt(0) & s[key].eq(0)
            s[key]=torch.where(first,self.observed[:,None],s[key])
        for name,value in (('q',q),('kp',kp),('kd',kd)):
            lo='q_min_rad' if name=='q' else name+'_min';hi='q_max_rad' if name=='q' else name+'_max'
            s[lo]=torch.where(m,torch.minimum(s[lo],value),s[lo]);s[hi]=torch.where(m,torch.maximum(s[hi],value),s[hi])
        s['minimum_position_margin_rad']=torch.where(m,torch.minimum(s['minimum_position_margin_rad'],position_margin),s['minimum_position_margin_rad'])
        first=m & (position_margin<0) & s['position_first_violation_sample'].eq(0)
        s['position_first_violation_sample']=torch.where(first,self.observed[:,None],s['position_first_violation_sample'])
        s['kp_sum']+=torch.where(m,kp,0);s['kd_sum']+=torch.where(m,kd,0)

    def sample(self,*,joint_position,limits,kp,kd,raw_load,held_load,**signals):
        expected=(self.num_envs,6)
        for value in (joint_position,kp,kd):
            if value.shape!=expected:raise ValueError('six canonical position/gain channels required')
            self.invalid |= self.active & ~torch.isfinite(value).all(-1)
        if limits.shape not in ((1,6,2),(self.num_envs,6,2)) or not torch.isfinite(limits).all():
            raise ValueError('finite canonical joint limits required')
        computed_bad=(joint_position<limits[...,0]) | (joint_position>limits[...,1])
        if not torch.equal(computed_bad,signals['joint_position_violation'].bool()):raise ValueError('position violation input differs from canonical limits')
        for value in (raw_load,held_load):
            if value.shape!=expected or not torch.isfinite(value).all() or bool((value<0).any()):raise ValueError('native raw/held impulse must be finite nonnegative six channels')
        if not torch.equal(signals['load'],torch.maximum(raw_load,held_load)):
            raise ValueError('enforced load differs from max(native rolling,native pulse)')
        self.current_raw_load=raw_load.double();self.current_held_load=held_load.double()
        self.current_velocity=signals['joint_velocity'].double()
        self.current_load=signals['load'].double()
        before=self.onset.clone();native_closed_before=self.native_closed.clone()
        super().sample(**signals)
        for i in (self.active & self.onset.gt(0)).nonzero().flatten().tolist():
            self.onset_trace.sample(i,int(self.count[i]),int(self.observed[i]),
                qvel=signals['joint_velocity'][i].detach().cpu().numpy(),
                rolling=raw_load[i].detach().cpu().numpy(),pulse=held_load[i].detach().cpu().numpy())
        axial=(signals['force_world'].double()*signals['axis'].double()).sum(-1)
        self._update(self.post50_stats,self.active & self.onset.gt(0) & self.count.gt(200) & self.count.le(800),
            signals['depth'].double().clamp(0.,.032),axial,signals['contact'].bool(),
            signals['joint_position_violation'].bool(),signals['joint_velocity'].double().abs(),signals['load'].double(),signals['saturated'].bool())
        self.raw_onset=torch.where(self.active & signals['contact'].bool() & self.raw_onset.eq(0),self.observed,self.raw_onset)
        q=joint_position.double();kp=kp.double();kd=kd.double()
        self._extra_update(self.extra_whole,self.active,q,limits,kp,kd)
        self._extra_update(self.extra_window,self.active & self.onset.gt(0) & self.count.le(800),q,limits,kp,kd)
        self._extra_update(self.extra_post50,self.active & self.onset.gt(0) & self.count.gt(200) & self.count.le(800),q,limits,kp,kd)
        for n in (200,800):self._copy(self.extra_horizons[n],self.extra_window,self.active & self.count.eq(n))
        for name,mask in (('first_physics_sample',self.observed.eq(1)),('onset_sample_200',self.count.eq(200)),('onset_sample_800',self.count.eq(800))):
            for i in (self.active & mask).nonzero().flatten().tolist():
                if self.instant_gains[name][i] is None:
                    self.instant_gains[name][i]=dict(sample=int(self.observed[i]),time_s=int(self.observed[i])*self.dt,kp=kp[i].tolist(),kd=kd[i].tolist())
        self._copy(self.extra_native,self.extra_window,self.active & self.count.eq(self.native_window_samples) & ~native_closed_before)
        for i in (self.active & self.onset.gt(0) & before.eq(0)).nonzero().flatten().tolist():
            self.accepted_gains[i]=dict(kp=kp[i].tolist(),kd=kd[i].tolist(),sample=int(self.observed[i]))
        for i in (self.active & self.onset.gt(0)).nonzero().flatten().tolist():
            self.last_depth[i]=dict(sample=int(self.observed[i]),time_s=int(self.observed[i])*self.dt,depth_m=max(0.,min(.032,float(signals['depth'][i]))))
        for n in (201,801):
            for i in (self.active & self.onset.gt(0) & self.count.eq(n)).nonzero().flatten().tolist():
                index=int(self.observed[i]);raw=float(signals['depth'][i]);t=index*self.dt
                self.literal[n][i]=dict(complete=True,sample=index,count_including_onset=n,target_time_s=(int(self.onset[i])+n-1)*self.dt,
                    lower_sample=index,upper_sample=index,lower_time_s=t,upper_time_s=t,
                    raw_qpos_m=raw,depth_m=max(0.,min(.032,raw)),lower_depth_m=max(0.,min(.032,raw)),upper_depth_m=max(0.,min(.032,raw)),
                    kp=kp[i].tolist(),kd=kd[i].tolist())

    def freeze_native_boundary(self,done_mask):
        """Latch observer-owned whole-attempt stats at the existing native event."""
        if done_mask.shape!=(self.num_envs,) or done_mask.dtype!=torch.bool:
            raise ValueError('native boundary mask differs')
        fresh=done_mask & ~self.native_boundary_closed
        for i in fresh.nonzero().flatten().tolist():
            self.onset_trace.mark_native(i,int(self.count[i]))
        self._copy(self.native_boundary_extra,self.extra_whole,fresh)
        self.native_boundary_sample=torch.where(fresh,self.observed,self.native_boundary_sample)
        self.native_boundary_closed |= fresh

    def freeze_native(self,done_mask):
        self._copy(self.extra_native,self.extra_window,done_mask.bool() & self.active & ~self.native_closed)
        super().freeze_native(done_mask)

    def _extra_row(self,s,i):
        n=int(s['samples'][i]);out={'samples':n}
        for k,v in s.items():
            if k in ('samples','kp_sum','kd_sum'):continue
            out[k]=v[i].tolist() if n else None
        first=out['position_first_violation_sample']
        out['position_first_violation_sample']=[int(x) or None for x in first] if first is not None else [None]*6
        out['position_first_violation_time_s']=[x*self.dt if x is not None else None for x in out['position_first_violation_sample']]
        for name in ('velocity','load','raw_load','held_load'):
            key=name+'_first_crossing_sample'
            out[key]=[int(x) or None for x in out[key]] if out[key] is not None else [None]*6
            out[name+'_crossing_duration_s']=[x*self.dt for x in out[name+'_crossing_samples']] if out[name+'_crossing_samples'] is not None else None
            out[name+'_first_crossing_time_from_reset_s']=[x*self.dt if x is not None else None for x in out[key]]
            out[name+'_first_crossing_time_from_contact_s']=[(x-int(self.onset[i]))*self.dt if x is not None and self.onset[i] else None for x in out[key]]
        out['margin_units']={'velocity':'rad/s','load':'Nms; enforced max(rolling,pulse)','raw_load':'Nms; native _rolling contact-masked baseline-subtracted window sum','held_load':'Nms; native _pulse control-step peak window sum'}
        for k in ('kp','kd'):out[k+'_mean']=(s[k+'_sum'][i]/n).tolist() if n else None
        return out

    def records(self):
        rows=super().records()
        for i,w in enumerate(rows):
            w['declared_native_window_s']=self.native_window_s
            w['contact_0_50ms']=w.pop('native_50ms')
            w.pop('native_50ms_semantics',None)
            w['contact_0_50ms']=self._row(self.snapshots[200] if self.complete[200][i] else self.window,i,self.complete[200][i])
            w['contact_0_200ms']=w['horizons']['800']
            w['contact_50_200ms']=self._row(self.post50_stats,i,self.complete[800][i])
            w['first_raw_contact_sample']=int(self.raw_onset[i]) or None
            w['position_and_gains']=dict(whole_attempt=self._extra_row(self.extra_whole,i),
                native_declared_prefix=self._extra_row(self.extra_native if self.native_closed[i] else self.extra_window,i),
                horizon_800=self._extra_row(self.extra_horizons[800] if self.complete[800][i] else self.extra_window,i),accepted_contact=self.accepted_gains[i])
            w['position_and_gains']['contact_50_200ms']=self._extra_row(self.extra_post50,i)
            w['position_and_gains']['native_policy_boundary']=self._extra_row(self.native_boundary_extra,i) if self.native_boundary_closed[i] else None
            w['position_and_gains']['native_policy_boundary_sample']=int(self.native_boundary_sample[i]) if self.native_boundary_closed[i] else None
            w['instant_applied_gains']={name:values[i] for name,values in self.instant_gains.items()}
            w['literal_depth']={} 
            for n,label in ((201,'50ms'),(801,'200ms')):
                target=(int(self.onset[i])+n-1)*self.dt if self.onset[i] else None
                w['literal_depth'][label]=self.literal[n][i] or dict(complete=False,sample=None,count_including_onset=n,target_time_s=target,
                    lower_sample=None,upper_sample=None,lower_time_s=None,upper_time_s=None,raw_qpos_m=None,depth_m=None,lower_depth_m=None,upper_depth_m=None,kp=None,kd=None)
                if self.literal[n][i] is None and self.last_depth[i] is not None:
                    last=self.last_depth[i]
                    if last['time_s']<target:
                        w['literal_depth'][label].update(lower_sample=last['sample'],lower_time_s=last['time_s'],lower_depth_m=last['depth_m'])
            w['depth_timestamp_semantics']='native onset-inclusive prefix200/800; literal201/801 relative to accepted sample timestamp; no interpolation or added steps'
        return rows

