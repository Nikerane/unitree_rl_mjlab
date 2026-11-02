"""Bounded native-producer onset trace; no simulator dependencies."""
import hashlib
from pathlib import Path
import numpy as np


class OnsetTrace:
    def __init__(self,worlds,capacity=881,dt=.00025):
        if worlds<=0 or worlds>32 or capacity!=881 or dt!=.00025:
            raise ValueError('trace requires <=32 worlds,881 slots and frozen dt')
        self.worlds=worlds;self.capacity=capacity;self.dt=dt;self.channels={}
        self.counts=np.zeros(worlds,dtype=np.int64)
        self.sample_index=np.zeros((worlds,capacity),dtype=np.int64)
        self.native_boundary=np.zeros((worlds,capacity),dtype=np.bool_)

    def sample(self,world,count,sample_index,*,qvel,rolling,pulse):
        if not 0<=world<self.worlds or count!=self.counts[world]+1 or count>self.capacity:
            raise ValueError('dense onset trace overflow,duplicate or gap')
        for name,value in (('qvel',qvel),('rolling',rolling),('pulse',pulse)):
            value=np.asarray(value)
            if value.shape!=(6,) or value.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(value).all():
                raise ValueError('native trace requires finite six-joint floating producer values')
            if name not in self.channels:self.channels[name]=np.zeros((self.worlds,self.capacity,6),dtype=value.dtype)
            if self.channels[name].dtype!=value.dtype:raise ValueError('native producer dtype changed')
            self.channels[name][world,count-1]=value
        self.sample_index[world,count-1]=sample_index
        self.counts[world]=count

    def mark_native(self,world,count):
        if count<=0:return  # Native misses/timeouts have summaries, no onset trace.
        if count>self.counts[world]:raise ValueError('native boundary not captured in dense trace')
        self.native_boundary[world,count-1]=True

    def save(self,path):
        path=Path(path)
        payload={**self.channels,'sample_index':self.sample_index,'native_boundary':self.native_boundary,
            'valid_counts':self.counts,'relative_time_s':np.arange(self.capacity,dtype=np.float64)*self.dt}
        with path.open('xb') as stream:np.savez_compressed(stream,**payload)
        return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'arrays':{name:{'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in payload.items()},
            'valid_counts':self.counts.tolist(),'capacity_per_world':self.capacity,
            'semantics':'onset sample included; literal200ms at slot800; native boundary marked; no precontact dense trace; padding invalid beyond valid_counts'}

    def safety_summaries(self,*,velocity_limit,load_caps):
        """Postprocess captured arrays only; authoritative inclusive endpoint801.

        No padding becomes evidence. Thresholds supplied by the pinned native
        observer contract, not fitted from observations.
        """
        caps=np.asarray(load_caps,dtype=np.float64)
        if caps.shape!=(6,) or not np.isfinite(caps).all() or not (caps>0).all() or not np.isfinite(velocity_limit) or not velocity_limit>0:
            raise ValueError('native safety thresholds required')
        windows=(('legacy_0_50ms_half_open',0,200,'[0,50ms)'),
            ('inclusive_0_50ms',0,201,'[0,50ms]'),
            ('legacy_0_200ms_half_open',0,800,'[0,200ms)'),
            ('inclusive_50_200ms',200,801,'[50ms,200ms]'),
            ('authoritative_0_200ms_inclusive',0,801,'[0,200ms]'))
        rows=[]
        for world in range(self.worlds):
            result={'world_id':world,'valid_count':int(self.counts[world]),
                'endpoint_convention':'onset-inclusive count1 at0s,count200 at49.75ms,count201 at50ms,count800 at199.75ms,count801 at200ms',
                'native_boundary_slots_zero_based':np.flatnonzero(self.native_boundary[world]).tolist(),'windows':{}}
            for name,start,stop,interval in windows:
                record={'interval':interval,'first_count_including_onset':start+1,'last_count_including_onset':stop,
                    'required_samples':stop-start,'authoritative':name=='authoritative_0_200ms_inclusive'}
                if self.counts[world]<stop or set(self.channels)!= {'qvel','rolling','pulse'}:
                    record.update(status='UNKNOWN_MISSING_ENDPOINT',channels=None)
                else:
                    record.update(status='COMPLETE',channels={})
                    for channel,values in (('velocity',np.abs(self.channels['qvel'][world,start:stop].astype(np.float64))),
                            ('raw_rolling_impulse',self.channels['rolling'][world,start:stop].astype(np.float64)),
                            ('held_impulse',self.channels['pulse'][world,start:stop].astype(np.float64))):
                        limit=velocity_limit if channel=='velocity' else caps
                        bad=values>limit
                        first=[int(np.flatnonzero(bad[:,j])[0])+start if bad[:,j].any() else None for j in range(6)]
                        record['channels'][channel]={'peak':values.max(axis=0).tolist(),
                            'minimum_margin':(limit-values).min(axis=0).tolist(),
                            'exceedance_samples':bad.sum(axis=0).tolist(),
                            'sample_count_times_dt_s':(bad.sum(axis=0)*self.dt).tolist(),
                            'first_count_including_onset':[i+1 if i is not None else None for i in first],
                            'first_time_from_contact_s':[i*self.dt if i is not None else None for i in first],
                            'first_absolute_sample_index':[int(self.sample_index[world,i]) if i is not None else None for i in first]}
                result['windows'][name]=record
            rows.append(result)
        return {'schema':'z1_onset_safety_endpoints_v1','velocity_limit_rad_s':float(velocity_limit),
            'load_caps_nms':caps.tolist(),'duration_semantics':'exceedance sample count times dt; inclusive endpoint samples do not define extra physical duration',
            'worlds':rows}
