"""Opt-in conserved two-tranche credit; original first-contact eligibility."""
import torch
from src.tasks.hammer.mdp.spline_curriculum import AcceptedSplineStrikeReward
from src.tasks.hammer.mdp.spline_credit_deadline import elapsed_contact_ticks


def validate_split_parameters(early_fraction, early_deadline_ms, reserve_deadline_ms):
    if (type(early_fraction) is not float or early_fraction not in (.5,1.)
            or type(early_deadline_ms) is not int or early_deadline_ms != 50
            or type(reserve_deadline_ms) is not int or reserve_deadline_ms != 200):
        raise ValueError('split requires alpha .5 or1. and exact integer50/200ms deadlines')


class SplitSplineStrikeReward(AcceptedSplineStrikeReward):
    """Pay alpha at50ms, reserve at200ms; native reset flushes both once."""
    def __init__(self,cfg,env):
        p=cfg.params
        validate_split_parameters(p['early_fraction'],p['early_deadline_ms'],p['reserve_deadline_ms'])
        if hasattr(env,'_spline_split_credit'):
            raise ValueError('only one split credit owner is allowed')
        super().__init__(cfg,env)
        self.early_fraction=p['early_fraction']
        self.early_deadline_ms=p['early_deadline_ms'];self.reserve_deadline_ms=p['reserve_deadline_ms']
        self.early_credit=torch.zeros(self.num_envs,device=self.device)
        self.reserve_credit=torch.zeros_like(self.early_credit)
        self.generation=torch.zeros(self.num_envs,dtype=torch.int64,device=self.device)
        self._discarded_early=torch.zeros_like(self.early_credit)
        self._discarded_reserve=torch.zeros_like(self.early_credit)
        self._resets=torch.zeros_like(self.generation)
        self.last_snapshot=None
        self.call_index=0
        env._spline_split_credit=self

    def reset(self,env_ids):
        ids=slice(None) if env_ids is None else env_ids
        self._discarded_early[ids]+=self.early_credit[ids]
        self._discarded_reserve[ids]+=self.reserve_credit[ids]
        self.early_credit[ids]=0.;self.reserve_credit[ids]=0.
        self.generation[ids]+=1;self._resets[ids]+=1
        super().reset(env_ids)

    def __call__(self,env,robot_cfg,early_fraction,early_deadline_ms,reserve_deadline_ms,width_m=.01):
        del robot_cfg
        validate_split_parameters(early_fraction,early_deadline_ms,reserve_deadline_ms)
        if (early_fraction!=self.early_fraction or early_deadline_ms!=self.early_deadline_ms
                or reserve_deadline_ms!=self.reserve_deadline_ms or width_m!=.01 or env.step_dt!=.02):
            raise ValueError('split parameters/width/policy clock differ from treatment')
        elapsed=elapsed_contact_ticks(env)
        if env.reset_buf.dtype!=torch.bool or env.reset_buf.shape!=(env.num_envs,):
            raise ValueError('split requires boolean per-world native reset flags')
        snapshot=dict(early_before=self.early_credit.clone(),reserve_before=self.reserve_credit.clone(),
            consumed_before=self._consumed.clone(),elapsed_ticks=elapsed.clone(),
            started=env._hammer_first_strike.started.clone(),native_reset=env.reset_buf.clone(),
            generation=self.generation.clone(),resets_since_call=self._resets.clone(),
            externally_discarded_early=self._discarded_early.clone(),
            externally_discarded_reserve=self._discarded_reserve.clone())
        accepted=self._accepted_onset(env,width_m).to(self.early_credit.dtype)
        self.early_credit+=accepted*early_fraction
        self.reserve_credit+=accepted*(1.-early_fraction)
        snapshot['new_accepted']=accepted;snapshot['consumed_after']=self._consumed.clone()
        for bucket,deadline in (('early',early_deadline_ms),('reserve',reserve_deadline_ms)):
            balance=getattr(self,bucket+'_credit')
            due=env.reset_buf | (env._hammer_first_strike.started & (elapsed>=round(deadline/1000/env.physics_dt)))
            snapshot[bucket+'_paid']=torch.where(due,balance,torch.zeros_like(balance))
            balance.masked_fill_(due,0.)
            snapshot[bucket+'_after']=balance.clone()
        self.call_index+=1
        snapshot["call_index"]=torch.full_like(self.generation,self.call_index)
        self.last_snapshot=snapshot
        self._discarded_early.zero_();self._discarded_reserve.zero_();self._resets.zero_()
        return (snapshot['early_paid']+snapshot['reserve_paid'])/env.step_dt
