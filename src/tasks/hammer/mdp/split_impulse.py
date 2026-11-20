"""Conserve one productive first-J50 budget across scheduled policy boundaries."""
import torch

from src.tasks.hammer.mdp.rewards import FirstStrikeDeliveredRewardTerm
from src.tasks.hammer.mdp.spline_curriculum import AcceptedSplineStrikeReward
from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS, elapsed_contact_ticks

ACCOUNT_FIELDS = ('earned','early_paid','reserve_paid','discarded','unpaid')


class SplitImpulseZeroBonus(AcceptedSplineStrikeReward):
    """Retain the historical acceptance reader without any binary credit."""
    def __call__(self, env, robot_cfg, width_m=.01):
        del robot_cfg
        return self._accepted_onset(env,width_m).float()*0.


class SplitImpulseDeliveredImpulse(FirstStrikeDeliveredRewardTerm):
    """Freeze J50 once; pay alpha at50ms, its complement at200ms.

    Output is a rate because RewardManager multiplies by policy dt. A reset
    discards only future unpaid credit; a due tranche still pays on that step.
    """
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        alpha = cfg.params['early_fraction']
        if type(alpha) not in (int,float) or alpha not in (1.,.5,.25):
            raise ValueError('split impulse fraction must be1,.5 or.25')
        if hasattr(env,'_split_impulse'):
            raise ValueError('only one split impulse ledger is allowed')
        self.early_fraction = float(alpha)
        for name in ('earned','early_paid','reserve_paid','discarded','unpaid_credit','post_cat_paid'):
            setattr(self,name,torch.zeros(self.num_envs,device=self.device))
        self._early_done = torch.zeros_like(self._paid)
        self._reserve_done = torch.zeros_like(self._paid)
        self.step = {name:torch.zeros(self.num_envs,device=self.device) for name in
                     (*ACCOUNT_FIELDS,'raw_impulse_payout','post_cat_impulse_payout')}
        env._split_impulse = self

    def reset(self,env_ids):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for name in ('earned','early_paid','reserve_paid','discarded','unpaid_credit','post_cat_paid',
                     '_early_done','_reserve_done'):
            getattr(self,name)[ids] = 0

    def _consume(self,env):
        tracker = env._hammer_first_strike
        freeze = tracker.finalized & (elapsed_contact_ticks(env) >= 200) & ~self._paid
        self._paid |= freeze
        return tracker,freeze & tracker.productive

    def __call__(self,env,i_ref=1.,saturate=False,early_fraction=1.,**params):
        if (saturate is not False or early_fraction != self.early_fraction
                or env.step_dt != .02 or env.physics_dt != .00025):
            raise ValueError('split impulse requires exact linear budget, fraction and clocks')
        for value in self.step.values():value.zero_()
        earned = super().__call__(env,i_ref=i_ref,saturate=False,**params)
        self.earned += earned
        self.unpaid_credit += earned
        elapsed = elapsed_contact_ticks(env)
        early_due = self._paid & ~self._early_done & (elapsed >= 200)
        reserve_due = self._paid & ~self._reserve_done & (elapsed >= 800)
        early = torch.where(early_due,self.earned*self.early_fraction,0.)
        reserve = torch.where(reserve_due,self.unpaid_credit-early,0.)
        self._early_done |= early_due
        self._reserve_done |= reserve_due
        self.early_paid += early
        self.reserve_paid += reserve
        self.unpaid_credit -= early+reserve
        discarded = torch.where(env.reset_buf,self.unpaid_credit,0.)
        self.discarded += discarded
        self.unpaid_credit -= discarded
        # Also consume future tranches on a premature reset even before J50.
        self._paid |= env.reset_buf
        self._early_done |= env.reset_buf
        self._reserve_done |= env.reset_buf
        for name,value in (('earned',earned),('early_paid',early),('reserve_paid',reserve),
                           ('discarded',discarded),('unpaid',self.unpaid_credit),
                           ('raw_impulse_payout',early+reserve)):
            self.step[name].copy_(value)
        return (early+reserve)/env.step_dt

    def records(self):
        return [dict(schema='split_impulse_accounting_v1',units='J50_divided_by_I_ref',
            earned=float(self.earned[i]),early_paid=float(self.early_paid[i]),
            reserve_paid=float(self.reserve_paid[i]),paid=float(self.early_paid[i]+self.reserve_paid[i]),
            discarded=float(self.discarded[i]),unpaid=float(self.unpaid_credit[i]),
            raw_impulse_payout=float(self.early_paid[i]+self.reserve_paid[i]),
            post_cat_impulse_payout=float(self.post_cat_paid[i])) for i in range(self.num_envs)]


def accounting_metric(env):
    """Read the one native CaT delta after its hook; never transform rewards."""
    from src.tasks.hammer.cat.keys import CAT_DELTA_KEY
    owner = env._split_impulse
    post = owner.step['raw_impulse_payout']*(1.-env.extras[CAT_DELTA_KEY])
    owner.step['post_cat_impulse_payout'].copy_(post)
    owner.post_cat_paid += post
    return owner.unpaid_credit


def event_observation(env,*,field):
    """Shared critic ledger: seen, age/200ms, frozen unpaid, next due/200ms."""
    if field not in CREDIT_EVENT_FIELDS:raise ValueError('unknown split impulse critic field')
    if not hasattr(env,'observation_manager'):
        return torch.zeros((env.num_envs,1),device=env.device)
    owner = getattr(env,'_split_impulse',None)
    if type(owner) is not SplitImpulseDeliveredImpulse:
        raise RuntimeError('split impulse observations require their exact ledger')
    elapsed = elapsed_contact_ticks(env)
    tracker = env._hammer_first_strike
    onset = tracker._substep_count-elapsed
    pending = owner.unpaid_credit > 0
    target = torch.where(owner._early_done,800,200)
    period = round(env.step_dt/env.physics_dt)
    due = ((onset+target+period-1)//period)*period
    until = (due-tracker._substep_count).clamp_min(0).float()/800
    values = (tracker.started.float(),(elapsed.float()/800).clamp(0.,1.),
              owner.unpaid_credit.clone(),torch.where(pending,until,0.))
    return values[CREDIT_EVENT_FIELDS.index(field)].unsqueeze(-1)
