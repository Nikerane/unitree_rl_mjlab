"""Opt-in accepted-credit deadlines and critic-only event context.

Payment timing changes the objective; no late safety gate or credit reversal is
introduced. Observations only read credit, never consume an eligibility event.
"""
import torch
from src.tasks.hammer.mdp.spline_curriculum import AcceptedSplineStrikeReward

DEADLINES_MS = (50, 100, 150, 200)
CREDIT_EVENT_FIELDS = ('credit_contact_seen', 'credit_contact_age',
                       'credit_unpaid', 'credit_time_to_payout')


def validate_credit_deadline(value):
    if type(value) is not int or value not in DEADLINES_MS:
        raise ValueError('credit deadline must be integer50,100,150 or200ms')
    return value


def elapsed_contact_ticks(env):
    """Recover integer onset tick from the tracker's existing float timestamp.

    The counter advances after event finalization and is reset per world; it is
    independent of randomized episode-length bookkeeping. Same admitted clock
    domain as the original horizon predicate, including its float64 recovery.
    """
    dt = env.physics_dt
    if dt not in (.002, .00025) or not 0. < env.cfg.episode_length_s <= 20.:
        raise ValueError('credit clock requires admitted physics dt and episode duration')
    tracker = env._hammer_first_strike
    onset = tracker.first_contact_time_s
    count = tracker._substep_count
    started = tracker.started
    shape = (env.num_envs,)
    if (onset.shape != shape or count.shape != shape or started.shape != shape
            or started.dtype != torch.bool or count.dtype not in (torch.int32, torch.int64)
            or not torch.isfinite(onset).all() or (onset < 0).any() or (count < 0).any()):
        raise ValueError('credit clock has invalid per-world state')
    ticks = count - torch.round(onset.to(torch.float64) / dt).to(torch.int64)
    if (started & (ticks < 0)).any():
        raise ValueError('credit clock precedes contact onset')
    return torch.where(started, ticks, torch.zeros_like(ticks))


class DeadlineSplineStrikeReward(AcceptedSplineStrikeReward):
    """Pay one accepted-event unit at deadline or earlier native reset, once."""
    def __init__(self, cfg, env):
        deadline = validate_credit_deadline(cfg.params['credit_deadline_ms'])
        if hasattr(env, '_spline_deadline_credit'):
            raise ValueError('only one deadline credit owner is allowed per environment')
        super().__init__(cfg, env)
        self.credit_deadline_ms = deadline
        self.unpaid_credit = torch.zeros(self.num_envs, device=self.device)
        env._spline_deadline_credit = self

    def reset(self, env_ids):
        super().reset(env_ids)
        self.unpaid_credit[slice(None) if env_ids is None else env_ids] = 0.

    def __call__(self, env, robot_cfg, credit_deadline_ms, width_m=.01):
        del robot_cfg
        if validate_credit_deadline(credit_deadline_ms) != self.credit_deadline_ms:
            raise ValueError('credit deadline differs from constructed treatment')
        elapsed = elapsed_contact_ticks(env)
        if env.reset_buf.dtype != torch.bool or env.reset_buf.shape != (env.num_envs,):
            raise ValueError('credit requires boolean per-world native reset flags')
        self.unpaid_credit += self._accepted_onset(env, width_m).to(self.unpaid_credit.dtype)
        ticks = round(self.credit_deadline_ms / 1000. / env.physics_dt)
        pay = env.reset_buf | (env._hammer_first_strike.started & (elapsed >= ticks))
        paid = torch.where(pay, self.unpaid_credit, torch.zeros_like(self.unpaid_credit))
        self.unpaid_credit.masked_fill_(pay, 0.)
        return paid / env.step_dt


def credit_event_observation(env, *, field, credit_deadline_ms):
    """Four raw bounded critic scalars, read after reward computation at runtime."""
    deadline = validate_credit_deadline(credit_deadline_ms)
    if field not in CREDIT_EVENT_FIELDS:
        raise ValueError('unknown credit event observation field')
    # ObservationManager probes shapes before assigning itself or RewardManager.
    # After that boundary, missing live state is a contract error, never zeros.
    if not hasattr(env, 'observation_manager'):
        return torch.zeros((env.num_envs, 1), device=env.device)
    credit = getattr(env, '_spline_deadline_credit', None)
    if type(credit) is not DeadlineSplineStrikeReward:
        raise RuntimeError('live credit observation requires its deadline credit owner')
    if credit.credit_deadline_ms != deadline:
        raise ValueError('observation credit deadline differs from reward owner')
    elapsed = elapsed_contact_ticks(env)
    started = env._hammer_first_strike.started
    balance = credit.unpaid_credit
    if (balance.shape != (env.num_envs,) or not torch.isfinite(balance).all()
            or not ((balance == 0) | (balance == 1)).all()
            or ((balance > 0) & ~started).any()):
        raise ValueError('credit observation has invalid unpaid event balance')
    full = round(.2 / env.physics_dt)
    if env.step_dt != .02:
        raise ValueError('credit observations require the admitted20ms policy period')
    period = round(env.step_dt / env.physics_dt)
    count = env._hammer_first_strike._substep_count
    onset = count - elapsed
    due = ((onset + round(deadline / 1000. / env.physics_dt) + period - 1) // period) * period
    until = (due - count).clamp_min(0)
    values = (started.to(balance.dtype),
              (elapsed.to(balance.dtype) / full).clamp(0., 1.),
              balance.clone(),
              torch.where(balance > 0, until.to(balance.dtype) / full, 0.))
    return values[CREDIT_EVENT_FIELDS.index(field)].unsqueeze(-1)
