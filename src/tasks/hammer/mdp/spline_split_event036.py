"""Read-only critic context for the existing two-tranche accepted-strike ledger."""
import torch

from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS, elapsed_contact_ticks
from src.tasks.hammer.mdp.spline_split_credit import SplitSplineStrikeReward, validate_split_parameters


def split_event_observation(env, *, field):
    """Read seen, age, total balance and time to the next pending policy payout.

    ObservationManager's initial shape probe precedes RewardManager construction.
    Once the observation manager exists, a missing live owner is an error. This
    reader never computes rewards, consumes an onset, or modifies a balance.
    Runtime observations follow reward computation and any automatic subset reset;
    reset worlds therefore expose their cleared owner and first-contact context.
    """
    if field not in CREDIT_EVENT_FIELDS:
        raise ValueError('unknown split event observation field')
    if not hasattr(env, 'observation_manager'):
        return torch.zeros((env.num_envs, 1), device=env.device)
    owner = getattr(env, '_spline_split_credit', None)
    if type(owner) is not SplitSplineStrikeReward:
        raise RuntimeError('live split event observation requires its exact split credit owner')
    validate_split_parameters(owner.early_fraction, owner.early_deadline_ms, owner.reserve_deadline_ms)
    if env.step_dt != .02:
        raise ValueError('split observations require the admitted20ms policy period')
    elapsed = elapsed_contact_ticks(env)
    started = env._hammer_first_strike.started
    early, reserve = owner.early_credit, owner.reserve_credit
    for balance, fraction in ((early, owner.early_fraction), (reserve, 1.-owner.early_fraction)):
        if (balance.shape != (env.num_envs,) or not balance.is_floating_point()
                or not bool(torch.isfinite(balance).all())
                or not bool(((balance == 0) | (balance == fraction)).all())
                or bool(((balance > 0) & ~started).any())):
            raise ValueError('split observation has invalid unpaid tranche balance')
    full = round(.2/env.physics_dt)
    period = round(env.step_dt/env.physics_dt)
    count = env._hammer_first_strike._substep_count
    onset = count-elapsed
    early_due = ((onset+round(.05/env.physics_dt)+period-1)//period)*period
    reserve_due = ((onset+round(.2/env.physics_dt)+period-1)//period)*period
    due = torch.where(early > 0, early_due, reserve_due)
    balance = early+reserve
    until = (due-count).clamp_min(0).to(balance.dtype)/full
    values = (
        started.to(balance.dtype),
        (elapsed.to(balance.dtype)/full).clamp(0.,1.),
        balance,
        torch.where(balance > 0, until, 0.),
    )
    return values[CREDIT_EVENT_FIELDS.index(field)].unsqueeze(-1)
