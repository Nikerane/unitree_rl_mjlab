"""Strict opt-in identity for the eight-cell deadline/event-critic screen."""
import json
from src.tasks.hammer.mdp.spline_credit_deadline import (
    DeadlineSplineStrikeReward, CREDIT_EVENT_FIELDS, credit_event_observation,
    validate_credit_deadline,
)
from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete


def experiment_identity(deadline, mode):
    validate_credit_deadline(deadline)
    if type(mode) is not str or mode not in ('current', 'event'):
        raise ValueError('credit critic mode must be current or event')
    return dict(schema_version='spline_credit_critic_v1', credit_deadline_ms=deadline,
                episode_contact_window_ms=200, critic_mode=mode, actor_width=43,
                critic_width=47 if mode == 'event' else 43,
                critic_event_fields=list(CREDIT_EVENT_FIELDS) if mode == 'event' else [],
                actor_inputs_unchanged=True)


def validate_experiment_identity(value):
    if not isinstance(value, dict):
        raise ValueError('credit critic identity must be a dictionary')
    expected = experiment_identity(value.get('credit_deadline_ms'), value.get('critic_mode'))
    if json.dumps(value,sort_keys=True,allow_nan=False) != json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('credit critic identity differs from exact declared schema')
    return expected


def credit_critic_metadata(env):
    """Return None for legacy tasks; require live owner, timer and exact new terms."""
    if 'accepted_spline_strike' not in env.reward_manager.active_terms:
        return None
    cfg = env.reward_manager.get_term_cfg('accepted_spline_strike')
    if not isinstance(cfg.func, DeadlineSplineStrikeReward):
        return None
    owner = getattr(env, '_spline_deadline_credit', None)
    if type(cfg.func) is not DeadlineSplineStrikeReward or owner is not cfg.func:
        raise ValueError('credit reward differs from its live deadline owner')
    deadline = validate_credit_deadline(cfg.params.get('credit_deadline_ms'))
    if cfg.weight != 1. or deadline != owner.credit_deadline_ms or cfg.params.get('width_m') != .01:
        raise ValueError('credit reward must preserve one geometric accepted-event unit and deadline')
    if set(cfg.params) != {'robot_cfg', 'width_m', 'credit_deadline_ms'}:
        raise ValueError('credit reward has unexpected parameters')
    terminal = env.termination_manager.get_term_cfg('nail_driven')
    if (terminal.func is not horizon_window_complete or terminal.params != {'window_s': .2}
            or terminal.time_out is not False):
        raise ValueError('credit screen requires unchanged200ms true termination')
    manager = env.observation_manager
    actor, critic = tuple(manager.active_terms['actor']), tuple(manager.active_terms['critic'])
    if set(actor) & set(CREDIT_EVENT_FIELDS):
        raise ValueError('credit context must remain critic-only')
    if critic == actor:
        mode = 'current'
    elif critic == (*actor, *CREDIT_EVENT_FIELDS):
        mode = 'event'
    else:
        raise ValueError('critic event observations have wrong semantic order')
    identity = experiment_identity(deadline, mode)
    if tuple(manager.group_obs_dim['actor']) != (43,) or tuple(manager.group_obs_dim['critic']) != (identity['critic_width'],):
        raise ValueError('credit actor/critic observation widths differ')
    if env.cfg.observations['critic'].enable_corruption is not False:
        raise ValueError('credit critic observations must not be corrupted')
    if mode == 'event':
        for field in CREDIT_EVENT_FIELDS:
            term = manager.get_term_cfg('critic', field)
            if (term.func is not credit_event_observation
                    or term.params != {'field': field, 'credit_deadline_ms': deadline}
                    or term.noise is not None or term.clip is not None or term.scale is not None
                    or term.history_length != 0 or term.delay_min_lag != 0
                    or term.delay_max_lag != 0 or term.delay_hold_prob != 0
                    or term.delay_update_period != 0):
                raise ValueError('critic event observations must preserve exact raw semantics')
    return identity
