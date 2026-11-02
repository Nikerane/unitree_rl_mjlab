"""Closed split/event036 identity, separate from the strict historical current43 marker."""
import math

import torch
from torch import nn

from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
from src.tasks.hammer.mdp.spline_split_credit import SplitSplineStrikeReward
from src.tasks.hammer.mdp.spline_split_event036 import split_event_observation
from src.tasks.hammer.rl.spline_split_credit_contract import (
    split_identity, credit_timing, validate_split_reward_controller, _need, _same,
)
from src.tasks.hammer.rl.spline_exploration_pilot import PILOT_ACTOR_CLASSES, PilotGaussian
from src.tasks.hammer.rl.spline_credit_critic import CreditEventCritic
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedNormalization


def split_event_identity(setting, early_fraction=.5):
    """Describe the two actual balances and their critic-only information."""
    _need(type(early_fraction) is float and early_fraction == .5,
          'event036 requires half credit at50ms and half at200ms')
    marker = split_identity(setting, early_fraction)
    timing = credit_timing(early_fraction)
    timing['observability'] = 'critic_only_raw_split_contact_age_balance_next_payout'
    marker.update(
        schema_version='spline_split_event036_v1', family='split_event036',
        reserve_fraction=1.-early_fraction, credit_timing=timing,
        critic_mode='event', critic_width=47, critic_event_fields=list(CREDIT_EVENT_FIELDS),
        critic_class='src.tasks.hammer.rl.spline_credit_critic:CreditEventCritic',
        critic_event_semantics=dict(
            credit_contact_seen='first_contact_started',
            credit_contact_age='elapsed_integer_physics_ticks_divided_by200ms_clipped_0_1',
            credit_unpaid='early_balance_plus_reserve_balance_in_accepted_event_units',
            credit_time_to_payout='time_to_next_nonzero_tranche_policy_boundary_divided_by200ms_zero_if_paid',
            read_order='after_reward_and_optional_auto_reset_no_reward_recompute'),
        added_rewards=dict(delivered_impulse=False, nail_depth_progress=False, depth_30mm_bonus=False))
    return marker


def validate_split_event_identity(value):
    _need(type(value) is dict, 'split event036 marker missing')
    expected = split_event_identity(value.get('setting'), value.get('early_fraction'))
    _need(_same(value, expected), 'split event036 marker differs from exact schema')
    return expected


EVENT_REWARD_WEIGHTS = dict(approach=0., nail_driven=0., nail_depth_delta=0., impact_progress=0.,
                   completion=0., action_rate=-.01, joint_pos_limits=-10., r_imit=0.,
                   delivered_impulse=0., r_tt=-1., speed_goal=-.8493676,
                   spline_path_cost=-.2, spline_progress=1., accepted_spline_strike=1.)


def validate_reward_weights(env, weights):
    """Check both configured and live reward sets for an explicitly admitted recipe."""
    _need(set(env.cfg.rewards) == set(weights), 'configured reward set differs')
    for name, weight in weights.items():
        _need(env.cfg.rewards[name].weight == weight, 'configured reward weight differs: '+name)
    # mjlab registers zero-weight terms too; compute() skips their evaluation.
    _need(set(env.reward_manager.active_terms) == set(weights), 'live reward set differs')
    for name in weights:
        _need(env.reward_manager.get_term_cfg(name).weight == weights[name], 'live reward weight differs: '+name)


def validate_native_constraints(env):
    """Require the unchanged velocity/impulse CaT configurations and live owners."""
    from src.tasks.hammer.cat.hook import CatSoftHook
    from src.tasks.hammer.mdp.impulse_bound import SubstepImpulseAccumulator
    from src.tasks.hammer.mdp.velocity_bound import SubstepPeakJointVel

    params = dict(use_vel=True, use_impulse=True, imp_limit=[.369,.246,.738,.369,.246,.0164],
                  imp_max_p=.2, imp_seed=.001, limit=3.1415, max_p=.5, min_p=0., tau=.95,
                  vel_detection='substep')
    metrics = getattr(env.metrics_manager, 'cfg', None)
    _need(type(metrics) is dict and 'cat_soft' in metrics, 'event036 requires live constraint manager configs')
    for configs in (env.cfg.metrics, metrics):
        term = configs['cat_soft']
        _need(_same({k:v for k,v in term.params.items() if k != 'robot_cfg'}, params)
              and term.per_substep is False and term.reduce == 'mean', 'event036 CaT parameters differ')
        joints = term.params['robot_cfg'].joint_names
        _need(tuple(joints) == tuple('joint'+str(i) for i in range(1,7)), 'event036 CaT arm joints differ')
        peak = configs['substep_peak_qv']
        _need((peak.func is SubstepPeakJointVel or type(peak.func) is SubstepPeakJointVel)
              and peak.per_substep is True and peak.reduce == 'mean' and peak.params == {},
              'event036 substep velocity tracker differs')
        impulse = configs['substep_impulse']
        _need((impulse.func is SubstepImpulseAccumulator or type(impulse.func) is SubstepImpulseAccumulator)
              and impulse.per_substep is True and impulse.reduce == 'last'
              and set(impulse.params) == {'sensor_name','robot_cfg','subtract_baseline','event_window_substeps'}
              and impulse.params['sensor_name'] == 'hammer_nail_contact'
              and impulse.params['subtract_baseline'] is True and impulse.params['event_window_substeps'] == 200
              and tuple(impulse.params['robot_cfg'].joint_names) == tuple('joint'+str(i) for i in range(1,7)),
              'event036 reaction impulse tracker differs')
    # The sampled controller converts the original25 x2ms window to200 x0.25ms.
    # Require the actual live owner and the same50ms window, never a25-tick runtime fallback.
    impulse = metrics['substep_impulse'].func
    _need(type(impulse) is SubstepImpulseAccumulator
          and impulse is getattr(env, '_hammer_substep_impulse', None)
          and impulse._window == 200 and impulse._subtract_baseline is True,
          'event036 live reaction impulse owner/window differs')
    hook = metrics['cat_soft'].func
    _need(type(hook) is CatSoftHook and hook._use_vel and hook._use_impulse
          and hook._max_p == .5 and hook._imp_max_p == .2 and hook._limit == 3.1415
          and hook._imp_seed == .001 and hook._vel_detection == 'substep'
          and hook._velocity_pressure is None and hook._cat.tau == .95 and hook._cat.min_p == 0.,
          'event036 live velocity/impulse CaT treatment differs')
    limit = hook._imp_limit
    _need(isinstance(limit, torch.Tensor) and limit.shape == (6,)
          and torch.equal(limit, limit.new_tensor(params['imp_limit'])), 'event036 live impulse limits differ')


def split_event_layout(env):
    """Validate the common split/controller and exact raw actor43/critic47 layout."""
    if 'accepted_spline_strike' not in env.reward_manager.active_terms:
        return None
    reward = env.reward_manager.get_term_cfg('accepted_spline_strike')
    manager = env.observation_manager
    configured = env.cfg.observations['critic'].terms
    uses_reader = any(term.func is split_event_observation for term in configured.values())
    is_split = isinstance(reward.func, SplitSplineStrikeReward) or hasattr(env, '_spline_split_credit')
    enriched = (tuple(manager.group_obs_dim['critic']) != (43,)
                or bool(set(manager.active_terms['critic']) & set(CREDIT_EVENT_FIELDS)))
    if not uses_reader and not (is_split and enriched):
        return None
    shared = validate_split_reward_controller(env)
    _need(shared is not None, 'split event036 requires its split ledger owner')
    setting, fraction = shared
    split_event_identity(setting, fraction)
    actor, critic = tuple(manager.active_terms['actor']), tuple(manager.active_terms['critic'])
    _need(not set(actor) & set(CREDIT_EVENT_FIELDS) and critic == (*actor, *CREDIT_EVENT_FIELDS),
          'split event036 requires four critic-only fields in exact order')
    _need(tuple(manager.group_obs_dim['actor']) == (43,) and tuple(manager.group_obs_dim['critic']) == (47,)
          and env.cfg.observations['critic'].enable_corruption is False,
          'split event036 requires uncorrupted actor43/critic47 observations')
    _need(tuple(manager.group_obs_term_dim['critic']) == (*manager.group_obs_term_dim['actor'], *((1,),)*4),
          'split event036 event field widths differ')
    for field in CREDIT_EVENT_FIELDS:
        term = manager.get_term_cfg('critic', field)
        _need(term.func is split_event_observation and term.params == {'field':field}
              and term.noise is None and term.clip is None and term.scale is None
              and term.history_length == 0 and term.delay_min_lag == 0 and term.delay_max_lag == 0
              and term.delay_hold_prob == 0 and term.delay_update_period == 0,
              'split event036 fields must preserve exact raw read-only semantics')
    return setting, fraction


def split_event_metadata(env):
    """Return None for old current43 split and all single-deadline recipes."""
    shared = split_event_layout(env)
    if shared is None:
        return None
    _need(not hasattr(env, '_z1_depth_credit031'), 'event036 must not carry a native depth reward owner')
    validate_reward_weights(env, EVENT_REWARD_WEIGHTS)
    validate_native_constraints(env)
    return split_event_identity(*shared)


def validate_split_event_models(actor, critic, full_metadata):
    _need(type(full_metadata) is dict, 'split event036 requires full metadata')
    marker = validate_split_event_identity(full_metadata.get('spline_split_event036'))
    _need(not any(key in full_metadata for key in (
        'spline_split_credit','spline_exploration_pilot','credit_critic_experiment',
        'spline_learning_variant','spline_depth_reward031','spline_reward_exploration032','spline_reward037')),
        'split event036 cannot claim mixed or historical current43 markers')
    validate_event_models(actor, critic, marker['setting'])


def validate_event_models(actor, critic, setting):
    """Check the unchanged learned Gaussian actor and event-aware critic models."""
    for group, model, cls, width, out in (
        ('actor', actor, PILOT_ACTOR_CLASSES[setting], 43, 13),
        ('critic', critic, CreditEventCritic, 47, 1),
    ):
        _need(type(model) is cls and model.obs_dim == width and tuple(model.obs_groups) == (group,),
              'split event036 model class/observation signature differs')
        for value in model.state_dict().values():
            _need(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                  and (not value.is_floating_point() or value.dtype == torch.float32),
                  'split event036 model state requires finite float32 tensors')
        layers = [layer for layer in model.mlp if isinstance(layer, nn.Linear)]
        _need(bool(layers) and layers[0].in_features == width and layers[-1].out_features == out,
              'split event036 layer signature differs')
        normalizer = model.obs_normalizer
        _need(type(normalizer) is ImpactSpeedNormalization and normalizer.physical.mean.shape == (40,)
              and normalizer.physical.eps == .01 and normalizer.physical.until is None,
              'split event036 physical40 normalizer differs')
    d = actor.distribution
    _need(actor.pilot_setting == setting and type(d) is PilotGaussian and d.output_dim == 13
          and d.std_type == 'scalar' and isinstance(d.std_param, nn.Parameter) and d.std_param.requires_grad
          and d.std_param.shape == (13,) and bool(torch.isfinite(d.std_param).all())
          and tuple(d.std_range) == (1e-6,1e6)
          and tuple(d.log_std_range) == (math.log(1e-6),math.log(1e6))
          and d.gain_entropy_weight == 1., 'split event036 Gaussian semantics differ')
    _need(critic.distribution is None, 'split event036 critic must be deterministic')
    mask = getattr(d, 'activity_mask', None)
    if mask is not None:
        _need(isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.shape[-1] == 13
              and bool((mask[:,:12] == 1).all()) and bool(((mask[:,12] == 0)|(mask[:,12] == 1)).all()),
              'split event036 live action activity mask differs')
