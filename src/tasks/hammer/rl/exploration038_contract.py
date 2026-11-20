"""Closed038 identity and strict live/model guards; historical guards stay exact."""
import math

import torch
from torch import nn

from src.tasks.hammer.config.z1.spline_exploration038_cfg import (
    SETTINGS, ExplorationDepthObserver038, _require_setting, _require_impulse_reward_weight,
)
from src.tasks.hammer.mdp.depth_credit031 import DepthCreditLedger031
from src.tasks.hammer.mdp.depth_reward031 import (
    depth_collect_metric031, depth_progress_reward031, depth_completion_reward031,
)
from src.tasks.hammer.mdp.first_strike import FirstStrikeEventTracker
from src.tasks.hammer.mdp.rewards import FirstStrikeDeliveredRewardTerm
from src.tasks.hammer.mdp.sampled_pd500 import ARM_KP, ARM_KD
from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
from src.tasks.hammer.mdp.spline_exploration_pilot import pilot_setting, pilot_startup_age
from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
from src.tasks.hammer.mdp.spline_split_credit import SplitSplineStrikeReward, validate_split_parameters
from src.tasks.hammer.mdp.spline_split_event036 import split_event_observation
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedNormalization
from src.tasks.hammer.rl.spline_reward037_contract import reward_identity, _validate_training_height
from src.tasks.hammer.rl.spline_split_credit_contract import _need, _same
from src.tasks.hammer.rl.spline_split_event036_contract import (
    EVENT_REWARD_WEIGHTS, validate_native_constraints, validate_reward_weights,
)

MARKER = 'spline_exploration038'
_OLD_MARKERS = ('spline_reward037', 'spline_split_credit', 'spline_split_event036',
    'spline_exploration_pilot', 'credit_critic_experiment', 'spline_learning_variant',
    'spline_depth_reward031', 'spline_reward_exploration032')


def identity(setting, impulse_reward_weight=4.):
    _require_setting(setting)
    _require_impulse_reward_weight(impulse_reward_weight)
    if impulse_reward_weight != 4.:
        _need(setting == 'independent', 'impulse reward treatment requires independent exploration038')
    fixed = setting == 'fixed_nominal'
    mode = 'independent' if fixed else setting
    marker = reward_identity('vic_baseline')
    marker.update(schema_version='spline_exploration038_v1', family='exploration038',
        setting=setting, gain_mode='fixed' if fixed else 'variable',
        actor_class='src.tasks.hammer.rl.reward_exploration038:ExplorationActor038',
        distribution_class='src.tasks.hammer.rl.reward_exploration038:ExplorationGaussian038',
        algorithm_class='src.tasks.hammer.rl.reward_exploration038:ExplorationPPO038',
        runner_class='src.tasks.hammer.rl.reward_exploration038:ExplorationRunner038',
        effective_gain_entropy_weight=0. if fixed else 1.,
        gain_entropy_mode='masked_out' if fixed else 'sum', likelihood_gain_mask=0 if fixed else 1,
        ignored_action_indices=list(range(6,12)) if fixed else [],
        exploration=dict(mode=mode, position_indices=list(range(6)),
            law={'independent':'learned_state_independent_Gaussian',
                 'dependent':'position_state_dependent_diagonal_Gaussian',
                 'gsde':'hidden_feature_gSDE'}[mode],
            gain_speed_law='existing_scalar_Gaussian', state_dependent_std=mode in ('dependent', 'gsde'),
            network_std_head=mode == 'dependent',
            gsde=mode == 'gsde', variance_epsilon=1e-6 if mode == 'gsde' else None,
            feature_map='actor_hidden_detached' if mode == 'gsde' else None,
            feature_dim='last_hidden_width' if mode == 'gsde' else None,
            sde_interval_steps=4 if mode == 'gsde' else None,
            per_world_matrix=mode == 'gsde', resample_on_reset=mode == 'gsde',
            initial_position_std_semantics=('shared_reset_population_sampled_position_rms_0.3; '
                                            'scored_marginal_includes_SB3_variance_epsilon; statewise_std_varies'
                                            if mode == 'gsde' else 'initial_std_0.3')),
        normalization=dict(physical_width=40, empirical_eps=.01,
            primed_before_first_action=True, frozen_during_rollout_and_ppo=True,
            update_after_ppo=True, tail='raw_startup_age_goal_pending_and_critic_event_fields'))
    marker['depth']['owner'] = ('src.tasks.hammer.config.z1.spline_exploration038_cfg:'
                              'ExplorationDepthObserver038')
    if impulse_reward_weight == 0.:
        marker.update(schema_version='spline_exploration038_impulse_reward_off_v1',
                      ablation='delivered_impulse_reward_off')
        marker['impulse']['weight'] = 0.
        marker['added_rewards']['delivered_impulse'] = False
    elif impulse_reward_weight == 8.:
        marker.update(schema_version='spline_exploration038_impulse_reward_8_v1',
                      ablation='delivered_impulse_reward_weight_8')
        marker['impulse']['weight'] = 8.
    return marker


def validate_identity(value):
    _need(type(value) is dict, 'exploration038 marker missing')
    if value.get('schema_version') == 'spline_exploration038_recovery_budget_v1':
        from src.tasks.hammer.rl.recovery_budget_campaign import exploration_identity
        expected=exploration_identity(value.get('recovery_cell'))
        _need(_same(value,expected), 'recovery exploration marker differs')
        return expected
    if value.get('schema_version') == 'spline_exploration038_safety_factorial_v1':
        from src.tasks.hammer.rl.safety_factorial_campaign import exploration_identity
        expected = exploration_identity(value.get('safety_factorial_cell'))
        _need(_same(value,expected), 'safety factorial exploration marker differs')
        return expected
    if value.get('schema_version') == 'spline_exploration038_split_impulse_v1':
        from src.tasks.hammer.rl.split_impulse_campaign import exploration_identity
        expected = exploration_identity(value.get('split_impulse_setting'))
        _need(_same(value,expected), 'split impulse exploration marker differs')
        return expected
    if value.get('schema_version') == 'spline_exploration038_reward_grid_v1':
        from src.tasks.hammer.rl.reward_grid_campaign import exploration_identity
        expected = exploration_identity(value.get('reward_grid_setting'))
        _need(_same(value,expected), 'reward grid exploration marker differs')
        return expected
    impulse = value.get('impulse')
    _need(type(impulse) is dict and 'weight' in impulse, 'exploration038 impulse identity missing')
    expected = identity(value.get('setting'), impulse_reward_weight=impulse['weight'])
    _need(_same(value, expected), 'exploration038 marker differs from exact recipe')
    return expected


def validate_metadata(value):
    _need(type(value) is dict, 'exploration038 requires full checkpoint metadata')
    marker = validate_identity(value.get(MARKER))
    _need(not any(key in value for key in _OLD_MARKERS),
          'exploration038 cannot claim old or mixed recipe markers')
    from src.tasks.hammer.rl.demo_progress_campaign import validate_metadata as validate_demo
    validate_demo(value)
    from src.tasks.hammer.rl.reward_grid_campaign import validate_metadata as validate_grid
    validate_grid(value)
    from src.tasks.hammer.rl.split_impulse_campaign import validate_metadata as validate_impulse
    validate_impulse(value)
    from src.tasks.hammer.rl.safety_factorial_campaign import validate_metadata as validate_factorial
    validate_factorial(value)
    from src.tasks.hammer.rl.recovery_budget_campaign import validate_metadata as validate_recovery
    validate_recovery(value)
    return marker


def _raw_term(term, func, params):
    return (term.func is func and term.params == params and term.noise is None
        and term.clip is None and term.scale is None and term.history_length == 0
        and term.delay_min_lag == 0 and term.delay_max_lag == 0
        and term.delay_hold_prob == 0 and term.delay_update_period == 0)


def _validate_controller(env, setting):
    fixed = setting == 'fixed_nominal'
    pilot = 'fixed_nominal' if fixed else 'vic_baseline'
    _need(pilot_setting(env.cfg) == pilot, 'exploration038 configured startup marker differs')
    controller = getattr(env, '_sampled_pd500', None)
    from src.tasks.hammer.mdp.human_reference import reset_human_reference
    from src.tasks.hammer.mdp.sampled_pd500 import HUMAN_STARTUP_RAMP_S
    event = env.cfg.events.get('sample_strike_route_signs')
    # Explicit zero preserves archival human-policy replay. Serialized controller
    # metadata still prevents silent cross-load between zero and 50ms delivery.
    ramps = (0., HUMAN_STARTUP_RAMP_S) if event is not None and event.func is reset_human_reference else (0.,)
    _need(controller is not None and controller.gain_mode == ('fixed' if fixed else 'variable')
        and controller.startup_ramp_s in ramps, 'exploration038 controller gain mode or ramp differs')
    age = controller.startup_age
    _need(isinstance(age, torch.Tensor) and age.is_floating_point() and age.shape == (env.num_envs,)
        and bool(torch.isfinite(age).all()) and bool(((age >= 0) & (age <= 1)).all()),
        'exploration038 startup age differs')
    for group in ('actor','critic'):
        _need(_raw_term(env.observation_manager.get_term_cfg(group,'startup_age'),
                        pilot_startup_age, {'setting':pilot}),
              'exploration038 live startup observation differs')
    metadata = controller.metadata()
    for name, expected in (('kp',list(ARM_KP)), ('kd',list(ARM_KD)),
                           ('pd_dt_s',.002), ('physics_dt_s',.00025), ('policy_dt_s',.02)):
        _need(_same(metadata.get(name),expected), 'exploration038 nominal controller/timing differs: '+name)
    for name, expected in (('arm_kp',ARM_KP), ('arm_kd',ARM_KD)):
        value = getattr(controller,name,None)
        _need(isinstance(value,torch.Tensor) and value.shape == (6,)
            and torch.equal(value,value.new_tensor(expected)), 'exploration038 nominal gains differ')
    if fixed:
        _need(metadata.get('version') == 'fixed_pd500_v1'
            and _same(metadata.get('ignored_action_indices'),list(range(6,12))),
            'exploration038 fixed controller must ignore precisely six gain actions')
        for name, expected in (('_kp',ARM_KP), ('_kd',ARM_KD)):
            value = getattr(controller.gains,name,None)
            _need(isinstance(value,torch.Tensor) and value.shape == (env.num_envs,6)
                and torch.equal(value,value.new_tensor(expected).expand_as(value)),
                'exploration038 fixed applied gain buffers differ from nominal')
    else:
        _need(metadata.get('version') == 'variable_impedance_pd500_v1'
            and metadata.get('gain_mode') == 'variable'
            and metadata.get('mapping_family') == 'author_v1_exponential'
            and metadata.get('C') == 1.25 and _same(metadata.get('p_bounds'),[-1.,1.])
            and 'ignored_action_indices' not in metadata, 'exploration038 variable mapping differs')


def _validate_split_event(env):
    """Check the actual split ledger and raw event47 layout for either controller."""
    _need('accepted_spline_strike' in env.reward_manager.active_terms,
          'exploration038 requires split accepted credit')
    cfg = env.reward_manager.get_term_cfg('accepted_spline_strike')
    owner = getattr(env,'_spline_split_credit',None)
    _need(type(owner) is SplitSplineStrikeReward and cfg.func is owner
        and not hasattr(env,'_spline_deadline_credit'), 'exploration038 split owner differs')
    params = cfg.params
    _need(set(params) == {'robot_cfg','width_m','early_fraction','early_deadline_ms','reserve_deadline_ms'},
          'exploration038 split parameters differ')
    validate_split_parameters(params['early_fraction'],params['early_deadline_ms'],params['reserve_deadline_ms'])
    _need(cfg.weight == 1. and params['width_m'] == .01 and params['early_fraction'] == .5
        and all(params[k] == getattr(owner,k) for k in
                ('early_fraction','early_deadline_ms','reserve_deadline_ms')),
        'exploration038 split balance/unit/timing differs')
    terminal = env.termination_manager.get_term_cfg('nail_driven')
    _need(terminal.func is horizon_window_complete and terminal.params == {'window_s':.2}
        and terminal.time_out is False, 'exploration038 requires unchanged200ms true termination')
    manager = env.observation_manager
    actor, critic = tuple(manager.active_terms['actor']), tuple(manager.active_terms['critic'])
    _need(not set(actor) & set(CREDIT_EVENT_FIELDS) and critic == (*actor,*CREDIT_EVENT_FIELDS)
        and tuple(manager.group_obs_dim['actor']) == (43,)
        and tuple(manager.group_obs_dim['critic']) == (47,)
        and env.cfg.observations['critic'].enable_corruption is False,
        'exploration038 requires uncorrupted actor43/critic47 event observations')
    _need(tuple(manager.group_obs_term_dim['critic']) == (*manager.group_obs_term_dim['actor'],*((1,),)*4),
          'exploration038 critic event widths differ')
    for field in CREDIT_EVENT_FIELDS:
        _need(_raw_term(manager.get_term_cfg('critic',field),split_event_observation,{'field':field}),
              'exploration038 event fields must preserve exact raw read-only semantics')


def metadata038(env):
    from src.tasks.hammer.rl import recovery_budget_campaign as recovery_api
    from src.tasks.hammer.mdp.recovery_budget import RecoveryDepthObserver
    recovery = recovery_api.live(env)
    owner_class = RecoveryDepthObserver if recovery is not None else ExplorationDepthObserver038
    owner = getattr(env,'_z1_depth_credit031',None)
    declared = getattr(env.cfg,'metrics',{}).get('native_depth_owner031')
    configured = declared is not None and (declared.func is owner_class
                                          or type(declared.func) is owner_class)
    if type(owner) is not owner_class and not configured:
        return None
    _need(type(owner) is owner_class and owner._env is env,
          'exploration038 requires exact live native depth observer')
    setting = owner.exploration_setting
    impulse_reward_weight = owner.impulse_reward_weight
    from src.tasks.hammer.rl import reward_grid_campaign as grid_api
    grid = grid_api.live(env)
    from src.tasks.hammer.rl import split_impulse_campaign as split_api
    split = split_api.live(env)
    from src.tasks.hammer.rl import safety_factorial_campaign as factorial_api
    factorial = factorial_api.live(env)
    marker = (recovery_api.exploration_identity(recovery['cell_id']) if recovery is not None else factorial_api.exploration_identity(factorial['cell_id']) if factorial is not None else
        split_api.exploration_identity(split['setting_id']) if split is not None else
        grid_api.exploration_identity(grid['setting_id']) if grid is not None else identity(setting, impulse_reward_weight=impulse_reward_weight))
    owner_keys = {'arm','nail_cfg','exploration_setting'}
    if impulse_reward_weight != 4.:
        owner_keys.add('impulse_reward_weight')
    _need(declared is not None and declared.func is owner_class
        and declared.per_substep is True and declared.reduce == 'last'
        and set(declared.params) == owner_keys
        and _same(declared.params.get('impulse_reward_weight', 4.), impulse_reward_weight)
        and declared.params['arm'] == 'A3' and declared.params['exploration_setting'] == setting
        and declared.params['nail_cfg'].name == 'nail_block'
        and tuple(declared.params['nail_cfg'].joint_names) == ('nail_slide',),
        'exploration038 configured depth observer differs')
    _validate_training_height(env,False)
    _validate_controller(env,setting)
    if recovery is not None:
        recovery_api.validate_event_layout(env,recovery)
    elif factorial is not None:
        factorial_api.validate_event_layout(env,factorial)
    elif split is not None:
        split_api.validate_event_layout(env,split)
    elif grid is None:
        _validate_split_event(env)
    else:
        grid_api.validate_event_layout(env,grid)
    validate_native_constraints(env)
    from src.tasks.hammer.rl.u22_lookahead import MARKER as LOOKAHEAD_MARKER, identity as lookahead_identity
    lookahead = getattr(env, LOOKAHEAD_MARKER, None)
    if lookahead is not None:
        _need(lookahead == lookahead_identity(lookahead.get('mode')), 'invalid U22 lookahead ablation')
    from src.tasks.hammer.rl.demo_progress_campaign import live as demo_live, weights as demo_weights
    demo = demo_live(env)
    expected_weights = (recovery_api.weights(recovery) if recovery is not None else factorial_api.weights(factorial) if factorial is not None else split_api.weights(split) if split is not None else grid_api.weights(grid) if grid is not None else demo_weights(demo) if demo is not None else
        dict(EVENT_REWARD_WEIGHTS,delivered_impulse=impulse_reward_weight,
             native_depth_progress031=1.,native_depth_completion031=1.,
             spline_progress=1. if lookahead is None else lookahead['progress_weight']))
    validate_reward_weights(env,expected_weights)
    _need(env.physics_dt == .00025 and env.step_dt == .02, 'exploration038 clocks differ')
    _need(owner.arm == 'A3' and type(owner.ledger) is DepthCreditLedger031
        and owner.ledger.num_envs == env.num_envs and owner.ledger.progress_enabled is True
        and owner.ledger.completion_enabled is True, 'exploration038 depth ledger differs')
    metrics = getattr(env.metrics_manager,'cfg',{})
    registered = metrics.get('native_depth_owner031')
    _need(registered is not None and registered.func is owner and registered.per_substep is True
        and registered.reduce == 'last' and set(registered.params) == owner_keys
        and _same(registered.params.get('impulse_reward_weight', 4.), impulse_reward_weight)
        and registered.params['arm'] == 'A3' and registered.params['exploration_setting'] == setting,
        'exploration038 live depth observer registration differs')
    nail = registered.params['nail_cfg']
    _need(nail is owner.nail_cfg and nail.name == 'nail_block'
        and tuple(nail.joint_names) == ('nail_slide',), 'exploration038 native depth coordinate differs')
    for configs in (env.cfg.metrics,metrics):
        collect = configs.get('native_depth_collect031')
        _need(collect is not None and collect.func is depth_collect_metric031 and collect.params == {}
            and collect.per_substep is False and collect.reduce == 'last', 'exploration038 collector differs')
        first = configs.get('first_strike')
        _need(first is not None and first.per_substep is True and first.params['window_substeps'] == 200,
              'exploration038 first50ms window differs')
    tracker = getattr(env,'_hammer_first_strike',None)
    _need(type(tracker) is FirstStrikeEventTracker and tracker._env is env and tracker._window == 200
        and metrics['first_strike'].func is tracker, 'exploration038 live first50ms tracker differs')
    for name, func in (('native_depth_progress031',depth_progress_reward031),
                       ('native_depth_completion031',depth_completion_reward031)):
        for term in (env.cfg.rewards[name],env.reward_manager.get_term_cfg(name)):
            _need(term.func is func and term.params == {}, 'exploration038 depth reward reader differs')
    for term, live in ((env.cfg.rewards['delivered_impulse'],False),
                       (env.reward_manager.get_term_cfg('delivered_impulse'),True)):
        from src.tasks.hammer.mdp.reward_grid import RewardGridDeliveredImpulse
        from src.tasks.hammer.mdp.split_impulse import SplitImpulseDeliveredImpulse
        delivered_class = SplitImpulseDeliveredImpulse if split is not None or factorial is not None or recovery is not None else RewardGridDeliveredImpulse if grid is not None else FirstStrikeDeliveredRewardTerm
        _need((type(term.func) is delivered_class if live else term.func is delivered_class)
            and term.params['i_ref'] == marker['impulse']['i_ref'] and term.params['saturate'] is False,
            'exploration038 historical normalized impulse differs')
    return marker


def observation_metadata038(env, marker):
    """Use the shared direct-reference check without older recipe recognizers."""
    from src.tasks.hammer.rl.runner import _DIRECT_JOINT_OBSERVATIONS, _joint_observation_metadata
    from src.tasks.hammer.rl.impact_speed_contract import _OBSERVATION_WIDTHS
    from src.tasks.hammer.mdp.impact_speed import impact_speed_goal, impact_speed_pending
    from src.tasks.hammer.mdp.spline_learning_variants import AGE_TAIL
    metadata = _joint_observation_metadata(env,scalar_tail=AGE_TAIL,
        critic_extra_tail=tuple(marker['critic_event_fields']))
    widths = (*_OBSERVATION_WIDTHS[:-2],1,1,1)
    _need(tuple(metadata['observation_names']) == (*_DIRECT_JOINT_OBSERVATIONS,*AGE_TAIL),
          'exploration038 direct-reference observation order differs')
    for group in ('actor','critic'):
        expected = (*widths,1,1,1,1) if group == 'critic' else widths
        _need(tuple(env.observation_manager.group_obs_term_dim[group]) == tuple((w,) for w in expected),
              'exploration038 observation widths differ')
        for name, func in (('impact_speed_goal',impact_speed_goal),('impact_speed_pending',impact_speed_pending)):
            _need(_raw_term(env.observation_manager.get_term_cfg(group,name),func,{}),
                  'exploration038 raw goal/pending differs')
    return {**metadata,'observation_term_widths':list(widths),
        'critic_observation_term_widths':[*widths,1,1,1,1],
        'observation_normalization':dict(type='physical40_empirical_fixed_tail_v1',
            empirical_width=40,fixed_tail=list(AGE_TAIL),
            critic_fixed_tail=[*AGE_TAIL,*marker['critic_event_fields']],empirical_eps=.01)}


def validate_models(actor, critic, metadata):
    from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
    from src.tasks.hammer.rl.reward_exploration038 import ExplorationActor038, ExplorationGaussian038
    from src.tasks.hammer.rl.spline_credit_critic import CreditEventCritic
    marker = validate_metadata(metadata)
    from src.tasks.hammer.rl.u22_lookahead import MARKER as LOOKAHEAD_MARKER
    lookahead = metadata.get(LOOKAHEAD_MARKER, {})
    from src.tasks.hammer.rl.demo_progress_campaign import MARKER as DEMO_MARKER
    masked = bool(lookahead.get('actor_columns')) or DEMO_MARKER in metadata or 'reward_grid_campaign' in metadata or 'split_impulse_campaign' in metadata or 'safety_factorial' in metadata or 'recovery_budget' in metadata
    for group, model, cls, width, out in (('actor',actor,ExplorationActor038,43,13),
                                         ('critic',critic,CreditEventCritic,47,1)):
        _need(type(model) is cls and model.obs_dim == width and tuple(model.obs_groups) == (group,),
              'exploration038 model class/observation signature differs')
        _need(all(isinstance(v,torch.Tensor) and bool(torch.isfinite(v).all())
            and (not v.is_floating_point() or v.dtype == torch.float32) for v in model.state_dict().values()),
            'exploration038 requires finite float32 model state')
        layers = [layer for layer in model.mlp if isinstance(layer,nn.Linear)]
        _need(len(layers) >= 2 and layers[0].in_features == width and layers[-1].out_features == out,
              'exploration038 layer signature differs')
        normalizer = model.obs_normalizer
        _need(type(normalizer) is ImpactSpeedNormalization and normalizer.physical.mean.shape == (40,)
            and normalizer.physical.eps == .01 and normalizer.physical.until is None,
            'exploration038 physical40 normalizer differs')
        _need(normalizer.mask_lookahead == (group == 'actor' and masked),
              'U22 actor lookahead mask differs from checkpoint contract')
    d = actor.distribution
    fixed = marker['setting'] == 'fixed_nominal'
    mode = marker['exploration']['mode']
    _need(type(d) is ExplorationGaussian038 and d.output_dim == 13 and d.mode == mode
        and type(d.fixed_nominal) is bool and d.fixed_nominal is fixed and d.setting == marker['setting']
        and d.std_type == 'scalar' and isinstance(d.std_param,nn.Parameter) and d.std_param.requires_grad
        and d.std_param.shape == (13,) and tuple(d.std_range) == (1e-6,1e6)
        and tuple(d.log_std_range) == (math.log(1e-6),math.log(1e6)),
        'exploration038 Gaussian semantics differ')
    _need(critic.distribution is None, 'exploration038 critic must be deterministic')
    _need((actor.std_head is not None) == (mode == 'dependent'), 'exploration038 dependent std head differs')
    if mode == 'dependent':
        _need(type(actor.std_head) is nn.Linear and actor.std_head.weight.shape == (6,43),
              'exploration038 dependent std head shape differs')
        dependent = d.position_distribution
        _need(type(dependent) is HeteroscedasticGaussianDistribution and dependent.output_dim == 6
            and dependent.std_type == 'log' and tuple(dependent.std_range) == (1e-6,1e6)
            and tuple(dependent.log_std_range) == (math.log(1e-6),math.log(1e6)),
            'exploration038 position-only dependent Gaussian differs')
    else:
        _need(d.position_distribution is None, 'exploration038 extraneous dependent Gaussian')
    _need(hasattr(d,'sde_log_std') == (mode == 'gsde'), 'exploration038 extraneous gSDE parameters')
    layers = [layer for layer in actor.mlp if isinstance(layer,nn.Linear)]
    _need(type(d.feature_dim) is int and d.feature_dim == layers[-2].out_features,
          'exploration038 hidden feature width differs')
    if mode == 'gsde':
        _need(d.feature_map == 'actor_hidden_detached' and d.variance_epsilon == 1e-6
            and isinstance(d.sde_log_std,nn.Parameter) and d.sde_log_std.requires_grad
            and d.sde_log_std.shape == (d.feature_dim,6), 'exploration038 hidden-feature gSDE differs')
    mask = getattr(d,'activity_mask',None)
    if mask is not None:
        _need(isinstance(mask,torch.Tensor) and mask.ndim == 2 and mask.shape[-1] == 13
            and bool((mask[:,:6] == 1).all())
            and bool((mask[:,6:12] == marker['likelihood_gain_mask']).all())
            and bool(((mask[:,12] == 0) | (mask[:,12] == 1)).all()),
            'exploration038 live activity mask differs')
