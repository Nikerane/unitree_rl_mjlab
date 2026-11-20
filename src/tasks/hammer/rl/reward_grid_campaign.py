"""Explicit immutable 29-setting reward-only campaign; old recipes stay exact."""
from copy import deepcopy
import json

MARKER = 'reward_grid_campaign'
SETTINGS = tuple(f'R{i:02d}' for i in range(1,30))
ZERO_REWARDS = ('native_depth_progress031','native_depth_completion031','speed_goal')


def identity(setting_id):
    if type(setting_id) is not str or setting_id not in SETTINGS:
        raise ValueError('reward grid setting must be R01 through R29')
    i = int(setting_id[1:])-1
    if i < 27:
        impulse,progress = (.5,1.,2.)[i//9],(.5,1.,2.)[(i//3)%3]
        bonus,amount = ('none','physical_contact','accepted_strike')[i%3],0. if i%3 == 0 else .1
    else:
        impulse,progress,bonus,amount = 1.,1.,('physical_contact' if i == 27 else 'accepted_strike'),1.
    return dict(schema='z1_reward_grid_campaign_v1',setting_id=setting_id,reference_id='U22',
        updates=200,impulse_event_weight=impulse,progress_weight=progress,bonus=bonus,bonus_amount=amount,
        episode_length_s=5.,impulse_window_s=.05,finalize_on_depth=False,productive_gate_m=.0005,
        event_payment='first_policy_boundary_at_or_after50ms_no_early_reset_flush',
        impulse_units='linear_J50_divided_by_I_ref_event_units_manager_dt_compensated',
        binary_manager_weight=1.,binary_amount_location='owner_output',
        completion_weight=0.,contact_horizon_s=.2,
        actor_mask='normalized_strike_ref_error',actor_columns=[31,32,33],critic_width=47,
        critic_context='accepted_first_event_seen_age_unpaid_time_to50ms_common_all_settings',
        startup_ramp_s=0.,guidance='ordered_geometric_path_no_clock',scope='reward_grid')


def validate_identity(value):
    if type(value) is not dict:
        raise ValueError('reward grid identity missing')
    expected = identity(value.get('setting_id'))
    if json.dumps(value,sort_keys=True,allow_nan=False) != json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('reward grid identity differs from exact recipe')
    return expected


def weights(marker):
    from src.tasks.hammer.rl.spline_split_event036_contract import EVENT_REWARD_WEIGHTS
    return dict(EVENT_REWARD_WEIGHTS,delivered_impulse=marker['impulse_event_weight'],
        native_depth_progress031=0.,native_depth_completion031=0.,speed_goal=0.,
        spline_progress=marker['progress_weight'])


def configure(cfg, setting_id):
    from src.tasks.hammer.config.z1.spline_exploration038_cfg import set_impulse_reward_weight
    from src.tasks.hammer.mdp.reward_grid import RewardGridEvent, RewardGridDeliveredImpulse, event_observation
    marker = identity(setting_id)
    if getattr(cfg,MARKER,None) is not None or getattr(cfg,'demo_progress_campaign',None) is not None:
        raise ValueError('reward grid cannot mix campaign recipes')
    # Keep the historical depth observer's exact instrumentation recipe. It
    # does not own delivered impulse credit or the new event's binary payment.
    set_impulse_reward_weight(cfg,8.)
    for name,weight in weights(marker).items(): cfg.rewards[name].weight = weight
    cfg.rewards['delivered_impulse'].func = RewardGridDeliveredImpulse
    event = cfg.rewards['accepted_spline_strike']
    event.func = RewardGridEvent
    event.params = dict(robot_cfg=event.params['robot_cfg'],width_m=.01,
        bonus=marker['bonus'],bonus_amount=marker['bonus_amount'])
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    for field in CREDIT_EVENT_FIELDS:
        cfg.observations['critic'].terms[field].func = event_observation
    cfg.episode_length_s = 5.
    cfg.metrics['first_strike'].params.update(finalize_on_depth=False,progress_eps=.0005)
    setattr(cfg,MARKER,marker)
    configured(cfg)
    return cfg


def configured(cfg):
    value = getattr(cfg,MARKER,None)
    if value is None: return None
    marker = validate_identity(value)
    if getattr(cfg,'demo_progress_campaign',None) is not None:
        raise ValueError('reward grid cannot mix demo campaign')
    observer = cfg.metrics['native_depth_owner031']
    if (observer.params.get('exploration_setting') != 'independent'
            or observer.params.get('impulse_reward_weight') != 8.):
        raise ValueError('reward grid requires unchanged independent8 depth instrumentation')
    from src.tasks.hammer.mdp.human_reference import reset_human_reference
    from src.tasks.hammer.mdp.reward_grid import RewardGridEvent, RewardGridDeliveredImpulse, event_observation
    event = cfg.events.get('sample_strike_route_signs')
    first = cfg.metrics['first_strike'].params
    if (event is None or event.func is not reset_human_reference or event.params != {'reference_id':'U22'}
            or cfg.episode_length_s != 5. or first.get('finalize_on_depth') is not False
            or first.get('progress_eps') != .0005
            or abs(first['window_substeps']*cfg.sim.mujoco.timestep-.05)>1e-9):
        raise ValueError('reward grid requires native U22 full50ms productive window and5s timeout')
    for name,weight in weights(marker).items():
        if name not in cfg.rewards or cfg.rewards[name].weight != weight:
            raise ValueError('reward grid configured reward differs: '+name)
    reward = cfg.rewards['accepted_spline_strike']
    if (reward.func is not RewardGridEvent or set(reward.params) != {'robot_cfg','width_m','bonus','bonus_amount'}
            or reward.params['width_m'] != .01 or reward.params['bonus'] != marker['bonus']
            or reward.params['bonus_amount'] != marker['bonus_amount']
            or cfg.rewards['delivered_impulse'].func is not RewardGridDeliveredImpulse):
        raise ValueError('reward grid event implementation differs')
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    if any(cfg.observations['critic'].terms[f].func is not event_observation for f in CREDIT_EVENT_FIELDS):
        raise ValueError('reward grid critic context differs')
    return marker


def live(env):
    marker = configured(env.cfg)
    if marker is None:return None
    if (getattr(env,'_u22_reference_clock',None) is not None
            or getattr(env,'u22_lookahead_ablation',None) is not None
            or env._sampled_pd500.startup_ramp_s != 0.
            or env._sampled_pd500.gain_mode != 'variable'):
        raise ValueError('reward grid requires zero-ramp native VIC guidance')
    from src.tasks.hammer.mdp.human_reference import human_metadata
    if human_metadata(env)['reference_id'] != 'U22':raise ValueError('reward grid reference differs')
    return marker


def exploration_identity(setting_id):
    from src.tasks.hammer.rl.exploration038_contract import identity as historical
    marker = identity(setting_id)
    result = historical('independent',8.)
    result.update(schema_version='spline_exploration038_reward_grid_v1',
        ablation='reward_grid',reward_grid_setting=setting_id)
    result['impulse']['weight'] = marker['impulse_event_weight']
    result['impulse'].update(units=marker['impulse_units'],reader='RewardGridDeliveredImpulse',payment=marker['event_payment'])
    result['depth']['progress']['weight'] = 0.
    result['depth']['completion']['weight'] = 0.
    result['added_rewards'].update(native_depth_progress031=False,native_depth_completion031=False)
    result['early_fraction'] = 1.
    result['reserve_fraction'] = 0.
    result['credit_timing'] = dict(schema_version='reward_grid_50ms_v1',
        boundary=marker['event_payment'],binary_bonus=marker['bonus'],binary_amount=marker['bonus_amount'])
    result['critic_event_semantics'].update(
        credit_unpaid='accepted_event_unit_unpaid_until50ms_independent_of_bonus',
        credit_time_to_payout='time_to50ms_policy_boundary_divided_by200ms_zero_if_paid')
    return result


def validate_metadata(metadata):
    value = metadata.get(MARKER)
    if value is None:
        if metadata.get('spline_exploration038',{}).get('schema_version') == 'spline_exploration038_reward_grid_v1':
            raise ValueError('reward grid marker missing')
        return None
    marker = validate_identity(value)
    from src.tasks.hammer.mdp.human_reference import reference_identity
    if (metadata.get('demo_progress_campaign') is not None or metadata.get('u22_lookahead_ablation') is not None
            or metadata.get('spline_exploration038') != exploration_identity(marker['setting_id'])
            or metadata.get('human_reference') != reference_identity('U22')):
        raise ValueError('reward grid checkpoint mixes incompatible identities')
    if metadata.get('reward_profile',{}).get('credit_timing') != exploration_identity(marker['setting_id'])['credit_timing']:
        raise ValueError('reward grid checkpoint payment timing differs')
    terms = metadata.get('reward_profile',{}).get('terms',{})
    if set(terms) != set(weights(marker)) or any(terms[n].get('weight') != w for n,w in weights(marker).items()):
        raise ValueError('reward grid checkpoint reward profile differs')
    bonus = terms['accepted_spline_strike']
    if (bonus['impl'] != 'src.tasks.hammer.mdp.reward_grid.RewardGridEvent'
            or bonus['params'].get('bonus') != marker['bonus']
            or bonus['params'].get('bonus_amount') != marker['bonus_amount']
            or terms['delivered_impulse']['impl'] != 'src.tasks.hammer.mdp.reward_grid.RewardGridDeliveredImpulse'):
        raise ValueError('reward grid checkpoint event recipe differs')
    return marker


def install_model_mask(actor,critic,metadata):
    if validate_metadata(metadata) is None:return
    if critic.obs_normalizer.mask_lookahead:raise ValueError('reward grid critic must retain lookahead')
    actor.obs_normalizer.mask_lookahead = True


def normalized_initial(value):
    """Validate then canonicalize only the approved reward differences to R14.

    Every model/normalizer/RNG/observation/physical hash remains untouched.
    Smoke and production cell bookkeeping are retained for callers to compare
    under their explicit budget rules.
    """
    result = deepcopy(value)
    marker = validate_metadata(result['contract'])
    if marker is None:raise ValueError('reward grid initialization identity missing')
    if result['cell'].get(MARKER) != marker:raise ValueError('initial cell/contract recipe differs')
    canonical = identity('R14')
    result['cell'][MARKER] = canonical
    contract = result['contract']
    contract[MARKER] = canonical
    contract['spline_exploration038'] = exploration_identity('R14')
    for name,weight in weights(canonical).items():contract['reward_profile']['terms'][name]['weight'] = weight
    contract['reward_profile']['terms']['accepted_spline_strike']['params'].update(bonus='physical_contact',bonus_amount=.1)
    contract['reward_profile']['credit_timing'] = contract['spline_exploration038']['credit_timing']
    return result


from src.tasks.hammer.rl.demo_progress_campaign import verify_zero_ledger, validate_runner_budget


def validate_event_layout(env,marker):
    """Validate the exact shared owner, payout parameters and raw critic47 tail."""
    from src.tasks.hammer.mdp.reward_grid import RewardGridEvent, event_observation
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
    from src.tasks.hammer.rl.exploration038_contract import _raw_term
    owner = getattr(env,'_reward_grid_event',None)
    cfg = env.reward_manager.get_term_cfg('accepted_spline_strike')
    declared = env.cfg.rewards['accepted_spline_strike'].params
    # Managers resolve SceneEntityCfg names into live IDs on their own copy.
    # Compare exact scalar recipe fields and validate the resolved target,
    # rather than requiring equality to the unresolved configuration object.
    expected_keys = {'robot_cfg','width_m','bonus','bonus_amount'}
    robot = cfg.params.get('robot_cfg')
    if (type(owner) is not RewardGridEvent or cfg.func is not owner
            or cfg.weight != 1. or owner._env is not env
            or owner.bonus != marker['bonus'] or owner.bonus_amount != marker['bonus_amount']
            or set(cfg.params) != expected_keys
            or any(cfg.params[key] != declared[key] for key in expected_keys-{'robot_cfg'})
            or robot.name != 'robot' or tuple(robot.site_names) != ('hammer_head_site',)
            or robot.site_ids != env._hammer_first_strike._site_ids
            or hasattr(env,'_spline_split_credit') or hasattr(env,'_spline_deadline_credit')):
        raise ValueError('reward grid live event owner differs')
    terminal = env.termination_manager.get_term_cfg('nail_driven')
    if (terminal.func is not horizon_window_complete or terminal.params != {'window_s':.2}
            or terminal.time_out is not False):raise ValueError('reward grid requires200ms contact horizon')
    manager = env.observation_manager
    actor,critic = tuple(manager.active_terms['actor']),tuple(manager.active_terms['critic'])
    if (set(actor)&set(CREDIT_EVENT_FIELDS) or critic != (*actor,*CREDIT_EVENT_FIELDS)
            or tuple(manager.group_obs_dim['actor']) != (43,) or tuple(manager.group_obs_dim['critic']) != (47,)
            or env.cfg.observations['critic'].enable_corruption is not False
            or tuple(manager.group_obs_term_dim['critic']) != (*manager.group_obs_term_dim['actor'],*((1,),)*4)):
        raise ValueError('reward grid requires uncorrupted actor43/critic47 layout')
    for field in CREDIT_EVENT_FIELDS:
        if not _raw_term(manager.get_term_cfg('critic',field),event_observation,{'field':field}):
            raise ValueError('reward grid raw critic context differs')
    tracker = env._hammer_first_strike
    if tracker._finalize_on_depth is not False or tracker._progress_eps != .0005:
        raise ValueError('reward grid live full50ms/productive gate differs')


def verify_initialization(actual_path, reference_path):
    """Reject any unqualified initialization drift before the first action."""
    actual = json.loads(actual_path.read_text())
    expected = json.loads(reference_path.read_text())
    if (actual['cell']['updates'] != 200 or actual['cell']['scope'] != 'reward_grid'
            or expected['cell']['updates'] != 1 or expected['cell']['scope'] != 'smoke'
            or actual['cell']['worlds'] != 1024 or expected['cell']['worlds'] != 1024
            or expected['cell'][MARKER]['setting_id'] != 'R14'):
        raise ValueError('reward grid requires same-seed qualified1024-world R14 smoke initialization')
    a,e = normalized_initial(actual),normalized_initial(expected)
    a['cell']['updates'],a['cell']['scope'] = 1,'smoke'
    if a != e:raise ValueError('reward grid initialization differs from qualification')
    return True
