"""Explicit immutable six-setting temporal split-impulse campaign; old recipes stay exact."""
from copy import deepcopy
import json

MARKER = 'split_impulse_campaign'
SETTINGS = tuple(f'S{i:02d}' for i in range(1,7))
ZERO_REWARDS = ('native_depth_progress031','native_depth_completion031','speed_goal')


def identity(setting_id):
    if type(setting_id) is not str or setting_id not in SETTINGS:
        raise ValueError('split impulse setting must be S01 through S06')
    i = int(setting_id[1:])-1
    return dict(schema='z1_split_impulse_campaign_v1',setting_id=setting_id,reference_id='U22',
        updates=200,impulse_event_weight=1.,progress_weight=.5 if i < 3 else 1.,
        early_fraction=(1.,.5,.25)[i%3],early_deadline_ms=50,reserve_deadline_ms=200,
        episode_length_s=5.,impulse_window_s=.05,finalize_on_depth=False,productive_gate_m=.0005,
        event_payment='first_policy_boundaries_at_or_after50_and200ms_discard_future_on_early_reset',
        impulse_units='linear_J50_divided_by_I_ref_event_units_manager_dt_compensated',
        binary_manager_weight=1.,binary_bonus=0.,completion_weight=0.,contact_horizon_s=.2,
        actor_mask='normalized_strike_ref_error',actor_columns=[31,32,33],critic_width=47,
        critic_context='first_contact_seen_age_unpaid_frozen_impulse_time_to_next_pending_payout',
        startup_ramp_s=0.,guidance='ordered_geometric_path_no_clock',scope='split_impulse')


def validate_identity(value):
    if type(value) is not dict:
        raise ValueError('split impulse identity missing')
    expected = identity(value.get('setting_id'))
    if json.dumps(value,sort_keys=True,allow_nan=False) != json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('split impulse identity differs from exact recipe')
    return expected


def weights(marker):
    from src.tasks.hammer.rl.spline_split_event036_contract import EVENT_REWARD_WEIGHTS
    return dict(EVENT_REWARD_WEIGHTS,delivered_impulse=marker['impulse_event_weight'],
        native_depth_progress031=0.,native_depth_completion031=0.,speed_goal=0.,
        spline_progress=marker['progress_weight'])


def configure(cfg, setting_id, *, reference_id='U22'):
    from src.tasks.hammer.config.z1.spline_exploration038_cfg import set_impulse_reward_weight
    from src.tasks.hammer.mdp.split_impulse import SplitImpulseZeroBonus, SplitImpulseDeliveredImpulse, event_observation
    marker = identity(setting_id)
    if any(getattr(cfg,key,None) is not None for key in (MARKER,'demo_progress_campaign','reward_grid_campaign')):
        raise ValueError('split impulse cannot mix campaign recipes')
    # Keep the historical depth observer's exact instrumentation recipe. It
    # does not own delivered impulse credit or the new event's binary payment.
    set_impulse_reward_weight(cfg,8.)
    for name,weight in weights(marker).items(): cfg.rewards[name].weight = weight
    cfg.rewards['delivered_impulse'].func = SplitImpulseDeliveredImpulse
    cfg.rewards['delivered_impulse'].params['early_fraction'] = marker['early_fraction']
    event = cfg.rewards['accepted_spline_strike']
    event.func = SplitImpulseZeroBonus
    event.params = dict(robot_cfg=event.params['robot_cfg'],width_m=.01)
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    for field in CREDIT_EVENT_FIELDS:
        cfg.observations['critic'].terms[field].func = event_observation
    cfg.episode_length_s = 5.
    cfg.metrics['first_strike'].params.update(finalize_on_depth=False,progress_eps=.0005)
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from src.tasks.hammer.mdp.split_impulse import accounting_metric
    cfg.metrics['split_impulse_accounting'] = MetricsTermCfg(func=accounting_metric,per_substep=False,reduce='last')
    setattr(cfg,MARKER,marker)
    configured(cfg, reference_id=reference_id)
    return cfg


def configured(cfg, *, reference_id='U22'):
    value = getattr(cfg,MARKER,None)
    if value is None: return None
    marker = validate_identity(value)
    if any(getattr(cfg,key,None) is not None for key in ('demo_progress_campaign','reward_grid_campaign')):
        raise ValueError('split impulse cannot mix demo campaign')
    observer = cfg.metrics['native_depth_owner031']
    if (observer.params.get('exploration_setting') != 'independent'
            or observer.params.get('impulse_reward_weight') != 8.):
        raise ValueError('split impulse requires unchanged independent8 depth instrumentation')
    from src.tasks.hammer.mdp.human_reference import FIVE_REFERENCE, reset_human_reference
    from src.tasks.hammer.mdp.split_impulse import SplitImpulseZeroBonus, SplitImpulseDeliveredImpulse, event_observation
    event = cfg.events.get('sample_strike_route_signs')
    first = cfg.metrics['first_strike'].params
    if (reference_id not in ('U22', FIVE_REFERENCE) or event is None or event.func is not reset_human_reference
            or event.params != {'reference_id':reference_id}
            or cfg.episode_length_s != 5. or first.get('finalize_on_depth') is not False
            or first.get('progress_eps') != .0005
            or abs(first['window_substeps']*cfg.sim.mujoco.timestep-.05)>1e-9):
        raise ValueError('split impulse requires native U22 full50ms productive window and5s timeout')
    for name,weight in weights(marker).items():
        if name not in cfg.rewards or cfg.rewards[name].weight != weight:
            raise ValueError('split impulse configured reward differs: '+name)
    reward = cfg.rewards['accepted_spline_strike']
    if (reward.func is not SplitImpulseZeroBonus or set(reward.params) != {'robot_cfg','width_m'}
            or reward.params['width_m'] != .01
            or cfg.rewards['delivered_impulse'].func is not SplitImpulseDeliveredImpulse
            or cfg.rewards['delivered_impulse'].params.get('early_fraction') != marker['early_fraction']):
        raise ValueError('split impulse event implementation differs')
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    _validate_impulse_params(cfg.rewards['delivered_impulse'].params,marker)
    from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
    from src.tasks.hammer.mdp.split_impulse import accounting_metric
    terminal = cfg.terminations['nail_driven']
    accounting = cfg.metrics.get('split_impulse_accounting')
    if ((cfg.sim.mujoco.timestep,cfg.decimation) not in ((.002,10),(.00025,80))
            or terminal.func is not horizon_window_complete or terminal.params != {'window_s':.2}
            or terminal.time_out is not False or accounting is None
            or accounting.func is not accounting_metric or accounting.per_substep is not False
            or accounting.params != {} or accounting.reduce != 'last'
            or list(cfg.metrics).index('split_impulse_accounting') <= list(cfg.metrics).index('cat_soft')):
        raise ValueError('split impulse clocks, horizon or accounting recipe differs')
    if any(cfg.observations['critic'].terms[f].func is not event_observation for f in CREDIT_EVENT_FIELDS):
        raise ValueError('split impulse critic context differs')
    from src.tasks.hammer.rl.exploration038_contract import _raw_term
    if any(not _raw_term(cfg.observations['critic'].terms[f],event_observation,{'field':f})
           for f in CREDIT_EVENT_FIELDS):raise ValueError('split impulse configured raw critic fields differ')
    return marker


def _validate_impulse_params(params,marker):
    expected_i_ref = exploration_identity(marker['setting_id'])['impulse']['i_ref']
    if (set(params) != {'i_ref','eps','nail_cfg','saturate','early_fraction'}
            or params['i_ref'] != expected_i_ref or params['saturate'] is not False
            or params['eps'] != .0005 or params['early_fraction'] != marker['early_fraction']):
        raise ValueError('split impulse normalized linear budget parameters differ')


def live(env):
    marker = configured(env.cfg)
    if marker is None:return None
    if (getattr(env,'_u22_reference_clock',None) is not None
            or getattr(env,'u22_lookahead_ablation',None) is not None
            or env._sampled_pd500.startup_ramp_s != 0.
            or env._sampled_pd500.gain_mode != 'variable'):
        raise ValueError('split impulse requires zero-ramp native VIC guidance')
    from src.tasks.hammer.mdp.human_reference import human_metadata
    if human_metadata(env)['reference_id'] != 'U22':raise ValueError('split impulse reference differs')
    return marker


def exploration_identity(setting_id):
    from src.tasks.hammer.rl.exploration038_contract import identity as historical
    marker = identity(setting_id)
    result = historical('independent',8.)
    result.update(schema_version='spline_exploration038_split_impulse_v1',
        ablation='split_impulse',split_impulse_setting=setting_id)
    result['impulse']['weight'] = marker['impulse_event_weight']
    result['impulse'].update(units=marker['impulse_units'],reader='SplitImpulseDeliveredImpulse',payment=marker['event_payment'])
    result['depth']['progress']['weight'] = 0.
    result['depth']['completion']['weight'] = 0.
    result['added_rewards'].update(native_depth_progress031=False,native_depth_completion031=False)
    result['early_fraction'] = marker['early_fraction']
    result['reserve_fraction'] = 1.-marker['early_fraction']
    result['credit_timing'] = dict(schema_version='split_impulse_50_200ms_v1',
        boundary=marker['event_payment'],early_fraction=marker['early_fraction'],
        reserve_fraction=1.-marker['early_fraction'],binary_bonus=0.)
    result['critic_event_semantics'].update(
        credit_unpaid='raw_frozen_eligible_J50_divided_by_I_ref_unpaid_zero_before_measurement',
        credit_time_to_payout='time_to_next_pending_policy_payout_divided_by200ms_zero_if_no_balance')
    return result


def validate_metadata(metadata):
    value = metadata.get(MARKER)
    if value is None:
        if metadata.get('spline_exploration038',{}).get('schema_version') == 'spline_exploration038_split_impulse_v1':
            raise ValueError('split impulse marker missing')
        return None
    marker = validate_identity(value)
    from src.tasks.hammer.mdp.human_reference import reference_identity
    if (metadata.get('demo_progress_campaign') is not None or metadata.get('reward_grid_campaign') is not None or metadata.get('u22_lookahead_ablation') is not None
            or metadata.get('spline_exploration038') != exploration_identity(marker['setting_id'])
            or metadata.get('human_reference') != reference_identity('U22')):
        raise ValueError('split impulse checkpoint mixes incompatible identities')
    if metadata.get('reward_profile',{}).get('credit_timing') != exploration_identity(marker['setting_id'])['credit_timing']:
        raise ValueError('split impulse checkpoint payment timing differs')
    terms = metadata.get('reward_profile',{}).get('terms',{})
    if set(terms) != set(weights(marker)) or any(terms[n].get('weight') != w for n,w in weights(marker).items()):
        raise ValueError('split impulse checkpoint reward profile differs')
    _validate_impulse_params(terms['delivered_impulse']['params'],marker)
    terminal = metadata['reward_profile'].get('termination',{})
    if (terminal.get('impl') != 'src.tasks.hammer.mdp.spline_horizon.horizon_window_complete'
            or terminal.get('params') != {'window_s':.2}):
        raise ValueError('split impulse checkpoint contact horizon differs')
    bonus = terms['accepted_spline_strike']
    if (bonus['impl'] != 'src.tasks.hammer.mdp.split_impulse.SplitImpulseZeroBonus'
            or set(bonus['params']) != {'robot_cfg','width_m'} or bonus['params'].get('width_m') != .01
            or terms['delivered_impulse']['params'].get('early_fraction') != marker['early_fraction']
            or terms['delivered_impulse']['impl'] != 'src.tasks.hammer.mdp.split_impulse.SplitImpulseDeliveredImpulse'):
        raise ValueError('split impulse checkpoint event recipe differs')
    return marker


def install_model_mask(actor,critic,metadata):
    if validate_metadata(metadata) is None:return
    if critic.obs_normalizer.mask_lookahead:raise ValueError('split impulse critic must retain lookahead')
    actor.obs_normalizer.mask_lookahead = True


def normalized_initial(value):
    """Validate then canonicalize only the approved reward differences to S04.

    Every model/normalizer/RNG/observation/physical hash remains untouched.
    Smoke and production cell bookkeeping are retained for callers to compare
    under their explicit budget rules.
    """
    result = deepcopy(value)
    marker = validate_metadata(result['contract'])
    if marker is None:raise ValueError('split impulse initialization identity missing')
    if result['cell'].get(MARKER) != marker:raise ValueError('initial cell/contract recipe differs')
    canonical = identity('S04')
    result['cell'][MARKER] = canonical
    contract = result['contract']
    contract[MARKER] = canonical
    contract['spline_exploration038'] = exploration_identity('S04')
    for name,weight in weights(canonical).items():contract['reward_profile']['terms'][name]['weight'] = weight
    contract['reward_profile']['terms']['delivered_impulse']['params']['early_fraction'] = canonical['early_fraction']
    contract['reward_profile']['credit_timing'] = contract['spline_exploration038']['credit_timing']
    return result


from src.tasks.hammer.rl.demo_progress_campaign import validate_runner_budget


def verify_zero_ledger(path):
    from src.tasks.hammer.rl.demo_progress_campaign import verify_zero_ledger as historical
    count = historical(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        for name in ('accepted_spline_strike','completion'):
            if row['reward_terms'].get(name) != dict(mean=0.,sum=0.,min=0.,max=0.):
                raise ValueError('disabled split impulse reward paid credit: '+name)
        diagnostic = row.get('split_impulse')
        if not isinstance(diagnostic,dict):raise ValueError('split impulse rollout accounting missing')
    return count


def validate_event_layout(env,marker):
    """Validate the exact historical200ms ledger and critic47 tail."""
    from src.tasks.hammer.mdp.split_impulse import event_observation
    from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
    return _validate_event_layout(env,marker,horizon_window_complete,{"window_s":.2},event_observation)


def _validate_event_layout(env,marker,terminal_func,terminal_params,reader):
    """Shared mechanical ledger checks; callers bind closed profile semantics."""
    from src.tasks.hammer.mdp.split_impulse import SplitImpulseZeroBonus, event_observation
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
    from src.tasks.hammer.rl.exploration038_contract import _raw_term
    from src.tasks.hammer.mdp.split_impulse import SplitImpulseDeliveredImpulse, accounting_metric
    owner = getattr(env,'_split_impulse',None)
    cfg = env.reward_manager.get_term_cfg('delivered_impulse')
    declared = env.cfg.rewards['delivered_impulse'].params
    bonus = env.reward_manager.get_term_cfg('accepted_spline_strike')
    accounting = env.metrics_manager.cfg.get('split_impulse_accounting')
    _validate_impulse_params(cfg.params,marker)
    if (type(owner) is not SplitImpulseDeliveredImpulse or cfg.func is not owner
            or cfg.weight != 1. or owner._env is not env
            or owner.early_fraction != marker['early_fraction']
            or cfg.params.get('early_fraction') != marker['early_fraction']
            or set(cfg.params) != set(declared)
            or type(bonus.func) is not SplitImpulseZeroBonus or bonus.weight != 1.
            or set(bonus.params) != {'robot_cfg','width_m'} or bonus.params['width_m'] != .01
            or bonus.params['robot_cfg'].name != 'robot'
            or tuple(bonus.params['robot_cfg'].site_names) != ('hammer_head_site',)
            or bonus.params['robot_cfg'].site_ids != env._hammer_first_strike._site_ids
            or accounting is None or accounting.func is not accounting_metric
            or accounting.per_substep is not False or accounting.params != {}
            or list(env.metrics_manager.cfg).index('split_impulse_accounting') <= list(env.metrics_manager.cfg).index('cat_soft')
            or hasattr(env,'_spline_split_credit') or hasattr(env,'_spline_deadline_credit')
            or hasattr(env,'_reward_grid_event')):
        raise ValueError('split impulse live ledger or accounting differs')
    terminal = env.termination_manager.get_term_cfg('nail_driven')
    if (terminal.func is not terminal_func or terminal.params != terminal_params
            or terminal.time_out is not False):raise ValueError('split impulse contact horizon differs')
    manager = env.observation_manager
    actor,critic = tuple(manager.active_terms['actor']),tuple(manager.active_terms['critic'])
    if (set(actor)&set(CREDIT_EVENT_FIELDS) or critic != (*actor,*CREDIT_EVENT_FIELDS)
            or tuple(manager.group_obs_dim['actor']) != (43,) or tuple(manager.group_obs_dim['critic']) != (47,)
            or env.cfg.observations['critic'].enable_corruption is not False
            or tuple(manager.group_obs_term_dim['critic']) != (*manager.group_obs_term_dim['actor'],*((1,),)*4)):
        raise ValueError('split impulse requires uncorrupted actor43/critic47 layout')
    for field in CREDIT_EVENT_FIELDS:
        if not _raw_term(manager.get_term_cfg('critic',field),reader,{'field':field}):
            raise ValueError('split impulse raw critic context differs')
    tracker = env._hammer_first_strike
    if tracker._finalize_on_depth is not False or tracker._progress_eps != .0005:
        raise ValueError('split impulse live full50ms/productive gate differs')


def verify_initialization(actual_path, reference_path):
    """Reject any unqualified initialization drift before the first action."""
    actual = json.loads(actual_path.read_text())
    expected = json.loads(reference_path.read_text())
    if (actual['cell']['updates'] != 200 or actual['cell']['scope'] != 'split_impulse'
            or expected['cell']['updates'] != 1 or expected['cell']['scope'] != 'smoke'
            or actual['cell']['worlds'] != 1024 or expected['cell']['worlds'] != 1024
            or expected['cell'][MARKER]['setting_id'] != 'S04'):
        raise ValueError('split impulse requires same-seed qualified1024-world S04 smoke initialization')
    a,e = normalized_initial(actual),normalized_initial(expected)
    a['cell']['updates'],a['cell']['scope'] = 1,'smoke'
    if a != e:raise ValueError('split impulse initialization differs from qualification')
    return True
