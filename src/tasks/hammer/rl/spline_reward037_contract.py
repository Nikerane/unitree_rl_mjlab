"""Closed reward037 checkpoint identity with unchanged split/event model semantics."""
from src.tasks.hammer.config.z1 import _FIC_CONTROLLED_DROP_I_REF_N_S as I_REF
from src.tasks.hammer.mdp.depth_credit031 import DepthCreditLedger031
from src.tasks.hammer.mdp.depth_reward031 import (
    depth_collect_metric031, depth_progress_reward031, depth_completion_reward031,
)
from src.tasks.hammer.mdp.first_strike import FirstStrikeEventTracker
from src.tasks.hammer.mdp.rewards import FirstStrikeDeliveredRewardTerm
from src.tasks.hammer.rl.depth_reward031_contract import depth_identity031
from src.tasks.hammer.rl.spline_split_event036_contract import (
    EVENT_REWARD_WEIGHTS, split_event_identity, split_event_layout,
    validate_event_models, validate_native_constraints, validate_reward_weights,
)
from src.tasks.hammer.rl.spline_split_credit_contract import _need, _same

_OLD_MARKERS = ('spline_split_credit','spline_split_event036','spline_exploration_pilot',
    'credit_critic_experiment','spline_learning_variant','spline_depth_reward031',
    'spline_reward_exploration032')


def reward_identity(setting, *, height_dr=False):
    _need(type(height_dr) is bool, 'reward037 height_dr must be a bool')
    marker = split_event_identity(setting)
    depth = depth_identity031('A3')
    marker.update(schema_version='spline_reward037_v1', family='reward037',
        added_rewards=dict(delivered_impulse=True,native_depth_progress031=True,
                           native_depth_completion031=True),
        impulse=dict(weight=4.,i_ref=I_REF,saturate=False,
            reader='FirstStrikeDeliveredRewardTerm',units='legacy_normalized_before_manager_dt',
            window_ms=50,payment='productive_first_window_once'),
        depth={key:depth[key] for key in ('measurement','progress','completion','collection','reset')},
        exploration=dict(law='learned_state_independent_Gaussian',std_type='scalar',
            position_indices=list(range(6)),state_dependent_std=False,gsde=False))
    marker['depth']['observability'] = 'actor43_critic47_hidden_native_depth_peak_and_earned_credit_after_rebound'
    marker['depth']['owner'] = 'src.tasks.hammer.config.z1.spline_reward037_cfg:RewardDepthObserver037'
    if height_dr:
        marker.update(schema_version='spline_reward037_height_dr_v1',family='reward037_height_dr',
            training_height=dict(entity='nail_block',body_names=['block','nail'],
                transform='shared_world_z_translation_of_board_and_nail',
                offset_range_m=[-.005,.005],distribution='uniform',operation='add',
                shared_random=True,axes=[2],cadence='each_reset_selected_worlds',
                baseline='default_model_body_positions_no_accumulation',
                protrusion='unchanged',native_nail_slide_depth='unchanged'))
    return marker


def validate_reward_identity(value):
    _need(type(value) is dict, 'reward037 marker missing')
    expected = reward_identity(value.get('setting'),
        height_dr=value.get('schema_version') == 'spline_reward037_height_dr_v1')
    _need(_same(value,expected), 'reward037 marker differs from exact recipe')
    return expected


def validate_reward_metadata(value):
    _need(type(value) is dict, 'reward037 requires full checkpoint metadata')
    marker = validate_reward_identity(value.get('spline_reward037'))
    _need(not any(key in value for key in _OLD_MARKERS), 'reward037 cannot claim old or mixed recipe markers')
    return marker


def _validate_training_height(env, height_dr):
    from mjlab.envs import mdp as envs_mdp
    name = 'randomize_board_nail_height037'
    live = getattr(getattr(env,'event_manager',None),'cfg',None)
    _need(not height_dr or type(live) is dict, 'reward037 height DR requires live reset event config')
    for configs in (env.cfg.events,live or {}):
        event = configs.get(name)
        if not height_dr:
            _need(event is None, 'base reward037 must not carry height DR')
            continue
        _need(event is not None and event.func is envs_mdp.dr.body_pos and event.mode == 'reset'
            and event.interval_range_s is None and event.is_global_time is False
            and event.min_step_count_between_reset == 0
            and set(event.params) == {'asset_cfg','ranges','axes','shared_random','operation','distribution'},
            'reward037 native height reset event differs')
        asset = event.params['asset_cfg']
        _need(asset.name == 'nail_block' and tuple(asset.body_names) == ('block','nail')
            and _same({k:v for k,v in event.params.items() if k != 'asset_cfg'},dict(
                ranges=[-.005,.005],axes=[2],shared_random=True,operation='add',distribution='uniform')),
            'reward037 height range or shared board/nail transform differs')


def reward_metadata(env):
    from src.tasks.hammer.config.z1.spline_reward037_cfg import RewardDepthObserver037
    owner = getattr(env,'_z1_depth_credit031',None)
    metric_cfg = getattr(getattr(env,'metrics_manager',None),'cfg',{})
    declared = getattr(env.cfg,'metrics',{}).get('native_depth_owner031')
    configured = declared is not None and (declared.func is RewardDepthObserver037
        or type(declared.func) is RewardDepthObserver037)
    if type(owner) is not RewardDepthObserver037 and not configured:
        return None
    _need(type(owner) is RewardDepthObserver037 and owner._env is env,
          'reward037 requires exact live native depth observer')
    _need(type(owner.height_dr) is bool, 'reward037 live height DR flag differs')
    height_dr = owner.height_dr
    owner_keys = {'arm','nail_cfg','reward_setting'} | ({'height_dr'} if height_dr else set())
    _need(declared is not None and declared.func is RewardDepthObserver037
        and declared.per_substep is True and declared.reduce == 'last'
        and set(declared.params) == owner_keys
        and declared.params.get('height_dr',False) is height_dr
        and declared.params['arm'] == 'A3' and declared.params['reward_setting'] == owner.reward_setting
        and declared.params['nail_cfg'].name == 'nail_block'
        and tuple(declared.params['nail_cfg'].joint_names) == ('nail_slide',),
        'reward037 configured depth observer differs')
    _validate_training_height(env,height_dr)
    marker = reward_identity(owner.reward_setting,height_dr=height_dr)
    _need(split_event_layout(env) == (owner.reward_setting,.5), 'reward037 split/event layout differs')
    validate_native_constraints(env)
    weights = dict(EVENT_REWARD_WEIGHTS,delivered_impulse=4.,
        native_depth_progress031=1.,native_depth_completion031=1.)
    validate_reward_weights(env,weights)
    _need(env.physics_dt == .00025 and env.step_dt == .02, 'reward037 clocks differ')
    _need(owner.arm == 'A3' and type(owner.ledger) is DepthCreditLedger031
        and owner.ledger.num_envs == env.num_envs and owner.ledger.progress_enabled is True
        and owner.ledger.completion_enabled is True, 'reward037 requires both unchanged depth ledger rewards')
    registered = metric_cfg.get('native_depth_owner031')
    _need(registered is not None and registered.func is owner and registered.per_substep is True
        and registered.reduce == 'last' and set(registered.params) == owner_keys
        and registered.params.get('height_dr',False) is height_dr
        and registered.params['arm'] == 'A3' and registered.params['reward_setting'] == owner.reward_setting,
        'reward037 depth observer registration differs')
    nail = registered.params['nail_cfg']
    _need(nail is owner.nail_cfg and nail.name == 'nail_block'
        and tuple(nail.joint_names) == ('nail_slide',), 'reward037 native depth coordinate differs')
    for configs in (env.cfg.metrics,metric_cfg):
        collect = configs.get('native_depth_collect031')
        _need(collect is not None and collect.func is depth_collect_metric031
            and collect.params == {} and collect.per_substep is False and collect.reduce == 'last',
            'reward037 depth collector differs')
        first = configs.get('first_strike')
        _need(first is not None and first.per_substep is True
            and first.params['window_substeps'] == 200, 'reward037 first50ms window differs')
    tracker = getattr(env,'_hammer_first_strike',None)
    _need(type(tracker) is FirstStrikeEventTracker and tracker._env is env and tracker._window == 200
        and metric_cfg['first_strike'].func is tracker, 'reward037 live first50ms tracker differs')
    for name,func in (('native_depth_progress031',depth_progress_reward031),
                      ('native_depth_completion031',depth_completion_reward031)):
        for term in (env.cfg.rewards[name],env.reward_manager.get_term_cfg(name)):
            _need(term.func is func and term.params == {}, 'reward037 depth reward reader differs')
    for term,live in ((env.cfg.rewards['delivered_impulse'],False),
                      (env.reward_manager.get_term_cfg('delivered_impulse'),True)):
        _need((type(term.func) is FirstStrikeDeliveredRewardTerm if live else term.func is FirstStrikeDeliveredRewardTerm)
            and term.params['i_ref'] == I_REF and term.params['saturate'] is False,
            'reward037 requires historical unsaturated normalized impulse')
    return marker


def validate_reward_models(actor,critic,metadata):
    marker = validate_reward_metadata(metadata)
    validate_event_models(actor,critic,marker['setting'])
