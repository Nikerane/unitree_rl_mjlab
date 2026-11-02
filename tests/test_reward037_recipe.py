"""Reward037 configuration and checkpoint guards on CPU-only live managers."""
from copy import deepcopy
from dataclasses import asdict
import importlib
from types import SimpleNamespace

import pytest
import torch

from tests.test_split_event036 import event_env
from tests.test_spline_exploration_pilot_contract import models
from tests.test_spline_speed_variants import config_value


def api(part='contract'):
    name = ('src.tasks.hammer.config.z1.spline_reward037_cfg' if part == 'cfg'
            else 'src.tasks.hammer.rl.spline_reward037_contract')
    assert importlib.util.find_spec(name), 'reward037 implementation missing'
    return importlib.import_module(name)


def live_env(setting='vic_baseline', *, height_dr=False):
    from mjlab.managers.metrics_manager import MetricsManager
    from src.tasks.hammer.mdp.rewards import FirstStrikeDeliveredRewardTerm
    env = event_env(setting)
    cfg = api('cfg').reward_env_cfg(play=False, setting=setting,height_dr=height_dr)
    # Keep the actual 036 split/controller/constraint owners, add real 031 managers.
    for group in ('metrics', 'rewards'):
        target = getattr(env.cfg, group)
        target.update({k:v for k,v in getattr(cfg, group).items() if k.startswith('native_depth_')})
    env.cfg.rewards['delivered_impulse'] = cfg.rewards['delivered_impulse']
    env.cfg.events = cfg.events
    env.event_manager = SimpleNamespace(cfg=deepcopy(cfg.events))
    env.common_step_counter = 0
    env.scene['nail_block'] = SimpleNamespace(joint_names=['nail_slide'], num_joints=1,
        data=SimpleNamespace(joint_pos=torch.zeros(env.num_envs,1)),
        find_joints=lambda names,preserve_order=False:([0],['nail_slide']))
    depth = MetricsManager({k:v for k,v in env.cfg.metrics.items() if k.startswith('native_depth_')},env)
    env.metrics_manager.cfg.update(depth.cfg)
    from tests.test_first_strike_event import _tracker
    tracker,_ = _tracker(batch=env.num_envs,window=200,physics_dt=.00025,quality_instrumentation=False)
    tracker._env = env
    env._hammer_first_strike = tracker
    first = deepcopy(env.cfg.metrics['first_strike'])
    first.func = tracker
    env.metrics_manager.cfg['first_strike'] = first
    env.reward_manager.active_terms = list(env.cfg.rewards)
    term = deepcopy(cfg.rewards['delivered_impulse'])
    term.func = FirstStrikeDeliveredRewardTerm(term,env)
    old = env.reward_manager.get_term_cfg
    env.reward_manager.get_term_cfg = lambda n: term if n == 'delivered_impulse' else old(n)
    return env


@pytest.mark.parametrize('setting,std', [('vic_baseline',.3), ('vic_position_low',.1)])
def test_recipe_adds_exact_historical_rewards_to_036_without_changing_plant(setting,std):
    from src.tasks.hammer.config.z1.spline_event_study036_cfg import event_env_cfg
    from src.tasks.hammer.config.z1.spline_depth_reward031_cfg import depth_env_cfg
    cfg = api('cfg').reward_env_cfg(play=False, setting=setting)
    baseline = event_env_cfg(play=False,payout='split',setting=setting)
    depth = depth_env_cfg(play=False,arm='A3')
    assert cfg.rewards['delivered_impulse'].weight == 4.
    assert cfg.rewards['delivered_impulse'].func is baseline.rewards['delivered_impulse'].func
    assert cfg.rewards['delivered_impulse'].params == baseline.rewards['delivered_impulse'].params
    assert cfg.rewards['delivered_impulse'].params['i_ref'] == 0.2799950838088989
    assert cfg.rewards['delivered_impulse'].params['saturate'] is False
    for name in ('native_depth_progress031','native_depth_completion031'):
        assert cfg.rewards[name] == depth.rewards[name]
    actual, expected = asdict(cfg), asdict(baseline)
    actual['rewards'].pop('native_depth_progress031')
    actual['rewards'].pop('native_depth_completion031')
    actual['rewards']['delivered_impulse']['weight'] = 0.
    actual['metrics'].pop('native_depth_owner031')
    actual['metrics'].pop('native_depth_collect031')
    assert config_value(actual) == config_value(expected)
    marker = api().reward_identity(setting)
    assert marker['initial_exploration_std'] == [std]*6+[.3]*6+[1.]
    assert marker['added_rewards'] == dict(delivered_impulse=True,
        native_depth_progress031=True,native_depth_completion031=True)
    assert marker['depth'].get('owner') == 'src.tasks.hammer.config.z1.spline_reward037_cfg:RewardDepthObserver037'


@pytest.mark.parametrize('setting', ['vic_baseline','vic_position_low'])
def test_live_identity_dispatches_only_037_and_preserves_event_observability(setting):
    from src.tasks.hammer.rl.impact_speed_contract import (
        _observation_metadata, _spline_reward_profile,
        validate_impact_speed_metadata, validate_impact_speed_models,
    )
    env = live_env(setting)
    marker = api().reward_metadata(env)
    assert marker == api().reward_identity(setting)
    assert marker['impulse']['weight'] == 4. and marker['impulse']['window_ms'] == 50
    assert marker['depth']['progress']['weight'] == marker['depth']['completion']['weight'] == 1.
    assert marker['critic_width'] == 47 and marker['actor_width'] == 43
    metadata = {'schema_version':'impact_speed_v1','spline_reward037':marker}
    a,c = models(setting,'event')
    a.distribution.std_param.data.fill_(.7)  # Learned std must remain admissible.
    validate_impact_speed_models(a,c,metadata)
    validate_impact_speed_metadata(deepcopy(metadata),metadata)
    assert _observation_metadata(env)['observation_widths'] == {'actor':43,'critic':47}
    profile = _spline_reward_profile(env)
    assert profile['terms']['delivered_impulse']['weight'] == 4.
    assert profile['credit_timing']['reserve_fraction'] == .5
    assert api().reward_metadata(event_env(setting)) is None
    from src.tasks.hammer.rl.spline_split_event036_contract import split_event_metadata
    with pytest.raises(ValueError): split_event_metadata(env)


@pytest.mark.parametrize('fault', ['impulse_weight','impulse_saturation','configured_impulse',
    'depth_weight','depth_flags','depth_owner','depth_collector','extra_reward',
    'constraint','event_scale','horizon','first_window_config','first_window_live',
    'declared_depth_setting','declared_depth_owner'])
def test_live_contract_rejects_reward_owner_and_shared_safety_drift(fault):
    env = live_env()
    if fault == 'impulse_weight': env.reward_manager.get_term_cfg('delivered_impulse').weight = 0.
    elif fault == 'impulse_saturation': env.reward_manager.get_term_cfg('delivered_impulse').params['saturate'] = True
    elif fault == 'configured_impulse': env.cfg.rewards['delivered_impulse'].weight = 0.
    elif fault == 'depth_weight': env.cfg.rewards['native_depth_completion031'].weight = 0.
    elif fault == 'depth_flags': env._z1_depth_credit031.ledger.completion_enabled = False
    elif fault == 'depth_owner': env._z1_depth_credit031 = object()
    elif fault == 'depth_collector': env.metrics_manager.cfg['native_depth_collect031'].per_substep = True
    elif fault == 'extra_reward': env.cfg.rewards['unexpected'] = env.cfg.rewards['spline_progress']
    elif fault == 'constraint': env.metrics_manager.cfg['cat_soft'].func._max_p = .2
    elif fault == 'event_scale': env.cfg.observations['critic'].terms['credit_unpaid'].scale = 2.
    elif fault == 'first_window_config': env.cfg.metrics['first_strike'].params['window_substeps'] = 25
    elif fault == 'first_window_live': env._hammer_first_strike._window = 25
    elif fault == 'declared_depth_setting': env.cfg.metrics['native_depth_owner031'].params['reward_setting'] = 'vic_position_low'
    elif fault == 'declared_depth_owner': env.cfg.metrics['native_depth_owner031'].func = object
    else: env.termination_manager.get_term_cfg('nail_driven').params['window_s'] = .05
    with pytest.raises(ValueError): api().reward_metadata(env)


@pytest.mark.parametrize('fault', ['old_marker','wrong_setting','wrong_critic','frozen_std','reward_coefficient'])
def test_checkpoint_rejects_old_mixed_or_mismatched_recipe(fault):
    from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models,validate_impact_speed_metadata
    metadata = {'schema_version':'impact_speed_v1','spline_reward037':api().reward_identity('vic_baseline')}
    a,c = models(mode='event')
    if fault == 'old_marker': metadata['spline_split_event036'] = {}
    elif fault == 'wrong_setting': a,_ = models('vic_position_low','event')
    elif fault == 'wrong_critic': _,c = models()
    elif fault == 'frozen_std': a.distribution.std_param.requires_grad_(False)
    else: metadata['spline_reward037']['impulse']['weight'] = 1.
    with pytest.raises(ValueError): validate_impact_speed_models(a,c,metadata)
    if fault in ('old_marker','reward_coefficient'):
        with pytest.raises(ValueError): validate_impact_speed_metadata(metadata,metadata)


def test_registration_keeps_existing_runner_and_ordinary_event_models():
    m = api('cfg')
    from mjlab.tasks.registry import list_tasks,load_env_cfg,load_rl_cfg,load_runner_cls
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    m.register_reward037_tasks()
    before = set(list_tasks())
    m.register_reward037_tasks()
    assert set(list_tasks()) == before
    for setting in ('vic_baseline','vic_position_low'):
        task = m.reward_task_id(setting)
        assert task in before and load_runner_cls(task) is HammerOnPolicyRunner
        cfg = load_rl_cfg(task)
        assert cfg.actor.distribution_cfg['class_name'].endswith(':PilotGaussian')
        assert cfg.critic.class_name.endswith(':CreditEventCritic')
        assert load_env_cfg(task).rewards['delivered_impulse'].weight == 4.
    with pytest.raises(ValueError): m.reward_task_id('fixed_nominal')


@pytest.mark.parametrize('setting', ['vic_baseline','vic_position_low'])
def test_height_dr_uses_shared_native_reset_translation_with_distinct_identity(setting):
    from mjlab.envs import mdp as envs_mdp
    cfgmod,contract = api('cfg'),api()
    base = cfgmod.reward_env_cfg(play=False,setting=setting)
    dr = cfgmod.reward_env_cfg(play=False,setting=setting,height_dr=True)
    event = dr.events['randomize_board_nail_height037']
    assert event.func is envs_mdp.dr.body_pos and event.mode == 'reset'
    assert event.min_step_count_between_reset == 0
    asset = event.params['asset_cfg']
    assert asset.name == 'nail_block' and tuple(asset.body_names) == ('block','nail')
    assert {k:v for k,v in event.params.items() if k != 'asset_cfg'} == dict(
        ranges=(-.005,.005),axes=[2],shared_random=True,operation='add',distribution='uniform')
    assert dr.metrics['native_depth_owner031'].params['height_dr'] is True
    actual,expected = asdict(dr),asdict(base)
    actual['events'].pop('randomize_board_nail_height037')
    actual['metrics']['native_depth_owner031']['params'].pop('height_dr')
    assert config_value(actual) == config_value(expected)
    assert 'height_dr' not in base.metrics['native_depth_owner031'].params
    assert cfgmod.reward_task_id(setting,height_dr=True) != cfgmod.reward_task_id(setting)
    marker = contract.reward_identity(setting,height_dr=True)
    assert marker['schema_version'] == 'spline_reward037_height_dr_v1'
    assert marker['training_height']['offset_range_m'] == [-.005,.005]
    assert 'training_height' not in contract.reward_identity(setting)
    assert contract.reward_metadata(live_env(setting,height_dr=True)) == marker
    contract.validate_reward_models(*models(setting,'event'),{'spline_reward037':marker})


@pytest.mark.parametrize('fault', ['range','independent','nail_only','startup','live_range','owner_flag','base_extra'])
def test_height_dr_contract_rejects_changed_range_geometry_or_reset_treatment(fault):
    env = live_env(height_dr=fault != 'base_extra')
    event = env.cfg.events.get('randomize_board_nail_height037')
    if fault == 'range': event.params['ranges'] = (-.01,.01)
    elif fault == 'independent': event.params['shared_random'] = False
    elif fault == 'nail_only': event.params['asset_cfg'].body_names = ('nail',)
    elif fault == 'startup': event.mode = 'startup'
    elif fault == 'live_range': env.event_manager.cfg['randomize_board_nail_height037'].params['ranges'] = (-.01,.01)
    elif fault == 'owner_flag': env._z1_depth_credit031.height_dr = False
    else:
        dr = api('cfg').reward_env_cfg(play=False,setting='vic_baseline',height_dr=True)
        env.cfg.events['randomize_board_nail_height037'] = dr.events['randomize_board_nail_height037']
    with pytest.raises(ValueError): api().reward_metadata(env)


@pytest.fixture(autouse=True)
def forbid_environment_or_checkpoint(monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    def forbidden(*a,**k):
        raise AssertionError('environment construction/checkpoint loading forbidden in CPU037 tests')
    monkeypatch.setattr(ManagerBasedRlEnv,'__init__',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
