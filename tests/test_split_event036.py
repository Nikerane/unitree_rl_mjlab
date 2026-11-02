"""CPU-only split-event036 tensors, configs and live-manager contracts."""
from copy import deepcopy
from dataclasses import asdict
import importlib
from types import SimpleNamespace

import pytest
import torch

from tests.test_spline_split_credit021 import setup
from tests.test_spline_boundary_credit import accepted_onset
from tests.test_spline_exploration_pilot_contract import pilot_env, models
from tests.test_spline_speed_variants import config_value


def api(part):
    names = {
        'mdp': 'src.tasks.hammer.mdp.spline_split_event036',
        'cfg': 'src.tasks.hammer.config.z1.spline_event_study036_cfg',
        'contract': 'src.tasks.hammer.rl.spline_split_event036_contract',
    }
    assert importlib.util.find_spec(names[part]), 'split event036 implementation missing'
    return importlib.import_module(names[part])


def observe(env):
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    return torch.cat([api('mdp').split_event_observation(env, field=f) for f in CREDIT_EVENT_FIELDS], -1)


@pytest.mark.parametrize('alpha', [.5, 1.])
def test_reader_reports_next_pending_tranche_without_recomputing_or_consuming(alpha):
    env, tracker, _, _, params, owner = setup(alpha, batch=3)
    accepted_onset(tracker)
    # Contact at ticks 1, 40, 80; next policy boundary is tick80.
    tracker.first_contact_time_s[:] = torch.tensor([1, 40, 80]) * .00025
    tracker._substep_count[:] = 80
    owner(env, **params)
    before = deepcopy(owner.last_snapshot)
    call = owner.call_index
    out = observe(env)
    torch.testing.assert_close(out, torch.tensor([
        [1., 79/800, 1., .04/.2], [1., 40/800, 1., .04/.2], [1., 0., 1., .06/.2],
    ]))
    assert owner.call_index == call
    assert all(torch.equal(before[k], owner.last_snapshot[k]) for k in before)
    tracker._substep_count[:] = 320
    owner(env, **params)
    out = observe(env)
    if alpha == .5:
        torch.testing.assert_close(out[:, 2:], torch.tensor([[.5,.14/.2],[.5,.14/.2],[.5,.14/.2]]))
    else:
        assert out[:, 2:].eq(0).all()
    tracker._substep_count[:] = 880
    owner(env, **params)
    assert observe(env)[:, 2:].eq(0).all()


@pytest.mark.parametrize('dt', [.00025, .002])
def test_all_onset_phases_round_each_next_payout_up_to_a_policy_boundary(dt):
    period = round(.02 / dt)
    env, tracker, _, _, params, owner = setup(.5, dt=dt, batch=period)
    accepted_onset(tracker)
    onset = torch.arange(1, period+1)
    tracker.first_contact_time_s[:] = onset.float()*dt
    tracker._substep_count[:] = period
    owner(env, **params)
    early = ((onset+round(.05/dt)+period-1)//period)*period
    reserve = ((onset+round(.2/dt)+period-1)//period)*period
    for step in range(1, 14):
        tracker._substep_count[:] = step*period
        if step > 1:
            owner(env, **params)
        expected = torch.where(early > step*period, early, reserve)
        expected = torch.where(reserve > step*period, expected-step*period, 0).float()*dt/.2
        torch.testing.assert_close(observe(env)[:,3], expected)


def test_no_contact_rejected_contact_native_flush_and_subset_reset():
    env, tracker, _, _, params, owner = setup(.5, batch=3)
    assert observe(env).eq(0).all()
    accepted_onset(tracker)
    tracker.v_precontact[1] = -1.
    tracker._substep_count[:] = 80
    owner(env, **params)
    assert observe(env)[:,2].tolist() == [1.,0.,1.]
    env.reset_buf[0] = True
    owner(env, **params)
    assert observe(env)[:,2].tolist() == [0.,0.,1.]
    owner.reset(torch.tensor([2]))
    tracker.started[2] = False
    tracker._substep_count[2] = 0
    assert observe(env)[2].eq(0).all()
    assert owner.early_credit.eq(0).all() and owner.reserve_credit.eq(0).all()


def test_probe_guard_only_permits_missing_owner_before_observation_manager_exists():
    env = SimpleNamespace(num_envs=2, device='cpu')
    assert observe(env).shape == (2,4) and observe(env).eq(0).all()
    env.observation_manager = object()
    with pytest.raises(RuntimeError):
        observe(env)
    with pytest.raises(ValueError):
        api('mdp').split_event_observation(env, field='wrong')


@pytest.mark.parametrize('fault', ['fraction', 'reserve', 'shape', 'nonfinite', 'clock', 'no_contact'])
def test_reader_fails_closed_on_invalid_balance_or_clock(fault):
    env, tracker, _, _, params, owner = setup(.5, batch=3)
    accepted_onset(tracker)
    owner(env, **params)
    if fault == 'fraction': owner.early_credit[0] = .25
    elif fault == 'reserve': owner.reserve_credit[0] = 1.
    elif fault == 'shape': owner.reserve_credit = torch.zeros(2)
    elif fault == 'nonfinite': owner.early_credit[0] = float('nan')
    elif fault == 'clock': env.step_dt = .01
    else: tracker.started[0] = False
    with pytest.raises(ValueError):
        observe(env)


RECIPES = [('split','vic_baseline'), ('200','vic_baseline'), ('split','vic_position_low'), ('50','vic_baseline')]


@pytest.mark.parametrize('payout,setting', RECIPES)
def test_four_configs_preserve_actor_controller_constraints_and_rewards(payout, setting):
    m = api('cfg')
    from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_env_cfg
    from src.tasks.hammer.config.z1.spline_split_credit_cfg import split_env_cfg
    cfg = m.event_env_cfg(play=False, payout=payout, setting=setting)
    baseline = (split_env_cfg(play=False, setting=setting) if payout == 'split' else
                pilot_env_cfg(play=False, setting=setting, credit_deadline_ms=int(payout), critic_mode='event'))
    actual, expected = asdict(cfg), asdict(baseline)
    if payout == 'split':
        for field in ('credit_contact_seen','credit_contact_age','credit_unpaid','credit_time_to_payout'):
            assert cfg.observations['critic'].terms[field].func is api('mdp').split_event_observation
        actual['observations']['critic'] = expected['observations']['critic']
    assert config_value(actual) == config_value(expected)
    assert cfg.rewards['delivered_impulse'].weight == 0
    assert cfg.rewards['nail_depth_delta'].weight == 0 and cfg.rewards['completion'].weight == 0
    assert cfg.metrics['cat_soft'].params['imp_limit'] == [.369,.246,.738,.369,.246,.0164]
    assert cfg.metrics['cat_soft'].params['max_p'] == .5
    assert cfg.terminations['nail_driven'].params == {'window_s':.2}


@pytest.mark.parametrize('payout,setting', [('200','vic_position_low'), ('50','vic_position_low'), ('100','vic_baseline'), (50,'vic_baseline'), ('split','fixed_nominal')])
def test_config_rejects_unapproved_recipe_combinations(payout, setting):
    with pytest.raises(ValueError):
        api('cfg').event_task_id(payout=payout, setting=setting)


def event_env(setting='vic_baseline'):
    from src.tasks.hammer.mdp.spline_split_credit import SplitSplineStrikeReward
    env = pilot_env(setting, 'event')
    env.cfg = api('cfg').event_env_cfg(play=False, payout='split', setting=setting)
    del env._spline_deadline_credit
    cfg = env.cfg.rewards['accepted_spline_strike']
    owner = SplitSplineStrikeReward(cfg, env)
    live = env.reward_manager.get_term_cfg('accepted_spline_strike')
    live.func, live.params = owner, deepcopy(cfg.params)
    env.reward_manager.active_terms = list(env.cfg.rewards)
    prior = env.reward_manager.get_term_cfg
    env.reward_manager.get_term_cfg = lambda n: prior(n) if n == 'accepted_spline_strike' else env.cfg.rewards[n]
    from src.tasks.hammer.rl.impact_speed_wrapper import configure_impact_speed_controller
    configure_impact_speed_controller(env.cfg,sampled_pd500=True,gain_mode='variable')
    env.metrics_manager = _native_constraint_manager(env)
    return env


@pytest.mark.parametrize('setting', ['vic_baseline','vic_position_low'])
def test_truthful_marker_accepts_event_models_and_legacy_split_stays_strict(setting):
    m = api('contract')
    env = event_env(setting)
    marker = m.split_event_metadata(env)
    assert marker['actor_width'] == 43 and marker['critic_width'] == 47
    assert marker['critic_event_fields'] == ['credit_contact_seen','credit_contact_age','credit_unpaid','credit_time_to_payout']
    assert marker['setting'] == setting and marker['credit_timing']['reserve_fraction'] == .5
    assert marker == m.validate_split_event_identity(marker)
    a,c = models(setting, 'event')
    m.validate_split_event_models(a,c,{'spline_split_event036':marker})
    from src.tasks.hammer.rl.spline_split_credit_contract import split_metadata
    with pytest.raises(ValueError):
        split_metadata(env)
    from tests.test_split_credit_contract021 import convert_env
    old = convert_env(pilot_env(setting), setting)
    assert m.split_event_metadata(old) is None
    assert split_metadata(old)['critic_width'] == 43
    assert m.split_event_metadata(pilot_env(setting,'event')) is None


@pytest.mark.parametrize('fault', ['owner','params','field_func','field_scale','field_width','order','actor_extra','critic_width','corruption','horizon','gains','age','delivered','depth','completion','cat_config','cat_live','impulse_limit','live_delivered','impulse_window_cfg','impulse_window_livecfg','impulse_window_live','new_reward'])
def test_live_split_event_metadata_rejects_observation_reward_controller_constraint_drift(fault):
    env = event_env()
    if fault == 'owner': env._spline_split_credit = object()
    elif fault == 'params': env.reward_manager.get_term_cfg('accepted_spline_strike').params['early_fraction'] = 1.
    elif fault == 'field_func': env.cfg.observations['critic'].terms['credit_unpaid'].func = lambda e: None
    elif fault == 'field_scale': env.cfg.observations['critic'].terms['credit_unpaid'].scale = 2.
    elif fault == 'field_width': env.observation_manager.group_obs_term_dim['critic'][-1] = (2,)
    elif fault == 'order': env.observation_manager.active_terms['critic'][-2:] = reversed(env.observation_manager.active_terms['critic'][-2:])
    elif fault == 'actor_extra': env.observation_manager.active_terms['actor'].append('credit_unpaid')
    elif fault == 'critic_width': env.observation_manager.group_obs_dim['critic'] = (43,)
    elif fault == 'corruption': env.cfg.observations['critic'].enable_corruption = True
    elif fault == 'horizon': env.termination_manager.get_term_cfg('nail_driven').params['window_s'] = .05
    elif fault == 'gains': env._sampled_pd500.arm_kp[0] += 1
    elif fault == 'age': env._sampled_pd500.startup_age[0] = 2
    elif fault == 'delivered': env.cfg.rewards['delivered_impulse'].weight = 4.
    elif fault == 'depth': env.cfg.rewards['nail_depth_delta'].weight = 600.
    elif fault == 'completion': env.cfg.rewards['completion'].weight = 1.
    elif fault == 'cat_config': env.cfg.metrics['cat_soft'].params['use_vel'] = False
    elif fault == 'cat_live': env.metrics_manager.cfg['cat_soft'].func._max_p = 0.
    elif fault == 'impulse_limit': env.metrics_manager.cfg['cat_soft'].func._imp_limit[3] = 100.
    elif fault == 'impulse_window_cfg': env.cfg.metrics['substep_impulse'].params['event_window_substeps'] = 25
    elif fault == 'impulse_window_livecfg': env.metrics_manager.cfg['substep_impulse'].params['event_window_substeps'] = 25
    elif fault == 'impulse_window_live': env.metrics_manager.cfg['substep_impulse'].func._window = 25
    elif fault == 'live_delivered':
        original = env.reward_manager.get_term_cfg
        delivered = deepcopy(original('delivered_impulse')); delivered.weight = 4.
        env.reward_manager.get_term_cfg = lambda n: delivered if n == 'delivered_impulse' else original(n)
    else:
        env.cfg.rewards['native_depth_progress031'] = deepcopy(env.cfg.rewards['spline_progress'])
        env.reward_manager.active_terms.append('native_depth_progress031')
    with pytest.raises(ValueError):
        api('contract').split_event_metadata(env)


@pytest.mark.parametrize('fault', ['oldcritic','wrongactor','mixed','extra','std_frozen','entropy','dtype','nonfinite','normalizer'])
def test_model_identity_rejects_masquerading_as_current_or_mixed_recipe(fault):
    m = api('contract')
    marker = m.split_event_metadata(event_env())
    a,c = models(mode='event')
    metadata = {'spline_split_event036': marker}
    if fault == 'oldcritic': _,c = models()
    elif fault == 'wrongactor': a,_ = models('fixed_nominal','event')
    elif fault == 'mixed': metadata['spline_split_credit'] = {}
    elif fault == 'extra': marker['extra'] = True
    elif fault == 'std_frozen': a.distribution.std_param.requires_grad_(False)
    elif fault == 'entropy': a.distribution.gain_entropy_weight = 1/6
    elif fault == 'dtype': c.double()
    elif fault == 'nonfinite': next(c.parameters()).data.reshape(-1)[0] = float('nan')
    else: c.obs_normalizer.physical.eps = .1
    with pytest.raises(ValueError):
        m.validate_split_event_models(a,c,metadata)


def test_actual_reward_manager_keeps_zero_weight_terms_in_active_schema():
    from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
    env = SimpleNamespace(num_envs=3,device='cpu')
    def forbidden(source):
        raise AssertionError('zero-weight reward must not be evaluated')
    def one(source):
        return torch.ones(source.num_envs)
    manager = RewardManager({'off':RewardTermCfg(func=forbidden,weight=0.),
                             'on':RewardTermCfg(func=one,weight=1.)},env)
    assert manager.active_terms == ['off','on']
    torch.testing.assert_close(manager.compute(.02),torch.full((3,),.02))
    live = event_env()
    live.reward_manager.active_terms = list(live.cfg.rewards)
    assert api('contract').split_event_metadata(live)['critic_width'] == 47


def test_fresh_event_identity_does_not_mutate_historical_or_sibling_marker():
    from scripts.z1_timing_noise020.split_contract import split_identity,credit_timing
    old = split_identity('vic_baseline'); timing = credit_timing(.5)
    before = deepcopy((old,timing))
    marker = api('contract').split_event_identity('vic_baseline')
    sibling = api('contract').split_event_identity('vic_baseline')
    marker['initial_exploration_std'][0] = .9
    marker['credit_timing']['early_fraction'] = .9
    assert (old,timing) == before
    assert sibling['initial_exploration_std'][0] == .3
    assert sibling['credit_timing']['early_fraction'] == .5
    assert split_identity('vic_baseline') == before[0] and credit_timing(.5) == before[1]


@pytest.mark.parametrize('reset_kind',['native_timeout','external_reset'])
def test_real_manager_subset_reset_clears_owner_and_tracker_before_next_observation(reset_kind):
    from mjlab.managers.reward_manager import RewardManager,RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsManager,MetricsTermCfg
    from mjlab.managers.observation_manager import ObservationManager,ObservationGroupCfg,ObservationTermCfg
    from tests.test_first_strike_event import _tracker
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    env,_,_,_,params,owner = setup(.5,batch=3)
    tracker, sample = _tracker(batch=3,physics_dt=.00025,quality_instrumentation=False)
    sample(head_z=.104,depth=0.,contact_on=False)
    sample(head_z=.102,depth=0.,contact_on=False)
    sample(head_z=.100,depth=0.,contact_on=True)
    env._hammer_first_strike = tracker
    env.max_episode_length_s = 20.
    # Existing live term instances avoid all scene/simulator construction.
    env.reward_manager = RewardManager({'accepted_spline_strike':RewardTermCfg(func=owner,weight=1.)},env)
    env.reward_manager.get_term_cfg('accepted_spline_strike').params = params
    env.metrics_manager = MetricsManager({'first_strike':MetricsTermCfg(func=tracker,per_substep=True,reduce='last')},env)
    # Managers deep-copy their configurations; expose the actual live instances.
    owner = env.reward_manager.get_term_cfg('accepted_spline_strike').func
    tracker = env.metrics_manager.cfg['first_strike'].func
    owner._env = tracker._env = env
    env._spline_split_credit, env._hammer_first_strike = owner, tracker
    env.observation_manager = ObservationManager({'critic':ObservationGroupCfg(terms={
        f:ObservationTermCfg(func=api('mdp').split_event_observation,params={'field':f})
        for f in CREDIT_EVENT_FIELDS},enable_corruption=False)},env)
    tracker._substep_count[:] = 80
    assert env.reward_manager.compute(.02).eq(0).all()
    torch.testing.assert_close(observe(env)[:,2],torch.ones(3))
    if reset_kind == 'native_timeout':
        env.reset_buf[0] = env.reset_time_outs[0] = True
        torch.testing.assert_close(env.reward_manager.compute(.02),torch.tensor([1.,0.,0.]))
    calls = owner.call_index
    # Actual _reset_idx order is observation reset, reward reset, metrics reset;
    # step computes the next observation only after all these resets complete.
    ids = torch.tensor([0])
    env.observation_manager.reset(ids)
    env.reward_manager.reset(ids)
    env.metrics_manager.reset(ids)
    out = env.observation_manager.compute()['critic']
    assert out[0].eq(0).all()
    torch.testing.assert_close(out[1:],torch.tensor([[1.,77/800,1.,.2],[1.,77/800,1.,.2]]))
    assert owner.call_index == calls
    assert owner.early_credit.tolist() == [0.,.5,.5] and owner.reserve_credit.tolist() == [0.,.5,.5]
    assert not tracker.started[0] and tracker.first_contact_time_s[0] == 0 and tracker._substep_count[0] == 0
    assert not owner._consumed[0]
    marker = api('contract').split_event_identity('vic_baseline')
    assert marker['critic_event_semantics']['read_order'] == 'after_reward_and_optional_auto_reset_no_reward_recompute'


def _native_constraint_manager(env):
    from mjlab.managers.metrics_manager import MetricsManager
    names = ['joint'+str(i) for i in range(1,7)]
    def find_joints(requested,preserve_order=False):
        assert list(requested) == names
        return list(range(6)),names.copy()
    robot = SimpleNamespace(joint_names=names,num_joints=6,find_joints=find_joints,
        data=SimpleNamespace(joint_pos=torch.zeros(env.num_envs,6),joint_vel=torch.zeros(env.num_envs,6),
                             _joint_dof_field=lambda field:torch.zeros(env.num_envs,6)))
    env.scene = {'robot':robot,'hammer_nail_contact':SimpleNamespace(data=SimpleNamespace(found=torch.zeros(env.num_envs,1)))}
    # Construct the actual metric classes and resolve their real SceneEntityCfgs.
    # No scene/simulator/physical environment is constructed or stepped.
    return MetricsManager({name:env.cfg.metrics[name] for name in
                           ('substep_peak_qv','substep_impulse','cat_soft')},env)


def test_real_training_overlay_and_native_constraint_manager_preserve_50ms_impulse_window():
    from src.tasks.hammer.rl.impact_speed_wrapper import configure_impact_speed_controller
    env = event_env()
    configure_impact_speed_controller(env.cfg,sampled_pd500=True,gain_mode='variable')
    env.metrics_manager = _native_constraint_manager(env)
    native = env.metrics_manager.cfg['substep_impulse'].func
    assert env.cfg.sim.mujoco.timestep == .00025 and env.cfg.decimation == 80
    assert env.cfg.metrics['substep_impulse'].params['event_window_substeps'] == 200
    assert native._window == 200 and native._window*env.physics_dt == .05
    assert env.metrics_manager.cfg['cat_soft'].func._imp_limit.tolist() == pytest.approx([.369,.246,.738,.369,.246,.0164])
    assert api('contract').split_event_metadata(env)['critic_width'] == 47


def test_registration_is_idempotent_exact_four_recipes_with_normal_runner():
    m = api('cfg')
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    before = set(list_tasks())
    m.register_event_study_tasks()
    after = set(list_tasks())
    ids = {m.event_task_id(payout=p,setting=s) for p,s in RECIPES}
    assert after-before <= ids and ids <= after
    m.register_event_study_tasks()
    assert set(list_tasks()) == after
    for task in ids:
        assert load_runner_cls(task) is HammerOnPolicyRunner
        assert load_rl_cfg(task).critic.class_name.endswith(':CreditEventCritic')
        assert len(load_env_cfg(task).observations['critic'].terms) == 18


@pytest.fixture(autouse=True)
def forbid_environment_or_checkpoint(monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    def forbidden(*a,**k):
        raise AssertionError('environment construction/checkpoint loading forbidden in CPU036 tests')
    monkeypatch.setattr(ManagerBasedRlEnv,'__init__',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
