"""Pure prospective pilot math: no physics, checkpoints or training runs."""
from copy import deepcopy
import importlib

import pytest
import torch
from tensordict import TensorDict

from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedGaussian, ImpactSpeedLowNoiseModel


def api():
    name = 'src.tasks.hammer.rl.spline_exploration_pilot'
    assert importlib.util.find_spec(name), 'new opt-in pilot module is missing'
    return importlib.import_module(name)


def obs(width=43):
    x = torch.arange(3 * width, dtype=torch.float32).reshape(3, width) / 100
    x[:, -1] = torch.tensor([1., 0., 1.])
    return TensorDict({'actor': x, 'critic': x.clone()}, batch_size=[3])


def actor(setting='vic_baseline', **overrides):
    kwargs = dict(obs=obs(), obs_groups={'actor': ['actor']}, obs_set='actor', output_dim=13,
                  hidden_dims=(16, 8), distribution_cfg={
                      'class_name': 'src.tasks.hammer.rl.spline_exploration_pilot:PilotGaussian',
                      'init_std': 1., 'std_type': 'scalar'})
    kwargs.update(overrides)
    return api().PILOT_ACTOR_CLASSES[setting](**kwargs)


def states_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_exact_four_opt_in_actors_and_only_six_position_stds_differ():
    m = api()
    assert m.PILOT_SETTINGS == ('vic_baseline', 'vic_position_low', 'vic_gain_mean', 'fixed_nominal')
    assert tuple(m.PILOT_ACTOR_CLASSES) == m.PILOT_SETTINGS
    for setting in m.PILOT_SETTINGS:
        torch.manual_seed(9)
        model = actor(setting)
        assert model.obs_dim == 43 and model.distribution.output_dim == 13
        assert model.pilot_setting == setting
        expected = torch.tensor(([.1] if setting == 'vic_position_low' else [.3]) * 6 + [.3]*6 + [1.])
        torch.testing.assert_close(model.distribution.std_param, expected, rtol=0, atol=0)
        assert type(model.distribution) is m.PilotGaussian
        assert tuple(model.distribution.std_range) == (1e-6, 1e6)
        state = model.state_dict()
        if setting == 'vic_baseline': baseline = deepcopy(state)
        else:
            state['distribution.std_param'] = baseline['distribution.std_param']
            states_equal(state, baseline)
    with pytest.raises(TypeError): m.PILOT_ACTOR_CLASSES['extra'] = object
    with pytest.raises(KeyError): m.PILOT_ACTOR_CLASSES['vic_both']


@pytest.mark.parametrize('setting', ['vic_baseline', 'vic_position_low', 'vic_gain_mean', 'fixed_nominal'])
def test_matching_copies_every_state_without_rng_or_reference_mutation(setting):
    m = api()
    reference = actor()
    reference.obs_normalizer.update(obs()['actor'])
    before = deepcopy(reference.state_dict())
    destination = actor(setting)
    rng = torch.random.get_rng_state().clone()
    m.initialize_matched_pilot_actor(destination, reference)
    assert torch.equal(rng, torch.random.get_rng_state())
    states_equal(reference.state_dict(), before)
    expected = deepcopy(before)
    if setting == 'vic_position_low': expected['distribution.std_param'][:6] = .1
    states_equal(destination.state_dict(), expected)
    torch.testing.assert_close(destination(obs()), reference(obs()), rtol=0, atol=0)


def test_gain_mean_entropy_gradient_is_exactly_sixfold_only_for_gains():
    baseline, mean = actor(), actor('vic_gain_mean')
    api().initialize_matched_pilot_actor(mean, baseline)
    for model in (baseline, mean):
        model(obs(), stochastic_output=True)
        model.distribution.entropy.sum().backward()
    expected = baseline.distribution.std_param.grad.clone()
    expected[6:12] /= 6
    torch.testing.assert_close(mean.distribution.std_param.grad, expected, rtol=1e-7, atol=1e-7)
    elem = mean.distribution._distribution.entropy()
    wanted = elem[:, :6].sum(-1) + elem[:, 6:12].sum(-1)/6 + elem[:, 12]*torch.tensor([1.,0.,1.])
    torch.testing.assert_close(mean.distribution.entropy, wanted)


def test_entropy_treatment_does_not_change_likelihood_kl_or_parameter_signature():
    base, mean = actor(), actor('vic_gain_mean')
    api().initialize_matched_pilot_actor(mean, base)
    base(obs(), stochastic_output=True)
    mean(obs(), stochastic_output=True)
    x = torch.linspace(-.8,.8,39).reshape(3,13)
    torch.testing.assert_close(base.distribution.log_prob(x), mean.distribution.log_prob(x), rtol=0, atol=0)
    old = base.distribution.params
    new = (old[0] + .2, old[1]*1.1, old[2])
    assert len(old) == 3 and old[2].shape == (3,13)
    torch.testing.assert_close(base.distribution.kl_divergence(old,new), mean.distribution.kl_divergence(old,new), rtol=0, atol=0)


def test_fixed_samples_all_thirteen_and_preserves_pending_mask_and_rng_draws():
    base, fixed = actor(), actor('fixed_nominal')
    api().initialize_matched_pilot_actor(fixed, base)
    rng = torch.random.get_rng_state().clone()
    sampled_base = base(obs(), stochastic_output=True)
    after_base = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng)
    sampled_fixed = fixed(obs(), stochastic_output=True)
    assert sampled_fixed.shape == (3,13)
    torch.testing.assert_close(sampled_base, sampled_fixed, rtol=0, atol=0)
    assert torch.equal(after_base, torch.random.get_rng_state())
    mask = fixed.distribution.params[2]
    assert mask[:, :6].eq(1).all() and mask[:, 6:12].eq(0).all()
    assert torch.equal(mask[:,12], torch.tensor([1.,0.,1.]))
    # Saved PPO masks are immutable snapshots, not aliases of the next transition.
    changed = obs(); changed['actor'][:,-1] = 0
    fixed(changed, stochastic_output=True)
    assert torch.equal(mask[:,12], torch.tensor([1.,0.,1.]))


@pytest.mark.parametrize('objective', ['likelihood', 'entropy', 'kl'])
def test_fixed_ignored_gain_gradients_are_zero_but_positions_remain_active(objective):
    d = actor('fixed_nominal').distribution
    mean = torch.linspace(-.4,.4,39).reshape(3,13).requires_grad_()
    d.activity_mask = torch.cat((torch.ones(3,6),torch.zeros(3,6),torch.tensor([[1.],[0.],[1.]])), -1)
    d.update(mean)
    if objective == 'likelihood': value = d.log_prob(torch.ones_like(mean))
    elif objective == 'entropy': value = d.entropy
    else:
        old = (torch.zeros_like(mean), torch.ones_like(mean), d.activity_mask.clone())
        value = d.kl_divergence(old,d.params)
    value.sum().backward()
    assert d.std_param.grad[6:12].eq(0).all()
    assert torch.count_nonzero(d.std_param.grad[:6]) == 6
    if objective != 'entropy':
        assert mean.grad[:,6:12].eq(0).all() and mean.grad[1,12] == 0
        assert torch.count_nonzero(mean.grad[:,:6]) == 18


@pytest.mark.parametrize('bad', ['width','group','outputs','deterministic','legacy_distribution','log_std','frozen_std','initial_std','bounds'])
def test_actor_refuses_malformed_signatures_and_unapproved_distribution_changes(bad):
    kwargs = {}
    cfg = {'class_name':'src.tasks.hammer.rl.spline_exploration_pilot:PilotGaussian','init_std':1.,'std_type':'scalar'}
    if bad == 'width': kwargs['obs'] = obs(42)
    if bad == 'group': kwargs.update(obs_set='critic',obs_groups={'critic':['critic']})
    if bad == 'outputs': kwargs['output_dim'] = 12
    if bad == 'deterministic': cfg = None
    if bad == 'legacy_distribution': cfg['class_name'] = 'src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedGaussian'
    if bad == 'log_std': cfg['std_type'] = 'log'
    if bad == 'frozen_std': cfg['learn_std'] = False
    if bad == 'initial_std': cfg['init_std'] = .3
    if bad == 'bounds': cfg['std_range'] = (.01,2.)
    kwargs['distribution_cfg'] = cfg
    with pytest.raises(ValueError): actor(**kwargs)


@pytest.mark.parametrize('corrupt', ['shape','unknown_state','architecture','normalizer','std','alias'])
def test_matching_rejects_invalid_reference_without_partial_mutation(corrupt):
    m = api(); destination, reference = actor('vic_position_low'), actor()
    if corrupt == 'shape': reference.mlp[0].weight = torch.nn.Parameter(reference.mlp[0].weight[:,:42])
    if corrupt == 'unknown_state': reference.register_buffer('unexpected', torch.ones(1))
    if corrupt == 'architecture': reference.mlp[1] = torch.nn.ReLU()
    if corrupt == 'normalizer': reference.obs_normalizer.physical.eps = .1
    if corrupt == 'std': reference.distribution.std_param.data[0] = .2
    if corrupt == 'alias': reference = destination
    before = deepcopy(destination.state_dict()); rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError): m.initialize_matched_pilot_actor(destination,reference)
    states_equal(destination.state_dict(),before)
    assert torch.equal(rng,torch.random.get_rng_state())


def test_legacy_model_retains_original_distribution_entropy_and_motor_masks():
    legacy = ImpactSpeedLowNoiseModel(obs(),{'actor':['actor']},'actor',13,hidden_dims=(16,8),
        distribution_cfg={'class_name':'src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedGaussian','init_std':1.,'std_type':'scalar'})
    assert type(legacy.distribution) is ImpactSpeedGaussian
    api().initialize_matched_pilot_actor(actor(),legacy)
    legacy(obs(),stochastic_output=True)
    assert legacy.distribution.activity_mask[:,:12].eq(1).all()
    torch.testing.assert_close(legacy.distribution.entropy,(legacy.distribution._distribution.entropy()*legacy.distribution.activity_mask).sum(-1))


@pytest.mark.parametrize('corrupt', ['entropy_weight','setting','output_signature','nonfinite','dtype','log_bounds'])
def test_matching_rejects_tampered_semantics_and_state_before_mutation(corrupt):
    m = api(); destination, reference = actor(), actor()
    if corrupt == 'entropy_weight': destination.distribution.gain_entropy_weight = .5
    if corrupt == 'setting': destination.pilot_setting = 'vic_both'
    if corrupt == 'output_signature': reference.distribution.output_dim = 12
    if corrupt == 'nonfinite': reference.mlp[0].weight.data[0,0] = float('nan')
    if corrupt == 'dtype': reference.double()
    if corrupt == 'log_bounds': reference.distribution.log_std_range[0] = -100.
    before = deepcopy(destination.state_dict()); rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError): m.initialize_matched_pilot_actor(destination,reference)
    states_equal(destination.state_dict(),before)
    assert torch.equal(rng,torch.random.get_rng_state())


def test_raw_tail_and_bad_pending_flag_use_unchanged_legacy_contract():
    model = actor('fixed_nominal')
    model.obs_normalizer.update(obs()['actor'])
    torch.testing.assert_close(model.obs_normalizer(obs()['actor'])[:,40:],obs()['actor'][:,40:],rtol=0,atol=0)
    bad = obs(); bad['actor'][0,-1] = .5
    with pytest.raises(ValueError,match='pending'): model(bad,stochastic_output=True)
    model(obs(),stochastic_output=True)
    params = model.distribution.params
    wrong = params[2].clone(); wrong[0,0] = 0
    with pytest.raises(ValueError,match='disagree'):
        model.distribution.kl_divergence(params,(params[0],params[1],wrong))


@pytest.mark.parametrize('shared', ['std', 'weight', 'weight_view', 'buffer', 'buffer_view', 'cross_key'])
def test_matching_rejects_any_cross_actor_storage_alias_without_mutation(shared):
    destination, reference = actor('vic_position_low'), actor()
    if shared == 'std':
        destination.distribution.std_param = reference.distribution.std_param
    elif shared in ('weight', 'weight_view'):
        source = reference.mlp[0].weight
        destination.mlp[0].weight = torch.nn.Parameter(source if shared == 'weight' else source.view_as(source))
    elif shared in ('buffer', 'buffer_view'):
        source = reference.obs_normalizer.physical._mean
        destination.obs_normalizer.physical._mean = source if shared == 'buffer' else source.view_as(source)
    else:
        # Compare every storage, not only corresponding state keys.
        destination.mlp[0].bias = torch.nn.Parameter(reference.mlp[2].weight.flatten()[:16])
    assert destination is not reference
    before_destination = deepcopy(destination.state_dict())
    before_reference = deepcopy(reference.state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match='storage'):
        api().initialize_matched_pilot_actor(destination, reference)
    states_equal(destination.state_dict(), before_destination)
    states_equal(reference.state_dict(), before_reference)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_matching_rejects_internal_destination_storage_alias_before_loading():
    destination, reference = actor(), actor()
    destination.mlp[0].bias = torch.nn.Parameter(destination.mlp[2].weight.flatten()[:16])
    before_destination = deepcopy(destination.state_dict())
    before_reference = deepcopy(reference.state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match='storage'):
        api().initialize_matched_pilot_actor(destination, reference)
    states_equal(destination.state_dict(), before_destination)
    states_equal(reference.state_dict(), before_reference)
    assert torch.equal(rng, torch.random.get_rng_state())
