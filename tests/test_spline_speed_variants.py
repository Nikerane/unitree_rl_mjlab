"""Isolated overnight treatments: geometric feedback and observed-bin credit."""

import importlib
from copy import deepcopy
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp import observations
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.mdp.spline_curriculum import SplineProgressReward, TASK


def variants():
    name = "src.tasks.hammer.mdp.spline_speed_variants"
    assert importlib.util.find_spec(name) is not None, "isolated O/P variants are missing"
    return importlib.import_module(name)


def config_value(value):
    """Compare factory-built configs without fresh-but-identical lambda addresses."""
    if is_dataclass(value):
        return {field.name: config_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(config_value(item) for item in value)
    if callable(value):
        code = getattr(value, "__code__", None)
        return (value.__module__, value.__qualname__,
                code.co_code if code is not None else None,
                code.co_consts if code is not None else None,
                config_value(getattr(value, "__defaults__", None)))
    return value


def scene(batch=2):
    head = torch.tensor([[.4, 0., .25]]).repeat(batch, 1)
    nail = torch.tensor([[.5, 0., .1]]).repeat(batch, 1)
    ref = SingleStrikeReference(batch, "cpu", path_shape="quintic", overshoot=.02)
    ref.update(head, nail, torch.zeros(batch, dtype=torch.long))
    env = SimpleNamespace(num_envs=batch, device="cpu", step_dt=.02,
        episode_length_buf=torch.ones(batch, dtype=torch.long),
        _strike_reference=ref,
        _hammer_first_strike=SimpleNamespace(started=torch.zeros(batch, dtype=torch.bool)),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=head[:, None])),
               "nail": SimpleNamespace(data=SimpleNamespace(site_pos_w=nail[:, None]))})
    params = dict(robot_cfg=SimpleNamespace(name="robot", site_ids=[0]),
                  nail_cfg=SimpleNamespace(name="nail", site_ids=[0]),
                  path_shape="quintic", overshoot=.02)
    return env, head, nail, ref, params


def test_reproduce_baseline_observation_disagreement_after_deviation_and_rebound():
    env, head, _, ref, params = scene()
    head[:] = ref.waypoint(torch.full((2,), .2))
    observations.strike_phase(env, **params)
    head[:] = ref.waypoint(torch.full((2,), .6))
    head[:, 1] += .06
    torch.testing.assert_close(observations.strike_phase(env, **params), torch.full((2, 1), .2))
    torch.testing.assert_close(observations.strike_ref_error(env, **params)[:, 2], torch.full((2,), .068))
    assert torch.equal(ref.diagnose_current_height(head)[1][:, 2] - head[:, 2], torch.zeros(2))
    head[:] = ref.waypoint(torch.full((2,), .6))
    observations.strike_phase(env, **params)
    head[:] = ref.waypoint(torch.full((2,), .2))
    torch.testing.assert_close(observations.strike_ref_error(env, **params)[:, 2], torch.full((2,), -.068))
    torch.testing.assert_close(ref.diagnose_current_height(head)[2], torch.zeros(2), atol=1e-7, rtol=0)


def test_reproduce_baseline_skipped_progress_credit():
    env, head, _, ref, params = scene()
    reward = SplineProgressReward(None, env)
    params = {"robot_cfg": params["robot_cfg"]}
    head[:] = ref.waypoint(torch.full((2,), .5 * .15 / .17))
    head[:, 1] += .06
    assert torch.equal(reward(env, **params), torch.zeros(2))
    head[:] = ref.waypoint(torch.full((2,), .99 * .15 / .17))
    torch.testing.assert_close(reward(env, **params) * .02, torch.full((2,), .99))


def test_observation_variant_corrects_deviation_rebound_and_keeps_latch():
    module = variants()
    env, head, _, ref, params = scene()
    head[:] = ref.waypoint(torch.full((2,), .2))
    module.current_height_phase(env, **params)
    head[:] = ref.waypoint(torch.full((2,), .6))
    head[:, 1] += .06
    torch.testing.assert_close(module.current_height_phase(env, **params), torch.full((2, 1), .6))
    torch.testing.assert_close(module.current_height_error(env, **params),
                               torch.tensor([[0., -.06, 0.]]).repeat(2, 1), atol=1e-7, rtol=0)
    torch.testing.assert_close(ref.peek(), torch.full((2,), .2))
    head[:] = ref.waypoint(torch.full((2,), .6))
    module.current_height_phase(env, **params)
    head[:] = ref.waypoint(torch.full((2,), .2))
    torch.testing.assert_close(module.current_height_phase(env, **params), torch.full((2, 1), .2))
    torch.testing.assert_close(module.current_height_error(env, **params), torch.zeros(2, 3), atol=1e-7, rtol=0)
    torch.testing.assert_close(ref.peek(), torch.full((2,), .6))
    env._hammer_first_strike.started[:] = True
    torch.testing.assert_close(module.current_height_phase(env, **params), torch.full((2, 1), .2))


def test_observation_variant_reset_reanchors_only_new_episode():
    module = variants()
    env, head, _, ref, params = scene()
    head[:] = ref.waypoint(torch.full((2,), .6))
    module.current_height_phase(env, **params)
    env.episode_length_buf[0] = 0
    head[0] = torch.tensor([.35, 0., .3])
    torch.testing.assert_close(module.current_height_phase(env, **params), torch.tensor([[0.], [.6]]))
    torch.testing.assert_close(module.current_height_error(env, **params), torch.zeros(2, 3), atol=1e-7, rtol=0)


def sample(collector, height, error=0., started=False):
    n = collector.num_envs
    collector.sample(approach_height=torch.as_tensor(height).float().expand(n),
                     error_m=torch.as_tensor(error).float().expand(n),
                     started=torch.as_tensor(started).bool().expand(n))


def test_bin_credit_never_fills_skipped_bins_and_cannot_repay_rebounds():
    c = variants().SplineBinVisits(2, "cpu")
    sample(c, .5, error=.06)
    sample(c, .99)
    torch.testing.assert_close(c.drain(), torch.full((2,), 1 / 16))
    for height in (.99, .99, .26, .99, .26):
        sample(c, height)
    torch.testing.assert_close(c.drain(), torch.full((2,), 1 / 16))
    assert c.visited.sum(-1).tolist() == [2, 2]


def test_bin_credit_counts_dense_substeps_once_and_has_unit_cap():
    c = variants().SplineBinVisits(2, "cpu")
    # All samples happen before one reward read; control cadence cannot lose visits.
    for height in torch.linspace(0., 1., 129):
        sample(c, height)
    torch.testing.assert_close(c.drain(), torch.ones(2))
    for height in torch.linspace(1., 0., 129):
        sample(c, height)
    assert torch.equal(c.drain(), torch.zeros(2))


def test_bin_credit_ignores_outside_domain_and_pays_initial_occupancy_once():
    c = variants().SplineBinVisits(2, "cpu")
    for height in (-.01, 1.01):
        sample(c, height)
    assert torch.equal(c.drain(), torch.zeros(2))
    for _ in range(5):
        sample(c, 0.)
    torch.testing.assert_close(c.drain(), torch.full((2,), 1 / 16))
    assert torch.equal(c.drain(), torch.zeros(2))


def test_bin_credit_flushes_precontact_earnings_but_never_collects_after_onset():
    c = variants().SplineBinVisits(2, "cpu")
    sample(c, .01)
    sample(c, .5, started=True)
    sample(c, .99, started=True)
    torch.testing.assert_close(c.drain(), torch.full((2,), 1 / 16))
    assert c.visited.sum(-1).tolist() == [1, 1]
    assert torch.equal(c.drain(), torch.zeros(2))


def test_bin_credit_subset_reset_drops_only_its_unpaid_credit():
    c = variants().SplineBinVisits(2, "cpu")
    sample(c, .1)
    c.reset(torch.tensor([0]))
    sample(c, .1)
    torch.testing.assert_close(c.drain(), torch.full((2,), 1 / 16))
    c.reset(None)
    assert c.visited.sum().item() == 0
    assert torch.equal(c.drain(), torch.zeros(2))


def test_progress_metric_and_reward_preserve_reference_and_deliver_queued_credit():
    module = variants()
    env, head, _, ref, params = scene()
    reward = module.VisitedBinsProgressReward(None, env)
    metric = module.SplineBinVisitMetric(None, env)
    head[:] = ref.waypoint(torch.full((2,), .01 * .15 / .17))
    metric(env, robot_cfg=params["robot_cfg"])
    head[:] = ref.waypoint(torch.full((2,), .99 * .15 / .17))
    metric(env, robot_cfg=params["robot_cfg"])
    env._hammer_first_strike.started[:] = True
    metric(env, robot_cfg=params["robot_cfg"])
    torch.testing.assert_close(reward(env) * .02, torch.full((2,), 2 / 16))
    assert torch.equal(reward(env), torch.zeros(2))
    assert torch.equal(ref.peek(), torch.zeros(2))
    reward.reset(torch.tensor([0]))
    env._hammer_first_strike.started[0] = False
    metric(env, robot_cfg=params["robot_cfg"])
    torch.testing.assert_close(reward(env) * .02, torch.tensor([1 / 16, 0.]))


@pytest.mark.parametrize("play", [False, True])
def test_variants_have_only_the_approved_config_deltas(play):
    module = variants()
    from mjlab.tasks.registry import load_env_cfg
    import src.tasks
    baseline = load_env_cfg(TASK, play=play)
    observed = load_env_cfg(module.TASKS["O"], play=play)
    progress = load_env_cfg(module.TASKS["P"], play=play)
    for group in ("actor", "critic"):
        assert observed.observations[group].terms["strike_phase"].func is module.current_height_phase
        assert observed.observations[group].terms["strike_ref_error"].func is module.current_height_error
    for group in ("actor", "critic"):
        for term in ("strike_phase", "strike_ref_error"):
            observed.observations[group].terms[term].func = baseline.observations[group].terms[term].func
    assert config_value(observed) == config_value(baseline)
    assert progress.rewards["spline_progress"].func is module.VisitedBinsProgressReward
    progress.rewards["spline_progress"] = deepcopy(baseline.rewards["spline_progress"])
    metric = progress.metrics.pop("spline_bin_visits")
    assert metric.per_substep and metric.func is module.SplineBinVisitMetric
    assert tuple(baseline.metrics).index("first_strike") < len(baseline.metrics)
    assert config_value(progress) == config_value(baseline)


@pytest.mark.integration
def test_live_contracts_reload_only_their_own_variant_and_keep_baseline_identity():
    module = variants()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from src.tasks.hammer.mdp.sampled_pd500 import configure_sampled_pd500, install_sampled_pd500
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata, validate_impact_speed_metadata
    contracts = {}
    for variant in ("B", "O", "P"):
        cfg = load_env_cfg(module.TASKS[variant])
        cfg.scene.num_envs = 2
        configure_sampled_pd500(cfg, gain_mode="variable")
        env = ManagerBasedRlEnv(cfg, device="cpu")
        try:
            install_sampled_pd500(env, gain_mode="variable")
            metadata = build_impact_speed_metadata(env, 1.)
            contracts[variant] = metadata
            assert ("spline_variant" in metadata) is (variant != "B")
            validate_impact_speed_metadata(deepcopy(metadata), build_impact_speed_metadata(env, 1.))
            if variant == "O":
                env.observation_manager.get_term_cfg("critic", "strike_phase").func = observations.strike_phase
                with pytest.raises(ValueError, match="observation|variant"):
                    build_impact_speed_metadata(env, 1.)
            if variant == "P":
                # The real manager executes first-strike before visit collection.
                terms = env.metrics_manager.active_terms
                assert terms.index("first_strike") < terms.index("spline_bin_visits")
                env.metrics_manager.cfg["spline_bin_visits"].per_substep = False
                with pytest.raises(ValueError, match="substep|variant"):
                    build_impact_speed_metadata(env, 1.)
        finally:
            env.close()
    for left in contracts:
        for right in contracts:
            if left != right:
                with pytest.raises(ValueError, match="mismatch"):
                    validate_impact_speed_metadata(contracts[left], contracts[right])
