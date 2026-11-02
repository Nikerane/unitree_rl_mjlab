"""Behavior contracts: no hover income, no repeated credit, no missed-hit bonus."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.references import SingleStrikeReference


def curriculum():
    name = "src.tasks.hammer.mdp.spline_curriculum"
    assert importlib.util.find_spec(name) is not None, "bounded spline rewards are missing"
    return importlib.import_module(name)


def environment(batch=2):
    head = torch.tensor([[0., 0., .2]]).repeat(batch, 1)
    nail = torch.tensor([[0., 0., .1]]).repeat(batch, 1)
    ref = SingleStrikeReference(batch, "cpu", path_shape="quintic", overshoot=.02)
    ref.update(head, nail, torch.zeros(batch, dtype=torch.long))
    tracker = SimpleNamespace(started=torch.zeros(batch, dtype=torch.bool),
                              v_precontact=torch.ones(batch), onset_head_w=head.clone())
    env = SimpleNamespace(num_envs=batch, device="cpu", step_dt=.02,
                          _strike_reference=ref, _hammer_first_strike=tracker,
                          scene={"robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=head[:, None]))})
    params = dict(robot_cfg=SimpleNamespace(name="robot", site_ids=[0]), width_m=.01)
    return env, head, tracker, params


def test_stationary_on_path_has_no_recurring_income_or_reference_mutation():
    module = curriculum()
    env, head, tracker, params = environment()
    path = module.SplinePathDeviationCost(None, env)
    progress = module.SplineProgressReward(None, env)
    for _ in range(4):
        assert torch.equal(path(env, **params), torch.zeros(2))
        assert torch.equal(progress(env, **params), torch.zeros(2))
    head[:, 2] = .15
    assert torch.equal(progress(env, **params) * env.step_dt, torch.full((2,), .5))
    assert torch.equal(env._strike_reference.peek(), torch.zeros(2))


def test_progress_pays_only_new_on_path_maximum_and_reset_is_per_world():
    module = curriculum()
    env, head, tracker, params = environment()
    term = module.SplineProgressReward(None, env)
    head[:, 2] = .15
    torch.testing.assert_close(term(env, **params) * .02, torch.full((2,), .5))
    head[:, 2] = .18
    assert torch.equal(term(env, **params), torch.zeros(2))
    head[:, 2] = .15
    assert torch.equal(term(env, **params), torch.zeros(2))
    head[0, 0] = .02
    head[:, 2] = .1
    torch.testing.assert_close(term(env, **params) * .02, torch.tensor([0., .5]))
    term.reset(torch.tensor([1]))
    head[:, 0] = 0
    head[:, 2] = .15
    torch.testing.assert_close(term(env, **params) * .02, torch.tensor([0., .5]))
    tracker.started[:] = True
    head[:, 2] = .1
    assert torch.equal(term(env, **params), torch.zeros(2))


def test_path_cost_is_bounded_zero_on_curve_and_closed_after_contact():
    module = curriculum()
    env, head, tracker, params = environment()
    term = module.SplinePathDeviationCost(None, env)
    head[:, 0] = torch.tensor([.01, .1])
    torch.testing.assert_close(term(env, **params), torch.tensor([1., 4.]))
    tracker.started[0] = True
    torch.testing.assert_close(term(env, **params), torch.tensor([0., 4.]))


@pytest.mark.parametrize("onset,speed,want", [((0., 0., .1), 1., 1.),
    ((.02, 0., .1), 1., 0.), ((0., 0., .15), 1., 0.), ((0., 0., .1), 0., 0.)])
def test_completion_requires_accepted_downward_contact_at_route_endpoint(onset, speed, want):
    module = curriculum()
    env, head, tracker, params = environment(1)
    term = module.AcceptedSplineStrikeReward(None, env)
    assert term(env, **params).item() == 0
    tracker.onset_head_w[:] = torch.tensor(onset)
    tracker.v_precontact[:] = speed
    tracker.started[:] = True
    # Reward-time rebound must not replace the accepted physics-onset position.
    head[:, 2] = .18
    assert (term(env, **params) * .02).item() == pytest.approx(want)
    assert term(env, **params).item() == 0
    term.reset(None)
    tracker.started[:] = False
    assert term(env, **params).item() == 0


def test_new_task_is_isolated_and_preserves_controller_geometry_and_safety():
    module = curriculum()
    from mjlab.tasks.registry import load_env_cfg
    import src.tasks
    old = load_env_cfg("Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed")
    cfg = load_env_cfg(module.TASK)
    assert tuple(cfg.actions) == ("joint_position", "joint_stiffness", "strike_speed")
    assert cfg.actions["joint_stiffness"] == old.actions["joint_stiffness"]
    assert cfg.actions["joint_position"] == old.actions["joint_position"]
    assert cfg.actions["strike_speed"].bounds == (1., 1.5)
    assert cfg.events == old.events
    assert cfg.metrics == old.metrics
    assert cfg.terminations["time_out"] == old.terminations["time_out"]
    assert cfg.terminations["nail_driven"].func is module.accepted_strike_window_complete
    assert cfg.sim == old.sim
    for name in ("approach", "nail_driven", "nail_depth_delta", "impact_progress",
                 "completion", "delivered_impulse", "r_imit"):
        assert cfg.rewards[name].weight == 0
        assert old.rewards[name].weight > 0
    assert cfg.rewards["action_rate"] == old.rewards["action_rate"]
    assert cfg.rewards["r_tt"] == old.rewards["r_tt"]
    assert old.actions["strike_speed"].bounds is None


def test_training_waits_full_postcontact_window_without_depth_success_reset():
    module = curriculum()
    env, head, tracker, params = environment(2)
    tracker.started[:] = torch.tensor([True, False])
    tracker.first_contact_time_s = torch.tensor([.01, 0.])
    tracker.elapsed_since_contact_s = torch.tensor([.049, 10.])
    # Training may randomize this counter; it must not shorten the contact window.
    env.episode_length_buf = torch.tensor([100, 100])
    assert torch.equal(module.accepted_strike_window_complete(env), torch.tensor([False, False]))
    tracker.elapsed_since_contact_s[0] = .05
    assert torch.equal(module.accepted_strike_window_complete(env), torch.tensor([True, False]))
