"""One-shot speed selection without changing motor actions."""

from __future__ import annotations

from types import SimpleNamespace
import math

import pytest
import torch

from mjlab.managers.action_manager import ActionManager
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from src.tasks.hammer.mdp.impact_speed import (
    ImpactSpeedActionCfg,
    ImpactSpeedCost,
    ImpactSpeedState,
    consume_speed_cost,
    impact_speed_goal,
    impact_speed_pending,
)


def test_policy_goal_latches_once_and_resets_only_subset():
    s = ImpactSpeedState(2, "cpu", (0.2, 0.8), "policy")
    s.reset(None)
    s.select(torch.tensor([[-1.0], [1.0]]))
    torch.testing.assert_close(s.goal_mps, torch.tensor([0.2, 0.8]))
    s.select(torch.zeros(2, 1))
    torch.testing.assert_close(s.goal_mps, torch.tensor([0.2, 0.8]))
    s.reset(torch.tensor([0]))
    assert s.pending.tolist() == [1.0, 0.0]
    assert s.goal_mps[1].item() == pytest.approx(0.8)


def test_external_goal_cannot_be_overwritten():
    s = ImpactSpeedState(3, "cpu", (0.2, 0.8), "external")
    s.reset(None)
    before = s.goal_mps.clone()
    s.select(torch.ones(3, 1))
    torch.testing.assert_close(s.goal_mps, before)
    assert not s.pending.any()


@pytest.mark.parametrize("raw", [math.nan, math.inf, -math.inf])
def test_selection_rejects_nonfinite_before_clipping(raw):
    s = ImpactSpeedState(1, "cpu", (0.2, 0.8), "policy")
    s.reset(None)
    with pytest.raises(ValueError, match="finite"):
        s.select(torch.tensor([[raw]]))
    assert s.pending.item() == 1


@pytest.mark.parametrize("bounds", [(0, 0.8), (-0.1, 0.8), (0.8, 0.2), (0.2, 0.2), (0.2, math.inf), (math.nan, 0.8), (True, 0.8), (0.2, False)])
def test_bounds_must_be_finite_positive_increasing_reals(bounds):
    with pytest.raises(ValueError, match="bounds"):
        ImpactSpeedState(1, "cpu", bounds, "policy")


@pytest.mark.parametrize("shape", [(2,), (1, 2), (2, 2)])
def test_selection_requires_one_column_per_world(shape):
    s = ImpactSpeedState(2, "cpu", (0.2, 0.8), "policy")
    with pytest.raises(ValueError, match="shape"):
        s.select(torch.zeros(shape))


def test_observation_sentinel_and_subset_reset():
    s = ImpactSpeedState(3, "cpu", (0.2, 0.8), "policy")
    torch.testing.assert_close(s.observation(), torch.tensor([[0., 1.]] * 3))
    s.select(torch.tensor([[-1.], [0.], [1.]]))
    torch.testing.assert_close(s.observation(), torch.tensor([[-1., 0.], [0., 0.], [1., 0.]]))
    s.cost_consumed[:] = True
    s.reset(torch.tensor([], dtype=torch.long))
    assert s.cost_consumed.all()
    s.reset(slice(1, 2))
    torch.testing.assert_close(s.observation(), torch.tensor([[-1., 0.], [0., 1.], [1., 0.]]))
    assert s.cost_consumed.tolist() == [True, False, True]


def test_external_reset_is_seeded_and_subset_owned():
    torch.manual_seed(42)
    a = ImpactSpeedState(3, "cpu", (0.2, 0.8), "external")
    a.reset(None)
    first = a.goal_mps.clone()
    torch.manual_seed(42)
    b = ImpactSpeedState(3, "cpu", (0.2, 0.8), "external")
    b.reset(None)
    torch.testing.assert_close(first, b.goal_mps)
    assert bool(((first >= 0.2) & (first <= 0.8)).all())
    a.reset(torch.tensor([1]))
    torch.testing.assert_close(a.goal_mps[[0, 2]], first[[0, 2]])
    torch.testing.assert_close(a.observation()[:, 1], torch.zeros(3))


def test_action_manager_owns_reset_and_latches_only_on_process():
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": object()})
    env.action_manager = ActionManager({"strike_speed": ImpactSpeedActionCfg(entity_name="robot", bounds=(0.2, 0.8), mode="policy")}, env)
    term = env.action_manager.get_term("strike_speed")
    assert term.action_dim == 1
    env.action_manager.reset()
    torch.testing.assert_close(impact_speed_goal(env), torch.zeros(2, 1))
    torch.testing.assert_close(impact_speed_pending(env), torch.ones(2, 1))
    env.action_manager.process_action(torch.tensor([[-2.], [2.]]))
    torch.testing.assert_close(term.state.goal_mps, torch.tensor([0.2, 0.8]))
    torch.testing.assert_close(term.processed_actions, torch.tensor([[-1.], [1.]]))
    torch.testing.assert_close(term.raw_action, term.raw_actions)
    for _ in range(4):
        env.action_manager.apply_action()
    env.action_manager.process_action(torch.zeros(2, 1))
    torch.testing.assert_close(term.state.goal_mps, torch.tensor([0.2, 0.8]))
    env.action_manager.reset(torch.tensor([0]))
    torch.testing.assert_close(impact_speed_pending(env), torch.tensor([[1.], [0.]]))
    assert term.state.goal_mps[1].item() == pytest.approx(0.8)


def test_external_goal_sampling_survives_later_action_manager_reset():
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": object()})
    env.action_manager = ActionManager({"strike_speed": ImpactSpeedActionCfg(entity_name="robot", bounds=(0.2, 0.8), mode="external")}, env)
    env.action_manager.reset()
    original = env.action_manager.get_term("strike_speed").state.goal_mps.clone()
    env.action_manager.process_action(torch.ones(2, 1))
    torch.testing.assert_close(env.action_manager.get_term("strike_speed").state.goal_mps, original)
    env.action_manager.reset(torch.tensor([], dtype=torch.long))
    torch.testing.assert_close(env.action_manager.get_term("strike_speed").state.goal_mps, original)


def test_action_cfg_requires_explicit_bounds():
    env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": object()})
    with pytest.raises(ValueError, match="bounds"):
        ImpactSpeedActionCfg(entity_name="robot", mode="policy").build(env)


def test_speed_cost_outcomes_and_second_read_are_exactly_once():
    state = ImpactSpeedState(5, "cpu", (0.2, 0.8), "policy")
    state.goal_mps[:] = 0.5
    finalized = torch.tensor([True, False, False, False, True])
    started = torch.tensor([True, True, False, False, True])
    done = torch.tensor([True, True, True, False, False])
    measured = torch.tensor([0.8, float("nan"), float("nan"), float("nan"), 0.2])
    expected = torch.tensor([0.25, 1.0, 1.0, 0.0, 0.25])
    actual = consume_speed_cost(state, finalized=finalized, started=started, measured_mps=measured, done=done)
    torch.testing.assert_close(actual, expected)
    assert state.cost_consumed.tolist() == [True, True, True, False, True]
    second = consume_speed_cost(state, finalized=finalized, started=started, measured_mps=measured, done=done)
    torch.testing.assert_close(second, torch.zeros(5))


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_finalized_speed_measurement_is_required_and_finite(bad):
    state = ImpactSpeedState(1, "cpu", (0.2, 0.8), "policy")
    measured = None if bad is None else torch.tensor([bad])
    with pytest.raises(ValueError, match="measurement|measured"):
        consume_speed_cost(state, finalized=torch.tensor([True]), started=torch.tensor([True]), measured_mps=measured, done=torch.tensor([True]))
    assert not state.cost_consumed.item()


def test_finalized_nonproductive_event_still_charges_speed_error():
    state = ImpactSpeedState(1, "cpu", (0.2, 0.8), "policy")
    state.goal_mps[:] = 0.5
    # Productivity is intentionally absent from the cost API; a finalized first strike is charged.
    value = consume_speed_cost(state, finalized=torch.tensor([True]), started=torch.tensor([True]), measured_mps=torch.tensor([0.8]), done=torch.tensor([False]))
    torch.testing.assert_close(value, torch.tensor([0.25]))


def test_event_charge_is_independent_of_control_dt():
    for dt in (0.01, 0.02, 0.04):
        raw = torch.tensor([0.25]) / dt
        coefficient = 0.4
        torch.testing.assert_close(raw * (-coefficient) * dt, torch.tensor([-0.1]))


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_actual_reward_manager_scales_speed_event_once(dt):
    state = ImpactSpeedState(1, "cpu", (0.2, 0.8), "policy")
    state.goal_mps[:] = 0.5
    tracker = SimpleNamespace(finalized=torch.tensor([True]), started=torch.tensor([True]), productive=torch.tensor([False]), v_precontact=torch.tensor([0.8]))
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=dt, reset_buf=torch.tensor([True]), extras={}, action_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(state=state)))
    env.reward_manager = RewardManager({"speed_goal": RewardTermCfg(func=ImpactSpeedCost, weight=-0.4)}, env, scale_by_dt=True)
    env._hammer_first_strike = tracker  # Metrics terms are constructed after reward terms.
    torch.testing.assert_close(env.reward_manager.compute(dt), torch.tensor([-0.1]))
    outcome = env.extras["impact_speed_outcome"]
    assert outcome["finalized"].tolist() == [True]
    torch.testing.assert_close(env.reward_manager.compute(dt), torch.tensor([0.0]))
    tracker.finalized[:] = False
    tracker.v_precontact[:] = 0.0
    state.reset(None)
    assert outcome["finalized"].tolist() == [True]
    torch.testing.assert_close(outcome["goal_mps"], torch.tensor([0.5]))
    torch.testing.assert_close(outcome["measured_mps"], torch.tensor([0.8]))


def test_terminal_snapshot_distinguishes_miss_incomplete_and_measured_event():
    state = ImpactSpeedState(3, "cpu", (0.2, 0.8), "policy")
    state.goal_mps[:] = torch.tensor([0.3, 0.4, 0.5])
    tracker = SimpleNamespace(
        finalized=torch.tensor([False, False, True]),
        started=torch.tensor([False, True, True]),
        productive=torch.tensor([False, False, False]),
        v_precontact=torch.tensor([0.0, 0.0, 0.8]),
    )
    env = SimpleNamespace(
        num_envs=3, device="cpu", step_dt=0.02, reset_buf=torch.tensor([True, True, True]),
        extras={}, action_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(state=state)),
        _hammer_first_strike=tracker,
    )
    rate = ImpactSpeedCost(RewardTermCfg(func=ImpactSpeedCost, weight=-0.4), env)(env)
    torch.testing.assert_close(rate, torch.tensor([50.0, 50.0, 12.5]))
    outcome = env.extras["impact_speed_outcome"]
    assert outcome["charged"].tolist() == [True, True, True]
    assert outcome["valid_measured"].tolist() == [False, False, True]
    assert outcome["miss"].tolist() == [True, False, False]
    assert outcome["incomplete"].tolist() == [False, True, False]
    assert outcome["finalized"].tolist() == [False, False, True]
    torch.testing.assert_close(outcome["cost"], torch.tensor([1.0, 1.0, 0.25]))
    assert torch.isnan(outcome["measured_mps"][:2]).all()
    state.reset(None)
    tracker.finalized[:] = False
    tracker.v_precontact[:] = -100.0
    torch.testing.assert_close(outcome["goal_mps"], torch.tensor([0.3, 0.4, 0.5]))
    torch.testing.assert_close(outcome["measured_mps"][2:], torch.tensor([0.8]))
