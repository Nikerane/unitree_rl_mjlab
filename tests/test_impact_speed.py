"""One-shot speed selection without changing motor actions."""

from __future__ import annotations

from types import SimpleNamespace
import math

import pytest
import torch

from mjlab.managers.action_manager import ActionManager
from src.tasks.hammer.mdp.impact_speed import (
    ImpactSpeedActionCfg,
    ImpactSpeedState,
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
