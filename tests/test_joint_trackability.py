"""Joint command-trackability formula contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.config.z1.joint_position_contract import JOINT_NAMES
from src.tasks.hammer.mdp.trackability import (
    joint_target_rmse,
    joint_target_squared_error,
    joint_trackability_cost,
)


def _reader_env(
    *,
    terminal: bool = False,
    active_terms: tuple[str, ...] = ("joint_position",),
):
    q_des = torch.tensor(
        [[0.2, 0.0, -0.1, 0.3, 0.1, 0.0, 99.0]], dtype=torch.float32
    )
    q_next = torch.tensor(
        [[0.1, 0.0, -0.1, 0.1, 0.0, 0.0, -99.0]], dtype=torch.float32
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos_target=q_des, joint_pos=q_next)
    )
    action_term = SimpleNamespace(
        target_ids=torch.arange(6), target_names=list(JOINT_NAMES)
    )
    action_manager = SimpleNamespace(
        active_terms=list(active_terms), get_term=lambda name: action_term
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        action_manager=action_manager,
        cfg=SimpleNamespace(auto_reset=False),
        reset_buf=torch.tensor([terminal]),
        step_dt=0.02,
    )
    arm_cfg = SimpleNamespace(
        name="robot", joint_ids=list(range(6)), joint_names=list(JOINT_NAMES)
    )
    return env, arm_cfg


def test_readers_use_public_applied_target_and_only_six_ordered_arm_joints() -> None:
    """Reading realized q, a seventh joint, or a reordered action must change this result."""
    env, arm_cfg = _reader_env()
    expected = torch.tensor([0.06])
    torch.testing.assert_close(joint_target_squared_error(env, arm_cfg), expected)
    torch.testing.assert_close(
        joint_target_rmse(env, arm_cfg), torch.sqrt(expected / 6.0)
    )
    torch.testing.assert_close(
        joint_trackability_cost(env, arm_cfg, 2.0), 2.0 * expected
    )


def test_reader_uses_current_state_even_when_terminal_mask_is_set() -> None:
    """The pure reader does not zero or replace state based on a terminal mask."""
    env, arm_cfg = _reader_env(terminal=True)
    torch.testing.assert_close(
        joint_target_squared_error(env, arm_cfg), torch.tensor([0.06])
    )


def test_reward_manager_dt_is_external_to_raw_trackability_cost() -> None:
    """Accidentally multiplying the raw term by dt would apply RewardManager scaling twice."""
    env, arm_cfg = _reader_env()
    raw = joint_trackability_cost(env, arm_cfg, 2.0)
    torch.testing.assert_close(raw, torch.tensor([0.12]))
    torch.testing.assert_close(raw * env.step_dt, torch.tensor([0.0024]))


def test_reader_accepts_exact_ordered_variable_impedance_action_signature() -> None:
    """Adding stiffness coordinates must not change the six-joint RTT value."""
    env, arm_cfg = _reader_env(
        active_terms=("joint_position", "joint_stiffness")
    )

    torch.testing.assert_close(
        joint_target_squared_error(env, arm_cfg), torch.tensor([0.06])
    )
    torch.testing.assert_close(
        joint_trackability_cost(env, arm_cfg, 1.0), torch.tensor([0.06])
    )


@pytest.mark.parametrize(
    "active_terms",
    (
        ("joint_stiffness", "joint_position"),
        ("joint_position", "renamed_stiffness"),
        ("joint_position", "joint_stiffness", "extra"),
    ),
    ids=("reversed", "renamed", "extra"),
)
def test_reader_rejects_every_other_multi_action_signature(
    active_terms: tuple[str, ...],
) -> None:
    env, arm_cfg = _reader_env(active_terms=active_terms)

    with pytest.raises(ValueError, match="requires exactly"):
        joint_target_squared_error(env, arm_cfg)


@pytest.mark.parametrize("k_tt", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_trackability_cost_rejects_nonpositive_or_nonfinite_gain(k_tt: float) -> None:
    env, arm_cfg = _reader_env()
    with pytest.raises(ValueError, match="k_tt"):
        joint_trackability_cost(env, arm_cfg, k_tt)


def test_reader_rejects_shape_mismatch_and_non_joint_action_task() -> None:
    env, arm_cfg = _reader_env()
    env.scene["robot"].data.joint_pos = torch.zeros(2, 7)
    with pytest.raises(ValueError, match="shape"):
        joint_target_squared_error(env, arm_cfg)

    env, arm_cfg = _reader_env(active_terms=("ik_hammer_head",))
    with pytest.raises(ValueError, match="requires exactly"):
        joint_target_squared_error(env, arm_cfg)
