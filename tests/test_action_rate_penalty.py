"""Action-history contracts for the hammer reward."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.rewards import action_rate_penalty


def _env(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    active_terms: tuple[str, ...] = ("joint_position", "joint_stiffness"),
    action_term_dim: tuple[int, ...] = (6, 6),
):
    return SimpleNamespace(
        action_manager=SimpleNamespace(
            action=current,
            prev_action=previous,
            active_terms=list(active_terms),
            action_term_dim=list(action_term_dim),
        )
    )


def test_named_action_rate_ignores_gain_only_changes() -> None:
    previous = torch.zeros((2, 12))
    current = previous.clone()
    current[:, 6:] = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.5, 0.25, -0.25],
            [-1.0, 1.0, -0.5, 0.5, -0.25, 0.25],
        ]
    )

    penalty = action_rate_penalty(_env(current, previous), "joint_position")

    torch.testing.assert_close(penalty, torch.zeros(2))


def test_named_action_rate_reads_only_the_requested_manager_slice() -> None:
    previous = torch.zeros((1, 12))
    current = torch.tensor(
        [[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0]]
    )

    penalty = action_rate_penalty(_env(current, previous), "joint_position")

    torch.testing.assert_close(penalty, torch.tensor([2.0]))


def test_unnamed_action_rate_preserves_full_action_behavior() -> None:
    previous = torch.zeros((1, 12))
    current = torch.tensor(
        [[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0]]
    )

    penalty = action_rate_penalty(_env(current, previous))

    torch.testing.assert_close(penalty, torch.tensor([2.5]))


def test_named_action_rate_rejects_an_unknown_term() -> None:
    actions = torch.zeros((1, 12))

    with pytest.raises(ValueError, match="unknown action term"):
        action_rate_penalty(_env(actions, actions.clone()), "renamed_stiffness")


@pytest.mark.parametrize(
    ("current", "previous", "dims", "match"),
    (
        (torch.zeros((2, 12)), torch.zeros((1, 12)), (6, 6), "matching"),
        (torch.zeros(12), torch.zeros(12), (6, 6), "two-dimensional"),
        (torch.zeros((1, 12)), torch.zeros((1, 12)), (6, 5), "dimensions"),
    ),
    ids=("history-shape", "rank", "term-width"),
)
def test_action_rate_rejects_malformed_manager_shapes(
    current: torch.Tensor,
    previous: torch.Tensor,
    dims: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        action_rate_penalty(
            _env(current, previous, action_term_dim=dims), "joint_position"
        )
