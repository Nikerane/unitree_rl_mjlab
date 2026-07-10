"""Unit tests for joint_impulse_excess (CaT constraint func, robot-side per-joint impulse).

Mirrors joint_velocity_excess: returns the RAW signed per-joint margin Λ_j − limit, shape (B, J)
(NOT clamped, NOT a probability, NOT a reward) — the CaT manager does all clamp/EMA/normalization.
It reads the substep-accumulated buffer that SubstepImpulseAccumulator stashes on the env.

Post-review hardening: ``limit`` is REQUIRED (no default) — the placeholder constant must never be
reachable as a silent fallback (an enforcement run against the 0.1 placeholder would be ~23–69×
tighter than the real per-joint caps and kill the strike).
"""

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.cat.constraints import joint_impulse_excess
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR


def _env_with_impulse(impulse: torch.Tensor):
  acc = SimpleNamespace(impulse=impulse)
  env = SimpleNamespace(num_envs=impulse.shape[0], device="cpu")
  setattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, acc)
  return env


def test_returns_raw_signed_margin_per_joint():
  imp = torch.tensor([[0.05, 0.20, 0.10, 0.0, 0.0, 0.0]])  # (1,6)
  c = joint_impulse_excess(_env_with_impulse(imp), limit=0.1)
  assert torch.allclose(c, imp - 0.1)
  assert c.shape == (1, 6)


def test_margin_negative_below_limit_not_clamped():
  imp = torch.zeros(2, 6)  # well under any positive limit
  c = joint_impulse_excess(_env_with_impulse(imp), limit=0.1)
  assert (c < 0).all()  # raw signed margin, NOT clamped to >= 0


def test_margin_positive_above_limit():
  imp = torch.full((3, 6), 0.5)
  c = joint_impulse_excess(_env_with_impulse(imp), limit=0.1)
  assert (c > 0).all()
  assert torch.allclose(c, torch.full((3, 6), 0.4))


def test_per_joint_limit_broadcasts():
  imp = torch.tensor([[0.10, 0.10, 0.10, 0.10, 0.10, 0.10]])
  per_joint = torch.tensor([0.05, 0.20, 0.05, 0.20, 0.05, 0.20])  # (J,)
  c = joint_impulse_excess(_env_with_impulse(imp), limit=per_joint)
  assert torch.allclose(c, imp - per_joint)  # joints 0,2,4 over; 1,3,5 under
  assert (c[0, [0, 2, 4]] > 0).all() and (c[0, [1, 3, 5]] < 0).all()


def test_raises_without_accumulator():
  env = SimpleNamespace(num_envs=1, device="cpu")  # no accumulator stashed
  with pytest.raises(RuntimeError, match="SubstepImpulseAccumulator"):
    joint_impulse_excess(env, limit=0.1)


def test_limit_is_required_no_silent_placeholder_fallback():
  # The 0.1 placeholder must NOT be reachable as a default (review finding #10).
  imp = torch.full((1, 6), 0.5)
  with pytest.raises(TypeError):
    joint_impulse_excess(_env_with_impulse(imp))  # no limit -> hard error, not placeholder
