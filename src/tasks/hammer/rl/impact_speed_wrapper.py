"""Pre-clip finite-action guard for the exact impact-speed action signature."""

from __future__ import annotations

import torch
from mjlab.rl import RslRlVecEnvWrapper

from src.tasks.hammer.mdp.impact_speed import (
  ImpactSpeedState,
  validate_speed_goal_weight,
)


IMPACT_SPEED_ACTION_SIGNATURE = ("joint_position", "joint_stiffness", "strike_speed")


class ImpactSpeedVecEnvWrapper(RslRlVecEnvWrapper):
  def step(self, actions: torch.Tensor):
    if not bool(torch.isfinite(actions).all()):
      raise ValueError("raw policy actions must be finite before clipping")
    return super().step(actions)


def wrap_impact_speed_env(env, clip_actions: float | None):
  """Guard only the experimental 13-coordinate environment."""
  terms = tuple(env.unwrapped.action_manager.active_terms)
  if (
    terms == IMPACT_SPEED_ACTION_SIGNATURE
    and env.unwrapped.action_manager.total_action_dim == 13
  ):
    return ImpactSpeedVecEnvWrapper(env, clip_actions=clip_actions)
  return RslRlVecEnvWrapper(env, clip_actions=clip_actions)


def require_impact_speed_experiment_cfg(env_cfg) -> None:
  """Reject an uncalibrated speed interval or event coefficient before env creation."""
  signature = tuple(env_cfg.actions)
  if "strike_speed" in signature and signature != IMPACT_SPEED_ACTION_SIGNATURE:
    raise ValueError("impact speed action signature must be exactly position, stiffness, speed")
  if signature != IMPACT_SPEED_ACTION_SIGNATURE:
    return
  speed = env_cfg.actions["strike_speed"]
  if speed.bounds is None:
    raise ValueError("impact speed bounds must be supplied explicitly")
  # Share the action state's strict domain validation without creating a native environment.
  ImpactSpeedState(1, "cpu", speed.bounds, speed.mode)
  validate_speed_goal_weight(env_cfg.rewards["speed_goal"].weight)


def configure_impact_speed_experiment(
  env_cfg,
  *,
  bounds: tuple[float, float] | None,
  cost: float | None,
  mode: str | None = None,
) -> None:
  """Apply explicit CLI calibration, then fail closed on omissions."""
  if (
    "strike_speed" in env_cfg.actions
    and tuple(env_cfg.actions) != IMPACT_SPEED_ACTION_SIGNATURE
  ):
    raise ValueError("impact speed action signature must be exactly position, stiffness, speed")
  if tuple(env_cfg.actions) != IMPACT_SPEED_ACTION_SIGNATURE:
    if bounds is not None or cost is not None or mode is not None:
      raise ValueError("impact speed options require the experimental task")
    return
  speed = env_cfg.actions["strike_speed"]
  if bounds is not None:
    speed.bounds = tuple(bounds)
  if mode is not None:
    speed.mode = mode
  if cost is not None:
    validate_speed_goal_weight(-cost)
    env_cfg.rewards["speed_goal"].weight = -cost
  require_impact_speed_experiment_cfg(env_cfg)
