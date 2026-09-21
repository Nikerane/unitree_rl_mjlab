"""Episode-scoped impact-speed goal selected by one policy action."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ImpactSpeedState:
  """Own the selected speed and once-per-episode accounting for each world."""

  def __init__(
    self,
    num_envs: int,
    device: str | torch.device,
    bounds: tuple[float, float],
    mode: Literal["external", "policy"],
  ) -> None:
    if (
      not isinstance(bounds, tuple)
      or len(bounds) != 2
      or any(isinstance(x, bool) or not isinstance(x, Real) for x in bounds)
      or not all(math.isfinite(float(x)) for x in bounds)
      or not 0 < bounds[0] < bounds[1]
    ):
      raise ValueError("bounds must be two finite reals with 0 < lower < upper")
    if mode not in ("external", "policy"):
      raise ValueError("mode must be 'external' or 'policy'")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
      raise ValueError("num_envs must be a positive integer")
    self._bounds = (float(bounds[0]), float(bounds[1]))
    self.mode = mode
    self.goal_mps = torch.full((num_envs,), sum(self._bounds) / 2, device=device)
    self.pending = torch.zeros(num_envs, device=device)
    self.cost_consumed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.reset(None)

  @property
  def bounds(self) -> tuple[float, float]:
    return self._bounds

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    if self.mode == "external":
      lo, hi = self.bounds
      self.goal_mps[ids] = lo + torch.rand_like(self.goal_mps[ids]) * (hi - lo)
      self.pending[ids] = 0.0
    else:
      self.goal_mps[ids] = sum(self.bounds) / 2
      self.pending[ids] = 1.0
    self.cost_consumed[ids] = False

  def select(self, raw: torch.Tensor) -> None:
    if not isinstance(raw, torch.Tensor) or raw.shape != (self.goal_mps.numel(), 1):
      raise ValueError(f"raw action shape must be {(self.goal_mps.numel(), 1)}")
    if not torch.is_floating_point(raw) or not bool(torch.isfinite(raw).all()):
      raise ValueError("raw action must contain finite floating values")
    lo, hi = self.bounds
    candidate = lo + 0.5 * (raw[:, 0].to(self.goal_mps.device).clamp(-1, 1) + 1) * (hi - lo)
    self.goal_mps.copy_(torch.where(self.pending.bool(), candidate, self.goal_mps))
    self.pending.zero_()

  def observation(self) -> torch.Tensor:
    lo, hi = self.bounds
    normalized = 2 * (self.goal_mps - lo) / (hi - lo) - 1
    return torch.stack((normalized, self.pending), dim=-1)


@dataclass(kw_only=True)
class ImpactSpeedActionCfg(ActionTermCfg):
  bounds: tuple[float, float] | None = None
  mode: Literal["external", "policy"] = "policy"

  def build(self, env: ManagerBasedRlEnv) -> ImpactSpeedAction:
    return ImpactSpeedAction(self, env)


class ImpactSpeedAction(ActionTerm):
  cfg: ImpactSpeedActionCfg

  def __init__(self, cfg: ImpactSpeedActionCfg, env: ManagerBasedRlEnv):
    if cfg.bounds is None:
      raise ValueError("impact speed bounds must be provided explicitly")
    super().__init__(cfg, env)
    self.state = ImpactSpeedState(self.num_envs, self.device, cfg.bounds, cfg.mode)
    self.raw_actions = torch.zeros((self.num_envs, 1), device=self.device)
    self.processed_actions = torch.zeros_like(self.raw_actions)

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self.raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self.state.select(actions)
    self.raw_actions.copy_(actions)
    self.processed_actions.copy_(actions.clamp(-1, 1))

  def apply_actions(self) -> None:
    pass

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.state.reset(env_ids)
    self.raw_actions[ids] = 0.0
    self.processed_actions[ids] = 0.0


def impact_speed_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.get_term("strike_speed").state.observation()[:, :1]


def impact_speed_pending(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.get_term("strike_speed").state.pending[:, None]


IMPACT_SPEED_OUTCOME_KEY = "impact_speed_outcome"


def consume_speed_cost(
  state: ImpactSpeedState,
  *,
  finalized: torch.Tensor,
  started: torch.Tensor,
  measured_mps: torch.Tensor | None,
  done: torch.Tensor,
) -> torch.Tensor:
  """Charge the first finalized strike or reset-boundary failure once per world.

  ``started`` is supplied for the caller's miss/incomplete label. Both failures
  have the same unit cost. A measured speed is required only for a new event.
  """
  shape = state.goal_mps.shape
  if any(not isinstance(x, torch.Tensor) or x.shape != shape or x.dtype != torch.bool
         for x in (finalized, started, done)):
    raise ValueError("finalized, started, and done must be boolean per-world tensors")
  eligible = ~state.cost_consumed
  event = eligible & finalized
  failure = eligible & ~finalized & done
  if measured_mps is None:
    if bool(event.any()):
      raise ValueError("finalized speed measurement is missing")
    measured = torch.zeros_like(state.goal_mps)
  else:
    if measured_mps.shape != shape or not torch.is_floating_point(measured_mps):
      raise ValueError("measured speed must be a floating per-world tensor")
    if bool((event & ~torch.isfinite(measured_mps)).any()):
      raise ValueError("finalized speed measurement must be finite")
    measured = torch.where(event, measured_mps, state.goal_mps)
  lo, hi = state.bounds
  error = ((measured - state.goal_mps) / (hi - lo)).square().clamp_max(1)
  cost = torch.where(event, error, torch.zeros_like(error))
  cost = torch.where(failure, torch.ones_like(cost), cost)
  state.cost_consumed |= event | failure
  return cost


class ImpactSpeedCost(ManagerTermBase):
  """Event-unit speed cost exposed as a rate for dt-scaled RewardManager."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
    validate_speed_goal_weight(cfg.weight)
    super().__init__(env)

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    del params
    tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR, None)
    if tracker is None:
      raise RuntimeError("ImpactSpeedCost requires the first-strike tracker")
    state = env.action_manager.get_term("strike_speed").state
    done = env.reset_buf.bool()
    eligible = ~state.cost_consumed
    event = eligible & tracker.finalized
    failure = eligible & ~tracker.finalized & done
    measured = tracker.v_precontact
    cost = consume_speed_cost(
      state, finalized=tracker.finalized, started=tracker.started,
      measured_mps=measured, done=done,
    )
    # Reward computation precedes auto-reset. Clone all transition evidence now:
    # tracker and speed state are cleared for reset worlds before step() returns.
    env.extras[IMPACT_SPEED_OUTCOME_KEY] = {
      "charged": (event | failure).clone(),
      "finalized": event.clone(),
      "valid_measured": event.clone(),
      "started": tracker.started.clone(),
      "miss": (failure & ~tracker.started).clone(),
      "incomplete": (failure & tracker.started).clone(),
      "done": done.clone(),
      "productive": (event & tracker.productive).clone(),
      "goal_mps": state.goal_mps.clone(),
      "measured_mps": torch.where(event, measured, torch.full_like(measured, float("nan"))).clone(),
      "cost": cost.clone(),
    }
    return cost / env.step_dt


def validate_speed_goal_weight(weight: float) -> None:
  if (
    isinstance(weight, bool)
    or not isinstance(weight, Real)
    or not math.isfinite(float(weight))
    or weight >= 0
  ):
    raise ValueError("speed_goal weight must be an explicit finite negative coefficient")
