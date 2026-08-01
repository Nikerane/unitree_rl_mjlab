"""Geometry and lifecycle state for the straight Cartesian hammer guideline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


GUIDELINE_NUM_GATES = 6
GUIDELINE_GATE_RADIUS_M = 0.015
GUIDELINE_CORRIDOR_RADIUS_M = 0.005
_ENV_GUIDELINE_ATTR = "_hammer_waypoint_guideline"


def project_to_reference(
  points: torch.Tensor,
  entry: torch.Tensor,
  nail: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Project ``points`` onto the finite segment from ``entry`` to ``nail``.

  Returns normalized segment progress and Euclidean distance to the nearest point on
  that segment, both with shape ``(num_envs,)``. A zero-length reference is treated
  as a point at ``entry`` with zero progress.
  """
  direction = nail - entry
  length_sq = torch.sum(direction * direction, dim=-1)
  valid = length_sq > 0
  safe_length_sq = torch.where(valid, length_sq, torch.ones_like(length_sq))
  progress = torch.sum((points - entry) * direction, dim=-1) / safe_length_sq
  progress = torch.where(valid, progress.clamp(0.0, 1.0), torch.zeros_like(progress))
  closest = entry + progress.unsqueeze(-1) * direction
  return progress, torch.linalg.vector_norm(points - closest, dim=-1)


def advance_ordered_gates(
  prev: torch.Tensor,
  curr: torch.Tensor,
  entry: torch.Tensor,
  nail: torch.Tensor,
  next_gate: torch.Tensor,
  *,
  num_gates: int = GUIDELINE_NUM_GATES,
  radius_m: float = GUIDELINE_GATE_RADIUS_M,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Advance the next ordered gate for every swept Cartesian segment.

  Gates are disks centered at fractions ``1/(num_gates + 1)`` through
  ``num_gates/(num_gates + 1)`` of the entry-to-nail segment. The function only
  recognizes forward crossings and evaluates the linearly interpolated swept point
  at each gate plane. It is stateless: callers own and supply the next unconsumed
  gate index.
  """
  direction = nail - entry
  length_sq = torch.sum(direction * direction, dim=-1)
  valid_reference = length_sq > 0
  safe_length_sq = torch.where(valid_reference, length_sq, torch.ones_like(length_sq))

  prev_progress = torch.sum((prev - entry) * direction, dim=-1) / safe_length_sq
  curr_progress = torch.sum((curr - entry) * direction, dim=-1) / safe_length_sq
  movement = curr - prev
  moving_forward = curr_progress > prev_progress

  index = next_gate.to(torch.long).clone()
  count = torch.zeros_like(index)
  for _ in range(min(num_gates, GUIDELINE_NUM_GATES)):
    active = valid_reference & moving_forward & (index < num_gates)
    gate_progress = (index + 1).to(curr_progress.dtype) / (num_gates + 1)
    crosses_plane = (prev_progress <= gate_progress) & (gate_progress <= curr_progress)
    denominator = curr_progress - prev_progress
    safe_denominator = torch.where(moving_forward, denominator, torch.ones_like(denominator))
    fraction = (gate_progress - prev_progress) / safe_denominator
    swept_at_plane = prev + fraction.unsqueeze(-1) * movement
    gate_center = entry + gate_progress.unsqueeze(-1) * direction
    in_disk = torch.linalg.vector_norm(swept_at_plane - gate_center, dim=-1) <= radius_m
    consumes_gate = active & crosses_plane & in_disk
    index = index + consumes_gate.to(index.dtype)
    count = count + consumes_gate.to(count.dtype)

  return count, index


class WaypointProgressTracker(ManagerTermBase):
  """Own ordered-gate state and update it once per 500 Hz physics substep."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    params = cfg.params
    self._robot = env.scene[params["robot_cfg"].name]
    self._nail_entity = env.scene[params["nail_cfg"].name]
    self._head_site_ids = params["robot_cfg"].site_ids
    self._nail_site_ids = params["nail_cfg"].site_ids
    self._dec = int(env.cfg.decimation)
    self._i = 0

    head = self._head_position()
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.entry = torch.zeros_like(head)
    self.nail = torch.zeros_like(head)
    self.previous_head = torch.zeros_like(head)
    self.next_gate = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.newly_crossed = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
    self.disarmed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    setattr(env, _ENV_GUIDELINE_ATTR, self)

  def _head_position(self) -> torch.Tensor:
    return self._robot.data.site_pos_w[:, self._head_site_ids].squeeze(1)

  def _nail_position(self) -> torch.Tensor:
    return self._nail_entity.data.site_pos_w[:, self._nail_site_ids].squeeze(1)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    idx = slice(None) if env_ids is None else env_ids
    self.initialized[idx] = False
    self.entry[idx] = 0.0
    self.nail[idx] = 0.0
    self.previous_head[idx] = 0.0
    self.next_gate[idx] = 0
    self.newly_crossed[idx] = 0
    self.disarmed[idx] = False
    return None

  def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
    del params
    if self._i % self._dec == 0:
      self.newly_crossed.zero_()

    head = self._head_position()
    nail = self._nail_position()
    was_initialized = self.initialized.clone()
    self.disarmed |= getattr(env, _ENV_FIRST_STRIKE_ATTR).started
    eligible = was_initialized & ~self.disarmed
    crossed, advanced = advance_ordered_gates(
      self.previous_head,
      head,
      self.entry,
      self.nail,
      self.next_gate,
    )
    crossed = torch.where(eligible, crossed, torch.zeros_like(crossed))
    self.next_gate = torch.where(eligible, advanced, self.next_gate)
    self.newly_crossed += crossed

    initializing = ~was_initialized
    self.entry = torch.where(initializing[:, None], head, self.entry)
    self.nail = torch.where(initializing[:, None], nail, self.nail)
    self.previous_head.copy_(head)
    self.initialized |= initializing
    self._i += 1
    return crossed.to(dtype=head.dtype)


def _guideline_tracker(env: "ManagerBasedRlEnv") -> WaypointProgressTracker:
  tracker = getattr(env, _ENV_GUIDELINE_ATTR, None)
  if not isinstance(tracker, WaypointProgressTracker):
    raise RuntimeError(
      "guideline reader needs WaypointProgressTracker wired as an always-on "
      "per-substep metric"
    )
  return tracker


def _reader_geometry(
  tracker: WaypointProgressTracker,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  head = tracker._head_position()
  current_nail = tracker._nail_position()
  initialized = tracker.initialized[:, None]
  entry = torch.where(initialized, tracker.entry, head)
  nail = torch.where(initialized, tracker.nail, current_nail)
  return head, entry, nail


def next_gate_vector(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Vector from the current hammer head to the next unvisited gate."""
  tracker = _guideline_tracker(env)
  head, entry, nail = _reader_geometry(tracker)
  fraction = (tracker.next_gate + 1).to(dtype=head.dtype) / (
    GUIDELINE_NUM_GATES + 1
  )
  gate = entry + fraction[:, None] * (nail - entry)
  vector = gate - head
  complete = tracker.next_gate >= GUIDELINE_NUM_GATES
  return torch.where(complete[:, None], torch.zeros_like(vector), vector)


def completed_gate_fraction(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Normalized ordered-gate completion, shaped as one observation column."""
  tracker = _guideline_tracker(env)
  return (tracker.next_gate.float() / GUIDELINE_NUM_GATES).unsqueeze(-1)


def guideline_perpendicular_error(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Distance from the current head to the frozen finite reference segment."""
  tracker = _guideline_tracker(env)
  head, entry, nail = _reader_geometry(tracker)
  _, distance = project_to_reference(head, entry, nail)
  return distance.unsqueeze(-1)


def ordered_gate_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Read the current control window's one-shot gate pulse, normalized to one."""
  tracker = _guideline_tracker(env)
  return tracker.newly_crossed.float() / GUIDELINE_NUM_GATES
