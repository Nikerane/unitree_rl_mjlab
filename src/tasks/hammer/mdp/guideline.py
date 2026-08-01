"""Pure geometry for the straight Cartesian hammer guideline."""

from __future__ import annotations

import torch


GUIDELINE_NUM_GATES = 6
GUIDELINE_GATE_RADIUS_M = 0.015
GUIDELINE_CORRIDOR_RADIUS_M = 0.005


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
