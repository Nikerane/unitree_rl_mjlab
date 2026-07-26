"""Pure tensor geometry for first-contact hammer-face quality."""

from __future__ import annotations

import torch


def contact_point_quality(
  *,
  found: torch.Tensor,
  force_contact: torch.Tensor,
  position_w: torch.Tensor,
  nail_top_w: torch.Tensor,
  nail_axis_w: torch.Tensor,
  nail_radius_m: float,
  num_slots: int,
  eps: float = 1e-9,
) -> tuple[
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
]:
  """Return contact centroid, radial error, quality, validity, and overflow.

  Contact positions are averaged with positive MuJoCo contact-frame normal
  force. The centroid offset from the nail top is projected onto the plane
  normal to the nail axis before its radial error is measured.
  """
  batch_size = found.shape[0]
  if nail_axis_w.ndim == 1:
    axis_w = nail_axis_w.unsqueeze(0).expand(batch_size, -1)
  else:
    axis_w = nail_axis_w

  overflow = found.amax(dim=1) > num_slots
  weights = torch.where(
    found > 0,
    force_contact[..., 0].clamp_min(0),
    torch.zeros_like(found),
  )

  input_finite = (
    torch.isfinite(found).all(dim=1)
    & torch.isfinite(force_contact).all(dim=(1, 2))
    & torch.isfinite(position_w).all(dim=(1, 2))
    & torch.isfinite(nail_top_w).all(dim=1)
    & torch.isfinite(axis_w).all(dim=1)
  )
  safe_weights = torch.where(
    torch.isfinite(weights), weights, torch.zeros_like(weights)
  )
  safe_positions = torch.where(
    torch.isfinite(position_w), position_w, torch.zeros_like(position_w)
  )
  total_weight = safe_weights.sum(dim=1)
  denominator = torch.where(
    total_weight > 0, total_weight, torch.ones_like(total_weight)
  )
  centroid_w = (
    safe_positions * safe_weights.unsqueeze(-1)
  ).sum(dim=1) / denominator.unsqueeze(-1)

  safe_axis = torch.where(
    torch.isfinite(axis_w), axis_w, torch.zeros_like(axis_w)
  )
  axis_norm = torch.linalg.vector_norm(safe_axis, dim=-1, keepdim=True)
  normalized_axis = safe_axis / torch.where(
    axis_norm > eps, axis_norm, torch.ones_like(axis_norm)
  )

  offset = centroid_w - nail_top_w
  radial_offset = offset - (
    offset * normalized_axis
  ).sum(dim=-1, keepdim=True) * normalized_axis
  radial_error = torch.linalg.vector_norm(radial_offset, dim=-1)

  radius = torch.as_tensor(
    nail_radius_m, dtype=radial_error.dtype, device=radial_error.device
  )
  safe_radius = torch.where(
    torch.isfinite(radius) & (radius > 0), radius, torch.ones_like(radius)
  )
  quality = torch.clamp(
    1.0 - (radial_error / safe_radius) ** 2,
    min=0.0,
    max=1.0,
  )

  valid = (
    input_finite
    & torch.isfinite(radius)
    & (radius > 0)
    & (axis_norm.squeeze(-1) > eps)
    & (total_weight > 0)
    & ~overflow
  )
  centroid_w = torch.where(
    valid.unsqueeze(-1), centroid_w, torch.zeros_like(centroid_w)
  )
  radial_error = torch.where(
    valid, radial_error, torch.zeros_like(radial_error)
  )
  quality = torch.where(valid, quality, torch.zeros_like(quality))
  return centroid_w, radial_error, quality, valid, overflow
