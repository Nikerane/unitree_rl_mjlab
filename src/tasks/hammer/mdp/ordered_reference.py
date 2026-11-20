"""Ordered Cartesian references for the opt-in human-path experiment.

Progress is spatial and remembers the active segment. This module uses no
simulator, controller, reward manager or clock-driven reference motion.
"""
from __future__ import annotations

import math
import torch


class OrderedPolylineReference:
  """Batched, locally ordered path reader with one committing ``update``.

  ``points`` ends at the mathematical nail. Reset only translates that path to
  the live nail; it never scales, deforms, or fits it to the realized head.
  A separate reset event must place the arm at the supplied starting IK pose.
  ``axis_tol`` is a geometric tracking/capture width, not a safety threshold.

  Candidates span sequential segments within the observed displacement budget,
  regardless of waypoint density. Every crossed vertex must lie near the
  observed motion chord, and advance is bounded by ``motion_budget_multiplier``
  times actual displacement (small curvature slack). Thus repeated calls or
  stationary heads cannot walk through a look-ahead window. This sampled
  geometry rule does not certify physical between-sample tracking.
  """

  path_shape = "human_polyline"
  followthrough_mode = "vertical"
  horizontal_detour_m = 0.0

  def __init__(self, points, num_envs: int, device, *, overshoot=.02,
               axis_tol=.01, lookahead_m=.01, motion_budget_multiplier=1.25):
    if type(num_envs) is not int or num_envs < 1:
      raise ValueError("num_envs must be a positive integer")
    if any(not math.isfinite(float(x)) for x in (overshoot, axis_tol, lookahead_m)):
      raise ValueError("reference distances must be finite")
    if overshoot < 0 or axis_tol <= 0 or lookahead_m < 0:
      raise ValueError("require nonnegative overshoot/lookahead and positive width")
    if (not math.isfinite(float(motion_budget_multiplier))
        or isinstance(motion_budget_multiplier, bool) or motion_budget_multiplier < 1):
      raise ValueError("motion budget multiplier must be finite and at least one")
    self.device, self.num_envs = torch.device(device), num_envs
    raw = torch.as_tensor(points, device=self.device, dtype=torch.float32)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 2 or not bool(torch.isfinite(raw).all()):
      raise ValueError("points must be finite (N,3), with N >= 2")
    self.original_points = raw.clone()
    keep = torch.cat((torch.ones(1, dtype=torch.bool, device=self.device),
                      (raw[1:] != raw[:-1]).any(-1)))
    clean = raw[keep]
    if len(clean) < 2:
      raise ValueError("path needs at least two distinct consecutive points")
    self.overshoot, self.axis_tol = float(overshoot), float(axis_tol)
    self.lookahead_m = float(lookahead_m)
    self.motion_budget_multiplier = float(motion_budget_multiplier)
    relative = clean - clean[-1]
    self._approach_length = torch.linalg.vector_norm(relative[1:]-relative[:-1], dim=-1).sum()
    if overshoot:
      relative = torch.cat((relative, relative[-1:]+relative.new_tensor([0., 0., -overshoot])))
    self._points = relative
    self._vectors = relative[1:] - relative[:-1]
    self._lengths = torch.linalg.vector_norm(self._vectors, dim=-1)
    self._cumulative = torch.cat((relative.new_zeros(1), self._lengths.cumsum(0)))
    self._total_length = self._cumulative[-1]
    self._distance = raw.new_zeros(num_envs)
    self._phi = raw.new_zeros(num_envs)
    self._anchored = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
    self._last_step = torch.full((num_envs,), -1, device=self.device, dtype=torch.long)
    self._last_head = raw.new_zeros(num_envs, 3)
    self._head0 = raw.new_zeros(num_envs, 3)
    self._nail = raw.new_zeros(num_envs, 3)
    self._target = raw.new_zeros(num_envs, 3)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self._anchored[ids] = False
    self._distance[ids] = self._phi[ids] = 0.
    self._last_step[ids] = -1

  def _check_head(self, head):
    if head.shape != (self.num_envs, 3):
      raise ValueError(f"head must have shape {(self.num_envs, 3)}")

  def _at_distance(self, distance):
    distance = distance.clamp(0, self._total_length)
    segment = torch.searchsorted(self._cumulative[1:], distance.contiguous(), right=False)
    segment = segment.clamp_max(len(self._lengths)-1)
    u = (distance-self._cumulative[segment])/self._lengths[segment]
    return self._points[segment]+u[..., None]*self._vectors[segment]

  def _projection(self, head):
    """Pure ordered candidate distance, closest waypoint and tracking error."""
    self._check_head(head)
    finite = torch.isfinite(head).all(-1)
    safe_head = torch.where(finite[:, None], head, self._last_head)
    segment = torch.searchsorted(self._cumulative[1:], self._distance.contiguous(), right=False)
    segment = segment.clamp_max(len(self._lengths)-1)
    ids = torch.arange(len(self._lengths), device=self.device)[None, :]
    a, vectors, lengths = self._points[:-1][None], self._vectors[None], self._lengths[None]
    local_head = safe_head-self._nail
    u = ((local_head[:, None]-a)*vectors).sum(-1)/lengths.square()
    distance = self._cumulative[ids]+u.clamp(0, 1)*lengths
    motion = safe_head-self._last_head
    traveled = torch.linalg.vector_norm(motion, dim=-1)
    budget = self._distance+self.motion_budget_multiplier*traveled
    valid = ((ids >= segment[:, None]) & (self._cumulative[ids] <= budget[:, None])
             & self._anchored[:, None] & finite[:, None])
    # Each crossed corner must be compatible with this observed local motion.
    vertices = self._points[1:][None]
    chord_u = ((vertices+self._nail[:, None]-self._last_head[:, None])*motion[:, None]).sum(-1)
    chord_u = chord_u/traveled.square().clamp_min(1e-18)[:, None]
    chord_nearest = self._last_head[:, None]+chord_u.clamp(0, 1)[..., None]*motion[:, None]
    crossed_close = torch.linalg.vector_norm(vertices+self._nail[:, None]-chord_nearest, dim=-1) <= self.axis_tol
    # Candidate j must cross ends of active segments s,...,j-1 in order.
    # Vertices before s are irrelevant; prefix products check every remaining
    # corner without a waypoint-count limit or a Python loop over the path.
    corner_ok = crossed_close | (ids < segment[:, None])
    prefix = corner_ok.to(torch.int32).cumprod(-1).bool()
    crossed = torch.cat((torch.ones_like(prefix[:, :1]), prefix[:, :-1]), -1)
    valid &= crossed
    distance = torch.minimum(distance, budget[:, None])
    distance = torch.maximum(distance, self._cumulative[ids])
    nearest = a+((distance-self._cumulative[ids])/lengths)[..., None]*vectors
    errors = torch.linalg.vector_norm(local_head[:, None]-nearest, dim=-1)
    errors = torch.where(valid, errors, torch.full_like(errors, torch.inf))
    chosen = errors.argmin(-1)
    worlds = torch.arange(self.num_envs, device=self.device)
    candidate = distance[worlds, chosen]
    # No finite candidate before anchoring: retain the declared first segment.
    candidate = torch.where(self._anchored, candidate, torch.zeros_like(candidate))
    nearest = self._at_distance(candidate)+self._nail
    error = torch.linalg.vector_norm(safe_head-nearest, dim=-1)
    # Nonfinite plant state must never acquire progress or poison the cursor.
    # Infinity is an explicit invalid diagnostic; the environment's existing
    # NaN guard remains responsible for rejecting the invalid physical state.
    error = torch.where(finite & self._anchored, error, torch.full_like(error, torch.inf))
    return candidate, nearest, error, traveled

  def update(self, head_w, nail_top_w, episode_step):
    self._check_head(head_w)
    if nail_top_w.shape != head_w.shape or episode_step.shape != (self.num_envs,):
      raise ValueError("nail/episode-step batch shape differs")
    finite = torch.isfinite(head_w).all(-1)
    logical_reset = ~self._anchored | (episode_step < self._last_step)
    # A reset is a state transition even if its first kinematic sample is bad.
    # Clear the old episode before considering whether new anchors are valid.
    self._anchored[logical_reset] = False
    self._distance[logical_reset] = self._phi[logical_reset] = 0.
    fresh = logical_reset & finite & torch.isfinite(nail_top_w).all(-1)
    if bool(fresh.any()):
      self._nail[fresh] = nail_top_w[fresh]
      self._head0[fresh] = nail_top_w[fresh]+self._points[0]
      self._target[fresh] = nail_top_w[fresh]+self._points[-1]
      self._distance[fresh] = self._phi[fresh] = 0.
      self._last_head[fresh] = head_w[fresh]
      self._anchored[fresh] = True
    candidate, _, error, traveled = self._projection(head_w)
    changed_step = episode_step != self._last_step
    eligible = changed_step & ~fresh & (error <= self.axis_tol) & (traveled > 1e-9)
    distance = torch.maximum(self._distance, candidate)
    self._distance.copy_(torch.where(eligible, distance, self._distance))
    self._phi.copy_(self._distance/self._total_length)
    self._last_head.copy_(torch.where((changed_step & finite)[:, None], head_w, self._last_head))
    self._last_step.copy_(episode_step)
    return self._phi.clone()

  def preview(self, head_w, episode_step=None):
    """Pure reward-time progress candidate, sharing update's motion gate."""
    candidate, _, error, traveled = self._projection(head_w)
    eligible = self._anchored & (error <= self.axis_tol) & (traveled > 1e-9)
    distance = torch.maximum(self._distance, candidate)
    return torch.where(eligible, distance, self._distance)/self._total_length

  def peek(self):
    return self._phi.clone()

  def anchor_error(self, head_w):
    """Pure distance from realized head to the declared reset start."""
    self._check_head(head_w)
    return torch.linalg.vector_norm(head_w-self._head0, dim=-1)

  def waypoint(self, phi):
    if phi.shape != (self.num_envs,):
      raise ValueError("phase must have one value per environment")
    return self._at_distance(phi*self._total_length)+self._nail

  def diagnose_ordered(self, head_w):
    distance, nearest, error, _ = self._projection(head_w)
    return distance/self._total_length, nearest, error

  def approach_coordinate(self, head_w):
    distance, _, error, traveled = self._projection(head_w)
    eligible = self._anchored & (error <= self.axis_tol) & (traveled > 1e-9)
    distance = torch.where(eligible, torch.maximum(distance, self._distance), self._distance)
    return (distance/self._approach_length).clamp(0, 1)

  def guidance(self, head_w):
    """Small upcoming waypoint; distinct from the nearest-point path cost."""
    progress = self.preview(head_w)*self._total_length
    return self._at_distance(progress+self.lookahead_m)+self._nail

  # Compatibility seams for existing reward classes. These are ORDERED spatial
  # geometry, never current-height geometry; new task metadata must say so.
  diagnose_current_height = diagnose_ordered
  approach_height_coordinate = approach_coordinate

  def reference_polyline(self, env_id=0, num_points=65):
    if not 0 <= env_id < self.num_envs or num_points < 2:
      raise ValueError("invalid environment or polyline sample count")
    if not bool(self._anchored[env_id]):
      raise RuntimeError("reference must be anchored before sampling")
    distance = torch.linspace(0, 1, num_points, device=self.device)*self._total_length
    return self._at_distance(distance)+self._nail[env_id]
