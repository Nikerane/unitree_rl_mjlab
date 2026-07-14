"""Pure CaT termination-probability math (Chane-Sane et al., IROS 2024, arXiv:2403.18765).

Port of the reference ``CaT`` helper (github.com/gepetto/constraints-as-terminations,
exts/.../tasks/utils/cat/constraint_manager.py), stripped of the IsaacLab ManagerBase wiring so the
δ math is unit-testable on CPU with no sim. The env-coupled ``ConstraintManager`` (which calls the
constraint funcs each step, stashes δ on the env, and feeds the learner) is the C3 layer; this is the
C0 core.

Per constraint term, per column (e.g. per joint):

    δ = min_p + clamp(c / c_max, 0, 1) · (max_p − min_p)      for c > 0  (else 0)

where ``c`` is the RAW signed margin (value − limit, NOT clamped, NOT a probability) returned by the
constraint func, and ``c_max`` is a Polyak EMA of the batch-max margin:

    c_max ← τ·c_max + (1−τ)·max_envs(c)        (seeded to the first batch max, floored at 1e-6)

This EMA is what makes CaT scale-free (δ depends on the violation relative to the current
population, not on hand-tuned units). It PERSISTS across episode resets — ``reset()`` clears only the
per-step ``probs``/``raw_constraints``, never ``running_maxes`` (it is a batch statistic).

Per-env aggregate δ = MAX over all terms and all columns — a soft logical-OR (``get_probs``).
``max_p`` is the per-term probability ceiling: 1.0 ⇒ a (in-incentive) hard constraint, <1 ⇒ soft.
"""

from __future__ import annotations

import torch


class CaT:
  """Computes per-env termination probabilities δ ∈ [0, max_p] from constraint violations."""

  def __init__(self, tau: float = 0.95, min_p: float = 0.0, device: str | torch.device = "cpu"):
    self.tau = float(tau)
    self.min_p = float(min_p)
    self.device = torch.device(device)
    # EMA of the batch-max margin per term; the scale-free normalizer. PERSISTS across resets.
    self.running_maxes: dict[str, torch.Tensor] = {}
    self.probs: dict[str, torch.Tensor] = {}            # per-term δ, shape (B, C); per-step
    self.raw_constraints: dict[str, torch.Tensor] = {}  # per-term raw margin, for logging; per-step

  def reset(self) -> None:
    """Clear the per-step buffers. ``running_maxes`` (the normalizer) is intentionally kept."""
    self.probs.clear()
    self.raw_constraints.clear()

  @staticmethod
  def delta_map(
    c: torch.Tensor, c_max: torch.Tensor, min_p: float, max_p: float
  ) -> torch.Tensor:
    """Per-column CaT termination probability: ``δ = min_p + clamp(c/c_max, 0, 1)·(max_p − min_p)``
    for ``c > 0``, else 0. Pure (no state) — the single home of the δ formula, shared by ``add``
    (velocity, every-step EMA) and ``CatSoftHook`` (impulse, per-column violation-masked EMA).
    Callers pre-floor ``c_max`` to their own normalizer floor (1e-6 here, ``imp_seed`` in the hook)."""
    normalized = (c / c_max).clamp(0.0, 1.0)  # broadcasts (B,C)/(1,C); c≤0 → clamps to 0
    return torch.where(c > 0.0, min_p + normalized * (max_p - min_p), torch.zeros_like(c))

  def add(self, name: str, constraint: torch.Tensor, max_p: float = 0.1) -> None:
    """Process one constraint term's raw margin ``constraint`` (shape (B,) or (B, C))."""
    if not torch.is_floating_point(constraint):
      constraint = constraint.float()
    if constraint.ndim == 1:
      constraint = constraint.unsqueeze(1)  # (B,) -> (B, 1)
    self.raw_constraints[name] = constraint

    # Batch-max margin per column, floored for numerical stability -> the EMA target.
    constraint_max = constraint.max(dim=0, keepdim=True)[0].clamp(min=1e-6)  # (1, C)
    if name in self.running_maxes:
      self.running_maxes[name] = self.tau * self.running_maxes[name] + (1.0 - self.tau) * constraint_max
    else:
      self.running_maxes[name] = constraint_max  # seed to the first batch max

    self.probs[name] = self.delta_map(constraint, self.running_maxes[name], self.min_p, max_p)

  def get_probs(self) -> torch.Tensor:
    """Per-env termination probability δ, shape (B,): MAX over all terms and columns (soft OR)."""
    if not self.probs:
      return torch.zeros(0, device=self.device)
    return torch.cat(list(self.probs.values()), dim=1).max(dim=1).values
