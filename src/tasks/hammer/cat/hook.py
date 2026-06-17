"""C3: env-side soft-CaT hook (full-step MetricsTerm).

Each control step, this term: evaluates the CaT constraint funcs, folds their raw margins into the C0
``CaT`` δ-math, and writes δ (``extras["cat_delta"]``) and ``r_pos`` (``extras["cat_r_pos"]``) into
``env.extras`` for ``CatPPO`` to consume. It is a **metrics** term, so it CANNOT feed ``reset_buf`` —
a soft violation never ends the episode (Decision 5). It runs at ``manager_based_rl_env.py:441``,
after ``reward_manager.compute`` (so ``_step_reward`` is fresh) and after the decimation loop
(``joint_vel`` current), but before ``_reset_idx`` — so δ describes the just-completed transition.
Returns δ (B,) as the metric value for episode-mean logging. Mirrors the proven
``SubstepPeakJointVel`` / ``CaTJointVelConstraint`` lifecycle in ``velocity_bound.py``.

See docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md §4 + the C3 design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from src.tasks.hammer.cat.constraint_manager import CaT
from src.tasks.hammer.cat.constraints import joint_velocity_excess
from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY
from src.tasks.hammer.mdp.velocity_bound import Z1_JOINT_VEL_LIMIT, _ARM_CFG

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Reward terms with negative weight. r_neg rides through the discount UNSCALED (Decision 1) — only the
# positive task return is multiplied by (1−δ). This is the complete negative set (CLAUDE.md live
# weights: action_rate=-0.01, joint_pos_limits=-10; every other term is positive).
_NEG_TERMS: tuple[str, ...] = ("action_rate", "joint_pos_limits")


class CatSoftHook(ManagerTermBase):
  """Full-step MetricsTerm: computes soft-CaT δ + r_pos and writes them to ``env.extras``."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    p = cfg.params
    self._cat = CaT(tau=p.get("tau", 0.95), min_p=p.get("min_p", 0.0), device=env.device)
    self._max_p: float = float(p.get("max_p", 0.5))
    self._limit: float = float(p.get("limit", Z1_JOINT_VEL_LIMIT))
    self._robot_cfg = p.get("robot_cfg", _ARM_CFG)
    self._robot_cfg.resolve(env.scene)
    # Negative-term column indices into reward_manager._step_reward. Resolved LAZILY on first __call__
    # (manager build order is not guaranteed, so the reward manager may not be fully built in __init__).
    self._neg_idx: list[int] | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    # Clears the per-step probs/raw buffers; running_maxes (the EMA normalizer) intentionally persists
    # (population statistic — see CaT.reset()).
    self._cat.reset()
    return None

  def _neg_indices(self, env: "ManagerBasedRlEnv") -> list[int]:
    if self._neg_idx is None:
      rm = env.reward_manager
      if not hasattr(rm, "_step_reward"):
        raise RuntimeError(
          "CatSoftHook needs reward_manager._step_reward (mjlab 1.4.0 internal). Pin mjlab or adapt "
          "the r_pos recipe in CatSoftHook._compute_r_pos."
        )
      if not getattr(rm, "_scale_by_dt", True):
        raise RuntimeError(
          "CatSoftHook assumes the reward manager scales by dt (reward_buf = rate*dt); r_pos is "
          "dt-scaled to match. reward_manager._scale_by_dt is False -- adapt _compute_r_pos."
        )
      if rm._step_reward.shape[1] != len(rm.active_terms):
        raise RuntimeError("CatSoftHook: reward_manager._step_reward column count != #active_terms.")
      # MF-3 guard: any negative-weight reward term NOT in _NEG_TERMS would be silently discounted by
      # (1-δ) under scale-positives -- the penalty-evasion exploit Decision 1 exists to prevent. The
      # augment-not-replace workflow means new penalty terms are expected; fail loudly if one appears.
      for n in rm.active_terms:
        if rm.get_term_cfg(n).weight < 0 and n not in _NEG_TERMS:
          raise RuntimeError(
            f"CatSoftHook: reward term '{n}' has negative weight {rm.get_term_cfg(n).weight} but is "
            f"not in _NEG_TERMS {_NEG_TERMS}; scale-positives would discount it by (1-δ) "
            f"(penalty-evasion exploit, Decision 1). Add it to _NEG_TERMS."
          )
      self._neg_idx = [rm.active_terms.index(n) for n in _NEG_TERMS if n in rm.active_terms]
    return self._neg_idx

  def _compute_r_pos(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
    """r_pos (sum of positive reward terms), dt-scaled to match the reward CatPPO discounts.

    ``_step_reward[:, k]`` is ``raw·weight`` (un-dt-scaled rate); ``reward_buf`` (what CatPPO scales)
    is dt-scaled. So compute the rate then multiply by ``step_dt``.
    """
    step = env.reward_manager._step_reward          # (B, num_terms), raw*weight (rate)
    r_total_rate = step.sum(dim=1)                  # == reward_buf / dt
    neg = self._neg_indices(env)
    r_neg_rate = step[:, neg].sum(dim=1) if neg else torch.zeros_like(r_total_rate)
    r_pos_rate = r_total_rate - r_neg_rate          # positive-term weighted rates
    return r_pos_rate * env.step_dt                 # dt-scaled domain (matches reward_buf)

  def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
    # δ: feed each constraint's RAW per-joint margin into the C0 CaT math (soft-OR over terms/cols).
    c = joint_velocity_excess(env, limit=self._limit, robot_cfg=self._robot_cfg)  # (B, J)
    self._cat.add("joint_velocity_excess", c, max_p=self._max_p)
    delta = self._cat.get_probs()                   # (B,) ∈ [min_p, max_p] on violation, else 0
    # Deliver to CatPPO via top-level extras (survives wrapper -> runner -> alg unmodified).
    env.extras[CAT_DELTA_KEY] = delta
    env.extras[CAT_R_POS_KEY] = self._compute_r_pos(env)
    return delta                                    # logged as Episode_Metrics/cat_soft mean
