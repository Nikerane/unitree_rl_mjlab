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
from src.tasks.hammer.cat.constraints import joint_impulse_excess, joint_velocity_excess
from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY
from src.tasks.hammer.mdp.impulse_bound import Z1_JOINT_IMPULSE_LIMIT, _joint_count
from src.tasks.hammer.mdp.velocity_bound import Z1_JOINT_VEL_LIMIT, _ARM_CFG

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Reward terms with negative weight. r_neg rides through the discount UNSCALED (Decision 1) — only the
# positive task return is multiplied by (1−δ). This is the complete negative set (CLAUDE.md live
# weights: action_rate=-0.01, joint_pos_limits=-10; every other term is positive).
_NEG_TERMS: tuple[str, ...] = ("action_rate", "joint_pos_limits")


class CatSoftHook(ManagerTermBase):
  """Full-step MetricsTerm: computes soft-CaT δ + r_pos and writes them to ``env.extras``."""

  @staticmethod
  def _validate_params(p: dict) -> None:
    """Loud construction-time guards for expressible misconfigurations (2026-07 review findings).

    Standalone (no env) so it is unit-testable; called from __init__ before any state is built.
    """
    use_vel = bool(p.get("use_vel", True))
    use_impulse = bool(p.get("use_impulse", False))
    if not (use_vel or use_impulse):
      raise RuntimeError(
        "CatSoftHook: use_vel and use_impulse are both False — the hook would emit a shape-(0,) δ "
        "that detonates later inside CatPPO. Enable at least one constraint."
      )
    if use_impulse and "imp_limit" not in p:
      raise RuntimeError(
        "CatSoftHook: use_impulse=True requires an explicit imp_limit (the derived per-joint caps "
        "from derive_impulse_thresholds.py). The Z1_JOINT_IMPULSE_LIMIT placeholder must never be "
        "a silent fallback — enforcing against it would be ~23–69x tighter than the real caps."
      )
    imp_max_p = float(p.get("imp_max_p", 0.0))
    min_p = float(p.get("min_p", 0.0))
    if use_impulse and 0.0 < imp_max_p < min_p:
      raise RuntimeError(
        f"CatSoftHook: imp_max_p={imp_max_p} < min_p={min_p} — δ would be INVERSELY graded "
        "(largest for the smallest violations). Set imp_max_p ≥ min_p (or 0 for log-only)."
      )
    il = p.get("imp_limit", None)
    if (
      use_impulse
      and imp_max_p > 0.0
      and isinstance(il, (int, float))
      and float(il) == Z1_JOINT_IMPULSE_LIMIT
    ):
      raise RuntimeError(
        "CatSoftHook: enforcement (imp_max_p > 0) against the Z1_JOINT_IMPULSE_LIMIT placeholder "
        "(0.1, ~23-69x tighter than the real caps) — env_cfgs passes the placeholder explicitly at "
        "C0, so flipping imp_max_p alone is the exact mistake this guard exists for. Pass the "
        "derived per-joint tensor from derive_impulse_thresholds.py as imp_limit."
      )

  def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    p = cfg.params
    self._validate_params(p)
    self._cat = CaT(tau=p.get("tau", 0.95), min_p=p.get("min_p", 0.0), device=env.device)
    self._max_p: float = float(p.get("max_p", 0.5))
    self._limit: float = float(p.get("limit", Z1_JOINT_VEL_LIMIT))
    self._robot_cfg = p.get("robot_cfg", _ARM_CFG)
    self._robot_cfg.resolve(env.scene)
    # Which constraints this arm enforces. Defaults reproduce the velocity-only -CaT-Soft arm; the
    # -CaT-Impulse arm sets use_vel=False, use_impulse=True for clean attribution (the two are
    # physically correlated, so a combined arm cannot attribute the safety gain — combined is C5).
    self._use_vel: bool = bool(p.get("use_vel", True))
    self._use_impulse: bool = bool(p.get("use_impulse", False))
    # Normalize the limit to a device-resident tensor: the C2 per-joint caps arrive as a (6,)
    # tensor or list from cfg params — a CPU tensor on a CUDA env or a plain list would only
    # detonate at the first constraint eval on GPU.
    self._imp_limit = torch.as_tensor(
      p.get("imp_limit", Z1_JOINT_IMPULSE_LIMIT), dtype=torch.float32, device=env.device
    )
    self._imp_max_p: float = float(p.get("imp_max_p", 0.0))  # 0.0 ⇒ C0 LOG-ONLY (δ_imp ≡ 0)
    # Normalizer DECAY FLOOR only (units no longer critical): the per-column scale self-seeds from
    # the first over-limit sample (CaT.add's own first-batch seeding), so a tiny floor cannot cause
    # cold-start saturation and no "excess p95" statistic is needed from the gate.
    self._imp_seed: float = float(p.get("imp_seed", 1e-3))
    # Slice-safe joint count (SceneEntityCfg.resolve optimizes all-joints to slice(None); len() raises).
    robot = env.scene[self._robot_cfg.name]
    J = _joint_count(robot.data.joint_pos, self._robot_cfg.joint_ids)
    self._imp_cmax = torch.full((1, J), self._imp_seed, device=env.device)
    self._imp_seeded = torch.zeros(1, J, dtype=torch.bool, device=env.device)
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

  def _add_impulse_constraint(self, env: "ManagerBasedRlEnv") -> None:
    """Fold the per-joint impact-impulse margin into the CaT soft-OR with a SPARSE-signal normalizer.

    The impulse margin is nonzero only while a contact window is open or just closed (per-event
    pulse semantics, impulse_bound.py), so the velocity batch-max EMA (which updates every step)
    would decay to the ~1e-6 floor between strikes and saturate δ to max_p on every contact
    (IMPULSE_CAT_IMPL_PLAN.md §4). Instead the normalizer ``_imp_cmax`` is updated PER-COLUMN and
    only on that joint's over-limit samples, FLOORED at ``imp_seed`` (the C0 histogram scale), so
    idle steps cannot collapse it and a violating joint cannot erode a compliant joint's scale.
    δ is written straight into the manager's per-term dicts — the CaT δ-engine itself is unchanged,
    and ``get_probs`` MAXes this term in with any velocity term (soft-OR).

    LOG-ONLY (``imp_max_p == 0``) is a hard invariant: δ_imp ≡ 0 regardless of ``min_p`` (the
    δ formula's min_p floor would otherwise leak min_p·(1−normalized) > 0 — 2026-07 review), the
    normalizer stays untouched, and only the RAW margin is recorded for the gate / histogram.
    """
    c = joint_impulse_excess(env, limit=self._imp_limit)  # (B,J) raw margin
    self._cat.raw_constraints["joint_impulse_excess"] = c
    if c.shape[1] != self._imp_cmax.shape[1]:
      raise RuntimeError(
        f"CatSoftHook: impulse margin has {c.shape[1]} joints (the accumulator's robot_cfg) but the "
        f"hook's normalizer has {self._imp_cmax.shape[1]} (the hook's robot_cfg) — the two joint "
        "sets must match, or δ would silently broadcast per-column normalizers onto wrong joints."
      )
    if self._imp_max_p <= 0.0:  # C0 log-only: a TRUE no-op beyond the raw-margin log
      self._cat.probs["joint_impulse_excess"] = torch.zeros_like(c)
      return
    batch_max = c.clamp_min(0.0).max(dim=0, keepdim=True).values  # (1,J)
    violated = batch_max > 0.0
    # First over-limit sample per column SEEDS the scale directly (CaT.add's first-batch seeding,
    # floored at imp_seed) — no cold-start saturation however small the floor; thereafter a
    # per-column violation-masked EMA (compliant joints keep their cmax).
    first = violated & ~self._imp_seeded
    ema = self._cat.tau * self._imp_cmax + (1.0 - self._cat.tau) * batch_max
    self._imp_cmax = torch.where(
      first, batch_max.clamp_min(self._imp_seed), torch.where(violated, ema, self._imp_cmax)
    )
    self._imp_seeded = self._imp_seeded | violated
    cmax = self._imp_cmax.clamp_min(self._imp_seed)
    normalized = (c / cmax).clamp(0.0, 1.0)
    self._cat.probs["joint_impulse_excess"] = torch.where(
      c > 0.0, self._cat.min_p + normalized * (self._imp_max_p - self._cat.min_p), torch.zeros_like(c)
    )

  def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
    # δ: feed each enabled constraint's RAW per-joint margin into the CaT math (soft-OR over terms/cols).
    if self._use_vel:
      c = joint_velocity_excess(env, limit=self._limit, robot_cfg=self._robot_cfg)  # (B, J)
      self._cat.add("joint_velocity_excess", c, max_p=self._max_p)
    if self._use_impulse:
      self._add_impulse_constraint(env)
    delta = self._cat.get_probs()                   # (B,) ∈ [min_p, max_p] on violation, else 0
    # Deliver to CatPPO via top-level extras (survives wrapper -> runner -> alg unmodified).
    env.extras[CAT_DELTA_KEY] = delta
    env.extras[CAT_R_POS_KEY] = self._compute_r_pos(env)
    return delta                                    # logged as Episode_Metrics/cat_soft mean
