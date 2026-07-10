"""Shared helper for the C2 impulse-CaT enforcement gate (IMPULSE_CAT_IMPL_PLAN.md C2).

``run_reference_strikes`` is **modeled on — not refactoring —**
``derive_impulse_thresholds.py:main()``: same env-build, site-helper, and open-loop
``SingleStrikeReference`` playback idiom, reused here to drive scripted reference strikes against a
live ``z1_hammer_env_cfg(cat_impulse=True)`` build. Unlike the C0 gate (which cross-checks the SHIPPED
accumulators against an independent script-side reimplementation), this helper reads ONLY shipped/live
objects — the ``SubstepImpulseAccumulator`` stashed on the env (``acc.impulse``, the per-event-pulse
Λ_j), the live ``CatSoftHook`` instance (``env.metrics_manager.cfg["cat_soft"].func`` — mjlab replaces
the class with the constructed instance in the manager's deepcopied cfg), ``env.extras["cat_delta"]``
(the hook's published δ), and ``reward_manager._step_reward`` (the per-term weighted rate, the same
accessor ``validate_rewards.py`` uses for its per-term table) — never a parallel reimplementation of
Λ_j, δ, or the reward terms.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.mdp.rewards import clamped_nail_depth

# Post-playback hold steps, mirrors derive_impulse_thresholds.py's HOLD_STEPS (lets a strike that
# lands on the last scripted waypoint finish settling / the success termination fire).
HOLD_STEPS = 6

_NAIL_CFG = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
_BAND_LO, _BAND_HI = 0.8, 1.0  # [3] band_fraction: worst-joint Λ within this fraction of J_limit


def run_reference_strikes(
  cfg, heights: tuple[float, ...], limit_override: float | None = None
) -> dict:
  """Drive one open-loop reference strike per height against a fresh env; return C2 gate stats.

  ``limit_override`` (a scalar factor) scales the LIVE hook's per-joint impulse limit in place,
  once before any strike: ``hook._imp_limit = hook._imp_limit * limit_override``. A fresh env is
  built per call, so there is nothing to restore.

  Returns a dict:
    max_excess:    list[float], one per strike (heights order) — worst-joint peak Λ_j minus the
                   (possibly overridden) limit, signed (negative ⇒ compliant).
    max_delta:     list[float], one per strike — peak ``env.extras["cat_delta"]`` observed.
    cmax:          list[float] (per joint) — the hook's ``_imp_cmax`` normalizer snapshotted ONCE
                   after all strikes.
    depth:         list[float], one per strike — final clamped nail depth (m).
    band_fraction: float — fraction of strikes whose worst-joint Λ_j/limit_j ratio falls in
                   [0.8, 1.0].
    binding_ratio: float — max over all strikes of the worst-joint Λ_j/limit_j ratio.
    reward_terms:  dict[str, float] — per-term weighted reward, SUMMED over every strike in this
                   run (dt-scaled to match ``reward_manager._episode_sums``' convention).
  """
  # auto_reset=False (mjlab-native, matches validate_rewards.py Phase M): with the default
  # auto_reset=True, a strike that triggers the nail_driven success termination is reset IN-STEP
  # (_reset_idx runs before env.step() returns), which zeroes the SubstepImpulseAccumulator (and
  # the nail position) before this function ever gets to read them — silently discarding exactly
  # the terminal strike's Λ_j (2026-07 review finding #5, see impulse_bound.py / validate_rewards.py
  # Phase M). auto_reset=False keeps the terminal state readable; the per-height loop below still
  # calls env.reset() explicitly before the next strike.
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device="cpu")
  assert env.num_envs == 1, "run_reference_strikes assumes num_envs=1 (single reference strike)"
  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  rcfg.resolve(env.scene)
  ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  ncfg.resolve(env.scene)

  def head() -> torch.Tensor:
    return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)

  def nail_top() -> torch.Tensor:
    return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

  # Live hook instance (post-construction, the manager replaced the class with this instance).
  hook = env.metrics_manager.cfg["cat_soft"].func
  if limit_override is not None:
    hook._imp_limit = hook._imp_limit * limit_override

  acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)  # SHIPPED SubstepImpulseAccumulator
  rm = env.reward_manager

  max_excess: list[float] = []
  max_delta: list[float] = []
  depth: list[float] = []
  ratios: list[float] = []  # per-strike worst-joint Λ_j / limit_j, feeds band_fraction/binding_ratio
  reward_terms: dict[str, float] = {name: 0.0 for name in rm.active_terms}

  for h in heights:
    env.reset()
    ref = SingleStrikeReference(1, env.device, approach_height=h)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length()
    peak_lambda = torch.zeros_like(hook._imp_limit)  # (J,) peak per-event-pulse Λ_j this strike
    peak_delta = 0.0
    for k in range(1, n + HOLD_STEPS + 1):
      target = ref.playback_target(min(k, n))
      action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
      env.step(action)
      peak_lambda = torch.maximum(peak_lambda, acc.impulse[0])
      delta = env.extras.get("cat_delta")
      if delta is not None:
        peak_delta = max(peak_delta, float(delta[0]))
      for idx, name in enumerate(rm.active_terms):
        reward_terms[name] += float(rm._step_reward[0, idx]) * env.step_dt
      if bool(env.reset_terminated.any()):  # success fired; auto_reset=False keeps state readable
        break

    limit = hook._imp_limit  # (J,) — the possibly-overridden limit, read live (not cached)
    max_excess.append(float((peak_lambda - limit).max()))
    max_delta.append(peak_delta)
    depth.append(float(clamped_nail_depth(env, _NAIL_CFG)[0]))
    ratios.append(float((peak_lambda / limit.clamp_min(1e-9)).max()))

  cmax = hook._imp_cmax[0].tolist()  # snapshot AFTER all strikes
  band_fraction = sum(1 for r in ratios if _BAND_LO <= r <= _BAND_HI) / len(ratios) if ratios else 0.0
  binding_ratio = max(ratios) if ratios else 0.0

  return {
    "max_excess": max_excess,
    "max_delta": max_delta,
    "cmax": cmax,
    "depth": depth,
    "band_fraction": band_fraction,
    "binding_ratio": binding_ratio,
    "reward_terms": reward_terms,
  }
