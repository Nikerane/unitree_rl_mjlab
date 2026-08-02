"""Log-only direct-reference measurement helpers.

The helpers in this module may replay the one production direct strike several
times to check deterministic plumbing.  Repeats are repeatability samples of
one controller, never independently varied impact intensities.  This module
cannot enable impulse enforcement or mutate the frozen per-joint caps.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from evaluation.guideline.qualify_reference import load_qualification_cfg
from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.mdp.rewards import clamped_nail_depth


HOLD_STEPS = 6


def _is_exact_frozen_cap(value) -> bool:
  actual = torch.as_tensor(value)
  if not torch.is_floating_point(actual):
    return False
  expected = torch.as_tensor(IMP_J_LIMIT, dtype=actual.dtype, device=actual.device)
  return actual.shape == expected.shape and torch.equal(actual, expected)


def assert_log_only_reference_contract(cfg, *, live_imp_limit=None) -> torch.Tensor:
  """Fail closed unless the config and optional live hook use frozen C0 settings."""
  params = cfg.metrics["cat_soft"].params
  imp_max_p = float(params["imp_max_p"])
  if imp_max_p != 0.0:
    raise RuntimeError(
      f"direct-reference measurement requires imp_max_p=0.0, got {imp_max_p}"
    )
  if not _is_exact_frozen_cap(params["imp_limit"]):
    raise RuntimeError(
      f"configured cap drifted from frozen IMP_J_LIMIT: {params['imp_limit']}"
    )
  if live_imp_limit is not None and not _is_exact_frozen_cap(live_imp_limit):
    raise RuntimeError(
      f"live cap drifted from frozen IMP_J_LIMIT: {live_imp_limit}"
    )
  return torch.tensor(IMP_J_LIMIT, dtype=torch.float32)


def load_direct_reference_c0_cfg():
  """Load the production training C0 config with passive evaluator settings."""
  from mjlab.tasks.registry import load_env_cfg

  import src.tasks  # noqa: F401 - populate the production task registry

  cfg = load_qualification_cfg(load_env_cfg)
  if set(cfg.actions) != {"ik_hammer_head"}:
    raise RuntimeError(
      "direct-reference measurement requires the fixed-impedance ik_hammer_head action only"
    )
  assert_log_only_reference_contract(cfg)
  return cfg


def direct_reference_repeat_indices(repeat_count: int) -> tuple[int, ...]:
  """Return labels for repeated observations of the same nominal strike."""
  if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
    raise ValueError(f"repeat_count must be a fixed positive integer, got {repeat_count!r}")
  return tuple(range(repeat_count))


def run_reference_strikes(cfg, repeat_count: int) -> dict:
  """Measure repeated executions of one default direct reference in log-only C0.

  The returned rows are deterministic repeatability observations.  The helper
  neither varies the reference constructor nor changes the live impulse cap.
  """
  repeat_indices = direct_reference_repeat_indices(repeat_count)
  assert_log_only_reference_contract(cfg)
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device="cpu")
  assert env.num_envs == 1, "run_reference_strikes assumes one environment"
  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  rcfg.resolve(env.scene)
  ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  ncfg.resolve(env.scene)
  nail_cfg = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
  nail_cfg.resolve(env.scene)

  def head() -> torch.Tensor:
    return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)

  def nail_top() -> torch.Tensor:
    return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

  hook = env.metrics_manager.cfg["cat_soft"].func
  frozen_limit = assert_log_only_reference_contract(
    cfg, live_imp_limit=hook._imp_limit
  ).to(device=env.device)
  acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)
  rm = env.reward_manager

  max_excess: list[float] = []
  max_delta: list[float] = []
  depth: list[float] = []
  lambda_to_cap: list[list[float]] = []
  reward_terms: dict[str, float] = {name: 0.0 for name in rm.active_terms}

  for _repeat_index in repeat_indices:
    env.reset()
    ref = SingleStrikeReference(1, env.device)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length()
    peak_lambda = torch.zeros_like(hook._imp_limit)
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
      if bool(env.reset_terminated.any()):
        break

    max_excess.append(float((peak_lambda - frozen_limit).max()))
    max_delta.append(peak_delta)
    depth.append(float(clamped_nail_depth(env, nail_cfg)[0]))
    lambda_to_cap.append((peak_lambda / frozen_limit).tolist())

  return {
    "sample_kind": "deterministic direct-reference repeatability",
    "repeat_indices": list(repeat_indices),
    "frozen_impulse_limit": list(IMP_J_LIMIT),
    "max_excess": max_excess,
    "max_delta": max_delta,
    "cmax": hook._imp_cmax[0].tolist(),
    "depth": depth,
    "lambda_to_cap": lambda_to_cap,
    "reward_terms": reward_terms,
  }
