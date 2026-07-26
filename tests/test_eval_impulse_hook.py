"""Adversarial-review I12 (2026-07-14): positive control for ``scripts/eval_impulse.py``'s
pre-reset snapshot hook — the C3-CSV correctness core.

The bug class the hook exists for: with mjlab's default ``auto_reset=True``, an env that
terminates inside ``env.step()`` is reset IN-STEP (``_reset_idx`` runs after
``metrics_manager.compute()`` but before ``env.step()`` returns), zeroing the shipped
accumulators' buffers. Reading Λ_j / delivered / depth after ``env.step()`` returns is therefore
too late for exactly the episodes the eval cares about (the completed ones). The hook patches
``metrics_manager.compute`` to snapshot the live buffers every control step, so the snapshot
taken on the terminal step still holds the pre-reset values.

This test drives the open-loop reference strike to success in an ``auto_reset=True`` env (the
same configuration ``eval_impulse.py`` runs under) and asserts BOTH halves of the control:
the snapshot holds nonzero strike measurements, AND the live buffers are already zeroed when
``env.step()`` returns — i.e. the hook is load-bearing, not decorative. Task 11 ran this
control interactively; it was never committed (the I12 finding).
"""

from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.references import SingleStrikeReference

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[1]

# scripts/ is not a package — load the script as a module the same way its .sh runner shells it.
_spec = importlib.util.spec_from_file_location(
  "eval_impulse_script", _REPO / "scripts" / "eval_impulse.py"
)
eval_impulse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_impulse)

HOLD_STEPS = 6  # post-playback settle steps, mirrors derive_impulse_thresholds.py


def _fixed_reference_action_tape() -> list[torch.Tensor]:
  task = eval_impulse.QUALITY_ARM_TASKS["F8"]
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    robot = env.scene["robot"]
    nail = env.scene["nail_block"]
    head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    head_cfg.resolve(env.scene)
    nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    nail_cfg.resolve(env.scene)
    head = lambda: robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)
    nail_top = lambda: nail.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)
    env.reset()
    reference = SingleStrikeReference(1, env.device, approach_height=0.10)
    reference.update(head(), nail_top(), torch.zeros(1, dtype=torch.long))
    tape = []
    for step in range(1, reference.playback_length() + HOLD_STEPS + 1):
      target = reference.playback_target(min(step, reference.playback_length()))
      action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
      tape.append(action.detach().clone())
      env.step(action)
      if bool(env.reset_terminated.any()):
        break
    assert bool(env.reset_terminated.any()), "reference tape did not reach contact/success"
    return tape
  finally:
    env.close()


def _single_control_trace(
  task: str, *, strict_quality: bool, action_tape: list[torch.Tensor]
) -> dict:
  if strict_quality:
    _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=True)
  else:
    cfg = eval_impulse.load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = float(
    len(action_tape) * cfg.sim.mujoco.timestep * cfg.decimation
  )
  cfg.metrics["cat_soft"].params["imp_max_p"] = 0.0
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    snapshot = eval_impulse._install_episode_hook(env)
    collector = eval_impulse._SampledTraceCollector(
      env,
      snapshot=snapshot,
      treatment=(
        eval_impulse.QUALITY_TASK_TO_ARM[task] if strict_quality else "F8"
      ),
      task=task,
      gamma=0.99,
      event_i_ref_n_s=0.3088,
      nail_geometry={
        "nail_axis": [0.0, 0.0, -1.0],
        "nail_xy_m": [0.5, 0.0],
        "nail_radius_m": 0.012,
        "source_sha256": "0" * 64,
      },
    )
    env.reset()
    for action in action_tape:
      env.step(action)
      if collector.completed:
        break
    assert len(collector.completed) == 1
    return copy.deepcopy(collector.completed[0])
  finally:
    env.close()


def _without_raw_quality_sensor_channels(trace: dict) -> dict:
  payload = eval_impulse._physical_trace_payload(trace)
  payload = copy.deepcopy(payload)
  for key in (
    "quality_found_count", "quality_normal_force_n",
    "quality_contact_position_m", "quality_contact_normal",
  ):
    del payload["physical"][key]
  return payload


def test_episode_hook_snapshots_pre_reset_buffers_on_the_terminal_step():
  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 1
  # auto_reset stays at its default (True) — the exact configuration the hook exists for.
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    snap = eval_impulse._install_episode_hook(env)

    robot = env.scene["robot"]
    nail_e = env.scene["nail_block"]
    head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    head_cfg.resolve(env.scene)
    nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    nail_cfg.resolve(env.scene)

    def head() -> torch.Tensor:
      return robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)

    def nail_top() -> torch.Tensor:
      return nail_e.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)

    env.reset()
    ref = SingleStrikeReference(1, env.device, approach_height=0.10)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length()

    struck = False
    for k in range(1, n + HOLD_STEPS + 1):
      target = ref.playback_target(min(k, n))
      action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
      env.step(action)
      if bool(env.reset_terminated.any()):
        struck = True
        break
    assert struck, "reference strike never triggered the success termination — cannot exercise the hook"

    # The terminal step just ran: compute() (hook fires, snapshot taken) -> _reset_idx (buffers
    # zeroed) -> env.step() returned. Positive control, both halves:
    # (a) the snapshot holds the pre-reset strike measurements...
    assert snap["perjoint"] is not None
    assert float(snap["perjoint"].max()) > 0.0, "hook snapshot lost the per-joint Λ episode peaks"
    assert float(snap["delivered"][0]) > 0.0, "hook snapshot lost the delivered impulse"
    assert float(snap["depth"][0]) > 0.02, (
      f"hook snapshot depth {float(snap['depth'][0]):.4f} m below the success regime — "
      "the strike terminated the episode, so the pre-reset depth must show it"
    )
    # (b) ...while the live buffers are ALREADY ZEROED — reading them post-step (the naive
    # pattern the hook replaces) would report an empty episode. If this half ever fails, the
    # reset path changed and the hook may no longer be needed; if (a) fails, the eval CSV is
    # silently recording zeros for every completed episode.
    acc = getattr(env, eval_impulse._ENV_SUBSTEP_IMPULSE_ATTR)
    dacc = getattr(env, eval_impulse._ENV_SUBSTEP_DELIVERED_ATTR)
    assert float(acc._episode_peak_perjoint.max()) == 0.0, (
      "live Λ buffer survived the in-step auto-reset — the hook's premise changed"
    )
    assert float(dacc.delivered[0]) == 0.0, (
      "live delivered buffer survived the in-step auto-reset — the hook's premise changed"
    )
  finally:
    env.close()


def test_install_episode_hook_requires_both_accumulators():
  # The guard path: a non-impulse task (no stashed accumulators) must be rejected loudly at
  # install time, not fail with silent Nones at CSV-write time.
  class _FakeEnv:
    pass

  with pytest.raises(RuntimeError, match="requires BOTH"):
    eval_impulse._install_episode_hook(_FakeEnv())


@pytest.mark.integration
def test_fixed_action_tape_preserves_physics_across_strict_quality_arms():
  """Passive quality sensing changes no plant channel; payouts are intentionally absent."""
  action_tape = _fixed_reference_action_tape()
  traces = {
    arm: _single_control_trace(task, strict_quality=True, action_tape=action_tape)
    for arm, task in eval_impulse.QUALITY_ARM_TASKS.items()
  }
  assert eval_impulse.compare_action_tape_physics(traces)["arms"] == (
    "D0", "F0", "F8", "FQ",
  )
  assert len({json.dumps(trace["action_tape"]) for trace in traces.values()}) == 1
  assert traces["F8"]["first_strike"]["started"] is True
  assert max(traces["F8"]["episode_peak_lambda"]) > 0.0

  uninstrumented_f8 = _single_control_trace(
    eval_impulse.QUALITY_ARM_TASKS["F8"],
    strict_quality=False,
    action_tape=action_tape,
  )
  assert eval_impulse._canonical_digest(
    _without_raw_quality_sensor_channels(traces["F8"])
  ) == eval_impulse._canonical_digest(
    _without_raw_quality_sensor_channels(uninstrumented_f8)
  )
