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
