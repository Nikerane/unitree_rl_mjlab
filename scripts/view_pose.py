"""Hold the Z1 hammer reset pose STATICALLY in viser, so you can orbit and inspect it.

Loads the full scene (robot + nail + floor + lights), sets the arm to NEAR_NAIL_JOINT_POS (the
committed reset), and holds it with a zero DiffIK action -- the arm is gravity-compensated
(gravcomp=1) so it does not droop -- with terminations off (no strike, no reset). Browser-based
viser; works on macOS, no mjpython.

  python scripts/view_pose.py

(The old ``--pose vertical`` IK candidate was REMOVED 2026-07-14: the 2026-07-06 L6 re-solve
established that a dead-vertical strike at the floor nail is kinematically infeasible with the
fixture grasp — the candidate was stale-rejected, not pending.)

Open the URL it prints; drag to rotate, scroll to zoom.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg


@dataclass(frozen=True)
class Cfg:
  viewer: str = "viser"       # "viser" (browser, mac-friendly) | "native"
  device: str | None = None


class HoldPolicy:
  """Zero DiffIK action -> hold the reset pose (gravcomp arm, so no droop)."""

  def __init__(self, action_dim: int, device: str | torch.device):
    self._n, self._device = action_dim, device

  def __call__(self, obs):  # noqa: ARG002
    del obs
    return torch.zeros((1, self._n), device=self._device)


def main(cfg: Cfg = Cfg()) -> None:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = z1_hammer_env_cfg(play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.terminations = {}                       # static: no strike, no reset

  agent_cfg = load_rl_cfg("Unitree-Z1-Hammer")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  policy = HoldPolicy(env.unwrapped.action_manager.total_action_dim, env.unwrapped.device)
  print(f"[view_pose] holding NEAR_NAIL reset pose in {cfg.viewer}; open the URL and orbit to inspect")

  if cfg.viewer == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()


if __name__ == "__main__":
  main(tyro.cli(Cfg))
