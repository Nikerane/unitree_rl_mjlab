"""Hold a single Z1 hammer reset pose STATICALLY in viser, so you can orbit and inspect it.

Loads the full scene (robot + nail + floor + lights), sets the arm to the chosen pose, and holds
it with a zero DiffIK action -- the arm is gravity-compensated (gravcomp=1) so it does not droop --
with terminations off (no strike, no reset). Browser-based viser; works on macOS, no mjpython.

  python scripts/view_pose.py --pose near       # current NEAR_NAIL_JOINT_POS (the committed reset)
  python scripts/view_pose.py --pose vertical    # 6-DoF orientation-aware IK candidate (dead-vertical strike)

Open the URL it prints; drag to rotate, scroll to zoom. Run two windows to compare the two poses.
"""

from __future__ import annotations

import dataclasses
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

# 6-DoF orientation-aware IK solution (pos at nail + strike axis straight down -> 0.05 deg tilt,
# vs the current NEAR_NAIL's 7.4 deg oblique). See IMPULSE_CAT_IMPL_PLAN.md §6b. Candidate only --
# not yet adopted as NEAR_NAIL_JOINT_POS (that needs an exact re-solve + the Phase-M re-verify).
VERTICAL_POSE = {
  "joint1": -0.10821, "joint2": 2.01760, "joint3": -1.83955,
  "joint4": 1.39277, "joint5": 0.0, "joint6": 1.23046, "jointGripper": -0.001,
}


@dataclass(frozen=True)
class Cfg:
  pose: str = "near"          # "near" (committed) | "vertical" (IK candidate)
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

  if cfg.pose == "vertical":
    robot = env_cfg.scene.entities["robot"]
    try:
      robot.init_state.joint_pos = dict(VERTICAL_POSE)
    except (dataclasses.FrozenInstanceError, AttributeError):
      robot.init_state = dataclasses.replace(robot.init_state, joint_pos=dict(VERTICAL_POSE))
  elif cfg.pose != "near":
    raise ValueError(f"--pose must be 'near' or 'vertical', got {cfg.pose!r}")

  agent_cfg = load_rl_cfg("Unitree-Z1-Hammer")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  policy = HoldPolicy(env.unwrapped.action_manager.total_action_dim, env.unwrapped.device)
  print(f"[view_pose] holding pose='{cfg.pose}' in {cfg.viewer}; open the URL and orbit to inspect")

  if cfg.viewer == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()


if __name__ == "__main__":
  main(tyro.cli(Cfg))
