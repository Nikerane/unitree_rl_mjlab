"""Watch the scripted single-strike reference (plan T1) drive the nail, open-loop.

This is NOT a trained policy — it replays the coded SingleStrikeReference
(src/tasks/hammer/mdp/references.py) by commanding the IK action toward each
phase waypoint, so you can visually confirm the reference path works before
spending GPU on training. Mirrors scripts/play.py's env construction; the
"policy" reads head + nail state from the scene and returns the open-loop
reference action.

Run (browser-based viser viewer — works on macOS, no mjpython needed):
    python scripts/play_reference.py
    python scripts/play_reference.py --num-envs 4 --approach-height 0.10

Then open the viser URL it prints. Episodes auto-reset on success, so the
strike repeats; screen-record that window to capture it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.mdp.references import get_strike_reference

TASK_ID = "Unitree-Z1-Hammer"


@dataclass(frozen=True)
class Cfg:
  num_envs: int = 1
  approach_height: float = 0.15
  viewer: str = "viser"  # "viser" (browser, mac-friendly) or "native"
  show_line: bool = True  # draw the reference head path as a red 3D polyline (viser; GUI-toggleable)
  device: str | None = None


def collect_head_path(env, policy: "ReferencePolicy") -> np.ndarray:
  """Roll the scripted reference once to capture the hammer-head path (world coords), then reset.

  Used only to draw the static reference polyline; the viewer re-runs the same open-loop strike,
  so the drawn line is exactly the path the head retraces. Breaks at the success auto-reset so the
  line is one clean strike (not a repeat).
  """
  obs, _ = env.reset()
  pts = [policy._head().squeeze(0).cpu().numpy().copy()]
  n = policy._ref.playback_length()
  for _ in range(n + 14):
    with torch.no_grad():
      action = policy(obs)
    obs = env.step(action)[0]
    if int(env.unwrapped.episode_length_buf[0]) == 0:  # success-reset -> strike complete
      break
    pts.append(policy._head().squeeze(0).cpu().numpy().copy())
  env.reset()
  return np.stack(pts).astype(np.float64)


class ReferencePolicy:
  """Open-loop: command the IK toward the scripted reference waypoint each step.

  Ignores the observation; reads head + nail_top from the live scene and uses
  the shared SingleStrikeReference (auto-anchored per episode) to pick the
  scripted target for the current per-env step (episode_length_buf).
  """

  def __init__(self, env, approach_height: float):
    raw = env.unwrapped
    self._raw = raw
    self._scale = Z1_HAMMER_DELTA_POS_SCALE
    self._ref = get_strike_reference(raw, approach_height=approach_height)
    rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    rcfg.resolve(raw.scene)
    ncfg.resolve(raw.scene)
    self._rcfg, self._ncfg = rcfg, ncfg

  def _head(self) -> torch.Tensor:
    r = self._raw.scene["robot"]
    return r.data.site_pos_w[:, self._rcfg.site_ids].squeeze(1)

  def _nail_top(self) -> torch.Tensor:
    n = self._raw.scene["nail_block"]
    return n.data.site_pos_w[:, self._ncfg.site_ids].squeeze(1)

  def __call__(self, obs) -> torch.Tensor:  # noqa: ARG002 - obs ignored by design
    del obs
    head = self._head()
    step = self._raw.episode_length_buf
    # Keep phase anchored (re-anchors itself when step==0 after each reset).
    self._ref.update(head, self._nail_top(), step)
    n = self._ref.playback_length()
    k = (step + 1).clamp(max=n)  # aim one scripted step ahead
    # playback_target takes a scalar step; envs share anchors only when reset
    # together. For multi-env, step per-env via the max (all reset in sync here
    # since they share reset timing); the single-env case is exact.
    target = self._ref.playback_target(int(k.max().item()))
    delta = (target - head) / self._scale
    return delta.clamp(-1.0, 1.0)


class _RefLineViewer(ViserPlayViewer):
  """ViserPlayViewer that draws the reference head path as a red polyline with a GUI on/off toggle."""

  head_path: np.ndarray | None = None

  def setup(self) -> None:
    super().setup()
    if self.head_path is None or len(self.head_path) < 2:
      return
    # scene offset is (0,0,0) for a single env; add it for safety so the line sits on the robot.
    off = np.asarray(getattr(self._scene, "_scene_offset", np.zeros(3)), dtype=np.float64).reshape(3)
    pts = self.head_path.astype(np.float64) + off[None, :]
    segs = np.stack([pts[:-1], pts[1:]], axis=1).astype(np.float32)  # (N-1, 2, 3): join the real points
    self._refline = self._server.scene.add_line_segments(
      "/reference_path", points=segs, colors=(230, 30, 15), line_width=3.0
    )
    with self._server.gui.add_folder("Reference"):
      toggle = self._server.gui.add_checkbox("Show reference path", True)

    @toggle.on_update
    def _(_event) -> None:
      self._refline.visible = toggle.value


def main(cfg: Cfg = Cfg()) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(TASK_ID, play=True)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  policy = ReferencePolicy(env, cfg.approach_height)
  print(f"[play_reference] open-loop scripted strike, num_envs={cfg.num_envs}, "
        f"approach_height={cfg.approach_height}, viewer={cfg.viewer}")

  if cfg.viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif cfg.show_line:
    path = collect_head_path(env, policy)
    viewer = _RefLineViewer(env, policy)
    viewer.head_path = path
    print(f"[play_reference] reference path drawn ({len(path)} pts) — toggle it under the "
          f"'Reference' folder in the viser GUI")
    viewer.run()
  else:
    ViserPlayViewer(env, policy).run()


if __name__ == "__main__":
  main(tyro.cli(Cfg))
