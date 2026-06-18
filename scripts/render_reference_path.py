"""Replay the scripted single-strike reference and trace its hammer-head path as a thin 3D line.

Re-rolls the open-loop SingleStrikeReference (the same drive as playback_reference.py) and draws
the hammer head's ACTUAL path as a thin red polyline that GROWS as the replay evolves -- each
frame connects the real head positions visited so far (no synthetic straight line; it just joins
the points the head actually passes through). Drawn in the 3D scene via the offscreen renderer's
debug-vis hook (env.render() calls env.update_visualizers, which we wrap). Headless on macOS with
plain python (no mjpython, no display, leave MUJOCO_GL unset).

Run:
    python scripts/render_reference_path.py
    python scripts/render_reference_path.py --distance 0.7 --line-radius 0.0010 --out-dir /tmp/hammer_refpath
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import tyro

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
    HAMMER_HEAD_SITE_NAME,
    Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.references import SingleStrikeReference


@dataclass(frozen=True)
class Cfg:
  out_dir: str = "/tmp/hammer_refpath"
  approach_height: float = 0.15
  extra_steps: int = 14          # push-through hold after the scripted descent
  width: int = 720
  height: int = 540
  distance: float | None = 0.85
  elevation: float | None = -25.0
  azimuth: float | None = None
  line_radius: float = 0.0012    # thin tube (metres); was 0.004
  fps: int = 10
  device: str = "cpu"


def main(cfg: Cfg) -> None:
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)

  env_cfg = z1_hammer_env_cfg(play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.terminations = {}                 # no reset: show the full strike + push-through
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  if cfg.distance is not None:
    env_cfg.viewer.distance = cfg.distance
  if cfg.elevation is not None:
    env_cfg.viewer.elevation = cfg.elevation
  if cfg.azimuth is not None:
    env_cfg.viewer.azimuth = cfg.azimuth

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode="rgb_array")

  head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  for c in (head_cfg, nail_cfg):
    c.resolve(env.scene)
  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  head = lambda: robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)
  nail_top = lambda: nail_e.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)
  head_np = lambda: head().squeeze(0).cpu().numpy().astype(np.float64).copy()

  # --- progressive trajectory: trace the REAL head positions visited so far ---
  trail: list[np.ndarray] = []
  _orig_update = env.update_visualizers
  RED = (0.95, 0.15, 0.05, 1.0)
  TIP = (1.0, 0.85, 0.0, 1.0)

  def draw(visualizer) -> None:
    _orig_update(visualizer)
    if not trail:
      return
    for i in range(len(trail) - 1):         # join the actual points (real path, grows each frame)
      visualizer.add_cylinder(trail[i], trail[i + 1], radius=cfg.line_radius, color=RED)
    visualizer.add_sphere(trail[-1], cfg.line_radius * 2.2, TIP)   # current head marker

  env.update_visualizers = draw             # render() picks this up via hasattr

  # --- replay the scripted reference open-loop, growing the trail ---
  env.reset()
  ref = SingleStrikeReference(1, env.device, approach_height=cfg.approach_height)
  ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
  n = ref.playback_length()

  frames: list[np.ndarray] = []
  trail.append(head_np())
  fr = env.render()
  if fr is not None:
    frames.append(np.asarray(fr))

  for k in range(1, n + cfg.extra_steps + 1):
    target = ref.playback_target(min(k, n))
    action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
    env.step(action)
    trail.append(head_np())                 # record the real head position, THEN render
    fr = env.render()
    if fr is not None:
      frames.append(np.asarray(fr))

  if frames:
    idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
    iio.imwrite(out / "montage.png", np.concatenate([frames[i] for i in idx], axis=1))
    try:
      iio.imwrite(out / "reference_path.mp4", np.stack(frames), fps=cfg.fps)
    except Exception as e:  # noqa: BLE001
      print(f"[refpath] mp4 skipped ({type(e).__name__}); PNGs only")
  span = np.ptp(np.stack(trail), axis=0) if len(trail) > 1 else np.zeros(3)
  print(f"[refpath] {len(frames)} frames, {len(trail)} trail points -> {out}/")
  print(f"[refpath] head path span (m): dx={span[0]:.4f} dy={span[1]:.4f} dz={span[2]:.4f}  "
        f"(near-vertical strike -> the line is genuinely straight)")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
