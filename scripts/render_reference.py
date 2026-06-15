"""Headless screenshots / video of the scripted T1 strike (no browser, no GPU).

Drives the open-loop SingleStrikeReference (same as play_reference.py) but uses
MuJoCo's offscreen renderer to write PNG frames + an MP4 — so the simulation can
be inspected from images alone (works on macOS; offscreen GL needs no display).

Run:
    python scripts/render_reference.py                 # -> /tmp/hammer_ref/
    python scripts/render_reference.py --out-dir foo --steps 30 --approach-height 0.10
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.mdp.references import SingleStrikeReference

TASK_ID = "Unitree-Z1-Hammer"


@dataclass(frozen=True)
class Cfg:
  out_dir: str = "/tmp/hammer_ref"
  steps: int = 30
  approach_height: float = 0.15
  width: int = 640
  height: int = 480
  # Camera overrides (None = keep the env's ViewerConfig). distance smaller =
  # zoom in; the default env cam sits 2 m from the base and the nail is tiny.
  distance: float | None = None
  elevation: float | None = None
  azimuth: float | None = None


def main(cfg: Cfg = Cfg()) -> None:
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  if cfg.distance is not None:
    env_cfg.viewer.distance = cfg.distance
  if cfg.elevation is not None:
    env_cfg.viewer.elevation = cfg.elevation
  if cfg.azimuth is not None:
    env_cfg.viewer.azimuth = cfg.azimuth
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
  env.reset()

  rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
  ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
  robot, nail = env.scene["robot"], env.scene["nail_block"]
  head = lambda: robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
  ntop = lambda: nail.data.site_pos_w[:, ncfg.site_ids].squeeze(1)
  depth = lambda: float(nail.data.joint_pos[0, 0])

  ref = SingleStrikeReference(1, env.device, approach_height=cfg.approach_height)
  ref.update(head(), ntop(), env.episode_length_buf)
  scale = Z1_HAMMER_DELTA_POS_SCALE

  frames, rows = [], []
  f0 = env.render()
  if f0 is not None:
    frames.append(np.asarray(f0)); iio.imwrite(out / "step_000_reset.png", frames[-1])
  rows.append(f"{'step':>4} {'phi':>6} {'nail_mm':>8} {'contact':>8}")

  sensor = env.scene["hammer_nail_contact"]
  for k in range(1, cfg.steps + 1):
    step_buf = env.episode_length_buf
    phi = ref.update(head(), ntop(), step_buf)
    n = ref.playback_length()
    target = ref.playback_target(min(int(step_buf.max().item()) + 1, n))
    action = ((target - head()) / scale).clamp(-1.0, 1.0)
    env.step(action)
    contact = bool((sensor.data.found > 0).any())
    rows.append(f"{k:>4} {float(phi[0]):>6.3f} {depth()*1000:>8.1f} {str(contact):>8}")
    fr = env.render()
    if fr is not None:
      fr = np.asarray(fr)
      frames.append(fr)
      # save a few labelled key frames
      if k in (5, 9, 10, 11, 13, 14):
        iio.imwrite(out / f"step_{k:03d}_d{int(round(depth()*1000)):02d}mm.png", fr)
    if int(env.episode_length_buf[0]) == 0:
      rows.append(f"  -- episode reset (success) after step {k} --")
      # keep rendering the next episode's start for context, then stop soon
      if k >= 14:
        break

  # montage of up to 6 evenly-spaced frames
  if frames:
    idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
    montage = np.concatenate([frames[i] for i in idx], axis=1)
    iio.imwrite(out / "montage.png", montage)
    try:
      iio.imwrite(out / "strike.mp4", np.stack(frames), fps=10)
    except Exception as e:  # ffmpeg may be absent; PNGs are the fallback
      print(f"[render] mp4 skipped ({type(e).__name__}); PNGs written")

  print("\n".join(rows))
  print(f"\n[render] wrote {len(frames)} frames + montage.png to {out}/")
  print("[render] key frames:", ", ".join(p.name for p in sorted(out.glob('step_*.png'))))


if __name__ == "__main__":
  main(tyro.cli(Cfg))
