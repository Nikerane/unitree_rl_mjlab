"""Headless offscreen render of a TRAINED policy rollout (no viewer) -> PNG frames + mp4.

Loads a checkpoint (like `scripts/play.py --agent trained`) but renders with MuJoCo's
offscreen renderer (like `scripts/render_reference.py`) so the trained policy can be
inspected from images alone -- no browser, no display. Works headless on macOS (CGL,
leave MUJOCO_GL unset) and on Linux/Vega (export MUJOCO_GL=egl first). CPU is fine:
the checkpoint is map_location'd to the chosen device.

Pull a checkpoint back from Vega first, e.g.:
    rsync -av vega:~/repos/unitree_rl_mjlab/logs/rsl_rl/z1_hammer/<run>/model_499.pt /tmp/

Then:
    python scripts/render_policy.py --checkpoint-file /tmp/model_499.pt
    python scripts/render_policy.py --task Unitree-Z1-Hammer-Track \
        --checkpoint-file /tmp/a_track_seed0_model_499.pt --steps 80 --distance 0.85 --elevation -25

Reads out: montage.png (6 evenly-spaced frames), frame_*.png, policy.mp4 (if ffmpeg),
and a per-step nail-depth / contact / reward table -- the Path-A read (strike vs slam vs press).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import tyro

import mjlab.tasks  # noqa: F401  (populate the task registry)
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


@dataclass(frozen=True)
class Cfg:
  checkpoint_file: str
  """Path to a model_*.pt checkpoint (pulled from Vega)."""
  task: str = "Unitree-Z1-Hammer"
  """Gym task id: Unitree-Z1-Hammer (A-BASE) or Unitree-Z1-Hammer-Track (A-TRACK)."""
  out_dir: str = "/tmp/hammer_policy"
  steps: int = 80
  width: int = 640
  height: int = 480
  # Camera overrides (None = keep the env's ViewerConfig). Smaller distance = zoom in.
  distance: float | None = 0.85
  elevation: float | None = -25.0
  azimuth: float | None = None
  device: str = "cpu"


def main(cfg: Cfg) -> None:
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  ckpt = Path(cfg.checkpoint_file)
  if not ckpt.exists():
    raise FileNotFoundError(f"checkpoint not found: {ckpt}")

  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  if cfg.distance is not None:
    env_cfg.viewer.distance = cfg.distance
  if cfg.elevation is not None:
    env_cfg.viewer.elevation = cfg.elevation
  if cfg.azimuth is not None:
    env_cfg.viewer.azimuth = cfg.azimuth

  agent_cfg = load_rl_cfg(cfg.task)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=cfg.device)
  policy = runner.get_inference_policy(device=cfg.device)

  nail = base_env.scene["nail_block"]
  sensor = base_env.scene["hammer_nail_contact"]
  depth_mm = lambda: float(nail.data.joint_pos[0, 0]) * 1000.0

  obs, _ = env.reset()
  frames: list[np.ndarray] = []
  rows = [f"{'step':>4} {'nail_mm':>8} {'contact':>8} {'reward':>9}"]
  f0 = base_env.render()
  if f0 is not None:
    frames.append(np.asarray(f0))
    iio.imwrite(out / "step_000_reset.png", frames[-1])

  for k in range(1, cfg.steps + 1):
    with torch.inference_mode():
      actions = policy(obs)
    step_out = env.step(actions)
    obs, rew = step_out[0], step_out[1]
    contact = bool((sensor.data.found > 0).any())
    rows.append(f"{k:>4} {depth_mm():>8.1f} {str(contact):>8} {float(rew[0]):>9.3f}")
    fr = base_env.render()
    if fr is not None:
      frames.append(np.asarray(fr))
    if int(base_env.episode_length_buf[0]) == 0:
      rows.append(f"  -- episode reset (success/timeout) after step {k} --")

  if frames:
    idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
    iio.imwrite(out / "montage.png", np.concatenate([frames[i] for i in idx], axis=1))
    for j, i in enumerate(idx):
      iio.imwrite(out / f"frame_{j}_idx{int(i):03d}.png", frames[i])
    try:
      iio.imwrite(out / "policy.mp4", np.stack(frames), fps=10)
    except Exception as e:  # ffmpeg may be absent; PNGs are the fallback
      print(f"[render] mp4 skipped ({type(e).__name__}); PNGs written")

  print("\n".join(rows))
  print(
    f"\n[render] {len(frames)} frames + montage.png -> {out}/  "
    f"(ckpt={ckpt.name}, task={cfg.task})"
  )
  print("[render] read montage.png / frame_*.png to watch the rollout.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
