"""Headless offscreen render of a TRAINED policy rollout (no viewer) -> PNG frames + mp4.

Loads a checkpoint (like `scripts/play.py --agent trained`) but renders with MuJoCo's
offscreen renderer (like `scripts/render_reference.py`) so the trained policy can be
inspected from images alone -- no browser, no display. Works headless on macOS (CGL,
leave MUJOCO_GL unset) and on Linux (export MUJOCO_GL=egl first). CPU is fine:
the checkpoint is map_location'd to the chosen device.

Checkpoints train locally under logs/rsl_rl/ (Lightning writes them in place via
scripts/lightning_pair.sh); for a remote box, rsync the .pt back first. Then:
    python scripts/render_policy.py --checkpoint-file /tmp/model_499.pt
    python scripts/render_policy.py --task Unitree-Z1-Hammer-Track \
        --checkpoint-file /tmp/a_track_seed0_model_499.pt --steps 80 --distance 0.85 --elevation -25

Reads out: montage.png (6 evenly-spaced frames), frame_*.png, policy.mp4,
trajectory.png, trace.npz, and metadata.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import tyro

import mjlab.tasks  # noqa: F401  (populate the task registry)
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from evaluation.analysis.fixed_reset_video_library import (
  ARTIFACT_FILENAMES,
  FIXED_RESET_ENVELOPE,
  load_fixed_reset,
  write_metadata,
  write_trajectory_png,
)
from scripts.eval_impulse import restore_reset_state
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME


@dataclass(frozen=True)
class Cfg:
  checkpoint_file: str
  """Path to a model_*.pt checkpoint (pulled from Vega)."""
  campaign: str
  """Registered campaign identity, e.g. fq4x8 or fq3x8."""
  arm: str
  """Registered treatment arm identity."""
  training_seed: int
  """Training seed for this checkpoint."""
  checkpoint_sha256: str
  """Expected SHA-256 of checkpoint_file, recorded in metadata."""
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
  fixed_reset_envelope: str = str(FIXED_RESET_ENVELOPE)
  """Stage-0 envelope containing the canonical shared fixed reset."""
  metadata_provenance: str = ""
  """Free-form immutable provenance note for this rendering invocation."""
  fps: int = 10
  """Frames per second for policy.mp4."""
  device: str = "cpu"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main(cfg: Cfg) -> None:
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  ckpt = Path(cfg.checkpoint_file)
  if not ckpt.exists():
    raise FileNotFoundError(f"checkpoint not found: {ckpt}")
  checkpoint_sha256 = _sha256(ckpt)
  if checkpoint_sha256 != cfg.checkpoint_sha256:
    raise ValueError(
      "checkpoint SHA-256 mismatch: "
      f"expected {cfg.checkpoint_sha256}, got {checkpoint_sha256}"
    )
  fixed_reset = load_fixed_reset(cfg.fixed_reset_envelope)

  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = 1
  # Keep the terminal strike state readable; auto-reset would replace it with
  # the next episode before its trajectory/contact/frame could be recorded.
  env_cfg.auto_reset = False
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
  head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  head_cfg.resolve(base_env.scene)
  robot = base_env.scene["robot"]
  head = lambda: (
    robot.data.site_pos_w[0, head_cfg.site_ids]
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
    .copy()
  )
  depth_mm = lambda: float(nail.data.joint_pos[0, 0]) * 1000.0

  env.reset()
  # restore_reset_state() is the established D2 replay helper and accepts the
  # complete reset record, rather than its nested reset_state mapping.
  restore_reset_state(base_env, fixed_reset)
  base_env.sim.forward()
  base_env.sim.sense()
  base_env.obs_buf = base_env.observation_manager.compute(update_history=True)
  obs = env.get_observations()
  frames: list[np.ndarray] = []
  head_positions = [head()]
  contacts = [bool((sensor.data.found[0] > 0).any())]
  actions_recorded: list[np.ndarray] = []
  rows = [f"{'step':>4} {'nail_mm':>8} {'contact':>8} {'reward':>9}"]
  f0 = base_env.render()
  if f0 is not None:
    frames.append(np.asarray(f0))
    iio.imwrite(out / "step_000_reset.png", frames[-1])

  for k in range(1, cfg.steps + 1):
    with torch.inference_mode():
      actions = policy(obs)
    actions_recorded.append(actions[0].detach().cpu().numpy().astype(np.float32).copy())
    step_out = env.step(actions)
    obs, rew, dones = step_out[0], step_out[1], step_out[2]
    contact = bool((sensor.data.found > 0).any())
    head_positions.append(head())
    contacts.append(contact)
    rows.append(f"{k:>4} {depth_mm():>8.1f} {str(contact):>8} {float(rew[0]):>9.3f}")
    fr = base_env.render()
    if fr is not None:
      frames.append(np.asarray(fr))
    if bool(dones[0]):
      rows.append(f"  -- episode boundary after step {k}; terminal frame retained --")
      break
  if not frames:
    raise RuntimeError("renderer produced no RGB frames")
  idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
  iio.imwrite(out / "montage.png", np.concatenate([frames[i] for i in idx], axis=1))
  for j, i in enumerate(idx):
    iio.imwrite(out / f"frame_{j}_idx{int(i):03d}.png", frames[i])
  iio.imwrite(out / "policy.mp4", np.stack(frames), fps=cfg.fps)
  trace = {
    "head_position_m": np.asarray(head_positions, dtype=np.float64),
    "contact": np.asarray(contacts, dtype=bool),
    "contact_live_at_control_boundary": np.asarray(contacts, dtype=bool),
    "control_step": np.arange(len(head_positions), dtype=np.int64),
    "action": np.asarray(actions_recorded[: len(head_positions) - 1], dtype=np.float32),
  }
  np.savez(out / "trace.npz", **trace)
  write_trajectory_png(trace, out / "trajectory.png")
  write_metadata(
    out / "metadata.json",
    {
      "campaign": cfg.campaign,
      "arm": cfg.arm,
      "training_seed": cfg.training_seed,
      "checkpoint_sha256": checkpoint_sha256,
      "checkpoint_file": str(ckpt),
      "reset_state_digest": fixed_reset["reset_state_digest"],
      "reset_envelope": str(cfg.fixed_reset_envelope),
      "fps": cfg.fps,
      "metadata_provenance": cfg.metadata_provenance,
      "contact_semantics": "live hammer_head_0/nail contact at control boundary",
      "artifacts": {name: _sha256(out / name) for name in ARTIFACT_FILENAMES},
    },
  )

  print("\n".join(rows))
  print(
    f"\n[render] {len(frames)} frames + artifact contract -> {out}/  "
    f"(ckpt={ckpt.name}, task={cfg.task})"
  )
  print("[render] read montage.png / trajectory.png to inspect the rollout.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
