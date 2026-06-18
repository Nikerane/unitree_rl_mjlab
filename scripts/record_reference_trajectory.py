"""Generate + record the scripted single-strike reference as a JOINT-SPACE q(t) dataset.

This is the "generate a trajectory in simulation, store it as a dataset" half of the
reference-tracking setup (the joint-space tracking reward consumes this file). It rolls
out the SingleStrikeReference OPEN-LOOP -- DiffIK driving the hammer head to the scripted
wind-up -> strike waypoints, exactly as docs/research/reward-design/playback_reference.py
certifies it (Phase M) -- and records the REALIZED arm joint trajectory q(t) together with
the strike phase phi(t). The dataset is therefore a phase-indexed lookup table phi -> q_ref:
the tracking reward reads q_ref at the live phi (the same contact-robust phase clock the obs
already compute), so timing variation near contact does not alias the target (Biemond 2013).

Deterministic: play=True zeroes reset/obs noise, so the recorded q(t) is reproducible and
anchored to the canonical reset pose (DeepMimic-style; reset randomization handling for the
tracking arm is a reward-side decision, not recorded here).

Run:
    python scripts/record_reference_trajectory.py
    python scripts/record_reference_trajectory.py --out src/tasks/hammer/data/reference_strike_qtraj.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab.tasks  # noqa: F401  (populate the task registry)
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
    HAMMER_HEAD_SITE_NAME,
    Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


@dataclass(frozen=True)
class Cfg:
  out: str = "src/tasks/hammer/data/reference_strike_qtraj.npz"
  approach_height: float = 0.15  # SingleStrikeReference default (matches training obs)
  hold_steps: int = 14           # push-through after descent: the strike completes here
                                 # (approach 0.15 succeeds ~step 20 per playback_reference)
  device: str = "cpu"


def main(cfg: Cfg) -> None:
  env_cfg = z1_hammer_env_cfg(play=True)
  env_cfg.scene.num_envs = 1
  # Disable terminations so the full strike + push-through is captured: with terminations on,
  # the nail-driven success auto-resets the arm mid-step and the deepest pose is lost.
  env_cfg.terminations = {}
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)

  head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
  nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  for c in (head_cfg, arm_cfg, nail_cfg):
    c.resolve(env.scene)
  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  sensor = env.scene["hammer_nail_contact"]

  head = lambda: robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)          # (1,3)
  nail_top = lambda: nail_e.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)     # (1,3)
  qpos = lambda: robot.data.joint_pos[:, arm_cfg.joint_ids].squeeze(0)           # (6,)
  depth = lambda: float(nail_e.data.joint_pos[0, 0])

  env.reset()
  ref = SingleStrikeReference(1, env.device, approach_height=cfg.approach_height)
  ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))  # anchor
  n = ref.playback_length()

  q_rows, phi_rows, head_rows, depth_rows = [], [], [], []
  # Record the anchored reset state at phi=0 first (so the dataset starts at the home pose).
  q_rows.append(qpos().cpu().numpy().copy())
  phi_rows.append(0.0)
  head_rows.append(head().squeeze(0).cpu().numpy().copy())
  depth_rows.append(depth())

  contact_step = None
  for k in range(1, n + cfg.hold_steps + 1):
    target = ref.playback_target(min(k, n))               # Cartesian head waypoint
    action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
    env.step(action)
    phi = ref.update(head(), nail_top(), env.episode_length_buf)  # training-consistent phase
    q_rows.append(qpos().cpu().numpy().copy())
    phi_rows.append(float(phi[0]))
    head_rows.append(head().squeeze(0).cpu().numpy().copy())
    depth_rows.append(depth())
    if contact_step is None and bool((sensor.data.found > 0).any()):
      contact_step = k

  q = np.stack(q_rows).astype(np.float32)        # (T, 6)
  phi = np.asarray(phi_rows, dtype=np.float32)   # (T,)
  head_w = np.stack(head_rows).astype(np.float32)
  depth_arr = np.asarray(depth_rows, dtype=np.float32)

  # phi is monotone-latched but record-order should be non-decreasing; assert for the lookup.
  assert np.all(np.diff(phi) >= -1e-6), "phi not monotone — phase lookup would be ill-defined"

  out = Path(cfg.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out,
    q_ref=q,                       # (T,6) realized arm joint trajectory
    phi=phi,                       # (T,)  strike phase at each row (lookup key)
    head_w=head_w,                 # (T,3) realized head path (validation/overlay)
    depth_mm=depth_arr * 1000.0,   # (T,)  nail depth, metadata
    joint_names=np.array(ARM_JOINTS),
    approach_height=np.float32(cfg.approach_height),
    step_dt=np.float32(env.step_dt),
    success_threshold_mm=np.float32(NAIL_SUCCESS_THRESHOLD * 1000.0),
  )

  print(f"[record] wrote {out}  ({q.shape[0]} steps, {q.shape[1]} joints)")
  print(f"[record] phi range [{phi.min():.3f}, {phi.max():.3f}]  "
        f"contact@step~{contact_step}  max depth {depth_arr.max()*1000:.1f} mm "
        f"(threshold {NAIL_SUCCESS_THRESHOLD*1000:.0f} mm)")
  print(f"[record] q_ref (deg) per phase quartile:")
  print(f"  {'phi':>5} " + " ".join(f"{j:>7}" for j in ARM_JOINTS))
  for target_phi in (0.0, 0.25, 0.5, 0.75, 1.0):
    i = int(np.argmin(np.abs(phi - target_phi)))
    degs = np.degrees(q[i])
    print(f"  {phi[i]:>5.2f} " + " ".join(f"{d:>7.1f}" for d in degs))


if __name__ == "__main__":
  main(tyro.cli(Cfg))
