"""Execute the fixed primary controlled-drop calibration."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import re
import subprocess

import torch

from mjlab.envs import ManagerBasedRlEnv

from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML
from src.tasks.hammer.calibration.controlled_drop_contract import (
  ControlledDropExecution,
  ControlledDropResult,
  ControlledDropTrial,
  _validate_primary_trial,
  build_primary_result,
)
from src.tasks.hammer.calibration.controlled_drop_env import (
  PRIMARY_AXIS,
  PRIMARY_DROP_FRICTION,
  PRIMARY_DROP_H0_M,
  PRIMARY_DROP_HALF_HEIGHT_M,
  PRIMARY_DROP_MASS_KG,
  PRIMARY_DROP_RADIUS_M,
  PRIMARY_PHYSICS_DT_S,
  PRIMARY_PROGRESS_EPS,
  PRIMARY_WINDOW_SUBSTEPS,
  make_controlled_drop_env_cfg,
)
from src.tasks.hammer.mdp.first_strike import (
  REASON_SUCCESS,
  REASON_WINDOW,
  FirstStrikeEventTracker,
)


_MAX_DROP_SUBSTEPS = 500
_FULL_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_EXPECTED_VERSIONS = {
  "mjlab": "1.4.0",
  "mujoco": "3.8.1",
  "mujoco-warp": "3.8.1",
}


def _validate_expected_revision(name: str, revision: str) -> None:
  if _FULL_GIT_REVISION.fullmatch(revision) is None:
    raise ValueError(f"{name} must be a full lowercase Git revision")


def capture_clean_execution_identity(
  *, expected_code_revision: str, expected_asset_revision: str
) -> tuple[str, str]:
  """Capture clean repository identities for one calibration boundary."""
  _validate_expected_revision("expected_code_revision", expected_code_revision)
  _validate_expected_revision("expected_asset_revision", expected_asset_revision)
  code_root = Path(__file__).resolve().parents[4]
  asset_root = Z1_HAMMER_XML.parents[2]
  code_revision = _capture_clean_repository(
    root=code_root,
    label="code",
    expected_revision=expected_code_revision,
  )
  asset_revision = _capture_clean_repository(
    root=asset_root,
    label="asset",
    expected_revision=expected_asset_revision,
  )
  return code_revision, asset_revision


def _capture_clean_repository(
  *, root: Path, label: str, expected_revision: str
) -> str:
  revision = _git_output(root, "rev-parse", "HEAD")
  if revision != expected_revision:
    raise ValueError(f"{label} revision differs from expected revision")
  status = _git_output(root, "status", "--porcelain", "--untracked-files=no")
  if status:
    raise ValueError(f"{label} repository has tracked changes")
  return revision


def _git_output(root: Path, *args: str) -> str:
  completed = subprocess.run(
    ["git", "-C", str(root), *args],
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def _live_library_versions() -> dict[str, str]:
  versions = {
    distribution: metadata.version(distribution)
    for distribution in _EXPECTED_VERSIONS
  }
  for distribution, expected in _EXPECTED_VERSIONS.items():
    if versions[distribution] != expected:
      raise RuntimeError(
        f"{distribution} version must be {expected}, got {versions[distribution]}"
      )
  return versions


def run_primary_calibration(
  *, device: str, expected_code_revision: str, expected_asset_revision: str
) -> ControlledDropResult:
  """Run exactly five fresh primary releases under one clean identity."""
  _validate_expected_revision("expected_code_revision", expected_code_revision)
  _validate_expected_revision("expected_asset_revision", expected_asset_revision)
  versions = _live_library_versions()
  code_revision, asset_revision = capture_clean_execution_identity(
    expected_code_revision=expected_code_revision,
    expected_asset_revision=expected_asset_revision,
  )

  trials: list[ControlledDropTrial] = []
  for trial_index in range(1, 6):
    trial = run_one_primary_drop(device=device, trial_index=trial_index)
    try:
      _validate_primary_trial(trial, expected_index=trial_index)
    except ValueError as error:
      raise ValueError(
        f"trial {trial_index} is invalid: {trial!r}; validation error: {error}"
      ) from error
    if trials and trial.reason != trials[0].reason:
      raise ValueError(
        f"trial {trial_index} has mixed finalization reason: {trial!r}; "
        f"expected {trials[0].reason!r}"
      )
    trials.append(trial)

  post_identity = capture_clean_execution_identity(
    expected_code_revision=expected_code_revision,
    expected_asset_revision=expected_asset_revision,
  )
  if post_identity != (code_revision, asset_revision):
    raise RuntimeError("repository identity changed during primary calibration")

  execution = ControlledDropExecution(
    code_revision=code_revision,
    asset_revision=asset_revision,
    device=device,
    backend="mujoco-warp",
    mujoco_version=versions["mujoco"],
    mujoco_warp_version=versions["mujoco-warp"],
    mjlab_version=versions["mjlab"],
    physics_dt_s=PRIMARY_PHYSICS_DT_S,
    h0_m=PRIMARY_DROP_H0_M,
    mass_kg=PRIMARY_DROP_MASS_KG,
    radius_m=PRIMARY_DROP_RADIUS_M,
    half_height_m=PRIMARY_DROP_HALF_HEIGHT_M,
    friction=PRIMARY_DROP_FRICTION,
    slide_axis=PRIMARY_AXIS,
    slide_damping=0.0,
    slide_frictionloss=0.0,
    tracker_axis=PRIMARY_AXIS,
    tracker_window_substeps=PRIMARY_WINDOW_SUBSTEPS,
    tracker_progress_eps=PRIMARY_PROGRESS_EPS,
  )
  return build_primary_result(execution, trials)


def run_one_primary_drop(*, device: str, trial_index: int) -> ControlledDropTrial:
  """Run one fresh passive release and return its public tracker result."""
  env = ManagerBasedRlEnv(cfg=make_controlled_drop_env_cfg(), device=device)
  try:
    assert env.num_envs == 1
    assert env.action_manager.action.shape == (1, 0)
    assert str(env.device) == device
    assert env.cfg.decimation == 1

    env.reset()
    drop_joint_id = env.sim.mj_model.joint("robot/drop_axis").id
    drop_dof_id = env.sim.mj_model.jnt_dofadr[drop_joint_id]
    release_velocity_m_s = float(env.sim.mj_data.qvel[drop_dof_id])
    assert release_velocity_m_s == 0.0

    tracker = env.metrics_manager.cfg["first_strike"].func
    assert isinstance(tracker, FirstStrikeEventTracker)
    for _ in range(_MAX_DROP_SUBSTEPS):
      env.step(torch.empty((1, 0), dtype=torch.float32, device=env.device))
      if bool(tracker.finalized[0].item()):
        break
    else:
      raise RuntimeError(
        f"primary drop did not finalize within {_MAX_DROP_SUBSTEPS} physics substeps"
      )

    reason_code = int(tracker.reason[0].item())
    if reason_code == REASON_SUCCESS:
      reason = "success"
    elif reason_code == REASON_WINDOW:
      reason = "window"
    else:
      raise RuntimeError(f"primary drop finalized with unknown reason {reason_code}")

    return ControlledDropTrial(
      trial_index=trial_index,
      h0_m=PRIMARY_DROP_H0_M,
      release_velocity_m_s=release_velocity_m_s,
      precontact_velocity_m_s=float(tracker.v_precontact[0].item()),
      impulse_n_s=float(tracker.delivered[0].item()),
      contacted=bool(tracker.started[0].item()),
      finalized=bool(tracker.finalized[0].item()),
      productive=bool(tracker.productive[0].item()),
      reason=reason,
      depth_at_contact_m=float(tracker.depth_at_contact[0].item()),
      peak_depth_m=float(tracker.peak_depth[0].item()),
    )
  finally:
    env.close()
