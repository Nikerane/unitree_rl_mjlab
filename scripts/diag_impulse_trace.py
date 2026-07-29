"""Impulse-CaT force-propagation diagnostics: the impulse analogue of diag_policy_trace.py.

Drives a SINGLE strike -- either the open-loop scripted reference (``--reference``, the Task-5
playback idiom) or a trained checkpoint (``--ckpt``, loaded exactly like diag_policy_trace.py) --
against a ``cat_impulse=True`` task (default ``Unitree-Z1-Hammer-CaT-Impulse``, which wires BOTH
the ``SubstepImpulseAccumulator`` / ``SubstepDeliveredImpulse`` accumulators AND the ``CatSoftHook``)
and records, at the true 500 Hz SUBSTEP rate:

  1. per-joint |qfrc_constraint_j|  -- the raw reaction, propagating down the arm chain
  2. object-side F_axial            -- the delivered contact force (weld/friction-immune)
  3. the SHIPPED accumulator's running per-joint Λ_j (``acc.impulse``, sliding-window semantics 2026-07-13)
  4. δ  -- env.extras["cat_delta"], written once per CONTROL step by CatSoftHook and naturally
     forward-filled across this step's substeps (env.extras is not cleared between steps; see
     manager_based_rl_env.py -- metrics_manager.compute_substep() runs INSIDE the decimation loop,
     metrics_manager.compute() -- which calls CatSoftHook -- runs once AFTER it)
  5. per-joint |q̇_j|

via a ``metrics_manager.compute_substep`` monkeypatch, hooked AFTER the shipped accumulators run
(mirrors ``derive_impulse_thresholds.py``:100-111 verbatim: patched() calls orig_substep() first).

Output: a 5-panel matplotlib figure (``trace.png``) plus the raw substep arrays (``trace.npz``).

Usage:
  python scripts/diag_impulse_trace.py --reference --out /tmp/impulse_trace_ref \
      --j-limit 1.640,3.280,1.640,1.640,1.640,1.640
  python scripts/diag_impulse_trace.py --ckpt /tmp/ckpt/c2/model_499.pt --out /tmp/impulse_trace_ckpt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

import mjlab.tasks  # noqa: F401  (register builtin tasks)
import src.tasks  # noqa: F401  (register hammer tasks)
from evaluation.analysis.fixed_reset_video_library import (
  expected_task,
  load_fixed_reset,
)
from scripts.eval_impulse import _PreIntegrationTracePhase, restore_reset_state
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR, _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.references import SingleStrikeReference

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
HOLD_STEPS = 6  # post-playback settle steps -- mirrors derive_impulse_thresholds.py / reward_design_util.py
PHYSICS_DT_S = 0.002
CONTROL_DECIMATION = 10
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION

_SUBSTEP_REQUIRED = frozenset(
  (
    "t_s",
    "head_position_m",
    "head_velocity_m_s",
    "contact",
    "nail_position_m",
    "nail_depth_m",
    "joint_velocity_rad_s",
    "joint_position_rad",
    "joint_velocity_post_integration_rad_s",
    "nail_depth_post_integration_m",
    "qfrc_constraint_abs",
    "lambda_windowed_constraint_read_n_m_s",
    "delivered_impulse_n_s",
    "axial_force_n",
    "control_step_index",
  )
)
_CONTROL_REQUIRED = frozenset(
  (
    "control_step",
    "action",
    "strike_phase_control",
    "strike_ref_error_control_m",
  )
)
_FIRST_STRIKE_FIELDS = frozenset(
  (
    "tracker_started",
    "tracker_finalized",
    "tracker_productive",
    "tracker_reason",
    "tracker_v_precontact_m_s",
    "tracker_delivered_n_s",
    "tracker_delivered_transverse_n_s",
    "tracker_peak_depth_m",
    "tracker_depth_at_contact_m",
    "tracker_first_contact_time_s",
  )
)
_QUALITY_FIELDS = frozenset(
  (
    "tracker_contact_point_w",
    "tracker_contact_error_m",
    "tracker_contact_quality",
    "tracker_contact_quality_valid",
    "tracker_contact_quality_overflow",
    "tracker_contact_normal_axiality",
  )
)
_SCALAR_FIELDS = frozenset(("first_strike_available", "quality_available"))


def _numpy(value) -> np.ndarray:
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().numpy().copy()
  return np.asarray(value).copy()


class _SubstepTraceRecorder:
  """One compact control/substep trace, captured after shipped metrics."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    env_idx: int,
    *,
    robot,
    nail,
    contact,
    net_force,
    arm_ids,
    head_ids,
    nail_site_ids,
    accumulator,
    delivered_accumulator,
  ):
    self.env = env
    self.env_idx = int(env_idx)
    self._robot = robot
    self._nail = nail
    self._contact = contact
    self._net_force = net_force
    self._arm_ids = arm_ids
    self._head_ids = head_ids
    self._nail_site_ids = nail_site_ids
    self._accumulator = accumulator
    self._delivered_accumulator = delivered_accumulator
    self._axis = torch.tensor(
      [0.0, 0.0, -1.0], device=env.device, dtype=torch.float32
    )
    self._phase = _PreIntegrationTracePhase()
    self._tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR, None)
    try:
      env.scene["hammer_nail_quality"]
      has_quality_sensor = True
    except (KeyError, TypeError):
      has_quality_sensor = False
    self.first_strike_available = self._tracker is not None
    self.quality_available = (
      self.first_strike_available and has_quality_sensor
    )
    self._substeps: dict[str, list[np.ndarray]] = {}
    self._actions: list[np.ndarray] = []
    self._strike_phase: list[float] = []
    self._strike_ref_error: list[np.ndarray] = []
    self._current_control_step: int | None = None

  def _append(self, key: str, value) -> None:
    self._substeps.setdefault(key, []).append(_numpy(value))

  def record_control_step(
    self,
    *,
    action,
    strike_phase,
    strike_ref_error_m,
  ) -> None:
    action_array = _numpy(action).astype(np.float32, copy=False).reshape(-1)
    error_array = _numpy(strike_ref_error_m).astype(
      np.float64, copy=False
    ).reshape(-1)
    phase_array = _numpy(strike_phase).reshape(-1)
    if action_array.shape != (3,):
      raise ValueError(
        f"control action must have shape (3,), got {action_array.shape}"
      )
    if error_array.shape != (3,):
      raise ValueError(
        "strike reference error must have shape (3,), got "
        f"{error_array.shape}"
      )
    if phase_array.size != 1:
      raise ValueError(
        f"strike phase must contain one value, got {phase_array.shape}"
      )
    if not (
      np.isfinite(action_array).all()
      and np.isfinite(error_array).all()
      and np.isfinite(phase_array).all()
    ):
      raise ValueError("nonfinite control-rate trace value")
    self._actions.append(action_array.copy())
    self._strike_phase.append(float(phase_array[0]))
    self._strike_ref_error.append(error_array.copy())
    self._current_control_step = len(self._actions) - 1

  def cache_pre_step(self) -> None:
    self._phase.cache_pre_step(
      joint_speed_rad_s=(
        self._robot.data.joint_vel[:, self._arm_ids]
      ),
      clamped_depth_m=self._nail.data.joint_pos[:, 0].clamp(0.0, 0.032),
    )

  def capture_substep(self) -> None:
    if self._current_control_step is None:
      raise RuntimeError(
        "substep captured before its control-rate action/reference row"
      )
    found = (self._contact.data.found > 0).any(dim=-1)
    force = self._net_force.data.force
    axis = self._axis.to(dtype=force.dtype)
    axial = (force * axis).sum(dim=-1).sum(dim=-1).clamp_min(0.0)
    sample = self._phase.capture(
      head_position_m=(
        self._robot.data.site_pos_w[:, self._head_ids].squeeze(1)
      ),
      contact=found,
      net_axial_force_n=axial,
      post_step_joint_speed_rad_s=(
        self._robot.data.joint_vel[:, self._arm_ids]
      ),
      post_step_clamped_depth_m=(
        self._nail.data.joint_pos[:, 0].clamp(0.0, 0.032)
      ),
    )
    i = self.env_idx
    self._append("head_position_m", sample["head_position_m"][i])
    self._append(
      "head_velocity_m_s",
      self._robot.data.site_vel_w[i, self._head_ids].squeeze(0)[:3],
    )
    self._append("contact", sample["contact"][i])
    self._append(
      "nail_position_m",
      self._nail.data.site_pos_w[i, self._nail_site_ids].squeeze(0),
    )
    self._append("nail_depth_m", sample["clamped_depth_m"][i])
    self._append(
      "joint_velocity_rad_s", sample["joint_speed_rad_s"][i]
    )
    self._append(
      "joint_position_rad",
      self._robot.data.joint_pos[i, self._arm_ids],
    )
    self._append(
      "joint_velocity_post_integration_rad_s",
      sample["post_step_joint_speed_rad_s"][i],
    )
    self._append(
      "nail_depth_post_integration_m",
      sample["tracker_depth_post_integration_m"][i],
    )
    self._append(
      "qfrc_constraint_abs",
      self._robot.data._joint_dof_field("qfrc_constraint")[
        i, self._arm_ids
      ].abs(),
    )
    self._append(
      "lambda_windowed_constraint_read_n_m_s",
      self._accumulator.impulse[i],
    )
    self._append(
      "delivered_impulse_n_s", self._delivered_accumulator.delivered[i]
    )
    self._append("axial_force_n", sample["net_axial_force_n"][i])
    self._append(
      "control_step_index",
      np.asarray(self._current_control_step, dtype=np.int64),
    )
    delta = self.env.extras.get("cat_delta")
    self._append(
      "cat_delta",
      0.0 if delta is None else delta[i],
    )

    if not self.first_strike_available:
      return
    tracker = self._tracker
    for key, value in (
      ("tracker_started", tracker.started[i]),
      ("tracker_finalized", tracker.finalized[i]),
      ("tracker_productive", tracker.productive[i]),
      ("tracker_reason", tracker.reason[i]),
      ("tracker_v_precontact_m_s", tracker.v_precontact[i]),
      ("tracker_delivered_n_s", tracker.delivered[i]),
      (
        "tracker_delivered_transverse_n_s",
        tracker.delivered_transverse[i],
      ),
      ("tracker_peak_depth_m", tracker.peak_depth[i]),
      ("tracker_depth_at_contact_m", tracker.depth_at_contact[i]),
      (
        "tracker_first_contact_time_s",
        tracker.first_contact_time_s[i],
      ),
    ):
      self._append(key, value)
    if self.quality_available:
      for key, value in (
        ("tracker_contact_point_w", tracker.contact_point_w[i]),
        ("tracker_contact_error_m", tracker.contact_error_m[i]),
        ("tracker_contact_quality", tracker.contact_quality[i]),
        (
          "tracker_contact_quality_valid",
          tracker.contact_quality_valid[i],
        ),
        (
          "tracker_contact_quality_overflow",
          tracker.contact_quality_overflow[i],
        ),
        (
          "tracker_contact_normal_axiality",
          tracker.contact_normal_axiality[i],
        ),
      ):
        self._append(key, value)

  def to_payload(self, *, physics_dt: float) -> dict[str, np.ndarray]:
    if not self._substeps:
      raise RuntimeError(
        "diag_impulse_trace: no substeps were recorded (empty rollout)."
      )
    payload = {
      key: np.stack(values)
      for key, values in self._substeps.items()
    }
    substeps = len(next(iter(self._substeps.values())))
    payload["t_s"] = np.arange(substeps, dtype=np.float64) * float(
      physics_dt
    )
    payload["control_step"] = np.arange(
      len(self._actions), dtype=np.int64
    )
    payload["action"] = np.stack(self._actions).astype(
      np.float32, copy=False
    )
    payload["strike_phase_control"] = np.asarray(
      self._strike_phase, dtype=np.float64
    )
    payload["strike_ref_error_control_m"] = np.stack(
      self._strike_ref_error
    ).astype(np.float64, copy=False)
    payload["first_strike_available"] = np.asarray(
      self.first_strike_available
    )
    payload["quality_available"] = np.asarray(self.quality_available)
    return payload


def _reset_checkpoint_for_inference(
  env,
  wrapped,
  fixed_reset_envelope: str | Path | None,
) -> tuple[object, dict | None, str | None]:
  """Reset checkpoint mode and return the observations used for inference.

  Historical diagnostics retain the wrapper's ordinary reset.  Companion
  records load and validate the approved fixed reset, initialize every manager
  through a normal environment reset, restore the realized state, refresh the
  observation/history buffer, and only then expose observations to the policy.

  ``restore_reset_state`` owns the MuJoCo ``forward``/``sense`` refresh.  Do
  not repeat those calls here: the observation refresh below is the next
  operation in the replay contract.
  """
  if fixed_reset_envelope is None:
    obs, _ = wrapped.reset()
    return obs, None, None

  fixed_reset = load_fixed_reset(fixed_reset_envelope)
  env.reset()
  restored_digest = restore_reset_state(env, fixed_reset)
  env.obs_buf = env.observation_manager.compute(update_history=True)
  obs = wrapped.get_observations()
  return obs, fixed_reset, restored_digest


def _install_substep_hook(
  env: ManagerBasedRlEnv, env_idx: int
) -> _SubstepTraceRecorder:
  """Install the evaluator phase cache and post-shipped-metric recorder."""
  robot = env.scene["robot"]
  nail = env.scene["nail_block"]
  contact = env.scene["hammer_nail_contact"]
  netf = env.scene["hammer_nail_impulse"]
  arm = SceneEntityCfg("robot", joint_names=ARM)
  head = SceneEntityCfg(
    "robot", site_names=(HAMMER_HEAD_SITE_NAME,)
  )
  nail_top = SceneEntityCfg(
    "nail_block", site_names=("nail_top",)
  )
  arm.resolve(env.scene)
  head.resolve(env.scene)
  nail_top.resolve(env.scene)

  acc_shipped = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
  dacc_shipped = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR, None)
  if acc_shipped is None or dacc_shipped is None:
    raise RuntimeError(
      "diag_impulse_trace requires BOTH the SubstepImpulseAccumulator and SubstepDeliveredImpulse "
      "metrics (cat_impulse=True) -- use --task Unitree-Z1-Hammer-CaT-Impulse (the default) or "
      "another task built with cat_impulse=True."
    )

  recorder = _SubstepTraceRecorder(
    env,
    env_idx,
    robot=robot,
    nail=nail,
    contact=contact,
    net_force=netf,
    arm_ids=arm.joint_ids,
    head_ids=head.site_ids,
    nail_site_ids=nail_top.site_ids,
    accumulator=acc_shipped,
    delivered_accumulator=dacc_shipped,
  )
  orig_step = env.sim.step

  def cache_preintegration_then_step() -> None:
    recorder.cache_pre_step()
    orig_step()

  env.sim.step = cache_preintegration_then_step
  orig_substep = env.metrics_manager.compute_substep

  def patched() -> None:
    # Deliberately first: the shipped 50 ms windowed Lambda read and delivered
    # accumulator must include the contact sample consumed by this substep.
    orig_substep()
    recorder.capture_substep()

  env.metrics_manager.compute_substep = patched  # type: ignore[method-assign]
  return recorder


def _validate_trace_payload(
  payload: Mapping[str, np.ndarray],
  *,
  physics_dt: float,
) -> dict[str, object]:
  """Validate rate alignment and return explicit rate/phase metadata."""
  if not math.isclose(
    float(physics_dt), PHYSICS_DT_S, rel_tol=0.0, abs_tol=1e-12
  ):
    raise ValueError(
      f"physics_dt must be exactly 2 ms ({PHYSICS_DT_S}), got {physics_dt}"
    )
  missing = (_SUBSTEP_REQUIRED | _CONTROL_REQUIRED | _SCALAR_FIELDS) - set(
    payload
  )
  if missing:
    raise ValueError(f"trace payload missing required arrays: {sorted(missing)}")

  arrays = {key: np.asarray(value) for key, value in payload.items()}
  for key in _SCALAR_FIELDS:
    if arrays[key].shape != ():
      raise ValueError(f"{key} must be a scalar availability flag")
  substeps = int(arrays["t_s"].shape[0])
  controls = int(arrays["control_step"].shape[0])
  if substeps <= 0 or controls <= 0:
    raise ValueError("trace payload must contain substep and control rows")

  first_strike_available = bool(arrays["first_strike_available"])
  quality_available = bool(arrays["quality_available"])
  present_first = _FIRST_STRIKE_FIELDS & set(arrays)
  present_quality = _QUALITY_FIELDS & set(arrays)
  if first_strike_available and present_first != _FIRST_STRIKE_FIELDS:
    raise ValueError(
      "first-strike instrumentation is available but snapshot arrays are "
      f"incomplete: {sorted(_FIRST_STRIKE_FIELDS - present_first)}"
    )
  if not first_strike_available and present_first:
    raise ValueError(
      "first-strike instrumentation unavailable but snapshot arrays were stored"
    )
  if quality_available and present_quality != _QUALITY_FIELDS:
    raise ValueError(
      "quality instrumentation is available but snapshot arrays are "
      f"incomplete: {sorted(_QUALITY_FIELDS - present_quality)}"
    )
  if not quality_available and present_quality:
    raise ValueError(
      "quality unavailable: omit quality measurements instead of storing zeros"
    )
  if quality_available and not first_strike_available:
    raise ValueError("quality cannot be available without first-strike tracking")

  control_fields = set(_CONTROL_REQUIRED)
  substep_fields = set(_SUBSTEP_REQUIRED) | present_first | present_quality
  optional_substep = {"cat_delta"} & set(arrays)
  substep_fields |= optional_substep
  known = substep_fields | control_fields | set(_SCALAR_FIELDS)
  unknown = set(arrays) - known
  if unknown:
    raise ValueError(f"trace payload has unclassified rate arrays: {sorted(unknown)}")

  for key in sorted(substep_fields):
    value = arrays[key]
    if value.ndim < 1 or value.shape[0] != substeps:
      raise ValueError(
        f"substep length mismatch for {key}: expected {substeps}, "
        f"got {value.shape}"
      )
  for key in sorted(control_fields):
    value = arrays[key]
    if value.ndim < 1 or value.shape[0] != controls:
      raise ValueError(
        f"control length mismatch for {key}: expected {controls}, "
        f"got {value.shape}"
      )
  for key, value in arrays.items():
    if value.dtype.hasobject:
      raise ValueError(f"object dtype is forbidden in trace payload: {key}")
    if not np.isfinite(value).all():
      raise ValueError(f"nonfinite trace payload array: {key}")

  expected_t = np.arange(substeps, dtype=np.float64) * PHYSICS_DT_S
  if not np.allclose(
    arrays["t_s"], expected_t, rtol=0.0, atol=1e-12
  ):
    raise ValueError("substep timestamps must have exact 2 ms spacing")
  expected_control = np.arange(controls, dtype=np.int64)
  if not np.array_equal(arrays["control_step"], expected_control):
    raise ValueError("control_step must equal arange(control_count)")
  expected_mapping = np.repeat(expected_control, CONTROL_DECIMATION)
  if not np.array_equal(arrays["control_step_index"], expected_mapping):
    raise ValueError(
      "control_step_index must map exactly ten 2 ms substeps to each "
      "20 ms control row"
    )

  expected_shapes = {
    "head_position_m": (substeps, 3),
    "head_velocity_m_s": (substeps, 3),
    "nail_position_m": (substeps, 3),
    "joint_velocity_rad_s": (substeps, 6),
    "joint_position_rad": (substeps, 6),
    "joint_velocity_post_integration_rad_s": (substeps, 6),
    "qfrc_constraint_abs": (substeps, 6),
    "lambda_windowed_constraint_read_n_m_s": (substeps, 6),
    "action": (controls, 3),
    "strike_ref_error_control_m": (controls, 3),
  }
  for key, expected_shape in expected_shapes.items():
    if arrays[key].shape != expected_shape:
      raise ValueError(
        f"{key} shape mismatch: expected {expected_shape}, "
        f"got {arrays[key].shape}"
      )

  phase_contract = {
    "head_position_m": "pre_integration_derived",
    "head_velocity_m_s": "pre_integration_derived",
    "contact": "pre_integration_derived",
    "nail_position_m": "pre_integration_derived",
    "axial_force_n": "pre_integration_derived",
    "joint_velocity_rad_s": "pre_integration",
    "nail_depth_m": "pre_integration",
    "joint_position_rad": "post_integration_legality",
    "joint_velocity_post_integration_rad_s": (
      "post_integration_legality"
    ),
    "nail_depth_post_integration_m": "post_integration_legality",
    "qfrc_constraint_abs": "post_shipped_metrics_current_substep",
    "lambda_windowed_constraint_read_n_m_s": (
      "post_shipped_accumulator_current_substep"
    ),
    "delivered_impulse_n_s": (
      "post_shipped_accumulator_current_substep"
    ),
    "first_strike_snapshot": (
      "post_shipped_metrics_current_substep"
      if first_strike_available
      else "not_available"
    ),
    "quality_snapshot": (
      "post_shipped_metrics_current_substep"
      if quality_available
      else "not_available"
    ),
  }
  return {
    "version": 1,
    "rates": {
      "substep": {
        "hz": 500,
        "dt_s": PHYSICS_DT_S,
        "count": substeps,
        "arrays": sorted(substep_fields),
      },
      "control": {
        "hz": 50,
        "dt_s": CONTROL_DT_S,
        "count": controls,
        "arrays": sorted(control_fields),
      },
      "scalar": {"arrays": sorted(_SCALAR_FIELDS)},
    },
    "phase_contract": phase_contract,
  }


def _canonical_json_bytes(value: Mapping) -> bytes:
  return json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, np.ndarray]) -> str:
  digest = hashlib.sha256()
  for key in sorted(payload):
    value = np.ascontiguousarray(np.asarray(payload[key]))
    header = {
      "key": key,
      "dtype": value.dtype.str,
      "shape": list(value.shape),
    }
    digest.update(_canonical_json_bytes(header))
    digest.update(value.tobytes(order="C"))
  return digest.hexdigest()


def validate_trace_leaf(path: str | Path) -> dict:
  """Fail closed unless a published trace leaf is complete and hash-bound."""
  leaf = Path(path)
  trace_path = leaf / "trace.npz"
  metadata_path = leaf / "metadata.json"
  if not trace_path.is_file() or not metadata_path.is_file():
    raise ValueError(f"incomplete trace leaf: {leaf}")
  try:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"invalid trace metadata: {metadata_path}") from error
  if not isinstance(metadata, dict):
    raise ValueError("trace metadata must be a JSON object")
  required_metadata = {
    "task",
    "checkpoint_sha256",
    "reset_state_digest",
    "code_revision",
    "asset_revision",
    "terminal_reason",
    "timing",
    "schema",
    "payload_sha256",
    "npz_sha256",
    "metadata_sha256",
  }
  missing = required_metadata - set(metadata)
  if missing:
    raise ValueError(f"incomplete trace metadata: {sorted(missing)}")
  if _sha256_file(trace_path) != metadata["npz_sha256"]:
    raise ValueError("trace NPZ SHA-256 mismatch")
  try:
    with np.load(trace_path, allow_pickle=False) as archive:
      payload = {key: archive[key] for key in archive.files}
  except (OSError, ValueError) as error:
    raise ValueError(f"invalid trace NPZ: {trace_path}") from error
  physics_dt = float(metadata["timing"]["physics_dt_s"])
  schema = _validate_trace_payload(payload, physics_dt=physics_dt)
  if schema != metadata["schema"]:
    raise ValueError("trace schema metadata mismatch")
  if _payload_sha256(payload) != metadata["payload_sha256"]:
    raise ValueError("trace payload SHA-256 mismatch")
  digest_metadata = dict(metadata)
  recorded_metadata_digest = digest_metadata.pop("metadata_sha256")
  if (
    hashlib.sha256(_canonical_json_bytes(digest_metadata)).hexdigest()
    != recorded_metadata_digest
  ):
    raise ValueError("trace metadata SHA-256 mismatch")
  return metadata


def _remove_path(path: Path) -> None:
  if path.is_dir():
    shutil.rmtree(path)
  elif path.exists():
    path.unlink()


def _publish_staged_leaf(staged: Path, final: Path) -> None:
  """Replace one leaf on the same filesystem, restoring the old leaf on error."""
  backup = final.with_name(f".{final.name}.backup-{uuid.uuid4().hex}")
  had_final = final.exists()
  if had_final:
    os.replace(final, backup)
  try:
    os.replace(staged, final)
  except BaseException:
    if had_final and backup.exists():
      os.replace(backup, final)
    raise
  if backup.exists():
    _remove_path(backup)


def _write_trace_leaf(
  out_dir: str | Path,
  payload: Mapping[str, np.ndarray],
  *,
  identity: Mapping[str, object],
  physics_dt: float,
) -> dict:
  """Validate, stage, hash, and publish one trace/metadata leaf."""
  schema = _validate_trace_payload(payload, physics_dt=physics_dt)
  required_identity = {
    "mode",
    "campaign",
    "arm",
    "training_seed",
    "task",
    "checkpoint_sha256",
    "reset_state_digest",
    "code_revision",
    "asset_revision",
    "terminal_reason",
  }
  missing_identity = required_identity - set(identity)
  if missing_identity:
    raise ValueError(
      f"trace identity missing fields: {sorted(missing_identity)}"
    )
  leaf = Path(out_dir).resolve()
  leaf.parent.mkdir(parents=True, exist_ok=True)
  staged = Path(
    tempfile.mkdtemp(
      prefix=f".{leaf.name}.staging-",
      dir=leaf.parent,
    )
  )
  try:
    trace_path = staged / "trace.npz"
    np.savez(trace_path, **payload)
    metadata = {
      **dict(identity),
      "timing": {
        "physics_dt_s": PHYSICS_DT_S,
        "control_decimation": CONTROL_DECIMATION,
        "control_dt_s": CONTROL_DT_S,
        "substep_count": int(np.asarray(payload["t_s"]).shape[0]),
        "control_step_count": int(
          np.asarray(payload["control_step"]).shape[0]
        ),
      },
      "schema": schema,
      "payload_sha256": _payload_sha256(payload),
      "npz_sha256": _sha256_file(trace_path),
    }
    metadata["metadata_sha256"] = hashlib.sha256(
      _canonical_json_bytes(metadata)
    ).hexdigest()
    (staged / "metadata.json").write_bytes(
      json.dumps(
        metadata,
        indent=2,
        sort_keys=True,
        allow_nan=False,
      ).encode("utf-8")
      + b"\n"
    )
    validate_trace_leaf(staged)
    _publish_staged_leaf(staged, leaf)
  finally:
    if staged.exists():
      _remove_path(staged)
  return metadata


def _actor_observation_term(
  env: ManagerBasedRlEnv,
  term_name: str,
  env_idx: int,
) -> torch.Tensor:
  """Read one already-computed actor observation without advancing history."""
  observations = env.obs_buf["actor"]
  if isinstance(observations, dict):
    return observations[term_name][env_idx]
  manager = env.observation_manager
  names = manager.active_terms["actor"]
  if term_name not in names:
    raise RuntimeError(f"actor observation has no {term_name!r} term")
  start = 0
  for name, shape in zip(
    names,
    manager.group_obs_term_dim["actor"],
    strict=True,
  ):
    width = int(np.prod(shape))
    if name == term_name:
      return observations[env_idx, start : start + width].reshape(shape)
    start += width
  raise RuntimeError(f"failed to locate actor observation term {term_name!r}")


def _record_control_row(
  recorder: _SubstepTraceRecorder,
  env: ManagerBasedRlEnv,
  action: torch.Tensor,
  env_idx: int,
) -> None:
  recorder.record_control_step(
    action=action[env_idx],
    strike_phase=_actor_observation_term(
      env, "strike_phase", env_idx
    ),
    strike_ref_error_m=_actor_observation_term(
      env, "strike_ref_error", env_idx
    ),
  )


def _terminal_reason(env: ManagerBasedRlEnv, env_idx: int) -> str:
  if bool(env.reset_terminated[env_idx]):
    return "terminated"
  if bool(env.reset_time_outs[env_idx]):
    return "timeout"
  return "done"


def _run_reference(
  args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], float, str, None]:
  """Open-loop scripted single strike (Task-5 idiom: reward_design_util.run_reference_strikes)."""
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  # auto_reset=False: a strike that fires the nail_driven success termination is otherwise reset
  # IN-STEP, zeroing the accumulators before we get to read the terminal strike's Λ_j (the same
  # 2026-07 review finding documented in impulse_bound.py / reward_design_util.py).
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=args.device)
  rec = _install_substep_hook(env, args.env_idx)

  robot = env.scene["robot"]
  nail_e = env.scene["nail_block"]
  rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  rcfg.resolve(env.scene)
  ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  ncfg.resolve(env.scene)

  def head() -> torch.Tensor:
    return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)

  def nail_top() -> torch.Tensor:
    return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

  env.reset()
  ref = SingleStrikeReference(args.num_envs, env.device, approach_height=args.approach_height)
  ref.update(head(), nail_top(), torch.zeros(args.num_envs, dtype=torch.long, device=env.device))
  n = ref.playback_length()
  terminal_reason = "step_limit"
  for k in range(1, n + HOLD_STEPS + 1):
    target = ref.playback_target(min(k, n))
    action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
    _record_control_row(rec, env, action, args.env_idx)
    env.step(action)
    done = env.reset_terminated | env.reset_time_outs
    if bool(done[args.env_idx]):
      terminal_reason = _terminal_reason(env, args.env_idx)
      break
  physics_dt = float(env.physics_dt)
  payload = rec.to_payload(physics_dt=physics_dt)
  env.close()
  return payload, physics_dt, terminal_reason, None


def _run_ckpt(
  args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], float, str, str | None]:
  """Roll out a trained checkpoint (loaded exactly like diag_policy_trace.py)."""
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task, play=args.play)
  env_cfg.scene.num_envs = args.num_envs
  # auto_reset=False (Task 6 deferred fix, 2026-07 review finding): with the default auto_reset=True
  # a success mid-trace resets the env IN-STEP (_reset_idx runs before env.step() returns), zeroing
  # the accumulators/nail state and silently splicing a fresh episode into what is meant to be ONE
  # continuous single-strike trace. Mirrors the SAME idiom already used by _run_reference above /
  # reward_design_util.run_reference_strikes: auto_reset=False + break on the first success keeps the
  # terminal state readable and the trace honest.
  env_cfg.auto_reset = False
  agent_cfg = load_rl_cfg(args.task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  rec = _install_substep_hook(env, args.env_idx)  # hook the raw env BEFORE wrapping
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

  obs, _fixed_reset, reset_digest = _reset_checkpoint_for_inference(
    env,
    wrapped,
    args.fixed_reset_envelope,
  )
  terminal_reason = "step_limit"
  for _ in range(args.nsteps):
    with torch.no_grad():
      actions = policy(obs)
    recorded_actions = (
      actions
      if wrapped.clip_actions is None
      else torch.clamp(
        actions, -wrapped.clip_actions, wrapped.clip_actions
      )
    )
    _record_control_row(rec, env, recorded_actions, args.env_idx)
    obs, rew, dones, extras = wrapped.step(actions)
    # Break on the TRACED env's done only -- success (reset_terminated) OR timeout (reset_time_outs;
    # reachable without --play: training cfg episode_length_s=20 s = 1000 control steps). Either way
    # ITS episode is over; tracing past it would splice a stale terminal state into the figure, and
    # auto_reset=False keeps that terminal state readable. Breaking on ANY env's done (the pre-
    # 2026-07-11 behavior) truncated an --env-idx > 0 trace mid-episode when another env finished
    # first.
    done = env.reset_terminated | env.reset_time_outs
    if bool(done[args.env_idx]):
      terminal_reason = _terminal_reason(env, args.env_idx)
      break
    other_done = torch.nonzero(done, as_tuple=False).flatten()
    if other_done.numel() > 0:
      # A NON-traced env finished: with auto_reset=False it arms _manual_reset_pending and the next
      # step() would raise mjlab's manual-reset RuntimeError. Partial-reset ONLY those envs (envs are
      # physically independent -- the traced env's state/accumulators/nail are untouched, and reset()
      # runs no substeps so the hook records nothing) and keep tracing the traced env's episode.
      env.reset(env_ids=other_done)
  physics_dt = float(env.physics_dt)
  payload = rec.to_payload(physics_dt=physics_dt)
  wrapped.close()
  return payload, physics_dt, terminal_reason, reset_digest


def _make_plot(
  payload: Mapping[str, np.ndarray],
  dt: float,
  j_limit: list[float] | None,
  out_dir: Path,
) -> None:
  t = np.asarray(payload["t_s"])
  n = len(t)
  if n == 0:
    raise RuntimeError("diag_impulse_trace: no substeps were recorded (empty rollout).")
  qfrc = np.asarray(payload["qfrc_constraint_abs"])
  f_axial = np.asarray(payload["axial_force_n"])
  impulse = np.asarray(
    payload["lambda_windowed_constraint_read_n_m_s"]
  )
  contact = np.asarray(payload["contact"], dtype=bool)
  qv = np.abs(np.asarray(payload["joint_velocity_rad_s"]))
  delta = np.asarray(payload.get("cat_delta", np.zeros(n)))

  labels = [f"joint{k + 1}" for k in range(6)]
  fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)

  ax = axes[0]
  for k in range(6):
    ax.plot(t, qfrc[:, k], label=labels[k])
  ax.set_ylabel("|qfrc_constraint|\n(N·m)")
  ax.set_title("(1) Per-joint reaction force -- chain propagation")
  ax.legend(loc="upper right", fontsize=7, ncol=3)

  ax = axes[1]
  ax.plot(t, f_axial, color="black")
  ax.set_ylabel("F_axial (N)")
  ax.set_title("(2) Object-side axial contact force (delivered, weld/friction-immune)")

  ax = axes[2]
  for k in range(6):
    ax.plot(t, impulse[:, k], label=labels[k])
  if j_limit is not None:
    for k in range(6):
      ax.axhline(j_limit[k], linestyle="--", color=f"C{k}", alpha=0.6, linewidth=1)
  ax.set_ylabel("Λ_j (N·m·s)")
  ax.set_title("(3) Shipped running 50 ms windowed constraint read vs J_limit" + (
    "" if j_limit is not None else "  [--j-limit not given: no cap lines]"
  ))
  ax.legend(loc="upper right", fontsize=7, ncol=3)

  ax = axes[3]
  ax.plot(t, delta, color="crimson")
  ax.set_ylabel("δ")
  ax.set_title("(4) CaT soft-violation probability (control-rate, forward-filled)")

  ax = axes[4]
  for k in range(6):
    ax.plot(t, qv[:, k], label=labels[k])
  ax.set_ylabel("|q̇_j| (rad/s)")
  ax.set_xlabel("time (s)")
  ax.set_title("(5) Per-joint speed")
  ax.legend(loc="upper right", fontsize=7, ncol=3)

  for ax in axes:  # shade contact windows on every panel
    ylo, yhi = ax.get_ylim()
    ax.fill_between(t, ylo, yhi, where=contact, color="gray", alpha=0.12, step="mid")
    ax.set_ylim(ylo, yhi)

  fig.tight_layout()
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / "trace.png", dpi=150)
  plt.close(fig)

  print(
    f"[diag_impulse_trace] wrote {out_dir / 'trace.png'} "
    f"({n} substeps, {n * dt:.3f}s, peak 50 ms read="
    f"{impulse.max(axis=0)})"
  )


def _build_arg_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default=None, help="trained checkpoint (.pt); XOR with --reference")
  ap.add_argument("--reference", action="store_true", help="open-loop scripted single strike; XOR with --ckpt")
  ap.add_argument("--task", default="Unitree-Z1-Hammer-CaT-Impulse",
                  help="must be a cat_impulse=True task (both accumulators + the hook wired)")
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--env-idx", type=int, default=0)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--out", default="/tmp/impulse_trace")
  ap.add_argument("--j-limit", default=None,
                  help="CSV of 6 floats (N·m·s), e.g. 1.640,3.280,1.640,1.640,1.640,1.640 -- "
                       "REQUIRED to draw the J_limit cap lines in panel 3. NEVER defaulted "
                       "(the placeholder Z1_JOINT_IMPULSE_LIMIT=0.1 is not a real cap); omitting "
                       "this flag draws no cap lines.")
  ap.add_argument("--approach-height", type=float, default=0.10, help="--reference mode: strike apex height (m)")
  ap.add_argument("--nsteps", type=int, default=80, help="--ckpt mode: control steps to roll out")
  ap.add_argument("--play", action="store_true", help="play-mode env cfg (zeroed reset/obs noise)")
  ap.add_argument(
    "--fixed-reset-envelope",
    default=None,
    help=(
      "approved fixed-reset JSON envelope for checkpoint companion mode; "
      "optional so historical checkpoint diagnostics remain available"
    ),
  )
  ap.add_argument("--campaign", default=None)
  ap.add_argument("--arm", default=None)
  ap.add_argument("--training-seed", type=int, default=None)
  ap.add_argument(
    "--checkpoint-sha256",
    default=None,
    help="expected checkpoint SHA-256 (required with --fixed-reset-envelope)",
  )
  ap.add_argument(
    "--code-revision",
    default=None,
    help="source revision input (required with --fixed-reset-envelope)",
  )
  ap.add_argument(
    "--asset-revision",
    default=None,
    help="asset revision input (required with --fixed-reset-envelope)",
  )
  return ap


def main() -> None:
  args = _build_arg_parser().parse_args()

  if bool(args.ckpt) == bool(args.reference):
    raise SystemExit("diag_impulse_trace: pass exactly one of --ckpt or --reference.")
  if args.device != "cpu":
    raise SystemExit("diag_impulse_trace: evaluation is CPU-only.")
  if args.num_envs < 1 or not 0 <= args.env_idx < args.num_envs:
    raise SystemExit(
      "diag_impulse_trace: --env-idx must select one configured environment."
    )
  if args.nsteps <= 0:
    raise SystemExit("diag_impulse_trace: --nsteps must be positive.")
  if args.reference and args.fixed_reset_envelope is not None:
    raise SystemExit(
      "diag_impulse_trace: --fixed-reset-envelope is checkpoint-only."
    )

  j_limit: list[float] | None = None
  if args.j_limit:
    j_limit = [float(x) for x in args.j_limit.split(",")]
    if len(j_limit) != 6 or not np.isfinite(j_limit).all():
      raise SystemExit(f"--j-limit must be exactly 6 comma-separated floats, got {len(j_limit)}: {args.j_limit}")

  checkpoint_sha256 = None
  if args.ckpt:
    checkpoint = Path(args.ckpt)
    if not checkpoint.is_file():
      raise SystemExit(f"diag_impulse_trace: checkpoint not found: {checkpoint}")
    checkpoint_sha256 = _sha256_file(checkpoint)
    if (
      args.checkpoint_sha256 is not None
      and args.checkpoint_sha256 != checkpoint_sha256
    ):
      raise SystemExit(
        "diag_impulse_trace: checkpoint SHA-256 mismatch: "
        f"expected {args.checkpoint_sha256}, got {checkpoint_sha256}"
      )

  if args.fixed_reset_envelope is not None:
    required = {
      "--campaign": args.campaign,
      "--arm": args.arm,
      "--training-seed": args.training_seed,
      "--checkpoint-sha256": args.checkpoint_sha256,
      "--code-revision": args.code_revision,
      "--asset-revision": args.asset_revision,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
      raise SystemExit(
        "diag_impulse_trace: fixed-reset companion mode requires "
        + ", ".join(missing)
      )
    if args.num_envs != 1 or args.env_idx != 0:
      raise SystemExit(
        "diag_impulse_trace: fixed-reset companion mode requires "
        "--num-envs 1 --env-idx 0."
      )
    registered_task = expected_task(args.campaign, args.arm)
    if args.task != registered_task:
      raise SystemExit(
        "diag_impulse_trace: task identity mismatch: "
        f"{args.campaign}/{args.arm} requires {registered_task}, "
        f"got {args.task}"
      )
    for name, revision in (
      ("code", args.code_revision),
      ("asset", args.asset_revision),
    ):
      if "dirty" in revision.lower():
        raise SystemExit(
          f"diag_impulse_trace: {name} revision must not be dirty: "
          f"{revision}"
        )

  payload, dt, terminal_reason, reset_digest = (
    _run_reference(args) if args.reference else _run_ckpt(args)
  )
  identity = {
    "mode": "reference" if args.reference else "checkpoint",
    "campaign": args.campaign,
    "arm": args.arm,
    "training_seed": args.training_seed,
    "task": args.task,
    "checkpoint_file": (
      None if args.ckpt is None else str(Path(args.ckpt).resolve())
    ),
    "checkpoint_sha256": checkpoint_sha256,
    "fixed_reset_envelope": args.fixed_reset_envelope,
    "reset_state_digest": reset_digest,
    "code_revision": args.code_revision,
    "asset_revision": args.asset_revision,
    "terminal_reason": terminal_reason,
    "j_limit_n_m_s": j_limit,
  }
  out_dir = Path(args.out)
  _write_trace_leaf(
    out_dir,
    payload,
    identity=identity,
    physics_dt=dt,
  )
  _make_plot(payload, dt, j_limit, out_dir)
  print(
    f"[diag_impulse_trace] wrote {out_dir / 'trace.npz'} and "
    f"{out_dir / 'metadata.json'}"
  )


if __name__ == "__main__":
  main()
