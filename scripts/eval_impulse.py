"""Checkpoint evaluation for impulse diagnostics and the first-strike campaign.

The legacy mean-action summary remains a diagnostic.  The sampled primary
evaluates the checkpoint in its exact C/D-prime/F/E task with stochastic
actions, training-equivalent reset noise, and independent action/reset/
observation RNG streams.  It records exactly two completed episodes from each
of 256 environments, with shared 500 Hz traces, episode metrics, frozen nail
geometry, and provenance written to ``summary.csv`` and compressed raw data.

Both paths snapshot metrics before auto-reset so episode-final accumulator
state cannot be lost. Campaign rows are written before fail-closed provenance,
sentinel, or nonfinite joint-velocity samples exit with status 2. Finite
hardware-rail exceedances remain in the simulation comparison and are reported
as hardware-speed qualifications rather than exclusions.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import math
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab.tasks  # noqa: F401  (register builtin tasks)
import src.tasks  # noqa: F401  (register hammer tasks)
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT
from src.tasks.hammer.mdp.first_strike import (
  REASON_SUCCESS,
  REASON_WINDOW,
  _ENV_FIRST_STRIKE_ATTR,
)
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR, _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.mdp.rewards import clamped_nail_depth
from evaluation.analysis.first_strike_campaign import (
    ARM_TASKS,
    EXPECTED_CONTROL_DECIMATION,
    EXPECTED_EPISODES_PER_ENV,
    EXPECTED_EPISODES_PER_SEED,
    EXPECTED_FIXED_ACTION_SIGNATURE,
    EXPECTED_FIXED_ACTION_SIGNATURE_SHA256,
    EXPECTED_FIXED_ACTUATOR_SIGNATURE,
    EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256,
    EXPECTED_PHYSICS_DT_S,
    HARDWARE_QVEL_LIMIT_RAD_S,
  REQUIRED_SAMPLED_COLUMNS,
  aggregate_episode_metrics,
  load_frozen_nail_geometry,
  summarize_episode,
)

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
TASK_TO_ARM = {task: arm for arm, task in ARM_TASKS.items()}
QUALITY_ARM_TASKS = {
  "F8": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
  "F0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
  "D0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
  "FQ": "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
}
QUALITY_TASK_TO_ARM = {task: arm for arm, task in QUALITY_ARM_TASKS.items()}
QUALITY_EVALUATION_REFERENCE_TASK = QUALITY_ARM_TASKS["FQ"]
RESET_SEED_OFFSET = 10_000_019
OBSERVATION_SEED_OFFSET = 20_000_033
ACTION_SEED_OFFSET = 30_000_041
PAYOUT_SEMANTICS = {
  "C": "actual_legacy_repeated",
  "D-prime": "actual_legacy_first_event",
  "F": "actual_event_linear",
  "E": "actual_event_saturated",
  "F8": "actual_event_linear",
  "F0": "actual_event_linear_speed_disabled",
  "D0": "actual_event_linear_delivered_disabled",
  "FQ": "actual_event_quality_bounded",
}

_TRACE_PHYSICAL_KEYS = (
  "contact", "head_position_m", "clamped_depth_m", "net_axial_force_n",
  "joint_speed_rad_s", "post_step_joint_speed_rad_s",
  "tracker_depth_post_integration_m", "quality_found_count",
  "quality_normal_force_n", "quality_contact_position_m",
  "quality_contact_normal",
)
_TRACE_EVENT_KEYS = (
  "tracker_started", "tracker_finalized", "tracker_productive",
  "tracker_reason", "event_cumulative_impulse_n_s",
  "tracker_contact_point_w", "tracker_contact_error_m",
  "tracker_contact_quality", "tracker_contact_quality_valid",
  "tracker_contact_quality_overflow", "tracker_first_contact_time_s",
  "tracker_contact_normal_axiality",
  "event_cumulative_transverse_impulse_n_s",
)
_TRACE_FIRST_STRIKE_PHYSICAL_KEYS = (
  "started", "finalized", "reason", "accepted_onset_index", "productive",
  "v_precontact_m_s", "delivered_n_s", "delivered_transverse_n_s",
  "contact_point_w", "contact_error_m", "contact_quality",
  "contact_quality_valid", "contact_quality_overflow", "first_contact_time_s",
  "contact_normal_axiality",
)
_TRACE_EPISODE_PHYSICAL_KEYS = (
  "overall_success", "episode_peak_lambda",
  "episode_delivered_accumulator_n_s", "episode_depth_m",
)
_INSTRUMENTATION_ONLY_TRACE_KEYS = {
  "physical": {
    "quality_found_count", "quality_normal_force_n",
    "quality_contact_position_m", "quality_contact_normal",
  },
  "event_trace": {
    "tracker_contact_point_w", "tracker_contact_error_m",
    "tracker_contact_quality", "tracker_contact_quality_valid",
    "tracker_contact_quality_overflow", "tracker_contact_normal_axiality",
  },
  "first_strike": {
    "contact_point_w", "contact_error_m", "contact_quality",
    "contact_quality_valid", "contact_quality_overflow",
    "contact_normal_axiality",
  },
}

FIELDNAMES = [
  "name", "ckpt_path", "checkpoint_path", "checkpoint_sha256",
  "task", "treatment", "training_seed",
  "num_envs", "nsteps", "episode_len_s", "seed", "imp_max_p",
  "impact_weight", "delivered_weight", "event_i_ref_n_s",
  "n_episodes",
  *[f"lambda_max_{j}" for j in ARM_JOINTS],
  *[f"lambda_p95_{j}" for j in ARM_JOINTS],
  "worst_ratio_max", "worst_ratio_mean",
  "delivered_mean", "delivered_std",
  "success_rate", "nail_depth_mean_mm", "nail_depth_std_mm",
  "ep_len_mean", "ep_len_std",
  # Physical-impossibility invariants (Tier-1 safety net, 2026-07-14) -- counted over BOTH
  # rollouts; MUST both be 0 or the script exits 2 after writing the row (see main()).
  "impossible_success_n", "lambda_dead_n",
  # sampled-action repeat (robustness check) -- condensed, NOT a second row (brief: rows=checkpoints)
  "n_episodes_sampled", "success_rate_sampled", "worst_ratio_max_sampled", "delivered_mean_sampled",
  "episodes_per_env_sampled",
  *sorted(REQUIRED_SAMPLED_COLUMNS - {
    "impossible_success_n", "lambda_dead_n",
  }),
  "action_rng_seed", "reset_rng_seed", "observation_rng_seed",
  "reset_position_noise_min_rad", "reset_position_noise_max_rad",
  "actor_observation_corruption", "critic_observation_corruption",
  "sampled_completion_rule", "sampled_actions_stochastic",
  "physics_dt_s", "control_decimation",
  "fixed_impedance_signature_sha256",
  "fixed_action_signature_sha256",
  "accepted_checkpoint_sha256",
  "accepted_manifest_sha256", "training_code_revision",
  "training_asset_revision",
  "sampled_trace_path", "sampled_trace_digest",
  "sampled_trace_artifact_sha256", "campaign_config_sha256",
  "treatment_config_sha256",
  "nail_asset_sha256", "host", "timestamp_utc",
  "git_hash", "git_revision", "git_dirty",
  "asset_git_hash", "asset_git_revision", "asset_git_dirty",
]


def _retain_first_completed(
  record: dict,
  *,
  counts: list[int],
  selected: list[dict],
  episodes_per_env: int = EXPECTED_EPISODES_PER_ENV,
) -> bool:
  """Retain one completion iff its environment has not reached the quota."""
  if not counts or episodes_per_env < 1:
    raise ValueError("counts and episodes_per_env must be positive")
  env_id = int(record["env_id"])
  if env_id < 0 or env_id >= len(counts):
    raise ValueError(f"completed episode has invalid env_id={env_id}")
  if counts[env_id] >= episodes_per_env:
    return False
  selected.append(record)
  counts[env_id] += 1
  return True


def _canonical_digest(value) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode()
  return hashlib.sha256(encoded).hexdigest()


def _physical_trace_payload(trace: Mapping) -> dict:
  """Extract replay-relevant channels, deliberately excluding reward payouts."""
  physical = trace.get("physical", {})
  event_trace = trace.get("event_trace", {})
  first_strike = trace.get("first_strike", {})
  missing = {
    "physical": [key for key in _TRACE_PHYSICAL_KEYS if key not in physical],
    "event_trace": [key for key in _TRACE_EVENT_KEYS if key not in event_trace],
    "first_strike": [
      key for key in _TRACE_FIRST_STRIKE_PHYSICAL_KEYS if key not in first_strike
    ],
    "episode_final": [
      key for key in _TRACE_EPISODE_PHYSICAL_KEYS if key not in trace
    ],
  }
  if any(missing.values()):
    raise ValueError(f"sampled trace missing physical channels: {missing}")
  return {
    "physical": {key: physical[key] for key in _TRACE_PHYSICAL_KEYS},
    "event_trace": {key: event_trace[key] for key in _TRACE_EVENT_KEYS},
    "first_strike": {
      key: first_strike[key] for key in _TRACE_FIRST_STRIKE_PHYSICAL_KEYS
    },
    "episode_final": {
      key: trace[key] for key in _TRACE_EPISODE_PHYSICAL_KEYS
    },
  }


def _physical_trace_digest(trace: Mapping) -> str:
  return _canonical_digest(_physical_trace_payload(trace))


def _plant_replay_payload(trace: Mapping) -> dict:
  """Physical replay projection when one trace lacks passive quality sensing.

  Strict F8/F0/D0/FQ comparison must use the full physical digest. This
  narrower projection exists only for the instrumentation on/off control: its
  removed fields cannot exist in an uninstrumented tracker and therefore say
  nothing about whether the action tape changed MuJoCo state.
  """
  payload = _physical_trace_payload(trace)
  return {
    section: {
      key: value
      for key, value in values.items()
      if key not in _INSTRUMENTATION_ONLY_TRACE_KEYS.get(section, set())
    }
    for section, values in payload.items()
  }


def _require_finite_trace_value(value, *, path: str) -> None:
  if isinstance(value, Mapping):
    for key, item in value.items():
      _require_finite_trace_value(item, path=f"{path}.{key}")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _require_finite_trace_value(item, path=f"{path}[{index}]")
  elif isinstance(value, (float, int, np.floating, np.integer)):
    if not math.isfinite(float(value)):
      raise ValueError(f"nonfinite physical trace value at {path}")
  elif value is None:
    if path.endswith("accepted_onset_index"):
      return
    raise ValueError(f"nonfinite physical trace value at {path}")


def _validate_quality_trace(trace: Mapping) -> None:
  event = trace["event_trace"]
  snapshot = trace["first_strike"]
  valid = np.asarray(event["tracker_contact_quality_valid"], dtype=bool)
  overflow = np.asarray(event["tracker_contact_quality_overflow"], dtype=bool)
  if np.any(valid & overflow):
    raise ValueError("overflowed quality cannot be valid in event stream")
  if bool(snapshot["contact_quality_valid"]) and bool(
    snapshot["contact_quality_overflow"]
  ):
    raise ValueError("overflowed quality cannot be valid in first-strike snapshot")
  if len(valid) and bool(snapshot["started"]):
    for stream_key, snapshot_key in (
      ("tracker_contact_point_w", "contact_point_w"),
      ("tracker_contact_error_m", "contact_error_m"),
      ("tracker_contact_quality", "contact_quality"),
      ("tracker_contact_quality_valid", "contact_quality_valid"),
      ("tracker_contact_quality_overflow", "contact_quality_overflow"),
      ("tracker_first_contact_time_s", "first_contact_time_s"),
      ("tracker_contact_normal_axiality", "contact_normal_axiality"),
      ("event_cumulative_transverse_impulse_n_s", "delivered_transverse_n_s"),
    ):
      if event[stream_key][-1] != snapshot[snapshot_key]:
        raise ValueError(
          f"first-strike snapshot disagrees with event stream for {snapshot_key}"
        )


def compare_action_tape_physics(
  traces: Mapping[str, Mapping],
) -> dict[str, object]:
  """Require byte-identical physical/event channels for F8/F0/D0/FQ replay."""
  if not traces:
    raise ValueError("action-tape replay requires at least one trace")
  digests: dict[str, str] = {}
  for arm, trace in traces.items():
    payload = _physical_trace_payload(trace)
    _require_finite_trace_value(payload, path=str(arm))
    _validate_quality_trace(trace)
    digest = _canonical_digest(payload)
    recorded = trace.get("trace_digest")
    if recorded is not None and str(recorded) != digest:
      raise ValueError(f"{arm}: recorded physical trace digest mismatch")
    digests[str(arm)] = digest
  if len(set(digests.values())) != 1:
    raise ValueError(f"physical/event replay drift across action tapes: {digests}")
  return {
    "arms": tuple(sorted(digests)),
    "physical_digest": next(iter(digests.values())),
  }


def _sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _verify_unchanged_file(
  path: str | Path, *, expected_sha256: str, label: str
) -> None:
  actual = _sha256_file(path)
  if actual != expected_sha256:
    raise RuntimeError(
      f"{label} changed during evaluation: expected {expected_sha256}, "
      f"found {actual}"
    )


def _verify_expected_checkpoint_sha256(
  path: str | Path, *, expected_sha256: str
) -> None:
  actual = _sha256_file(path)
  if actual != expected_sha256:
    raise RuntimeError(
      "accepted checkpoint SHA-256 mismatch before rollout: "
      f"expected {expected_sha256}, found {actual}"
    )


@contextmanager
def _frozen_checkpoint(
  source: str | Path, *, snapshot_parent: str | Path
):
  """Load both policies from one private byte snapshot of the checkpoint."""
  source_path = Path(source).resolve(strict=True)
  source_sha256 = _sha256_file(source_path)
  with TemporaryDirectory(
    prefix=".eval-impulse-checkpoint-", dir=Path(snapshot_parent)
  ) as directory:
    snapshot_path = Path(directory) / source_path.name
    shutil.copyfile(source_path, snapshot_path)
    snapshot_sha256 = _sha256_file(snapshot_path)
    if snapshot_sha256 != source_sha256:
      raise RuntimeError(
        "checkpoint changed while creating the immutable evaluation snapshot"
      )
    try:
      yield source_path, snapshot_path, snapshot_sha256
    finally:
      _verify_unchanged_file(
        source_path,
        expected_sha256=snapshot_sha256,
        label="checkpoint",
      )


def _json_values(tensor: torch.Tensor):
  """Convert a tensor to JSON data while retaining nonfinite qvel as null."""
  def clean(value):
    if isinstance(value, list):
      return [clean(item) for item in value]
    if isinstance(value, (bool, int)):
      return value
    number = float(value)
    return number if math.isfinite(number) else None

  return clean(tensor.detach().cpu().tolist())


class _TorchRngStream:
  """A frozen torch RNG stream isolated from the ambient/device stream."""

  def __init__(self, seed: int, device: str):
    self.seed = int(seed)
    self.device = torch.device(device)
    generator = torch.Generator(device=self.device)
    generator.manual_seed(self.seed)
    self._state = generator.get_state()

  def run(self, function):
    if self.device.type == "cuda":
      ambient = torch.cuda.get_rng_state(self.device)
      torch.cuda.set_rng_state(self._state, self.device)
      try:
        return function()
      finally:
        self._state = torch.cuda.get_rng_state(self.device)
        torch.cuda.set_rng_state(ambient, self.device)
    ambient = torch.random.get_rng_state()
    torch.random.set_rng_state(self._state)
    try:
      return function()
    finally:
      self._state = torch.random.get_rng_state()
      torch.random.set_rng_state(ambient)


class _PreIntegrationTracePhase:
  """Pair stale derived channels with state cached immediately before mj_step."""

  def __init__(self):
    self._joint_speed_rad_s: torch.Tensor | None = None
    self._clamped_depth_m: torch.Tensor | None = None

  def cache_pre_step(
    self,
    *,
    joint_speed_rad_s: torch.Tensor,
    clamped_depth_m: torch.Tensor,
  ) -> None:
    if self._joint_speed_rad_s is not None or self._clamped_depth_m is not None:
      raise RuntimeError(
        "unconsumed pre-integration cache before the next sim.step"
      )
    self._joint_speed_rad_s = joint_speed_rad_s.detach().clone()
    self._clamped_depth_m = clamped_depth_m.detach().clone()

  def capture(
    self,
    *,
    head_position_m: torch.Tensor,
    contact: torch.Tensor,
    net_axial_force_n: torch.Tensor,
    post_step_joint_speed_rad_s: torch.Tensor,
    post_step_clamped_depth_m: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    if self._joint_speed_rad_s is None or self._clamped_depth_m is None:
      raise RuntimeError("sim.step must cache pre-integration state before capture")
    sample = {
      "head_position_m": head_position_m.detach().clone(),
      "contact": contact.detach().clone(),
      "net_axial_force_n": net_axial_force_n.detach().clone(),
      "joint_speed_rad_s": self._joint_speed_rad_s,
      "clamped_depth_m": self._clamped_depth_m,
      "post_step_joint_speed_rad_s": (
        post_step_joint_speed_rad_s.detach().clone()
      ),
      "tracker_depth_post_integration_m": (
        post_step_clamped_depth_m.detach().clone()
      ),
    }
    self._joint_speed_rad_s = None
    self._clamped_depth_m = None
    return sample


def _install_evaluator_rng_streams(
  env: ManagerBasedRlEnv,
  *,
  reset_seed: int,
  observation_seed: int,
) -> tuple[_TorchRngStream, _TorchRngStream]:
  """Isolate training-matched reset and observation randomness."""
  reset_stream = _TorchRngStream(reset_seed, env.device)
  observation_stream = _TorchRngStream(observation_seed, env.device)

  original_reset_idx = env._reset_idx

  def reset_idx(env_ids=None):
    return reset_stream.run(lambda: original_reset_idx(env_ids))

  env._reset_idx = reset_idx  # type: ignore[method-assign]

  original_observation_compute = env.observation_manager.compute

  def observation_compute(*args, **kwargs):
    return observation_stream.run(
      lambda: original_observation_compute(*args, **kwargs)
    )

  env.observation_manager.compute = observation_compute  # type: ignore[method-assign]
  return reset_stream, observation_stream


def _ensure_first_strike_instrumentation(env_cfg) -> None:
  """Install the event tracker on C without changing its reward treatment."""
  if "first_strike" not in env_cfg.metrics:
    reference = load_env_cfg(ARM_TASKS["E"], play=False)
    env_cfg.metrics["first_strike"] = copy.deepcopy(
      reference.metrics["first_strike"]
    )


def _freeze_config_value(value):
  """Convert config objects to deterministic, JSON-safe identity data."""
  if isinstance(value, Mapping):
    return {
      str(key): _freeze_config_value(item)
      for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }
  if isinstance(value, (tuple, list)):
    return [_freeze_config_value(item) for item in value]
  if isinstance(value, set):
    return sorted(_freeze_config_value(item) for item in value)
  if is_dataclass(value):
    return {
      field.name: _freeze_config_value(getattr(value, field.name))
      for field in fields(value)
    }
  if callable(value):
    return f"{value.__module__}.{value.__qualname__}"
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, (str, bool, int, float)) or value is None:
    return value
  return repr(value)


def _policy_observation_signature(env_cfg) -> dict:
  return _freeze_config_value(env_cfg.observations["actor"])


def _treatment_reward_signature(env_cfg) -> dict:
  return {
    name: _freeze_config_value(cfg)
    for name, cfg in sorted(env_cfg.rewards.items())
  }


def _strict_evaluation_config_signature(env_cfg) -> dict:
  """Bank the complete fixed-impedance evaluator configuration identity."""
  return {
    "observations": _policy_observation_signature(env_cfg),
    "rewards": _treatment_reward_signature(env_cfg),
    "actions": _freeze_config_value(env_cfg.actions),
    "actuators": _freeze_config_value(
      env_cfg.scene.entities["robot"].articulation.actuators
    ),
    "metrics": _freeze_config_value(env_cfg.metrics),
    "sensors": _freeze_config_value(env_cfg.scene.sensors or ()),
    "terminations": _freeze_config_value(env_cfg.terminations),
    "sim": _freeze_config_value(env_cfg.sim),
    "decimation": int(env_cfg.decimation),
  }


def build_strict_quality_evaluation_cfg(task: str, *, play: bool):
  """Build a quality-instrumented strict eval config without changing treatment."""
  if task not in QUALITY_TASK_TO_ARM:
    raise ValueError(
      f"strict quality evaluation requires one of {tuple(QUALITY_TASK_TO_ARM)}, got {task!r}"
    )
  training_cfg = load_env_cfg(task, play=play)
  evaluation_cfg = copy.deepcopy(training_cfg)
  reference_cfg = load_env_cfg(QUALITY_EVALUATION_REFERENCE_TASK, play=play)
  quality_sensor = next(
    sensor for sensor in (reference_cfg.scene.sensors or ())
    if sensor.name == "hammer_nail_quality"
  )
  if not any(
    sensor.name == "hammer_nail_quality"
    for sensor in (evaluation_cfg.scene.sensors or ())
  ):
    evaluation_cfg.scene.sensors = (
      *(evaluation_cfg.scene.sensors or ()), copy.deepcopy(quality_sensor)
    )
  evaluation_cfg.metrics["first_strike"] = copy.deepcopy(
    reference_cfg.metrics["first_strike"]
  )
  training_observation = _canonical_digest(
    _policy_observation_signature(training_cfg)
  )
  evaluation_observation = _canonical_digest(
    _policy_observation_signature(evaluation_cfg)
  )
  training_reward = _canonical_digest(_treatment_reward_signature(training_cfg))
  evaluation_reward = _canonical_digest(
    _treatment_reward_signature(evaluation_cfg)
  )
  if training_observation != evaluation_observation:
    raise ValueError("quality instrumentation changed policy observations")
  if training_reward != evaluation_reward:
    raise ValueError("quality instrumentation changed treatment rewards")
  return training_cfg, evaluation_cfg, {
    "training_config_sha256": _canonical_digest(
      _strict_evaluation_config_signature(training_cfg)
    ),
    "evaluation_config_sha256": _canonical_digest(
      _strict_evaluation_config_signature(evaluation_cfg)
    ),
    "training_policy_observation_sha256": training_observation,
    "evaluation_policy_observation_sha256": evaluation_observation,
    "training_treatment_reward_sha256": training_reward,
    "evaluation_treatment_reward_sha256": evaluation_reward,
  }


def _validate_sampled_env_contract(env_cfg, task: str) -> dict:
  if task not in TASK_TO_ARM and task not in QUALITY_TASK_TO_ARM:
    raise ValueError(
      "sampled first-strike evaluation requires a registered legacy or strict "
      f"quality task, got {task!r}"
    )
  treatment = TASK_TO_ARM.get(task, QUALITY_TASK_TO_ARM.get(task))
  impact = env_cfg.rewards["impact_progress"]
  delivered = env_cfg.rewards["delivered_impulse"]
  impact_weight = float(impact.weight)
  delivered_weight = float(delivered.weight)
  expected_weights = {
    "F0": (0.0, 2.0), "D0": (8.0, 0.0),
  }.get(treatment, (8.0, 2.0))
  if (impact_weight, delivered_weight) != expected_weights:
    raise ValueError(
      f"{task}: configured maximize weights must be exactly "
      f"{expected_weights[0]}/{expected_weights[1]}, got "
      f"{impact_weight}/{delivered_weight}"
    )
  reset_range = tuple(
    float(value)
    for value in env_cfg.events["reset_robot_joints"].params["position_range"]
  )
  if reset_range != (-0.05, 0.05):
    raise ValueError(
      f"{task}: sampled evaluation requires training reset noise (-0.05, 0.05)"
    )
  actor_corruption = bool(env_cfg.observations["actor"].enable_corruption)
  critic_corruption = bool(env_cfg.observations["critic"].enable_corruption)
  if not actor_corruption or critic_corruption:
    raise ValueError(
      f"{task}: actor corruption must be on and critic corruption must be off"
    )
  if float(env_cfg.metrics["cat_soft"].params["imp_max_p"]) != 0.0:
    raise ValueError(f"{task}: sampled evaluation requires imp_max_p=0")
  impulse_limits = tuple(
    float(value)
    for value in env_cfg.metrics["cat_soft"].params["imp_limit"]
  )
  if impulse_limits != tuple(float(value) for value in IMP_J_LIMIT):
    raise ValueError(
      f"{task}: sampled evaluation impulse limits do not match IMP_J_LIMIT"
    )
  actuator_signature = tuple(
    (
      type(actuator).__name__,
      tuple(actuator.target_names_expr),
      float(actuator.stiffness),
      float(actuator.damping),
      float(actuator.effort_limit),
      float(actuator.armature),
    )
    for actuator in env_cfg.scene.entities["robot"].articulation.actuators
  )
  if actuator_signature != EXPECTED_FIXED_ACTUATOR_SIGNATURE:
    raise ValueError(f"{task}: fixed-impedance actuator signature drift")
  def freeze_config_value(value):
    if isinstance(value, dict):
      return tuple(
        (str(key), freeze_config_value(item))
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
      )
    if isinstance(value, (tuple, list)):
      return tuple(freeze_config_value(item) for item in value)
    return value

  action_signature = tuple(
    (
      name,
      type(action).__name__,
      tuple(
        (
          field.name,
          freeze_config_value(getattr(action, field.name)),
        )
        for field in fields(action)
      ),
    )
    for name, action in env_cfg.actions.items()
  )
  if action_signature != EXPECTED_FIXED_ACTION_SIGNATURE:
    raise ValueError(f"{task}: fixed action signature drift")
  physics_dt_s = _physics_timestep_s(env_cfg)
  if not math.isclose(physics_dt_s, EXPECTED_PHYSICS_DT_S, rel_tol=0.0, abs_tol=1e-15):
    raise ValueError(f"{task}: physics timestep must remain 0.002 s (500 Hz)")
  decimation = int(env_cfg.decimation)
  if decimation != EXPECTED_CONTROL_DECIMATION:
    raise ValueError(f"{task}: control decimation must remain 10")
  return {
    "treatment": treatment,
    "impact_weight": impact_weight,
    "delivered_weight": delivered_weight,
    "reset_position_noise_min_rad": reset_range[0],
    "reset_position_noise_max_rad": reset_range[1],
    "actor_observation_corruption": actor_corruption,
    "critic_observation_corruption": critic_corruption,
    "event_i_ref_n_s": float(delivered.params["i_ref"]),
    "impulse_limits_n_m_s": list(impulse_limits),
    "physics_dt_s": physics_dt_s,
    "control_decimation": decimation,
    "fixed_impedance_signature_sha256": (
      EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256
    ),
    "fixed_action_signature_sha256": EXPECTED_FIXED_ACTION_SIGNATURE_SHA256,
  }


def _campaign_config_digest(
  *,
  contract: dict,
  num_envs: int,
  episode_len_s: float,
  physics_dt_s: float,
  decimation: int,
  reset_seed: int,
  observation_seed: int,
  action_seed: int,
) -> str:
  """Hash only the comparison protocol shared by all intended treatments."""
  shared_contract = {
    key: contract[key]
    for key in (
      "impact_weight",
      "delivered_weight",
      "reset_position_noise_min_rad",
      "reset_position_noise_max_rad",
      "actor_observation_corruption",
      "critic_observation_corruption",
      "impulse_limits_n_m_s",
      "fixed_impedance_signature_sha256",
      "fixed_action_signature_sha256",
    )
  }
  return _canonical_digest(
    {
      "schema_version": 2,
      "num_envs": num_envs,
      "episodes_per_env": EXPECTED_EPISODES_PER_ENV,
      "episode_len_s": episode_len_s,
      "physics_dt_s": physics_dt_s,
      "decimation": decimation,
      "imp_max_p": 0.0,
      "contract": shared_contract,
      "rng_streams": {
        "reset": reset_seed,
        "observation": observation_seed,
        "action": action_seed,
      },
    }
  )


def _treatment_config_digest(*, task: str, contract: dict) -> str:
  """Hash reward semantics that intentionally differ between campaign arms."""
  treatment = str(contract["treatment"])
  return _canonical_digest(
    {
      "schema_version": 1,
      "task": task,
      "treatment": treatment,
      "payout_semantics": PAYOUT_SEMANTICS[treatment],
      "impact_weight": float(contract["impact_weight"]),
      "delivered_weight": float(contract["delivered_weight"]),
      "event_i_ref_n_s": float(contract["event_i_ref_n_s"]),
    }
  )


def _physics_timestep_s(env_cfg) -> float:
  """Return the live MuJoCo integration timestep from SimulationCfg."""
  timestep = float(env_cfg.sim.mujoco.timestep)
  if not math.isfinite(timestep) or timestep <= 0.0:
    raise ValueError(f"invalid MuJoCo timestep {timestep!r}")
  return timestep


class _SampledTraceCollector:
  """Extend the evaluator hook with one shared 500 Hz trace per episode."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    snapshot: dict,
    treatment: str,
    task: str,
    gamma: float,
    event_i_ref_n_s: float,
    nail_geometry: dict,
  ):
    self.env = env
    self.snapshot = snapshot
    self.treatment = treatment
    self.task = task
    self.gamma = float(gamma)
    self.event_i_ref_n_s = float(event_i_ref_n_s)
    self.nail_geometry = copy.deepcopy(nail_geometry)
    self.completed: list[dict] = []
    self.accepted_counts = [0] * env.num_envs
    self._seen_counts = [0] * env.num_envs
    self._substep_start = [0] * env.num_envs
    self._control_start = [0] * env.num_envs
    self._substeps: dict[str, list[torch.Tensor]] = {
      key: []
      for key in (
        "contact", "head_position_m", "clamped_depth_m",
        "net_axial_force_n", "joint_speed_rad_s",
        "post_step_joint_speed_rad_s", "tracker_depth_post_integration_m",
        "tracker_started", "tracker_finalized", "tracker_productive",
        "tracker_reason", "event_cumulative_impulse_n_s",
        "quality_found_count", "quality_normal_force_n",
        "quality_contact_position_m", "quality_contact_normal",
        "tracker_contact_point_w", "tracker_contact_error_m",
        "tracker_contact_quality", "tracker_contact_quality_valid",
        "tracker_contact_quality_overflow", "tracker_first_contact_time_s",
        "tracker_contact_normal_axiality",
        "event_cumulative_transverse_impulse_n_s",
      )
    }
    self._actions: list[torch.Tensor] = []
    self._impact_payout: list[torch.Tensor] = []
    self._delivered_payout: list[torch.Tensor] = []

    robot = env.scene["robot"]
    nail = env.scene["nail_block"]
    contact = env.scene["hammer_nail_contact"]
    net_force = env.scene["hammer_nail_impulse"]
    head_cfg = SceneEntityCfg(
      "robot", site_names=(HAMMER_HEAD_SITE_NAME,)
    )
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
    head_cfg.resolve(env.scene)
    arm_cfg.resolve(env.scene)
    self._robot = robot
    self._nail = nail
    self._contact = contact
    self._net_force = net_force
    self._head_ids = head_cfg.site_ids
    self._arm_ids = arm_cfg.joint_ids
    self._tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR, None)
    if self._tracker is None:
      raise RuntimeError("sampled evaluator requires FirstStrikeEventTracker")
    try:
      self._quality_sensor = env.scene["hammer_nail_quality"]
    except KeyError:
      self._quality_sensor = None
    names = env.reward_manager.active_terms
    self._impact_idx = names.index("impact_progress")
    self._delivered_idx = names.index("delivered_impulse")
    self._axis = torch.tensor(
      self.nail_geometry["nail_axis"], dtype=torch.float32, device=env.device
    )
    self._phase = _PreIntegrationTracePhase()
    self._install()

  def _install(self) -> None:
    original_sim_step = self.env.sim.step

    def cache_preintegration_then_step() -> None:
      self._phase.cache_pre_step(
        joint_speed_rad_s=self._robot.data.joint_vel[:, self._arm_ids],
        clamped_depth_m=self._nail.data.joint_pos[:, 0].clamp(0.0, 0.032),
      )
      original_sim_step()

    self.env.sim.step = cache_preintegration_then_step
    original_substep = self.env.metrics_manager.compute_substep

    def capture_substep() -> None:
      original_substep()
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
      for key, value in sample.items():
        self._substeps[key].append(value)
      self._substeps["tracker_started"].append(
        self._tracker.started.detach().clone()
      )
      self._substeps["tracker_finalized"].append(
        self._tracker.finalized.detach().clone()
      )
      self._substeps["tracker_productive"].append(
        self._tracker.productive.detach().clone()
      )
      self._substeps["tracker_reason"].append(
        self._tracker.reason.detach().clone()
      )
      self._substeps["event_cumulative_impulse_n_s"].append(
        self._tracker.delivered.detach().clone()
      )
      self._substeps["tracker_contact_point_w"].append(
        self._tracker.contact_point_w.detach().clone()
      )
      self._substeps["tracker_contact_error_m"].append(
        self._tracker.contact_error_m.detach().clone()
      )
      self._substeps["tracker_contact_quality"].append(
        self._tracker.contact_quality.detach().clone()
      )
      self._substeps["tracker_contact_quality_valid"].append(
        self._tracker.contact_quality_valid.detach().clone()
      )
      self._substeps["tracker_contact_quality_overflow"].append(
        self._tracker.contact_quality_overflow.detach().clone()
      )
      self._substeps["tracker_first_contact_time_s"].append(
        self._tracker.first_contact_time_s.detach().clone()
      )
      self._substeps["tracker_contact_normal_axiality"].append(
        self._tracker.contact_normal_axiality.detach().clone()
      )
      self._substeps["event_cumulative_transverse_impulse_n_s"].append(
        self._tracker.delivered_transverse.detach().clone()
      )
      if self._quality_sensor is None:
        batch = self.env.num_envs
        device = self.env.device
        self._substeps["quality_found_count"].append(
          torch.zeros(batch, dtype=torch.long, device=device)
        )
        self._substeps["quality_normal_force_n"].append(
          torch.zeros(batch, 1, device=device)
        )
        self._substeps["quality_contact_position_m"].append(
          torch.zeros(batch, 1, 3, device=device)
        )
        self._substeps["quality_contact_normal"].append(
          torch.zeros(batch, 1, 3, device=device)
        )
      else:
        quality = self._quality_sensor.data
        self._substeps["quality_found_count"].append(
          (quality.found > 0).sum(dim=-1).detach().clone()
        )
        self._substeps["quality_normal_force_n"].append(
          quality.force[..., 0].detach().clone()
        )
        self._substeps["quality_contact_position_m"].append(
          quality.pos.detach().clone()
        )
        self._substeps["quality_contact_normal"].append(
          quality.normal.detach().clone()
        )

    self.env.metrics_manager.compute_substep = capture_substep  # type: ignore[method-assign]

    original_compute = self.env.metrics_manager.compute

    def capture_completed() -> None:
      original_compute()
      step_reward = self.env.reward_manager._step_reward
      self._actions.append(self.env.action_manager.action.detach().clone())
      self._impact_payout.append(
        (step_reward[:, self._impact_idx] * self.env.step_dt).detach().clone()
      )
      self._delivered_payout.append(
        (step_reward[:, self._delivered_idx] * self.env.step_dt).detach().clone()
      )
      done_ids = torch.nonzero(self.env.reset_buf, as_tuple=False).flatten()
      for env_id in done_ids.tolist():
        self._capture_completed(env_id)

    self.env.metrics_manager.compute = capture_completed  # type: ignore[method-assign]

  def _slice_substep(self, key: str, env_id: int) -> torch.Tensor:
    start = self._substep_start[env_id]
    values = self._substeps[key][start:]
    if not values:
      raise RuntimeError("completed episode has no captured substeps")
    return torch.stack([value[env_id] for value in values])

  def _slice_control(self, values: list[torch.Tensor], env_id: int) -> torch.Tensor:
    start = self._control_start[env_id]
    selected = values[start:]
    if not selected:
      raise RuntimeError("completed episode has no captured control steps")
    return torch.stack([value[env_id] for value in selected])

  def _capture_completed(self, env_id: int) -> None:
    ordinal = self._seen_counts[env_id]
    self._seen_counts[env_id] += 1
    started_stream = self._slice_substep("tracker_started", env_id).bool()
    accepted = torch.nonzero(started_stream, as_tuple=False).flatten()
    accepted_onset = int(accepted[0]) if accepted.numel() else None
    reason_code = int(self._tracker.reason[env_id])
    reason = {
      REASON_SUCCESS: "success",
      REASON_WINDOW: "window",
    }.get(reason_code, "none")
    physical = {
      key: _json_values(self._slice_substep(key, env_id))
      for key in _TRACE_PHYSICAL_KEYS
    }
    event_trace = {
      key: _json_values(self._slice_substep(key, env_id))
      for key in _TRACE_EVENT_KEYS
    }
    actions = self._slice_control(self._actions, env_id)
    impact = self._slice_control(self._impact_payout, env_id)
    delivered_payout = self._slice_control(self._delivered_payout, env_id)
    event_delivered = float(self._tracker.delivered[env_id])
    trace = {
      "episode_id": f"{self.treatment}-env{env_id}-episode{ordinal}",
      "env_id": env_id,
      "episode_ordinal": ordinal,
      "arm": self.treatment,
      "task": self.task,
      "physics_dt_s": float(self.env.physics_dt),
      "phase_contract": {
        "physical_channels": "pre_integration",
        "tracker_derived_channels": "pre_integration",
        "tracker_nail_depth": "post_integration",
        "tracker_depth_lead_substeps": 1,
      },
      "nail_geometry": copy.deepcopy(self.nail_geometry),
      "physical": physical,
      "event_trace": event_trace,
      "action_tape": _json_values(actions),
      "action_tape_digest": _canonical_digest(_json_values(actions)),
      "payout_semantics": PAYOUT_SEMANTICS[self.treatment],
      "payout_is_counterfactual": False,
      "first_strike": {
        "started": bool(self._tracker.started[env_id]),
        "finalized": bool(self._tracker.finalized[env_id]),
        "reason": reason,
        "accepted_onset_index": accepted_onset,
        "productive": bool(self._tracker.productive[env_id]),
        "v_precontact_m_s": float(self._tracker.v_precontact[env_id]),
        "delivered_n_s": event_delivered,
        "delivered_transverse_n_s": float(
          self._tracker.delivered_transverse[env_id]
        ),
        "contact_point_w": _json_values(self._tracker.contact_point_w[env_id]),
        "contact_error_m": float(self._tracker.contact_error_m[env_id]),
        "contact_quality": float(self._tracker.contact_quality[env_id]),
        "contact_quality_valid": bool(
          self._tracker.contact_quality_valid[env_id]
        ),
        "contact_quality_overflow": bool(
          self._tracker.contact_quality_overflow[env_id]
        ),
        "first_contact_time_s": float(
          self._tracker.first_contact_time_s[env_id]
        ),
        "contact_normal_axiality": float(
          self._tracker.contact_normal_axiality[env_id]
        ),
        "saturated": (
          self.treatment == "E"
          and event_delivered >= self.event_i_ref_n_s
        ),
      },
      "overall_success": bool(self.env.reset_terminated[env_id]),
      "reward": {
        "gamma": self.gamma,
        "impact_payout": _json_values(impact),
        "delivered_payout": _json_values(delivered_payout),
        "source": "actual_reward_manager_weighted_once",
      },
      "episode_peak_lambda": _json_values(
        self.snapshot["perjoint"][env_id]
      ),
      "episode_delivered_accumulator_n_s": float(
        self.snapshot["delivered"][env_id]
      ),
      "episode_depth_m": float(self.snapshot["depth"][env_id]),
    }
    trace["trace_digest"] = _physical_trace_digest(trace)
    _retain_first_completed(
      trace,
      counts=self.accepted_counts,
      selected=self.completed,
      episodes_per_env=EXPECTED_EPISODES_PER_ENV,
    )
    self._substep_start[env_id] = len(self._substeps["contact"])
    self._control_start[env_id] = len(self._impact_payout)

  @property
  def quota_complete(self) -> bool:
    return all(count == EXPECTED_EPISODES_PER_ENV for count in self.accepted_counts)


def _invariant_violations(rec: dict) -> tuple[int, int]:
  """Theorem-level physical-consistency counters over one rollout's episode records.

  Tier-1 safety net (2026-07-14, zero-Lambda incident): the first GPU smoke produced
  success=1.0 / depth=32 mm with Lambda == 0.0 and delivered == 0.0 -- physically
  impossible with live instrumentation (a nail cannot be driven without contact
  impulse), yet nothing complained. These counters make every summary row
  self-certifying; they use only exact-zero comparisons (the accumulators are exact
  zeros absent contact), so there are no tunable thresholds to drift.

  impossible_success: episode terminated on SUCCESS but delivered == 0 or Lambda == 0
    -- the nail was driven with no recorded impulse: dead instrument, never physics.
  lambda_dead: object-side path saw delivered axial impulse > 0 while the robot-side
    qfrc path recorded Lambda == 0 -- the two INDEPENDENT measurement paths disagree
    in the one direction that is impossible (axial force delivered to the nail implies
    a reaction impulse in the arm). The reverse (Lambda > 0, delivered == 0) is a
    legitimate lateral graze and is NOT counted.
  """
  lam_worst = [float(lam.max()) for lam in rec["lam"]]
  impossible = sum(
    1 for s, d, lw in zip(rec["succ"], rec["delivered"], lam_worst)
    if s and (d <= 0.0 or lw <= 0.0)
  )
  lam_dead = sum(
    1 for d, lw in zip(rec["delivered"], lam_worst) if d > 0.0 and lw <= 0.0
  )
  return impossible, lam_dead


def _install_episode_hook(env: ManagerBasedRlEnv) -> dict:
  """Snapshot the shipped accumulators' PRE-auto-reset buffers once per control step.

  See module docstring "Auto-reset correctness". Hooking ``metrics_manager.compute`` (not
  ``compute_substep``) is deliberate: reading the MANAGER's own per_substep-averaged ``_step_values``
  would re-introduce the substep-dilution ``joint_impulse_peak`` was written to avoid (impulse_bound.py
  docstring); this hook instead reads the LIVE accumulator objects directly (their monotone-max /
  cumulative-sum buffers), exactly like diag_impulse_trace.py's substep hook does one level down.
  """
  acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
  dacc = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR, None)
  if acc is None or dacc is None:
    raise RuntimeError(
      "eval_impulse requires BOTH the SubstepImpulseAccumulator and SubstepDeliveredImpulse metrics "
      "(cat_impulse=True) -- use --task Unitree-Z1-Hammer-CaT-Impulse (the default) or another task "
      "built with cat_impulse=True."
    )
  nail_cfg = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
  nail_cfg.resolve(env.scene)

  snap: dict = {"perjoint": None, "delivered": None, "depth": None}
  orig_compute = env.metrics_manager.compute

  def patched() -> None:
    orig_compute()
    snap["perjoint"] = acc._episode_peak_perjoint.detach().clone()  # (B,6) pre-reset
    snap["delivered"] = dacc.delivered.detach().clone()             # (B,) pre-reset
    snap["depth"] = clamped_nail_depth(env, nail_cfg).detach().clone()  # (B,) pre-reset

  env.metrics_manager.compute = patched  # type: ignore[method-assign]
  return snap


def _rollout(
  env_cfg,
  agent_cfg,
  runner_cls,
  ckpt: str,
  device: str,
  nsteps: int,
  seed: int,
  stochastic: bool,
) -> dict:
  """Roll out ONE checkpoint for ``nsteps`` control steps; return per-episode records."""
  # Fresh cfg per rollout: env construction mutates the cfg in place (the registry's load_env_cfg
  # deep-copies for exactly this reason), and this function runs TWICE per checkpoint (mean +
  # sampled) -- without the copy the second env would be built from a construction-mutated cfg.
  env_cfg = copy.deepcopy(env_cfg)
  torch.manual_seed(seed)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  snap = _install_episode_hook(env)  # hook the raw env BEFORE wrapping
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs, _ = wrapped.reset()
  ep_len = torch.zeros(env.num_envs, device=env.device)
  lam_records: list[torch.Tensor] = []
  delivered_records: list[float] = []
  depth_records: list[float] = []
  ep_len_records: list[float] = []
  succ_records: list[bool] = []

  for _ in range(nsteps):
    with torch.no_grad():
      actions = policy(obs, stochastic_output=stochastic)
    obs, rew, dones, extras = wrapped.step(actions)
    del rew, extras
    ep_len += 1.0
    # Exact success signal, set inside env.step() BEFORE the in-step auto-reset and not touched
    # again until the NEXT env.step() call -- same idiom as diag_policy_trace.py / diag_impulse_trace.py.
    succ_mask = env.reset_terminated.clone()
    done_idx = torch.nonzero(dones, as_tuple=False).flatten()
    if done_idx.numel() > 0:
      for i in done_idx.tolist():
        lam_records.append(snap["perjoint"][i].cpu())
        delivered_records.append(float(snap["delivered"][i]))
        depth_records.append(float(snap["depth"][i]))
        ep_len_records.append(float(ep_len[i]))
        succ_records.append(bool(succ_mask[i]))
      ep_len[done_idx] = 0.0

  env.close()
  return {
    "lam": lam_records,
    "delivered": delivered_records,
    "depth": depth_records,
    "ep_len": ep_len_records,
    "succ": succ_records,
  }


def _rollout_balanced_sampled(
  env_cfg,
  agent_cfg,
  runner_cls,
  ckpt: str,
  device: str,
  *,
  task: str,
  contract: dict,
  reset_seed: int,
  observation_seed: int,
  action_seed: int,
  nail_geometry: dict,
) -> dict:
  """The sampled branch of the existing snapshot path, stopped by episode quota."""
  env_cfg = copy.deepcopy(env_cfg)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  snapshot = _install_episode_hook(env)
  _install_evaluator_rng_streams(
    env,
    reset_seed=reset_seed,
    observation_seed=observation_seed,
  )
  gamma = float(getattr(agent_cfg.algorithm, "gamma", 0.99))
  collector = _SampledTraceCollector(
    env,
    snapshot=snapshot,
    treatment=contract["treatment"],
    task=task,
    gamma=gamma,
    event_i_ref_n_s=contract["event_i_ref_n_s"],
    nail_geometry=nail_geometry,
  )
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  action_stream = _TorchRngStream(action_seed, device)
  obs, _ = wrapped.reset()

  # Two timeout-length episodes are a deterministic fail-safe, not the sample
  # size.  The accepted population is always first-two completions per env.
  max_control_steps = (
    EXPECTED_EPISODES_PER_ENV * env.max_episode_length
    + EXPECTED_EPISODES_PER_ENV
  )
  control_steps = 0
  while not collector.quota_complete and control_steps < max_control_steps:
    with torch.no_grad():
      actions = action_stream.run(
        lambda: policy(obs, stochastic_output=True)
      )
    obs, _, _, _ = wrapped.step(actions)
    control_steps += 1
  if not collector.quota_complete:
    missing = {
      env_id: count
      for env_id, count in enumerate(collector.accepted_counts)
      if count != EXPECTED_EPISODES_PER_ENV
    }
    env.close()
    raise RuntimeError(
      "sampled evaluator failed the exact first-two-completions quota "
      f"within {max_control_steps} control steps: {missing}"
    )

  selected = collector.completed
  expected = env.num_envs * EXPECTED_EPISODES_PER_ENV
  if expected != EXPECTED_EPISODES_PER_SEED:
    env.close()
    raise RuntimeError(
      f"campaign protocol requires 256x2=512 episodes, got {env.num_envs}x"
      f"{EXPECTED_EPISODES_PER_ENV}={expected}"
    )
  if len(selected) != expected:
    env.close()
    raise RuntimeError(
      f"balanced collector retained {len(selected)} episodes, expected {expected}"
    )
  episode_metrics = [
    summarize_episode(trace, nail_geometry=nail_geometry) for trace in selected
  ]
  aggregate = aggregate_episode_metrics(
    episode_metrics, expected_episode_count=EXPECTED_EPISODES_PER_SEED
  )
  result = {
    "lam": [
      torch.tensor(trace["episode_peak_lambda"], dtype=torch.float32)
      for trace in selected
    ],
    "delivered": [
      float(trace["episode_delivered_accumulator_n_s"]) for trace in selected
    ],
    "depth": [float(trace["episode_depth_m"]) for trace in selected],
    "ep_len": [float(len(trace["reward"]["impact_payout"])) for trace in selected],
    "succ": [bool(trace["overall_success"]) for trace in selected],
    "episodes": selected,
    "episode_metrics": episode_metrics,
    "sampled_aggregate": aggregate,
    "control_steps": control_steps,
  }
  env.close()
  return result


def _persist_sampled_traces(
  *,
  out_dir: Path,
  name: str,
  sampled_rec: dict,
  task: str,
  contract: dict,
  training_seed: int,
  reset_seed: int,
  observation_seed: int,
  action_seed: int,
  nail_geometry: dict,
  provenance: dict,
  mean_rollout_invariants: dict,
  evaluation_contract: dict | None = None,
) -> dict:
  payload = {
    "schema_version": 3,
    "selection": "first two completed episodes from each of 256 environments",
    "expected_episode_count": EXPECTED_EPISODES_PER_SEED,
    "treatment": contract["treatment"],
    "task": task,
    "training_seed": training_seed,
    "weights": {
      "impact_progress": contract["impact_weight"],
      "delivered_impulse": contract["delivered_weight"],
    },
    "event_i_ref_n_s": contract["event_i_ref_n_s"],
    "impulse_limits_n_m_s": contract["impulse_limits_n_m_s"],
    "imp_max_p": 0.0,
    "rng_streams": {
      "reset": reset_seed,
      "observation": observation_seed,
      "action": action_seed,
    },
    "evaluation_contract": evaluation_contract or {},
    "nail_geometry": nail_geometry,
    "provenance": provenance,
    "mean_rollout_invariants": mean_rollout_invariants,
    "control_steps_until_quota": sampled_rec["control_steps"],
    "episodes": sampled_rec["episodes"],
  }
  digest = _canonical_digest(payload)
  payload["payload_digest"] = digest
  safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
  checkpoint_prefix = str(provenance["checkpoint_sha256"])[:16]
  path = out_dir / (
    f"{safe_name}_{checkpoint_prefix}_{digest}_sampled_traces.npz"
  )
  encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
  if path.exists():
    with np.load(path, allow_pickle=False) as saved:
      if saved.files != ["payload_json"] or str(saved["payload_json"]) != encoded:
        raise RuntimeError(
          f"content-addressed sampled artifact collision at {path}"
        )
  else:
    np.savez_compressed(
      path, payload_json=np.asarray(encoded, dtype=np.str_)
    )
  return {
    "path": str(path),
    "payload_digest": digest,
    "artifact_sha256": _sha256_file(path),
  }


def _mean_std(xs: list[float]) -> tuple[float, float]:
  if not xs:
    return 0.0, 0.0
  t = torch.tensor(xs, dtype=torch.float32)
  return float(t.mean()), float(t.std() if len(xs) > 1 else 0.0)


def _worst_ratios(lam_records: list[torch.Tensor], j_limit: torch.Tensor) -> tuple[float, float]:
  """Per-episode worst-joint Λ_j/J_limit_j; returns (max over episodes, mean over episodes)."""
  if not lam_records:
    return 0.0, 0.0
  lam = torch.stack(lam_records)  # (N, 6)
  ratio = lam / j_limit.clamp_min(1e-9)
  worst_per_ep = ratio.amax(dim=1)  # (N,)
  return float(worst_per_ep.max()), float(worst_per_ep.mean())


def _git_provenance(repo_root: Path) -> dict:
  """Return full revision plus fail-closed tracked and untracked status."""
  try:
    revision = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=repo_root, capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()
    status = subprocess.run(
      ["git", "status", "--porcelain=v1", "--untracked-files=all"],
      cwd=repo_root, capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()
    return {
      "revision": revision,
      "dirty": bool(status),
      "status": status,
    }
  except Exception:
    return {
      "revision": "unknown",
      "dirty": True,
      "status": "git provenance unavailable",
    }


def _verify_unchanged_git(
  repo_root: Path, *, expected: dict, label: str
) -> None:
  actual = _git_provenance(repo_root)
  if actual != expected:
    raise RuntimeError(
      f"{label} provenance changed during evaluation: "
      f"before={expected!r}, after={actual!r}"
    )


def _training_seed(name: str, explicit: int | None) -> int:
  if explicit is not None:
    if explicit < 0:
      raise ValueError("--training-seed must be non-negative")
    return explicit
  match = re.search(r"(?:seed|_s)([0-9]+)(?:$|[^0-9])", name)
  if match is None:
    raise ValueError(
      "training seed is provenance, not evaluator RNG: pass --training-seed "
      "or include seedN in --name"
    )
  return int(match.group(1))


def _enforce_postwrite_invariants(
  *,
  name: str,
  row: dict,
  impossible_success_n: int,
  lambda_dead_n: int,
  provenance_invalid: bool,
  repo_hash: str,
  asset_hash: str,
) -> str:
  """Apply post-write hard gates while retaining finite qvel exceedances ITT."""

  finite_exceedance_rate = float(
    row["qvel_finite_exceedance_rate_sampled"]
  )
  nonfinite_rate = float(row["qvel_nonfinite_rate_sampled"])
  if finite_exceedance_rate > 0.0:
    print(
      f"[eval_impulse] HARDWARE-SPEED QUALIFICATION for '{name}': "
      f"finite qvel rail exceedance rate={finite_exceedance_rate} at "
      f"{HARDWARE_QVEL_LIMIT_RAD_S} rad/s. Episodes/seeds/rows are not excluded "
      "from the simulation reward comparison; no hardware-safe claim is allowed.",
      file=sys.stderr,
    )
    qualification = "simulation_only_hardware_speed_unqualified"
  else:
    qualification = "no_observed_qvel_rail_exceedance_not_hardware_certification"
  if (
    impossible_success_n > 0
    or lambda_dead_n > 0
    or provenance_invalid
    or nonfinite_rate > 0.0
  ):
    print(
      f"[eval_impulse] INVARIANT VIOLATION for '{name}': "
      f"impossible_success_n={impossible_success_n} (success episodes with zero "
      "recorded impulse -- a nail cannot be driven without impulse), "
      f"lambda_dead_n={lambda_dead_n} (object-side impulse delivered while the "
      "robot-side qfrc path read exactly zero), "
      f"qvel_nonfinite_rate_sampled={nonfinite_rate} "
      "(nonfinite pre/post 500 Hz qvel is invalid), "
      f"qvel_violation_rate_sampled={finite_exceedance_rate} "
      f"(finite rail={HARDWARE_QVEL_LIMIT_RAD_S} rad/s is retained ITT), "
      f"git_hash={repo_hash}, asset_git_hash={asset_hash}. "
      "The row is invalid for comparison.",
      file=sys.stderr,
    )
    raise SystemExit(2)
  return qualification


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Unitree-Z1-Hammer-CaT-Impulse",
                  choices=tuple(TASK_TO_ARM | QUALITY_TASK_TO_ARM),
                  help="exact registered C/D-prime/F/E treatment task used to train this checkpoint")
  ap.add_argument("--ckpt", required=True, help="checkpoint .pt path")
  ap.add_argument("--name", default=None, help="row label; defaults to the checkpoint's parent dir name")
  ap.add_argument("--training-seed", type=int, default=None,
                  help="independent policy-training seed (inferred from seedN in --name if omitted)")
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--nsteps", type=int, default=400,
                  help="control steps for the diagnostic mean-action rollout only")
  ap.add_argument("--sampled-nsteps", type=int, default=None,
                  help="retired: sampled primary stops only at the exact 2-per-env episode quota")
  ap.add_argument("--episode-len-s", type=float, default=4.0,
                  help="finite eval episode horizon (s), overriding the play cfg's near-infinite "
                       "default -- otherwise an undertrained policy that never strikes never yields a "
                       "completed episode. 4.0s = 200 control steps @ 50 Hz; converged single strikes "
                       "finish in ~8-15 steps, so this only bounds failure-mode episodes.")
  ap.add_argument("--imp-max-p", type=float, default=0.0,
                  help="forced eval-time value (default 0 -- log-only/pure instrumentation, "
                       "equivalent to --env.metrics.cat-soft.params.imp-max-p 0 on scripts/train.py)")
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--seed", type=int, default=2026072900,
                  help="base used only to derive frozen evaluator RNG streams")
  ap.add_argument("--reset-seed", type=int, default=None)
  ap.add_argument("--observation-seed", type=int, default=None)
  ap.add_argument("--action-seed", type=int, default=None)
  ap.add_argument("--accepted-manifest-sha256", default="")
  ap.add_argument("--expected-checkpoint-sha256", default="")
  ap.add_argument("--training-code-revision", default="")
  ap.add_argument("--training-asset-revision", default="")
  ap.add_argument(
    "--nail-asset",
    default=str(
      Path(__file__).resolve().parents[2]
      / "safe_impact_manipulation"
      / "hammer_z1_env"
      / "assets"
      / "nail_block_scene.xml"
    ),
  )
  ap.add_argument("--out", default="/tmp/eval_impulse", help="directory for summary.csv (appended)")
  ap.add_argument("--csv-name", default="summary.csv")
  args = ap.parse_args()

  name = args.name or Path(args.ckpt).resolve().parent.name
  training_seed = _training_seed(name, args.training_seed)
  if args.sampled_nsteps is not None:
    raise ValueError(
      "--sampled-nsteps is forbidden: the sampled primary stops at exactly "
      "the first two completed episodes per environment"
    )
  if args.num_envs != 256:
    raise ValueError("--num-envs must be exactly 256 for the 512-episode campaign")
  if args.imp_max_p != 0.0:
    raise ValueError("--imp-max-p is frozen at 0 for the fixed-impedance campaign")
  reset_seed = (
    args.reset_seed
    if args.reset_seed is not None
    else args.seed + RESET_SEED_OFFSET
  )
  observation_seed = (
    args.observation_seed
    if args.observation_seed is not None
    else args.seed + OBSERVATION_SEED_OFFSET
  )
  action_seed = (
    args.action_seed
    if args.action_seed is not None
    else args.seed + ACTION_SEED_OFFSET
  )
  if len({reset_seed, observation_seed, action_seed}) != 3:
    raise ValueError("reset, observation, and action RNG seeds must be distinct")
  manifest_identity = (
    args.accepted_manifest_sha256,
    args.training_code_revision,
    args.training_asset_revision,
  )
  if any(manifest_identity) and not all(manifest_identity):
    raise ValueError(
      "accepted manifest hash and both training revisions must be supplied together"
    )
  if all(manifest_identity):
    for label, value, length in (
      ("accepted manifest SHA-256", args.accepted_manifest_sha256, 64),
      ("training code revision", args.training_code_revision, 40),
      ("training asset revision", args.training_asset_revision, 40),
    ):
      if len(value) != length or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{label} must be {length} hexadecimal characters")
  if args.expected_checkpoint_sha256 and (
    len(args.expected_checkpoint_sha256) != 64
    or any(
      c not in "0123456789abcdefABCDEF"
      for c in args.expected_checkpoint_sha256
    )
  ):
    raise ValueError(
      "expected checkpoint SHA-256 must be 64 hexadecimal characters"
    )

  # Training cfg is deliberate: play cfg zeros reset noise and disables actor
  # observation corruption, which would make the sampled campaign off-policy.
  config_identities = {}
  if args.task in QUALITY_TASK_TO_ARM:
    _, env_cfg, config_identities = build_strict_quality_evaluation_cfg(
      args.task, play=False
    )
  else:
    env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.episode_length_s = args.episode_len_s
  if "cat_soft" not in env_cfg.metrics:
    raise RuntimeError(f"--task {args.task} has no 'cat_soft' metric -- not a cat_impulse=True task.")
  env_cfg.metrics["cat_soft"].params["imp_max_p"] = args.imp_max_p
  if args.task not in QUALITY_TASK_TO_ARM:
    _ensure_first_strike_instrumentation(env_cfg)
  contract = _validate_sampled_env_contract(env_cfg, args.task)
  agent_cfg = load_rl_cfg(args.task)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  nail_geometry = load_frozen_nail_geometry(args.nail_asset)
  asset_repo = Path(args.nail_asset).resolve().parents[2]
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  code_repo = Path(__file__).resolve().parents[1]
  code_git = _git_provenance(code_repo)
  asset_git = _git_provenance(asset_repo)
  if all(manifest_identity):
    if code_git["revision"] != args.training_code_revision:
      raise RuntimeError("training/evaluation code revision mismatch")
    if asset_git["revision"] != args.training_asset_revision:
      raise RuntimeError("training/evaluation asset revision mismatch")

  j_limit = torch.tensor(IMP_J_LIMIT, dtype=torch.float32)
  assert j_limit.numel() == len(ARM_JOINTS), f"IMP_J_LIMIT must have {len(ARM_JOINTS)} entries"

  print(f"[eval_impulse] name={name} ckpt={args.ckpt} task={args.task} "
        f"num_envs={args.num_envs} nsteps={args.nsteps} episode_len_s={args.episode_len_s} "
        f"training_seed={training_seed} reset_seed={reset_seed} "
        f"observation_seed={observation_seed} action_seed={action_seed} "
        f"device={args.device} imp_max_p={args.imp_max_p}")

  if args.expected_checkpoint_sha256:
    _verify_expected_checkpoint_sha256(
      args.ckpt, expected_sha256=args.expected_checkpoint_sha256
    )
  with _frozen_checkpoint(
    args.ckpt, snapshot_parent=out_dir
  ) as (checkpoint_path, checkpoint_load_path, checkpoint_sha256):
    if (
      args.expected_checkpoint_sha256
      and checkpoint_sha256 != args.expected_checkpoint_sha256
    ):
      raise RuntimeError(
        "accepted checkpoint SHA-256 mismatch while creating snapshot"
      )
    # --- Primary: mean-action rollout ---
    mean_rec = _rollout(
      env_cfg, agent_cfg, runner_cls, str(checkpoint_load_path), args.device,
      args.nsteps, args.seed, stochastic=False,
    )
    n_ep = len(mean_rec["ep_len"])
    if n_ep == 0:
      print(f"[eval_impulse] ERROR: ZERO completed episodes in the mean-action rollout for '{name}' "
            f"(num_envs={args.num_envs}, nsteps={args.nsteps}, episode_len_s={args.episode_len_s}) -- "
            f"NO row written. Keep nsteps >= 2 x episode_len_s x 50 control steps so even a "
            f"never-succeeding checkpoint times out into >=2 episodes/env (see the "
            f"scripts/eval_impulse.sh header).", file=sys.stderr)
      raise SystemExit(1)
    # --- Primary sampled rollout: stochastic policy, exact balanced quota ---
    sampled_rec = _rollout_balanced_sampled(
      env_cfg,
      agent_cfg,
      runner_cls,
      str(checkpoint_load_path),
      args.device,
      task=args.task,
      contract=contract,
      reset_seed=reset_seed,
      observation_seed=observation_seed,
      action_seed=action_seed,
      nail_geometry=nail_geometry,
    )

  if n_ep < 2 * args.num_envs:
    print(f"[eval_impulse] WARNING: n_episodes={n_ep} < {2 * args.num_envs} -- below the pinned "
          f">=2 episodes/env floor for num_envs={args.num_envs}. The row is still written "
          f"(n_episodes is a column), but its tail statistics are under-populated vs the protocol.",
          file=sys.stderr)
  lam_max = [0.0] * len(ARM_JOINTS)
  lam_p95 = [0.0] * len(ARM_JOINTS)
  if n_ep > 0:
    lam_stack = torch.stack(mean_rec["lam"])  # (N, 6)
    lam_max = lam_stack.amax(dim=0).tolist()
    lam_p95 = torch.quantile(lam_stack, 0.95, dim=0).tolist()
  worst_max, worst_mean = _worst_ratios(mean_rec["lam"], j_limit)
  delivered_mean, delivered_std = _mean_std(mean_rec["delivered"])
  depth_mean, depth_std = _mean_std([d * 1000.0 for d in mean_rec["depth"]])
  ep_len_mean, ep_len_std = _mean_std(mean_rec["ep_len"])
  success_rate = (sum(mean_rec["succ"]) / n_ep) if n_ep > 0 else 0.0
  print(f"[eval_impulse] mean-action: n_episodes={n_ep} success_rate={success_rate:.3f} "
        f"worst_ratio_max={worst_max:.4f} delivered_mean={delivered_mean:.4f} "
        f"lambda_max={['%.4f' % v for v in lam_max]}")

  n_ep_s = len(sampled_rec["ep_len"])
  worst_max_s, _ = _worst_ratios(sampled_rec["lam"], j_limit)
  delivered_mean_s, _ = _mean_std(sampled_rec["delivered"])
  success_rate_s = (sum(sampled_rec["succ"]) / n_ep_s) if n_ep_s > 0 else 0.0
  print(f"[eval_impulse] sampled-action (primary): n_episodes={n_ep_s} "
        f"success_rate={success_rate_s:.3f} worst_ratio_max={worst_max_s:.4f} "
        f"delivered_mean={delivered_mean_s:.4f}")
  _verify_unchanged_git(
    code_repo, expected=code_git, label="code"
  )
  _verify_unchanged_git(
    asset_repo, expected=asset_git, label="asset"
  )
  _verify_unchanged_file(
    args.nail_asset,
    expected_sha256=nail_geometry["source_sha256"],
    label="nail asset",
  )
  campaign_config_sha256 = _campaign_config_digest(
    contract=contract,
    num_envs=args.num_envs,
    episode_len_s=args.episode_len_s,
    physics_dt_s=_physics_timestep_s(env_cfg),
    decimation=contract["control_decimation"],
    reset_seed=reset_seed,
    observation_seed=observation_seed,
    action_seed=action_seed,
  )
  treatment_config_sha256 = _treatment_config_digest(
    task=args.task, contract=contract
  )
  # --- Tier-1 physical-impossibility invariants (both rollouts) ---
  imp_m, dead_m = _invariant_violations(mean_rec)
  imp_s, dead_s = _invariant_violations(sampled_rec)
  impossible_success_n = imp_m + imp_s
  lambda_dead_n = dead_m + dead_s
  provenance = {
    "code_git": code_git,
    "asset_git": asset_git,
    "checkpoint_sha256": checkpoint_sha256,
    "accepted_checkpoint_sha256": (
      args.expected_checkpoint_sha256 or checkpoint_sha256
    ),
    "campaign_config_sha256": campaign_config_sha256,
    "treatment_config_sha256": treatment_config_sha256,
    "nail_asset_sha256": nail_geometry["source_sha256"],
    "accepted_manifest_sha256": args.accepted_manifest_sha256,
    "training_code_revision": args.training_code_revision,
    "training_asset_revision": args.training_asset_revision,
  }
  evaluation_contract = {
    "base_rng_seed": args.seed,
    "num_envs": args.num_envs,
    "episodes_per_env": EXPECTED_EPISODES_PER_ENV,
    "episode_len_s": args.episode_len_s,
    "mean_nsteps": args.nsteps,
    "completion_rule": "first_two_completions_per_environment",
    "stochastic_actions": True,
    "reset_position_noise_rad": [
      contract["reset_position_noise_min_rad"],
      contract["reset_position_noise_max_rad"],
    ],
    "actor_observation_corruption": contract["actor_observation_corruption"],
    "critic_observation_corruption": contract["critic_observation_corruption"],
    "physics_dt_s": contract["physics_dt_s"],
    "control_decimation": contract["control_decimation"],
    "fixed_impedance_signature_sha256": (
      contract["fixed_impedance_signature_sha256"]
    ),
    "fixed_action_signature_sha256": (
      contract["fixed_action_signature_sha256"]
    ),
    "strict_config_identities": config_identities,
  }
  trace_artifact = _persist_sampled_traces(
    out_dir=out_dir,
    name=name,
    sampled_rec=sampled_rec,
    task=args.task,
    contract=contract,
    training_seed=training_seed,
    reset_seed=reset_seed,
    observation_seed=observation_seed,
    action_seed=action_seed,
    nail_geometry=nail_geometry,
    provenance=provenance,
    mean_rollout_invariants={
      "impossible_success_n": imp_m,
      "lambda_dead_n": dead_m,
    },
    evaluation_contract=evaluation_contract,
  )
  repo_hash = (
    f"{code_git['revision']}-dirty"
    if code_git["dirty"]
    else str(code_git["revision"])
  )
  asset_hash = (
    f"{asset_git['revision']}-dirty"
    if asset_git["dirty"]
    else str(asset_git["revision"])
  )

  row = {
    "name": name,
    "ckpt_path": str(checkpoint_path),
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_sha256": checkpoint_sha256,
    "accepted_checkpoint_sha256": (
      args.expected_checkpoint_sha256 or checkpoint_sha256
    ),
    "task": args.task,
    "treatment": contract["treatment"],
    "training_seed": training_seed,
    "num_envs": args.num_envs,
    "nsteps": args.nsteps,
    "episode_len_s": args.episode_len_s,
    "seed": args.seed,
    "imp_max_p": args.imp_max_p,
    "impact_weight": contract["impact_weight"],
    "delivered_weight": contract["delivered_weight"],
    "event_i_ref_n_s": contract["event_i_ref_n_s"],
    "n_episodes": n_ep,
    **{f"lambda_max_{j}": lam_max[k] for k, j in enumerate(ARM_JOINTS)},
    **{f"lambda_p95_{j}": lam_p95[k] for k, j in enumerate(ARM_JOINTS)},
    "worst_ratio_max": worst_max,
    "worst_ratio_mean": worst_mean,
    "delivered_mean": delivered_mean,
    "delivered_std": delivered_std,
    "success_rate": success_rate,
    "nail_depth_mean_mm": depth_mean,
    "nail_depth_std_mm": depth_std,
    "ep_len_mean": ep_len_mean,
    "ep_len_std": ep_len_std,
    "impossible_success_n": impossible_success_n,
    "lambda_dead_n": lambda_dead_n,
    "n_episodes_sampled": n_ep_s,
    "success_rate_sampled": success_rate_s,
    "worst_ratio_max_sampled": worst_max_s,
    "delivered_mean_sampled": delivered_mean_s,
    "episodes_per_env_sampled": EXPECTED_EPISODES_PER_ENV,
    **sampled_rec["sampled_aggregate"],
    "action_rng_seed": action_seed,
    "reset_rng_seed": reset_seed,
    "observation_rng_seed": observation_seed,
    "reset_position_noise_min_rad": contract["reset_position_noise_min_rad"],
    "reset_position_noise_max_rad": contract["reset_position_noise_max_rad"],
    "actor_observation_corruption": contract["actor_observation_corruption"],
    "critic_observation_corruption": contract["critic_observation_corruption"],
    "sampled_completion_rule": "first_two_completions_per_environment",
    "sampled_actions_stochastic": True,
    "physics_dt_s": contract["physics_dt_s"],
    "control_decimation": contract["control_decimation"],
    "fixed_impedance_signature_sha256": (
      contract["fixed_impedance_signature_sha256"]
    ),
    "fixed_action_signature_sha256": (
      contract["fixed_action_signature_sha256"]
    ),
    "accepted_manifest_sha256": args.accepted_manifest_sha256,
    "training_code_revision": args.training_code_revision,
    "training_asset_revision": args.training_asset_revision,
    "sampled_trace_path": str(Path(trace_artifact["path"]).resolve()),
    "sampled_trace_digest": trace_artifact["payload_digest"],
    "sampled_trace_artifact_sha256": trace_artifact["artifact_sha256"],
    "campaign_config_sha256": campaign_config_sha256,
    "treatment_config_sha256": treatment_config_sha256,
    "nail_asset_sha256": nail_geometry["source_sha256"],
    "host": socket.gethostname(),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_hash": repo_hash,
    "git_revision": code_git["revision"],
    "git_dirty": code_git["dirty"],
    "asset_git_hash": asset_hash,
    "asset_git_revision": asset_git["revision"],
    "asset_git_dirty": asset_git["dirty"],
  }

  # Finiteness guard: a silent NaN/inf in the thesis CSV is exactly the failure mode to prevent --
  # refuse the whole row loudly instead of appending a poisoned value.
  nonfinite = {
    k: v for k, v in row.items()
    if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(v)
  }
  if nonfinite:
    print(f"[eval_impulse] ERROR: non-finite value(s) in the row for '{name}': {nonfinite} -- "
          f"NO row written.", file=sys.stderr)
    raise SystemExit(1)

  csv_path = out_dir / args.csv_name
  write_header = not csv_path.exists()
  if not write_header:
    # Schema drift must never misalign rows: an existing file appended to by a DIFFERENT
    # FIELDNAMES vintage would silently shift every column right of the drift point.
    with open(csv_path, newline="") as f:
      existing_header = next(csv.reader(f), None)
    if existing_header != FIELDNAMES:
      print(f"[eval_impulse] ERROR: {csv_path} has a different header schema "
            f"({len(existing_header) if existing_header else 0} cols) than this script's FIELDNAMES "
            f"({len(FIELDNAMES)} cols) -- appending would misalign rows. Re-run with FRESH=1 "
            f"(scripts/eval_impulse.sh) to wipe the output dir, or move the stale CSV aside.",
            file=sys.stderr)
      raise SystemExit(1)
  with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
      writer.writeheader()
    writer.writerow(row)
  print(f"[eval_impulse] appended row to {csv_path}")

  # Fail CLOSED on invariant violations, AFTER writing the row (the counts are columns,
  # so the violation is on the record; the nonzero exit makes eval_impulse.sh count a
  # FAIL instead of presenting the row as a clean result).
  provenance_invalid = (
    bool(code_git["dirty"])
    or bool(asset_git["dirty"])
    or code_git["revision"] == "unknown"
    or asset_git["revision"] == "unknown"
  )
  _enforce_postwrite_invariants(
    name=name,
    row=row,
    impossible_success_n=impossible_success_n,
    lambda_dead_n=lambda_dead_n,
    provenance_invalid=provenance_invalid,
    repo_hash=repo_hash,
    asset_hash=asset_hash,
  )


if __name__ == "__main__":
  main()
