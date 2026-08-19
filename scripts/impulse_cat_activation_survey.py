"""No-learning qualification of diagnostic impulse-CaT activation for frozen Z1 VIC."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


# Executing ``python scripts/...`` otherwise prefers the editable-installed main checkout over this
# isolated worktree.  Pin imports to the script's own repository before loading task registrations.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))


PROVISIONAL_CAPS_N_M_S = (0.82, 1.64, 0.82, 0.82, 0.82, 0.82)
# Optional plumbing-only threshold requested on 2026-08-14: apply one joint-agnostic 0.9 factor to
# every provisional project threshold.  Which joint crosses remains an observed outcome.
# It is neither a hardware limit nor a proposed scientific treatment. It is always analyzed only
# after the provisional threshold so plumbing evidence cannot be mistaken for provisional evidence.
DIAGNOSTIC_LIMITS_N_M_S = (0.738, 1.476, 0.738, 0.738, 0.738, 0.738)
VIC_TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT"
)
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
FIXED_SEED = 2026081202
TRAINING_LIKE_SEED = 2
FIXED_ENVS = 64
TRAINING_LIKE_ENVS = 4096
TRAINING_LIKE_STEPS = 24
PHYSICS_DT_S = 0.002
CONTROL_DECIMATION = 10
IMPULSE_WINDOW_SUBSTEPS = 25
CANDIDATE_IMP_MAX_P = (0.05, 0.1, 0.2, 0.3, 0.5)
CAT_TAU = 0.95
CAT_MIN_P = 0.0
IMPULSE_SEED = 1e-3
VELOCITY_LIMIT_RAD_S = 3.1415
VELOCITY_MAX_P = 0.5
VELOCITY_DETECTION = "substep"
EXPECTED_CHECKPOINT_SHA256 = "c1544b779e78e7323bf02ce7b0f165745ee63b5aad9f930f9eea64eb6ea4ea77"
EXPECTED_FIXED_POPULATION_SHA256 = "320efae8c21b3303c5dc3f18ae7df00e887653abf26aa8fb413803cf78d094cb"
EVALUATION_CHECKPOINTS: Mapping[str, str] = MappingProxyType(
  {
    "diag90_control": "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3",
    "diag90_target": "ddd7ac4c855160bff1db2af52e642d960dab2bef34e41dd2532002185eb36d15",
    "bridge_p0_control": "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3",
    "bridge_p02_target": "57000e958bbafa2c62929652d3b76fd6ed571c9867bee3735c14baf0ca57d8de",
    "dose_p0_control": "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3",
    "dose_p01_target": "92f1d97c8ff1476cb26c0b648478a0bc3c522e4e1eb7087389fb8e0d6bf73f86",
    "dose_p02_target": "57000e958bbafa2c62929652d3b76fd6ed571c9867bee3735c14baf0ca57d8de",
    "dose_p03_target": "4c0a665fffc077d488630a28b258c1593050f969b4f6227147c3594097dc4efd",
    "dose_p025_exploratory": "5efc45c11c02dd400ad7d417bcdeefe9d271038ab43007f08a2820ceca0e744d",
    "p02_uniform09": "a99593b263a74944d60ac412bb1da733a36a29a1cd9f4eeeaed89906372595df",
    "p02_joint_stress": "509cc26a2e521a935bcbc8c342040c95c7d2e3d7fc105a52d5d9450518bd2ec7",
  }
)
TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S: Mapping[str, tuple[float, ...]] = (
  MappingProxyType(
    {
      "p02_uniform09": (0.738, 1.476, 0.738, 0.738, 0.738, 0.738),
      "p02_joint_stress": (0.369, 0.246, 0.738, 0.369, 0.246, 0.0164),
    }
  )
)
RESET_RNG_OFFSET = 10_000_019
OBSERVATION_RNG_OFFSET = 20_000_033
ACTION_RNG_OFFSET = 30_000_041
POLICY_EVALUATION_STOCHASTIC_SEEDS = (2, 2026081701, 2026081702)
APPROVED_EVALUATOR_BASE_REVISION = "030942f34ac4a79131c1b70206d3c4acd58da79a"
EXPECTED_EVALUATION_ASSET_REVISION = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
_ASSET_REPO = _REPO_ROOT.parent / "safe_impact_manipulation"


@dataclass(frozen=True)
class EvaluationRngSeeds:
  """Matched evaluator-only Torch RNG identities derived from one population seed."""

  reset: int
  observation: int
  action: int

  @classmethod
  def from_evaluation_seed(cls, seed: int) -> EvaluationRngSeeds:
    return cls(
      reset=int(seed) + RESET_RNG_OFFSET,
      observation=int(seed) + OBSERVATION_RNG_OFFSET,
      action=int(seed) + ACTION_RNG_OFFSET,
    )


class _TorchRngStream:
  """A frozen Torch RNG stream isolated from the ambient device stream."""

  def __init__(self, seed: int, device: str):
    import torch

    self.seed = int(seed)
    self.device = torch.device(device)
    generator = torch.Generator(device=self.device)
    generator.manual_seed(self.seed)
    self._state = generator.get_state()

  def run(self, function):
    import torch

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


def _install_evaluator_rng_streams(
  env: Any,
  *,
  reset_seed: int,
  observation_seed: int,
) -> tuple[_TorchRngStream, _TorchRngStream]:
  """Isolate evaluator reset and observation randomness from policy sampling."""
  reset_stream = _TorchRngStream(reset_seed, env.device)
  observation_stream = _TorchRngStream(observation_seed, env.device)

  original_reset_idx = env._reset_idx

  def reset_idx(env_ids=None):
    return reset_stream.run(lambda: original_reset_idx(env_ids))

  env._reset_idx = reset_idx

  original_observation_compute = env.observation_manager.compute

  def observation_compute(*args, **kwargs):
    return observation_stream.run(
      lambda: original_observation_compute(*args, **kwargs)
    )

  env.observation_manager.compute = observation_compute
  return reset_stream, observation_stream


def validate_checkpoint_role(checkpoint: Path, role: str) -> str:
  if checkpoint.name != "model_499.pt" or role not in EVALUATION_CHECKPOINTS:
    raise ValueError("post-training evaluation requires a known role and model_499.pt")
  actual = _sha256(checkpoint.resolve(strict=True))
  if actual != EVALUATION_CHECKPOINTS[role]:
    raise RuntimeError("role/checkpoint SHA-256 mismatch")
  return actual


def validate_training_cap_identity(
  role: str, training_caps_n_m_s: object | None
) -> tuple[float, ...] | None:
  """Fail closed on the immutable training-cap identity for two-boundary roles."""
  expected = TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S.get(role)
  if expected is None:
    if training_caps_n_m_s is not None:
      raise ValueError("training-cap identity is only valid for two-boundary roles")
    return None
  if not isinstance(training_caps_n_m_s, (list, tuple)) or len(training_caps_n_m_s) != 6:
    raise ValueError("training-cap identity must contain six finite numeric values")
  if any(
    isinstance(cap, bool)
    or not isinstance(cap, (int, float))
    or not np.isfinite(cap)
    for cap in training_caps_n_m_s
  ):
    raise ValueError("training-cap identity must contain six finite numeric values")
  actual = tuple(float(cap) for cap in training_caps_n_m_s)
  if actual != expected:
    raise ValueError("training-cap identity mismatch for checkpoint role")
  return actual


def shadow_impulse_cat(
  margins: np.ndarray,
  *,
  tau: float,
  seed: float,
  max_p: float,
  valid: np.ndarray | None = None,
) -> np.ndarray:
  """Replay the shipped violation-masked impulse normalizer without touching the environment.

  ``margins`` has shape ``(control_steps, environments, joints)`` and is the raw
  ``Lambda - cap`` telemetry from the log-only hook.  With the shipped ``min_p=0`` this replay is
  linear in ``max_p``, so one frozen rollout supports an exact offline dose comparison.
  """
  margins = np.asarray(margins, dtype=np.float64)
  if margins.ndim != 3 or not np.isfinite(margins).all():
    raise ValueError("margins must be finite with shape (steps, envs, joints)")
  if not (0.0 <= tau < 1.0):
    raise ValueError("tau must lie in [0, 1)")
  if not np.isfinite(seed) or seed <= 0.0:
    raise ValueError("seed must be finite and positive")
  if not np.isfinite(max_p) or not (0.0 <= max_p <= 1.0):
    raise ValueError("max_p must lie in [0, 1]")

  steps, envs, joints = margins.shape
  if valid is None:
    valid_array = np.ones((steps, envs), dtype=bool)
  else:
    valid_array = np.asarray(valid, dtype=bool)
    if valid_array.shape != (steps, envs):
      raise ValueError("valid must have shape (steps, envs)")
  replay_margins = np.where(valid_array[:, :, None], margins, -1.0)
  cmax = np.full(joints, seed, dtype=np.float64)
  seeded = np.zeros(joints, dtype=bool)
  result = np.zeros_like(margins)
  for step in range(steps):
    batch_max = np.maximum(replay_margins[step], 0.0).max(axis=0)
    violated = batch_max > 0.0
    first = violated & ~seeded
    ema = tau * cmax + (1.0 - tau) * batch_max
    cmax = np.where(first, np.maximum(batch_max, seed), np.where(violated, ema, cmax))
    seeded |= violated
    normalized = np.clip(replay_margins[step] / np.maximum(cmax, seed), 0.0, 1.0)
    result[step] = np.where(replay_margins[step] > 0.0, max_p * normalized, 0.0)
  if not np.isfinite(result).all():
    raise RuntimeError("shadow impulse-CaT replay produced a non-finite value")
  return result


def contact_events(
  contact: np.ndarray,
  *,
  terminal: np.ndarray,
  episode_id: np.ndarray,
  physics_dt_s: float,
) -> tuple[list[dict[str, object]], np.ndarray]:
  """Measure contiguous physical contacts at substep rate and retain censoring status."""
  contact = np.asarray(contact, dtype=bool)
  terminal = np.asarray(terminal, dtype=bool)
  episode_id = np.asarray(episode_id, dtype=np.int64)
  if contact.ndim != 2 or not (contact.shape == terminal.shape == episode_id.shape):
    raise ValueError("contact, terminal, and episode_id must share shape (substeps, envs)")
  if not np.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
    raise ValueError("physics_dt_s must be finite and positive")

  substeps, envs = contact.shape
  labels = np.full(contact.shape, -1, dtype=np.int64)
  events: list[dict[str, object]] = []
  for env_id in range(envs):
    start: int | None = None
    event_episode: int | None = None
    for substep in range(substeps):
      current_episode = int(episode_id[substep, env_id])
      if start is not None and current_episode != event_episode:
        end = substep - 1
        events.append(
          _contact_event_record(
            len(events), env_id, int(event_episode), start, end, physics_dt_s,
            right_censored=True, censor_reason="episode_boundary",
          )
        )
        start = None
        event_episode = None
      if bool(contact[substep, env_id]):
        if start is None:
          start = substep
          event_episode = current_episode
        labels[substep, env_id] = len(events)

      if start is None:
        continue
      terminal_now = bool(terminal[substep, env_id])
      released_now = not bool(contact[substep, env_id])
      trace_end = substep == substeps - 1
      if terminal_now or released_now or trace_end:
        end = substep if not released_now else substep - 1
        # A sampled release is a completed event even if the control boundary also terminates.
        reason = None if released_now else ("terminal" if terminal_now else "trace_end")
        events.append(
          _contact_event_record(
            len(events), env_id, int(event_episode), start, end, physics_dt_s,
            right_censored=reason is not None, censor_reason=reason,
          )
        )
        start = None
        event_episode = None
  return events, labels


def _contact_event_record(
  event_id: int,
  env_id: int,
  episode_id: int,
  start: int,
  end: int,
  physics_dt_s: float,
  *,
  right_censored: bool,
  censor_reason: str | None,
) -> dict[str, object]:
  count = end - start + 1
  return {
    "event_id": event_id,
    "env_id": env_id,
    "episode_id": episode_id,
    "start_substep": start,
    "end_substep": end,
    "contact_substeps": count,
    "observed_duration_ms": count * physics_dt_s * 1000.0,
    "right_censored": right_censored,
    "censor_reason": censor_reason,
  }


def contiguous_activation_events(
  active: np.ndarray,
  done: np.ndarray,
  delta_impulse: np.ndarray,
) -> list[dict[str, int | float]]:
  """Group consecutive 50 Hz activation reads, splitting at episode boundaries."""
  active = np.asarray(active, dtype=bool)
  done = np.asarray(done, dtype=bool)
  delta_impulse = np.asarray(delta_impulse, dtype=np.float64)
  if active.ndim != 2 or not (active.shape == done.shape == delta_impulse.shape):
    raise ValueError("active, done, and delta_impulse must share shape (steps, envs)")
  if not np.isfinite(delta_impulse).all():
    raise ValueError("delta_impulse must be finite")
  if ((delta_impulse < 0.0) | (delta_impulse > 1.0)).any():
    raise ValueError("delta_impulse must lie in [0, 1]")

  steps, envs = active.shape
  events: list[dict[str, int | float]] = []
  for env_id in range(envs):
    start: int | None = None
    survival = 1.0
    reads = 0
    for step in range(steps):
      if bool(active[step, env_id]):
        if start is None:
          start = step
          survival = 1.0
          reads = 0
        survival *= 1.0 - float(delta_impulse[step, env_id])
        reads += 1
      close = start is not None and (
        bool(done[step, env_id])
        or not bool(active[step, env_id])
        or step == steps - 1
      )
      if close:
        end = step if bool(active[step, env_id]) else step - 1
        right_censored = (
          step == steps - 1
          and bool(active[step, env_id])
          and not bool(done[step, env_id])
        )
        events.append(
          {
            "env_id": env_id,
            "start_step": start,
            "end_step": end,
            "reads": reads,
            "event_pressure": round(1.0 - survival, 12),
            "right_censored": right_censored,
            "censor_reason": "trace_end" if right_censored else None,
          }
        )
        start = None
  return events


class _LiveSurveyRecorder:
  """Capture existing hook/accumulator state before auto-reset can clear it."""

  _CONTROL_FIELDS = (
    "delta_velocity_per_joint",
    "delta_impulse_per_joint",
    "delta_velocity",
    "delta_impulse",
    "delta",
    "winner_constraint",
    "winner_joint",
    "lambda_per_joint",
    "cap_utilization_per_joint",
    "raw_margin_per_joint",
    "positive_margin_per_joint",
  )
  _TASK_SCALAR_FIELDS = (
    "success",
    "timeout",
    "nail_depth_m",
    "delivered_total_n_s",
    "first_strike_productive",
    "first_strike_delivered_n_s",
  )
  _TASK_JOINT_FIELDS = (
    "substep_peak_qv_per_joint",
    "vic_p",
    "vic_kp",
    "vic_kd",
  )

  def __init__(self, env: Any):
    import torch

    from mjlab.envs.mdp.actions.actions import JointPositionAction
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from src.tasks.hammer.cat.hook import CatSoftHook
    from src.tasks.hammer.mdp.first_strike import (
      FirstStrikeEventTracker,
      _ENV_FIRST_STRIKE_ATTR,
    )
    from src.tasks.hammer.mdp.impulse_bound import (
      SubstepDeliveredImpulse,
      SubstepImpulseAccumulator,
      _ENV_SUBSTEP_DELIVERED_ATTR,
      _ENV_SUBSTEP_IMPULSE_ATTR,
    )
    from src.tasks.hammer.mdp.variable_impedance import JointStiffnessAction
    from src.tasks.hammer.mdp.velocity_bound import (
      SubstepPeakJointVel,
      _ENV_SUBSTEP_ATTR,
    )
    from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH

    self._torch = torch
    self._env = env
    manager = env.metrics_manager
    if "cat_soft" not in manager.active_terms:
      raise RuntimeError("activation survey requires the existing cat_soft metric")
    hook_index = manager.active_terms.index("cat_soft")
    hook = manager._term_cfgs[hook_index].func
    if not isinstance(hook, CatSoftHook):
      raise RuntimeError("cat_soft metric is not the existing CatSoftHook")
    if hook._imp_max_p != 0.0:
      raise RuntimeError("survey trajectories must keep impulse CaT log-only (imp_max_p=0)")
    accumulator = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
    if type(accumulator) is not SubstepImpulseAccumulator:
      raise RuntimeError("activation survey requires the shipped substep impulse accumulator")
    delivered = getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR, None)
    if type(delivered) is not SubstepDeliveredImpulse:
      raise RuntimeError("activation survey requires the shipped delivered-impulse accumulator")
    first_strike = getattr(env, _ENV_FIRST_STRIKE_ATTR, None)
    if type(first_strike) is not FirstStrikeEventTracker:
      raise RuntimeError("activation survey requires the shipped first-strike tracker")
    velocity = getattr(env, _ENV_SUBSTEP_ATTR, None)
    if type(velocity) is not SubstepPeakJointVel:
      raise RuntimeError("activation survey requires the shipped substep velocity tracker")
    expected_trackers = {
      "first_strike": (first_strike, "first-strike tracker"),
      "substep_impulse": (accumulator, "substep impulse accumulator"),
      "substep_delivered": (delivered, "delivered-impulse accumulator"),
      "substep_peak_qv": (velocity, "substep velocity tracker"),
    }
    for term_name, (tracker, label) in expected_trackers.items():
      if term_name not in manager.active_terms:
        raise RuntimeError(f"activation survey requires the shipped {label}")
      term_index = manager.active_terms.index(term_name)
      if manager._term_cfgs[term_index].func is not tracker:
        raise RuntimeError(
          f"activation survey {label} is not the exact configured metric term"
        )

    action_manager = env.action_manager
    if tuple(action_manager.active_terms) != ("joint_position", "joint_stiffness"):
      raise RuntimeError(
        "activation survey requires the exact VIC action pair "
        "('joint_position', 'joint_stiffness')"
      )
    position_action = action_manager.get_term("joint_position")
    stiffness_action = action_manager.get_term("joint_stiffness")
    if type(position_action) is not JointPositionAction:
      raise RuntimeError("activation survey joint_position action identity drift")
    if type(stiffness_action) is not JointStiffnessAction:
      raise RuntimeError("activation survey joint_stiffness action identity drift")

    nail_cfg = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
    nail_cfg.resolve(env.scene)

    self._hook = hook
    self._accumulator = accumulator
    self._delivered = delivered
    self._first_strike = first_strike
    self._velocity = velocity
    self._stiffness_action = stiffness_action
    self._nail = env.scene[nail_cfg.name]
    self._nail_joint_ids = nail_cfg.joint_ids
    self._contact = env.scene["hammer_nail_contact"]
    self._episode_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._control: dict[str, list[Any]] = {name: [] for name in self._CONTROL_FIELDS}
    self._control.update(
      {name: [] for name in (*self._TASK_SCALAR_FIELDS, *self._TASK_JOINT_FIELDS)}
    )
    self._control["policy_action"] = []
    self._control["done"] = []
    self._control["episode_id"] = []
    self._substep_contact: list[Any] = []
    self._substep_rolling: list[Any] = []
    self._substep_episode_id: list[Any] = []
    self._active_limit: Any | None = None

    original_substep = manager.compute_substep

    def capture_substep() -> None:
      original_substep()
      in_contact = (self._contact.data.found > 0).any(dim=-1)
      rolling = self._accumulator._rolling
      if in_contact.shape != (env.num_envs,) or rolling.shape != (env.num_envs, len(JOINT_NAMES)):
        raise RuntimeError("live 500 Hz contact/rolling-impulse shape drift")
      self._substep_contact.append(in_contact.detach().clone())
      self._substep_rolling.append(rolling.detach().clone())
      self._substep_episode_id.append(self._episode_id.detach().clone())

    manager.compute_substep = capture_substep  # type: ignore[method-assign]

    original_compute = manager.compute

    def capture_control() -> None:
      original_compute()
      telemetry = self._hook.constraint_telemetry()
      if self._active_limit is None:
        self._active_limit = telemetry["active_limit_per_joint"].detach().clone()
      elif not torch.equal(self._active_limit, telemetry["active_limit_per_joint"]):
        raise RuntimeError("live impulse threshold changed inside one frozen rollout")
      for name in self._CONTROL_FIELDS:
        self._control[name].append(telemetry[name].detach().clone())
      done = (env.reset_terminated | env.reset_time_outs).detach().clone()
      nail_depth = self._nail.data.joint_pos[:, self._nail_joint_ids].squeeze(1)
      stiffness = self._stiffness_action.telemetry
      task_values = {
        "success": env.reset_terminated,
        "timeout": env.reset_time_outs,
        "nail_depth_m": nail_depth.clamp(0.0, NAIL_GOAL_DEPTH),
        "delivered_total_n_s": self._delivered.delivered,
        "first_strike_productive": self._first_strike.productive,
        "first_strike_delivered_n_s": self._first_strike.delivered,
        "substep_peak_qv_per_joint": self._velocity.peak_qv_joint,
        "vic_p": stiffness.p,
        "vic_kp": stiffness.kp,
        "vic_kd": stiffness.kd,
      }
      policy_action = action_manager.action
      scalar_shape = (env.num_envs,)
      joint_shape = (env.num_envs, len(JOINT_NAMES))
      for name in self._TASK_SCALAR_FIELDS:
        if task_values[name].shape != scalar_shape:
          raise RuntimeError(
            f"live survey field {name} has shape {task_values[name].shape}, "
            f"expected {scalar_shape}"
          )
      for name in self._TASK_JOINT_FIELDS:
        if task_values[name].shape != joint_shape:
          raise RuntimeError(
            f"live survey field {name} has shape {task_values[name].shape}, "
            f"expected {joint_shape}"
          )
      if policy_action.shape != (env.num_envs, 12):
        raise RuntimeError(
          f"live survey policy_action has shape {policy_action.shape}, "
          f"expected {(env.num_envs, 12)}"
        )
      for name, value in task_values.items():
        self._control[name].append(value.detach().clone())
      self._control["policy_action"].append(policy_action.detach().clone())
      self._control["done"].append(done)
      self._control["episode_id"].append(self._episode_id.detach().clone())
      self._episode_id += done.long()

    manager.compute = capture_control  # type: ignore[method-assign]

  def numpy_trace(self) -> dict[str, np.ndarray]:
    torch = self._torch
    if not self._control["delta"] or not self._substep_contact:
      raise RuntimeError("live survey recorder captured no rollout data")
    trace = {
      name: torch.stack(values).detach().cpu().numpy()
      for name, values in self._control.items()
    }
    trace.update(
      {
        "substep_contact": torch.stack(self._substep_contact).detach().cpu().numpy(),
        "substep_rolling_per_joint": (
          torch.stack(self._substep_rolling).detach().cpu().numpy()
        ),
        "substep_episode_id": (
          torch.stack(self._substep_episode_id).detach().cpu().numpy()
        ),
        "active_limit_per_joint": self._active_limit.detach().cpu().numpy(),
      }
    )
    steps = trace["delta"].shape[0]
    scalar_shape = (steps, self._env.num_envs)
    joint_shape = (*scalar_shape, len(JOINT_NAMES))
    for name in (
      "delta_velocity_per_joint",
      "delta_impulse_per_joint",
      "lambda_per_joint",
      "cap_utilization_per_joint",
      "raw_margin_per_joint",
      "positive_margin_per_joint",
      *self._TASK_JOINT_FIELDS,
    ):
      if trace[name].shape != joint_shape:
        raise RuntimeError(f"live survey field {name} has shape {trace[name].shape}, expected {joint_shape}")
    for name in (
      "delta_velocity",
      "delta_impulse",
      "delta",
      "winner_constraint",
      "winner_joint",
      "done",
      "episode_id",
      *self._TASK_SCALAR_FIELDS,
    ):
      if trace[name].shape != scalar_shape:
        raise RuntimeError(f"live survey field {name} has shape {trace[name].shape}, expected {scalar_shape}")
    if trace["policy_action"].shape != (*scalar_shape, 12):
      raise RuntimeError(
        f"live survey field policy_action has shape {trace['policy_action'].shape}, "
        f"expected {(*scalar_shape, 12)}"
      )
    if trace["active_limit_per_joint"].shape != (len(JOINT_NAMES),):
      raise RuntimeError("live active impulse threshold does not have shape (6,)")
    expected_limit = np.asarray(
      PROVISIONAL_CAPS_N_M_S, dtype=trace["active_limit_per_joint"].dtype
    )
    if not np.array_equal(trace["active_limit_per_joint"], expected_limit):
      raise RuntimeError("live active impulse threshold differs from the provisional survey cap")
    if not np.array_equal(
      trace["delta_impulse_per_joint"],
      np.zeros_like(trace["delta_impulse_per_joint"]),
    ):
      raise RuntimeError("imp_max_p=0 did not produce zero per-joint impulse activation")
    if not np.array_equal(trace["delta_impulse"], np.zeros_like(trace["delta_impulse"])):
      raise RuntimeError("imp_max_p=0 did not produce identically zero impulse activation")
    expected = np.maximum(trace["delta_velocity"], trace["delta_impulse"])
    if not np.array_equal(trace["delta"], expected):
      raise RuntimeError("live aggregate delta differs from exact max soft-OR")
    if not np.array_equal(
      trace["delta_velocity"], trace["delta_velocity_per_joint"].max(axis=2)
    ):
      raise RuntimeError("live scalar velocity delta differs from its per-joint maximum")
    if not np.array_equal(
      trace["delta_impulse"], trace["delta_impulse_per_joint"].max(axis=2)
    ):
      raise RuntimeError("live scalar impulse delta differs from its per-joint maximum")
    raw_margin = trace["lambda_per_joint"] - trace["active_limit_per_joint"]
    if not np.array_equal(trace["raw_margin_per_joint"], raw_margin):
      raise RuntimeError("live raw impulse margin differs from Lambda minus cap")
    if not np.array_equal(
      trace["positive_margin_per_joint"], np.maximum(raw_margin, 0.0)
    ):
      raise RuntimeError("live positive impulse margin differs from max(Lambda minus cap, 0)")
    expected_utilization = trace["lambda_per_joint"] / trace["active_limit_per_joint"]
    if not np.allclose(
      trace["cap_utilization_per_joint"], expected_utilization, rtol=1e-6, atol=0.0
    ):
      raise RuntimeError("live cap utilization differs from Lambda divided by cap")
    for name, value in trace.items():
      if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise RuntimeError(f"live survey field {name} contains a non-finite value")
    return trace


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _initial_population_sha256(env: Any) -> str:
  robot = env.scene["robot"].data
  nail = env.scene["nail_block"].data
  payload = {
    "robot_joint_pos": robot.joint_pos.detach().cpu().tolist(),
    "robot_joint_vel": robot.joint_vel.detach().cpu().tolist(),
    "nail_joint_pos": nail.joint_pos.detach().cpu().tolist(),
    "nail_joint_vel": nail.joint_vel.detach().cpu().tolist(),
  }
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  return hashlib.sha256(encoded).hexdigest()


def _prepare_survey_measurement_config(
  env_cfg: Any, provisional_caps: tuple[float, ...]
) -> dict[str, object]:
  """Fail closed on every live setting assumed by the offline impulse-CaT replay."""
  from src.tasks.hammer.mdp.impulse_bound import SubstepImpulseAccumulator

  params = env_cfg.metrics["cat_soft"].params
  expected_cat = {
    "use_vel": True,
    "use_impulse": True,
    "imp_max_p": 0.0,
    "imp_seed": IMPULSE_SEED,
    "tau": CAT_TAU,
    "min_p": CAT_MIN_P,
    "limit": VELOCITY_LIMIT_RAD_S,
    "max_p": VELOCITY_MAX_P,
    "vel_detection": VELOCITY_DETECTION,
  }
  for name, expected in expected_cat.items():
    if params.get(name) != expected:
      raise RuntimeError(
        f"frozen survey requires cat_soft.{name}={expected!r}, got {params.get(name)!r}"
      )
  if tuple(provisional_caps) != PROVISIONAL_CAPS_N_M_S:
    raise RuntimeError("survey provisional threshold snapshot drifted from project configuration")
  impulse_term = env_cfg.metrics["substep_impulse"]
  if impulse_term.func is not SubstepImpulseAccumulator:
    raise RuntimeError("survey substep_impulse.func is not SubstepImpulseAccumulator")
  if impulse_term.per_substep is not True:
    raise RuntimeError("survey substep_impulse.per_substep must be True")
  if impulse_term.reduce != "last":
    raise RuntimeError("survey substep_impulse.reduce must be 'last'")
  impulse_params = impulse_term.params
  if impulse_params.get("sensor_name") != "hammer_nail_contact":
    raise RuntimeError("survey substep_impulse.sensor_name drifted from hammer_nail_contact")
  if impulse_params.get("subtract_baseline") is not True:
    raise RuntimeError("survey substep_impulse.subtract_baseline must be True")
  if impulse_params.get("event_window_substeps") != IMPULSE_WINDOW_SUBSTEPS:
    raise RuntimeError(
      "survey substep_impulse.event_window_substeps drifted from 25 physics substeps"
    )
  robot_cfg = impulse_params.get("robot_cfg")
  joint_names = None if robot_cfg is None else tuple(robot_cfg.joint_names)
  if joint_names != JOINT_NAMES:
    raise RuntimeError(
      f"survey substep_impulse.joint_names must be {JOINT_NAMES}, got {joint_names}"
    )
  if not np.isclose(
    float(env_cfg.sim.mujoco.timestep), PHYSICS_DT_S, rtol=0.0, atol=0.0
  ):
    raise RuntimeError("survey physics timestep drifted from 2 ms")
  if int(env_cfg.decimation) != CONTROL_DECIMATION:
    raise RuntimeError("survey control decimation drifted from 10 physics substeps")
  contact_row_params = env_cfg.metrics["substep_impulse_rows"].params
  if contact_row_params.get("enabled") is not False:
    raise RuntimeError("survey requires the unused contact-row diagnostic to remain disabled")

  # Measurement-only override: with imp_max_p=0 this changes margins/utilization but cannot change
  # physics, actions, rewards, PPO data, or aggregate delta.
  params["imp_limit"] = list(provisional_caps)
  return {
    "cat_replay": {
      "tau": CAT_TAU,
      "min_p": CAT_MIN_P,
      "imp_seed": IMPULSE_SEED,
      "imp_max_p_live": 0.0,
    },
    "velocity_cat": {
      "limit_rad_s": VELOCITY_LIMIT_RAD_S,
      "max_p": VELOCITY_MAX_P,
      "detection": VELOCITY_DETECTION,
    },
    "physics_dt_s": PHYSICS_DT_S,
    "control_decimation": CONTROL_DECIMATION,
    "impulse_window_substeps": IMPULSE_WINDOW_SUBSTEPS,
    "contact_row_diagnostic_enabled": False,
  }


def _run_population(
  *,
  checkpoint: Path,
  device: str,
  num_envs: int,
  seed: int,
  rng_seeds: EvaluationRngSeeds,
  steps: int | None,
  stochastic: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
  """Run a frozen policy without calling storage collection, an optimizer, or a learner update."""
  if rng_seeds != EvaluationRngSeeds.from_evaluation_seed(seed):
    raise ValueError("RNG streams must match evaluation seed")

  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from tensordict import TensorDict

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.tasks.hammer.config.z1.env_cfgs import PROVISIONAL_IMP_J_LIMIT

  env_cfg = load_env_cfg(VIC_TASK, play=False)
  agent_cfg = load_rl_cfg(VIC_TASK)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  env_cfg.auto_reset = steps is not None
  if tuple(PROVISIONAL_IMP_J_LIMIT) != PROVISIONAL_CAPS_N_M_S:
    raise RuntimeError("survey provisional threshold snapshot drifted from project configuration")
  measurement_protocol = _prepare_survey_measurement_config(
    env_cfg, tuple(PROVISIONAL_IMP_J_LIMIT)
  )
  if agent_cfg.num_steps_per_env != TRAINING_LIKE_STEPS:
    raise RuntimeError("registered PPO rollout length is no longer 24 control steps")

  torch.manual_seed(seed)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  recorder = _LiveSurveyRecorder(env)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(VIC_TASK) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    torch.manual_seed(seed)
    if stochastic:
      _install_evaluator_rng_streams(
        env,
        reset_seed=rng_seeds.reset,
        observation_seed=rng_seeds.observation,
      )
    observations, _ = wrapped.reset()
    population_hash = _initial_population_sha256(env)
    if not stochastic:
      _install_evaluator_rng_streams(
        env,
        reset_seed=rng_seeds.reset,
        observation_seed=rng_seeds.observation,
      )
    action_stream = _TorchRngStream(rng_seeds.action, device)
    completed = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    control_steps = steps if steps is not None else env.max_episode_length
    for _ in range(control_steps):
      with torch.no_grad():
        if stochastic:
          actions = action_stream.run(
            lambda: policy(observations, stochastic_output=True)
          )
        else:
          actions = policy(observations, stochastic_output=False)
      observations, _, _, _ = wrapped.step(actions)
      if steps is not None:
        continue
      done = env.reset_terminated | env.reset_time_outs
      completed |= done
      if bool(completed.all()):
        break
      done_ids = torch.nonzero(done, as_tuple=False).flatten()
      if done_ids.numel():
        cpu_state = torch.get_rng_state()
        device_obj = torch.device(env.device)
        cuda_state = (
          torch.cuda.get_rng_state(device_obj) if device_obj.type == "cuda" else None
        )
        try:
          reset_obs, _ = env.reset(env_ids=done_ids)
        finally:
          torch.set_rng_state(cpu_state)
          if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device_obj)
        reset_td = TensorDict(reset_obs, batch_size=[env.num_envs])
        merged = observations.clone()
        merged[done_ids] = reset_td[done_ids]
        observations = merged
    if steps is None and not bool(completed.all()):
      raise RuntimeError(f"fixed survey completed only {int(completed.sum())}/{num_envs} worlds")
    trace = recorder.numpy_trace()
    metadata = {
      "seed": seed,
      "num_envs": num_envs,
      "control_steps": int(trace["delta"].shape[0]),
      "policy_mode": "sampled" if stochastic else "mean",
      "auto_reset": bool(env_cfg.auto_reset),
      "initial_population_sha256": population_hash,
      "rng_streams": asdict(rng_seeds),
      **measurement_protocol,
    }
  finally:
    wrapped.close()
  return trace, metadata


def _finite_summary(values: list[float]) -> dict[str, float | None]:
  if not values:
    return {"median": None, "p95": None, "max": None}
  array = np.asarray(values, dtype=np.float64)
  return {
    "median": float(np.median(array)),
    "p95": float(np.quantile(array, 0.95)),
    "max": float(np.max(array)),
  }


def _finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
  flat = np.asarray(values, dtype=np.float64).reshape(-1)
  if flat.size == 0:
    return {"p50": None, "p95": None, "p99": None, "max": None}
  if not np.isfinite(flat).all():
    raise ValueError("quantile input must be finite")
  return {
    "p50": float(np.quantile(flat, 0.50)),
    "p95": float(np.quantile(flat, 0.95)),
    "p99": float(np.quantile(flat, 0.99)),
    "max": float(np.max(flat)),
  }


def _native_cap_margins(lambda_per_joint: np.ndarray, caps: tuple[float, ...]) -> np.ndarray:
  """Subtract caps in the live trace dtype, then promote margins for stable offline replay."""
  lam = np.asarray(lambda_per_joint)
  if lam.ndim != 3 or not np.issubdtype(lam.dtype, np.floating) or not np.isfinite(lam).all():
    raise ValueError(
      "lambda_per_joint must be a finite floating array with shape (steps, envs, joints)"
    )
  cap_native = np.asarray(caps, dtype=lam.dtype)
  if cap_native.shape != (lam.shape[2],):
    raise ValueError("caps must contain one value per joint")
  return (lam - cap_native).astype(np.float64)


def _candidate_activation_window_summaries(
  *,
  active: np.ndarray,
  done: np.ndarray,
  delta_velocity: np.ndarray,
  unit_delta_impulse: np.ndarray,
  max_p: float,
) -> list[dict[str, object]]:
  """Record separate CaT components and winner counts for each 50 Hz activation window."""
  active = np.asarray(active, dtype=bool)
  done = np.asarray(done, dtype=bool)
  delta_velocity = np.asarray(delta_velocity, dtype=np.float64)
  unit_delta_impulse = np.asarray(unit_delta_impulse, dtype=np.float64)
  if active.ndim != 2 or not (
    active.shape == done.shape == delta_velocity.shape == unit_delta_impulse.shape
  ):
    raise ValueError("candidate activation fields must share shape (steps, envs)")
  if not np.isfinite(delta_velocity).all() or not np.isfinite(unit_delta_impulse).all():
    raise ValueError("candidate activation deltas must be finite")
  if not np.isfinite(max_p) or not (0.0 <= max_p <= 1.0):
    raise ValueError("candidate imp_max_p must lie in [0, 1]")
  if (
    (delta_velocity < 0.0).any()
    or (delta_velocity > 1.0).any()
    or (unit_delta_impulse < 0.0).any()
    or (unit_delta_impulse > 1.0).any()
  ):
    raise ValueError("candidate activation deltas must lie in [0, 1]")
  if bool((active & (unit_delta_impulse <= 0.0)).any()):
    raise ValueError("an active impulse window must have positive unit impulse activation")

  delta_impulse = max_p * unit_delta_impulse
  combined = np.maximum(delta_velocity, delta_impulse)
  windows = contiguous_activation_events(
    active,
    done,
    np.where(active, delta_impulse, 0.0),
  )
  summaries: list[dict[str, object]] = []
  for event in windows:
    env_id = int(event["env_id"])
    start = int(event["start_step"])
    end = int(event["end_step"])
    velocity_values = delta_velocity[start : end + 1, env_id]
    impulse_values = delta_impulse[start : end + 1, env_id]
    combined_values = combined[start : end + 1, env_id]
    impulse_wins = impulse_values > velocity_values
    velocity_wins = velocity_values > impulse_values
    summaries.append(
      {
        **event,
        "delta_velocity": _finite_summary(velocity_values.tolist()),
        "delta_impulse": _finite_summary(impulse_values.tolist()),
        "combined_delta": _finite_summary(combined_values.tolist()),
        "winner_reads": {
          "velocity": int(velocity_wins.sum()),
          "impulse": int(impulse_wins.sum()),
          "tie": int((~(velocity_wins | impulse_wins)).sum()),
        },
      }
    )
  return summaries


def _observed_segment_peaks(
  values: np.ndarray,
  episode_id: np.ndarray,
  valid: np.ndarray,
) -> np.ndarray:
  peaks: list[np.ndarray] = []
  _, envs, _ = values.shape
  for env_id in range(envs):
    ids = np.unique(episode_id[valid[:, env_id], env_id])
    for current in ids:
      mask = valid[:, env_id] & (episode_id[:, env_id] == current)
      if bool(mask.any()):
        peaks.append(values[mask, env_id].max(axis=0))
  if not peaks:
    raise RuntimeError("survey contains no valid episodes")
  return np.asarray(peaks, dtype=np.float64)


def _associate_violating_reads(
  trace: dict[str, np.ndarray],
  *,
  active: np.ndarray,
  margins: np.ndarray,
  unit_per_joint: np.ndarray,
  event_labels: np.ndarray,
  valid: np.ndarray,
) -> tuple[list[dict[str, object]], dict[int, list[tuple[int, int]]]]:
  lam = np.asarray(trace["lambda_per_joint"], dtype=np.float64)
  rolling = np.asarray(trace["substep_rolling_per_joint"], dtype=np.float64)
  substep_episode = np.asarray(trace["substep_episode_id"], dtype=np.int64)
  control_episode = np.asarray(trace["episode_id"], dtype=np.int64)
  steps, envs, joints = lam.shape
  if rolling.shape != (steps * CONTROL_DECIMATION, envs, joints):
    raise RuntimeError("500 Hz rolling trace does not align with the 50 Hz control trace")
  if unit_per_joint.shape != lam.shape:
    raise RuntimeError("normalized impulse-CaT trace does not align with Lambda")

  records: list[dict[str, object]] = []
  by_event: dict[int, list[tuple[int, int]]] = {}
  for step, env_id in np.argwhere(active & valid):
    responsible_joint = int(np.argmax(unit_per_joint[step, env_id]))
    violating_joints = np.flatnonzero(margins[step, env_id] > 0.0).astype(int).tolist()
    block_start = int(step) * CONTROL_DECIMATION
    block_end = block_start + CONTROL_DECIMATION
    current_episode = int(control_episode[step, env_id])

    joint_associations: dict[str, dict[str, object]] = {}
    for joint in violating_joints:
      block = rolling[block_start:block_end, env_id, joint]
      source = block_start + int(np.argmax(block))
      if not np.isclose(
        rolling[source, env_id, joint],
        lam[step, env_id, joint],
        rtol=0.0,
        atol=2e-6,
      ):
        raise RuntimeError(
          "50 Hz Lambda does not match a 500 Hz rolling-window source sample"
        )
      window_start = max(0, source - IMPULSE_WINDOW_SUBSTEPS + 1)
      indices = np.arange(window_start, source + 1)
      indices = indices[substep_episode[indices, env_id] == current_episode]
      event_ids = tuple(
        int(value)
        for value in np.unique(event_labels[indices, env_id])
        if int(value) >= 0
      )
      if not event_ids:
        raise RuntimeError("a positive contact-masked impulse margin has no source contact event")
      joint_associations[str(joint)] = {
        "source_substep": source,
        "contact_event_ids": list(event_ids),
        "physical_event_association_ambiguous": len(event_ids) > 1,
      }

    responsible_association = joint_associations[str(responsible_joint)]
    all_event_ids = sorted(
      {
        int(event_id)
        for association in joint_associations.values()
        for event_id in association["contact_event_ids"]
      }
    )
    record = {
      "control_step": int(step),
      "env_id": int(env_id),
      "episode_id": current_episode,
      "responsible_joint": responsible_joint,
      "violating_joints": violating_joints,
      "source_substep": int(responsible_association["source_substep"]),
      "responsible_contact_event_ids": list(
        responsible_association["contact_event_ids"]
      ),
      "contact_event_ids": all_event_ids,
      "physical_event_association_ambiguous": len(all_event_ids) > 1,
      "joint_associations": joint_associations,
    }
    records.append(record)
    for event_id in all_event_ids:
      by_event.setdefault(event_id, []).append((int(step), int(env_id)))
  return records, by_event


def _physical_event_read_counts(
  records: list[dict[str, object]], *, event_ids: list[int]
) -> list[dict[str, int]]:
  """Count unambiguous reads per physical event without duplicating multi-event windows."""
  result: list[dict[str, int]] = []
  for event_id in event_ids:
    exclusive: set[tuple[int, int]] = set()
    ambiguous: set[tuple[int, int]] = set()
    for record in records:
      associated = [int(value) for value in record["contact_event_ids"]]
      if event_id not in associated:
        continue
      pair = (int(record["control_step"]), int(record["env_id"]))
      (exclusive if len(associated) == 1 else ambiguous).add(pair)
    result.append(
      {
        "event_id": int(event_id),
        "exclusive_reads": len(exclusive),
        "ambiguous_read_associations": len(ambiguous),
      }
    )
  return result


def _utility_summary(
  trace: dict[str, np.ndarray],
  *,
  episode_id: np.ndarray,
  done: np.ndarray,
  valid: np.ndarray,
  substep_contact: np.ndarray,
  substep_episode_id: np.ndarray,
  first_episode_only: bool,
) -> dict[str, object]:
  """Reduce exact pre-reset task, velocity, and VIC-action telemetry."""
  scalar_shape = episode_id.shape
  joint_shape = (*scalar_shape, len(JOINT_NAMES))
  boolean_fields = ("success", "timeout", "first_strike_productive")
  scalar_float_fields = (
    "nail_depth_m",
    "delivered_total_n_s",
    "first_strike_delivered_n_s",
  )
  joint_float_fields = (
    "substep_peak_qv_per_joint",
    "vic_p",
    "vic_kp",
    "vic_kd",
  )
  arrays: dict[str, np.ndarray] = {}
  for name in boolean_fields:
    value = np.asarray(trace[name])
    if value.shape != scalar_shape or value.dtype != np.bool_:
      raise ValueError(f"{name} must be boolean with shape (steps, envs)")
    arrays[name] = value
  for name in scalar_float_fields:
    value = np.asarray(trace[name])
    if (
      value.shape != scalar_shape
      or not np.issubdtype(value.dtype, np.floating)
      or not np.isfinite(value).all()
    ):
      raise ValueError(f"{name} must be finite floating with shape (steps, envs)")
    arrays[name] = value.astype(np.float64)
  for name in joint_float_fields:
    value = np.asarray(trace[name])
    if (
      value.shape != joint_shape
      or not np.issubdtype(value.dtype, np.floating)
      or not np.isfinite(value).all()
    ):
      raise ValueError(f"{name} must be finite floating with shape (steps, envs, 6)")
    arrays[name] = value.astype(np.float64)

  terminal = done & valid
  success = arrays["success"]
  timeout = arrays["timeout"]
  productive = arrays["first_strike_productive"]
  if not np.array_equal(done, success | timeout):
    raise ValueError("done must be the exact union of success and timeout")

  qv_segment_peaks = _observed_segment_peaks(
    arrays["substep_peak_qv_per_joint"], episode_id, valid
  )
  qv_violations = qv_segment_peaks > VELOCITY_LIMIT_RAD_S
  velocity_compliance = {
    "limit_rad_s": VELOCITY_LIMIT_RAD_S,
    "segments": int(qv_segment_peaks.shape[0]),
    "any_joint_violating_segments": int(qv_violations.any(axis=1).sum()),
    "any_joint_violation_rate": float(qv_violations.any(axis=1).mean()),
    "max_joint_speed_rad_s": _finite_quantiles(qv_segment_peaks.max(axis=1)),
    "per_joint": {
      name: {
        "violating_segments": int(qv_violations[:, joint].sum()),
        "violation_rate": float(qv_violations[:, joint].mean()),
        "peak_speed_rad_s": _finite_quantiles(qv_segment_peaks[:, joint]),
      }
      for joint, name in enumerate(JOINT_NAMES)
    },
  }

  first_contact_records: list[dict[str, object]] = []
  steps, envs = scalar_shape
  for env_id in range(envs):
    for current_episode in np.unique(episode_id[valid[:, env_id], env_id]):
      control_mask = valid[:, env_id] & (
        episode_id[:, env_id] == current_episode
      )
      contact_indices = np.flatnonzero(
        substep_contact[:, env_id]
        & (substep_episode_id[:, env_id] == current_episode)
      )
      contact_indices = contact_indices[
        control_mask[np.minimum(contact_indices // CONTROL_DECIMATION, steps - 1)]
      ]
      if contact_indices.size == 0:
        continue
      contact_step = int(contact_indices[0] // CONTROL_DECIMATION)
      precontact_step = contact_step - 1
      if precontact_step < 0 or not bool(control_mask[precontact_step]):
        precontact_step_or_none = None
      else:
        precontact_step_or_none = precontact_step

      record: dict[str, object] = {
        "env_id": env_id,
        "episode_id": int(current_episode),
        "contact_control_step": contact_step,
        "precontact_control_step": precontact_step_or_none,
      }
      for name in ("vic_p", "vic_kp", "vic_kd"):
        values = arrays[name]
        record[f"{name}_precontact"] = (
          None
          if precontact_step_or_none is None
          else values[precontact_step_or_none, env_id].tolist()
        )
        record[f"{name}_at_contact"] = values[contact_step, env_id].tolist()
      record["right_censored"] = not bool((terminal[:, env_id] & control_mask).any())
      first_contact_records.append(record)

  return {
    "task_field_scope": (
      "complete first episodes"
      if first_episode_only
      else "24-step stochastic task fragments; unfinished and post-reset segments are censored"
    ),
    "terminal_counts": {
      "population": envs if first_episode_only else int(qv_segment_peaks.shape[0]),
      "terminal": int(terminal.sum()),
      "success": int((success & terminal).sum()),
      "timeout": int((timeout & terminal).sum()),
      "productive_first_strike": int((productive & terminal).sum()),
    },
    "terminal_nail_depth_m": _finite_quantiles(arrays["nail_depth_m"][terminal]),
    "terminal_delivered_total_n_s": _finite_quantiles(
      arrays["delivered_total_n_s"][terminal]
    ),
    "terminal_first_strike_delivered_n_s": _finite_quantiles(
      arrays["first_strike_delivered_n_s"][terminal]
    ),
    "velocity_limit_compliance": velocity_compliance,
    "first_contact_vic_gains": {
      "segments_with_contact": len(first_contact_records),
      "right_censored_segments": sum(
        bool(record["right_censored"]) for record in first_contact_records
      ),
      "records": first_contact_records,
    },
  }


def summarize_population(
  trace: dict[str, np.ndarray],
  *,
  caps: tuple[float, ...],
  first_episode_only: bool,
) -> dict[str, object]:
  """Summarize cap binding, physical events, and exact offline CaT doses for one population."""
  lam_native = np.asarray(trace["lambda_per_joint"])
  if (
    lam_native.ndim != 3
    or lam_native.shape[2] != len(JOINT_NAMES)
    or not np.issubdtype(lam_native.dtype, np.floating)
    or not np.isfinite(lam_native).all()
  ):
    raise ValueError("lambda_per_joint must be finite with shape (steps, envs, 6)")
  lam = lam_native.astype(np.float64)
  steps, envs, joints = lam.shape
  scalar_shape = (steps, envs)
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  done = np.asarray(trace["done"], dtype=bool)
  delta_velocity = np.asarray(trace["delta_velocity"], dtype=np.float64)
  delta_impulse_actual = np.asarray(trace["delta_impulse"], dtype=np.float64)
  delta_actual = np.asarray(trace["delta"], dtype=np.float64)
  for name, value in (
    ("episode_id", episode_id),
    ("done", done),
    ("delta_velocity", delta_velocity),
    ("delta_impulse", delta_impulse_actual),
    ("delta", delta_actual),
  ):
    if value.shape != scalar_shape:
      raise ValueError(f"{name} must have shape (steps, envs)")
  if not np.array_equal(delta_impulse_actual, np.zeros_like(delta_impulse_actual)):
    raise ValueError("log-only impulse delta is not identically zero")
  if not np.array_equal(delta_actual, np.maximum(delta_velocity, delta_impulse_actual)):
    raise ValueError("actual combined delta is not the exact max soft-OR")

  cap_array = np.asarray(caps, dtype=np.float64)
  if cap_array.shape != (joints,) or not np.isfinite(cap_array).all() or (cap_array <= 0).any():
    raise ValueError("caps must contain six finite positive values")
  valid = episode_id == 0 if first_episode_only else np.ones(scalar_shape, dtype=bool)
  margins = _native_cap_margins(lam_native, caps)
  unit_per_joint = shadow_impulse_cat(
    margins,
    tau=CAT_TAU,
    seed=IMPULSE_SEED,
    max_p=1.0,
    valid=valid,
  )
  unit_delta = unit_per_joint.max(axis=2)
  active = unit_delta > 0.0
  active_valid = active & valid

  sub_contact = np.asarray(trace["substep_contact"], dtype=bool)
  sub_episode = np.asarray(trace["substep_episode_id"], dtype=np.int64)
  expected_sub_shape = (steps * CONTROL_DECIMATION, envs)
  if sub_contact.shape != expected_sub_shape or sub_episode.shape != expected_sub_shape:
    raise ValueError("500 Hz contact trace is not aligned with control-rate telemetry")
  terminal = np.zeros(expected_sub_shape, dtype=bool)
  terminal[CONTROL_DECIMATION - 1 :: CONTROL_DECIMATION] = done
  physical_events, event_labels = contact_events(
    sub_contact,
    terminal=terminal,
    episode_id=sub_episode,
    physics_dt_s=PHYSICS_DT_S,
  )
  relevant_events = [
    event for event in physical_events
    if not first_episode_only or int(event["episode_id"]) == 0
  ]
  read_records, event_reads = _associate_violating_reads(
    trace,
    active=active,
    margins=margins,
    unit_per_joint=unit_per_joint,
    event_labels=event_labels,
    valid=valid,
  )
  associated_ids = sorted(event_reads)
  associated_events = [physical_events[event_id] for event_id in associated_ids]
  activation_windows = contiguous_activation_events(
    active_valid,
    done,
    np.where(active_valid, unit_delta, 0.0),
  )
  utility = _utility_summary(
    trace,
    episode_id=episode_id,
    done=done,
    valid=valid,
    substep_contact=sub_contact,
    substep_episode_id=sub_episode,
    first_episode_only=first_episode_only,
  )

  observed_segment_peaks = _observed_segment_peaks(lam, episode_id, valid)
  observed_segment_margins = _observed_segment_peaks(margins, episode_id, valid)
  segment_utilization = observed_segment_peaks / cap_array
  segment_violating = observed_segment_margins > 0.0
  segment_compliance = {
    "segments": int(segment_utilization.shape[0]),
    "any_joint_violating_segments": int(segment_violating.any(axis=1).sum()),
    "any_joint_violation_rate": float(segment_violating.any(axis=1).mean()),
    "max_joint_utilization": _finite_quantiles(segment_utilization.max(axis=1)),
    "per_joint": {
      name: {
        "violating_segments": int(segment_violating[:, joint].sum()),
        "violation_rate": float(segment_violating[:, joint].mean()),
        "utilization": _finite_quantiles(segment_utilization[:, joint]),
        "positive_margin_n_m_s": _finite_quantiles(
          observed_segment_margins[segment_violating[:, joint], joint]
        ),
      }
      for joint, name in enumerate(JOINT_NAMES)
    },
  }
  max_utilization = observed_segment_peaks.max(axis=0) / cap_array
  median_utilization = np.median(observed_segment_peaks, axis=0) / cap_array
  violating_by_joint = (margins > 0.0) & valid[:, :, None]
  per_joint: dict[str, object] = {}
  for joint, name in enumerate(JOINT_NAMES):
    joint_event_ids = {
      int(event_id)
      for record in read_records
      if str(joint) in record["joint_associations"]
      for event_id in record["joint_associations"][str(joint)]["contact_event_ids"]
    }
    responsible_event_ids = {
      int(event_id)
      for record in read_records
      if int(record["responsible_joint"]) == joint
      for event_id in record["responsible_contact_event_ids"]
    }
    per_joint[name] = {
      "cap_n_m_s": float(cap_array[joint]),
      "observed_segment_peak_median_n_m_s": float(
        np.median(observed_segment_peaks[:, joint])
      ),
      "observed_segment_peak_p95_n_m_s": float(
        np.quantile(observed_segment_peaks[:, joint], 0.95)
      ),
      "observed_segment_peak_max_n_m_s": float(np.max(observed_segment_peaks[:, joint])),
      "median_cap_utilization": float(median_utilization[joint]),
      "max_cap_utilization": float(max_utilization[joint]),
      "violating_reads": int(violating_by_joint[:, :, joint].sum()),
      "associated_physical_events": len(joint_event_ids),
      "responsible_physical_events": len(responsible_event_ids),
    }

  observed_durations = [float(event["observed_duration_ms"]) for event in relevant_events]
  complete_durations = [
    float(event["observed_duration_ms"])
    for event in relevant_events
    if not bool(event["right_censored"])
  ]
  associated_durations = [
    float(event["observed_duration_ms"]) for event in associated_events
  ]
  associated_complete_durations = [
    float(event["observed_duration_ms"])
    for event in associated_events
    if not bool(event["right_censored"])
  ]
  associated_censored_durations = [
    float(event["observed_duration_ms"])
    for event in associated_events
    if bool(event["right_censored"])
  ]

  candidates: dict[str, object] = {}
  active_count = int(active_valid.sum())
  for max_p in CANDIDATE_IMP_MAX_P:
    delta_impulse = max_p * unit_delta
    combined = np.maximum(delta_velocity, delta_impulse)
    impulse_wins = active_valid & (delta_impulse > delta_velocity)
    velocity_masks = active_valid & (delta_velocity > delta_impulse)
    ties = active_valid & ~(impulse_wins | velocity_masks)
    candidate_windows = _candidate_activation_window_summaries(
      active=active_valid,
      done=done,
      delta_velocity=delta_velocity,
      unit_delta_impulse=np.where(active_valid, unit_delta, 0.0),
      max_p=max_p,
    )
    pressures = [float(event["event_pressure"]) for event in candidate_windows]
    completed_pressures = [
      float(event["event_pressure"])
      for event in candidate_windows
      if not bool(event["right_censored"])
    ]
    read_counts = [int(event["reads"]) for event in candidate_windows]
    candidates[str(max_p)] = {
      "active_reads": active_count,
      "delta_impulse_active_mean": (
        float(delta_impulse[active_valid].mean()) if active_count else 0.0
      ),
      "delta_impulse_max": float(delta_impulse[valid].max()) if bool(valid.any()) else 0.0,
      "combined_delta_max": float(combined[valid].max()) if bool(valid.any()) else 0.0,
      "impulse_winner_reads": int(impulse_wins.sum()),
      "velocity_masked_reads": int(velocity_masks.sum()),
      "tie_reads": int(ties.sum()),
      "impulse_win_share": int(impulse_wins.sum()) / active_count if active_count else 0.0,
      "velocity_mask_share": int(velocity_masks.sum()) / active_count if active_count else 0.0,
      "reads_per_activation_window": read_counts,
      "event_pressure_observed_prefixes": pressures,
      "event_pressure_all_observed_summary": _finite_summary(pressures),
      "event_pressure_completed_only_summary": _finite_summary(completed_pressures),
      "right_censored_activation_windows": sum(
        bool(event["right_censored"]) for event in candidate_windows
      ),
      "activation_window_summaries": candidate_windows,
    }

  return {
    "shape": {"control_steps": steps, "environments": envs, "joints": joints},
    "caps_n_m_s": list(caps),
    "actual_log_only_invariants": {
      "delta_impulse_identically_zero": True,
      "combined_delta_exact_max": True,
    },
    "binding": {
      "violating_reads": active_count,
      "violating_activation_windows": len(activation_windows),
      "associated_physical_events": len(associated_ids),
      "ambiguous_multi_event_reads": sum(
        len(record["contact_event_ids"]) > 1 for record in read_records
      ),
      "reads": read_records,
      "activation_window_read_counts": [int(event["reads"]) for event in activation_windows],
      "physical_event_read_counts": _physical_event_read_counts(
        read_records, event_ids=associated_ids
      ),
    },
    "physical_contact_duration_ms": {
      "events": len(relevant_events),
      "right_censored_events": sum(bool(event["right_censored"]) for event in relevant_events),
      "all_observed_prefixes": _finite_summary(observed_durations),
      "completed_events_only": _finite_summary(complete_durations),
      "cap_associated_observed_prefixes": _finite_summary(associated_durations),
      "cap_associated_completed_only": _finite_summary(associated_complete_durations),
      "cap_associated_censored_prefixes": _finite_summary(associated_censored_durations),
    },
    "per_joint": per_joint,
    "observed_peak_scope": (
      "complete first episode"
      if first_episode_only
      else "24-step rollout episode segments; boundary fragments may be censored"
    ),
    "observed_peak_segments": int(observed_segment_peaks.shape[0]),
    "segment_compliance": segment_compliance,
    "utility": utility,
    "candidate_imp_max_p": candidates,
  }


def run_survey(*, checkpoint: Path, output_dir: Path, device: str) -> dict[str, object]:
  checkpoint = checkpoint.resolve(strict=True)
  output_dir = output_dir.absolute()
  if checkpoint.name != "model_499.pt":
    raise ValueError("survey requires the frozen final model_499.pt checkpoint")
  checkpoint_sha256 = _sha256(checkpoint)
  if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
    raise RuntimeError("frozen VIC seed-2 checkpoint SHA-256 mismatch")
  if output_dir.exists() or output_dir.is_symlink():
    raise FileExistsError(f"refusing to overwrite survey output: {output_dir}")
  output_dir.mkdir(parents=True)

  fixed_trace, fixed_protocol = _run_population(
    checkpoint=checkpoint,
    device=device,
    num_envs=FIXED_ENVS,
    seed=FIXED_SEED,
    rng_seeds=EvaluationRngSeeds.from_evaluation_seed(FIXED_SEED),
    steps=None,
    stochastic=False,
  )
  if fixed_protocol["initial_population_sha256"] != EXPECTED_FIXED_POPULATION_SHA256:
    raise RuntimeError("fixed survey initial population differs from the banked seed-2 evaluator")
  np.savez_compressed(output_dir / "fixed_trace.npz", **fixed_trace)
  # Provisional thresholds are always analyzed before the diagnostic plumbing threshold.
  fixed_provisional = summarize_population(
    fixed_trace, caps=PROVISIONAL_CAPS_N_M_S, first_episode_only=True
  )
  fixed_diagnostic = summarize_population(
    fixed_trace, caps=DIAGNOSTIC_LIMITS_N_M_S, first_episode_only=True
  )

  stochastic_trace, stochastic_protocol = _run_population(
    checkpoint=checkpoint,
    device=device,
    num_envs=TRAINING_LIKE_ENVS,
    seed=TRAINING_LIKE_SEED,
    rng_seeds=EvaluationRngSeeds.from_evaluation_seed(TRAINING_LIKE_SEED),
    steps=TRAINING_LIKE_STEPS,
    stochastic=True,
  )
  np.savez_compressed(output_dir / "training_like_trace.npz", **stochastic_trace)
  stochastic_provisional = summarize_population(
    stochastic_trace, caps=PROVISIONAL_CAPS_N_M_S, first_episode_only=False
  )
  stochastic_diagnostic = summarize_population(
    stochastic_trace, caps=DIAGNOSTIC_LIMITS_N_M_S, first_episode_only=False
  )

  payload: dict[str, object] = {
    "schema_version": 1,
    "purpose": "no-learning impulse-CaT binding, attribution, and offline dose survey",
    "task": VIC_TASK,
    "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
    "thresholds": {
      "provisional_project_caps_n_m_s": list(PROVISIONAL_CAPS_N_M_S),
      "diagnostic_only_caps_n_m_s": list(DIAGNOSTIC_LIMITS_N_M_S),
      "hardware_limit_claimed": False,
      "global_repeated_peak_multiplier": 1.0,
      "retained_time_basis_ms": 27.333333333333332,
      "time_basis_status": "historical protocol-censored contact prefix",
    },
    "measurement": {
      "physics_dt_s": PHYSICS_DT_S,
      "physics_sample_hz": 500,
      "control_decimation": CONTROL_DECIMATION,
      "control_sample_hz": 50,
      "impulse_window_substeps": IMPULSE_WINDOW_SUBSTEPS,
      "impulse_window_ms": IMPULSE_WINDOW_SUBSTEPS * PHYSICS_DT_S * 1000.0,
      "quantity": (
        "baseline-subtracted contact-masked absolute joint reaction impulse in a 50 ms "
        "sliding window"
      ),
    },
    "candidate_imp_max_p": list(CANDIDATE_IMP_MAX_P),
    "populations": {
      "fixed_mean": {
        "protocol": fixed_protocol,
        "provisional_caps": fixed_provisional,
        "diagnostic_only": fixed_diagnostic,
      },
      "training_like_sampled": {
        "protocol": stochastic_protocol,
        "provenance_note": (
          "fresh sampled rollout from post-update model_499.pt; training-like tensor shape, not "
          "the checkpoint's stored pre-update behavior rollout"
        ),
        "provisional_caps": stochastic_provisional,
        "diagnostic_only": stochastic_diagnostic,
      },
    },
  }
  encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
  (output_dir / "summary.json").write_text(encoded)
  return payload


def _git_revision(repository: Path) -> str:
  try:
    result = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=repository,
      check=True,
      capture_output=True,
      text=True,
    )
  except (OSError, subprocess.CalledProcessError) as error:
    raise RuntimeError(f"cannot read Git revision for {repository}") from error
  revision = result.stdout.strip()
  if len(revision) != 40 or any(
    character not in "0123456789abcdef" for character in revision
  ):
    raise RuntimeError(f"Git revision for {repository} is not full lowercase hexadecimal")
  return revision


def _evaluation_revisions() -> dict[str, str]:
  return {
    "code_revision": _git_revision(_REPO_ROOT),
    "asset_revision": _git_revision(_ASSET_REPO),
  }


def _require_live_impulse_log_only(protocol: Mapping[str, object]) -> None:
  cat_replay = protocol.get("cat_replay")
  if not isinstance(cat_replay, Mapping) or cat_replay.get("imp_max_p_live") != 0.0:
    raise RuntimeError("post-training evaluation requires live imp_max_p=0")


def _policy_population_payload(
  trace: dict[str, np.ndarray],
  protocol: dict[str, object],
  *,
  stochastic: bool,
) -> dict[str, object]:
  _require_live_impulse_log_only(protocol)
  payload = {
    "protocol": protocol,
    "provisional_caps": summarize_population(
      trace,
      caps=PROVISIONAL_CAPS_N_M_S,
      first_episode_only=True,
    ),
    "diagnostic_only": summarize_population(
      trace,
      caps=DIAGNOSTIC_LIMITS_N_M_S,
      first_episode_only=True,
    ),
  }
  if stochastic:
    for threshold in ("provisional_caps", "diagnostic_only"):
      summary = payload[threshold]
      summary["observed_peak_scope"] = (
        "initial 24-step rollout episode segments; unfinished segments are censored"
      )
      summary["utility"]["task_field_scope"] = (
        "initial 24-step stochastic task fragments; unfinished segments are censored"
      )
  return payload


def _segment_endpoint(summary: Mapping[str, object]) -> tuple[int, int, float]:
  compliance = summary.get("segment_compliance")
  if not isinstance(compliance, Mapping):
    raise ValueError("summary is missing segment_compliance")
  segments = compliance.get("segments")
  violations = compliance.get("any_joint_violating_segments")
  stored_rate = compliance.get("any_joint_violation_rate")
  if type(segments) is not int or segments <= 0:
    raise ValueError("summary segment count must be a positive integer")
  if type(violations) is not int or not (0 <= violations <= segments):
    raise ValueError("summary violating-segment count is invalid")
  if not isinstance(stored_rate, (int, float)) or not np.isfinite(stored_rate):
    raise ValueError("summary violation rate must be finite")
  rate = violations / segments
  if not np.isclose(float(stored_rate), rate, rtol=0.0, atol=1e-15):
    raise ValueError("summary violation rate differs from segment counts")
  return segments, violations, rate


def _utilization_quantiles(summary: Mapping[str, object]) -> dict[str, float]:
  compliance = summary["segment_compliance"]
  quantiles = compliance.get("max_joint_utilization")
  if not isinstance(quantiles, Mapping):
    raise ValueError("summary is missing max-joint utilization quantiles")
  result: dict[str, float] = {}
  for name in ("p50", "p95", "p99", "max"):
    value = quantiles.get(name)
    if not isinstance(value, (int, float)) or not np.isfinite(value):
      raise ValueError(f"summary utilization {name} must be finite")
    result[name] = float(value)
  return result


def _is_lower_hex(value: object, *, length: int) -> bool:
  return (
    isinstance(value, str)
    and len(value) == length
    and all(character in "0123456789abcdef" for character in value)
  )


def validate_evaluation_revision(
  repository: Path,
  expected_revision: str,
  *,
  approved_base_revision: str = APPROVED_EVALUATOR_BASE_REVISION,
) -> str:
  """Require a real evaluator commit descended from the approved training-code base."""
  if not _is_lower_hex(expected_revision, length=40):
    raise ValueError("expected evaluation revision must be full lowercase 40-hex")
  if not _is_lower_hex(approved_base_revision, length=40):
    raise ValueError("approved evaluator base revision must be full lowercase 40-hex")
  repository = repository.resolve(strict=True)

  def git(*arguments: str) -> subprocess.CompletedProcess[str]:
    try:
      return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
      )
    except OSError as error:
      raise RuntimeError("cannot execute Git evaluation-lineage validation") from error

  expected_commit = git("cat-file", "-e", f"{expected_revision}^{{commit}}")
  if expected_commit.returncode != 0:
    raise RuntimeError("expected evaluation revision does not exist as a Git commit")
  approved_commit = git("cat-file", "-e", f"{approved_base_revision}^{{commit}}")
  if approved_commit.returncode != 0:
    raise RuntimeError("approved evaluator base revision does not exist as a Git commit")
  ancestry = git(
    "merge-base", "--is-ancestor", approved_base_revision, expected_revision
  )
  if ancestry.returncode == 1:
    raise RuntimeError("evaluation revision is not a descendant of the approved base")
  if ancestry.returncode != 0:
    raise RuntimeError("cannot establish evaluation revision ancestry")
  return expected_revision


def _validate_exact_measurement_protocol(
  protocol: Mapping[str, object], *, label: str
) -> None:
  cat_replay = protocol.get("cat_replay")
  if not isinstance(cat_replay, Mapping) or cat_replay.get("imp_max_p_live") != 0.0:
    raise ValueError(f"{label} must record live imp_max_p=0")
  if dict(cat_replay) != {
    "tau": CAT_TAU,
    "min_p": CAT_MIN_P,
    "imp_seed": IMPULSE_SEED,
    "imp_max_p_live": 0.0,
  }:
    raise ValueError(f"{label} impulse-CaT measurement protocol drifted")
  if protocol.get("velocity_cat") != {
    "limit_rad_s": VELOCITY_LIMIT_RAD_S,
    "max_p": VELOCITY_MAX_P,
    "detection": VELOCITY_DETECTION,
  }:
    raise ValueError(f"{label} velocity-CaT measurement protocol drifted")
  expected_scalars = {
    "physics_dt_s": PHYSICS_DT_S,
    "control_decimation": CONTROL_DECIMATION,
    "impulse_window_substeps": IMPULSE_WINDOW_SUBSTEPS,
    "contact_row_diagnostic_enabled": False,
  }
  for name, expected in expected_scalars.items():
    actual = protocol.get(name)
    if type(actual) is not type(expected) or actual != expected:
      raise ValueError(f"{label} {name} drifted from the exact protocol")


def _validate_fixed_population_protocol_pair(
  control: Mapping[str, object], target: Mapping[str, object]
) -> None:
  expected_rng = asdict(EvaluationRngSeeds.from_evaluation_seed(FIXED_SEED))
  for protocol in (control, target):
    if type(protocol.get("seed")) is not int or protocol.get("seed") != FIXED_SEED:
      raise ValueError("fixed population seed drifted from the exact protocol")
    if (
      type(protocol.get("num_envs")) is not int
      or protocol.get("num_envs") != FIXED_ENVS
    ):
      raise ValueError("fixed population num_envs drifted from the exact protocol")
    if protocol.get("initial_population_sha256") != EXPECTED_FIXED_POPULATION_SHA256:
      raise ValueError("fixed population hash drifted from the banked population")
    if protocol.get("policy_mode") != "mean":
      raise ValueError("fixed population policy mode must be mean")
    if protocol.get("auto_reset") is not False:
      raise ValueError("fixed population auto_reset must be false")
    if protocol.get("rng_streams") != expected_rng:
      raise ValueError("fixed population RNG streams drifted from the evaluation seed")
    control_steps = protocol.get("control_steps")
    if type(control_steps) is not int or control_steps <= 0:
      raise ValueError("fixed population control_steps must be a positive observed outcome")
    _validate_exact_measurement_protocol(protocol, label="fixed population")


def _validate_sampled_population_protocol_pair(
  control: Mapping[str, object], target: Mapping[str, object], *, seed: int
) -> None:
  expected_rng = asdict(EvaluationRngSeeds.from_evaluation_seed(seed))
  for protocol in (control, target):
    if type(protocol.get("seed")) is not int or protocol.get("seed") != seed:
      raise ValueError("sampled population seed drifted from its seed key")
    if (
      type(protocol.get("num_envs")) is not int
      or protocol.get("num_envs") != TRAINING_LIKE_ENVS
    ):
      raise ValueError("sampled population num_envs drifted from the exact protocol")
    if (
      type(protocol.get("control_steps")) is not int
      or protocol.get("control_steps") != TRAINING_LIKE_STEPS
    ):
      raise ValueError("sampled population control_steps must be exactly 24")
    if protocol.get("policy_mode") != "sampled":
      raise ValueError("sampled population policy mode must be sampled")
    if protocol.get("auto_reset") is not True:
      raise ValueError("sampled population auto_reset must be true")
    if protocol.get("rng_streams") != expected_rng:
      raise ValueError("sampled population RNG streams drifted from the evaluation seed")
    population_hash = protocol.get("initial_population_sha256")
    if not _is_lower_hex(population_hash, length=64):
      raise ValueError("sampled population hash must be full lowercase SHA-256")
    _validate_exact_measurement_protocol(protocol, label="sampled population")
  if control.get("initial_population_sha256") != target.get(
    "initial_population_sha256"
  ):
    raise ValueError("sampled populations must use the same initial population hash")


def compare_population_summaries(
  control: Mapping[str, object], target: Mapping[str, object]
) -> dict[str, object]:
  """Compare matched episode-segment summaries without using overlapping read counts."""
  if control.get("caps_n_m_s") != target.get("caps_n_m_s"):
    raise ValueError("paired summaries must use the same cap vector")
  control_segments, control_violations, control_risk = _segment_endpoint(control)
  target_segments, target_violations, target_risk = _segment_endpoint(target)
  if control_segments != target_segments:
    raise ValueError("paired summaries must contain the same number of episode segments")
  control_utilization = _utilization_quantiles(control)
  target_utilization = _utilization_quantiles(target)
  return {
    "statistical_unit": "paired episode segment",
    "controller_reads_are_independent": False,
    "paired_segments": control_segments,
    "any_joint_violation": {
      "control": {
        "violating_segments": control_violations,
        "risk": control_risk,
      },
      "target": {
        "violating_segments": target_violations,
        "risk": target_risk,
      },
      "target_minus_control_absolute_risk_difference": target_risk - control_risk,
      "target_over_control_risk_ratio": (
        target_risk / control_risk if control_risk > 0.0 else None
      ),
    },
    "max_joint_utilization_target_minus_control": {
      name: target_utilization[name] - control_utilization[name]
      for name in ("p50", "p95", "p99", "max")
    },
  }


def compare_policy_evaluations(
  control: Mapping[str, object],
  target: Mapping[str, object],
  *,
  expected_code_revision: str,
  expected_roles: tuple[str, str] = ("diag90_control", "diag90_target"),
) -> dict[str, object]:
  """Build pure comparisons bound to one explicitly approved evaluator revision."""
  if (
    not isinstance(expected_roles, tuple)
    or len(expected_roles) != 2
    or not all(isinstance(role, str) and role for role in expected_roles)
    or expected_roles[0] == expected_roles[1]
  ):
    raise ValueError("expected_roles must be an exact ordered pair of distinct role names")
  control_role, target_role = expected_roles
  control_checkpoint = control.get("checkpoint")
  target_checkpoint = target.get("checkpoint")
  if not isinstance(control_checkpoint, Mapping) or not isinstance(
    target_checkpoint, Mapping
  ):
    raise ValueError("policy summaries must record checkpoint roles")
  if control_checkpoint.get("role") != control_role or target_checkpoint.get(
    "role"
  ) != target_role:
    raise ValueError("policy summaries must be ordered control then target")
  for checkpoint, role in (
    (control_checkpoint, control_role),
    (target_checkpoint, target_role),
  ):
    expected_hash = EVALUATION_CHECKPOINTS.get(role)
    if expected_hash is None or checkpoint.get("sha256") != expected_hash:
      raise ValueError(f"{role} checkpoint SHA does not match the frozen role")
  if control.get("task") != VIC_TASK or target.get("task") != VIC_TASK:
    raise ValueError("paired policy summaries must record the exact task")
  control_code_revision = control.get("code_revision")
  target_code_revision = target.get("code_revision")
  if not _is_lower_hex(expected_code_revision, length=40):
    raise ValueError("expected evaluation code revision must be full lowercase 40-hex")
  if not _is_lower_hex(control_code_revision, length=40) or not _is_lower_hex(
    target_code_revision, length=40
  ):
    raise ValueError("paired policy summaries require a full lowercase code revision")
  if control_code_revision != target_code_revision:
    raise ValueError("paired policy summaries must use the same code revision")
  if control_code_revision != expected_code_revision:
    raise ValueError(
      "paired policy summaries must use the exact expected evaluation code revision"
    )
  if (
    control.get("asset_revision") != EXPECTED_EVALUATION_ASSET_REVISION
    or target.get("asset_revision") != EXPECTED_EVALUATION_ASSET_REVISION
  ):
    raise ValueError("paired policy summaries must use the frozen asset revision")
  if control.get("protocol") != target.get("protocol"):
    raise ValueError("paired policy summaries must use the same evaluation protocol")
  protocol = control.get("protocol")
  if not isinstance(protocol, Mapping) or protocol.get("stochastic_seeds") != list(
    POLICY_EVALUATION_STOCHASTIC_SEEDS
  ):
    raise ValueError("paired policy summaries must use the exact stochastic seed tuple")
  if protocol.get("live_imp_max_p") != 0.0:
    raise ValueError("paired policy summaries must record live imp_max_p=0")
  if (
    type(protocol.get("fixed_seed")) is not int
    or protocol.get("fixed_seed") != FIXED_SEED
  ):
    raise ValueError("paired policy summaries must record the exact fixed seed")
  if protocol.get("threshold_summary_order") != [
    "provisional_caps", "diagnostic_only"
  ]:
    raise ValueError("paired policy summaries have the wrong threshold summary order")
  if protocol.get("primary_statistical_unit") != "initial episode segment":
    raise ValueError("paired policy summaries have the wrong statistical unit")
  if protocol.get("controller_reads_are_independent") is not False:
    raise ValueError("paired policy summaries must declare controller reads non-independent")

  control_populations = control.get("populations")
  target_populations = target.get("populations")
  if not isinstance(control_populations, Mapping) or not isinstance(
    target_populations, Mapping
  ):
    raise ValueError("policy summaries are missing populations")
  threshold_order = ("provisional_caps", "diagnostic_only")

  def compare_population_pair(
    control_population: object,
    target_population: object,
    *,
    population_kind: str,
    seed: int | None = None,
  ) -> dict[str, object]:
    if not isinstance(control_population, Mapping) or not isinstance(
      target_population, Mapping
    ):
      raise ValueError("paired population payload is invalid")
    control_protocol = control_population.get("protocol")
    target_protocol = target_population.get("protocol")
    if not isinstance(control_protocol, Mapping) or not isinstance(
      target_protocol, Mapping
    ):
      raise ValueError("paired population protocol is invalid")
    if population_kind == "fixed":
      _validate_fixed_population_protocol_pair(control_protocol, target_protocol)
    else:
      if seed is None:
        raise RuntimeError("sampled comparison requires its declared seed")
      _validate_sampled_population_protocol_pair(
        control_protocol, target_protocol, seed=seed
      )
    result: dict[str, object] = {}
    for threshold, expected_caps in (
      ("provisional_caps", PROVISIONAL_CAPS_N_M_S),
      ("diagnostic_only", DIAGNOSTIC_LIMITS_N_M_S),
    ):
      control_summary = control_population.get(threshold)
      target_summary = target_population.get(threshold)
      if not isinstance(control_summary, Mapping) or not isinstance(
        target_summary, Mapping
      ):
        raise ValueError(f"paired populations are missing {threshold}")
      if control_summary.get("caps_n_m_s") != list(
        expected_caps
      ) or target_summary.get("caps_n_m_s") != list(expected_caps):
        raise ValueError(f"{threshold} must use its exact cap vector")
      result[threshold] = compare_population_summaries(
        control_summary, target_summary
      )
    return result

  fixed = compare_population_pair(
    control_populations.get("fixed_mean"),
    target_populations.get("fixed_mean"),
    population_kind="fixed",
  )
  control_stochastic = control_populations.get("training_like_sampled")
  target_stochastic = target_populations.get("training_like_sampled")
  if not isinstance(control_stochastic, Mapping) or not isinstance(
    target_stochastic, Mapping
  ):
    raise ValueError("policy summaries are missing stochastic populations")
  expected_seed_keys = tuple(str(seed) for seed in POLICY_EVALUATION_STOCHASTIC_SEEDS)
  if (
    tuple(control_stochastic) != expected_seed_keys
    or tuple(target_stochastic) != expected_seed_keys
  ):
    raise ValueError("policy summaries must contain the exact seed-keyed populations")
  stochastic = {
    seed: compare_population_pair(
      control_stochastic[seed],
      target_stochastic[seed],
      population_kind="sampled",
      seed=int(seed),
    )
    for seed in expected_seed_keys
  }
  return {
    "roles": {"control": control_role, "target": target_role},
    "threshold_order": list(threshold_order),
    "statistical_unit": "paired episode segment",
    "controller_reads_are_independent": False,
    "fixed_mean": fixed,
    "training_like_sampled": stochastic,
  }


def run_policy_evaluation(
  checkpoint: Path,
  role: str,
  output_dir: Path,
  device: str,
  stochastic_seeds: tuple[int, ...],
  training_caps_n_m_s: object | None = None,
) -> dict[str, object]:
  """Run one frozen role on the prespecified matched post-training populations."""
  seeds = tuple(stochastic_seeds)
  if seeds != POLICY_EVALUATION_STOCHASTIC_SEEDS:
    raise ValueError(
      "post-training evaluation requires the exact stochastic seed tuple "
      f"{POLICY_EVALUATION_STOCHASTIC_SEEDS}"
    )

  checkpoint = checkpoint.resolve(strict=True)
  output_dir = output_dir.absolute()
  if output_dir.exists() or output_dir.is_symlink():
    raise FileExistsError(
      f"refusing to overwrite policy evaluation output: {output_dir}"
    )
  checkpoint_sha256 = validate_checkpoint_role(checkpoint, role)
  training_cap_identity = validate_training_cap_identity(role, training_caps_n_m_s)
  revisions = _evaluation_revisions()
  revisions["code_revision"] = validate_evaluation_revision(
    _REPO_ROOT, revisions["code_revision"]
  )
  output_dir.mkdir(parents=True)

  fixed_trace, fixed_protocol = _run_population(
    checkpoint=checkpoint,
    device=device,
    num_envs=FIXED_ENVS,
    seed=FIXED_SEED,
    rng_seeds=EvaluationRngSeeds.from_evaluation_seed(FIXED_SEED),
    steps=None,
    stochastic=False,
  )
  _require_live_impulse_log_only(fixed_protocol)
  if fixed_protocol["initial_population_sha256"] != EXPECTED_FIXED_POPULATION_SHA256:
    raise RuntimeError("fixed evaluation initial population differs from the banked evaluator")
  np.savez_compressed(output_dir / "fixed_trace.npz", **fixed_trace)
  fixed_payload = _policy_population_payload(
    fixed_trace, fixed_protocol, stochastic=False
  )

  stochastic_payloads: dict[str, object] = {}
  for seed in seeds:
    trace, population_protocol = _run_population(
      checkpoint=checkpoint,
      device=device,
      num_envs=TRAINING_LIKE_ENVS,
      seed=seed,
      rng_seeds=EvaluationRngSeeds.from_evaluation_seed(seed),
      steps=TRAINING_LIKE_STEPS,
      stochastic=True,
    )
    _require_live_impulse_log_only(population_protocol)
    trace_name = f"training_like_seed_{seed}_trace.npz"
    np.savez_compressed(output_dir / trace_name, **trace)
    stochastic_payloads[str(seed)] = {
      "trace": trace_name,
      **_policy_population_payload(
        trace, population_protocol, stochastic=True
      ),
    }

  payload: dict[str, object] = {
    "schema_version": 2,
    "purpose": "matched no-learning comparison of frozen diagnostic impulse-CaT policies",
    "task": VIC_TASK,
    "checkpoint": {
      "role": role,
      "path": str(checkpoint),
      "sha256": checkpoint_sha256,
    },
    **revisions,
    "protocol": {
      "live_imp_max_p": 0.0,
      "fixed_seed": FIXED_SEED,
      "stochastic_seeds": list(seeds),
      "threshold_summary_order": ["provisional_caps", "diagnostic_only"],
      "primary_statistical_unit": "initial episode segment",
      "controller_reads_are_independent": False,
    },
    "thresholds": {
      "provisional_project_caps_n_m_s": list(PROVISIONAL_CAPS_N_M_S),
      "diagnostic_only_caps_n_m_s": list(DIAGNOSTIC_LIMITS_N_M_S),
      "hardware_limit_claimed": False,
    },
    "populations": {
      "fixed_mean": {"trace": "fixed_trace.npz", **fixed_payload},
      "training_like_sampled": stochastic_payloads,
    },
  }
  if training_cap_identity is not None:
    payload["training_cap_identity_n_m_s"] = list(training_cap_identity)
  encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
  (output_dir / "summary.json").write_text(encoded)
  return payload


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-role", choices=tuple(EVALUATION_CHECKPOINTS))
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--training-caps-n-m-s", type=json.loads)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.checkpoint_role is None:
    payload = run_survey(
      checkpoint=args.checkpoint,
      output_dir=args.output_dir,
      device=args.device,
    )
    for name, population in payload["populations"].items():
      provisional = population["provisional_caps"]
      print(
        f"{name}: provisional-cap reads={provisional['binding']['violating_reads']} "
        f"physical_events={provisional['binding']['associated_physical_events']}"
      )
  else:
    run_policy_evaluation(
      checkpoint=args.checkpoint,
      role=args.checkpoint_role,
      output_dir=args.output_dir,
      device=args.device,
      stochastic_seeds=POLICY_EVALUATION_STOCHASTIC_SEEDS,
      training_caps_n_m_s=args.training_caps_n_m_s,
    )
  print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
  main()
