#!/usr/bin/env python3
"""Lean frozen-policy evaluation for the five Z1 diagonal-start specialists."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from scripts import impulse_cat_activation_survey as survey


_TASK_PREFIX = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
  "JointPosition-VariableImpedance-TT-DiagonalFixedStart"
)
TRAINING_CAPS_N_M_S = (0.369, 0.246, 0.738, 0.369, 0.246, 0.0164)
TRAINING_CODE_REVISION = "b17e7fafbd498bb2a7f9f375a82be5872cf3e9f3"
EXPECTED_ASSET_REVISION = survey.EXPECTED_EVALUATION_ASSET_REVISION
POLICY_SPECS: Mapping[str, Mapping[str, object]] = MappingProxyType(
  {
    "fixed_m40mm_p02": MappingProxyType(
      {
        "task": f"{_TASK_PREFIX}M40mm-Persistent",
        "offset_mm": -40,
        "route_sign": -1,
        "joint_pose_rad": (0.0, 1.487947605122, -0.319221074008, -1.190426531114, -0.0013, 1.5544),
        "sha256": "a429523ded414e569783166677ee7d3725f3d1ef542d9792eab7733a1da07504",
      }
    ),
    "fixed_m20mm_p02": MappingProxyType(
      {
        "task": f"{_TASK_PREFIX}M20mm-Persistent",
        "offset_mm": -20,
        "route_sign": -1,
        "joint_pose_rad": (0.0, 1.549360913, -0.372460482, -1.198600431, -0.0013, 1.5544),
        "sha256": "1155bb102a080040b68ce483d40b7caf78b5049dffb596f1b813223597fbcd64",
      }
    ),
    "fixed_0mm_p02": MappingProxyType(
      {
        "task": f"{_TASK_PREFIX}0mm-Persistent",
        "offset_mm": 0,
        "route_sign": 0,
        "joint_pose_rad": (0.0, 1.606, -0.4301, -1.1976, -0.0013, 1.5544),
        "sha256": "2fa4dafd9a5291768ad459a368b8805bafa9571662a3a6876f4a3f2ded21f9e6",
      }
    ),
    "fixed_p20mm_p02": MappingProxyType(
      {
        "task": f"{_TASK_PREFIX}P20mm-Persistent",
        "offset_mm": 20,
        "route_sign": 1,
        "joint_pose_rad": (0.0, 1.658994499, -0.491428010, -1.189266489, -0.0013, 1.5544),
        "sha256": "f803f97f43a916c8e9af9c5756f46424d94dc686cd34153024443615940f17a4",
      }
    ),
    "fixed_p40mm_p02": MappingProxyType(
      {
        "task": f"{_TASK_PREFIX}P40mm-Persistent",
        "offset_mm": 40,
        "route_sign": 1,
        "joint_pose_rad": (0.0, 1.709237021603, -0.555950051788, -1.174986969814, -0.0013, 1.5544),
        "sha256": "c6b3b0b9267345efc0b316f20d708b09edb2ae49b2a86fc5f0d984f542161d00",
      }
    ),
  }
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def validate_checkpoint(checkpoint: Path, *, role: str) -> tuple[Path, str]:
  if role not in POLICY_SPECS:
    raise ValueError("unknown fixed-start specialist role")
  checkpoint = Path(checkpoint)
  if checkpoint.name != "model_499.pt" or checkpoint.is_symlink() or not checkpoint.is_file():
    raise ValueError("checkpoint must be a regular non-symlink model_499.pt")
  resolved = checkpoint.resolve(strict=True)
  digest = _sha256(resolved)
  if digest != POLICY_SPECS[role]["sha256"]:
    raise RuntimeError("fixed-start role/checkpoint SHA-256 mismatch")
  return resolved, digest


def validate_specialist_config(env_cfg, qualified_env_cfg, *, role: str) -> None:
  """Fail closed on the fixed start and straight-reference training treatment."""
  if role not in POLICY_SPECS:
    raise ValueError("unknown fixed-start specialist role")
  spec = POLICY_SPECS[role]
  sign = int(spec["route_sign"])
  from src.tasks.hammer.mdp.references import (
    reset_joints_to_strike_route_starts,
    sample_strike_route_signs,
  )

  sampler = env_cfg.events["sample_strike_route_signs"]
  if sampler.func is not sample_strike_route_signs:
    raise RuntimeError("fixed-start sampler callable drifted")
  if sampler.params != {
    "horizontal_detour_m": 0.0,
    "followthrough_mode": "strike_axis",
    "fixed_route_sign": sign,
  }:
    raise RuntimeError("fixed-start sampler treatment drifted")
  route_reset = env_cfg.events["reset_strike_route_joints"]
  if route_reset.func is not reset_joints_to_strike_route_starts:
    raise RuntimeError("fixed-start reset callable drifted")
  rows = tuple(tuple(row) for row in route_reset.params["route_joint_positions"])
  if len(rows) != 3 or rows[sign + 1] != tuple(spec["joint_pose_rad"]):
    raise RuntimeError("fixed-start reset pose drifted")
  imitation = env_cfg.rewards["r_imit"]
  if (
    imitation.weight != 0.2
    or imitation.params.get("sigma") != 0.05
    or imitation.params.get("followthrough_mode") != "strike_axis"
    or imitation.params.get("horizontal_detour_m", 0.0) != 0.0
    or "r_imit_anneal" in env_cfg.curriculum
  ):
    raise RuntimeError("fixed-start imitation treatment drifted")
  for group in env_cfg.observations.values():
    for term_name in ("strike_phase", "strike_ref_error"):
      params = group.terms[term_name].params
      if params.get("followthrough_mode") != "strike_axis":
        raise RuntimeError("fixed-start observation reference drifted")
  if tuple(env_cfg.actions) != ("joint_position", "joint_stiffness"):
    raise RuntimeError("fixed-start action layout drifted")
  for action_name in env_cfg.actions:
    action = env_cfg.actions[action_name]
    qualified_action = qualified_env_cfg.actions[action_name]
    if type(action) is not type(qualified_action) or vars(action) != vars(qualified_action):
      raise RuntimeError(f"fixed-start {action_name} mapping drifted")
  default_pose = env_cfg.scene.entities["robot"].init_state.joint_pos
  qualified_default = qualified_env_cfg.scene.entities["robot"].init_state.joint_pos
  if default_pose != qualified_default:
    raise RuntimeError("fixed-start robot default pose drifted")


def _quantiles(values: list[float]) -> dict[str, float | None]:
  if not values:
    return {"median": None, "p95": None, "max": None}
  array = np.asarray(values, dtype=np.float64)
  return {
    "median": float(np.median(array)),
    "p95": float(np.quantile(array, 0.95)),
    "max": float(np.max(array)),
  }


def straight_tracking_summary(
  trace: Mapping[str, np.ndarray], *, expected_sign: int
) -> dict[str, object]:
  """Summarize straight-guide error with one equal-weight unit per initial episode."""
  required = {
    "episode_id",
    "route_sign",
    "strike_phase",
    "imitation_eligible",
    "hammer_head_pos_w",
    "assigned_reference_waypoint_w",
    "straight_reference_waypoint_w",
  }
  if not required.issubset(trace):
    raise ValueError("fixed-start trace is missing reference telemetry")
  episode_id = np.asarray(trace["episode_id"])
  signs = np.asarray(trace["route_sign"])
  phase = np.asarray(trace["strike_phase"])
  eligible = np.asarray(trace["imitation_eligible"])
  head = np.asarray(trace["hammer_head_pos_w"], dtype=np.float64)
  assigned = np.asarray(trace["assigned_reference_waypoint_w"], dtype=np.float64)
  straight = np.asarray(trace["straight_reference_waypoint_w"], dtype=np.float64)
  scalar_shape = episode_id.shape
  if len(scalar_shape) != 2:
    raise ValueError("fixed-start episode telemetry must have shape (steps, envs)")
  if signs.shape != scalar_shape or not bool((signs == expected_sign).all()):
    raise ValueError("fixed-start trace sign does not match its specialist")
  if phase.shape != scalar_shape or eligible.shape != scalar_shape or eligible.dtype != np.bool_:
    raise ValueError("fixed-start phase or imitation eligibility shape drifted")
  vector_shape = (*scalar_shape, 3)
  if head.shape != vector_shape or assigned.shape != vector_shape or straight.shape != vector_shape:
    raise ValueError("fixed-start Cartesian telemetry shape drifted")
  if not np.isfinite(phase).all() or not np.isfinite(head).all() or not np.isfinite(assigned).all():
    raise ValueError("fixed-start reference telemetry must be finite")
  if not np.array_equal(assigned, straight):
    raise ValueError("fixed-start specialist reference is not a straight guide")

  initial = episode_id == 0
  error = np.linalg.norm(head - assigned, axis=2)
  episode_rmse: list[float] = []
  within_10mm: list[float] = []
  max_phase_all: list[float] = []
  no_progress = 0
  for env_id in range(scalar_shape[1]):
    initial_eligible = initial[:, env_id] & eligible[:, env_id]
    max_phase_all.append(
      float(np.max(phase[initial_eligible, env_id]))
      if bool(initial_eligible.any())
      else 0.0
    )
    progressed = initial_eligible & (phase[:, env_id] >= 0.05)
    if not bool(progressed.any()):
      no_progress += 1
      continue
    values = error[progressed, env_id]
    episode_rmse.append(float(np.sqrt(np.mean(values * values))))
    within_10mm.append(float(np.mean(values <= 0.010)))
  return {
    "population_episodes": int(scalar_shape[1]),
    "eligible_episodes": len(episode_rmse),
    "no_progress_episodes": no_progress,
    "progress_coverage_rate": len(episode_rmse) / int(scalar_shape[1]),
    "eligibility_rule": "initial episode, pre-contact imitation gate, phase >= 0.05",
    "conditional_progress_aggregate_equal_episode_rmse_m": (
      float(np.sqrt(np.mean(np.square(episode_rmse)))) if episode_rmse else None
    ),
    "conditional_progress_episode_rmse_m": _quantiles(episode_rmse),
    "conditional_progress_within_10mm_fraction_episode_mean": (
      float(np.mean(within_10mm)) if within_10mm else None
    ),
    "all_episode_max_precontact_phase": _quantiles(max_phase_all),
  }


def _first_strike_summary(trace: Mapping[str, np.ndarray]) -> dict[str, object]:
  episode_id = np.asarray(trace["episode_id"])
  done = np.asarray(trace["done"], dtype=bool)
  started = np.asarray(trace["first_strike_started"], dtype=bool)
  finalized = np.asarray(trace["first_strike_finalized"], dtype=bool)
  productive = np.asarray(trace["first_strike_productive"], dtype=bool)
  delivered = np.asarray(trace["first_strike_delivered_n_s"], dtype=np.float64)
  speed = np.asarray(
    trace["first_strike_precontact_nail_axial_velocity_m_s"], dtype=np.float64
  )
  if (
    speed.shape != episode_id.shape
    or done.shape != episode_id.shape
    or started.shape != episode_id.shape
    or finalized.shape != episode_id.shape
    or productive.shape != episode_id.shape
    or delivered.shape != episode_id.shape
    or not np.isfinite(speed).all()
    or not np.isfinite(delivered).all()
  ):
    raise ValueError("first-strike outcome telemetry shape drifted")
  terminal = done & (episode_id == 0)
  accepted = terminal & started
  productive_terminal = terminal & productive
  population = int(episode_id.shape[1])
  terminals = int(terminal.sum())
  return {
    "population_fragments": population,
    "terminal_episodes": terminals,
    "terminal_episode_coverage": terminals / population,
    "accepted_strikes": int(accepted.sum()),
    "finalized_strikes": int((terminal & finalized).sum()),
    "productive_strikes": int(productive_terminal.sum()),
    "accepted_strike_rate_among_terminal_episodes": (
      float(accepted.sum() / terminal.sum()) if bool(terminal.any()) else None
    ),
    "productive_strike_rate_among_terminal_episodes": (
      float(productive_terminal.sum() / terminal.sum())
      if bool(terminal.any())
      else None
    ),
    "accepted_strike_fraction_of_population": float(accepted.sum() / population),
    "productive_strike_fraction_of_population": float(
      productive_terminal.sum() / population
    ),
    "unconditional_first_strike_delivered_n_s": _quantiles(
      delivered[terminal].tolist()
    ),
    "accepted_strike_delivered_n_s": _quantiles(delivered[accepted].tolist()),
    "productive_strike_delivered_n_s": _quantiles(
      delivered[productive_terminal].tolist()
    ),
    "accepted_strike_precontact_nail_axial_velocity_m_s": _quantiles(
      speed[accepted].tolist()
    ),
  }


def _population_summary(
  trace: dict[str, np.ndarray],
  protocol: dict[str, object],
  *,
  expected_sign: int,
  expected_task: str,
) -> dict[str, object]:
  if protocol.get("route_telemetry") is not True:
    raise RuntimeError("fixed-start evaluation did not enable reference telemetry")
  if protocol.get("forced_route_sign") is not None:
    raise RuntimeError("fixed-start evaluation must use the task's pre-reset specialist sampler")
  if protocol.get("actor_only_checkpoint_load") is not True:
    raise RuntimeError("fixed-start evaluation must load only the frozen actor")
  if protocol.get("task") != expected_task:
    raise RuntimeError("fixed-start evaluation task identity drifted")
  if protocol.get("training_config") != {
    "imp_max_p": 0.2,
    "imp_limit_n_m_s": list(TRAINING_CAPS_N_M_S),
  }:
    raise RuntimeError("fixed-start evaluation training treatment identity drifted")
  cat_replay = protocol.get("cat_replay")
  if not isinstance(cat_replay, Mapping) or cat_replay.get("imp_max_p_live") != 0.0:
    raise RuntimeError("fixed-start evaluation did not keep impulse CaT log-only")
  task_and_constraints = survey.summarize_population(
    trace, caps=TRAINING_CAPS_N_M_S, first_episode_only=True
  )
  utility = task_and_constraints["utility"]
  utility["task_field_scope"] = (
    "initial-episode portions within censored 24-step fragments; terminal counts "
    "include only observed terminals"
    if protocol.get("auto_reset") is True
    else "complete first episodes"
  )
  return {
    "protocol": protocol,
    "tracking": straight_tracking_summary(trace, expected_sign=expected_sign),
    "task_and_constraints": task_and_constraints,
    "first_strike_outcomes": _first_strike_summary(trace),
  }


def run_fixed_start_evaluation(
  *, checkpoint: Path, role: str, output_dir: Path, device: str
) -> dict[str, object]:
  """Evaluate one specialist on its own fixed physical start with four frozen populations."""
  checkpoint, checkpoint_sha = validate_checkpoint(checkpoint, role=role)
  spec = POLICY_SPECS[role]
  output_dir = Path(output_dir).absolute()
  if output_dir.exists() or output_dir.is_symlink():
    raise FileExistsError(f"refusing to overwrite fixed-start evaluation output: {output_dir}")
  revisions = survey._evaluation_revisions()
  revisions["code_revision"] = survey.validate_evaluation_revision(
    survey._REPO_ROOT,
    revisions["code_revision"],
    approved_base_revision=TRAINING_CODE_REVISION,
  )
  if revisions.get("asset_revision") != EXPECTED_ASSET_REVISION:
    raise RuntimeError("fixed-start evaluation asset revision mismatch")
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  specialist_cfg = load_env_cfg(str(spec["task"]), play=False)
  qualified_cfg = load_env_cfg(survey.VIC_TASK, play=False)
  validate_specialist_config(specialist_cfg, qualified_cfg, role=role)
  output_dir.mkdir(parents=True)

  def run_population(*, seed: int, num_envs: int, steps: int | None, stochastic: bool):
    trace, protocol = survey._run_population(
      checkpoint=checkpoint,
      device=device,
      num_envs=num_envs,
      seed=seed,
      rng_seeds=survey.EvaluationRngSeeds.from_evaluation_seed(seed),
      steps=steps,
      stochastic=stochastic,
      task_id=str(spec["task"]),
      source_imp_max_p=0.2,
      source_imp_limit_n_m_s=TRAINING_CAPS_N_M_S,
      forced_route_sign=None,
      route_telemetry=True,
    )
    return trace, _population_summary(
      trace,
      protocol,
      expected_sign=int(spec["route_sign"]),
      expected_task=str(spec["task"]),
    )

  fixed_trace, fixed_summary = run_population(
    seed=survey.FIXED_SEED,
    num_envs=survey.FIXED_ENVS,
    steps=None,
    stochastic=False,
  )
  fixed_name = "fixed_mean_trace.npz"
  np.savez_compressed(output_dir / fixed_name, **fixed_trace)
  sampled: dict[str, object] = {}
  for seed in survey.POLICY_EVALUATION_STOCHASTIC_SEEDS:
    trace, summary = run_population(
      seed=seed,
      num_envs=survey.TRAINING_LIKE_ENVS,
      steps=survey.TRAINING_LIKE_STEPS,
      stochastic=True,
    )
    name = f"training_like_seed_{seed}_trace.npz"
    np.savez_compressed(output_dir / name, **trace)
    sampled[str(seed)] = {
      "trace": name,
      "trace_sha256": _sha256(output_dir / name),
      **summary,
    }

  payload: dict[str, object] = {
    "schema_version": 1,
    "purpose": "frozen actor-only own-start evaluation of one fixed-start specialist",
    "policy": {
      "role": role,
      "task": spec["task"],
      "offset_mm": spec["offset_mm"],
      "route_sign": spec["route_sign"],
      "checkpoint": str(checkpoint),
      "checkpoint_sha256": checkpoint_sha,
    },
    **revisions,
    "training_treatment": {
      "seed": 2,
      "iterations": 500,
      "imp_max_p": 0.2,
      "imp_limit_n_m_s": list(TRAINING_CAPS_N_M_S),
      "r_imit_weight": 0.2,
      "r_imit_sigma_m": 0.05,
      "r_imit_schedule": "persistent",
    },
    "evaluation_protocol": {
      "actor_only": True,
      "live_imp_max_p": 0.0,
      "fixed_mean": [survey.FIXED_ENVS, "complete first episode"],
      "sampled_seeds": list(survey.POLICY_EVALUATION_STOCHASTIC_SEEDS),
      "sampled_shape": [survey.TRAINING_LIKE_ENVS, survey.TRAINING_LIKE_STEPS],
      "sampled_scope": "censored 24-step robustness fragments, not complete episodes",
      "tracking_unit": "initial episode with phase >= 0.05 while imitation-eligible",
    },
    "populations": {
      "fixed_mean": {
        "trace": fixed_name,
        "trace_sha256": _sha256(output_dir / fixed_name),
        **fixed_summary,
      },
      "training_like_sampled": sampled,
    },
  }
  (output_dir / "summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
  )
  return payload


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-role", required=True, choices=tuple(POLICY_SPECS))
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--device", default="cpu")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  run_fixed_start_evaluation(
    checkpoint=args.checkpoint,
    role=args.checkpoint_role,
    output_dir=args.output_dir,
    device=args.device,
  )
  print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
  main()
