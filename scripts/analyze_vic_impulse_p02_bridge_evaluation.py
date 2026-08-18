#!/usr/bin/env python3
"""Pre-registered native-horizon analysis for the Z1 impulse-CaT p=0.2 bridge."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import numpy as np

from scripts import analyze_vic_impulse_diag90_500_evaluation as historical
from scripts import impulse_cat_activation_survey as survey


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026081703
STOCHASTIC_SEEDS = survey.POLICY_EVALUATION_STOCHASTIC_SEEDS
EXPECTED_ROLES = ("bridge_p0_control", "bridge_p02_target")


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
  return historical._quantiles(np.asarray(values, dtype=np.float64))


def paired_mean_ratio_bootstrap(
  control: np.ndarray,
  target: np.ndarray,
  *,
  resamples: int = BOOTSTRAP_RESAMPLES,
  seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
  """Bootstrap a ratio of paired environment-level means without dropping invalid draws."""
  control_values = np.asarray(control, dtype=np.float64)
  target_values = np.asarray(target, dtype=np.float64)
  if (
    control_values.ndim != 1
    or target_values.shape != control_values.shape
    or control_values.size == 0
    or not np.isfinite(control_values).all()
    or not np.isfinite(target_values).all()
    or type(resamples) is not int
    or resamples <= 0
    or type(seed) is not int
    or seed < 0
  ):
    raise ValueError("paired means require aligned finite vectors and a valid bootstrap protocol")

  observed_control = float(control_values.mean())
  observed_target = float(target_values.mean())
  base = {
    "resampling_unit": "whole paired environment ID",
    "controller_reads_are_inferential_units": False,
    "environments": int(control_values.size),
    "resamples": resamples,
    "seed": seed,
    "point_target_over_control_mean_ratio": (
      observed_target / observed_control if observed_control > 0.0 else None
    ),
  }
  if observed_control <= 0.0:
    return {
      **base,
      "valid": False,
      "lower_97_5": None,
      "invalid_nonpositive_control_resamples": resamples,
      "failure_reason": "observed control mean is nonpositive",
    }

  ratios = np.empty(resamples, dtype=np.float64)
  invalid = 0
  rng = np.random.default_rng(seed)
  offset = 0
  while offset < resamples:
    count = min(256, resamples - offset)
    indices = rng.integers(0, control_values.size, size=(count, control_values.size))
    control_means = control_values[indices].mean(axis=1)
    target_means = target_values[indices].mean(axis=1)
    invalid_mask = control_means <= 0.0
    invalid += int(invalid_mask.sum())
    batch_ratios = np.full(count, np.nan, dtype=np.float64)
    np.divide(
      target_means,
      control_means,
      out=batch_ratios,
      where=~invalid_mask,
    )
    ratios[offset : offset + count] = batch_ratios
    offset += count
  if invalid:
    return {
      **base,
      "valid": False,
      "lower_97_5": None,
      "invalid_nonpositive_control_resamples": invalid,
      "failure_reason": "at least one paired resample had a nonpositive control mean",
    }
  return {
    **base,
    "valid": True,
    "lower_97_5": float(np.quantile(ratios, 0.025)),
    "invalid_nonpositive_control_resamples": 0,
    "failure_reason": None,
  }


def evaluate_gate_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
  """Apply all six bridge gates to one stochastic population, never to pooled reads."""
  rho = metrics["rho"]
  per_joint = metrics["per_joint"]
  if not isinstance(rho, Mapping) or not isinstance(per_joint, Mapping):
    raise ValueError("bridge gate metrics are missing rho or per-joint endpoints")
  control_rho = rho["control"]
  target_rho = rho["target"]
  if not isinstance(control_rho, Mapping) or not isinstance(target_rho, Mapping):
    raise ValueError("bridge gate rho endpoints are invalid")
  if any(float(control_rho[name]) <= 0.0 for name in ("p95", "p99")):
    raise ValueError("control rho p95 and p99 must be finite and positive")
  rho_reductions = {
    quantile: (float(control_rho[quantile]) - float(target_rho[quantile]))
    / float(control_rho[quantile])
    for quantile in ("p95", "p99")
  }
  if any(not np.isfinite(value) for value in rho_reductions.values()):
    raise ValueError("control rho p95 and p99 must be finite and positive")

  control_joint = np.asarray(per_joint["control_p99"], dtype=np.float64)
  target_joint = np.asarray(per_joint["target_p99"], dtype=np.float64)
  control_violates = np.asarray(per_joint["control_any_violation"], dtype=bool)
  target_violates = np.asarray(per_joint["target_any_violation"], dtype=bool)
  if not (
    control_joint.shape == target_joint.shape == control_violates.shape == target_violates.shape
    == (len(survey.JOINT_NAMES),)
  ) or not np.isfinite(control_joint).all() or not np.isfinite(target_joint).all():
    raise ValueError("per-joint bridge endpoints must be finite six-vectors")
  considered = control_joint >= 0.10
  relative_joint_increase = np.full(control_joint.shape, None, dtype=object)
  for joint in np.flatnonzero(considered):
    relative_joint_increase[joint] = float(
      (target_joint[joint] - control_joint[joint]) / control_joint[joint]
    )
  joint_tail_pass = all(
    float(relative_joint_increase[joint]) < 0.10
    for joint in np.flatnonzero(considered)
  )
  newly_violating = (~control_violates) & target_violates

  risk_upper = float(metrics["provisional_risk_difference_upper_97_5"])
  velocity_upper = float(metrics["velocity_risk_difference_upper_97_5"])
  success_raw = metrics["success_difference"]
  productive_raw = metrics["productive_strike_difference"]
  success_difference = None if success_raw is None else float(success_raw)
  productive_difference = None if productive_raw is None else float(productive_raw)
  delivered_lower_raw = metrics["first_event_delivered_ratio_lower_97_5"]
  delivered_lower = None if delivered_lower_raw is None else float(delivered_lower_raw)
  finite_scalars = (risk_upper, velocity_upper)
  if not np.isfinite(finite_scalars).all() or (
    success_difference is not None and not np.isfinite(success_difference)
  ) or (
    productive_difference is not None and not np.isfinite(productive_difference)
  ) or (
    delivered_lower is not None and not np.isfinite(delivered_lower)
  ):
    raise ValueError("bridge gate endpoints must be finite")

  gates = {
    "provisional_risk_reduction": {
      "pass": risk_upper < -0.005,
      "upper_97_5": risk_upper,
      "comparison": "<",
      "threshold": -0.005,
    },
    "global_rho_reduction": {
      "pass": all(value >= 0.10 for value in rho_reductions.values()),
      "relative_reduction": rho_reductions,
      "comparison": ">=",
      "threshold": 0.10,
    },
    "joint_tail_noninferiority": {
      "pass": joint_tail_pass and not bool(newly_violating.any()),
      "control_p99_at_least_0_10": considered.tolist(),
      "relative_p99_increase": relative_joint_increase.tolist(),
      "newly_violating_joint": newly_violating.tolist(),
      "comparison": "<",
      "threshold": 0.10,
    },
    "velocity_noninferiority": {
      "pass": velocity_upper <= 0.001,
      "upper_97_5": velocity_upper,
      "comparison": "<=",
      "threshold": 0.001,
    },
    "utility_noninferiority": {
      "pass": (
        success_difference is not None
        and productive_difference is not None
        and success_difference >= -0.01
        and productive_difference >= -0.01
      ),
      "success_difference": success_difference,
      "productive_strike_difference": productive_difference,
      "comparison": ">=",
      "threshold": -0.01,
    },
    "delivered_impulse_retention": {
      "pass": delivered_lower is not None and delivered_lower >= 0.90,
      "lower_97_5": delivered_lower,
      "comparison": ">=",
      "threshold": 0.90,
    },
    "identity_finiteness_native_claims": {
      "pass": metrics.get("identity_finiteness_native_claims") is True,
    },
  }
  return {"pass": all(gate["pass"] for gate in gates.values()), **gates}


def _initial_joint_utilization(
  trace: Mapping[str, np.ndarray], caps: Sequence[float]
) -> np.ndarray:
  lam = np.asarray(trace["lambda_per_joint"], dtype=np.float64)
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  cap_array = np.asarray(caps, dtype=np.float64)
  valid = episode_id == 0
  if (
    lam.ndim != 3
    or episode_id.shape != lam.shape[:2]
    or cap_array.shape != (lam.shape[2],)
    or not np.isfinite(lam).all()
    or not np.isfinite(cap_array).all()
    or bool((cap_array <= 0.0).any())
    or not bool(valid.any(axis=0).all())
  ):
    raise ValueError("trace lacks finite per-environment/per-joint initial utilization")
  return np.where(valid[:, :, None], lam / cap_array, -np.inf).max(axis=0)


def _initial_joint_violation_flags(
  trace: Mapping[str, np.ndarray], caps: Sequence[float]
) -> np.ndarray:
  """Classify violations from Lambda-cap subtraction in Lambda's native dtype."""
  lam = np.asarray(trace["lambda_per_joint"])
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  if lam.ndim != 3 or episode_id.shape != lam.shape[:2]:
    raise ValueError("trace lacks aligned initial-episode Lambda reads")
  valid = episode_id == 0
  margins = survey._native_cap_margins(lam, tuple(caps))
  return np.where(valid[:, :, None], margins, -np.inf).max(axis=0) > 0.0


def _terminal_endpoint_by_environment(
  trace: Mapping[str, np.ndarray], name: str
) -> dict[str, np.ndarray | int]:
  """Extract one initial-episode terminal value per completed environment, in env-ID order."""
  done = np.asarray(trace["done"], dtype=bool)
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  values = np.asarray(trace[name])
  terminal = done & (episode_id == 0)
  if values.shape != terminal.shape:
    raise ValueError(f"terminal field {name} is not an aligned scalar trace")
  counts = terminal.sum(axis=0)
  if bool((counts > 1).any()):
    raise ValueError(f"initial episode terminates more than once for {name}")
  completed = counts == 1
  env_ids = np.flatnonzero(completed)
  selected = np.asarray(
    [values[np.flatnonzero(terminal[:, env_id])[0], env_id] for env_id in env_ids],
    dtype=values.dtype,
  )
  return {
    "env_ids": env_ids,
    "values": selected,
    "completed_environments": int(completed.sum()),
    "right_censored_environments": int((~completed).sum()),
  }


def _binary_difference(control: np.ndarray, target: np.ndarray) -> float:
  control_flags = np.asarray(control)
  target_flags = np.asarray(target)
  if control_flags.dtype != np.bool_ or target_flags.shape != control_flags.shape:
    raise ValueError("paired utility endpoints must be aligned boolean vectors")
  return float(target_flags.mean() - control_flags.mean())


def _observed_duration(trace: Mapping[str, np.ndarray]) -> dict[str, object]:
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  done = np.asarray(trace["done"], dtype=bool)
  valid = episode_id == 0
  if done.shape != valid.shape:
    raise ValueError("done and episode_id must be aligned")
  reads = valid.sum(axis=0)
  terminated = (done & valid).sum(axis=0) == 1
  return {
    "unit": "native observed initial-episode prefix",
    "duration_ms": _quantiles(reads.astype(np.float64) * 20.0),
    "completed_environments": int(terminated.sum()),
    "right_censored_environments": int((~terminated).sum()),
  }


def _observed_first_contact_prefix_lambda(
  trace: Mapping[str, np.ndarray]
) -> dict[str, object]:
  lam = np.asarray(trace["lambda_per_joint"], dtype=np.float64)
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  contact = np.asarray(trace["substep_contact"], dtype=bool)
  sub_episode = np.asarray(trace["substep_episode_id"], dtype=np.int64)
  if contact.shape != sub_episode.shape or contact.shape[1] != lam.shape[1]:
    raise ValueError("contact and Lambda traces must be aligned")
  decimation = contact.shape[0] // lam.shape[0]
  if decimation <= 0 or decimation * lam.shape[0] != contact.shape[0]:
    raise ValueError("contact trace must contain an integer substep decimation")
  peaks: list[np.ndarray] = []
  for env_id in range(lam.shape[1]):
    indices = np.flatnonzero(contact[:, env_id] & (sub_episode[:, env_id] == 0))
    if not indices.size:
      continue
    end = int(indices[0])
    while (
      end + 1 < contact.shape[0]
      and bool(contact[end + 1, env_id])
      and int(sub_episode[end + 1, env_id]) == 0
    ):
      end += 1
    steps = np.unique(np.arange(int(indices[0]), end + 1) // decimation)
    steps = steps[(steps < lam.shape[0]) & (episode_id[steps, env_id] == 0)]
    if steps.size:
      peaks.append(lam[steps, env_id].max(axis=0))
  values = np.asarray(peaks, dtype=np.float64)
  return {
    "scope": "native controller reads overlapping the observed first-contact prefix",
    "complete_physical_event": False,
    "prefixes_with_contact": len(peaks),
    "global_max_lambda_n_m_s": _quantiles(values.max(axis=1) if values.size else values),
    "per_joint_lambda_n_m_s": {
      name: _quantiles(values[:, joint] if values.size else values)
      for joint, name in enumerate(survey.JOINT_NAMES)
    },
  }


def _action_descriptors(trace: Mapping[str, np.ndarray]) -> dict[str, object]:
  actions = np.asarray(trace["policy_action"])
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  if (
    actions.shape != (*episode_id.shape, 12)
    or not np.issubdtype(actions.dtype, np.floating)
    or not np.isfinite(actions).all()
  ):
    raise ValueError("policy_action must be finite with shape (steps, envs, 12)")
  valid_actions = actions[episode_id == 0]
  return {
    f"action_{index}": _quantiles(valid_actions[:, index])
    for index in range(12)
  }


def _secondary_descriptors(
  trace: Mapping[str, np.ndarray], population_summary: Mapping[str, object]
) -> dict[str, object]:
  threshold = population_summary.get("provisional_caps")
  if not isinstance(threshold, Mapping):
    raise ValueError("population summary is missing provisional_caps")
  binding = threshold.get("binding")
  utility = threshold.get("utility")
  if not isinstance(binding, Mapping) or not isinstance(utility, Mapping):
    raise ValueError("population summary is missing binding or utility telemetry")
  gains = utility.get("first_contact_vic_gains")
  if not isinstance(gains, Mapping):
    raise ValueError("population summary is missing first-contact VIC gains")
  physical_contact = threshold.get("physical_contact_duration_ms")
  if not isinstance(physical_contact, Mapping):
    raise ValueError("population summary is missing physical-contact censoring telemetry")
  valid = np.asarray(trace["episode_id"], dtype=np.int64) == 0
  reads_per_env = valid.sum(axis=0)
  nail_depth = _terminal_endpoint_by_environment(trace, "nail_depth_m")
  delivered = _terminal_endpoint_by_environment(trace, "delivered_total_n_s")
  return {
    "episode_duration_ms": _observed_duration(trace),
    "physical_contact_censoring": dict(physical_contact),
    "controller_read_counts": {
      "observed_initial_episode_reads_per_environment": _quantiles(reads_per_env),
      "violating_reads": binding.get("violating_reads"),
      "activation_window_read_counts": binding.get("activation_window_read_counts"),
      "physical_event_read_counts": binding.get("physical_event_read_counts"),
    },
    "observed_first_contact_prefix_lambda": _observed_first_contact_prefix_lambda(trace),
    "nail_depth_m": {
      "completed_environments": nail_depth["completed_environments"],
      "right_censored_environments": nail_depth["right_censored_environments"],
      **_quantiles(nail_depth["values"]),
    },
    "cumulative_delivered_impulse_n_s": {
      "completed_environments": delivered["completed_environments"],
      "right_censored_environments": delivered["right_censored_environments"],
      **_quantiles(delivered["values"]),
    },
    "vic_gains": historical._compact_gain_summary(gains),
    "policy_action": _action_descriptors(trace),
  }


def _validate_live_trace(trace: Mapping[str, np.ndarray]) -> None:
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  velocity = np.asarray(trace["delta_velocity"])
  impulse = np.asarray(trace["delta_impulse"])
  combined = np.asarray(trace["delta"])
  if not (
    velocity.shape == impulse.shape == combined.shape == episode_id.shape
  ) or not all(np.isfinite(value).all() for value in (velocity, impulse, combined)):
    raise ValueError("live CaT traces must be finite and aligned")
  if not np.array_equal(impulse, np.zeros_like(impulse)):
    raise ValueError("bridge evaluation must keep live impulse CaT log-only")
  if not np.array_equal(combined, np.maximum(velocity, impulse)):
    raise ValueError("bridge evaluation combined delta must be the exact max soft-OR")
  _action_descriptors(trace)


def evaluate_population_pair(
  control_trace: Mapping[str, np.ndarray],
  target_trace: Mapping[str, np.ndarray],
  control_summary: Mapping[str, object],
  target_summary: Mapping[str, object],
  *,
  inferential: bool,
) -> dict[str, object]:
  """Analyze one matched population; fixed-64 may request descriptors but never gates."""
  _validate_live_trace(control_trace)
  _validate_live_trace(target_trace)
  caps = survey.PROVISIONAL_CAPS_N_M_S
  control_utilization = _initial_joint_utilization(control_trace, caps)
  target_utilization = _initial_joint_utilization(target_trace, caps)
  if control_utilization.shape != target_utilization.shape:
    raise ValueError("paired populations must contain the same environment IDs")
  control_rho = control_utilization.max(axis=1)
  target_rho = target_utilization.max(axis=1)
  control_violation = _initial_joint_violation_flags(control_trace, caps)
  target_violation = _initial_joint_violation_flags(target_trace, caps)
  result: dict[str, object] = {
    "statistical_unit": "whole paired environment ID",
    "controller_reads_are_inferential_units": False,
    "per_environment_per_joint_utilization": {
      "control": {
        name: _quantiles(control_utilization[:, joint])
        for joint, name in enumerate(survey.JOINT_NAMES)
      },
      "target": {
        name: _quantiles(target_utilization[:, joint])
        for joint, name in enumerate(survey.JOINT_NAMES)
      },
    },
    "secondary_descriptors": {
      "control": _secondary_descriptors(control_trace, control_summary),
      "target": _secondary_descriptors(target_trace, target_summary),
    },
    "claim_boundary": {
      "native_observed_exposure_only": True,
      "complete_contact_claim_authorized": False,
      "actuator_loading_or_hardware_safety_claim_authorized": False,
    },
  }
  if not inferential:
    result["descriptive_only"] = True
    return result

  impulse_bootstrap = historical.paired_binary_risk_bootstrap(
    control_violation.any(axis=1),
    target_violation.any(axis=1),
    resamples=BOOTSTRAP_RESAMPLES,
    seed=BOOTSTRAP_SEED,
  )
  control_velocity = historical._initial_episode_velocity_peaks(control_trace)
  target_velocity = historical._initial_episode_velocity_peaks(target_trace)
  velocity_bootstrap = historical.paired_binary_risk_bootstrap(
    (control_velocity > survey.VELOCITY_LIMIT_RAD_S).any(axis=1),
    (target_velocity > survey.VELOCITY_LIMIT_RAD_S).any(axis=1),
    resamples=BOOTSTRAP_RESAMPLES,
    seed=BOOTSTRAP_SEED,
  )
  terminal_endpoints = {
    role: {
      name: _terminal_endpoint_by_environment(trace, name)
      for name in (
        "success",
        "first_strike_productive",
        "first_strike_delivered_n_s",
      )
    }
    for role, trace in (("control", control_trace), ("target", target_trace))
  }
  expected_environments = control_utilization.shape[0]
  terminal_complete = all(
    endpoint["completed_environments"] == expected_environments
    for role in terminal_endpoints.values()
    for endpoint in role.values()
  )
  if terminal_complete:
    control_delivered = terminal_endpoints["control"][
      "first_strike_delivered_n_s"
    ]["values"]
    target_delivered = terminal_endpoints["target"][
      "first_strike_delivered_n_s"
    ]["values"]
    delivered_bootstrap = paired_mean_ratio_bootstrap(
      control_delivered, target_delivered
    )
    success_difference = _binary_difference(
      terminal_endpoints["control"]["success"]["values"],
      terminal_endpoints["target"]["success"]["values"],
    )
    productive_difference = _binary_difference(
      terminal_endpoints["control"]["first_strike_productive"]["values"],
      terminal_endpoints["target"]["first_strike_productive"]["values"],
    )
  else:
    delivered_bootstrap = {
      "resampling_unit": "whole paired environment ID",
      "controller_reads_are_inferential_units": False,
      "environments": expected_environments,
      "resamples": BOOTSTRAP_RESAMPLES,
      "seed": BOOTSTRAP_SEED,
      "point_target_over_control_mean_ratio": None,
      "valid": False,
      "lower_97_5": None,
      "invalid_nonpositive_control_resamples": 0,
      "failure_reason": "initial episode fragment is censored in at least one arm",
      "completed_environments": {
        role: endpoint["first_strike_delivered_n_s"]["completed_environments"]
        for role, endpoint in terminal_endpoints.items()
      },
      "right_censored_environments": {
        role: endpoint["first_strike_delivered_n_s"]["right_censored_environments"]
        for role, endpoint in terminal_endpoints.items()
      },
    }
    success_difference = None
    productive_difference = None
  endpoints = {
    "provisional_risk": impulse_bootstrap,
    "rho": {"control": _quantiles(control_rho), "target": _quantiles(target_rho)},
    "per_joint": {
      "control_p99": np.quantile(control_utilization, 0.99, axis=0).tolist(),
      "target_p99": np.quantile(target_utilization, 0.99, axis=0).tolist(),
      "control_any_violation": control_violation.any(axis=0).tolist(),
      "target_any_violation": target_violation.any(axis=0).tolist(),
    },
    "true_velocity_risk": velocity_bootstrap,
    "success_difference": success_difference,
    "productive_strike_difference": productive_difference,
    "first_event_delivered_impulse_ratio": delivered_bootstrap,
  }
  result["endpoints"] = endpoints
  result["verdict"] = evaluate_gate_metrics(
    {
      "provisional_risk_difference_upper_97_5": impulse_bootstrap["interval_95"][
        "target_minus_control_risk"
      ]["high"],
      "rho": endpoints["rho"],
      "per_joint": endpoints["per_joint"],
      "velocity_risk_difference_upper_97_5": velocity_bootstrap["interval_95"][
        "target_minus_control_risk"
      ]["high"],
      "success_difference": endpoints["success_difference"],
      "productive_strike_difference": endpoints["productive_strike_difference"],
      "first_event_delivered_ratio_lower_97_5": delivered_bootstrap["lower_97_5"],
      "identity_finiteness_native_claims": (
        terminal_complete and delivered_bootstrap["valid"]
      ),
    }
  )
  return result


def assemble_analysis(
  *,
  fixed_result: Mapping[str, object],
  stochastic_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
  expected = tuple(str(seed) for seed in STOCHASTIC_SEEDS)
  if tuple(stochastic_results) != expected:
    raise ValueError("analysis requires the exact separate stochastic populations")
  return {
    "schema_version": 1,
    "purpose": "pre-registered native-observed impulse-CaT p=0.2 bridge screen",
    "analysis_protocol": {
      "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
      "bootstrap_seed": BOOTSTRAP_SEED,
      "resampling_unit": "whole paired environment ID",
      "fixed_mean_is_inferential": False,
      "stochastic_populations_reported_separately": True,
    },
    "fixed_mean_descriptive_only": dict(fixed_result),
    "training_like_by_seed": dict(stochastic_results),
  }


def analyze_evaluation_pair(
  control_leaf: Path,
  target_leaf: Path,
  *,
  expected_code_revision: str,
) -> dict[str, object]:
  """Validate two immutable five-artifact leaves and analyze their matched populations."""
  leaves = {
    "control": Path(control_leaf).resolve(strict=True),
    "target": Path(target_leaf).resolve(strict=True),
  }
  manifests = {
    role: historical.validate_leaf_manifest(leaf) for role, leaf in leaves.items()
  }
  summaries: dict[str, dict[str, object]] = {}
  for role, leaf in leaves.items():
    summary = json.loads((leaf / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    historical._require_finite_json(summary)
    summaries[role] = summary
  survey.compare_policy_evaluations(
    summaries["control"],
    summaries["target"],
    expected_code_revision=expected_code_revision,
    expected_roles=EXPECTED_ROLES,
  )

  fixed_traces = {
    role: historical._load_trace(leaf / "evaluation" / "fixed_trace.npz")
    for role, leaf in leaves.items()
  }
  fixed = evaluate_population_pair(
    fixed_traces["control"],
    fixed_traces["target"],
    summaries["control"]["populations"]["fixed_mean"],
    summaries["target"]["populations"]["fixed_mean"],
    inferential=False,
  )
  stochastic: dict[str, object] = {}
  for seed in STOCHASTIC_SEEDS:
    seed_key = str(seed)
    traces = {
      role: historical._load_trace(
        leaf / "evaluation" / f"training_like_seed_{seed}_trace.npz"
      )
      for role, leaf in leaves.items()
    }
    stochastic[seed_key] = evaluate_population_pair(
      traces["control"],
      traces["target"],
      summaries["control"]["populations"]["training_like_sampled"][seed_key],
      summaries["target"]["populations"]["training_like_sampled"][seed_key],
      inferential=True,
    )
  payload = assemble_analysis(fixed_result=fixed, stochastic_results=stochastic)
  payload["provenance"] = {
    "expected_roles": list(EXPECTED_ROLES),
    "code_revision": expected_code_revision,
    "asset_revision": survey.EXPECTED_EVALUATION_ASSET_REVISION,
    "independently_rehashed_artifacts": manifests,
    "identity_finiteness_native_claim_gate": True,
  }
  return payload


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--control-leaf", required=True, type=Path)
  parser.add_argument("--target-leaf", required=True, type=Path)
  parser.add_argument("--expected-code-revision", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  payload = analyze_evaluation_pair(
    args.control_leaf,
    args.target_leaf,
    expected_code_revision=args.expected_code_revision,
  )
  args.output.write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
  )
  print(args.output)


if __name__ == "__main__":
  main()
