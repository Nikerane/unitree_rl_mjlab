#!/usr/bin/env python3
"""Compact analysis for the frozen Z1 diagnostic-0.9 policy evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ARTIFACT_NAMES = (
  "fixed_trace.npz",
  "training_like_seed_2_trace.npz",
  "training_like_seed_2026081701_trace.npz",
  "training_like_seed_2026081702_trace.npz",
  "summary.json",
)
CONTROL_LEAF = "41386821_0_diag90_control"
TARGET_LEAF = "41386821_1_diag90_target"
EVALUATOR_REVISION = "142098b8af4cbd4b52777012b27f477f0e9aa358"
ASSET_REVISION = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026081703
STOCHASTIC_SEEDS = (2, 2026081701, 2026081702)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def validate_leaf_manifest(leaf: Path) -> dict[str, dict[str, object]]:
  """Validate the evaluator's exact-five manifest and rehash every artifact."""
  leaf = Path(leaf).resolve(strict=True)
  evaluation = leaf / "evaluation"
  manifest = leaf / "SHA256SUMS"
  if not evaluation.is_dir() or not manifest.is_file() or manifest.is_symlink():
    raise ValueError(f"invalid evaluation leaf layout: {leaf}")
  children = tuple(sorted(path.name for path in evaluation.iterdir()))
  if children != tuple(sorted(EXPECTED_ARTIFACT_NAMES)):
    raise ValueError("evaluation leaf must contain exactly the five frozen artifacts")
  rows = manifest.read_text(encoding="utf-8").splitlines()
  if len(rows) != len(EXPECTED_ARTIFACT_NAMES):
    raise ValueError("SHA256SUMS must contain exactly five rows")
  result: dict[str, dict[str, object]] = {}
  for row, expected_name in zip(rows, EXPECTED_ARTIFACT_NAMES, strict=True):
    expected_suffix = f"  evaluation/{expected_name}"
    if not row.endswith(expected_suffix):
      raise ValueError("SHA256SUMS artifact order or path drifted")
    expected_digest = row[: -len(expected_suffix)]
    if len(expected_digest) != 64 or any(
      character not in "0123456789abcdef" for character in expected_digest
    ):
      raise ValueError("SHA256SUMS contains a malformed SHA-256")
    path = evaluation / expected_name
    if not path.is_file() or path.is_symlink():
      raise ValueError("evaluation artifacts must be regular non-symlink files")
    actual_digest = _sha256(path)
    if actual_digest != expected_digest:
      raise ValueError(f"artifact SHA-256 mismatch: {expected_name}")
    result[expected_name] = {
      "sha256": actual_digest,
      "size_bytes": path.stat().st_size,
    }
  return result


def _initial_episode_rho(
  trace: Mapping[str, np.ndarray], caps: Sequence[float]
) -> np.ndarray:
  lam = np.asarray(trace["lambda_per_joint"])
  episode_id = np.asarray(trace["episode_id"])
  cap_array = np.asarray(caps, dtype=np.float64)
  if (
    lam.ndim != 3
    or episode_id.shape != lam.shape[:2]
    or cap_array.shape != (lam.shape[2],)
    or not np.issubdtype(lam.dtype, np.floating)
    or not np.isfinite(lam).all()
    or not np.isfinite(cap_array).all()
    or bool((cap_array <= 0.0).any())
  ):
    raise ValueError("trace and caps must define finite positive initial-episode utilization")
  valid = episode_id == 0
  if not bool(valid.any(axis=0).all()):
    raise ValueError("every environment must contain an initial-episode read")
  utilization = lam.astype(np.float64) / cap_array
  return np.where(valid[:, :, None], utilization, -np.inf).max(axis=(0, 2))


def paired_binary_risk_bootstrap(
  control: np.ndarray,
  target: np.ndarray,
  *,
  resamples: int,
  seed: int,
) -> dict[str, object]:
  """Bootstrap a matched binary endpoint without treating reads as independent."""
  control_flags = np.asarray(control)
  target_flags = np.asarray(target)
  if (
    control_flags.ndim != 1
    or target_flags.shape != control_flags.shape
    or control_flags.dtype != np.bool_
    or target_flags.dtype != np.bool_
    or control_flags.size == 0
  ):
    raise ValueError("paired binary endpoints must be nonempty aligned boolean vectors")
  if type(resamples) is not int or resamples <= 0:
    raise ValueError("resamples must be a positive integer")
  if type(seed) is not int or seed < 0:
    raise ValueError("seed must be a nonnegative integer")

  control_risk = float(control_flags.mean())
  target_risk = float(target_flags.mean())
  risk_differences = np.empty(resamples, dtype=np.float64)
  risk_ratios = np.full(resamples, np.nan, dtype=np.float64)
  rng = np.random.default_rng(seed)
  envs = control_flags.size
  offset = 0
  while offset < resamples:
    count = min(256, resamples - offset)
    indices = rng.integers(0, envs, size=(count, envs))
    sampled_control = control_flags[indices].mean(axis=1)
    sampled_target = target_flags[indices].mean(axis=1)
    risk_differences[offset : offset + count] = sampled_target - sampled_control
    positive_control = sampled_control > 0.0
    risk_ratios[offset : offset + count][positive_control] = (
      sampled_target[positive_control] / sampled_control[positive_control]
    )
    offset += count

  def interval(values: np.ndarray) -> dict[str, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
      return None
    low, high = np.quantile(finite, (0.025, 0.975))
    return {"low": float(low), "high": float(high)}

  return {
    "resampling_unit": "whole paired environment ID",
    "controller_reads_are_inferential_units": False,
    "environments": envs,
    "resamples": resamples,
    "seed": seed,
    "point": {
      "control_risk": control_risk,
      "target_risk": target_risk,
      "target_minus_control_risk": target_risk - control_risk,
      "target_over_control_risk": (
        target_risk / control_risk if control_risk > 0.0 else None
      ),
    },
    "interval_95": {
      "target_minus_control_risk": interval(risk_differences),
      "target_over_control_risk": interval(risk_ratios),
    },
    "undefined_ratio_resamples": int((~np.isfinite(risk_ratios)).sum()),
  }


def paired_initial_episode_bootstrap(
  control_trace: Mapping[str, np.ndarray],
  target_trace: Mapping[str, np.ndarray],
  *,
  caps: Sequence[float],
  resamples: int,
  seed: int,
) -> dict[str, object]:
  """Bootstrap paired initial episodes by resampling complete environment IDs."""
  control_rho = _initial_episode_rho(control_trace, caps)
  target_rho = _initial_episode_rho(target_trace, caps)
  if control_rho.shape != target_rho.shape:
    raise ValueError("paired arms must contain the same environment IDs")

  control_violates = control_rho > 1.0
  target_violates = target_rho > 1.0
  binary = paired_binary_risk_bootstrap(
    control_violates, target_violates, resamples=resamples, seed=seed
  )
  return {
    **{key: value for key, value in binary.items() if key not in ("point", "interval_95")},
    "point": {
      f"{key.removesuffix('_risk')}_violation_risk": value
      for key, value in binary["point"].items()
    },
    "interval_95": {
      f"{key.removesuffix('_risk')}_violation_risk": value
      for key, value in binary["interval_95"].items()
    },
  }


def _survey_api() -> Any:
  try:
    from scripts import impulse_cat_activation_survey as survey
  except ImportError:
    import impulse_cat_activation_survey as survey
  return survey


def _require_finite_json(value: object, *, path: str = "$") -> None:
  if isinstance(value, Mapping):
    for key, child in value.items():
      _require_finite_json(child, path=f"{path}.{key}")
  elif isinstance(value, list):
    for index, child in enumerate(value):
      _require_finite_json(child, path=f"{path}[{index}]")
  elif isinstance(value, float) and not np.isfinite(value):
    raise ValueError(f"non-finite JSON value at {path}")


def _load_trace(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as archive:
    trace = {name: archive[name] for name in archive.files}
  for name, value in trace.items():
    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
      raise ValueError(f"non-finite trace field {path.name}:{name}")
  return trace


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
  flat = np.asarray(values, dtype=np.float64).reshape(-1)
  if flat.size == 0:
    return {"p50": None, "p95": None, "p99": None, "max": None}
  if not np.isfinite(flat).all():
    raise ValueError("descriptive endpoint contains a non-finite value")
  return {
    "p50": float(np.quantile(flat, 0.50)),
    "p95": float(np.quantile(flat, 0.95)),
    "p99": float(np.quantile(flat, 0.99)),
    "max": float(np.max(flat)),
  }


def _initial_episode_velocity_peaks(trace: Mapping[str, np.ndarray]) -> np.ndarray:
  velocity = np.asarray(trace["substep_peak_qv_per_joint"], dtype=np.float64)
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  valid = episode_id == 0
  if (
    velocity.ndim != 3
    or episode_id.shape != velocity.shape[:2]
    or not np.isfinite(velocity).all()
    or not valid.any(axis=0).all()
  ):
    raise ValueError("trace lacks aligned finite initial-episode velocity reads")
  return np.where(valid[:, :, None], velocity, -np.inf).max(axis=0)


def _observed_first_contact_prefix_utilization(
  trace: Mapping[str, np.ndarray], caps: Sequence[float]
) -> dict[str, object]:
  lam = np.asarray(trace["lambda_per_joint"], dtype=np.float64)
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  contact = np.asarray(trace["substep_contact"], dtype=bool)
  sub_episode = np.asarray(trace["substep_episode_id"], dtype=np.int64)
  cap_array = np.asarray(caps, dtype=np.float64)
  control_decimation = contact.shape[0] // lam.shape[0]
  values: list[float] = []
  for env_id in range(lam.shape[1]):
    indices = np.flatnonzero(contact[:, env_id] & (sub_episode[:, env_id] == 0))
    if indices.size == 0:
      continue
    start = int(indices[0])
    end = start
    while end + 1 < contact.shape[0] and bool(contact[end + 1, env_id]) and int(
      sub_episode[end + 1, env_id]
    ) == 0:
      end += 1
    steps = np.unique(np.arange(start, end + 1) // control_decimation)
    steps = steps[(steps < lam.shape[0]) & (episode_id[steps, env_id] == 0)]
    if steps.size:
      values.append(float(np.max(lam[steps, env_id] / cap_array)))
  return {
    "unit": "observed first-contact physical-event prefix",
    "scope": "native controller reads overlapping the terminal-cut first-contact prefix",
    "equal_horizon_across_arms": False,
    "complete_physical_event": False,
    "causal_or_full_contact_interpretation_authorized": False,
    "observed_prefixes_with_contact": len(values),
    "rho": _quantiles(np.asarray(values)),
  }


def _terminal_values(trace: Mapping[str, np.ndarray], name: str) -> np.ndarray:
  done = np.asarray(trace["done"], dtype=bool)
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  values = np.asarray(trace[name])
  terminal = done & (episode_id == 0)
  if values.shape[:2] != terminal.shape:
    raise ValueError(f"terminal field {name} is not aligned")
  selected = values[terminal]
  if selected.shape[0] != terminal.shape[1]:
    raise ValueError(f"every initial episode must terminate exactly once for {name}")
  return selected


def _episode_duration_ms(trace: Mapping[str, np.ndarray]) -> np.ndarray:
  done = np.asarray(trace["done"], dtype=bool)
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  terminal = done & (episode_id == 0)
  durations: list[float] = []
  for env_id in range(done.shape[1]):
    steps = np.flatnonzero(terminal[:, env_id])
    if steps.size != 1:
      raise ValueError("each initial episode must have exactly one terminal step")
    durations.append((int(steps[0]) + 1) * 20.0)
  return np.asarray(durations, dtype=np.float64)


def _compact_gain_summary(gains: Mapping[str, object]) -> dict[str, object]:
  records = gains["records"]
  if not isinstance(records, list):
    raise ValueError("first-contact gain records must be a list")
  result: dict[str, object] = {
    "segments_with_contact": gains["segments_with_contact"],
    "right_censored_segments": gains["right_censored_segments"],
    "per_joint": {},
  }
  per_joint = result["per_joint"]
  for joint in range(6):
    joint_summary: dict[str, object] = {}
    for timing in ("precontact", "at_contact"):
      for field in ("vic_p", "vic_kp", "vic_kd"):
        values = [
          record[f"{field}_{timing}"][joint]
          for record in records
          if record[f"{field}_{timing}"] is not None
        ]
        array = np.asarray(values, dtype=np.float64)
        joint_summary[f"{field}_{timing}"] = {
          "mean": float(array.mean()) if array.size else None,
          **_quantiles(array),
        }
    per_joint[f"joint{joint + 1}"] = joint_summary
  return result


def _activation_window_pressure_by_associated_physical_event_status(
  summary: Mapping[str, object],
  trace: Mapping[str, np.ndarray],
  *,
  survey: Any,
) -> dict[str, object]:
  steps, envs = np.asarray(trace["done"]).shape
  terminal = np.zeros((steps * survey.CONTROL_DECIMATION, envs), dtype=bool)
  terminal[survey.CONTROL_DECIMATION - 1 :: survey.CONTROL_DECIMATION] = np.asarray(
    trace["done"], dtype=bool
  )
  events, _ = survey.contact_events(
    trace["substep_contact"],
    terminal=terminal,
    episode_id=trace["substep_episode_id"],
    physics_dt_s=survey.PHYSICS_DT_S,
  )
  by_id = {int(event["event_id"]): event for event in events}
  binding = summary["binding"]
  candidate = summary["candidate_imp_max_p"]["0.5"]
  reads = binding["reads"]
  buckets: dict[str, list[float]] = {
    "completed_unambiguous": [],
    "completed_ambiguous": [],
    "right_censored_unambiguous": [],
    "right_censored_ambiguous": [],
  }
  for window in candidate["activation_window_summaries"]:
    associated = [
      read
      for read in reads
      if int(read["env_id"]) == int(window["env_id"])
      and int(window["start_step"]) <= int(read["control_step"]) <= int(window["end_step"])
    ]
    event_ids = {
      int(event_id) for read in associated for event_id in read["contact_event_ids"]
    }
    if not event_ids:
      raise ValueError("an activating window has no physical-event association")
    ambiguous = len(event_ids) != 1 or any(
      bool(read["physical_event_association_ambiguous"]) for read in associated
    )
    censored = any(bool(by_id[event_id]["right_censored"]) for event_id in event_ids)
    key = (
      ("right_censored" if censored else "completed")
      + ("_ambiguous" if ambiguous else "_unambiguous")
    )
    buckets[key].append(float(window["event_pressure"]))
  return {
    "unit": "contiguous 50 Hz impulse-activation window",
    "unique_physical_event_dose": False,
    "classification_basis": "censoring and ambiguity of associated physical event IDs",
    "groups": {
      key: {
        "activation_windows": len(values),
        "activation_window_pressure": _quantiles(np.asarray(values)),
      }
      for key, values in buckets.items()
    },
  }


def _compact_cat_attribution(
  summary: Mapping[str, object],
  trace: Mapping[str, np.ndarray],
  *,
  caps: Sequence[float],
  survey: Any,
) -> dict[str, object]:
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  valid = episode_id == 0
  velocity = np.asarray(trace["delta_velocity"], dtype=np.float64)
  live_impulse = np.asarray(trace["delta_impulse"], dtype=np.float64)
  live_combined = np.asarray(trace["delta"], dtype=np.float64)
  if not (
    velocity.shape == live_impulse.shape == live_combined.shape == valid.shape
  ) or not all(
    np.isfinite(values).all()
    for values in (velocity, live_impulse, live_combined)
  ):
    raise ValueError("live CaT component traces must be finite and aligned")
  if not np.array_equal(live_impulse, np.zeros_like(live_impulse)):
    raise ValueError("live impulse delta must remain identically zero")
  if not np.array_equal(live_combined, np.maximum(velocity, live_impulse)):
    raise ValueError("live combined delta is not the exact max of its components")

  lam = np.asarray(trace["lambda_per_joint"])
  margins = survey._native_cap_margins(lam, tuple(caps))
  unit_per_joint = survey.shadow_impulse_cat(
    margins,
    tau=survey.CAT_TAU,
    seed=survey.IMPULSE_SEED,
    max_p=1.0,
    valid=valid,
  )
  offline_impulse = 0.5 * unit_per_joint.max(axis=2)
  offline_combined = np.maximum(velocity, offline_impulse)
  if not np.array_equal(offline_combined, np.maximum(velocity, offline_impulse)):
    raise RuntimeError("offline combined delta is not the exact max of its components")
  active = valid & (offline_impulse > 0.0)
  candidate = summary["candidate_imp_max_p"]["0.5"]
  expected = {
    "active_reads": int(active.sum()),
    "delta_impulse_active_mean": (
      float(offline_impulse[active].mean()) if bool(active.any()) else 0.0
    ),
    "delta_impulse_max": float(offline_impulse[valid].max()),
    "combined_delta_max": float(offline_combined[valid].max()),
  }
  for name, value in expected.items():
    stored = candidate[name]
    if not isinstance(stored, (int, float)) or not np.isclose(
      float(stored), float(value), rtol=0.0, atol=1e-7
    ):
      raise ValueError(f"stored CaT candidate {name} differs from exact replay")

  def component_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    selected = values[mask]
    return {
      "mean": float(selected.mean()) if selected.size else None,
      **_quantiles(selected),
    }

  return {
    "scope": "all native initial-episode controller reads",
    "live_trace": {
      "delta_velocity": component_summary(velocity, valid),
      "delta_impulse": component_summary(live_impulse, valid),
      "combined_delta": component_summary(live_combined, valid),
      "delta_impulse_identically_zero": True,
      "combined_delta_exact_max": True,
    },
    "offline_replay_at_p_0_5": {
      "delta_velocity": component_summary(velocity, valid),
      "delta_impulse": component_summary(offline_impulse, valid),
      "combined_delta": component_summary(offline_combined, valid),
      "active_read_components": {
        "delta_velocity": component_summary(velocity, active),
        "delta_impulse": component_summary(offline_impulse, active),
        "combined_delta": component_summary(offline_combined, active),
      },
      "combined_delta_exact_max": True,
      "summary_replay_consistent": True,
    },
  }


def _compact_threshold(
  summary: Mapping[str, object],
  trace: Mapping[str, np.ndarray],
  *,
  survey: Any,
) -> dict[str, object]:
  caps = summary["caps_n_m_s"]
  compliance = summary["segment_compliance"]
  compact_joints: dict[str, object] = {}
  for name, values in compliance["per_joint"].items():
    cap = float(summary["per_joint"][name]["cap_n_m_s"])
    utilization = values["utilization"]
    compact_joints[name] = {
      "lambda_peak_n_m_s": {
        key: (None if value is None else float(value) * cap)
        for key, value in utilization.items()
      },
      "utilization": utilization,
      "positive_margin_n_m_s": values["positive_margin_n_m_s"],
      "violating_segments": values["violating_segments"],
      "violation_rate": values["violation_rate"],
      "violating_reads": summary["per_joint"][name]["violating_reads"],
      "associated_physical_events": summary["per_joint"][name][
        "associated_physical_events"
      ],
      "responsible_physical_events": summary["per_joint"][name][
        "responsible_physical_events"
      ],
    }
  candidate = summary["candidate_imp_max_p"]["0.5"]
  binding = summary["binding"]
  return {
    "caps_n_m_s": caps,
    "segments": compliance["segments"],
    "any_joint_violating_segments": compliance["any_joint_violating_segments"],
    "any_joint_violation_rate": compliance["any_joint_violation_rate"],
    "rho": compliance["max_joint_utilization"],
    "observed_first_contact_prefix_utilization": (
      _observed_first_contact_prefix_utilization(trace, caps)
    ),
    "per_joint": compact_joints,
    "binding_reads_and_events": {
      "violating_reads": binding["violating_reads"],
      "activation_windows": binding["violating_activation_windows"],
      "associated_physical_events": binding["associated_physical_events"],
      "ambiguous_multi_event_reads": binding["ambiguous_multi_event_reads"],
      "activation_window_read_counts": _quantiles(
        np.asarray(binding["activation_window_read_counts"])
      ),
      "physical_event_exclusive_read_counts": _quantiles(
        np.asarray(
          [value["exclusive_reads"] for value in binding["physical_event_read_counts"]]
        )
      ),
    },
    "offline_cat_at_p_0_5": {
      "active_reads": candidate["active_reads"],
      "delta_impulse_active_mean": candidate["delta_impulse_active_mean"],
      "delta_impulse_max": candidate["delta_impulse_max"],
      "combined_delta_max": candidate["combined_delta_max"],
      "impulse_winner_reads": candidate["impulse_winner_reads"],
      "velocity_masked_reads": candidate["velocity_masked_reads"],
      "tie_reads": candidate["tie_reads"],
      "impulse_win_share": candidate["impulse_win_share"],
      "velocity_mask_share": candidate["velocity_mask_share"],
      "component_summaries_and_invariants": _compact_cat_attribution(
        summary, trace, caps=caps, survey=survey
      ),
      "activation_window_pressure_by_associated_physical_event_status": (
        _activation_window_pressure_by_associated_physical_event_status(
          summary, trace, survey=survey
        )
      ),
    },
    "physical_contact": summary["physical_contact_duration_ms"],
  }


def _compact_utility(
  summary: Mapping[str, object], trace: Mapping[str, np.ndarray]
) -> dict[str, object]:
  utility = summary["utility"]
  first_delivered = _terminal_values(trace, "first_strike_delivered_n_s")
  total_delivered = _terminal_values(trace, "delivered_total_n_s")
  nail_depth = _terminal_values(trace, "nail_depth_m")
  return {
    "task_scope": "complete initial episodes only; post-reset episodes excluded",
    "terminal_value_source": "episode_id == 0 and done",
    "terminal_counts": utility["terminal_counts"],
    "first_strike_delivered_n_s": {
      "mean": float(np.mean(first_delivered)),
      **_quantiles(first_delivered),
    },
    "delivered_total_n_s": {"mean": float(np.mean(total_delivered)), **_quantiles(total_delivered)},
    "nail_depth_m": {"mean": float(np.mean(nail_depth)), **_quantiles(nail_depth)},
    "episode_duration_ms": _quantiles(_episode_duration_ms(trace)),
    "episode_reward": {
      "available": False,
      "trace_field_present": "reward" in trace,
      "reason": "the frozen evaluator trace did not record reward",
    },
    "true_500hz_velocity": utility["velocity_limit_compliance"],
    "first_contact_vic_gains": _compact_gain_summary(
      utility["first_contact_vic_gains"]
    ),
  }


def _population_result(
  control_summary: Mapping[str, object],
  target_summary: Mapping[str, object],
  control_trace: Mapping[str, np.ndarray],
  target_trace: Mapping[str, np.ndarray],
  *,
  survey: Any,
) -> dict[str, object]:
  thresholds: dict[str, object] = {}
  for name, caps in (
    ("provisional_caps", survey.PROVISIONAL_CAPS_N_M_S),
    ("diagnostic_only", survey.DIAGNOSTIC_LIMITS_N_M_S),
  ):
    thresholds[name] = {
      "control": _compact_threshold(control_summary[name], control_trace, survey=survey),
      "target": _compact_threshold(target_summary[name], target_trace, survey=survey),
      "paired_initial_episode_bootstrap": paired_initial_episode_bootstrap(
        control_trace,
        target_trace,
        caps=caps,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
      ),
    }
  control_velocity = _initial_episode_velocity_peaks(control_trace)
  target_velocity = _initial_episode_velocity_peaks(target_trace)
  velocity_bootstrap = paired_binary_risk_bootstrap(
    (control_velocity > survey.VELOCITY_LIMIT_RAD_S).any(axis=1),
    (target_velocity > survey.VELOCITY_LIMIT_RAD_S).any(axis=1),
    resamples=BOOTSTRAP_RESAMPLES,
    seed=BOOTSTRAP_SEED,
  )
  control_utility = _compact_utility(control_summary["provisional_caps"], control_trace)
  target_utility = _compact_utility(target_summary["provisional_caps"], target_trace)
  return {
    "thresholds_in_required_order": thresholds,
    "utility": {
      "control": control_utility,
      "target": target_utility,
      "first_event_delivered_impulse_target_over_control_mean_ratio": (
        target_utility["first_strike_delivered_n_s"]["mean"]
        / control_utility["first_strike_delivered_n_s"]["mean"]
      ),
      "paired_true_velocity_violation_bootstrap": velocity_bootstrap,
    },
  }


def _pooled_descriptive(
  traces: Mapping[str, list[Mapping[str, np.ndarray]]], *, survey: Any
) -> dict[str, object]:
  result: dict[str, object] = {
    "scope": "descriptive concatenation of three inference/reset replicas; no pooled CI or p-value",
    "environments_per_arm": sum(
      np.asarray(trace["episode_id"]).shape[1] for trace in traces["control"]
    ),
    "thresholds_in_required_order": {},
  }
  for name, caps in (
    ("provisional_caps", survey.PROVISIONAL_CAPS_N_M_S),
    ("diagnostic_only", survey.DIAGNOSTIC_LIMITS_N_M_S),
  ):
    threshold: dict[str, object] = {}
    for role in ("control", "target"):
      rho = np.concatenate([_initial_episode_rho(trace, caps) for trace in traces[role]])
      threshold[role] = {
        "violating_segments": int((rho > 1.0).sum()),
        "violation_risk": float((rho > 1.0).mean()),
        "rho": _quantiles(rho),
      }
    result["thresholds_in_required_order"][name] = threshold
  for role in ("control", "target"):
    velocities = np.concatenate(
      [_initial_episode_velocity_peaks(trace) for trace in traces[role]], axis=0
    )
    delivered = np.concatenate(
      [_terminal_values(trace, "first_strike_delivered_n_s") for trace in traces[role]]
    )
    result[f"{role}_utility"] = {
      "success": sum(
        int(_terminal_values(trace, "success").sum()) for trace in traces[role]
      ),
      "productive_first_strike": sum(
        int(_terminal_values(trace, "first_strike_productive").sum())
        for trace in traces[role]
      ),
      "first_strike_delivered_n_s_mean": float(delivered.mean()),
      "true_velocity_violating_segments": int(
        ((velocities > survey.VELOCITY_LIMIT_RAD_S).any(axis=1)).sum()
      ),
      "true_velocity_violation_risk": float(
        ((velocities > survey.VELOCITY_LIMIT_RAD_S).any(axis=1)).mean()
      ),
      "true_velocity_peak_rad_s": _quantiles(velocities.max(axis=1)),
    }
  result["first_event_delivered_impulse_target_over_control_mean_ratio"] = (
    result["target_utility"]["first_strike_delivered_n_s_mean"]
    / result["control_utility"]["first_strike_delivered_n_s_mean"]
  )
  return result


def _native_verdict() -> dict[str, object]:
  branch_texts = (
    (
      "Lower diagnostic violation probability and lower p95/p99 utilization, with retained "
      "task and velocity behavior: learned graded impulse reduction for this training seed."
    ),
    (
      "Lower mean but unchanged violation rate or tail: behavioral softening, not boundary "
      "confinement."
    ),
    (
      "Lower J3 exposure with higher exposure at another joint or in velocity: "
      "redistribution/trade-off, not clean enforcement."
    ),
    (
      "Active, winning impulse pressure with unchanged tail: calibration/treatment-strength "
      "problem."
    ),
    (
      "Absent or too-rare provisional-cap violations: report exactly, \"impulse constraint is "
      "empirically nonbinding under the surveyed population.\""
    ),
  )
  return {
    "selected_branch": 3,
    "text": branch_texts[2],
    "preregistered_branches_verbatim": [
      {"branch": index, "text": text, "selected": index == 3}
      for index, text in enumerate(branch_texts, start=1)
    ],
    "branch_5_scope_application": (
      "applies only to the fixed mean-action provisional-cap population; it is not the "
      "stochastic verdict"
    ),
    "native_observed_endpoint_valid": True,
    "full_contact_or_causal_softening_interpretation_authorized": False,
    "shadow_required_to_resolve_episode_horizon_confound": True,
    "event_dose_calibration_complete": False,
    "follow_up_plan_required": True,
    "imp_max_p_increase_authorized": False,
  }


def _compact_provenance(
  summaries: Mapping[str, Mapping[str, object]],
  manifests: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
  control = summaries["control"]
  target = summaries["target"]

  def population_protocol(
    role: str, population: str, seed: str | None = None
  ) -> dict[str, object]:
    payload = summaries[role]["populations"][population]
    if seed is not None:
      payload = payload[seed]
    return dict(payload["protocol"])

  stochastic = {
    str(seed): {
      role: population_protocol(role, "training_like_sampled", str(seed))
      for role in ("control", "target")
    }
    for seed in STOCHASTIC_SEEDS
  }
  return {
    "scheduler_execution_record": {
      "source": "authoritative Task-8 execution record",
      "slurm_job": "41386821_[0-1]",
      "control": {"state": "COMPLETED", "exit_code": "0:0", "stderr_empty": True},
      "target": {"state": "COMPLETED", "exit_code": "0:0", "stderr_empty": True},
    },
    "summary_declared_fail_closed_identities": {
      "validation_scope": (
        "declared evaluator identities checked against frozen constants and paired-summary "
        "contracts before analysis"
      ),
      "task": control["task"],
      "code_revision": control["code_revision"],
      "asset_revision": control["asset_revision"],
      "checkpoints": {
        role: {
          "role": summaries[role]["checkpoint"]["role"],
          "sha256": summaries[role]["checkpoint"]["sha256"],
        }
        for role in ("control", "target")
      },
      "evaluation_protocol": {
        "control": control["protocol"],
        "target": target["protocol"],
      },
      "populations": {
        "fixed_mean": {
          role: population_protocol(role, "fixed_mean")
          for role in ("control", "target")
        },
        "training_like_sampled": stochastic,
      },
    },
    "independently_rehashed_local_artifacts": manifests,
    "claim_boundary": {
      "checkpoint_binaries_locally_banked_and_rehashed": False,
      "checkpoint_identity_status": (
        "summary-declared role/hash checked fail-closed; checkpoint binaries are not in the "
        "downloaded evidence root"
      ),
      "latent_action_noise_equality_trace_auditable": False,
      "latent_action_noise_status": (
        "procedural matched-RNG contract declared by identical action stream seeds; no latent "
        "noise tensor is stored in the traces"
      ),
      "scheduler_status_derived_from_trace": False,
      "scheduler_status_source": "authoritative Task-8 execution record",
      "local_artifact_hash_status": (
        "all downloaded trace and summary bytes independently rehashed against their respective "
        "remote SHA256SUMS manifests"
      ),
    },
    "leaves": {"control": CONTROL_LEAF, "target": TARGET_LEAF},
    "all_manifest_hashes_match": True,
    "all_summary_and_trace_numeric_values_finite": True,
  }


def analyze_result_root(root: Path) -> dict[str, object]:
  """Validate the frozen leaves and build compact sufficient statistics."""
  root = Path(root).resolve(strict=True)
  survey = _survey_api()
  leaves = {
    "control": root / CONTROL_LEAF,
    "target": root / TARGET_LEAF,
  }
  manifests = {role: validate_leaf_manifest(path) for role, path in leaves.items()}
  summaries: dict[str, object] = {}
  for role, leaf in leaves.items():
    summaries[role] = json.loads((leaf / "evaluation" / "summary.json").read_text())
    _require_finite_json(summaries[role])
  survey.compare_policy_evaluations(
    summaries["control"], summaries["target"], expected_code_revision=EVALUATOR_REVISION
  )

  fixed_traces = {
    role: _load_trace(leaf / "evaluation" / "fixed_trace.npz")
    for role, leaf in leaves.items()
  }
  fixed = _population_result(
    summaries["control"]["populations"]["fixed_mean"],
    summaries["target"]["populations"]["fixed_mean"],
    fixed_traces["control"],
    fixed_traces["target"],
    survey=survey,
  )

  stochastic: dict[str, object] = {}
  pooled: dict[str, list[Mapping[str, np.ndarray]]] = {"control": [], "target": []}
  for seed in STOCHASTIC_SEEDS:
    traces = {
      role: _load_trace(
        leaf / "evaluation" / f"training_like_seed_{seed}_trace.npz"
      )
      for role, leaf in leaves.items()
    }
    for role in pooled:
      pooled[role].append(traces[role])
    stochastic[str(seed)] = _population_result(
      summaries["control"]["populations"]["training_like_sampled"][str(seed)],
      summaries["target"]["populations"]["training_like_sampled"][str(seed)],
      traces["control"],
      traces["target"],
      survey=survey,
    )

  return {
    "schema_version": 2,
    "purpose": "compact sufficient statistics for the frozen diagnostic-0.9 post-training pair",
    "provenance": _compact_provenance(summaries, manifests),
    "analysis_protocol": {
      "bootstrap_seed": BOOTSTRAP_SEED,
      "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
      "resampling_unit": "whole paired environment ID",
      "reads_are_inferential_units": False,
      "threshold_order": ["provisional_caps", "diagnostic_only"],
      "stochastic_replica_order": list(STOCHASTIC_SEEDS),
    },
    "fixed_mean": fixed,
    "training_like_replicas_before_pooling": stochastic,
    "pooled_training_like_descriptive_only": _pooled_descriptive(pooled, survey=survey),
    "native_verdict": _native_verdict(),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  payload = analyze_result_root(args.input_root)
  encoded = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n"
  args.output.write_text(encoded, encoding="utf-8")
  print(args.output)


if __name__ == "__main__":
  main()
