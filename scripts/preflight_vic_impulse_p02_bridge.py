#!/usr/bin/env python3
"""Validate the zero-learning p=0.2 impulse-CaT bridge against frozen traces."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import analyze_vic_impulse_diag90_500_evaluation as historical
from scripts import impulse_cat_activation_survey as survey


CANDIDATE_IMP_MAX_P = 0.2
EXPECTED_SEED2_TRACE_SHA256 = {
  "control": "a2666a6f9cde2f5607f8d7bc8cde696578f3aab65247166d20b6c530359c1547",
  "target": "62967852edd9e3a3abaa3dbf9de5b77faae1e0cc4854e9b41ead92c010c6b54d",
}
EXPECTED_CONTROL_ACTIVATION = {
  "fixed_mean": {
    "active_reads": 0,
    "distinct_positive_doses": 0,
    "impulse_winner_reads": 0,
  },
  "2": {
    "active_reads": 35,
    "distinct_positive_doses": 35,
    "impulse_winner_reads": 35,
  },
  "2026081701": {
    "active_reads": 30,
    "distinct_positive_doses": 29,
    "impulse_winner_reads": 30,
  },
  "2026081702": {
    "active_reads": 35,
    "distinct_positive_doses": 35,
    "impulse_winner_reads": 35,
  },
}
_TRACE_NAMES = {
  "fixed_mean": "fixed_trace.npz",
  "2": "training_like_seed_2_trace.npz",
  "2026081701": "training_like_seed_2026081701_trace.npz",
  "2026081702": "training_like_seed_2026081702_trace.npz",
}


def _load_json(value: Path | Mapping[str, object], *, label: str) -> dict[str, object]:
  if isinstance(value, Mapping):
    payload = dict(value)
  else:
    path = Path(value).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"{label} must contain a JSON object")
  historical._require_finite_json(payload)
  return payload


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} is missing or invalid")
  return value


def _population_payload(summary: Mapping[str, object], population: str) -> Mapping[str, Any]:
  populations = _require_mapping(summary.get("populations"), label="summary populations")
  if population == "fixed_mean":
    payload = populations.get("fixed_mean")
  else:
    stochastic = _require_mapping(
      populations.get("training_like_sampled"), label="stochastic populations"
    )
    payload = stochastic.get(population)
  payload = _require_mapping(payload, label=f"population {population}")
  return _require_mapping(
    payload.get("provisional_caps"), label=f"population {population} provisional caps"
  )


def _banked_population(
  banked_analysis: Mapping[str, object], population: str
) -> Mapping[str, Any]:
  if population == "fixed_mean":
    payload = banked_analysis.get("fixed_mean")
  else:
    stochastic = _require_mapping(
      banked_analysis.get("training_like_replicas_before_pooling"),
      label="banked stochastic populations",
    )
    payload = stochastic.get(population)
  payload = _require_mapping(payload, label=f"banked population {population}")
  thresholds = _require_mapping(
    payload.get("thresholds_in_required_order"),
    label=f"banked population {population} thresholds",
  )
  return _require_mapping(
    thresholds.get("provisional_caps"),
    label=f"banked population {population} provisional caps",
  )


def _numeric_tree_equal(actual: object, expected: object) -> bool:
  if isinstance(actual, Mapping) and isinstance(expected, Mapping):
    return actual.keys() == expected.keys() and all(
      _numeric_tree_equal(actual[key], expected[key]) for key in actual
    )
  if isinstance(actual, list) and isinstance(expected, list):
    return len(actual) == len(expected) and all(
      _numeric_tree_equal(left, right) for left, right in zip(actual, expected, strict=True)
    )
  if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
    return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-15))
  return actual == expected


def _require_descriptors(summary: Mapping[str, object], *, population: str) -> None:
  binding = _require_mapping(summary.get("binding"), label=f"{population} binding")
  for name in (
    "violating_reads",
    "violating_activation_windows",
    "activation_window_read_counts",
    "physical_event_read_counts",
  ):
    if name not in binding:
      raise ValueError(f"{population} binding is missing {name}")
  physical = _require_mapping(
    summary.get("physical_contact_duration_ms"),
    label=f"{population} physical-contact duration",
  )
  for name in (
    "events",
    "right_censored_events",
    "all_observed_prefixes",
    "completed_events_only",
    "cap_associated_observed_prefixes",
    "cap_associated_completed_only",
    "cap_associated_censored_prefixes",
  ):
    if name not in physical:
      raise ValueError(f"{population} physical-contact duration is missing {name}")


def _pressure_summary(values: np.ndarray) -> dict[str, float | None]:
  return historical._quantiles(np.asarray(values, dtype=np.float64))


def _candidate_population(
  trace: Mapping[str, np.ndarray],
  stored_summary: Mapping[str, object],
  *,
  population: str,
  candidate_imp_max_p: float,
) -> dict[str, object]:
  _require_descriptors(stored_summary, population=population)
  recomputed_summary = survey.summarize_population(
    dict(trace),
    caps=survey.PROVISIONAL_CAPS_N_M_S,
    first_episode_only=True,
  )
  candidates = _require_mapping(
    stored_summary.get("candidate_imp_max_p"),
    label=f"{population} stored candidates",
  )
  stored_candidate = candidates.get(str(candidate_imp_max_p))
  stored_candidate = _require_mapping(
    stored_candidate, label=f"{population} stored p=0.2 candidate"
  )
  historical._require_finite_json(stored_candidate, path=f"$.{population}.candidate")
  recomputed_candidate = recomputed_summary["candidate_imp_max_p"][
    str(candidate_imp_max_p)
  ]
  if not _numeric_tree_equal(stored_candidate, recomputed_candidate):
    raise ValueError(f"{population} stored p=0.2 candidate differs from raw-trace replay")

  lam = np.asarray(trace["lambda_per_joint"])
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  valid = episode_id == 0
  margins = survey._native_cap_margins(lam, survey.PROVISIONAL_CAPS_N_M_S)
  delta_impulse_per_joint = survey.shadow_impulse_cat(
    margins,
    tau=survey.CAT_TAU,
    seed=survey.IMPULSE_SEED,
    max_p=candidate_imp_max_p,
    valid=valid,
  )
  delta_impulse = delta_impulse_per_joint.max(axis=2)
  delta_velocity = np.asarray(trace["delta_velocity"], dtype=np.float64)
  if delta_velocity.shape != valid.shape or not np.isfinite(delta_velocity).all():
    raise ValueError(f"{population} delta_velocity is missing, nonfinite, or misaligned")
  combined = np.maximum(delta_velocity, delta_impulse)
  if not np.isfinite(delta_impulse).all() or not np.isfinite(combined).all():
    raise ValueError(f"{population} pressure replay is non-finite")
  active = valid & (delta_impulse > 0.0)
  impulse_wins = active & (delta_impulse > delta_velocity)
  velocity_masks = active & (delta_velocity > delta_impulse)
  ties = active & ~(impulse_wins | velocity_masks)
  active_reads = int(active.sum())
  distinct_positive = int(np.unique(delta_impulse[active]).size)
  if population != "fixed_mean":
    if active_reads == 0:
      raise ValueError(f"{population} p=0.2 pressure is nonbinding")
    if distinct_positive < 2:
      raise ValueError(f"{population} p=0.2 pressure is nonzero but not graded")
    if not bool(impulse_wins.any()):
      raise ValueError(f"{population} p=0.2 impulse pressure is completely masked by velocity")

  observed = {
    "active_reads": active_reads,
    "distinct_positive_doses": distinct_positive,
    "impulse_winner_reads": int(impulse_wins.sum()),
  }
  expected = EXPECTED_CONTROL_ACTIVATION[population]
  if observed != expected:
    raise ValueError(
      f"{population} frozen p=0.2 activation counts drifted: {observed} != {expected}"
    )
  if int(velocity_masks.sum()) != 0 or int(ties.sum()) != 0:
    raise ValueError(f"{population} frozen p=0.2 winner/masking counts drifted")

  binding = stored_summary["binding"]
  physical = stored_summary["physical_contact_duration_ms"]
  candidate_pressure = {
    "unit": "contiguous 50 Hz impulse-activation window",
    "unique_physical_event_dose": False,
    "observed_prefixes": list(
      recomputed_candidate["event_pressure_observed_prefixes"]
    ),
    "all_observed_summary": dict(
      recomputed_candidate["event_pressure_all_observed_summary"]
    ),
    "completed_only_summary": dict(
      recomputed_candidate["event_pressure_completed_only_summary"]
    ),
    "right_censored_windows": int(
      recomputed_candidate["right_censored_activation_windows"]
    ),
  }
  native_rho = historical._initial_episode_rho(
    trace, survey.PROVISIONAL_CAPS_N_M_S
  )
  observed_prefix = historical._observed_first_contact_prefix_utilization(
    trace, survey.PROVISIONAL_CAPS_N_M_S
  )
  return {
    "active_reads": active_reads,
    "distinct_positive_doses": distinct_positive,
    "winner_reads": {
      "impulse": int(impulse_wins.sum()),
      "velocity_masked": int(velocity_masks.sum()),
      "tie": int(ties.sum()),
    },
    "combined_delta_exact_max": bool(
      np.array_equal(combined, np.maximum(delta_velocity, delta_impulse))
    ),
    "delta_velocity": _pressure_summary(delta_velocity[valid]),
    "delta_impulse": _pressure_summary(delta_impulse[valid]),
    "combined_delta": _pressure_summary(combined[valid]),
    "activation_window_read_counts": list(binding["activation_window_read_counts"]),
    "activation_window_pressure": candidate_pressure,
    "physical_event_read_counts": list(binding["physical_event_read_counts"]),
    "physical_contact": dict(physical),
    "native_initial_episode_rho": _pressure_summary(native_rho),
    "observed_first_contact_prefix_rho": observed_prefix["rho"],
    "observed_first_contact_prefix": observed_prefix,
    "per_joint": recomputed_summary["per_joint"],
    "positive_margin_per_joint": {
      name: recomputed_summary["segment_compliance"]["per_joint"][name][
        "positive_margin_n_m_s"
      ]
      for name in survey.JOINT_NAMES
    },
  }


def _require_banked_match(
  banked: Mapping[str, object],
  role: str,
  population_result: Mapping[str, object],
  *,
  population: str,
) -> None:
  role_payload = _require_mapping(
    banked.get(role), label=f"banked analysis {population} {role}"
  )
  expected_native = _require_mapping(
    role_payload.get("rho"), label=f"banked analysis {population} {role} rho"
  )
  expected_prefix = _require_mapping(
    role_payload.get("observed_first_contact_prefix_utilization"),
    label=f"banked analysis {population} {role} observed prefix",
  )
  if not _numeric_tree_equal(
    population_result["native_initial_episode_rho"], expected_native
  ) or not _numeric_tree_equal(
    population_result["observed_first_contact_prefix"], expected_prefix
  ):
    raise ValueError(
      f"banked analysis {population} {role} exposure differs from frozen raw traces"
    )


def _direction_gate(
  control: Mapping[str, object], target: Mapping[str, object], *, population: str
) -> dict[str, object]:
  native = {
    quantile: float(target["native_initial_episode_rho"][quantile])
    - float(control["native_initial_episode_rho"][quantile])
    for quantile in ("p95", "p99")
  }
  prefix = {
    quantile: float(target["observed_first_contact_prefix_rho"][quantile])
    - float(control["observed_first_contact_prefix_rho"][quantile])
    for quantile in ("p95", "p99")
  }
  passed = all(value < 0.0 for value in (*native.values(), *prefix.values()))
  if not passed:
    raise ValueError(
      f"{population} exposure direction is not strictly lower in all four provisional-cap tails"
    )
  return {
    "native_initial_episode_rho_target_minus_control": native,
    "observed_first_contact_prefix_rho_target_minus_control": prefix,
    "all_four_strictly_negative": True,
  }


def evaluate_preflight(
  control_leaf: Path,
  historical_target_leaf: Path,
  banked_analysis: Path | Mapping[str, object],
  *,
  candidate_imp_max_p: float = CANDIDATE_IMP_MAX_P,
) -> dict[str, object]:
  """Recompute and validate the one approved lower-dose bridge preflight."""
  if type(candidate_imp_max_p) is not float or candidate_imp_max_p != 0.2:
    raise ValueError("bridge candidate imp_max_p must be exactly 0.2")
  leaves = {
    "control": Path(control_leaf).resolve(strict=True),
    "target": Path(historical_target_leaf).resolve(strict=True),
  }
  manifests = {
    role: historical.validate_leaf_manifest(leaf) for role, leaf in leaves.items()
  }
  analysis = _load_json(banked_analysis, label="banked historical analysis")
  provenance = _require_mapping(analysis.get("provenance"), label="banked provenance")
  banked_manifests = _require_mapping(
    provenance.get("independently_rehashed_local_artifacts"),
    label="banked artifact manifests",
  )
  if not _numeric_tree_equal(manifests, banked_manifests):
    raise ValueError("banked analysis artifact manifests differ from frozen leaves")

  summaries = {
    role: _load_json(leaf / "evaluation" / "summary.json", label=f"{role} summary")
    for role, leaf in leaves.items()
  }
  population_results: dict[str, dict[str, dict[str, object]]] = {}
  gates: dict[str, object] = {}
  for population, trace_name in _TRACE_NAMES.items():
    results: dict[str, dict[str, object]] = {}
    for role in ("control", "target"):
      trace = historical._load_trace(leaves[role] / "evaluation" / trace_name)
      stored = _population_payload(summaries[role], population)
      if role == "control":
        results[role] = _candidate_population(
          trace,
          stored,
          population=population,
          candidate_imp_max_p=candidate_imp_max_p,
        )
      else:
        native_rho = historical._initial_episode_rho(
          trace, survey.PROVISIONAL_CAPS_N_M_S
        )
        prefix = historical._observed_first_contact_prefix_utilization(
          trace, survey.PROVISIONAL_CAPS_N_M_S
        )
        results[role] = {
          "native_initial_episode_rho": _pressure_summary(native_rho),
          "observed_first_contact_prefix_rho": prefix["rho"],
          "observed_first_contact_prefix": prefix,
        }
    banked_population = _banked_population(analysis, population)
    for role in ("control", "target"):
      _require_banked_match(
        banked_population, role, results[role], population=population
      )
    gates[population] = _direction_gate(
      results["control"], results["target"], population=population
    )
    population_results[population] = results

  return {
    "schema_version": 1,
    "purpose": "zero-learning p=0.2 impulse-CaT bridge preflight",
    "candidate_imp_max_p": candidate_imp_max_p,
    "caps_n_m_s": list(survey.PROVISIONAL_CAPS_N_M_S),
    "caps_status": "project-defined simulation thresholds; not hardware limits",
    "fixed_mean": {
      **population_results["fixed_mean"]["control"],
      "descriptive_only": True,
      "empirically_nonbinding": True,
    },
    "training_like_sampled": {
      population: population_results[population]["control"]
      for population in ("2", "2026081701", "2026081702")
    },
    "exposure_direction_gate": {
      "comparison": "historical p=0.5 target minus p=0 control",
      "threshold": "provisional caps only",
      "populations": gates,
      "all_populations_pass": True,
    },
    "provenance": {
      "frozen_leaf_manifests": manifests,
      "banked_analysis_reconciled": True,
    },
    "claim_boundary": {
      "no_learning": True,
      "no_simulation": True,
      "dose_selected_or_optimized": False,
      "complete_physical_event_claim": False,
      "hardware_limit_claim": False,
    },
    "preflight_pass": True,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--control-leaf", required=True, type=Path)
  parser.add_argument("--historical-target-leaf", required=True, type=Path)
  parser.add_argument("--banked-analysis", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  output = args.output.absolute()
  if output.exists() or output.is_symlink():
    raise FileExistsError(f"refusing to overwrite preflight output: {output}")
  payload = evaluate_preflight(
    args.control_leaf, args.historical_target_leaf, args.banked_analysis
  )
  manifest_hashes = payload["provenance"]["frozen_leaf_manifests"]
  for role in ("control", "target"):
    actual = manifest_hashes[role]["training_like_seed_2_trace.npz"]["sha256"]
    if actual != EXPECTED_SEED2_TRACE_SHA256[role]:
      raise ValueError(f"{role} frozen seed-2 trace SHA-256 mismatch")
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
    json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
    encoding="utf-8",
  )
  print(output)


if __name__ == "__main__":
  main()
