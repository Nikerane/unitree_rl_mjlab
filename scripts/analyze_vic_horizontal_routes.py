#!/usr/bin/env python3
"""Thin route-aware analysis for the Z1 horizontal-route pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts import analyze_vic_impulse_diag90_500_evaluation as historical
from scripts import analyze_vic_impulse_p02_bridge_evaluation as bridge
from scripts import impulse_cat_activation_survey as survey
from scripts.evaluate_vic_horizontal_routes import (
  EXPECTED_ASSET_REVISION,
  POLICY_SPECS,
  ROUTE_LABEL_BY_SIGN,
  TRAINING_CAPS_N_M_S,
)


CORE_PHASE_MIN = 0.125
CORE_PHASE_MAX = 0.375
HORIZONTAL_DETOUR_M = 0.020


def _finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  if values.size == 0:
    return {"p05": None, "p50": None, "p95": None}
  if not np.isfinite(values).all():
    raise ValueError("route metric values must be finite")
  return {
    "p05": float(np.quantile(values, 0.05)),
    "p50": float(np.quantile(values, 0.50)),
    "p95": float(np.quantile(values, 0.95)),
  }


def _route_waypoints(straight: np.ndarray, phase: np.ndarray) -> np.ndarray:
  amplitude = HORIZONTAL_DETOUR_M * np.sin(2.0 * math.pi * phase) ** 2
  candidates = np.repeat(straight[:, None, :], 3, axis=1)
  candidates[:, 0, 0] -= amplitude
  candidates[:, 2, 0] += amplitude
  return candidates


def _wilson_lower(successes: int, total: int, *, z: float = 1.96) -> float | None:
  if total <= 0:
    return None
  proportion = successes / total
  denominator = 1.0 + z * z / total
  center = proportion + z * z / (2.0 * total)
  radius = z * math.sqrt(
    proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
  )
  return (center - radius) / denominator


def tracking_metrics(
  trace: Mapping[str, np.ndarray], *, forced_sign: int
) -> dict[str, object]:
  """Classify initial episodes using pre-contact samples from the separable detour core."""
  if type(forced_sign) is not int or forced_sign not in ROUTE_LABEL_BY_SIGN:
    raise ValueError("forced_sign must be one of -1, 0, +1")
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  scalar_shape = episode_id.shape
  if episode_id.ndim != 2:
    raise ValueError("episode_id must have shape (steps, envs)")
  route_sign = np.asarray(trace["route_sign"])
  phase = np.asarray(trace["strike_phase"], dtype=np.float64)
  eligible = np.asarray(trace["imitation_eligible"])
  head = np.asarray(trace["hammer_head_pos_w"], dtype=np.float64)
  assigned = np.asarray(trace["assigned_reference_waypoint_w"], dtype=np.float64)
  straight = np.asarray(trace["straight_reference_waypoint_w"], dtype=np.float64)
  if route_sign.shape != scalar_shape or phase.shape != scalar_shape:
    raise ValueError("route sign and phase must align with episode_id")
  if eligible.shape != scalar_shape or eligible.dtype != np.bool_:
    raise ValueError("imitation eligibility must be an aligned bool array")
  if any(value.shape != (*scalar_shape, 3) for value in (head, assigned, straight)):
    raise ValueError("route positions must have shape (steps, envs, 3)")
  if not all(np.isfinite(value).all() for value in (phase, head, assigned, straight)):
    raise ValueError("route telemetry must be finite")
  if not np.array_equal(route_sign, np.full(scalar_shape, forced_sign, dtype=route_sign.dtype)):
    raise ValueError("trace route signs differ from the forced route")

  initial_precontact = (episode_id == 0) & eligible
  core = initial_precontact & (phase >= CORE_PHASE_MIN) & (phase <= CORE_PHASE_MAX)
  assigned_error = np.linalg.norm(head - assigned, axis=2)
  core_errors = assigned_error[core]
  envs = scalar_shape[1]
  closest_counts = {label: 0 for label in ROUTE_LABEL_BY_SIGN.values()}
  margins: list[float] = []
  responses: list[float] = []
  episode_records: list[dict[str, object]] = []
  correct = 0
  ambiguous = 0
  episodes_with_core = 0
  for env_id in range(envs):
    mask = core[:, env_id]
    eligible_mask = initial_precontact[:, env_id]
    record: dict[str, object] = {
      "env_id": env_id,
      "eligible_precontact_reads": int(eligible_mask.sum()),
      "core_reads": int(mask.sum()),
      "max_precontact_phase": (
        float(np.max(phase[eligible_mask, env_id])) if bool(eligible_mask.any()) else None
      ),
      "assigned_rmse_m": None,
      "next_best_rmse_m": None,
      "assigned_vs_next_rmse_margin_m": None,
      "median_horizontal_response_m": None,
      "closest_route": None,
      "ambiguous_tie": False,
    }
    if not bool(mask.any()):
      episode_records.append(record)
      continue
    episodes_with_core += 1
    candidates = _route_waypoints(straight[mask, env_id], phase[mask, env_id])
    residual = head[mask, env_id, None, :] - candidates
    rmses = np.sqrt(np.mean(np.sum(residual * residual, axis=2), axis=0))
    minimum = float(np.min(rmses))
    tied = np.flatnonzero(np.isclose(rmses, minimum, rtol=1e-7, atol=1e-9))
    is_ambiguous = tied.size != 1
    closest_sign = None if is_ambiguous else int(tied[0]) - 1
    if is_ambiguous:
      ambiguous += 1
    else:
      closest_counts[ROUTE_LABEL_BY_SIGN[closest_sign]] += 1
      correct += closest_sign == forced_sign
    assigned_index = forced_sign + 1
    next_best = float(np.min(np.delete(rmses, assigned_index)))
    margin = next_best - float(rmses[assigned_index])
    response = float(np.median(head[mask, env_id, 0] - straight[mask, env_id, 0]))
    margins.append(margin)
    responses.append(response)
    record.update(
      {
        "assigned_rmse_m": float(rmses[assigned_index]),
        "next_best_rmse_m": next_best,
        "assigned_vs_next_rmse_margin_m": margin,
        "median_horizontal_response_m": response,
        "closest_route": (
          None if closest_sign is None else ROUTE_LABEL_BY_SIGN[closest_sign]
        ),
        "ambiguous_tie": is_ambiguous,
      }
    )
    episode_records.append(record)

  error_summary = _finite_quantiles(core_errors)
  error_summary["rmse"] = (
    float(np.sqrt(np.mean(core_errors * core_errors))) if core_errors.size else None
  )
  coverage = episodes_with_core / envs if envs else 0.0
  correct_fraction = correct / episodes_with_core if episodes_with_core else None
  wilson_lower = _wilson_lower(correct, episodes_with_core)
  margin_summary = _finite_quantiles(np.asarray(margins))
  response_summary = _finite_quantiles(np.asarray(responses))
  response_p50 = response_summary["p50"]
  correct_side = (
    None
    if response_p50 is None
    else (
      response_p50 < 0.0
      if forced_sign < 0
      else response_p50 > 0.0
      if forced_sign > 0
      else True
    )
  )
  assessable = coverage >= 0.90
  classification_reliable = correct_fraction is not None and correct_fraction >= 0.90
  above_chance = correct_fraction is not None and correct_fraction > 0.50
  wilson_above_chance = wilson_lower is not None and wilson_lower > 1.0 / 3.0
  positive_margin = margin_summary["p50"] is not None and margin_summary["p50"] > 0.0
  conditioning = bool(
    assessable and above_chance and wilson_above_chance and positive_margin and correct_side
  )
  return {
    "core_phase_interval": [CORE_PHASE_MIN, CORE_PHASE_MAX],
    "eligibility": {
      "initial_episode_precontact_reads": int(initial_precontact.sum()),
      "core_reads": int(core.sum()),
      "episodes_with_core": episodes_with_core,
      "population_environments": envs,
      "assessable_coverage": coverage,
      "route_progress_failure": episodes_with_core == 0,
    },
    "assigned_reference_error_m": error_summary,
    "episode_route_classification": {
      "closest_route_counts": closest_counts,
      "ambiguous_tie_episodes": ambiguous,
      "correct_episodes": correct,
      "correct_fraction": correct_fraction,
      "wilson_lower_95": wilson_lower,
      "assigned_vs_next_rmse_margin_m": margin_summary,
    },
    "horizontal_response_m": response_summary,
    "episode_records": episode_records,
    "diagnostic_rubric": {
      "assessable_coverage_ge_0_90": assessable,
      "assigned_classification_ge_0_90": classification_reliable,
      "classification_gt_chance": above_chance,
      "wilson_lower_gt_one_third": wilson_above_chance,
      "positive_median_margin": positive_margin,
      "correct_side_sign": correct_side,
      "route_conditioning_evidence": conditioning,
      "reliable_cell_before_cross_route_ordering": bool(
        assessable and classification_reliable and positive_margin and correct_side
      ),
    },
  }
def signed_route_ordering(route_results: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
  expected = ("rminus", "r0", "rplus")
  if tuple(route_results) != expected:
    raise ValueError("route results must be ordered R-, R0, R+")
  medians: dict[str, float] = {}
  for route in expected:
    tracking = route_results[route].get("tracking")
    if not isinstance(tracking, Mapping):
      raise ValueError("route result is missing tracking metrics")
    response = tracking.get("horizontal_response_m")
    value = response.get("p50") if isinstance(response, Mapping) else None
    if not isinstance(value, (int, float)) or not np.isfinite(value):
      return {
        "strict_rminus_lt_r0_lt_rplus": False,
        "reason": "one or more routes have no eligible core episode",
        "median_response_m": {route: None for route in expected},
        "adjacent_gaps_m": {"r0_minus_rminus": None, "rplus_minus_r0": None},
      }
    medians[route] = float(value)
  gaps = {
    "r0_minus_rminus": medians["r0"] - medians["rminus"],
    "rplus_minus_r0": medians["rplus"] - medians["r0"],
  }
  return {
    "strict_rminus_lt_r0_lt_rplus": all(value > 0.0 for value in gaps.values()),
    "median_response_m": medians,
    "adjacent_gaps_m": gaps,
  }


def assess_policy_population(
  route_results: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
  ordering = signed_route_ordering(route_results)
  cell_reliable: dict[str, bool] = {}
  conditioning: dict[str, bool] = {}
  for route in ("rminus", "r0", "rplus"):
    tracking = route_results[route].get("tracking")
    rubric = tracking.get("diagnostic_rubric") if isinstance(tracking, Mapping) else None
    if not isinstance(rubric, Mapping):
      raise ValueError("route result is missing its diagnostic rubric")
    cell_reliable[route] = rubric.get("reliable_cell_before_cross_route_ordering") is True
    conditioning[route] = rubric.get("route_conditioning_evidence") is True
  strict = ordering["strict_rminus_lt_r0_lt_rplus"] is True
  all_cells = all(cell_reliable.values())
  return {
    "strict_signed_ordering": strict,
    "all_route_cells_reliable": all_cells,
    "all_route_cells_above_chance": all(conditioning.values()),
    "reliable_route_conditioning": strict and all_cells,
    "route_cell_reliable": cell_reliable,
    "route_cell_conditioning_evidence": conditioning,
    "signed_ordering": ordering,
  }


def analyze_population(
  trace: Mapping[str, np.ndarray], *, forced_sign: int
) -> dict[str, object]:
  """Report route behavior first and reuse qualified safety/utility summaries."""
  summary = survey.summarize_population(
    dict(trace), caps=TRAINING_CAPS_N_M_S, first_episode_only=True
  )
  utility = summary["utility"]
  if not isinstance(utility, Mapping):
    raise ValueError("population summary is missing utility telemetry")
  gains = utility["first_contact_vic_gains"]
  if not isinstance(gains, Mapping):
    raise ValueError("population summary is missing gain telemetry")
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  initial = episode_id == 0
  observed: dict[str, np.ndarray] = {}
  for field in (
    "delivered_total_n_s",
    "first_strike_delivered_n_s",
    "nail_depth_m",
  ):
    values = np.asarray(trace[field], dtype=np.float64)
    if values.shape != episode_id.shape or not np.isfinite(values).all():
      raise ValueError(f"{field} must align with initial-episode telemetry")
    observed[field] = np.asarray(
      [values[initial[:, env_id], env_id].max() for env_id in range(values.shape[1])],
      dtype=np.float64,
    )
  success = np.asarray(trace["success"], dtype=bool)
  productive = np.asarray(trace["first_strike_productive"], dtype=bool)
  if success.shape != episode_id.shape or productive.shape != episode_id.shape:
    raise ValueError("success/productive fields must align with initial episodes")
  success_seen = sum(
    bool(success[initial[:, env_id], env_id].any()) for env_id in range(success.shape[1])
  )
  productive_seen = sum(
    bool(productive[initial[:, env_id], env_id].any())
    for env_id in range(productive.shape[1])
  )
  return {
    "tracking": tracking_metrics(trace, forced_sign=forced_sign),
    "task_utility": {
      "terminal_counts": utility["terminal_counts"],
      "terminal_nail_depth_m": utility["terminal_nail_depth_m"],
      "terminal_delivered_total_n_s": utility["terminal_delivered_total_n_s"],
      "terminal_first_strike_delivered_n_s": utility[
        "terminal_first_strike_delivered_n_s"
      ],
      "initial_episode_observed": {
        "segments": episode_id.shape[1],
        "success_seen": success_seen,
        "productive_seen": productive_seen,
        "delivered_total_peak_n_s": survey._finite_quantiles(
          observed["delivered_total_n_s"]
        ),
        "first_strike_delivered_peak_n_s": survey._finite_quantiles(
          observed["first_strike_delivered_n_s"]
        ),
        "nail_depth_peak_m": survey._finite_quantiles(observed["nail_depth_m"]),
      },
    },
    "impulse_caps": {
      "caps_n_m_s": summary["caps_n_m_s"],
      "segment_compliance": summary["segment_compliance"],
      "per_joint": summary["per_joint"],
    },
    "true_velocity_risk": utility["velocity_limit_compliance"],
    "episode_duration_ms": bridge._observed_duration(trace),
    "vic_gains": historical._compact_gain_summary(gains),
  }


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _load_summary(leaf: Path) -> dict[str, object]:
  summary = json.loads((leaf / "evaluation" / "summary.json").read_text(encoding="utf-8"))
  historical._require_finite_json(summary)
  return summary


def _validate_summary(
  summary: Mapping[str, object], *, role: str, route: str, expected_code_revision: str
) -> int:
  if summary.get("schema_version") != 1:
    raise ValueError("horizontal evaluation summary schema mismatch")
  spec = POLICY_SPECS[role]
  checkpoint = summary.get("checkpoint")
  if not isinstance(checkpoint, Mapping) or checkpoint.get("role") != role:
    raise ValueError("horizontal evaluation checkpoint role mismatch")
  if checkpoint.get("sha256") != spec["sha256"]:
    raise ValueError("horizontal evaluation checkpoint hash mismatch")
  if summary.get("task") != spec["task"]:
    raise ValueError("horizontal evaluation task mismatch")
  sign = next(value for value, label in ROUTE_LABEL_BY_SIGN.items() if label == route)
  if summary.get("route") != {"label": route, "sign": sign}:
    raise ValueError("horizontal evaluation route mismatch")
  if summary.get("code_revision") != expected_code_revision:
    raise ValueError("horizontal evaluation code revision mismatch")
  if summary.get("asset_revision") != EXPECTED_ASSET_REVISION:
    raise ValueError("horizontal evaluation asset revision mismatch")
  if summary.get("training_cap_identity_n_m_s") != list(TRAINING_CAPS_N_M_S):
    raise ValueError("horizontal evaluation training-cap identity mismatch")
  protocol = summary.get("protocol")
  if not isinstance(protocol, Mapping):
    raise ValueError("horizontal evaluation protocol is missing")
  if protocol.get("actor_only_checkpoint_load") is not True or protocol.get(
    "live_imp_max_p"
  ) != 0.0:
    raise ValueError("horizontal evaluation is not actor-only/log-only")
  if protocol.get("fixed_seed") != survey.FIXED_SEED or protocol.get(
    "stochastic_seeds"
  ) != list(survey.POLICY_EVALUATION_STOCHASTIC_SEEDS):
    raise ValueError("horizontal evaluation population seeds mismatch")
  populations = summary.get("populations")
  if not isinstance(populations, Mapping):
    raise ValueError("horizontal evaluation populations are missing")
  sampled = populations.get("training_like_sampled")
  expected_seeds = tuple(str(seed) for seed in survey.POLICY_EVALUATION_STOCHASTIC_SEEDS)
  if not isinstance(sampled, Mapping) or tuple(sampled) != expected_seeds:
    raise ValueError("horizontal evaluation stochastic populations mismatch")
  expected_entries = (
    (populations.get("fixed_mean"), survey.FIXED_SEED, "fixed_trace.npz"),
    *(
      (sampled[str(seed)], seed, f"training_like_seed_{seed}_trace.npz")
      for seed in survey.POLICY_EVALUATION_STOCHASTIC_SEEDS
    ),
  )
  for population, seed, trace_name in expected_entries:
    if not isinstance(population, Mapping) or population.get("trace") != trace_name:
      raise ValueError("horizontal evaluation trace identity mismatch")
    population_protocol = population.get("protocol")
    if not isinstance(population_protocol, Mapping):
      raise ValueError("horizontal population protocol is missing")
    if population_protocol.get("seed") != seed or population_protocol.get(
      "forced_route_sign"
    ) != sign:
      raise ValueError("horizontal population seed/route mismatch")
    if population_protocol.get("actor_only_checkpoint_load") is not True:
      raise ValueError("horizontal population was not actor-only")
    cat_replay = population_protocol.get("cat_replay")
    if not isinstance(cat_replay, Mapping) or cat_replay.get("imp_max_p_live") != 0.0:
      raise ValueError("horizontal population live impulse p is not zero")
    if population_protocol.get("training_config") != {
      "imp_max_p": 0.2,
      "imp_limit_n_m_s": list(TRAINING_CAPS_N_M_S),
    }:
      raise ValueError("horizontal population training config mismatch")
    population_hash = population_protocol.get("initial_population_sha256")
    if not isinstance(population_hash, str) or len(population_hash) != 64 or any(
      character not in "0123456789abcdef" for character in population_hash
    ):
      raise ValueError("horizontal population hash is not lowercase SHA-256")
  return sign


def analyze_evaluation_leaves(
  leaves: Mapping[str, Mapping[str, Path]], *, expected_code_revision: str
) -> dict[str, object]:
  """Validate six immutable cells and report fixed/seed populations separately."""
  expected_roles = tuple(POLICY_SPECS)
  expected_routes = ("rminus", "r0", "rplus")
  if tuple(leaves) != expected_roles or any(
    tuple(leaves[role]) != expected_routes for role in expected_roles
  ):
    raise ValueError("analysis requires the exact ordered two-policy by three-route leaves")
  if len(expected_code_revision) != 40 or any(
    character not in "0123456789abcdef" for character in expected_code_revision
  ):
    raise ValueError("expected code revision must be full lowercase 40-hex")
  resolved = {
    role: {route: Path(leaves[role][route]).resolve(strict=True) for route in expected_routes}
    for role in expected_roles
  }
  flat = [leaf for routes in resolved.values() for leaf in routes.values()]
  if len(set(flat)) != 6:
    raise ValueError("each policy-route cell must use a distinct immutable leaf")
  manifests: dict[str, dict[str, object]] = {}
  summaries: dict[str, dict[str, dict[str, object]]] = {}
  for role in expected_roles:
    manifests[role] = {}
    summaries[role] = {}
    for route in expected_routes:
      leaf = resolved[role][route]
      manifests[role][route] = historical.validate_leaf_manifest(leaf)
      summary = _load_summary(leaf)
      _validate_summary(
        summary, role=role, route=route, expected_code_revision=expected_code_revision
      )
      summaries[role][route] = summary

  # All six cells use the same physical population/RNG identity; only policy and forced sign vary.
  for population_key in ("fixed_mean", *map(str, survey.POLICY_EVALUATION_STOCHASTIC_SEEDS)):
    hashes = []
    for role in expected_roles:
      for route in expected_routes:
        populations = summaries[role][route]["populations"]
        population = (
          populations["fixed_mean"]
          if population_key == "fixed_mean"
          else populations["training_like_sampled"][population_key]
        )
        hashes.append(population["protocol"]["initial_population_sha256"])
    if len(set(hashes)) != 1:
      raise ValueError("horizontal cells do not share the same initial population")

  policies: dict[str, object] = {}
  for role in expected_roles:
    fixed_routes = {
      route: analyze_population(
        historical._load_trace(resolved[role][route] / "evaluation/fixed_trace.npz"),
        forced_sign=next(sign for sign, label in ROUTE_LABEL_BY_SIGN.items() if label == route),
      )
      for route in expected_routes
    }
    sampled: dict[str, object] = {}
    for seed in survey.POLICY_EVALUATION_STOCHASTIC_SEEDS:
      routes = {
        route: analyze_population(
          historical._load_trace(
            resolved[role][route]
            / f"evaluation/training_like_seed_{seed}_trace.npz"
          ),
          forced_sign=next(
            sign for sign, label in ROUTE_LABEL_BY_SIGN.items() if label == route
          ),
        )
        for route in expected_routes
      }
      sampled[str(seed)] = {"routes": routes, "assessment": assess_policy_population(routes)}
    policies[role] = {
      "fixed64": {"routes": fixed_routes, "assessment": assess_policy_population(fixed_routes)},
      "training_like_by_seed": sampled,
    }

  payload = {
    "schema_version": 1,
    "purpose": "diagnostic frozen-policy horizontal route-conditioning assessment",
    "rubric": {
      "core_phase_interval": [CORE_PHASE_MIN, CORE_PHASE_MAX],
      "minimum_assessable_coverage": 0.90,
      "minimum_reliable_classification": 0.90,
      "route_conditioning_classification_strictly_above": 0.50,
      "wilson_lower_95_strictly_above": 1.0 / 3.0,
      "ties_are_ambiguous_and_incorrect": True,
      "strict_cross_route_signed_ordering_required": True,
      "diagnostic_not_hypothesis_test": True,
    },
    "policies": policies,
    "provenance": {
      "expected_code_revision": expected_code_revision,
      "asset_revision": EXPECTED_ASSET_REVISION,
      "checkpoint_sha256": {
        role: spec["sha256"] for role, spec in POLICY_SPECS.items()
      },
      "training_caps_n_m_s": list(TRAINING_CAPS_N_M_S),
      "live_imp_max_p": 0.0,
      "manifests": manifests,
      "manifest_sha256": {
        role: {
          route: _sha256(resolved[role][route] / "SHA256SUMS")
          for route in expected_routes
        }
        for role in expected_roles
      },
    },
    "claim_limits": {
      "training_seeds": 1,
      "evaluation_replicas_are_training_seeds": False,
      "soft_pressure_not_clamp": True,
      "hard_enforcement_claim_authorized": False,
      "hardware_safety_claim_authorized": False,
      "trajectory_generalization_claim_authorized": False,
    },
  }
  historical._require_finite_json(payload)
  return payload


def encode_analysis(payload: Mapping[str, object]) -> str:
  historical._require_finite_json(payload)
  return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  for role in POLICY_SPECS:
    for route in ("rminus", "r0", "rplus"):
      parser.add_argument(f"--{role.replace('_', '-')}-{route}-leaf", required=True, type=Path)
  parser.add_argument("--expected-code-revision", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  values = vars(args)
  leaves = {
    role: {
      route: values[f"{role}_{route}_leaf"]
      for route in ("rminus", "r0", "rplus")
    }
    for role in POLICY_SPECS
  }
  payload = analyze_evaluation_leaves(
    leaves, expected_code_revision=args.expected_code_revision
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(encode_analysis(payload), encoding="utf-8")
  print(args.output)


if __name__ == "__main__":
  main()
