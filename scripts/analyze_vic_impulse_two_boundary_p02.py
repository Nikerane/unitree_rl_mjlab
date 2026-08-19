#!/usr/bin/env python3
"""Descriptive frozen-policy analysis for the Z1 two-boundary p=.2 diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts import analyze_vic_impulse_diag90_500_evaluation as historical
from scripts import impulse_cat_activation_survey as survey


EXPECTED_ROLES = ("p02_uniform09", "p02_joint_stress")
EXPECTED_CODE_REVISION = "7642a0558970fb6ef450efe981809a0d48873456"
CAP_GEOMETRIES = {
  "own_training_caps": {
    role: list(survey.TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S[role])
    for role in EXPECTED_ROLES
  },
  "other_training_caps": {
    "p02_uniform09": list(survey.TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S["p02_joint_stress"]),
    "p02_joint_stress": list(survey.TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S["p02_uniform09"]),
  },
  "provisional_project_caps": list(survey.PROVISIONAL_CAPS_N_M_S),
}
STOCHASTIC_SEEDS = survey.POLICY_EVALUATION_STOCHASTIC_SEEDS


def claim_limits() -> dict[str, object]:
  """The non-negotiable scope of this single-seed simulation diagnostic."""
  return {
    "training_seeds": 1,
    "evaluation_replicas_are_training_seeds": False,
    "native_observed_horizon_only": True,
    "soft_pressure_not_clamp": True,
    "provisional_project_caps_not_manufacturer_limits": True,
    "hardware_safety_claim_authorized": False,
    "complete_contact_claim_authorized": False,
    "hard_enforcement_claim_authorized": False,
    "optimal_boundary_or_dose_claim_authorized": False,
    "individual_joint_causality_claim_authorized": False,
    "controller_reads_are_inferential_units": False,
  }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
  return historical._quantiles(np.asarray(values, dtype=np.float64))


def _valid_lambda(trace: Mapping[str, np.ndarray], *, first_episode_only: bool) -> np.ndarray:
  lam = np.asarray(trace["lambda_per_joint"])
  episode_id = np.asarray(trace["episode_id"], dtype=np.int64)
  if (
    lam.ndim != 3
    or episode_id.shape != lam.shape[:2]
    or lam.shape[2] != len(survey.JOINT_NAMES)
    or not np.issubdtype(lam.dtype, np.floating)
    or not np.isfinite(lam).all()
  ):
    raise ValueError("trace must contain finite aligned six-joint Lambda telemetry")
  valid = episode_id == 0 if first_episode_only else np.ones(episode_id.shape, dtype=bool)
  if not bool(valid.any()):
    raise ValueError("trace has no valid Lambda controller reads")
  return lam[valid].astype(np.float64)


def _raw_lambda_descriptors(
  trace: Mapping[str, np.ndarray], *, first_episode_only: bool
) -> dict[str, dict[str, float | None]]:
  values = _valid_lambda(trace, first_episode_only=first_episode_only)
  return {
    joint: _quantiles(values[:, index])
    for index, joint in enumerate(survey.JOINT_NAMES)
  }


def _descriptive_classification(summary: Mapping[str, object]) -> dict[str, bool]:
  """Label observed geometry without converting it into a pass/fail scientific verdict."""
  compliance = summary["segment_compliance"]
  per_joint = summary["per_joint"]
  if not isinstance(compliance, Mapping) or not isinstance(per_joint, Mapping):
    raise ValueError("survey summary lacks compliance geometry")
  violations = int(compliance["any_joint_violating_segments"])
  active_joints = sum(
    int(per_joint[name]["violating_reads"]) > 0 for name in survey.JOINT_NAMES
  )
  active = int(summary["binding"]["violating_reads"])
  return {
    "boundary_compliance": violations == 0,
    "partial_compliance": violations > 0 and active > 0,
    "distributed_load": active_joints > 1,
    "bottleneck_transfer": False,
    "global_suppression": False,
    "task_output_trade_off": False,
    "velocity_risk_transfer": False,
    "no_meaningful_response": active == 0,
  }


def _compact_survey_summary(summary: Mapping[str, object]) -> dict[str, object]:
  """Keep required event/attribution endpoints, not replay-sized read records."""
  binding = summary["binding"]
  physical = summary["physical_contact_duration_ms"]
  candidate = summary["candidate_imp_max_p"]["0.2"]
  utility = summary["utility"]
  if not all(isinstance(value, Mapping) for value in (binding, physical, candidate, utility)):
    raise ValueError("survey summary cannot be compacted")
  return {
    "caps_n_m_s": summary["caps_n_m_s"],
    "shape": summary["shape"],
    "actual_log_only_invariants": summary["actual_log_only_invariants"],
    "observed_peak_scope": summary["observed_peak_scope"],
    "observed_peak_segments": summary["observed_peak_segments"],
    "segment_compliance": summary["segment_compliance"],
    "per_joint": summary["per_joint"],
    "binding": {
      key: binding[key]
      for key in (
        "violating_reads",
        "violating_activation_windows",
        "associated_physical_events",
        "ambiguous_multi_event_reads",
      )
    }
    | {
      "activation_window_read_counts": _quantiles(
        np.asarray(binding["activation_window_read_counts"], dtype=np.float64)
      ),
      "physical_event_read_counts": {
        "events": len(binding["physical_event_read_counts"]),
        "exclusive_reads": _quantiles(
          np.asarray(
            [entry["exclusive_reads"] for entry in binding["physical_event_read_counts"]],
            dtype=np.float64,
          )
        ),
        "ambiguous_read_associations": _quantiles(
          np.asarray(
            [
              entry["ambiguous_read_associations"]
              for entry in binding["physical_event_read_counts"]
            ],
            dtype=np.float64,
          )
        ),
      },
    },
    "physical_contact_duration_ms": physical,
    "candidate_imp_max_p_0_2": {
      key: candidate[key]
      for key in (
        "active_reads",
        "delta_impulse_active_mean",
        "delta_impulse_max",
        "combined_delta_max",
        "impulse_winner_reads",
        "velocity_masked_reads",
        "tie_reads",
        "impulse_win_share",
        "velocity_mask_share",
        "event_pressure_all_observed_summary",
        "event_pressure_completed_only_summary",
        "right_censored_activation_windows",
      )
    }
    | {
      "reads_per_activation_window": _quantiles(
        np.asarray(candidate["reads_per_activation_window"], dtype=np.float64)
      ),
      "activation_windows": len(candidate["activation_window_summaries"]),
    },
    "utility": {
      "task_field_scope": utility["task_field_scope"],
      "terminal_counts": utility["terminal_counts"],
      "terminal_nail_depth_m": utility["terminal_nail_depth_m"],
      "terminal_delivered_total_n_s": utility["terminal_delivered_total_n_s"],
      "terminal_first_strike_delivered_n_s": utility["terminal_first_strike_delivered_n_s"],
      "velocity_limit_compliance": utility["velocity_limit_compliance"],
      "first_contact_vic_gains": historical._compact_gain_summary(
        utility["first_contact_vic_gains"]
      ),
    },
  }


def analyze_population(
  trace: Mapping[str, np.ndarray],
  *,
  caps: Sequence[float],
  first_episode_only: bool,
) -> dict[str, object]:
  """Compose survey measurement, preserving its native-dtype margin/event logic."""
  cap_tuple = tuple(float(value) for value in caps)
  summary = survey.summarize_population(
    dict(trace), caps=cap_tuple, first_episode_only=first_episode_only
  )
  utility = summary["utility"]
  if not isinstance(utility, Mapping):
    raise ValueError("survey summary lacks utility telemetry")
  compact = _compact_survey_summary(summary)
  return {
    "raw_lambda_n_m_s": _raw_lambda_descriptors(
      trace, first_episode_only=first_episode_only
    ),
    "survey_summary": compact,
    "counterfactual_cat_p02": compact["candidate_imp_max_p_0_2"],
    "true_velocity_risk": utility["velocity_limit_compliance"],
    "task_utility": {
      "terminal_counts": utility["terminal_counts"],
      "terminal_first_strike_delivered_n_s": utility[
        "terminal_first_strike_delivered_n_s"
      ],
      "terminal_delivered_total_n_s": utility["terminal_delivered_total_n_s"],
      "terminal_nail_depth_m": utility["terminal_nail_depth_m"],
      "first_contact_vic_gains": historical._compact_gain_summary(
        utility["first_contact_vic_gains"]
      ),
    },
    "classification": _descriptive_classification(summary),
  }


def _cap_cross_analysis(
  role: str, trace: Mapping[str, np.ndarray], *, first_episode_only: bool
) -> dict[str, object]:
  return {
    name: analyze_population(trace, caps=caps[role] if isinstance(caps, Mapping) else caps,
                             first_episode_only=first_episode_only)
    for name, caps in CAP_GEOMETRIES.items()
  }


def _raw_lambda_comparison(
  uniform: Mapping[str, object], stress: Mapping[str, object]
) -> dict[str, object]:
  result: dict[str, object] = {}
  for joint in survey.JOINT_NAMES:
    uniform_q = uniform["raw_lambda_n_m_s"][joint]
    stress_q = stress["raw_lambda_n_m_s"][joint]
    result[joint] = {
      "p02_uniform09": uniform_q,
      "p02_joint_stress": stress_q,
      "joint_stress_minus_uniform": {
        name: float(stress_q[name] - uniform_q[name])
        for name in ("p50", "p95", "p99", "max")
      },
    }
  return result


def _population_pair(
  traces: Mapping[str, Mapping[str, np.ndarray]], *, first_episode_only: bool
) -> dict[str, object]:
  if tuple(traces) != EXPECTED_ROLES:
    raise ValueError("population traces require the exact ordered two roles")
  roles = {
    role: _cap_cross_analysis(role, traces[role], first_episode_only=first_episode_only)
    for role in EXPECTED_ROLES
  }
  return {
    "descriptive_only": True,
    "controller_reads_are_inferential_units": False,
    "roles": roles,
    "raw_lambda_cross_policy": _raw_lambda_comparison(
      roles["p02_uniform09"]["own_training_caps"],
      roles["p02_joint_stress"]["own_training_caps"],
    ),
  }


def assemble_analysis(
  *, populations: Mapping[str, object], provenance: Mapping[str, object]
) -> dict[str, object]:
  expected_seeds = tuple(str(seed) for seed in STOCHASTIC_SEEDS)
  fixed = populations.get("fixed64_descriptive_only")
  stochastic = populations.get("training_like_by_seed")
  if not isinstance(fixed, Mapping) or not isinstance(stochastic, Mapping) or tuple(stochastic) != expected_seeds:
    raise ValueError("analysis requires separate fixed-64 and three stochastic populations")
  return {
    "schema_version": 1,
    "purpose": "descriptive Z1 two-boundary impulse-CaT p=.2 behavior diagnostic",
    "analysis_protocol": {
      "fixed64_descriptive_only": True,
      "stochastic_populations_reported_separately": True,
      "controller_reads_are_inferential_units": False,
      "cap_geometries": CAP_GEOMETRIES,
    },
    "populations": dict(populations),
    "provenance": dict(provenance),
    "claim_limits": claim_limits(),
  }


def _load_summary(leaf: Path) -> dict[str, object]:
  summary = json.loads((leaf / "evaluation" / "summary.json").read_text(encoding="utf-8"))
  historical._require_finite_json(summary)
  if summary.get("schema_version") != 2:
    raise ValueError("two-boundary evaluation requires summary schema version 2")
  return summary


def _validate_role_training_caps(role: str, summary: Mapping[str, object]) -> None:
  checkpoint = summary.get("checkpoint")
  if not isinstance(checkpoint, Mapping) or checkpoint.get("role") != role:
    raise ValueError("evaluation summary role mismatch")
  if checkpoint.get("sha256") != survey.EVALUATION_CHECKPOINTS[role]:
    raise ValueError("evaluation summary checkpoint hash mismatch")
  survey.validate_training_cap_identity(role, summary.get("training_cap_identity_n_m_s"))


def analyze_evaluation_leaves(
  leaves: Mapping[str, Path], *, expected_code_revision: str = EXPECTED_CODE_REVISION
) -> dict[str, object]:
  """Fail closed on both immutable leaves, then describe each population separately."""
  if tuple(leaves) != EXPECTED_ROLES:
    raise ValueError("analysis requires the exact ordered two-boundary roles")
  resolved = {role: Path(leaves[role]).resolve(strict=True) for role in EXPECTED_ROLES}
  if len(set(resolved.values())) != len(resolved):
    raise ValueError("each role must use a distinct immutable evaluation leaf")
  manifests = {role: historical.validate_leaf_manifest(leaf) for role, leaf in resolved.items()}
  summaries = {role: _load_summary(leaf) for role, leaf in resolved.items()}
  for role in EXPECTED_ROLES:
    _validate_role_training_caps(role, summaries[role])
  survey.compare_policy_evaluations(
    summaries["p02_uniform09"], summaries["p02_joint_stress"],
    expected_code_revision=expected_code_revision,
    expected_roles=EXPECTED_ROLES,
  )

  fixed_traces = {
    role: historical._load_trace(leaf / "evaluation" / "fixed_trace.npz")
    for role, leaf in resolved.items()
  }
  populations: dict[str, object] = {
    "fixed64_descriptive_only": _population_pair(fixed_traces, first_episode_only=True),
    "training_like_by_seed": {},
  }
  for seed in STOCHASTIC_SEEDS:
    traces = {
      role: historical._load_trace(
        resolved[role] / "evaluation" / f"training_like_seed_{seed}_trace.npz"
      )
      for role in EXPECTED_ROLES
    }
    populations["training_like_by_seed"][str(seed)] = _population_pair(
      traces, first_episode_only=True
    )
  provenance = {
    "expected_code_revision": expected_code_revision,
    "asset_revision": survey.EXPECTED_EVALUATION_ASSET_REVISION,
    "checkpoint_sha256": {role: survey.EVALUATION_CHECKPOINTS[role] for role in EXPECTED_ROLES},
    "training_cap_identity_n_m_s": {
      role: list(survey.TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S[role])
      for role in EXPECTED_ROLES
    },
    "live_imp_max_p": 0.0,
    "manifests": manifests,
    "manifest_sha256": {
      role: historical._sha256(leaf / "SHA256SUMS") for role, leaf in resolved.items()
    },
    "offline_log_only_control_starting_condition_replay": {
      "scope": "joint-stress cap vector replay before either policy was trained",
      "stochastic_initial_episodes_any_boundary_violation_range": [0.9875, 0.9912],
      "joint2_responsible_read_share_range": [0.757, 0.797],
      "joint6_responsible_read_share_range": [0.137, 0.145],
      "impulse_outer_winner_share_range": [0.9968, 0.9985],
      "interpretation": "starting-condition descriptor, not a paired control or stop gate",
    },
  }
  payload = assemble_analysis(populations=populations, provenance=provenance)
  historical._require_finite_json(payload)
  return payload


def encode_analysis(payload: Mapping[str, object]) -> str:
  historical._require_finite_json(payload)
  return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_analysis(output: Path, payload: Mapping[str, object]) -> None:
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(encode_analysis(payload), encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--uniform09-leaf", required=True, type=Path)
  parser.add_argument("--joint-stress-leaf", required=True, type=Path)
  parser.add_argument("--expected-code-revision", default=EXPECTED_CODE_REVISION)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  payload = analyze_evaluation_leaves(
    {"p02_uniform09": args.uniform09_leaf, "p02_joint_stress": args.joint_stress_leaf},
    expected_code_revision=args.expected_code_revision,
  )
  write_analysis(args.output, payload)
  print(args.output)


if __name__ == "__main__":
  main()
