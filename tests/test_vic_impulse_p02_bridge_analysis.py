"""Pre-registered gates for the provisional-cap impulse-CaT p=0.2 bridge."""

from __future__ import annotations

import copy
import importlib

import numpy as np
import pytest


analysis = importlib.import_module("scripts.analyze_vic_impulse_p02_bridge_evaluation")


def _passing_gate_metrics() -> dict[str, object]:
  return {
    "provisional_risk_difference_upper_97_5": -0.006,
    "rho": {
      "control": {"p95": 1.0, "p99": 1.0},
      "target": {"p95": 0.89, "p99": 0.89},
    },
    "per_joint": {
      "control_p99": [0.2, 0.09, 0.5, 0.1, 0.3, 0.4],
      "target_p99": [0.219, 1.0, 0.4, 0.109, 0.2, 0.3],
      "control_any_violation": [False, False, True, False, False, False],
      "target_any_violation": [False, False, False, False, False, False],
    },
    "velocity_risk_difference_upper_97_5": 0.001,
    "success_difference": -0.01,
    "productive_strike_difference": -0.01,
    "first_event_delivered_ratio_lower_97_5": 0.90,
    "identity_finiteness_native_claims": True,
  }


def test_gate_boundaries_encode_every_strict_and_inclusive_rule():
  result = analysis.evaluate_gate_metrics(_passing_gate_metrics())

  assert result["pass"] is True
  assert result["provisional_risk_reduction"]["comparison"] == "<"
  assert result["provisional_risk_reduction"]["threshold"] == -0.005
  assert result["global_rho_reduction"]["comparison"] == ">="
  assert result["joint_tail_noninferiority"]["comparison"] == "<"
  assert result["velocity_noninferiority"]["comparison"] == "<="
  assert result["utility_noninferiority"]["comparison"] == ">="
  assert result["delivered_impulse_retention"]["comparison"] == ">="


@pytest.mark.parametrize(
  ("mutation", "gate"),
  (
    ("risk_equal", "provisional_risk_reduction"),
    ("rho_below", "global_rho_reduction"),
    ("joint_equal", "joint_tail_noninferiority"),
    ("new_joint_violation", "joint_tail_noninferiority"),
    ("velocity_above", "velocity_noninferiority"),
    ("success_below", "utility_noninferiority"),
    ("productive_below", "utility_noninferiority"),
    ("delivered_below", "delivered_impulse_retention"),
    ("identity_false", "identity_finiteness_native_claims"),
  ),
)
def test_each_failed_boundary_fails_its_population_gate(mutation: str, gate: str):
  metrics = _passing_gate_metrics()
  if mutation == "risk_equal":
    metrics["provisional_risk_difference_upper_97_5"] = -0.005
  elif mutation == "rho_below":
    metrics["rho"]["target"]["p99"] = np.nextafter(0.9, np.inf)
  elif mutation == "joint_equal":
    metrics["per_joint"]["target_p99"][0] = np.nextafter(0.22, np.inf)
  elif mutation == "new_joint_violation":
    metrics["per_joint"]["target_any_violation"][1] = True
  elif mutation == "velocity_above":
    metrics["velocity_risk_difference_upper_97_5"] = 0.0010001
  elif mutation == "success_below":
    metrics["success_difference"] = -0.0100001
  elif mutation == "productive_below":
    metrics["productive_strike_difference"] = -0.0100001
  elif mutation == "delivered_below":
    metrics["first_event_delivered_ratio_lower_97_5"] = 0.8999999
  else:
    metrics["identity_finiteness_native_claims"] = False

  result = analysis.evaluate_gate_metrics(metrics)

  assert result["pass"] is False
  assert result[gate]["pass"] is False


def test_literal_strict_risk_boundary_uses_adjacent_floats_without_tolerance():
  below = _passing_gate_metrics()
  below["provisional_risk_difference_upper_97_5"] = np.nextafter(-0.005, -np.inf)
  above = _passing_gate_metrics()
  above["provisional_risk_difference_upper_97_5"] = np.nextafter(-0.005, np.inf)

  assert analysis.evaluate_gate_metrics(below)["provisional_risk_reduction"]["pass"] is True
  assert analysis.evaluate_gate_metrics(above)["provisional_risk_reduction"]["pass"] is False


def test_literal_rho_and_joint_boundaries_use_adjacent_floats_without_tolerance():
  rho_pass = _passing_gate_metrics()
  rho_pass["rho"]["target"]["p99"] = np.nextafter(0.9, -np.inf)
  rho_fail = _passing_gate_metrics()
  rho_fail["rho"]["target"]["p99"] = np.nextafter(0.9, np.inf)
  joint_pass = _passing_gate_metrics()
  joint_pass["per_joint"]["target_p99"][0] = np.nextafter(0.22, -np.inf)
  joint_fail = _passing_gate_metrics()
  joint_fail["per_joint"]["target_p99"][0] = np.nextafter(0.22, np.inf)

  assert analysis.evaluate_gate_metrics(rho_pass)["global_rho_reduction"]["pass"] is True
  assert analysis.evaluate_gate_metrics(rho_fail)["global_rho_reduction"]["pass"] is False
  assert analysis.evaluate_gate_metrics(joint_pass)["joint_tail_noninferiority"]["pass"] is True
  assert analysis.evaluate_gate_metrics(joint_fail)["joint_tail_noninferiority"]["pass"] is False


def test_literal_inclusive_velocity_utility_and_delivered_boundaries():
  passing = analysis.evaluate_gate_metrics(_passing_gate_metrics())
  assert passing["velocity_noninferiority"]["pass"] is True
  assert passing["utility_noninferiority"]["pass"] is True
  assert passing["delivered_impulse_retention"]["pass"] is True

  velocity = _passing_gate_metrics()
  velocity["velocity_risk_difference_upper_97_5"] = np.nextafter(0.001, np.inf)
  success = _passing_gate_metrics()
  success["success_difference"] = np.nextafter(-0.01, -np.inf)
  productive = _passing_gate_metrics()
  productive["productive_strike_difference"] = np.nextafter(-0.01, -np.inf)
  delivered = _passing_gate_metrics()
  delivered["first_event_delivered_ratio_lower_97_5"] = np.nextafter(0.90, -np.inf)

  assert analysis.evaluate_gate_metrics(velocity)["velocity_noninferiority"]["pass"] is False
  assert analysis.evaluate_gate_metrics(success)["utility_noninferiority"]["pass"] is False
  assert analysis.evaluate_gate_metrics(productive)["utility_noninferiority"]["pass"] is False
  assert analysis.evaluate_gate_metrics(delivered)["delivered_impulse_retention"]["pass"] is False


def test_delivered_impulse_bootstrap_resamples_paired_environment_means():
  result = analysis.paired_mean_ratio_bootstrap(
    np.array([1.0, 2.0, 3.0, 4.0]),
    np.array([0.9, 1.8, 2.7, 3.6]),
  )

  assert result["resampling_unit"] == "whole paired environment ID"
  assert result["controller_reads_are_inferential_units"] is False
  assert result["resamples"] == 10_000
  assert result["seed"] == 2026081703
  assert result["point_target_over_control_mean_ratio"] == pytest.approx(0.9)
  assert result["lower_97_5"] == pytest.approx(0.9)
  assert result["valid"] is True


def test_terminal_extraction_preserves_environment_identity_across_staggered_order():
  control = _trace(0.5)
  target = _trace(0.5)
  control["done"] = np.array(
    [[True, False, False, False], [False, True, True, True]], dtype=bool
  )
  target["done"] = np.array(
    [[False, True, True, True], [True, False, False, False]], dtype=bool
  )
  control["first_strike_delivered_n_s"] = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 3.0, 4.0]], dtype=np.float32
  )
  target["first_strike_delivered_n_s"] = np.array(
    [[0.0, 1.8, 2.7, 3.6], [0.9, 0.0, 0.0, 0.0]], dtype=np.float32
  )

  control_endpoint = analysis._terminal_endpoint_by_environment(
    control, "first_strike_delivered_n_s"
  )
  target_endpoint = analysis._terminal_endpoint_by_environment(
    target, "first_strike_delivered_n_s"
  )
  result = analysis.paired_mean_ratio_bootstrap(
    control_endpoint["values"], target_endpoint["values"]
  )

  np.testing.assert_array_equal(control_endpoint["env_ids"], np.arange(4))
  np.testing.assert_allclose(control_endpoint["values"], [1.0, 2.0, 3.0, 4.0])
  np.testing.assert_allclose(target_endpoint["values"], [0.9, 1.8, 2.7, 3.6])
  assert result["lower_97_5"] == pytest.approx(0.9)


@pytest.mark.parametrize(
  "control",
  (
    np.array([0.0, 0.0]),
    np.array([0.0, 1.0]),
  ),
)
def test_delivered_impulse_bootstrap_fails_on_nonpositive_observed_or_resampled_control(
  control: np.ndarray,
):
  result = analysis.paired_mean_ratio_bootstrap(control, np.ones_like(control))

  assert result["valid"] is False
  assert result["lower_97_5"] is None
  assert result["invalid_nonpositive_control_resamples"] >= 1


def _trace(lambda_scale: float, delivered_scale: float = 1.0) -> dict[str, np.ndarray]:
  steps, envs = 2, 4
  lam = np.zeros((steps, envs, 6), dtype=np.float32)
  lam[:, :, 0] = lambda_scale
  done = np.array([[False] * envs, [True] * envs])
  actions = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
  actions = np.broadcast_to(actions, (steps, envs, 12)).copy()
  contact = np.zeros((steps * 10, envs), dtype=bool)
  contact[10:, :] = True
  return {
    "lambda_per_joint": lam,
    "episode_id": np.zeros((steps, envs), dtype=np.int64),
    "delta_velocity": np.zeros((steps, envs), dtype=np.float32),
    "delta_impulse": np.zeros((steps, envs), dtype=np.float32),
    "delta": np.zeros((steps, envs), dtype=np.float32),
    "done": done,
    "success": done.copy(),
    "timeout": np.zeros_like(done),
    "first_strike_productive": done.copy(),
    "first_strike_delivered_n_s": np.where(done, delivered_scale, 0.0).astype(np.float32),
    "delivered_total_n_s": np.where(done, delivered_scale + 0.1, 0.0).astype(np.float32),
    "nail_depth_m": np.where(done, 0.032, 0.0).astype(np.float32),
    "substep_peak_qv_per_joint": np.ones((steps, envs, 6), dtype=np.float32),
    "vic_p": np.zeros((steps, envs, 6), dtype=np.float32),
    "vic_kp": np.full((steps, envs, 6), 1000.0, dtype=np.float32),
    "vic_kd": np.full((steps, envs, 6), 100.0, dtype=np.float32),
    "policy_action": actions,
    "substep_contact": contact,
    "substep_episode_id": np.zeros_like(contact, dtype=np.int64),
  }


def _population_summary() -> dict[str, object]:
  gains = {
    "segments_with_contact": 4,
    "right_censored_segments": 0,
    "records": [],
  }
  return {
    "provisional_caps": {
      "physical_contact_duration_ms": {
        "events": 4,
        "right_censored_events": 4,
        "all_observed_prefixes": {"median": 20.0, "p95": 20.0, "max": 20.0},
      },
      "binding": {
        "violating_reads": 8,
        "activation_window_read_counts": [2, 2, 2, 2],
        "physical_event_read_counts": [
          {"event_id": i, "exclusive_reads": 2, "overlapping_reads": 0}
          for i in range(4)
        ],
      },
      "utility": {"first_contact_vic_gains": gains},
    }
  }


def test_synthetic_population_passes_and_emits_every_mandatory_descriptor():
  control = _trace(1.2, 1.0)
  target = _trace(0.8, 0.95)
  result = analysis.evaluate_population_pair(
    control,
    target,
    _population_summary(),
    _population_summary(),
    inferential=True,
  )

  assert result["verdict"]["pass"] is True
  descriptors = result["secondary_descriptors"]
  for role in ("control", "target"):
    assert set(descriptors[role]) >= {
      "episode_duration_ms",
      "physical_contact_censoring",
      "controller_read_counts",
      "observed_first_contact_prefix_lambda",
      "nail_depth_m",
      "cumulative_delivered_impulse_n_s",
      "vic_gains",
      "policy_action",
    }
    assert tuple(descriptors[role]["policy_action"]) == tuple(
      f"action_{index}" for index in range(12)
    )
  assert result["claim_boundary"] == {
    "native_observed_exposure_only": True,
    "complete_contact_claim_authorized": False,
    "actuator_loading_or_hardware_safety_claim_authorized": False,
  }


def test_native_float32_margin_controls_exact_boundary_risk_and_new_joint_flags():
  caps = np.asarray(analysis.survey.PROVISIONAL_CAPS_N_M_S, dtype=np.float32)
  control = _trace(0.0)
  target = _trace(0.0)
  control["lambda_per_joint"][:, :, 0] = caps[0]
  target["lambda_per_joint"][:, :, 0] = caps[0]
  target["lambda_per_joint"][:, :, 1] = caps[1]

  exact = analysis.evaluate_population_pair(
    control, target, _population_summary(), _population_summary(), inferential=True
  )

  assert exact["endpoints"]["provisional_risk"]["point"] == {
    "control_risk": 0.0,
    "target_risk": 0.0,
    "target_minus_control_risk": 0.0,
    "target_over_control_risk": None,
  }
  assert exact["endpoints"]["per_joint"]["target_any_violation"] == [False] * 6

  target["lambda_per_joint"][:, :, 1] = np.nextafter(caps[1], np.float32(np.inf))
  above = analysis.evaluate_population_pair(
    control, target, _population_summary(), _population_summary(), inferential=True
  )

  assert above["endpoints"]["provisional_risk"]["point"]["target_risk"] == 1.0
  assert above["endpoints"]["per_joint"]["target_any_violation"][1] is True
  assert above["verdict"]["joint_tail_noninferiority"]["pass"] is False


def test_censored_population_emits_counts_and_fails_utility_delivered_and_identity_gates():
  control = _trace(1.2, 1.0)
  target = _trace(0.8, 0.95)
  target["done"][:, 0] = False
  target["success"][:, 0] = False
  target["first_strike_productive"][:, 0] = False

  result = analysis.evaluate_population_pair(
    control, target, _population_summary(), _population_summary(), inferential=True
  )

  target_duration = result["secondary_descriptors"]["target"]["episode_duration_ms"]
  assert target_duration["completed_environments"] == 3
  assert target_duration["right_censored_environments"] == 1
  assert result["endpoints"]["success_difference"] is None
  assert result["endpoints"]["productive_strike_difference"] is None
  ratio = result["endpoints"]["first_event_delivered_impulse_ratio"]
  assert ratio["valid"] is False
  assert ratio["failure_reason"] == "initial episode fragment is censored in at least one arm"
  assert result["verdict"]["utility_noninferiority"]["pass"] is False
  assert result["verdict"]["delivered_impulse_retention"]["pass"] is False
  assert result["verdict"]["identity_finiteness_native_claims"]["pass"] is False


def test_population_analysis_rejects_missing_mandatory_contact_censoring_descriptor():
  summary = _population_summary()
  summary["provisional_caps"].pop("physical_contact_duration_ms")

  with pytest.raises(ValueError, match="physical-contact censoring"):
    analysis.evaluate_population_pair(
      _trace(1.2),
      _trace(0.8, 0.95),
      summary,
      _population_summary(),
      inferential=True,
    )


def test_fixed_population_is_descriptive_and_stochastic_verdicts_are_not_pooled():
  passing = {"verdict": {"pass": True}}
  result = analysis.assemble_analysis(
    fixed_result={"descriptive": True},
    stochastic_results={str(seed): copy.deepcopy(passing) for seed in analysis.STOCHASTIC_SEEDS},
  )

  assert result["fixed_mean_descriptive_only"] == {"descriptive": True}
  assert list(result["training_like_by_seed"]) == [
    "2", "2026081701", "2026081702"
  ]
  assert all(
    population["verdict"]["pass"]
    for population in result["training_like_by_seed"].values()
  )
  assert "pooled" not in str(result).lower()


def test_assemble_analysis_rejects_a_missing_stochastic_population():
  with pytest.raises(ValueError, match="exact separate stochastic populations"):
    analysis.assemble_analysis(
      fixed_result={},
      stochastic_results={"2": {"verdict": {"pass": True}}},
    )
