"""Contract for the two-boundary p=.2 frozen-policy evidence bank."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from scripts import impulse_cat_activation_survey as survey


def test_two_boundary_analysis_declares_the_exact_roles_and_cap_geometries():
  analysis = importlib.import_module("scripts.analyze_vic_impulse_two_boundary_p02")

  assert analysis.EXPECTED_ROLES == ("p02_uniform09", "p02_joint_stress")
  assert analysis.CAP_GEOMETRIES == {
    "own_training_caps": {
      "p02_uniform09": [0.738, 1.476, 0.738, 0.738, 0.738, 0.738],
      "p02_joint_stress": [0.369, 0.246, 0.738, 0.369, 0.246, 0.0164],
    },
    "other_training_caps": {
      "p02_uniform09": [0.369, 0.246, 0.738, 0.369, 0.246, 0.0164],
      "p02_joint_stress": [0.738, 1.476, 0.738, 0.738, 0.738, 0.738],
    },
    "provisional_project_caps": list(survey.PROVISIONAL_CAPS_N_M_S),
  }


def _trace(lambda_values: tuple[float, float]) -> dict[str, np.ndarray]:
  steps, envs = 2, 2
  lam = np.zeros((steps, envs, 6), dtype=np.float32)
  lam[:, :, 0] = lambda_values[0]
  lam[:, :, 1] = lambda_values[1]
  contact = np.ones((steps * survey.CONTROL_DECIMATION, envs), dtype=bool)
  action = np.broadcast_to(
    np.arange(12, dtype=np.float32).reshape(1, 1, 12), (steps, envs, 12)
  ).copy()
  return {
    "lambda_per_joint": lam,
    "episode_id": np.zeros((steps, envs), dtype=np.int64),
    "done": np.array([[False, False], [True, True]], dtype=bool),
    "delta_velocity": np.full((steps, envs), 0.1, dtype=np.float32),
    "delta_impulse": np.zeros((steps, envs), dtype=np.float32),
    "delta": np.full((steps, envs), 0.1, dtype=np.float32),
    "success": np.array([[False, False], [True, True]], dtype=bool),
    "timeout": np.zeros((steps, envs), dtype=bool),
    "first_strike_productive": np.array([[False, False], [True, True]], dtype=bool),
    "first_strike_delivered_n_s": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    "delivered_total_n_s": np.array([[0.0, 0.0], [1.1, 1.1]], dtype=np.float32),
    "nail_depth_m": np.array([[0.0, 0.0], [0.03, 0.03]], dtype=np.float32),
    "substep_peak_qv_per_joint": np.ones((steps, envs, 6), dtype=np.float32),
    "vic_p": np.zeros((steps, envs, 6), dtype=np.float32),
    "vic_kp": np.full((steps, envs, 6), 1000.0, dtype=np.float32),
    "vic_kd": np.full((steps, envs, 6), 100.0, dtype=np.float32),
    "policy_action": action,
    "substep_contact": contact,
    "substep_episode_id": np.zeros_like(contact, dtype=np.int64),
    "substep_rolling_per_joint": np.repeat(lam, survey.CONTROL_DECIMATION, axis=0),
  }


def test_population_cross_analysis_reuses_survey_and_reports_required_descriptors():
  analysis = importlib.import_module("scripts.analyze_vic_impulse_two_boundary_p02")

  result = analysis.analyze_population(
    _trace((0.8, 0.2)),
    caps=(0.738, 1.476, 0.738, 0.738, 0.738, 0.738),
    first_episode_only=True,
  )

  assert result["raw_lambda_n_m_s"]["joint1"] == {
    "p50": pytest.approx(0.8),
    "p95": pytest.approx(0.8),
    "p99": pytest.approx(0.8),
    "max": pytest.approx(0.8),
  }
  assert result["survey_summary"]["actual_log_only_invariants"] == {
    "delta_impulse_identically_zero": True,
    "combined_delta_exact_max": True,
  }
  assert result["survey_summary"]["binding"]["violating_reads"] == 4
  assert result["true_velocity_risk"]["limit_rad_s"] == survey.VELOCITY_LIMIT_RAD_S
  assert result["classification"]["boundary_compliance"] is False
  assert result["classification"]["partial_compliance"] is True


def test_assembly_requires_separate_fixed_and_three_stochastic_populations():
  analysis = importlib.import_module("scripts.analyze_vic_impulse_two_boundary_p02")
  fake = {"fixed64_descriptive_only": {}, "training_like_by_seed": {}}

  with pytest.raises(ValueError, match="separate fixed-64"):
    analysis.assemble_analysis(populations=fake, provenance={})


def test_claim_boundary_rejects_hard_enforcement_and_read_level_inference():
  analysis = importlib.import_module("scripts.analyze_vic_impulse_two_boundary_p02")
  limits = analysis.claim_limits()

  assert limits["training_seeds"] == 1
  assert limits["soft_pressure_not_clamp"] is True
  assert limits["hardware_safety_claim_authorized"] is False
  assert limits["complete_contact_claim_authorized"] is False
  assert limits["controller_reads_are_inferential_units"] is False
