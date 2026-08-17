"""Tests for the compact diagnostic-0.9 post-training comparison."""

import importlib
import hashlib

import numpy as np
import pytest


def _trace_from_rho(rho: list[float]) -> dict[str, np.ndarray]:
  values = np.asarray(rho, dtype=np.float32)
  trace = np.zeros((2, values.size, 6), dtype=np.float32)
  trace[0, :, 0] = values
  return {
    "lambda_per_joint": trace,
    "episode_id": np.zeros((2, values.size), dtype=np.int64),
  }


def test_paired_bootstrap_resamples_whole_environment_ids_across_arms():
  """Independent arm resampling would invent nonzero risk-difference uncertainty."""
  try:
    analysis = importlib.import_module(
      "scripts.analyze_vic_impulse_diag90_500_evaluation"
    )
  except ModuleNotFoundError:
    pytest.fail("paired diagnostic evaluation analysis helper is missing")

  control = _trace_from_rho([0.5, 1.5, 0.6, 1.6])
  target = _trace_from_rho([0.8, 1.2, 0.9, 1.3])
  result = analysis.paired_initial_episode_bootstrap(
    control,
    target,
    caps=(1.0,) * 6,
    resamples=10_000,
    seed=2026081703,
  )

  assert result["resampling_unit"] == "whole paired environment ID"
  assert result["resamples"] == 10_000
  assert result["seed"] == 2026081703
  assert result["point"]["control_violation_risk"] == pytest.approx(0.5)
  assert result["point"]["target_violation_risk"] == pytest.approx(0.5)
  assert result["point"]["target_minus_control_violation_risk"] == pytest.approx(0.0)
  assert result["interval_95"]["target_minus_control_violation_risk"] == {
    "low": 0.0,
    "high": 0.0,
  }


def test_binary_risk_bootstrap_preserves_pairing_for_velocity_tradeoff():
  """Resampling velocity flags independently would break the matched-world contrast."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )

  result = analysis.paired_binary_risk_bootstrap(
    np.array([False, True, False, True]),
    np.array([False, True, False, True]),
    resamples=10_000,
    seed=2026081703,
  )

  assert result["point"]["target_minus_control_risk"] == 0.0
  assert result["interval_95"]["target_minus_control_risk"] == {
    "low": 0.0,
    "high": 0.0,
  }


def test_frozen_leaf_validation_rehashes_every_manifest_artifact(tmp_path):
  """Trusting manifest text without rehashing would accept changed evaluator evidence."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )
  leaf = tmp_path / "41386821_0_diag90_control"
  evaluation = leaf / "evaluation"
  evaluation.mkdir(parents=True)
  rows = []
  for name in analysis.EXPECTED_ARTIFACT_NAMES:
    path = evaluation / name
    path.write_bytes(f"frozen {name}".encode())
    rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  evaluation/{name}")
  (leaf / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")

  validated = analysis.validate_leaf_manifest(leaf)
  assert tuple(validated) == analysis.EXPECTED_ARTIFACT_NAMES

  (evaluation / analysis.EXPECTED_ARTIFACT_NAMES[0]).write_bytes(b"changed")
  with pytest.raises(ValueError, match="SHA-256"):
    analysis.validate_leaf_manifest(leaf)
