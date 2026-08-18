"""Preregistered analysis contract for the four-policy impulse-CaT dose curve."""

from __future__ import annotations

import importlib
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import impulse_cat_activation_survey as survey


analysis = importlib.import_module("scripts.analyze_vic_impulse_dose_curve")


def test_dose_curve_roles_and_hashes_are_exact_and_ordered():
  assert analysis.EXPECTED_ROLES == (
    "dose_p0_control",
    "dose_p01_target",
    "dose_p02_target",
    "dose_p03_target",
  )
  assert dict(analysis.EXPECTED_CHECKPOINTS) == {
    "dose_p0_control": (
      "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3"
    ),
    "dose_p01_target": (
      "92f1d97c8ff1476cb26c0b648478a0bc3c522e4e1eb7087389fb8e0d6bf73f86"
    ),
    "dose_p02_target": (
      "57000e958bbafa2c62929652d3b76fd6ed571c9867bee3735c14baf0ca57d8de"
    ),
    "dose_p03_target": (
      "4c0a665fffc077d488630a28b258c1593050f969b4f6227147c3594097dc4efd"
    ),
  }
  assert dict(analysis.EXPECTED_IMP_MAX_P) == {
    "dose_p0_control": 0.0,
    "dose_p01_target": 0.1,
    "dose_p02_target": 0.2,
    "dose_p03_target": 0.3,
  }


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


@pytest.mark.parametrize(
  ("field", "failure_value", "gate"),
  (
    ("risk", -0.005, "provisional_risk_reduction"),
    ("rho", np.nextafter(0.9, np.inf), "global_rho_reduction"),
    ("joint", np.nextafter(0.22, np.inf), "joint_tail_noninferiority"),
    ("velocity", np.nextafter(0.001, np.inf), "velocity_noninferiority"),
    ("success", np.nextafter(-0.01, -np.inf), "utility_noninferiority"),
    ("productive", np.nextafter(-0.01, -np.inf), "utility_noninferiority"),
    ("delivered", np.nextafter(0.90, -np.inf), "delivered_impulse_retention"),
  ),
)
def test_dose_gate_composition_preserves_bridge_boundaries(
  field: str, failure_value: float, gate: str
):
  passing = _passing_gate_metrics()
  failing = _passing_gate_metrics()
  if field == "risk":
    failing["provisional_risk_difference_upper_97_5"] = failure_value
  elif field == "rho":
    failing["rho"]["target"]["p99"] = failure_value
  elif field == "joint":
    failing["per_joint"]["target_p99"][0] = failure_value
  elif field == "velocity":
    failing["velocity_risk_difference_upper_97_5"] = failure_value
  elif field == "success":
    failing["success_difference"] = failure_value
  elif field == "productive":
    failing["productive_strike_difference"] = failure_value
  else:
    failing["first_event_delivered_ratio_lower_97_5"] = failure_value

  assert analysis.evaluate_gate_metrics(passing)["pass"] is True
  assert analysis.evaluate_gate_metrics(failing)[gate]["pass"] is False


def test_dose_gate_composition_preserves_new_joint_and_identity_failures():
  new_joint = _passing_gate_metrics()
  new_joint["per_joint"]["target_any_violation"][1] = True
  identity = _passing_gate_metrics()
  identity["identity_finiteness_native_claims"] = False

  assert analysis.evaluate_gate_metrics(new_joint)["joint_tail_noninferiority"][
    "pass"
  ] is False
  assert analysis.evaluate_gate_metrics(identity)[
    "identity_finiteness_native_claims"
  ]["pass"] is False


def _passing_pair_result() -> dict[str, object]:
  return {
    "fixed_mean_descriptive_only": {
      "descriptive_only": True,
      "secondary_descriptors": {"control": {}, "target": {}},
    },
    "training_like_by_seed": {
      str(seed): {"verdict": {"pass": True}}
      for seed in (2, 2026081701, 2026081702)
    },
  }


@pytest.mark.parametrize(
  "failed_role", ("dose_p01_target", "dose_p02_target", "dose_p03_target")
)
def test_each_dose_fails_independently_without_pooling_or_rescue(failed_role: str):
  pairs = {
    role: _passing_pair_result()
    for role in analysis.EXPECTED_ROLES[1:]
  }
  pairs[failed_role]["training_like_by_seed"]["2026081701"]["verdict"][
    "pass"
  ] = False

  result = analysis.assemble_analysis(pair_results=copy.deepcopy(pairs))

  assert list(result["dose_results"]) == list(analysis.EXPECTED_ROLES[1:])
  for role, dose in result["dose_results"].items():
    assert dose["overall_verdict"]["pass"] is (role != failed_role)
    assert list(dose["training_like_by_seed"]) == ["2", "2026081701", "2026081702"]
  assert result["claim_limits"] == {
    "training_seeds": 1,
    "native_observed_horizon_only": True,
    "soft_pressure_not_clamp": True,
    "provisional_project_caps_not_manufacturer_limits": True,
    "hardware_safety_claim_authorized": False,
    "complete_contact_claim_authorized": False,
    "optimal_dose_claim_authorized": False,
  }
  encoded = str(result).lower()
  assert "pooled" not in encoded
  assert "rescue" not in encoded


def test_assemble_analysis_rejects_missing_extra_or_misordered_doses():
  pairs = {role: _passing_pair_result() for role in analysis.EXPECTED_ROLES[1:]}
  for invalid in (
    {key: value for key, value in pairs.items() if key != "dose_p02_target"},
    {**pairs, "unexpected": _passing_pair_result()},
    dict(reversed(tuple(pairs.items()))),
  ):
    with pytest.raises(ValueError, match="exact ordered dose roles"):
      analysis.assemble_analysis(pair_results=invalid)


CODE_REVISION = "a" * 40


def _population_protocol(seed: int, *, fixed: bool) -> dict[str, object]:
  return {
    "seed": seed,
    "num_envs": survey.FIXED_ENVS if fixed else survey.TRAINING_LIKE_ENVS,
    "control_steps": 2 if fixed else survey.TRAINING_LIKE_STEPS,
    "policy_mode": "mean" if fixed else "sampled",
    "auto_reset": not fixed,
    "initial_population_sha256": survey.EXPECTED_FIXED_POPULATION_SHA256,
    "rng_streams": {
      "reset": seed + survey.RESET_RNG_OFFSET,
      "observation": seed + survey.OBSERVATION_RNG_OFFSET,
      "action": seed + survey.ACTION_RNG_OFFSET,
    },
    "cat_replay": {
      "tau": survey.CAT_TAU,
      "min_p": survey.CAT_MIN_P,
      "imp_seed": survey.IMPULSE_SEED,
      "imp_max_p_live": 0.0,
    },
    "velocity_cat": {
      "limit_rad_s": survey.VELOCITY_LIMIT_RAD_S,
      "max_p": survey.VELOCITY_MAX_P,
      "detection": survey.VELOCITY_DETECTION,
    },
    "physics_dt_s": survey.PHYSICS_DT_S,
    "control_decimation": survey.CONTROL_DECIMATION,
    "impulse_window_substeps": survey.IMPULSE_WINDOW_SUBSTEPS,
    "contact_row_diagnostic_enabled": False,
  }


def _trace(lambda_value: float, *, delivered: float = 1.0) -> dict[str, np.ndarray]:
  steps, envs = 2, 4
  lam = np.zeros((steps, envs, 6), dtype=np.float32)
  lam[:, :, 0] = lambda_value
  done = np.array([[False] * envs, [True] * envs], dtype=bool)
  contact = np.zeros((steps * survey.CONTROL_DECIMATION, envs), dtype=bool)
  contact[survey.CONTROL_DECIMATION :, :] = True
  actions = np.broadcast_to(
    np.arange(12, dtype=np.float32).reshape(1, 1, 12), (steps, envs, 12)
  ).copy()
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
    "first_strike_delivered_n_s": np.where(done, delivered, 0.0).astype(np.float32),
    "delivered_total_n_s": np.where(done, delivered + 0.1, 0.0).astype(np.float32),
    "nail_depth_m": np.where(done, 0.032, 0.0).astype(np.float32),
    "substep_peak_qv_per_joint": np.ones((steps, envs, 6), dtype=np.float32),
    "vic_p": np.zeros((steps, envs, 6), dtype=np.float32),
    "vic_kp": np.full((steps, envs, 6), 1000.0, dtype=np.float32),
    "vic_kd": np.full((steps, envs, 6), 100.0, dtype=np.float32),
    "policy_action": actions,
    "substep_contact": contact,
    "substep_episode_id": np.zeros_like(contact, dtype=np.int64),
  }


def _threshold_summary(caps: tuple[float, ...], *, violating: bool) -> dict[str, object]:
  segments = 4
  violating_segments = segments if violating else 0
  gains = {"segments_with_contact": 4, "right_censored_segments": 0, "records": []}
  return {
    "caps_n_m_s": list(caps),
    "segment_compliance": {
      "segments": segments,
      "any_joint_violating_segments": violating_segments,
      "any_joint_violation_rate": violating_segments / segments,
      "max_joint_utilization": {
        "p50": 1.2 if violating else 0.8,
        "p95": 1.2 if violating else 0.8,
        "p99": 1.2 if violating else 0.8,
        "max": 1.2 if violating else 0.8,
      },
    },
    "binding": {
      "violating_reads": 8 if violating else 0,
      "activation_window_read_counts": [2, 2, 2, 2] if violating else [],
      "physical_event_read_counts": [
        {"event_id": index, "exclusive_reads": 2, "overlapping_reads": 0}
        for index in range(4)
      ],
    },
    "physical_contact_duration_ms": {
      "events": 4,
      "right_censored_events": 4,
      "all_observed_prefixes": {"median": 20.0, "p95": 20.0, "max": 20.0},
    },
    "utility": {"first_contact_vic_gains": gains},
  }


def _summary(role: str, *, control: bool) -> dict[str, object]:
  def population(seed: int, *, fixed: bool) -> dict[str, object]:
    return {
      "protocol": _population_protocol(seed, fixed=fixed),
      "provisional_caps": _threshold_summary(
        survey.PROVISIONAL_CAPS_N_M_S, violating=control
      ),
      "diagnostic_only": _threshold_summary(
        survey.DIAGNOSTIC_LIMITS_N_M_S, violating=control
      ),
    }

  return {
    "schema_version": 2,
    "task": survey.VIC_TASK,
    "checkpoint": {"role": role, "sha256": analysis.EXPECTED_CHECKPOINTS[role]},
    "code_revision": CODE_REVISION,
    "asset_revision": survey.EXPECTED_EVALUATION_ASSET_REVISION,
    "protocol": {
      "live_imp_max_p": 0.0,
      "fixed_seed": survey.FIXED_SEED,
      "stochastic_seeds": list(survey.POLICY_EVALUATION_STOCHASTIC_SEEDS),
      "threshold_summary_order": ["provisional_caps", "diagnostic_only"],
      "primary_statistical_unit": "initial episode segment",
      "controller_reads_are_independent": False,
    },
    "populations": {
      "fixed_mean": population(survey.FIXED_SEED, fixed=True),
      "training_like_sampled": {
        str(seed): population(seed, fixed=False)
        for seed in survey.POLICY_EVALUATION_STOCHASTIC_SEEDS
      },
    },
  }


def _write_leaf(tmp_path: Path, role: str) -> Path:
  leaf = tmp_path / role
  evaluation = leaf / "evaluation"
  evaluation.mkdir(parents=True)
  control = role == "dose_p0_control"
  trace = _trace(1.2 if control else 0.8, delivered=1.0 if control else 0.95)
  names = (
    "fixed_trace.npz",
    "training_like_seed_2_trace.npz",
    "training_like_seed_2026081701_trace.npz",
    "training_like_seed_2026081702_trace.npz",
  )
  for name in names:
    np.savez(evaluation / name, **trace)
  (evaluation / "summary.json").write_text(
    json.dumps(_summary(role, control=control), sort_keys=True, allow_nan=False) + "\n"
  )
  _refresh_manifest(leaf)
  return leaf


def _refresh_manifest(leaf: Path) -> None:
  evaluation = leaf / "evaluation"
  artifacts = (
    "fixed_trace.npz",
    "training_like_seed_2_trace.npz",
    "training_like_seed_2026081701_trace.npz",
    "training_like_seed_2026081702_trace.npz",
    "summary.json",
  )
  (leaf / "SHA256SUMS").write_text(
    "".join(
      f"{hashlib.sha256((evaluation / name).read_bytes()).hexdigest()}  evaluation/{name}\n"
      for name in artifacts
    )
  )


def _four_leaves(tmp_path: Path) -> dict[str, Path]:
  return {role: _write_leaf(tmp_path, role) for role in analysis.EXPECTED_ROLES}


def test_four_leaf_all_pass_curve_is_finite_complete_and_byte_deterministic(tmp_path: Path):
  leaves = _four_leaves(tmp_path)

  first = analysis.analyze_evaluation_curve(
    leaves, expected_code_revision=CODE_REVISION
  )
  second = analysis.analyze_evaluation_curve(
    leaves, expected_code_revision=CODE_REVISION
  )

  assert all(
    result["overall_verdict"]["pass"]
    for result in first["dose_results"].values()
  )
  assert first["provenance"]["expected_roles"] == list(analysis.EXPECTED_ROLES)
  assert first["provenance"]["imp_max_p_during_training"] == dict(
    analysis.EXPECTED_IMP_MAX_P
  )
  assert set(first["provenance"]["independently_rehashed_artifacts"]) == set(
    analysis.EXPECTED_ROLES
  )
  assert first["provenance"]["manifest_sha256"] == {
    role: hashlib.sha256((leaf / "SHA256SUMS").read_bytes()).hexdigest()
    for role, leaf in leaves.items()
  }
  for dose in first["dose_results"].values():
    assert dose["fixed_mean_descriptive_only"]["descriptive_only"] is True
    for population in dose["training_like_by_seed"].values():
      assert set(population["secondary_descriptors"]["target"]) >= {
        "episode_duration_ms",
        "physical_contact_censoring",
        "controller_read_counts",
        "observed_first_contact_prefix_lambda",
        "nail_depth_m",
        "cumulative_delivered_impulse_n_s",
        "vic_gains",
        "policy_action",
      }
      assert set(population["endpoints"]) >= {
        "provisional_risk",
        "rho",
        "per_joint",
        "true_velocity_risk",
        "success_difference",
        "productive_strike_difference",
        "first_event_delivered_impulse_ratio",
      }
  assert analysis.encode_analysis(first) == analysis.encode_analysis(second)
  json.loads(analysis.encode_analysis(first))


def test_curve_rejects_missing_extra_misordered_or_duplicate_leaf_roles(tmp_path: Path):
  leaves = _four_leaves(tmp_path)
  invalid_inputs = (
    {role: leaf for role, leaf in leaves.items() if role != "dose_p02_target"},
    {**leaves, "unexpected": leaves["dose_p03_target"]},
    dict(reversed(tuple(leaves.items()))),
    {**leaves, "dose_p01_target": leaves["dose_p0_control"]},
  )
  for invalid in invalid_inputs:
    with pytest.raises(ValueError, match="exact ordered policy roles|distinct evaluation leaf"):
      analysis.analyze_evaluation_curve(
        invalid, expected_code_revision=CODE_REVISION
      )


def test_curve_rehashes_every_raw_artifact_and_rejects_manifest_drift(tmp_path: Path):
  leaves = _four_leaves(tmp_path)
  trace = leaves["dose_p01_target"] / "evaluation" / "fixed_trace.npz"
  trace.write_bytes(trace.read_bytes() + b"drift")

  with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
    analysis.analyze_evaluation_curve(leaves, expected_code_revision=CODE_REVISION)


@pytest.mark.parametrize(
  ("mutation", "message"),
  (
    ("schema", "schema version 2"),
    ("task", "exact task"),
    ("role", "ordered control then target"),
    ("checkpoint", "checkpoint SHA"),
    ("code", "same code revision"),
    ("asset", "frozen asset revision"),
    ("live", "same evaluation protocol"),
    ("caps", "exact cap vector"),
    ("population", "same initial population hash"),
    ("rng", "RNG streams"),
    ("nonfinite", "non-finite JSON"),
  ),
)
def test_curve_fails_closed_on_summary_identity_or_protocol_drift(
  tmp_path: Path, mutation: str, message: str
):
  leaves = _four_leaves(tmp_path)
  leaf = leaves["dose_p01_target"]
  path = leaf / "evaluation" / "summary.json"
  payload = json.loads(path.read_text())
  if mutation == "schema":
    payload["schema_version"] = 1
  elif mutation == "task":
    payload["task"] = "wrong-task"
  elif mutation == "role":
    payload["checkpoint"]["role"] = "dose_p0_control"
  elif mutation == "checkpoint":
    payload["checkpoint"]["sha256"] = "0" * 64
  elif mutation == "code":
    payload["code_revision"] = "b" * 40
  elif mutation == "asset":
    payload["asset_revision"] = "b" * 40
  elif mutation == "live":
    payload["protocol"]["live_imp_max_p"] = 0.1
  elif mutation == "caps":
    payload["populations"]["fixed_mean"]["provisional_caps"]["caps_n_m_s"][0] = 0.81
  elif mutation == "population":
    payload["populations"]["training_like_sampled"]["2"]["protocol"][
      "initial_population_sha256"
    ] = "b" * 64
  elif mutation == "rng":
    payload["populations"]["training_like_sampled"]["2"]["protocol"][
      "rng_streams"
    ]["action"] += 1
  else:
    payload["protocol"]["bad"] = float("nan")
  path.write_text(json.dumps(payload, allow_nan=True))
  _refresh_manifest(leaf)

  with pytest.raises(ValueError, match=message):
    analysis.analyze_evaluation_curve(leaves, expected_code_revision=CODE_REVISION)


@pytest.mark.parametrize("mutation", ("nonfinite", "live_impulse", "wrong_delta"))
def test_curve_rejects_nonfinite_or_invalid_live_cat_traces(
  tmp_path: Path, mutation: str
):
  leaves = _four_leaves(tmp_path)
  leaf = leaves["dose_p01_target"]
  path = leaf / "evaluation" / "training_like_seed_2_trace.npz"
  with np.load(path, allow_pickle=False) as archive:
    trace = {name: archive[name] for name in archive.files}
  if mutation == "nonfinite":
    trace["lambda_per_joint"][0, 0, 0] = np.nan
  elif mutation == "live_impulse":
    trace["delta_impulse"][0, 0] = 0.1
  else:
    trace["delta"][0, 0] = 0.1
  np.savez(path, **trace)
  _refresh_manifest(leaf)

  with pytest.raises(ValueError, match="non-finite trace field|log-only|exact max soft-OR"):
    analysis.analyze_evaluation_curve(leaves, expected_code_revision=CODE_REVISION)


def test_censored_utility_fragment_fails_only_its_dose(tmp_path: Path):
  leaves = _four_leaves(tmp_path)
  leaf = leaves["dose_p01_target"]
  path = leaf / "evaluation" / "training_like_seed_2_trace.npz"
  with np.load(path, allow_pickle=False) as archive:
    trace = {name: archive[name] for name in archive.files}
  trace["done"][:, 0] = False
  trace["success"][:, 0] = False
  trace["first_strike_productive"][:, 0] = False
  np.savez(path, **trace)
  _refresh_manifest(leaf)

  result = analysis.analyze_evaluation_curve(
    leaves, expected_code_revision=CODE_REVISION
  )

  assert result["dose_results"]["dose_p01_target"]["overall_verdict"]["pass"] is False
  assert result["dose_results"]["dose_p02_target"]["overall_verdict"]["pass"] is True
  assert result["dose_results"]["dose_p03_target"]["overall_verdict"]["pass"] is True


def test_cli_requires_four_immutable_role_flags_and_regenerates_identical_bytes(
  tmp_path: Path,
):
  leaves = _four_leaves(tmp_path / "leaves")
  outputs = (tmp_path / "first.json", tmp_path / "second.json")
  command = [
    sys.executable,
    str(Path(analysis.__file__)),
    "--dose-p0-control-leaf",
    str(leaves["dose_p0_control"]),
    "--dose-p01-target-leaf",
    str(leaves["dose_p01_target"]),
    "--dose-p02-target-leaf",
    str(leaves["dose_p02_target"]),
    "--dose-p03-target-leaf",
    str(leaves["dose_p03_target"]),
    "--expected-code-revision",
    CODE_REVISION,
  ]
  for output in outputs:
    result = subprocess.run(
      [*command, "--output", str(output)],
      cwd=Path(__file__).resolve().parents[1],
      capture_output=True,
      text=True,
      timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(output)

  assert outputs[0].read_bytes() == outputs[1].read_bytes()
