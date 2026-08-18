"""Tests for the zero-learning impulse-CaT p=0.2 bridge preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import analyze_vic_impulse_diag90_500_evaluation as historical_analysis
from scripts import impulse_cat_activation_survey as survey


_LEAF_NAMES = {
  "control": "41386821_0_diag90_control",
  "target": "41386821_1_diag90_target",
}
_POPULATIONS = {
  "fixed_mean": ("fixed_trace.npz", 64, 0, 0),
  "2": ("training_like_seed_2_trace.npz", 4096, 35, 35),
  "2026081701": (
    "training_like_seed_2026081701_trace.npz",
    4096,
    30,
    29,
  ),
  "2026081702": (
    "training_like_seed_2026081702_trace.npz",
    4096,
    35,
    35,
  ),
}


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_trace(
  *,
  envs: int,
  active_reads: int,
  distinct_positive_doses: int,
  target: bool,
  velocity_delta: float = 0.0,
) -> dict[str, np.ndarray]:
  steps = 2
  joint_shape = (steps, envs, 6)
  scalar_shape = (steps, envs)
  lambda_per_joint = np.zeros(joint_shape, dtype=np.float32)
  lambda_per_joint[0, :, 0] = np.float32(0.4 if target else 0.6)
  if active_reads:
    margins = np.linspace(0.01, 0.18, distinct_positive_doses, dtype=np.float32)
    if distinct_positive_doses < active_reads:
      margins = np.concatenate(
        (margins, np.repeat(margins[-1], active_reads - distinct_positive_doses))
      )
    lambda_per_joint[0, :active_reads, 0] = np.float32(0.82) + margins

  episode_id = np.zeros(scalar_shape, dtype=np.int64)
  done = np.zeros(scalar_shape, dtype=bool)
  done[1] = True
  success = done.copy()
  timeout = np.zeros(scalar_shape, dtype=bool)
  productive = done.copy()
  delta_velocity = np.full(scalar_shape, velocity_delta, dtype=np.float32)
  delta_impulse = np.zeros(scalar_shape, dtype=np.float32)

  substep_shape = (steps * survey.CONTROL_DECIMATION, envs)
  substep_contact = np.zeros(substep_shape, dtype=bool)
  contacted_envs = min(envs, max(active_reads, 100))
  substep_contact[0, :contacted_envs] = True
  substep_episode_id = np.zeros(substep_shape, dtype=np.int64)
  rolling = np.zeros((*substep_shape, 6), dtype=np.float32)
  rolling[0] = lambda_per_joint[0]

  return {
    "lambda_per_joint": lambda_per_joint,
    "episode_id": episode_id,
    "done": done,
    "delta_velocity": delta_velocity,
    "delta_impulse": delta_impulse,
    "delta": delta_velocity.copy(),
    "substep_contact": substep_contact,
    "substep_episode_id": substep_episode_id,
    "substep_rolling_per_joint": rolling,
    "success": success,
    "timeout": timeout,
    "first_strike_productive": productive,
    "nail_depth_m": np.zeros(scalar_shape, dtype=np.float32),
    "delivered_total_n_s": np.zeros(scalar_shape, dtype=np.float32),
    "first_strike_delivered_n_s": np.zeros(scalar_shape, dtype=np.float32),
    "substep_peak_qv_per_joint": np.zeros(joint_shape, dtype=np.float32),
    "vic_p": np.zeros(joint_shape, dtype=np.float32),
    "vic_kp": np.zeros(joint_shape, dtype=np.float32),
    "vic_kd": np.zeros(joint_shape, dtype=np.float32),
  }


def _population_summary(trace: dict[str, np.ndarray]) -> dict[str, object]:
  return survey.summarize_population(
    trace,
    caps=survey.PROVISIONAL_CAPS_N_M_S,
    first_episode_only=True,
  )


def _banked_population(
  control_trace: dict[str, np.ndarray], target_trace: dict[str, np.ndarray]
) -> dict[str, object]:
  def role(trace: dict[str, np.ndarray]) -> dict[str, object]:
    rho = historical_analysis._initial_episode_rho(
      trace, survey.PROVISIONAL_CAPS_N_M_S
    )
    return {
      "rho": historical_analysis._quantiles(rho),
      "observed_first_contact_prefix_utilization": (
        historical_analysis._observed_first_contact_prefix_utilization(
          trace, survey.PROVISIONAL_CAPS_N_M_S
        )
      ),
    }

  return {
    "thresholds_in_required_order": {
      "provisional_caps": {
        "control": role(control_trace),
        "target": role(target_trace),
      }
    }
  }


def _write_manifest(leaf: Path) -> dict[str, dict[str, object]]:
  rows = []
  result = {}
  for name in historical_analysis.EXPECTED_ARTIFACT_NAMES:
    artifact = leaf / "evaluation" / name
    digest = _sha256(artifact)
    rows.append(f"{digest}  evaluation/{name}")
    result[name] = {"sha256": digest, "size_bytes": artifact.stat().st_size}
  (leaf / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
  return result


def _build_evidence(root: Path) -> tuple[Path, Path, Path]:
  role_traces: dict[str, dict[str, dict[str, np.ndarray]]] = {}
  manifests = {}
  for role, leaf_name in _LEAF_NAMES.items():
    target = role == "target"
    leaf = root / leaf_name
    evaluation = leaf / "evaluation"
    evaluation.mkdir(parents=True)
    populations: dict[str, object] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    for population, (filename, envs, active_reads, distinct_doses) in _POPULATIONS.items():
      trace = _make_trace(
        envs=envs,
        active_reads=0 if target else active_reads,
        distinct_positive_doses=0 if target else distinct_doses,
        target=target,
      )
      traces[population] = trace
      np.savez_compressed(evaluation / filename, **trace)
      payload = {"provisional_caps": _population_summary(trace)}
      if population == "fixed_mean":
        populations[population] = payload
      else:
        populations.setdefault("training_like_sampled", {})[population] = payload
    summary = {"populations": populations}
    (evaluation / "summary.json").write_text(
      json.dumps(summary, allow_nan=False), encoding="utf-8"
    )
    role_traces[role] = traces
    manifests[role] = _write_manifest(leaf)

  analysis = {
    "schema_version": 2,
    "provenance": {"independently_rehashed_local_artifacts": manifests},
    "fixed_mean": _banked_population(
      role_traces["control"]["fixed_mean"], role_traces["target"]["fixed_mean"]
    ),
    "training_like_replicas_before_pooling": {
      seed: _banked_population(
        role_traces["control"][seed], role_traces["target"][seed]
      )
      for seed in ("2", "2026081701", "2026081702")
    },
  }
  analysis_path = root / "analysis.json"
  analysis_path.write_text(json.dumps(analysis, allow_nan=False), encoding="utf-8")
  return root / _LEAF_NAMES["control"], root / _LEAF_NAMES["target"], analysis_path


@pytest.fixture(scope="module")
def frozen_evidence(tmp_path_factory):
  return _build_evidence(tmp_path_factory.mktemp("p02_preflight_frozen"))


def _copy_evidence(
  frozen_evidence: tuple[Path, Path, Path], destination: Path
) -> tuple[Path, Path, Path]:
  control, target, analysis = frozen_evidence
  copied_control = destination / control.name
  copied_target = destination / target.name
  shutil.copytree(control, copied_control)
  shutil.copytree(target, copied_target)
  copied_analysis = destination / analysis.name
  shutil.copy2(analysis, copied_analysis)
  return copied_control, copied_target, copied_analysis


def _rewrite_summary(leaf: Path, mutate) -> None:
  path = leaf / "evaluation" / "summary.json"
  payload = json.loads(path.read_text(encoding="utf-8"))
  mutate(payload)
  path.write_text(json.dumps(payload), encoding="utf-8")
  _write_manifest(leaf)


def _refresh_banked_manifest(analysis_path: Path, role: str, leaf: Path) -> None:
  payload = json.loads(analysis_path.read_text(encoding="utf-8"))
  payload["provenance"]["independently_rehashed_local_artifacts"][role] = (
    historical_analysis.validate_leaf_manifest(leaf)
  )
  analysis_path.write_text(json.dumps(payload), encoding="utf-8")


def _refresh_banked_exposure(
  analysis_path: Path,
  *,
  population: str,
  role: str,
  trace: dict[str, np.ndarray],
) -> None:
  payload = json.loads(analysis_path.read_text(encoding="utf-8"))
  if population == "fixed_mean":
    banked = payload["fixed_mean"]
  else:
    banked = payload["training_like_replicas_before_pooling"][population]
  provisional = banked["thresholds_in_required_order"]["provisional_caps"][role]
  provisional["rho"] = historical_analysis._quantiles(
    historical_analysis._initial_episode_rho(
      trace, survey.PROVISIONAL_CAPS_N_M_S
    )
  )
  provisional["observed_first_contact_prefix_utilization"] = (
    historical_analysis._observed_first_contact_prefix_utilization(
      trace, survey.PROVISIONAL_CAPS_N_M_S
    )
  )
  analysis_path.write_text(json.dumps(payload), encoding="utf-8")


def test_preflight_banks_exact_p02_attribution_and_exposure_gate(frozen_evidence):
  """Returning defaults would hide whether the lower dose independently activates."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  result = evaluate_preflight(*frozen_evidence)

  assert result["candidate_imp_max_p"] == 0.2
  assert result["preflight_pass"] is True
  assert result["fixed_mean"]["active_reads"] == 0
  for seed, (_, _, active_reads, distinct_doses) in list(_POPULATIONS.items())[1:]:
    population = result["training_like_sampled"][seed]
    assert population["active_reads"] == active_reads
    assert population["distinct_positive_doses"] == distinct_doses
    assert population["winner_reads"] == {
      "impulse": active_reads,
      "velocity_masked": 0,
      "tie": 0,
    }
    assert population["combined_delta_exact_max"] is True
    assert population["activation_window_read_counts"]
    assert len(population["activation_window_pressure"]["observed_prefixes"]) == active_reads
    assert population["activation_window_pressure"]["unique_physical_event_dose"] is False
    assert population["physical_contact"]["right_censored_events"] == 0
    assert population["native_initial_episode_rho"]["max"] > 1.0
    assert population["observed_first_contact_prefix_rho"]["max"] > 1.0
  assert result["exposure_direction_gate"]["all_populations_pass"] is True


def test_preflight_rehashes_both_frozen_manifests(frozen_evidence, tmp_path):
  """Trusting manifest text would accept a trace changed after the historical analysis."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  control, target, analysis = _copy_evidence(frozen_evidence, tmp_path)
  with (control / "evaluation" / "fixed_trace.npz").open("ab") as stream:
    stream.write(b"tampered")

  with pytest.raises(ValueError, match="SHA-256"):
    evaluate_preflight(control, target, analysis)


def test_preflight_rejects_any_candidate_other_than_exactly_p02(frozen_evidence):
  """Dose drift would turn the one predeclared bridge into an unapproved sweep."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  with pytest.raises(ValueError, match="exactly 0.2"):
    evaluate_preflight(*frozen_evidence, candidate_imp_max_p=0.3)


def test_preflight_rejects_nonfinite_stored_pressure(frozen_evidence, tmp_path):
  """A NaN pressure must stop rather than survive JSON compaction."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  control, target, analysis = _copy_evidence(frozen_evidence, tmp_path)
  _rewrite_summary(
    control,
    lambda payload: payload["populations"]["training_like_sampled"]["2"][
      "provisional_caps"
    ]["candidate_imp_max_p"]["0.2"].__setitem__("delta_impulse_max", float("nan")),
  )
  _refresh_banked_manifest(analysis, "control", control)

  with pytest.raises(ValueError, match="non-finite"):
    evaluate_preflight(control, target, analysis)


def test_preflight_rejects_a_stored_nonmax_combination(frozen_evidence, tmp_path):
  """Adding the two pressures instead of max soft-OR must invalidate the preflight."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  control, target, analysis = _copy_evidence(frozen_evidence, tmp_path)

  def corrupt(payload):
    candidate = payload["populations"]["training_like_sampled"]["2"][
      "provisional_caps"
    ]["candidate_imp_max_p"]["0.2"]
    candidate["combined_delta_max"] += 0.1

  _rewrite_summary(control, corrupt)
  _refresh_banked_manifest(analysis, "control", control)
  with pytest.raises(ValueError, match="stored p=0.2 candidate"):
    evaluate_preflight(control, target, analysis)


@pytest.mark.parametrize("descriptor", ("read_count", "duration"))
def test_preflight_rejects_finite_stored_descriptor_corruption(
  frozen_evidence, tmp_path, descriptor
):
  """Finite stored descriptors must still agree exactly with the frozen raw trace."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  control, target, analysis = _copy_evidence(frozen_evidence, tmp_path)

  def corrupt(payload):
    summary = payload["populations"]["training_like_sampled"]["2"][
      "provisional_caps"
    ]
    if descriptor == "read_count":
      summary["binding"]["activation_window_read_counts"][0] += 1
    else:
      summary["physical_contact_duration_ms"]["all_observed_prefixes"][
        "median"
      ] += 1.0

  _rewrite_summary(control, corrupt)
  _refresh_banked_manifest(analysis, "control", control)
  with pytest.raises(ValueError, match="stored (binding|physical-contact) descriptors"):
    evaluate_preflight(control, target, analysis)


@pytest.mark.parametrize(
  "missing_path",
  ("activation_window_read_counts", "right_censored_events"),
)
def test_preflight_rejects_missing_read_duration_or_censor_descriptors(
  frozen_evidence, tmp_path, missing_path
):
  """Dropping exposure descriptors would let shorter or censored behavior look gentler."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  control, target, analysis = _copy_evidence(frozen_evidence, tmp_path)

  def corrupt(payload):
    summary = payload["populations"]["training_like_sampled"]["2"][
      "provisional_caps"
    ]
    if missing_path == "activation_window_read_counts":
      del summary["binding"][missing_path]
    else:
      del summary["physical_contact_duration_ms"][missing_path]

  _rewrite_summary(control, corrupt)
  _refresh_banked_manifest(analysis, "control", control)
  with pytest.raises((KeyError, ValueError), match=missing_path):
    evaluate_preflight(control, target, analysis)


def test_preflight_rejects_complete_velocity_masking(frozen_evidence, tmp_path):
  """Nonzero impulse pressure is not independently active when velocity wins every read."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  control, target, analysis = _copy_evidence(frozen_evidence, tmp_path)
  trace_path = control / "evaluation" / "training_like_seed_2_trace.npz"
  trace = historical_analysis._load_trace(trace_path)
  trace["delta_velocity"].fill(0.3)
  trace["delta"] = trace["delta_velocity"].copy()
  np.savez_compressed(trace_path, **trace)

  def replace_summary(payload):
    payload["populations"]["training_like_sampled"]["2"][
      "provisional_caps"
    ] = _population_summary(trace)

  _rewrite_summary(control, replace_summary)
  _refresh_banked_manifest(analysis, "control", control)
  with pytest.raises(ValueError, match="completely masked"):
    evaluate_preflight(control, target, analysis)


def test_preflight_rejects_ungraded_positive_pressure(frozen_evidence, tmp_path):
  """A single saturated positive value would not demonstrate a graded lower-dose response."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  control, target, analysis = _copy_evidence(frozen_evidence, tmp_path)
  trace_path = control / "evaluation" / "training_like_seed_2_trace.npz"
  trace = historical_analysis._load_trace(trace_path)
  trace["lambda_per_joint"][0, :35, 0] = np.float32(0.92)
  trace["substep_rolling_per_joint"][0] = trace["lambda_per_joint"][0]
  np.savez_compressed(trace_path, **trace)

  def replace_summary(payload):
    payload["populations"]["training_like_sampled"]["2"][
      "provisional_caps"
    ] = _population_summary(trace)

  _rewrite_summary(control, replace_summary)
  _refresh_banked_manifest(analysis, "control", control)
  with pytest.raises(ValueError, match="graded"):
    evaluate_preflight(control, target, analysis)


def test_preflight_rejects_reversed_observed_prefix_direction(frozen_evidence, tmp_path):
  """A historical target tail that rises on the contact prefix must stop training."""
  from scripts.preflight_vic_impulse_p02_bridge import evaluate_preflight

  control, target, analysis_path = _copy_evidence(frozen_evidence, tmp_path)
  trace_path = target / "evaluation" / "training_like_seed_2_trace.npz"
  trace = historical_analysis._load_trace(trace_path)
  trace["lambda_per_joint"][0, :100, 0] = np.float32(1.2)
  trace["substep_rolling_per_joint"][0] = trace["lambda_per_joint"][0]
  np.savez_compressed(trace_path, **trace)

  def replace_summary(payload):
    payload["populations"]["training_like_sampled"]["2"][
      "provisional_caps"
    ] = _population_summary(trace)

  _rewrite_summary(target, replace_summary)
  _refresh_banked_manifest(analysis_path, "target", target)
  _refresh_banked_exposure(
    analysis_path,
    population="2",
    role="target",
    trace=trace,
  )

  with pytest.raises(
    ValueError,
    match=r"2 exposure direction.*observed_first_contact_prefix_rho\.p99",
  ):
    evaluate_preflight(control, target, analysis_path)
