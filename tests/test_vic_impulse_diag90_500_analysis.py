"""Tests for the compact diagnostic-0.9 post-training comparison."""

import importlib
import hashlib
import json
from pathlib import Path

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


def test_first_contact_utilization_is_labeled_as_censored_observed_prefix():
  """Calling a terminal-cut prefix a complete event would hide unequal horizons."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )
  trace = {
    "lambda_per_joint": np.full((2, 1, 6), 0.5, dtype=np.float32),
    "episode_id": np.zeros((2, 1), dtype=np.int64),
    "substep_contact": np.ones((20, 1), dtype=bool),
    "substep_episode_id": np.zeros((20, 1), dtype=np.int64),
  }

  result = analysis._observed_first_contact_prefix_utilization(
    trace, caps=(1.0,) * 6
  )

  assert result["unit"] == "observed first-contact physical-event prefix"
  assert result["scope"] == (
    "native controller reads overlapping the terminal-cut first-contact prefix"
  )
  assert result["equal_horizon_across_arms"] is False
  assert result["complete_physical_event"] is False
  assert result["causal_or_full_contact_interpretation_authorized"] is False
  assert result["observed_prefixes_with_contact"] == 1
  assert "segments_with_contact" not in result


def test_banked_claim_keeps_native_endpoint_but_rejects_causal_contact_interpretation():
  """The bank must not turn a valid native endpoint into an unconfounded causal claim."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )

  verdict = analysis._native_verdict()

  assert verdict["native_observed_endpoint_valid"] is True
  assert verdict["full_contact_or_causal_softening_interpretation_authorized"] is False
  assert verdict["shadow_required_to_resolve_episode_horizon_confound"] is True
  assert [branch["text"] for branch in verdict["preregistered_branches_verbatim"]] == [
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
  ]
  assert [branch["selected"] for branch in verdict["preregistered_branches_verbatim"]] == [
    False,
    False,
    True,
    False,
    False,
  ]


def test_pressure_compaction_names_activation_windows_not_physical_event_dose():
  """An activation-window survival product must not be banked as unique-event dose."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )
  survey = importlib.import_module("scripts.impulse_cat_activation_survey")
  trace = {
    "done": np.array([[False, True, False, True]]),
    "substep_contact": np.array(
      [
        [True, True, True, True],
        [False, True, False, False],
        [False, True, True, True],
        [False, True, False, True],
        *([[False, True, False, True]] * 6),
      ]
    ),
    "substep_episode_id": np.zeros((10, 4), dtype=np.int64),
  }
  summary = {
    "binding": {
      "reads": [
        {
          "env_id": 0,
          "control_step": 0,
          "contact_event_ids": [0],
          "physical_event_association_ambiguous": False,
        },
        {
          "env_id": 1,
          "control_step": 0,
          "contact_event_ids": [1],
          "physical_event_association_ambiguous": False,
        },
        {
          "env_id": 2,
          "control_step": 0,
          "contact_event_ids": [2, 3],
          "physical_event_association_ambiguous": True,
        },
        {
          "env_id": 3,
          "control_step": 0,
          "contact_event_ids": [4, 5],
          "physical_event_association_ambiguous": True,
        },
      ]
    },
    "candidate_imp_max_p": {
      "0.5": {
        "activation_window_summaries": [
          {"env_id": env_id, "start_step": 0, "end_step": 0, "event_pressure": 0.1 * (env_id + 1)}
          for env_id in range(4)
        ]
      }
    },
  }

  result = analysis._activation_window_pressure_by_associated_physical_event_status(
    summary, trace, survey=survey
  )

  assert result["unit"] == "contiguous 50 Hz impulse-activation window"
  assert result["unique_physical_event_dose"] is False
  assert result["classification_basis"] == (
    "censoring and ambiguity of associated physical event IDs"
  )
  assert result["groups"]["completed_unambiguous"]["activation_windows"] == 1
  assert result["groups"]["completed_ambiguous"]["activation_windows"] == 1
  assert result["groups"]["right_censored_unambiguous"]["activation_windows"] == 1
  assert result["groups"]["right_censored_ambiguous"]["activation_windows"] == 1


def _cat_trace() -> dict[str, np.ndarray]:
  lam = np.zeros((2, 1, 6), dtype=np.float32)
  lam[:, 0, 0] = 1.2
  velocity = np.array([[0.2], [0.7]], dtype=np.float32)
  return {
    "lambda_per_joint": lam,
    "episode_id": np.zeros((2, 1), dtype=np.int64),
    "delta_velocity": velocity,
    "delta_impulse": np.zeros((2, 1), dtype=np.float32),
    "delta": velocity.copy(),
  }


def test_cat_compaction_banks_both_components_and_exact_max_invariants():
  """Winner counts alone cannot reproduce velocity, impulse, or combined pressure."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )
  survey = importlib.import_module("scripts.impulse_cat_activation_survey")
  summary = {
    "candidate_imp_max_p": {
      "0.5": {
        "active_reads": 2,
        "delta_impulse_active_mean": 0.5,
        "delta_impulse_max": 0.5,
        "combined_delta_max": 0.7,
      }
    }
  }

  result = analysis._compact_cat_attribution(
    summary, _cat_trace(), caps=(1.0,) * 6, survey=survey
  )

  assert result["live_trace"]["delta_velocity"]["max"] == pytest.approx(0.7)
  assert result["live_trace"]["delta_impulse"]["max"] == 0.0
  assert result["live_trace"]["combined_delta"]["max"] == pytest.approx(0.7)
  assert result["live_trace"]["combined_delta_exact_max"] is True
  offline = result["offline_replay_at_p_0_5"]
  assert offline["delta_velocity"]["p50"] == pytest.approx(0.45)
  assert offline["delta_impulse"]["p50"] == pytest.approx(0.5)
  assert offline["combined_delta"]["p50"] == pytest.approx(0.6)
  assert offline["combined_delta_exact_max"] is True
  assert offline["summary_replay_consistent"] is True


def test_cat_compaction_rejects_a_nonmax_live_combined_trace():
  """A corrupted combined delta must fail rather than be summarized as valid evidence."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )
  survey = importlib.import_module("scripts.impulse_cat_activation_survey")
  trace = _cat_trace()
  trace["delta"][0, 0] = 0.3
  summary = {
    "candidate_imp_max_p": {
      "0.5": {
        "active_reads": 2,
        "delta_impulse_active_mean": 0.5,
        "delta_impulse_max": 0.5,
        "combined_delta_max": 0.7,
      }
    }
  }

  with pytest.raises(ValueError, match="live combined delta"):
    analysis._compact_cat_attribution(
      summary, trace, caps=(1.0,) * 6, survey=survey
    )


def _gain_record(*, pre_p: float, pre_kp: float, at_p: float, at_kp: float):
  return {
    "vic_p_precontact": [0.0, 0.0, 0.0, pre_p, 0.0, 0.0],
    "vic_kp_precontact": [0.0, 0.0, 0.0, pre_kp, 0.0, 0.0],
    "vic_kd_precontact": [0.0, 0.0, 0.0, 90.0, 0.0, 0.0],
    "vic_p_at_contact": [0.0, 0.0, 0.0, at_p, 0.0, 0.0],
    "vic_kp_at_contact": [0.0, 0.0, 0.0, at_kp, 0.0, 0.0],
    "vic_kd_at_contact": [0.0, 0.0, 0.0, 110.0, 0.0, 0.0],
  }


def test_gain_compaction_keeps_precontact_and_at_contact_distributions_distinct():
  """At-contact equality must not erase a real precontact policy difference."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )
  gains = {
    "segments_with_contact": 2,
    "right_censored_segments": 0,
    "records": [
      _gain_record(pre_p=1.0, pre_kp=1250.0, at_p=1.0, at_kp=1250.0),
      _gain_record(pre_p=-1.0, pre_kp=800.0, at_p=1.0, at_kp=1250.0),
    ],
  }

  result = analysis._compact_gain_summary(gains)["per_joint"]["joint4"]

  assert result["vic_p_precontact"]["mean"] == 0.0
  assert result["vic_kp_precontact"]["mean"] == 1025.0
  assert result["vic_p_at_contact"]["mean"] == 1.0
  assert result["vic_kp_at_contact"]["mean"] == 1250.0


def test_utility_compaction_uses_only_complete_initial_episode_terminals():
  """Post-reset task values must never leak into initial-episode utility."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )
  gains = {
    "segments_with_contact": 1,
    "right_censored_segments": 0,
    "records": [_gain_record(pre_p=1.0, pre_kp=1250.0, at_p=1.0, at_kp=1250.0)],
  }
  summary = {
    "utility": {
      "terminal_counts": {
        "population": 1,
        "terminal": 1,
        "success": 1,
        "timeout": 0,
        "productive_first_strike": 1,
      },
      "velocity_limit_compliance": {"segments": 1},
      "first_contact_vic_gains": gains,
    }
  }
  trace = {
    "done": np.array([[True], [True]]),
    "episode_id": np.array([[0], [1]], dtype=np.int64),
    "first_strike_delivered_n_s": np.array([[0.4], [9.0]], dtype=np.float32),
    "delivered_total_n_s": np.array([[0.5], [9.0]], dtype=np.float32),
    "nail_depth_m": np.array([[0.02], [9.0]], dtype=np.float32),
  }

  result = analysis._compact_utility(summary, trace)

  assert result["terminal_value_source"] == "episode_id == 0 and done"
  assert result["first_strike_delivered_n_s"]["mean"] == pytest.approx(0.4)
  assert result["episode_duration_ms"]["p50"] == 20.0
  assert result["episode_reward"] == {
    "available": False,
    "trace_field_present": False,
    "reason": "the frozen evaluator trace did not record reward",
  }


def _provenance_summary(role: str, checkpoint_sha: str) -> dict[str, object]:
  def protocol(seed: int, population_hash: str) -> dict[str, object]:
    return {
      "seed": seed,
      "num_envs": 4,
      "initial_population_sha256": population_hash,
      "rng_streams": {"reset": seed + 1, "observation": seed + 2, "action": seed + 3},
      "policy_mode": "sampled",
      "auto_reset": True,
    }

  return {
    "task": "exact-task",
    "code_revision": "1" * 40,
    "asset_revision": "2" * 40,
    "checkpoint": {"role": role, "sha256": checkpoint_sha},
    "protocol": {"live_imp_max_p": 0.0},
    "populations": {
      "fixed_mean": {"protocol": protocol(10, "a" * 64)},
      "training_like_sampled": {
        "2": {"protocol": protocol(2, "b" * 64)},
        "2026081701": {"protocol": protocol(2026081701, "c" * 64)},
        "2026081702": {"protocol": protocol(2026081702, "d" * 64)},
      },
    },
  }


def test_provenance_separates_rehashed_artifacts_from_summary_declarations():
  """Local artifact verification must not imply unavailable checkpoint/job proof."""
  analysis = importlib.import_module(
    "scripts.analyze_vic_impulse_diag90_500_evaluation"
  )
  summaries = {
    "control": _provenance_summary("diag90_control", "e" * 64),
    "target": _provenance_summary("diag90_target", "f" * 64),
  }
  manifests = {
    "control": {"summary.json": {"sha256": "3" * 64, "size_bytes": 123}},
    "target": {"summary.json": {"sha256": "4" * 64, "size_bytes": 456}},
  }

  result = analysis._compact_provenance(summaries, manifests)

  declared = result["summary_declared_fail_closed_identities"]
  assert declared["task"] == "exact-task"
  assert declared["checkpoints"]["control"] == {
    "role": "diag90_control",
    "sha256": "e" * 64,
  }
  assert declared["populations"]["training_like_sampled"]["2"]["control"][
    "rng_streams"
  ] == {"reset": 3, "observation": 4, "action": 5}
  assert result["independently_rehashed_local_artifacts"] == manifests
  boundary = result["claim_boundary"]
  assert boundary["checkpoint_binaries_locally_banked_and_rehashed"] is False
  assert boundary["latent_action_noise_equality_trace_auditable"] is False
  assert boundary["scheduler_status_derived_from_trace"] is False
  assert boundary["scheduler_status_source"] == "authoritative Task-8 execution record"
  assert boundary["local_artifact_hash_status"] == (
    "all downloaded trace and summary bytes independently rehashed against their respective "
    "remote SHA256SUMS manifests"
  )


def test_banked_analysis_schema_preserves_corrected_semantics_and_claim_boundary():
  """A stale bank must fail after semantic fields are renamed or claim scope narrows."""
  bank = (
    Path(__file__).resolve().parents[1]
    / "docs/results/assets/2026-08-17_z1_impulse_diag90_500_evaluation/analysis.json"
  )
  payload = json.loads(bank.read_text(encoding="utf-8"))

  assert payload["schema_version"] == 2
  verdict = payload["native_verdict"]
  assert verdict["native_observed_endpoint_valid"] is True
  assert verdict["full_contact_or_causal_softening_interpretation_authorized"] is False
  assert len(verdict["preregistered_branches_verbatim"]) == 5
  assert verdict["preregistered_branches_verbatim"][2]["selected"] is True
  provenance = payload["provenance"]
  assert "summary_declared_fail_closed_identities" in provenance
  assert "independently_rehashed_local_artifacts" in provenance
  fixed = payload["fixed_mean"]["thresholds_in_required_order"]["provisional_caps"]
  for role in ("control", "target"):
    threshold = fixed[role]
    assert "observed_first_contact_prefix_utilization" in threshold
    cat = threshold["offline_cat_at_p_0_5"]
    assert "component_summaries_and_invariants" in cat
    assert "activation_window_pressure_by_associated_physical_event_status" in cat
    assert "event_pressure_by_physical_censoring_and_ambiguity" not in cat
