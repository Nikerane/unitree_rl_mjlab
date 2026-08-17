"""Contracts for matched post-training evaluation of the diagnostic-0.9 pair."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import impulse_cat_activation_survey as survey


CONTROL_SHA = "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3"
CODE_REVISION = "a" * 40
ASSET_REVISION = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
STOCHASTIC_SEEDS = (2, 2026081701, 2026081702)


def _protocol(seed: int, *, fixed: bool, live_imp_max_p: float = 0.0) -> dict[str, object]:
  return {
    "seed": seed,
    "num_envs": survey.FIXED_ENVS if fixed else survey.TRAINING_LIKE_ENVS,
    "control_steps": 7 if fixed else survey.TRAINING_LIKE_STEPS,
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
      "imp_max_p_live": live_imp_max_p,
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


def test_paired_evaluator_cli_selects_an_immutable_checkpoint_role(monkeypatch):
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "impulse_cat_activation_survey.py",
      "--checkpoint-role",
      "diag90_target",
      "--checkpoint",
      "/tmp/model_499.pt",
      "--output-dir",
      "/tmp/evaluation",
      "--device",
      "cuda:0",
    ],
  )

  args = survey._parse_args()

  assert args.checkpoint_role == "diag90_target"
  assert args.checkpoint == Path("/tmp/model_499.pt")
  assert args.output_dir == Path("/tmp/evaluation")
  assert args.device == "cuda:0"


def test_policy_evaluation_runs_one_fixed_and_three_exact_seed_keyed_populations(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  checkpoint = tmp_path / "model_499.pt"
  checkpoint.write_bytes(b"checkpoint")
  output_dir = tmp_path / "evaluation"
  calls: list[tuple[str, object]] = []
  traces: dict[int, dict[str, np.ndarray]] = {}
  summary_calls: list[tuple[int, tuple[float, ...], bool, int]] = []

  def validate(path: Path, role: str) -> str:
    calls.append(("validate", role))
    assert path == checkpoint.resolve()
    return CONTROL_SHA

  def run_population(**kwargs):
    calls.append(("run", dict(kwargs)))
    seed = int(kwargs["seed"])
    trace = {"trace_seed": np.asarray([seed], dtype=np.int64)}
    traces[seed] = trace
    return trace, _protocol(seed, fixed=kwargs["steps"] is None)

  def summarize(
    trace: dict[str, np.ndarray], *, caps: tuple[float, ...], first_episode_only: bool
  ) -> dict[str, object]:
    seed = int(trace["trace_seed"][0])
    summary_calls.append((seed, caps, first_episode_only, id(trace)))
    return {
      "trace_seed": seed,
      "caps_n_m_s": list(caps),
      "observed_peak_scope": "complete first episode",
      "utility": {"task_field_scope": "complete first episodes"},
    }

  monkeypatch.setattr(survey, "validate_checkpoint_role", validate)
  monkeypatch.setattr(survey, "_run_population", run_population)
  monkeypatch.setattr(survey, "summarize_population", summarize)
  monkeypatch.setattr(
    survey,
    "_evaluation_revisions",
    lambda: {"code_revision": CODE_REVISION, "asset_revision": ASSET_REVISION},
    raising=False,
  )

  payload = survey.run_policy_evaluation(
    checkpoint=checkpoint,
    role="diag90_control",
    output_dir=output_dir,
    device="cuda:0",
    stochastic_seeds=STOCHASTIC_SEEDS,
  )

  assert calls[0] == ("validate", "diag90_control")
  population_calls = [entry[1] for entry in calls if entry[0] == "run"]
  assert [call["seed"] for call in population_calls] == [
    survey.FIXED_SEED,
    *STOCHASTIC_SEEDS,
  ]
  assert population_calls[0] == {
    "checkpoint": checkpoint.resolve(),
    "device": "cuda:0",
    "num_envs": survey.FIXED_ENVS,
    "seed": survey.FIXED_SEED,
    "rng_seeds": survey.EvaluationRngSeeds.from_evaluation_seed(survey.FIXED_SEED),
    "steps": None,
    "stochastic": False,
  }
  for call, seed in zip(population_calls[1:], STOCHASTIC_SEEDS, strict=True):
    assert call == {
      "checkpoint": checkpoint.resolve(),
      "device": "cuda:0",
      "num_envs": survey.TRAINING_LIKE_ENVS,
      "seed": seed,
      "rng_seeds": survey.EvaluationRngSeeds.from_evaluation_seed(seed),
      "steps": survey.TRAINING_LIKE_STEPS,
      "stochastic": True,
    }

  assert summary_calls == [
    (
      survey.FIXED_SEED,
      survey.PROVISIONAL_CAPS_N_M_S,
      True,
      id(traces[survey.FIXED_SEED]),
    ),
    (
      survey.FIXED_SEED,
      survey.DIAGNOSTIC_LIMITS_N_M_S,
      True,
      id(traces[survey.FIXED_SEED]),
    ),
    *[
      item
      for seed in STOCHASTIC_SEEDS
      for item in (
        (seed, survey.PROVISIONAL_CAPS_N_M_S, True, id(traces[seed])),
        (seed, survey.DIAGNOSTIC_LIMITS_N_M_S, True, id(traces[seed])),
      )
    ],
  ]
  assert payload["checkpoint"] == {
    "role": "diag90_control",
    "path": str(checkpoint.resolve()),
    "sha256": CONTROL_SHA,
  }
  assert payload["code_revision"] == CODE_REVISION
  assert payload["asset_revision"] == ASSET_REVISION
  assert payload["protocol"]["stochastic_seeds"] == list(STOCHASTIC_SEEDS)
  assert payload["protocol"]["live_imp_max_p"] == 0.0
  assert list(payload["populations"]["training_like_sampled"]) == [
    str(seed) for seed in STOCHASTIC_SEEDS
  ]
  assert payload["populations"]["fixed_mean"]["provisional_caps"][
    "observed_peak_scope"
  ] == "complete first episode"
  for population in payload["populations"]["training_like_sampled"].values():
    assert population["provisional_caps"]["observed_peak_scope"] == (
      "initial 24-step rollout episode segments; unfinished segments are censored"
    )
    assert population["provisional_caps"]["utility"]["task_field_scope"] == (
      "initial 24-step stochastic task fragments; unfinished segments are censored"
    )
  assert (output_dir / "fixed_trace.npz").is_file()
  for seed in STOCHASTIC_SEEDS:
    assert (output_dir / f"training_like_seed_{seed}_trace.npz").is_file()
  assert json.loads((output_dir / "summary.json").read_text()) == payload


def test_policy_evaluation_refuses_wrong_seed_protocol_and_output_reuse(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  checkpoint = tmp_path / "model_499.pt"
  checkpoint.write_bytes(b"checkpoint")
  output_dir = tmp_path / "evaluation"
  run_calls = 0

  def unexpected_population(**_kwargs):
    nonlocal run_calls
    run_calls += 1
    raise AssertionError("a rejected evaluation must not construct a population")

  monkeypatch.setattr(survey, "_run_population", unexpected_population)

  with pytest.raises(ValueError, match="exact stochastic seed tuple"):
    survey.run_policy_evaluation(
      checkpoint=checkpoint,
      role="diag90_control",
      output_dir=output_dir,
      device="cpu",
      stochastic_seeds=(2,),
    )
  assert run_calls == 0

  output_dir.mkdir()
  with pytest.raises(FileExistsError, match="refusing to overwrite policy evaluation output"):
    survey.run_policy_evaluation(
      checkpoint=checkpoint,
      role="diag90_control",
      output_dir=output_dir,
      device="cpu",
      stochastic_seeds=STOCHASTIC_SEEDS,
    )
  assert run_calls == 0


def test_policy_evaluation_rejects_nonzero_live_impulse_pressure(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  checkpoint = tmp_path / "model_499.pt"
  checkpoint.write_bytes(b"checkpoint")
  trace = {"trace_seed": np.asarray([survey.FIXED_SEED], dtype=np.int64)}

  monkeypatch.setattr(survey, "validate_checkpoint_role", lambda *_args: CONTROL_SHA)
  monkeypatch.setattr(
    survey,
    "_evaluation_revisions",
    lambda: {"code_revision": CODE_REVISION, "asset_revision": ASSET_REVISION},
    raising=False,
  )
  monkeypatch.setattr(
    survey,
    "_run_population",
    lambda **_kwargs: (
      trace,
      _protocol(survey.FIXED_SEED, fixed=True, live_imp_max_p=0.5),
    ),
  )

  with pytest.raises(RuntimeError, match="live imp_max_p=0"):
    survey.run_policy_evaluation(
      checkpoint=checkpoint,
      role="diag90_control",
      output_dir=tmp_path / "evaluation",
      device="cpu",
      stochastic_seeds=STOCHASTIC_SEEDS,
    )


def _segment_summary(
  *,
  caps: tuple[float, ...],
  segments: int,
  violating_segments: int,
  violating_reads: int,
  utilization: tuple[float, float, float, float],
) -> dict[str, object]:
  return {
    "caps_n_m_s": list(caps),
    "binding": {"violating_reads": violating_reads},
    "segment_compliance": {
      "segments": segments,
      "any_joint_violating_segments": violating_segments,
      "any_joint_violation_rate": violating_segments / segments,
      "max_joint_utilization": dict(
        zip(("p50", "p95", "p99", "max"), utilization, strict=True)
      ),
    },
  }


def test_population_comparison_uses_paired_segments_not_overlapping_reads():
  control = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=100,
    violating_segments=10,
    violating_reads=1,
    utilization=(0.8, 1.2, 1.4, 1.6),
  )
  target = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=100,
    violating_segments=5,
    violating_reads=999_999,
    utilization=(0.7, 1.0, 1.1, 1.3),
  )

  comparison = survey.compare_population_summaries(control, target)

  assert comparison["statistical_unit"] == "paired episode segment"
  assert comparison["controller_reads_are_independent"] is False
  assert comparison["paired_segments"] == 100
  assert comparison["any_joint_violation"] == {
    "control": {"violating_segments": 10, "risk": 0.1},
    "target": {"violating_segments": 5, "risk": 0.05},
    "target_minus_control_absolute_risk_difference": pytest.approx(-0.05),
    "target_over_control_risk_ratio": pytest.approx(0.5),
  }
  assert comparison["max_joint_utilization_target_minus_control"] == {
    "p50": pytest.approx(-0.1),
    "p95": pytest.approx(-0.2),
    "p99": pytest.approx(-0.3),
    "max": pytest.approx(-0.3),
  }


def test_population_comparison_fails_closed_on_unpaired_or_inconsistent_summaries():
  control = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=100,
    violating_segments=10,
    violating_reads=10,
    utilization=(0.8, 1.2, 1.4, 1.6),
  )
  target = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=101,
    violating_segments=5,
    violating_reads=5,
    utilization=(0.7, 1.0, 1.1, 1.3),
  )
  with pytest.raises(ValueError, match="same number of episode segments"):
    survey.compare_population_summaries(control, target)

  target["segment_compliance"]["segments"] = 100
  target["segment_compliance"]["any_joint_violation_rate"] = 0.9
  with pytest.raises(ValueError, match="violation rate differs from segment counts"):
    survey.compare_population_summaries(control, target)

  target["segment_compliance"]["any_joint_violation_rate"] = 0.05
  target["caps_n_m_s"] = list(survey.PROVISIONAL_CAPS_N_M_S)
  with pytest.raises(ValueError, match="same cap vector"):
    survey.compare_population_summaries(control, target)


def test_population_comparison_reports_undefined_ratio_for_zero_control_risk():
  control = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=8,
    violating_segments=0,
    violating_reads=0,
    utilization=(0.5, 0.6, 0.7, 0.8),
  )
  target = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=8,
    violating_segments=1,
    violating_reads=40,
    utilization=(0.6, 0.8, 1.1, 1.2),
  )

  comparison = survey.compare_population_summaries(control, target)

  assert comparison["any_joint_violation"][
    "target_minus_control_absolute_risk_difference"
  ] == pytest.approx(0.125)
  assert comparison["any_joint_violation"]["target_over_control_risk_ratio"] is None


def _evaluation_payload(
  role: str,
  provisional_summary: dict[str, object],
  diagnostic_summary: dict[str, object],
) -> dict[str, object]:
  return {
    "task": survey.VIC_TASK,
    "checkpoint": {"role": role, "sha256": survey.EVALUATION_CHECKPOINTS[role]},
    "code_revision": CODE_REVISION,
    "asset_revision": ASSET_REVISION,
    "protocol": {
      "live_imp_max_p": 0.0,
      "fixed_seed": survey.FIXED_SEED,
      "stochastic_seeds": list(STOCHASTIC_SEEDS),
      "threshold_summary_order": ["provisional_caps", "diagnostic_only"],
      "primary_statistical_unit": "initial episode segment",
      "controller_reads_are_independent": False,
    },
    "populations": {
      "fixed_mean": {
        "protocol": _protocol(survey.FIXED_SEED, fixed=True),
        "provisional_caps": copy.deepcopy(provisional_summary),
        "diagnostic_only": copy.deepcopy(diagnostic_summary),
      },
      "training_like_sampled": {
        str(seed): {
          "protocol": _protocol(seed, fixed=False),
          "provisional_caps": copy.deepcopy(provisional_summary),
          "diagnostic_only": copy.deepcopy(diagnostic_summary),
        }
        for seed in STOCHASTIC_SEEDS
      },
    },
  }


def test_policy_comparison_pairs_roles_population_protocols_and_threshold_order():
  control_provisional = _segment_summary(
    caps=survey.PROVISIONAL_CAPS_N_M_S,
    segments=64,
    violating_segments=8,
    violating_reads=300,
    utilization=(0.8, 1.2, 1.4, 1.6),
  )
  control_diagnostic = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=64,
    violating_segments=8,
    violating_reads=300,
    utilization=(0.9, 1.3, 1.5, 1.7),
  )
  target_provisional = _segment_summary(
    caps=survey.PROVISIONAL_CAPS_N_M_S,
    segments=64,
    violating_segments=4,
    violating_reads=2,
    utilization=(0.7, 1.0, 1.1, 1.3),
  )
  target_diagnostic = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=64,
    violating_segments=4,
    violating_reads=2,
    utilization=(0.8, 1.1, 1.2, 1.4),
  )

  control = _evaluation_payload(
    "diag90_control", control_provisional, control_diagnostic
  )
  target = _evaluation_payload(
    "diag90_target", target_provisional, target_diagnostic
  )
  target["populations"]["fixed_mean"]["protocol"]["control_steps"] = 9

  comparison = survey.compare_policy_evaluations(control, target)

  assert comparison["roles"] == {
    "control": "diag90_control",
    "target": "diag90_target",
  }
  assert comparison["threshold_order"] == ["provisional_caps", "diagnostic_only"]
  assert comparison["fixed_mean"]["diagnostic_only"]["any_joint_violation"][
    "target_over_control_risk_ratio"
  ] == pytest.approx(0.5)
  assert list(comparison["training_like_sampled"]) == [
    str(seed) for seed in STOCHASTIC_SEEDS
  ]


@pytest.mark.parametrize(
  ("mutation", "message"),
  (
    ("checkpoint", "checkpoint SHA"),
    ("code_revision", "code revision"),
    ("asset_revision", "asset revision"),
    ("live_imp_max_p", "live imp_max_p=0"),
  ),
)
def test_policy_comparison_fails_closed_on_identity_or_log_only_drift(
  mutation: str, message: str
):
  provisional = _segment_summary(
    caps=survey.PROVISIONAL_CAPS_N_M_S,
    segments=64,
    violating_segments=4,
    violating_reads=4,
    utilization=(0.7, 1.0, 1.1, 1.3),
  )
  diagnostic = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=64,
    violating_segments=4,
    violating_reads=4,
    utilization=(0.8, 1.1, 1.2, 1.4),
  )
  control = _evaluation_payload("diag90_control", provisional, diagnostic)
  target = _evaluation_payload("diag90_target", provisional, diagnostic)
  if mutation == "checkpoint":
    target["checkpoint"]["sha256"] = "f" * 64
  elif mutation in ("code_revision", "asset_revision"):
    target[mutation] = "c" * 40
  else:
    control["protocol"]["live_imp_max_p"] = 0.5
    target["protocol"]["live_imp_max_p"] = 0.5

  with pytest.raises(ValueError, match=message):
    survey.compare_policy_evaluations(control, target)


def _valid_policy_pair() -> tuple[dict[str, object], dict[str, object]]:
  provisional = _segment_summary(
    caps=survey.PROVISIONAL_CAPS_N_M_S,
    segments=64,
    violating_segments=4,
    violating_reads=4,
    utilization=(0.7, 1.0, 1.1, 1.3),
  )
  diagnostic = _segment_summary(
    caps=survey.DIAGNOSTIC_LIMITS_N_M_S,
    segments=64,
    violating_segments=6,
    violating_reads=8,
    utilization=(0.8, 1.1, 1.2, 1.4),
  )
  return (
    _evaluation_payload("diag90_control", provisional, diagnostic),
    _evaluation_payload("diag90_target", provisional, diagnostic),
  )


@pytest.mark.parametrize(
  ("population", "slot", "wrong_caps"),
  (
    ("fixed_mean", "provisional_caps", survey.DIAGNOSTIC_LIMITS_N_M_S),
    ("fixed_mean", "diagnostic_only", survey.PROVISIONAL_CAPS_N_M_S),
    ("2", "provisional_caps", survey.DIAGNOSTIC_LIMITS_N_M_S),
    ("2", "diagnostic_only", survey.PROVISIONAL_CAPS_N_M_S),
  ),
)
def test_policy_comparison_binds_each_threshold_slot_to_its_frozen_cap_vector(
  population: str, slot: str, wrong_caps: tuple[float, ...]
):
  control, target = _valid_policy_pair()
  for payload in (control, target):
    selected = (
      payload["populations"]["fixed_mean"]
      if population == "fixed_mean"
      else payload["populations"]["training_like_sampled"][population]
    )
    selected[slot]["caps_n_m_s"] = list(wrong_caps)

  with pytest.raises(ValueError, match=rf"{slot}.*exact cap vector"):
    survey.compare_policy_evaluations(control, target)


@pytest.mark.parametrize(
  ("mutation", "message"),
  (
    ("task", "exact task"),
    ("threshold_order", "threshold summary order"),
    ("statistical_unit", "statistical unit"),
    ("read_independence", "controller reads"),
    ("fixed_seed", "fixed seed"),
    ("code_revision_missing", "code revision"),
    ("code_revision_malformed", "code revision"),
    ("asset_revision", "asset revision"),
    ("top_live_p", "live imp_max_p=0"),
    ("fixed_population_seed", "fixed population seed"),
    ("fixed_envs", "fixed population num_envs"),
    ("fixed_hash", "fixed population hash"),
    ("fixed_mode", "fixed population policy mode"),
    ("fixed_auto_reset", "fixed population auto_reset"),
    ("sampled_seed", "sampled population seed"),
    ("sampled_envs", "sampled population num_envs"),
    ("sampled_steps", "sampled population control_steps"),
    ("sampled_mode", "sampled population policy mode"),
    ("sampled_auto_reset", "sampled population auto_reset"),
    ("sampled_rng", "sampled population RNG streams"),
    ("population_live_p", "live imp_max_p=0"),
  ),
)
def test_policy_comparison_rejects_common_mode_protocol_drift(
  mutation: str, message: str
):
  control, target = _valid_policy_pair()
  payloads = (control, target)
  if mutation == "task":
    for payload in payloads:
      payload["task"] = "wrong-task"
  elif mutation == "threshold_order":
    for payload in payloads:
      payload["protocol"]["threshold_summary_order"] = [
        "diagnostic_only", "provisional_caps"
      ]
  elif mutation == "statistical_unit":
    for payload in payloads:
      payload["protocol"]["primary_statistical_unit"] = "controller read"
  elif mutation == "read_independence":
    for payload in payloads:
      payload["protocol"]["controller_reads_are_independent"] = True
  elif mutation == "fixed_seed":
    for payload in payloads:
      payload["protocol"]["fixed_seed"] = 7
  elif mutation == "code_revision_missing":
    for payload in payloads:
      payload.pop("code_revision")
  elif mutation == "code_revision_malformed":
    for payload in payloads:
      payload["code_revision"] = "z" * 40
  elif mutation == "asset_revision":
    for payload in payloads:
      payload["asset_revision"] = "c" * 40
  elif mutation == "top_live_p":
    for payload in payloads:
      payload["protocol"]["live_imp_max_p"] = 0.5
  elif mutation.startswith("fixed_"):
    protocols = [
      payload["populations"]["fixed_mean"]["protocol"] for payload in payloads
    ]
    field, value = {
      "fixed_population_seed": ("seed", 7),
      "fixed_envs": ("num_envs", 63),
      "fixed_hash": ("initial_population_sha256", "c" * 64),
      "fixed_mode": ("policy_mode", "sampled"),
      "fixed_auto_reset": ("auto_reset", True),
    }[mutation]
    for protocol in protocols:
      protocol[field] = value
  else:
    protocols = [
      payload["populations"]["training_like_sampled"]["2"]["protocol"]
      for payload in payloads
    ]
    if mutation == "sampled_seed":
      for protocol in protocols:
        protocol["seed"] = 3
    elif mutation == "sampled_envs":
      for protocol in protocols:
        protocol["num_envs"] = 4095
    elif mutation == "sampled_steps":
      for protocol in protocols:
        protocol["control_steps"] = 23
    elif mutation == "sampled_mode":
      for protocol in protocols:
        protocol["policy_mode"] = "mean"
    elif mutation == "sampled_auto_reset":
      for protocol in protocols:
        protocol["auto_reset"] = False
    elif mutation == "sampled_rng":
      for protocol in protocols:
        protocol["rng_streams"]["action"] += 1
    else:
      for protocol in protocols:
        protocol["cat_replay"]["imp_max_p_live"] = 0.5

  with pytest.raises(ValueError, match=message):
    survey.compare_policy_evaluations(control, target)


def test_sampled_control_steps_must_match_even_when_one_arm_claims_twenty_four():
  control, target = _valid_policy_pair()
  target["populations"]["training_like_sampled"]["2"]["protocol"][
    "control_steps"
  ] = 23

  with pytest.raises(ValueError, match="sampled population control_steps"):
    survey.compare_policy_evaluations(control, target)
