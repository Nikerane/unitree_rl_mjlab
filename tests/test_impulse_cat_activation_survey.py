"""Pure analysis tests for the no-learning impulse-CaT activation survey."""

import numpy as np
import pytest

import scripts.impulse_cat_activation_survey as survey

from scripts.impulse_cat_activation_survey import (
  DIAGNOSTIC_LIMITS_N_M_S,
  EXPECTED_CHECKPOINT_SHA256,
  EXPECTED_FIXED_POPULATION_SHA256,
  FIXED_ENVS,
  FIXED_SEED,
  PROVISIONAL_CAPS_N_M_S,
  _associate_violating_reads,
  _candidate_activation_window_summaries,
  _native_cap_margins,
  _physical_event_read_counts,
  _prepare_survey_measurement_config,
  contact_events,
  contiguous_activation_events,
  shadow_impulse_cat,
)


CONTROL_SHA = "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3"
TARGET_SHA = "ddd7ac4c855160bff1db2af52e642d960dab2bef34e41dd2532002185eb36d15"


def test_evaluation_checkpoint_roles_are_exact_and_fail_closed(tmp_path, monkeypatch):
  checkpoint = tmp_path / "model_499.pt"
  checkpoint.write_bytes(b"control")
  monkeypatch.setattr(survey, "_sha256", lambda _: CONTROL_SHA)

  assert dict(survey.EVALUATION_CHECKPOINTS) == {
    "diag90_control": CONTROL_SHA,
    "diag90_target": TARGET_SHA,
  }
  with pytest.raises(TypeError):
    survey.EVALUATION_CHECKPOINTS["unexpected_role"] = CONTROL_SHA
  assert survey.validate_checkpoint_role(checkpoint, "diag90_control") == CONTROL_SHA
  with pytest.raises(RuntimeError, match="role/checkpoint SHA-256 mismatch"):
    survey.validate_checkpoint_role(checkpoint, "diag90_target")


def test_contiguous_activation_events_count_reads_pressure_and_reset_boundaries():
  active = np.array(
    [
      [False, True],
      [True, True],
      [True, True],
      [False, True],
      [True, False],
    ],
    dtype=bool,
  )
  done = np.array(
    [
      [False, False],
      [False, True],
      [False, False],
      [False, False],
      [True, False],
    ],
    dtype=bool,
  )
  delta_impulse = np.array(
    [
      [0.0, 0.1],
      [0.2, 0.2],
      [0.3, 0.3],
      [0.0, 0.4],
      [0.5, 0.0],
    ],
    dtype=np.float64,
  )

  events = contiguous_activation_events(active, done, delta_impulse)

  assert events == [
    {
      "env_id": 0, "start_step": 1, "end_step": 2, "reads": 2,
      "event_pressure": 0.44, "right_censored": False, "censor_reason": None,
    },
    {
      "env_id": 0, "start_step": 4, "end_step": 4, "reads": 1,
      "event_pressure": 0.5, "right_censored": False, "censor_reason": None,
    },
    {
      "env_id": 1, "start_step": 0, "end_step": 1, "reads": 2,
      "event_pressure": 0.28, "right_censored": False, "censor_reason": None,
    },
    {
      "env_id": 1, "start_step": 2, "end_step": 3, "reads": 2,
      "event_pressure": 0.58, "right_censored": False, "censor_reason": None,
    },
  ]


def test_contiguous_activation_events_reject_nonfinite_or_misaligned_inputs():
  import pytest

  with pytest.raises(ValueError, match="shape"):
    contiguous_activation_events(
      np.zeros((2, 1), dtype=bool),
      np.zeros((3, 1), dtype=bool),
      np.zeros((2, 1)),
    )
  with pytest.raises(ValueError, match="finite"):
    contiguous_activation_events(
      np.ones((1, 1), dtype=bool),
      np.zeros((1, 1), dtype=bool),
      np.array([[np.nan]]),
    )


def test_contact_events_measure_500hz_duration_and_mark_terminal_censoring():
  contact = np.array(
    [
      [False, True],
      [True, True],
      [True, False],
      [False, True],
      [True, True],
    ],
    dtype=bool,
  )
  terminal = np.array(
    [
      [False, False],
      [False, False],
      [False, False],
      [False, False],
      [True, False],
    ],
    dtype=bool,
  )
  episode_id = np.array(
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    dtype=np.int64,
  )

  events, labels = contact_events(
    contact,
    terminal=terminal,
    episode_id=episode_id,
    physics_dt_s=0.002,
  )

  assert events == [
    {
      "event_id": 0,
      "env_id": 0,
      "episode_id": 0,
      "start_substep": 1,
      "end_substep": 2,
      "contact_substeps": 2,
      "observed_duration_ms": 4.0,
      "right_censored": False,
      "censor_reason": None,
    },
    {
      "event_id": 1,
      "env_id": 0,
      "episode_id": 0,
      "start_substep": 4,
      "end_substep": 4,
      "contact_substeps": 1,
      "observed_duration_ms": 2.0,
      "right_censored": True,
      "censor_reason": "terminal",
    },
    {
      "event_id": 2,
      "env_id": 1,
      "episode_id": 0,
      "start_substep": 0,
      "end_substep": 1,
      "contact_substeps": 2,
      "observed_duration_ms": 4.0,
      "right_censored": False,
      "censor_reason": None,
    },
    {
      "event_id": 3,
      "env_id": 1,
      "episode_id": 0,
      "start_substep": 3,
      "end_substep": 4,
      "contact_substeps": 2,
      "observed_duration_ms": 4.0,
      "right_censored": True,
      "censor_reason": "trace_end",
    },
  ]
  assert labels.tolist() == [[-1, 2], [0, 2], [0, -1], [-1, 3], [1, 3]]


def test_contact_release_takes_precedence_when_terminal_is_observed_on_same_substep():
  events, _ = contact_events(
    np.array([[True], [False]]),
    terminal=np.array([[False], [True]]),
    episode_id=np.array([[0], [0]]),
    physics_dt_s=0.002,
  )

  assert events[0]["observed_duration_ms"] == 2.0
  assert events[0]["right_censored"] is False
  assert events[0]["censor_reason"] is None


def test_shadow_impulse_cat_replays_violation_masked_normalizer_and_scales_dose():
  margins = np.array(
    [
      [[-0.1, 0.2], [0.1, -0.1]],
      [[0.2, 0.1], [0.05, 0.3]],
      [[-0.1, -0.1], [-0.2, -0.2]],
    ],
    dtype=np.float64,
  )

  unit = shadow_impulse_cat(margins, tau=0.5, seed=0.01, max_p=1.0)
  dose = shadow_impulse_cat(margins, tau=0.5, seed=0.01, max_p=0.2)

  np.testing.assert_allclose(
    unit,
    np.array(
      [
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0, 0.4], [1.0 / 3.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0]],
      ]
    ),
  )
  np.testing.assert_allclose(dose, 0.2 * unit)
  assert np.isfinite(unit).all()
  assert unit.shape == margins.shape


def test_shadow_impulse_cat_excludes_invalid_reset_episodes_from_population_ema():
  margins = np.array(
    [
      [[0.5], [0.5]],
      [[10.0], [0.5]],
    ],
    dtype=np.float64,
  )
  valid = np.array([[True, True], [False, True]], dtype=bool)

  unit = shadow_impulse_cat(
    margins,
    tau=0.95,
    seed=0.001,
    max_p=1.0,
    valid=valid,
  )

  np.testing.assert_allclose(unit[:, :, 0], [[1.0, 1.0], [0.0, 1.0]])


def test_cap_margins_are_subtracted_before_float64_promotion():
  """Offline replay must classify the same near-boundary float32 margin as the live hook."""
  lam = np.array([[[np.float32(0.82)]]], dtype=np.float32)

  margins = _native_cap_margins(lam, (0.82,))

  assert margins.dtype == np.float64
  assert margins.item() == 0.0
  assert (lam.astype(np.float64) - np.array([0.82], dtype=np.float64)).item() != 0.0


def test_segment_compliance_counts_repeated_violating_reads_once_per_episode_segment():
  """A four-read cap crossing is one episode-level compliance observation, not four."""
  lam = np.full((4, 1, 6), 0.1, dtype=np.float32)
  lam[:, 0, 0] = 1.0
  rolling = np.zeros((40, 1, 6), dtype=np.float32)
  for step in range(4):
    rolling[(step + 1) * 10 - 1, 0] = lam[step, 0]
  trace = {
    "lambda_per_joint": lam,
    "episode_id": np.zeros((4, 1), dtype=np.int64),
    "done": np.array([[False], [False], [False], [True]]),
    "delta_velocity": np.zeros((4, 1), dtype=np.float64),
    "delta_impulse": np.zeros((4, 1), dtype=np.float64),
    "delta": np.zeros((4, 1), dtype=np.float64),
    "substep_contact": np.ones((40, 1), dtype=bool),
    "substep_episode_id": np.zeros((40, 1), dtype=np.int64),
    "substep_rolling_per_joint": rolling,
  }

  summary = survey.summarize_population(
    trace, caps=PROVISIONAL_CAPS_N_M_S, first_episode_only=True
  )

  compliance = summary["segment_compliance"]
  assert compliance["segments"] == 1
  assert compliance["any_joint_violating_segments"] == 1
  assert compliance["any_joint_violation_rate"] == 1.0
  assert compliance["max_joint_utilization"] == {
    "p50": pytest.approx(1.0 / 0.82),
    "p95": pytest.approx(1.0 / 0.82),
    "p99": pytest.approx(1.0 / 0.82),
    "max": pytest.approx(1.0 / 0.82),
  }
  assert compliance["per_joint"]["joint1"] == {
    "violating_segments": 1,
    "violation_rate": 1.0,
    "utilization": {
      "p50": pytest.approx(1.0 / 0.82),
      "p95": pytest.approx(1.0 / 0.82),
      "p99": pytest.approx(1.0 / 0.82),
      "max": pytest.approx(1.0 / 0.82),
    },
    "positive_margin_n_m_s": {
      "p50": pytest.approx(0.18),
      "p95": pytest.approx(0.18),
      "p99": pytest.approx(0.18),
      "max": pytest.approx(0.18),
    },
  }
  assert compliance["per_joint"]["joint2"]["violating_segments"] == 0
  assert compliance["per_joint"]["joint2"]["positive_margin_n_m_s"] == {
    "p50": None, "p95": None, "p99": None, "max": None,
  }


def test_registered_vic_survey_config_matches_exact_offline_replay_contract():
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-VariableImpedance-TT",
    play=False,
  )

  protocol = _prepare_survey_measurement_config(cfg, PROVISIONAL_CAPS_N_M_S)

  assert cfg.metrics["cat_soft"].params["imp_limit"] == list(PROVISIONAL_CAPS_N_M_S)
  assert cfg.metrics["substep_impulse_rows"].params["enabled"] is False
  assert protocol == {
    "cat_replay": {"tau": 0.95, "min_p": 0.0, "imp_seed": 0.001, "imp_max_p_live": 0.0},
    "velocity_cat": {"limit_rad_s": 3.1415, "max_p": 0.5, "detection": "substep"},
    "physics_dt_s": 0.002,
    "control_decimation": 10,
    "impulse_window_substeps": 25,
    "contact_row_diagnostic_enabled": False,
  }


@pytest.mark.parametrize(
  ("name", "wrong"),
  (("limit", 3.0), ("max_p", 0.4), ("vel_detection", "control_rate")),
)
def test_survey_config_fails_closed_on_velocity_cat_identity_drift(name: str, wrong: object):
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-VariableImpedance-TT",
    play=False,
  )
  cfg.metrics["cat_soft"].params[name] = wrong

  with pytest.raises(RuntimeError, match=name):
    _prepare_survey_measurement_config(cfg, PROVISIONAL_CAPS_N_M_S)


@pytest.mark.parametrize(
  ("field", "wrong"),
  (
    ("func", object()),
    ("per_substep", False),
    ("reduce", "mean"),
    ("sensor_name", "wrong_contact"),
    ("subtract_baseline", False),
    ("event_window_substeps", 24),
    (
      "joint_names",
      tuple(reversed(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"))),
    ),
  ),
)
def test_survey_config_fails_closed_on_impulse_measurement_identity_drift(
  field: str, wrong: object
):
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-VariableImpedance-TT",
    play=False,
  )
  term = cfg.metrics["substep_impulse"]
  if field in ("func", "per_substep", "reduce"):
    setattr(term, field, wrong)
  elif field == "joint_names":
    term.params["robot_cfg"] = SceneEntityCfg("robot", joint_names=wrong)
  else:
    term.params[field] = wrong

  with pytest.raises(RuntimeError, match=field):
    _prepare_survey_measurement_config(cfg, PROVISIONAL_CAPS_N_M_S)


def test_violating_read_attributes_responsible_joint_by_normalized_cat_activation():
  trace = {
    "lambda_per_joint": np.array([[[1.5, 1.4, 0.0, 0.0, 0.0, 0.0]]]),
    "substep_rolling_per_joint": np.zeros((10, 1, 6), dtype=np.float64),
    "substep_episode_id": np.zeros((10, 1), dtype=np.int64),
    "episode_id": np.zeros((1, 1), dtype=np.int64),
  }
  trace["substep_rolling_per_joint"][-1, 0] = trace["lambda_per_joint"][0, 0]
  event_labels = np.zeros((10, 1), dtype=np.int64)
  margins = np.array([[[0.5, 0.4, -1.0, -1.0, -1.0, -1.0]]])
  # Joint 1 has the larger raw excess, but joint 2 has the larger normalized CaT response.
  unit_per_joint = np.array([[[0.25, 1.0, 0.0, 0.0, 0.0, 0.0]]])

  records, _ = _associate_violating_reads(
    trace,
    active=np.array([[True]]),
    margins=margins,
    unit_per_joint=unit_per_joint,
    event_labels=event_labels,
    valid=np.array([[True]]),
  )

  assert records[0]["responsible_joint"] == 1
  assert records[0]["violating_joints"] == [0, 1]


def test_violating_read_associates_each_joint_with_its_own_source_event():
  trace = {
    "lambda_per_joint": np.zeros((4, 1, 6), dtype=np.float64),
    "substep_rolling_per_joint": np.zeros((40, 1, 6), dtype=np.float64),
    "substep_episode_id": np.zeros((40, 1), dtype=np.int64),
    "episode_id": np.zeros((4, 1), dtype=np.int64),
  }
  trace["lambda_per_joint"][3, 0, :2] = [1.5, 1.4]
  trace["substep_rolling_per_joint"][30, 0, 0] = 1.5
  trace["substep_rolling_per_joint"][39, 0, 1] = 1.4
  margins = np.full((4, 1, 6), -1.0, dtype=np.float64)
  margins[3, 0, :2] = [0.5, 0.4]
  unit_per_joint = np.zeros((4, 1, 6), dtype=np.float64)
  unit_per_joint[3, 0, :2] = [0.5, 1.0]
  event_labels = np.full((40, 1), -1, dtype=np.int64)
  event_labels[10, 0] = 0
  event_labels[35, 0] = 1

  records, by_event = _associate_violating_reads(
    trace,
    active=np.array([[False], [False], [False], [True]]),
    margins=margins,
    unit_per_joint=unit_per_joint,
    event_labels=event_labels,
    valid=np.ones((4, 1), dtype=bool),
  )

  assert records == [
    {
      "control_step": 3,
      "env_id": 0,
      "episode_id": 0,
      "responsible_joint": 1,
      "violating_joints": [0, 1],
      "source_substep": 39,
      "responsible_contact_event_ids": [1],
      "contact_event_ids": [0, 1],
      "physical_event_association_ambiguous": True,
      "joint_associations": {
        "0": {
          "source_substep": 30,
          "contact_event_ids": [0],
          "physical_event_association_ambiguous": False,
        },
        "1": {
          "source_substep": 39,
          "contact_event_ids": [1],
          "physical_event_association_ambiguous": False,
        },
      },
    }
  ]
  assert by_event == {0: [(3, 0)], 1: [(3, 0)]}


def test_candidate_windows_record_separate_deltas_winners_and_event_pressure():
  summaries = _candidate_activation_window_summaries(
    active=np.array([[True], [True], [False], [True]]),
    done=np.array([[False], [True], [False], [False]]),
    delta_velocity=np.array([[0.1], [0.4], [0.0], [0.05]]),
    unit_delta_impulse=np.array([[1.0], [0.5], [0.0], [0.25]]),
    max_p=0.2,
  )

  assert len(summaries) == 2
  first = summaries[0]
  assert first["env_id"] == 0
  assert first["start_step"] == 0
  assert first["end_step"] == 1
  assert first["reads"] == 2
  assert first["event_pressure"] == 0.28
  assert first["delta_velocity"]["max"] == 0.4
  assert first["delta_impulse"]["max"] == 0.2
  assert first["combined_delta"]["max"] == 0.4
  assert first["winner_reads"] == {"velocity": 1, "impulse": 1, "tie": 0}
  assert summaries[1]["winner_reads"] == {"velocity": 0, "impulse": 0, "tie": 1}
  assert summaries[1]["event_pressure"] == 0.05
  assert summaries[1]["right_censored"] is True
  assert summaries[1]["censor_reason"] == "trace_end"


def test_physical_event_read_counts_do_not_double_count_ambiguous_reads_as_exclusive():
  records = [
    {"control_step": 0, "env_id": 0, "contact_event_ids": [3]},
    {"control_step": 1, "env_id": 0, "contact_event_ids": [3, 4]},
    {"control_step": 2, "env_id": 0, "contact_event_ids": [4]},
  ]

  counts = _physical_event_read_counts(records, event_ids=[3, 4])

  assert counts == [
    {"event_id": 3, "exclusive_reads": 1, "ambiguous_read_associations": 1},
    {"event_id": 4, "exclusive_reads": 1, "ambiguous_read_associations": 1},
  ]


def test_banked_canary_uniform_ninety_percent_diagnostic_threshold_activates_reproducibly():
  import json
  from pathlib import Path

  artifact = (
    Path(__file__).resolve().parents[1]
    / "docs/results/assets/2026-08-14_z1_vic_seed2_canary/victt_seed2_eval.json"
  )
  payload = json.loads(artifact.read_text())
  episodes = payload["episodes"]
  peaks = np.asarray([row["joint_impulse_peak_n_m_s"] for row in episodes])
  provisional = np.asarray(PROVISIONAL_CAPS_N_M_S)
  diagnostic = np.asarray(DIAGNOSTIC_LIMITS_N_M_S)

  assert PROVISIONAL_CAPS_N_M_S == (0.82, 1.64, 0.82, 0.82, 0.82, 0.82)
  assert DIAGNOSTIC_LIMITS_N_M_S == (0.738, 1.476, 0.738, 0.738, 0.738, 0.738)
  assert EXPECTED_CHECKPOINT_SHA256 == payload["checkpoint"]["sha256"]
  assert EXPECTED_FIXED_POPULATION_SHA256 == payload["initial_population_sha256"]
  assert FIXED_ENVS == payload["protocol"]["num_envs"] == 64
  assert FIXED_SEED == payload["protocol"]["evaluation_seed"] == 2026081202
  assert len(episodes) == 64
  assert int((peaks > provisional).any(axis=1).sum()) == 0
  assert int((peaks > diagnostic).any(axis=1).sum()) == 60
  assert (peaks > diagnostic).sum(axis=0).tolist() == [0, 0, 60, 0, 0, 0]
  assert peaks.max(axis=0)[2] == 0.8111855387687683
