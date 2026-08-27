from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_vic_diagonal_fixed_starts as evaluator


def test_specialist_specs_bind_all_five_frozen_policies():
  assert {
    role: (spec["offset_mm"], spec["route_sign"], spec["sha256"])
    for role, spec in evaluator.POLICY_SPECS.items()
  } == {
    "fixed_m40mm_p02": (
      -40,
      -1,
      "a429523ded414e569783166677ee7d3725f3d1ef542d9792eab7733a1da07504",
    ),
    "fixed_m20mm_p02": (
      -20,
      -1,
      "1155bb102a080040b68ce483d40b7caf78b5049dffb596f1b813223597fbcd64",
    ),
    "fixed_0mm_p02": (0, 0, "2fa4dafd9a5291768ad459a368b8805bafa9571662a3a6876f4a3f2ded21f9e6"),
    "fixed_p20mm_p02": (20, 1, "f803f97f43a916c8e9af9c5756f46424d94dc686cd34153024443615940f17a4"),
    "fixed_p40mm_p02": (40, 1, "c6b3b0b9267345efc0b316f20d708b09edb2ae49b2a86fc5f0d984f542161d00"),
  }


def test_straight_tracking_summary_uses_equal_episode_units_and_ignores_start_anchor():
  steps, envs = 4, 2
  assigned = np.zeros((steps, envs, 3), dtype=np.float32)
  head = assigned.copy()
  head[1:, 0, 0] = 0.001
  head[1, 1, 0] = 0.009
  trace = {
    "episode_id": np.zeros((steps, envs), dtype=np.int64),
    "route_sign": np.full((steps, envs), -1, dtype=np.int64),
    "strike_phase": np.array(
      [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=np.float32
    ),
    "imitation_eligible": np.array(
      [[True, True], [True, True], [True, False], [True, False]], dtype=bool
    ),
    "hammer_head_pos_w": head,
    "assigned_reference_waypoint_w": assigned,
    "straight_reference_waypoint_w": assigned.copy(),
  }

  summary = evaluator.straight_tracking_summary(trace, expected_sign=-1)

  assert summary["eligible_episodes"] == 2
  assert summary["no_progress_episodes"] == 0
  assert summary["progress_coverage_rate"] == 1.0
  assert summary["conditional_progress_aggregate_equal_episode_rmse_m"] == pytest.approx(
    np.sqrt((0.001**2 + 0.009**2) / 2.0)
  )
  assert summary["conditional_progress_episode_rmse_m"]["median"] == pytest.approx(
    0.005
  )
  assert summary["conditional_progress_episode_rmse_m"]["p95"] == pytest.approx(
    0.0086
  )
  assert summary["conditional_progress_episode_rmse_m"]["max"] == pytest.approx(
    0.009
  )
  assert summary["conditional_progress_within_10mm_fraction_episode_mean"] == (
    pytest.approx(1.0)
  )


def test_checkpoint_validation_is_role_and_hash_bound(tmp_path, monkeypatch):
  checkpoint = tmp_path / "model_499.pt"
  checkpoint.write_bytes(b"frozen")
  monkeypatch.setattr(
    evaluator,
    "_sha256",
    lambda _: evaluator.POLICY_SPECS["fixed_0mm_p02"]["sha256"],
  )

  resolved, digest = evaluator.validate_checkpoint(
    checkpoint, role="fixed_0mm_p02"
  )
  assert resolved == checkpoint.resolve()
  assert digest == evaluator.POLICY_SPECS["fixed_0mm_p02"]["sha256"]
  with pytest.raises(RuntimeError, match="role/checkpoint"):
    evaluator.validate_checkpoint(checkpoint, role="fixed_p20mm_p02")


def test_first_strike_summary_separates_coverage_from_axial_velocity():
  trace = {
    "episode_id": np.zeros((1, 2), dtype=np.int64),
    "done": np.ones((1, 2), dtype=bool),
    "first_strike_started": np.array([[True, False]], dtype=bool),
    "first_strike_finalized": np.array([[True, False]], dtype=bool),
    "first_strike_productive": np.array([[True, False]], dtype=bool),
    "first_strike_delivered_n_s": np.array([[0.4, 0.0]], dtype=np.float32),
    "first_strike_precontact_nail_axial_velocity_m_s": np.array(
      [[2.0, 0.0]], dtype=np.float32
    ),
  }

  summary = evaluator._first_strike_summary(trace)

  assert summary["terminal_episode_coverage"] == 1.0
  assert summary["accepted_strike_rate_among_terminal_episodes"] == 0.5
  assert summary["accepted_strike_fraction_of_population"] == 0.5
  assert summary["accepted_strike_precontact_nail_axial_velocity_m_s"] == {
    "median": 2.0,
    "p95": 2.0,
    "max": 2.0,
  }
  assert summary["unconditional_first_strike_delivered_n_s"]["median"] == (
    pytest.approx(0.2)
  )
  assert summary["accepted_strike_delivered_n_s"]["median"] == pytest.approx(0.4)


def test_orchestrator_runs_exact_four_populations_without_forcing_route(
  tmp_path, monkeypatch
):
  role = "fixed_0mm_p02"
  checkpoint = tmp_path / "model_499.pt"
  checkpoint.write_bytes(b"frozen")
  output_dir = tmp_path / "evaluation"
  calls = []
  monkeypatch.setattr(
    evaluator, "_sha256", lambda _: evaluator.POLICY_SPECS[role]["sha256"]
  )
  monkeypatch.setattr(
    evaluator.survey,
    "_evaluation_revisions",
    lambda: {
      "code_revision": "1" * 40,
      "asset_revision": evaluator.EXPECTED_ASSET_REVISION,
    },
  )
  monkeypatch.setattr(
    evaluator.survey, "validate_evaluation_revision", lambda *args, **kwargs: "1" * 40
  )
  monkeypatch.setattr(
    evaluator, "validate_specialist_config", lambda cfg, qualified, role: None
  )
  import mjlab.tasks.registry

  monkeypatch.setattr(
    mjlab.tasks.registry, "load_env_cfg", lambda task, play: SimpleNamespace()
  )

  def run_population(**kwargs):
    calls.append(kwargs)
    return {"sample": np.zeros((1,), dtype=np.float32)}, {"task": kwargs["task_id"]}

  monkeypatch.setattr(evaluator.survey, "_run_population", run_population)
  monkeypatch.setattr(
    evaluator,
    "_population_summary",
    lambda trace, protocol, **kwargs: {"checked": True},
  )

  payload = evaluator.run_fixed_start_evaluation(
    checkpoint=checkpoint,
    role=role,
    output_dir=output_dir,
    device="cuda:0",
  )

  assert [(call["num_envs"], call["steps"], call["stochastic"]) for call in calls] == [
    (64, None, False),
    (4096, 24, True),
    (4096, 24, True),
    (4096, 24, True),
  ]
  assert [call["seed"] for call in calls] == [
    evaluator.survey.FIXED_SEED,
    *evaluator.survey.POLICY_EVALUATION_STOCHASTIC_SEEDS,
  ]
  for call in calls:
    assert call["task_id"] == evaluator.POLICY_SPECS[role]["task"]
    assert call["source_imp_max_p"] == 0.2
    assert call["source_imp_limit_n_m_s"] == evaluator.TRAINING_CAPS_N_M_S
    assert call["forced_route_sign"] is None
    assert call["route_telemetry"] is True
  assert payload["evaluation_protocol"]["actor_only"] is True
  assert (output_dir / "fixed_mean_trace.npz").is_file()
  assert (output_dir / "summary.json").is_file()
  with pytest.raises(FileExistsError):
    evaluator.run_fixed_start_evaluation(
      checkpoint=checkpoint,
      role=role,
      output_dir=output_dir,
      device="cuda:0",
    )


@pytest.mark.integration
def test_all_registered_specialist_configs_match_evaluation_treatment():
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  for role, spec in evaluator.POLICY_SPECS.items():
    evaluator.validate_specialist_config(
      load_env_cfg(str(spec["task"]), play=False),
      load_env_cfg(evaluator.survey.VIC_TASK, play=False),
      role=role,
    )


@pytest.mark.integration
def test_specialist_config_preflight_rejects_reset_action_and_default_pose_drift():
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  role = "fixed_p20mm_p02"
  task = str(evaluator.POLICY_SPECS[role]["task"])

  cfg = load_env_cfg(task, play=False)
  cfg.events["sample_strike_route_signs"].func = object()
  with pytest.raises(RuntimeError, match="sampler callable"):
    evaluator.validate_specialist_config(
      cfg, load_env_cfg(evaluator.survey.VIC_TASK, play=False), role=role
    )

  cfg = load_env_cfg(task, play=False)
  cfg.actions["joint_stiffness"].C += 0.01
  with pytest.raises(RuntimeError, match="joint_stiffness mapping"):
    evaluator.validate_specialist_config(
      cfg, load_env_cfg(evaluator.survey.VIC_TASK, play=False), role=role
    )

  cfg = load_env_cfg(task, play=False)
  cfg.scene.entities["robot"].init_state.joint_pos["joint2"] += 0.01
  with pytest.raises(RuntimeError, match="default pose"):
    evaluator.validate_specialist_config(
      cfg, load_env_cfg(evaluator.survey.VIC_TASK, play=False), role=role
    )
