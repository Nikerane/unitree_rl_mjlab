"""Contracts for the frozen R-/R0/R+ horizontal-route evaluation."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import analyze_vic_horizontal_routes as analysis
from scripts import evaluate_vic_horizontal_routes as evaluator
from scripts import impulse_cat_activation_survey as survey


CAPS = (0.369, 0.246, 0.738, 0.369, 0.246, 0.0164)


def test_policy_and_route_identities_bind_the_two_terminal_training_artifacts():
  assert evaluator.POLICY_SPECS == {
    "horizontal_routes_annealed": {
      "task": evaluator.HORIZONTAL_ANNEALED_TASK,
      "sha256": "2a434c50457455daab10b5ff32e1ee26e3b189b0092a315f3e881bb7f960156a",
    },
    "horizontal_routes_persistent": {
      "task": evaluator.HORIZONTAL_PERSISTENT_TASK,
      "sha256": "feb22dd5e5740e96348387c846ef395143316354f7e3163b5684f24db58f4e9e",
    },
  }
  assert evaluator.ROUTE_LABEL_BY_SIGN == {-1: "rminus", 0: "r0", 1: "rplus"}
  assert evaluator.TRAINING_CAPS_N_M_S == CAPS


@pytest.mark.parametrize(
  ("role", "task"),
  (
    (
      "horizontal_routes_annealed",
      "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
      "JointPosition-VariableImpedance-TT-HorizontalRoutes-Annealed",
    ),
    (
      "horizontal_routes_persistent",
      "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
      "JointPosition-VariableImpedance-TT-HorizontalRoutes-Persistent",
    ),
  ),
)
def test_registered_horizontal_config_is_validated_before_log_only_override(role, task):
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(task, play=False)
  protocol = evaluator.prepare_horizontal_measurement_config(cfg, role=role)

  assert protocol["training_config"] == {
    "imp_max_p": 0.2,
    "imp_limit_n_m_s": list(CAPS),
  }
  assert protocol["cat_replay"]["imp_max_p_live"] == 0.0
  assert cfg.metrics["cat_soft"].params["imp_max_p"] == 0.0
  assert cfg.metrics["cat_soft"].params["imp_limit"] == list(survey.PROVISIONAL_CAPS_N_M_S)


def test_forced_route_reset_preserves_sampler_then_overwrites_only_reset_envs():
  events: list[str] = []

  class Ref:
    def __init__(self):
      self.signs = torch.tensor([-1, 0, 1], dtype=torch.int8)

    def route_signs(self):
      return self.signs.clone()

    def set_route_signs(self, signs):
      self.signs.copy_(signs)

  ref = Ref()

  def sampled_reset(env_ids=None):
    events.append("sampled")
    ids = torch.arange(3) if env_ids is None else env_ids
    ref.signs[ids] = -1

  env = SimpleNamespace(num_envs=3, device="cpu", _reset_idx=sampled_reset)
  survey._install_forced_route_sign(env, ref=ref, route_sign=1)

  env._reset_idx(torch.tensor([1]))

  assert events == ["sampled"]
  assert ref.signs.tolist() == [-1, 1, 1]


@pytest.mark.integration
def test_live_recorder_emits_exact_route_and_imitation_gate_telemetry():
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(evaluator.HORIZONTAL_ANNEALED_TASK, play=False)
  cfg.scene.num_envs = 2
  cfg.auto_reset = True
  evaluator.prepare_horizontal_measurement_config(
    cfg, role="horizontal_routes_annealed"
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  recorder = survey._LiveSurveyRecorder(env, route_telemetry=True)
  try:
    env.reset(seed=20260822)
    recorder.route_reference.set_route_signs(torch.tensor([-1, 1]))
    env.step(torch.zeros((2, 12), device=env.device))
    trace = recorder.numpy_trace()

    assert trace["route_sign"].shape == (1, 2)
    assert trace["strike_phase"].shape == (1, 2)
    for name in (
      "hammer_head_pos_w",
      "assigned_reference_waypoint_w",
      "straight_reference_waypoint_w",
    ):
      assert trace[name].shape == (1, 2, 3)
      assert np.isfinite(trace[name]).all()
    assert trace["imitation_eligible"].shape == (1, 2)
    assert trace["imitation_eligible"].dtype == np.bool_
    r_imit = env.reward_manager.get_term_cfg("r_imit").func
    assert np.array_equal(trace["imitation_eligible"][-1], (~r_imit._contacted).cpu())
    amplitude = 0.020 * np.sin(2.0 * np.pi * trace["strike_phase"]) ** 2
    expected_x = (
      trace["straight_reference_waypoint_w"][:, :, 0]
      + trace["route_sign"] * amplitude
    )
    np.testing.assert_allclose(
      trace["assigned_reference_waypoint_w"][:, :, 0], expected_x, atol=1e-7
    )
    np.testing.assert_array_equal(
      trace["assigned_reference_waypoint_w"][:, :, 1:],
      trace["straight_reference_waypoint_w"][:, :, 1:],
    )
  finally:
    env.close()


def _route_trace(sign: int, head_x: tuple[float, float]) -> dict[str, np.ndarray]:
  steps, envs = 2, 2
  phase = np.full((steps, envs), 0.25, dtype=np.float32)
  straight = np.zeros((steps, envs, 3), dtype=np.float32)
  assigned = straight.copy()
  assigned[:, :, 0] = sign * 0.020
  head = straight.copy()
  head[:, 0, 0] = head_x[0]
  head[:, 1, 0] = head_x[1]
  return {
    "episode_id": np.zeros((steps, envs), dtype=np.int64),
    "route_sign": np.full((steps, envs), sign, dtype=np.int8),
    "strike_phase": phase,
    "hammer_head_pos_w": head,
    "assigned_reference_waypoint_w": assigned,
    "straight_reference_waypoint_w": straight,
    "imitation_eligible": np.ones((steps, envs), dtype=bool),
  }


def test_tracking_core_classifies_each_initial_episode_by_lowest_route_rmse():
  result = analysis.tracking_metrics(_route_trace(1, (0.018, 0.019)), forced_sign=1)

  assert result["eligibility"]["initial_episode_precontact_reads"] == 4
  assert result["eligibility"]["core_reads"] == 4
  assert result["eligibility"]["episodes_with_core"] == 2
  assert result["eligibility"]["assessable_coverage"] == 1.0
  assert result["eligibility"]["route_progress_failure"] is False
  assert result["assigned_reference_error_m"]["rmse"] == pytest.approx(
    np.sqrt((4e-6 + 1e-6) / 2)
  )
  classification = result["episode_route_classification"]
  assert classification["closest_route_counts"] == {
    "rminus": 0,
    "r0": 0,
    "rplus": 2,
  }
  assert classification["correct_episodes"] == 2
  assert classification["correct_fraction"] == 1.0
  assert classification["assigned_vs_next_rmse_margin_m"]["p50"] == pytest.approx(0.017)
  assert classification["wilson_lower_95"] > 1 / 3
  record = result["episode_records"][0]
  assert record["env_id"] == 0
  assert record["eligible_precontact_reads"] == 2
  assert record["core_reads"] == 2
  assert record["closest_route"] == "rplus"
  assert record["ambiguous_tie"] is False
  assert record["max_precontact_phase"] == pytest.approx(0.25)
  assert record["assigned_rmse_m"] == pytest.approx(0.002)
  assert record["next_best_rmse_m"] == pytest.approx(0.018)
  assert record["assigned_vs_next_rmse_margin_m"] == pytest.approx(0.016)
  assert record["median_horizontal_response_m"] == pytest.approx(0.018)
  assert result["horizontal_response_m"]["p50"] == pytest.approx(0.0185)
  assert result["diagnostic_rubric"] == {
    "assessable_coverage_ge_0_90": True,
    "assigned_classification_ge_0_90": True,
    "classification_gt_chance": True,
    "wilson_lower_gt_one_third": True,
    "positive_median_margin": True,
    "correct_side_sign": True,
    "route_conditioning_evidence": True,
    "reliable_cell_before_cross_route_ordering": True,
  }


def test_tracking_core_marks_no_precontact_core_progress_instead_of_claiming_tracking():
  trace = _route_trace(-1, (-0.02, -0.02))
  trace["strike_phase"].fill(0.5)

  result = analysis.tracking_metrics(trace, forced_sign=-1)

  assert result["eligibility"]["core_reads"] == 0
  assert result["eligibility"]["episodes_with_core"] == 0
  assert result["eligibility"]["route_progress_failure"] is True
  assert result["episode_route_classification"]["correct_fraction"] is None


def test_assigned_reference_error_weights_initial_episodes_not_controller_reads():
  trace = _route_trace(1, (0.012, 0.020))
  trace = {
    name: np.concatenate((value, value[:1]), axis=0)
    for name, value in trace.items()
  }
  trace["imitation_eligible"][1:, 0] = False

  result = analysis.tracking_metrics(trace, forced_sign=1)

  # The two episode RMSEs are 8 mm and 0 mm, even though they contribute one and
  # three eligible controller reads respectively.
  errors = result["assigned_reference_error_m"]
  assert errors["p50"] == pytest.approx(0.004)
  assert errors["p95"] == pytest.approx(0.0076)
  assert errors["rmse"] == pytest.approx(np.sqrt((0.008**2 + 0.0**2) / 2.0))


def test_equal_candidate_rmse_is_ambiguous_and_incorrect():
  trace = _route_trace(1, (0.010, 0.010))

  result = analysis.tracking_metrics(trace, forced_sign=1)

  classification = result["episode_route_classification"]
  assert classification["ambiguous_tie_episodes"] == 2
  assert classification["correct_episodes"] == 0
  assert classification["correct_fraction"] == 0.0
  assert result["episode_records"][0]["closest_route"] is None
  assert result["episode_records"][0]["ambiguous_tie"] is True


def test_signed_route_ordering_requires_rminus_then_r0_then_rplus_response():
  ordered = {
    "rminus": {"tracking": analysis.tracking_metrics(_route_trace(-1, (-0.019, -0.018)), forced_sign=-1)},
    "r0": {"tracking": analysis.tracking_metrics(_route_trace(0, (-0.001, 0.001)), forced_sign=0)},
    "rplus": {"tracking": analysis.tracking_metrics(_route_trace(1, (0.018, 0.019)), forced_sign=1)},
  }

  result = analysis.signed_route_ordering(ordered)

  assert result["strict_rminus_lt_r0_lt_rplus"] is True
  assert result["median_response_m"] == pytest.approx(
    {"rminus": -0.0185, "r0": 0.0, "rplus": 0.0185}
  )
  assert result["adjacent_gaps_m"] == pytest.approx(
    {"r0_minus_rminus": 0.0185, "rplus_minus_r0": 0.0185}
  )

  assessment = analysis.assess_policy_population(ordered)
  assert assessment["strict_signed_ordering"] is True
  assert assessment["all_route_cells_reliable"] is True
  assert assessment["reliable_route_conditioning"] is True


def test_population_analysis_keeps_route_primary_and_safety_utility_secondary():
  from tests.test_vic_impulse_two_boundary_p02_analysis import _trace as safety_trace

  trace = safety_trace((0.8, 0.2))
  trace.update(_route_trace(1, (0.018, 0.019)))

  result = analysis.analyze_population(trace, forced_sign=1)

  assert result["tracking"]["episode_route_classification"]["correct_fraction"] == 1.0
  assert result["task_utility"]["terminal_counts"] == {
    "population": 2,
    "terminal": 2,
    "success": 2,
    "timeout": 0,
    "productive_first_strike": 2,
  }
  assert result["task_utility"]["terminal_first_strike_delivered_n_s"]["p50"] == 1.0
  assert result["task_utility"]["initial_episode_observed"]["segments"] == 2
  assert result["task_utility"]["initial_episode_observed"]["success_seen"] == 2
  assert result["task_utility"]["initial_episode_observed"]["productive_seen"] == 2
  assert result["task_utility"]["initial_episode_observed"]["delivered_total_peak_n_s"]["p50"] == pytest.approx(1.1)
  assert result["impulse_caps"]["caps_n_m_s"] == list(CAPS)
  assert result["impulse_caps"]["segment_compliance"]["any_joint_violating_segments"] == 2
  assert result["impulse_caps"]["per_joint"]["joint1"]["violating_reads"] == 4
  assert result["true_velocity_risk"]["limit_rad_s"] == survey.VELOCITY_LIMIT_RAD_S
  assert result["episode_duration_ms"]["unit"] == "native observed initial-episode prefix"
  assert result["vic_gains"]["segments_with_contact"] == 2


def test_evaluator_writes_one_forced_route_over_exact_fixed_and_stochastic_populations(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  checkpoint = tmp_path / "model_499.pt"
  checkpoint.write_bytes(b"annealed")
  expected_sha = evaluator.POLICY_SPECS["horizontal_routes_annealed"]["sha256"]
  monkeypatch.setattr(evaluator, "_sha256", lambda _: expected_sha)
  monkeypatch.setattr(
    evaluator,
    "_evaluation_revisions",
    lambda: {"code_revision": "1" * 40, "asset_revision": survey.EXPECTED_EVALUATION_ASSET_REVISION},
  )
  monkeypatch.setattr(
    evaluator,
    "validate_evaluation_revision",
    lambda repository, revision: revision,
  )

  def fake_population(**kwargs):
    assert kwargs["task_id"] == evaluator.HORIZONTAL_ANNEALED_TASK
    assert kwargs["source_imp_max_p"] == 0.2
    assert kwargs["source_imp_limit_n_m_s"] == CAPS
    assert kwargs["forced_route_sign"] == -1
    num_envs = kwargs["num_envs"]
    steps = 3 if kwargs["steps"] is None else kwargs["steps"]
    scalar = (steps, num_envs)
    trace = {
      "episode_id": np.zeros(scalar, dtype=np.int64),
      "route_sign": np.full(scalar, -1, dtype=np.int8),
      "strike_phase": np.full(scalar, 0.25, dtype=np.float32),
      "hammer_head_pos_w": np.zeros((*scalar, 3), dtype=np.float32),
      "assigned_reference_waypoint_w": np.zeros((*scalar, 3), dtype=np.float32),
      "straight_reference_waypoint_w": np.zeros((*scalar, 3), dtype=np.float32),
      "imitation_eligible": np.ones(scalar, dtype=bool),
      "delta_impulse": np.zeros(scalar, dtype=np.float32),
    }
    return trace, {
      "seed": kwargs["seed"],
      "num_envs": num_envs,
      "control_steps": steps,
      "policy_mode": "sampled" if kwargs["stochastic"] else "mean",
      "auto_reset": kwargs["steps"] is not None,
      "initial_population_sha256": "a" * 64,
      "rng_streams": vars(kwargs["rng_seeds"]),
      "cat_replay": {"imp_max_p_live": 0.0},
      "training_config": {
        "imp_max_p": 0.2,
        "imp_limit_n_m_s": list(CAPS),
      },
      "forced_route_sign": -1,
      "actor_only_checkpoint_load": True,
    }

  monkeypatch.setattr(survey, "_run_population", fake_population)
  output = tmp_path / "output"

  payload = evaluator.run_horizontal_route_evaluation(
    checkpoint=checkpoint,
    role="horizontal_routes_annealed",
    route_sign=-1,
    output_dir=output,
    device="cpu",
  )

  assert payload["schema_version"] == 1
  assert payload["checkpoint"] == {
    "role": "horizontal_routes_annealed",
    "path": str(checkpoint.resolve()),
    "sha256": expected_sha,
  }
  assert payload["route"] == {"label": "rminus", "sign": -1}
  assert payload["protocol"]["live_imp_max_p"] == 0.0
  assert payload["protocol"]["actor_only_checkpoint_load"] is True
  assert tuple(payload["populations"]["training_like_sampled"]) == (
    "2",
    "2026081701",
    "2026081702",
  )
  assert sorted(path.name for path in output.iterdir()) == [
    "fixed_trace.npz",
    "summary.json",
    "training_like_seed_2026081701_trace.npz",
    "training_like_seed_2026081702_trace.npz",
    "training_like_seed_2_trace.npz",
  ]
  with np.load(output / "fixed_trace.npz", allow_pickle=False) as trace:
    assert np.array_equal(trace["route_sign"], np.full((3, 64), -1, dtype=np.int8))


def _analysis_leaf(tmp_path: Path, role: str, sign: int, code: str) -> Path:
  from tests.test_vic_impulse_two_boundary_p02_analysis import _trace as safety_trace

  label = evaluator.ROUTE_LABEL_BY_SIGN[sign]
  leaf = tmp_path / role / label
  output = leaf / "evaluation"
  output.mkdir(parents=True)
  head_x = {
    -1: (-0.019, -0.018),
    0: (-0.001, 0.001),
    1: (0.018, 0.019),
  }[sign]
  trace = safety_trace((0.8, 0.2))
  trace.update(_route_trace(sign, head_x))
  names = (
    "fixed_trace.npz",
    "training_like_seed_2_trace.npz",
    "training_like_seed_2026081701_trace.npz",
    "training_like_seed_2026081702_trace.npz",
  )
  for name in names:
    np.savez_compressed(output / name, **trace)
  task = evaluator.POLICY_SPECS[role]["task"]
  protocol = lambda seed: {  # noqa: E731
    "seed": seed,
    "initial_population_sha256": "a" * 64,
    "forced_route_sign": sign,
    "actor_only_checkpoint_load": True,
    "cat_replay": {"imp_max_p_live": 0.0},
    "training_config": {"imp_max_p": 0.2, "imp_limit_n_m_s": list(CAPS)},
  }
  summary = {
    "schema_version": 1,
    "task": task,
    "checkpoint": {
      "role": role,
      "path": f"/frozen/{role}/model_499.pt",
      "sha256": evaluator.POLICY_SPECS[role]["sha256"],
    },
    "code_revision": code,
    "asset_revision": evaluator.EXPECTED_ASSET_REVISION,
    "route": {"label": label, "sign": sign},
    "training_cap_identity_n_m_s": list(CAPS),
    "protocol": {
      "actor_only_checkpoint_load": True,
      "live_imp_max_p": 0.0,
      "fixed_seed": survey.FIXED_SEED,
      "stochastic_seeds": list(survey.POLICY_EVALUATION_STOCHASTIC_SEEDS),
    },
    "populations": {
      "fixed_mean": {"trace": "fixed_trace.npz", "protocol": protocol(survey.FIXED_SEED)},
      "training_like_sampled": {
        str(seed): {
          "trace": f"training_like_seed_{seed}_trace.npz",
          "protocol": protocol(seed),
        }
        for seed in survey.POLICY_EVALUATION_STOCHASTIC_SEEDS
      },
    },
  }
  (output / "summary.json").write_text(
    json.dumps(summary, sort_keys=True, allow_nan=False), encoding="utf-8"
  )
  manifest_names = (*names, "summary.json")
  (leaf / "SHA256SUMS").write_text(
    "".join(
      f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  evaluation/{name}\n"
      for name in manifest_names
    ),
    encoding="utf-8",
  )
  return leaf


def test_six_leaf_analysis_validates_manifests_and_keeps_each_seed_separate(tmp_path: Path):
  code = "c" * 40
  leaves = {
    role: {
      evaluator.ROUTE_LABEL_BY_SIGN[sign]: _analysis_leaf(tmp_path, role, sign, code)
      for sign in (-1, 0, 1)
    }
    for role in evaluator.POLICY_SPECS
  }

  payload = analysis.analyze_evaluation_leaves(leaves, expected_code_revision=code)

  assert tuple(payload["policies"]) == tuple(evaluator.POLICY_SPECS)
  for policy in payload["policies"].values():
    assert policy["fixed64"]["assessment"]["reliable_route_conditioning"] is True
    assert tuple(policy["training_like_by_seed"]) == ("2", "2026081701", "2026081702")
    assert all(
      population["assessment"]["reliable_route_conditioning"]
      for population in policy["training_like_by_seed"].values()
    )
  assert payload["provenance"]["checkpoint_sha256"] == {
    role: spec["sha256"] for role, spec in evaluator.POLICY_SPECS.items()
  }
  encoded = analysis.encode_analysis(payload)
  assert encoded == analysis.encode_analysis(payload)
  assert "NaN" not in encoded and "Infinity" not in encoded
