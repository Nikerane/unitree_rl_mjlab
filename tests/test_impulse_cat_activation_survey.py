"""Pure analysis tests for the no-learning impulse-CaT activation survey."""

from dataclasses import dataclass
import inspect
from pathlib import Path
from types import SimpleNamespace

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
BRIDGE_TARGET_SHA = "57000e958bbafa2c62929652d3b76fd6ed571c9867bee3735c14baf0ca57d8de"
DOSE_P01_SHA = "92f1d97c8ff1476cb26c0b648478a0bc3c522e4e1eb7087389fb8e0d6bf73f86"
DOSE_P03_SHA = "4c0a665fffc077d488630a28b258c1593050f969b4f6227147c3594097dc4efd"
DOSE_P025_SHA = "5efc45c11c02dd400ad7d417bcdeefe9d271038ab43007f08a2820ceca0e744d"


def _install_fake_population_runtime(monkeypatch, events):
  import torch

  import mjlab.envs
  import mjlab.rl
  import mjlab.tasks.registry

  @dataclass
  class AgentCfg:
    num_steps_per_env: int = survey.TRAINING_LIKE_STEPS
    clip_actions: float = 1.0

  class Env:
    def __init__(self, *, cfg, device, render_mode):
      del render_mode
      self.device = device
      self.num_envs = cfg.scene.num_envs
      self.control_steps = 0
      self.max_episode_length = 1
      self.observation_manager = SimpleNamespace(compute=self._compute_observation)

    def _reset_idx(self, env_ids=None):
      del env_ids
      events.append("reset")

    def _compute_observation(self):
      events.append("observation")
      return torch.zeros((self.num_envs, 1))

  class Wrapper:
    def __init__(self, env, *, clip_actions):
      del clip_actions
      self.env = env

    def reset(self):
      self.env._reset_idx()
      return self.env.observation_manager.compute(), {}

    def step(self, actions):
      del actions
      self.env.control_steps += 1
      return self.env.observation_manager.compute(), None, None, None

    def close(self):
      pass

  class Runner:
    def __init__(self, wrapped, cfg, *, device):
      del wrapped, cfg, device

    def load(self, checkpoint, *, load_cfg, strict, map_location):
      del checkpoint, load_cfg, strict, map_location

    def get_inference_policy(self, *, device):
      del device
      return lambda observations, *, stochastic_output: torch.zeros_like(observations)

  class Recorder:
    def __init__(self, env):
      self.env = env

    def numpy_trace(self):
      scalar_shape = (self.env.control_steps, self.env.num_envs)
      joint_shape = (*scalar_shape, 6)
      return {
        "delta_velocity": np.zeros(scalar_shape, dtype=np.float32),
        "delta_impulse": np.zeros(scalar_shape, dtype=np.float32),
        "delta": np.zeros(scalar_shape, dtype=np.float32),
        "lambda_per_joint": np.zeros(joint_shape, dtype=np.float32),
      }

  env_cfg = SimpleNamespace(
    scene=SimpleNamespace(num_envs=None),
    seed=None,
    auto_reset=None,
  )
  monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", Env)
  monkeypatch.setattr(mjlab.rl, "MjlabOnPolicyRunner", Runner)
  monkeypatch.setattr(mjlab.rl, "RslRlVecEnvWrapper", Wrapper)
  monkeypatch.setattr(mjlab.tasks.registry, "load_env_cfg", lambda task, play: env_cfg)
  monkeypatch.setattr(mjlab.tasks.registry, "load_rl_cfg", lambda task: AgentCfg())
  monkeypatch.setattr(mjlab.tasks.registry, "load_runner_cls", lambda task: Runner)
  monkeypatch.setattr(survey, "_LiveSurveyRecorder", Recorder)
  monkeypatch.setattr(survey, "_prepare_survey_measurement_config", lambda cfg, caps: {})
  monkeypatch.setattr(survey, "_initial_population_sha256", lambda env: "population")

  def install_streams(env, *, reset_seed, observation_seed):
    del env, reset_seed, observation_seed
    events.append("install")

  monkeypatch.setattr(survey, "_install_evaluator_rng_streams", install_streams)


def _run_fake_population(*, stochastic, rng_seeds):
  return survey._run_population(
    checkpoint=Path("unused-model_499.pt"),
    device="cpu",
    num_envs=1,
    seed=2,
    rng_seeds=rng_seeds,
    steps=1,
    stochastic=stochastic,
  )


def test_evaluation_checkpoint_roles_are_exact_and_fail_closed(tmp_path, monkeypatch):
  checkpoint = tmp_path / "model_499.pt"
  checkpoint.write_bytes(b"control")
  monkeypatch.setattr(survey, "_sha256", lambda _: CONTROL_SHA)

  assert dict(survey.EVALUATION_CHECKPOINTS) == {
    "diag90_control": CONTROL_SHA,
    "diag90_target": TARGET_SHA,
    "bridge_p0_control": CONTROL_SHA,
    "bridge_p02_target": BRIDGE_TARGET_SHA,
    "dose_p0_control": CONTROL_SHA,
    "dose_p01_target": DOSE_P01_SHA,
    "dose_p02_target": BRIDGE_TARGET_SHA,
    "dose_p03_target": DOSE_P03_SHA,
    "dose_p025_exploratory": DOSE_P025_SHA,
    "p02_uniform09": "a99593b263a74944d60ac412bb1da733a36a29a1cd9f4eeeaed89906372595df",
    "p02_joint_stress": "509cc26a2e521a935bcbc8c342040c95c7d2e3d7fc105a52d5d9450518bd2ec7",
  }
  with pytest.raises(TypeError):
    survey.EVALUATION_CHECKPOINTS["unexpected_role"] = CONTROL_SHA
  assert inspect.signature(survey.compare_policy_evaluations).parameters[
    "expected_roles"
  ].default == ("diag90_control", "diag90_target")
  assert survey.validate_checkpoint_role(checkpoint, "diag90_control") == CONTROL_SHA
  with pytest.raises(RuntimeError, match="role/checkpoint SHA-256 mismatch"):
    survey.validate_checkpoint_role(checkpoint, "diag90_target")


def test_two_boundary_roles_bind_exact_checkpoint_and_training_cap_identities():
  """The new p=0.2 diagnostic leaves cannot be relabelled across cap geometries."""
  assert dict(survey.TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S) == {
    "p02_uniform09": (0.738, 1.476, 0.738, 0.738, 0.738, 0.738),
    "p02_joint_stress": (0.369, 0.246, 0.738, 0.369, 0.246, 0.0164),
  }
  assert survey.EVALUATION_CHECKPOINTS["p02_uniform09"] == (
    "a99593b263a74944d60ac412bb1da733a36a29a1cd9f4eeeaed89906372595df"
  )
  assert survey.EVALUATION_CHECKPOINTS["p02_joint_stress"] == (
    "509cc26a2e521a935bcbc8c342040c95c7d2e3d7fc105a52d5d9450518bd2ec7"
  )
  with pytest.raises(TypeError):
    survey.TWO_BOUNDARY_P02_TRAINING_CAPS_N_M_S["p02_uniform09"] = (0.0,) * 6
  assert survey.validate_training_cap_identity(
    "p02_uniform09", [0.738, 1.476, 0.738, 0.738, 0.738, 0.738]
  ) == (0.738, 1.476, 0.738, 0.738, 0.738, 0.738)
  with pytest.raises(ValueError, match="training-cap identity mismatch"):
    survey.validate_training_cap_identity(
      "p02_uniform09", [0.369, 0.246, 0.738, 0.369, 0.246, 0.0164]
    )
  with pytest.raises(ValueError, match="six finite numeric values"):
    survey.validate_training_cap_identity("p02_joint_stress", [0.369, float("nan")])


def test_environment_rng_consumption_cannot_advance_policy_action_noise():
  import torch

  seeds = survey.EvaluationRngSeeds.from_evaluation_seed(101)
  action = survey._TorchRngStream(seeds.action, "cpu")
  reset = survey._TorchRngStream(seeds.reset, "cpu")
  observation = survey._TorchRngStream(seeds.observation, "cpu")
  action_reference = survey._TorchRngStream(seeds.action, "cpu")

  action_first = action.run(lambda: torch.rand(4))
  reset.run(lambda: torch.rand(100))
  observation.run(lambda: torch.rand(200))
  action_second = action.run(lambda: torch.rand(4))
  reference_first = action_reference.run(lambda: torch.rand(4))
  reference_second = action_reference.run(lambda: torch.rand(4))

  torch.testing.assert_close(action_first, reference_first)
  torch.testing.assert_close(action_second, reference_second)


def test_evaluation_roles_derive_identical_rng_streams_from_one_base_seed():
  seeds_by_role = {
    role: survey.EvaluationRngSeeds.from_evaluation_seed(2)
    for role in survey.EVALUATION_CHECKPOINTS
  }

  assert seeds_by_role == {
    "diag90_control": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "diag90_target": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "bridge_p0_control": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "bridge_p02_target": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "dose_p0_control": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "dose_p01_target": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "dose_p02_target": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "dose_p03_target": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "dose_p025_exploratory": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "p02_uniform09": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
    "p02_joint_stress": survey.EvaluationRngSeeds(
      reset=10_000_021,
      observation=20_000_035,
      action=30_000_043,
    ),
  }


def test_four_role_local_smoke_seam_runs_two_envs_for_eight_steps(
  tmp_path, monkeypatch
):
  """The real smoke can use this seam once all four frozen checkpoint files are local."""
  events = []
  _install_fake_population_runtime(monkeypatch, events)
  role_hashes = {
    "dose_p0_control": CONTROL_SHA,
    "dose_p01_target": DOSE_P01_SHA,
    "dose_p02_target": BRIDGE_TARGET_SHA,
    "dose_p03_target": DOSE_P03_SHA,
  }
  checkpoint_roles: dict[Path, str] = {}
  for role in role_hashes:
    checkpoint = tmp_path / role / "model_499.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(role.encode())
    checkpoint_roles[checkpoint.resolve()] = role
  monkeypatch.setattr(
    survey,
    "_sha256",
    lambda checkpoint: role_hashes[checkpoint_roles[checkpoint.resolve()]],
  )

  population_hashes = []
  for checkpoint, role in checkpoint_roles.items():
    assert survey.validate_checkpoint_role(checkpoint, role) == role_hashes[role]
    trace, protocol = survey._run_population(
      checkpoint=checkpoint,
      device="cpu",
      num_envs=2,
      seed=2,
      rng_seeds=survey.EvaluationRngSeeds.from_evaluation_seed(2),
      steps=8,
      stochastic=True,
    )
    assert protocol["num_envs"] == 2
    assert protocol["control_steps"] == 8
    assert protocol["rng_streams"] == {
      "reset": 10_000_021,
      "observation": 20_000_035,
      "action": 30_000_043,
    }
    assert trace["delta_velocity"].shape == (8, 2)
    assert trace["delta_impulse"].shape == (8, 2)
    assert trace["delta"].shape == (8, 2)
    assert trace["lambda_per_joint"].shape == (8, 2, 6)
    assert all(np.isfinite(value).all() for value in trace.values())
    assert np.array_equal(trace["delta_impulse"], np.zeros((8, 2)))
    assert np.array_equal(
      trace["delta"], np.maximum(trace["delta_velocity"], trace["delta_impulse"])
    )
    population_hashes.append(protocol["initial_population_sha256"])
  assert population_hashes == ["population"] * 4


def test_stochastic_population_installs_rng_streams_before_initial_reset(
  monkeypatch,
):
  events = []
  _install_fake_population_runtime(monkeypatch, events)

  _run_fake_population(
    stochastic=True,
    rng_seeds=survey.EvaluationRngSeeds.from_evaluation_seed(2),
  )

  assert events[:3] == ["install", "reset", "observation"]


def test_fixed_mean_population_preserves_initial_reset_before_stream_install(
  monkeypatch,
):
  events = []
  _install_fake_population_runtime(monkeypatch, events)

  _run_fake_population(
    stochastic=False,
    rng_seeds=survey.EvaluationRngSeeds.from_evaluation_seed(2),
  )

  assert events[:3] == ["reset", "observation", "install"]


def test_run_population_rejects_rng_streams_not_derived_from_evaluation_seed(
  monkeypatch,
):
  events = []
  _install_fake_population_runtime(monkeypatch, events)

  with pytest.raises(ValueError, match="RNG streams must match evaluation seed"):
    _run_fake_population(
      stochastic=True,
      rng_seeds=survey.EvaluationRngSeeds(
        reset=10_000_022,
        observation=20_000_035,
        action=30_000_043,
      ),
    )


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
    "success": np.zeros((4, 1), dtype=bool),
    "timeout": np.array([[False], [False], [False], [True]]),
    "nail_depth_m": np.zeros((4, 1), dtype=np.float32),
    "delivered_total_n_s": np.zeros((4, 1), dtype=np.float32),
    "first_strike_productive": np.zeros((4, 1), dtype=bool),
    "first_strike_delivered_n_s": np.zeros((4, 1), dtype=np.float32),
    "substep_peak_qv_per_joint": np.zeros((4, 1, 6), dtype=np.float32),
    "vic_p": np.zeros((4, 1, 6), dtype=np.float32),
    "vic_kp": np.full((4, 1, 6), 1000.0, dtype=np.float32),
    "vic_kd": np.full((4, 1, 6), 100.0, dtype=np.float32),
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


def test_segment_compliance_uses_native_margin_at_float32_cap_boundary():
  """A native zero margin must not become a violation after float64 utilization reporting."""
  caps = (0.1,) * 6
  lam = np.full((1, 1, 6), np.float32(0.05), dtype=np.float32)
  lam[0, 0, 0] = np.float32(0.1)
  trace = {
    "lambda_per_joint": lam,
    "episode_id": np.zeros((1, 1), dtype=np.int64),
    "done": np.ones((1, 1), dtype=bool),
    "success": np.zeros((1, 1), dtype=bool),
    "timeout": np.ones((1, 1), dtype=bool),
    "nail_depth_m": np.zeros((1, 1), dtype=np.float32),
    "delivered_total_n_s": np.zeros((1, 1), dtype=np.float32),
    "first_strike_productive": np.zeros((1, 1), dtype=bool),
    "first_strike_delivered_n_s": np.zeros((1, 1), dtype=np.float32),
    "substep_peak_qv_per_joint": np.zeros((1, 1, 6), dtype=np.float32),
    "vic_p": np.zeros((1, 1, 6), dtype=np.float32),
    "vic_kp": np.full((1, 1, 6), 1000.0, dtype=np.float32),
    "vic_kd": np.full((1, 1, 6), 100.0, dtype=np.float32),
    "delta_velocity": np.zeros((1, 1), dtype=np.float64),
    "delta_impulse": np.zeros((1, 1), dtype=np.float64),
    "delta": np.zeros((1, 1), dtype=np.float64),
    "substep_contact": np.zeros((10, 1), dtype=bool),
    "substep_episode_id": np.zeros((10, 1), dtype=np.int64),
    "substep_rolling_per_joint": np.zeros((10, 1, 6), dtype=np.float32),
  }

  summary = survey.summarize_population(trace, caps=caps, first_episode_only=True)

  assert float(lam[0, 0, 0]) / caps[0] > 1.0
  compliance = summary["segment_compliance"]
  assert compliance["any_joint_violating_segments"] == 0
  assert compliance["any_joint_violation_rate"] == 0.0
  assert compliance["per_joint"]["joint1"]["violating_segments"] == 0
  assert compliance["per_joint"]["joint1"]["positive_margin_n_m_s"] == {
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


@pytest.mark.integration
def test_live_vic_recorder_captures_utility_velocity_and_gains_before_auto_reset():
  """Dropping the pre-reset hook would replace commanded gains with reset defaults."""
  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(survey.VIC_TASK, play=False)
  cfg.scene.num_envs = 2
  cfg.auto_reset = True
  _prepare_survey_measurement_config(cfg, PROVISIONAL_CAPS_N_M_S)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  recorder = survey._LiveSurveyRecorder(env)
  try:
    env.reset(seed=20260817)
    env.episode_length_buf.fill_(env.max_episode_length - 1)
    stiffness = torch.tensor(
      [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0], device=env.device
    ).repeat(2, 1)
    actions = torch.zeros((2, 12), device=env.device)
    actions[:, 6:] = stiffness

    env.step(actions)
    trace = recorder.numpy_trace()

    for name in (
      "success",
      "timeout",
      "nail_depth_m",
      "delivered_total_n_s",
      "first_strike_productive",
      "first_strike_delivered_n_s",
    ):
      assert trace[name].shape == (1, 2)
    for name in (
      "substep_peak_qv_per_joint",
      "vic_p",
      "vic_kp",
      "vic_kd",
    ):
      assert trace[name].shape == (1, 2, 6)
    assert trace["policy_action"].shape == (1, 2, 12)
    for value in trace.values():
      if np.issubdtype(value.dtype, np.floating):
        assert np.isfinite(value).all()

    assert trace["success"].tolist() == [[False, False]]
    assert trace["timeout"].tolist() == [[True, True]]
    np.testing.assert_allclose(trace["vic_p"][0], stiffness.cpu().numpy())
    np.testing.assert_allclose(trace["policy_action"][0], actions.cpu().numpy())
    np.testing.assert_allclose(
      env.action_manager.get_term("joint_stiffness").telemetry.p.cpu().numpy(),
      np.zeros((2, 6)),
    )
  finally:
    env.close()


@pytest.mark.integration
def test_live_vic_recorder_rejects_action_or_tracker_identity_drift():
  """Same-shaped side objects must not substitute for configured VIC seams."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(survey.VIC_TASK, play=False)
  cfg.scene.num_envs = 2
  _prepare_survey_measurement_config(cfg, PROVISIONAL_CAPS_N_M_S)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  try:
    stiffness = env.action_manager._terms["joint_stiffness"]
    env.action_manager._terms["joint_stiffness"] = object()
    with pytest.raises(RuntimeError, match="joint_stiffness action identity"):
      survey._LiveSurveyRecorder(env)
    env.action_manager._terms["joint_stiffness"] = stiffness

    index = env.metrics_manager.active_terms.index("first_strike")
    env.metrics_manager._term_cfgs[index].func = object()
    with pytest.raises(RuntimeError, match="first-strike tracker"):
      survey._LiveSurveyRecorder(env)
  finally:
    env.close()


def test_population_summary_reports_task_velocity_and_first_contact_gain_tradeoffs():
  """Omitting terminal utility or contact-time gains hides policy-shaping tradeoffs."""
  lam = np.full((2, 2, 6), 0.1, dtype=np.float32)
  contact = np.zeros((20, 2), dtype=bool)
  contact[10, :] = True
  qv = np.array(
    [
      [[1.0] * 6, [1.0] * 6],
      [[3.0] * 6, [1.0, 1.0, 3.2, 1.0, 1.0, 1.0]],
    ],
    dtype=np.float32,
  )
  vic_p = np.array(
    [
      [[-0.5] * 6, [-0.25] * 6],
      [[0.5] * 6, [0.25] * 6],
    ],
    dtype=np.float32,
  )
  trace = {
    "lambda_per_joint": lam,
    "episode_id": np.zeros((2, 2), dtype=np.int64),
    "done": np.array([[False, False], [True, True]]),
    "success": np.array([[False, False], [True, False]]),
    "timeout": np.array([[False, False], [False, True]]),
    "nail_depth_m": np.array([[0.0, 0.0], [0.032, 0.01]]),
    "delivered_total_n_s": np.array([[0.0, 0.0], [0.5, 0.2]]),
    "first_strike_productive": np.array(
      [[False, False], [True, False]]
    ),
    "first_strike_delivered_n_s": np.array([[0.0, 0.0], [0.4, 0.1]]),
    "substep_peak_qv_per_joint": qv,
    "vic_p": vic_p,
    "vic_kp": 1000.0 + 100.0 * vic_p,
    "vic_kd": 100.0 + 10.0 * vic_p,
    "delta_velocity": np.zeros((2, 2), dtype=np.float64),
    "delta_impulse": np.zeros((2, 2), dtype=np.float64),
    "delta": np.zeros((2, 2), dtype=np.float64),
    "substep_contact": contact,
    "substep_episode_id": np.zeros((20, 2), dtype=np.int64),
    "substep_rolling_per_joint": np.zeros((20, 2, 6), dtype=np.float32),
  }

  summary = survey.summarize_population(
    trace, caps=PROVISIONAL_CAPS_N_M_S, first_episode_only=True
  )

  utility = summary["utility"]
  assert utility["task_field_scope"] == "complete first episodes"
  assert utility["terminal_counts"] == {
    "population": 2,
    "terminal": 2,
    "success": 1,
    "timeout": 1,
    "productive_first_strike": 1,
  }
  assert utility["terminal_nail_depth_m"] == {
    "p50": pytest.approx(0.021),
    "p95": pytest.approx(0.0309),
    "p99": pytest.approx(0.03178),
    "max": pytest.approx(0.032),
  }
  assert utility["terminal_delivered_total_n_s"]["max"] == pytest.approx(0.5)
  assert utility["terminal_first_strike_delivered_n_s"]["p50"] == pytest.approx(0.25)
  velocity = utility["velocity_limit_compliance"]
  assert velocity["segments"] == 2
  assert velocity["any_joint_violating_segments"] == 1
  assert velocity["per_joint"]["joint3"]["violating_segments"] == 1
  contacts = utility["first_contact_vic_gains"]
  assert contacts["segments_with_contact"] == 2
  assert contacts["records"][0] == {
    "env_id": 0,
    "episode_id": 0,
    "contact_control_step": 1,
    "precontact_control_step": 0,
    "vic_p_precontact": pytest.approx([-0.5] * 6),
    "vic_kp_precontact": pytest.approx([950.0] * 6),
    "vic_kd_precontact": pytest.approx([95.0] * 6),
    "vic_p_at_contact": pytest.approx([0.5] * 6),
    "vic_kp_at_contact": pytest.approx([1050.0] * 6),
    "vic_kd_at_contact": pytest.approx([105.0] * 6),
    "right_censored": False,
  }


def test_stochastic_utility_summary_labels_task_values_as_censored_fragments():
  """A 24-step prefix must never be presented as complete task behavior."""
  trace = {
    "lambda_per_joint": np.full((1, 1, 6), 0.1, dtype=np.float32),
    "episode_id": np.zeros((1, 1), dtype=np.int64),
    "done": np.zeros((1, 1), dtype=bool),
    "success": np.zeros((1, 1), dtype=bool),
    "timeout": np.zeros((1, 1), dtype=bool),
    "nail_depth_m": np.zeros((1, 1), dtype=np.float32),
    "delivered_total_n_s": np.zeros((1, 1), dtype=np.float32),
    "first_strike_productive": np.zeros((1, 1), dtype=bool),
    "first_strike_delivered_n_s": np.zeros((1, 1), dtype=np.float32),
    "substep_peak_qv_per_joint": np.zeros((1, 1, 6), dtype=np.float32),
    "vic_p": np.zeros((1, 1, 6), dtype=np.float32),
    "vic_kp": np.full((1, 1, 6), 1000.0, dtype=np.float32),
    "vic_kd": np.full((1, 1, 6), 100.0, dtype=np.float32),
    "delta_velocity": np.zeros((1, 1), dtype=np.float64),
    "delta_impulse": np.zeros((1, 1), dtype=np.float64),
    "delta": np.zeros((1, 1), dtype=np.float64),
    "substep_contact": np.zeros((10, 1), dtype=bool),
    "substep_episode_id": np.zeros((10, 1), dtype=np.int64),
    "substep_rolling_per_joint": np.zeros((10, 1, 6), dtype=np.float32),
  }

  summary = survey.summarize_population(
    trace, caps=PROVISIONAL_CAPS_N_M_S, first_episode_only=False
  )

  assert summary["utility"]["task_field_scope"] == (
    "24-step stochastic task fragments; unfinished and post-reset segments are censored"
  )
  assert summary["utility"]["terminal_counts"]["terminal"] == 0


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
