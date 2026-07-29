"""Contracts for the fixed-reset 500 Hz single-policy impulse recorder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import diag_impulse_trace as diag


class _OrderedSim:
  def __init__(self, events: list[str]):
    self._events = events

  def forward(self) -> None:
    self._events.append("sim.forward")

  def sense(self) -> None:
    self._events.append("sim.sense")


class _OrderedObservationManager:
  def __init__(self, events: list[str]):
    self._events = events

  def compute(self, *, update_history: bool = False):
    self._events.append(f"observation.compute:{update_history}")
    return {"actor": "restored-observation"}


class _OrderedEnv:
  def __init__(self, events: list[str]):
    self._events = events
    self.sim = _OrderedSim(events)
    self.observation_manager = _OrderedObservationManager(events)
    self.obs_buf = None

  def reset(self):
    self._events.append("env.reset")
    return {"actor": "random-reset-observation"}, {}


class _OrderedWrapper:
  def __init__(self, events: list[str]):
    self._events = events

  def get_observations(self):
    self._events.append("wrapped.get_observations")
    return "policy-observation"


def test_fixed_reset_restore_refreshes_observation_before_policy_inference(
  monkeypatch,
):
  """A restored state, not the discarded random reset, must feed the policy."""
  events: list[str] = []
  env = _OrderedEnv(events)
  wrapped = _OrderedWrapper(events)
  fixed_reset = {
    "reset_state_digest": "approved-digest",
    "reset_state": {"realized": {}},
  }

  def load_fixed_reset(path):
    events.append(f"load_fixed_reset:{path}")
    return fixed_reset

  def restore_reset_state(actual_env, reset):
    assert actual_env is env
    assert reset is fixed_reset
    events.append("restore_reset_state")
    # The established replay helper owns the one forward/sense refresh.
    actual_env.sim.forward()
    actual_env.sim.sense()
    return reset["reset_state_digest"]

  monkeypatch.setattr(diag, "load_fixed_reset", load_fixed_reset, raising=False)
  monkeypatch.setattr(
    diag, "restore_reset_state", restore_reset_state, raising=False
  )

  obs, restored, digest = diag._reset_checkpoint_for_inference(
    env,
    wrapped,
    "fixed-reset.json",
  )

  assert obs == "policy-observation"
  assert restored is fixed_reset
  assert digest == "approved-digest"
  assert env.obs_buf == {"actor": "restored-observation"}
  assert events == [
    "load_fixed_reset:fixed-reset.json",
    "env.reset",
    "restore_reset_state",
    "sim.forward",
    "sim.sense",
    "observation.compute:True",
    "wrapped.get_observations",
  ]


def test_tampered_fixed_reset_fails_before_observation_or_inference(monkeypatch):
  events: list[str] = []
  env = _OrderedEnv(events)
  wrapped = _OrderedWrapper(events)
  fixed_reset = {
    "reset_state_digest": "tampered",
    "reset_state": {"realized": {}},
  }

  monkeypatch.setattr(
    diag, "load_fixed_reset", lambda _path: fixed_reset, raising=False
  )

  def reject_tampered(_env, _reset):
    events.append("restore_reset_state")
    raise ValueError("recorded reset state digest mismatch")

  monkeypatch.setattr(
    diag, "restore_reset_state", reject_tampered, raising=False
  )

  with pytest.raises(ValueError, match="digest mismatch"):
    diag._reset_checkpoint_for_inference(env, wrapped, "tampered.json")

  assert events == ["env.reset", "restore_reset_state"]
  assert env.obs_buf is None


def test_historical_checkpoint_reset_remains_available_without_fixed_envelope():
  events: list[str] = []
  env = _OrderedEnv(events)

  class HistoricalWrapper(_OrderedWrapper):
    def reset(self):
      self._events.append("wrapped.reset")
      return "historical-policy-observation", {}

  obs, restored, digest = diag._reset_checkpoint_for_inference(
    env,
    HistoricalWrapper(events),
    None,
  )

  assert obs == "historical-policy-observation"
  assert restored is None
  assert digest is None
  assert events == ["wrapped.reset"]


def _valid_payload(*, substeps: int = 20, controls: int = 2) -> dict[str, np.ndarray]:
  assert substeps == controls * 10
  control_step_index = np.repeat(np.arange(controls, dtype=np.int64), 10)
  payload = {
    "t_s": np.arange(substeps, dtype=np.float64) * 0.002,
    "head_position_m": np.zeros((substeps, 3), dtype=np.float64),
    "head_velocity_m_s": np.zeros((substeps, 3), dtype=np.float64),
    "contact": np.zeros(substeps, dtype=bool),
    "nail_position_m": np.zeros((substeps, 3), dtype=np.float64),
    "nail_depth_m": np.zeros(substeps, dtype=np.float64),
    "joint_velocity_rad_s": np.zeros((substeps, 6), dtype=np.float64),
    "joint_position_rad": np.zeros((substeps, 6), dtype=np.float64),
    "joint_velocity_post_integration_rad_s": np.zeros(
      (substeps, 6), dtype=np.float64
    ),
    "nail_depth_post_integration_m": np.zeros(substeps, dtype=np.float64),
    "qfrc_constraint_abs": np.zeros((substeps, 6), dtype=np.float64),
    "lambda_windowed_constraint_read_n_m_s": np.zeros(
      (substeps, 6), dtype=np.float64
    ),
    "delivered_impulse_n_s": np.zeros(substeps, dtype=np.float64),
    "axial_force_n": np.zeros(substeps, dtype=np.float64),
    "control_step_index": control_step_index,
    "tracker_started": np.zeros(substeps, dtype=bool),
    "tracker_finalized": np.zeros(substeps, dtype=bool),
    "tracker_productive": np.zeros(substeps, dtype=bool),
    "tracker_reason": np.zeros(substeps, dtype=np.int64),
    "tracker_v_precontact_m_s": np.zeros(substeps, dtype=np.float64),
    "tracker_delivered_n_s": np.zeros(substeps, dtype=np.float64),
    "tracker_delivered_transverse_n_s": np.zeros(
      substeps, dtype=np.float64
    ),
    "tracker_peak_depth_m": np.zeros(substeps, dtype=np.float64),
    "tracker_depth_at_contact_m": np.zeros(substeps, dtype=np.float64),
    "tracker_first_contact_time_s": np.zeros(substeps, dtype=np.float64),
    "control_step": np.arange(controls, dtype=np.int64),
    "action": np.zeros((controls, 3), dtype=np.float32),
    "strike_phase_control": np.zeros(controls, dtype=np.float64),
    "strike_ref_error_control_m": np.zeros((controls, 3), dtype=np.float64),
    "first_strike_available": np.asarray(True),
    "quality_available": np.asarray(False),
  }
  return payload


def _identity() -> dict[str, object]:
  return {
    "mode": "checkpoint",
    "campaign": "fq3x8",
    "arm": "FQ",
    "training_seed": 23,
    "task": "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
    "checkpoint_sha256": "a" * 64,
    "fixed_reset_envelope": "fixed-reset.json",
    "reset_state_digest": "b" * 64,
    "code_revision": "c" * 40,
    "asset_revision": "d" * 40,
    "terminal_reason": "step_limit",
  }


def test_cli_accepts_optional_fixed_reset_without_breaking_historical_modes():
  parser = diag._build_arg_parser()
  companion = parser.parse_args(
    [
      "--ckpt",
      "model.pt",
      "--fixed-reset-envelope",
      "fixed-reset.json",
    ]
  )
  historical = parser.parse_args(["--ckpt", "model.pt"])
  reference = parser.parse_args(["--reference"])

  assert companion.fixed_reset_envelope == "fixed-reset.json"
  assert historical.fixed_reset_envelope is None
  assert reference.reference is True


def test_substep_hook_separates_preintegration_and_postintegration_state(
  monkeypatch,
):
  """A one-substep state transition catches phase drift and Lambda lag."""

  class ResolvedCfg:
    def __init__(self, name, *, joint_names=None, site_names=None):
      del name
      self.joint_ids = list(range(6)) if joint_names is not None else None
      self.site_ids = [0] if site_names is not None else None

    def resolve(self, _scene):
      return None

  monkeypatch.setattr(diag, "SceneEntityCfg", ResolvedCfg)
  events: list[str] = []
  robot_data = SimpleNamespace(
    joint_pos=torch.full((1, 6), 1.0),
    joint_vel=torch.full((1, 6), 2.0),
    site_pos_w=torch.tensor([[[0.1, 0.2, 0.3]]]),
    site_vel_w=torch.tensor([[[0.4, 0.5, 0.6, 0.0, 0.0, 0.0]]]),
    qfrc=torch.full((1, 6), -4.0),
  )
  robot_data._joint_dof_field = lambda name: (
    robot_data.qfrc
    if name == "qfrc_constraint"
    else pytest.fail(f"unexpected field {name}")
  )
  nail_data = SimpleNamespace(
    joint_pos=torch.tensor([[0.010]]),
    site_pos_w=torch.tensor([[[0.5, 0.0, 0.02]]]),
  )
  accumulator = SimpleNamespace(impulse=torch.full((1, 6), 1.0))
  delivered = SimpleNamespace(delivered=torch.tensor([0.1]))

  def sim_step():
    events.append("sim.step")
    robot_data.joint_pos.fill_(7.0)
    robot_data.joint_vel.fill_(8.0)
    nail_data.joint_pos.fill_(0.020)

  def orig_substep():
    events.append("orig_substep")
    accumulator.impulse.fill_(2.0)
    delivered.delivered.fill_(0.2)

  env = SimpleNamespace(
    scene={
      "robot": SimpleNamespace(data=robot_data),
      "nail_block": SimpleNamespace(data=nail_data),
      "hammer_nail_contact": SimpleNamespace(
        data=SimpleNamespace(found=torch.ones(1, 1))
      ),
      "hammer_nail_impulse": SimpleNamespace(
        data=SimpleNamespace(force=torch.tensor([[[0.0, 0.0, -5.0]]]))
      ),
    },
    sim=SimpleNamespace(step=sim_step),
    metrics_manager=SimpleNamespace(compute_substep=orig_substep),
    device="cpu",
    physics_dt=0.002,
    extras={},
  )
  setattr(env, diag._ENV_SUBSTEP_IMPULSE_ATTR, accumulator)
  setattr(env, diag._ENV_SUBSTEP_DELIVERED_ATTR, delivered)

  recorder = diag._install_substep_hook(env, 0)
  recorder.record_control_step(
    action=np.array([0.1, 0.2, 0.3]),
    strike_phase=0.25,
    strike_ref_error_m=np.array([0.01, 0.02, 0.03]),
  )
  env.sim.step()
  env.metrics_manager.compute_substep()
  payload = recorder.to_payload(physics_dt=0.002)

  assert events == ["sim.step", "orig_substep"]
  np.testing.assert_allclose(payload["head_position_m"][0], [0.1, 0.2, 0.3])
  np.testing.assert_allclose(payload["head_velocity_m_s"][0], [0.4, 0.5, 0.6])
  np.testing.assert_allclose(payload["joint_velocity_rad_s"][0], 2.0)
  assert payload["nail_depth_m"][0] == pytest.approx(0.010)
  np.testing.assert_allclose(
    payload["joint_position_rad"][0], 7.0
  )
  np.testing.assert_allclose(
    payload["joint_velocity_post_integration_rad_s"][0], 8.0
  )
  assert payload["nail_depth_post_integration_m"][0] == pytest.approx(0.020)
  # orig_substep ran first, so this includes the current contact sample.
  np.testing.assert_allclose(
    payload["lambda_windowed_constraint_read_n_m_s"][0], 2.0
  )
  assert payload["delivered_impulse_n_s"][0] == pytest.approx(0.2)
  assert payload["control_step_index"].tolist() == [0]


def test_payload_has_exact_2ms_substep_rate_and_separate_control_rate_arrays():
  payload = _valid_payload()
  schema = diag._validate_trace_payload(payload, physics_dt=0.002)

  np.testing.assert_array_equal(
    payload["t_s"], np.arange(20, dtype=np.float64) * 0.002
  )
  assert payload["action"].shape == (2, 3)
  assert payload["strike_phase_control"].shape == (2,)
  assert payload["strike_ref_error_control_m"].shape == (2, 3)
  assert payload["control_step_index"].tolist() == [0] * 10 + [1] * 10
  assert schema["rates"]["substep"]["dt_s"] == 0.002
  assert schema["rates"]["control"]["dt_s"] == 0.02
  assert schema["phase_contract"]["joint_velocity_rad_s"] == "pre_integration"
  assert (
    schema["phase_contract"]["joint_position_rad"]
    == "post_integration_legality"
  )
  assert (
    schema["phase_contract"]["joint_velocity_post_integration_rad_s"]
    == "post_integration_legality"
  )
  assert (
    schema["phase_contract"][
      "lambda_windowed_constraint_read_n_m_s"
    ]
    == "post_shipped_accumulator_current_substep"
  )


def test_quality_unavailable_omits_quality_measurements_instead_of_storing_zeros():
  payload = _valid_payload()
  diag._validate_trace_payload(payload, physics_dt=0.002)
  assert not bool(payload["quality_available"])
  assert not any(key.startswith("tracker_contact_quality") for key in payload)
  assert "tracker_contact_point_w" not in payload

  payload["tracker_contact_quality"] = np.zeros(20)
  with pytest.raises(ValueError, match="quality unavailable"):
    diag._validate_trace_payload(payload, physics_dt=0.002)


@pytest.mark.parametrize(
  ("mutation", "message"),
  [
    (
      lambda payload: payload.__setitem__(
        "head_position_m", payload["head_position_m"][:-1]
      ),
      "substep length",
    ),
    (
      lambda payload: payload["qfrc_constraint_abs"].__setitem__(
        (3, 2), np.nan
      ),
      "nonfinite",
    ),
    (
      lambda payload: payload["t_s"].__setitem__(1, 0.003),
      "2 ms",
    ),
    (
      lambda payload: payload["control_step_index"].__setitem__(0, 2),
      "control_step_index",
    ),
  ],
)
def test_payload_validation_rejects_nonfinite_misaligned_or_wrong_rate_data(
  mutation,
  message,
):
  payload = _valid_payload()
  mutation(payload)
  with pytest.raises(ValueError, match=message):
    diag._validate_trace_payload(payload, physics_dt=0.002)


def test_atomic_leaf_publish_binds_hashes_and_preserves_old_leaf_on_failure(
  tmp_path,
):
  leaf = tmp_path / "FQ" / "23"
  leaf.mkdir(parents=True)
  (leaf / "old-sentinel").write_text("old", encoding="utf-8")

  metadata = diag._write_trace_leaf(
    leaf,
    _valid_payload(),
    identity=_identity(),
    physics_dt=0.002,
  )

  assert not (leaf / "old-sentinel").exists()
  assert (leaf / "trace.npz").is_file()
  assert (leaf / "metadata.json").is_file()
  assert not tuple(leaf.parent.glob(f".{leaf.name}.staging-*"))
  assert not tuple(leaf.parent.glob(f".{leaf.name}.backup-*"))
  assert metadata["checkpoint_sha256"] == "a" * 64
  assert metadata["reset_state_digest"] == "b" * 64
  assert metadata["payload_sha256"]
  assert metadata["npz_sha256"] == hashlib.sha256(
    (leaf / "trace.npz").read_bytes()
  ).hexdigest()
  on_disk = json.loads((leaf / "metadata.json").read_text(encoding="utf-8"))
  assert on_disk == metadata
  assert diag.validate_trace_leaf(leaf)["payload_sha256"] == metadata[
    "payload_sha256"
  ]

  previous_npz = (leaf / "trace.npz").read_bytes()
  invalid = _valid_payload()
  invalid["axial_force_n"][0] = np.inf
  with pytest.raises(ValueError, match="nonfinite"):
    diag._write_trace_leaf(
      leaf,
      invalid,
      identity=_identity(),
      physics_dt=0.002,
    )
  assert (leaf / "trace.npz").read_bytes() == previous_npz

  partial = tmp_path / ".partial.staging-deadbeef"
  partial.mkdir()
  np.savez(partial / "trace.npz", **_valid_payload())
  with pytest.raises(ValueError, match="incomplete"):
    diag.validate_trace_leaf(partial)


def test_interrupted_publish_recovers_the_single_parked_backup(
  tmp_path,
  monkeypatch,
):
  """A restart must restore the good leaf parked before the publish rename."""
  final = tmp_path / "FQ" / "23"
  final.mkdir(parents=True)
  (final / "old-sentinel").write_text("old", encoding="utf-8")
  staged = tmp_path / ".staged" / "FQ" / "23"
  staged.mkdir(parents=True)
  (staged / "new-sentinel").write_text("new", encoding="utf-8")
  real_replace = diag.os.replace

  def interrupt_after_parking(source, destination):
    real_replace(source, destination)
    if Path(source) == final and ".backup-" in Path(destination).name:
      raise SystemExit("simulated process interruption")

  monkeypatch.setattr(diag.os, "replace", interrupt_after_parking)
  with pytest.raises(SystemExit, match="simulated process interruption"):
    diag._publish_staged_leaf(staged, final)

  assert not final.exists()
  assert len(tuple(final.parent.glob(f".{final.name}.backup-*"))) == 1

  monkeypatch.setattr(diag.os, "replace", real_replace)
  diag.recover_trace_leaf_backup(final)

  assert (final / "old-sentinel").read_text(encoding="utf-8") == "old"
  assert not tuple(final.parent.glob(f".{final.name}.backup-*"))
  assert (staged / "new-sentinel").is_file()


def test_backup_recovery_fails_closed_when_multiple_candidates_exist(tmp_path):
  final = tmp_path / "FQ" / "23"
  final.parent.mkdir(parents=True)
  for suffix in ("first", "second"):
    final.with_name(f".{final.name}.backup-{suffix}").mkdir()

  with pytest.raises(ValueError, match="ambiguous"):
    diag.recover_trace_leaf_backup(final)


def test_noncompanion_output_preserves_exact_legacy_npz_and_directory(
  tmp_path,
):
  """Historical runs overwrite their two files without replacing the directory."""
  out = tmp_path / "historical"
  out.mkdir()
  (out / "unrelated.txt").write_text("keep", encoding="utf-8")
  (out / "metadata.json").write_text("pre-existing", encoding="utf-8")
  payload = _valid_payload()
  payload["qfrc_constraint_abs"][:] = 2.0
  payload["axial_force_n"][:] = 3.0
  payload["lambda_windowed_constraint_read_n_m_s"][:] = 4.0
  payload["delivered_impulse_n_s"][:] = 5.0
  payload["joint_velocity_rad_s"][:] = -6.0
  payload["cat_delta"] = np.full(20, 0.25)
  j_limit = [1.64, 3.28, 1.64, 1.64, 1.64, 1.64]

  result = diag._write_outputs(
    out,
    payload,
    physics_dt=0.002,
    j_limit=j_limit,
    companion_identity=None,
  )

  assert result is None
  assert (out / "trace.png").is_file()
  assert (out / "unrelated.txt").read_text(encoding="utf-8") == "keep"
  assert (out / "metadata.json").read_text(encoding="utf-8") == "pre-existing"
  with np.load(out / "trace.npz") as trace:
    assert trace.files == [
      "t",
      "qfrc",
      "f_axial",
      "impulse",
      "delivered",
      "contact",
      "qv",
      "delta",
      "j_limit",
    ]
    np.testing.assert_array_equal(trace["t"], np.arange(20) * 0.002)
    np.testing.assert_array_equal(trace["qfrc"], np.full((20, 6), 2.0))
    np.testing.assert_array_equal(trace["f_axial"], np.full(20, 3.0))
    np.testing.assert_array_equal(trace["impulse"], np.full((20, 6), 4.0))
    np.testing.assert_array_equal(trace["delivered"], np.full(20, 5.0))
    np.testing.assert_array_equal(trace["contact"], np.zeros(20, dtype=bool))
    np.testing.assert_array_equal(trace["qv"], np.full((20, 6), 6.0))
    np.testing.assert_array_equal(trace["delta"], np.full(20, 0.25))
    np.testing.assert_array_equal(trace["j_limit"], np.asarray(j_limit))


def _resign_metadata(path: Path, mutation) -> None:
  metadata = json.loads(path.read_text(encoding="utf-8"))
  mutation(metadata)
  metadata.pop("metadata_sha256", None)
  encoded = json.dumps(
    metadata,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")
  metadata["metadata_sha256"] = hashlib.sha256(encoded).hexdigest()
  path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


@pytest.mark.parametrize(
  ("mutation", "message"),
  [
    (lambda value: value.pop("mode"), "mode"),
    (lambda value: value.__setitem__("mode", "reference"), "mode"),
    (lambda value: value.__setitem__("campaign", None), "campaign"),
    (lambda value: value.__setitem__("arm", ""), "arm"),
    (lambda value: value.__setitem__("training_seed", None), "training_seed"),
    (
      lambda value: value.__setitem__("checkpoint_sha256", "a" * 63),
      "checkpoint_sha256",
    ),
    (
      lambda value: value.__setitem__("reset_state_digest", None),
      "reset_state_digest",
    ),
    (
      lambda value: value.__setitem__("code_revision", "c" * 39),
      "code_revision",
    ),
    (
      lambda value: value.__setitem__("asset_revision", "g" * 40),
      "asset_revision",
    ),
    (
      lambda value: value.__setitem__("fixed_reset_envelope", None),
      "fixed_reset_envelope",
    ),
    (
      lambda value: value.__setitem__("terminal_reason", None),
      "terminal_reason",
    ),
  ],
)
def test_companion_identity_mutations_fail_on_write_and_resume(
  tmp_path,
  mutation,
  message,
):
  bad_identity = _identity()
  mutation(bad_identity)
  with pytest.raises(ValueError, match=message):
    diag._write_trace_leaf(
      tmp_path / "rejected",
      _valid_payload(),
      identity=bad_identity,
      physics_dt=0.002,
    )

  leaf = tmp_path / "published"
  diag._write_trace_leaf(
    leaf,
    _valid_payload(),
    identity=_identity(),
    physics_dt=0.002,
  )
  _resign_metadata(leaf / "metadata.json", mutation)
  with pytest.raises(ValueError, match=message):
    diag.validate_trace_leaf(leaf)
