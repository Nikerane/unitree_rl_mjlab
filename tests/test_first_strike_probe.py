"""Pure contract tests for the corrected first-strike CPU qualification probe."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROBE_DIR = (
  Path(__file__).resolve().parents[1]
  / "docs/results/assets/2026-07-25_first_strike_reward"
)
sys.path.insert(0, str(PROBE_DIR))

import probe_first_strike as probe


def _trace(offset: float = 0.0) -> dict[str, object]:
  action_tape = [[0.0, 0.0, -1.0], [0.0, 0.0, -0.5]]
  physical = {
    "contact": [False] * 10 + [True] * 10,
    "head_position_m": (
      [[0.0, 0.0, 0.100 + offset]] * 10
      + [[0.0, 0.0, 0.097 + offset]] * 10
    ),
    "clamped_depth_m": [0.0] * 10 + [0.001] * 10,
    "net_axial_force_n": [0.0] * 10 + [12.0] * 10,
    "joint_speed_rad_s": [[0.0] * 6] * 10 + [[0.1] * 6] * 10,
  }
  return {
    "action_tape": action_tape,
    "action_tape_digest": probe.canonical_digest(action_tape),
    "physical": physical,
    "trace_digest": probe.canonical_digest(physical),
  }


def _arm(
  *,
  impact: list[float],
  delivered: list[float],
  finalized: int | None,
  productive: bool,
) -> dict[str, object]:
  combined = [a + b for a, b in zip(impact, delivered, strict=True)]
  return {
    "discounted": sum((0.99**i) * value for i, value in enumerate(combined)),
    "undiscounted": sum(combined),
    "impact_stream": impact,
    "delivered_stream": delivered,
    "impact_payout_steps": [i for i, value in enumerate(impact) if value > 0.0],
    "delivered_payout_steps": [
      i for i, value in enumerate(delivered) if value > 0.0
    ],
    "finalization_step": finalized,
    "finalization_reason": "success" if finalized is not None else "none",
    "productive": productive,
  }


def _record(
  episode_id: str,
  checkpoint: dict[str, object],
  *,
  reset_seed: int,
  action_seed: int,
) -> dict[str, object]:
  c = _arm(
    impact=[0.0, 0.16],
    delivered=[0.0, 0.04],
    finalized=None,
    productive=False,
  )
  dprime = _arm(
    impact=[0.0, 0.16],
    delivered=[0.0, 0.04],
    finalized=1,
    productive=True,
  )
  f = copy.deepcopy(dprime)
  e = copy.deepcopy(dprime)
  return {
    "episode_id": episode_id,
    "checkpoint_path": checkpoint["path"],
    "stratum": checkpoint["stratum"],
    "training_seed": checkpoint["training_seed"],
    "split": checkpoint["split"],
    "reset_seed": reset_seed,
    "action_seed": action_seed,
    "physical_trace_digests": {
      "C": "",
      "D-prime": "",
      "F": "",
      "E": "",
    },
    "returns": {"C": c, "D-prime": dprime, "F": f, "E": e},
    "tracker_streams": {
      arm: {
        "started": [False, True],
        "finalized": [False, True],
        "productive": [False, True],
        "reason": [0, 1],
      }
      for arm in ("D-prime", "F", "E")
    },
    "dprime_audit": {
      "wrapper_parent_impact_raw": [0.0, 1.0],
      "c_legacy_impact_raw": [0.0, 1.0],
      "wrapper_parent_delivered_raw": [0.0, 1.0],
      "c_legacy_delivered_raw": [0.0, 1.0],
      "impact_latch_oracle": 1.0,
      "impact_manager_payout_raw": 1.0,
      "delivered_sum_oracle": 1.0,
      "delivered_manager_payout_raw": 1.0,
    },
  }


def _raw_fixture(tmp_path: Path) -> dict[str, object]:
  checkpoint_paths = [tmp_path / "alpha.pt", tmp_path / "beta.pt"]
  checkpoint_paths[0].write_bytes(b"checkpoint-alpha")
  checkpoint_paths[1].write_bytes(b"checkpoint-beta")
  checkpoints = [
    {
      "stratum": "alpha",
      "training_seed": 0,
      "split": "calibration",
      "path": checkpoint_paths[0].name,
      "sha256": probe.sha256_path(checkpoint_paths[0]),
    },
    {
      "stratum": "beta",
      "training_seed": 2,
      "split": "validation",
      "path": checkpoint_paths[1].name,
      "sha256": probe.sha256_path(checkpoint_paths[1]),
    },
  ]
  manifest = {
    "schema_version": 2,
    "checkpoints": checkpoints,
    "reset_seeds": [11],
    "action_seeds": [21],
    "expected_episode_count": 2,
    "tasks": {
      "C": "task-c",
      "D-prime": "task-dprime",
      "F": "task-f",
      "E": "task-e",
    },
    "reward_weights": {
      arm: {"impact_progress": 8.0, "delivered_impulse": 2.0}
      for arm in ("C", "D-prime", "F", "E")
    },
    "relevant_source_sha256": {},
  }
  traces = [
    {"episode_id": "alpha_s0_r11", **_trace(0.0)},
    {"episode_id": "beta_s2_r11", **_trace(0.01)},
  ]
  records = [
    _record("alpha_s0_r11", checkpoints[0], reset_seed=11, action_seed=21),
    _record("beta_s2_r11", checkpoints[1], reset_seed=11, action_seed=21),
  ]
  for trace, record in zip(traces, records, strict=True):
    record["physical_trace_digests"] = {
      arm: trace["trace_digest"] for arm in ("C", "D-prime", "F", "E")
    }
  return {
    "schema_version": 3,
    "manifest": manifest,
    "physical_traces": traces,
    "reward_records": records,
    "phase_rows": [
      {
        "phase": i,
        "matched_event_digest": "literal-event",
        "contact_detected": True,
        "productive": True,
        "v_precontact_m_s": 1.5,
        "v_true_m_s": 1.5,
        "impact_payout_count": 1,
        "delivered_payout_count": 1,
        "delivered_n_s": 0.3088,
        "depth_at_contact_m": 0.001,
        "peak_depth_m": 0.030,
        "progress_m": 0.029,
        "finalization_reason": "success",
      }
      for i in range(10)
    ],
    "normalizer_samples": {
      "legacy_terminal_n_s": 0.6094,
      "event_values_n_s": [0.3088] * 8,
    },
    "runtime_s": 1.25,
    "development_replay": {
      "D-prime": {"late_payout": 0.0},
      "F": {"late_payout": 0.0},
      "E": {"late_payout": 0.0},
    },
    "crash": None,
  }


def test_legacy_reference_keeps_success_termination_and_reads_terminal_snapshot():
  class FakeReference:
    def playback_length(self):
      return 3

    def playback_target(self, _step):
      return torch.zeros(1, 3)

  class FakeEnv:
    def __init__(self):
      self.device = "cpu"
      self.terminations = {"nail_driven": object()}
      self.cfg = SimpleNamespace(auto_reset=False)
      self.steps = 0
      self.acc = SimpleNamespace(delivered=torch.zeros(1))
      self.termination_manager = SimpleNamespace(
        get_term=lambda name: torch.tensor(
          [name == "nail_driven" and self.steps == 2]
        )
      )

    def step(self, _action):
      self.steps += 1
      self.acc.delivered[0] = (0.1, 0.6094, 9.0)[self.steps - 1]
      terminated = torch.tensor([self.steps == 2])
      truncated = torch.tensor([False])
      return {}, torch.zeros(1), terminated, truncated, {}

  env = FakeEnv()
  result = probe.drive_reference_episode(
    env,
    FakeReference(),
    head=lambda: torch.zeros(1, 3),
    delivered_accumulator=env.acc,
    action_scale=0.05,
    hold_steps=6,
  )

  assert "nail_driven" in env.terminations
  assert env.steps == 2
  assert result == {
    "nail_driven": True,
    "terminated": True,
    "truncated": False,
    "steps": 2,
    "delivered_n_s": pytest.approx(0.6094),
  }

  bad = FakeEnv()
  bad.termination_manager = SimpleNamespace(
    get_term=lambda _name: torch.tensor([False])
  )
  with pytest.raises(RuntimeError, match="nail_driven"):
    probe.drive_reference_episode(
      bad,
      FakeReference(),
      head=lambda: torch.zeros(1, 3),
      delivered_accumulator=bad.acc,
      action_scale=0.05,
      hold_steps=6,
    )


def test_collect_failure_reasons_reports_all_failed_strata_and_overall(
  tmp_path: Path,
):
  raw = _raw_fixture(tmp_path)
  summary = probe.build_summary(raw, manifest_sha256="1" * 64, raw_sha256="2" * 64)
  summary["population"]["completed_episodes"] = 1
  raw["reward_records"][1]["checkpoint_path"] = "alpha.pt"
  raw["physical_traces"][0]["trace_digest"] = "3" * 64
  raw["reward_records"][1]["returns"]["D-prime"]["finalization_step"] = 0

  reasons = probe.collect_failure_reasons(
    summary,
    raw,
    root=tmp_path,
    verify_artifact_hashes=False,
    verify_source_hashes=False,
  )

  assert [reason.split(":", 1)[0] for reason in reasons] == [
    "population",
    "checkpoint_binding",
    "digests",
    "replay_semantics",
  ]
  assert "alpha" in reasons[2]
  assert "beta" in reasons[1]


def test_raw_schema_stores_one_physical_trace_per_episode(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  path = tmp_path / "raw.npz"

  probe.write_raw(raw, path)
  loaded = probe.read_raw(path)
  probe.validate_persisted_raw(
    path,
    root=tmp_path,
    expected_checkpoints=raw["manifest"]["checkpoints"],
    expected_reset_seeds=(11,),
    expected_action_seeds=(21,),
    verify_source_hashes=False,
  )

  assert set(loaded) == probe.RAW_SCHEMA_KEYS
  assert len(loaded["physical_traces"]) == 2
  assert len(loaded["reward_records"]) == 2
  assert {row["episode_id"] for row in loaded["physical_traces"]} == {
    "alpha_s0_r11",
    "beta_s2_r11",
  }
  assert all("physical" not in row for row in loaded["reward_records"])
  assert all(
    set(row["returns"]) == {"C", "D-prime", "F", "E"}
    for row in loaded["reward_records"]
  )


def test_summary_is_aggregate_only_and_binds_raw_sha256(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  manifest_path = tmp_path / "manifest.json"
  raw_path = tmp_path / "raw.npz"
  manifest_path.write_text(json.dumps(raw["manifest"], sort_keys=True))
  probe.write_raw(raw, raw_path)

  summary = probe.build_summary(
    raw,
    manifest_sha256=probe.sha256_path(manifest_path),
    raw_sha256=probe.sha256_path(raw_path),
    raw_size_bytes=raw_path.stat().st_size,
  )
  encoded = json.dumps(summary, sort_keys=True)

  assert summary["artifact_sha256"]["raw"] == probe.sha256_path(raw_path)
  assert summary["artifact_sha256"]["manifest"] == probe.sha256_path(manifest_path)
  assert summary["raw_artifact"] == {
    "schema_version": 3,
    "size_bytes": raw_path.stat().st_size,
    "physical_trace_count": 2,
    "reward_record_count": 2,
  }
  assert "episode_rows" not in encoded
  assert "physical_traces" not in encoded
  assert "action_tape" not in encoded
  assert summary["population"]["completed_episodes"] == 2
  assert set(summary["aggregates"]["by_arm"]) == {"C", "D-prime", "F", "E"}


def test_validator_recomputes_action_and_trace_digests(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  path = tmp_path / "raw.npz"
  probe.write_raw(raw, path)
  probe.validate_persisted_raw(
    path,
    root=tmp_path,
    expected_checkpoints=raw["manifest"]["checkpoints"],
    expected_reset_seeds=(11,),
    expected_action_seeds=(21,),
    verify_source_hashes=False,
  )

  raw["physical_traces"][0]["physical"]["head_position_m"][1][2] = 123.0
  probe.write_raw(raw, path)
  with pytest.raises(ValueError, match="trace digest.*alpha"):
    probe.validate_persisted_raw(
      path,
      root=tmp_path,
      expected_checkpoints=raw["manifest"]["checkpoints"],
      expected_reset_seeds=(11,),
      expected_action_seeds=(21,),
      verify_source_hashes=False,
    )


def test_validator_binds_checkpoint_path_to_stratum_seed_and_split(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  raw["reward_records"][1]["checkpoint_path"] = "alpha.pt"

  with pytest.raises(ValueError, match="checkpoint tuple.*beta"):
    probe.validate_raw_payload(
      raw,
      root=tmp_path,
      expected_checkpoints=raw["manifest"]["checkpoints"],
      expected_reset_seeds=(11,),
      expected_action_seeds=(21,),
      verify_checkpoint_hashes=True,
      verify_source_hashes=False,
    )


def test_manifest_hashes_relevant_source_files_before_payout(tmp_path: Path):
  source_a = tmp_path / "first_strike.py"
  source_b = tmp_path / "probe_first_strike.py"
  source_a.write_text("tracker = 'production'\n")
  source_b.write_text("probe = 'production'\n")
  paths = ("first_strike.py", "probe_first_strike.py")

  hashes = probe.hash_relevant_sources(tmp_path, paths)
  assert hashes == {
    "first_strike.py": probe.sha256_path(source_a),
    "probe_first_strike.py": probe.sha256_path(source_b),
  }

  hashes["first_strike.py"] = "f" * 64
  with pytest.raises(ValueError, match="relevant source hash.*first_strike"):
    probe.validate_file_hashes(tmp_path, hashes, label="relevant source")


def test_dprime_replay_latches_first_impact_and_sums_delivered(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  record = raw["reward_records"][0]
  record["returns"]["C"] = _arm(
    impact=[0.0, 0.16, 0.24],
    delivered=[0.0, 0.04, 0.06],
    finalized=None,
    productive=False,
  )
  record["returns"]["D-prime"] = _arm(
    impact=[0.0, 0.0, 0.16],
    delivered=[0.0, 0.0, 0.10],
    finalized=2,
    productive=True,
  )
  for arm in ("F", "E"):
    record["returns"][arm] = _arm(
      impact=[0.0, 0.0, 0.16],
      delivered=[0.0, 0.0, 0.10],
      finalized=2,
      productive=True,
    )
  for arm in ("D-prime", "F", "E"):
    record["tracker_streams"][arm] = {
      "started": [False, True, True],
      "finalized": [False, False, True],
      "productive": [False, False, True],
      "reason": [0, 0, 1],
    }
  record["dprime_audit"] = {
    "wrapper_parent_impact_raw": [0.0, 1.0, 1.5],
    "c_legacy_impact_raw": [0.0, 1.0, 1.5],
    "wrapper_parent_delivered_raw": [0.0, 2.0, 3.0],
    "c_legacy_delivered_raw": [0.0, 2.0, 3.0],
    "impact_latch_oracle": 1.0,
    "impact_manager_payout_raw": 1.0,
    "delivered_sum_oracle": 5.0,
    "delivered_manager_payout_raw": 5.0,
  }

  probe.validate_reward_record(record)
  assert record["returns"]["D-prime"]["impact_stream"] == [0.0, 0.0, 0.16]
  assert record["returns"]["D-prime"]["delivered_stream"] == [0.0, 0.0, 0.10]


def test_dprime_f_e_share_finalization_payout_step(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  record = raw["reward_records"][0]
  for arm in ("D-prime", "F", "E"):
    assert record["returns"][arm]["finalization_step"] == 1
    assert record["returns"][arm]["impact_payout_steps"] == [1]
    assert record["returns"][arm]["delivered_payout_steps"] == [1]
  probe.validate_reward_record(record)

  record["returns"]["E"] = _arm(
    impact=[0.16, 0.0],
    delivered=[0.04, 0.0],
    finalized=0,
    productive=True,
  )
  record["tracker_streams"]["E"] = {
    "started": [True, True],
    "finalized": [True, True],
    "productive": [True, True],
    "reason": [1, 1],
  }
  with pytest.raises(ValueError, match="shared finalization"):
    probe.validate_reward_record(record)


def test_all_arms_keep_weights_eight_and_two(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  probe.validate_reward_weights(raw["manifest"]["reward_weights"])

  raw["manifest"]["reward_weights"]["D-prime"]["impact_progress"] = 5.0
  with pytest.raises(ValueError, match="weights 8/2.*D-prime"):
    probe.validate_reward_weights(raw["manifest"]["reward_weights"])

  assert probe.RESET_SEEDS == tuple(range(2026072700, 2026072732))
  assert probe.ACTION_SEEDS == tuple(range(2026072800, 2026072832))
  assert set(probe.TASKS) == {"C", "D-prime", "F", "E"}


def test_policy_sampling_is_stochastic_once():
  calls = []

  def policy(obs, *, stochastic_output):
    calls.append((obs, stochastic_output))
    return torch.tensor([[2.0, -2.0, 0.25]])

  action = probe.sample_stochastic_action(policy, "literal-observation", clip=1.0)

  assert calls == [("literal-observation", True)]
  assert torch.equal(action, torch.tensor([[1.0, -1.0, 0.25]]))


def test_invalid_summary_is_durable_after_partial_crash(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  raw["physical_traces"] = raw["physical_traces"][:1]
  raw["reward_records"] = raw["reward_records"][:1]
  raw["crash"] = "RuntimeError: checkpoint beta failed"
  manifest_path = tmp_path / "manifest.json"
  raw_path = tmp_path / "raw.npz"
  summary_path = tmp_path / "summary.json"
  manifest_path.write_text(json.dumps(raw["manifest"], sort_keys=True))

  valid = probe.persist_final_artifacts(
    raw,
    manifest_path=manifest_path,
    raw_path=raw_path,
    summary_path=summary_path,
    root=tmp_path,
    expected_checkpoints=raw["manifest"]["checkpoints"],
    expected_reset_seeds=(11,),
    expected_action_seeds=(21,),
    verify_source_hashes=False,
  )

  assert valid is False
  with np.load(raw_path, allow_pickle=False) as saved:
    assert set(saved.files) == probe.NPZ_KEYS
  summary = json.loads(summary_path.read_text())
  assert summary["valid"] is False
  assert "episode_rows" not in json.dumps(summary)
  assert any("crash" in item for item in summary["failure_reasons"])
  assert any("population" in item for item in summary["failure_reasons"])


def test_recursive_schema_rejects_duplicate_trace_payload(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  raw["reward_records"][0]["physical_traces"] = {
    arm: copy.deepcopy(raw["physical_traces"][0]["physical"])
    for arm in ("C", "D-prime", "F", "E")
  }
  with pytest.raises(ValueError, match="raw schema"):
    probe.validate_raw_payload(
      raw,
      root=tmp_path,
      expected_checkpoints=raw["manifest"]["checkpoints"],
      expected_reset_seeds=(11,),
      expected_action_seeds=(21,),
      verify_source_hashes=False,
    )


def test_manifest_provenance_tree_and_versions(tmp_path: Path):
  repo_file = tmp_path / "src/tasks/hammer/mdp/rewards.py"
  asset_file = tmp_path / "hammer_z1_env/assets/hammer.xml"
  repo_file.parent.mkdir(parents=True)
  asset_file.parent.mkdir(parents=True)
  repo_file.write_text("reward = 1\n")
  asset_file.write_text("<mujoco/>\n")

  provenance = probe.build_relevant_provenance(
    tmp_path,
    tmp_path,
    repo_paths=("src/tasks/hammer/mdp/rewards.py",),
    asset_paths=("hammer_z1_env/assets/hammer.xml",),
    require_git_clean=False,
  )

  assert provenance["tree_sha256"] == probe.canonical_digest(
    provenance["file_sha256"]
  )
  assert set(provenance["versions"]) == {
    "python",
    "torch",
    "numpy",
    "mjlab",
    "mujoco",
    "mujoco_warp",
    "rsl_rl",
  }
  asset_file.write_text("<mujoco model='changed'/>\n")
  with pytest.raises(ValueError, match="relevant source hash.*hammer.xml"):
    probe.validate_relevant_provenance(
      provenance,
      tmp_path,
      tmp_path,
      require_git_clean=False,
    )


def test_manager_term_scaling_applies_weight_and_dt_once():
  iterable_terms = [
    ("impact_progress", [8.0]),
    ("delivered_impulse", [2.0]),
  ]
  assert probe.scaled_treatment_terms(iterable_terms, step_dt=0.02) == (
    pytest.approx(0.16),
    pytest.approx(0.04),
  )


def test_population_rejects_duplicate_old_or_relabelled_rows(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  probe.validate_population(
    raw,
    expected_checkpoints=raw["manifest"]["checkpoints"],
    expected_reset_seeds=(11,),
    expected_action_seeds=(21,),
  )

  raw["reward_records"][1] = copy.deepcopy(raw["reward_records"][0])
  raw["physical_traces"][1] = copy.deepcopy(raw["physical_traces"][0])
  with pytest.raises(ValueError, match="population.*duplicate|duplicate.*population"):
    probe.validate_population(
      raw,
      expected_checkpoints=raw["manifest"]["checkpoints"],
      expected_reset_seeds=(11,),
      expected_action_seeds=(21,),
    )
