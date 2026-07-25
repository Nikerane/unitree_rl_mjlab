"""Pure contract tests for the corrected first-strike CPU qualification probe."""

from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROBE_DIR = Path(__file__).resolve().parents[1] / (
  "docs/results/assets/2026-07-25_first_strike_reward"
)
sys.path.insert(0, str(PROBE_DIR))

import probe_first_strike as probe


def _trace(offset: float = 0.0) -> dict[str, object]:
  action_tape = [[0.0, 0.0, -1.0], [0.0, 0.0, -0.5]]
  physical = {
    "contact": [False] * 10 + [True] * 10,
    "head_position_m": [[0.0, 0.0, 0.100 + offset]] * 10
    + [[0.0, 0.0, 0.097 + offset]] * 10,
    "clamped_depth_m": [0.0] * 10 + [0.001] * 10,
    "net_axial_force_n": [0.0] * 10 + [12.0] * 10,
    "joint_speed_rad_s": [[0.0] * 6] * 10 + [[0.1] * 6] * 10,
  }
  return {
    "action_tape": action_tape, "physical": physical,
    "action_tape_digest": probe.canonical_digest(action_tape),
    "trace_digest": probe.canonical_digest(physical),
  }


def _arm(*, impact, delivered, finalized, productive) -> dict[str, object]:
  combined = [a + b for a, b in zip(impact, delivered, strict=True)]
  return {
    "discounted": sum((0.99**i) * value for i, value in enumerate(combined)),
    "undiscounted": sum(combined), "impact_stream": impact,
    "delivered_stream": delivered,
    "impact_payout_steps": [i for i, value in enumerate(impact) if value > 0.0],
    "delivered_payout_steps": [
      i for i, value in enumerate(delivered) if value > 0.0
    ],
    "finalization_step": finalized, "productive": productive,
    "finalization_reason": "success" if finalized is not None else "none",
  }


def _record(
  episode_id: str,
  checkpoint: dict[str, object],
  *,
  reset_seed: int,
  action_seed: int,
) -> dict[str, object]:
  c = _arm(impact=[0.0, 0.16], delivered=[0.0, 0.04],
           finalized=None, productive=False)
  dprime = _arm(impact=[0.0, 0.16], delivered=[0.0, 0.04],
                finalized=1, productive=True)
  return {
    "episode_id": episode_id, "checkpoint_path": checkpoint["path"],
    "stratum": checkpoint["stratum"], "split": checkpoint["split"],
    "training_seed": checkpoint["training_seed"],
    "reset_seed": reset_seed, "action_seed": action_seed,
    "physical_trace_digests": {arm: "" for arm in probe.TASKS},
    "returns": {"C": c, **{
      arm: copy.deepcopy(dprime) for arm in ("D-prime", "F", "E")
    }},
    "tracker_streams": {
      arm: {
        "started": [False, True], "finalized": [False, True],
        "productive": [False, True], "reason": [0, 1],
      }
      for arm in ("D-prime", "F", "E")
    },
    "dprime_audit": {
      **{key: [0.0, 1.0] for key in (
        "wrapper_parent_impact_raw", "c_legacy_impact_raw",
        "wrapper_parent_delivered_raw", "c_legacy_delivered_raw",
      )},
      **{key: 1.0 for key in (
        "impact_latch_oracle", "impact_manager_payout_raw",
        "delivered_sum_oracle", "delivered_manager_payout_raw",
      )},
    },
  }


def _raw_fixture(tmp_path: Path) -> dict[str, object]:
  checkpoint_paths = [tmp_path / "alpha.pt", tmp_path / "beta.pt"]
  checkpoint_paths[0].write_bytes(b"checkpoint-alpha")
  checkpoint_paths[1].write_bytes(b"checkpoint-beta")
  checkpoints = [
    {
      "stratum": stratum, "training_seed": seed, "split": split,
      "path": path.name, "sha256": probe.sha256_path(path),
    }
    for (stratum, seed, split), path in zip(
      (("alpha", 0, "calibration"), ("beta", 2, "validation")),
      checkpoint_paths, strict=True
    )
  ]
  task_contract = {
    arm: {
      "task_id": f"task-{arm}", "impact_class": "LiteralImpact",
      "delivered_class": "LiteralDelivered",
      "tracker_class": None if arm == "C" else "LiteralTracker",
      "tracker_per_substep": None if arm == "C" else True,
      "impact_weight": 8.0, "delivered_weight": 2.0, "v_expected": 1.0,
      "i_ref": 0.6094 if arm in ("C", "D-prime") else 0.3088,
      "saturate": True if arm == "E" else (False if arm == "F" else None),
      "imp_max_p": 0.0, "physics_dt_s": 0.002, "step_dt_s": 0.02,
      "decimation": 10,
      "non_treatment_digest": "literal-config",
    }
    for arm in ("C", "D-prime", "F", "E")
  }
  manifest = {
    "schema_version": 3, "frozen_before_payout": True,
    "selection_policy": "complete_frozen_bank_no_adaptation",
    "checkpoints": checkpoints, "reset_seeds": [11], "action_seeds": [21],
    "action_mode": "stochastic_once_replayed",
    "gamma": 0.99, "physics_dt_s": 0.002, "step_dt_s": 0.02,
    "decimation": 10,
    "normalizers_n_s": {"legacy": 0.6094, "first_strike_success": 0.3088},
    "expected_episode_count": 2,
    "tasks": {arm: f"task-{arm}" for arm in ("C", "D-prime", "F", "E")},
    "reward_weights": {
      arm: {"impact_progress": 8.0, "delivered_impulse": 2.0}
      for arm in ("C", "D-prime", "F", "E")
    },
    "reference_params": copy.deepcopy(probe.REFERENCE_PARAMS),
    "repo": {"head": "literal-repo", "dirty": False},
    "assets": {"head": "literal-assets", "dirty": False},
    "task_config_digest": "literal-task-config", "task_contract": task_contract,
    "relevant_source_sha256": {},
    "relevant_provenance": {
      "repo_paths": [],
      "asset_paths": [],
      "file_sha256": {},
      "tree_sha256": probe.canonical_digest({}),
      "versions": {key: "literal" for key in (
        "python", "torch", "numpy", "mjlab", "mujoco", "mujoco_warp", "rsl_rl"
      )},
    },
    "stochastic_policy_proof": [
      {
        "checkpoint_path": row["path"], "same_seed": 101, "different_seed": 202,
        "sample_a": [[0.1, 0.2, 0.3]], "sample_b": [[0.1, 0.2, 0.3]],
        "sample_different": [[0.2, 0.3, 0.4]],
        "deterministic_mean": [[0.0, 0.0, 0.0]],
      }
      for row in checkpoints
    ],
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
    "schema_version": 4,
    "manifest": manifest,
    "physical_traces": traces,
    "reward_records": records,
    "phase_rows": [
      {
        "phase": i, "matched_event_digest": "literal-event",
        "contact_detected": True, "productive": True,
        "v_precontact_m_s": 1.5, "v_true_m_s": 1.5,
        "impact_payout_count": 1, "delivered_payout_count": 1,
        "delivered_n_s": 0.3088, "depth_at_contact_m": 0.001,
        "peak_depth_m": 0.030, "progress_m": 0.029,
        "finalization_reason": "success",
      }
      for i in range(10)
    ],
    "normalizer_samples": {
      "legacy_terminal_n_s": 0.6094,
      "legacy_reference_params": copy.deepcopy(probe.REFERENCE_PARAMS),
      "event_values_n_s": [0.3088] * 8,
      "event_reference_params": copy.deepcopy(probe.REFERENCE_PARAMS),
    },
    "runtime_s": 1.25,
    "development_replay": {
      arm: {"late_payout": 0.0} for arm in ("D-prime", "F", "E")
    },
    "crash": None,
  }


def _summary(raw: dict[str, object], **kwargs):
  options = {
    "manifest_sha256": "1" * 64,
    "raw_sha256": "2" * 64,
    "expected_checkpoints": raw["manifest"]["checkpoints"],
    "expected_reset_seeds": (11,),
    "expected_action_seeds": (21,),
  }
  options.update(kwargs)
  return probe.build_summary(raw, **options)


def _validate(raw: dict[str, object], root: Path):
  return probe.validate_raw_payload(
    raw,
    root=root,
    expected_checkpoints=raw["manifest"]["checkpoints"],
    expected_reset_seeds=(11,),
    expected_action_seeds=(21,),
    verify_source_hashes=False,
  )


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
  summary = _summary(raw)
  summary["population"]["completed_episodes"] = 1
  raw["reward_records"][1]["checkpoint_path"] = "alpha.pt"
  for trace in raw["physical_traces"]:
    trace["trace_digest"] = "3" * 64
    trace["action_tape"].append([0.0, 0.0, 0.0])
  for record in raw["reward_records"]:
    record["returns"]["D-prime"]["finalization_step"] = 0

  reasons = probe.collect_failure_reasons(
    summary,
    raw,
    root=tmp_path,
    expected_checkpoints=raw["manifest"]["checkpoints"],
    expected_reset_seeds=(11,),
    expected_action_seeds=(21,),
    verify_artifact_hashes=False,
    verify_source_hashes=False,
  )

  assert [reason.split(":", 1)[0] for reason in reasons] == [
    "population",
    "checkpoint_binding",
    "digests",
    "replay_semantics",
    "stream_lengths",
  ]
  assert "beta" in reasons[1]
  for reason in reasons[2:]:
    assert "alpha_s0_r11" in reason
    assert "beta_s2_r11" in reason


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

  summary = _summary(
    raw,
    manifest_sha256=probe.sha256_path(manifest_path),
    raw_sha256=probe.sha256_path(raw_path),
    raw_size_bytes=raw_path.stat().st_size,
  )
  encoded = json.dumps(summary, sort_keys=True)

  assert summary["artifact_sha256"]["raw"] == probe.sha256_path(raw_path)
  assert summary["artifact_sha256"]["manifest"] == probe.sha256_path(manifest_path)
  assert summary["raw_artifact"] == {
    "schema_version": 4,
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
    _validate(raw, tmp_path)


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

  record = _raw_fixture(tmp_path)["reward_records"][0]
  record["tracker_streams"]["F"]["productive"][0] = True
  record["tracker_streams"]["F"]["reason"][0] = 2
  assert {record["returns"][arm]["finalization_step"] for arm in (
    "D-prime", "F", "E"
  )} == {1}
  with pytest.raises(ValueError, match="tracker streams.*F.*productive|reason"):
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


def test_policy_sampling_and_live_proof_are_bound(tmp_path: Path):
  calls = []

  def policy(obs, *, stochastic_output):
    calls.append((obs, stochastic_output))
    return torch.tensor([[2.0, -2.0, 0.25]])

  action = probe.sample_stochastic_action(policy, "literal-observation", clip=1.0)

  assert calls == [("literal-observation", True)]
  assert torch.equal(action, torch.tensor([[1.0, -1.0, 0.25]]))

  class LiteralPolicy:
    def __call__(self, observation, *, stochastic_output):
      mean = observation + 0.25
      return mean + torch.randn_like(mean) if stochastic_output else mean

  proof = probe.prove_stochastic_policy(
    LiteralPolicy(),
    torch.zeros(1, 3),
    checkpoint_path="literal.pt",
    same_seed=101, different_seed=202, clip=10.0,
  )

  assert proof["sample_a"] == proof["sample_b"]
  assert proof["sample_a"] != proof["sample_different"]
  assert proof["sample_a"] != proof["deterministic_mean"]

  raw = _raw_fixture(tmp_path)
  summary = _summary(raw)
  assert summary["stochastic_policy"] == {
    "checkpoint_count": 2,
    "passed": True,
  }
  assert next(
    gate for gate in summary["gates"] if gate["id"] == "stochastic_policy"
  )["passed"]


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
    raw, manifest_path=manifest_path, raw_path=raw_path,
    summary_path=summary_path, root=tmp_path,
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


def test_recursive_schema_rejects_nested_payloads_and_bad_scalars(tmp_path: Path):
  raw = _raw_fixture(tmp_path)
  raw["reward_records"][0]["physical_traces"] = {
    arm: copy.deepcopy(raw["physical_traces"][0]["physical"])
    for arm in ("C", "D-prime", "F", "E")
  }
  with pytest.raises(ValueError, match="schema mismatch.*reward_records"):
    _validate(raw, tmp_path)

  raw = _raw_fixture(tmp_path)
  raw["normalizer_samples"]["renamed_full_trace"] = copy.deepcopy(
    raw["physical_traces"][0]["physical"]
  )
  with pytest.raises(ValueError, match="normalizer_samples"):
    _validate(raw, tmp_path)

  clean = _raw_fixture(tmp_path)
  summary = _summary(clean)
  summary["aggregates"]["renamed_substep_payload"] = {
    "trajectory_samples": copy.deepcopy(clean["physical_traces"][0]["physical"])
  }
  with pytest.raises(ValueError, match="aggregates"):
    probe._validate_summary_schema(summary)

  mutations = (
    ("legacy_terminal_n_s", lambda item: item["normalizer_samples"].update(
      legacy_terminal_n_s="0.6094"
    )),
    ("runtime_s", lambda item: item.update(runtime_s=None)),
    ("contact_detected", lambda item: item["phase_rows"][0].update(
      contact_detected=1
    )),
  )
  for target, mutate in mutations:
    raw = _raw_fixture(tmp_path)
    mutate(raw)
    with pytest.raises(ValueError, match=target):
      _validate(raw, tmp_path)
  summary = _summary(_raw_fixture(tmp_path))
  summary["aggregates"]["by_arm"]["C"]["discounted_mean"] = float("nan")
  with pytest.raises(ValueError, match="discounted_mean"):
    probe._validate_summary_schema(summary)


def test_manifest_provenance_tree_versions_and_passed_roots(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  repo_file = tmp_path / "src/tasks/hammer/mdp/rewards.py"
  asset_file = tmp_path / "hammer_z1_env/assets/hammer.xml"
  repo_file.parent.mkdir(parents=True)
  asset_file.parent.mkdir(parents=True)
  repo_file.write_text("reward = 1\n")
  asset_file.write_text("<mujoco/>\n")

  provenance = probe.build_relevant_provenance(
    tmp_path, tmp_path,
    repo_paths=("src/tasks/hammer/mdp/rewards.py",),
    asset_paths=("hammer_z1_env/assets/hammer.xml",),
    require_git_clean=False,
  )

  assert provenance["tree_sha256"] == probe.canonical_digest(
    provenance["file_sha256"])
  assert set(provenance["versions"]) == {
    "python", "torch", "numpy", "mjlab", "mujoco", "mujoco_warp", "rsl_rl"
  }
  asset_file.write_text("<mujoco model='changed'/>\n")
  with pytest.raises(ValueError, match="relevant source hash.*hammer.xml"):
    probe.validate_relevant_provenance(
      provenance, tmp_path, tmp_path, require_git_clean=False,
    )

  assert "assets_root" in inspect.signature(probe.collect_gate_results).parameters
  raw = _raw_fixture(tmp_path)
  raw["manifest"]["relevant_provenance"] = {"literal": "provenance"}
  summary = _summary(raw)
  asset_root = tmp_path / "assets-root"
  calls = []

  monkeypatch.setattr(
    probe, "validate_relevant_provenance",
    lambda provenance, repo, assets, **_: calls.append((provenance, repo, assets)),
  )
  probe.collect_gate_results(
    summary, raw, root=tmp_path, assets_root=asset_root,
    expected_checkpoints=raw["manifest"]["checkpoints"],
    expected_reset_seeds=(11,),
    expected_action_seeds=(21,),
    verify_artifact_hashes=False,
    verify_source_hashes=True,
  )

  assert calls == [(raw["manifest"]["relevant_provenance"], tmp_path, asset_root)]


def test_manager_scaling_and_population_uniqueness(tmp_path: Path):
  iterable_terms = [("impact_progress", [8.0]), ("delivered_impulse", [2.0])]
  assert probe.scaled_treatment_terms(iterable_terms, step_dt=0.02) == (
    pytest.approx(0.16), pytest.approx(0.04),
  )

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
