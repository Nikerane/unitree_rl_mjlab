#!/usr/bin/env python
"""Frozen CPU qualification for the C / D-prime / F / E reward packages."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
ASSETS_ROOT = ROOT.parent / "safe_impact_manipulation"

RESET_SEEDS = tuple(range(2026072700, 2026072732))
ACTION_SEEDS = tuple(range(2026072800, 2026072832))
STRATA = ("mx_maxoff", "dc_imponly", "dc_delonly", "mx_maxmax")

_CHECKPOINT_DATA = (
  ("mx_maxoff", 0, "logs/rsl_rl/z1_hammer/2026-07-20_22-07-49_mx_maxoff_seed0/model_499.pt", "e9abae170d6204a1330003ac8bf3140f6b0d8f6a275546ee14affd0a69dead11"),
  ("mx_maxoff", 1, "logs/rsl_rl/z1_hammer/2026-07-20_22-07-49_mx_maxoff_seed1/model_499.pt", "80f173787031d1d49b32844053ad93bad6aed2c650ed1eaae5f36f84cdcd7dd7"),
  ("mx_maxoff", 2, "logs/rsl_rl/z1_hammer/2026-07-20_22-07-49_mx_maxoff_seed2/model_499.pt", "19de601caec04d1e1a0900201710428f951c2a10ecaa0629228b54e0f089226a"),
  ("dc_imponly", 0, "logs/rsl_rl/z1_hammer/2026-07-21_00-40-17_dc_imponly_seed0/model_499.pt", "c5b909bf1b12d637b78964ee501e5e20239c1e900bfb00c9df6cb10387909bc7"),
  ("dc_imponly", 1, "logs/rsl_rl/z1_hammer/2026-07-21_00-40-17_dc_imponly_seed1/model_499.pt", "a4b2b42a46711581785e91b111bc1222a22b7b383077784f1f1c55634913f69d"),
  ("dc_imponly", 2, "logs/rsl_rl/z1_hammer/2026-07-21_00-40-17_dc_imponly_seed2/model_499.pt", "2c71a0535d4a7f4ab74449626d2b230ba66971f334a1e4241bbf76f815c72809"),
  ("dc_delonly", 0, "logs/rsl_rl/z1_hammer/2026-07-21_00-42-50_dc_delonly_seed0/model_499.pt", "6abb3fd34e3230e0c56c689992b70f535f573466f056d6b1a01652788fdec933"),
  ("dc_delonly", 1, "logs/rsl_rl/z1_hammer/2026-07-21_00-42-50_dc_delonly_seed1/model_499.pt", "33dd521894c3db4e71af0ee400400114d1833ba7877287de1baf0911905cdf13"),
  ("dc_delonly", 2, "logs/rsl_rl/z1_hammer/2026-07-21_00-41-51_dc_delonly_seed2/model_499.pt", "3584ca961b8a0ded094241dfecfd157d942f394542fdcc17b86002e6f5c30861"),
  ("mx_maxmax", 0, "logs/rsl_rl/z1_hammer/2026-07-20_22-07-49_mx_maxmax_seed0/model_499.pt", "9dc802c2c0fa2770be5733f6baa728b5fbe75834ab8ca899c20b88123facd7d9"),
  ("mx_maxmax", 1, "logs/rsl_rl/z1_hammer/2026-07-20_22-07-49_mx_maxmax_seed1/model_499.pt", "6af2075dbc99e769e4694227019b633721d51d1d3fd71598c6e29db2ee0b3676"),
  ("mx_maxmax", 2, "logs/rsl_rl/z1_hammer/2026-07-20_22-07-49_mx_maxmax_seed2/model_499.pt", "92d771d498e3f929b3e9974338b30c555d73295fb0797b291ddd594104c95ca1"),
)
CHECKPOINTS = tuple(
  {
    "stratum": stratum,
    "training_seed": seed,
    "split": "calibration" if seed in (0, 1) else "validation",
    "path": path,
    "sha256": digest,
  }
  for stratum, seed, path, digest in _CHECKPOINT_DATA
)

REFERENCE_PARAMS = {
  "approach_height": 0.15,
  "min_windup_clearance": 0.05,
  "overshoot": 0.15,
  "windup_speed": 0.02,
  "descent_speed": 0.05,
  "axis_tol": 0.05,
  "descent_margin": 0.01,
}
TASKS = {
  "C": "Unitree-Z1-Hammer-CaT-Impulse",
  "D-prime": "Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy",
  "F": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
  "E": "Unitree-Z1-Hammer-CaT-Impulse-Event",
}
MANDATED_SOURCES = (
  "src/tasks/hammer/mdp/first_strike.py",
  "src/tasks/hammer/mdp/rewards.py",
  "src/tasks/hammer/config/z1/env_cfgs.py",
  "src/tasks/hammer/config/z1/__init__.py",
  "docs/results/assets/2026-07-25_first_strike_reward/probe_first_strike.py",
)
STOCHASTIC_PROOF_SEEDS = (2026072898, 2026072899)


_seq = lambda item: ("seq", item)
_map = lambda item, keys=None: (
  "map", item, None if keys is None else frozenset(keys)
)
_one_of = lambda *types: ("one_of", *types)
VEC = _seq(float)
MATRIX = _seq(VEC)
HASH_MAP = _map(str)
ARM_HASHES = _map(str, TASKS)
TRACKER_SCHEMA = {
  **{key: _seq(bool) for key in ("started", "finalized", "productive")},
  "reason": _seq(int),
}
RETURN_SCHEMA = {
  "discounted": float, "undiscounted": float,
  "impact_stream": VEC, "delivered_stream": VEC,
  "impact_payout_steps": _seq(int), "delivered_payout_steps": _seq(int),
  "finalization_step": _one_of(int, type(None)),
  "finalization_reason": str, "productive": bool,
}
DPRIME_AUDIT_SCHEMA = {
  **{key: VEC for key in (
    "wrapper_parent_impact_raw", "c_legacy_impact_raw",
    "wrapper_parent_delivered_raw", "c_legacy_delivered_raw",
  )},
  **{key: float for key in (
    "impact_latch_oracle", "impact_manager_payout_raw",
    "delivered_sum_oracle", "delivered_manager_payout_raw",
  )},
}
PHYSICAL_SCHEMA = {
  "contact": _seq(bool),
  "head_position_m": MATRIX, "joint_speed_rad_s": MATRIX,
  "clamped_depth_m": VEC, "net_axial_force_n": VEC,
}
TRACE_SCHEMA = {
  "episode_id": str, "action_tape": MATRIX, "action_tape_digest": str,
  "physical": PHYSICAL_SCHEMA, "trace_digest": str,
}
REWARD_RECORD_SCHEMA = {
  "episode_id": str, "checkpoint_path": str, "stratum": str,
  "training_seed": int, "split": str, "reset_seed": int, "action_seed": int,
  "physical_trace_digests": ARM_HASHES,
  "returns": _map(RETURN_SCHEMA, TASKS),
  "tracker_streams": _map(TRACKER_SCHEMA, ("D-prime", "F", "E")),
  "dprime_audit": DPRIME_AUDIT_SCHEMA,
}
REFERENCE_SCHEMA = {key: float for key in REFERENCE_PARAMS}
CHECKPOINT_SCHEMA = {
  "stratum": str, "training_seed": int, "split": str,
  "path": str, "sha256": str,
}
PROOF_SCHEMA = {
  "checkpoint_path": str, "same_seed": int, "different_seed": int,
  "sample_a": MATRIX, "sample_b": MATRIX,
  "sample_different": MATRIX, "deterministic_mean": MATRIX,
}
GIT_STATE_SCHEMA = {"head": str, "dirty": bool}
VERSIONS_SCHEMA = {key: str for key in (
  "python", "torch", "numpy", "mjlab", "mujoco", "mujoco_warp", "rsl_rl"
)}
TASK_CONTRACT_SCHEMA = {
  "task_id": str, "impact_class": str, "delivered_class": str,
  "tracker_class": _one_of(str, type(None)),
  "tracker_per_substep": _one_of(bool, type(None)),
  "impact_weight": float, "delivered_weight": float,
  "v_expected": float, "i_ref": float,
  "saturate": _one_of(bool, type(None)),
  "imp_max_p": float, "physics_dt_s": float, "step_dt_s": float,
  "decimation": int, "non_treatment_digest": str,
}
MANIFEST_SCHEMA = {
  "schema_version": int, "frozen_before_payout": bool, "selection_policy": str,
  "checkpoints": _seq(CHECKPOINT_SCHEMA),
  "reset_seeds": _seq(int), "action_seeds": _seq(int), "action_mode": str,
  "gamma": float, "physics_dt_s": float, "step_dt_s": float, "decimation": int,
  "normalizers_n_s": {"legacy": float, "first_strike_success": float},
  "tasks": _map(str, TASKS), "reference_params": REFERENCE_SCHEMA,
  "expected_episode_count": int, "repo": GIT_STATE_SCHEMA,
  "assets": GIT_STATE_SCHEMA, "task_config_digest": str,
  "reward_weights": _map(
    {"impact_progress": float, "delivered_impulse": float}, TASKS
  ),
  "task_contract": _map(TASK_CONTRACT_SCHEMA, TASKS),
  "relevant_source_sha256": HASH_MAP,
  "relevant_provenance": {
    "repo_paths": _seq(str), "asset_paths": _seq(str),
    "file_sha256": HASH_MAP, "tree_sha256": str, "versions": VERSIONS_SCHEMA,
  },
  "stochastic_policy_proof": _seq(PROOF_SCHEMA),
}
PHASE_SCHEMA = {
  "phase": int, "matched_event_digest": str,
  "contact_detected": bool, "productive": bool,
  "v_precontact_m_s": float, "v_true_m_s": float,
  "impact_payout_count": int, "delivered_payout_count": int,
  "delivered_n_s": float, "depth_at_contact_m": float,
  "peak_depth_m": float, "progress_m": float, "finalization_reason": str,
}
NORMALIZER_SCHEMA = {
  "legacy_terminal_n_s": float,
  "legacy_reference_params": REFERENCE_SCHEMA,
  "event_values_n_s": VEC, "event_reference_params": REFERENCE_SCHEMA,
}
DEVELOPMENT_SCHEMA = _map({"late_payout": float}, ("D-prime", "F", "E"))
RAW_SCHEMA = {
  "schema_version": int, "manifest": MANIFEST_SCHEMA,
  "physical_traces": _seq(TRACE_SCHEMA),
  "reward_records": _seq(REWARD_RECORD_SCHEMA),
  "phase_rows": _seq(PHASE_SCHEMA), "normalizer_samples": NORMALIZER_SCHEMA,
  "runtime_s": float, "development_replay": DEVELOPMENT_SCHEMA,
  "crash": _one_of(str, type(None)),
}
GATE_SCHEMA = {"id": str, "target": str, "passed": bool, "reason": str}
ARM_AGGREGATE_SCHEMA = {
  "episode_count": int, "discounted_mean": float, "undiscounted_mean": float,
  "impact_payout_episode_count": int,
  "delivered_payout_episode_count": int,
}
STRATUM_ARM_SCHEMA = {"episode_count": int, "undiscounted_mean": float}
SUMMARY_SCHEMA = {
  "schema_version": int, "valid": bool,
  "failure_reasons": _seq(str), "gates": _seq(GATE_SCHEMA),
  "artifact_sha256": {"manifest": str, "raw": str},
  "raw_artifact": {
    "schema_version": int, "size_bytes": int,
    "physical_trace_count": int, "reward_record_count": int,
  },
  "population": {
    "expected_episodes": int, "completed_episodes": int,
    "checkpoint_groups": int, "paired_seed_count": int,
  },
  "normalizers": {
    "legacy": {"configured_n_s": float, "observed_n_s": float},
    "first_strike_success": {
      "configured_n_s": float, "sample_count": int,
      "mean_n_s": float, "sd_n_s": float, "range_n_s": float,
    },
  },
  "phase_invariance": {"count": int, "passed": bool},
  "stochastic_policy": {"checkpoint_count": int, "passed": bool},
  "development_replay": DEVELOPMENT_SCHEMA,
  "aggregates": {
    "by_arm": _map(ARM_AGGREGATE_SCHEMA, TASKS),
    "by_stratum": _map(_map(STRATUM_ARM_SCHEMA, TASKS)),
  },
  "provenance": {
    "repo": GIT_STATE_SCHEMA, "assets": GIT_STATE_SCHEMA,
    "task_config_digest": str, "relevant_tree_sha256": str,
    "versions": VERSIONS_SCHEMA,
  },
}

RAW_SCHEMA_KEYS = frozenset(RAW_SCHEMA)
PHYSICAL_KEYS = frozenset(PHYSICAL_SCHEMA)
NPZ_KEYS = frozenset({"payload_json"})


def canonical_digest(value: Any) -> str:
  payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
  return hashlib.sha256(payload.encode()).hexdigest()


def sha256_path(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _validate_schema(value: Any, schema: Any, path: str) -> None:
  if isinstance(schema, dict):
    if type(value) is not dict or set(value) != set(schema):
      raise ValueError(f"schema mismatch at {path}")
    for key, child_schema in schema.items():
      _validate_schema(value[key], child_schema, f"{path}.{key}")
    return
  if isinstance(schema, tuple):
    tag = schema[0]
    if tag == "seq":
      if type(value) is not list:
        raise ValueError(f"{path} must be list")
      for index, child in enumerate(value):
        _validate_schema(child, schema[1], f"{path}[{index}]")
      return
    if tag == "map":
      if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{path} must be string-keyed map")
      expected_keys = schema[2]
      if expected_keys is not None and set(value) != expected_keys:
        raise ValueError(f"schema mismatch at {path}")
      for key, child in value.items():
        _validate_schema(child, schema[1], f"{path}.{key}")
      return
    if tag == "one_of":
      if not any(type(value) is allowed for allowed in schema[1:]):
        names = "/".join(allowed.__name__ for allowed in schema[1:])
        raise ValueError(f"{path} must be {names}")
      if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
      return
    raise RuntimeError(f"unknown schema tag: {tag}")
  if type(value) is not schema:
    raise ValueError(f"{path} must be {schema.__name__}")
  if schema is float and not math.isfinite(value):
    raise ValueError(f"{path} must be finite")


def hash_relevant_sources(root: Path, paths: Sequence[str]) -> dict[str, str]:
  return {path: sha256_path(root / path) for path in sorted(paths)}


def validate_file_hashes(root: Path, hashes: dict[str, str], *, label: str) -> None:
  for relative, expected in sorted(hashes.items()):
    path = root / relative
    if not path.is_file() or sha256_path(path) != expected:
      raise ValueError(f"{label} hash mismatch: {relative}")


def _package_versions() -> dict[str, str]:
  def version(distribution: str) -> str:
    try:
      return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
      return "not-installed"

  return {
    "python": platform.python_version(),
    "torch": version("torch"),
    "numpy": version("numpy"),
    "mjlab": version("mjlab"),
    "mujoco": version("mujoco"),
    "mujoco_warp": version("mujoco-warp"),
    "rsl_rl": version("rsl-rl-lib"),
  }


def _targeted_git_clean(root: Path, paths: Sequence[str]) -> None:
  if not paths:
    return
  status = subprocess.run(
    ["git", "-C", str(root), "status", "--porcelain", "--", *paths],
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  if status:
    raise ValueError(f"relevant source paths are dirty: {status}")
  for relative in paths:
    subprocess.run(
      ["git", "-C", str(root), "ls-files", "--error-unmatch", relative],
      check=True,
      capture_output=True,
      text=True,
    )


def _default_repo_provenance_paths(root: Path) -> tuple[str, ...]:
  paths = {
    path.relative_to(root).as_posix()
    for path in (root / "src/tasks/hammer").rglob("*.py")
    if path.is_file()
  }
  paths.add("src/assets/robots/unitree_z1/z1_constants.py")
  paths.update(MANDATED_SOURCES)
  return tuple(sorted(paths))


def _default_asset_provenance_paths(assets_root: Path) -> tuple[str, ...]:
  base = assets_root / "hammer_z1_env/assets"
  return tuple(
    sorted(
      path.relative_to(assets_root).as_posix()
      for path in base.rglob("*")
      if path.is_file()
    )
  )


def build_relevant_provenance(
  repo_root: Path,
  assets_root: Path,
  *,
  repo_paths: Sequence[str] | None = None,
  asset_paths: Sequence[str] | None = None,
  require_git_clean: bool = True,
) -> dict[str, Any]:
  repo_paths = tuple(repo_paths or _default_repo_provenance_paths(repo_root))
  asset_paths = tuple(asset_paths or _default_asset_provenance_paths(assets_root))
  if require_git_clean:
    _targeted_git_clean(repo_root, repo_paths)
    _targeted_git_clean(assets_root, asset_paths)
  hashes = {
    **{
      f"repo:{path}": sha256_path(repo_root / path)
      for path in sorted(repo_paths)
    },
    **{
      f"assets:{path}": sha256_path(assets_root / path)
      for path in sorted(asset_paths)
    },
  }
  return {
    "repo_paths": list(repo_paths),
    "asset_paths": list(asset_paths),
    "file_sha256": hashes,
    "tree_sha256": canonical_digest(hashes),
    "versions": _package_versions(),
  }


def validate_relevant_provenance(
  provenance: dict[str, Any],
  repo_root: Path,
  assets_root: Path,
  *,
  require_git_clean: bool = True,
) -> None:
  current = build_relevant_provenance(
    repo_root,
    assets_root,
    repo_paths=provenance["repo_paths"],
    asset_paths=provenance["asset_paths"],
    require_git_clean=require_git_clean,
  )
  for key, expected in provenance["file_sha256"].items():
    if current["file_sha256"].get(key) != expected:
      raise ValueError(f"relevant source hash mismatch: {key.split(':', 1)[1]}")
  if current != provenance:
    raise ValueError("relevant source tree or runtime versions changed")


def _git_state(path: Path) -> dict[str, Any]:
  head = subprocess.run(
    ["git", "-C", str(path), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  dirty = bool(
    subprocess.run(
      ["git", "-C", str(path), "status", "--porcelain"],
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
  )
  return {"head": head, "dirty": dirty}


def validate_reward_weights(weights: dict[str, Any]) -> None:
  if set(weights) != set(TASKS):
    raise ValueError("all four task weights are required")
  for arm in TASKS:
    if weights[arm] != {
      "impact_progress": 8.0,
      "delivered_impulse": 2.0,
    }:
      raise ValueError(f"weights 8/2 required for {arm}")


def _task_contract() -> dict[str, Any]:
  from mjlab.tasks.registry import load_env_cfg

  contract: dict[str, Any] = {}
  configs = {arm: load_env_cfg(task, play=False) for arm, task in TASKS.items()}
  for arm, cfg in configs.items():
    delivered = cfg.rewards["delivered_impulse"]
    impact = cfg.rewards["impact_progress"]
    tracker = cfg.metrics.get("first_strike")
    contract[arm] = {
      "task_id": TASKS[arm],
      "impact_class": getattr(impact.func, "__name__", type(impact.func).__name__),
      "delivered_class": getattr(
        delivered.func, "__name__", type(delivered.func).__name__
      ),
      "tracker_class": (
        None
        if tracker is None
        else getattr(tracker.func, "__name__", type(tracker.func).__name__)
      ),
      "tracker_per_substep": None if tracker is None else tracker.per_substep,
      "impact_weight": float(impact.weight),
      "delivered_weight": float(delivered.weight),
      "v_expected": float(impact.params["v_expected"]),
      "i_ref": float(delivered.params["i_ref"]),
      "saturate": delivered.params.get("saturate"),
      "imp_max_p": float(cfg.metrics["cat_soft"].params["imp_max_p"]),
      "physics_dt_s": float(cfg.sim.mujoco.timestep),
      "step_dt_s": float(cfg.sim.mujoco.timestep * cfg.decimation),
      "decimation": int(cfg.decimation),
      "non_treatment_digest": canonical_digest(
        {
          "actions": str(cfg.actions),
          "terminations": str(cfg.terminations),
          "other_rewards": {
            name: str(term)
            for name, term in cfg.rewards.items()
            if name not in {"impact_progress", "delivered_impulse"}
          },
          "cat_soft": str(cfg.metrics["cat_soft"].params),
          "actuators": str(cfg.scene.entities["robot"].articulation.actuators),
        }
      ),
    }
  return contract


def validate_task_contract(contract: dict[str, Any]) -> None:
  expected_classes = {
    "C": ("ImpactProgressTerm", "DeliveredImpulseTerm", None),
    "D-prime": (
      "FirstStrikeLegacyImpactRewardTerm",
      "FirstStrikeLegacyDeliveredRewardTerm",
      "FirstStrikeEventTracker",
    ),
    "F": (
      "FirstStrikeImpactRewardTerm",
      "FirstStrikeDeliveredRewardTerm",
      "FirstStrikeEventTracker",
    ),
    "E": (
      "FirstStrikeImpactRewardTerm",
      "FirstStrikeDeliveredRewardTerm",
      "FirstStrikeEventTracker",
    ),
  }
  if set(contract) != set(TASKS):
    raise ValueError("live task contract requires C/D-prime/F/E")
  for arm, row in contract.items():
    if row["task_id"] != TASKS[arm]:
      raise ValueError(f"task id mismatch for {arm}")
    if (
      row["impact_class"],
      row["delivered_class"],
      row["tracker_class"],
    ) != expected_classes[arm]:
      raise ValueError(f"task class/tracker mismatch for {arm}")
    expected_tracker_substep = None if arm == "C" else True
    if row["tracker_per_substep"] is not expected_tracker_substep:
      raise ValueError(f"tracker per-substep mismatch for {arm}")
    if row["impact_weight"] != 8.0 or row["delivered_weight"] != 2.0:
      raise ValueError(f"weights 8/2 required for {arm}")
    if row["v_expected"] != 1.0 or row["imp_max_p"] != 0.0:
      raise ValueError(f"v_expected/imp_max_p mismatch for {arm}")
    if (
      row["physics_dt_s"] != 0.002
      or row["step_dt_s"] != 0.02
      or row["decimation"] != 10
    ):
      raise ValueError(f"dt/decimation mismatch for {arm}")
  if contract["C"]["i_ref"] != 0.6094 or contract["D-prime"]["i_ref"] != 0.6094:
    raise ValueError("legacy normalizer mismatch")
  if contract["E"]["i_ref"] != 0.3088 or contract["F"]["i_ref"] != 0.3088:
    raise ValueError("event normalizer mismatch")
  if contract["E"]["saturate"] is not True or contract["F"]["saturate"] is not False:
    raise ValueError("F/E saturation treatment mismatch")
  if any(
    contract[arm]["non_treatment_digest"] != contract["C"]["non_treatment_digest"]
    for arm in ("D-prime", "F", "E")
  ):
    raise ValueError("non-treatment task configuration differs")


def frozen_manifest() -> dict[str, Any]:
  provenance = build_relevant_provenance(
    ROOT, ASSETS_ROOT, require_git_clean=True
  )
  checkpoint_hashes = {
    row["path"]: sha256_path(ROOT / row["path"]) for row in CHECKPOINTS
  }
  for row in CHECKPOINTS:
    if checkpoint_hashes[row["path"]] != row["sha256"]:
      raise ValueError(f"checkpoint hash mismatch: {row['path']}")
  contract = _task_contract()
  validate_task_contract(contract)
  weights = {
    arm: {
      "impact_progress": contract[arm]["impact_weight"],
      "delivered_impulse": contract[arm]["delivered_weight"],
    }
    for arm in TASKS
  }
  validate_reward_weights(weights)
  return {
    "schema_version": 3,
    "frozen_before_payout": True,
    "selection_policy": "complete_frozen_bank_no_adaptation",
    "checkpoints": copy.deepcopy(list(CHECKPOINTS)),
    "reset_seeds": list(RESET_SEEDS),
    "action_seeds": list(ACTION_SEEDS),
    "action_mode": "stochastic_C_sampled_once_replayed_C_Dprime_F_E",
    "gamma": 0.99,
    "physics_dt_s": 0.002,
    "step_dt_s": 0.02,
    "decimation": 10,
    "normalizers_n_s": {"legacy": 0.6094, "first_strike_success": 0.3088},
    "tasks": copy.deepcopy(TASKS),
    "reference_params": copy.deepcopy(REFERENCE_PARAMS),
    "expected_episode_count": 384,
    "repo": _git_state(ROOT),
    "assets": _git_state(ASSETS_ROOT),
    "task_config_digest": canonical_digest(contract),
    "reward_weights": weights,
    "task_contract": contract,
    "relevant_source_sha256": hash_relevant_sources(ROOT, MANDATED_SOURCES),
    "relevant_provenance": provenance,
    "stochastic_policy_proof": [],
  }


def scaled_treatment_terms(
  iterable_terms: Iterable[tuple[str, Sequence[float]]], *, step_dt: float
) -> tuple[float, float]:
  values = {name: float(value[0]) for name, value in iterable_terms}
  return (values["impact_progress"] * step_dt,
          values["delivered_impulse"] * step_dt)


def sample_stochastic_action(policy: Callable[..., Any], obs: Any, *, clip: float):
  return policy(obs, stochastic_output=True).clamp(-clip, clip)


def prove_stochastic_policy(policy, observation, *, checkpoint_path: str,
                            same_seed: int, different_seed: int,
                            clip: float) -> dict[str, Any]:
  import torch

  rng_state = torch.random.get_rng_state()
  try:
    torch.manual_seed(same_seed)
    sample_a = sample_stochastic_action(policy, observation, clip=clip)
    torch.manual_seed(same_seed)
    sample_b = sample_stochastic_action(policy, observation, clip=clip)
    torch.manual_seed(different_seed)
    sample_different = sample_stochastic_action(policy, observation, clip=clip)
    deterministic = policy(observation, stochastic_output=False).clamp(-clip, clip)
  finally:
    torch.random.set_rng_state(rng_state)
  return {
    "checkpoint_path": checkpoint_path, "same_seed": same_seed,
    "different_seed": different_seed,
    "sample_a": sample_a.detach().cpu().tolist(),
    "sample_b": sample_b.detach().cpu().tolist(),
    "sample_different": sample_different.detach().cpu().tolist(),
    "deterministic_mean": deterministic.detach().cpu().tolist(),
  }


def validate_stochastic_policy_proof(
  rows: list[dict[str, Any]], checkpoints: Sequence[dict[str, Any]]
) -> None:
  expected_paths = [row["path"] for row in checkpoints]
  if [row["checkpoint_path"] for row in rows] != expected_paths:
    raise ValueError("stochastic proof must cover frozen checkpoints in order")
  failures = []
  for row in rows:
    checks = (
      (row["same_seed"] == row["different_seed"], "proof seeds must differ"),
      (row["sample_a"] != row["sample_b"], "same seed was not reproducible"),
      (row["sample_a"] == row["sample_different"], "different seeds matched"),
      (row["sample_a"] == row["deterministic_mean"], "stochastic sample equals mean"),
    )
    failures.extend(
      f"{row['checkpoint_path']}: {message}" for failed, message in checks if failed
    )
  if failures:
    raise ValueError("; ".join(failures))


def drive_reference_episode(
  env,
  reference,
  *,
  head: Callable[[], Any],
  delivered_accumulator,
  action_scale: float,
  hold_steps: int,
) -> dict[str, Any]:
  import torch

  if bool(env.cfg.auto_reset):
    raise RuntimeError("legacy reference requires auto_reset=False")
  terminations = getattr(env, "terminations", None)
  if terminations is None:
    terminations = env.cfg.terminations
  if "nail_driven" not in terminations:
    raise RuntimeError("legacy reference requires nail_driven termination")
  length = reference.playback_length()
  for step in range(1, length + hold_steps + 1):
    target = reference.playback_target(min(step, length))
    action = ((target - head()) / action_scale).clamp(-1.0, 1.0)
    _, _, terminated, truncated, _ = env.step(action)
    is_terminated = bool(terminated[0])
    is_truncated = bool(truncated[0])
    if is_terminated or is_truncated:
      nail_driven = bool(env.termination_manager.get_term("nail_driven")[0])
      result = {
        "nail_driven": nail_driven,
        "terminated": is_terminated,
        "truncated": is_truncated,
        "steps": step,
        "delivered_n_s": float(delivered_accumulator.delivered[0]),
      }
      if not (is_terminated and not is_truncated and nail_driven):
        raise RuntimeError(f"reference did not terminate specifically on nail_driven: {result}")
      return result
  raise RuntimeError("reference exhausted playback plus six hold steps without nail_driven")


def _build_env(task: str, *, play: bool, disable_success: bool = False):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task, play=play)
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  if disable_success:
    cfg.terminations.pop("nail_driven", None)
  return ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)


def _reference_once(*, event: bool, height: float = 0.15, speed: float = 0.05):
  import torch
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from src.assets.robots.unitree_z1.z1_constants import (
    HAMMER_HEAD_SITE_NAME,
    Z1_HAMMER_DELTA_POS_SCALE,
  )
  from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR
  from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR
  from src.tasks.hammer.mdp.references import SingleStrikeReference

  env = _build_env(TASKS["E"] if event else TASKS["C"], play=True)
  try:
    head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    head_cfg.resolve(env.scene)
    nail_cfg.resolve(env.scene)
    robot, nail = env.scene["robot"], env.scene["nail_block"]

    def head():
      return robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)

    def nail_top():
      return nail.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)

    env.reset(seed=12345)
    params = dict(REFERENCE_PARAMS)
    params["approach_height"] = height
    params["descent_speed"] = speed
    ref = SingleStrikeReference(1, env.device, **params)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long))
    terminal = drive_reference_episode(
      env,
      ref,
      head=head,
      delivered_accumulator=getattr(env, _ENV_SUBSTEP_DELIVERED_ATTR),
      action_scale=Z1_HAMMER_DELTA_POS_SCALE,
      hold_steps=6,
    )
    result = {**terminal, "reference_params": params}
    if event:
      tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR)
      result.update(
        {
          "event_finalized": bool(tracker.finalized[0]),
          "event_productive": bool(tracker.productive[0]),
          "event_reason": int(tracker.reason[0]),
          "event_delivered_n_s": float(tracker.delivered[0]),
        }
      )
    return result
  finally:
    env.close()


def _phase_probe() -> list[dict[str, Any]]:
  import torch
  from src.tasks.hammer.mdp.first_strike import (
    REASON_SUCCESS,
    FirstStrikeEventTracker,
  )
  from src.tasks.hammer.mdp.rewards import (
    FirstStrikeDeliveredRewardTerm,
    FirstStrikeImpactRewardTerm,
  )

  physical_event = [
    {"head_z": 0.097, "depth": 0.001, "contact": True, "force": 100.0},
    {"head_z": 0.096, "depth": 0.030, "contact": True, "force": 50.0},
  ]
  event_digest = canonical_digest(physical_event)
  rows = []
  for phase in range(10):
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(1, 1, 3)))
    nail = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 1)))
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 1)))
    impulse = SimpleNamespace(data=SimpleNamespace(force=torch.zeros(1, 1, 3)))
    env = SimpleNamespace(
      num_envs=1,
      device="cpu",
      physics_dt=0.002,
      scene={
        "robot": robot,
        "nail_block": nail,
        "hammer_nail_contact": contact,
        "hammer_nail_impulse": impulse,
      },
    )
    cfg = SimpleNamespace(
      params={
        "contact_sensor_name": "hammer_nail_contact",
        "impulse_sensor_name": "hammer_nail_impulse",
        "robot_cfg": SimpleNamespace(name="robot", site_ids=[0]),
        "nail_cfg": SimpleNamespace(name="nail_block", joint_ids=[0]),
        "axis": (0.0, 0.0, -1.0),
        "window_substeps": 25,
        "progress_eps": 5e-4,
      }
    )
    tracker = FirstStrikeEventTracker(cfg=cfg, env=env)
    impact = FirstStrikeImpactRewardTerm(cfg=None, env=env)
    delivered = FirstStrikeDeliveredRewardTerm(cfg=None, env=env)
    impact_count = delivered_count = 0

    def substep(head_z: float, depth: float, on: bool, force: float = 0.0):
      robot.data.site_pos_w[0, 0, 2] = head_z
      nail.data.joint_pos[0, 0] = depth
      contact.data.found[0, 0] = float(on)
      impulse.data.force.zero_()
      impulse.data.force[0, 0, 2] = -force
      tracker(env)

    substep(0.103, 0.0, False)
    substep(0.100, 0.0, False)
    for global_sample in range(30):
      if global_sample == phase:
        sample = physical_event[0]
      elif global_sample == phase + 1:
        sample = physical_event[1]
      elif global_sample == phase + 12:
        sample = {
          "head_z": 0.090,
          "depth": 0.032,
          "contact": True,
          "force": 500.0,
        }
      else:
        sample = {
          "head_z": 0.100,
          "depth": 0.0,
          "contact": False,
          "force": 0.0,
        }
      substep(sample["head_z"], sample["depth"], sample["contact"], sample["force"])
      if global_sample % 10 == 9:
        impact_count += int(float(impact(env, v_expected=1.0)[0]) > 0.0)
        delivered_count += int(float(delivered(env, i_ref=0.3088)[0]) > 0.0)
    rows.append(
      {
        "phase": phase,
        "matched_event_digest": event_digest,
        "contact_detected": bool(tracker.finalized[0]),
        "productive": bool(tracker.productive[0]),
        "v_precontact_m_s": float(tracker.v_precontact[0]),
        "v_true_m_s": (0.100 - 0.097) / 0.002,
        "impact_payout_count": impact_count,
        "delivered_payout_count": delivered_count,
        "delivered_n_s": float(tracker.delivered[0]),
        "depth_at_contact_m": float(tracker.depth_at_contact[0]),
        "peak_depth_m": float(tracker.peak_depth[0]),
        "progress_m": float(tracker.peak_depth[0] - tracker.depth_at_contact[0]),
        "finalization_reason": (
          "success" if int(tracker.reason[0]) == REASON_SUCCESS else "window"
        ),
      }
    )
  return rows


def _capture_setup(env):
  import torch
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME

  robot, nail = env.scene["robot"], env.scene["nail_block"]
  contact, netf = env.scene["hammer_nail_contact"], env.scene["hammer_nail_impulse"]
  head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  arm_cfg = SceneEntityCfg(
    "robot", joint_names=tuple(f"joint{i}" for i in range(1, 7))
  )
  head_cfg.resolve(env.scene)
  arm_cfg.resolve(env.scene)
  trace: dict[str, Any] = {}
  axis = torch.tensor([0.0, 0.0, -1.0])
  original = env.metrics_manager.compute_substep

  def start() -> None:
    trace.clear()
    trace.update(
      {
        "contact": [],
        "head_position_m": [],
        "clamped_depth_m": [],
        "net_axial_force_n": [],
        "joint_speed_rad_s": [],
      }
    )

  def capture() -> None:
    original()
    trace["contact"].append(bool((contact.data.found[0] > 0).any()))
    trace["head_position_m"].append(
      robot.data.site_pos_w[0, head_cfg.site_ids]
      .squeeze(0)
      .detach()
      .cpu()
      .tolist()
    )
    trace["clamped_depth_m"].append(
      float(nail.data.joint_pos[0, 0].clamp(0.0, 0.032))
    )
    trace["net_axial_force_n"].append(
      float((netf.data.force * axis).sum(-1).sum(-1).clamp_min(0.0)[0])
    )
    trace["joint_speed_rad_s"].append(
      robot.data.joint_vel[0, arm_cfg.joint_ids].detach().cpu().tolist()
    )

  env.metrics_manager.compute_substep = capture
  return trace, start


@contextmanager
def _parent_reader_audit(envs: dict[str, Any]) -> Iterator[dict[str, dict[str, list[float]]]]:
  from src.tasks.hammer.mdp.rewards import DeliveredImpulseTerm, ImpactProgressTerm

  original_impact = ImpactProgressTerm.__call__
  original_delivered = DeliveredImpulseTerm.__call__
  sinks: dict[str, dict[str, list[float]]] = {
    arm: {"impact": [], "delivered": []} for arm in ("C", "D-prime")
  }
  for arm in sinks:
    envs[arm].reward_manager.get_term_cfg("impact_progress").func._probe_sink = (
      sinks[arm]["impact"]
    )
    envs[arm].reward_manager.get_term_cfg("delivered_impulse").func._probe_sink = (
      sinks[arm]["delivered"]
    )

  def impact_audit(self, env, **params):
    raw = original_impact(self, env, **params)
    sink = getattr(self, "_probe_sink", None)
    if sink is not None:
      sink.append(float(raw[0]))
    return raw

  def delivered_audit(self, env, **params):
    raw = original_delivered(self, env, **params)
    sink = getattr(self, "_probe_sink", None)
    if sink is not None:
      sink.append(float(raw[0]))
    return raw

  ImpactProgressTerm.__call__ = impact_audit
  DeliveredImpulseTerm.__call__ = delivered_audit
  try:
    yield sinks
  finally:
    ImpactProgressTerm.__call__ = original_impact
    DeliveredImpulseTerm.__call__ = original_delivered


def _tracker_step(env) -> dict[str, Any] | None:
  from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR

  tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR, None)
  if tracker is None:
    return None
  return {
    "started": bool(tracker.started[0]),
    "finalized": bool(tracker.finalized[0]),
    "productive": bool(tracker.productive[0]),
    "reason": int(tracker.reason[0]),
  }


def _return_record(
  impact: list[float],
  delivered: list[float],
  tracker_steps: list[dict[str, Any]],
) -> dict[str, Any]:
  combined = [a + b for a, b in zip(impact, delivered, strict=True)]
  finalization_step = next(
    (index for index, row in enumerate(tracker_steps) if row["finalized"]), None
  )
  final = (
    {"productive": False, "reason": 0}
    if finalization_step is None
    else tracker_steps[finalization_step]
  )
  return {
    "discounted": sum(
      (0.99**index) * value for index, value in enumerate(combined)
    ),
    "undiscounted": sum(combined),
    "impact_stream": impact,
    "delivered_stream": delivered,
    "impact_payout_steps": [
      index for index, value in enumerate(impact) if value > 0.0
    ],
    "delivered_payout_steps": [
      index for index, value in enumerate(delivered) if value > 0.0
    ],
    "finalization_step": finalization_step,
    "finalization_reason": {0: "none", 1: "success", 2: "window"}.get(
      int(final["reason"]), "none"
    ),
    "productive": bool(final["productive"]),
  }


def _tracker_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
  return {
    key: [row[key] for row in rows]
    for key in ("started", "finalized", "productive", "reason")
  }


def _dprime_audit(
  c_parent: dict[str, list[float]],
  d_parent: dict[str, list[float]],
  tracker: dict[str, list[Any]],
  d_return: dict[str, Any],
) -> dict[str, Any]:
  if len(c_parent["impact"]) != len(d_parent["impact"]):
    raise RuntimeError("C/D-prime parent impact stream length mismatch")
  if len(c_parent["delivered"]) != len(d_parent["delivered"]):
    raise RuntimeError("C/D-prime parent delivered stream length mismatch")
  final = d_return["finalization_step"]
  if final is None:
    impact_oracle = delivered_oracle = 0.0
  else:
    indices = [i for i in range(final + 1) if tracker["started"][i]]
    positives = [d_parent["impact"][i] for i in indices if d_parent["impact"][i] > 0]
    impact_oracle = positives[0] if positives else 0.0
    delivered_oracle = sum(
      max(0.0, d_parent["delivered"][i]) for i in indices
    )
    if not d_return["productive"]:
      impact_oracle = delivered_oracle = 0.0
  return {
    "wrapper_parent_impact_raw": copy.deepcopy(d_parent["impact"]),
    "c_legacy_impact_raw": copy.deepcopy(c_parent["impact"]),
    "wrapper_parent_delivered_raw": copy.deepcopy(d_parent["delivered"]),
    "c_legacy_delivered_raw": copy.deepcopy(c_parent["delivered"]),
    "impact_latch_oracle": impact_oracle,
    "impact_manager_payout_raw": sum(d_return["impact_stream"]) / (8.0 * 0.02),
    "delivered_sum_oracle": delivered_oracle,
    "delivered_manager_payout_raw": sum(d_return["delivered_stream"])
    / (2.0 * 0.02),
  }


def _empty_raw(manifest: dict[str, Any]) -> dict[str, Any]:
  return {
    "schema_version": 4,
    "manifest": manifest,
    "physical_traces": [],
    "reward_records": [],
    "phase_rows": [],
    "normalizer_samples": {
      "legacy_terminal_n_s": 0.0,
      "legacy_reference_params": copy.deepcopy(REFERENCE_PARAMS),
      "event_values_n_s": [],
      "event_reference_params": copy.deepcopy(REFERENCE_PARAMS),
    },
    "runtime_s": 0.0,
    "development_replay": {
      arm: {"late_payout": -1.0} for arm in ("D-prime", "F", "E")
    },
    "crash": None,
  }


def _validate_rows(
  rows: Iterable[dict[str, Any]], check: Callable[[dict[str, Any]], None]
) -> None:
  failures = []
  for row in sorted(rows, key=lambda item: item["episode_id"]):
    try:
      check(row)
    except Exception as exc:
      failures.append(f"{type(exc).__name__}: {exc}: {row['episode_id']}")
  if failures:
    raise ValueError("; ".join(failures))


def validate_population(
  raw: dict[str, Any],
  *,
  expected_checkpoints: Sequence[dict[str, Any]] = CHECKPOINTS,
  expected_reset_seeds: Sequence[int] = RESET_SEEDS,
  expected_action_seeds: Sequence[int] = ACTION_SEEDS,
) -> None:
  expected_ids = {
    f"{checkpoint['stratum']}_s{checkpoint['training_seed']}_r{reset_seed}"
    for checkpoint in expected_checkpoints
    for reset_seed in expected_reset_seeds
  }
  trace_ids = [row["episode_id"] for row in raw["physical_traces"]]
  record_ids = [row["episode_id"] for row in raw["reward_records"]]
  if len(trace_ids) != len(set(trace_ids)) or len(record_ids) != len(set(record_ids)):
    raise ValueError("population contains duplicate episode ids")
  if set(trace_ids) != expected_ids or set(record_ids) != expected_ids:
    raise ValueError("population has missing, extra, old, or relabelled rows")
  if len(trace_ids) != len(expected_ids):
    raise ValueError("population count mismatch")
  expected_pairs = set(zip(expected_reset_seeds, expected_action_seeds, strict=True))
  by_checkpoint: dict[tuple[Any, ...], set[tuple[int, int]]] = {}
  for row in raw["reward_records"]:
    key = (
      row["stratum"],
      row["training_seed"],
      row["split"],
    )
    by_checkpoint.setdefault(key, set()).add((row["reset_seed"], row["action_seed"]))
  expected_keys = {
    (row["stratum"], row["training_seed"], row["split"])
    for row in expected_checkpoints
  }
  if set(by_checkpoint) != expected_keys:
    raise ValueError("population checkpoint groups are missing or relabelled")
  if any(pairs != expected_pairs for pairs in by_checkpoint.values()):
    raise ValueError("population paired seeds are missing, duplicated, or swapped")


def _checkpoint_binding_failures(
  records: list[dict[str, Any]], checkpoints: Sequence[dict[str, Any]]
) -> list[str]:
  by_path = {row["path"]: row for row in checkpoints}
  failures = []
  for record in records:
    frozen = by_path.get(record["checkpoint_path"])
    actual = (
      record["stratum"],
      record["training_seed"],
      record["split"],
    )
    expected = (
      None
      if frozen is None
      else (frozen["stratum"], frozen["training_seed"], frozen["split"])
    )
    if actual != expected:
      failures.append(str(record["stratum"]))
  return sorted(set(failures))


def _validate_replay_semantics(record: dict[str, Any]) -> None:
  for arm, payload in record["returns"].items():
    n = len(payload["impact_stream"])
    if len(payload["delivered_stream"]) != n:
      raise ValueError("reward stream length mismatch")
    if payload["impact_payout_steps"] != [
      i for i, value in enumerate(payload["impact_stream"]) if value > 0
    ]:
      raise ValueError("impact payout indices do not match stream")
    if payload["delivered_payout_steps"] != [
      i for i, value in enumerate(payload["delivered_stream"]) if value > 0
    ]:
      raise ValueError("delivered payout indices do not match stream")
    combined = [
      a + b
      for a, b in zip(
        payload["impact_stream"], payload["delivered_stream"], strict=True
      )
    ]
    if abs(payload["undiscounted"] - sum(combined)) > 1e-8:
      raise ValueError("undiscounted return does not recompute")
    discounted = sum((0.99**i) * value for i, value in enumerate(combined))
    if abs(payload["discounted"] - discounted) > 1e-8:
      raise ValueError("discounted return does not recompute")
  tracker_final = {}
  for arm, tracker in record["tracker_streams"].items():
    lengths = {len(tracker[key]) for key in TRACKER_SCHEMA}
    if len(lengths) != 1:
      raise ValueError("tracker stream lengths differ")
    final = next(
      (i for i, value in enumerate(tracker["finalized"]) if value), None
    )
    tracker_final[arm] = final
    payload = record["returns"][arm]
    if payload["finalization_step"] != final:
      raise ValueError("return finalization differs from tracker")
    if final is not None:
      if any(not value for value in tracker["finalized"][final:]):
        raise ValueError("tracker finalization must remain latched")
      productive = bool(tracker["productive"][final])
      reason = int(tracker["reason"][final])
      if payload["productive"] != productive:
        raise ValueError("return productivity differs from tracker")
      if payload["finalization_reason"] != {1: "success", 2: "window"}.get(
        reason, "none"
      ):
        raise ValueError("return reason differs from tracker")
      all_payouts = (
        payload["impact_payout_steps"] + payload["delivered_payout_steps"]
      )
      if productive:
        if any(step != final for step in all_payouts):
          raise ValueError("productive payout must occur only at finalization")
        if len(payload["impact_payout_steps"]) > 1 or len(
          payload["delivered_payout_steps"]
        ) > 1:
          raise ValueError("productive component may pay at most once")
      elif all_payouts:
        raise ValueError("unproductive finalization must consume without payout")
      if any(
        value > 0
        for stream in (payload["impact_stream"], payload["delivered_stream"])
        for value in stream[final + 1 :]
      ):
        raise ValueError("delayed payout after finalization")
  if len(set(tracker_final.values())) != 1:
    raise ValueError("shared finalization boundary differs for D-prime/F/E")
  baseline = record["tracker_streams"]["D-prime"]
  for arm in ("F", "E"):
    differing = [
      key for key in TRACKER_SCHEMA
      if record["tracker_streams"][arm][key] != baseline[key]
    ]
    if differing:
      raise ValueError(
        f"tracker streams differ for {arm}: {', '.join(differing)}"
      )


def _validate_dprime_payout(record: dict[str, Any]) -> None:
  audit = record["dprime_audit"]
  if audit["wrapper_parent_impact_raw"] != audit["c_legacy_impact_raw"]:
    raise ValueError("D-prime impact parent differs from C legacy oracle")
  if audit["wrapper_parent_delivered_raw"] != audit["c_legacy_delivered_raw"]:
    raise ValueError("D-prime delivered parent differs from C legacy oracle")
  if abs(audit["impact_latch_oracle"] - audit["impact_manager_payout_raw"]) > 1e-6:
    raise ValueError("D-prime impact production payout differs from latch oracle")
  if abs(
    audit["delivered_sum_oracle"] - audit["delivered_manager_payout_raw"]
  ) > 1e-6:
    raise ValueError("D-prime delivered production payout differs from sum oracle")


def validate_reward_record(record: dict[str, Any]) -> None:
  _validate_schema(record, REWARD_RECORD_SCHEMA, "reward_record")
  _validate_replay_semantics(record)
  _validate_dprime_payout(record)


def _validate_digests(raw: dict[str, Any]) -> None:
  record_by_id = {row["episode_id"]: row for row in raw["reward_records"]}
  def check(trace: dict[str, Any]) -> None:
    episode_id = trace["episode_id"]
    if trace["action_tape_digest"] != canonical_digest(trace["action_tape"]):
      raise ValueError("action digest mismatch")
    if trace["trace_digest"] != canonical_digest(trace["physical"]):
      raise ValueError("trace digest mismatch")
    record = record_by_id.get(episode_id)
    if record is None:
      raise ValueError("reward record missing")
    if any(
      digest != trace["trace_digest"]
      for digest in record["physical_trace_digests"].values()
    ):
      raise ValueError("physical trace inequality")

  _validate_rows(raw["physical_traces"], check)


def _validate_stream_lengths(raw: dict[str, Any], *, decimation: int = 10) -> None:
  trace_by_id = {row["episode_id"]: row for row in raw["physical_traces"]}
  def check(record: dict[str, Any]) -> None:
    trace = trace_by_id.get(record["episode_id"])
    if trace is None:
      raise ValueError("physical trace missing")
    actions = len(trace["action_tape"])
    physical_lengths = {len(trace["physical"][key]) for key in PHYSICAL_KEYS}
    if physical_lengths != {decimation * actions}:
      raise ValueError("substeps must equal decimation×actions")
    for arm, payload in record["returns"].items():
      if len(payload["impact_stream"]) != actions:
        raise ValueError(f"stream/action length mismatch: {arm}")
    for arm, tracker in record["tracker_streams"].items():
      if len(tracker["started"]) != actions:
        raise ValueError(f"tracker/action length mismatch: {arm}")

  _validate_rows(raw["reward_records"], check)


def validate_raw_payload(
  raw: dict[str, Any],
  *,
  root: Path = ROOT,
  assets_root: Path = ASSETS_ROOT,
  expected_checkpoints: Sequence[dict[str, Any]] = CHECKPOINTS,
  expected_reset_seeds: Sequence[int] = RESET_SEEDS,
  expected_action_seeds: Sequence[int] = ACTION_SEEDS,
  verify_checkpoint_hashes: bool = True,
  verify_source_hashes: bool = True,
) -> None:
  _validate_schema(raw, RAW_SCHEMA, "raw")
  if raw["schema_version"] != 4:
    raise ValueError("raw schema version mismatch")
  _validate_rows(raw["reward_records"], _validate_replay_semantics)
  _validate_rows(raw["reward_records"], _validate_dprime_payout)
  validate_population(
    raw,
    expected_checkpoints=expected_checkpoints,
    expected_reset_seeds=expected_reset_seeds,
    expected_action_seeds=expected_action_seeds,
  )
  binding = _checkpoint_binding_failures(raw["reward_records"], expected_checkpoints)
  if binding:
    raise ValueError(f"checkpoint tuple/path mismatch: {', '.join(binding)}")
  _validate_digests(raw)
  decimation = int(raw["manifest"].get("decimation", 10))
  _validate_stream_lengths(raw, decimation=decimation)
  validate_reward_weights(raw["manifest"]["reward_weights"])
  if verify_checkpoint_hashes:
    validate_file_hashes(
      root,
      {row["path"]: row["sha256"] for row in expected_checkpoints},
      label="checkpoint",
    )
  if verify_source_hashes and raw["manifest"].get("relevant_source_sha256"):
    validate_file_hashes(
      root,
      raw["manifest"]["relevant_source_sha256"],
      label="relevant source",
    )
    validate_relevant_provenance(
      raw["manifest"]["relevant_provenance"],
      root,
      assets_root,
      require_git_clean=True,
    )


def write_raw(raw: dict[str, Any], path: Path) -> None:
  import numpy as np

  payload = json.dumps(
    raw, sort_keys=True, separators=(",", ":"), allow_nan=False
  )
  np.savez_compressed(path, payload_json=np.asarray(payload))


def read_raw(path: Path) -> dict[str, Any]:
  import numpy as np

  with np.load(path, allow_pickle=False) as saved:
    if set(saved.files) != NPZ_KEYS:
      raise ValueError("NPZ schema mismatch")
    payload = json.loads(str(saved["payload_json"]))
  return payload


def validate_persisted_raw(path: Path, **kwargs) -> dict[str, Any]:
  raw = read_raw(path)
  validate_raw_payload(raw, **kwargs)
  return raw


def _validate_phase(rows: list[dict[str, Any]]) -> None:
  if len(rows) != 10 or sorted(row["phase"] for row in rows) != list(range(10)):
    raise ValueError("phase invariance must be 10/10")
  baseline = rows[0]
  invariant = (
    "matched_event_digest",
    "productive",
    "v_precontact_m_s",
    "delivered_n_s",
    "depth_at_contact_m",
    "peak_depth_m",
    "progress_m",
    "finalization_reason",
  )
  if not all(row["contact_detected"] for row in rows):
    raise ValueError("phase invariance contact recall must be 10/10")
  if any(
    any(row[key] != baseline[key] for key in invariant) for row in rows[1:]
  ):
    raise ValueError("phase event values are not invariant")
  for row in rows:
    if row["productive"] and (
      row["impact_payout_count"] != 1 or row["delivered_payout_count"] != 1
    ):
      raise ValueError("phase productive payout must occur exactly once")
    tolerance = max(0.02, 0.02 * abs(row["v_true_m_s"]))
    if abs(row["v_precontact_m_s"] - row["v_true_m_s"]) > tolerance:
      raise ValueError("phase finite-difference velocity tolerance exceeded")


def _validate_normalizers(samples: dict[str, Any]) -> None:
  legacy = float(samples["legacy_terminal_n_s"])
  if abs(legacy / 0.6094 - 1.0) > 0.01:
    raise ValueError("legacy terminal normalizer exceeds 1%")
  if samples.get("legacy_reference_params", REFERENCE_PARAMS) != REFERENCE_PARAMS:
    raise ValueError("legacy reference parameters mismatch")
  values = samples["event_values_n_s"]
  if len(values) != 8 or any(value <= 0 for value in values):
    raise ValueError("event normalizer requires eight productive successes")
  event_mean = mean(values)
  if any(abs(value / event_mean - 1.0) > 0.01 for value in values):
    raise ValueError("event samples differ by more than 1%")
  if abs(0.3088 / event_mean - 1.0) > 0.01:
    raise ValueError("configured event normalizer differs by more than 1%")
  if samples.get("event_reference_params", REFERENCE_PARAMS) != REFERENCE_PARAMS:
    raise ValueError("event reference parameters mismatch")


def _validate_development(development: dict[str, Any]) -> None:
  if set(development) != {"D-prime", "F", "E"}:
    raise ValueError("development replay requires D-prime/F/E")
  for arm, result in development.items():
    if result["late_payout"] != 0.0:
      raise ValueError(f"late payout in development replay: {arm}")


def _gate(
  gate_id: str, target: str, check: Callable[[], None]
) -> dict[str, Any]:
  try:
    check()
  except Exception as exc:
    return {
      "id": gate_id,
      "target": target,
      "passed": False,
      "reason": f"{type(exc).__name__}: {exc}",
    }
  return {"id": gate_id, "target": target, "passed": True, "reason": ""}


def collect_gate_results(
  summary: dict[str, Any],
  raw: dict[str, Any],
  *,
  root: Path = ROOT,
  assets_root: Path = ASSETS_ROOT,
  manifest_path: Path | None = None,
  raw_path: Path | None = None,
  expected_checkpoints: Sequence[dict[str, Any]] = CHECKPOINTS,
  expected_reset_seeds: Sequence[int] = RESET_SEEDS,
  expected_action_seeds: Sequence[int] = ACTION_SEEDS,
  verify_artifact_hashes: bool = True,
  verify_source_hashes: bool = True,
) -> list[dict[str, Any]]:
  def population_check():
    validate_population(
      raw,
      expected_checkpoints=expected_checkpoints,
      expected_reset_seeds=expected_reset_seeds,
      expected_action_seeds=expected_action_seeds,
    )
    expected = len(expected_checkpoints) * len(expected_reset_seeds)
    if summary["population"]["completed_episodes"] != expected:
      raise ValueError("summary completed count mismatch")

  def binding_check():
    failures = _checkpoint_binding_failures(raw["reward_records"], expected_checkpoints)
    if failures:
      raise ValueError(f"checkpoint tuple/path mismatch: {', '.join(failures)}")

  def stochastic_check():
    validate_stochastic_policy_proof(
      raw["manifest"]["stochastic_policy_proof"], expected_checkpoints
    )
    if summary["stochastic_policy"] != {
      "checkpoint_count": len(expected_checkpoints),
      "passed": True,
    }:
      raise ValueError("stochastic proof summary mismatch")

  specs = (
    ("schema", "recursive typed raw schema",
     lambda: _validate_schema(raw, RAW_SCHEMA, "raw")),
    ("population", "exact checkpoint×seed population", population_check),
    ("checkpoint_binding", "frozen checkpoint tuples", binding_check),
    ("digests", "action/trace/physical digests", lambda: _validate_digests(raw)),
    ("replay_semantics", "D-prime/F/E tracker and event payout semantics",
     lambda: _validate_rows(raw["reward_records"], _validate_replay_semantics)),
    ("dprime_payout", "D-prime production parent and payout fidelity",
     lambda: _validate_rows(raw["reward_records"], _validate_dprime_payout)),
    ("stream_lengths", "manager streams/actions/substeps",
     lambda: _validate_stream_lengths(raw, decimation=raw["manifest"]["decimation"])),
    ("weights", "equal 8/2 treatment weights",
     lambda: validate_reward_weights(raw["manifest"]["reward_weights"])),
    ("stochastic_policy", "live seeded stochastic-policy behavior", stochastic_check),
    ("normalizers", "legacy/event reference gates",
     lambda: _validate_normalizers(raw["normalizer_samples"])),
    ("phase_invariance", "10/10 phase invariance",
     lambda: _validate_phase(raw["phase_rows"])),
    ("late_payout", "delayed-recontact replay",
     lambda: _validate_development(raw["development_replay"])),
  )
  gates = [_gate(*spec) for spec in specs]
  if verify_source_hashes:
    gates.append(
      _gate(
        "source_binding",
        "relevant source tree unchanged",
        lambda: validate_relevant_provenance(
          raw["manifest"]["relevant_provenance"],
          root,
          assets_root,
          require_git_clean=True,
        ),
      )
    )
  if verify_artifact_hashes:
    def artifact_check():
      if manifest_path is None or raw_path is None:
        raise ValueError("artifact paths missing")
      if summary["artifact_sha256"]["manifest"] != sha256_path(manifest_path):
        raise ValueError("external manifest byte hash mismatch")
      if summary["artifact_sha256"]["raw"] != sha256_path(raw_path):
        raise ValueError("raw byte hash mismatch")
      external = json.loads(manifest_path.read_text())
      if external != raw["manifest"]:
        raise ValueError("embedded manifest binding mismatch")
      if summary["raw_artifact"]["size_bytes"] != raw_path.stat().st_size:
        raise ValueError("raw byte size mismatch")

    gates.append(_gate("artifact_binding", "manifest/raw persisted bytes", artifact_check))
  if raw.get("crash"):
    gates.append(
      {
        "id": "crash",
        "target": "complete execution",
        "passed": False,
        "reason": str(raw["crash"]),
      }
    )
  return gates


def collect_failure_reasons(
  summary: dict[str, Any], raw: dict[str, Any], **kwargs
) -> list[str]:
  return [
    f"{gate['id']}: {gate['reason']}"
    for gate in collect_gate_results(summary, raw, **kwargs)
    if not gate["passed"]
  ]


def _aggregate_rows(raw: dict[str, Any]) -> dict[str, Any]:
  by_arm: dict[str, Any] = {}
  for arm in TASKS:
    rows = [record["returns"][arm] for record in raw["reward_records"]]
    by_arm[arm] = {
      "episode_count": len(rows),
      "discounted_mean": mean([row["discounted"] for row in rows]) if rows else 0.0,
      "undiscounted_mean": mean([row["undiscounted"] for row in rows]) if rows else 0.0,
      "impact_payout_episode_count": sum(bool(row["impact_payout_steps"]) for row in rows),
      "delivered_payout_episode_count": sum(bool(row["delivered_payout_steps"]) for row in rows),
    }
  by_stratum: dict[str, Any] = {}
  strata = sorted({row["stratum"] for row in raw["reward_records"]})
  for stratum in strata:
    by_stratum[stratum] = {}
    for arm in TASKS:
      values = [
        row["returns"][arm]["undiscounted"]
        for row in raw["reward_records"]
        if row["stratum"] == stratum
      ]
      by_stratum[stratum][arm] = {
        "episode_count": len(values),
        "undiscounted_mean": mean(values) if values else 0.0,
      }
  return {"by_arm": by_arm, "by_stratum": by_stratum}


def build_summary(
  raw: dict[str, Any],
  *,
  manifest_sha256: str,
  raw_sha256: str,
  raw_size_bytes: int = 0,
  root: Path = ROOT,
  assets_root: Path = ASSETS_ROOT,
  manifest_path: Path | None = None,
  raw_path: Path | None = None,
  expected_checkpoints: Sequence[dict[str, Any]] = CHECKPOINTS,
  expected_reset_seeds: Sequence[int] = RESET_SEEDS,
  expected_action_seeds: Sequence[int] = ACTION_SEEDS,
  verify_artifact_hashes: bool = False,
  verify_source_hashes: bool = False,
) -> dict[str, Any]:
  legacy = float(raw["normalizer_samples"]["legacy_terminal_n_s"])
  values = list(raw["normalizer_samples"]["event_values_n_s"])
  proof_rows = raw["manifest"].get("stochastic_policy_proof", [])
  try:
    validate_stochastic_policy_proof(proof_rows, expected_checkpoints)
    stochastic_passed = True
  except Exception:
    stochastic_passed = False
  summary: dict[str, Any] = {
    "schema_version": 4,
    "valid": False,
    "failure_reasons": [],
    "gates": [],
    "artifact_sha256": {"manifest": manifest_sha256, "raw": raw_sha256},
    "raw_artifact": {
      "schema_version": raw["schema_version"],
      "size_bytes": raw_size_bytes,
      "physical_trace_count": len(raw["physical_traces"]),
      "reward_record_count": len(raw["reward_records"]),
    },
    "population": {
      "expected_episodes": len(expected_checkpoints) * len(expected_reset_seeds),
      "completed_episodes": len(raw["reward_records"]),
      "checkpoint_groups": len(
        {
          (
            row["checkpoint_path"],
            row["stratum"],
            row["training_seed"],
            row["split"],
          )
          for row in raw["reward_records"]
        }
      ),
      "paired_seed_count": len(expected_reset_seeds),
    },
    "normalizers": {
      "legacy": {"configured_n_s": 0.6094, "observed_n_s": legacy},
      "first_strike_success": {
        "configured_n_s": 0.3088,
        "sample_count": len(values),
        "mean_n_s": mean(values) if values else 0.0,
        "sd_n_s": pstdev(values) if len(values) > 1 else 0.0,
        "range_n_s": max(values) - min(values) if values else 0.0,
      },
    },
    "phase_invariance": {
      "count": len(raw["phase_rows"]),
      "passed": len(raw["phase_rows"]) == 10,
    },
    "stochastic_policy": {
      "checkpoint_count": len(proof_rows),
      "passed": stochastic_passed,
    },
    "development_replay": copy.deepcopy(raw["development_replay"]),
    "aggregates": _aggregate_rows(raw),
    "provenance": {
      "repo": copy.deepcopy(raw["manifest"].get("repo", {})),
      "assets": copy.deepcopy(raw["manifest"].get("assets", {})),
      "task_config_digest": raw["manifest"].get("task_config_digest", ""),
      "relevant_tree_sha256": raw["manifest"].get(
        "relevant_provenance", {}
      ).get("tree_sha256", ""),
      "versions": copy.deepcopy(
        raw["manifest"].get("relevant_provenance", {}).get("versions", {})
      ),
    },
  }
  gates = collect_gate_results(
    summary,
    raw,
    root=root,
    assets_root=assets_root,
    manifest_path=manifest_path,
    raw_path=raw_path,
    expected_checkpoints=expected_checkpoints,
    expected_reset_seeds=expected_reset_seeds,
    expected_action_seeds=expected_action_seeds,
    verify_artifact_hashes=verify_artifact_hashes,
    verify_source_hashes=verify_source_hashes,
  )
  summary["gates"] = gates
  summary["failure_reasons"] = [
    f"{gate['id']}: {gate['reason']}" for gate in gates if not gate["passed"]
  ]
  summary["valid"] = all(gate["passed"] for gate in gates)
  return summary


def _validate_summary_schema(summary: dict[str, Any]) -> None:
  _validate_schema(summary, SUMMARY_SCHEMA, "summary")
  if summary["schema_version"] != 4:
    raise ValueError("summary schema version mismatch")
  expected_valid = all(gate["passed"] for gate in summary["gates"])
  if summary["valid"] is not expected_valid:
    raise ValueError("summary valid flag is not gate conjunction")
  expected_reasons = [
    f"{gate['id']}: {gate['reason']}"
    for gate in summary["gates"]
    if not gate["passed"]
  ]
  if summary["failure_reasons"] != expected_reasons:
    raise ValueError("summary failure reasons are not derived from current gates")


def persist_final_artifacts(
  raw: dict[str, Any],
  *,
  manifest_path: Path,
  raw_path: Path,
  summary_path: Path,
  root: Path = ROOT,
  assets_root: Path = ASSETS_ROOT,
  expected_checkpoints: Sequence[dict[str, Any]] = CHECKPOINTS,
  expected_reset_seeds: Sequence[int] = RESET_SEEDS,
  expected_action_seeds: Sequence[int] = ACTION_SEEDS,
  verify_source_hashes: bool = True,
) -> bool:
  write_raw(raw, raw_path)
  raw_hash = sha256_path(raw_path)
  manifest_hash = sha256_path(manifest_path)
  reloaded = read_raw(raw_path)
  summary = build_summary(
    reloaded,
    manifest_sha256=manifest_hash,
    raw_sha256=raw_hash,
    raw_size_bytes=raw_path.stat().st_size,
    root=root,
    assets_root=assets_root,
    manifest_path=manifest_path,
    raw_path=raw_path,
    expected_checkpoints=expected_checkpoints,
    expected_reset_seeds=expected_reset_seeds,
    expected_action_seeds=expected_action_seeds,
    verify_artifact_hashes=True,
    verify_source_hashes=verify_source_hashes,
  )
  _validate_summary_schema(summary)
  summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
  if summary["valid"]:
    validate_persisted_artifacts(
      manifest_path=manifest_path,
      raw_path=raw_path,
      summary_path=summary_path,
      root=root,
      assets_root=assets_root,
      expected_checkpoints=expected_checkpoints,
      expected_reset_seeds=expected_reset_seeds,
      expected_action_seeds=expected_action_seeds,
      verify_source_hashes=verify_source_hashes,
    )
  return bool(summary["valid"])


def validate_persisted_artifacts(
  *,
  manifest_path: Path,
  raw_path: Path,
  summary_path: Path,
  root: Path = ROOT,
  assets_root: Path = ASSETS_ROOT,
  expected_checkpoints: Sequence[dict[str, Any]] = CHECKPOINTS,
  expected_reset_seeds: Sequence[int] = RESET_SEEDS,
  expected_action_seeds: Sequence[int] = ACTION_SEEDS,
  verify_source_hashes: bool = True,
) -> None:
  raw = validate_persisted_raw(
    raw_path,
    root=root,
    assets_root=assets_root,
    expected_checkpoints=expected_checkpoints,
    expected_reset_seeds=expected_reset_seeds,
    expected_action_seeds=expected_action_seeds,
    verify_source_hashes=verify_source_hashes,
  )
  summary = json.loads(summary_path.read_text())
  _validate_summary_schema(summary)
  expected = build_summary(
    raw,
    manifest_sha256=sha256_path(manifest_path),
    raw_sha256=sha256_path(raw_path),
    raw_size_bytes=raw_path.stat().st_size,
    root=root,
    assets_root=assets_root,
    manifest_path=manifest_path,
    raw_path=raw_path,
    expected_checkpoints=expected_checkpoints,
    expected_reset_seeds=expected_reset_seeds,
    expected_action_seeds=expected_action_seeds,
    verify_artifact_hashes=True,
    verify_source_hashes=verify_source_hashes,
  )
  if summary != expected:
    raise ValueError("persisted summary aggregates/gates do not recompute")


def record_completed_checkpoint(
  raw: dict[str, Any],
  checkpoint_traces: list[dict[str, Any]],
  checkpoint_records: list[dict[str, Any]],
  *,
  raw_path: Path,
  runtime_s: float,
) -> None:
  raw["physical_traces"].extend(copy.deepcopy(checkpoint_traces))
  raw["reward_records"].extend(copy.deepcopy(checkpoint_records))
  raw["runtime_s"] = runtime_s
  write_raw(raw, raw_path)


def _run_bank(
  raw: dict[str, Any], *, manifest_path: Path, raw_path: Path, started: float
) -> None:
  import torch
  from dataclasses import asdict
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from tensordict import TensorDict

  envs = {arm: _build_env(task, play=False) for arm, task in TASKS.items()}
  captures = {arm: _capture_setup(env) for arm, env in envs.items()}
  agent_cfg = load_rl_cfg(TASKS["C"])
  wrapped_c = RslRlVecEnvWrapper(envs["C"], clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(TASKS["C"]) or MjlabOnPolicyRunner)(
    wrapped_c, asdict(agent_cfg), device="cpu"
  )

  def reset_obs(env, seed: int):
    obs, _ = env.reset(seed=seed)
    return TensorDict(obs, batch_size=[1])

  proof_rows = []
  for checkpoint in raw["manifest"]["checkpoints"]:
    runner.load(
      str(ROOT / checkpoint["path"]),
      load_cfg={"actor": True},
      strict=True,
      map_location="cpu",
    )
    policy = runner.get_inference_policy(device="cpu")
    observation = reset_obs(envs["C"], RESET_SEEDS[0])
    with torch.inference_mode():
      proof_rows.append(
        prove_stochastic_policy(
          policy,
          observation,
          checkpoint_path=checkpoint["path"],
          same_seed=STOCHASTIC_PROOF_SEEDS[0],
          different_seed=STOCHASTIC_PROOF_SEEDS[1],
          clip=agent_cfg.clip_actions,
        )
      )
  raw["manifest"]["stochastic_policy_proof"] = proof_rows
  manifest_path.write_text(
    json.dumps(raw["manifest"], sort_keys=True, indent=2, allow_nan=False) + "\n"
  )
  validate_stochastic_policy_proof(proof_rows, raw["manifest"]["checkpoints"])

  with _parent_reader_audit(envs) as audit:
    def generate(policy, reset_seed: int, action_seed: int, *, stochastic: bool):
      actions: list[list[float]] = []
      impact: list[float] = []
      delivered: list[float] = []
      tracker_steps: list[dict[str, Any]] = []
      obs = reset_obs(envs["C"], reset_seed)
      torch.manual_seed(action_seed)
      captures["C"][1]()
      audit["C"]["impact"].clear()
      audit["C"]["delivered"].clear()
      for _ in range(envs["C"].max_episode_length):
        with torch.inference_mode():
          if stochastic:
            action = sample_stochastic_action(
              policy, obs, clip=agent_cfg.clip_actions
            )
          else:
            action = policy(obs, stochastic_output=False).clamp(
              -agent_cfg.clip_actions, agent_cfg.clip_actions
            )
        actions.append(action[0].detach().cpu().tolist())
        obs_dict, _, terminated, truncated, _ = envs["C"].step(action)
        obs = TensorDict(obs_dict, batch_size=[1])
        i, d = scaled_treatment_terms(
          envs["C"].reward_manager.get_active_iterable_terms(0),
          step_dt=float(envs["C"].step_dt),
        )
        impact.append(i)
        delivered.append(d)
        if bool(terminated[0] | truncated[0]):
          break
      return (
        actions,
        copy.deepcopy(captures["C"][0]),
        _return_record(impact, delivered, tracker_steps),
        copy.deepcopy(audit["C"]),
      )

    def replay(arm: str, actions: list[list[float]], reset_seed: int):
      env = envs[arm]
      reset_obs(env, reset_seed)
      captures[arm][1]()
      if arm == "D-prime":
        audit[arm]["impact"].clear()
        audit[arm]["delivered"].clear()
      impact: list[float] = []
      delivered: list[float] = []
      tracker_steps: list[dict[str, Any]] = []
      for values in actions:
        action = torch.tensor([values], dtype=torch.float32)
        env.step(action)
        i, d = scaled_treatment_terms(
          env.reward_manager.get_active_iterable_terms(0),
          step_dt=float(env.step_dt),
        )
        impact.append(i)
        delivered.append(d)
        tracker = _tracker_step(env)
        if tracker is not None:
          tracker_steps.append(tracker)
      return (
        copy.deepcopy(captures[arm][0]),
        _return_record(impact, delivered, tracker_steps),
        _tracker_columns(tracker_steps),
        copy.deepcopy(audit.get(arm, {})),
      )

    for checkpoint in raw["manifest"]["checkpoints"]:
      path = ROOT / checkpoint["path"]
      if sha256_path(path) != checkpoint["sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint['path']}")
      runner.load(
        str(path), load_cfg={"actor": True}, strict=True, map_location="cpu"
      )
      policy = runner.get_inference_policy(device="cpu")
      checkpoint_traces: list[dict[str, Any]] = []
      checkpoint_records: list[dict[str, Any]] = []
      for reset_seed, action_seed in zip(RESET_SEEDS, ACTION_SEEDS, strict=True):
        actions, trace_c, return_c, c_parent = generate(
          policy, reset_seed, action_seed, stochastic=True
        )
        arm_traces = {"C": trace_c}
        arm_returns = {"C": return_c}
        trackers: dict[str, Any] = {}
        d_parent: dict[str, list[float]] = {}
        for arm in ("D-prime", "F", "E"):
          trace, result, tracker, parent = replay(arm, actions, reset_seed)
          arm_traces[arm] = trace
          arm_returns[arm] = result
          trackers[arm] = tracker
          if arm == "D-prime":
            d_parent = parent
        trace_digests = {
          arm: canonical_digest(trace) for arm, trace in arm_traces.items()
        }
        if len(set(trace_digests.values())) != 1:
          raise RuntimeError(
            f"physical trace inequality: {checkpoint['stratum']}/"
            f"s{checkpoint['training_seed']}/reset{reset_seed}"
          )
        episode_id = (
          f"{checkpoint['stratum']}_s{checkpoint['training_seed']}_r{reset_seed}"
        )
        checkpoint_traces.append(
          {
            "episode_id": episode_id,
            "action_tape": actions,
            "action_tape_digest": canonical_digest(actions),
            "physical": trace_c,
            "trace_digest": trace_digests["C"],
          }
        )
        checkpoint_records.append(
          {
            "episode_id": episode_id,
            "checkpoint_path": checkpoint["path"],
            "stratum": checkpoint["stratum"],
            "training_seed": checkpoint["training_seed"],
            "split": checkpoint["split"],
            "reset_seed": reset_seed,
            "action_seed": action_seed,
            "physical_trace_digests": trace_digests,
            "returns": arm_returns,
            "tracker_streams": trackers,
            "dprime_audit": _dprime_audit(
              c_parent, d_parent, trackers["D-prime"], arm_returns["D-prime"]
            ),
          }
        )
      record_completed_checkpoint(
        raw,
        checkpoint_traces,
        checkpoint_records,
        raw_path=raw_path,
        runtime_s=time.monotonic() - started,
      )
      print(
        f"bank: {checkpoint['stratum']} seed "
        f"{checkpoint['training_seed']} complete",
        flush=True,
      )

    development_ckpt = next(
      item
      for item in CHECKPOINTS
      if item["stratum"] == "mx_maxmax" and item["training_seed"] == 2
    )
    runner.load(
      str(ROOT / development_ckpt["path"]),
      load_cfg={"actor": True},
      strict=True,
      map_location="cpu",
    )
    policy = runner.get_inference_policy(device="cpu")
    actions, _, _, _ = generate(policy, 12345, 12345, stochastic=False)
    actions = actions + [[0.0, 0.0, 0.0] for _ in range(5)]
    development_envs = {
      arm: _build_env(TASKS[arm], play=False, disable_success=True)
      for arm in ("D-prime", "F", "E")
    }
    for arm, development_env in development_envs.items():
      reset_obs(development_env, 12345)
      impact: list[float] = []
      delivered: list[float] = []
      tracker_steps: list[dict[str, Any]] = []
      for values in actions:
        action = torch.tensor([values], dtype=torch.float32)
        _, _, _, truncated, _ = development_env.step(action)
        i, d = scaled_treatment_terms(
          development_env.reward_manager.get_active_iterable_terms(0),
          step_dt=float(development_env.step_dt),
        )
        impact.append(i)
        delivered.append(d)
        tracker_steps.append(_tracker_step(development_env))
        if bool(truncated[0]):
          break
      result = _return_record(impact, delivered, tracker_steps)
      payout_steps = sorted(
        set(result["impact_payout_steps"] + result["delivered_payout_steps"])
      )
      late = 0.0
      if payout_steps:
        first = payout_steps[0]
        late = sum(result["impact_stream"][first + 1 :]) + sum(
          result["delivered_stream"][first + 1 :]
        )
      raw["development_replay"][arm] = {"late_payout": late}
    for development_env in development_envs.values():
      development_env.close()

  for env in envs.values():
    env.close()


def main() -> int:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  started = time.monotonic()
  manifest_path = OUT / "bank_manifest.json"
  raw_path = OUT / "probe_raw.npz"
  summary_path = OUT / "probe_summary.json"

  manifest = frozen_manifest()
  manifest_path.write_text(
    json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
  )
  raw = _empty_raw(manifest)
  crash: Exception | None = None
  try:
    raw["phase_rows"] = _phase_probe()
    legacy = _reference_once(event=False)
    raw["normalizer_samples"]["legacy_terminal_n_s"] = legacy["delivered_n_s"]
    raw["normalizer_samples"]["legacy_reference_params"] = legacy[
      "reference_params"
    ]
    event_values = []
    for _ in range(8):
      result = _reference_once(event=True)
      if not (
        result["terminated"]
        and not result["truncated"]
        and result["nail_driven"]
        and result["event_finalized"]
        and result["event_productive"]
        and result["event_reason"] == 1
      ):
        raise RuntimeError(f"event reference was not productive success: {result}")
      event_values.append(result["event_delivered_n_s"])
    raw["normalizer_samples"]["event_values_n_s"] = event_values
    raw["normalizer_samples"]["event_reference_params"] = copy.deepcopy(
      REFERENCE_PARAMS
    )
    _validate_normalizers(raw["normalizer_samples"])
    _run_bank(
      raw, manifest_path=manifest_path, raw_path=raw_path, started=started
    )
  except Exception as exc:
    crash = exc
    raw["crash"] = f"{type(exc).__name__}: {exc}"
  finally:
    raw["runtime_s"] = time.monotonic() - started
    valid = persist_final_artifacts(
      raw,
      manifest_path=manifest_path,
      raw_path=raw_path,
      summary_path=summary_path,
    )
    summary = json.loads(summary_path.read_text())
    print(
      json.dumps(
        {
          "valid": valid,
          "runtime_s": raw["runtime_s"],
          "completed_episodes": len(raw["reward_records"]),
          "failure_reasons": summary["failure_reasons"],
          "manifest_sha256": summary["artifact_sha256"]["manifest"],
          "raw_sha256": summary["artifact_sha256"]["raw"],
        },
        indent=2,
      ),
      flush=True,
    )
  return 0 if valid and crash is None else 1


if __name__ == "__main__":
  raise SystemExit(main())
