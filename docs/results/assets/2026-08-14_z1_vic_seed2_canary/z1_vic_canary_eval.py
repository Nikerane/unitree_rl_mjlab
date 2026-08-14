#!/usr/bin/env python3
"""One-shot, fail-closed 64-world evaluator for the Z1 VIC-TT canary.

This is an ephemeral wrapper around the banked FIC evaluator at code revision
``ca5e83bc84bc979bac857088a9330f2a2bd8034b``.  It deliberately lives outside
the repository.  Scientific parameters are not CLI options: the only runtime
inputs are immutable artifact locations and the independently recorded final
checkpoint SHA-256.

The wrapper imports and patches the existing evaluator in memory.  It reuses
that evaluator's first-terminal capture, unfinished-world reset handling,
500-Hz joint-velocity accumulator, impulse capture, and summary implementation.
The patch admits only the qualified VIC-TT config and adds live action, gain,
and actuator-force validation/capture.  No repository file is modified.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
from importlib.metadata import version as package_version
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

# Repository imports must remain read-only even when the detached Vega worktree
# has no pre-existing bytecode cache.
sys.dont_write_bytecode = True


EXPECTED_CODE_REVISION = "ca5e83bc84bc979bac857088a9330f2a2bd8034b"
EXPECTED_ASSET_REVISION = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
EXPECTED_FIC_EVALUATOR_SHA256 = (
    "b6be88e377bc7db9c0b0a6161a7ee7bfd22a343f30e99d74956683db0e6c2e3f"
)
EXPECTED_INITIAL_POPULATION_SHA256 = (
    "320efae8c21b3303c5dc3f18ae7df00e887653abf26aa8fb413803cf78d094cb"
)
EXPECTED_MODEL_0_SHA256 = (
    "f5353960ee0362ed467649ae5344dcc01fdd050f8e893deea9460fb8ccf5dec3"
)
EXPECTED_TELEMETRY_SHA256 = (
    "402d5dbe92125facf301cbfcf84c0b36109dd01c6b7978060fcb5a379dd5f62a"
)
VIC_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-VariableImpedance-TT"
)
FIC_TT_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-Fixed-TT"
)
ACTION_TERMS = ("joint_position", "joint_stiffness")
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
CONTROL_IDS = (0, 5, 1, 2, 3, 4)
OBSERVATION_WIDTH = 40
ACTION_WIDTH = 12
C = 1.25
EVALUATION_SEED = 2026081202
TRAINING_SEED = 2
NUM_ENVS = 64
EPISODE_LENGTH_S = 4.0
PHYSICS_DT_S = 0.002
CONTROL_DECIMATION = 10
MAX_CONTROL_STEPS = 200
JOINT_VELOCITY_LIMIT_RAD_S = 3.1415
JOINT_IMPULSE_CAP_N_M_S = (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
NOMINAL_KP = (1000.0, 1500.0, 1000.0, 1000.0, 1000.0, 1000.0)
NOMINAL_KD = (100.0, 150.0, 100.0, 100.0, 100.0, 100.0)
FORCE_RANGE_N_M = (
    (-30.0, 30.0),
    (-60.0, 60.0),
    (-30.0, 30.0),
    (-30.0, 30.0),
    (-30.0, 30.0),
    (-30.0, 30.0),
)
EXPECTED_GPU = "NVIDIA A100-SXM4-40GB"
EXPECTED_PACKAGES = {"mjlab": "1.4.0", "mujoco": "3.8.1", "mujoco-warp": "3.8.1"}
OUTPUT_BASENAME = "victt_seed2_eval.json"
CHECKPOINT_BASENAME = "model_499.pt"
TELEMETRY_BASENAME = "victt_seed2_telemetry.json"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT = re.compile(r"^(?P<job>[0-9]+)_victt_seed2$")


class EvaluationError(RuntimeError):
    """Raised when any frozen identity, schema, or runtime invariant drifts."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def finite_number(value: object, *, name: str) -> float:
    require(isinstance(value, Real) and not isinstance(value, bool), f"{name} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{name} must be finite")
    return number


def finite_vector(
    value: object,
    *,
    name: str,
    width: int = 6,
    lower: float | Sequence[float] | None = None,
    upper: float | Sequence[float] | None = None,
    positive: bool = False,
) -> list[float]:
    require(isinstance(value, list) and len(value) == width, f"{name} must contain {width} values")
    result = [finite_number(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if positive:
        require(all(item > 0.0 for item in result), f"{name} must be strictly positive")
    if lower is not None:
        lower_values = [float(lower)] * width if isinstance(lower, Real) else list(lower)
        require(len(lower_values) == width, f"{name} lower bound width drifted")
        require(all(item >= bound for item, bound in zip(result, lower_values, strict=True)), f"{name} is below its bound")
    if upper is not None:
        upper_values = [float(upper)] * width if isinstance(upper, Real) else list(upper)
        require(len(upper_values) == width, f"{name} upper bound width drifted")
        require(all(item <= bound for item, bound in zip(result, upper_values, strict=True)), f"{name} is above its bound")
    return result


def require_finite_json(value: object, path: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            require(isinstance(key, str), f"{path} contains a non-string key")
            require_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_json(item, f"{path}[{index}]")
    elif value is None or isinstance(value, (str, bool)):
        return
    elif isinstance(value, Real):
        require(math.isfinite(float(value)), f"{path} contains a non-finite number")
    else:
        raise EvaluationError(f"{path} contains a non-JSON value: {type(value).__name__}")


def exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, object]:
    require(isinstance(value, Mapping), f"{name} must be a mapping")
    require(set(value) == expected, f"{name} schema drifted: {sorted(set(value))}")
    return value


def git_identity(path: Path, *, expected_revision: str) -> dict[str, str]:
    require(path.is_dir(), f"repository is unavailable: {path}")
    top = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "--show-toplevel"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(Path(top).resolve() == path.resolve(), f"repository root drifted: {path}")
    revision = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(_HEX40.fullmatch(revision) is not None, f"repository has no full revision: {path}")
    require(revision == expected_revision, f"repository revision mismatch: {path}")
    require(status == "", f"repository is not clean: {path}")
    return {"revision": revision, "status": status}


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_checkpoint_layout(checkpoint: Path, repo_root: Path) -> tuple[Path, str]:
    require(checkpoint.is_absolute(), "checkpoint path must be absolute")
    require(checkpoint.is_file() and not checkpoint.is_symlink(), "checkpoint must be a regular non-symlink file")
    require(checkpoint.name == CHECKPOINT_BASENAME, f"checkpoint basename must be {CHECKPOINT_BASENAME}")
    run_dir = checkpoint.parent
    require(run_dir.parent.name == "z1_hammer", "checkpoint is outside the canonical z1_hammer run leaf")
    require(run_dir.parent.parent.name == "rsl_rl", "checkpoint is outside the canonical rsl_rl run leaf")
    require(run_dir.parent.parent.parent.name == "logs", "checkpoint is outside the canonical logs run leaf")
    attempt = run_dir.parent.parent.parent.parent
    match = _ATTEMPT.fullmatch(attempt.name)
    require(match is not None, "checkpoint attempt leaf is not <job>_victt_seed2")
    job = match.group("job")
    require(run_dir.name.endswith(f"_victt_seed2_job{job}"), "checkpoint training run name does not match its Slurm job")
    require(attempt.parent.name == EXPECTED_ASSET_REVISION, "checkpoint path asset revision drifted")
    require(attempt.parent.parent.name == EXPECTED_CODE_REVISION, "checkpoint path code revision drifted")
    require(attempt.parent.parent.parent.name == "runs", "checkpoint is outside the canonical campaign runs tree")
    require(attempt.parent.parent.parent.parent.name == "z1-vic-prototype", "checkpoint is outside the z1-vic-prototype campaign")
    require(attempt.parent.parent.parent.parent.parent.name == "campaigns", "checkpoint is outside a campaigns root")
    require(not is_within(checkpoint, repo_root), "checkpoint must be outside the code repository")
    return attempt, job


_SUMMARY_KEYS = {
    "mean",
    "std",
    "minimum",
    "p05",
    "median",
    "p95",
    "maximum",
    "lower_bound_occupancy",
    "upper_bound_occupancy",
}
_ROLLOUT_TELEMETRY_KEYS = {
    "schema_version",
    "source",
    "action_terms",
    "joint_names",
    "gain_action_indices",
    "raw_action_clip",
    "rollout_steps_per_env",
    "sample_count",
    "temporal_provenance",
    "deterministic_gaussian_mean",
    "sampled_action",
    "gaussian_exploration_std",
}


def validate_rollout_telemetry(value: object, *, expected_samples: int) -> dict[str, object]:
    telemetry = exact_keys(value, _ROLLOUT_TELEMETRY_KEYS, name="VIC rollout telemetry")
    require(telemetry["schema_version"] == 1, "VIC rollout telemetry schema version drifted")
    require(telemetry["source"] == "cat_rollout_storage", "VIC rollout telemetry source drifted")
    require(telemetry["action_terms"] == list(ACTION_TERMS), "VIC rollout telemetry action signature drifted")
    require(telemetry["joint_names"] == list(JOINT_NAMES), "VIC rollout telemetry joint order drifted")
    require(telemetry["gain_action_indices"] == list(range(6, 12)), "VIC rollout telemetry gain indices drifted")
    require(telemetry["raw_action_clip"] == 1.0, "VIC rollout telemetry raw clip drifted")
    require(telemetry["rollout_steps_per_env"] == 24, "VIC rollout telemetry rollout length drifted")
    require(telemetry["sample_count"] == expected_samples and expected_samples > 0, "VIC rollout telemetry sample count drifted")
    require(
        telemetry["temporal_provenance"]
        == {
            "rollout_generated_by": "pre_update_behavior_policy",
            "checkpoint_weights": "post_update",
        },
        "VIC rollout telemetry temporal provenance drifted",
    )
    for source in ("deterministic_gaussian_mean", "sampled_action"):
        forms = exact_keys(telemetry[source], {"raw", "clipped"}, name=source)
        for form in ("raw", "clipped"):
            summary = exact_keys(forms[form], _SUMMARY_KEYS, name=f"{source}.{form}")
            vectors: dict[str, list[float]] = {}
            for key in _SUMMARY_KEYS:
                occupancy = key.endswith("occupancy")
                vectors[key] = finite_vector(
                    summary[key],
                    name=f"{source}.{form}.{key}",
                    lower=0.0 if occupancy else None,
                    upper=1.0 if occupancy else None,
                )
            require(all(item >= 0.0 for item in vectors["std"]), f"{source}.{form}.std is negative")
            for ordered in zip(
                vectors["minimum"],
                vectors["p05"],
                vectors["median"],
                vectors["p95"],
                vectors["maximum"],
                strict=True,
            ):
                require(ordered[0] <= ordered[1] <= ordered[2] <= ordered[3] <= ordered[4], f"{source}.{form} quantiles are unordered")
            if form == "clipped":
                for key in ("mean", "minimum", "p05", "median", "p95", "maximum"):
                    finite_vector(summary[key], name=f"{source}.clipped.{key}", lower=-1.0, upper=1.0)
    finite_vector(telemetry["gaussian_exploration_std"], name="gaussian_exploration_std", positive=True)
    require_finite_json(telemetry, "VIC rollout telemetry")
    json.dumps(telemetry, allow_nan=False, sort_keys=True)
    return dict(telemetry)


def load_checkpoint_identity(torch: Any, checkpoint: Path, expected_sha256: str) -> tuple[dict[str, object], dict[str, object]]:
    require(_HEX64.fullmatch(expected_sha256) is not None, "expected checkpoint SHA-256 must be 64 lowercase hex")
    actual_sha256 = sha256_file(checkpoint)
    require(actual_sha256 == expected_sha256, "model_499.pt SHA-256 mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(state, Mapping), "model_499.pt must contain a mapping")

    tensors: list[Any] = []

    def visit(value: object) -> None:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(state)
    require(tensors, "model_499.pt contains no tensors")
    require(all(bool(torch.isfinite(item).all()) for item in tensors), "model_499.pt contains a non-finite tensor")
    require(type(state.get("iter")) is int and state["iter"] == 499, "model_499.pt iteration identity drifted")
    infos = state.get("infos")
    require(isinstance(infos, Mapping), "model_499.pt infos are missing")
    telemetry = validate_rollout_telemetry(
        infos.get("vic_rollout_telemetry"), expected_samples=24 * 4096
    )
    return (
        {
            "path": str(checkpoint),
            "basename": checkpoint.name,
            "sha256": actual_sha256,
            "iteration": 499,
            "finite_tensor_count": len(tensors),
        },
        telemetry,
    )


def load_model_0_identity(
    torch: Any, checkpoint: Path
) -> tuple[dict[str, object], dict[str, object]]:
    require(
        checkpoint.name == "model_0.pt"
        and checkpoint.is_file()
        and not checkpoint.is_symlink(),
        "model_0.pt is missing or is not a regular file",
    )
    actual_sha256 = sha256_file(checkpoint)
    require(
        actual_sha256 == EXPECTED_MODEL_0_SHA256,
        "model_0.pt SHA-256 mismatch",
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(state, Mapping), "model_0.pt must contain a mapping")

    tensors: list[Any] = []

    def visit(value: object) -> None:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(state)
    require(tensors, "model_0.pt contains no tensors")
    require(
        all(bool(torch.isfinite(item).all()) for item in tensors),
        "model_0.pt contains a non-finite tensor",
    )
    require(
        type(state.get("iter")) is int and state["iter"] == 0,
        "model_0.pt iteration identity drifted",
    )
    infos = state.get("infos")
    require(isinstance(infos, Mapping), "model_0.pt infos are missing")
    telemetry = validate_rollout_telemetry(
        infos.get("vic_rollout_telemetry"), expected_samples=24 * 4096
    )
    return (
        {
            "path": str(checkpoint),
            "basename": checkpoint.name,
            "sha256": actual_sha256,
            "iteration": 0,
            "finite_tensor_count": len(tensors),
        },
        telemetry,
    )


def validate_training_telemetry_payload(
    payload: object,
    model_0_telemetry: Mapping[str, object],
    checkpoint_telemetry: Mapping[str, object],
) -> dict[str, object]:
    artifact = exact_keys(
        payload,
        {
            "schema_version",
            "task",
            "training_seed",
            "checkpoint_iterations",
            "vic_rollout_telemetry",
            "final_gain_classification",
            "r_imit_curriculum",
            "impossible_success",
        },
        name="canonical telemetry artifact",
    )
    require_finite_json(artifact, "canonical telemetry artifact")
    require(artifact["schema_version"] == 1, "canonical telemetry artifact schema drifted")
    require(artifact["task"] == VIC_TASK, "canonical telemetry artifact task drifted")
    require(artifact["training_seed"] == TRAINING_SEED, "canonical telemetry artifact seed drifted")
    require(
        artifact["checkpoint_iterations"]
        == [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499],
        "canonical telemetry checkpoint schedule drifted",
    )
    records = exact_keys(
        artifact["vic_rollout_telemetry"],
        {"model_0", "model_499"},
        name="canonical telemetry checkpoint records",
    )
    model_0 = validate_rollout_telemetry(
        records["model_0"], expected_samples=24 * 4096
    )
    model_499 = validate_rollout_telemetry(
        records["model_499"], expected_samples=24 * 4096
    )
    require(
        model_0 == model_0_telemetry,
        "canonical telemetry model_0 record differs from model_0.pt infos",
    )
    require(
        model_499 == checkpoint_telemetry,
        "canonical telemetry model_499 record differs from checkpoint infos",
    )

    clipped = model_499["deterministic_gaussian_mean"]["clipped"]
    means = finite_vector(clipped["mean"], name="final deterministic gain mean", lower=-1.0, upper=1.0)
    stds = finite_vector(clipped["std"], name="final deterministic gain std", lower=0.0)
    lower_occupancy = finite_vector(
        clipped["lower_bound_occupancy"],
        name="final deterministic lower occupancy",
        lower=0.0,
        upper=1.0,
    )
    upper_occupancy = finite_vector(
        clipped["upper_bound_occupancy"],
        name="final deterministic upper occupancy",
        lower=0.0,
        upper=1.0,
    )
    all_zero = all(item == 0.0 for item in means) and all(
        item == 0.0 for item in stds
    )
    nonzero_spread = any(item > 0.0 for item in stds)
    bound_pinned_joints = [
        joint
        for joint, lower, upper in zip(
            JOINT_NAMES, lower_occupancy, upper_occupancy, strict=True
        )
        if lower == 1.0 or upper == 1.0
    ]
    recomputed_classification = {
        "all_zero": all_zero,
        "nonzero_spread": nonzero_spread,
        "bound_pinned_joints": bound_pinned_joints,
    }
    require(
        artifact["final_gain_classification"] == recomputed_classification,
        "canonical final gain classification was not independently recomputable",
    )
    require(not all_zero, "final deterministic clipped gain telemetry is all zero")
    require(nonzero_spread, "final deterministic clipped gain telemetry has no spread")
    require(
        not bound_pinned_joints,
        f"final deterministic clipped gain telemetry is bound pinned: {bound_pinned_joints}",
    )

    curriculum = exact_keys(
        artifact["r_imit_curriculum"],
        {"scalar", "observed", "final_weight"},
        name="canonical r_imit curriculum telemetry",
    )
    require(
        curriculum["scalar"] == "Curriculum/r_imit_anneal/weight",
        "canonical r_imit curriculum scalar drifted",
    )
    require(curriculum["final_weight"] == 0.0, "canonical final r_imit weight is not zero")
    curriculum_events = curriculum["observed"]
    require(
        isinstance(curriculum_events, list) and len(curriculum_events) == 500,
        "canonical r_imit curriculum must contain all 500 iterations",
    )
    curriculum_rows: list[Mapping[str, object]] = []
    for index, event in enumerate(curriculum_events):
        row = exact_keys(event, {"iteration", "value"}, name=f"r_imit event {index}")
        require(
            type(row["iteration"]) is int and row["iteration"] == index,
            f"r_imit event {index} iteration drifted",
        )
        finite_number(row["value"], name=f"r_imit event {index} value")
        curriculum_rows.append(row)
    stages = (0.10, 0.08, 0.06, 0.04, 0.02, 0.00)
    anchors = {0: 0.10, 50: 0.08, 100: 0.06, 150: 0.04, 200: 0.02, 250: 0.00, 499: 0.00}
    boundaries = {
        49: (0.08, 0.10),
        99: (0.06, 0.08),
        149: (0.04, 0.06),
        199: (0.02, 0.04),
        249: (0.00, 0.02),
    }
    tolerance = 1e-6
    for iteration, expected in anchors.items():
        require(
            abs(float(curriculum_rows[iteration]["value"]) - expected) <= tolerance,
            f"canonical r_imit curriculum anchor drifted at iteration {iteration}",
        )
    for iteration, row in enumerate(curriculum_rows):
        weight = float(row["value"])
        if iteration in boundaries:
            lower, upper = boundaries[iteration]
            require(
                lower - tolerance <= weight <= upper + tolerance,
                f"canonical r_imit curriculum boundary drifted at iteration {iteration}",
            )
        else:
            expected = stages[min(iteration // 50, 5)]
            require(
                abs(weight - expected) <= tolerance,
                f"canonical r_imit curriculum stage drifted at iteration {iteration}",
            )
    for stage in stages:
        require(
            any(abs(float(row["value"]) - stage) <= tolerance for row in curriculum_rows),
            f"canonical r_imit curriculum plateau is missing: {stage}",
        )
    require(
        curriculum_events[-1]["value"] == 0.0,
        "last observed r_imit curriculum weight is not zero",
    )

    impossible = exact_keys(
        artifact["impossible_success"],
        {"scalar", "observed", "all_zero"},
        name="canonical impossible_success telemetry",
    )
    require(
        impossible["scalar"] == "Episode_Metrics/impossible_success",
        "canonical impossible_success scalar drifted",
    )
    events = impossible["observed"]
    require(
        isinstance(events, list) and len(events) == 500,
        "canonical impossible_success must contain all 500 iterations",
    )
    for index, event in enumerate(events):
        row = exact_keys(
            event, {"iteration", "value"}, name=f"impossible_success event {index}"
        )
        require(
            type(row["iteration"]) is int and row["iteration"] == index,
            f"impossible_success event {index} iteration drifted",
        )
        require(
            finite_number(row["value"], name=f"impossible_success event {index} value")
            == 0.0,
            "impossible_success must be identically zero",
        )
    require(
        impossible["all_zero"] is True,
        "canonical impossible_success all-zero classification drifted",
    )
    return {
        "model_0_matches_checkpoint": True,
        "model_499_matches_checkpoint": True,
        "final_gain_classification": recomputed_classification,
        "training_gain_gate_pass": True,
        "r_imit_final_weight_zero": True,
        "impossible_success_gate_pass": True,
    }


def load_consolidated_telemetry(
    attempt: Path,
    model_0_identity: Mapping[str, object],
    model_0_telemetry: Mapping[str, object],
    checkpoint_telemetry: Mapping[str, object],
) -> dict[str, object]:
    path = attempt / TELEMETRY_BASENAME
    require(
        path.is_file() and not path.is_symlink(),
        f"required telemetry artifact is missing: {path}",
    )
    actual_sha256 = sha256_file(path)
    require(
        actual_sha256 == EXPECTED_TELEMETRY_SHA256,
        "canonical telemetry artifact SHA-256 mismatch",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read canonical telemetry artifact: {path}") from exc
    validation = validate_training_telemetry_payload(
        payload, model_0_telemetry, checkpoint_telemetry
    )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "payload_schema_version": 1,
        "model_0_checkpoint": dict(model_0_identity),
        **validation,
    }


class VectorStats:
    """GPU-resident streaming six-vector statistics with deferred checks."""

    def __init__(self, torch: Any, template: Any, *, name: str, lower: Any = None, upper: Any = None):
        self.torch = torch
        self.name = name
        self.count = 0
        self.minimum = torch.full((6,), torch.inf, dtype=torch.float64, device=template.device)
        self.maximum = torch.full((6,), -torch.inf, dtype=torch.float64, device=template.device)
        self.total = torch.zeros(6, dtype=torch.float64, device=template.device)
        self.total_sq = torch.zeros(6, dtype=torch.float64, device=template.device)
        self.finite = torch.ones((), dtype=torch.bool, device=template.device)
        self.in_bounds = torch.ones((), dtype=torch.bool, device=template.device)
        self.lower = None if lower is None else torch.as_tensor(lower, dtype=template.dtype, device=template.device)
        self.upper = None if upper is None else torch.as_tensor(upper, dtype=template.dtype, device=template.device)

    def update(self, values: Any) -> None:
        require(values.ndim == 2 and values.shape[1] == 6, f"{self.name} must have shape (N, 6)")
        if values.shape[0] == 0:
            return
        self.count += int(values.shape[0])
        self.finite &= self.torch.isfinite(values).all()
        if self.lower is not None:
            self.in_bounds &= (values >= self.lower).all()
        if self.upper is not None:
            self.in_bounds &= (values <= self.upper).all()
        work = values.to(dtype=self.torch.float64)
        self.minimum = self.torch.minimum(self.minimum, work.amin(dim=0))
        self.maximum = self.torch.maximum(self.maximum, work.amax(dim=0))
        self.total += work.sum(dim=0)
        self.total_sq += work.square().sum(dim=0)

    def snapshot(self) -> dict[str, object]:
        require(self.count > 0, f"{self.name} has no samples")
        require(bool(self.finite.item()), f"{self.name} contains a non-finite value")
        require(bool(self.in_bounds.item()), f"{self.name} violates its bound")
        mean = self.total / self.count
        variance = (self.total_sq / self.count - mean.square()).clamp_min(0.0)

        def values(tensor: Any) -> list[float]:
            result = [float(item) for item in tensor.detach().cpu().tolist()]
            require(len(result) == 6 and all(math.isfinite(item) for item in result), f"{self.name} summary is non-finite")
            return result

        return {
            "sample_count": self.count,
            "mean": values(mean),
            "std": values(self.torch.sqrt(variance)),
            "minimum": values(self.minimum),
            "maximum": values(self.maximum),
        }


class VicRuntimeCapture:
    """Capture and validate the first episode's live VIC channels."""

    def __init__(self, torch: Any, env: Any, *, clip_actions: float):
        self.torch = torch
        self.env = env
        require(clip_actions == 1.0, "runtime wrapper requires exact action clip 1.0")
        require(tuple(env.action_manager.active_terms) == ACTION_TERMS, "live VIC action signature drifted")
        require(list(env.action_manager.action_term_dim) == [6, 6], "live VIC action dimensions drifted")
        require(env.action_manager.total_action_dim == ACTION_WIDTH, "live VIC action width drifted")
        self.position = env.action_manager.get_term("joint_position")
        self.stiffness = env.action_manager.get_term("joint_stiffness")
        telemetry = self.stiffness.telemetry
        require(telemetry.mapping_family == "author_v1_exponential", "live VIC mapping family drifted")
        require(telemetry.C == C, "live VIC C drifted")
        require(telemetry.p_bounds == (-1.0, 1.0), "live VIC p bounds drifted")
        require(tuple(telemetry.joint_names) == JOINT_NAMES, "live VIC joint order drifted")
        self.control_ids = tuple(int(item) for item in telemetry.control_ids.detach().cpu().tolist())
        require(self.control_ids == CONTROL_IDS, "live VIC native control IDs drifted")
        require(tuple(float(item) for item in telemetry.nominal_kp.detach().cpu().tolist()) == NOMINAL_KP, "live VIC nominal Kp drifted")
        require(tuple(float(item) for item in telemetry.nominal_kd.detach().cpu().tolist()) == NOMINAL_KD, "live VIC nominal Kd drifted")
        require(tuple(getattr(self.position, "target_names", ())) == JOINT_NAMES, "live position target order drifted")
        self.model = env.sim.model
        self.active = torch.ones(NUM_ENVS, dtype=torch.bool, device=env.device)
        self.control_counts = torch.zeros(NUM_ENVS, dtype=torch.long, device=env.device)
        self.substep_counts = torch.zeros(NUM_ENVS, dtype=torch.long, device=env.device)
        self.pending_executed = None
        template = telemetry.p
        self.stats = {
            "policy_mean_raw_position": VectorStats(torch, template, name="raw position action"),
            "policy_mean_raw_stiffness": VectorStats(torch, template, name="raw stiffness action"),
            "executed_position_action": VectorStats(torch, template, name="executed position action", lower=-1.0, upper=1.0),
            "executed_stiffness_action": VectorStats(torch, template, name="executed stiffness action", lower=-1.0, upper=1.0),
            "stiffness_p": VectorStats(torch, template, name="stiffness p", lower=-1.0, upper=1.0),
            "gain_multiplier": VectorStats(torch, template, name="gain multiplier", lower=1.0 / C, upper=C),
            "kp": VectorStats(torch, template, name="Kp", lower=[item / C for item in NOMINAL_KP], upper=[item * C for item in NOMINAL_KP]),
            "kd": VectorStats(torch, template, name="Kd", lower=[item / math.sqrt(C) for item in NOMINAL_KD], upper=[item * math.sqrt(C) for item in NOMINAL_KD]),
            "actuator_force_n_m": VectorStats(torch, template, name="actuator force", lower=[item[0] for item in FORCE_RANGE_N_M], upper=[item[1] for item in FORCE_RANGE_N_M]),
        }
        self.flags = {
            name: torch.ones((), dtype=torch.bool, device=env.device)
            for name in (
                "manager_action_matches_clipped_policy_mean",
                "stiffness_raw_matches_executed_action",
                "p_matches_executed_action",
                "gain_map_exact",
                "native_gain_fields_match",
                "native_force_within_range",
            )
        }
        self.force_range_initial = self.model.actuator_forcerange.clone()
        self.force_limited_initial = self.model.actuator_forcelimited.clone()
        all_ids = set(range(int(self.model.nu)))
        self.non_target_ids = tuple(sorted(all_ids - set(self.control_ids)))
        self.non_target_gain_initial = self.model.actuator_gainprm[:, self.non_target_ids].clone()
        self.non_target_bias_initial = self.model.actuator_biasprm[:, self.non_target_ids].clone()
        self.live_metadata: dict[str, object] | None = None
        self.started = False
        self.original_compute_substep = env.metrics_manager.compute_substep

        def capture_substep() -> None:
            self.original_compute_substep()
            self._capture_substep()

        self.capture_substep = capture_substep
        env.metrics_manager.compute_substep = capture_substep

    def start_population(self) -> None:
        require(not self.started, "evaluation population reset more than once")
        self.started = True

    def set_live_metadata(self, metadata: Mapping[str, object]) -> None:
        require(self.live_metadata is None, "live VIC metadata captured more than once")
        self.live_metadata = dict(metadata)

    def before_step(self, raw_action: Any) -> None:
        require(self.started, "runtime capture was not started at the population boundary")
        require(isinstance(raw_action, self.torch.Tensor), "mean policy output must be a tensor")
        require(tuple(raw_action.shape) == (NUM_ENVS, ACTION_WIDTH), "mean policy output must have shape (64, 12)")
        require(raw_action.device == self.active.device, "mean policy output device drifted")
        active = self.active
        self.stats["policy_mean_raw_position"].update(raw_action[active, :6])
        self.stats["policy_mean_raw_stiffness"].update(raw_action[active, 6:])
        executed = raw_action.clamp(-1.0, 1.0)
        self.stats["executed_position_action"].update(executed[active, :6])
        self.stats["executed_stiffness_action"].update(executed[active, 6:])
        self.control_counts[active] += 1
        self.pending_executed = executed

    def after_step(self) -> None:
        require(self.pending_executed is not None, "runtime step has no pending executed action")
        done = self.env.reset_terminated | self.env.reset_time_outs
        require(done.shape == (NUM_ENVS,) and done.dtype == self.torch.bool, "live done mask drifted")
        self.active &= ~done
        self.pending_executed = None

    def _ordered_force_range(self) -> Any:
        ranges = self.model.actuator_forcerange
        ids = list(self.control_ids)
        if ranges.ndim == 2:
            ordered = ranges[ids].unsqueeze(0).expand(NUM_ENVS, -1, -1)
        elif ranges.ndim == 3:
            ordered = ranges[:, ids, :]
        else:
            raise EvaluationError("live actuator force-range rank drifted")
        require(tuple(ordered.shape) == (NUM_ENVS, 6, 2), "live actuator force-range shape drifted")
        return ordered

    def _capture_substep(self) -> None:
        require(self.pending_executed is not None, "substep capture occurred outside a policy step")
        telemetry = self.stiffness.telemetry
        active = self.active
        p = telemetry.p
        multiplier = telemetry.multiplier
        kp = telemetry.kp
        kd = telemetry.kd
        force = self.env.sim.data.actuator_force[:, list(self.control_ids)]
        require(tuple(p.shape) == (NUM_ENVS, 6), "live stiffness p shape drifted")
        require(tuple(force.shape) == (NUM_ENVS, 6), "live actuator-force shape drifted")
        self.stats["stiffness_p"].update(p[active])
        self.stats["gain_multiplier"].update(multiplier[active])
        self.stats["kp"].update(kp[active])
        self.stats["kd"].update(kd[active])
        self.stats["actuator_force_n_m"].update(force[active])
        self.substep_counts[active] += 1
        executed = self.pending_executed
        manager_action = self.env.action_manager.action
        expected_multiplier = self.torch.pow(p.new_tensor(C), p)
        expected_kp = expected_multiplier * p.new_tensor(NOMINAL_KP)
        expected_kd = self.torch.sqrt(expected_multiplier) * p.new_tensor(NOMINAL_KD)
        control_ids = list(self.control_ids)
        gainprm = self.model.actuator_gainprm[:, control_ids, 0]
        bias_kp = self.model.actuator_biasprm[:, control_ids, 1]
        bias_kd = self.model.actuator_biasprm[:, control_ids, 2]
        force_range = self._ordered_force_range()
        self.flags["manager_action_matches_clipped_policy_mean"] &= (manager_action == executed).all()
        self.flags["stiffness_raw_matches_executed_action"] &= (self.stiffness.raw_action == executed[:, 6:]).all()
        self.flags["p_matches_executed_action"] &= (p == executed[:, 6:]).all()
        self.flags["gain_map_exact"] &= (
            self.torch.isclose(multiplier, expected_multiplier, rtol=1.0e-6, atol=1.0e-7).all()
            & self.torch.isclose(kp, expected_kp, rtol=1.0e-6, atol=1.0e-5).all()
            & self.torch.isclose(kd, expected_kd, rtol=1.0e-6, atol=1.0e-6).all()
        )
        self.flags["native_gain_fields_match"] &= (
            (gainprm == kp).all() & (bias_kp == -kp).all() & (bias_kd == -kd).all()
        )
        self.flags["native_force_within_range"] &= (
            self.torch.isfinite(force).all()
            & (force >= force_range[:, :, 0]).all()
            & (force <= force_range[:, :, 1]).all()
        )

    def restore_hook(self) -> None:
        self.env.metrics_manager.compute_substep = self.original_compute_substep

    def snapshot(self, episodes: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
        require(len(episodes) == NUM_ENVS, "runtime capture requires 64 completed episodes")
        require(not bool(self.active.any().item()), "runtime capture still has unfinished worlds")
        episode_steps = [int(row["episode_steps"]) for row in episodes]
        control_counts = [int(item) for item in self.control_counts.detach().cpu().tolist()]
        substep_counts = [int(item) for item in self.substep_counts.detach().cpu().tolist()]
        require(control_counts == episode_steps, "runtime action sample counts differ from episode steps")
        require(substep_counts == [item * CONTROL_DECIMATION for item in episode_steps], "runtime gain/force sample counts are not 500-Hz complete")
        flags = {name: bool(value.item()) for name, value in self.flags.items()}
        require(all(flags.values()), f"live VIC runtime invariant failed: {flags}")
        require(self.live_metadata is not None, "live VIC runner metadata was not captured")
        require(self.torch.equal(self.force_range_initial, self.model.actuator_forcerange), "native actuator force ranges changed during evaluation")
        require(self.torch.equal(self.force_limited_initial, self.model.actuator_forcelimited), "native actuator force-limit flags changed during evaluation")
        require(self.torch.equal(self.non_target_gain_initial, self.model.actuator_gainprm[:, self.non_target_ids]), "non-target native gains changed during evaluation")
        require(self.torch.equal(self.non_target_bias_initial, self.model.actuator_biasprm[:, self.non_target_ids]), "non-target native biases changed during evaluation")
        ranges = self._ordered_force_range()[0].detach().cpu().tolist()
        require(tuple(tuple(float(item) for item in row) for row in ranges) == FORCE_RANGE_N_M, "canonical per-joint force ranges drifted")
        stats = {name: value.snapshot() for name, value in self.stats.items()}
        return (
            {
                "schema_version": 1,
                "source": "first_episode_runtime_capture",
                "action_terms": list(ACTION_TERMS),
                "action_width": ACTION_WIDTH,
                "joint_names": list(JOINT_NAMES),
                "control_ids": list(self.control_ids),
                "mapping_family": "author_v1_exponential",
                "C": C,
                "executed_action_bounds": [-1.0, 1.0],
                "nominal_kp": list(NOMINAL_KP),
                "nominal_kd": list(NOMINAL_KD),
                "force_range_n_m": [list(row) for row in FORCE_RANGE_N_M],
                "control_sample_count": sum(control_counts),
                "physics_substep_sample_count": sum(substep_counts),
                "per_env_control_sample_count": control_counts,
                "per_env_physics_substep_sample_count": substep_counts,
                "statistics": stats,
                "runtime_invariants": flags,
                "force_limits_unchanged": True,
                "non_target_native_fields_unchanged": True,
            },
            dict(self.live_metadata),
        )


def validate_episode_outcomes(episodes: object, summary: object) -> dict[str, object]:
    require(isinstance(episodes, list) and len(episodes) == NUM_ENVS, "evaluation must contain exactly 64 episodes")
    require(isinstance(summary, Mapping), "evaluation summary must be a mapping")
    success = 0
    productive = 0
    qvel_legal = 0
    impulse_legal = 0
    expected_ids = list(range(NUM_ENVS))
    actual_ids: list[int] = []
    for index, row in enumerate(episodes):
        require(isinstance(row, Mapping), f"episode {index} must be a mapping")
        env_id = row.get("env_id")
        require(type(env_id) is int, f"episode {index} env_id must be an integer")
        actual_ids.append(env_id)
        steps = row.get("episode_steps")
        require(type(steps) is int and 1 <= steps <= MAX_CONTROL_STEPS, f"episode {index} step count drifted")
        require(type(row.get("success")) is bool, f"episode {index} success must be bool")
        require(type(row.get("first_strike_productive")) is bool, f"episode {index} productivity must be bool")
        success += int(row["success"])
        productive += int(row["first_strike_productive"])
        for key in (
            "precontact_velocity_m_s",
            "first_event_impulse_n_s",
            "nail_depth_m",
            "joint_target_rmse_rad",
            "joint_target_error_max_rad",
            "reference_error_mean_m",
            "reference_error_max_m",
        ):
            finite_number(row.get(key), name=f"episode {index}.{key}")
        reference_samples = row.get("reference_error_samples")
        require(type(reference_samples) is int and 1 <= reference_samples <= steps, f"episode {index} reference sample count drifted")
        qvel = finite_vector(row.get("joint_velocity_peak_rad_s"), name=f"episode {index}.qvel", lower=0.0)
        qvel_ok = all(item <= JOINT_VELOCITY_LIMIT_RAD_S for item in qvel)
        require(row.get("joint_velocity_legal") is qvel_ok, f"episode {index} qvel legality flag drifted")
        qvel_legal += int(qvel_ok)
        peaks = finite_vector(row.get("joint_impulse_peak_n_m_s"), name=f"episode {index}.joint_impulse", lower=0.0)
        utilization = finite_vector(row.get("joint_impulse_utilization"), name=f"episode {index}.joint_impulse_utilization", lower=0.0)
        expected_utilization = [item / cap for item, cap in zip(peaks, JOINT_IMPULSE_CAP_N_M_S, strict=True)]
        require(utilization == expected_utilization, f"episode {index} impulse utilization drifted")
        impulse_legal += int(all(item <= cap for item, cap in zip(peaks, JOINT_IMPULSE_CAP_N_M_S, strict=True)))
    require(actual_ids == expected_ids, "episodes are not in exact env_id order 0..63")
    expected_success_rate = success / NUM_ENVS
    expected_productive_rate = productive / NUM_ENVS
    require(summary.get("task_success_n") == success, "summary success count was not recomputed exactly")
    require(summary.get("task_success_rate") == expected_success_rate, "summary success rate was not recomputed exactly")
    require(summary.get("productive_first_strike_n") == productive, "summary productive count was not recomputed exactly")
    require(summary.get("productive_first_strike_rate") == expected_productive_rate, "summary productive rate was not recomputed exactly")
    qvel_summary = summary.get("joint_velocity_peak_rad_s")
    impulse_summary = summary.get("joint_impulse_utilization")
    require(isinstance(qvel_summary, Mapping), "summary qvel block is missing")
    require(isinstance(impulse_summary, Mapping), "summary impulse block is missing")
    require(qvel_summary.get("all_joints_legal_n") == qvel_legal, "summary qvel legal count drifted")
    require(qvel_summary.get("all_joints_legal_rate") == qvel_legal / NUM_ENVS, "summary qvel legal rate drifted")
    require(impulse_summary.get("all_joints_at_or_below_cap_n") == impulse_legal, "summary impulse legal count drifted")
    require(impulse_summary.get("all_joints_at_or_below_cap_rate") == impulse_legal / NUM_ENVS, "summary impulse legal rate drifted")
    return {
        "task_success_n_recomputed": success,
        "productive_first_strike_n_recomputed": productive,
        "qvel_500hz_all_joints_legal_n_recomputed": qvel_legal,
        "joint_impulse_all_joints_at_or_below_cap_n_recomputed": impulse_legal,
        "success_gate_at_least_58_of_64": success >= 58,
        "productive_gate_at_least_58_of_64": productive >= 58,
        "qvel_gate_64_of_64": qvel_legal == NUM_ENVS,
        "impulse_gate_64_of_64": impulse_legal == NUM_ENVS,
    }


def validate_protocol_payload(payload: Mapping[str, object], *, pilot: Any, expected_checkpoint_sha256: str) -> dict[str, object]:
    require(payload.get("schema_version") == 3, "VIC evaluation schema version drifted")
    require(payload.get("task") == VIC_TASK, "VIC evaluation task drifted")
    require(payload.get("training_seed") == TRAINING_SEED, "VIC evaluation training seed drifted")
    require(
        payload.get("protocol")
        == {
            "evaluation_seed": EVALUATION_SEED,
            "num_envs": NUM_ENVS,
            "episodes_per_env": 1,
            "episode_length_s": EPISODE_LENGTH_S,
            "auto_reset": False,
            "policy_mode": "mean",
            "physics_dt_s": PHYSICS_DT_S,
            "physics_sample_hz": 500,
            "control_decimation": CONTROL_DECIMATION,
            "max_control_steps": MAX_CONTROL_STEPS,
        },
        "VIC evaluation protocol drifted",
    )
    checkpoint = payload.get("checkpoint")
    require(isinstance(checkpoint, Mapping) and checkpoint.get("sha256") == expected_checkpoint_sha256, "VIC evaluation checkpoint identity drifted")
    require(payload.get("code_git") == {"revision": EXPECTED_CODE_REVISION, "status": ""}, "VIC evaluation code identity drifted")
    require(payload.get("asset_git") == {"revision": EXPECTED_ASSET_REVISION, "status": ""}, "VIC evaluation asset identity drifted")
    require(
        payload.get("initial_population_sha256")
        == EXPECTED_INITIAL_POPULATION_SHA256,
        "VIC evaluation initial population drifted from the banked direct-reference population",
    )
    contract = payload.get("treatment_contract")
    require(isinstance(contract, Mapping), "VIC evaluation treatment contract is missing")
    require(contract.get("task") == VIC_TASK, "VIC treatment task drifted")
    require(contract.get("action_terms") == list(ACTION_TERMS), "VIC treatment action terms drifted")
    require(contract.get("action_term_dims") == [6, 6], "VIC treatment action dimensions drifted")
    require(contract.get("action_dim") == ACTION_WIDTH, "VIC treatment action width drifted")
    require(contract.get("observation_width") == OBSERVATION_WIDTH, "VIC treatment observation width drifted")
    variable = contract.get("variable_impedance")
    require(isinstance(variable, Mapping) and variable.get("C") == C, "VIC treatment C drifted")
    require(variable.get("joint_names") == list(JOINT_NAMES), "VIC treatment gain-joint order drifted")
    require(variable.get("p_bounds") == [-1.0, 1.0], "VIC treatment p bounds drifted")
    live = payload.get("live_runner_metadata")
    require(isinstance(live, Mapping), "live VIC runner metadata is missing")
    require(live.get("action_type") == "joint_position_variable_impedance", "live VIC action type drifted")
    require(live.get("action_terms") == list(ACTION_TERMS), "live VIC action signature drifted")
    require(live.get("action_term_dims") == [6, 6] and live.get("action_dim") == ACTION_WIDTH, "live VIC action width drifted")
    widths = live.get("observation_widths")
    require(widths == {"actor": OBSERVATION_WIDTH, "critic": OBSERVATION_WIDTH}, "live VIC observation widths drifted")
    live_variable = live.get("variable_impedance")
    require(isinstance(live_variable, Mapping) and live_variable.get("C") == C, "live VIC metadata C drifted")
    require(live_variable.get("control_ids") == list(CONTROL_IDS), "live VIC metadata control IDs drifted")
    runtime = payload.get("vic_runtime_telemetry")
    require(isinstance(runtime, Mapping), "VIC runtime telemetry is missing")
    require(runtime.get("action_terms") == list(ACTION_TERMS) and runtime.get("action_width") == ACTION_WIDTH, "runtime action signature drifted")
    invariants = runtime.get("runtime_invariants")
    require(isinstance(invariants, Mapping) and invariants and all(value is True for value in invariants.values()), "runtime action/gain/force invariant failed")
    require(runtime.get("force_limits_unchanged") is True, "runtime force limits changed")
    require(runtime.get("non_target_native_fields_unchanged") is True, "runtime non-target fields changed")
    require_finite_json(runtime, "VIC runtime telemetry")
    training_artifact = payload.get("canonical_training_telemetry_artifact")
    require(
        isinstance(training_artifact, Mapping),
        "canonical training telemetry artifact identity is missing",
    )
    require(
        training_artifact.get("model_0_matches_checkpoint") is True
        and training_artifact.get("model_499_matches_checkpoint") is True,
        "canonical training checkpoint telemetry validation drifted",
    )
    require(
        training_artifact.get("training_gain_gate_pass") is True,
        "canonical training gain gate drifted",
    )
    require(
        training_artifact.get("r_imit_final_weight_zero") is True,
        "canonical training curriculum gate drifted",
    )
    require(
        training_artifact.get("impossible_success_gate_pass") is True,
        "canonical impossible_success gate drifted",
    )
    outcomes = validate_episode_outcomes(payload.get("episodes"), payload.get("summary"))
    recomputed_summary = pilot._summary(payload["episodes"])
    require(recomputed_summary == payload["summary"], "full evaluator summary differs on independent recomputation")
    recorded = payload.get("validation")
    require(isinstance(recorded, Mapping), "VIC evaluation validation block is missing")
    for key, value in outcomes.items():
        require(recorded.get(key) == value, f"recorded validation drifted for {key}")
    require(recorded.get("all_runtime_invariants") is True, "recorded runtime invariant gate drifted")
    require(
        recorded.get("training_gain_gate_pass")
        is training_artifact["training_gain_gate_pass"],
        "recorded training gain gate drifted",
    )
    require(
        recorded.get("impossible_success_gate_pass")
        is training_artifact["impossible_success_gate_pass"],
        "recorded impossible_success gate drifted",
    )
    expected_pass = all(
        outcomes[key]
        for key in (
            "success_gate_at_least_58_of_64",
            "productive_gate_at_least_58_of_64",
            "qvel_gate_64_of_64",
            "impulse_gate_64_of_64",
        )
    )
    require(recorded.get("canary_behavior_gate_pass") is expected_pass, "recorded canary behavior gate drifted")
    require(
        recorded.get("canary_all_gates_pass") is expected_pass,
        "recorded overall canary gate drifted",
    )
    require_finite_json(payload, "VIC evaluation payload")
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return outcomes


def build_vic_validator(pilot: Any, variable_module: Any, observation_module: Any) -> Any:
    original = pilot.validate_fic_contract

    def validate(task: str, env_cfg: Any, agent_cfg: Any) -> dict[str, object]:
        require(task == VIC_TASK, f"unsupported VIC canary task: {task}")
        require(tuple(env_cfg.actions) == ACTION_TERMS, "VIC config action signature drifted")
        stiffness = env_cfg.actions["joint_stiffness"]
        require(type(stiffness) is variable_module.JointStiffnessActionCfg, "VIC config stiffness action type drifted")
        require(stiffness.entity_name == "robot", "VIC config stiffness entity drifted")
        require(tuple(stiffness.joint_names) == JOINT_NAMES, "VIC config stiffness joint order drifted")
        require(stiffness.C == C and stiffness.clip is None, "VIC config gain envelope drifted")
        require(tuple(env_cfg.events) == ("reset_robot_joints", "reset_nail", "expand_variable_impedance_model_fields"), "VIC config event signature drifted")
        startup = env_cfg.events["expand_variable_impedance_model_fields"]
        require(startup.func is variable_module.expand_variable_impedance_model_fields, "VIC startup expansion implementation drifted")
        require(startup.mode == "startup" and startup.params == {}, "VIC startup expansion config drifted")
        require(startup.interval_range_s is None and startup.is_global_time is False and startup.min_step_count_between_reset == 0, "VIC startup expansion scheduling drifted")
        for group in ("actor", "critic"):
            action_obs = env_cfg.observations[group].terms["actions"]
            require(action_obs.func is observation_module.last_action, f"{group} action observation implementation drifted")
            require(action_obs.params == {"action_name": "joint_position"}, f"{group} action observation source drifted")
        action_rate = env_cfg.rewards["action_rate"]
        require(action_rate.params == {"action_name": "joint_position"}, "VIC action-rate source drifted")

        stripped = copy.deepcopy(env_cfg)
        stripped.actions.pop("joint_stiffness")
        stripped.events.pop("expand_variable_impedance_model_fields")
        for group in ("actor", "critic"):
            stripped.observations[group].terms["actions"].params = {}
        stripped.rewards["action_rate"].params = {}
        base = original(FIC_TT_TASK, stripped, agent_cfg)
        require(base["r_tt_enabled"] is True and base["r_tt_k_tt"] == 1.0, "VIC base treatment is not exact FIC-TT")
        base["task"] = VIC_TASK
        base.pop("action_term")
        base.update(
            {
                "action_terms": list(ACTION_TERMS),
                "action_term_dims": [6, 6],
                "action_dim": ACTION_WIDTH,
                "action_observation_source": "joint_position",
                "action_rate_source": "joint_position",
                "variable_impedance": {
                    "mapping_family": "author_v1_exponential",
                    "C": C,
                    "p_bounds": [-1.0, 1.0],
                    "joint_names": list(JOINT_NAMES),
                    "native_model_fields": ["actuator_gainprm", "actuator_biasprm"],
                    "nominal_kp": list(NOMINAL_KP),
                    "nominal_kd": list(NOMINAL_KD),
                },
            }
        )
        return base

    return validate


def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, object], Path]:
    require(sys.flags.optimize == 0, "optimized Python is forbidden")
    require(os.environ.get("PYTHONOPTIMIZE", "") == "", "PYTHONOPTIMIZE must be empty")
    require(not args.repo_root.is_symlink(), "repo root must not be a symlink")
    require(not args.checkpoint.is_symlink(), "checkpoint must not be a symlink")
    repo_root = args.repo_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.absolute()
    require(args.repo_root.is_absolute(), "repo root must be an absolute path")
    require(args.checkpoint.is_absolute(), "checkpoint must be an absolute path")
    require(args.output.is_absolute(), "output must be an absolute path")
    require(repo_root.is_dir(), "repo root must be a directory")
    require(output.name == OUTPUT_BASENAME, f"output basename must be {OUTPUT_BASENAME}")
    require(output.parent.is_dir(), "output parent must already exist")
    require(not output.exists() and not output.is_symlink(), "refusing to overwrite evaluator output")
    code_identity = git_identity(repo_root, expected_revision=EXPECTED_CODE_REVISION)
    evaluator_path = repo_root / "evaluation/joint_position/evaluate_fic_pilot.py"
    require(evaluator_path.is_file() and not evaluator_path.is_symlink(), "banked FIC evaluator is missing")
    require(sha256_file(evaluator_path) == EXPECTED_FIC_EVALUATOR_SHA256, "banked FIC evaluator bytes drifted")
    attempt, job = validate_checkpoint_layout(checkpoint, repo_root)
    require(output.parent.resolve() == attempt.resolve(), "evaluation output must be written directly into the canary attempt")

    sys.path.insert(0, str(repo_root))
    import torch

    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.device_count() == 1, "evaluation requires exactly one visible CUDA device")
    gpu = torch.cuda.get_device_name(0)
    require(gpu == EXPECTED_GPU, f"unexpected evaluation GPU: {gpu}")
    packages = {name: package_version(name) for name in EXPECTED_PACKAGES}
    require(packages == EXPECTED_PACKAGES, f"runtime package versions drifted: {packages}")

    from evaluation.joint_position import evaluate_fic_pilot as pilot
    from mjlab.envs.mdp import observations as observation_module
    import src.tasks.hammer.config.z1  # noqa: F401
    import src.tasks.hammer.mdp.variable_impedance as variable_module
    import src.tasks.hammer.rl.runner as runner_module
    from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML

    require(Path(pilot.__file__).resolve() == evaluator_path.resolve(), "imported FIC evaluator path drifted")
    require(pilot.SEED == EVALUATION_SEED, "banked evaluator seed drifted")
    require(pilot.NUM_ENVS == NUM_ENVS, "banked evaluator environment count drifted")
    require(pilot.EPISODE_LENGTH_S == EPISODE_LENGTH_S, "banked evaluator horizon drifted")
    require(pilot.OBSERVATION_WIDTH == OBSERVATION_WIDTH, "banked evaluator observation width drifted")
    require(tuple(pilot.JOINT_IMPULSE_CAP_N_M_S) == JOINT_IMPULSE_CAP_N_M_S, "banked evaluator impulse caps drifted")
    require(pilot.JOINT_VELOCITY_LIMIT_RAD_S == JOINT_VELOCITY_LIMIT_RAD_S, "banked evaluator qvel limit drifted")
    asset_root = Z1_HAMMER_XML.parents[2].resolve(strict=True)
    asset_identity = git_identity(asset_root, expected_revision=EXPECTED_ASSET_REVISION)
    require(not is_within(output, asset_root), "evaluation output must be outside the asset repository")
    require(not is_within(output, repo_root), "evaluation output must be outside the code repository")

    checkpoint_identity, checkpoint_telemetry = load_checkpoint_identity(
        torch, checkpoint, args.expected_checkpoint_sha256
    )
    model_0_identity, model_0_telemetry = load_model_0_identity(
        torch, checkpoint.parent / "model_0.pt"
    )
    telemetry_artifact = load_consolidated_telemetry(
        attempt,
        model_0_identity,
        model_0_telemetry,
        checkpoint_telemetry,
    )

    original_wrapper = pilot.RslRlVecEnvWrapper
    original_live_validator = pilot._validate_live_observation_contract
    original_publisher = pilot._publish_new_json
    capture_holder: dict[str, Any] = {}

    class CaptureWrapper(original_wrapper):
        def __init__(self, env: Any, clip_actions: float | None = None):
            super().__init__(env, clip_actions=clip_actions)
            require(clip_actions == 1.0, "VIC evaluation wrapper clip drifted")
            capture = VicRuntimeCapture(torch, env, clip_actions=float(clip_actions))
            require(not capture_holder, "more than one evaluation wrapper was constructed")
            capture_holder["capture"] = capture

        def reset(self):
            observations, extras = super().reset()
            capture_holder["capture"].start_population()
            return observations, extras

        def step(self, actions):
            capture = capture_holder["capture"]
            capture.before_step(actions)
            result = super().step(actions)
            capture.after_step()
            return result

        def close(self):
            capture = capture_holder.get("capture")
            if capture is not None:
                capture.restore_hook()
            return super().close()

    def validate_live(env: Any) -> None:
        original_live_validator(env)
        require(env.cfg.scene.num_envs == NUM_ENVS and env.num_envs == NUM_ENVS, "live environment count drifted")
        require(env.cfg.episode_length_s == EPISODE_LENGTH_S, "live episode horizon drifted")
        require(env.cfg.auto_reset is False, "live auto_reset must be false")
        require(env.cfg.seed == EVALUATION_SEED, "live evaluation seed drifted")
        require(env.cfg.sim.mujoco.timestep == PHYSICS_DT_S, "live physics timestep drifted")
        require(env.cfg.decimation == CONTROL_DECIMATION, "live control decimation drifted")
        require(env.max_episode_length == MAX_CONTROL_STEPS, "live maximum control-step horizon drifted")
        require(tuple(env.action_manager.active_terms) == ACTION_TERMS, "live action signature drifted")
        require(env.action_manager.total_action_dim == ACTION_WIDTH, "live action width drifted")
        metadata = runner_module._get_hammer_metadata(env, f"victt_seed2_job{job}", raw_policy_clip=1.0)
        capture_holder["capture"].set_live_metadata(metadata)

    def publish(output_path: Path, payload: Mapping[str, object]) -> None:
        mutable = dict(payload)
        capture = capture_holder.get("capture")
        require(capture is not None, "VIC runtime capture was not constructed")
        runtime, live_metadata = capture.snapshot(mutable["episodes"])
        mutable["schema_version"] = 3
        mutable["protocol"] = {
            **mutable["protocol"],
            "physics_dt_s": PHYSICS_DT_S,
            "physics_sample_hz": 500,
            "control_decimation": CONTROL_DECIMATION,
            "max_control_steps": MAX_CONTROL_STEPS,
        }
        mutable["checkpoint"] = checkpoint_identity
        mutable["evaluator"] = {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "banked_fic_evaluator_path": str(evaluator_path),
            "banked_fic_evaluator_sha256": EXPECTED_FIC_EVALUATOR_SHA256,
            "reuse_mode": "in_memory_import_and_patch",
        }
        mutable["runtime"] = {
            "device": "cuda:0",
            "gpu": gpu,
            "packages": packages,
        }
        mutable["checkpoint_vic_rollout_telemetry"] = checkpoint_telemetry
        mutable["canonical_training_telemetry_artifact"] = telemetry_artifact
        mutable["live_runner_metadata"] = live_metadata
        mutable["vic_runtime_telemetry"] = runtime
        outcomes = validate_episode_outcomes(mutable["episodes"], mutable["summary"])
        behavior_gate_pass = all(
            outcomes[key]
            for key in (
                "success_gate_at_least_58_of_64",
                "productive_gate_at_least_58_of_64",
                "qvel_gate_64_of_64",
                "impulse_gate_64_of_64",
            )
        )
        training_gain_gate_pass = telemetry_artifact["training_gain_gate_pass"] is True
        impossible_success_gate_pass = (
            telemetry_artifact["impossible_success_gate_pass"] is True
        )
        mutable["validation"] = {
            **outcomes,
            "all_runtime_invariants": True,
            "training_gain_gate_pass": training_gain_gate_pass,
            "impossible_success_gate_pass": impossible_success_gate_pass,
            "canary_behavior_gate_pass": behavior_gate_pass,
            "canary_all_gates_pass": (
                behavior_gate_pass
                and training_gain_gate_pass
                and impossible_success_gate_pass
            ),
        }
        require(git_identity(repo_root, expected_revision=EXPECTED_CODE_REVISION) == code_identity, "post-rollout code identity changed")
        require(git_identity(asset_root, expected_revision=EXPECTED_ASSET_REVISION) == asset_identity, "post-rollout asset identity changed")
        require(sha256_file(checkpoint) == args.expected_checkpoint_sha256, "checkpoint changed during evaluation")
        validate_protocol_payload(mutable, pilot=pilot, expected_checkpoint_sha256=args.expected_checkpoint_sha256)
        payload.clear()
        payload.update(mutable)
        original_publisher(output_path, payload)

    pilot.validate_fic_contract = build_vic_validator(pilot, variable_module, observation_module)
    pilot.RslRlVecEnvWrapper = CaptureWrapper
    pilot._validate_live_observation_contract = validate_live
    pilot._publish_new_json = publish
    try:
        payload = pilot.evaluate_checkpoint(
            VIC_TASK,
            checkpoint,
            output,
            "cuda:0",
            training_seed=TRAINING_SEED,
        )
    finally:
        pilot.RslRlVecEnvWrapper = original_wrapper
        pilot._validate_live_observation_contract = original_live_validator
        pilot._publish_new_json = original_publisher

    require(output.is_file() and not output.is_symlink(), "evaluation did not publish a regular JSON artifact")
    expected_disk = json_bytes(payload) + b"\n"
    require(output.read_bytes() == expected_disk, "published evaluation bytes differ from returned payload")
    validate_protocol_payload(payload, pilot=pilot, expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    require(git_identity(repo_root, expected_revision=EXPECTED_CODE_REVISION) == code_identity, "post-publish code identity changed")
    require(git_identity(asset_root, expected_revision=EXPECTED_ASSET_REVISION) == asset_identity, "post-publish asset identity changed")
    require(sha256_file(checkpoint) == args.expected_checkpoint_sha256, "checkpoint changed after evaluation")
    return payload, output


def _fake_rollout_telemetry() -> dict[str, object]:
    stats = {
        "mean": [0.0] * 6,
        "std": [0.1] * 6,
        "minimum": [-1.0] * 6,
        "p05": [-0.5] * 6,
        "median": [0.0] * 6,
        "p95": [0.5] * 6,
        "maximum": [1.0] * 6,
        "lower_bound_occupancy": [0.01] * 6,
        "upper_bound_occupancy": [0.01] * 6,
    }
    return {
        "schema_version": 1,
        "source": "cat_rollout_storage",
        "action_terms": list(ACTION_TERMS),
        "joint_names": list(JOINT_NAMES),
        "gain_action_indices": list(range(6, 12)),
        "raw_action_clip": 1.0,
        "rollout_steps_per_env": 24,
        "sample_count": 24 * 4096,
        "temporal_provenance": {
            "rollout_generated_by": "pre_update_behavior_policy",
            "checkpoint_weights": "post_update",
        },
        "deterministic_gaussian_mean": {"raw": copy.deepcopy(stats), "clipped": copy.deepcopy(stats)},
        "sampled_action": {"raw": copy.deepcopy(stats), "clipped": copy.deepcopy(stats)},
        "gaussian_exploration_std": [0.2] * 6,
    }


def _fake_training_telemetry(
    checkpoint_telemetry: Mapping[str, object]
) -> dict[str, object]:
    curriculum = []
    for iteration in range(500):
        stages = (0.10, 0.08, 0.06, 0.04, 0.02, 0.00)
        curriculum.append(
            {"iteration": iteration, "value": stages[min(iteration // 50, 5)]}
        )
    impossible = [
        {"iteration": iteration, "value": 0.0} for iteration in range(500)
    ]
    return {
        "schema_version": 1,
        "task": VIC_TASK,
        "training_seed": TRAINING_SEED,
        "checkpoint_iterations": [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499],
        "vic_rollout_telemetry": {
            "model_0": copy.deepcopy(checkpoint_telemetry),
            "model_499": copy.deepcopy(checkpoint_telemetry),
        },
        "final_gain_classification": {
            "all_zero": False,
            "nonzero_spread": True,
            "bound_pinned_joints": [],
        },
        "r_imit_curriculum": {
            "scalar": "Curriculum/r_imit_anneal/weight",
            "observed": curriculum,
            "final_weight": 0.0,
        },
        "impossible_success": {
            "scalar": "Episode_Metrics/impossible_success",
            "observed": impossible,
            "all_zero": True,
        },
    }


def run_self_test() -> None:
    telemetry = _fake_rollout_telemetry()
    assert validate_rollout_telemetry(telemetry, expected_samples=24 * 4096) == telemetry
    bad = copy.deepcopy(telemetry)
    bad["deterministic_gaussian_mean"]["clipped"]["maximum"][0] = 1.0001
    try:
        validate_rollout_telemetry(bad, expected_samples=24 * 4096)
    except EvaluationError:
        pass
    else:
        raise AssertionError("clipped telemetry bound mutation was accepted")
    bad = copy.deepcopy(telemetry)
    bad["gaussian_exploration_std"][2] = float("nan")
    try:
        validate_rollout_telemetry(bad, expected_samples=24 * 4096)
    except EvaluationError:
        pass
    else:
        raise AssertionError("non-finite telemetry mutation was accepted")
    training = _fake_training_telemetry(telemetry)
    training_validation = validate_training_telemetry_payload(
        training, telemetry, telemetry
    )
    assert training_validation["training_gain_gate_pass"] is True
    assert training_validation["impossible_success_gate_pass"] is True
    for mutation, message in (
        (lambda value: value["vic_rollout_telemetry"].pop("model_0"), "missing model_0"),
        (
            lambda value: value["final_gain_classification"].update(
                {"all_zero": True}
            ),
            "false gain classification",
        ),
        (
            lambda value: value["impossible_success"]["observed"][0].update(
                {"value": 1.0}
            ),
            "nonzero impossible_success",
        ),
        (
            lambda value: value["r_imit_curriculum"]["observed"].pop(0),
            "incomplete curriculum stream",
        ),
        (
            lambda value: value["impossible_success"]["observed"].pop(0),
            "incomplete impossible_success stream",
        ),
    ):
        bad_training = copy.deepcopy(training)
        mutation(bad_training)
        try:
            validate_training_telemetry_payload(
                bad_training, telemetry, telemetry
            )
        except EvaluationError:
            pass
        else:
            raise AssertionError(f"{message} mutation was accepted")
    forged_model_0 = copy.deepcopy(telemetry)
    forged_model_0["deterministic_gaussian_mean"]["clipped"]["mean"][0] = 0.5
    try:
        validate_training_telemetry_payload(
            training, forged_model_0, telemetry
        )
    except EvaluationError:
        pass
    else:
        raise AssertionError("forged model_0 checkpoint telemetry was accepted")

    episodes = []
    for env_id in range(NUM_ENVS):
        peaks = [cap * 0.5 for cap in JOINT_IMPULSE_CAP_N_M_S]
        episodes.append(
            {
                "env_id": env_id,
                "episode_steps": 10,
                "success": True,
                "timeout": False,
                "first_strike_started": True,
                "first_strike_finalized": True,
                "first_strike_productive": True,
                "first_strike_reason": "success",
                "precontact_velocity_m_s": 1.0,
                "first_event_impulse_n_s": 0.25,
                "nail_depth_m": 0.032,
                "joint_impulse_peak_n_m_s": peaks,
                "joint_impulse_utilization": [0.5] * 6,
                "joint_velocity_peak_rad_s": [3.1415] * 6,
                "joint_velocity_legal": True,
                "joint_target_rmse_rad": 0.1,
                "joint_target_error_max_rad": 0.2,
                "reference_error_mean_m": 0.01,
                "reference_error_max_m": 0.02,
                "reference_error_samples": 10,
            }
        )
    summary = {
        "task_success_n": 64,
        "task_success_rate": 1.0,
        "productive_first_strike_n": 64,
        "productive_first_strike_rate": 1.0,
        "joint_velocity_peak_rad_s": {"all_joints_legal_n": 64, "all_joints_legal_rate": 1.0},
        "joint_impulse_utilization": {"all_joints_at_or_below_cap_n": 64, "all_joints_at_or_below_cap_rate": 1.0},
    }
    outcomes = validate_episode_outcomes(episodes, summary)
    assert outcomes["qvel_gate_64_of_64"] is True
    assert outcomes["impulse_gate_64_of_64"] is True
    mutated = copy.deepcopy(summary)
    mutated["task_success_n"] = 63
    try:
        validate_episode_outcomes(episodes, mutated)
    except EvaluationError:
        pass
    else:
        raise AssertionError("recomputed success mutation was accepted")
    unsafe = copy.deepcopy(episodes)
    unsafe[0]["joint_velocity_peak_rad_s"][0] = 3.1416
    unsafe[0]["joint_velocity_legal"] = False
    unsafe_summary = copy.deepcopy(summary)
    unsafe_summary["joint_velocity_peak_rad_s"] = {
        "all_joints_legal_n": 63,
        "all_joints_legal_rate": 63 / 64,
    }
    unsafe_outcomes = validate_episode_outcomes(unsafe, unsafe_summary)
    assert unsafe_outcomes["qvel_gate_64_of_64"] is False
    print("SELF_TEST_PASS telemetry_schema=1 episodes=64 boundary_qvel=3.1415 mutations_rejected=9")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--self-test"]:
        run_self_test()
        return 0
    args = parse_args(argv)
    payload, output = run_evaluation(args)
    validation = payload["validation"]
    output_sha256 = sha256_file(output)
    print(f"Z1_VIC_EVAL_OUTPUT={output}")
    print(f"Z1_VIC_EVAL_OUTPUT_SHA256={output_sha256}")
    print(f"Z1_VIC_EVAL_SCRIPT_SHA256={sha256_file(Path(__file__).resolve())}")
    print(f"Z1_VIC_EVAL_CHECKPOINT_SHA256={payload['checkpoint']['sha256']}")
    print(
        "Z1_VIC_EVAL_COUNTS "
        f"success={validation['task_success_n_recomputed']}/{NUM_ENVS} "
        f"productive={validation['productive_first_strike_n_recomputed']}/{NUM_ENVS} "
        f"qvel_legal={validation['qvel_500hz_all_joints_legal_n_recomputed']}/{NUM_ENVS} "
        f"impulse_legal={validation['joint_impulse_all_joints_at_or_below_cap_n_recomputed']}/{NUM_ENVS}"
    )
    if validation["canary_all_gates_pass"] is not True:
        print("Z1_VIC_EVAL_FAIL: canary gate failed; diagnostic JSON was preserved", file=sys.stderr)
        return 3
    print("Z1_VIC_EVAL_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Z1_VIC_EVAL_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
