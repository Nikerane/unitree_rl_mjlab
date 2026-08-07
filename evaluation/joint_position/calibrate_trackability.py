"""Deterministically calibrate the fixed-joint one-step tracking cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np

from evaluation.joint_position.qualify_joint_action import (
    canonical_sha256,
    normalized_action_for_target,
    scheduled_replay_targets,
)
from src.tasks.hammer.mdp.first_strike import REASON_SUCCESS


REQUIRED_SEEDS = tuple(range(1000, 1016))
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
FIC0_TASK_ID = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
    "CProgress-Vel-Delivered4-JointPosition-Fixed"
)
TARGET_Q90_COST = 0.1
MIN_Q90 = 1e-8
MAX_PRE_DT_CUMULATIVE_COST = 5.0


def derive_k_tt(
    squared_errors: object, target_q90_cost: float = TARGET_Q90_COST
) -> float:
    """Derive ``k_tt`` from one unique trajectory's 90th-percentile error."""
    if (
        isinstance(target_q90_cost, bool)
        or not isinstance(target_q90_cost, (int, float))
        or not math.isfinite(float(target_q90_cost))
        or float(target_q90_cost) <= 0.0
    ):
        raise ValueError("target_q90_cost must be finite and strictly positive")
    try:
        errors = np.asarray(squared_errors, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("squared_errors must be numeric") from exc
    if (
        errors.ndim != 1
        or errors.size == 0
        or not np.all(np.isfinite(errors))
        or np.any(errors < 0.0)
    ):
        raise ValueError("squared_errors must be a nonempty finite nonnegative vector")
    q90 = float(np.quantile(errors, 0.90))
    if not math.isfinite(q90) or q90 < MIN_Q90:
        raise ValueError(f"q90 must be finite and at least {MIN_Q90}")
    gain = float(target_q90_cost) / q90
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("derived k_tt must be finite and strictly positive")
    return gain


def _canonical_payload_sha(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_hash(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _required_revision(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase Git commit")
    return value


def _validated_run(run: object, *, expected_seed: int) -> tuple[dict[str, Any], np.ndarray]:
    if not isinstance(run, dict):
        raise ValueError("each calibration run must be a mapping")
    if run.get("seed") != expected_seed or isinstance(run.get("seed"), bool):
        raise ValueError("calibration runs must be ordered seeds 1000 through 1015")
    if run.get("terminal_transition_captured") is not True:
        raise ValueError("every calibration run must capture its terminal transition")
    try:
        targets = np.asarray(run.get("applied_target_tape_rad"), dtype=np.float64)
        errors = np.asarray(run.get("squared_errors_rad2"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration run targets and errors must be numeric") from exc
    if (
        targets.ndim != 2
        or targets.shape[0] == 0
        or targets.shape[1] != len(JOINT_NAMES)
        or errors.ndim != 1
        or errors.shape[0] != targets.shape[0]
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(errors))
        or np.any(errors < 0.0)
    ):
        raise ValueError("each run needs matching finite [step,6] targets and squared errors")
    applied_target_hash = canonical_sha256(targets.tolist())
    qualified_target_hash = _required_hash(
        run.get("qualified_applied_target_tape_sha256"),
        name="qualified_applied_target_tape_sha256",
    )
    if applied_target_hash != qualified_target_hash:
        raise ValueError(
            f"seed {expected_seed} live target tape does not match its qualified replay"
        )
    row = {
        "seed": expected_seed,
        "sample_count": int(errors.size),
        "terminal_transition_captured": True,
        "applied_target_tape_sha256": applied_target_hash,
        "squared_errors_sha256": canonical_sha256(errors.tolist()),
    }
    return row, errors


def build_trackability_payload(
    *,
    runs: Sequence[dict[str, Any]],
    source_qualification_payload_sha256: str,
    source_qualification_code_revision: str,
    source_asset_revision: str,
    calibration_code_revision: str,
    physics_dt_s: float,
    control_decimation: int,
) -> dict[str, Any]:
    """Build a PASS artifact from one trajectory plus 15 identity witnesses."""
    if (
        isinstance(physics_dt_s, bool)
        or not isinstance(physics_dt_s, (int, float))
        or not math.isfinite(float(physics_dt_s))
        or float(physics_dt_s) <= 0.0
    ):
        raise ValueError("physics_dt_s must be finite and strictly positive")
    if (
        isinstance(control_decimation, bool)
        or not isinstance(control_decimation, int)
        or control_decimation <= 0
    ):
        raise ValueError("control_decimation must be a positive non-bool integer")
    measured_physics_dt_s = float(physics_dt_s)
    reward_manager_dt_s = measured_physics_dt_s * control_decimation
    if len(runs) != len(REQUIRED_SEEDS):
        raise ValueError("calibration requires exactly 16 repeatability runs")
    rows: list[dict[str, Any]] = []
    canonical_errors: np.ndarray | None = None
    for run, seed in zip(runs, REQUIRED_SEEDS, strict=True):
        row, errors = _validated_run(run, expected_seed=seed)
        rows.append(row)
        if canonical_errors is None:
            canonical_errors = errors
        elif (
            row["sample_count"] != rows[0]["sample_count"]
            or row["applied_target_tape_sha256"]
            != rows[0]["applied_target_tape_sha256"]
            or row["squared_errors_sha256"] != rows[0]["squared_errors_sha256"]
        ):
            raise ValueError(
                "repeatability witness differs from the unique seed-1000 canonical trajectory"
            )
    assert canonical_errors is not None
    q90 = float(np.quantile(canonical_errors, 0.90))
    k_tt = derive_k_tt(canonical_errors)
    pre_dt_cost = float(k_tt * np.sum(canonical_errors, dtype=np.float64))
    if not math.isfinite(pre_dt_cost) or pre_dt_cost > MAX_PRE_DT_CUMULATIVE_COST:
        raise ValueError(
            "canonical pre-dt cumulative cost exceeds the preregistered 5.0 ceiling"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "joint_names": list(JOINT_NAMES),
        "physics_dt_s": measured_physics_dt_s,
        "control_decimation": control_decimation,
        "reward_manager_dt_s": reward_manager_dt_s,
        "target_q90_cost": TARGET_Q90_COST,
        "q90_squared_error_rad2": q90,
        "k_tt": k_tt,
        "canonical_seed": REQUIRED_SEEDS[0],
        "canonical_sample_count": int(canonical_errors.size),
        "canonical_pre_dt_cumulative_cost": pre_dt_cost,
        "canonical_returned_dose": pre_dt_cost * reward_manager_dt_s,
        "canonical_applied_target_tape_sha256": rows[0][
            "applied_target_tape_sha256"
        ],
        "canonical_squared_errors_rad2": canonical_errors.tolist(),
        "canonical_squared_errors_sha256": rows[0]["squared_errors_sha256"],
        "repeatability_rows": rows,
        "source_qualification_payload_sha256": _required_hash(
            source_qualification_payload_sha256,
            name="source_qualification_payload_sha256",
        ),
        "source_qualification_code_revision": _required_revision(
            source_qualification_code_revision,
            name="source_qualification_code_revision",
        ),
        "source_asset_revision": _required_revision(
            source_asset_revision, name="source_asset_revision"
        ),
        "calibration_code_revision": _required_revision(
            calibration_code_revision, name="calibration_code_revision"
        ),
        "decision": "PASS",
    }
    payload["payload_sha256"] = _canonical_payload_sha(payload)
    return payload


def write_canonical_json(path: str | Path, payload: object) -> None:
    """Serialize deterministically while rejecting NaN and infinity."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_joint_position_contract(path: str | Path) -> Any:
    """Late-bound source loader, kept patchable for CLI boundary tests."""
    from src.tasks.hammer.config.z1.joint_position_contract import (
        load_joint_position_contract as load,
    )

    return load(path)


def load_joint_trackability_contract(path: str | Path, *, source_contract: Any) -> Any:
    """Late-bound consumer loader used for the CLI's pre-PASS round trip."""
    from src.tasks.hammer.config.z1.joint_position_contract import (
        load_joint_trackability_contract as load,
    )

    return load(path, source_contract=source_contract)


def publish_validated_json(
    path: str | Path, payload: object, *, source_contract: Any
) -> None:
    """Strict-load deterministic sibling bytes before atomically publishing them."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_canonical_json(temporary, payload)
        load_joint_trackability_contract(
            temporary, source_contract=source_contract
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_git_worktree(path: Path, *, label: str) -> None:
    """Fail before recording a commit as provenance for dirty worktree content."""
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(f"{label} worktree is dirty; refusing calibration provenance")


def _scheduled_replay_tape(source_contract: Any) -> np.ndarray:
    """Reconstruct the preregistered zero-order hold used by qualification."""
    rows = source_contract.per_seed_replay_rows
    if not rows:
        raise ValueError("source contract has no replay rows")
    playback_length = int(rows[0]["source"]["playback_length"])
    scheduled = scheduled_replay_targets(
        source_contract.source_target_tape_rad,
        playback_length=playback_length,
    )
    for row in rows:
        if int(row["source"]["playback_length"]) != playback_length:
            raise ValueError("qualified source playback lengths are not repeatable")
        expected_count = int(row["replay"]["target_count"])
        if expected_count <= 0 or expected_count > len(scheduled):
            raise ValueError("qualified replay target count exceeds the scheduled tape")
    return scheduled


def _require_accepted_terminal(env: Any, expected_replay: Any) -> None:
    """Reject a same-length replay that ended for anything but productive success."""
    tracker = getattr(env, "_hammer_first_strike", None)
    if (
        expected_replay.get("terminal_reason") != "success"
        or not bool(env.reset_buf[0])
        or not bool(env.reset_terminated[0])
        or bool(env.reset_time_outs[0])
        or tracker is None
        or not bool(tracker.finalized[0])
        or not bool(tracker.productive[0])
        or int(tracker.reason[0]) != REASON_SUCCESS
    ):
        raise RuntimeError(
            "calibration replay must reproduce the banked productive success terminal"
        )


def _validated_cfg_timing(cfg: Any, source_contract: Any) -> tuple[float, int]:
    """Read the live config timing and bind it to the qualified source timing."""
    physics_dt_s = cfg.sim.mujoco.timestep
    control_decimation = cfg.decimation
    if (
        isinstance(physics_dt_s, bool)
        or not isinstance(physics_dt_s, (int, float))
        or not math.isfinite(float(physics_dt_s))
        or float(physics_dt_s) <= 0.0
        or float(physics_dt_s) != float(source_contract.physics_dt_s)
    ):
        raise RuntimeError("live FIC-0 physics timestep does not match the source contract")
    if (
        isinstance(control_decimation, bool)
        or not isinstance(control_decimation, int)
        or control_decimation <= 0
        or control_decimation != source_contract.control_decimation
    ):
        raise RuntimeError("live FIC-0 control decimation does not match the source contract")
    return float(physics_dt_s), control_decimation


def _rollout_repeated_trajectory(
    source_contract: Any,
) -> tuple[list[dict[str, Any]], float, int]:
    """Replay the banked tape once per fixed reset and read every post-step error."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.registry import load_env_cfg

    import src.tasks  # noqa: F401 - populate task registry
    from src.tasks.hammer.mdp.trackability import joint_target_squared_error

    cfg = load_env_cfg(FIC0_TASK_ID, play=True)
    physics_dt_s, control_decimation = _validated_cfg_timing(cfg, source_contract)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        arm_cfg = SceneEntityCfg(
            "robot", joint_names=JOINT_NAMES, preserve_order=True
        )
        arm_cfg.resolve(env.scene)
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_position")
        target_ids = action_term.target_ids
        normalized = normalized_action_for_target(
            _scheduled_replay_tape(source_contract),
            source_contract.default_joint_pos_rad,
            source_contract.scale_rad,
        )
        runs: list[dict[str, Any]] = []
        for seed, qualification_row in zip(
            source_contract.seeds,
            source_contract.per_seed_replay_rows,
            strict=True,
        ):
            expected_replay = qualification_row["replay"]
            expected_count = int(expected_replay["target_count"])
            env.reset(seed=int(seed))
            targets: list[list[float]] = []
            errors: list[float] = []
            for row in normalized:
                action = torch.tensor(
                    row[None, :], dtype=torch.float32, device=env.device
                )
                env.step(action)
                applied = (
                    robot.data.joint_pos_target[0, target_ids]
                    .detach()
                    .cpu()
                    .to(torch.float64)
                    .tolist()
                )
                error = float(joint_target_squared_error(env, arm_cfg)[0])
                targets.append(applied)
                errors.append(error)
                if bool(env.reset_buf[0]):
                    break
            terminal = bool(env.reset_buf[0])
            if len(errors) != expected_count or not terminal:
                raise RuntimeError(
                    "calibration did not reproduce the complete accepted-contact terminal replay"
                )
            _require_accepted_terminal(env, expected_replay)
            runs.append(
                {
                    "seed": int(seed),
                    "applied_target_tape_rad": targets,
                    "qualified_applied_target_tape_sha256": expected_replay[
                        "replay_applied_target_tape_sha256"
                    ],
                    "squared_errors_rad2": errors,
                    "terminal_transition_captured": terminal,
                }
            )
        return runs, physics_dt_s, control_decimation
    finally:
        env.close()


def run_calibration(qualification_path: str | Path) -> dict[str, Any]:
    """Load, replay, cross-check, and calibrate one banked qualification tape."""
    from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML

    contract = load_joint_position_contract(qualification_path)
    root = Path(__file__).resolve().parents[2]
    asset_root = Path(
        subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=Path(Z1_HAMMER_XML).parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    _require_clean_git_worktree(root, label="code")
    _require_clean_git_worktree(asset_root, label="asset")
    code_revision = _git_revision(root)
    asset_revision = _git_revision(asset_root)
    if asset_revision != contract.source_asset_revision:
        raise RuntimeError("live asset revision differs from the qualified asset revision")
    runs, physics_dt_s, control_decimation = _rollout_repeated_trajectory(contract)
    _require_clean_git_worktree(root, label="code")
    _require_clean_git_worktree(asset_root, label="asset")
    if _git_revision(root) != code_revision:
        raise RuntimeError("code HEAD changed during calibration rollout")
    if _git_revision(asset_root) != asset_revision:
        raise RuntimeError("asset HEAD changed during calibration rollout")
    return build_trackability_payload(
        runs=runs,
        source_qualification_payload_sha256=contract.payload_sha256,
        source_qualification_code_revision=contract.source_code_revision,
        source_asset_revision=asset_revision,
        calibration_code_revision=code_revision,
        physics_dt_s=physics_dt_s,
        control_decimation=control_decimation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = run_calibration(args.qualification)
    source_contract = load_joint_position_contract(args.qualification)
    publish_validated_json(
        args.out, payload, source_contract=source_contract
    )
    print(
        "joint trackability calibration: PASS "
        f"q90={payload['q90_squared_error_rad2']:.9g} "
        f"k_tt={payload['k_tt']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
