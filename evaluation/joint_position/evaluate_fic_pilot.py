"""Evaluate one matched fixed-impedance FIC pilot checkpoint.

This deliberately small evaluator is separate from the historical Cartesian
campaign evaluator: it admits only the two joint-position treatments and
records the first completed episode from each member of one fixed 64-world
population.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from src.tasks.hammer.config.z1.joint_position_contract import JOINT_NAMES
from src.tasks.hammer.mdp.first_strike import (
    REASON_SUCCESS,
    REASON_WINDOW,
    _ENV_FIRST_STRIKE_ATTR,
)
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH


PARENT_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
    "CProgress-Vel-Delivered4"
)
FIC0_TASK = f"{PARENT_TASK}-JointPosition-Fixed"
FICTT_TASK = f"{FIC0_TASK}-TT"
TASKS = (FIC0_TASK, FICTT_TASK)

SEED = 2026081202
NUM_ENVS = 64
EPISODE_LENGTH_S = 4.0
I_REF_N_S = 0.2799950838088989
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity(path: Path) -> dict[str, str]:
    revision = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if _REVISION_RE.fullmatch(revision) is None or status:
        raise RuntimeError(f"repository must be clean at a full commit: {path}")
    return {"revision": revision, "status": status}


def _cfg_value(mapping: Mapping[str, Any], name: str) -> Any:
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"missing required {name}") from exc


def validate_fic_contract(task: str, env_cfg, agent_cfg) -> dict[str, object]:
    """Fail closed unless a loaded config is exactly one calibrated FIC arm."""
    if task not in TASKS:
        raise ValueError(f"unsupported FIC pilot task: {task}")
    if _finite_number(getattr(agent_cfg, "clip_actions", None), name="clip_actions") != 1.0:
        raise ValueError("FIC pilot requires raw policy clip 1.0")
    if getattr(getattr(agent_cfg, "actor", None), "obs_normalization", None) is not True:
        raise ValueError("FIC pilot requires actor observation normalization")

    actions = getattr(env_cfg, "actions", {})
    if tuple(actions) != ("joint_position",):
        raise ValueError("FIC pilot requires exactly the joint_position action")
    action = actions["joint_position"]
    if (
        tuple(getattr(action, "actuator_names", ())) != JOINT_NAMES
        or tuple(getattr(action, "scale", ())) != JOINT_NAMES
        or tuple(getattr(action, "clip", ())) != JOINT_NAMES
        or getattr(action, "use_default_offset", None) is not True
    ):
        raise ValueError("FIC pilot joint action contract drift")

    rewards = getattr(env_cfg, "rewards", {})
    delivered = _cfg_value(rewards, "delivered_impulse")
    if _finite_number(getattr(delivered, "weight", None), name="D4 weight") != 4.0:
        raise ValueError("FIC pilot requires D4 weight 4.0")
    i_ref = _finite_number(
        getattr(delivered, "params", {}).get("i_ref"), name="delivered i_ref"
    )
    if i_ref != I_REF_N_S:
        raise ValueError(f"FIC pilot delivered i_ref drift: {i_ref}")
    if _finite_number(
        getattr(_cfg_value(rewards, "r_waypoint_progress"), "weight", None),
        name="P weight",
    ) != 8.0:
        raise ValueError("FIC pilot requires P weight 8.0")

    metrics = getattr(env_cfg, "metrics", {})
    cat = _cfg_value(metrics, "cat_soft")
    cat_params = getattr(cat, "params", {})
    if (
        cat_params.get("use_vel") is not True
        or cat_params.get("use_impulse") is not True
        or cat_params.get("vel_detection") != "substep"
        or _finite_number(cat_params.get("imp_max_p"), name="imp_max_p") != 0.0
    ):
        raise ValueError("FIC pilot CaT contract drift")
    if getattr(_cfg_value(metrics, "substep_impulse_rows"), "params", {}).get("enabled") is not False:
        raise ValueError("FIC pilot requires row-attribution disabled")

    r_tt = rewards.get("r_tt")
    if task == FIC0_TASK:
        if r_tt is not None:
            raise ValueError("FIC-0 must not contain r_tt")
    elif (
        r_tt is None
        or _finite_number(getattr(r_tt, "weight", None), name="r_tt weight") != -1.0
        or _finite_number(getattr(r_tt, "params", {}).get("k_tt"), name="r_tt k_tt") != 1.0
    ):
        raise ValueError("FIC-TT r_tt contract drift")

    return {
        "task": task,
        "action_term": "joint_position",
        "joint_names": list(JOINT_NAMES),
        "actor_obs_normalization": True,
        "d4_weight": 4.0,
        "p_weight": 8.0,
        "delivered_impulse_i_ref_n_s": i_ref,
        "velocity_cat_substep": True,
        "impulse_cat_log_only": True,
        "row_attribution_enabled": False,
        "r_tt_enabled": task == FICTT_TASK,
        "r_tt_k_tt": 1.0 if task == FICTT_TASK else None,
    }


def _reason(value: object) -> str:
    code = int(value)
    if code == REASON_SUCCESS:
        return "success"
    if code == REASON_WINDOW:
        return "window"
    return "none"


def _finite_six(value: torch.Tensor) -> list[float]:
    values = [float(item) for item in value.detach().cpu().tolist()]
    if len(values) != len(JOINT_NAMES) or not all(math.isfinite(item) for item in values):
        raise RuntimeError("joint impulse peak must contain six finite values")
    return values


def capture_first_terminals(
    *,
    records: dict[int, dict[str, object]],
    done: torch.Tensor,
    terminated: torch.Tensor,
    timed_out: torch.Tensor,
    steps: torch.Tensor,
    tracker,
    nail_depth: torch.Tensor,
    joint_impulse_peak: torch.Tensor,
) -> tuple[int, ...]:
    """Copy each first terminal state before any caller-reset can mutate it."""
    ids = torch.nonzero(done, as_tuple=False).flatten().detach().cpu().tolist()
    for env_id in ids:
        if env_id in records:
            continue
        record = {
            "env_id": int(env_id),
            "success": bool(terminated[env_id]),
            "timeout": bool(timed_out[env_id]),
            "episode_steps": int(steps[env_id].item()),
            "first_strike_started": bool(tracker.started[env_id]),
            "first_strike_finalized": bool(tracker.finalized[env_id]),
            "first_strike_productive": bool(tracker.productive[env_id]),
            "first_strike_reason": _reason(tracker.reason[env_id]),
            "precontact_velocity_m_s": float(tracker.v_precontact[env_id]),
            "first_event_impulse_n_s": float(tracker.delivered[env_id]),
            "nail_depth_m": float(nail_depth[env_id]),
            "joint_impulse_peak_n_m_s": _finite_six(joint_impulse_peak[env_id]),
        }
        if not all(
            math.isfinite(float(record[key]))
            for key in ("precontact_velocity_m_s", "first_event_impulse_n_s", "nail_depth_m")
        ):
            raise RuntimeError("terminal first-strike metrics must be finite")
        records[env_id] = record
    return tuple(int(env_id) for env_id in ids)


def _population_hash(env) -> str:
    robot = env.scene["robot"].data
    nail = env.scene["nail_block"].data
    payload = {
        "robot_joint_pos": robot.joint_pos.detach().cpu().tolist(),
        "robot_joint_vel": robot.joint_vel.detach().cpu().tolist(),
        "nail_joint_pos": nail.joint_pos.detach().cpu().tolist(),
        "nail_joint_vel": nail.joint_vel.detach().cpu().tolist(),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _publish_new_json(output: Path, payload: Mapping[str, object]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluator output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    impulses = np.asarray([row["first_event_impulse_n_s"] for row in records], dtype=float)
    peaks = np.asarray([row["joint_impulse_peak_n_m_s"] for row in records], dtype=float)
    return {
        "task_success_n": sum(bool(row["success"]) for row in records),
        "task_success_rate": float(np.mean([bool(row["success"]) for row in records])),
        "productive_first_strike_n": sum(
            bool(row["first_strike_productive"]) for row in records
        ),
        "productive_first_strike_rate": float(
            np.mean([bool(row["first_strike_productive"]) for row in records])
        ),
        "first_event_impulse_n_s": {
            "mean": float(np.mean(impulses)),
            "std": float(np.std(impulses)),
            "min": float(np.min(impulses)),
            "max": float(np.max(impulses)),
        },
        "joint_impulse_peak_n_m_s": {
            "p95": np.quantile(peaks, 0.95, axis=0).tolist(),
            "max": np.max(peaks, axis=0).tolist(),
        },
    }


def _ordered_complete_records(records: Mapping[int, dict[str, object]]) -> list[dict[str, object]]:
    """Return the fixed population only when every expected first episode exists."""
    expected = set(range(NUM_ENVS))
    actual = set(records)
    if actual != expected:
        raise RuntimeError(
            f"incomplete first-episode population: {len(actual)}/{NUM_ENVS}"
        )
    return [records[index] for index in range(NUM_ENVS)]


def evaluate_checkpoint(
    task: str, checkpoint: Path, output: Path, device: str = "cpu"
) -> dict[str, object]:
    """Evaluate a mean policy on the frozen 64-world first-episode population."""
    checkpoint = Path(checkpoint)
    output = Path(output)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluator output: {output}")

    env_cfg = load_env_cfg(task, play=False)
    agent_cfg = load_rl_cfg(task)
    contract = validate_fic_contract(task, env_cfg, agent_cfg)
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.episode_length_s = EPISODE_LENGTH_S
    env_cfg.auto_reset = False
    env_cfg.seed = SEED
    torch.manual_seed(SEED)

    root = Path(__file__).resolve().parents[2]
    from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML

    code_git = _git_identity(root)
    asset_git = _git_identity(Z1_HAMMER_XML.parents[2])
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR, None)
        accumulator = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
        if tracker is None or accumulator is None:
            raise RuntimeError("FIC pilot requires first-strike and joint-impulse instrumentation")
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        # Runner construction can consume Torch RNG state; reset the global stream at
        # the population boundary as well as supplying the environment seed above.
        torch.manual_seed(SEED)
        observations, _ = wrapped.reset()
        population_hash = _population_hash(env)
        records: dict[int, dict[str, object]] = {}
        steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=env.device)
        max_steps = env.max_episode_length
        for _ in range(max_steps):
            with torch.no_grad():
                actions = policy(observations, stochastic_output=False)
            observations, _, _, _ = wrapped.step(actions)
            steps += 1
            done = env.reset_terminated | env.reset_time_outs
            done_ids = capture_first_terminals(
                records=records,
                done=done,
                terminated=env.reset_terminated,
                timed_out=env.reset_time_outs,
                steps=steps,
                tracker=tracker,
                nail_depth=env.scene["nail_block"].data.joint_pos[:, 0].clamp(
                    0.0, NAIL_GOAL_DEPTH
                ),
                joint_impulse_peak=accumulator._episode_peak_perjoint,
            )
            if len(records) == NUM_ENVS:
                break
            if done_ids:
                ids = torch.tensor(done_ids, device=env.device, dtype=torch.long)
                obs_dict, _ = env.reset(env_ids=ids)
                observations = TensorDict(obs_dict, batch_size=[env.num_envs])
                steps[ids] = 0
        ordered = _ordered_complete_records(records)
        payload: dict[str, object] = {
            "schema_version": 1,
            "task": task,
            "protocol": {
                "seed": SEED,
                "num_envs": NUM_ENVS,
                "episodes_per_env": 1,
                "episode_length_s": EPISODE_LENGTH_S,
                "auto_reset": False,
                "policy_mode": "mean",
            },
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha256_file(checkpoint)},
            "code_git": code_git,
            "asset_git": asset_git,
            "treatment_contract": contract,
            "initial_population_sha256": population_hash,
            "episodes": ordered,
            "summary": _summary(ordered),
        }
    finally:
        wrapped.close()
    _publish_new_json(output, payload)
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    payload = evaluate_checkpoint(args.task, args.checkpoint, args.output, args.device)
    print(
        f"FIC pilot evaluation complete output={args.output} "
        f"success={payload['summary']['task_success_n']}/{NUM_ENVS}"
    )


if __name__ == "__main__":
    main()
