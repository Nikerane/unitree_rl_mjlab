"""Qualify the production scripted strike against the Cartesian guideline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from src.tasks.hammer.mdp.guideline import (
    GUIDELINE_CORRIDOR_RADIUS_M as CORRIDOR_RADIUS_M,
    GUIDELINE_NUM_GATES as REQUIRED_GATES,
)


TASK_ID = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0"
REQUIRED_SEEDS = tuple(range(1000, 1016))
RESET_POSITION_RANGE_RAD = (-0.05, 0.05)
QVEL_LIMIT_RAD_S = 3.1415
HOLD_STEPS = 6
_REQUIRED_ROW_FIELDS = (
    "seed",
    "gates_crossed",
    "contact_seen",
    "nail_progress_m",
    "corridor_max_m",
    "qvel_peak_rad_s",
    "finite",
    "reset_arm_qpos_rad",
)


def _row_failures(row: dict[str, Any]) -> list[str]:
    missing = [name for name in _REQUIRED_ROW_FIELDS if name not in row]
    if missing:
        return [f"missing {name}" for name in missing]

    numeric = (
        row["gates_crossed"],
        row["nail_progress_m"],
        row["corridor_max_m"],
        row["qvel_peak_rad_s"],
    )
    reset_qpos = row["reset_arm_qpos_rad"]
    try:
        numeric_finite = all(math.isfinite(float(value)) for value in numeric)
        reset_finite = len(reset_qpos) == 6 and all(
            math.isfinite(float(value)) for value in reset_qpos
        )
    except (TypeError, ValueError):
        numeric_finite = False
        reset_finite = False
    if not bool(row["finite"]) or not numeric_finite or not reset_finite:
        return ["non-finite value"]

    failures: list[str] = []
    if int(row["gates_crossed"]) < REQUIRED_GATES:
        failures.append("fewer than six gates")
    if not bool(row["contact_seen"]):
        failures.append("no contact")
    if float(row["nail_progress_m"]) <= 0.0:
        failures.append("no nail progress")
    if float(row["corridor_max_m"]) > CORRIDOR_RADIUS_M:
        failures.append("corridor above 0.005 m")
    if float(row["qvel_peak_rad_s"]) > QVEL_LIMIT_RAD_S:
        failures.append("qvel above 3.1415 rad/s")
    return failures


def summarize_qualification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the fail-closed 16-reset decision rule and return JSON-ready data."""
    seeds = [row.get("seed") for row in rows]
    exact_seed_set = seeds == list(REQUIRED_SEEDS)

    annotated_rows: list[dict[str, Any]] = []
    aggregate_failures: list[str] = []
    if not exact_seed_set:
        aggregate_failures.append(
            "seed set must be exactly 1000--1015 with one row per seed"
        )

    try:
        reset_tuples = [
            tuple(float(value) for value in row["reset_arm_qpos_rad"])
            for row in rows
        ]
        distinct_reset_state = (
            exact_seed_set
            and all(len(values) == 6 for values in reset_tuples)
            and len(set(reset_tuples)) == len(REQUIRED_SEEDS)
        )
    except (KeyError, TypeError, ValueError):
        distinct_reset_state = False
    if exact_seed_set and not distinct_reset_state:
        aggregate_failures.append(
            "reset_arm_qpos_rad must contain 16 distinct realized training resets"
        )

    for row in rows:
        failures = _row_failures(row)
        if not exact_seed_set:
            failures = list(failures)
            failures.append("invalid seed set")
        elif not distinct_reset_state:
            failures = list(failures)
            failures.append("duplicate or invalid realized reset state")
        qualifies = not failures
        annotated_rows.append(dict(row, qualifies=qualifies, failures=failures))
        if exact_seed_set and distinct_reset_state:
            seed = row.get("seed", "unknown")
            aggregate_failures.extend(f"seed {seed}: {reason}" for reason in failures)

    qualified = sum(bool(row["qualifies"]) for row in annotated_rows)
    return {
        "schema_version": 1,
        "required_seeds": list(REQUIRED_SEEDS),
        "thresholds": {
            "required_gates": REQUIRED_GATES,
            "corridor_radius_m": CORRIDOR_RADIUS_M,
            "qvel_limit_rad_s": QVEL_LIMIT_RAD_S,
        },
        "passed": exact_seed_set and qualified == len(REQUIRED_SEEDS),
        "qualified_resets": qualified,
        "required_resets": len(REQUIRED_SEEDS),
        "distinct_reset_arm_qpos": distinct_reset_state,
        "failures": aggregate_failures,
        "rows": annotated_rows,
    }


def load_qualification_cfg(loader: Callable[..., Any]) -> Any:
    """Load the production training config and apply passive evaluator settings."""
    cfg = loader(TASK_ID, play=False)
    required_metrics = {"first_strike", "waypoint_progress", "cat_soft"}
    if not required_metrics <= set(cfg.metrics):
        raise RuntimeError("production qualification metrics are missing")
    if float(cfg.metrics["cat_soft"].params["imp_max_p"]) != 0.0:
        raise RuntimeError("qualification requires production imp_max_p=0.0")
    reset_range = tuple(
        cfg.events["reset_robot_joints"].params["position_range"]
    )
    if reset_range != RESET_POSITION_RANGE_RAD:
        raise RuntimeError(
            "qualification requires reset position_range "
            f"{RESET_POSITION_RANGE_RAD}, got {reset_range}"
        )
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    return cfg


def canonical_qvel_trace(pre: Any, post: Any):
    """Return every pre-integration state plus the final post-integration state."""
    import numpy as np

    pre_array = np.asarray(pre, dtype=np.float64)
    post_array = np.asarray(post, dtype=np.float64)
    if pre_array.ndim != 2 or post_array.shape != pre_array.shape:
        raise ValueError(
            f"qvel pre/post must have matching [substep,joint] shapes, got "
            f"{pre_array.shape} and {post_array.shape}"
        )
    if len(post_array) == 0:
        return pre_array.copy()
    return np.concatenate((pre_array, post_array[-1:]), axis=0)


def corridor_window_errors(samples: list[dict[str, Any]]) -> list[float]:
    """Select gate-1 transition through accepted first-contact onset, inclusively."""
    errors: list[float] = []
    entered = False
    for sample in samples:
        entered |= int(sample["gates_crossed"]) >= 1
        if entered:
            errors.append(float(sample["error_m"]))
        if bool(sample["accepted_contact"]):
            break
    return errors


def trace_numeric_is_finite(
    trace: dict[str, Any],
    *,
    initial_depth: float,
    reset_head: Any,
    reset_arm_qpos: Any,
    entry: Any,
    nail: Any,
) -> bool:
    """Reject any non-finite value in the complete recorded numeric tape."""
    import numpy as np

    channels = [
        np.asarray([initial_depth, trace["physics_dt_s"]], dtype=np.float64),
        np.asarray(reset_head, dtype=np.float64),
        np.asarray(reset_arm_qpos, dtype=np.float64),
        np.asarray(entry, dtype=np.float64),
        np.asarray(nail, dtype=np.float64),
        np.asarray(trace["qvel_pre"], dtype=np.float64),
        np.asarray(trace["qvel_post"], dtype=np.float64),
        np.asarray(trace["path"], dtype=np.float64),
        np.asarray(trace["depth"], dtype=np.float64),
        np.asarray(trace["actions"], dtype=np.float64),
        np.asarray(
            [
                (sample["gates_crossed"], sample["error_m"])
                for sample in trace["samples"]
            ],
            dtype=np.float64,
        ),
        np.asarray(list(trace["gate_centers"].values()), dtype=np.float64),
    ]
    if trace["contact_point"] is not None:
        channels.append(np.asarray(trace["contact_point"], dtype=np.float64))
    return all(np.isfinite(channel).all() for channel in channels)


def run_required_seeds(run_one: Callable[[int], dict[str, Any]]) -> list[dict[str, Any]]:
    """Run every preregistered reset in its fixed order; never select successes."""
    return [run_one(seed) for seed in REQUIRED_SEEDS]


def write_result_tables(summary: dict[str, Any], out: Path) -> None:
    """Bank JSON and a losslessly JSON-encoded CSV view of the same reset rows."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "qualification.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    rows = summary["rows"]
    fieldnames = list(rows[0]) if rows else []
    with (out / "per_reset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value, sort_keys=True) for key, value in row.items()}
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity() -> dict[str, Any]:
    from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML

    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    external_root = Path(
        subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=Path(Z1_HAMMER_XML).parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    external_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=external_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    external_tracked_dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=no"),
            cwd=external_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    files = {
        "qualification_script": Path(__file__).resolve(),
        "guideline_geometry": root / "src/tasks/hammer/mdp/guideline.py",
        "scripted_reference": root / "src/tasks/hammer/mdp/references.py",
        "production_config": root / "src/tasks/hammer/config/z1/env_cfgs.py",
        "base_hammer_config": root / "src/tasks/hammer/hammer_env_cfg.py",
        "nail_entity": root / "src/tasks/hammer/nail_block.py",
        "z1_hammer_xml": Path(Z1_HAMMER_XML),
    }
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "external_safe_impact_manipulation": {
            "git_revision": external_revision,
            "tracked_dirty": external_tracked_dirty,
        },
        "runtime_versions": {
            distribution: importlib.metadata.version(distribution)
            for distribution in ("mjlab", "mujoco", "mujoco-warp", "torch")
        },
        "sha256": {name: _sha256_file(path) for name, path in files.items()},
    }


def _as_numpy(tensor):
    return tensor.detach().cpu().numpy().astype("float64", copy=True)


def run_cpu_qualification() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Replay the training-reset reference with complete 500 Hz instrumentation."""
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.registry import load_env_cfg

    import src.tasks  # noqa: F401 - populate the production task registry
    from src.assets.robots.unitree_z1.z1_constants import (
        ARM_JOINT_NAMES,
        HAMMER_HEAD_SITE_NAME,
        Z1_HAMMER_DELTA_POS_SCALE,
    )
    from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR
    from src.tasks.hammer.mdp.guideline import (
        _ENV_GUIDELINE_ATTR,
        next_gate_vector,
        project_to_reference,
    )
    from src.tasks.hammer.mdp.references import SingleStrikeReference
    from src.tasks.hammer.mdp.rewards import clamped_nail_depth

    cfg = load_qualification_cfg(load_env_cfg)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    robot = env.scene["robot"]
    nail = env.scene["nail_block"]
    contact_sensor = env.scene["hammer_nail_contact"]
    first_tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR)
    guideline_tracker = getattr(env, _ENV_GUIDELINE_ATTR)

    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    head_cfg.resolve(env.scene)
    nail_top_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    nail_top_cfg.resolve(env.scene)
    nail_joint_cfg = SceneEntityCfg("nail_block", joint_names=("nail_slide",))
    nail_joint_cfg.resolve(env.scene)

    def head_position():
        return robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)

    def nail_top_position():
        return nail.data.site_pos_w[:, nail_top_cfg.site_ids].squeeze(1)

    active: dict[str, Any] | None = None
    original_sim_step = env.sim.step
    original_substep = env.metrics_manager.compute_substep

    def preintegration_then_step() -> None:
        if active is not None:
            active["qvel_pre"].append(
                _as_numpy(robot.data.joint_vel[0, arm_cfg.joint_ids])
            )
        original_sim_step()

    def record_substep() -> None:
        if active is not None and bool(guideline_tracker.initialized[0]):
            index = int(guideline_tracker.next_gate[0])
            if index < REQUIRED_GATES and index not in active["gate_centers"]:
                center = head_position() + next_gate_vector(env)
                active["gate_centers"][index] = _as_numpy(center[0])

        original_substep()
        if active is None:
            return

        head = head_position()
        qvel = _as_numpy(robot.data.joint_vel[0, arm_cfg.joint_ids])
        depth = float(clamped_nail_depth(env, nail_joint_cfg)[0])
        accepted = bool(first_tracker.started[0])
        raw_contact = bool((contact_sensor.data.found[0] > 0).any())
        gates = int(guideline_tracker.next_gate[0])
        if bool(guideline_tracker.initialized[0]):
            _, error = project_to_reference(
                head, guideline_tracker.entry, guideline_tracker.nail
            )
            error_m = float(error[0])
        else:
            error_m = 0.0

        active["qvel_post"].append(qvel)
        active["path"].append(_as_numpy(head[0]))
        active["depth"].append(depth)
        active["samples"].append(
            {
                "gates_crossed": gates,
                "accepted_contact": accepted,
                "raw_contact": raw_contact,
                "error_m": error_m,
            }
        )
        if accepted and active["contact_point"] is None:
            active["contact_point"] = _as_numpy(head[0])
            active["contact_path_index"] = len(active["path"]) - 1

    env.sim.step = preintegration_then_step
    env.metrics_manager.compute_substep = record_substep
    traces: dict[int, dict[str, Any]] = {}

    def run_one(seed: int) -> dict[str, Any]:
        nonlocal active
        active = None
        env.reset(seed=seed)
        initial_depth = float(clamped_nail_depth(env, nail_joint_cfg)[0])
        reset_head = _as_numpy(head_position()[0])
        reset_arm_qpos = _as_numpy(
            robot.data.joint_pos[0, arm_cfg.joint_ids]
        ).tolist()
        reference = SingleStrikeReference(1, env.device)
        reference.update(head_position(), nail_top_position(), env.episode_length_buf)
        playback_length = reference.playback_length()
        trace: dict[str, Any] = {
            "seed": seed,
            "qvel_pre": [],
            "qvel_post": [],
            "path": [reset_head],
            "depth": [],
            "actions": [],
            "samples": [],
            "gate_centers": {},
            "contact_point": None,
            "contact_path_index": None,
            "actions_finite": True,
            "physics_dt_s": float(env.physics_dt),
            "control_decimation": int(env.cfg.decimation),
            "arm_joint_names": list(ARM_JOINT_NAMES),
        }
        active = trace
        for control_index in range(1, playback_length + HOLD_STEPS + 1):
            target = reference.playback_target(min(control_index, playback_length))
            action = ((target - head_position()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(
                -1.0, 1.0
            )
            trace["actions"].append(_as_numpy(action[0]))
            trace["actions_finite"] &= bool(torch.isfinite(action).all())
            env.step(action)
            if bool(env.reset_buf[0]):
                break
        active = None

        pre = np.asarray(trace["qvel_pre"], dtype=np.float64)
        post = np.asarray(trace["qvel_post"], dtype=np.float64)
        qvel = canonical_qvel_trace(pre, post)
        corridor_errors = corridor_window_errors(trace["samples"])
        depths = np.asarray(trace["depth"], dtype=np.float64)
        path = np.asarray(trace["path"], dtype=np.float64)
        gates = max((sample["gates_crossed"] for sample in trace["samples"]), default=0)
        accepted_contact = any(
            sample["accepted_contact"] for sample in trace["samples"]
        )
        raw_contact = any(sample["raw_contact"] for sample in trace["samples"])
        entry = _as_numpy(guideline_tracker.entry[0])
        frozen_nail = _as_numpy(guideline_tracker.nail[0])
        finite = trace_numeric_is_finite(
            trace,
            initial_depth=initial_depth,
            reset_head=reset_head,
            reset_arm_qpos=reset_arm_qpos,
            entry=entry,
            nail=frozen_nail,
        )
        qvel_peak = float(np.max(np.abs(qvel))) if qvel.size else 0.0
        nail_progress = (
            float(np.max(depths) - initial_depth) if depths.size else 0.0
        )
        corridor_max = max(corridor_errors, default=0.0)
        trace["entry"] = entry
        trace["nail"] = frozen_nail
        trace["gate_centers"] = [
            trace["gate_centers"][index]
            for index in sorted(trace["gate_centers"])
        ]
        traces[seed] = trace
        return {
            "seed": seed,
            "gates_crossed": gates,
            "contact_seen": accepted_contact,
            "raw_contact_seen": raw_contact,
            "nail_progress_m": nail_progress,
            "corridor_max_m": corridor_max,
            "qvel_peak_rad_s": qvel_peak,
            "finite": finite,
            "reset_arm_qpos_rad": reset_arm_qpos,
            "substeps": len(post),
        }

    try:
        rows = run_required_seeds(run_one)
    finally:
        env.sim.step = original_sim_step
        env.metrics_manager.compute_substep = original_substep
        env.close()
    return rows, traces


def _corridor_polygon(entry, nail, axes: tuple[int, int], radius: float):
    import numpy as np

    segment = nail[list(axes)] - entry[list(axes)]
    length = float(np.linalg.norm(segment))
    if length == 0.0:
        return None
    normal = np.array((-segment[1], segment[0])) / length * radius
    start = entry[list(axes)]
    end = nail[list(axes)]
    return np.stack((start + normal, end + normal, end - normal, start - normal))


def _gate_disk_points(center, entry, nail, radius: float):
    """Sample a 3-D disk perpendicular to the frozen production guideline."""
    import numpy as np

    direction = nail - entry
    direction = direction / np.linalg.norm(direction)
    helper = np.array((0.0, 0.0, 1.0))
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.array((0.0, 1.0, 0.0))
    basis_a = np.cross(direction, helper)
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(direction, basis_a)
    angle = np.linspace(0.0, 2.0 * np.pi, 65)
    return center + radius * (
        np.cos(angle)[:, None] * basis_a + np.sin(angle)[:, None] * basis_b
    )


def representative_plot_title(summary: dict[str, Any]) -> str:
    row = next(row for row in summary["rows"] if row["seed"] == 1000)
    seed_status = "PASS" if row["qualifies"] else "FAIL"
    aggregate_status = "PASS" if summary["passed"] else "FAIL"
    return (
        f"Representative 1/16, seed 1000 (chosen a priori): {seed_status} "
        f"({row['corridor_max_m'] * 1000:.3f} mm vs 5.000 mm)\n"
        f"Aggregate {summary['qualified_resets']}/16: {aggregate_status} | "
        "production guideline: frozen reset-head anchor to frozen nail top | "
        "scored interval: gate 1 to accepted contact"
    )


def _configure_projection_x_ticks(ax: Any, *, centered: bool) -> None:
    """Keep the narrow x-z projection label readable at artifact resolution."""
    if centered:
        left, right = ax.get_xlim()
        ax.set_xticks([(left + right) / 2.0])
        return

    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))


def plot_representative_trace(
    trace: dict[str, Any], summary: dict[str, Any], path: Path
) -> None:
    """Plot the seed-1000 production guideline and realized substep trace."""
    import matplotlib.pyplot as plt
    import numpy as np

    from src.tasks.hammer.mdp.guideline import GUIDELINE_GATE_RADIUS_M

    realized = np.asarray(trace["path"], dtype=np.float64)
    entry = np.asarray(trace["entry"], dtype=np.float64)
    nail = np.asarray(trace["nail"], dtype=np.float64)
    gates = np.asarray(trace["gate_centers"], dtype=np.float64)
    contact = trace["contact_point"]
    if len(gates) != REQUIRED_GATES:
        raise RuntimeError(f"representative plot needs six production gates, got {len(gates)}")
    gate_one_sample = next(
        (
            index
            for index, sample in enumerate(trace["samples"])
            if int(sample["gates_crossed"]) >= 1
        ),
        None,
    )
    if gate_one_sample is None:
        raise RuntimeError("representative plot needs a credited gate-1 transition")
    gate_one_path_index = gate_one_sample + 1
    contact_index = trace["contact_path_index"]
    if contact_index is not None:
        realized = realized[: int(contact_index) + 1]
    pre_gate_path = realized[: gate_one_path_index + 1]
    scored_path = realized[gate_one_path_index:]

    fig, axes_array = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(left=0.07, right=0.77, bottom=0.12, top=0.78, wspace=0.30)
    for panel_index, (ax, projection, labels) in enumerate(
        zip(
            axes_array,
            ((0, 2), (0, 1)),
            (("x (m)", "z (m)"), ("x (m)", "y (m)")),
        )
    ):
        polygon = _corridor_polygon(entry, nail, projection, CORRIDOR_RADIUS_M)
        if polygon is not None:
            ax.fill(
                polygon[:, 0],
                polygon[:, 1],
                color="tab:green",
                alpha=0.18,
                label="5 mm-radius corridor",
            )
        ax.plot(
            [entry[projection[0]], nail[projection[0]]],
            [entry[projection[1]], nail[projection[1]]],
            "k--",
            linewidth=1.8,
            label="production guideline",
        )
        for index, center in enumerate(gates):
            disk = _gate_disk_points(
                center, entry, nail, GUIDELINE_GATE_RADIUS_M
            )
            ax.plot(
                disk[:, projection[0]],
                disk[:, projection[1]],
                color="tab:orange",
                linewidth=1.0,
                alpha=0.8,
                label="production-derived gates" if index == 0 else None,
            )
        ax.plot(
            pre_gate_path[:, projection[0]],
            pre_gate_path[:, projection[1]],
            color="0.6",
            linewidth=1.2,
            linestyle=":",
            label="pre-gate-1 path (not scored)",
        )
        ax.plot(
            scored_path[:, projection[0]],
            scored_path[:, projection[1]],
            color="tab:blue",
            linewidth=1.8,
            label="scored gate 1 to accepted contact",
        )
        if contact is not None:
            contact_array = np.asarray(contact)
            ax.scatter(
                contact_array[projection[0]],
                contact_array[projection[1]],
                marker="x",
                s=70,
                linewidth=2.0,
                color="tab:red",
                label="first accepted contact",
                zorder=5,
            )
        ax.scatter(entry[projection[0]], entry[projection[1]], color="black", s=22)
        ax.scatter(nail[projection[0]], nail[projection[1]], color="black", s=22)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.autoscale_view()
        _configure_projection_x_ticks(ax, centered=panel_index == 0)
    axes_array[1].legend(
        fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0
    )
    fig.suptitle(representative_plot_title(summary), y=0.97, fontsize=14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows, traces = run_cpu_qualification()
    summary = summarize_qualification(rows)
    summary.update(
        {
            "task_id": TASK_ID,
            "configuration": {
                "play": False,
                "device": "cpu",
                "num_envs": 1,
                "auto_reset": False,
                "fixed_impedance": True,
                "imp_max_p": 0.0,
                "reset_position_range_rad": list(RESET_POSITION_RANGE_RAD),
                "reset_arm_joint_names": traces[1000]["arm_joint_names"],
            },
            "sampling": {
                "physics_dt_s": traces[1000]["physics_dt_s"],
                "control_decimation": traces[1000]["control_decimation"],
                "qvel_states": "preintegration[:] plus terminal postintegration[-1]",
                "path": "postintegration production-metrics substeps",
            },
            "contact_gate": "production FirstStrikeEventTracker.started accepted onset",
            "corridor_window": "production gate-1 transition through accepted onset inclusive",
            "representative_plot_seed": 1000,
            "source_identity": _source_identity(),
        }
    )
    write_result_tables(summary, args.out)
    plot_representative_trace(
        traces[1000], summary, args.out / "reference_xz_xy.png"
    )
    summary["artifact_sha256"] = {
        "per_reset.csv": _sha256_file(args.out / "per_reset.csv"),
        "reference_xz_xy.png": _sha256_file(args.out / "reference_xz_xy.png"),
    }
    (args.out / "qualification.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    print("seed gates contact nail_mm corridor_mm qvel_rad_s finite pass")
    for row in summary["rows"]:
        print(
            f"{row['seed']:4d} {row['gates_crossed']:5d} "
            f"{str(row['contact_seen']):>7s} {row['nail_progress_m'] * 1000:7.3f} "
            f"{row['corridor_max_m'] * 1000:11.3f} "
            f"{row['qvel_peak_rad_s']:10.4f} {str(row['finite']):>6s} "
            f"{str(row['qualifies']):>5s}"
        )
    print(
        f"qualification: {summary['qualified_resets']}/{summary['required_resets']} "
        f"PASS={summary['passed']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
