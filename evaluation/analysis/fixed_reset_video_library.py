"""Pure contracts for the deterministic 56-policy fixed-reset video library."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


EXPECTED = {
    "fq4x8": {
        "F8": range(8, 16),
        "F0": range(8, 16),
        "D0": range(8, 16),
        "FQ-min": range(8, 16),
    },
    "fq3x8": {
        "F8": range(16, 24),
        "B8": range(16, 24),
        "FQ": range(16, 24),
    },
}

SOURCE_ROOT = Path(__file__).resolve().parents[2]
FIXED_RESET_ENVELOPE = SOURCE_ROOT / (
    "docs/results/assets/2026-07-29_lambda_feasibility_stage0/"
    "lambda_feasibility_stage0_inputs.json"
)
APPROVED_FIXED_RESET_DIGEST = (
    "bde511ec2adc42e5365e1e46f45ff1fb43223a93c31c6fbfb4580352e445e319"
)
RENDERER_SOURCE_SCOPE = ("src", "scripts", "evaluation/analysis")
RENDERER_ASSET_SCOPE = ("hammer_z1_env/assets",)
RENDERER_PROVENANCE_FIELDS = frozenset(
    ("code_revision", "asset_revision", "source_scope", "asset_scope", "device")
)
ARTIFACT_FILENAMES = ("policy.mp4", "montage.png", "trajectory.png", "trace.npz")
RENDERER_CONTRACT = {
    "version": 1,
    "frame_width_px": 960,
    "frame_height_px": 720,
    "camera_distance_m": 0.85,
    "camera_elevation_deg": -25.0,
    "camera_azimuth_deg": 135.0,
    "fps": 10,
}
TIMING_CONTRACT = {
    "physics_dt_s": 0.002,
    "control_decimation": 10,
    "control_dt_s": 0.02,
}
SUBSTEP_RENDERER_CONTRACT = {
    **RENDERER_CONTRACT,
    "fps": 50,
    "frame_substep_stride": 2,
}
SUBSTEP_TRACE_KEYS = (
    "substep_head_position_m",
    "substep_contact",
    "substep_nail_depth_m",
    "substep_arm_qvel_rad_s",
    "substep_arm_qvel_pre_rad_s",
    "arm_joint_names",
    "substep_gate_index",
    "substep_perpendicular_error_m",
    "substep_d_start_m",
    "substep_f_best",
    "substep_control_step",
    "substep_is_control_boundary",
    "substep_episode_index",
    "control_step_reward_terms",
    "control_step_reward_term_names",
    "control_step_reward_total",
    "guideline_entry_m",
    "guideline_nail_m",
    "guideline_gate_centers_m",
    "guideline_reference_length_m",
    "physics_dt_s",
    "control_decimation",
    "executed_control_steps",
)
WAVE1_ARTIFACT_FILENAMES = ("policy.mp4", "montage.png", "trajectory.png", "trace.npz")
WAVE1_REVISION_FIELDS = ("training_revision", "asset_revision", "analysis_revision")
# Wave 1 recorded no execution device; its CPU provenance rests on a documented
# inferred bridge (rendered on a CPU-only machine, bit-identical shared channels
# against the hard-guarded CPU diagnostic). Every campaign AFTER wave1 must record
# the device directly, so the bridge is never needed again. "requested" is what the
# renderer was asked for; the two "actual_*" fields are read back off the live env
# and a real tensor, which is what catches a silent accelerator promotion.
EXECUTION_DEVICE_FIELDS = (
    "requested",
    "actual_env_device",
    "actual_tensor_device",
    "platform",
)
# Exempt = every campaign that predates the field, not merely wave1: fq4x8/fq3x8
# artifacts were rendered before it existed, so requiring it would reject them
# retroactively. Only campaigns from wave2 onward must record a device.
EXECUTION_DEVICE_CAMPAIGN_EXEMPT = frozenset({"wave1", "fq4x8", "fq3x8"})
# Mirrors src.tasks.hammer.mdp.guideline.GUIDELINE_GATE_RADIUS_M; duplicated so this
# module stays import-light. test_wave1_gate_disk_radius_matches_the_tracker guards drift.
GUIDELINE_GATE_RADIUS_M = 0.015
TASK_BY_CAMPAIGN_ARM = {
    ("wave1", "C0"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
    ("wave1", "G"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate",
    ("wave1", "P"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress",
    ("wave2", "C0"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
    ("wave2", "G"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate",
    ("wave2", "P"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress",
    ("fq4x8", "F8"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    ("fq4x8", "F0"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
    ("fq4x8", "D0"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
    ("fq4x8", "FQ-min"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
    ("fq3x8", "F8"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    ("fq3x8", "B8"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
    ("fq3x8", "FQ"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
    ("wave3", "P+V"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel"
    ),
    ("presentation3", "P+V"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel"
    ),
    ("presentation3", "P+D4"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Delivered4"
    ),
    ("presentation3", "P+V+D4"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered4"
    ),
    ("impulse6", "s0d0"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0"
    ),
    ("impulse6", "s0d4"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4"
    ),
    ("impulse6", "s0d16"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16"
    ),
    ("impulse6", "s8d0"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0"
    ),
    # This centre cell is intentionally the existing presentation3 P+V+D4 task.
    ("impulse6", "s8d4"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered4"
    ),
    ("impulse6", "s8d16"): (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16"
    ),
}

# --- Treatment-faithful plot semantics ----------------------------------------------------------
# A figure asserts a treatment. Drawing gate disks for an arm that was never paid for crossing a
# gate is a false claim about what the policy learned, so geometry and title are derived from the
# validated TASK (and, via the drift guard in tests, from the registered reward configuration) --
# never from a directory name.
#   none      trajectory only: the arm received no guidance payout at all.
#   gates     entry->nail line + six 15 mm disks: r_gate pays a one-shot pulse for CROSSING a disk,
#             so the radius is the tolerance the policy was actually rewarded against.
#   waypoints entry->nail line + six ordered point markers: r_waypoint_progress pays shaped credit
#             for APPROACHING ordered target points; there is no radius in its payout, so drawing
#             disks would invent a tolerance the reward never had.
GEOMETRY_NONE = "none"
GEOMETRY_GATES = "gates"
GEOMETRY_WAYPOINTS = "waypoints"
GUIDELINE_GATE_COUNT = 6
QVEL_LIMIT_RAD_S = 3.1415
_GEOMETRY_CAPTION = {
    GEOMETRY_NONE: "no guidance geometry drawn (none was rewarded)",
    GEOMETRY_GATES: "dashed black = tracker entry->nail · blue disks = 15 mm gates",
    GEOMETRY_WAYPOINTS: (
        "dashed black = tracker entry->nail · numbered diamonds = ordered waypoints"
    ),
}


@dataclasses.dataclass(frozen=True)
class Treatment:
    """What a policy was actually trained on — the only honest source for plot geometry."""

    headline: str
    guidance: str
    velocity: str
    impulse: str
    geometry: str


_VELOCITY_MEASURED_ONLY = "velocity measured, not enforced"
_VELOCITY_SOFT_CAT = "velocity max_p 0.5 (500 Hz substep peak)"
_IMPULSE_LOG_ONLY = "impulse log-only"
_NO_GUIDELINE = Treatment(
    headline="unguided strike (pre-guideline campaign)",
    guidance="no guideline state, no guidance reward",
    velocity=_VELOCITY_MEASURED_ONLY,
    impulse=_IMPULSE_LOG_ONLY,
    geometry=GEOMETRY_NONE,
)
TREATMENT_BY_TASK: dict[str, Treatment] = {
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0": Treatment(
        headline="unguided strike (guideline state observed only)",
        guidance="guideline state observed; no guidance reward",
        velocity=_VELOCITY_MEASURED_ONLY,
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_NONE,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate": Treatment(
        headline="gate-guided strike",
        guidance="ordered gate reward, weight 8.0",
        velocity=_VELOCITY_MEASURED_ONLY,
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_GATES,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress": Treatment(
        headline="progress-guided strike",
        guidance="progress weight 8.0",
        velocity=_VELOCITY_MEASURED_ONLY,
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_WAYPOINTS,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel": Treatment(
        headline="progress-guided strike with soft velocity-CaT",
        guidance="progress weight 8.0",
        velocity=_VELOCITY_SOFT_CAT,
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_WAYPOINTS,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Delivered4": Treatment(
        headline="progress-guided strike with delivered weight 4.0",
        guidance="progress weight 8.0",
        velocity=_VELOCITY_MEASURED_ONLY,
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_WAYPOINTS,
    ),
    (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered4"
    ): Treatment(
        headline=(
            "impulse screen S=8 D=4 · I-CaT log-only: progress-guided strike with soft velocity-CaT "
            "and delivered weight 4.0"
        ),
        guidance="waypoint w=8 (progress weight 8.0)",
        velocity="V-CaT 0.5 @ 500 Hz (soft velocity-CaT; max_p 0.5)",
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_WAYPOINTS,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0": Treatment(
        headline="impulse screen S=0 D=0 · I-CaT log-only",
        guidance="waypoint w=8 (progress weight 8.0)",
        velocity="V-CaT 0.5 @ 500 Hz (soft velocity-CaT; max_p 0.5)",
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_WAYPOINTS,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4": Treatment(
        headline="impulse screen S=0 D=4 · I-CaT log-only",
        guidance="waypoint w=8 (progress weight 8.0)",
        velocity="V-CaT 0.5 @ 500 Hz (soft velocity-CaT; max_p 0.5)",
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_WAYPOINTS,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16": Treatment(
        headline="impulse screen S=0 D=16 · I-CaT log-only",
        guidance="waypoint w=8 (progress weight 8.0)",
        velocity="V-CaT 0.5 @ 500 Hz (soft velocity-CaT; max_p 0.5)",
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_WAYPOINTS,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0": Treatment(
        headline="impulse screen S=8 D=0 · I-CaT log-only",
        guidance="waypoint w=8 (progress weight 8.0)",
        velocity="V-CaT 0.5 @ 500 Hz (soft velocity-CaT; max_p 0.5)",
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_WAYPOINTS,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16": Treatment(
        headline="impulse screen S=8 D=16 · I-CaT log-only",
        guidance="waypoint w=8 (progress weight 8.0)",
        velocity="V-CaT 0.5 @ 500 Hz (soft velocity-CaT; max_p 0.5)",
        impulse=_IMPULSE_LOG_ONLY,
        geometry=GEOMETRY_WAYPOINTS,
    ),
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear": _NO_GUIDELINE,
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0": _NO_GUIDELINE,
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0": _NO_GUIDELINE,
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality": _NO_GUIDELINE,
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded": _NO_GUIDELINE,
}


def treatment_for_task(task: str) -> Treatment:
    """Return the plot treatment for a registered task, or fail closed."""
    try:
        return TREATMENT_BY_TASK[task]
    except KeyError as error:
        raise ValueError(f"unregistered task treatment identity: {task}") from error


def trajectory_outcome(trace: Mapping[str, Any], *, success: bool) -> dict:
    """Read the plotted outcome off the trace itself, never off a caller's claim."""
    return {
        "gates": int(np.asarray(trace["substep_gate_index"]).max()),
        "peak_qvel_rad_s": float(
            np.abs(np.asarray(trace["substep_arm_qvel_rad_s"], dtype=float)).max()
        ),
        "success": bool(success),
    }


def compose_trajectory_title(
    *,
    campaign: str,
    arm: str,
    training_seed: int,
    treatment: Treatment,
    outcome: Mapping[str, Any],
) -> str:
    """Three lines: who this is, what it was trained on, and what it actually did."""
    peak = float(outcome["peak_qvel_rad_s"])
    return (
        f"{campaign} · {arm} · seed {training_seed} · {treatment.headline}\n"
        f"Training: {treatment.guidance} · {treatment.velocity} · {treatment.impulse}\n"
        f"Result: gates {int(outcome['gates'])}/{GUIDELINE_GATE_COUNT}"
        f" · peak |q̇| {peak:.4f}/{QVEL_LIMIT_RAD_S} rad/s"
        f" · {'success' if outcome['success'] else 'no success'}"
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_task(campaign: str, arm: str) -> str:
    """Return the single task allowed for a registered campaign/treatment arm."""
    try:
        return TASK_BY_CAMPAIGN_ARM[(campaign, arm)]
    except KeyError as error:
        raise ValueError(f"unregistered campaign/arm task identity: {campaign}/{arm}") from error


def _reset_record(payload: Mapping[str, Any]) -> dict:
    if "resets" in payload:
        resets = payload["resets"]
        if not isinstance(resets, Mapping) or not isinstance(resets.get("fixed_reset"), Mapping):
            raise ValueError("reset envelope has no fixed_reset record")
        payload = resets["fixed_reset"]
    return dict(payload)


def load_fixed_reset(path: str | Path = FIXED_RESET_ENVELOPE) -> dict:
    """Load the shared reset record and fail closed if its realized-state digest drifts."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read fixed reset envelope: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("fixed reset envelope must be a JSON object")
    record = _reset_record(payload)
    reset_state = record.get("reset_state")
    if not isinstance(reset_state, Mapping) or not isinstance(reset_state.get("realized"), Mapping):
        raise ValueError("fixed reset record has no realized reset state")
    digest = hashlib.sha256(_canonical_json(reset_state["realized"])).hexdigest()
    if digest != APPROVED_FIXED_RESET_DIGEST:
        raise ValueError("fixed reset state digest is not the approved digest")
    if record.get("reset_state_digest") != digest:
        raise ValueError("fixed reset state digest mismatch")
    return record


def validate_inventory(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require the exact registered 56-policy membership and unique checkpoints."""
    try:
        identities = {
            (str(row["campaign"]), str(row["arm"]), int(row["training_seed"]))
            for row in rows
        }
        checkpoints = {str(row["checkpoint_sha256"]) for row in rows}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint membership or uniqueness mismatch") from error
    expected = {
        (campaign, arm, seed)
        for campaign, arms in EXPECTED.items()
        for arm, seeds in arms.items()
        for seed in seeds
    }
    if identities != expected or len(rows) != 56 or len(checkpoints) != 56:
        raise ValueError("checkpoint membership or uniqueness mismatch")


def first_episode_frame_count(episode_lengths: Sequence[int]) -> int:
    """Count post-step frames strictly before an auto-reset boundary."""
    for index, length in enumerate(episode_lengths):
        if int(length) == 0:
            return index
    return len(episode_lengths)


def _trace_geometry(trace: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the sampled path plus fixed-reset geometry needed to interpret it."""
    positions = np.asarray(trace["head_position_m"], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("trace head_position_m must have shape [steps, 3]")
    if not np.isfinite(positions).all():
        raise ValueError("trace head_position_m must be finite")
    contact = np.asarray(trace.get("contact", np.zeros(len(positions), dtype=bool)), dtype=bool)
    if contact.shape != (len(positions),):
        raise ValueError("trace contact must have one value per head position")
    reference = np.asarray(trace.get("reference_polyline_m"), dtype=float)
    if reference.shape not in ((2, 3), (3, 3)) or not np.isfinite(reference).all():
        raise ValueError("trace reference_polyline_m must be finite with shape (2, 3) or (3, 3)")
    nail_top = np.asarray(trace.get("nail_top_m"), dtype=float)
    if nail_top.shape != (3,) or not np.isfinite(nail_top).all():
        raise ValueError("trace nail_top_m must be finite with shape (3,)")
    return positions, contact, reference, nail_top


def _draw_trajectory(
    axis: Any,
    positions: np.ndarray,
    contact: np.ndarray,
    reference: np.ndarray,
    nail_top: np.ndarray,
    ordinate: int,
) -> None:
    """Draw one view; the reference is contextual evidence, never a reward claim."""
    axis.plot(
        reference[:, 0],
        reference[:, ordinate],
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="SingleStrikeReference (observation only)",
        zorder=1,
    )
    if len(positions) > 1:
        points = positions[:, [0, ordinate]].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        axis.add_collection(
            LineCollection(
                segments,
                cmap="viridis",
                array=np.linspace(0.0, 1.0, len(segments)),
                linewidth=1.5,
                zorder=2,
            )
        )
    axis.scatter(positions[0, 0], positions[0, ordinate], c="#2ca02c", s=24, zorder=4)
    if contact.any():
        axis.scatter(positions[contact, 0], positions[contact, ordinate], c="#d62728", s=15, zorder=5)
    if ordinate == 2:
        axis.plot(
            [nail_top[0], nail_top[0]],
            [nail_top[2], nail_top[2] - 0.06],
            color="#8c564b",
            linewidth=2.0,
            zorder=0,
        )
    else:
        axis.scatter(nail_top[0], nail_top[1], c="#8c564b", marker="+", s=40, zorder=3)


def write_trajectory_png(trace: Mapping[str, Any], path: str | Path) -> None:
    """Write x-z and x-y diagnostics with the anchored reference as context only."""
    positions, contact, reference, nail_top = _trace_geometry(trace)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=False)
    for axis, ordinate, label in ((axes[0], 2, "z (m)"), (axes[1], 1, "y (m)")):
        _draw_trajectory(axis, positions, contact, reference, nail_top, ordinate)
        axis.set_xlabel("x (m)")
        axis.set_ylabel(label)
        axis.set_aspect("auto")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.25)
    axes[0].set_title("x-z trajectory")
    axes[1].set_title("x-y trajectory")
    fig.suptitle("Hammer-head trajectory · green=start · viridis=normalized time · red=contact")
    fig.text(
        0.5,
        0.035,
        "SingleStrikeReference (observation only) · black dashed · not rewarded",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0.02, 0.11, 0.98, 0.90])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _grid_limits(traces: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    xz = np.concatenate(
        [
            np.vstack((
                np.asarray(trace["head_position_m"], dtype=float)[:, [0, 2]],
                np.asarray(trace["reference_polyline_m"], dtype=float)[:, [0, 2]],
                np.asarray(trace["nail_top_m"], dtype=float)[None, [0, 2]],
            ))
            for trace in traces
        ]
    )
    span = np.ptp(xz, axis=0)
    padding = np.maximum(span * 0.05, 0.01)
    return (
        (float(xz[:, 0].min() - padding[0]), float(xz[:, 0].max() + padding[0])),
        (float(xz[:, 1].min() - padding[1]), float(xz[:, 1].max() + padding[1])),
    )


def _library_grid_limits(root: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute the one x/z limit pair shared by all 56 comparison panels."""
    traces: list[dict[str, np.ndarray]] = []
    for campaign, arms in EXPECTED.items():
        for arm, seeds in arms.items():
            for seed in seeds:
                trace_path = root / campaign / arm / str(seed) / "trace.npz"
                try:
                    with np.load(trace_path) as loaded:
                        trace = {key: np.asarray(loaded[key]) for key in loaded.files}
                except (OSError, ValueError, KeyError) as error:
                    raise ValueError(f"campaign trace unreadable: {trace_path}") from error
                _trace_geometry(trace)
                traces.append(trace)
    return _grid_limits(traces)


def write_campaign_trajectory_grid(
    root: str | Path, campaign: str, path: str | Path
) -> dict[str, Any]:
    """Write one common-axis x-z panel per registered arm/seed in a campaign."""
    if campaign not in EXPECTED:
        raise ValueError(f"unregistered campaign: {campaign}")
    root = Path(root)
    panels: list[tuple[str, int, dict[str, np.ndarray]]] = []
    for arm, seeds in EXPECTED[campaign].items():
        for seed in seeds:
            trace_path = root / campaign / arm / str(seed) / "trace.npz"
            try:
                with np.load(trace_path) as loaded:
                    trace = {key: np.asarray(loaded[key]) for key in loaded.files}
            except (OSError, ValueError, KeyError) as error:
                raise ValueError(f"campaign trace unreadable: {trace_path}") from error
            _trace_geometry(trace)
            panels.append((arm, seed, trace))
    xlim, zlim = _library_grid_limits(root)
    arms = tuple(EXPECTED[campaign])
    seeds = tuple(next(iter(EXPECTED[campaign].values())))
    fig, axes = plt.subplots(
        len(arms), len(seeds), figsize=(2.0 * len(seeds), 2.0 * len(arms)), sharex=True, sharey=True
    )
    panel_limits: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row, arm in enumerate(arms):
        for column, seed in enumerate(seeds):
            axis = axes[row, column]
            trace = next(value for panel_arm, panel_seed, value in panels if (panel_arm, panel_seed) == (arm, seed))
            positions, contact, reference, nail_top = _trace_geometry(trace)
            _draw_trajectory(axis, positions, contact, reference, nail_top, 2)
            axis.set_xlim(xlim)
            axis.set_ylim(zlim)
            panel_limits.append((tuple(axis.get_xlim()), tuple(axis.get_ylim())))
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.set_title(f"{arm} · seed {seed}", fontsize=7)
            axis.tick_params(labelsize=6)
    fig.suptitle(
        f"{campaign}: fixed-reset hammer-head trajectories (x-z)\n"
        "green=start · viridis=normalized time · red=contact · brown=nail axis\n"
        "black dashed=SingleStrikeReference (observation only; r_imit disabled/not rewarded)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "common_xlim": xlim,
        "common_zlim": zlim,
        "panel_limits": tuple(panel_limits),
    }


def write_library_index(
    root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    generation_command: str,
    validation_command: str,
) -> Path:
    """Write the clickable, hash-bound index for a fully rendered library."""
    validate_inventory(rows)
    if not generation_command or not validation_command:
        raise ValueError("generation and validation commands are required")
    root = Path(root)
    grids = ("fq4x8_trajectories_grid.png", "fq3x8_trajectories_grid.png")
    for grid in grids:
        if not (root / grid).is_file():
            raise ValueError(f"index grid link is missing: {root / grid}")
    lines = [
        "# Fixed-reset 56-policy video library",
        "",
        "Fixed-reset qualitative comparison only; this is neither a reward/impulse causal analysis nor a best-episode selection.",
        "",
        "The dashed `SingleStrikeReference (observation only)` line in each trajectory was available as an observation, but none of these 56 arms enabled the separate `r_imit` tracking reward. It is neither an optimal path nor a rewarded path.",
        "black dashed=SingleStrikeReference (observation only; r_imit disabled/not rewarded)",
        "Each policy's `trajectory.png` contains the corresponding x-y top view beside the x-z side view.",
        "",
        "Rollout identity (`code_revision` and `renderer_provenance`) is recorded separately from `presentation_generator_revision`, which identifies the code that generated `trajectory.png`.",
        "",
        "The accepted historical FQ3×8 manifest has no `clean_state` field, so source-tree cleanliness is unavailable and is not retroactively certified for those 24 rows. Their checkpoint, training revision, asset revision, manifest, and content-hash identities remain frozen.",
        "",
        "Large MP4 and per-policy media remain local and untracked under the repository's existing no-tracked-MP4 convention. This README and the two compact comparison grids are intended for named-file banking.",
        "",
        "- [FQ4×8 trajectory grid](fq4x8_trajectories_grid.png)",
        "- [FQ3×8 trajectory grid](fq3x8_trajectories_grid.png)",
        "",
        "## Reproduction",
        "",
        f"Generation command: `{generation_command}`",
        "",
        f"Validation command: `{validation_command}`",
        "",
        "## Output SHA-256",
        "",
        *[f"- `{grid}`: `{_sha256(root / grid)}`" for grid in grids],
        "",
        "| policy | checkpoint SHA-256 | output SHA-256 | artifacts and provenance |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        leaf = policy_artifact_dir(root, row)
        metadata_path = leaf / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"artifact metadata is unreadable: {metadata_path}") from error
        if metadata.get("checkpoint_sha256") != row["checkpoint_sha256"]:
            raise ValueError(f"artifact checkpoint hash mismatch: {leaf}")
        output_names = (*ARTIFACT_FILENAMES, "metadata.json")
        for name in output_names:
            if not (leaf / name).is_file():
                raise ValueError(f"index artifact link is missing: {leaf / name}")
        relative = leaf.relative_to(root).as_posix()
        links = " · ".join(
            f"[{name}]({relative}/{name})" for name in output_names
        )
        hashes = "<br>".join(f"{name}: `{_sha256(leaf / name)}`" for name in output_names)
        lines.append(
            f"| {row['campaign']}/{row['arm']}/{row['training_seed']} | "
            f"{row['checkpoint_sha256']} | {hashes} | {links} |"
        )
    index = root / "README.md"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def _metadata_digest(metadata: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in metadata.items() if key != "metadata_payload_sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def write_metadata(path: str | Path, metadata: Mapping[str, Any]) -> dict:
    """Write provenance with a canonical checksum over the metadata payload itself."""
    payload = dict(metadata)
    payload["metadata_payload_sha256"] = _metadata_digest(payload)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def policy_artifact_dir(root: str | Path, row: Mapping[str, Any]) -> Path:
    return Path(root) / str(row["campaign"]) / str(row["arm"]) / str(int(row["training_seed"]))


def _readable_artifact(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"artifact is missing or empty: {path}")
    try:
        if path.suffix == ".png":
            image = iio.imread(path)
            if np.asarray(image).size == 0:
                raise ValueError("empty image")
        elif path.suffix == ".mp4":
            frame = next(iter(iio.imiter(path)))
            if np.asarray(frame).size == 0:
                raise ValueError("empty video")
        elif path.suffix == ".npz":
            with np.load(path) as trace:
                _trace_geometry({key: np.asarray(trace[key]) for key in trace.files})
    except (OSError, StopIteration, ValueError, KeyError) as error:
        raise ValueError(f"artifact is unreadable: {path}: {error}") from error


def _validate_library_tree(root: Path) -> None:
    """Reject unregistered policy directories while permitting campaign files."""
    for child in root.iterdir():
        if child.is_dir() and child.name not in EXPECTED:
            raise ValueError(f"unexpected campaign directory: {child}")
    for campaign, arms in EXPECTED.items():
        campaign_dir = root / campaign
        if not campaign_dir.is_dir():
            raise ValueError(f"missing campaign directory: {campaign_dir}")
        for arm_dir in campaign_dir.iterdir():
            if not arm_dir.is_dir():  # documented campaign-level files are allowed.
                continue
            if arm_dir.name not in arms:
                raise ValueError(f"unexpected arm directory: {arm_dir}")
            expected_seeds = {str(seed) for seed in arms[arm_dir.name]}
            for leaf in arm_dir.iterdir():
                if leaf.is_dir() and leaf.name not in expected_seeds:
                    raise ValueError(f"unexpected policy leaf directory: {leaf}")
        for arm, seeds in arms.items():
            arm_dir = campaign_dir / arm
            if not arm_dir.is_dir():
                raise ValueError(f"missing arm directory: {arm_dir}")
            for seed in seeds:
                leaf = arm_dir / str(seed)
                if not leaf.is_dir():
                    raise ValueError(f"missing policy leaf directory: {leaf}")
                if any(child.is_dir() for child in leaf.iterdir()):
                    raise ValueError(f"unexpected nested policy directory: {leaf}")


def _is_revision(value: object) -> bool:
    text = str(value)
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_metadata_provenance(
    metadata: Mapping[str, Any],
    row: Mapping[str, Any],
    leaf: Path,
    *,
    expected_renderer_provenance: Mapping[str, Any] | None = None,
) -> None:
    """Cross-bind rollout and training identities for final and resume validation."""
    renderer = metadata.get("renderer_provenance")
    if not isinstance(renderer, Mapping):
        raise ValueError(f"artifact renderer provenance missing: {leaf}")
    if (
        set(renderer) != RENDERER_PROVENANCE_FIELDS
        or renderer.get("source_scope") != list(RENDERER_SOURCE_SCOPE)
        or renderer.get("asset_scope") != list(RENDERER_ASSET_SCOPE)
    ):
        raise ValueError(f"artifact renderer provenance scope mismatch: {leaf}")
    for key in ("code_revision", "asset_revision"):
        if not _is_revision(renderer.get(key)) or renderer.get(key) != metadata.get(key):
            raise ValueError(f"artifact renderer provenance {key} mismatch: {leaf}")
    if renderer.get("device") != "cpu":
        raise ValueError(f"artifact renderer provenance must use CPU: {leaf}")
    if expected_renderer_provenance is not None and dict(renderer) != dict(
        expected_renderer_provenance
    ):
        raise ValueError(f"artifact renderer provenance mismatch: {leaf}")

    training = metadata.get("training_provenance")
    fields = (
        "checkpoint_path",
        "training_code_revision",
        "training_asset_revision",
        "accepted_training_manifest_sha256",
    )
    try:
        expected_training = {key: row[key] for key in fields}
    except KeyError as error:
        raise ValueError(f"artifact training provenance inventory incomplete: {leaf}") from error
    if (
        not isinstance(training, Mapping)
        or any(training.get(key) != expected_training[key] for key in fields)
        or not isinstance(expected_training["checkpoint_path"], str)
        or not expected_training["checkpoint_path"]
        or not _is_revision(expected_training["training_code_revision"])
        or not _is_revision(expected_training["training_asset_revision"])
        or not _is_sha256(expected_training["accepted_training_manifest_sha256"])
    ):
        raise ValueError(f"artifact training provenance mismatch: {leaf}")


def _validate_metadata_contract(
    metadata: Mapping[str, Any],
    row: Mapping[str, Any],
    leaf: Path,
    *,
    expected_renderer_provenance: Mapping[str, Any] | None = None,
) -> None:
    for key in ("task", "code_revision", "asset_revision"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"artifact metadata missing {key}: {leaf}")
    if metadata["task"] != expected_task(str(row["campaign"]), str(row["arm"])):
        raise ValueError(f"artifact task mismatch: {leaf}")
    for key in ("code_revision", "asset_revision"):
        if key in row and metadata[key] != row[key]:
            raise ValueError(f"artifact {key} mismatch: {leaf}")
    if not _is_revision(metadata.get("presentation_generator_revision")):
        raise ValueError(f"artifact presentation generator revision invalid: {leaf}")
    _validate_metadata_provenance(
        metadata,
        row,
        leaf,
        expected_renderer_provenance=expected_renderer_provenance,
    )
    if metadata.get("renderer_contract") != RENDERER_CONTRACT:
        raise ValueError(f"artifact renderer contract mismatch: {leaf}")
    if metadata.get("timing") != TIMING_CONTRACT:
        raise ValueError(f"artifact timing contract mismatch: {leaf}")
    rollout = metadata.get("rollout")
    if not isinstance(rollout, Mapping):
        raise ValueError(f"artifact rollout metadata missing: {leaf}")
    try:
        requested = int(rollout["requested_control_steps"])
        executed = int(rollout["executed_control_steps"])
        frame_count = int(rollout["frame_count"])
        auto_reset_enabled = rollout["auto_reset_enabled"]
        boundary = rollout["terminal_boundary"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"artifact rollout metadata malformed: {leaf}") from error
    if requested < executed or executed < 0 or frame_count != executed + 1:
        raise ValueError(f"artifact rollout counts mismatch: {leaf}")
    if auto_reset_enabled is not False:
        raise ValueError(f"artifact rollout auto-reset must be disabled: {leaf}")
    if not isinstance(boundary, Mapping) or not isinstance(boundary.get("detected"), bool):
        raise ValueError(f"artifact terminal boundary metadata malformed: {leaf}")
    if boundary["detected"]:
        if boundary.get("step") != executed or not isinstance(boundary.get("reason"), str):
            raise ValueError(f"artifact terminal boundary mismatch: {leaf}")
    elif boundary.get("step") is not None or boundary.get("reason") != "step_limit":
        raise ValueError(f"artifact terminal boundary mismatch: {leaf}")
    dimensions = metadata.get("output_dimensions_px")
    expected_dimensions = {
        "frame": [RENDERER_CONTRACT["frame_width_px"], RENDERER_CONTRACT["frame_height_px"]],
        "montage": [
            RENDERER_CONTRACT["frame_width_px"] * min(6, frame_count),
            RENDERER_CONTRACT["frame_height_px"],
        ],
    }
    if dimensions != expected_dimensions:
        raise ValueError(f"artifact output dimensions mismatch: {leaf}")


def _validate_trace_rollout(trace_path: Path, rollout: Mapping[str, Any]) -> None:
    """Cross-bind declared frame/control counts to the recorded control trace."""
    try:
        with np.load(trace_path) as trace:
            positions = np.asarray(trace["head_position_m"])
            contact = np.asarray(trace["contact"])
            actions = np.asarray(trace["action"])
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"artifact rollout trace unreadable: {trace_path}") from error
    executed = int(rollout["executed_control_steps"])
    if (
        positions.shape != (executed + 1, 3)
        or contact.shape != (executed + 1,)
        or actions.ndim != 2
        or len(actions) != executed
    ):
        raise ValueError(f"artifact rollout trace counts mismatch: {trace_path}")


def _validate_media_properties(leaf: Path, rollout: Mapping[str, Any]) -> None:
    """Measure MP4/PNG properties rather than trusting declared metadata."""
    video_path = leaf / "policy.mp4"
    montage_path = leaf / "montage.png"
    try:
        video_metadata = iio.immeta(video_path)
        fps = float(video_metadata["fps"])
        width, height = tuple(video_metadata["size"])
        frame_count = sum(1 for _ in iio.imiter(video_path))
        montage = np.asarray(iio.imread(montage_path))
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"artifact media properties unavailable: {leaf}") from error
    expected_width = RENDERER_CONTRACT["frame_width_px"]
    expected_height = RENDERER_CONTRACT["frame_height_px"]
    expected_frames = int(rollout["frame_count"])
    if (
        (width, height) != (expected_width, expected_height)
        or not np.isfinite(fps)
        or abs(fps - RENDERER_CONTRACT["fps"]) > 1e-9
        or frame_count != expected_frames
        or montage.ndim < 2
        or montage.shape[:2]
        != (expected_height, expected_width * min(6, expected_frames))
    ):
        raise ValueError(f"artifact media properties mismatch: {leaf}")


def validate_policy_artifacts(root: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict:
    """Validate a complete, uniformly reset, hash-bound fixed-reset video library."""
    validate_inventory(rows)
    root = Path(root)
    _validate_library_tree(root)
    fixed_reset = load_fixed_reset(FIXED_RESET_ENVELOPE)
    fixed_digest = fixed_reset["reset_state_digest"]
    validated: list[dict] = []
    requested_steps: set[int] = set()
    code_revisions: set[str] = set()
    asset_revisions: set[str] = set()
    presentation_revisions: set[str] = set()
    reference_polyline: np.ndarray | None = None
    nail_top: np.ndarray | None = None
    for row in rows:
        leaf = policy_artifact_dir(root, row)
        metadata_path = leaf / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"artifact metadata is unreadable: {metadata_path}") from error
        if not isinstance(metadata, Mapping):
            raise ValueError(f"artifact metadata is not an object: {metadata_path}")
        for key in ("campaign", "arm", "training_seed", "checkpoint_sha256"):
            if str(metadata.get(key)) != str(row[key]):
                raise ValueError(f"artifact metadata identity mismatch: {leaf}")
        if metadata.get("reset_state_digest") != fixed_digest:
            raise ValueError(f"artifact reset digest mismatch: {leaf}")
        if metadata.get("metadata_payload_sha256") != _metadata_digest(metadata):
            raise ValueError(f"artifact metadata digest mismatch: {leaf}")
        _validate_metadata_contract(metadata, row, leaf)
        artifact_hashes = metadata.get("artifacts")
        if not isinstance(artifact_hashes, Mapping):
            raise ValueError(f"artifact hashes missing: {leaf}")
        for filename in ARTIFACT_FILENAMES:
            artifact_path = leaf / filename
            _readable_artifact(artifact_path)
            if artifact_hashes.get(filename) != _sha256(artifact_path):
                raise ValueError(f"artifact SHA-256 mismatch: {artifact_path}")
        _validate_trace_rollout(leaf / "trace.npz", metadata["rollout"])
        with np.load(leaf / "trace.npz") as trace:
            _, _, current_reference, current_nail_top = _trace_geometry(
                {key: np.asarray(trace[key]) for key in trace.files}
            )
        if reference_polyline is None:
            reference_polyline = current_reference
            nail_top = current_nail_top
        elif not np.array_equal(current_reference, reference_polyline):
            raise ValueError(f"artifact reference geometry differs: {leaf}")
        elif not np.array_equal(current_nail_top, nail_top):
            raise ValueError(f"artifact nail geometry differs: {leaf}")
        _validate_media_properties(leaf, metadata["rollout"])
        requested_steps.add(int(metadata["rollout"]["requested_control_steps"]))
        code_revisions.add(str(metadata["code_revision"]))
        asset_revisions.add(str(metadata["asset_revision"]))
        presentation_revisions.add(str(metadata["presentation_generator_revision"]))
        validated.append(dict(metadata))
    if len(requested_steps) != 1:
        raise ValueError("artifact rollout requested control steps differ")
    if len(code_revisions) != 1 or len(asset_revisions) != 1:
        raise ValueError("artifact code or asset revision differs")
    if len(presentation_revisions) != 1:
        raise ValueError("artifact presentation generator revision differs")
    return {"fixed_reset": fixed_reset, "artifacts": validated}


# --- Wave-1 500 Hz substep evidence -----------------------------------------


def _as_array(trace: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        return np.asarray(trace[key])
    except KeyError as error:
        raise ValueError(f"substep trace is missing {key}") from error


def validate_substep_trace(trace: Mapping[str, Any]) -> dict:
    """Prove a trace is genuinely per-physics-substep and first-episode only.

    A 50 Hz control-rate array repeated to substep length is the failure this
    guards against: it has the right shape and the wrong content.
    """
    for key in SUBSTEP_TRACE_KEYS:
        if key not in trace:
            raise ValueError(f"substep trace is missing {key}")

    positions = _as_array(trace, "substep_head_position_m").astype(float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("substep_head_position_m must have shape [substeps, 3]")
    if not np.isfinite(positions).all():
        raise ValueError("substep_head_position_m must be finite")

    decimation = int(_as_array(trace, "control_decimation"))
    control_steps = int(_as_array(trace, "executed_control_steps"))
    physics_dt = float(_as_array(trace, "physics_dt_s"))
    if decimation <= 0 or control_steps <= 0 or not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("substep trace timing must be positive and finite")
    substeps = len(positions)
    if substeps != decimation * control_steps:
        raise ValueError(
            "substep trace does not sit on the control grid: "
            f"{substeps} samples != {decimation} x {control_steps}"
        )

    per_substep = (
        "substep_contact",
        "substep_nail_depth_m",
        "substep_perpendicular_error_m",
        "substep_d_start_m",
        "substep_f_best",
        "substep_gate_index",
        "substep_control_step",
        "substep_is_control_boundary",
        "substep_episode_index",
    )
    for key in per_substep:
        values = _as_array(trace, key)
        if values.shape != (substeps,):
            raise ValueError(f"{key} must have one value per substep")
    joint_names = _as_array(trace, "arm_joint_names")
    for key in ("substep_arm_qvel_rad_s", "substep_arm_qvel_pre_rad_s"):
        qvel = _as_array(trace, key).astype(float)
        if qvel.ndim != 2 or len(qvel) != substeps or not np.isfinite(qvel).all():
            raise ValueError(f"{key} must be finite with one row per substep")
        if joint_names.ndim != 1 or qvel.shape[1] != len(joint_names):
            raise ValueError(f"{key} columns must be named by arm_joint_names")
    for key in ("substep_nail_depth_m", "substep_perpendicular_error_m",
                "substep_d_start_m", "substep_f_best"):
        if not np.isfinite(_as_array(trace, key).astype(float)).all():
            raise ValueError(f"{key} must be finite")

    episode = _as_array(trace, "substep_episode_index").astype(np.int64)
    if int(episode.max(initial=0)) != 0 or int(episode.min(initial=0)) != 0:
        raise ValueError("substep trace contains post-reset samples from a later episode")
    control_step = _as_array(trace, "substep_control_step").astype(np.int64)
    if not np.array_equal(control_step, np.arange(substeps) // decimation):
        raise ValueError("substep_control_step must index its own control window")
    boundary = _as_array(trace, "substep_is_control_boundary").astype(bool)
    expected_boundary = np.zeros(substeps, dtype=bool)
    expected_boundary[decimation - 1 :: decimation] = True
    if not np.array_equal(boundary, expected_boundary):
        raise ValueError("substep_is_control_boundary must mark every control window end")

    payouts = _as_array(trace, "control_step_reward_terms").astype(float)
    names = _as_array(trace, "control_step_reward_term_names")
    if payouts.ndim != 2 or len(payouts) != control_steps or not np.isfinite(payouts).all():
        raise ValueError("control_step_reward_terms must be finite [control_steps, terms]")
    if names.ndim != 1 or len(names) != payouts.shape[1]:
        raise ValueError("control_step_reward_term_names must name every payout column")
    totals = _as_array(trace, "control_step_reward_total").astype(float)
    if totals.shape != (control_steps,) or not np.isfinite(totals).all():
        raise ValueError("control_step_reward_total must be finite, one value per control step")
    tolerance = 1e-6 * max(1.0, float(np.abs(totals).max(initial=0.0)))
    if not np.allclose(payouts.sum(axis=1), totals, atol=tolerance, rtol=0.0):
        raise ValueError(
            "control_step_reward_terms must sum to control_step_reward_total; a payout "
            "matrix that disagrees with the recorded step reward was not measured together"
        )

    entry = _as_array(trace, "guideline_entry_m").astype(float)
    nail = _as_array(trace, "guideline_nail_m").astype(float)
    gates = _as_array(trace, "guideline_gate_centers_m").astype(float)
    if entry.shape != (3,) or nail.shape != (3,) or not np.isfinite(entry).all() or not np.isfinite(nail).all():
        raise ValueError("guideline entry/nail must be finite 3-vectors")
    if gates.shape != (6, 3) or not np.isfinite(gates).all():
        raise ValueError("guideline_gate_centers_m must be six finite 3-vectors")
    reference_length = float(_as_array(trace, "guideline_reference_length_m"))
    if not np.isfinite(reference_length) or reference_length <= 0:
        raise ValueError("guideline_reference_length_m must be positive and finite")

    if not np.array_equal(positions[0], entry):
        raise ValueError(
            "guideline_entry_m must be the anchored first substep head position; "
            "geometry from a different episode was pasted in"
        )
    gate_index = _as_array(trace, "substep_gate_index").astype(np.int64)
    if gate_index.min(initial=0) < 0 or gate_index.max(initial=0) > 6:
        raise ValueError("substep_gate_index must stay within [0, 6]")
    if np.any(np.diff(gate_index) < 0):
        raise ValueError("substep_gate_index must be non-decreasing within an episode")

    # --- the 500 Hz proof -------------------------------------------------
    # A control-rate path repeated to substep length puts ALL of its displacement on
    # the transitions that cross a control-window boundary and none inside a window.
    # With decimation d, an honestly sampled path puts ~(d-1)/d of its displacement on
    # interior transitions. Measuring the SHARE of motion (rather than counting distinct
    # values) is robust both to float noise sprinkled on a fake and to an arm that
    # genuinely holds still: a still window contributes zero to numerator and
    # denominator alike.
    #
    # The share alone is NOT sufficient: a zero-order hold whose phase is offset by
    # s != 0 puts every transition inside a window and scores 1.000. So the sample
    # COUNT is checked too -- any hold, at any phase, can only produce as many distinct
    # positions as there were control steps. The two clauses are complementary: noise
    # sprinkled on a repeat inflates the count but leaves the motion on boundaries;
    # a phase-shifted hold keeps motion interior but cannot inflate the count.
    #
    # Ceiling: together these refute repetition/upsampling of a control-rate array,
    # which is the failure they exist to catch. They cannot refute a wholly synthetic
    # path fabricated with smooth intra-window motion -- no content-only test can.
    # Provenance (the renderer hook that wrote the file) is what rules that out.
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    crosses_boundary = ((np.arange(1, substeps) % decimation) == 0)
    total_motion = float(steps.sum())
    if total_motion <= 0.0:
        raise ValueError(
            "substep positions are not genuinely per-substep: the head never moved"
        )
    interior_motion_share = float(steps[~crosses_boundary].sum()) / total_motion
    unique_positions = len(np.unique(positions, axis=0))
    if interior_motion_share < 0.5:
        raise ValueError(
            "substep positions are not genuinely per-substep: only "
            f"{interior_motion_share:.3f} of the head displacement happens inside "
            "control windows, which is the signature of a repeated control-rate path"
        )
    if unique_positions <= control_steps + 1:
        raise ValueError(
            "substep positions are not genuinely per-substep: "
            f"{unique_positions} distinct samples cannot come from {substeps} "
            f"substeps unless a control-rate path was held across each window"
        )

    return {
        "substep_count": substeps,
        "executed_control_steps": control_steps,
        "control_decimation": decimation,
        "physics_dt_s": physics_dt,
        "sample_rate_hz": 1.0 / physics_dt,
        "unique_substep_positions": unique_positions,
        "interior_motion_share": interior_motion_share,
        "total_head_path_length_m": total_motion,
        "reference_length_m": reference_length,
    }


def _wave1_axis_half_span(trace: Mapping[str, Any]) -> tuple[float, dict]:
    positions = np.asarray(trace["substep_head_position_m"], dtype=float)
    stacked = np.vstack((
        positions,
        np.asarray(trace["guideline_entry_m"], dtype=float)[None, :],
        np.asarray(trace["guideline_nail_m"], dtype=float)[None, :],
        np.asarray(trace["guideline_gate_centers_m"], dtype=float),
    ))
    span = float(np.max(np.ptp(stacked, axis=0)))
    half = max(span, 0.02) * 0.6  # one shared metric scale for both panels
    centers = {
        axis: float(0.5 * (stacked[:, axis].max() + stacked[:, axis].min()))
        for axis in (0, 1, 2)
    }
    return half, centers


def write_substep_trajectory_png(
    trace: Mapping[str, Any],
    path: str | Path,
    *,
    title: str = "",
    treatment: Treatment | None = None,
) -> dict:
    """Plot the 500 Hz path against the tracker's frozen entry->nail guideline.

    ``treatment=None`` reproduces the LEGACY figure byte-for-byte (line + six gate disks). The
    Wave-1/Wave-2 leaves record their trajectory.png SHA-256 inside hash-bound manifests, so that
    path must never drift. Pass a ``Treatment`` to draw only the geometry the arm was paid for.
    """
    report = validate_substep_trace(trace)
    geometry = GEOMETRY_GATES if treatment is None else treatment.geometry
    positions = np.asarray(trace["substep_head_position_m"], dtype=float)
    contact = np.asarray(trace["substep_contact"], dtype=bool)
    boundary = np.asarray(trace["substep_is_control_boundary"], dtype=bool)
    entry = np.asarray(trace["guideline_entry_m"], dtype=float)
    nail = np.asarray(trace["guideline_nail_m"], dtype=float)
    gates = np.asarray(trace["guideline_gate_centers_m"], dtype=float)
    reference = np.vstack((entry, nail))
    half, centers = _wave1_axis_half_span(trace)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for axis, ordinate, label in ((axes[0], 2, "z (m)"), (axes[1], 1, "y (m)")):
        if geometry != GEOMETRY_NONE:
            axis.plot(
                reference[:, 0],
                reference[:, ordinate],
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="tracker guideline (entry -> nail)",
                zorder=1,
            )
        if geometry == GEOMETRY_GATES:
            for center in gates:
                axis.add_patch(
                    plt.Circle(
                        (center[0], center[ordinate]),
                        GUIDELINE_GATE_RADIUS_M,
                        facecolor="none",
                        edgecolor="#1f77b4",
                        linewidth=0.8,
                        alpha=0.8,
                        zorder=2,
                    )
                )
        elif geometry == GEOMETRY_WAYPOINTS:
            # Ordered target POINTS -- r_waypoint_progress rewards approach, not disk entry.
            for order, center in enumerate(gates, start=1):
                axis.scatter(
                    center[0],
                    center[ordinate],
                    marker="D",
                    facecolors="none",
                    edgecolors="#1f77b4",
                    s=22,
                    linewidths=0.9,
                    zorder=2,
                )
                axis.annotate(
                    str(order),
                    (center[0], center[ordinate]),
                    fontsize=6,
                    color="#1f77b4",
                    xytext=(3, 3),
                    textcoords="offset points",
                    zorder=2,
                )
        if len(positions) > 1:
            points = positions[:, [0, ordinate]].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            axis.add_collection(
                LineCollection(
                    segments,
                    cmap="viridis",
                    array=np.linspace(0.0, 1.0, len(segments)),
                    linewidth=1.4,
                    zorder=3,
                )
            )
        axis.scatter(
            positions[boundary, 0],
            positions[boundary, ordinate],
            facecolors="none",
            edgecolors="#444444",
            s=26,
            linewidths=0.7,
            label="control-step boundary",
            zorder=4,
        )
        axis.scatter(positions[0, 0], positions[0, ordinate], c="#2ca02c", s=42, zorder=6,
                     label="start")
        if contact.any():
            axis.scatter(positions[contact, 0], positions[contact, ordinate], c="#d62728",
                         s=14, zorder=7, label="contact")
        axis.set_xlim(centers[0] - half, centers[0] + half)
        axis.set_ylim(centers[ordinate] - half, centers[ordinate] + half)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (m)")
        axis.set_ylabel(label)
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.25)
    axes[0].set_title("x-z side view")
    axes[1].set_title("x-y top view")
    axes[0].legend(fontsize=7, loc="best")
    if treatment is None:
        caption = (
            f"\n{report['substep_count']} substeps @ {report['sample_rate_hz']:.0f} Hz"
            " · dashed black = WaypointProgressTracker entry->nail · viridis = normalized time"
        )
    else:
        caption = (
            f"\n{report['substep_count']} substeps @ {report['sample_rate_hz']:.0f} Hz"
            f" · {_GEOMETRY_CAPTION[geometry]} · viridis = normalized time"
        )
    fig.suptitle((title or "Hammer-head 500 Hz trajectory") + caption)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88 if treatment is None else 0.80])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)

    result = {
        **report,
        "reference_source": (
            None
            if geometry == GEOMETRY_NONE
            else "waypoint_progress_tracker_entry_to_nail"
        ),
        # None when nothing was drawn, so a downstream consumer cannot cite a guideline the
        # figure deliberately does not show (2026-08 review).
        "reference_endpoints_m": None if geometry == GEOMETRY_NONE else reference,
        "gate_disk_count": int(len(gates)) if geometry == GEOMETRY_GATES else 0,
        "waypoint_marker_count": (
            int(len(gates)) if geometry == GEOMETRY_WAYPOINTS else 0
        ),
        "geometry": geometry,
        "contact_sample_count": int(contact.sum()),
        "control_boundary_marker_count": int(boundary.sum()),
        "axis_half_span_m": {
            "xz": float(0.5 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0])),
            "xy": float(0.5 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0])),
        },
        "axis_x_half_span_m": {
            "xz": float(0.5 * (axes[0].get_xlim()[1] - axes[0].get_xlim()[0])),
            "xy": float(0.5 * (axes[1].get_xlim()[1] - axes[1].get_xlim()[0])),
        },
    }
    plt.close(fig)
    return result


def validate_execution_device(metadata: Mapping[str, Any], leaf: Path) -> dict:
    """Require a directly recorded CPU execution device on this fixed-reset leaf."""
    block = metadata.get("execution_device")
    if not isinstance(block, Mapping) or set(block) != set(EXECUTION_DEVICE_FIELDS):
        raise ValueError(f"execution device block missing or malformed: {leaf}")
    platform_name = block.get("platform")
    if not isinstance(platform_name, str) or not platform_name.strip():
        raise ValueError(f"execution device platform not recorded: {leaf}")
    for field in ("requested", "actual_env_device", "actual_tensor_device"):
        value = block.get(field)
        if not isinstance(value, str) or value.split(":")[0].strip().lower() != "cpu":
            raise ValueError(
                f"fixed-reset library must run on CPU; {field}={value!r}: {leaf}"
            )
    return dict(block)


def validate_wave1_policy_artifacts(
    leaf: str | Path, expectations: Mapping[str, Any], *, campaign: str = "wave1"
) -> dict:
    """Validate one rendered substep leaf against its frozen policy manifest row.

    Named for the campaign that introduced the contract; `campaign` selects the
    identity it is validated against. Only wave1 is exempt from recording a device.
    """
    leaf = Path(leaf)
    metadata_path = leaf / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"wave1 metadata unreadable: {metadata_path}") from error
    if not isinstance(metadata, Mapping):
        raise ValueError(f"wave1 metadata is not an object: {metadata_path}")

    arm = str(expectations["arm"])
    if metadata.get("campaign") != campaign or str(metadata.get("arm")) != arm:
        raise ValueError(f"wave1 metadata identity mismatch: {leaf}")
    try:
        recorded_seed = int(metadata.get("training_seed"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"wave1 metadata seed malformed: {leaf}") from error
    if recorded_seed != int(expectations["training_seed"]):
        raise ValueError(f"wave1 metadata seed mismatch: {leaf}")
    if metadata.get("task") != expected_task(campaign, arm):
        raise ValueError(f"wave1 metadata task mismatch: {leaf}")
    if metadata.get("checkpoint_sha256") != expectations["checkpoint_sha256"]:
        raise ValueError(f"wave1 checkpoint hash mismatch: {leaf}")
    if metadata.get("reset_state_digest") != expectations["reset_state_digest"]:
        raise ValueError(f"wave1 fixed-reset digest mismatch: {leaf}")
    revisions = {}
    for field, key in zip(WAVE1_REVISION_FIELDS,
                          ("training", "asset", "analysis"), strict=True):
        value = metadata.get(field)
        if not _is_revision(value) or value != expectations[field.replace("_revision", "") + "_revision"]:
            raise ValueError(f"wave1 {field} mismatch: {leaf}")
        revisions[key] = str(value)
    if revisions["training"] == revisions["analysis"]:
        raise ValueError(
            f"wave1 analysis revision must be recorded separately from training: {leaf}"
        )
    if metadata.get("renderer_contract") != SUBSTEP_RENDERER_CONTRACT:
        raise ValueError(f"wave1 renderer contract mismatch: {leaf}")
    if metadata.get("timing") != TIMING_CONTRACT:
        raise ValueError(f"wave1 timing contract mismatch: {leaf}")
    if metadata.get("metadata_payload_sha256") != _metadata_digest(metadata):
        raise ValueError(f"wave1 metadata payload digest mismatch: {leaf}")
    execution_device = (
        None
        if campaign in EXECUTION_DEVICE_CAMPAIGN_EXEMPT
        else validate_execution_device(metadata, leaf)
    )
    rollout = metadata.get("rollout")
    if not isinstance(rollout, Mapping) or rollout.get("auto_reset_enabled") is not False:
        raise ValueError(f"wave1 rollout metadata malformed: {leaf}")
    boundary = rollout.get("terminal_boundary")
    if not isinstance(boundary, Mapping) or not isinstance(boundary.get("detected"), bool):
        raise ValueError(f"wave1 terminal boundary malformed: {leaf}")
    executed_declared = rollout.get("executed_control_steps")
    if boundary["detected"]:
        if boundary.get("step") != executed_declared or not isinstance(
            boundary.get("reason"), str
        ):
            raise ValueError(f"wave1 terminal boundary mismatch: {leaf}")
    elif boundary.get("step") is not None or boundary.get("reason") != "step_limit":
        raise ValueError(f"wave1 terminal boundary mismatch: {leaf}")

    recorded_hashes = metadata.get("artifacts")
    if not isinstance(recorded_hashes, Mapping):
        raise ValueError(f"wave1 artifact hashes missing: {leaf}")
    for name in WAVE1_ARTIFACT_FILENAMES:
        artifact = leaf / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"wave1 artifact missing or empty: {artifact}")
        if recorded_hashes.get(name) != _sha256(artifact):
            raise ValueError(f"wave1 artifact SHA-256 mismatch: {artifact}")
    try:
        with np.load(leaf / "trace.npz") as loaded:
            trace = {key: loaded[key] for key in loaded.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"wave1 trace unreadable: {leaf}") from error
    report = validate_substep_trace(trace)

    try:
        video_metadata = iio.immeta(leaf / "policy.mp4")
        fps = float(video_metadata["fps"])
        width, height = tuple(video_metadata["size"])
        frame_count = sum(1 for _ in iio.imiter(leaf / "policy.mp4"))
        image = np.asarray(iio.imread(leaf / "trajectory.png"))
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"wave1 media properties unavailable: {leaf}") from error
    stride = SUBSTEP_RENDERER_CONTRACT["frame_substep_stride"]
    expected_frames = report["substep_count"] // stride
    if (
        abs(fps - SUBSTEP_RENDERER_CONTRACT["fps"]) > 1e-9
        or (width, height) != (
            SUBSTEP_RENDERER_CONTRACT["frame_width_px"],
            SUBSTEP_RENDERER_CONTRACT["frame_height_px"],
        )
        or frame_count != expected_frames
        or int(rollout.get("frame_count", -1)) != frame_count
        or int(rollout.get("substep_count", -1)) != report["substep_count"]
        or int(rollout.get("executed_control_steps", -1))
        != report["executed_control_steps"]
        or image.size == 0
    ):
        raise ValueError(f"wave1 media properties mismatch: {leaf}")

    return {
        **report,
        "measured_fps": fps,
        "measured_frame_count": frame_count,
        "revisions": revisions,
        "execution_device": execution_device,
        "artifact_sha256": {
            name: _sha256(leaf / name) for name in WAVE1_ARTIFACT_FILENAMES
        },
    }
