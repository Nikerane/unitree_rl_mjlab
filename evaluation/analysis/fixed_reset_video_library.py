"""Pure contracts for the deterministic 56-policy fixed-reset video library."""

from __future__ import annotations

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
TASK_BY_CAMPAIGN_ARM = {
    ("fq4x8", "F8"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    ("fq4x8", "F0"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
    ("fq4x8", "D0"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
    ("fq4x8", "FQ-min"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
    ("fq3x8", "F8"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    ("fq3x8", "B8"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
    ("fq3x8", "FQ"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
}


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
