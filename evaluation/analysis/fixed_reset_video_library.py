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

FIXED_RESET_ENVELOPE = Path(
    "docs/results/assets/2026-07-29_lambda_feasibility_stage0/"
    "lambda_feasibility_stage0_inputs.json"
)
ARTIFACT_FILENAMES = ("policy.mp4", "montage.png", "trajectory.png", "trace.npz")


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


def _reset_record(payload: Mapping[str, Any]) -> dict:
    if "resets" in payload:
        resets = payload["resets"]
        if not isinstance(resets, Mapping) or not isinstance(resets.get("fixed_reset"), Mapping):
            raise ValueError("reset envelope has no fixed_reset record")
        payload = resets["fixed_reset"]
    return dict(payload)


def load_fixed_reset(path: str | Path) -> dict:
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


def write_trajectory_png(trace: Mapping[str, Any], path: str | Path) -> None:
    """Write a shared-axis x-z / x-y hammer-head trajectory diagnostic."""
    positions = np.asarray(trace["head_position_m"], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("trace head_position_m must have shape [steps, 3]")
    if not np.isfinite(positions).all():
        raise ValueError("trace head_position_m must be finite")
    contact = np.asarray(trace.get("contact", np.zeros(len(positions), dtype=bool)), dtype=bool)
    if contact.shape != (len(positions),):
        raise ValueError("trace contact must have one value per head position")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=False, layout="constrained")
    for axis, ordinate, label in ((axes[0], 2, "z (m)"), (axes[1], 1, "y (m)")):
        axis.plot(positions[:, 0], positions[:, ordinate], color="#1f77b4", linewidth=1.5)
        if contact.any():
            axis.scatter(positions[contact, 0], positions[contact, ordinate], c="#d62728", s=15, label="contact")
            axis.legend(loc="best")
        axis.set_xlabel("x (m)")
        axis.set_ylabel(label)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
    axes[0].set_title("x-z trajectory")
    axes[1].set_title("x-y trajectory")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
                positions = np.asarray(trace["head_position_m"])
                contact = np.asarray(trace["contact"])
                if positions.ndim != 2 or positions.shape[1] != 3 or contact.shape != (len(positions),):
                    raise ValueError("invalid trace arrays")
    except (OSError, StopIteration, ValueError, KeyError) as error:
        raise ValueError(f"artifact is unreadable: {path}") from error


def validate_policy_artifacts(root: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict:
    """Validate a complete, uniformly reset, hash-bound fixed-reset video library."""
    validate_inventory(rows)
    fixed_reset = load_fixed_reset(FIXED_RESET_ENVELOPE)
    fixed_digest = fixed_reset["reset_state_digest"]
    validated: list[dict] = []
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
        artifact_hashes = metadata.get("artifacts")
        if not isinstance(artifact_hashes, Mapping):
            raise ValueError(f"artifact hashes missing: {leaf}")
        for filename in ARTIFACT_FILENAMES:
            artifact_path = leaf / filename
            _readable_artifact(artifact_path)
            if artifact_hashes.get(filename) != _sha256(artifact_path):
                raise ValueError(f"artifact SHA-256 mismatch: {artifact_path}")
        validated.append(dict(metadata))
    return {"fixed_reset": fixed_reset, "artifacts": validated}
