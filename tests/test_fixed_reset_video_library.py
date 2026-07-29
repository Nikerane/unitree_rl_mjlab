"""Contracts for one fixed-reset rollout artifact per registered policy."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from evaluation.analysis.fixed_reset_video_library import (
    EXPECTED,
    first_episode_frame_count,
    load_fixed_reset,
    validate_inventory,
    validate_policy_artifacts,
)


FIXED_DIGEST = "bde511ec2adc42e5365e1e46f45ff1fb43223a93c31c6fbfb4580352e445e319"
RENDER_CONTRACT = {
    "version": 1,
    "frame_width_px": 960,
    "frame_height_px": 720,
    "camera_distance_m": 0.85,
    "camera_elevation_deg": -25.0,
    "camera_azimuth_deg": 135.0,
    "fps": 10,
}
TIMING = {
    "physics_dt_s": 0.002,
    "control_decimation": 10,
    "control_dt_s": 0.02,
}
TASKS = {
    ("fq4x8", "F8"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    ("fq4x8", "F0"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
    ("fq4x8", "D0"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
    ("fq4x8", "FQ-min"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
    ("fq3x8", "F8"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    ("fq3x8", "B8"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
    ("fq3x8", "FQ"): "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_rows() -> list[dict]:
    """The 56 registered policy identities with distinct checkpoint digests."""
    return [
        {
            "campaign": campaign,
            "arm": arm,
            "training_seed": seed,
            "checkpoint_sha256": hashlib.sha256(
                f"{campaign}/{arm}/{seed}".encode()
            ).hexdigest(),
        }
        for campaign, arms in EXPECTED.items()
        for arm, seeds in arms.items()
        for seed in seeds
    ]


def canonical_fixed_reset_payload() -> dict:
    payload = json.loads(
        Path(
            "docs/results/assets/2026-07-29_lambda_feasibility_stage0/"
            "lambda_feasibility_stage0_inputs.json"
        ).read_text()
    )
    return copy.deepcopy(payload["resets"]["fixed_reset"])


def _write_png(path: Path, *, width: int = 960, height: int = 720) -> None:
    iio.imwrite(path, np.zeros((height, width, 3), dtype=np.uint8))


def _write_mp4(path: Path, *, frame_count: int = 1, fps: int = 10) -> None:
    iio.imwrite(path, np.zeros((frame_count, 720, 960, 3), dtype=np.uint8), fps=fps)


def _write_metadata(path: Path, metadata: dict) -> None:
    metadata["metadata_payload_sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    path.write_text(json.dumps(metadata, sort_keys=True))


def write_complete_fake_library(root: Path, *, reset_digest: str) -> None:
    fixture_mp4 = root / "fixture_policy.mp4"
    fixture_montage = root / "fixture_montage.png"
    _write_mp4(fixture_mp4)
    _write_png(fixture_montage)
    for row in expected_rows():
        leaf = root / row["campaign"] / row["arm"] / str(row["training_seed"])
        leaf.mkdir(parents=True)
        os.link(fixture_mp4, leaf / "policy.mp4")
        os.link(fixture_montage, leaf / "montage.png")
        _write_png(leaf / "trajectory.png")
        np.savez(
            leaf / "trace.npz",
            head_position_m=np.zeros((1, 3)),
            contact=np.zeros(1),
            action=np.zeros((0, 3)),
        )
        metadata = {
            **row,
            "task": TASKS[(row["campaign"], row["arm"])],
            "code_revision": "c" * 40,
            "asset_revision": "a" * 40,
            "reset_state_digest": reset_digest,
            "renderer_contract": RENDER_CONTRACT,
            "timing": TIMING,
            "rollout": {
                "requested_control_steps": 80,
                "executed_control_steps": 0,
                "frame_count": 1,
                "terminal_boundary": {
                    "detected": False,
                    "step": None,
                    "reason": "step_limit",
                },
            },
            "output_dimensions_px": {"frame": [960, 720], "montage": [960, 720]},
            "artifacts": {
                name: _sha256(leaf / name)
                for name in ("policy.mp4", "montage.png", "trajectory.png", "trace.npz")
            },
        }
        _write_metadata(leaf / "metadata.json", metadata)


def corrupt_one_metadata_reset_digest(root: Path) -> None:
    path = next(root.glob("*/*/*/metadata.json"))
    payload = json.loads(path.read_text())
    payload["reset_state_digest"] = "0" * 64
    del payload["metadata_payload_sha256"]
    _write_metadata(path, payload)


def test_inventory_requires_exact_56_members():
    """Removing one registered policy must make the artifact set ineligible."""
    rows = expected_rows()
    validate_inventory(rows)
    with pytest.raises(ValueError, match="membership"):
        validate_inventory(rows[:-1])


def test_fixed_reset_digest_is_verified(tmp_path):
    """A changed realized joint position must not be accepted as the fixed reset."""
    payload = canonical_fixed_reset_payload()
    path = tmp_path / "reset.json"
    path.write_text(json.dumps(payload))
    assert load_fixed_reset(path)["reset_state_digest"] == FIXED_DIGEST
    payload["reset_state"]["realized"]["robot_joint_pos"][0] += 0.01
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        load_fixed_reset(path)


def test_episode_boundary_excludes_the_auto_reset_frame():
    """A post-step zero is already the next episode and must be excluded."""
    assert first_episode_frame_count([1, 2, 3, 0, 1]) == 3


def test_artifact_validator_rejects_mixed_reset_digest(tmp_path):
    """One policy rendered from another reset cannot enter the comparison grid."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    corrupt_one_metadata_reset_digest(tmp_path)
    with pytest.raises(ValueError, match="reset"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_rejects_noncanonical_render_contract(tmp_path):
    """A 640 px rendering cannot be mixed into the frozen 960 px library."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    path = next(tmp_path.glob("*/*/*/metadata.json"))
    metadata = json.loads(path.read_text())
    metadata["renderer_contract"]["frame_width_px"] = 640
    del metadata["metadata_payload_sha256"]
    _write_metadata(path, metadata)
    with pytest.raises(ValueError, match="renderer contract"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_requires_auditable_rollout_metadata(tmp_path):
    """A video without task, revision, timing, and terminal evidence is ineligible."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    path = next(tmp_path.glob("*/*/*/metadata.json"))
    metadata = json.loads(path.read_text())
    del metadata["rollout"]
    del metadata["metadata_payload_sha256"]
    _write_metadata(path, metadata)
    with pytest.raises(ValueError, match="rollout"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_rejects_unexpected_policy_leaf(tmp_path):
    """An unregistered seed directory must not silently evade the exact-56 check."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    (tmp_path / "fq4x8" / "F8" / "999").mkdir()
    with pytest.raises(ValueError, match="unexpected"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_rejects_rollout_counts_inconsistent_with_trace(tmp_path):
    """Metadata cannot claim a rendered control step absent from trace.npz."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    path = next(tmp_path.glob("*/*/*/metadata.json"))
    metadata = json.loads(path.read_text())
    metadata["rollout"]["executed_control_steps"] = 1
    metadata["rollout"]["frame_count"] = 2
    metadata["output_dimensions_px"]["montage"] = [1920, 720]
    del metadata["metadata_payload_sha256"]
    _write_metadata(path, metadata)
    with pytest.raises(ValueError, match="rollout"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_rejects_actual_mp4_metadata_mismatch(tmp_path):
    """A 9 FPS/two-frame MP4 cannot claim the canonical 10 FPS/one-frame rollout."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    video = next(tmp_path.glob("*/*/*/policy.mp4"))
    video.unlink()
    _write_mp4(video, frame_count=2, fps=9)
    metadata_path = video.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["artifacts"]["policy.mp4"] = _sha256(video)
    del metadata["metadata_payload_sha256"]
    _write_metadata(metadata_path, metadata)
    with pytest.raises(ValueError, match="media"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_rejects_actual_montage_dimensions_mismatch(tmp_path):
    """A montage whose real width differs from the contract cannot be admitted."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    montage = next(tmp_path.glob("*/*/*/montage.png"))
    montage.unlink()
    _write_png(montage, width=959)
    metadata_path = montage.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["artifacts"]["montage.png"] = _sha256(montage)
    del metadata["metadata_payload_sha256"]
    _write_metadata(metadata_path, metadata)
    with pytest.raises(ValueError, match="media"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_rejects_wrong_task_without_row_task_field(tmp_path):
    """The campaign/arm contract, not an optional manifest field, pins the task."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    metadata_path = tmp_path / "fq4x8" / "F8" / "8" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["task"] = "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality"
    del metadata["metadata_payload_sha256"]
    _write_metadata(metadata_path, metadata)
    with pytest.raises(ValueError, match="task"):
        validate_policy_artifacts(tmp_path, expected_rows())
