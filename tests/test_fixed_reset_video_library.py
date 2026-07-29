"""Contracts for one fixed-reset rollout artifact per registered policy."""

from __future__ import annotations

import copy
import hashlib
import json
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


def _write_png(path: Path) -> None:
    iio.imwrite(path, np.zeros((2, 2, 3), dtype=np.uint8))


def _write_mp4(path: Path) -> None:
    iio.imwrite(path, np.zeros((2, 2, 2, 3), dtype=np.uint8), fps=10)


def write_complete_fake_library(root: Path, *, reset_digest: str) -> None:
    for row in expected_rows():
        leaf = root / row["campaign"] / row["arm"] / str(row["training_seed"])
        leaf.mkdir(parents=True)
        _write_mp4(leaf / "policy.mp4")
        _write_png(leaf / "montage.png")
        _write_png(leaf / "trajectory.png")
        np.savez(leaf / "trace.npz", head_position_m=np.zeros((1, 3)), contact=np.zeros(1))
        metadata = {
            **row,
            "reset_state_digest": reset_digest,
            "artifacts": {
                name: _sha256(leaf / name)
                for name in ("policy.mp4", "montage.png", "trajectory.png", "trace.npz")
            },
        }
        metadata["metadata_payload_sha256"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        (leaf / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))


def corrupt_one_metadata_reset_digest(root: Path) -> None:
    path = next(root.glob("*/*/*/metadata.json"))
    payload = json.loads(path.read_text())
    payload["reset_state_digest"] = "0" * 64
    del payload["metadata_payload_sha256"]
    payload["metadata_payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True))


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
