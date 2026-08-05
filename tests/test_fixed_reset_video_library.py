"""Contracts for one fixed-reset rollout artifact per registered policy."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import imageio.v3 as iio
import numpy as np
import pytest
import torch

from evaluation.analysis import render_fixed_reset_video_library as library_driver
from evaluation.analysis.fixed_reset_video_library import (
    EXPECTED,
    first_episode_frame_count,
    load_fixed_reset,
    validate_inventory,
    validate_policy_artifacts,
    write_campaign_trajectory_grid,
    write_library_index,
    write_trajectory_png,
)
from scripts import render_policy
from src.tasks.hammer.mdp.references import SingleStrikeReference


FIXED_DIGEST = "bde511ec2adc42e5365e1e46f45ff1fb43223a93c31c6fbfb4580352e445e319"
PRESENTATION_REVISION = "d" * 40
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
RENDER_PROVENANCE = {
    "code_revision": "c" * 40,
    "asset_revision": "a" * 40,
    "source_scope": ["src", "scripts", "evaluation/analysis"],
    "asset_scope": ["hammer_z1_env/assets"],
    "device": "cpu",
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
            "checkpoint_path": f"/training/{campaign}/{arm}/{seed}/model_499.pt",
            "checkpoint_sha256": hashlib.sha256(
                f"{campaign}/{arm}/{seed}".encode()
            ).hexdigest(),
            "training_code_revision": "1" * 40,
            "training_asset_revision": "2" * 40,
            "accepted_training_manifest_sha256": "f" * 64,
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
            reference_polyline_m=np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]
            ),
            nail_top_m=np.array([0.0, 0.0, 0.0]),
        )
        metadata = {
            **row,
            "task": TASKS[(row["campaign"], row["arm"])],
            "code_revision": "c" * 40,
            "asset_revision": "a" * 40,
            "presentation_generator_revision": PRESENTATION_REVISION,
            "reset_state_digest": reset_digest,
            "renderer_contract": RENDER_CONTRACT,
            "timing": TIMING,
            "rollout": {
                "requested_control_steps": 80,
                "executed_control_steps": 0,
                "frame_count": 1,
                "auto_reset_enabled": False,
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
            "renderer_provenance": copy.deepcopy(RENDER_PROVENANCE),
            "training_provenance": {
                key: row[key]
                for key in (
                    "checkpoint_path",
                    "training_code_revision",
                    "training_asset_revision",
                    "accepted_training_manifest_sha256",
                )
            },
        }
        _write_metadata(leaf / "metadata.json", metadata)


def corrupt_one_metadata_reset_digest(root: Path) -> None:
    path = next(root.glob("*/*/*/metadata.json"))
    payload = json.loads(path.read_text())
    payload["reset_state_digest"] = "0" * 64
    del payload["metadata_payload_sha256"]
    _write_metadata(path, payload)


def _valid_contract_metadata(row: dict) -> dict:
    return {
        "task": TASKS[(row["campaign"], row["arm"])],
        "code_revision": RENDER_PROVENANCE["code_revision"],
        "asset_revision": RENDER_PROVENANCE["asset_revision"],
        "presentation_generator_revision": PRESENTATION_REVISION,
        "renderer_contract": RENDER_CONTRACT,
        "timing": TIMING,
        "rollout": {
            "requested_control_steps": 80,
            "executed_control_steps": 0,
            "frame_count": 1,
            "auto_reset_enabled": False,
            "terminal_boundary": {
                "detected": False,
                "step": None,
                "reason": "step_limit",
            },
        },
        "output_dimensions_px": {"frame": [960, 720], "montage": [960, 720]},
        "renderer_provenance": copy.deepcopy(RENDER_PROVENANCE),
        "training_provenance": {
            key: row[key]
            for key in (
                "checkpoint_path",
                "training_code_revision",
                "training_asset_revision",
                "accepted_training_manifest_sha256",
            )
        },
    }


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


def test_fixed_reset_rejects_self_consistent_alternate_realized_state(tmp_path):
    """Recomputing the inner digest cannot redefine the approved library reset."""
    payload = canonical_fixed_reset_payload()
    payload["reset_state"]["realized"]["robot_joint_pos"][0] += 0.01
    payload["reset_state_digest"] = hashlib.sha256(
        json.dumps(
            payload["reset_state"]["realized"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path = tmp_path / "alternate-reset.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="approved"):
        load_fixed_reset(path)


def test_default_fixed_reset_path_is_independent_of_caller_cwd(tmp_path, monkeypatch):
    """The library's default reset envelope belongs to its source checkout."""
    monkeypatch.chdir(tmp_path)

    assert load_fixed_reset()["reset_state_digest"] == FIXED_DIGEST


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.pop("renderer_provenance"), "renderer provenance"),
        (
            lambda metadata: metadata["renderer_provenance"].update(
                {"code_revision": "e" * 40}
            ),
            "renderer provenance",
        ),
        (
            lambda metadata: metadata["renderer_provenance"].update({"device": "cuda"}),
            "CPU",
        ),
        (
            lambda metadata: metadata["training_provenance"].update(
                {"training_code_revision": "e" * 40}
            ),
            "training provenance",
        ),
    ],
)
def test_final_validator_cross_binds_renderer_and_training_provenance(
    tmp_path, monkeypatch, mutation, message
):
    """Final admission must enforce the same frozen provenance as resume."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    metadata_path = tmp_path / "fq4x8" / "F8" / "8" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    mutation(metadata)
    del metadata["metadata_payload_sha256"]
    _write_metadata(metadata_path, metadata)
    monkeypatch.setattr(
        "evaluation.analysis.fixed_reset_video_library._readable_artifact",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "evaluation.analysis.fixed_reset_video_library._validate_media_properties",
        lambda _leaf, _rollout: None,
    )

    with pytest.raises(ValueError, match=message):
        validate_policy_artifacts(tmp_path, expected_rows())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda renderer: renderer.pop("source_scope"),
        lambda renderer: renderer.update({"asset_scope": ["other/assets"]}),
        lambda renderer: renderer.update({"unreviewed_scope": ["extra"]}),
    ],
)
def test_metadata_contract_requires_complete_renderer_provenance_schema(
    tmp_path, mutation
):
    """Final and resume validation must bind the same complete renderer schema."""
    row = expected_rows()[0]
    metadata = _valid_contract_metadata(row)
    mutation(metadata["renderer_provenance"])

    with pytest.raises(ValueError, match="renderer provenance"):
        library_driver._validate_metadata_contract(metadata, row, tmp_path / "leaf")


def test_metadata_contract_rejects_missing_frozen_training_inventory_fields(tmp_path):
    """Missing metadata and row fields cannot agree merely because both read as None."""
    complete_row = expected_rows()[0]
    row = {
        key: complete_row[key]
        for key in ("campaign", "arm", "training_seed", "checkpoint_sha256")
    }
    metadata = _valid_contract_metadata(complete_row)
    metadata["training_provenance"] = {}

    with pytest.raises(ValueError, match="training provenance"):
        library_driver._validate_metadata_contract(metadata, row, tmp_path / "leaf")


@pytest.mark.parametrize("malformed_artifacts", [None, [], "not-a-map"])
def test_malformed_artifact_map_invalidates_resume_without_raising(
    tmp_path, malformed_artifacts
):
    """A malformed metadata map is an invalid leaf that can be rerendered."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    row = expected_rows()[0]
    leaf = library_driver.policy_artifact_dir(tmp_path, row)
    metadata_path = leaf / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["artifacts"] = malformed_artifacts
    del metadata["metadata_payload_sha256"]
    _write_metadata(metadata_path, metadata)

    assert not library_driver._policy_artifact_is_valid(
        tmp_path,
        row,
        canonical_fixed_reset_payload(),
        render_provenance=RENDER_PROVENANCE,
    )


def _write_accepted_checkpoint_manifests(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Create byte-verified, schema-complete accepted manifests for the real builder."""
    import csv

    roots = {campaign: tmp_path / "checkpoints" / campaign for campaign in EXPECTED}
    manifests = {
        "fq4x8": tmp_path / "fq4x8_accepted.tsv",
        "fq3x8": tmp_path / "fq3x8_accepted.tsv",
    }
    rows_by_campaign: dict[str, list[dict[str, str]]] = {campaign: [] for campaign in EXPECTED}
    for row in expected_rows():
        campaign = row["campaign"]
        checkpoint = (
            roots[campaign]
            / f"{campaign}_{row['arm'].lower()}_seed{row['training_seed']}"
            / "model_499.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(
            f"{campaign}/{row['arm']}/{row['training_seed']}".encode("utf-8")
        )
        checkpoint_sha256 = _sha256(checkpoint)
        if campaign == "fq4x8":
            manifest_row = {
                "campaign": campaign,
                "disposition": "accepted",
                "arm": row["arm"],
                "short": row["arm"].lower().replace("-min", ""),
                "task": TASKS[(campaign, row["arm"])],
                "training_seed": str(row["training_seed"]),
                "checkpoint_path": f"/vega/frozen/{checkpoint.parent.name}/model_499.pt",
                "checkpoint_sha256": checkpoint_sha256,
                "training_attempt": "attempt1",
                "retry_history": "none",
                "code_revision": "1" * 40,
                "asset_revision": "2" * 40,
                "campaign_config_sha256": "5" * 64,
                "treatment_config_sha256": "6" * 64,
                "reader_sha256": "7" * 64,
                "normalizer_sha256": "8" * 64,
                "treatment_reward_sha256": "9" * 64,
                "fixed_action_signature_sha256": "a" * 64,
                "fixed_impedance_signature_sha256": "b" * 64,
                "cap_signature_sha256": "c" * 64,
                "clean_state": "true",
            }
        else:
            manifest_row = {
                "campaign": campaign,
                "disposition": "accepted",
                "arm": row["arm"],
                "short": row["arm"].lower(),
                "task": TASKS[(campaign, row["arm"])],
                "training_seed": str(row["training_seed"]),
                "run_name": f"{campaign}_{row['arm'].lower()}_seed{row['training_seed']}",
                "checkpoint_path": f"/vega/frozen/{checkpoint.parent.name}/model_499.pt",
                "checkpoint_sha256": checkpoint_sha256,
                "training_code_revision": "3" * 40,
                "training_asset_revision": "4" * 40,
                "slurm_array_id": "1_0",
                "retry_history": "none",
            }
        rows_by_campaign[campaign].append(manifest_row)
    for campaign, manifest in manifests.items():
        fieldnames = list(rows_by_campaign[campaign][0])
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows_by_campaign[campaign])
    return manifests["fq4x8"], manifests["fq3x8"], roots


def _fixture_manifest_hashes(fq4: Path, fq3: Path) -> dict[str, str]:
    return {"fq4x8": _sha256(fq4), "fq3x8": _sha256(fq3)}


def test_inventory_builder_freezes_verified_training_provenance_in_deterministic_order(tmp_path):
    """Changing source rows or checkpoint bytes cannot yield a different eligible inventory."""
    fq4, fq3, roots = _write_accepted_checkpoint_manifests(tmp_path)
    rows = library_driver.build_inventory(
        fq4, fq3, roots, expected_manifest_sha256=_fixture_manifest_hashes(fq4, fq3)
    )

    assert len(rows) == 56
    assert [(row["campaign"], row["arm"], row["training_seed"]) for row in rows] == [
        (row["campaign"], row["arm"], row["training_seed"]) for row in expected_rows()
    ]
    assert rows[0]["training_code_revision"] == "1" * 40
    assert rows[0]["training_asset_revision"] == "2" * 40
    assert rows[-1]["training_code_revision"] == "3" * 40
    assert rows[-1]["training_asset_revision"] == "4" * 40
    assert all(Path(row["_checkpoint_file"]).is_file() for row in rows)

    inventory = tmp_path / "checkpoint_inventory.tsv"
    library_driver.write_inventory(rows, inventory)
    assert _sha256(inventory) == inventory.with_suffix(".tsv.sha256").read_text().split()[0]


@pytest.mark.parametrize(
    ("mutation", "message", "refresh_manifest_hashes"),
    [
        (
            lambda fq4, fq3, roots: fq4.write_text(fq4.read_text() + "\n"),
            "manifest SHA-256",
            False,
        ),
        (
            lambda fq4, fq3, roots: _replace_tsv_cell(fq3, 1, "training_seed", "16"),
            "membership",
            True,
        ),
        (
            lambda fq4, fq3, roots: _replace_tsv_cell(fq4, 1, "task", "wrong-task"),
            "task",
            True,
        ),
        (
            lambda fq4, fq3, roots: next(roots["fq4x8"].rglob("model_499.pt")).unlink(),
            "checkpoint",
            False,
        ),
        (
            lambda fq4, fq3, roots: _replace_tsv_cell(fq3, 1, "arm", "unexpected"),
            "arm",
            True,
        ),
        (
            lambda fq4, fq3, roots: _replace_tsv_cell(fq4, 1, "clean_state", "false"),
            "clean",
            True,
        ),
    ],
)
def test_inventory_builder_rejects_unfrozen_or_ineligible_manifest_input(
    tmp_path, mutation, message, refresh_manifest_hashes
):
    """A wrong hash, identity, task, file, arm, or dirty training state must stop rendering."""
    fq4, fq3, roots = _write_accepted_checkpoint_manifests(tmp_path)
    expected_hashes = _fixture_manifest_hashes(fq4, fq3)
    mutation(fq4, fq3, roots)
    if refresh_manifest_hashes:
        expected_hashes = _fixture_manifest_hashes(fq4, fq3)

    with pytest.raises(ValueError, match=message):
        library_driver.build_inventory(
            fq4, fq3, roots, expected_manifest_sha256=expected_hashes
        )


def _replace_tsv_cell(path: Path, row_index: int, field: str, value: str) -> None:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
        fieldnames = tuple(rows[0])
    rows[row_index][field] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _commit_fixture_repo(root: Path, files: dict[str, str]) -> str:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _provenance_repositories(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "unitree_rl_mjlab"
    asset = tmp_path / "safe_impact_manipulation"
    source_revision = _commit_fixture_repo(
        source,
        {
            "src/task.py": "TASK = 1\n",
            "scripts/render_policy.py": "RENDER = 1\n",
            "evaluation/analysis/contract.py": "CONTRACT = 1\n",
            "docs/results/ignored.md": "result\n",
        },
    )
    asset_revision = _commit_fixture_repo(
        asset,
        {
            "hammer_z1_env/assets/hammer.xml": "<mujoco/>\n",
            "notes/ignored.md": "note\n",
        },
    )
    return source, asset, source_revision, asset_revision


def _bind_fixture_repositories(monkeypatch, source: Path, asset: Path) -> None:
    monkeypatch.setattr(library_driver, "SOURCE_ROOT", source, raising=False)
    monkeypatch.setattr(library_driver, "ASSET_ROOT", asset, raising=False)


def test_render_provenance_derives_revisions_and_ignores_unimported_dirty_files(
    tmp_path, monkeypatch
):
    """Caller labels and unrelated docs cannot define or block renderer provenance."""
    source, asset, source_revision, asset_revision = _provenance_repositories(tmp_path)
    _bind_fixture_repositories(monkeypatch, source, asset)
    (source / "docs/results/ignored.md").write_text("dirty result\n")
    (asset / "notes/ignored.md").write_text("dirty note\n")

    provenance = library_driver.capture_render_provenance()

    assert provenance["code_revision"] == source_revision
    assert provenance["asset_revision"] == asset_revision


@pytest.mark.parametrize(
    ("repository", "relative", "message"),
    [
        ("source", "src/task.py", "source scope"),
        ("asset", "hammer_z1_env/assets/hammer.xml", "asset scope"),
    ],
)
def test_render_provenance_rejects_dirty_import_or_hammer_asset_scope(
    tmp_path, monkeypatch, repository, relative, message
):
    """Any tracked or untracked mutation that can affect rendering must fail closed."""
    source, asset, _, _ = _provenance_repositories(tmp_path)
    _bind_fixture_repositories(monkeypatch, source, asset)
    root = source if repository == "source" else asset
    (root / relative).write_text("dirty\n")

    with pytest.raises(ValueError, match=message):
        library_driver.capture_render_provenance()


def test_render_provenance_is_rechecked_after_batch(tmp_path, monkeypatch):
    """A source mutation during rendering must invalidate the recorded revision."""
    source, asset, _, _ = _provenance_repositories(tmp_path)
    _bind_fixture_repositories(monkeypatch, source, asset)
    before = library_driver.capture_render_provenance()
    (source / "evaluation/analysis/contract.py").write_text("changed during batch\n")

    with pytest.raises(ValueError, match="source scope"):
        library_driver.require_unchanged_render_provenance(before)


def test_driver_binds_source_and_assets_to_the_paths_imported_by_the_checkout():
    """The renderer cannot validate one checkout or asset repo and execute another."""
    from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML

    assert library_driver.SOURCE_ROOT == Path(library_driver.__file__).resolve().parents[2]
    assert library_driver.ASSET_ROOT == Z1_HAMMER_XML.resolve().parents[2]


def test_renderer_process_uses_checked_source_for_script_cwd_and_pythonpath(
    tmp_path, monkeypatch
):
    """A caller PYTHONPATH cannot make the checked renderer import another repo."""
    source = tmp_path / "source"
    checked_probe = source / "src" / "project_probe.py"
    checked_probe.parent.mkdir(parents=True)
    (checked_probe.parent / "__init__.py").write_text("")
    checked_probe.write_text("ORIGIN = __file__\n")
    rogue = tmp_path / "rogue"
    rogue_probe = rogue / "src" / "project_probe.py"
    rogue_probe.parent.mkdir(parents=True)
    (rogue_probe.parent / "__init__.py").write_text("")
    rogue_probe.write_text("ORIGIN = __file__\n")
    script = source / "scripts" / "render_policy.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import os, pathlib, sys\n"
        "from src import project_probe\n"
        "pathlib.Path(sys.argv[1]).write_text(\n"
        "    os.getcwd() + '\\n' + __file__ + '\\n' + project_probe.ORIGIN\n"
        ")\n"
    )
    monkeypatch.setattr(library_driver, "SOURCE_ROOT", source)
    monkeypatch.setenv("PYTHONPATH", str(rogue))
    observed = tmp_path / "observed.txt"

    library_driver.run_renderer_process([str(observed)])

    cwd, executed, imported = observed.read_text().splitlines()
    assert Path(cwd) == source
    assert Path(executed) == script
    assert Path(imported) == checked_probe


def test_child_project_import_preflight_resolves_beneath_checked_source():
    """The real child import graph must resolve its project modules in this worktree."""
    library_driver.verify_project_import_roots()


def test_inventory_is_portable_lf_only_and_runtime_path_is_not_serialized(tmp_path):
    """Frozen inventory bytes must not bind consumers to the transfer machine."""
    fq4, fq3, roots = _write_accepted_checkpoint_manifests(tmp_path)
    rows = library_driver.build_inventory(
        fq4, fq3, roots, expected_manifest_sha256=_fixture_manifest_hashes(fq4, fq3)
    )
    inventory = tmp_path / "checkpoint_inventory.tsv"
    library_driver.write_inventory(rows, inventory)

    payload = inventory.read_bytes()
    assert b"\r\n" not in payload
    assert str(tmp_path).encode() not in payload
    assert b"checkpoint_cache_path" in payload.splitlines()[0]
    assert all(not Path(row["checkpoint_cache_path"]).is_absolute() for row in rows)


def test_existing_inventory_and_sidecar_must_be_byte_identical(tmp_path):
    """Resume must stop if the committed membership bytes were edited or replaced."""
    fq4, fq3, roots = _write_accepted_checkpoint_manifests(tmp_path)
    rows = library_driver.build_inventory(
        fq4, fq3, roots, expected_manifest_sha256=_fixture_manifest_hashes(fq4, fq3)
    )
    inventory = tmp_path / "checkpoint_inventory.tsv"
    library_driver.write_inventory(rows, inventory)
    inventory.write_bytes(inventory.read_bytes() + b"tampered\n")

    with pytest.raises(ValueError, match="existing inventory"):
        library_driver.write_inventory(rows, inventory)


@pytest.mark.parametrize("missing", ["inventory", "sidecar"])
def test_matching_partial_inventory_pair_is_recovered_atomically(tmp_path, missing):
    """A crash between the two atomic writes must be safely resumable."""
    fq4, fq3, roots = _write_accepted_checkpoint_manifests(tmp_path)
    rows = library_driver.build_inventory(
        fq4, fq3, roots, expected_manifest_sha256=_fixture_manifest_hashes(fq4, fq3)
    )
    inventory = tmp_path / "checkpoint_inventory.tsv"
    sidecar = library_driver.write_inventory(rows, inventory)
    expected_inventory, expected_sidecar = inventory.read_bytes(), sidecar.read_bytes()
    (inventory if missing == "inventory" else sidecar).unlink()

    library_driver.write_inventory(rows, inventory)

    assert inventory.read_bytes() == expected_inventory
    assert sidecar.read_bytes() == expected_sidecar


def test_manifest_headers_and_registered_short_are_exact(tmp_path):
    """Schema additions and short-label drift cannot enter the frozen inventory."""
    fq4, fq3, roots = _write_accepted_checkpoint_manifests(tmp_path)
    _replace_tsv_cell(fq3, 0, "short", "not-f8")
    with pytest.raises(ValueError, match="short"):
        library_driver.build_inventory(
            fq4,
            fq3,
            roots,
            expected_manifest_sha256=_fixture_manifest_hashes(fq4, fq3),
        )

    fq4, fq3, roots = _write_accepted_checkpoint_manifests(tmp_path / "header")
    text = fq4.read_text().splitlines()
    text[0] += "\textra"
    text[1] += "\tvalue"
    fq4.write_text("\n".join(text) + "\n")
    with pytest.raises(ValueError, match="header"):
        library_driver.build_inventory(
            fq4,
            fq3,
            roots,
            expected_manifest_sha256=_fixture_manifest_hashes(fq4, fq3),
        )


def test_resume_contract_requires_horizon_80_and_exact_five_files(tmp_path):
    """Short rollouts or renderer scratch files cannot be reused as final leaves."""
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    for filename in (*library_driver.ARTIFACT_FILENAMES, "metadata.json"):
        (leaf / filename).write_text("x")
    metadata = {"rollout": {"requested_control_steps": 80}}
    assert library_driver.resume_leaf_shape_is_valid(leaf, metadata)

    metadata["rollout"]["requested_control_steps"] = 79
    assert not library_driver.resume_leaf_shape_is_valid(leaf, metadata)
    metadata["rollout"]["requested_control_steps"] = 80
    (leaf / "frame_0_idx000.png").write_text("scratch")
    assert not library_driver.resume_leaf_shape_is_valid(leaf, metadata)


def test_renderer_scratch_is_removed_and_staged_leaf_replaces_final_atomically(tmp_path):
    """Finalization must publish only the staged five-file leaf and remove scratch."""
    final_leaf = tmp_path / "fq4x8" / "F8" / "8"
    final_leaf.mkdir(parents=True)
    (final_leaf / "old.txt").write_text("old")
    staged_leaf = tmp_path / ".staging" / "fq4x8" / "F8" / "8"
    staged_leaf.mkdir(parents=True)
    for filename in (*library_driver.ARTIFACT_FILENAMES, "metadata.json"):
        (staged_leaf / filename).write_text("new")
    (staged_leaf / "step_000_reset.png").write_text("scratch")
    (staged_leaf / "frame_0_idx000.png").write_text("scratch")

    library_driver.remove_renderer_scratch(staged_leaf)
    library_driver.replace_policy_leaf(staged_leaf, final_leaf)

    assert {path.name for path in final_leaf.iterdir()} == set(
        (*library_driver.ARTIFACT_FILENAMES, "metadata.json")
    )
    assert not staged_leaf.exists()


def test_resume_restores_one_interrupted_replacement_backup_before_validation(
    tmp_path, monkeypatch
):
    """A crash after parking the old leaf must reuse that complete backup."""
    row = {
        **expected_rows()[0],
        "_checkpoint_file": "unused-model_499.pt",
        "task": TASKS[("fq4x8", "F8")],
    }
    final_leaf = library_driver.policy_artifact_dir(tmp_path, row)
    backup = final_leaf.with_name(f".{final_leaf.name}.backup-interrupted")
    backup.mkdir(parents=True)
    (backup / "sentinel").write_text("old leaf")
    monkeypatch.setattr(
        library_driver,
        "_policy_artifact_is_valid",
        lambda output_root, candidate, fixed_reset, *, render_provenance: (
            library_driver.policy_artifact_dir(output_root, candidate) / "sentinel"
        ).is_file(),
    )
    monkeypatch.setattr(
        library_driver,
        "run_renderer_process",
        lambda _arguments: pytest.fail("restored backup should be validated before rendering"),
    )

    reused = library_driver.run_single_policy_renderer(
        row,
        canonical_fixed_reset_payload(),
        tmp_path,
        render_provenance=RENDER_PROVENANCE,
    )

    assert reused
    assert (final_leaf / "sentinel").read_text() == "old leaf"
    assert not backup.exists()


def test_resume_removes_one_stale_generated_backup_when_final_leaf_exists(tmp_path):
    """A completed replacement may leave one generated backup to clean up."""
    final_leaf = tmp_path / "fq4x8" / "F8" / "8"
    final_leaf.mkdir(parents=True)
    backup = final_leaf.with_name(f".{final_leaf.name}.backup-stale")
    backup.mkdir()

    library_driver.recover_policy_leaf_backup(final_leaf)

    assert final_leaf.is_dir()
    assert not backup.exists()


def test_resume_fails_closed_on_ambiguous_replacement_backups(tmp_path):
    """Two interrupted candidates cannot be selected without inventing history."""
    final_leaf = tmp_path / "fq4x8" / "F8" / "8"
    final_leaf.parent.mkdir(parents=True)
    for suffix in ("one", "two"):
        final_leaf.with_name(f".{final_leaf.name}.backup-{suffix}").mkdir()

    with pytest.raises(ValueError, match="ambiguous"):
        library_driver.recover_policy_leaf_backup(final_leaf)


def test_policy_staging_root_is_a_sibling_of_the_library(tmp_path):
    """A killed render may leave a sibling scratch root, never an unexpected library child."""
    library = tmp_path / "library"
    staging = library_driver.make_policy_staging_root(library)
    try:
        assert staging.parent == library.parent
        assert staging != library
    finally:
        staging.rmdir()


def test_final_validation_prints_the_six_required_counters():
    """Successful fail-closed validation must emit the complete review summary."""
    rows = expected_rows()
    validation = {
        "artifacts": [
            {
                "reset_state_digest": FIXED_DIGEST,
                "rollout": {"auto_reset_enabled": False},
            }
            for _ in rows
        ]
    }
    assert library_driver.format_validation_counters(rows, validation) == (
        "policies=56\n"
        "unique_checkpoint_sha256=56\n"
        "unique_reset_state_digest=1\n"
        "missing_artifacts=0\n"
        "invalid_video=0\n"
        "second_episode_contamination=0"
    )


def test_validation_counter_is_derived_from_auto_reset_evidence():
    """The contamination count must be computed from admitted rollout metadata."""
    rows = expected_rows()
    validation = {
        "artifacts": [
            {
                "reset_state_digest": FIXED_DIGEST,
                "rollout": {"auto_reset_enabled": index == 0},
            }
            for index, _row in enumerate(rows)
        ]
    }

    assert library_driver.format_validation_counters(rows, validation).endswith(
        "second_episode_contamination=1"
    )


def test_metadata_contract_rejects_auto_reset_enabled_rollout(tmp_path):
    """A rollout with automatic episode replacement cannot enter the library."""
    row = expected_rows()[0]
    metadata = {
        "task": TASKS[(row["campaign"], row["arm"])],
        "code_revision": RENDER_PROVENANCE["code_revision"],
        "asset_revision": RENDER_PROVENANCE["asset_revision"],
        "presentation_generator_revision": PRESENTATION_REVISION,
        "renderer_contract": RENDER_CONTRACT,
        "timing": TIMING,
        "rollout": {
            "requested_control_steps": 80,
            "executed_control_steps": 0,
            "frame_count": 1,
            "auto_reset_enabled": True,
            "terminal_boundary": {
                "detected": False,
                "step": None,
                "reason": "step_limit",
            },
        },
        "output_dimensions_px": {"frame": [960, 720], "montage": [960, 720]},
        "renderer_provenance": copy.deepcopy(RENDER_PROVENANCE),
        "training_provenance": {
            key: row[key]
            for key in (
                "checkpoint_path",
                "training_code_revision",
                "training_asset_revision",
                "accepted_training_manifest_sha256",
            )
        },
    }

    with pytest.raises(ValueError, match="auto-reset"):
        library_driver._validate_metadata_contract(metadata, row, tmp_path / "leaf")


def test_metadata_contract_requires_valid_presentation_revision(tmp_path):
    """Every trajectory presentation must name its own 40-hex generator revision."""
    row = expected_rows()[0]
    metadata = {
        "task": TASKS[(row["campaign"], row["arm"])],
        "code_revision": RENDER_PROVENANCE["code_revision"],
        "asset_revision": RENDER_PROVENANCE["asset_revision"],
        "presentation_generator_revision": "not-a-revision",
        "renderer_contract": RENDER_CONTRACT,
        "timing": TIMING,
        "rollout": {
            "requested_control_steps": 80,
            "executed_control_steps": 0,
            "frame_count": 1,
            "auto_reset_enabled": False,
            "terminal_boundary": {
                "detected": False,
                "step": None,
                "reason": "step_limit",
            },
        },
        "output_dimensions_px": {"frame": [960, 720], "montage": [960, 720]},
        "renderer_provenance": copy.deepcopy(RENDER_PROVENANCE),
        "training_provenance": {
            key: row[key]
            for key in (
                "checkpoint_path",
                "training_code_revision",
                "training_asset_revision",
                "accepted_training_manifest_sha256",
            )
        },
    }

    with pytest.raises(ValueError, match="presentation"):
        library_driver._validate_metadata_contract(metadata, row, tmp_path / "leaf")


def test_final_validator_requires_one_presentation_revision_across_library(
    tmp_path, monkeypatch
):
    """A mixed trajectory-presentation generation cannot be reported as one library."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    metadata_path = tmp_path / "fq4x8" / "F8" / "8" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["presentation_generator_revision"] = "e" * 40
    del metadata["metadata_payload_sha256"]
    _write_metadata(metadata_path, metadata)
    monkeypatch.setattr(
        "evaluation.analysis.fixed_reset_video_library._readable_artifact",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "evaluation.analysis.fixed_reset_video_library._validate_media_properties",
        lambda _leaf, _rollout: None,
    )

    with pytest.raises(ValueError, match="presentation"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_requires_the_anchored_reference_polyline(tmp_path):
    """Every trajectory must preserve its finite reference polyline."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    trace_path = next(tmp_path.glob("*/*/*/trace.npz"))
    with np.load(trace_path) as trace:
        np.savez(
            trace_path,
            head_position_m=trace["head_position_m"],
            contact=trace["contact"],
            action=trace["action"],
            nail_top_m=trace["nail_top_m"],
        )
    metadata_path = trace_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["artifacts"]["trace.npz"] = _sha256(trace_path)
    del metadata["metadata_payload_sha256"]
    _write_metadata(metadata_path, metadata)

    with pytest.raises(ValueError, match="reference_polyline_m"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_preserves_historical_three_vertex_reference(tmp_path, monkeypatch):
    """The historical 56-policy traces retain their accepted three-vertex geometry."""
    monkeypatch.setattr(
        "evaluation.analysis.fixed_reset_video_library.load_fixed_reset",
        lambda _path: {"reset_state_digest": FIXED_DIGEST},
    )
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)

    validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_requires_identical_reference_geometry_for_the_frozen_reset(tmp_path):
    """A different valid-looking three-vertex reference cannot enter the shared-reset library."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    trace_path = next(tmp_path.glob("*/*/*/trace.npz"))
    with np.load(trace_path) as trace:
        mutated = {key: trace[key] for key in trace.files}
    mutated["reference_polyline_m"] = mutated["reference_polyline_m"].copy()
    mutated["reference_polyline_m"][1, 0] += 0.01
    np.savez(trace_path, **mutated)
    metadata_path = trace_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["artifacts"]["trace.npz"] = _sha256(trace_path)
    del metadata["metadata_payload_sha256"]
    _write_metadata(metadata_path, metadata)

    with pytest.raises(ValueError, match="reference geometry differs"):
        validate_policy_artifacts(tmp_path, expected_rows())


def test_artifact_validator_accepts_a_uniform_direct_two_endpoint_reference(tmp_path, monkeypatch):
    """Direct-reference artifacts may use the reset-head and follow-through endpoints."""
    monkeypatch.setattr(
        "evaluation.analysis.fixed_reset_video_library.load_fixed_reset",
        lambda _path: {"reset_state_digest": FIXED_DIGEST},
    )
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    for trace_path in tmp_path.glob("*/*/*/trace.npz"):
        with np.load(trace_path) as trace:
            mutated = {key: trace[key] for key in trace.files}
        mutated["reference_polyline_m"] = mutated["reference_polyline_m"][[0, 2]]
        np.savez(trace_path, **mutated)
        metadata_path = trace_path.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["artifacts"]["trace.npz"] = _sha256(trace_path)
        del metadata["metadata_payload_sha256"]
        _write_metadata(metadata_path, metadata)

    validate_policy_artifacts(tmp_path, expected_rows())


def test_renderer_reference_vertices_match_the_anchored_phi_waypoints():
    """The renderer saves the exact direct-reference phi={0, 1} endpoints."""
    reference = SingleStrikeReference(1, "cpu", overshoot=0.02)
    head = torch.tensor([[0.50, 0.01, 0.12]])
    nail = torch.tensor([[0.48, -0.02, 0.10]])
    reference.update(head, nail, torch.zeros(1, dtype=torch.long))

    assert callable(getattr(render_policy, "_reference_polyline_m", None))
    actual = render_policy._reference_polyline_m(reference, "cpu")
    expected = np.array(
        [
            head.numpy()[0],
            [0.48, -0.02, 0.08],
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    "reference",
    (
        np.zeros((1, 3)),
        np.zeros((4, 3)),
        np.zeros((2, 2)),
        np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
    ),
)
def test_trajectory_plot_rejects_invalid_reference_polylines(tmp_path, reference):
    """Only finite direct or historical reference polylines are valid trace geometry."""
    with pytest.raises(ValueError, match="reference_polyline_m"):
        write_trajectory_png(
            {
                "head_position_m": np.array([[0.0, 0.0, 0.0]]),
                "reference_polyline_m": reference,
                "nail_top_m": np.array([0.0, 0.0, 0.0]),
            },
            tmp_path / "trajectory.png",
        )


def test_trajectory_plot_draws_the_reference_as_a_dashed_observation_line(tmp_path, monkeypatch):
    """The reference is visibly distinct from the realized, reward-free path."""
    observed: list[dict] = []
    from matplotlib.axes import Axes

    original_plot = Axes.plot

    def capture_plot(self, *args, **kwargs):
        observed.append(kwargs)
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
    write_trajectory_png(
        {
            "head_position_m": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, -0.1]]),
            "contact": np.array([False, True]),
            "reference_polyline_m": np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]
            ),
            "nail_top_m": np.array([0.0, 0.0, 0.0]),
        },
        tmp_path / "trajectory.png",
    )

    assert any(
        call.get("label") == "SingleStrikeReference (observation only)"
        and call.get("color") == "black"
        and call.get("linestyle") == "--"
        for call in observed
    )


def test_trajectory_plot_uses_readable_figure_level_layout(tmp_path, monkeypatch):
    """Near-vertical traces must fill both panels without data-overlay annotations."""
    from evaluation.analysis import fixed_reset_video_library as video_contract

    original_close = video_contract.plt.close
    monkeypatch.setattr(video_contract.plt, "close", lambda _figure: None)
    before = set(video_contract.plt.get_fignums())
    write_trajectory_png(
        {
            "head_position_m": np.array(
                [
                    [0.500, 0.010, 0.300],
                    [0.503, 0.006, 0.235],
                    [0.497, -0.002, 0.165],
                    [0.501, -0.006, 0.100],
                ]
            ),
            "contact": np.array([False, False, False, True]),
            "reference_polyline_m": np.array(
                [[0.500, 0.010, 0.300], [0.500, 0.000, 0.350], [0.500, 0.000, 0.080]]
            ),
            "nail_top_m": np.array([0.500, 0.000, 0.100]),
        },
        tmp_path / "trajectory.png",
    )
    figure_number = (set(video_contract.plt.get_fignums()) - before).pop()
    figure = video_contract.plt.figure(figure_number)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    assert len(figure.axes) == 2
    assert all(axis.get_aspect() == "auto" for axis in figure.axes)
    assert all(axis.get_legend() is None for axis in figure.axes)
    annotation = next(
        text for text in figure.texts
        if text.get_text() == "SingleStrikeReference (observation only) · black dashed · not rewarded"
    )
    annotation_box = annotation.get_window_extent(renderer)
    for axis in figure.axes:
        assert len(axis.get_xticklabels()) <= 6
        assert len(axis.get_yticklabels()) <= 6
        assert not annotation_box.overlaps(axis.get_tightbbox(renderer))
    original_close(figure)


def test_campaign_grids_and_index_cover_each_registered_policy_once(tmp_path, monkeypatch):
    """Campaign artifacts provide common side-view limits and resolvable policy links."""
    from matplotlib.figure import Figure

    subtitles: list[str] = []
    original_suptitle = Figure.suptitle

    def capture_suptitle(self, text, *args, **kwargs):
        subtitles.append(str(text))
        return original_suptitle(self, text, *args, **kwargs)

    monkeypatch.setattr(Figure, "suptitle", capture_suptitle)
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    rows = expected_rows()
    fq4 = write_campaign_trajectory_grid(
        tmp_path, "fq4x8", tmp_path / "fq4x8_trajectories_grid.png"
    )
    fq3 = write_campaign_trajectory_grid(
        tmp_path, "fq3x8", tmp_path / "fq3x8_trajectories_grid.png"
    )
    index = write_library_index(
        tmp_path,
        rows,
        generation_command="python generate_fixed_reset_library.py --frozen-inventory",
        validation_command="python validate_fixed_reset_library.py --all-56",
    )

    assert fq4["common_xlim"] and fq4["common_zlim"]
    assert fq3["common_xlim"] and fq3["common_zlim"]
    assert len(fq4["panel_limits"]) == 32
    assert len(fq3["panel_limits"]) == 24
    assert all(limits == (fq4["common_xlim"], fq4["common_zlim"]) for limits in fq4["panel_limits"])
    assert all(limits == (fq3["common_xlim"], fq3["common_zlim"]) for limits in fq3["panel_limits"])
    assert all(
        "black dashed=SingleStrikeReference (observation only; r_imit disabled/not rewarded)"
        in subtitle
        for subtitle in subtitles
    )
    assert iio.imread(tmp_path / "fq4x8_trajectories_grid.png").size > 0
    assert iio.imread(tmp_path / "fq3x8_trajectories_grid.png").size > 0
    text = index.read_text()
    assert "black dashed=SingleStrikeReference (observation only; r_imit disabled/not rewarded)" in text
    assert "## Reproduction" in text
    assert "Generation command:" in text
    assert "Validation command:" in text
    assert "`python generate_fixed_reset_library.py --frozen-inventory`" in text
    assert "`python validate_fixed_reset_library.py --all-56`" in text
    assert _sha256(tmp_path / "fq4x8_trajectories_grid.png") in text
    assert _sha256(tmp_path / "fq3x8_trajectories_grid.png") in text
    for relative in re.findall(r"\]\(([^)]+)\)", text):
        assert (index.parent / relative).is_file()
    for row in rows:
        identity = f"{row['campaign']}/{row['arm']}/{row['training_seed']}"
        assert text.count(f"| {identity} |") == 1
        assert text.count(row["checkpoint_sha256"]) == 1
        assert f"[metadata.json]({identity}/metadata.json)" in text
        assert _sha256(tmp_path / identity / "metadata.json") in text
        for artifact in ("policy.mp4", "montage.png", "trajectory.png", "trace.npz"):
            assert (tmp_path / row["campaign"] / row["arm"] / str(row["training_seed"]) / artifact).is_file()


def test_both_campaign_grids_use_one_limit_pair_from_all_56_traces(tmp_path):
    """A campaign-specific outlier must expand every panel in both comparison grids."""
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    outlier_path = tmp_path / "fq3x8" / "FQ" / "23" / "trace.npz"
    with np.load(outlier_path) as trace:
        outlier = {key: trace[key] for key in trace.files}
    outlier["head_position_m"] = np.array([[2.0, 0.0, 0.0]])
    np.savez(outlier_path, **outlier)

    fq4 = write_campaign_trajectory_grid(tmp_path, "fq4x8", tmp_path / "fq4.png")
    fq3 = write_campaign_trajectory_grid(tmp_path, "fq3x8", tmp_path / "fq3.png")

    shared = (fq4["common_xlim"], fq4["common_zlim"])
    assert shared == (fq3["common_xlim"], fq3["common_zlim"])
    assert len(fq4["panel_limits"]) == 32
    assert len(fq3["panel_limits"]) == 24
    assert all(limits == shared for limits in (*fq4["panel_limits"], *fq3["panel_limits"]))


# --- Wave-1 500 Hz substep evidence -----------------------------------------


def _wave1_trace(
    *,
    control_steps: int = 4,
    decimation: int = 10,
    upsampled_50hz: bool = False,
    episode_index_tail: int = 0,
) -> dict[str, np.ndarray]:
    """Build a synthetic but structurally honest Wave-1 substep trace."""
    entry = np.array([0.5, 0.0, 0.25], dtype=np.float64)
    nail = np.array([0.5, 0.0, 0.10], dtype=np.float64)
    n = control_steps * decimation
    if upsampled_50hz:
        # One position per CONTROL step, each repeated `decimation` times.
        control_path = entry + np.linspace(0.0, 1.0, control_steps)[:, None] * (
            nail - entry
        )
        positions = np.repeat(control_path, decimation, axis=0)
    else:
        positions = entry + np.linspace(0.0, 1.0, n)[:, None] * (nail - entry)
    gate_centers = entry + (
        np.arange(1, 7, dtype=np.float64)[:, None] / 7.0
    ) * (nail - entry)
    contact = np.zeros(n, dtype=bool)
    contact[-3:] = True
    episode_index = np.zeros(n, dtype=np.int64)
    if episode_index_tail:
        episode_index[-episode_index_tail:] = 1
    boundary = np.zeros(n, dtype=bool)
    boundary[decimation - 1 :: decimation] = True
    return {
        "substep_head_position_m": positions,
        "substep_contact": contact,
        "substep_nail_depth_m": np.linspace(0.0, 0.032, n),
        "substep_arm_qvel_rad_s": np.zeros((n, 6), dtype=np.float64),
        "substep_arm_qvel_pre_rad_s": np.zeros((n, 6), dtype=np.float64),
        "arm_joint_names": np.array([f"joint{i}" for i in range(1, 7)]),
        "substep_gate_index": np.clip(
            np.arange(n) // max(1, n // 7), 0, 6
        ).astype(np.int64),
        "substep_perpendicular_error_m": np.zeros(n, dtype=np.float64),
        "substep_d_start_m": np.full(n, 0.021, dtype=np.float64),
        "substep_f_best": np.linspace(0.0, 1.0, n),
        "substep_control_step": (np.arange(n) // decimation).astype(np.int64),
        "substep_is_control_boundary": boundary,
        "substep_episode_index": episode_index,
        "control_step_reward_terms": np.zeros((control_steps, 3), dtype=np.float64),
        "control_step_reward_term_names": np.array(
            ["approach", "completion", "r_gate"]
        ),
        "control_step_reward_total": np.zeros(control_steps, dtype=np.float64),
        "guideline_entry_m": entry,
        "guideline_nail_m": nail,
        "guideline_gate_centers_m": gate_centers,
        "guideline_reference_length_m": np.float64(
            float(np.linalg.norm(nail - entry))
        ),
        "physics_dt_s": np.float64(0.002),
        "control_decimation": np.int64(decimation),
        "executed_control_steps": np.int64(control_steps),
    }


def test_wave1_substep_trace_accepts_a_genuine_500hz_trace():
    """A trace with real intra-window motion is accepted and reports its rate."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    report = validate_substep_trace(_wave1_trace())

    assert report["sample_rate_hz"] == pytest.approx(500.0)
    assert report["substep_count"] == 40
    assert report["executed_control_steps"] == 4
    # 9 of every 10 transitions are interior, so an honest path puts ~0.9 of its
    # displacement inside control windows.
    assert report["interior_motion_share"] == pytest.approx(0.9, abs=0.05)


def test_wave1_substep_trace_rejects_upsampled_50hz_positions():
    """A 50 Hz array repeated to 500 Hz length must be refused, not relabeled."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    with pytest.raises(ValueError, match="not genuinely per-substep"):
        validate_substep_trace(_wave1_trace(upsampled_50hz=True))


def test_wave1_substep_trace_rejects_post_reset_samples():
    """Any sample from a second episode invalidates the first-episode trace."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    with pytest.raises(ValueError, match="post-reset"):
        validate_substep_trace(_wave1_trace(episode_index_tail=5))


def test_wave1_substep_trace_rejects_substep_count_off_the_control_grid():
    """The substep count must be exactly decimation x executed control steps."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    trace = _wave1_trace()
    for key in ("substep_head_position_m", "substep_contact"):
        trace[key] = trace[key][:-1]
    with pytest.raises(ValueError, match="control grid"):
        validate_substep_trace(trace)


def test_wave1_substep_trace_requires_tracker_geometry_and_payouts():
    """Missing tracker geometry or manager payouts must fail closed."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    for missing in (
        "guideline_entry_m",
        "guideline_nail_m",
        "guideline_gate_centers_m",
        "control_step_reward_terms",
        "substep_f_best",
        "substep_d_start_m",
    ):
        trace = _wave1_trace()
        del trace[missing]
        with pytest.raises(ValueError, match="missing"):
            validate_substep_trace(trace)


def test_wave1_trajectory_png_uses_tracker_geometry_not_single_strike_reference(tmp_path):
    """The dashed guideline must be the tracker entry->nail segment."""
    from evaluation.analysis.fixed_reset_video_library import (
        write_substep_trajectory_png,
    )

    trace = _wave1_trace()
    path = tmp_path / "trajectory.png"
    report = write_substep_trajectory_png(trace, path)

    assert path.is_file()
    assert np.asarray(iio.imread(path)).size > 0
    assert report["reference_source"] == "waypoint_progress_tracker_entry_to_nail"
    np.testing.assert_allclose(
        report["reference_endpoints_m"],
        np.vstack((trace["guideline_entry_m"], trace["guideline_nail_m"])),
    )
    assert report["gate_disk_count"] == 6
    assert report["contact_sample_count"] == 3
    assert report["control_boundary_marker_count"] == 4
    # Each panel must be metrically square, and both panels must share one scale.
    for panel in ("xz", "xy"):
        assert report["axis_x_half_span_m"][panel] == pytest.approx(
            report["axis_half_span_m"][panel]
        )
    assert report["axis_half_span_m"]["xz"] == pytest.approx(
        report["axis_half_span_m"]["xy"]
    )


def test_wave1_trajectory_png_refuses_a_trace_that_failed_validation(tmp_path):
    """The plotter must not silently draw an upsampled 50 Hz trace."""
    from evaluation.analysis.fixed_reset_video_library import (
        write_substep_trajectory_png,
    )

    with pytest.raises(ValueError, match="not genuinely per-substep"):
        write_substep_trajectory_png(
            _wave1_trace(upsampled_50hz=True), tmp_path / "bad.png"
        )


def test_wave1_renderer_contract_is_50fps_stride_two():
    """One RGB frame per two physics substeps, encoded at 50 fps."""
    from evaluation.analysis.fixed_reset_video_library import (
        SUBSTEP_RENDERER_CONTRACT,
    )

    assert SUBSTEP_RENDERER_CONTRACT["fps"] == 50
    assert SUBSTEP_RENDERER_CONTRACT["frame_substep_stride"] == 2
    for key in ("frame_width_px", "frame_height_px", "camera_distance_m",
                "camera_elevation_deg", "camera_azimuth_deg"):
        assert SUBSTEP_RENDERER_CONTRACT[key] == RENDER_CONTRACT[key]


def test_wave1_campaign_arms_resolve_to_the_registered_guideline_tasks():
    """Wave-1 arm identities must map to the exact trained task IDs."""
    from evaluation.analysis.fixed_reset_video_library import expected_task

    prefix = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
    assert expected_task("wave1", "C0") == prefix + "C0"
    assert expected_task("wave1", "G") == prefix + "CGate"
    assert expected_task("wave1", "P") == prefix + "CProgress"


def _write_valid_wave1_leaf(tmp_path):
    """Render-free stand-in for a complete, self-consistent Wave-1 leaf."""
    from evaluation.analysis.fixed_reset_video_library import (
        SUBSTEP_RENDERER_CONTRACT,
        WAVE1_ARTIFACT_FILENAMES,
        write_metadata,
        write_substep_trajectory_png,
    )

    leaf = tmp_path / "c0_seed2"
    leaf.mkdir()
    trace = _wave1_trace()
    np.savez(leaf / "trace.npz", **trace)
    write_substep_trajectory_png(trace, leaf / "trajectory.png")
    frames = np.zeros((20, 720, 960, 3), dtype=np.uint8)
    frames[:, ::2, ::2] = 255
    iio.imwrite(leaf / "policy.mp4", frames, fps=50)
    iio.imwrite(leaf / "montage.png", np.concatenate(list(frames[:6]), axis=1))
    expectations = {
        "arm": "C0",
        "training_seed": 2,
        "checkpoint_sha256": "a" * 64,
        "training_revision": "b" * 40,
        "asset_revision": "c" * 40,
        "analysis_revision": "d" * 40,
        "reset_state_digest": FIXED_DIGEST,
    }
    write_metadata(
        leaf / "metadata.json",
        {
            "campaign": "wave1",
            "arm": "C0",
            "training_seed": 2,
            "task": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
            "checkpoint_sha256": "a" * 64,
            "training_revision": "b" * 40,
            "asset_revision": "c" * 40,
            "analysis_revision": "d" * 40,
            "reset_state_digest": FIXED_DIGEST,
            "renderer_contract": SUBSTEP_RENDERER_CONTRACT,
            "timing": TIMING,
            "rollout": {
                "requested_control_steps": 80,
                "executed_control_steps": 4,
                "substep_count": 40,
                "frame_count": 20,
                "auto_reset_enabled": False,
                "terminal_boundary": {"detected": True, "step": 4, "reason": "terminated"},
            },
            "artifacts": {
                name: _sha256(leaf / name) for name in WAVE1_ARTIFACT_FILENAMES
            },
        },
    )
    return leaf, expectations


def test_wave1_artifact_validator_binds_checkpoint_reset_and_three_revisions(tmp_path):
    """A rendered leaf must bind its checkpoint, fixed reset, and all revisions."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )

    leaf, expectations = _write_valid_wave1_leaf(tmp_path)
    report = validate_wave1_policy_artifacts(leaf, expectations)

    assert report["substep_count"] == 40
    assert report["measured_fps"] == pytest.approx(50.0)
    assert report["measured_frame_count"] == 20
    assert report["revisions"] == {
        "training": "b" * 40, "asset": "c" * 40, "analysis": "d" * 40,
    }


def test_wave1_artifact_validator_rejects_swapped_media(tmp_path):
    """Replacing a validated artifact must break its recorded content hash."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )

    leaf, expectations = _write_valid_wave1_leaf(tmp_path)
    frames = np.full((20, 720, 960, 3), 7, dtype=np.uint8)
    frames[:, ::3, ::3] = 240
    iio.imwrite(leaf / "policy.mp4", frames, fps=50)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_wave1_policy_artifacts(leaf, expectations)


def test_wave1_artifact_validator_rejects_an_edited_metadata_payload(tmp_path):
    """Editing metadata without re-deriving its payload digest must fail closed."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )

    leaf, expectations = _write_valid_wave1_leaf(tmp_path)
    metadata = json.loads((leaf / "metadata.json").read_text())
    metadata["rollout"]["requested_control_steps"] = 999
    (leaf / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="payload digest mismatch"):
        validate_wave1_policy_artifacts(leaf, expectations)


def test_wave1_artifact_validator_rejects_a_boundary_that_contradicts_the_rollout(tmp_path):
    """A terminal boundary must land on the last executed control step."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
        write_metadata,
    )

    leaf, expectations = _write_valid_wave1_leaf(tmp_path)
    metadata = json.loads((leaf / "metadata.json").read_text())
    metadata.pop("metadata_payload_sha256")
    metadata["rollout"]["terminal_boundary"]["step"] = 3
    write_metadata(leaf / "metadata.json", metadata)

    with pytest.raises(ValueError, match="terminal boundary mismatch"):
        validate_wave1_policy_artifacts(leaf, expectations)


def test_wave1_artifact_validator_rejects_a_checkpoint_hash_mismatch(tmp_path):
    """A leaf whose metadata hash differs from the frozen manifest must fail."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )

    leaf, expectations = _write_valid_wave1_leaf(tmp_path)
    metadata = json.loads((leaf / "metadata.json").read_text())
    from evaluation.analysis.fixed_reset_video_library import write_metadata

    metadata.pop("metadata_payload_sha256")
    metadata["checkpoint_sha256"] = "e" * 64
    write_metadata(leaf / "metadata.json", metadata)

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        validate_wave1_policy_artifacts(leaf, expectations)


def test_wave1_gate_disk_radius_matches_the_tracker():
    """The plotted disk radius must not drift from the tracker's gate radius."""
    from evaluation.analysis import fixed_reset_video_library as lib
    from src.tasks.hammer.mdp.guideline import GUIDELINE_GATE_RADIUS_M

    assert lib.GUIDELINE_GATE_RADIUS_M == GUIDELINE_GATE_RADIUS_M


# --- Wave-1 renderer (scripts/render_policy.py) ------------------------------


def _recorded_substeps(
    *, control_steps: int = 4, decimation: int = 10, episode_tail: int = 0
) -> dict:
    """Per-substep recordings as the render hook accumulates them."""
    entry = np.array([0.5, 0.0, 0.25])
    nail = np.array([0.5, 0.0, 0.10])
    n = control_steps * decimation
    fractions = np.linspace(0.0, 1.0, n)
    episode = [0] * n
    for i in range(episode_tail):
        episode[n - 1 - i] = 1
    return {
        "head_positions": [entry + f * (nail - entry) for f in fractions],
        "contacts": [bool(f > 0.93) for f in fractions],
        "nail_depths": list(np.linspace(0.0, 0.032, n)),
        "arm_qvels": [np.zeros(6) for _ in range(n)],
        "arm_qvels_pre": [np.zeros(6) for _ in range(n)],
        "arm_joint_names": [f"joint{i}" for i in range(1, 7)],
        "control_steps_recorded": [i // decimation for i in range(n)],
        "gate_indices": [min(6, int(f * 7)) for f in fractions],
        "perpendicular_errors": [0.0] * n,
        "d_starts": [0.021] * n,
        "f_bests": list(fractions),
        "episode_indices": episode,
        "control_step_payouts": [np.zeros(3) for _ in range(control_steps)],
        "control_step_totals": [0.0] * control_steps,
        "payout_names": ["approach", "completion", "r_gate"],
        "entry": entry,
        "nail": nail,
        "physics_dt_s": 0.002,
        "control_decimation": decimation,
        "executed_control_steps": control_steps,
    }


def test_wave1_frame_indices_are_every_second_substep():
    """One RGB frame per two physics substeps, taken at the end of each pair."""
    indices = render_policy.wave1_frame_substep_indices(40, 2)

    assert indices.tolist() == list(range(1, 40, 2))
    assert len(indices) == 20


def test_wave1_trace_builder_produces_a_validated_500hz_trace():
    """The builder emits a trace the frozen validator accepts as genuine 500 Hz."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    trace = render_policy.build_wave1_substep_trace(**_recorded_substeps())
    report = validate_substep_trace(trace)

    assert report["substep_count"] == 40
    assert report["executed_control_steps"] == 4
    assert report["sample_rate_hz"] == pytest.approx(500.0)
    assert trace["guideline_gate_centers_m"].shape == (6, 3)
    # Payouts stay at CONTROL rate; they are never broadcast to substep length.
    assert trace["control_step_reward_terms"].shape == (4, 3)


def test_wave1_trace_builder_refuses_substep_broadcast_reward_payouts():
    """Repeating a control-rate payout once per substep is fabrication, not data."""
    recorded = _recorded_substeps()
    recorded["control_step_payouts"] = [np.zeros(3) for _ in range(40)]
    recorded["control_step_totals"] = [0.0] * 40

    with pytest.raises(ValueError, match="control rate"):
        render_policy.build_wave1_substep_trace(**recorded)


def test_wave1_trace_builder_rejects_post_reset_substeps():
    """Samples recorded after an episode boundary must not reach the trace."""
    with pytest.raises(ValueError, match="post-reset"):
        render_policy.build_wave1_substep_trace(**_recorded_substeps(episode_tail=4))


def test_wave1_trace_builder_rejects_a_partial_control_window():
    """A truncated final window would misalign every control-rate join."""
    recorded = _recorded_substeps()
    for key in ("head_positions", "contacts", "nail_depths", "arm_qvels",
                "arm_qvels_pre", "gate_indices", "perpendicular_errors", "d_starts",
                "f_bests", "episode_indices", "control_steps_recorded"):
        recorded[key] = recorded[key][:-1]

    with pytest.raises(ValueError, match="control grid"):
        render_policy.build_wave1_substep_trace(**recorded)


def test_wave1_renderer_requires_a_distinct_analysis_revision():
    """Analysis provenance must be recorded separately from the training revision."""
    training = "a" * 40
    with pytest.raises(ValueError, match="analysis_revision"):
        render_policy.validate_wave1_revisions(
            training_revision=training, asset_revision="b" * 40,
            analysis_revision=training,
        )
    with pytest.raises(ValueError, match="analysis_revision"):
        render_policy.validate_wave1_revisions(
            training_revision=training, asset_revision="b" * 40,
            analysis_revision="",
        )
    assert render_policy.validate_wave1_revisions(
        training_revision=training, asset_revision="b" * 40,
        analysis_revision="c" * 40,
    ) == {"training": training, "asset": "b" * 40, "analysis": "c" * 40}


# --- reviewer-supplied attacks on the 500 Hz proof ---------------------------


def _repeat_fake(noise: float = 0.0, hold: int = 10, seed: int = 0) -> dict:
    """A control-rate path repeated to substep length, optionally perturbed."""
    trace = _wave1_trace()
    entry = trace["guideline_entry_m"]
    nail = trace["guideline_nail_m"]
    control = entry + np.linspace(0.0, 1.0, 4)[:, None] * (nail - entry)
    positions = np.repeat(control, 10, axis=0)
    if hold < 10:
        # staircase: nudge the first few substeps of each window, then hold
        for window in range(4):
            for i in range(10 - hold):
                positions[window * 10 + i] = control[window] + (i + 1) * 1e-9
    if noise:
        rng = np.random.default_rng(seed)
        positions = positions + rng.normal(0.0, noise, positions.shape)
    positions[0] = entry  # keep the anchored-entry invariant satisfied
    trace["substep_head_position_m"] = positions
    return trace


@pytest.mark.parametrize(
    "fake",
    [
        pytest.param(_repeat_fake(), id="exact-repeat"),
        pytest.param(_repeat_fake(noise=1e-12), id="repeat-plus-float-noise"),
        pytest.param(_repeat_fake(noise=1e-9, seed=3), id="repeat-plus-larger-noise"),
        pytest.param(_repeat_fake(hold=8), id="staircase-with-nudges"),
    ],
)
def test_wave1_substep_trace_rejects_every_repeated_control_rate_fake(fake):
    """Sprinkling float noise on a repeated 50 Hz path must not buy acceptance."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    with pytest.raises(ValueError, match="not genuinely per-substep"):
        validate_substep_trace(fake)


def test_wave1_substep_trace_accepts_a_trace_whose_arm_genuinely_settles():
    """An honest capture that moves, strikes, then holds still must NOT be rejected."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    trace = _wave1_trace()
    positions = trace["substep_head_position_m"].copy()
    positions[20:] = positions[20]  # arm parks after the strike; bitwise identical
    trace["substep_head_position_m"] = positions
    contact = np.zeros(40, dtype=bool)
    contact[19:] = True
    trace["substep_contact"] = contact

    report = validate_substep_trace(trace)

    assert report["substep_count"] == 40
    assert report["interior_motion_share"] > 0.5


def _zero_order_hold_fake(phase: int) -> dict:
    """A 50 Hz path held across each control window, at an arbitrary phase.

    Phases other than 0 put every transition INSIDE a window, which defeats a
    motion-share test on its own; only the distinct-sample count catches them.
    """
    trace = _wave1_trace()
    entry = trace["guideline_entry_m"]
    nail = trace["guideline_nail_m"]
    control = entry + np.linspace(0.0, 1.0, 5)[:, None] * (nail - entry)
    index = np.clip((np.arange(40) + phase) // 10, 0, 4)
    positions = control[index]
    positions[0] = entry
    trace["substep_head_position_m"] = positions
    return trace


@pytest.mark.parametrize("phase", [0, 1, 5, 9])
def test_wave1_substep_trace_rejects_every_zero_order_hold_phase(phase):
    """A held control-rate path must be refused at every alignment, not just phase 0."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    with pytest.raises(ValueError, match="not genuinely per-substep"):
        validate_substep_trace(_zero_order_hold_fake(phase))


def test_wave1_substep_trace_rejects_payouts_that_contradict_the_step_reward():
    """A payout matrix pasted beside a real total was not measured with it."""
    from evaluation.analysis.fixed_reset_video_library import validate_substep_trace

    trace = _wave1_trace()
    trace["control_step_reward_total"] = np.full(4, 0.25)
    with pytest.raises(ValueError, match="must sum to"):
        validate_substep_trace(trace)


def test_wave1_trace_only_capture_requests_no_frames():
    """A trace-only screening pass must render nothing while still tracing."""
    full = render_policy.wave1_frame_substep_indices(40, 2)
    screening = render_policy.wave1_frame_substep_indices(40, 2, capture=False)

    assert full.tolist() == list(range(1, 40, 2))
    assert screening.size == 0


# --- Wave-2: recorded execution device (supersedes Wave-1's inferred CPU bridge) ---


def _write_valid_wave2_leaf(tmp_path, *, device_block="default"):
    """Wave-2 leaf: identical to a Wave-1 leaf plus a recorded execution device."""
    from evaluation.analysis.fixed_reset_video_library import (
        SUBSTEP_RENDERER_CONTRACT,
        WAVE1_ARTIFACT_FILENAMES,
        write_metadata,
        write_substep_trajectory_png,
    )

    leaf = tmp_path / "p_seed4"
    leaf.mkdir()
    trace = _wave1_trace()
    np.savez(leaf / "trace.npz", **trace)
    write_substep_trajectory_png(trace, leaf / "trajectory.png")
    frames = np.zeros((20, 720, 960, 3), dtype=np.uint8)
    frames[:, ::2, ::2] = 255
    iio.imwrite(leaf / "policy.mp4", frames, fps=50)
    iio.imwrite(leaf / "montage.png", np.concatenate(list(frames[:6]), axis=1))
    expectations = {
        "arm": "P",
        "training_seed": 4,
        "checkpoint_sha256": "a" * 64,
        "training_revision": "b" * 40,
        "asset_revision": "c" * 40,
        "analysis_revision": "d" * 40,
        "reset_state_digest": FIXED_DIGEST,
    }
    payload = {
        "campaign": "wave2",
        "arm": "P",
        "training_seed": 4,
        "task": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress",
        "checkpoint_sha256": "a" * 64,
        "training_revision": "b" * 40,
        "asset_revision": "c" * 40,
        "analysis_revision": "d" * 40,
        "reset_state_digest": FIXED_DIGEST,
        "renderer_contract": SUBSTEP_RENDERER_CONTRACT,
        "timing": TIMING,
        "rollout": {
            "requested_control_steps": 80,
            "executed_control_steps": 4,
            "substep_count": 40,
            "frame_count": 20,
            "auto_reset_enabled": False,
            "terminal_boundary": {"detected": True, "step": 4, "reason": "terminated"},
        },
        "artifacts": {name: _sha256(leaf / name) for name in WAVE1_ARTIFACT_FILENAMES},
    }
    if device_block == "default":
        device_block = {
            "requested": "cpu",
            "actual_env_device": "cpu",
            "actual_tensor_device": "cpu",
            "platform": "Darwin-25.5.0-arm64",
        }
    if device_block is not None:
        payload["execution_device"] = device_block
    write_metadata(leaf / "metadata.json", payload)
    return leaf, expectations


def test_expected_task_resolves_every_wave2_arm():
    """Wave 2 replicates the same three tasks; the registry must know them."""
    from evaluation.analysis.fixed_reset_video_library import expected_task

    assert expected_task("wave2", "C0").endswith("Guideline-C0")
    assert expected_task("wave2", "G").endswith("Guideline-CGate")
    assert expected_task("wave2", "P").endswith("Guideline-CProgress")


def test_wave2_artifact_validator_reports_the_recorded_execution_device(tmp_path):
    """A Wave-2 leaf records its device directly instead of inferring it."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )

    leaf, expectations = _write_valid_wave2_leaf(tmp_path)
    report = validate_wave1_policy_artifacts(leaf, expectations, campaign="wave2")

    assert report["execution_device"] == {
        "requested": "cpu",
        "actual_env_device": "cpu",
        "actual_tensor_device": "cpu",
        "platform": "Darwin-25.5.0-arm64",
    }


def test_wave2_artifact_validator_rejects_a_missing_device_block(tmp_path):
    """Wave 2 may not fall back to Wave-1's inferred-CPU bridge."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )

    leaf, expectations = _write_valid_wave2_leaf(tmp_path, device_block=None)
    with pytest.raises(ValueError, match="execution device"):
        validate_wave1_policy_artifacts(leaf, expectations, campaign="wave2")


@pytest.mark.parametrize(
    "field", ["requested", "actual_env_device", "actual_tensor_device"]
)
def test_wave2_artifact_validator_requires_cpu_in_every_device_field(tmp_path, field):
    """A CUDA render must be rejected however the accelerator leaked in."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )

    block = {
        "requested": "cpu",
        "actual_env_device": "cpu",
        "actual_tensor_device": "cpu",
        "platform": "Linux-x86_64",
    }
    block[field] = "cuda:0"
    leaf, expectations = _write_valid_wave2_leaf(tmp_path, device_block=block)
    with pytest.raises(ValueError, match="must run on CPU"):
        validate_wave1_policy_artifacts(leaf, expectations, campaign="wave2")


def test_wave2_artifact_validator_rejects_an_empty_platform(tmp_path):
    """Platform is provenance, not decoration; a blank value records nothing."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )

    leaf, expectations = _write_valid_wave2_leaf(
        tmp_path,
        device_block={
            "requested": "cpu",
            "actual_env_device": "cpu",
            "actual_tensor_device": "cpu",
            "platform": "",
        },
    )
    with pytest.raises(ValueError, match="execution device"):
        validate_wave1_policy_artifacts(leaf, expectations, campaign="wave2")


def test_wave1_leaf_still_validates_without_a_device_block(tmp_path):
    """Frozen Wave-1 artifacts keep their documented inferred-CPU bridge."""
    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )

    leaf, expectations = _write_valid_wave1_leaf(tmp_path)
    report = validate_wave1_policy_artifacts(leaf, expectations)

    assert report["execution_device"] is None


def test_renderer_builds_a_device_block_its_own_validator_accepts():
    """The producer and the validator must agree, or the field is decoration."""
    import torch

    from evaluation.analysis.fixed_reset_video_library import (
        validate_execution_device,
    )

    block = render_policy.wave1_execution_device(
        requested="cpu",
        env_device="cpu",
        tensor=torch.zeros(3),
    )

    assert validate_execution_device({"execution_device": block}, Path("leaf")) == block
    assert block["platform"]


def test_renderer_device_block_reports_a_promoted_tensor_device():
    """A tensor that silently moved off CPU must be recorded, not normalised away."""

    class _FakeTensor:
        device = "cuda:0"

    block = render_policy.wave1_execution_device(
        requested="cpu", env_device="cpu", tensor=_FakeTensor()
    )

    assert block["actual_tensor_device"] == "cuda:0"


# --- Treatment-faithful plot semantics (P+V wave) ------------------------------------------------
# A plot that draws gate disks for an arm that was never paid for crossing gates asserts a
# treatment the policy never received. Geometry and title must be DERIVED from the validated
# task/reward configuration, never from a directory name, and the frozen Wave-1/Wave-2 renders
# must stay byte-identical (they are hash-bound inside their manifests).

_C0_TASK = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0"
_G_TASK = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate"
_P_TASK = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress"
_PV_TASK = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel"
_PD4_TASK = f"{_P_TASK}-Delivered4"
_PVD4_TASK = f"{_PV_TASK}-Delivered4"


def test_treatment_geometry_follows_the_reward_the_arm_actually_received():
    from evaluation.analysis.fixed_reset_video_library import treatment_for_task

    assert treatment_for_task(_C0_TASK).geometry == "none"
    assert treatment_for_task(_G_TASK).geometry == "gates"
    assert treatment_for_task(_P_TASK).geometry == "waypoints"
    assert treatment_for_task(_PV_TASK).geometry == "waypoints"


def test_c0_treatment_states_the_state_is_observed_but_not_rewarded():
    from evaluation.analysis.fixed_reset_video_library import treatment_for_task

    assert (
        treatment_for_task(_C0_TASK).guidance
        == "guideline state observed; no guidance reward"
    )


def test_only_the_velocity_arm_reports_velocity_cat_enforcement():
    from evaluation.analysis.fixed_reset_video_library import treatment_for_task

    for task in (_C0_TASK, _G_TASK, _P_TASK):
        velocity = treatment_for_task(task).velocity
        assert "not enforced" in velocity and "max_p" not in velocity
    velocity = treatment_for_task(_PV_TASK).velocity
    assert "max_p 0.5" in velocity and "500 Hz" in velocity


def test_every_guideline_arm_reports_the_impulse_constraint_as_log_only():
    from evaluation.analysis.fixed_reset_video_library import treatment_for_task

    for task in (_C0_TASK, _G_TASK, _P_TASK, _PV_TASK):
        assert treatment_for_task(task).impulse == "impulse log-only"


def test_treatment_lookup_fails_closed_on_an_unregistered_task():
    from evaluation.analysis.fixed_reset_video_library import treatment_for_task

    with pytest.raises(ValueError, match="unregistered"):
        treatment_for_task("Unitree-Z1-Hammer-Some-Future-Arm")


def test_title_carries_identity_treatment_and_outcome():
    from evaluation.analysis.fixed_reset_video_library import (
        compose_trajectory_title,
        treatment_for_task,
    )

    title = compose_trajectory_title(
        campaign="wave3",
        arm="P+V",
        training_seed=4,
        treatment=treatment_for_task(_PV_TASK),
        outcome={"gates": 6, "peak_qvel_rad_s": 3.02, "success": True},
    )
    assert "wave3" in title and "P+V" in title and "seed 4" in title
    assert "progress weight 8.0" in title            # guidance treatment
    assert "max_p 0.5" in title                      # velocity treatment
    assert "impulse log-only" in title               # impulse treatment
    assert "gates 6/6" in title                      # outcome: gates
    assert "3.1415" in title and "3.02" in title     # outcome: peak qvel vs the limit
    assert "success" in title


def test_title_does_not_claim_success_when_the_rollout_hit_the_step_limit():
    from evaluation.analysis.fixed_reset_video_library import (
        compose_trajectory_title,
        treatment_for_task,
    )

    title = compose_trajectory_title(
        campaign="wave3",
        arm="P+V",
        training_seed=6,
        treatment=treatment_for_task(_PV_TASK),
        outcome={"gates": 0, "peak_qvel_rad_s": 4.9, "success": False},
    )
    assert "no success" in title
    assert "gates 0/6" in title


def test_outcome_is_read_off_the_trace_not_asserted_by_the_caller():
    from evaluation.analysis.fixed_reset_video_library import trajectory_outcome

    trace = _wave1_trace()
    outcome = trajectory_outcome(trace, success=True)
    assert outcome["gates"] == int(np.asarray(trace["substep_gate_index"]).max())
    assert outcome["peak_qvel_rad_s"] == pytest.approx(
        float(np.abs(np.asarray(trace["substep_arm_qvel_rad_s"])).max())
    )
    assert outcome["success"] is True


def test_c0_plot_draws_no_reference_line_gate_disks_or_waypoints(tmp_path):
    from evaluation.analysis.fixed_reset_video_library import (
        treatment_for_task,
        write_substep_trajectory_png,
    )

    report = write_substep_trajectory_png(
        _wave1_trace(), tmp_path / "c0.png", treatment=treatment_for_task(_C0_TASK)
    )
    assert report["reference_source"] is None
    assert report["gate_disk_count"] == 0
    assert report["waypoint_marker_count"] == 0
    assert (tmp_path / "c0.png").is_file()


def test_gate_arm_plot_draws_the_reference_line_and_six_gate_disks(tmp_path):
    from evaluation.analysis.fixed_reset_video_library import (
        treatment_for_task,
        write_substep_trajectory_png,
    )

    report = write_substep_trajectory_png(
        _wave1_trace(), tmp_path / "g.png", treatment=treatment_for_task(_G_TASK)
    )
    assert report["reference_source"] == "waypoint_progress_tracker_entry_to_nail"
    assert report["gate_disk_count"] == 6
    assert report["waypoint_marker_count"] == 0


def test_progress_arms_draw_the_reference_line_and_six_ordered_waypoints(tmp_path):
    from evaluation.analysis.fixed_reset_video_library import (
        treatment_for_task,
        write_substep_trajectory_png,
    )

    for name, task in (("p", _P_TASK), ("pv", _PV_TASK)):
        report = write_substep_trajectory_png(
            _wave1_trace(), tmp_path / f"{name}.png", treatment=treatment_for_task(task)
        )
        assert report["reference_source"] == "waypoint_progress_tracker_entry_to_nail"
        assert report["waypoint_marker_count"] == 6
        assert report["gate_disk_count"] == 0


def test_frozen_render_path_reproduces_a_REAL_frozen_figure_byte_for_byte(tmp_path):
    """Re-render a frozen Wave-2 leaf's trace and match the SHA-256 in its own manifest.

    Rendering the same trace twice with the CURRENT code only proves determinism: edit the
    legacy branch (layout rect, caption text, artist order) and both renders shift together
    while every hash-bound frozen manifest silently breaks. This pins an EXTERNAL hash
    recorded before the treatment work existed, so the legacy path cannot drift.
    """
    from evaluation.analysis.fixed_reset_video_library import (
        write_substep_trajectory_png,
    )

    leaf = Path("evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_p_seed4")
    if not (leaf / "trace.npz").is_file():
        pytest.skip("frozen Wave-2 evidence tree not present in this checkout")
    recorded = json.loads((leaf / "metadata.json").read_text())["artifacts"]["trajectory.png"]
    with np.load(leaf / "trace.npz") as loaded:
        trace = {key: loaded[key] for key in loaded.files}

    path = tmp_path / "trajectory.png"
    report = write_substep_trajectory_png(
        trace, path, title=f"wave1 / P / seed 4"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded
    assert report["reference_source"] == "waypoint_progress_tracker_entry_to_nail"
    assert report["gate_disk_count"] == 6
    assert report["waypoint_marker_count"] == 0


def test_the_velocity_arm_has_a_registered_campaign_task_identity():
    from evaluation.analysis.fixed_reset_video_library import expected_task

    assert expected_task("wave3", "P+V") == _PV_TASK


def test_presentation3_resolves_all_three_registered_task_identities():
    from evaluation.analysis.fixed_reset_video_library import expected_task

    assert expected_task("presentation3", "P+V") == _PV_TASK
    assert expected_task("presentation3", "P+D4") == _PD4_TASK
    assert expected_task("presentation3", "P+V+D4") == _PVD4_TASK


def test_treatment_table_matches_the_registered_environment_configuration():
    """Drift guard: the plot labels are only honest if they match the real configs."""
    from mjlab.tasks.registry import load_env_cfg

    from evaluation.analysis.fixed_reset_video_library import treatment_for_task
    import src.tasks.hammer.config.z1  # noqa: F401

    for task, reward_key, geometry in (
        (_C0_TASK, None, "none"),
        (_G_TASK, "r_gate", "gates"),
        (_P_TASK, "r_waypoint_progress", "waypoints"),
        (_PV_TASK, "r_waypoint_progress", "waypoints"),
        (_PD4_TASK, "r_waypoint_progress", "waypoints"),
        (_PVD4_TASK, "r_waypoint_progress", "waypoints"),
    ):
        cfg = load_env_cfg(task)
        treatment = treatment_for_task(task)
        assert treatment.geometry == geometry
        guidance_rewards = {"r_gate", "r_waypoint_progress"} & set(cfg.rewards)
        assert guidance_rewards == ({reward_key} if reward_key else set())
        if reward_key:
            assert f"{cfg.rewards[reward_key].weight:.1f}" in treatment.guidance
        # Velocity label must match whether the hook actually enforces velocity.
        params = cfg.metrics["cat_soft"].params
        enforced = bool(params["use_vel"])
        assert ("max_p" in treatment.velocity) is enforced
        if enforced:
            assert f"max_p {params['max_p']}" in treatment.velocity
            assert params["vel_detection"] == "substep"
        # Impulse label must match the log-only invariant.
        assert params["imp_max_p"] == 0.0
        assert treatment.impulse == "impulse log-only"
        if task.endswith("Delivered4"):
            assert cfg.rewards["delivered_impulse"].weight == 4.0
            assert "delivered weight 4.0" in treatment.headline


# --- render_policy plot wiring -------------------------------------------------------------------

def _render_cfg(campaign, arm, seed=4):
    return render_policy.Cfg(
        checkpoint_file="model.pt",
        campaign=campaign,
        arm=arm,
        training_seed=seed,
        checkpoint_sha256="0" * 64,
        code_revision="a" * 40,
        asset_revision="b" * 40,
    )


def test_frozen_campaigns_keep_the_legacy_substep_plot_call():
    # Wave-1/Wave-2 trajectory.png hashes live in frozen manifests: re-rendering must not
    # change the call, so no treatment is passed and the legacy title is preserved verbatim.
    for campaign, arm in (("wave1", "P"), ("wave2", "P"), ("wave2", "C0")):
        kwargs = render_policy.substep_plot_kwargs(
            _render_cfg(campaign, arm), _wave1_trace(), terminal_reason="terminated"
        )
        assert kwargs == {"title": f"wave1 / {arm} / seed 4"}


def test_new_campaigns_get_treatment_faithful_geometry_and_a_full_title():
    kwargs = render_policy.substep_plot_kwargs(
        _render_cfg("wave3", "P+V"), _wave1_trace(), terminal_reason="terminated"
    )
    assert kwargs["treatment"].geometry == "waypoints"
    title = kwargs["title"]
    assert "wave3" in title and "P+V" in title and "seed 4" in title
    assert "progress weight 8.0" in title
    assert "max_p 0.5" in title
    assert "impulse log-only" in title
    assert "3.1415" in title
    assert "success" in title and "no success" not in title


@pytest.mark.parametrize(
    "arm, expected_phrase",
    (
        ("P+V", "soft velocity-CaT"),
        ("P+D4", "delivered weight 4.0"),
        ("P+V+D4", "delivered weight 4.0"),
    ),
)
def test_presentation3_titles_describe_the_actual_training_treatment(arm, expected_phrase):
    kwargs = render_policy.substep_plot_kwargs(
        _render_cfg("presentation3", arm), _wave1_trace(), terminal_reason="terminated"
    )
    assert kwargs["treatment"].geometry == "waypoints"
    assert expected_phrase in kwargs["title"]
    assert "impulse log-only" in kwargs["title"]


def test_plot_success_comes_from_the_terminal_reason_not_an_assumption():
    for reason, expected in (
        ("terminated", "· success"),
        ("timeout", "· no success"),
        ("step_limit", "· no success"),
    ):
        title = render_policy.substep_plot_kwargs(
            _render_cfg("wave3", "P+V"), _wave1_trace(), terminal_reason=reason
        )["title"]
        assert title.endswith(expected), (reason, title)


def test_plot_wiring_fails_closed_on_an_unregistered_campaign_arm():
    with pytest.raises(ValueError, match="unregistered"):
        render_policy.substep_plot_kwargs(
            _render_cfg("wave3", "G+P"), _wave1_trace(), terminal_reason="terminated"
        )
