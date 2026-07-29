"""Contracts for one fixed-reset rollout artifact per registered policy."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import imageio.v3 as iio
import numpy as np
import pytest

from evaluation.analysis import render_fixed_reset_video_library as library_driver
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
    source = tmp_path / "source"
    asset = tmp_path / "asset"
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


def test_render_provenance_derives_revisions_and_ignores_unimported_dirty_files(tmp_path):
    """Caller labels and unrelated docs cannot define or block renderer provenance."""
    source, asset, source_revision, asset_revision = _provenance_repositories(tmp_path)
    (source / "docs/results/ignored.md").write_text("dirty result\n")
    (asset / "notes/ignored.md").write_text("dirty note\n")

    provenance = library_driver.capture_render_provenance(source, asset)

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
    tmp_path, repository, relative, message
):
    """Any tracked or untracked mutation that can affect rendering must fail closed."""
    source, asset, _, _ = _provenance_repositories(tmp_path)
    root = source if repository == "source" else asset
    (root / relative).write_text("dirty\n")

    with pytest.raises(ValueError, match=message):
        library_driver.capture_render_provenance(source, asset)


def test_render_provenance_is_rechecked_after_batch(tmp_path):
    """A source mutation during rendering must invalidate the recorded revision."""
    source, asset, _, _ = _provenance_repositories(tmp_path)
    before = library_driver.capture_render_provenance(source, asset)
    (source / "evaluation/analysis/contract.py").write_text("changed during batch\n")

    with pytest.raises(ValueError, match="source scope"):
        library_driver.require_unchanged_render_provenance(before, source, asset)


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
