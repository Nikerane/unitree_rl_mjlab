from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from evaluation.analysis import fq4x8_manifests as manifests
from evaluation.analysis import fq4x8_task9 as task9
from evaluation.analysis import plot_first_strike_quality_campaign as quality_plot
from tests import test_first_strike_quality_campaign as quality_fixtures
from tests.test_fq4x8_manifests import valid_evaluation_rows, valid_training_rows


def _write_minimal_frozen_inputs(tmp_path: Path, monkeypatch):
    training_rows = valid_training_rows()
    training_bytes = manifests.serialize_training_manifest(training_rows).encode()
    training_sha256 = hashlib.sha256(training_bytes).hexdigest()

    evaluation_rows = valid_evaluation_rows(training_rows)
    for row in evaluation_rows:
        row["accepted_training_manifest_sha256"] = training_sha256
        row["evaluation_attempt"] = manifests.EXPECTED_EVALUATION_ATTEMPT
        row["evaluation_retry_history"] = (
            manifests.EXPECTED_EVALUATION_RETRY_HISTORY
        )
    evaluation_bytes = manifests.serialize_evaluation_manifest(evaluation_rows).encode()

    stream = io.StringIO(newline="")
    integer_fields = (
        "training_seed",
        "num_envs",
        "episodes_per_env_sampled",
        "n_episodes_sampled",
        "reset_rng_seed",
        "observation_rng_seed",
        "action_rng_seed",
    )
    writer = csv.DictWriter(
        stream, fieldnames=("name", "sampled_trace_path", *integer_fields)
    )
    writer.writeheader()
    writer.writerow(
        {
            "name": "fq4x8_f8_seed8",
            "sampled_trace_path": "/trace.npz",
            "training_seed": "8",
            "num_envs": "256",
            "episodes_per_env_sampled": "2",
            "n_episodes_sampled": "512",
            "reset_rng_seed": "2036072919",
            "observation_rng_seed": "2046072933",
            "action_rng_seed": "2056072941",
        }
    )
    summary_bytes = stream.getvalue().encode()

    paths = {
        "accepted_training_manifest": tmp_path / "accepted_training.tsv",
        "accepted_evaluation_manifest": tmp_path / "accepted_evaluations.tsv",
        "summary": tmp_path / "summary.csv",
    }
    for key, payload in (
        ("accepted_training_manifest", training_bytes),
        ("accepted_evaluation_manifest", evaluation_bytes),
        ("summary", summary_bytes),
    ):
        paths[key].write_bytes(payload)

    monkeypatch.setattr(task9, "EXPECTED_TRAINING_MANIFEST_SHA256", training_sha256)
    monkeypatch.setattr(
        task9,
        "EXPECTED_EVALUATION_MANIFEST_SHA256",
        hashlib.sha256(evaluation_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        task9,
        "EXPECTED_SUMMARY_SHA256",
        hashlib.sha256(summary_bytes).hexdigest(),
    )
    return paths, training_rows, evaluation_rows


def test_load_frozen_inputs_validates_and_returns_the_decoded_inputs(
    tmp_path, monkeypatch
) -> None:
    paths, training_rows, evaluation_rows = _write_minimal_frozen_inputs(
        tmp_path, monkeypatch
    )

    loaded = task9.load_frozen_inputs(
        paths["accepted_training_manifest"],
        paths["accepted_evaluation_manifest"],
        paths["summary"],
    )

    assert loaded["training_rows"] == training_rows
    assert loaded["evaluation_rows"] == evaluation_rows
    assert loaded["summary_rows"] == [
        {
            "name": "fq4x8_f8_seed8",
            "sampled_trace_path": "/trace.npz",
            "training_seed": 8,
            "num_envs": 256,
            "episodes_per_env_sampled": 2,
            "n_episodes_sampled": 512,
            "reset_rng_seed": 2036072919,
            "observation_rng_seed": 2046072933,
            "action_rng_seed": 2056072941,
        }
    ]
    assert loaded["input_provenance"] == {
        key: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in paths.items()
    }


def test_run_task9_publishes_the_exact_canonical_bundle(
    tmp_path, monkeypatch
) -> None:
    paths, _, _ = _write_minimal_frozen_inputs(tmp_path, monkeypatch)
    analysis = {"valid": True, "campaign": "fq4x8", "finite": 1.25}

    def render(rows, accepted_evaluations, *, output_dir):
        assert rows == [
            {
                "name": "fq4x8_f8_seed8",
                "sampled_trace_path": "/trace.npz",
                "training_seed": 8,
                "num_envs": 256,
                "episodes_per_env_sampled": 2,
                "n_episodes_sampled": 512,
                "reset_rng_seed": 2036072919,
                "observation_rng_seed": 2046072933,
                "action_rng_seed": 2056072941,
            }
        ]
        assert len(accepted_evaluations) == 32
        output = Path(output_dir)
        paired = output / "paired_seed_effects.png"
        contact = output / "aggregate_nail_plane_contact_map.png"
        paired.write_bytes(b"\x89PNG\r\npaired")
        contact.write_bytes(b"\x89PNG\r\ncontact")
        return {
            "analysis": analysis,
            "artifacts": [str(paired), str(contact)],
        }

    monkeypatch.setattr(task9, "render_quality_figures", render)
    output_dir = tmp_path / "task9_corrected"

    provenance = task9.run_task9(
        paths["accepted_training_manifest"],
        paths["accepted_evaluation_manifest"],
        paths["summary"],
        output_dir,
    )

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "aggregate_nail_plane_contact_map.png",
        "analysis.json",
        "paired_seed_effects.png",
        "task9_artifacts.sha256",
    ]
    expected_json = (
        json.dumps(analysis, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    assert (output_dir / "analysis.json").read_bytes() == expected_json

    lines = (output_dir / "task9_artifacts.sha256").read_text().splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == [
        "aggregate_nail_plane_contact_map.png",
        "analysis.json",
        "paired_seed_effects.png",
    ]
    for line in lines:
        expected_sha256, basename = line.split("  ", 1)
        assert "/" not in basename
        assert expected_sha256 == hashlib.sha256(
            (output_dir / basename).read_bytes()
        ).hexdigest()
    assert provenance["output"]["directory"] == str(output_dir)
    assert provenance["output"]["artifact_sha256"] == {
        line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in lines
    }


@pytest.mark.parametrize(
    "input_name",
    ("accepted_training_manifest", "accepted_evaluation_manifest", "summary"),
)
def test_load_frozen_inputs_rejects_one_byte_drift(
    tmp_path, monkeypatch, input_name
) -> None:
    paths, _, _ = _write_minimal_frozen_inputs(tmp_path, monkeypatch)
    paths[input_name].write_bytes(paths[input_name].read_bytes() + b"x")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        task9.load_frozen_inputs(
            paths["accepted_training_manifest"],
            paths["accepted_evaluation_manifest"],
            paths["summary"],
        )


def test_load_frozen_inputs_rejects_hash_correct_cross_inconsistent_manifests(
    tmp_path, monkeypatch
) -> None:
    paths, _, evaluation_rows = _write_minimal_frozen_inputs(tmp_path, monkeypatch)
    evaluation_rows[0]["checkpoint_sha256"] = "0" * 64
    evaluation_bytes = manifests.serialize_evaluation_manifest(evaluation_rows).encode()
    paths["accepted_evaluation_manifest"].write_bytes(evaluation_bytes)
    monkeypatch.setattr(
        task9,
        "EXPECTED_EVALUATION_MANIFEST_SHA256",
        hashlib.sha256(evaluation_bytes).hexdigest(),
    )

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        task9.load_frozen_inputs(
            paths["accepted_training_manifest"],
            paths["accepted_evaluation_manifest"],
            paths["summary"],
        )


@pytest.mark.parametrize("existing_kind", ("file", "directory", "dangling_symlink"))
def test_run_task9_rejects_every_existing_output_inode_before_loading(
    tmp_path, existing_kind
) -> None:
    output = tmp_path / "occupied"
    if existing_kind == "file":
        output.write_text("keep")
    elif existing_kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
        assert os.path.lexists(output) and not output.exists()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        task9.run_task9("missing-training", "missing-evaluation", "missing-summary", output)


def test_run_task9_requires_an_existing_output_parent_before_loading(tmp_path) -> None:
    output = tmp_path / "missing-parent" / "result"
    with pytest.raises(ValueError, match="output parent must be an existing directory"):
        task9.run_task9("missing-training", "missing-evaluation", "missing-summary", output)


def test_run_task9_rejects_invalid_analysis_and_removes_only_its_stage(
    tmp_path, monkeypatch
) -> None:
    paths, _, _ = _write_minimal_frozen_inputs(tmp_path, monkeypatch)
    unrelated = tmp_path / ".task9_corrected.stage-owner"
    unrelated.mkdir()
    (unrelated / "keep").write_text("owned elsewhere")

    def render(rows, manifest, *, output_dir):
        del rows, manifest
        output = Path(output_dir)
        paired = output / "paired_seed_effects.png"
        contact = output / "aggregate_nail_plane_contact_map.png"
        paired.write_bytes(b"paired")
        contact.write_bytes(b"contact")
        return {
            "analysis": {"valid": False, "invalidation_reasons": ["bad"]},
            "artifacts": [str(paired), str(contact)],
        }

    monkeypatch.setattr(task9, "render_quality_figures", render)
    output = tmp_path / "task9_corrected"
    with pytest.raises(ValueError, match="analysis is invalid"):
        task9.run_task9(
            paths["accepted_training_manifest"],
            paths["accepted_evaluation_manifest"],
            paths["summary"],
            output,
        )

    assert not os.path.lexists(output)
    assert (unrelated / "keep").read_text() == "owned elsewhere"
    assert sorted(path.name for path in tmp_path.glob(".task9_corrected.stage-*")) == [
        unrelated.name
    ]


@pytest.mark.parametrize(
    "registered_names",
    (
        ("paired_seed_effects.png",),
        ("paired_seed_effects.png", "unexpected.png"),
        (
            "paired_seed_effects.png",
            "aggregate_nail_plane_contact_map.png",
            "third.png",
        ),
    ),
)
def test_run_task9_rejects_unexpected_renderer_artifacts_and_cleans_stage(
    tmp_path, monkeypatch, registered_names
) -> None:
    paths, _, _ = _write_minimal_frozen_inputs(tmp_path, monkeypatch)

    def render(rows, manifest, *, output_dir):
        del rows, manifest
        output = Path(output_dir)
        artifacts = []
        for name in registered_names:
            path = output / name
            path.write_bytes(name.encode())
            artifacts.append(str(path))
        return {"analysis": {"valid": True}, "artifacts": artifacts}

    monkeypatch.setattr(task9, "render_quality_figures", render)
    output = tmp_path / "task9_corrected"
    with pytest.raises(ValueError, match="two Task-9 PNGs"):
        task9.run_task9(
            paths["accepted_training_manifest"],
            paths["accepted_evaluation_manifest"],
            paths["summary"],
            output,
        )
    assert not os.path.lexists(output)
    assert list(tmp_path.glob(".task9_corrected.stage-*")) == []


def test_run_task9_cleans_stage_when_json_serialization_fails(
    tmp_path, monkeypatch
) -> None:
    paths, _, _ = _write_minimal_frozen_inputs(tmp_path, monkeypatch)

    def render(rows, manifest, *, output_dir):
        del rows, manifest
        output = Path(output_dir)
        paired = output / "paired_seed_effects.png"
        contact = output / "aggregate_nail_plane_contact_map.png"
        paired.write_bytes(b"paired")
        contact.write_bytes(b"contact")
        return {
            "analysis": {"valid": True, "nonfinite": float("nan")},
            "artifacts": [str(paired), str(contact)],
        }

    monkeypatch.setattr(task9, "render_quality_figures", render)
    output = tmp_path / "task9_corrected"
    with pytest.raises(ValueError, match="Out of range float"):
        task9.run_task9(
            paths["accepted_training_manifest"],
            paths["accepted_evaluation_manifest"],
            paths["summary"],
            output,
        )
    assert not os.path.lexists(output)
    assert list(tmp_path.glob(".task9_corrected.stage-*")) == []


def test_run_task9_cleans_stage_when_atomic_publication_fails(
    tmp_path, monkeypatch
) -> None:
    paths, _, _ = _write_minimal_frozen_inputs(tmp_path, monkeypatch)

    def render(rows, manifest, *, output_dir):
        del rows, manifest
        output = Path(output_dir)
        paired = output / "paired_seed_effects.png"
        contact = output / "aggregate_nail_plane_contact_map.png"
        paired.write_bytes(b"paired")
        contact.write_bytes(b"contact")
        return {
            "analysis": {"valid": True},
            "artifacts": [str(paired), str(contact)],
        }

    def fail_rename(source, destination):
        del source, destination
        raise OSError("publication failed")

    monkeypatch.setattr(task9, "render_quality_figures", render)
    monkeypatch.setattr(task9, "_publish_directory_noreplace", fail_rename)
    output = tmp_path / "task9_corrected"
    with pytest.raises(OSError, match="publication failed"):
        task9.run_task9(
            paths["accepted_training_manifest"],
            paths["accepted_evaluation_manifest"],
            paths["summary"],
            output,
        )
    assert not os.path.lexists(output)
    assert list(tmp_path.glob(".task9_corrected.stage-*")) == []


def test_run_task9_never_replaces_a_destination_created_at_publication(
    tmp_path, monkeypatch
) -> None:
    paths, _, _ = _write_minimal_frozen_inputs(tmp_path, monkeypatch)

    def render(rows, manifest, *, output_dir):
        del rows, manifest
        output = Path(output_dir)
        paired = output / "paired_seed_effects.png"
        contact = output / "aggregate_nail_plane_contact_map.png"
        paired.write_bytes(b"paired")
        contact.write_bytes(b"contact")
        return {
            "analysis": {"valid": True},
            "artifacts": [str(paired), str(contact)],
        }

    real_publish = task9._publish_directory_noreplace
    competing_inode = None

    def collide_then_publish(source, destination):
        nonlocal competing_inode
        destination.mkdir()
        competing_inode = destination.lstat().st_ino
        return real_publish(source, destination)

    monkeypatch.setattr(task9, "render_quality_figures", render)
    monkeypatch.setattr(task9, "_publish_directory_noreplace", collide_then_publish)
    output = tmp_path / "task9_corrected"
    with pytest.raises(FileExistsError):
        task9.run_task9(
            paths["accepted_training_manifest"],
            paths["accepted_evaluation_manifest"],
            paths["summary"],
            output,
        )

    assert output.lstat().st_ino == competing_inode
    assert list(output.iterdir()) == []
    assert list(tmp_path.glob(".task9_corrected.stage-*")) == []


def test_main_requires_exactly_the_four_named_arguments() -> None:
    with pytest.raises(SystemExit) as error:
        task9.main([])
    assert error.value.code == 2

    with pytest.raises(SystemExit) as error:
        task9.main(
            [
                "--accepted-training-manifest",
                "train.tsv",
                "--accepted-evaluation-manifest",
                "eval.tsv",
                "--summary",
                "summary.csv",
                "--output-dir",
                "out",
                "--unexpected",
                "no",
            ]
        )
    assert error.value.code == 2

    with pytest.raises(SystemExit) as error:
        task9.main(
            [
                "--accepted-training-m",
                "train.tsv",
                "--accepted-evaluation-manifest",
                "eval.tsv",
                "--summary",
                "summary.csv",
                "--output-dir",
                "out",
            ]
        )
    assert error.value.code == 2


def test_main_runs_once_and_emits_provenance_as_one_json_object(
    monkeypatch, capsys
) -> None:
    expected = {
        "inputs": {"summary": {"sha256": "a" * 64}},
        "output": {"directory": "out"},
        "runtime": {"python": "test"},
    }
    calls = []

    def run(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(task9, "run_task9", run)
    result = task9.main(
        [
            "--accepted-training-manifest",
            "train.tsv",
            "--accepted-evaluation-manifest",
            "eval.tsv",
            "--summary",
            "summary.csv",
            "--output-dir",
            "out",
        ]
    )

    assert result == 0
    assert calls == [("train.tsv", "eval.tsv", "summary.csv", "out")]
    assert capsys.readouterr().out == json.dumps(expected, sort_keys=True) + "\n"


def _write_full_shape_frozen_inputs(root: Path):
    rows, evaluation_rows = quality_fixtures._campaign_rows(root)
    training_rows = []
    for accepted in evaluation_rows:
        arm = accepted["arm"]
        training_rows.append(
            {
                "campaign": manifests.CAMPAIGN_NAME,
                "disposition": "accepted",
                "arm": arm,
                "short": accepted["short"],
                "task": accepted["task"],
                "training_seed": accepted["training_seed"],
                "checkpoint_path": accepted["checkpoint_path"],
                "checkpoint_sha256": accepted["checkpoint_sha256"],
                "training_attempt": "attempt1",
                "retry_history": "attempt1:accepted",
                "code_revision": accepted["code_revision"],
                "asset_revision": accepted["asset_revision"],
                "campaign_config_sha256": accepted["campaign_config_sha256"],
                "treatment_config_sha256": accepted["treatment_config_sha256"],
                "reader_sha256": manifests.EXPECTED_READER_SHA256[arm],
                "normalizer_sha256": manifests.EXPECTED_NORMALIZER_SHA256[arm],
                "treatment_reward_sha256": (
                    quality_fixtures.EXPECTED_REWARD_HASHES[arm]
                ),
                "fixed_action_signature_sha256": accepted[
                    "fixed_action_signature_sha256"
                ],
                "fixed_impedance_signature_sha256": accepted[
                    "fixed_impedance_signature_sha256"
                ],
                "cap_signature_sha256": accepted["cap_signature_sha256"],
                "clean_state": True,
            }
        )
    training_bytes = manifests.serialize_training_manifest(training_rows).encode()
    training_sha256 = hashlib.sha256(training_bytes).hexdigest()
    for row, accepted in zip(rows, evaluation_rows, strict=True):
        row["accepted_manifest_sha256"] = training_sha256
        accepted["accepted_training_manifest_sha256"] = training_sha256
        quality_fixtures._write_artifact(
            root / f"{row['name']}.npz", row, accepted
        )
    evaluation_bytes = manifests.serialize_evaluation_manifest(evaluation_rows).encode()

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    summary_bytes = stream.getvalue().encode()

    paths = {
        "accepted_training_manifest": root / "accepted_training.tsv",
        "accepted_evaluation_manifest": root / "accepted_evaluations.tsv",
        "summary": root / "summary.csv",
    }
    paths["accepted_training_manifest"].write_bytes(training_bytes)
    paths["accepted_evaluation_manifest"].write_bytes(evaluation_bytes)
    paths["summary"].write_bytes(summary_bytes)
    return paths


def test_real_full_shape_fixture_runs_the_analyzer_once_and_publishes(
    tmp_path, monkeypatch
) -> None:
    paths = _write_full_shape_frozen_inputs(tmp_path)
    monkeypatch.setattr(
        task9,
        "EXPECTED_TRAINING_MANIFEST_SHA256",
        hashlib.sha256(paths["accepted_training_manifest"].read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        task9,
        "EXPECTED_EVALUATION_MANIFEST_SHA256",
        hashlib.sha256(paths["accepted_evaluation_manifest"].read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        task9,
        "EXPECTED_SUMMARY_SHA256",
        hashlib.sha256(paths["summary"].read_bytes()).hexdigest(),
    )
    real_analyzer = quality_plot.analyze_quality_campaign
    calls = 0

    def counted_analyzer(rows, manifest):
        nonlocal calls
        calls += 1
        return real_analyzer(rows, manifest)

    monkeypatch.setattr(quality_plot, "analyze_quality_campaign", counted_analyzer)
    output = tmp_path / "task9_corrected"
    task9.run_task9(
        paths["accepted_training_manifest"],
        paths["accepted_evaluation_manifest"],
        paths["summary"],
        output,
    )

    analysis = json.loads((output / "analysis.json").read_text())
    assert calls == 1
    assert analysis["valid"] is True
    assert len(analysis["seed_aggregates"]) == 32
    assert sum(row["n_episodes_sampled"] for row in analysis["seed_aggregates"]) == 16_384
    assert sorted(path.name for path in output.iterdir()) == [
        "aggregate_nail_plane_contact_map.png",
        "analysis.json",
        "paired_seed_effects.png",
        "task9_artifacts.sha256",
    ]


def test_run_task9_serializes_numpy_analysis_values_and_reports_resolved_modules(
    tmp_path, monkeypatch
) -> None:
    paths, _, _ = _write_minimal_frozen_inputs(tmp_path, monkeypatch)

    def render(rows, manifest, *, output_dir):
        del rows, manifest
        output = Path(output_dir)
        paired = output / "paired_seed_effects.png"
        contact = output / "aggregate_nail_plane_contact_map.png"
        paired.write_bytes(b"paired")
        contact.write_bytes(b"contact")
        return {
            "analysis": {
                "valid": True,
                "scalar": np.int64(3),
                "vector": np.asarray([1.25, 2.5]),
            },
            "artifacts": [str(paired), str(contact)],
        }

    monkeypatch.setattr(task9, "render_quality_figures", render)
    output = tmp_path / "result"
    provenance = task9.run_task9(
        paths["accepted_training_manifest"],
        paths["accepted_evaluation_manifest"],
        paths["summary"],
        output,
    )

    assert json.loads((output / "analysis.json").read_text()) == {
        "valid": True,
        "scalar": 3,
        "vector": [1.25, 2.5],
    }
    assert provenance["output"]["directory"] == str(output.resolve())
    assert set(provenance["runtime"]) >= {"analysis_module", "renderer_module"}
    assert provenance["runtime"]["analysis_module"] == str(
        Path(task9.quality_analysis.__file__).resolve()
    )
    assert provenance["runtime"]["renderer_module"] == str(
        Path(task9.quality_plot.__file__).resolve()
    )
    assert all(
        record["path"] == str(paths[name].resolve())
        for name, record in provenance["inputs"].items()
    )
