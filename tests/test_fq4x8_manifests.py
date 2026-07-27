"""Tests for the fail-closed fq4x8 accepted-manifest tooling.

Frozen constants (arms, seeds, task ids, reward/action/impedance hashes) are
imported from ``first_strike_quality_campaign`` / ``first_strike_campaign``
rather than restated, so these tests exercise the same source of truth the
implementation binds against.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
import shutil
import sys
from pathlib import Path

import pytest

import evaluation.analysis.first_strike_campaign as legacy
import evaluation.analysis.first_strike_quality_campaign as quality
import evaluation.analysis.fq4x8_manifests as manifests


def _hex(label: str, length: int) -> str:
    if length == 64:
        return hashlib.sha256(label.encode()).hexdigest()
    if length == 40:
        return hashlib.sha1(label.encode()).hexdigest()
    raise ValueError(f"unsupported hex length: {length}")


SHARED_CODE_REVISION = _hex("code-revision", 40)
SHARED_ASSET_REVISION = _hex("asset-revision", 40)
SHARED_CAMPAIGN_CONFIG_SHA256 = _hex("campaign-config", 64)
TRAINING_MANIFEST_SHA256 = _hex("training-manifest-bytes", 64)


def _training_row(arm: str, seed: int) -> dict:
    return {
        "campaign": manifests.CAMPAIGN_NAME,
        "disposition": "accepted",
        "arm": arm,
        "short": manifests.SHORT[arm],
        "task": manifests.TASKS[arm],
        "training_seed": seed,
        "checkpoint_path": f"/checkpoints/{arm}/seed{seed}/model_499.pt",
        "checkpoint_sha256": _hex(f"checkpoint:{arm}:{seed}", 64),
        "training_attempt": "attempt1",
        "retry_history": "attempt0:infra_failure;attempt1:accepted",
        "code_revision": SHARED_CODE_REVISION,
        "asset_revision": SHARED_ASSET_REVISION,
        "campaign_config_sha256": SHARED_CAMPAIGN_CONFIG_SHA256,
        "treatment_config_sha256": manifests.EXPECTED_TREATMENT_CONFIG_SHA256[arm],
        "reader_sha256": manifests.EXPECTED_READER_SHA256[arm],
        "normalizer_sha256": manifests.EXPECTED_NORMALIZER_SHA256[arm],
        "treatment_reward_sha256": quality.EXPECTED_TREATMENT_REWARD_SHA256[arm],
        "fixed_action_signature_sha256": legacy.EXPECTED_FIXED_ACTION_SIGNATURE_SHA256,
        "fixed_impedance_signature_sha256": legacy.EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256,
        "cap_signature_sha256": manifests.EXPECTED_CAP_SIGNATURE_SHA256,
        "clean_state": True,
    }


def valid_training_rows() -> list[dict]:
    return [
        _training_row(arm, seed)
        for arm in manifests.LABELS
        for seed in sorted(manifests.SEEDS)
    ]


def _evaluation_row(training_row: dict) -> dict:
    arm = training_row["arm"]
    seed = training_row["training_seed"]
    observation_sha = _hex(f"observation:{arm}", 64)
    row = {
        "campaign": manifests.CAMPAIGN_NAME,
        "disposition": "accepted",
        "arm": arm,
        "short": training_row["short"],
        "task": training_row["task"],
        "training_seed": seed,
        "checkpoint_path": training_row["checkpoint_path"],
        "checkpoint_sha256": training_row["checkpoint_sha256"],
        "code_revision": training_row["code_revision"],
        "asset_revision": training_row["asset_revision"],
        "campaign_config_sha256": training_row["campaign_config_sha256"],
        "treatment_config_sha256": training_row["treatment_config_sha256"],
        "fixed_action_signature_sha256": training_row["fixed_action_signature_sha256"],
        "fixed_impedance_signature_sha256": training_row["fixed_impedance_signature_sha256"],
        "cap_signature_sha256": training_row["cap_signature_sha256"],
        "clean_state": True,
        "accepted_training_manifest_sha256": TRAINING_MANIFEST_SHA256,
        "evaluation_attempt": "eval-attempt1",
        "evaluation_retry_history": "eval-attempt1:accepted",
        "training_config_sha256": _hex(f"training-config:{arm}:{seed}", 64),
        "evaluation_config_sha256": _hex(f"evaluation-config:{arm}:{seed}", 64),
        "training_policy_observation_sha256": observation_sha,
        "evaluation_policy_observation_sha256": observation_sha,
        "training_treatment_reward_sha256": training_row["treatment_reward_sha256"],
        "evaluation_treatment_reward_sha256": training_row["treatment_reward_sha256"],
        "reset_rng_seed": legacy.CAMPAIGN_EVALUATOR_RNG["reset"],
        "observation_rng_seed": legacy.CAMPAIGN_EVALUATOR_RNG["observation"],
        "action_rng_seed": legacy.CAMPAIGN_EVALUATOR_RNG["action"],
        "sampled_trace_digest": _hex(f"trace-digest:{arm}:{seed}", 64),
        "sampled_trace_artifact_sha256": _hex(f"trace-artifact:{arm}:{seed}", 64),
        "num_envs": 256,
        "episodes_per_env_sampled": 2,
        "n_episodes_sampled": 512,
    }
    for sentinel in manifests.SENTINELS:
        row[sentinel] = 0
    return row


def valid_evaluation_rows(training_rows: list[dict]) -> list[dict]:
    return [_evaluation_row(row) for row in training_rows]


def _mutate(rows: list[dict], index: int, **overrides) -> list[dict]:
    rows = copy.deepcopy(rows)
    rows[index] = {**rows[index], **overrides}
    return rows


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_fully_valid_32_row_pair_validates_cleanly():
    training_rows = valid_training_rows()
    evaluation_rows = valid_evaluation_rows(training_rows)
    # Must not raise.
    manifests.validate_training_manifest(training_rows)
    manifests.validate_evaluation_manifest(
        evaluation_rows, training_rows, TRAINING_MANIFEST_SHA256
    )


def test_training_manifest_does_not_contain_evaluation_attempt_fields():
    forbidden = {
        "evaluation_attempt",
        "evaluation_retry_history",
        "reset_rng_seed",
        "observation_rng_seed",
        "action_rng_seed",
        "n_episodes_sampled",
        "accepted_training_manifest_sha256",
    }
    assert forbidden.isdisjoint(manifests.TRAINING_FIELDS)


# ---------------------------------------------------------------------------
# Round-trip and determinism
# ---------------------------------------------------------------------------


def test_training_manifest_round_trip():
    rows = valid_training_rows()
    text = manifests.serialize_training_manifest(rows)
    parsed = manifests.parse_training_manifest(text)
    expected = sorted(rows, key=lambda row: (manifests.LABELS.index(row["arm"]), row["training_seed"]))
    assert parsed == expected


def test_evaluation_manifest_round_trip():
    training_rows = valid_training_rows()
    rows = valid_evaluation_rows(training_rows)
    text = manifests.serialize_evaluation_manifest(rows)
    parsed = manifests.parse_evaluation_manifest(text)
    expected = sorted(rows, key=lambda row: (manifests.LABELS.index(row["arm"]), row["training_seed"]))
    assert parsed == expected


def test_training_manifest_serialization_is_deterministic_under_shuffle():
    rows = valid_training_rows()
    shuffled = list(rows)
    random.Random(20260726).shuffle(shuffled)
    assert manifests.serialize_training_manifest(rows) == manifests.serialize_training_manifest(
        shuffled
    )


def test_evaluation_manifest_serialization_is_deterministic_under_shuffle():
    training_rows = valid_training_rows()
    rows = valid_evaluation_rows(training_rows)
    shuffled = list(rows)
    random.Random(20260727).shuffle(shuffled)
    assert manifests.serialize_evaluation_manifest(
        rows
    ) == manifests.serialize_evaluation_manifest(shuffled)


def test_training_manifest_serialization_is_sorted_by_arm_then_seed():
    rows = valid_training_rows()
    text = manifests.serialize_training_manifest(rows)
    body_lines = text.strip("\n").split("\n")[1:]
    arms = [line.split("\t")[manifests.TRAINING_FIELDS.index("arm")] for line in body_lines]
    seeds = [
        int(line.split("\t")[manifests.TRAINING_FIELDS.index("training_seed")])
        for line in body_lines
    ]
    expected_arms = [arm for arm in manifests.LABELS for _ in sorted(manifests.SEEDS)]
    assert arms == expected_arms
    assert seeds == sorted(manifests.SEEDS) * len(manifests.LABELS)


# ---------------------------------------------------------------------------
# Training manifest rejections
# ---------------------------------------------------------------------------


def test_training_manifest_rejects_31_rows():
    rows = valid_training_rows()[:-1]
    with pytest.raises(ValueError, match="32 rows"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_33_rows():
    rows = valid_training_rows()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="32 rows"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_uneven_per_arm_count():
    # Drop F0/seed8 and duplicate F8/seed8: F8 now holds 9 rows (one
    # duplicated), F0 holds 7. Total stays 32. Because SEEDS is a closed
    # 8-element space (8..15), any 9-in-one-arm distribution over the frozen
    # 32-slot grid necessarily involves a repeated seed -- the duplicate
    # check and the per-arm-count check are the same underlying violation
    # here, and either is a legitimate rejection of this malformed shape.
    rows = [row for row in valid_training_rows() if not (row["arm"] == "F0" and row["training_seed"] == 8)]
    rows.append(copy.deepcopy(_training_row("F8", 8)))
    assert len(rows) == 32
    with pytest.raises(ValueError):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_missing_seed():
    rows = [
        row for row in valid_training_rows() if not (row["arm"] == "F8" and row["training_seed"] == 11)
    ]
    rows.append(copy.deepcopy(_training_row("D0", 8)))
    assert len(rows) == 32
    with pytest.raises(ValueError):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_duplicate_arm_seed():
    rows = valid_training_rows()
    rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_seed_outside_range():
    rows = _mutate(valid_training_rows(), 0, training_seed=16)
    with pytest.raises(ValueError, match="8..15"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_unknown_arm():
    rows = _mutate(valid_training_rows(), 0, arm="F9")
    with pytest.raises(ValueError, match="unknown arm"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_misspelled_short():
    rows = _mutate(valid_training_rows(), 0, short="f9")
    with pytest.raises(ValueError, match="short"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_short_task_mismatch():
    # f8 paired with the D0 task.
    rows = _mutate(valid_training_rows(), 0, task=manifests.TASKS["D0"])
    with pytest.raises(ValueError, match="task"):
        manifests.validate_training_manifest(rows)


@pytest.mark.parametrize("field", ["code_revision", "asset_revision"])
def test_training_manifest_rejects_dirty_revision(field):
    rows = valid_training_rows()
    dirty = rows[0][field] + "-dirty"
    for row in rows:
        row[field] = dirty
    with pytest.raises(ValueError, match="40-hex"):
        manifests.validate_training_manifest(rows)


@pytest.mark.parametrize("field", ["code_revision", "asset_revision"])
def test_training_manifest_rejects_short_revision(field):
    rows = valid_training_rows()
    short_value = rows[0][field][:-1]
    for row in rows:
        row[field] = short_value
    with pytest.raises(ValueError, match="40-hex"):
        manifests.validate_training_manifest(rows)


@pytest.mark.parametrize("field", ["code_revision", "asset_revision"])
def test_training_manifest_rejects_non_hex_revision(field):
    rows = valid_training_rows()
    non_hex = "g" * 40
    for row in rows:
        row[field] = non_hex
    with pytest.raises(ValueError, match="40-hex"):
        manifests.validate_training_manifest(rows)


@pytest.mark.parametrize("field", ["code_revision", "asset_revision"])
def test_training_manifest_rejects_missing_revision(field):
    rows = valid_training_rows()
    for row in rows:
        row[field] = ""
    with pytest.raises(ValueError, match="40-hex"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_treatment_reward_hash_mismatch():
    rows = _mutate(valid_training_rows(), 0, treatment_reward_sha256=_hex("wrong-reward", 64))
    with pytest.raises(ValueError, match="treatment_reward_sha256"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_reader_hash_mismatch():
    rows = _mutate(valid_training_rows(), 0, reader_sha256=_hex("wrong-reader", 64))
    with pytest.raises(ValueError, match="reader_sha256"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_normalizer_hash_mismatch():
    rows = _mutate(valid_training_rows(), 0, normalizer_sha256=_hex("wrong-normalizer", 64))
    with pytest.raises(ValueError, match="normalizer_sha256"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_missing_checkpoint_sha256():
    rows = _mutate(valid_training_rows(), 0, checkpoint_sha256="")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_malformed_checkpoint_sha256():
    rows = _mutate(valid_training_rows(), 0, checkpoint_sha256="zz" * 32)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_dirty_clean_state():
    rows = _mutate(valid_training_rows(), 0, clean_state=False)
    with pytest.raises(ValueError, match="clean_state"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_missing_field():
    rows = valid_training_rows()
    del rows[0]["checkpoint_sha256"]
    with pytest.raises(ValueError, match="missing field"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_retry_history_not_ending_in_accepted_attempt():
    rows = _mutate(valid_training_rows(), 0, retry_history="attempt0:infra_failure")
    with pytest.raises(ValueError, match="retry history"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_wrong_campaign():
    rows = _mutate(valid_training_rows(), 0, campaign="fsr4x8")
    with pytest.raises(ValueError, match="campaign"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_non_accepted_disposition():
    rows = _mutate(valid_training_rows(), 0, disposition="pending")
    with pytest.raises(ValueError, match="disposition"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_missing_training_attempt():
    rows = _mutate(valid_training_rows(), 0, training_attempt="", retry_history=":accepted")
    with pytest.raises(ValueError, match="training_attempt"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_bogus_training_attempt_not_matching_retry_history():
    # A "bogus" attempt label alone is not what makes this invalid -- what closes the hole is
    # that it must match the final, accepted entry of its own retry history.
    rows = _mutate(valid_training_rows(), 0, training_attempt="bogus")
    with pytest.raises(ValueError, match="retry history"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_treatment_config_hash_mismatch():
    rows = _mutate(valid_training_rows(), 0, treatment_config_sha256=_hex("wrong-treatment-config", 64))
    with pytest.raises(ValueError, match="treatment_config_sha256"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_fixed_action_signature_mismatch():
    rows = _mutate(
        valid_training_rows(), 0, fixed_action_signature_sha256=_hex("wrong-action-sig", 64)
    )
    with pytest.raises(ValueError, match="fixed_action_signature_sha256"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_fixed_impedance_signature_mismatch():
    rows = _mutate(
        valid_training_rows(), 0, fixed_impedance_signature_sha256=_hex("wrong-impedance-sig", 64)
    )
    with pytest.raises(ValueError, match="fixed_impedance_signature_sha256"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_cap_signature_mismatch():
    rows = _mutate(valid_training_rows(), 0, cap_signature_sha256=_hex("wrong-cap-sig", 64))
    with pytest.raises(ValueError, match="cap_signature_sha256"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_non_model_499_checkpoint_basename():
    rows = _mutate(valid_training_rows(), 0, checkpoint_path="/checkpoints/F8/seed8/model_250.pt")
    with pytest.raises(ValueError, match="model_499.pt"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_duplicate_checkpoint_path():
    rows = valid_training_rows()
    rows[1] = {**rows[1], "checkpoint_path": rows[0]["checkpoint_path"]}
    with pytest.raises(ValueError, match="checkpoint_path is duplicated"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_duplicate_checkpoint_sha256():
    rows = valid_training_rows()
    rows[1] = {**rows[1], "checkpoint_sha256": rows[0]["checkpoint_sha256"]}
    with pytest.raises(ValueError, match="checkpoint_sha256 is duplicated"):
        manifests.validate_training_manifest(rows)


def test_training_manifest_rejects_eight_copies_of_seed_8_per_arm():
    """Reproduces the exact independent-review attack: eight duplicate seed-8
    rows per arm (32 rows total, right count, wrong content) must still fail
    closed -- this was the HIGH-1 hole (the old hand-rolled awk validator let
    this reach the CUDA guard)."""
    rows = []
    for arm in manifests.LABELS:
        for _ in manifests.SEEDS:
            rows.append(_training_row(arm, 8))
    assert len(rows) == 32
    with pytest.raises(ValueError):
        manifests.validate_training_manifest(rows)


# ---------------------------------------------------------------------------
# CLI (validate-accepted-manifest): exercises the real subprocess entry point
# the Slurm launcher invokes, end to end.
# ---------------------------------------------------------------------------


def _run_cli(manifest_path, *, expected_code_revision, expected_asset_revision):
    import subprocess
    import sys as _sys

    return subprocess.run(
        [
            _sys.executable,
            "-m",
            "evaluation.analysis.fq4x8_manifests",
            "validate-accepted-manifest",
            "--manifest",
            str(manifest_path),
            "--expected-code-revision",
            expected_code_revision,
            "--expected-asset-revision",
            expected_asset_revision,
        ],
        capture_output=True,
        text=True,
    )


def test_cli_accepts_valid_32_row_manifest_and_emits_32_rows(tmp_path):
    rows = valid_training_rows()
    manifest_path = tmp_path / "accepted_training_checkpoints.tsv"
    manifest_path.write_text(manifests.serialize_training_manifest(rows))

    result = _run_cli(
        manifest_path,
        expected_code_revision=SHARED_CODE_REVISION,
        expected_asset_revision=SHARED_ASSET_REVISION,
    )

    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line]
    assert len(lines) == 32
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for line in lines:
        fields = line.split("\t")
        assert len(fields) == 7
        assert fields[0] == manifest_sha


def test_cli_rejects_manifest_with_duplicate_seeds(tmp_path):
    rows = []
    for arm in manifests.LABELS:
        for _ in manifests.SEEDS:
            rows.append(_training_row(arm, 8))
    manifest_path = tmp_path / "accepted_training_checkpoints.tsv"
    manifest_path.write_text(manifests.serialize_training_manifest(rows))

    result = _run_cli(
        manifest_path,
        expected_code_revision=SHARED_CODE_REVISION,
        expected_asset_revision=SHARED_ASSET_REVISION,
    )

    assert result.returncode == 2
    assert "MANIFEST_FAIL" in result.stderr
    assert result.stdout == ""


def test_cli_rejects_code_revision_not_matching_expected(tmp_path):
    rows = valid_training_rows()
    manifest_path = tmp_path / "accepted_training_checkpoints.tsv"
    manifest_path.write_text(manifests.serialize_training_manifest(rows))

    result = _run_cli(
        manifest_path,
        expected_code_revision=_hex("different-code-revision", 40),
        expected_asset_revision=SHARED_ASSET_REVISION,
    )

    assert result.returncode == 2
    assert "code_revision does not equal expected code revision" in result.stderr


# ---------------------------------------------------------------------------
# Evaluation manifest rejections
# ---------------------------------------------------------------------------


def _valid_pair() -> tuple[list[dict], list[dict]]:
    training_rows = valid_training_rows()
    return training_rows, valid_evaluation_rows(training_rows)


def test_evaluation_manifest_rejects_31_rows():
    training_rows, rows = _valid_pair()
    with pytest.raises(ValueError, match="32 rows"):
        manifests.validate_evaluation_manifest(rows[:-1], training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_33_rows():
    training_rows, rows = _valid_pair()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="32 rows"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_duplicate_arm_seed():
    training_rows, rows = _valid_pair()
    rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_seed_outside_range():
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, training_seed=7)
    with pytest.raises(ValueError, match="8..15"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_unknown_task():
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, task="Unitree-Z1-Hammer-Bogus")
    with pytest.raises(ValueError, match="task"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_checkpoint_not_in_training_manifest():
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, checkpoint_sha256=_hex("unlisted-checkpoint", 64))
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_training_manifest_sha_mismatch_in_row():
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, accepted_training_manifest_sha256=_hex("different-manifest", 64))
    with pytest.raises(ValueError, match="accepted_training_manifest_sha256"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_training_manifest_sha_mismatch_in_supplied_argument():
    training_rows, rows = _valid_pair()
    with pytest.raises(ValueError, match="accepted_training_manifest_sha256"):
        manifests.validate_evaluation_manifest(rows, training_rows, _hex("other-manifest", 64))


def test_evaluation_manifest_rejects_episode_count_not_512():
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, n_episodes_sampled=511)
    with pytest.raises(ValueError, match="n_episodes_sampled"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


@pytest.mark.parametrize("sentinel", quality.SENTINELS)
def test_evaluation_manifest_rejects_any_nonzero_sentinel(sentinel):
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, **{sentinel: 1})
    with pytest.raises(ValueError, match=sentinel):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


@pytest.mark.parametrize(
    "field", ["reset_rng_seed", "observation_rng_seed", "action_rng_seed"]
)
def test_evaluation_manifest_rejects_wrong_rng_stream_value(field):
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, **{field: rows[0][field] + 1})
    with pytest.raises(ValueError, match=field):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_treatment_reward_hash_not_matching_frozen_expectation():
    training_rows, rows = _valid_pair()
    wrong = _hex("wrong-treatment-reward", 64)
    rows = _mutate(
        rows,
        0,
        training_treatment_reward_sha256=wrong,
        evaluation_treatment_reward_sha256=wrong,
    )
    with pytest.raises(ValueError, match="treatment-reward|treatment_reward"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_training_evaluation_observation_hash_mismatch():
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, evaluation_policy_observation_sha256=_hex("other-observation", 64))
    with pytest.raises(ValueError, match="observation"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


@pytest.mark.parametrize("key", quality.STRICT_CONFIG_IDENTITY_KEYS)
def test_evaluation_manifest_rejects_malformed_strict_identity_hash(key):
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, **{key: "not-a-hash"})
    with pytest.raises(ValueError, match=key):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_wrong_num_envs():
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, num_envs=128)
    with pytest.raises(ValueError, match="num_envs"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_bad_training_row_when_cross_checking():
    training_rows, rows = _valid_pair()
    training_rows = _mutate(training_rows, 0, clean_state=False)
    with pytest.raises(ValueError, match="clean_state"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


def test_evaluation_manifest_rejects_evaluation_retry_history_not_ending_in_accepted_attempt():
    training_rows, rows = _valid_pair()
    rows = _mutate(rows, 0, evaluation_retry_history="eval-attempt0:infra_failure")
    with pytest.raises(ValueError, match="retry history"):
        manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)


# ---------------------------------------------------------------------------
# build-training-manifest: scans real run directories and builds TRAINING_FIELDS
# rows.  The config-identity derivation (which in production loads the live
# registered env cfg via mjlab/torch) is injected as a stub here so these
# tests need neither a GPU nor mjlab -- only ``test_cli_build_training_*``
# monkeypatches the real ``default_config_identity`` entry point, and only to
# replace it with the same stub.
# ---------------------------------------------------------------------------


def _run_dir_name(arm: str, seed: int, *, timestamp: str = "2026-07-27_20-36-22") -> str:
    return f"{timestamp}_fq4x8_{manifests.SHORT[arm]}_seed{seed}"


def _write_checkpoint(run_dir: Path, *, content: bytes) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "model_499.pt").write_bytes(content)


def _populate_valid_run_dirs(root: Path) -> None:
    for arm in manifests.LABELS:
        for seed in sorted(manifests.SEEDS):
            run_dir = root / _run_dir_name(arm, seed)
            _write_checkpoint(run_dir, content=f"checkpoint:{arm}:{seed}".encode())


def _stub_config_identity(arm: str, task: str) -> tuple[str, str]:
    assert task == manifests.TASKS[arm], (arm, task)
    return SHARED_CAMPAIGN_CONFIG_SHA256, manifests.EXPECTED_TREATMENT_CONFIG_SHA256[arm]


def _build_kwargs(**overrides) -> dict:
    kwargs = dict(
        code_revision=SHARED_CODE_REVISION,
        asset_revision=SHARED_ASSET_REVISION,
        config_identity=_stub_config_identity,
    )
    kwargs.update(overrides)
    return kwargs


def test_build_training_manifest_happy_path_returns_valid_32_row_manifest(tmp_path):
    _populate_valid_run_dirs(tmp_path)
    rows = manifests.build_training_manifest(
        tmp_path, attempt="attempt1", **_build_kwargs()
    )
    manifests.validate_training_manifest(rows)  # must not raise
    assert len(rows) == 32
    identities = {(row["arm"], row["training_seed"]) for row in rows}
    expected = {(arm, seed) for arm in manifests.LABELS for seed in manifests.SEEDS}
    assert identities == expected
    for row in rows:
        checkpoint_path = Path(row["checkpoint_path"])
        assert checkpoint_path.is_absolute()
        assert checkpoint_path.name == "model_499.pt"
        assert row["checkpoint_sha256"] == hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
        assert row["training_attempt"] == "attempt1"
        assert row["retry_history"] == "attempt1:accepted"
        assert row["clean_state"] is True
        assert row["disposition"] == "accepted"
        assert row["campaign"] == manifests.CAMPAIGN_NAME
        assert row["campaign_config_sha256"] == SHARED_CAMPAIGN_CONFIG_SHA256


def test_build_training_manifest_is_deterministic_regardless_of_directory_order(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    pairs = [(arm, seed) for arm in manifests.LABELS for seed in sorted(manifests.SEEDS)]
    for arm, seed in pairs:
        _write_checkpoint(
            root_a / _run_dir_name(arm, seed), content=f"checkpoint:{arm}:{seed}".encode()
        )
    for arm, seed in reversed(pairs):
        _write_checkpoint(
            root_b / _run_dir_name(arm, seed), content=f"checkpoint:{arm}:{seed}".encode()
        )

    rows_a = manifests.build_training_manifest(root_a, **_build_kwargs())
    rows_b = manifests.build_training_manifest(root_b, **_build_kwargs())
    text_a = manifests.serialize_training_manifest(rows_a).replace(str(root_a), "<ROOT>")
    text_b = manifests.serialize_training_manifest(rows_b).replace(str(root_b), "<ROOT>")
    assert text_a == text_b


def test_build_training_manifest_rejects_missing_run_dir(tmp_path):
    _populate_valid_run_dirs(tmp_path)
    shutil.rmtree(tmp_path / _run_dir_name("FQ-min", 15))
    with pytest.raises(ValueError, match="missing"):
        manifests.build_training_manifest(tmp_path, **_build_kwargs())


def test_build_training_manifest_rejects_duplicate_identity(tmp_path):
    _populate_valid_run_dirs(tmp_path)
    _write_checkpoint(
        tmp_path / _run_dir_name("F8", 8, timestamp="2026-07-27_21-00-00"),
        content=b"checkpoint:F8:8:dup",
    )
    with pytest.raises(ValueError, match="duplicate"):
        manifests.build_training_manifest(tmp_path, **_build_kwargs())


def test_build_training_manifest_rejects_seed_outside_range(tmp_path):
    _populate_valid_run_dirs(tmp_path)
    _write_checkpoint(
        tmp_path / _run_dir_name("F8", 16), content=b"checkpoint:F8:16"
    )
    with pytest.raises(ValueError, match="8..15"):
        manifests.build_training_manifest(tmp_path, **_build_kwargs())


def test_build_training_manifest_rejects_unknown_short(tmp_path):
    _populate_valid_run_dirs(tmp_path)
    bad_dir = tmp_path / "2026-07-27_20-36-22_fq4x8_zz_seed8"
    _write_checkpoint(bad_dir, content=b"unknown-short")
    with pytest.raises(ValueError, match="short"):
        manifests.build_training_manifest(tmp_path, **_build_kwargs())


def test_build_training_manifest_rejects_unparseable_run_dir_name(tmp_path):
    _populate_valid_run_dirs(tmp_path)
    (tmp_path / "not_a_run_dir").mkdir()
    with pytest.raises(ValueError, match="does not parse"):
        manifests.build_training_manifest(tmp_path, **_build_kwargs())


def test_build_training_manifest_rejects_missing_checkpoint_file(tmp_path):
    _populate_valid_run_dirs(tmp_path)
    (tmp_path / _run_dir_name("F0", 9) / "model_499.pt").unlink()
    with pytest.raises(ValueError, match="model_499.pt"):
        manifests.build_training_manifest(tmp_path, **_build_kwargs())


def test_build_training_manifest_rejects_zero_length_checkpoint(tmp_path):
    _populate_valid_run_dirs(tmp_path)
    (tmp_path / _run_dir_name("D0", 12) / "model_499.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="zero-length"):
        manifests.build_training_manifest(tmp_path, **_build_kwargs())


def test_build_training_manifest_rejects_unreadable_checkpoint(tmp_path):
    _populate_valid_run_dirs(tmp_path)
    bad = tmp_path / _run_dir_name("F8", 8) / "model_499.pt"
    os.chmod(bad, 0o000)
    try:
        with pytest.raises(ValueError, match="unreadable"):
            manifests.build_training_manifest(tmp_path, **_build_kwargs())
    finally:
        os.chmod(bad, 0o644)


@pytest.mark.parametrize("field", ["code_revision", "asset_revision"])
def test_build_training_manifest_rejects_dirty_revision_arg(tmp_path, field):
    _populate_valid_run_dirs(tmp_path)
    with pytest.raises(ValueError, match="40-hex"):
        manifests.build_training_manifest(tmp_path, **_build_kwargs(**{field: "not-40-hex"}))


def test_build_training_manifest_rejects_config_identity_not_matching_frozen_expectation(tmp_path):
    _populate_valid_run_dirs(tmp_path)

    def bad_identity(arm, task):
        campaign_hash, treatment_hash = _stub_config_identity(arm, task)
        if arm == "F8":
            treatment_hash = _hex("wrong-treatment-config", 64)
        return campaign_hash, treatment_hash

    with pytest.raises(ValueError, match="treatment_config_sha256"):
        manifests.build_training_manifest(
            tmp_path, **_build_kwargs(config_identity=bad_identity)
        )


def test_build_training_manifest_rejects_non_uniform_campaign_config_hash(tmp_path):
    _populate_valid_run_dirs(tmp_path)

    def varying_identity(arm, task):
        _, treatment_hash = _stub_config_identity(arm, task)
        return _hex(f"campaign-config:{arm}", 64), treatment_hash

    with pytest.raises(ValueError, match="campaign_config_sha256"):
        manifests.build_training_manifest(
            tmp_path, **_build_kwargs(config_identity=varying_identity)
        )


# ---------------------------------------------------------------------------
# CLI (build-training-manifest): exercises the real subprocess-free entry
# point in-process so ``default_config_identity`` can be monkeypatched to the
# same GPU/mjlab-free stub used above.
# ---------------------------------------------------------------------------


def test_cli_build_training_manifest_happy_path(tmp_path, monkeypatch):
    _populate_valid_run_dirs(tmp_path)
    monkeypatch.setattr(manifests, "default_config_identity", _stub_config_identity)
    out_path = tmp_path / "accepted_training_checkpoints.tsv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fq4x8_manifests",
            "build-training-manifest",
            "--log-root",
            str(tmp_path),
            "--code-revision",
            SHARED_CODE_REVISION,
            "--asset-revision",
            SHARED_ASSET_REVISION,
            "--attempt",
            "attempt1",
            "--out",
            str(out_path),
        ],
    )
    returncode = manifests._command_line()
    assert returncode == 0
    rows = manifests.parse_training_manifest(out_path.read_text(encoding="utf-8"))
    manifests.validate_training_manifest(rows)
    assert len(rows) == 32


def test_cli_build_training_manifest_fails_closed_and_writes_nothing(tmp_path, monkeypatch, capsys):
    _populate_valid_run_dirs(tmp_path)
    (tmp_path / _run_dir_name("D0", 10) / "model_499.pt").unlink()
    monkeypatch.setattr(manifests, "default_config_identity", _stub_config_identity)
    out_path = tmp_path / "accepted_training_checkpoints.tsv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fq4x8_manifests",
            "build-training-manifest",
            "--log-root",
            str(tmp_path),
            "--code-revision",
            SHARED_CODE_REVISION,
            "--asset-revision",
            SHARED_ASSET_REVISION,
            "--out",
            str(out_path),
        ],
    )
    returncode = manifests._command_line()
    assert returncode == 2
    captured = capsys.readouterr()
    assert "MANIFEST_FAIL" in captured.err
    assert not out_path.exists()
