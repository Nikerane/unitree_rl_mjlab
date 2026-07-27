"""Tests for the fail-closed fq4x8 accepted-manifest tooling.

Frozen constants (arms, seeds, task ids, reward/action/impedance hashes) are
imported from ``first_strike_quality_campaign`` / ``first_strike_campaign``
rather than restated, so these tests exercise the same source of truth the
implementation binds against.
"""

from __future__ import annotations

import copy
import hashlib
import random

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
