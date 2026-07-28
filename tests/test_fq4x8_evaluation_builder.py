"""build-evaluation-manifest: fail-closed acceptance of one immutable attempt.

The load-bearing scientific test in this file is
``test_weak_but_valid_checkpoint_is_accepted``: acceptance must depend ONLY on
provenance, schema and validity sentinels.  If a poorly-performing but
structurally valid checkpoint could be rejected, the builder would silently
become a post-outcome seed-selection mechanism and the campaign's paired 4x8
design would be invalid.  Do not weaken that test to make another one pass.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.analysis import fq4x8_manifests as manifests
from evaluation.analysis import first_strike_quality_campaign as quality

from tests.test_fq4x8_manifests import (
    TRAINING_MANIFEST_SHA256,
    _hex,
    valid_evaluation_rows,
    valid_training_rows,
)

EVAL_ATTEMPT = "attempt2"
EVAL_RETRY_HISTORY = "attempt1:infra_failure;attempt2:accepted"
EVAL_CODE_REVISION = "ffac038c6ee5ad903fbd0feb33311b7c562699e4"
NAIL_ASSET_SHA256 = _hex("nail-asset", 64)
NAIL_GEOMETRY = {
    "nail_axis": [0.0, 0.0, -1.0],
    "nail_xy_m": [0.5, 0.0],
    "nail_radius_m": 0.012,
    "source_sha256": NAIL_ASSET_SHA256,
}
# The evaluator's OWN treatment label, which is NOT the arm label: arm "F8"
# is written as "F".  Verified against the real attempt2 artifacts -- an
# earlier fixture that assumed treatment == arm hid a builder defect.
EVALUATOR_TREATMENT = {"F8": "F", "F0": "F0", "D0": "D0", "FQ-min": "FQ"}
EVALUATOR_CAMPAIGN_CONFIG_SHA256 = {
    arm: _hex(f"evaluator-campaign-config:{arm}", 64)
    for arm in manifests.LABELS
}
EVALUATOR_TREATMENT_CONFIG_SHA256 = {
    arm: _hex(f"evaluator-treatment-config:{arm}", 64)
    for arm in manifests.LABELS
}
QUALITY_SENSOR_SLOT_COUNT = 8


# ---------------------------------------------------------------------------
# fixture: a synthetic-but-structurally-faithful immutable evaluation attempt
# ---------------------------------------------------------------------------
def _episode(env_id: int, ordinal: int, *, weak: bool = False, **overrides) -> dict:
    """One sampled episode.  ``weak`` = valid instrumentation, poor outcome."""
    sample_count = 4
    quality_value = 0.0 if weak else 0.83
    contact_error = 0.0 if weak else 0.012 * float(np.sqrt(1.0 - quality_value))
    contact_point = (
        [0.0, 0.0, 0.0]
        if weak
        else [0.5 + contact_error, 0.0, 0.1]
    )
    contact_normal = [0.0, 0.0, -1.0]
    contact = [False] * sample_count if weak else [False, True, True, False]
    found_first = [0] * sample_count if weak else [0, 1, 1, 0]
    normal_force_first = (
        [0.0] * sample_count if weak else [0.0, 10.0, 8.0, 0.0]
    )
    raw_point_first = (
        [[0.0, 0.0, 0.0]] * sample_count
        if weak
        else [
            [0.0, 0.0, 0.0],
            contact_point,
            contact_point,
            [0.0, 0.0, 0.0],
        ]
    )
    raw_normal_first = (
        [[0.0, 0.0, 0.0]] * sample_count
        if weak
        else [
            [0.0, 0.0, 0.0],
            contact_normal,
            contact_normal,
            [0.0, 0.0, 0.0],
        ]
    )
    found = [
        [value] + [0] * (QUALITY_SENSOR_SLOT_COUNT - 1)
        for value in found_first
    ]
    normal_force = [
        [value] + [0.0] * (QUALITY_SENSOR_SLOT_COUNT - 1)
        for value in normal_force_first
    ]
    raw_points = [
        [list(value)]
        + [[0.0, 0.0, 0.0] for _ in range(QUALITY_SENSOR_SLOT_COUNT - 1)]
        for value in raw_point_first
    ]
    raw_normals = [
        [list(value)]
        + [[0.0, 0.0, 0.0] for _ in range(QUALITY_SENSOR_SLOT_COUNT - 1)]
        for value in raw_normal_first
    ]
    latched_points = (
        [[0.0, 0.0, 0.0]] * sample_count
        if weak
        else [[0.0, 0.0, 0.0], contact_point, contact_point, contact_point]
    )
    latched_scalars = (
        [0.0] * sample_count
        if weak
        else [0.0, quality_value, quality_value, quality_value]
    )
    latched_errors = (
        [0.0] * sample_count
        if weak
        else [0.0, contact_error, contact_error, contact_error]
    )
    latched_valid = (
        [False] * sample_count
        if weak
        else [False, True, True, True]
    )
    latched_started = (
        [False] * sample_count
        if weak
        else [False, True, True, True]
    )
    latched_time = (
        [0.0] * sample_count
        if weak
        else [0.0, 0.004, 0.004, 0.004]
    )
    latched_axiality = (
        [0.0] * sample_count
        if weak
        else [0.0, 1.0, 1.0, 1.0]
    )
    qvel = [[0.0] * 6 for _ in range(sample_count)]
    episode = {
        "env_id": env_id,
        "episode_ordinal": ordinal,
        "episode_depth_m": 0.0 if weak else 0.021,
        # per-joint vector, exactly as the real schema-v3 traces record it
        "episode_peak_lambda": [0.0] * 6 if weak else [0.26, 0.2, 0.16, 0.1, 0.08, 0.05],
        "episode_delivered_accumulator_n_s": 0.0 if weak else 0.1,
        "overall_success": False if weak else True,
        "trace_digest": _hex(f"trace:{env_id}:{ordinal}", 64),
        "reset_state_digest": _hex(f"reset:{env_id}:{ordinal}", 64),
        "reset_contract_version": 1,
        "first_strike": {
            # A real no-contact episode has contact_quality_valid == False:
            # the tracker initialises it False and only writes it at onset
            # (src/tasks/hammer/mdp/first_strike.py). An earlier fixture set it
            # True while started was False -- physically impossible, and it
            # would have let a future "valid reading?" gate pass this test while
            # rejecting every real weak episode.
            "started": False if weak else True,
            "accepted_onset_index": None if weak else 1,
            "contact_point_w": contact_point,
            "contact_error_m": contact_error,
            "contact_quality": quality_value,
            "contact_quality_valid": False if weak else True,
            "contact_quality_overflow": False,
            "first_contact_time_s": 0.0 if weak else 0.004,
            "contact_normal_axiality": 0.0 if weak else 1.0,
        },
        # Instrument liveness: the tracker appends one reading per physics
        # substep regardless of contact, so a weak episode has a FULL-length
        # all-zero stream (in reality the longest, since it never terminates
        # early).  Length must match the dense physical series below.
        "event_trace": {
            "tracker_started": latched_started,
            "tracker_contact_point_w": latched_points,
            "tracker_contact_error_m": latched_errors,
            "tracker_contact_quality": latched_scalars,
            "tracker_contact_quality_valid": latched_valid,
            "tracker_contact_quality_overflow": [False] * sample_count,
            "tracker_first_contact_time_s": latched_time,
            "tracker_contact_normal_axiality": latched_axiality,
        },
        "physical": {
            "contact": contact,
            "joint_speed_rad_s": qvel,
            "post_step_joint_speed_rad_s": [list(row) for row in qvel],
            "quality_found_count": found,
            "quality_normal_force_n": normal_force,
            "quality_contact_position_m": raw_points,
            "quality_contact_normal": raw_normals,
        },
    }
    episode.update(overrides)
    return episode


def _strict_config_identities(training_row: dict) -> dict:
    """Evaluator identities are frozen per arm, never derived per seed."""
    arm = training_row["arm"]
    observation_sha = _hex(f"observation:{arm}", 64)
    return {
        "training_config_sha256": _hex(f"training-config:{arm}", 64),
        "evaluation_config_sha256": _hex(f"evaluation-config:{arm}", 64),
        "training_policy_observation_sha256": observation_sha,
        "evaluation_policy_observation_sha256": observation_sha,
        "training_treatment_reward_sha256": training_row["treatment_reward_sha256"],
        "evaluation_treatment_reward_sha256": training_row["treatment_reward_sha256"],
    }


def _artifact_provenance(
    training_row: dict,
    *,
    training_manifest_sha256: str = TRAINING_MANIFEST_SHA256,
) -> dict:
    arm = training_row["arm"]
    return {
        "code_git": {"revision": EVAL_CODE_REVISION, "dirty": False},
        "asset_git": {
            "revision": training_row["asset_revision"],
            "dirty": False,
        },
        "checkpoint_sha256": training_row["checkpoint_sha256"],
        "accepted_checkpoint_sha256": training_row["checkpoint_sha256"],
        "campaign_config_sha256": EVALUATOR_CAMPAIGN_CONFIG_SHA256[arm],
        "treatment_config_sha256": EVALUATOR_TREATMENT_CONFIG_SHA256[arm],
        "nail_asset_sha256": NAIL_ASSET_SHA256,
        "accepted_manifest_sha256": training_manifest_sha256,
        "training_code_revision": training_row["code_revision"],
        "training_asset_revision": training_row["asset_revision"],
    }


def _evaluation_contract(training_row: dict) -> dict:
    return {
        "num_envs": manifests.EXPECTED_NUM_ENVS,
        "episodes_per_env": manifests.EXPECTED_EPISODES_PER_ENV,
        "fixed_action_signature_sha256": training_row[
            "fixed_action_signature_sha256"
        ],
        "fixed_impedance_signature_sha256": training_row[
            "fixed_impedance_signature_sha256"
        ],
        "strict_config_identities": _strict_config_identities(training_row),
    }


def _payload(
    training_row: dict,
    *,
    weak: bool = False,
    training_manifest_sha256: str = TRAINING_MANIFEST_SHA256,
    **overrides,
) -> dict:
    arm = training_row["arm"]
    seed = training_row["training_seed"]
    episodes = [
        _episode(env_id, ordinal, weak=weak)
        for env_id in range(manifests.EXPECTED_NUM_ENVS)
        for ordinal in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    payload = {
        "schema_version": 3,
        "expected_episode_count": manifests.EXPECTED_N_EPISODES,
        "task": training_row["task"],
        "treatment": EVALUATOR_TREATMENT[arm],
        "training_seed": seed,
        "weights": {
            "impact_progress": quality.WEIGHTS[arm][0],
            "delivered_impulse": quality.WEIGHTS[arm][1],
        },
        "impulse_limits_n_m_s": list(quality.IMPULSE_LIMITS),
        "rng_streams": dict(manifests.EVALUATOR_RNG),
        "imp_max_p": 0.0,
        "evaluation_contract": _evaluation_contract(training_row),
        "nail_geometry": dict(NAIL_GEOMETRY),
        "provenance": _artifact_provenance(
            training_row,
            training_manifest_sha256=training_manifest_sha256,
        ),
        "mean_rollout_invariants": {
            "impossible_success_n": 0,
            "lambda_dead_n": 0,
        },
        "episodes": episodes,
    }
    payload.update(overrides)
    return payload


def _write_artifact(
    directory: Path, name: str, payload: dict, *, allow_nan: bool = False
) -> tuple[str, str, str]:
    """Content-address and persist one payload exactly as the evaluator does.

    ``allow_nan`` exists only so a test can forge an artifact the real
    evaluator could never write (``_digest`` rejects nonfinite values), and so
    prove the builder still refuses it.
    """
    if allow_nan:
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=True
            ).encode()
        ).hexdigest()
    else:
        digest = manifests._digest(payload)
    payload = dict(payload, payload_digest=digest)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=allow_nan)
    path = directory / f"{name}_{digest}_sampled_traces.npz"
    np.savez_compressed(
        path, payload_json=np.frombuffer(encoded.encode("utf-8"), dtype=np.uint8)
    )
    return str(path), digest, hashlib.sha256(path.read_bytes()).hexdigest()


def _summary_row(
    training_row: dict,
    trace_path: str,
    digest: str,
    artifact_sha: str,
    *,
    training_manifest_sha256: str = TRAINING_MANIFEST_SHA256,
) -> dict:
    return {
        "name": f"fq4x8_{training_row['short']}_seed{training_row['training_seed']}",
        "task": training_row["task"],
        "training_seed": training_row["training_seed"],
        "checkpoint_path": training_row["checkpoint_path"],
        "checkpoint_sha256": training_row["checkpoint_sha256"],
        "accepted_checkpoint_sha256": training_row["checkpoint_sha256"],
        "accepted_manifest_sha256": training_manifest_sha256,
        "training_code_revision": training_row["code_revision"],
        "training_asset_revision": training_row["asset_revision"],
        "git_revision": EVAL_CODE_REVISION,
        "git_dirty": "False",
        "asset_git_revision": training_row["asset_revision"],
        "asset_git_dirty": "False",
        "campaign_config_sha256": EVALUATOR_CAMPAIGN_CONFIG_SHA256[
            training_row["arm"]
        ],
        "treatment_config_sha256": EVALUATOR_TREATMENT_CONFIG_SHA256[
            training_row["arm"]
        ],
        "nail_asset_sha256": NAIL_ASSET_SHA256,
        "fixed_action_signature_sha256": training_row["fixed_action_signature_sha256"],
        "fixed_impedance_signature_sha256": training_row["fixed_impedance_signature_sha256"],
        "reset_rng_seed": manifests.EVALUATOR_RNG["reset"],
        "observation_rng_seed": manifests.EVALUATOR_RNG["observation"],
        "action_rng_seed": manifests.EVALUATOR_RNG["action"],
        "sampled_trace_path": trace_path,
        "sampled_trace_digest": digest,
        "sampled_trace_artifact_sha256": artifact_sha,
        "num_envs": manifests.EXPECTED_NUM_ENVS,
        "episodes_per_env_sampled": manifests.EXPECTED_EPISODES_PER_ENV,
        "n_episodes_sampled": manifests.EXPECTED_N_EPISODES,
        "impossible_success_n": 0,
        "lambda_dead_n": 0,
    }


def _make_attempt(
    tmp_path: Path,
    *,
    training_rows=None,
    weak_arms=(),
    order=None,
    payload_overrides=None,
    summary_overrides=None,
    drop=None,
    duplicate=None,
    training_manifest_sha256=TRAINING_MANIFEST_SHA256,
) -> tuple[Path, list[dict]]:
    training_rows = training_rows or valid_training_rows()
    attempt = tmp_path / EVAL_ATTEMPT
    attempt.mkdir(parents=True, exist_ok=True)
    rows = []
    ordered = list(training_rows) if order is None else order(list(training_rows))
    for training_row in ordered:
        ident = (training_row["arm"], training_row["training_seed"])
        if drop == ident:
            continue
        overrides = dict((payload_overrides or {}).get(ident, {}))
        allow_nan = bool(overrides.pop("_allow_nan", False))
        payload = _payload(
            training_row,
            weak=training_row["arm"] in weak_arms,
            training_manifest_sha256=training_manifest_sha256,
            **overrides,
        )
        name = f"fq4x8_{training_row['short']}_seed{training_row['training_seed']}"
        path, digest, artifact_sha = _write_artifact(
            attempt, name, payload, allow_nan=allow_nan
        )
        row = _summary_row(
            training_row,
            path,
            digest,
            artifact_sha,
            training_manifest_sha256=training_manifest_sha256,
        )
        row.update((summary_overrides or {}).get(ident, {}))
        rows.append(row)
    if duplicate is not None:
        rows.append(dict(rows[duplicate]))
    _write_summary(attempt, rows)
    return attempt, training_rows


def _write_summary(attempt: Path, rows: list[dict]) -> None:
    with (attempt / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build(attempt: Path, training_rows: list[dict], **kwargs):
    params = {
        "training_rows": training_rows,
        "training_manifest_sha256": TRAINING_MANIFEST_SHA256,
        "evaluation_attempt": EVAL_ATTEMPT,
        "evaluation_retry_history": EVAL_RETRY_HISTORY,
        "expected_evaluation_code_revision": EVAL_CODE_REVISION,
    }
    params.update(kwargs)
    return manifests.build_evaluation_manifest(attempt, **params)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------
def test_valid_attempt_builds_exactly_32_rows(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path)
    rows = _build(attempt, training_rows)
    assert len(rows) == 32
    manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)
    assert {(r["arm"], r["training_seed"]) for r in rows} == {
        (arm, seed) for arm in manifests.LABELS for seed in manifests.SEEDS
    }
    for row in rows:
        assert list(row) == list(manifests.EVALUATION_FIELDS)
        assert row["evaluation_attempt"] == EVAL_ATTEMPT
        assert row["disposition"] == "accepted"


def test_evaluation_retry_history_is_explicit_and_preserved(tmp_path):
    """The accepted rows must retain the failed first attempt, verbatim."""
    attempt, training_rows = _make_attempt(tmp_path)
    rows = manifests.build_evaluation_manifest(
        attempt,
        training_rows=training_rows,
        training_manifest_sha256=TRAINING_MANIFEST_SHA256,
        evaluation_attempt=EVAL_ATTEMPT,
        evaluation_retry_history=EVAL_RETRY_HISTORY,
        expected_evaluation_code_revision=EVAL_CODE_REVISION,
    )

    assert {
        row["evaluation_retry_history"] for row in rows
    } == {"attempt1:infra_failure;attempt2:accepted"}


@pytest.mark.parametrize(
    "retry_history",
    [
        "attempt2:accepted",
        "attempt1:rejected;attempt2:accepted",
        " attempt1:infra_failure;attempt2:accepted ",
    ],
)
def test_evaluation_retry_history_requires_exact_owner_approved_chain(
    tmp_path, retry_history
):
    attempt, training_rows = _make_attempt(tmp_path)

    with pytest.raises(ValueError, match="retry history|retry_history"):
        _build(
            attempt,
            training_rows,
            evaluation_retry_history=retry_history,
        )


def test_attempt_directory_name_must_equal_evaluation_attempt(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path)

    with pytest.raises(ValueError, match="attempt directory|evaluation_attempt"):
        _build(
            attempt,
            training_rows,
            evaluation_attempt="attempt3",
            evaluation_retry_history=(
                "attempt1:infra_failure;attempt3:accepted"
            ),
        )


def test_cli_forwards_explicit_evaluation_retry_history(
    tmp_path, monkeypatch
):
    training_rows = valid_training_rows()
    training_manifest = tmp_path / "accepted_training_checkpoints.tsv"
    training_manifest.write_text(
        manifests.serialize_training_manifest(training_rows),
        encoding="utf-8",
    )
    training_manifest_sha256 = hashlib.sha256(
        training_manifest.read_bytes()
    ).hexdigest()
    attempt, _ = _make_attempt(
        tmp_path,
        training_rows=training_rows,
        training_manifest_sha256=training_manifest_sha256,
    )
    out = tmp_path / "accepted_evaluations.tsv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fq4x8_manifests",
            "build-evaluation-manifest",
            "--attempt-dir",
            str(attempt),
            "--training-manifest",
            str(training_manifest),
            "--training-manifest-sha256",
            training_manifest_sha256,
            "--evaluation-attempt",
            EVAL_ATTEMPT,
            "--evaluation-retry-history",
            EVAL_RETRY_HISTORY,
            "--expected-evaluation-code-revision",
            EVAL_CODE_REVISION,
            "--out",
            str(out),
        ],
    )

    try:
        returncode = manifests._command_line()
    except SystemExit as error:
        pytest.fail(f"CLI rejected the retry-history flag: {error}")

    assert returncode == 0
    rows = manifests.parse_evaluation_manifest(
        out.read_text(encoding="utf-8")
    )
    assert {
        row["evaluation_retry_history"] for row in rows
    } == {"attempt1:infra_failure;attempt2:accepted"}


def test_produced_manifest_round_trips_through_validator(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path)
    rows = _build(attempt, training_rows)
    text = manifests.serialize_evaluation_manifest(rows)
    reparsed = manifests.parse_evaluation_manifest(text)
    manifests.validate_evaluation_manifest(
        reparsed, training_rows, TRAINING_MANIFEST_SHA256
    )


def test_rows_are_sorted_by_frozen_arm_order_then_seed(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path, order=lambda r: list(reversed(r)))
    rows = _build(attempt, training_rows)
    expected = [
        (arm, seed) for arm in manifests.LABELS for seed in sorted(manifests.SEEDS)
    ]
    assert [(r["arm"], r["training_seed"]) for r in rows] == expected


def test_shuffled_inputs_produce_byte_identical_output(tmp_path):
    forward, training_rows = _make_attempt(tmp_path / "fwd")
    reverse, _ = _make_attempt(tmp_path / "rev", order=lambda r: list(reversed(r)))
    a = manifests.serialize_evaluation_manifest(_build(forward, training_rows))
    b = manifests.serialize_evaluation_manifest(_build(reverse, training_rows))
    assert a.replace(str(forward), "<A>") == b.replace(str(reverse), "<A>")


# ---------------------------------------------------------------------------
# THE load-bearing scientific safeguard
# ---------------------------------------------------------------------------
def test_weak_but_valid_checkpoint_is_accepted(tmp_path):
    """A checkpoint that never drives the nail must still be ACCEPTED.

    Acceptance is a provenance/validity judgement, never a performance one.
    If this test ever fails, the builder has become a seed-selection filter.
    """
    attempt, training_rows = _make_attempt(tmp_path, weak_arms={"F0", "D0"})
    rows = _build(attempt, training_rows)
    assert len(rows) == 32
    manifests.validate_evaluation_manifest(rows, training_rows, TRAINING_MANIFEST_SHA256)
    assert all(row["disposition"] == "accepted" for row in rows)


def test_performance_changes_nothing_but_content_addresses(tmp_path):
    """ALL 32 checkpoints weak vs ALL strong must differ in content hashes ONLY.

    Stronger than asserting acceptance: this proves no performance quantity
    leaks into ANY emitted column. It fails the moment a reward, success rate,
    depth or speed reaches the manifest -- which is the only way this builder
    could become a post-outcome selection mechanism.
    """
    strong_attempt, training_rows = _make_attempt(tmp_path / "strong")
    weak_attempt, _ = _make_attempt(
        tmp_path / "weak", weak_arms=set(manifests.LABELS)
    )
    strong = _build(strong_attempt, training_rows)
    weak = _build(weak_attempt, training_rows)

    differing = {
        key
        for strong_row, weak_row in zip(strong, weak, strict=True)
        for key in strong_row
        if strong_row[key] != weak_row[key]
    }
    assert differing == {"sampled_trace_digest", "sampled_trace_artifact_sha256"}


def test_builder_emits_no_performance_field(tmp_path):
    """No decision/outcome field may leak into the accepted manifest."""
    attempt, training_rows = _make_attempt(tmp_path)
    rows = _build(attempt, training_rows)
    forbidden = set(quality.DECISION_FIELDS) | {
        "success_rate", "delivered_mean", "nail_depth_mean_mm", "overall_success",
        "worst_ratio_max", "reward",
    }
    for row in rows:
        assert not (set(row) & forbidden)


# ---------------------------------------------------------------------------
# fail-closed: quota / identity
# ---------------------------------------------------------------------------
def test_missing_row_fails(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path, drop=("F0", 11))
    with pytest.raises(ValueError, match="31|missing|identity"):
        _build(attempt, training_rows)


def test_duplicate_row_fails(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path, duplicate=0)
    with pytest.raises(ValueError, match="duplicate|33"):
        _build(attempt, training_rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0"),
        ("training_seed", 99),
        ("checkpoint_sha256", _hex("wrong-checkpoint", 64)),
        ("checkpoint_path", "/checkpoints/elsewhere/model_499.pt"),
    ],
)
def test_wrong_identity_fails(tmp_path, field, value):
    attempt, training_rows = _make_attempt(
        tmp_path, summary_overrides={("F8", 8): {field: value}}
    )
    with pytest.raises(ValueError):
        _build(attempt, training_rows)


def test_artifact_sha_tampering_fails(tmp_path):
    attempt, training_rows = _make_attempt(
        tmp_path,
        summary_overrides={("F8", 8): {"sampled_trace_artifact_sha256": _hex("t", 64)}},
    )
    with pytest.raises(ValueError, match="artifact|sha256"):
        _build(attempt, training_rows)


def test_payload_digest_tampering_fails(tmp_path):
    """Forge the artifact AND repair its summary SHA: the digest must still catch it.

    Rewriting the file alone is caught by the outer artifact-SHA check, so this
    test repairs that SHA first -- otherwise it would never exercise the inner
    payload-digest defence at all.
    """
    import csv as _csv

    attempt, training_rows = _make_attempt(tmp_path)
    target = next(attempt.glob("fq4x8_f8_seed8_*.npz"))
    with np.load(target, allow_pickle=False) as handle:
        payload = json.loads(handle["payload_json"].tobytes().decode("utf-8"))
    payload["training_seed"] = 99  # digest no longer describes the payload
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    np.savez_compressed(
        target, payload_json=np.frombuffer(encoded.encode("utf-8"), dtype=np.uint8)
    )

    summary_path = attempt / "summary.csv"
    with summary_path.open(newline="") as handle:
        rows = list(_csv.DictReader(handle))
    forged = hashlib.sha256(target.read_bytes()).hexdigest()
    for row in rows:
        if row["sampled_trace_path"] == str(target):
            row["sampled_trace_artifact_sha256"] = forged
    _write_summary(attempt, rows)

    with pytest.raises(ValueError, match="digest"):
        _build(attempt, training_rows)


def test_missing_artifact_file_fails(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path)
    next(attempt.glob("fq4x8_f8_seed8_*.npz")).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _build(attempt, training_rows)


# ---------------------------------------------------------------------------
# fail-closed: provenance
# ---------------------------------------------------------------------------
def test_manifest_sha_mismatch_fails(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        _build(attempt, training_rows, training_manifest_sha256=_hex("other", 64))


def test_evaluation_code_revision_mismatch_fails(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path)
    with pytest.raises(ValueError, match="revision"):
        _build(attempt, training_rows, expected_evaluation_code_revision="0" * 40)


@pytest.mark.parametrize("field", ["git_dirty", "asset_git_dirty"])
def test_dirty_evaluation_row_fails(tmp_path, field):
    attempt, training_rows = _make_attempt(
        tmp_path, summary_overrides={("F8", 8): {field: "True"}}
    )
    with pytest.raises(ValueError, match="dirty"):
        _build(attempt, training_rows)


def test_training_revision_mismatch_fails(tmp_path):
    attempt, training_rows = _make_attempt(
        tmp_path, summary_overrides={("F8", 8): {"training_code_revision": "0" * 40}}
    )
    with pytest.raises(ValueError, match="revision"):
        _build(attempt, training_rows)


@pytest.mark.parametrize(
    "key", ["reset_rng_seed", "observation_rng_seed", "action_rng_seed"]
)
def test_rng_identity_mismatch_fails(tmp_path, key):
    attempt, training_rows = _make_attempt(
        tmp_path, summary_overrides={("F8", 8): {key: 12345}}
    )
    with pytest.raises(ValueError, match="rng|seed"):
        _build(attempt, training_rows)


# ---------------------------------------------------------------------------
# fail-closed: sentinels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sentinel", ["impossible_success_n", "lambda_dead_n"])
def test_nonzero_summary_sentinel_fails(tmp_path, sentinel):
    attempt, training_rows = _make_attempt(
        tmp_path, summary_overrides={("F8", 8): {sentinel: 1}}
    )
    with pytest.raises(ValueError, match="sentinel|" + sentinel):
        _build(attempt, training_rows)


def test_quality_overflow_episode_fails(tmp_path):
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    episodes[7]["physical"]["quality_found_count"][1][0] = 9
    episodes[7]["event_trace"]["tracker_contact_quality_overflow"][1] = True
    episodes[7]["first_strike"]["contact_quality_overflow"] = True
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )
    with pytest.raises(ValueError, match="overflow"):
        _build(attempt, training_rows)


def test_nonfinite_episode_fails(tmp_path):
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    episodes[3]["episode_depth_m"] = float("nan")
    attempt, training_rows = _make_attempt(
        tmp_path,
        payload_overrides={("F8", 8): {"episodes": episodes, "_allow_nan": True}},
    )
    with pytest.raises(ValueError, match="nonfinite|finite"):
        _build(attempt, training_rows)


def test_null_inside_per_joint_vector_fails_general_numeric_validity(tmp_path):
    """A null buried in the 6-wide per-joint lambda vector must fail closed.

    Uses null rather than inf deliberately: inf is rejected earlier by the
    payload-digest guard (``allow_nan=False``), so an inf-based test never
    reaches the general numerical-validity gate. Null is JSON-legal, so it
    hashes fine and must still be rejected without crashing on the real
    six-wide vector channel.
    """
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    episodes[9]["episode_peak_lambda"] = [0.26, 0.2, None, 0.1, 0.08, 0.05]
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )
    with pytest.raises(ValueError, match="episode_peak_lambda|JSON numbers"):
        _build(attempt, training_rows)


def test_artifact_outside_attempt_directory_fails(tmp_path):
    """An artifact from ANOTHER attempt must be refused even if self-consistent.

    Without this, summary.csv can point at a different attempt's artifact and
    re-hash it self-consistently, yielding a manifest labelled one attempt whose
    rows are a MIXTURE of attempts -- artifact-level post-outcome selection.
    """
    import csv as _csv
    import shutil

    attempt, training_rows = _make_attempt(tmp_path)
    outside = tmp_path / "other_attempt"
    outside.mkdir()
    target = next(attempt.glob("fq4x8_f8_seed8_*.npz"))
    moved = outside / target.name
    shutil.move(str(target), str(moved))

    with (attempt / "summary.csv").open(newline="") as handle:
        rows = list(_csv.DictReader(handle))
    for row in rows:
        if Path(row["sampled_trace_path"]).name == moved.name:
            row["sampled_trace_path"] = str(moved)
    _write_summary(attempt, rows)

    with pytest.raises(ValueError, match="outside the attempt directory"):
        _build(attempt, training_rows)


def test_artifact_from_another_checkpoint_fails(tmp_path):
    """The artifact's own recorded checkpoint must match the accepted one."""
    attempt, training_rows = _make_attempt(
        tmp_path,
        payload_overrides={
            ("F8", 8): {
                "provenance": {
                    "checkpoint_sha256": _hex("some-other-checkpoint", 64),
                    "accepted_manifest_sha256": TRAINING_MANIFEST_SHA256,
                }
            }
        },
    )
    with pytest.raises(ValueError, match="not produced by the accepted checkpoint"):
        _build(attempt, training_rows)


@pytest.mark.parametrize(
    "path,bad_value",
    [
        pytest.param(
            ("accepted_checkpoint_sha256",),
            _hex("wrong-accepted-checkpoint", 64),
            id="accepted-checkpoint",
        ),
        pytest.param(
            ("accepted_manifest_sha256",),
            _hex("wrong-accepted-training-manifest", 64),
            id="accepted-training-manifest",
        ),
        pytest.param(
            ("training_code_revision",),
            "1" * 40,
            id="training-code-revision",
        ),
        pytest.param(
            ("training_asset_revision",),
            "2" * 40,
            id="training-asset-revision",
        ),
        pytest.param(
            ("code_git", "revision"),
            "3" * 40,
            id="evaluation-code-revision",
        ),
        pytest.param(
            ("code_git", "dirty"),
            True,
            id="dirty-evaluation-code",
        ),
        pytest.param(
            ("asset_git", "revision"),
            "4" * 40,
            id="evaluation-asset-revision",
        ),
        pytest.param(
            ("asset_git", "dirty"),
            True,
            id="dirty-evaluation-asset",
        ),
        pytest.param(
            ("nail_asset_sha256",),
            _hex("wrong-nail-asset", 64),
            id="nail-asset",
        ),
    ],
)
def test_payload_provenance_is_bound_to_accepted_evidence(
    tmp_path, path, bad_value
):
    """A self-consistently rehashed payload cannot rewrite its provenance."""
    training_rows = valid_training_rows()
    training_row = next(
        row
        for row in training_rows
        if (row["arm"], row["training_seed"]) == ("F8", 8)
    )
    provenance = _artifact_provenance(training_row)
    if len(path) == 1:
        provenance[path[0]] = bad_value
    else:
        provenance[path[0]][path[1]] = bad_value
    attempt, training_rows = _make_attempt(
        tmp_path,
        training_rows=training_rows,
        payload_overrides={("F8", 8): {"provenance": provenance}},
    )

    with pytest.raises(ValueError, match="provenance|revision|dirty|nail"):
        _build(attempt, training_rows)


@pytest.mark.parametrize("mutation", ["missing", "blank", "malformed"])
def test_summary_nail_asset_identity_requires_recorded_64_hex(
    tmp_path, mutation
):
    attempt, training_rows = _make_attempt(tmp_path)
    with (attempt / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if mutation == "missing":
            row.pop("nail_asset_sha256")
        elif mutation == "blank":
            row["nail_asset_sha256"] = ""
        else:
            row["nail_asset_sha256"] = "not-a-sha256"
    _write_summary(attempt, rows)

    with pytest.raises(ValueError, match="nail_asset_sha256"):
        _build(attempt, training_rows)


def test_payload_nail_geometry_source_matches_provenance(tmp_path):
    attempt, training_rows = _make_attempt(
        tmp_path,
        payload_overrides={
            ("F8", 8): {
                "nail_geometry": {
                    "source_sha256": _hex("wrong-nail-geometry-source", 64)
                }
            }
        },
    )

    with pytest.raises(ValueError, match="nail.*source|nail.*provenance"):
        _build(attempt, training_rows)


@pytest.mark.parametrize(
    "mutation",
    [
        "fixed-action",
        "fixed-impedance",
        "num-envs",
        "episodes-per-env",
        "expected-episode-count",
        "imp-max-p",
        "impact-weight",
        "delivered-weight",
        "impulse-limits",
    ],
)
def test_payload_contract_is_frozen(tmp_path, mutation):
    """Digest-valid payloads with a rewritten evaluation contract must fail."""
    training_rows = valid_training_rows()
    training_row = next(
        row
        for row in training_rows
        if (row["arm"], row["training_seed"]) == ("F8", 8)
    )
    overrides = {}
    if mutation in {
        "fixed-action",
        "fixed-impedance",
        "num-envs",
        "episodes-per-env",
    }:
        contract = _evaluation_contract(training_row)
        if mutation == "fixed-action":
            contract["fixed_action_signature_sha256"] = _hex(
                "wrong-action-signature", 64
            )
        elif mutation == "fixed-impedance":
            contract["fixed_impedance_signature_sha256"] = _hex(
                "wrong-impedance-signature", 64
            )
        elif mutation == "num-envs":
            contract["num_envs"] = 255
        else:
            contract["episodes_per_env"] = 1
        overrides["evaluation_contract"] = contract
    elif mutation == "expected-episode-count":
        overrides["expected_episode_count"] = 511
    elif mutation == "imp-max-p":
        overrides["imp_max_p"] = 0.1
    elif mutation in {"impact-weight", "delivered-weight"}:
        weights = {
            "impact_progress": quality.WEIGHTS["F8"][0],
            "delivered_impulse": quality.WEIGHTS["F8"][1],
        }
        if mutation == "impact-weight":
            weights["impact_progress"] = 7.9
        else:
            weights["delivered_impulse"] = 1.9
        overrides["weights"] = weights
    else:
        overrides["impulse_limits_n_m_s"] = [0.1] * 6

    attempt, training_rows = _make_attempt(
        tmp_path,
        training_rows=training_rows,
        payload_overrides={("F8", 8): overrides},
    )

    with pytest.raises(
        ValueError,
        match="action|impedance|quota|num_envs|episodes|imp_max_p|weight|impulse",
    ):
        _build(attempt, training_rows)


@pytest.mark.parametrize("mutation", ["blank", "absent"])
def test_unknown_provenance_fails_closed(tmp_path, mutation):
    """Unrecorded provenance is NOT evidence of clean provenance."""
    import csv as _csv

    attempt, training_rows = _make_attempt(tmp_path)
    with (attempt / "summary.csv").open(newline="") as handle:
        rows = list(_csv.DictReader(handle))
    for row in rows:
        if mutation == "blank":
            row["git_dirty"] = ""
        else:
            row.pop("git_dirty", None)
    _write_summary(attempt, rows)

    with pytest.raises(ValueError, match="blank|absent"):
        _build(attempt, training_rows)


def test_sparse_tracker_stream_fails_liveness(tmp_path):
    """A stream shorter than the substep series is a dead/sparse instrument.

    Guards the scenario where a future evaluator switches to event-window
    recording: 'non-empty' would silently start meaning 'made contact', turning
    liveness into a performance filter against weak seeds.
    """
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    episodes[11]["event_trace"]["tracker_contact_quality"] = [0.83]  # 1 vs 4 substeps
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )
    with pytest.raises(ValueError, match="liveness"):
        _build(attempt, training_rows)


def test_dead_instrument_episode_fails(tmp_path):
    """No tracker readings at all == dead instrument (NOT a weak policy)."""
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    episodes[5]["event_trace"]["tracker_contact_quality"] = []
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )
    with pytest.raises(ValueError, match="liveness"):
        _build(attempt, training_rows)


@pytest.mark.parametrize("mutation", ["missing", "non-list"])
def test_missing_or_non_list_physical_contact_fails_liveness(tmp_path, mutation):
    """The dense physical clock is required even when tracker samples exist."""
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    if mutation == "missing":
        episodes[13]["physical"].pop("contact")
    else:
        episodes[13]["physical"]["contact"] = "not-a-physical-series"
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )

    with pytest.raises(ValueError, match="liveness"):
        _build(attempt, training_rows)


@pytest.mark.parametrize("bad_value", [None, 0, "false"])
def test_overflow_snapshot_requires_a_present_json_boolean(tmp_path, bad_value):
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    if bad_value is None:
        episodes[17]["first_strike"].pop("contact_quality_overflow")
    else:
        episodes[17]["first_strike"]["contact_quality_overflow"] = bad_value
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )

    with pytest.raises(ValueError, match="quality_nonfinite_n"):
        _build(attempt, training_rows)


@pytest.mark.parametrize("mutation", ["missing", "non-list", "non-boolean"])
def test_malformed_event_overflow_stream_fails_quality_nonfinite(
    tmp_path, mutation
):
    """Malformed required overflow evidence is nonfinite, not true overflow."""
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    event = episodes[18]["event_trace"]
    if mutation == "missing":
        event.pop("tracker_contact_quality_overflow")
    elif mutation == "non-list":
        event["tracker_contact_quality_overflow"] = "false"
    else:
        event["tracker_contact_quality_overflow"][1] = 0
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )

    with pytest.raises(ValueError, match="quality_nonfinite_n"):
        _build(attempt, training_rows)


def test_no_contact_true_overflow_snapshot_is_nonfinite_not_overflow(tmp_path):
    """Without an accepted onset, a True overflow snapshot is malformed."""
    episodes = [
        _episode(e, o, weak=True)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    episodes[20]["first_strike"]["contact_quality_overflow"] = True
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )

    with pytest.raises(ValueError, match="quality_nonfinite_n"):
        _build(attempt, training_rows)


def test_liveness_and_malformed_snapshot_sentinels_are_independent():
    """A dead dense stream cannot hide malformed immutable snapshot evidence."""
    episode = _episode(0, 0)
    episode["event_trace"]["tracker_contact_quality"] = []
    episode["first_strike"].pop("contact_quality_overflow")

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["liveness_failure_n"] == 1
    assert sentinels["quality_nonfinite_n"] == 1


def test_liveness_and_out_of_range_snapshot_onset_are_independent():
    """Known physical length still bounds onset when tracker liveness is dead."""
    episode = _episode(0, 0)
    episode["event_trace"]["tracker_contact_quality"] = []
    episode["first_strike"]["accepted_onset_index"] = 99

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["liveness_failure_n"] == 1
    assert sentinels["quality_nonfinite_n"] == 1


def test_liveness_and_missing_raw_quality_geometry_are_independent():
    """A dead tracker stream cannot hide missing required raw geometry."""
    episode = _episode(0, 0)
    episode["event_trace"]["tracker_contact_quality"] = []
    episode["physical"].pop("quality_found_count")

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["liveness_failure_n"] == 1
    assert sentinels["quality_nonfinite_n"] == 1


def test_dead_liveness_and_wrong_raw_slot_width_are_independent():
    """A dead tracker stream cannot hide the one-slot fallback signature."""

    episode = _episode(0, 0)
    episode["event_trace"]["tracker_contact_quality"] = []
    _resize_raw_quality_slots(episode, 1)

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["liveness_failure_n"] == 1
    assert sentinels["quality_nonfinite_n"] == 1


def test_liveness_and_nonfinite_qvel_gate_are_independent():
    """Dead tracker liveness cannot suppress general qvel validation."""
    episode = _episode(0, 0)
    episode["event_trace"]["tracker_contact_quality"] = []
    episode["physical"]["joint_speed_rad_s"][1][0] = None

    with pytest.raises(ValueError, match="joint_speed_rad_s"):
        manifests._derive_validity_sentinels(
            [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
        )


def test_disagreeing_onset_overflow_is_nonfinite_not_true_overflow():
    """Only three valid onset representations agreeing True count overflow."""
    episode = _episode(0, 0)
    episode["first_strike"]["contact_quality_overflow"] = True

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["quality_overflow_n"] == 0
    assert sentinels["quality_nonfinite_n"] == 1


def test_liveness_and_onset_overflow_disagreement_are_independent():
    """Dead quality samples cannot hide malformed onset-overflow evidence."""
    episode = _episode(0, 0)
    episode["event_trace"]["tracker_contact_quality"] = []
    episode["first_strike"]["contact_quality_overflow"] = True

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["liveness_failure_n"] == 1
    assert sentinels["quality_overflow_n"] == 0
    assert sentinels["quality_nonfinite_n"] == 1


def test_liveness_and_aligned_true_onset_overflow_are_independent():
    """Dead quality samples cannot hide valid all-True onset overflow."""
    episode = _episode(0, 0)
    episode["event_trace"]["tracker_contact_quality"] = []
    episode["physical"]["quality_found_count"][1][0] = 9
    episode["event_trace"]["tracker_contact_quality_overflow"][1] = True
    episode["first_strike"]["contact_quality_overflow"] = True

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["liveness_failure_n"] == 1
    assert sentinels["quality_overflow_n"] == 1
    assert sentinels["quality_nonfinite_n"] == 0


def test_liveness_and_malformed_event_overflow_are_independent():
    """Dead quality samples cannot hide missing required event overflow."""
    episode = _episode(0, 0)
    episode["event_trace"]["tracker_contact_quality"] = []
    episode["event_trace"].pop("tracker_contact_quality_overflow")

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["liveness_failure_n"] == 1
    assert sentinels["quality_nonfinite_n"] == 1


def test_missing_physical_contact_and_event_overflow_are_independent():
    """Raw quality length still exposes malformed overflow without contact."""
    episode = _episode(0, 0)
    episode["physical"].pop("contact")
    episode["event_trace"].pop("tracker_contact_quality_overflow")

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["liveness_failure_n"] == 1
    assert sentinels["quality_nonfinite_n"] == 1


def test_missing_physical_contact_and_aligned_overflow_are_independent():
    """Raw quality length still exposes genuine overflow without contact."""
    episode = _episode(0, 0)
    episode["physical"].pop("contact")
    episode["physical"]["quality_found_count"][1][0] = 9
    episode["event_trace"]["tracker_contact_quality_overflow"][1] = True
    episode["first_strike"]["contact_quality_overflow"] = True

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["liveness_failure_n"] == 1
    assert sentinels["quality_overflow_n"] == 1
    assert sentinels["quality_nonfinite_n"] == 0


def test_aligned_true_onset_overflow_counts_as_overflow():
    """All three accepted-onset representations agreeing True is overflow."""
    episode = _episode(0, 0)
    episode["physical"]["quality_found_count"][1][0] = 9
    episode["event_trace"]["tracker_contact_quality_overflow"][1] = True
    episode["first_strike"]["contact_quality_overflow"] = True

    sentinels = manifests._derive_validity_sentinels(
        [episode], nail_geometry=NAIL_GEOMETRY, prefix="test"
    )

    assert sentinels["quality_overflow_n"] == 1
    assert sentinels["quality_nonfinite_n"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda episode: episode["physical"].pop("quality_found_count"),
            id="missing-found-count",
        ),
        pytest.param(
            lambda episode: episode["physical"].__setitem__(
                "quality_contact_position_m", [[0.0]] * 4
            ),
            id="malformed-contact-position",
        ),
        pytest.param(
            lambda episode: episode["physical"]["quality_contact_normal"][1][
                0
            ].__setitem__(0, None),
            id="nonfinite-contact-normal",
        ),
    ],
)
def test_invalid_raw_onset_geometry_fails_quality_nonfinite(tmp_path, mutation):
    """Missing, malformed, and nonfinite raw onset geometry are invalid."""
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    mutation(episodes[19])
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )

    with pytest.raises(ValueError, match="quality_nonfinite_n"):
        _build(attempt, training_rows)


def test_no_contact_full_length_zero_quality_streams_are_accepted(tmp_path):
    """No contact is a valid zero-quality outcome, never dead instrumentation."""
    attempt, training_rows = _make_attempt(
        tmp_path, weak_arms=set(manifests.LABELS)
    )

    rows = _build(attempt, training_rows)

    assert len(rows) == 32
    assert all(row["liveness_failure_n"] == 0 for row in rows)
    assert all(row["quality_nonfinite_n"] == 0 for row in rows)


def _resize_raw_quality_slots(episode: dict, width: int) -> None:
    """Keep the first raw slot and resize the instrument signature."""

    physical = episode["physical"]
    for key in ("quality_found_count", "quality_normal_force_n"):
        zero = 0 if key == "quality_found_count" else 0.0
        physical[key] = [
            [row[0]] + [zero] * (width - 1) for row in physical[key]
        ]
    for key in ("quality_contact_position_m", "quality_contact_normal"):
        physical[key] = [
            [list(row[0])]
            + [[0.0, 0.0, 0.0] for _ in range(width - 1)]
            for row in physical[key]
        ]


@pytest.mark.parametrize(
    "width",
    [
        pytest.param(1, id="missing-sensor-one-slot-fallback"),
        pytest.param(7, id="undersized-seven-slot"),
        pytest.param(9, id="oversized-nine-slot"),
    ],
)
def test_raw_quality_instrument_requires_exactly_eight_slots(tmp_path, width):
    """A digest-bound episode with the wrong sensor width must fail closed."""

    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    _resize_raw_quality_slots(episodes[23], width)
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )

    with pytest.raises(ValueError, match="quality_nonfinite_n"):
        _build(attempt, training_rows)


def test_short_episode_quota_fails(tmp_path):
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ][:-1]
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )
    with pytest.raises(ValueError, match="quota|511|episode"):
        _build(attempt, training_rows)


def test_non_schema_v3_artifact_fails(tmp_path):
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"schema_version": 2}}
    )
    with pytest.raises(ValueError, match="schema"):
        _build(attempt, training_rows)


# ---------------------------------------------------------------------------
# atomic, non-clobbering write
# ---------------------------------------------------------------------------
_NEGATIVE_EVALUATION_INTEGER_FIELDS = (
    "training_seed",
    "reset_rng_seed",
    "observation_rng_seed",
    "action_rng_seed",
    "num_envs",
    "episodes_per_env_sampled",
    "n_episodes_sampled",
    "quality_overflow_n",
    "quality_nonfinite_n",
    "liveness_failure_n",
    "impossible_success_n",
    "lambda_dead_n",
    "quota_failure_n",
)

_NON_EXACT_EVALUATION_INTEGERS = (
    pytest.param("training_seed", 8.5, id="fractional-seed"),
    pytest.param("training_seed", True, id="boolean-seed"),
    pytest.param("reset_rng_seed", 2_036_072_919.5, id="fractional-reset-rng"),
    pytest.param("reset_rng_seed", False, id="boolean-reset-rng"),
    pytest.param(
        "observation_rng_seed", 2_046_072_933.5, id="fractional-observation-rng"
    ),
    pytest.param("observation_rng_seed", False, id="boolean-observation-rng"),
    pytest.param("action_rng_seed", 2_056_072_941.5, id="fractional-action-rng"),
    pytest.param("action_rng_seed", False, id="boolean-action-rng"),
    pytest.param("num_envs", 256.5, id="fractional-num-envs"),
    pytest.param("num_envs", False, id="boolean-num-envs"),
    pytest.param(
        "episodes_per_env_sampled", 2.5, id="fractional-episodes-per-env"
    ),
    pytest.param(
        "episodes_per_env_sampled", False, id="boolean-episodes-per-env"
    ),
    pytest.param("n_episodes_sampled", 512.5, id="fractional-episode-count"),
    pytest.param("n_episodes_sampled", False, id="boolean-episode-count"),
    *(
        pytest.param(sentinel, bad, id=f"{kind}-{sentinel}")
        for sentinel in (
            "quality_overflow_n",
            "quality_nonfinite_n",
            "liveness_failure_n",
            "impossible_success_n",
            "lambda_dead_n",
            "quota_failure_n",
        )
        for bad, kind in ((0.5, "fractional"), (False, "boolean"))
    ),
    *(
        pytest.param(field, -1, id=f"negative-{field}")
        for field in _NEGATIVE_EVALUATION_INTEGER_FIELDS
    ),
)


@pytest.mark.parametrize("field,bad_value", _NON_EXACT_EVALUATION_INTEGERS)
def test_evaluation_validator_rejects_non_exact_integers(field, bad_value):
    training_rows = valid_training_rows()
    rows = valid_evaluation_rows(training_rows)
    rows[0][field] = bad_value

    with pytest.raises(ValueError, match=field):
        manifests.validate_evaluation_manifest(
            rows, training_rows, TRAINING_MANIFEST_SHA256
        )


@pytest.mark.parametrize("field,bad_value", _NON_EXACT_EVALUATION_INTEGERS)
def test_write_rejects_non_exact_integers_without_serializing(
    tmp_path, field, bad_value
):
    training_rows = valid_training_rows()
    rows = valid_evaluation_rows(training_rows)
    rows[0][field] = bad_value
    out = tmp_path / "accepted_evaluations.tsv"

    with pytest.raises(ValueError, match=field):
        manifests.write_evaluation_manifest(
            rows,
            out,
            training_rows=training_rows,
            training_manifest_sha256=TRAINING_MANIFEST_SHA256,
        )

    assert not out.exists()


def test_fractional_quality_overflow_never_serializes_as_zero():
    training_rows = valid_training_rows()
    rows = valid_evaluation_rows(training_rows)
    rows[0]["quality_overflow_n"] = 0.5

    with pytest.raises(ValueError, match="quality_overflow_n"):
        manifests.serialize_evaluation_manifest(rows)


@pytest.mark.parametrize("field", _NEGATIVE_EVALUATION_INTEGER_FIELDS)
def test_negative_integer_never_serializes(field):
    training_rows = valid_training_rows()
    rows = valid_evaluation_rows(training_rows)
    rows[0][field] = -1

    with pytest.raises(ValueError, match=field):
        manifests.serialize_evaluation_manifest(rows)


def test_write_refuses_to_overwrite_existing_manifest(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path)
    rows = _build(attempt, training_rows)
    out = tmp_path / "accepted_evaluations.tsv"
    kw = dict(training_rows=training_rows, training_manifest_sha256=TRAINING_MANIFEST_SHA256)
    manifests.write_evaluation_manifest(rows, out, **kw)
    original = out.read_bytes()
    with pytest.raises(FileExistsError):
        manifests.write_evaluation_manifest(rows, out, **kw)
    assert out.read_bytes() == original


def test_competing_writer_at_publication_is_not_overwritten(tmp_path, monkeypatch):
    training_rows = valid_training_rows()
    rows = valid_evaluation_rows(training_rows)
    out = tmp_path / "accepted_evaluations.tsv"
    competing_bytes = b"published by the competing writer\n"
    real_link = os.link
    real_replace = Path.replace

    def link_after_competitor(source, destination, *args, **kwargs):
        Path(destination).write_bytes(competing_bytes)
        return real_link(source, destination, *args, **kwargs)

    def replace_after_competitor(source, destination):
        Path(destination).write_bytes(competing_bytes)
        return real_replace(source, destination)

    # Cover the inherited replacement publication and the desired no-replace
    # publication with the same deterministic race injection.
    monkeypatch.setattr(os, "link", link_after_competitor)
    monkeypatch.setattr(Path, "replace", replace_after_competitor)

    with pytest.raises(FileExistsError):
        manifests.write_evaluation_manifest(
            rows,
            out,
            training_rows=training_rows,
            training_manifest_sha256=TRAINING_MANIFEST_SHA256,
        )

    assert out.read_bytes() == competing_bytes
    assert not list(tmp_path.glob(f".{out.name}.*.tmp"))


def test_repeated_builds_to_separate_paths_are_byte_identical(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path)
    first = tmp_path / "one.tsv"
    second = tmp_path / "two.tsv"
    kw = dict(training_rows=training_rows, training_manifest_sha256=TRAINING_MANIFEST_SHA256)
    sha_a = manifests.write_evaluation_manifest(_build(attempt, training_rows), first, **kw)
    sha_b = manifests.write_evaluation_manifest(_build(attempt, training_rows), second, **kw)
    assert first.read_bytes() == second.read_bytes()
    assert sha_a == sha_b == hashlib.sha256(first.read_bytes()).hexdigest()


def test_write_rejects_semantically_invalid_rows(tmp_path):
    """The writer must run the FULL validator, not just a structural round-trip.

    Reviewer A reproduction: rows with a wrong task and a nonzero sentinel were
    serialized, written, and returned a SHA -- because parse_evaluation_manifest
    only checks structure and types.
    """
    attempt, training_rows = _make_attempt(tmp_path)
    rows = _build(attempt, training_rows)
    rows[0]["task"] = "WRONG"
    rows[0]["quality_overflow_n"] = 7
    out = tmp_path / "accepted_evaluations.tsv"
    with pytest.raises(ValueError):
        manifests.write_evaluation_manifest(
            rows, out, training_rows=training_rows,
            training_manifest_sha256=TRAINING_MANIFEST_SHA256,
        )
    assert not out.exists()


@pytest.mark.parametrize(
    "field",
    [
        "campaign_config_sha256",
        "treatment_config_sha256",
        "fixed_action_signature_sha256",
        "fixed_impedance_signature_sha256",
    ],
)
def test_evaluator_config_drift_is_not_silently_masked(tmp_path, field):
    """Evaluator config identities must be cross-bound, not overwritten.

    Reviewer A reproduction: setting these to garbage in summary.csv produced 32
    accepted rows carrying the TRAINING manifest's values -- masking the drift.
    """
    attempt, training_rows = _make_attempt(
        tmp_path, summary_overrides={("F8", 8): {field: "garbage"}}
    )
    with pytest.raises(ValueError, match=field + "|identity|match"):
        _build(attempt, training_rows)


@pytest.mark.parametrize(
    "field", ["campaign_config_sha256", "treatment_config_sha256"]
)
@pytest.mark.parametrize("mutation", ["missing", "blank", "malformed"])
def test_evaluator_config_identity_requires_recorded_64_hex(
    tmp_path, field, mutation
):
    """Removing the producer's config identity must invalidate the attempt."""
    attempt, training_rows = _make_attempt(tmp_path)
    with (attempt / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if mutation == "missing":
            row.pop(field)
        elif mutation == "blank":
            row[field] = ""
        else:
            row[field] = "not-a-sha256"
    _write_summary(attempt, rows)

    with pytest.raises(ValueError, match=field):
        _build(attempt, training_rows)


@pytest.mark.parametrize(
    "field", ["campaign_config_sha256", "treatment_config_sha256"]
)
def test_all_arm_identical_evaluator_config_identity_fails(tmp_path, field):
    """One plausible-looking garbage digest cannot label all four arms."""
    shared_garbage = _hex(f"all-arm-garbage:{field}", 64)
    training_rows = valid_training_rows()
    summary_overrides = {
        (arm, seed): {field: shared_garbage}
        for arm in manifests.LABELS
        for seed in manifests.SEEDS
    }
    payload_overrides = {}
    for training_row in training_rows:
        provenance = _artifact_provenance(training_row)
        provenance[field] = shared_garbage
        payload_overrides[
            (training_row["arm"], training_row["training_seed"])
        ] = {"provenance": provenance}
    attempt, training_rows = _make_attempt(
        tmp_path,
        training_rows=training_rows,
        summary_overrides=summary_overrides,
        payload_overrides=payload_overrides,
    )

    with pytest.raises(ValueError, match="identical across all four arms"):
        _build(attempt, training_rows)


@pytest.mark.parametrize(
    "field", ["campaign_config_sha256", "treatment_config_sha256"]
)
def test_summary_evaluator_config_identity_must_match_payload_provenance(
    tmp_path, field
):
    """The unhashed summary identity must equal the digest-bound payload copy."""
    training_rows = valid_training_rows()
    training_row = next(
        row
        for row in training_rows
        if (row["arm"], row["training_seed"]) == ("F8", 8)
    )
    provenance = _artifact_provenance(training_row)
    provenance[field] = _hex(f"payload-drift:{field}", 64)
    attempt, training_rows = _make_attempt(
        tmp_path,
        training_rows=training_rows,
        payload_overrides={("F8", 8): {"provenance": provenance}},
    )

    with pytest.raises(ValueError, match=field + "|provenance"):
        _build(attempt, training_rows)


def test_evaluator_config_derivation_remains_distinct_from_training_manifest(
    tmp_path,
):
    """Valid evaluator identities differ; the 39 fields retain training values."""
    attempt, training_rows = _make_attempt(tmp_path)
    rows = _build(attempt, training_rows)
    training_index = {
        (row["arm"], row["training_seed"]): row for row in training_rows
    }
    with (attempt / "summary.csv").open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    for summary, row in zip(summary_rows, rows, strict=True):
        key = (row["arm"], row["training_seed"])
        training_row = training_index[key]
        for field in ("campaign_config_sha256", "treatment_config_sha256"):
            assert summary[field] != training_row[field]
            assert row[field] == training_row[field]


@pytest.mark.parametrize(
    "keys",
    [
        ("training_config_sha256",),
        ("evaluation_config_sha256",),
        (
            "training_policy_observation_sha256",
            "evaluation_policy_observation_sha256",
        ),
        (
            "training_treatment_reward_sha256",
            "evaluation_treatment_reward_sha256",
        ),
    ],
)
def test_strict_config_identities_are_consistent_per_arm(tmp_path, keys):
    """One seed cannot carry a distinct strict identity from its seven peers."""
    training_rows = valid_training_rows()
    training_row = next(
        row
        for row in training_rows
        if (row["arm"], row["training_seed"]) == ("F8", 8)
    )
    contract = _evaluation_contract(training_row)
    one_seed_identity = _hex(
        "one-seed-drift:" + ",".join(keys), 64
    )
    for key in keys:
        contract["strict_config_identities"][key] = one_seed_identity
    attempt, training_rows = _make_attempt(
        tmp_path,
        training_rows=training_rows,
        payload_overrides={("F8", 8): {"evaluation_contract": contract}},
    )

    with pytest.raises(
        ValueError,
        match="consistent|strict_config|treatment_reward",
    ):
        _build(attempt, training_rows)


@pytest.mark.parametrize("identity_key", manifests.STRICT_CONFIG_IDENTITY_KEYS)
@pytest.mark.parametrize("mutation", ["missing", "blank", "malformed"])
def test_strict_config_identity_requires_recorded_64_hex(
    tmp_path, identity_key, mutation
):
    training_rows = valid_training_rows()
    training_row = next(
        row
        for row in training_rows
        if (row["arm"], row["training_seed"]) == ("F8", 8)
    )
    contract = _evaluation_contract(training_row)
    identities = contract["strict_config_identities"]
    if mutation == "missing":
        identities.pop(identity_key)
    elif mutation == "blank":
        identities[identity_key] = ""
    else:
        identities[identity_key] = "not-a-sha256"
    attempt, training_rows = _make_attempt(
        tmp_path,
        training_rows=training_rows,
        payload_overrides={("F8", 8): {"evaluation_contract": contract}},
    )

    with pytest.raises(
        ValueError,
        match=identity_key + "|strict_config_identities",
    ):
        _build(attempt, training_rows)


def test_fractional_schema_version_is_rejected(tmp_path):
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"schema_version": 3.9}}
    )
    with pytest.raises(ValueError, match="schema"):
        _build(attempt, training_rows)


def test_fractional_artifact_training_seed_is_rejected(tmp_path):
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"training_seed": 8.9}}
    )
    with pytest.raises(ValueError, match="training_seed"):
        _build(attempt, training_rows)


@pytest.mark.parametrize(
    "stream,bad_value",
    [
        pytest.param("reset", 2_036_072_919.0, id="integral-float-reset"),
        pytest.param(
            "observation", 2_046_072_933.0, id="integral-float-observation"
        ),
        pytest.param("action", 2_056_072_941.0, id="integral-float-action"),
        pytest.param("reset", True, id="boolean"),
        pytest.param("reset", -1, id="negative"),
    ],
)
def test_artifact_rng_streams_require_exact_nonnegative_integers(
    tmp_path, stream, bad_value
):
    rng_streams = {
        "reset": 2_036_072_919,
        "observation": 2_046_072_933,
        "action": 2_056_072_941,
    }
    rng_streams[stream] = bad_value
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"rng_streams": rng_streams}}
    )

    with pytest.raises(ValueError, match=f"rng_streams|{stream}|integer"):
        _build(attempt, training_rows)


def test_fractional_summary_sentinel_is_rejected(tmp_path):
    """'0.5' must not be truncated to a passing 0."""
    attempt, training_rows = _make_attempt(
        tmp_path, summary_overrides={("F8", 8): {"impossible_success_n": "0.5"}}
    )
    with pytest.raises(ValueError, match="impossible_success_n|integer"):
        _build(attempt, training_rows)


@pytest.mark.parametrize(
    ("sentinel", "mutate"),
    [
        pytest.param(
            "impossible_success_n",
            lambda episode: episode.__setitem__(
                "episode_delivered_accumulator_n_s", 0.0
            ),
            id="sampled-impossible-success",
        ),
        pytest.param(
            "lambda_dead_n",
            lambda episode: (
                episode.__setitem__("overall_success", False),
                episode.__setitem__("episode_peak_lambda", [0.0] * 6),
            ),
            id="sampled-lambda-dead",
        ),
    ],
)
def test_forged_zero_summary_rejects_nonzero_sampled_invariant(
    tmp_path, sentinel, mutate
):
    """Digest-bound sampled episodes, not unhashed CSV, determine the count."""
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    mutate(episodes[23])
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )

    with pytest.raises(ValueError, match=f"{sentinel}|digest-bound|disagrees"):
        _build(attempt, training_rows)


@pytest.mark.parametrize("sentinel", ["impossible_success_n", "lambda_dead_n"])
def test_forged_zero_summary_rejects_nonzero_mean_invariant(tmp_path, sentinel):
    mean_counts = {"impossible_success_n": 0, "lambda_dead_n": 0}
    mean_counts[sentinel] = 1
    attempt, training_rows = _make_attempt(
        tmp_path,
        payload_overrides={
            ("F8", 8): {"mean_rollout_invariants": mean_counts}
        },
    )

    with pytest.raises(ValueError, match=f"{sentinel}|digest-bound|disagrees"):
        _build(attempt, training_rows)


@pytest.mark.parametrize("sentinel", ["impossible_success_n", "lambda_dead_n"])
def test_nonzero_summary_rejects_zero_digest_bound_invariant(tmp_path, sentinel):
    attempt, training_rows = _make_attempt(
        tmp_path, summary_overrides={("F8", 8): {sentinel: 1}}
    )

    with pytest.raises(ValueError, match=f"{sentinel}.*digest-bound|disagrees"):
        _build(attempt, training_rows)


@pytest.mark.parametrize("bad_count", [0.5, True])
@pytest.mark.parametrize("sentinel", ["impossible_success_n", "lambda_dead_n"])
def test_mean_invariant_counts_require_exact_nonnegative_integers(
    tmp_path, sentinel, bad_count
):
    mean_counts = {"impossible_success_n": 0, "lambda_dead_n": 0}
    mean_counts[sentinel] = bad_count
    attempt, training_rows = _make_attempt(
        tmp_path,
        payload_overrides={
            ("F8", 8): {"mean_rollout_invariants": mean_counts}
        },
    )

    with pytest.raises(ValueError, match=f"{sentinel}|integer|boolean"):
        _build(attempt, training_rows)


def test_post_onset_overflow_does_not_change_accepted_snapshot_scope(tmp_path):
    """Only overflow at the tracker-accepted onset belongs to this sentinel."""
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    episodes[29]["physical"]["quality_found_count"][3][0] = 9
    episodes[29]["event_trace"]["tracker_contact_quality_overflow"][3] = True
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )

    rows = _build(attempt, training_rows)

    accepted = next(
        row
        for row in rows
        if row["arm"] == "F8" and row["training_seed"] == 8
    )
    assert accepted["quality_overflow_n"] == 0


def test_duplicate_episode_ordinals_fail_quota(tmp_path):
    """512 episodes with the right per-env COUNT but wrong ordinals must fail."""
    episodes = [
        _episode(e, 0)  # every ordinal 0 -> (env, ordinal) pairs are duplicated
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for _ in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )
    with pytest.raises(ValueError, match="quota"):
        _build(attempt, training_rows)


def test_empty_depth_is_rejected_by_general_numeric_validity(tmp_path):
    """An empty depth value is not a general finite measurement."""
    episodes = [
        _episode(e, o)
        for e in range(manifests.EXPECTED_NUM_ENVS)
        for o in range(manifests.EXPECTED_EPISODES_PER_ENV)
    ]
    episodes[4]["episode_depth_m"] = []
    attempt, training_rows = _make_attempt(
        tmp_path, payload_overrides={("F8", 8): {"episodes": episodes}}
    )
    with pytest.raises(ValueError, match="episode_depth_m|finite JSON number"):
        _build(attempt, training_rows)


def test_write_uses_a_unique_temporary_file(tmp_path):
    """A fixed .tmp name lets a concurrent writer swap the published bytes."""
    attempt, training_rows = _make_attempt(tmp_path)
    rows = _build(attempt, training_rows)
    out = tmp_path / "accepted_evaluations.tsv"
    digest = manifests.write_evaluation_manifest(
        rows, out, training_rows=training_rows,
        training_manifest_sha256=TRAINING_MANIFEST_SHA256,
    )
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    assert not (tmp_path / "accepted_evaluations.tsv.tmp").exists()
    assert not list(tmp_path.glob("*.tmp*"))


def test_write_leaves_no_partial_file_on_validation_failure(tmp_path):
    attempt, training_rows = _make_attempt(tmp_path)
    rows = _build(attempt, training_rows)
    rows[0]["disposition"] = "rejected"  # validator must refuse this
    out = tmp_path / "accepted_evaluations.tsv"
    with pytest.raises(ValueError):
        manifests.write_evaluation_manifest(
            rows, out, training_rows=training_rows,
            training_manifest_sha256=TRAINING_MANIFEST_SHA256,
        )
    assert not out.exists()
    assert not list(tmp_path.glob("*.tmp*"))
