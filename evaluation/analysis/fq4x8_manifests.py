"""Deterministic, fail-closed manifest tooling for the ``fq4x8`` campaign.

Two manifests are produced/consumed here, both plain tab-separated files:

- ``accepted_training_checkpoints.tsv`` (32 rows: 4 arms x 8 seeds) binds
  each accepted training checkpoint to its frozen treatment/task/action/
  impedance/cap identity.  It carries no evaluation-attempt fields.
- ``accepted_evaluations.tsv`` (32 rows, one per training row) additionally
  binds the strict evaluation contract: the SHA-256 of the training manifest
  it consumed, the checkpoint identity it evaluated, the three frozen
  evaluator RNG streams, the six ``STRICT_CONFIG_IDENTITY_KEYS``, the
  schema-v3 payload hashes, the 256x2=512 sampled-episode quota, and the six
  zero-required sentinels.

All frozen identities are imported from ``first_strike_quality_campaign``
(and, through it, ``first_strike_campaign``) rather than re-declared here, so
a future change to the campaign contract cannot silently drift out of step
with this tooling -- a second source of truth that can drift is a defect.

Row order is never semantically meaningful, but serialized TSV bytes MUST be
deterministic: rows are always written sorted by ``(arm order, training
seed)`` (the order of ``LABELS`` x ascending seed) regardless of the order
supplied by the caller.

Every ``validate_*`` function raises ``ValueError`` -- never warns, never
drops a row, never accepts a missing field -- on the first contract
violation it finds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from evaluation.analysis import first_strike_campaign as legacy
from evaluation.analysis import first_strike_quality_campaign as quality


CAMPAIGN_NAME = quality.CAMPAIGN_NAME
LABELS = quality.LABELS
SHORT = quality.SHORT
TASKS = quality.TASKS
SEEDS = frozenset(quality.SEEDS)
SENTINELS = quality.SENTINELS
STRICT_CONFIG_IDENTITY_KEYS = quality.STRICT_CONFIG_IDENTITY_KEYS
EVALUATOR_RNG = legacy.CAMPAIGN_EVALUATOR_RNG
EXPECTED_NUM_ENVS = legacy.EXPECTED_NUM_ENVS
EXPECTED_EPISODES_PER_ENV = legacy.EXPECTED_EPISODES_PER_ENV
EXPECTED_N_EPISODES = legacy.EXPECTED_EPISODES_PER_SEED

_RETRY_STATUSES = frozenset({"accepted", "infra_failure", "rejected"})
EXPECTED_EVALUATION_ATTEMPT = "attempt2"
EXPECTED_EVALUATION_RETRY_HISTORY = "attempt1:infra_failure;attempt2:accepted"


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


# Derived (not re-declared) frozen expectations: each is a pure function of
# constants already frozen in ``first_strike_quality_campaign`` /
# ``first_strike_campaign``.
EXPECTED_READER_SHA256 = {
    label: _digest(
        {
            "impact_reader": quality.IMPACT_READERS[label],
            "delivered_reader": quality.DELIVERED_READER,
        }
    )
    for label in LABELS
}
EXPECTED_NORMALIZER_SHA256 = {
    label: _digest({"speed_normalizer_m_s": quality.SPEED_NORMALIZER[label]})
    for label in LABELS
}
EXPECTED_TREATMENT_CONFIG_SHA256 = {
    label: _digest(
        {
            "campaign": CAMPAIGN_NAME,
            "arm": label,
            "short": SHORT[label],
            "task": TASKS[label],
        }
    )
    for label in LABELS
}
EXPECTED_CAP_SIGNATURE_SHA256 = _digest(
    {"impulse_limits_n_m_s": list(quality.IMPULSE_LIMITS)}
)


TRAINING_FIELDS = (
    "campaign",
    "disposition",
    "arm",
    "short",
    "task",
    "training_seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "training_attempt",
    "retry_history",
    "code_revision",
    "asset_revision",
    "campaign_config_sha256",
    "treatment_config_sha256",
    "reader_sha256",
    "normalizer_sha256",
    "treatment_reward_sha256",
    "fixed_action_signature_sha256",
    "fixed_impedance_signature_sha256",
    "cap_signature_sha256",
    "clean_state",
)

# Identity fields shared verbatim by both manifests -- present so an
# evaluation row can be cross-verified against its training row without a
# second lookup table.
_IDENTITY_FIELDS = (
    "campaign",
    "disposition",
    "arm",
    "short",
    "task",
    "training_seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "code_revision",
    "asset_revision",
    "campaign_config_sha256",
    "treatment_config_sha256",
    "fixed_action_signature_sha256",
    "fixed_impedance_signature_sha256",
    "cap_signature_sha256",
    "clean_state",
)

EVALUATION_FIELDS = (
    _IDENTITY_FIELDS
    + (
        "accepted_training_manifest_sha256",
        "evaluation_attempt",
        "evaluation_retry_history",
    )
    + STRICT_CONFIG_IDENTITY_KEYS
    + (
        "reset_rng_seed",
        "observation_rng_seed",
        "action_rng_seed",
        "sampled_trace_digest",
        "sampled_trace_artifact_sha256",
        "num_envs",
        "episodes_per_env_sampled",
        "n_episodes_sampled",
    )
    + SENTINELS
)

_BOOL_FIELDS = frozenset({"clean_state"})
_TRAINING_INT_FIELDS = frozenset({"training_seed"})
_EVALUATION_INT_FIELDS = frozenset(
    {
        "training_seed",
        "reset_rng_seed",
        "observation_rng_seed",
        "action_rng_seed",
        "num_envs",
        "episodes_per_env_sampled",
        "n_episodes_sampled",
    }
) | set(SENTINELS)

_INTEGER_TEXT = re.compile(r"^(0|[1-9][0-9]*)$")


def _exact_int(value, *, label: str, prefix: str) -> int:
    """Parse a nonnegative integer without lossy coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{prefix}: {label} must be an integer, got a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(
                f"{prefix}: {label} must be a nonnegative integer, got {value}"
            )
        return value
    if isinstance(value, str) and _INTEGER_TEXT.match(value.strip()):
        return int(value.strip())
    raise ValueError(f"{prefix}: {label} is not an exact integer ({value!r})")


# ---------------------------------------------------------------------------
# TSV formatting (pure, no validation beyond structural completeness)
# ---------------------------------------------------------------------------


def _sort_key(row: Mapping) -> tuple[int, int]:
    arm = row.get("arm")
    if arm not in LABELS:
        raise ValueError(f"unknown arm: {arm!r}")
    seed = _exact_int(
        row["training_seed"], label="training_seed", prefix="manifest serialization"
    )
    return (LABELS.index(arm), seed)


def _format_cell(
    field: str, value: object, *, bool_fields: frozenset, int_fields: frozenset
) -> str:
    if field in bool_fields:
        text = "true" if legacy._parse_bool(value) else "false"
    elif field in int_fields:
        text = str(
            _exact_int(value, label=field, prefix="manifest serialization")
        )
    else:
        text = "" if value is None else str(value)
    if "\t" in text or "\n" in text:
        raise ValueError(f"{field}: value must not contain a tab or newline: {value!r}")
    return text


def _serialize(
    rows: Sequence[Mapping],
    *,
    fields: tuple[str, ...],
    bool_fields: frozenset,
    int_fields: frozenset,
) -> str:
    rows = list(rows)
    for row in rows:
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError(f"row is missing field(s): {missing}")
    ordered = sorted(rows, key=_sort_key)
    lines = ["\t".join(fields)]
    for row in ordered:
        lines.append(
            "\t".join(
                _format_cell(field, row[field], bool_fields=bool_fields, int_fields=int_fields)
                for field in fields
            )
        )
    return "\n".join(lines) + "\n"


def _parse(
    text: str,
    *,
    fields: tuple[str, ...],
    bool_fields: frozenset,
    int_fields: frozenset,
) -> list[dict]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        raise ValueError("manifest text is empty")
    header = tuple(lines[0].split("\t"))
    if header != fields:
        raise ValueError("manifest header does not match the frozen field contract")
    rows: list[dict] = []
    for line_number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(fields):
            raise ValueError(
                f"line {line_number}: expected {len(fields)} columns, found {len(cells)}"
            )
        row: dict = {}
        for field, cell in zip(fields, cells):
            if field in bool_fields:
                row[field] = legacy._parse_bool(cell)
            elif field in int_fields:
                row[field] = _exact_int(
                    cell, label=field, prefix=f"line {line_number}"
                )
            else:
                row[field] = cell
        rows.append(row)
    return rows


def serialize_training_manifest(rows: Sequence[Mapping]) -> str:
    """Serialize training-checkpoint rows to deterministic TSV text.

    Output is sorted by ``(arm order, training_seed)`` regardless of the
    input order, so byte-identical rows always produce byte-identical text.
    """

    return _serialize(
        rows, fields=TRAINING_FIELDS, bool_fields=_BOOL_FIELDS, int_fields=_TRAINING_INT_FIELDS
    )


def parse_training_manifest(text: str) -> list[dict]:
    """Parse TSV text produced by :func:`serialize_training_manifest`."""

    return _parse(
        text, fields=TRAINING_FIELDS, bool_fields=_BOOL_FIELDS, int_fields=_TRAINING_INT_FIELDS
    )


def serialize_evaluation_manifest(rows: Sequence[Mapping]) -> str:
    """Serialize evaluation rows to deterministic TSV text.

    Output is sorted by ``(arm order, training_seed)`` regardless of the
    input order, so byte-identical rows always produce byte-identical text.
    """

    return _serialize(
        rows,
        fields=EVALUATION_FIELDS,
        bool_fields=_BOOL_FIELDS,
        int_fields=_EVALUATION_INT_FIELDS,
    )


def parse_evaluation_manifest(text: str) -> list[dict]:
    """Parse TSV text produced by :func:`serialize_evaluation_manifest`."""

    return _parse(
        text,
        fields=EVALUATION_FIELDS,
        bool_fields=_BOOL_FIELDS,
        int_fields=_EVALUATION_INT_FIELDS,
    )


# ---------------------------------------------------------------------------
# Validation (fail-closed: every violation raises ValueError)
# ---------------------------------------------------------------------------


def _validate_retry_history(value: object, *, accepted_attempt: object, prefix: str) -> None:
    text = str(value)
    if not text.strip():
        raise ValueError(f"{prefix}: retry history is missing")
    if "\t" in text or "\n" in text:
        raise ValueError(f"{prefix}: retry history must not contain a tab or newline")
    parsed: list[tuple[str, str]] = []
    for entry in text.split(";"):
        parts = entry.split(":")
        if len(parts) != 2 or not parts[0].strip() or parts[1].strip() not in _RETRY_STATUSES:
            raise ValueError(f"{prefix}: retry history entry {entry!r} is malformed")
        parsed.append((parts[0].strip(), parts[1].strip()))
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{prefix}: retry history has duplicate attempt labels")
    if parsed[-1] != (str(accepted_attempt).strip(), "accepted"):
        raise ValueError(f"{prefix}: retry history must end with the accepted attempt")


def _validate_core_identity(
    rows: Sequence[Mapping], *, context: str
) -> dict[tuple[str, int], Mapping]:
    """Structural arm/short/task/seed checks shared by both manifests.

    Fails closed on: unknown arm, short<->arm mismatch, task<->arm mismatch,
    non-integer/out-of-range seed, duplicate (arm, seed), and any arm not
    holding exactly the frozen seeds 8..15.
    """

    identity: dict[tuple[str, int], Mapping] = {}
    for row in rows:
        arm = row.get("arm")
        if arm not in LABELS:
            raise ValueError(f"{context}: unknown arm {arm!r}")
        seed = _exact_int(
            row.get("training_seed"),
            label="training_seed",
            prefix=f"{context}: {arm}",
        )
        if seed not in SEEDS:
            raise ValueError(f"{context}: {arm}/seed{seed}: training_seed must be 8..15")
        key = (arm, seed)
        if key in identity:
            raise ValueError(f"{context}: duplicate row for {arm}/seed{seed}")
        if row.get("short") != SHORT[arm]:
            raise ValueError(
                f"{context}: {arm}/seed{seed}: short {row.get('short')!r} does not match arm"
            )
        if row.get("task") != TASKS[arm]:
            raise ValueError(
                f"{context}: {arm}/seed{seed}: task {row.get('task')!r} does not match arm"
            )
        identity[key] = row
    for label in LABELS:
        seeds = sorted(seed for (arm, seed) in identity if arm == label)
        if seeds != sorted(SEEDS):
            raise ValueError(
                f"{context}: {label} must have exactly seeds 8..15, found {seeds}"
            )
    return identity


def validate_training_manifest(rows: Sequence[Mapping]) -> None:
    """Fail closed on any drift from the frozen 32-row training contract."""

    rows = list(rows)
    if len(rows) != 32:
        raise ValueError(f"training manifest must contain exactly 32 rows, found {len(rows)}")
    for row in rows:
        missing = [field for field in TRAINING_FIELDS if field not in row]
        if missing:
            raise ValueError(f"training row is missing field(s): {missing}")

    identity = _validate_core_identity(rows, context="training manifest")

    code_revisions: set[str] = set()
    asset_revisions: set[str] = set()
    campaign_config_hashes: set[str] = set()
    checkpoint_paths: set[str] = set()
    checkpoint_hashes: set[str] = set()
    for (arm, seed), row in identity.items():
        prefix = f"training manifest: {arm}/seed{seed}"
        if row["campaign"] != CAMPAIGN_NAME:
            raise ValueError(f"{prefix}: campaign must be {CAMPAIGN_NAME!r}")
        if row["disposition"] != "accepted":
            raise ValueError(f"{prefix}: disposition must be 'accepted'")
        checkpoint_path = str(row["checkpoint_path"])
        if not checkpoint_path.strip():
            raise ValueError(f"{prefix}: checkpoint_path is missing")
        if Path(checkpoint_path).name != "model_499.pt":
            raise ValueError(
                f"{prefix}: checkpoint_path must be a model_499.pt checkpoint, "
                f"got {checkpoint_path!r}"
            )
        if checkpoint_path in checkpoint_paths:
            raise ValueError(f"{prefix}: checkpoint_path is duplicated across the manifest")
        checkpoint_paths.add(checkpoint_path)
        if not legacy._is_hex_digest(row["checkpoint_sha256"], 64):
            raise ValueError(f"{prefix}: checkpoint_sha256 is missing or malformed")
        checkpoint_hash = str(row["checkpoint_sha256"]).lower()
        if checkpoint_hash in checkpoint_hashes:
            raise ValueError(f"{prefix}: checkpoint_sha256 is duplicated across the manifest")
        checkpoint_hashes.add(checkpoint_hash)
        if not str(row["training_attempt"]).strip():
            raise ValueError(f"{prefix}: training_attempt is missing")
        _validate_retry_history(
            row["retry_history"], accepted_attempt=row["training_attempt"], prefix=prefix
        )
        for key in ("code_revision", "asset_revision"):
            if not legacy._is_hex_digest(row[key], 40):
                raise ValueError(f"{prefix}: {key} must be a clean 40-hex revision")
        code_revisions.add(row["code_revision"])
        asset_revisions.add(row["asset_revision"])
        if not legacy._is_hex_digest(row["campaign_config_sha256"], 64):
            raise ValueError(f"{prefix}: campaign_config_sha256 is missing or malformed")
        campaign_config_hashes.add(row["campaign_config_sha256"])
        if row["treatment_config_sha256"] != EXPECTED_TREATMENT_CONFIG_SHA256[arm]:
            raise ValueError(
                f"{prefix}: treatment_config_sha256 does not match the frozen treatment identity"
            )
        if row["reader_sha256"] != EXPECTED_READER_SHA256[arm]:
            raise ValueError(f"{prefix}: reader_sha256 does not match the frozen reader identity")
        if row["normalizer_sha256"] != EXPECTED_NORMALIZER_SHA256[arm]:
            raise ValueError(
                f"{prefix}: normalizer_sha256 does not match the frozen normalizer identity"
            )
        if row["treatment_reward_sha256"] != quality.EXPECTED_TREATMENT_REWARD_SHA256[arm]:
            raise ValueError(
                f"{prefix}: treatment_reward_sha256 does not match the frozen expectation"
            )
        if row["fixed_action_signature_sha256"] != legacy.EXPECTED_FIXED_ACTION_SIGNATURE_SHA256:
            raise ValueError(f"{prefix}: fixed_action_signature_sha256 mismatch")
        if (
            row["fixed_impedance_signature_sha256"]
            != legacy.EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256
        ):
            raise ValueError(f"{prefix}: fixed_impedance_signature_sha256 mismatch")
        if row["cap_signature_sha256"] != EXPECTED_CAP_SIGNATURE_SHA256:
            raise ValueError(f"{prefix}: cap_signature_sha256 mismatch")
        try:
            clean = legacy._parse_bool(row["clean_state"])
        except ValueError as error:
            raise ValueError(f"{prefix}: clean_state is not a valid boolean") from error
        if not clean:
            raise ValueError(f"{prefix}: clean_state must be true")

    if len(code_revisions) != 1:
        raise ValueError("training manifest: code_revision must be identical across all 32 rows")
    if len(asset_revisions) != 1:
        raise ValueError("training manifest: asset_revision must be identical across all 32 rows")
    if len(campaign_config_hashes) != 1:
        raise ValueError(
            "training manifest: campaign_config_sha256 must be identical across all 32 rows"
        )


def validate_evaluation_manifest(
    rows: Sequence[Mapping],
    training_rows: Sequence[Mapping],
    training_manifest_sha256: str,
) -> None:
    """Fail closed on any drift from the frozen 32-row evaluation contract.

    Cross-verifies every row against ``training_rows`` (validated with
    :func:`validate_training_manifest` as part of this call) and against the
    supplied ``training_manifest_sha256``.
    """

    training_rows = list(training_rows)
    validate_training_manifest(training_rows)
    if not legacy._is_hex_digest(training_manifest_sha256, 64):
        raise ValueError("training_manifest_sha256 must be a 64-hex SHA-256 digest")
    training_identity = _validate_core_identity(training_rows, context="training manifest")

    rows = list(rows)
    if len(rows) != 32:
        raise ValueError(f"evaluation manifest must contain exactly 32 rows, found {len(rows)}")
    for row in rows:
        missing = [field for field in EVALUATION_FIELDS if field not in row]
        if missing:
            raise ValueError(f"evaluation row is missing field(s): {missing}")

    identity = _validate_core_identity(rows, context="evaluation manifest")

    for (arm, seed), row in identity.items():
        prefix = f"evaluation manifest: {arm}/seed{seed}"
        training_row = training_identity[(arm, seed)]

        if row["campaign"] != CAMPAIGN_NAME:
            raise ValueError(f"{prefix}: campaign must be {CAMPAIGN_NAME!r}")
        if row["disposition"] != "accepted":
            raise ValueError(f"{prefix}: disposition must be 'accepted'")

        for key in (
            "checkpoint_path",
            "checkpoint_sha256",
            "code_revision",
            "asset_revision",
            "campaign_config_sha256",
            "treatment_config_sha256",
            "fixed_action_signature_sha256",
            "fixed_impedance_signature_sha256",
            "cap_signature_sha256",
        ):
            if row[key] != training_row[key]:
                raise ValueError(
                    f"{prefix}: {key} does not match the checkpoint bound in the training "
                    "manifest"
                )

        try:
            clean = legacy._parse_bool(row["clean_state"])
        except ValueError as error:
            raise ValueError(f"{prefix}: clean_state is not a valid boolean") from error
        if not clean:
            raise ValueError(f"{prefix}: clean_state must be true")

        if (
            not legacy._is_hex_digest(row["accepted_training_manifest_sha256"], 64)
            or row["accepted_training_manifest_sha256"] != training_manifest_sha256
        ):
            raise ValueError(
                f"{prefix}: accepted_training_manifest_sha256 does not match the supplied "
                "training manifest"
            )

        if not str(row["evaluation_attempt"]).strip():
            raise ValueError(f"{prefix}: evaluation_attempt is missing")
        _validate_retry_history(
            row["evaluation_retry_history"],
            accepted_attempt=row["evaluation_attempt"],
            prefix=prefix,
        )

        for key in STRICT_CONFIG_IDENTITY_KEYS:
            if not legacy._is_hex_digest(row[key], 64):
                raise ValueError(f"{prefix}: {key} is missing or malformed")
        if row["training_policy_observation_sha256"] != row["evaluation_policy_observation_sha256"]:
            raise ValueError(f"{prefix}: training/evaluation policy-observation identity mismatch")
        if row["training_treatment_reward_sha256"] != row["evaluation_treatment_reward_sha256"]:
            raise ValueError(f"{prefix}: training/evaluation treatment-reward identity mismatch")
        if row["training_treatment_reward_sha256"] != quality.EXPECTED_TREATMENT_REWARD_SHA256[arm]:
            raise ValueError(
                f"{prefix}: treatment_reward_sha256 does not match the frozen expectation"
            )
        if row["training_treatment_reward_sha256"] != training_row["treatment_reward_sha256"]:
            raise ValueError(
                f"{prefix}: evaluation treatment-reward identity does not match the training "
                "manifest"
            )

        for key, expected in (
            ("reset_rng_seed", EVALUATOR_RNG["reset"]),
            ("observation_rng_seed", EVALUATOR_RNG["observation"]),
            ("action_rng_seed", EVALUATOR_RNG["action"]),
        ):
            if _exact_int(row[key], label=key, prefix=prefix) != expected:
                raise ValueError(f"{prefix}: {key} must be the frozen value {expected}")

        for key in ("sampled_trace_digest", "sampled_trace_artifact_sha256"):
            if not legacy._is_hex_digest(row[key], 64):
                raise ValueError(f"{prefix}: {key} is missing or malformed")

        if (
            _exact_int(row["num_envs"], label="num_envs", prefix=prefix)
            != EXPECTED_NUM_ENVS
        ):
            raise ValueError(f"{prefix}: num_envs must be {EXPECTED_NUM_ENVS}")
        if (
            _exact_int(
                row["episodes_per_env_sampled"],
                label="episodes_per_env_sampled",
                prefix=prefix,
            )
            != EXPECTED_EPISODES_PER_ENV
        ):
            raise ValueError(
                f"{prefix}: episodes_per_env_sampled must be {EXPECTED_EPISODES_PER_ENV}"
            )
        if (
            _exact_int(
                row["n_episodes_sampled"],
                label="n_episodes_sampled",
                prefix=prefix,
            )
            != EXPECTED_N_EPISODES
        ):
            raise ValueError(
                f"{prefix}: n_episodes_sampled must be exactly {EXPECTED_N_EPISODES}"
            )

        for key in SENTINELS:
            if _exact_int(row[key], label=key, prefix=prefix) != 0:
                raise ValueError(f"{prefix}: sentinel {key} is nonzero")


# ---------------------------------------------------------------------------
# build-training-manifest: scan real training run directories and build the
# 21-field TRAINING_FIELDS rows validate_training_manifest expects.  Fails
# closed on every drift the caller could hand us; never returns a manifest
# validate_training_manifest itself would reject.
# ---------------------------------------------------------------------------

# Verified cluster layout: <log_root>/<TIMESTAMP>_fq4x8_<short>_seed<N>/model_499.pt
_RUN_DIR_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_fq4x8_(?P<short>[a-z0-9]+)_seed(?P<seed>\d+)$"
)
_LABEL_BY_SHORT = {short: label for label, short in SHORT.items()}


def default_config_identity(arm: str, task: str) -> tuple[str, str]:
    """Derive ``(campaign_config_sha256, treatment_config_sha256)`` from the
    LIVE registered env cfg for ``task`` -- never hand-written.

    Reuses ``scripts.eval_impulse._validate_sampled_env_contract``, which
    fails closed on any drift from the frozen fixed-impedance/action
    contract, then asserts the maximize-reward weights it actually observed
    there match this arm's frozen ``WEIGHTS`` before trusting the frozen
    treatment-config digest. (``contract["treatment"]`` is not used for this
    cross-check: F8 shares its exact registered task with the legacy "F"
    arm, so the shared helper's own arm-label lookup resolves to "F", not
    "F8".)

    The weights are NOT sufficient on their own: **D0 and FQ-min both have
    weights (8, 0)**, so a weights-only check is blind between them and would
    happily label an FQ-min checkpoint as D0 -- corrupting the very contrast
    this campaign exists to measure. ``task`` is therefore pinned to the arm's
    frozen registered task first; the weights check then guards the remaining
    pairs (F8/F0/D0), whose weights genuinely differ.

    mjlab/torch are imported lazily so importing this module never requires
    them -- only calling this function (the production default) does.
    """

    from mjlab.tasks.registry import load_env_cfg

    from scripts.eval_impulse import _validate_sampled_env_contract

    expected_task = TASKS.get(arm)
    if expected_task is None:
        raise ValueError(f"arm/task: unknown arm {arm!r}")
    if task != expected_task:
        raise ValueError(
            f"arm/task mismatch: arm {arm!r} is frozen to task "
            f"{expected_task!r}, got {task!r}"
        )

    env_cfg = load_env_cfg(task, play=False)
    contract = _validate_sampled_env_contract(env_cfg, task)
    observed_weights = (contract["impact_weight"], contract["delivered_weight"])
    if observed_weights != quality.WEIGHTS[arm]:
        raise ValueError(
            f"{arm}: live cfg maximize weights {observed_weights} do not match "
            f"the frozen weights {quality.WEIGHTS[arm]} for task {task!r}"
        )
    campaign_config_sha256 = _digest(
        {
            "schema_version": 1,
            "physics_dt_s": contract["physics_dt_s"],
            "control_decimation": contract["control_decimation"],
            "impulse_limits_n_m_s": contract["impulse_limits_n_m_s"],
            "fixed_impedance_signature_sha256": contract["fixed_impedance_signature_sha256"],
            "fixed_action_signature_sha256": contract["fixed_action_signature_sha256"],
        }
    )
    return campaign_config_sha256, EXPECTED_TREATMENT_CONFIG_SHA256[arm]


def build_training_manifest(
    log_root: str | Path,
    *,
    code_revision: str,
    asset_revision: str,
    attempt: str = "attempt1",
    config_identity: Callable[[str, str], tuple[str, str]] | None = None,
) -> list[dict]:
    """Scan ``log_root`` for the 32 accepted fq4x8 run directories and build
    validated TRAINING_FIELDS rows.

    Fails closed (raises ``ValueError``) on any missing/duplicate/extra
    identity, out-of-range seed, unparseable run-dir name, missing/zero-
    length/unreadable checkpoint, dirty code/asset revision, or a derived
    config identity that does not match the frozen expectation for its
    treatment.  Never returns a partial manifest: the returned rows have
    already passed :func:`validate_training_manifest`.
    """

    if not legacy._is_hex_digest(code_revision, 40):
        raise ValueError("code_revision must be a clean 40-hex revision")
    if not legacy._is_hex_digest(asset_revision, 40):
        raise ValueError("asset_revision must be a clean 40-hex revision")
    if not str(attempt).strip():
        raise ValueError("attempt must not be empty")
    identity_fn = config_identity if config_identity is not None else default_config_identity

    root = Path(log_root)
    if not root.is_dir():
        raise ValueError(f"log root is not a directory: {root}")

    found: dict[tuple[str, int], Path] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = _RUN_DIR_PATTERN.match(entry.name)
        if match is None:
            raise ValueError(f"run directory name does not parse: {entry.name}")
        short = match.group("short")
        if short not in _LABEL_BY_SHORT:
            raise ValueError(f"run directory has unknown short {short!r}: {entry.name}")
        arm = _LABEL_BY_SHORT[short]
        seed = int(match.group("seed"))
        if seed not in SEEDS:
            raise ValueError(f"{entry.name}: training_seed must be 8..15, got {seed}")
        key = (arm, seed)
        if key in found:
            raise ValueError(
                f"duplicate run directory for {arm}/seed{seed}: "
                f"{found[key].name} and {entry.name}"
            )
        found[key] = entry

    expected_keys = {(arm, seed) for arm in LABELS for seed in SEEDS}
    missing = sorted(expected_keys - set(found))
    if missing:
        raise ValueError(f"missing run directories for: {missing}")
    extra = sorted(set(found) - expected_keys)
    if extra:
        raise ValueError(f"unexpected run directories for: {extra}")

    identity_cache: dict[str, tuple[str, str]] = {}
    rows: list[dict] = []
    for arm in LABELS:
        for seed in sorted(SEEDS):
            run_dir = found[(arm, seed)]
            prefix = f"{arm}/seed{seed}"
            checkpoint_path = run_dir / "model_499.pt"
            if not checkpoint_path.is_file():
                raise ValueError(f"{prefix}: model_499.pt is missing in {run_dir}")
            if checkpoint_path.stat().st_size <= 0:
                raise ValueError(f"{prefix}: checkpoint is zero-length: {checkpoint_path}")
            try:
                checkpoint_sha256 = legacy._sha256(checkpoint_path)
            except OSError as error:
                raise ValueError(f"{prefix}: checkpoint is unreadable: {error}") from error

            task = TASKS[arm]
            if arm not in identity_cache:
                identity_cache[arm] = identity_fn(arm, task)
            campaign_config_sha256, treatment_config_sha256 = identity_cache[arm]

            rows.append(
                {
                    "campaign": CAMPAIGN_NAME,
                    "disposition": "accepted",
                    "arm": arm,
                    "short": SHORT[arm],
                    "task": task,
                    "training_seed": seed,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "training_attempt": str(attempt),
                    "retry_history": f"{attempt}:accepted",
                    "code_revision": code_revision,
                    "asset_revision": asset_revision,
                    "campaign_config_sha256": campaign_config_sha256,
                    "treatment_config_sha256": treatment_config_sha256,
                    "reader_sha256": EXPECTED_READER_SHA256[arm],
                    "normalizer_sha256": EXPECTED_NORMALIZER_SHA256[arm],
                    "treatment_reward_sha256": quality.EXPECTED_TREATMENT_REWARD_SHA256[arm],
                    "fixed_action_signature_sha256": legacy.EXPECTED_FIXED_ACTION_SIGNATURE_SHA256,
                    "fixed_impedance_signature_sha256": (
                        legacy.EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256
                    ),
                    "cap_signature_sha256": EXPECTED_CAP_SIGNATURE_SHA256,
                    "clean_state": True,
                }
            )

    validate_training_manifest(rows)
    return rows


# ---------------------------------------------------------------------------
# build-evaluation-manifest: turn ONE immutable, already-verified evaluation
# attempt into the 39-field accepted_evaluations.tsv.
#
# ACCEPTANCE IS A PROVENANCE + VALIDITY JUDGEMENT, NEVER A PERFORMANCE ONE.
# No outcome is emitted or used for selection; only the frozen exact-zero and
# numerical-validity predicates are evaluated. A checkpoint that never once
# drives the nail is ACCEPTED as long as its provenance binds and its
# instrumentation was alive -- otherwise this builder would silently become a
# post-outcome seed-selection mechanism and destroy the paired 4x8 design. See
# ``tests/test_fq4x8_evaluation_builder.py::test_weak_but_valid_checkpoint_is_accepted``.
#
# Digest strategy: ``payload_digest`` is the canonical digest of the WHOLE
# schema-v3 payload, which contains every episode's ``trace_digest`` and
# ``reset_state_digest``.  Recomputing it here therefore cryptographically
# binds all 512 per-episode digests per checkpoint without re-walking them.
# That those banked digests correctly describe the physics is established
# separately, by the read-only verification battery run against the immutable
# attempt before this builder is ever invoked.
# ---------------------------------------------------------------------------

_SUMMARY_NAME = "summary.csv"


def _decode_sampled_artifact(path: Path, *, prefix: str) -> dict:
    """Decode one schema-v3 sampled-trace artifact.  No pickle, ever."""
    try:
        with np.load(path, allow_pickle=False) as handle:
            if "payload_json" not in handle:
                raise ValueError(f"{prefix}: artifact has no payload_json member")
            encoded = handle["payload_json"]
            if encoded.dtype != np.uint8 or encoded.ndim != 1:
                raise ValueError(f"{prefix}: payload_json must be 1-D uint8")
            text = encoded.tobytes().decode("utf-8")
    except OSError as error:
        raise ValueError(f"{prefix}: artifact is unreadable: {error}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"{prefix}: payload_json is not valid UTF-8") from error
    return json.loads(text)


def _finite_json_number(
    value: object, *, label: str, prefix: str, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {label} must be a finite JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{prefix}: {label} must be a finite JSON number")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"{prefix}: {label} must be nonnegative")
    return parsed


def _finite_json_array(value: object, *, label: str, prefix: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{prefix}: {label} is malformed") from error
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{prefix}: {label} must contain only JSON numbers")
    parsed = raw.astype(float, copy=False)
    if not np.isfinite(parsed).all():
        raise ValueError(f"{prefix}: {label} contains nonfinite values")
    return parsed


def _episode_invariant_inputs(
    episode: Mapping,
    *,
    index: int,
    prefix: str,
    physical_sample_count: int | None,
) -> tuple[bool, float, float]:
    """Validate general numeric channels and return exact-zero inputs.

    The final two predicates below intentionally mirror the evaluator's
    ``_invariant_violations`` implementation: no epsilon or tunable threshold
    is introduced because the accumulators are exact zero absent contact.
    """

    episode_prefix = f"{prefix}: episode {index}"
    _finite_json_number(
        episode.get("episode_depth_m"),
        label="episode_depth_m",
        prefix=episode_prefix,
    )
    peak_lambda = _finite_json_array(
        episode.get("episode_peak_lambda"),
        label="episode_peak_lambda",
        prefix=episode_prefix,
    )
    if peak_lambda.shape != (6,) or np.any(peak_lambda < 0.0):
        raise ValueError(
            f"{episode_prefix}: episode_peak_lambda must be six nonnegative values"
        )
    delivered = _finite_json_number(
        episode.get("episode_delivered_accumulator_n_s"),
        label="episode_delivered_accumulator_n_s",
        prefix=episode_prefix,
        nonnegative=True,
    )
    success = episode.get("overall_success")
    if type(success) is not bool:
        raise ValueError(
            f"{episode_prefix}: overall_success must be a JSON boolean"
        )

    # qvel is a separate general numerical-validity gate, not part of
    # quality_nonfinite_n. If liveness already proved the physical clock
    # malformed, that sentinel is the clearer failure and qvel shape cannot be
    # interpreted against an absent clock.
    if physical_sample_count is not None:
        physical = episode.get("physical")
        if not isinstance(physical, Mapping):
            raise ValueError(f"{episode_prefix}: physical channels are missing")
        qvel = _finite_json_array(
            physical.get("joint_speed_rad_s"),
            label="joint_speed_rad_s",
            prefix=episode_prefix,
        )
        qvel_post = _finite_json_array(
            physical.get("post_step_joint_speed_rad_s"),
            label="post_step_joint_speed_rad_s",
            prefix=episode_prefix,
        )
        if (
            qvel.ndim != 2
            or qvel.shape[0] != physical_sample_count
            or qvel.shape[1] < 1
            or qvel_post.shape != qvel.shape
        ):
            raise ValueError(
                f"{episode_prefix}: qvel series must match the physical substeps"
            )

    return success, delivered, float(np.max(peak_lambda))


def _derive_validity_sentinels(
    episodes: Sequence[Mapping],
    *,
    nail_geometry: Mapping,
    prefix: str,
) -> dict:
    """Derive all six sentinels from digest-bound episode evidence."""

    overflow = nonfinite = liveness = 0
    impossible_success = lambda_dead = 0
    per_env: Counter[tuple[int, int]] = Counter()
    quota_ok = len(episodes) == EXPECTED_N_EPISODES

    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"{prefix}: episode {index} is not a mapping")
        event = episode.get("event_trace")
        physical = episode.get("physical")
        stream = (
            event.get("tracker_contact_quality")
            if isinstance(event, Mapping)
            else None
        )
        substeps = (
            physical.get("contact")
            if isinstance(physical, Mapping)
            else None
        )
        live = (
            isinstance(stream, list)
            and bool(stream)
            and isinstance(substeps, list)
            and bool(substeps)
            and len(stream) == len(substeps)
        )
        if not live:
            liveness += 1

        physical_sample_count = (
            len(substeps)
            if isinstance(substeps, list) and substeps
            else None
        )
        episode_nonfinite = False
        snapshot_valid = True
        try:
            quality.validate_first_contact_snapshot(
                episode,
                physical_sample_count=physical_sample_count,
            )
        except ValueError:
            episode_nonfinite = True
            snapshot_valid = False
        raw_quality_valid = True
        raw_quality_sample_count: int | None = None
        try:
            raw_quality_sample_count = quality.validate_first_contact_raw_quality(
                episode, physical_sample_count=physical_sample_count
            )
        except ValueError:
            episode_nonfinite = True
            raw_quality_valid = False
        overflow_evidence_valid = snapshot_valid and raw_quality_valid
        overflowed = False
        overflow_sample_count = (
            physical_sample_count
            if physical_sample_count is not None
            else raw_quality_sample_count
        )
        if overflow_evidence_valid and overflow_sample_count is not None:
            try:
                overflowed = quality.first_contact_overflow_at_onset(
                    episode, physical_sample_count=overflow_sample_count
                )
            except ValueError:
                episode_nonfinite = True
                overflow_evidence_valid = False
            else:
                if overflowed:
                    overflow += 1

        success, delivered, lam_worst = _episode_invariant_inputs(
            episode,
            index=index,
            prefix=prefix,
            physical_sample_count=physical_sample_count,
        )
        # Exact evaluator predicates; these are exact-zero checks because the
        # validated inputs above are finite and nonnegative.
        if success and (delivered <= 0.0 or lam_worst <= 0.0):
            impossible_success += 1
        if delivered > 0.0 and lam_worst <= 0.0:
            lambda_dead += 1

        if (
            live
            and snapshot_valid
            and raw_quality_valid
            and overflow_evidence_valid
        ):
            try:
                quality.validate_first_contact_quality(
                    episode, nail_geometry=nail_geometry
                )
            except quality.QualityOverflowError:
                pass
            except ValueError:
                episode_nonfinite = True
        if episode_nonfinite:
            nonfinite += 1

        try:
            coordinate = (
                _exact_int(
                    episode["env_id"],
                    label="env_id",
                    prefix=f"{prefix}: episode {index}",
                ),
                _exact_int(
                    episode["episode_ordinal"],
                    label="episode_ordinal",
                    prefix=f"{prefix}: episode {index}",
                ),
            )
        except (KeyError, ValueError):
            quota_ok = False
        else:
            per_env[coordinate] += 1

    expected_population = Counter(
        {
            (env_id, ordinal): 1
            for env_id in range(EXPECTED_NUM_ENVS)
            for ordinal in range(EXPECTED_EPISODES_PER_ENV)
        }
    )
    quota_ok = quota_ok and per_env == expected_population
    return {
        "quality_overflow_n": overflow,
        "quality_nonfinite_n": nonfinite,
        "liveness_failure_n": liveness,
        "impossible_success_n": impossible_success,
        "lambda_dead_n": lambda_dead,
        # Only zero/nonzero is meaningful for this population predicate.
        "quota_failure_n": 0 if quota_ok else 1,
    }


def _require_clean(row: Mapping, key: str, *, prefix: str) -> None:
    """Require an explicitly-clean provenance flag.

    Fails closed on a MISSING column and on a BLANK value: "we did not record
    whether this was dirty" is not evidence of cleanliness, and the
    preregistration treats unknown provenance as campaign-invalidating exactly
    like dirty provenance.
    """
    if key not in row:
        raise ValueError(f"{prefix}: provenance column {key} is absent")
    value = str(row[key]).strip().lower()
    if value == "":
        raise ValueError(f"{prefix}: provenance column {key} is blank")
    if value not in ("false", "0", "no"):
        raise ValueError(f"{prefix}: {key} is dirty ({row[key]!r})")


def build_evaluation_manifest(
    attempt_dir: str | Path,
    *,
    training_rows: Sequence[Mapping],
    training_manifest_sha256: str,
    evaluation_attempt: str,
    evaluation_retry_history: str,
    expected_evaluation_code_revision: str,
) -> list[dict]:
    """Build the validated 32-row evaluation manifest for one attempt.

    Fails closed on any missing/duplicate/extra identity, any drift from the
    frozen training manifest, any provenance or RNG mismatch, any dirty
    revision, any artifact whose SHA-256 or payload digest does not recompute,
    any non-schema-v3 payload, and any nonzero sentinel.  Never returns a
    partial manifest: the rows returned have already passed
    :func:`validate_evaluation_manifest`.
    """

    training_rows = list(training_rows)
    validate_training_manifest(training_rows)
    if not legacy._is_hex_digest(training_manifest_sha256, 64):
        raise ValueError("training_manifest_sha256 must be a 64-hex SHA-256 digest")
    if not legacy._is_hex_digest(expected_evaluation_code_revision, 40):
        raise ValueError(
            "expected_evaluation_code_revision must be a clean 40-hex revision"
        )
    if not str(evaluation_attempt).strip():
        raise ValueError("evaluation_attempt must not be empty")

    training_index = {
        (
            row["arm"],
            _exact_int(
                row["training_seed"],
                label="training_seed",
                prefix="training manifest",
            ),
        ): row
        for row in training_rows
    }

    attempt = Path(attempt_dir)
    attempt_root = attempt.resolve()
    if attempt_root.name != str(evaluation_attempt):
        raise ValueError(
            "evaluation attempt directory name must equal evaluation_attempt: "
            f"{attempt_root.name!r} != {evaluation_attempt!r}"
        )
    if evaluation_attempt != EXPECTED_EVALUATION_ATTEMPT:
        raise ValueError(
            f"evaluation_attempt must be {EXPECTED_EVALUATION_ATTEMPT!r}"
        )
    if evaluation_retry_history != EXPECTED_EVALUATION_RETRY_HISTORY:
        raise ValueError(
            "evaluation retry history must be exactly "
            f"{EXPECTED_EVALUATION_RETRY_HISTORY!r}"
        )
    summary_path = attempt / _SUMMARY_NAME
    if not summary_path.is_file():
        raise ValueError(f"evaluation attempt has no {_SUMMARY_NAME}: {attempt}")
    with summary_path.open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    built: dict[tuple[str, int], dict] = {}
    # evaluator-derived config identities, kept for the per-arm consistency pass
    evaluator_config: dict[tuple[str, int], dict] = {}
    strict_config: dict[tuple[str, int], Mapping] = {}
    for summary in summary_rows:
        name = str(summary.get("name", "")).strip()
        parts = name.split("_")
        if len(parts) != 3 or parts[0] != CAMPAIGN_NAME:
            raise ValueError(f"evaluation row name does not parse: {name!r}")
        short = parts[1]
        if short not in _LABEL_BY_SHORT:
            raise ValueError(f"{name}: unknown arm short {short!r}")
        arm = _LABEL_BY_SHORT[short]
        try:
            seed = _exact_int(
                summary["training_seed"], label="training_seed", prefix=name
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name}: training_seed is missing or malformed") from error
        if seed not in SEEDS:
            raise ValueError(f"{name}: training_seed must be 8..15, got {seed}")
        if parts[2] != f"seed{seed}":
            raise ValueError(f"{name}: row name disagrees with training_seed {seed}")
        key = (arm, seed)
        prefix = f"evaluation attempt: {arm}/seed{seed}"
        if key in built:
            raise ValueError(f"{prefix}: duplicate evaluation row")
        training_row = training_index.get(key)
        if training_row is None:
            raise ValueError(f"{prefix}: no accepted-training row for this identity")

        # --- identity + provenance cross-binding -------------------------
        if summary.get("task") != TASKS[arm]:
            raise ValueError(
                f"{prefix}: task {summary.get('task')!r} is not the frozen task for {arm}"
            )
        for column, expected, label in (
            ("checkpoint_path", training_row["checkpoint_path"], "checkpoint_path"),
            ("checkpoint_sha256", training_row["checkpoint_sha256"], "checkpoint_sha256"),
            (
                "accepted_checkpoint_sha256",
                training_row["checkpoint_sha256"],
                "accepted_checkpoint_sha256",
            ),
            (
                "accepted_manifest_sha256",
                training_manifest_sha256,
                "accepted_manifest_sha256",
            ),
            ("training_code_revision", training_row["code_revision"], "code revision"),
            ("training_asset_revision", training_row["asset_revision"], "asset revision"),
            ("asset_git_revision", training_row["asset_revision"], "asset revision"),
            # Action/impedance signatures ARE the same quantity in both files
            # (verified byte-identical across all 32 real rows), so they are
            # cross-bound.  campaign_config_sha256 / treatment_config_sha256 are
            # NOT: the evaluator derives them differently from
            # default_config_identity (they differ on all 32 real rows, with
            # different cardinality -- the signature of a different derivation,
            # not of drift). Asserting equality there would encode a false
            # identity, so those two are checked for per-arm CONSISTENCY below
            # instead, which still catches a seed evaluated under a odd config.
            (
                "fixed_action_signature_sha256",
                training_row["fixed_action_signature_sha256"],
                "fixed_action_signature_sha256",
            ),
            (
                "fixed_impedance_signature_sha256",
                training_row["fixed_impedance_signature_sha256"],
                "fixed_impedance_signature_sha256",
            ),
        ):
            if summary.get(column) != expected:
                raise ValueError(
                    f"{prefix}: {label} does not match the accepted training manifest"
                )
        if summary.get("git_revision") != expected_evaluation_code_revision:
            raise ValueError(
                f"{prefix}: evaluation code revision does not match the expected "
                "evaluation revision"
            )
        _require_clean(summary, "git_dirty", prefix=prefix)
        _require_clean(summary, "asset_git_dirty", prefix=prefix)
        for column in ("campaign_config_sha256", "treatment_config_sha256"):
            if not legacy._is_hex_digest(summary.get(column), 64):
                raise ValueError(f"{prefix}: {column} is missing or malformed")
        if not legacy._is_hex_digest(summary.get("nail_asset_sha256"), 64):
            raise ValueError(f"{prefix}: nail_asset_sha256 is missing or malformed")

        for column, expected in (
            ("reset_rng_seed", EVALUATOR_RNG["reset"]),
            ("observation_rng_seed", EVALUATOR_RNG["observation"]),
            ("action_rng_seed", EVALUATOR_RNG["action"]),
        ):
            if _exact_int(summary.get(column), label=column, prefix=prefix) != expected:
                raise ValueError(f"{prefix}: {column} is not the frozen rng seed")

        # --- artifact: exists, SHA recomputes, payload digest recomputes ---
        # The artifact MUST live inside the attempt being accepted.  Without
        # this, a summary row can point at another attempt's artifact and
        # self-consistently re-hash it, silently producing a manifest labelled
        # one attempt whose rows are a MIXTURE of attempts -- artifact-level
        # post-outcome selection, below the layer seed-level checks police.
        trace_path = Path(str(summary.get("sampled_trace_path", ""))).resolve()
        if not trace_path.is_relative_to(attempt_root):
            raise ValueError(
                f"{prefix}: sampled trace lives outside the attempt directory: {trace_path}"
            )
        if not trace_path.is_file():
            raise ValueError(f"{prefix}: sampled trace artifact is missing: {trace_path}")
        artifact_sha256 = legacy._sha256(trace_path)
        if artifact_sha256 != summary.get("sampled_trace_artifact_sha256"):
            raise ValueError(f"{prefix}: sampled trace artifact sha256 does not recompute")

        payload = _decode_sampled_artifact(trace_path, prefix=prefix)
        if _exact_int(
            payload.get("schema_version", -1), label="schema_version", prefix=prefix
        ) != 3:
            raise ValueError(
                f"{prefix}: sampled trace is not schema v3 "
                f"(found {payload.get('schema_version')!r})"
            )
        recorded_digest = payload.pop("payload_digest", None)
        try:
            recomputed_digest = _digest(payload)
        except ValueError as error:
            raise ValueError(
                f"{prefix}: payload carries nonfinite values and cannot be hashed: {error}"
            ) from error
        if recorded_digest != recomputed_digest:
            raise ValueError(f"{prefix}: sampled trace payload digest does not recompute")
        if summary.get("sampled_trace_digest") != recomputed_digest:
            raise ValueError(
                f"{prefix}: summary sampled_trace_digest disagrees with the artifact"
            )

        # The artifact must describe THIS identity -- a summary row must not be
        # able to point at another checkpoint's (internally consistent) traces.
        # ``task`` is unique per arm, so (task, training_seed) already pins the
        # identity exactly; the artifact's ``treatment`` field is deliberately
        # NOT checked here because the evaluator writes its own treatment label
        # ("F" for arm "F8"), and re-deriving that mapping in this module would
        # create a second source of truth that can drift.
        if payload.get("task") != TASKS[arm]:
            raise ValueError(f"{prefix}: artifact task does not match the frozen task")
        if _exact_int(
            payload.get("training_seed", -1), label="artifact training_seed", prefix=prefix
        ) != seed:
            raise ValueError(f"{prefix}: artifact training_seed does not match the row")
        artifact_rng_streams = payload.get("rng_streams")
        if (
            not isinstance(artifact_rng_streams, Mapping)
            or set(artifact_rng_streams) != set(EVALUATOR_RNG)
        ):
            raise ValueError(f"{prefix}: artifact rng_streams are not the frozen streams")
        for stream, expected in EVALUATOR_RNG.items():
            observed = _exact_int(
                artifact_rng_streams[stream],
                label=f"artifact rng_streams.{stream}",
                prefix=prefix,
            )
            if observed != expected:
                raise ValueError(
                    f"{prefix}: artifact rng_streams.{stream} is not the frozen stream"
                )
        # The artifact records WHICH checkpoint produced it.  Binding that to the
        # accepted-training row closes the last forgeable gap: without it, an
        # artifact from a different checkpoint (or a different attempt) verifies
        # perfectly as long as its own digests are self-consistent.
        artifact_provenance = payload.get("provenance")
        if not isinstance(artifact_provenance, Mapping):
            raise ValueError(f"{prefix}: artifact carries no payload provenance")
        if artifact_provenance.get("checkpoint_sha256") != training_row["checkpoint_sha256"]:
            raise ValueError(
                f"{prefix}: artifact was not produced by the accepted checkpoint"
            )
        for column, expected in (
            ("accepted_checkpoint_sha256", training_row["checkpoint_sha256"]),
            ("accepted_manifest_sha256", training_manifest_sha256),
            ("training_code_revision", training_row["code_revision"]),
            ("training_asset_revision", training_row["asset_revision"]),
            ("campaign_config_sha256", summary["campaign_config_sha256"]),
            ("treatment_config_sha256", summary["treatment_config_sha256"]),
            ("nail_asset_sha256", summary["nail_asset_sha256"]),
        ):
            if artifact_provenance.get(column) != expected:
                raise ValueError(
                    f"{prefix}: payload provenance {column} does not match "
                    "the accepted evidence"
                )
        nail_geometry = payload.get("nail_geometry")
        if (
            not isinstance(nail_geometry, Mapping)
            or nail_geometry.get("source_sha256")
            != artifact_provenance["nail_asset_sha256"]
        ):
            raise ValueError(
                f"{prefix}: payload nail geometry source does not match "
                "nail asset provenance"
            )

        for payload_key, summary_prefix in (
            ("code_git", "git"),
            ("asset_git", "asset_git"),
        ):
            binding = artifact_provenance.get(payload_key)
            if not isinstance(binding, Mapping):
                raise ValueError(f"{prefix}: payload provenance {payload_key} is missing")
            if binding.get("revision") != summary[f"{summary_prefix}_revision"]:
                raise ValueError(
                    f"{prefix}: payload provenance {payload_key} revision mismatch"
                )
            if binding.get("dirty") is not False:
                raise ValueError(
                    f"{prefix}: payload provenance {payload_key} is dirty"
                )

        evaluation_contract = payload.get("evaluation_contract")
        if not isinstance(evaluation_contract, Mapping):
            raise ValueError(f"{prefix}: artifact carries no evaluation_contract")
        for contract_key, summary_key in (
            ("fixed_action_signature_sha256", "fixed_action_signature_sha256"),
            (
                "fixed_impedance_signature_sha256",
                "fixed_impedance_signature_sha256",
            ),
        ):
            if evaluation_contract.get(contract_key) != summary[summary_key]:
                raise ValueError(
                    f"{prefix}: payload {contract_key} does not match "
                    "the accepted evaluation contract"
                )
        for contract_key, expected in (
            ("num_envs", EXPECTED_NUM_ENVS),
            ("episodes_per_env", EXPECTED_EPISODES_PER_ENV),
        ):
            if (
                _exact_int(
                    evaluation_contract.get(contract_key),
                    label=f"evaluation_contract.{contract_key}",
                    prefix=prefix,
                )
                != expected
            ):
                raise ValueError(
                    f"{prefix}: evaluation_contract.{contract_key} does not "
                    "match the frozen quota"
                )
        if (
            _exact_int(
                payload.get("expected_episode_count"),
                label="expected_episode_count",
                prefix=prefix,
            )
            != EXPECTED_N_EPISODES
        ):
            raise ValueError(
                f"{prefix}: expected_episode_count does not match the frozen quota"
            )
        imp_max_p = payload.get("imp_max_p")
        if isinstance(imp_max_p, bool) or imp_max_p != 0.0:
            raise ValueError(f"{prefix}: payload imp_max_p must be exactly 0.0")
        impact_weight, delivered_weight = quality.WEIGHTS[arm]
        if payload.get("weights") != {
            "impact_progress": impact_weight,
            "delivered_impulse": delivered_weight,
        }:
            raise ValueError(f"{prefix}: payload weights do not match the frozen arm")
        if payload.get("impulse_limits_n_m_s") != list(quality.IMPULSE_LIMITS):
            raise ValueError(
                f"{prefix}: payload impulse limits do not match the frozen limits"
            )

        identities = evaluation_contract.get("strict_config_identities")
        if not isinstance(identities, Mapping):
            raise ValueError(f"{prefix}: artifact carries no strict_config_identities")
        missing = [key_ for key_ in STRICT_CONFIG_IDENTITY_KEYS if key_ not in identities]
        if missing:
            raise ValueError(f"{prefix}: strict_config_identities missing {missing}")
        for identity_key in STRICT_CONFIG_IDENTITY_KEYS:
            if not legacy._is_hex_digest(identities[identity_key], 64):
                raise ValueError(
                    f"{prefix}: strict_config_identities.{identity_key} is "
                    "missing or malformed"
                )

        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"{prefix}: artifact carries no episode list")
        sentinels = _derive_validity_sentinels(
            episodes,
            nail_geometry=nail_geometry,
            prefix=prefix,
        )
        mean_invariants = payload.get("mean_rollout_invariants")
        if not isinstance(mean_invariants, Mapping):
            raise ValueError(
                f"{prefix}: payload mean_rollout_invariants is missing or malformed"
            )
        for column in ("impossible_success_n", "lambda_dead_n"):
            try:
                mean_count = _exact_int(
                    mean_invariants[column],
                    label=f"mean_rollout_invariants.{column}",
                    prefix=prefix,
                )
            except KeyError as error:
                raise ValueError(
                    f"{prefix}: mean_rollout_invariants.{column} is missing"
                ) from error
            combined_count = sentinels[column] + mean_count
            try:
                summary_count = _exact_int(
                    summary[column], label=f"sentinel {column}", prefix=prefix
                )
            except KeyError as error:
                raise ValueError(f"{prefix}: sentinel {column} is missing") from error
            if summary_count != combined_count:
                raise ValueError(
                    f"{prefix}: sentinel {column} disagrees with the "
                    f"digest-bound total ({summary_count} != {combined_count})"
                )
            sentinels[column] = combined_count
        for column, value in sentinels.items():
            if value != 0:
                raise ValueError(f"{prefix}: sentinel {column} is nonzero ({value})")

        evaluator_config[key] = {
            column: summary.get(column)
            for column in ("campaign_config_sha256", "treatment_config_sha256")
        }
        strict_config[key] = identities
        built[key] = {
            "campaign": CAMPAIGN_NAME,
            "disposition": "accepted",
            "arm": arm,
            "short": short,
            "task": TASKS[arm],
            "training_seed": seed,
            "checkpoint_path": training_row["checkpoint_path"],
            "checkpoint_sha256": training_row["checkpoint_sha256"],
            "code_revision": training_row["code_revision"],
            "asset_revision": training_row["asset_revision"],
            "campaign_config_sha256": training_row["campaign_config_sha256"],
            "treatment_config_sha256": training_row["treatment_config_sha256"],
            "fixed_action_signature_sha256": training_row[
                "fixed_action_signature_sha256"
            ],
            "fixed_impedance_signature_sha256": training_row[
                "fixed_impedance_signature_sha256"
            ],
            "cap_signature_sha256": training_row["cap_signature_sha256"],
            "clean_state": True,
            "accepted_training_manifest_sha256": training_manifest_sha256,
            "evaluation_attempt": str(evaluation_attempt),
            "evaluation_retry_history": str(evaluation_retry_history),
            **{key_: identities[key_] for key_ in STRICT_CONFIG_IDENTITY_KEYS},
            "reset_rng_seed": EVALUATOR_RNG["reset"],
            "observation_rng_seed": EVALUATOR_RNG["observation"],
            "action_rng_seed": EVALUATOR_RNG["action"],
            "sampled_trace_digest": recomputed_digest,
            "sampled_trace_artifact_sha256": artifact_sha256,
            "num_envs": _exact_int(
                summary.get("num_envs"), label="num_envs", prefix=prefix
            ),
            "episodes_per_env_sampled": _exact_int(
                summary.get("episodes_per_env_sampled"),
                label="episodes_per_env_sampled", prefix=prefix,
            ),
            "n_episodes_sampled": _exact_int(
                summary.get("n_episodes_sampled"),
                label="n_episodes_sampled", prefix=prefix,
            ),
            # All six sentinels are proven zero above. The quality/liveness/quota
            # counts and sampled exact-zero invariants are recomputed from the
            # digest-bound episodes; the mean exact-zero counts come from the
            # digest-bound payload. The unhashed summary totals must agree.
            **{name_: sentinels[name_] for name_ in SENTINELS},
        }

    # Per-arm consistency of the evaluator-derived config identities.  These are
    # per-arm quantities, so all eight seeds of an arm MUST report the same
    # value; a lone disagreeing seed means that checkpoint was evaluated under a
    # different configuration than its siblings, which the paired design cannot
    # tolerate.  (They are deliberately NOT compared to the training manifest --
    # see the cross-bind block above.)
    for column in ("campaign_config_sha256", "treatment_config_sha256"):
        by_arm: dict[str, set] = {}
        for (arm, _seed), observed in evaluator_config.items():
            by_arm.setdefault(arm, set()).add(observed[column])
        for arm, values in sorted(by_arm.items()):
            if len(values) != 1:
                raise ValueError(
                    f"evaluation attempt: {arm}: {column} is not consistent across "
                    f"its eight seeds ({len(values)} distinct values)"
                )
        if len({next(iter(values)) for values in by_arm.values()}) == 1:
            raise ValueError(
                f"evaluation attempt: {column} is identical across all four arms"
            )

    for identity_key in STRICT_CONFIG_IDENTITY_KEYS:
        by_arm: dict[str, set] = {}
        for (arm, _seed), identities in strict_config.items():
            by_arm.setdefault(arm, set()).add(identities[identity_key])
        for arm, values in sorted(by_arm.items()):
            if len(values) != 1:
                raise ValueError(
                    f"evaluation attempt: {arm}: strict_config_identities."
                    f"{identity_key} is not consistent across its eight seeds"
                )

    expected_keys = {(arm, seed) for arm in LABELS for seed in SEEDS}
    missing_keys = sorted(expected_keys - set(built))
    if missing_keys:
        raise ValueError(f"evaluation attempt is missing rows for: {missing_keys}")
    extra_keys = sorted(set(built) - expected_keys)
    if extra_keys:
        raise ValueError(f"evaluation attempt has unexpected rows for: {extra_keys}")

    rows = [built[(arm, seed)] for arm in LABELS for seed in sorted(SEEDS)]
    validate_evaluation_manifest(rows, training_rows, training_manifest_sha256)
    return rows


def write_evaluation_manifest(
    rows: Sequence[Mapping],
    path: str | Path,
    *,
    training_rows: Sequence[Mapping],
    training_manifest_sha256: str,
) -> str:
    """Fully validate, then atomically publish ``rows`` to a NEW ``path``.

    ``training_rows`` / ``training_manifest_sha256`` are REQUIRED: without them
    the writer could only check structure and types, and a semantically invalid
    manifest (wrong task, nonzero sentinel) would serialize, publish and return
    a SHA as though it were sound.

    Refuses to overwrite an existing file, publishes via an exclusively-created
    uniquely-named temporary in the destination directory (so a concurrent
    writer cannot swap the bytes between hashing and publication), and leaves
    no partial artifact behind on any failure.  Returns the SHA-256 of the
    bytes actually published.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing evaluation manifest: {destination}"
        )
    rows = list(rows)
    for row in rows:
        if row.get("disposition") != "accepted":
            raise ValueError(
                "accepted_evaluations.tsv may only contain accepted rows, found "
                f"disposition {row.get('disposition')!r}"
            )
    # the FULL contract, not merely a structural round-trip
    validate_evaluation_manifest(rows, training_rows, training_manifest_sha256)
    text = serialize_evaluation_manifest(rows)
    validate_evaluation_manifest(
        parse_evaluation_manifest(text), training_rows, training_manifest_sha256
    )
    payload = text.encode()

    handle, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with open(handle, "wb") as stream:
            stream.write(payload)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite an existing evaluation manifest: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# CLI: mirrors first_strike_campaign's ``validate-accepted-manifest`` command so
# the Slurm launcher can validate a snapshot of the accepted-training manifest
# through one Python entry point, before any GPU work. Emits one TSV line per
# validated row -- (manifest_sha256, short, training_seed, task, name,
# checkpoint_path, checkpoint_sha256) -- so the launcher never has to
# re-parse or reinterpret the manifest columns itself.
# ---------------------------------------------------------------------------


def _command_line() -> int:
    parser = argparse.ArgumentParser(
        description="fq4x8 accepted-training-manifest validation support"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser(
        "validate-accepted-manifest",
        help="validate and emit the frozen fq4x8 accepted-training manifest",
    )
    manifest.add_argument("--manifest", required=True)
    manifest.add_argument("--expected-code-revision", required=True)
    manifest.add_argument("--expected-asset-revision", required=True)

    build = subparsers.add_parser(
        "build-training-manifest",
        help="scan fq4x8 training run directories and emit the accepted-training manifest",
    )
    build.add_argument("--log-root", required=True)
    build.add_argument("--code-revision", required=True)
    build.add_argument("--asset-revision", required=True)
    build.add_argument("--attempt", default="attempt1")
    build.add_argument("--out")

    build_eval = subparsers.add_parser(
        "build-evaluation-manifest",
        help="turn one immutable evaluation attempt into accepted_evaluations.tsv",
    )
    build_eval.add_argument("--attempt-dir", required=True)
    build_eval.add_argument("--training-manifest", required=True)
    build_eval.add_argument("--training-manifest-sha256", required=True)
    build_eval.add_argument("--evaluation-attempt", required=True)
    build_eval.add_argument("--evaluation-retry-history", required=True)
    build_eval.add_argument("--expected-evaluation-code-revision", required=True)
    build_eval.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "validate-accepted-manifest":
        return _validate_accepted_manifest_command(args)
    if args.command == "build-training-manifest":
        return _build_training_manifest_command(args)
    if args.command == "build-evaluation-manifest":
        return _build_evaluation_manifest_command(args)
    parser.error("unsupported command")
    return 2  # pragma: no cover -- parser.error exits the process


def _build_evaluation_manifest_command(args) -> int:
    try:
        manifest_path = Path(args.training_manifest)
        actual_sha256 = legacy._sha256(manifest_path)
        if actual_sha256 != args.training_manifest_sha256:
            raise ValueError(
                "training manifest sha256 does not match the frozen digest: "
                f"expected {args.training_manifest_sha256}, found {actual_sha256}"
            )
        training_rows = parse_training_manifest(
            manifest_path.read_text(encoding="utf-8")
        )
        rows = build_evaluation_manifest(
            args.attempt_dir,
            training_rows=training_rows,
            training_manifest_sha256=actual_sha256,
            evaluation_attempt=args.evaluation_attempt,
            evaluation_retry_history=args.evaluation_retry_history,
            expected_evaluation_code_revision=args.expected_evaluation_code_revision,
        )
        out_path = Path(args.out)
        manifest_sha256 = write_evaluation_manifest(
            rows,
            out_path,
            training_rows=training_rows,
            training_manifest_sha256=actual_sha256,
        )
        # Re-validate the bytes actually on disk, in-process, before success.
        written_rows = parse_evaluation_manifest(out_path.read_text(encoding="utf-8"))
        validate_evaluation_manifest(written_rows, training_rows, actual_sha256)
    except Exception as error:  # noqa: BLE001 -- fail-closed CLI boundary
        print(f"MANIFEST_FAIL: {error}", file=sys.stderr)
        return 2

    print(f"{out_path}\t{manifest_sha256}\t{len(rows)}")
    return 0


def _build_training_manifest_command(args) -> int:
    try:
        rows = build_training_manifest(
            args.log_root,
            code_revision=args.code_revision,
            asset_revision=args.asset_revision,
            attempt=args.attempt,
        )
        text = serialize_training_manifest(rows)
        out_path = (
            Path(args.out)
            if args.out
            else Path(args.log_root) / "accepted_training_checkpoints.tsv"
        )
        out_path.write_text(text, encoding="utf-8")
        # Re-parse and re-validate the bytes actually on disk, in-process,
        # before declaring success -- guards against a serialization bug
        # silently emitting a manifest the module's own validator rejects.
        written_rows = parse_training_manifest(out_path.read_text(encoding="utf-8"))
        validate_training_manifest(written_rows)
    except Exception as error:  # noqa: BLE001 -- fail-closed CLI boundary
        print(f"MANIFEST_FAIL: {error}", file=sys.stderr)
        return 2

    manifest_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"{out_path}\t{manifest_sha256}\t{len(rows)}")
    return 0


def _validate_accepted_manifest_command(args) -> int:
    manifest_path = Path(args.manifest)
    try:
        if not legacy._is_hex_digest(args.expected_code_revision, 40):
            raise ValueError("expected code revision must be a full 40-hex revision")
        if not legacy._is_hex_digest(args.expected_asset_revision, 40):
            raise ValueError("expected asset revision must be a full 40-hex revision")
        rows = parse_training_manifest(manifest_path.read_text(encoding="utf-8"))
        validate_training_manifest(rows)
        for row in rows:
            prefix = f"{row['arm']}/seed{row['training_seed']}"
            if row["code_revision"] != args.expected_code_revision:
                raise ValueError(f"{prefix}: code_revision does not equal expected code revision")
            if row["asset_revision"] != args.expected_asset_revision:
                raise ValueError(
                    f"{prefix}: asset_revision does not equal expected asset revision"
                )
    except Exception as error:  # noqa: BLE001 -- fail-closed CLI boundary
        print(f"MANIFEST_FAIL: {error}", file=sys.stderr)
        return 2

    manifest_sha256 = legacy._sha256(manifest_path)
    for row in sorted(rows, key=_sort_key):
        name = f"{CAMPAIGN_NAME}_{row['short']}_seed{row['training_seed']}"
        print(
            "\t".join(
                (
                    manifest_sha256,
                    str(row["short"]),
                    str(row["training_seed"]),
                    str(row["task"]),
                    name,
                    str(row["checkpoint_path"]),
                    str(row["checkpoint_sha256"]),
                )
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(_command_line())
