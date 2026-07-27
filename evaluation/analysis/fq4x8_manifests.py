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
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

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


# ---------------------------------------------------------------------------
# TSV formatting (pure, no validation beyond structural completeness)
# ---------------------------------------------------------------------------


def _sort_key(row: Mapping) -> tuple[int, int]:
    arm = row.get("arm")
    if arm not in LABELS:
        raise ValueError(f"unknown arm: {arm!r}")
    return (LABELS.index(arm), int(row["training_seed"]))


def _format_cell(
    field: str, value: object, *, bool_fields: frozenset, int_fields: frozenset
) -> str:
    if field in bool_fields:
        text = "true" if legacy._parse_bool(value) else "false"
    elif field in int_fields:
        text = str(int(value))
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
                row[field] = int(cell)
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
        try:
            seed = int(row.get("training_seed"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{context}: {arm}: training_seed must be an integer, "
                f"got {row.get('training_seed')!r}"
            ) from error
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
            if int(row[key]) != expected:
                raise ValueError(f"{prefix}: {key} must be the frozen value {expected}")

        for key in ("sampled_trace_digest", "sampled_trace_artifact_sha256"):
            if not legacy._is_hex_digest(row[key], 64):
                raise ValueError(f"{prefix}: {key} is missing or malformed")

        if int(row["num_envs"]) != EXPECTED_NUM_ENVS:
            raise ValueError(f"{prefix}: num_envs must be {EXPECTED_NUM_ENVS}")
        if int(row["episodes_per_env_sampled"]) != EXPECTED_EPISODES_PER_ENV:
            raise ValueError(
                f"{prefix}: episodes_per_env_sampled must be {EXPECTED_EPISODES_PER_ENV}"
            )
        if int(row["n_episodes_sampled"]) != EXPECTED_N_EPISODES:
            raise ValueError(
                f"{prefix}: n_episodes_sampled must be exactly {EXPECTED_N_EPISODES}"
            )

        for key in SENTINELS:
            if int(row[key]) != 0:
                raise ValueError(f"{prefix}: sentinel {key} is nonzero")


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
    args = parser.parse_args()
    if args.command != "validate-accepted-manifest":
        parser.error("unsupported command")

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
