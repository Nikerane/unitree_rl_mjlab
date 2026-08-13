"""Lean analysis for the registered first-contact quality campaign.

The campaign has four human-facing arms (F8, F0, D0, FQ-min).  The evaluator
still records the quality treatment as ``FQ``; that raw identifier is never
allowed to leak into the registered reporting labels or alter the frozen
inference family.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from evaluation.analysis import first_strike_campaign as legacy


CAMPAIGN_NAME = "fq4x8"
LABELS = ("F8", "F0", "D0", "FQ-min")
SEEDS = tuple(range(8, 16))
RAW_TREATMENT = {"F8": "F", "F0": "F0", "D0": "D0", "FQ-min": "FQ"}
_LABEL_BY_RAW_TREATMENT = {
    raw_treatment: label for label, raw_treatment in RAW_TREATMENT.items()
}
SHORT = {"F8": "f8", "F0": "f0", "D0": "d0", "FQ-min": "fq"}
TASKS = {
    "F8": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    "F0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
    "D0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
    "FQ-min": "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
}
WEIGHTS = {
    "F8": (8.0, 2.0),
    "F0": (0.0, 2.0),
    "D0": (8.0, 0.0),
    "FQ-min": (8.0, 0.0),
}
IMPACT_READERS = {
    "F8": "FirstStrikeImpactRewardTerm",
    "F0": "FirstStrikeImpactRewardTerm",
    "D0": "FirstStrikeImpactRewardTerm",
    "FQ-min": "FirstStrikeQualityImpactRewardTerm",
}
DELIVERED_READER = "FirstStrikeDeliveredRewardTerm"
SPEED_NORMALIZER = {
    "F8": 1.0,
    "F0": 1.0,
    "D0": 1.0,
    "FQ-min": 1.4598331451416016,
}
EXPECTED_TREATMENT_REWARD_SHA256 = {
    "F8": "47d993698852dc939c753d978e41c9124e24c470e0595439f004d2b561d76fb9",
    "F0": "a8fdd61dc521a6a4294945d94e52dde8fde6560ee963976c08251e0996d42f9c",
    "D0": "c2c8f069f5f744eb063514b0c4a20e75ed9b271b382f2397281c04e7ac12f325",
    "FQ-min": "5bdd740a32cb730e1e63392422387dff5df2c8812d52587060249fe2b54a09a7",
}
DECISION_FIELDS = (
    "first_contact_quality_sampled",
    "first_window_useful_speed_mean_sampled",
    "first_window_success_rate_sampled",
    "overall_success_rate_sampled",
    "event_window_depth_gain_mean_sampled",
)
SENTINELS = (
    "quality_overflow_n",
    "quality_nonfinite_n",
    "liveness_failure_n",
    "impossible_success_n",
    "lambda_dead_n",
    "quota_failure_n",
)
STRICT_CONFIG_IDENTITY_KEYS = (
    "training_config_sha256",
    "evaluation_config_sha256",
    "training_policy_observation_sha256",
    "evaluation_policy_observation_sha256",
    "training_treatment_reward_sha256",
    "evaluation_treatment_reward_sha256",
)
PAYOUT_SEMANTICS = {
    "F": "actual_event_linear",
    "F0": "actual_event_linear_speed_disabled",
    "D0": "actual_event_linear_delivered_disabled",
    "FQ": "actual_event_quality_bounded",
}
PRIMARY_CONTRASTS = (
    ("F0_minus_F8", "F0", "F8"),
    ("D0_minus_F8", "D0", "F8"),
    ("FQ-min_minus_D0", "FQ-min", "D0"),
)
IMPULSE_LIMITS = legacy.EXPECTED_IMPULSE_LIMITS_N_M_S
# Frozen in the pinned evaluator configuration. This is an outcome-blind
# instrument-identity/readability prerequisite, not an inferred performance
# threshold or evidence that eight slots are universally sufficient.
EXPECTED_QUALITY_SENSOR_SLOTS = 8
_FLOAT32_EPS = float(np.finfo(np.float32).eps)
_FLOAT32_RECOMPUTE_ATOL = {
    "contact_point_w": 0.0,
    "contact_error_m": 1e-9,
    "contact_quality": 2.0 * _FLOAT32_EPS,
    "contact_normal_axiality": _FLOAT32_EPS,
}


def _frozen_row(label: str, seed: int) -> dict:
    impact, delivered = WEIGHTS[label]
    return {
        "campaign": CAMPAIGN_NAME,
        "treatment": label,
        "raw_treatment": RAW_TREATMENT[label],
        "short": SHORT[label],
        "task": TASKS[label],
        "training_seed": seed,
        "impact_weight": impact,
        "delivered_weight": delivered,
        "impact_reader": IMPACT_READERS[label],
        "delivered_reader": DELIVERED_READER,
        "speed_normalizer_m_s": SPEED_NORMALIZER[label],
        "fixed_impedance_signature_sha256":
            legacy.EXPECTED_FIXED_IMPEDANCE_SIGNATURE_SHA256,
        "fixed_action_signature_sha256":
            legacy.EXPECTED_FIXED_ACTION_SIGNATURE_SHA256,
        "impulse_limits_n_m_s": list(IMPULSE_LIMITS),
        "training_iterations": 500,
        "training_num_envs": 4096,
        "checkpoint_filename": "model_499.pt",
        "evaluation_num_envs": 256,
        "evaluation_episodes_per_env": 2,
        "decision_fields": list(DECISION_FIELDS),
    }


FROZEN_QUALITY_CAMPAIGN_MATRIX = tuple(
    _frozen_row(label, seed) for label in LABELS for seed in SEEDS
)


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def exact_paired_sign_flip(
    treatment: Sequence[float], control: Sequence[float]
) -> dict:
    """Exhaustive two-sided paired sign-flip sensitivity analysis."""

    treatment = _finite_vector(treatment, name="treatment")
    control = _finite_vector(control, name="control")
    if treatment.shape != control.shape:
        raise ValueError("paired samples must have matching shapes")
    differences = treatment - control
    nonzero = differences[np.abs(differences) > 1e-15]
    zero_count = int(differences.size - nonzero.size)
    observed = abs(float(np.mean(nonzero))) if nonzero.size else 0.0
    if not nonzero.size:
        p_value = 1.0
        assignments = 1
    else:
        assignments = 2 ** int(nonzero.size)
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=int(nonzero.size)):
            statistic = abs(float(np.mean(nonzero * np.asarray(signs))))
            extreme += statistic >= observed - 1e-15
        p_value = extreme / assignments
    return {
        "n_pairs": int(differences.size),
        "zero_difference_count": zero_count,
        "assignment_count": assignments,
        "mean_paired_difference": float(np.mean(differences)),
        "two_sided_exact_p": p_value,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjustment, preserving the caller's labels."""

    if not p_values:
        return {}
    checked = {}
    for name, value in p_values.items():
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Holm inputs must be finite probabilities")
        checked[str(name)] = value
    ordered = sorted(checked, key=checked.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * checked[name]))
        adjusted[name] = running
    return {name: adjusted[name] for name in checked}


def paired_bootstrap_ratio(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    samples: int = 100_000,
    seed: int = 20_260_726,
) -> dict:
    """Seed-matched PCG64 bootstrap interval for a ratio of arm means."""

    treatment = _finite_vector(treatment, name="treatment")
    control = _finite_vector(control, name="control")
    if treatment.shape != control.shape:
        raise ValueError("paired samples must have matching shapes")
    if samples <= 0:
        raise ValueError("samples must be positive")
    treatment_mean = float(treatment.mean())
    control_mean = float(control.mean())
    if not np.isfinite(control_mean) or control_mean <= 0.0:
        raise ValueError("ratio bootstrap requires a positive denominator")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, treatment.size, size=(samples, treatment.size))
    treatment_means = treatment[indices].mean(axis=1)
    control_means = control[indices].mean(axis=1)
    if not np.isfinite(control_means).all() or np.any(control_means <= 0.0):
        raise ValueError("ratio bootstrap requires a positive denominator")
    ratios = treatment_means / control_means
    return {
        "estimate": treatment_mean / control_mean,
        "one_sided_95_lower": float(np.quantile(ratios, 0.05)),
        "two_sided_95_interval": [
            float(np.quantile(ratios, 0.025)),
            float(np.quantile(ratios, 0.975)),
        ],
        "samples": int(samples),
        "seed": int(seed),
    }


def _paired_bootstrap_difference(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    samples: int = 100_000,
    seed: int = 20_260_726,
) -> dict:
    treatment = _finite_vector(treatment, name="treatment")
    control = _finite_vector(control, name="control")
    if treatment.shape != control.shape:
        raise ValueError("paired samples must have matching shapes")
    differences = treatment - control
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    boot = differences[indices].mean(axis=1)
    return {
        "estimate": float(differences.mean()),
        "one_sided_95_lower": float(np.quantile(boot, 0.05)),
        "one_sided_95_upper": float(np.quantile(boot, 0.95)),
        "two_sided_95_interval": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "samples": samples,
        "seed": seed,
    }


def _treatment_digest(row: Mapping) -> str:
    raw = str(row["raw_treatment"])
    value = {
        "schema_version": 1,
        "task": str(row["task"]),
        "treatment": raw,
        "payout_semantics": PAYOUT_SEMANTICS[raw],
        "impact_weight": float(row["impact_weight"]),
        "delivered_weight": float(row["delivered_weight"]),
        "event_i_ref_n_s": 0.3088,
    }
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _require_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def validate_quality_campaign_contract(
    rows: Sequence[Mapping],
    accepted_evaluation_manifest: Sequence[Mapping],
) -> None:
    """Fail closed on any drift from the registered 32-row campaign."""

    # Local import avoids the module-import cycle: fq4x8_manifests derives its
    # frozen identities from this module.
    from evaluation.analysis import fq4x8_manifests as manifests

    rows = list(rows)
    manifest = list(accepted_evaluation_manifest)
    if len(rows) != 32:
        raise ValueError("quality campaign requires exactly 32 evaluation rows")
    if len(manifest) != 32:
        raise ValueError("quality campaign requires 32 accepted-evaluation rows")
    canonical_fields = set(manifests.EVALUATION_FIELDS)
    for accepted in manifest:
        if set(accepted) != canonical_fields:
            missing = sorted(canonical_fields - set(accepted))
            unexpected = sorted(set(accepted) - canonical_fields)
            raise ValueError(
                "accepted-evaluation row does not match the canonical 39-field "
                f"schema (missing={missing}, unexpected={unexpected})"
            )

    row_index = {
        (str(row.get("treatment")), int(row.get("training_seed", -1))): row
        for row in rows
    }
    manifest_index = {
        (str(row.get("arm")), int(row.get("training_seed", -1))): row
        for row in manifest
    }
    expected_rows = {
        (item["raw_treatment"], item["training_seed"])
        for item in FROZEN_QUALITY_CAMPAIGN_MATRIX
    }
    expected_manifest = {
        (item["treatment"], item["training_seed"])
        for item in FROZEN_QUALITY_CAMPAIGN_MATRIX
    }
    if set(row_index) != expected_rows or len(row_index) != 32:
        raise ValueError("evaluation row identities do not match the frozen matrix")
    if set(manifest_index) != expected_manifest or len(manifest_index) != 32:
        raise ValueError("accepted-evaluation identities do not match the frozen matrix")

    for frozen in FROZEN_QUALITY_CAMPAIGN_MATRIX:
        label, raw, seed = (
            frozen["treatment"],
            frozen["raw_treatment"],
            frozen["training_seed"],
        )
        row = row_index[(raw, seed)]
        accepted = manifest_index[(label, seed)]
        prefix = f"{label}/seed{seed}"
        if accepted.get("campaign") != CAMPAIGN_NAME:
            raise ValueError(f"{prefix}: campaign must be {CAMPAIGN_NAME}")
        if accepted.get("disposition") != "accepted":
            raise ValueError(f"{prefix}: disposition must be accepted")
        if not str(accepted.get("evaluation_attempt", "")).strip():
            raise ValueError(f"{prefix}: accepted evaluation attempt is missing")
        if (
            accepted.get("evaluation_attempt")
            != manifests.EXPECTED_EVALUATION_ATTEMPT
            or accepted.get("evaluation_retry_history")
            != manifests.EXPECTED_EVALUATION_RETRY_HISTORY
        ):
            raise ValueError(f"{prefix}: frozen evaluation attempt mismatch")
        manifests._validate_retry_history(
            accepted.get("evaluation_retry_history"),
            accepted_attempt=accepted["evaluation_attempt"],
            prefix=prefix,
        )
        for key, expected, category in (
            ("arm", label, "arm"),
            ("short", frozen["short"], "short"),
            ("task", frozen["task"], "task"),
            ("training_seed", seed, "training seed"),
            (
                "fixed_impedance_signature_sha256",
                frozen["fixed_impedance_signature_sha256"],
                "gains",
            ),
            (
                "fixed_action_signature_sha256",
                frozen["fixed_action_signature_sha256"],
                "action",
            ),
            (
                "cap_signature_sha256",
                manifests.EXPECTED_CAP_SIGNATURE_SHA256,
                "caps",
            ),
        ):
            if accepted.get(key) != expected:
                raise ValueError(f"{prefix}: frozen {category} mismatch")
        try:
            clean_state = legacy._parse_bool(accepted.get("clean_state"))
        except ValueError as error:
            raise ValueError(f"{prefix}: invalid clean_state") from error
        if not clean_state:
            raise ValueError(f"{prefix}: clean_state must be true")

        _require_equal(row.get("task"), frozen["task"], f"{prefix}: task mismatch")
        _require_equal(
            row.get("fixed_impedance_signature_sha256"),
            frozen["fixed_impedance_signature_sha256"],
            f"{prefix}: frozen gains mismatch",
        )
        _require_equal(
            row.get("fixed_action_signature_sha256"),
            frozen["fixed_action_signature_sha256"],
            f"{prefix}: frozen action mismatch",
        )
        _require_equal(
            float(row.get("impact_weight", np.nan)),
            frozen["impact_weight"],
            f"{prefix}: weights mismatch",
        )
        _require_equal(
            float(row.get("delivered_weight", np.nan)),
            frozen["delivered_weight"],
            f"{prefix}: weights mismatch",
        )
        raw_quota = (
            row.get("num_envs"),
            row.get("episodes_per_env_sampled"),
            row.get("n_episodes_sampled"),
        )
        manifest_quota = (
            accepted.get("num_envs"),
            accepted.get("episodes_per_env_sampled"),
            accepted.get("n_episodes_sampled"),
        )
        if raw_quota != (256, 2, 512) or manifest_quota != (256, 2, 512):
            raise ValueError(f"{prefix}: strict 256x2 quota mismatch")
        if float(row.get("imp_max_p", np.nan)) != 0.0:
            raise ValueError(f"{prefix}: imp_max_p must be zero")
        checkpoint_path = str(row.get("checkpoint_path", ""))
        if Path(checkpoint_path).name != "model_499.pt":
            raise ValueError(f"{prefix}: checkpoint mismatch")
        if accepted.get("checkpoint_path") != checkpoint_path:
            raise ValueError(f"{prefix}: checkpoint path binding mismatch")
        for dirty_key in ("git_dirty", "asset_git_dirty"):
            try:
                row_dirty = legacy._parse_bool(row.get(dirty_key))
            except ValueError as error:
                raise ValueError(f"{prefix}: invalid dirty provenance") from error
            if row_dirty:
                raise ValueError(f"{prefix}: dirty provenance is forbidden")
        for sentinel in SENTINELS:
            try:
                sentinel_value = manifests._exact_int(
                    accepted.get(sentinel),
                    label=f"sentinel {sentinel}",
                    prefix=prefix,
                )
            except ValueError as error:
                raise ValueError(
                    f"{prefix}: sentinel {sentinel} is not an exact "
                    "nonnegative integer"
                ) from error
            if sentinel_value != 0:
                raise ValueError(f"{prefix}: sentinel {sentinel} is nonzero")
        for key, expected in (
            ("action_rng_seed", legacy.CAMPAIGN_EVALUATOR_RNG["action"]),
            ("reset_rng_seed", legacy.CAMPAIGN_EVALUATOR_RNG["reset"]),
            ("observation_rng_seed", legacy.CAMPAIGN_EVALUATOR_RNG["observation"]),
        ):
            if int(row.get(key, -1)) != expected or accepted.get(key) != expected:
                raise ValueError(f"{prefix}: frozen evaluator RNG mismatch")
        for key in (
            "checkpoint_sha256",
            "accepted_checkpoint_sha256",
            "accepted_manifest_sha256",
            "campaign_config_sha256",
            "treatment_config_sha256",
            "nail_asset_sha256",
            "sampled_trace_digest",
            "sampled_trace_artifact_sha256",
        ):
            if not legacy._is_hex_digest(row.get(key), 64):
                raise ValueError(f"{prefix}: invalid hash binding for {key}")
        if row["accepted_checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError(f"{prefix}: accepted checkpoint binding mismatch")
        for key in (
            "checkpoint_sha256",
            "sampled_trace_digest",
            "sampled_trace_artifact_sha256",
        ):
            if not legacy._is_hex_digest(accepted.get(key), 64):
                raise ValueError(f"{prefix}: invalid manifest hash for {key}")
            if accepted[key] != row[key]:
                raise ValueError(f"{prefix}: manifest binding mismatch for {key}")
        for key in ("campaign_config_sha256", "treatment_config_sha256"):
            if not legacy._is_hex_digest(accepted.get(key), 64):
                raise ValueError(f"{prefix}: invalid manifest hash for {key}")
        if not legacy._is_hex_digest(
            accepted.get("accepted_training_manifest_sha256"), 64
        ) or accepted["accepted_training_manifest_sha256"] != row[
            "accepted_manifest_sha256"
        ]:
            raise ValueError(f"{prefix}: accepted manifest binding mismatch")
        for key in STRICT_CONFIG_IDENTITY_KEYS:
            if not legacy._is_hex_digest(accepted.get(key), 64):
                raise ValueError(
                    f"{prefix}: invalid strict configuration hash for {key}"
                )
        if accepted["training_policy_observation_sha256"] != accepted[
            "evaluation_policy_observation_sha256"
        ]:
            raise ValueError(f"{prefix}: observation identity mismatch")
        if accepted["training_treatment_reward_sha256"] != accepted[
            "evaluation_treatment_reward_sha256"
        ]:
            raise ValueError(f"{prefix}: reward identity mismatch")
        if accepted["training_treatment_reward_sha256"] != (
            EXPECTED_TREATMENT_REWARD_SHA256[label]
        ):
            raise ValueError(f"{prefix}: frozen reward semantics mismatch")
        evaluation_code_revision = row.get("git_revision")
        training_code_revision = row.get("training_code_revision")
        manifest_code_revision = accepted.get("code_revision")
        if not all(
            legacy._is_hex_digest(value, 40)
            for value in (
                evaluation_code_revision,
                training_code_revision,
                manifest_code_revision,
            )
        ):
            raise ValueError(f"{prefix}: invalid git revision")
        if manifest_code_revision != training_code_revision:
            raise ValueError(f"{prefix}: training code revision binding mismatch")

        evaluation_asset_revision = row.get("asset_git_revision")
        training_asset_revision = row.get("training_asset_revision")
        manifest_asset_revision = accepted.get("asset_revision")
        if not all(
            legacy._is_hex_digest(value, 40)
            for value in (
                evaluation_asset_revision,
                training_asset_revision,
                manifest_asset_revision,
            )
        ):
            raise ValueError(f"{prefix}: invalid asset_git revision")
        if not (
            evaluation_asset_revision
            == training_asset_revision
            == manifest_asset_revision
        ):
            raise ValueError(f"{prefix}: asset_git revision binding mismatch")
        if row["treatment_config_sha256"] != _treatment_digest(frozen):
            raise ValueError(f"{prefix}: treatment configuration binding mismatch")
        artifact = Path(str(row.get("sampled_trace_path", "")))
        if not artifact.is_file():
            raise ValueError(f"{prefix}: sampled artifact is missing")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != row[
            "sampled_trace_artifact_sha256"
        ]:
            raise ValueError(f"{prefix}: sampled artifact binding mismatch")

    common_row_fields = (
        "git_revision",
        "asset_git_revision",
        "training_code_revision",
        "training_asset_revision",
        "accepted_manifest_sha256",
        "nail_asset_sha256",
        "fixed_impedance_signature_sha256",
        "fixed_action_signature_sha256",
        "action_rng_seed",
        "reset_rng_seed",
        "observation_rng_seed",
    )
    for key in common_row_fields:
        if len({str(row.get(key)) for row in rows}) != 1:
            raise ValueError(f"campaign: treatment-shared {key} mismatch")
    for key in (
        "training_policy_observation_sha256",
        "evaluation_policy_observation_sha256",
    ):
        if len({str(item.get(key)) for item in manifest}) != 1:
            raise ValueError(f"campaign: treatment-shared {key} mismatch")
    if len({str(item.get("campaign_config_sha256")) for item in manifest}) != 1:
        raise ValueError(
            "campaign: canonical training campaign_config_sha256 mismatch"
        )
    for label in LABELS:
        accepted_arm = [
            item for item in manifest if item.get("arm") == label
        ]
        raw_arm = [
            row for row in rows if row.get("treatment") == RAW_TREATMENT[label]
        ]
        for key in (
            "training_config_sha256",
            "evaluation_config_sha256",
            "training_treatment_reward_sha256",
            "evaluation_treatment_reward_sha256",
        ):
            if len({str(item.get(key)) for item in accepted_arm}) != 1:
                raise ValueError(
                    f"{label}: per-treatment {key} identity mismatch"
                )
        if (
            len(
                {
                    str(item.get("treatment_config_sha256"))
                    for item in accepted_arm
                }
            )
            != 1
            or accepted_arm[0].get("treatment_config_sha256")
            != manifests.EXPECTED_TREATMENT_CONFIG_SHA256[label]
        ):
            raise ValueError(
                f"{label}: canonical treatment_config_sha256 identity mismatch"
            )
        # The raw evaluator and canonical training manifest derive these two
        # identities differently. Each derivation must be internally stable
        # across an arm, but equality between the two sources is not meaningful.
        for key in ("campaign_config_sha256", "treatment_config_sha256"):
            if len({str(item.get(key)) for item in raw_arm}) != 1:
                raise ValueError(
                    f"{label}: per-treatment {key} identity mismatch"
                )


def _quality_payload_identity_reasons(
    row: Mapping, accepted: Mapping, payload: Mapping
) -> list[str]:
    """Cross-bind schema-v3 payload identities to row and accepted manifest."""

    prefix = f"{accepted.get('arm')}/seed{accepted.get('training_seed')}"
    reasons: list[str] = []
    evaluation = payload.get("evaluation_contract", {})
    strict = evaluation.get("strict_config_identities", {})
    if not isinstance(strict, Mapping) or set(strict) != set(
        STRICT_CONFIG_IDENTITY_KEYS
    ):
        reasons.append(f"{prefix}: payload must contain all six strict identities")
    else:
        for key in STRICT_CONFIG_IDENTITY_KEYS:
            if (
                not legacy._is_hex_digest(strict.get(key), 64)
                or strict[key] != accepted.get(key)
            ):
                reasons.append(f"{prefix}: strict {key} manifest binding mismatch")
        if strict["training_policy_observation_sha256"] != strict[
            "evaluation_policy_observation_sha256"
        ]:
            reasons.append(f"{prefix}: observation identity mismatch")
        if strict["training_treatment_reward_sha256"] != strict[
            "evaluation_treatment_reward_sha256"
        ]:
            reasons.append(f"{prefix}: reward identity mismatch")
        expected_reward = EXPECTED_TREATMENT_REWARD_SHA256.get(
            str(accepted.get("arm"))
        )
        if strict["training_treatment_reward_sha256"] != expected_reward:
            reasons.append(f"{prefix}: frozen reward semantics mismatch")
    expected_payload = {
        "treatment": row.get("treatment"),
        "task": row.get("task"),
        "training_seed": int(row.get("training_seed", -1)),
        "event_i_ref_n_s": float(row.get("event_i_ref_n_s", np.nan)),
        "imp_max_p": 0.0,
    }
    for key, expected in expected_payload.items():
        if payload.get(key) != expected:
            reasons.append(f"{prefix}: payload {key} binding mismatch")
    if payload.get("weights") != {
        "impact_progress": float(row.get("impact_weight", np.nan)),
        "delivered_impulse": float(row.get("delivered_weight", np.nan)),
    }:
        reasons.append(f"{prefix}: payload weight binding mismatch")
    if evaluation.get("fixed_impedance_signature_sha256") != row.get(
        "fixed_impedance_signature_sha256"
    ):
        reasons.append(f"{prefix}: payload gains binding mismatch")
    if evaluation.get("fixed_action_signature_sha256") != row.get(
        "fixed_action_signature_sha256"
    ):
        reasons.append(f"{prefix}: payload action binding mismatch")
    if payload.get("rng_streams") != {
        "reset": row.get("reset_rng_seed"),
        "observation": row.get("observation_rng_seed"),
        "action": row.get("action_rng_seed"),
    }:
        reasons.append(f"{prefix}: payload RNG binding mismatch")
    provenance = payload.get("provenance", {})
    provenance_bindings = {
        "checkpoint_sha256": row.get("checkpoint_sha256"),
        "accepted_checkpoint_sha256": row.get("accepted_checkpoint_sha256"),
        "campaign_config_sha256": row.get("campaign_config_sha256"),
        "treatment_config_sha256": row.get("treatment_config_sha256"),
        "nail_asset_sha256": row.get("nail_asset_sha256"),
        "accepted_manifest_sha256": row.get("accepted_manifest_sha256"),
        "training_code_revision": row.get("training_code_revision"),
        "training_asset_revision": row.get("training_asset_revision"),
    }
    for key, expected in provenance_bindings.items():
        if provenance.get(key) != expected:
            reasons.append(f"{prefix}: payload provenance {key} mismatch")
    for payload_key, row_prefix in (("code_git", "git"), ("asset_git", "asset_git")):
        binding = provenance.get(payload_key, {})
        if (
            binding.get("revision") != row.get(f"{row_prefix}_revision")
            or binding.get("dirty") is not False
        ):
            reasons.append(f"{prefix}: payload {payload_key} revision mismatch")
    return reasons


def _validate_common_nail_geometry(geometries: Sequence[Mapping]) -> dict:
    geometries = [dict(geometry) for geometry in geometries]
    if not geometries:
        raise ValueError("common nail geometry is missing")
    required = {"nail_axis", "nail_xy_m", "nail_radius_m", "source_sha256"}
    if not required <= set(geometries[0]):
        raise ValueError("common nail geometry fields are malformed")
    expected = {key: geometries[0][key] for key in required}
    try:
        axis = np.asarray(expected["nail_axis"], dtype=float)
        xy = np.asarray(expected["nail_xy_m"], dtype=float)
        radius = float(expected["nail_radius_m"])
    except (TypeError, ValueError) as error:
        raise ValueError("common nail geometry is malformed") from error
    if (
        axis.shape != (3,)
        or xy.shape != (2,)
        or not np.isfinite(axis).all()
        or not np.isfinite(xy).all()
        or not np.isfinite(radius)
        or radius <= 0.0
        or not legacy._is_hex_digest(expected["source_sha256"], 64)
    ):
        raise ValueError("common nail geometry is malformed")
    projections = []
    for geometry in geometries[1:]:
        if not required <= set(geometry):
            raise ValueError("common nail geometry fields are malformed")
        projections.append({key: geometry[key] for key in required})
    if any(projection != expected for projection in projections):
        raise ValueError("common nail frame/radius/provenance mismatch")
    return expected


def _nail_plane(
    point: np.ndarray, geometry: Mapping
) -> tuple[float, float, float]:
    axis = np.asarray(geometry["nail_axis"], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("nail axis is nonfinite or malformed")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("nail axis must be nonzero")
    axis /= axis_norm
    reference = np.asarray([1.0, 0.0, 0.0])
    first = reference - np.dot(reference, axis) * axis
    if np.linalg.norm(first) < 1e-12:
        reference = np.asarray([0.0, 1.0, 0.0])
        first = reference - np.dot(reference, axis) * axis
    first /= np.linalg.norm(first)
    second = np.cross(first, axis)
    origin = np.asarray(
        [geometry["nail_xy_m"][0], geometry["nail_xy_m"][1], 0.0],
        dtype=float,
    )
    offset = point - origin
    x = float(np.dot(offset, first))
    y = float(np.dot(offset, second))
    return x, y, float(np.hypot(x, y))


class QualityOverflowError(ValueError):
    """The accepted-onset overflow evidence is valid and aligned."""


def _numeric_quality_array(value: object, *, label: str) -> np.ndarray:
    """Decode a JSON numeric array without accepting bool/string coercions."""

    try:
        object_values = np.asarray(value, dtype=object)
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is malformed") from error
    if raw.dtype.kind not in "iuf" or any(
        isinstance(item, (bool, np.bool_)) for item in object_values.flat
    ):
        raise ValueError(f"{label} must contain only JSON numbers")
    values = raw.astype(float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains nonfinite values")
    return values


def _quality_bool_stream(
    value: object, *, label: str, sample_count: int
) -> list[bool]:
    if (
        not isinstance(value, list)
        or len(value) != sample_count
        or any(type(item) is not bool for item in value)
    ):
        raise ValueError(
            f"{label} must be one JSON boolean per physical substep"
        )
    return value


def _finite_quality_scalar(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a finite JSON number")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be a finite JSON number")
    return parsed


def _as_finite_float32(value: np.ndarray, *, label: str) -> np.ndarray:
    """Cast finite numeric evidence without permitting float32 overflow."""

    values64 = np.asarray(value, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        values32 = values64.astype(np.float32)
    if not np.isfinite(values32).all():
        raise ValueError(f"{label} exceeds the float32 range")
    return values32


def _within_float32_forward_error(
    actual: object, expected: object, *, field: str
) -> bool:
    """Compare CPU/CUDA float32 replay under one field-specific bound."""

    actual64 = np.asarray(actual, dtype=float)
    expected64 = np.asarray(expected, dtype=float)
    if (
        actual64.shape != expected64.shape
        or not np.isfinite(actual64).all()
        or not np.isfinite(expected64).all()
    ):
        return False
    try:
        _as_finite_float32(actual64, label=field)
        expected32 = _as_finite_float32(expected64, label=field)
    except ValueError:
        return False
    return bool(
        np.all(
            np.abs(actual64 - expected32.astype(float))
            <= _FLOAT32_RECOMPUTE_ATOL[field]
        )
    )


def validate_first_contact_snapshot(
    trace: Mapping, *, physical_sample_count: int | None = None
) -> int | None:
    """Validate required immutable snapshot fields without dense trace data."""

    first = trace.get("first_strike")
    if not isinstance(first, Mapping):
        raise ValueError("first-strike quality snapshot is missing")
    if "started" not in first or type(first["started"]) is not bool:
        raise ValueError("first_strike.started must be a JSON boolean")
    if "accepted_onset_index" not in first:
        raise ValueError("accepted_onset_index is missing")
    onset_raw = first["accepted_onset_index"]
    if onset_raw is None:
        onset = None
    elif isinstance(onset_raw, bool) or not isinstance(onset_raw, int):
        raise ValueError("quality onset index is malformed")
    else:
        onset = onset_raw
    if onset is not None and (
        onset < 0
        or (
            physical_sample_count is not None
            and onset >= physical_sample_count
        )
    ):
        raise ValueError("quality onset index is malformed")
    if first["started"] != (onset is not None):
        raise ValueError("quality snapshot started/onset mismatch")

    if "contact_quality_overflow" not in first:
        raise ValueError("missing contact_quality_overflow snapshot")
    snapshot_overflow = first["contact_quality_overflow"]
    if type(snapshot_overflow) is not bool:
        raise ValueError(
            "contact_quality_overflow snapshot must be a JSON boolean"
        )
    if onset is None and snapshot_overflow:
        raise ValueError(
            "contact_quality_overflow cannot be true without an accepted onset"
        )

    required = (
        "contact_point_w",
        "contact_error_m",
        "contact_quality",
        "contact_quality_valid",
        "first_contact_time_s",
        "contact_normal_axiality",
    )
    missing = [key for key in required if key not in first]
    if missing:
        raise ValueError(f"missing quality field: {missing[0]}")
    point = _numeric_quality_array(
        first["contact_point_w"], label="contact_point_w"
    )
    if point.shape != (3,):
        raise ValueError("contact_point_w must contain three finite values")
    for key in (
        "contact_error_m",
        "contact_quality",
        "first_contact_time_s",
        "contact_normal_axiality",
    ):
        _finite_quality_scalar(first[key], label=key)
    if first["contact_error_m"] < 0.0:
        raise ValueError("contact_error_m must be nonnegative")
    if not 0.0 <= first["contact_quality"] <= 1.0:
        raise ValueError("contact_quality must lie in [0, 1]")
    if first["first_contact_time_s"] < 0.0:
        raise ValueError("first_contact_time_s must be nonnegative")
    if not 0.0 <= first["contact_normal_axiality"] <= 1.0:
        raise ValueError("contact_normal_axiality must lie in [0, 1]")
    if onset is None and first["first_contact_time_s"] != 0.0:
        raise ValueError(
            "first_contact_time_s must be zero without an accepted onset"
        )
    if type(first["contact_quality_valid"]) is not bool:
        raise ValueError("contact_quality_valid must be a JSON boolean")
    return onset


def _validated_raw_quality_arrays(
    physical: Mapping, *, physical_sample_count: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        found = _numeric_quality_array(
            physical["quality_found_count"], label="quality_found_count"
        )
        forces = _numeric_quality_array(
            physical["quality_normal_force_n"],
            label="quality_normal_force_n",
        )
        positions = _numeric_quality_array(
            physical["quality_contact_position_m"],
            label="quality_contact_position_m",
        )
        normals = _numeric_quality_array(
            physical["quality_contact_normal"],
            label="quality_contact_normal",
        )
    except KeyError as error:
        raise ValueError(f"missing raw quality slot: {error.args[0]}") from error
    if (
        found.ndim != 2
        or found.shape[0] < 1
        or (
            physical_sample_count is not None
            and found.shape[0] != physical_sample_count
        )
        or forces.shape != found.shape
        or positions.shape != (*found.shape, 3)
        or normals.shape != (*found.shape, 3)
    ):
        raise ValueError("raw quality slot shape mismatch")
    if found.shape[1] != EXPECTED_QUALITY_SENSOR_SLOTS:
        raise ValueError(
            "raw quality sensor must have exactly "
            f"{EXPECTED_QUALITY_SENSOR_SLOTS} slots"
        )
    if (
        np.any(found < 0.0)
        or not np.equal(found, np.floor(found)).all()
    ):
        raise ValueError("raw quality slots contain nonfinite or invalid values")
    forces = _as_finite_float32(
        forces, label="quality_normal_force_n"
    )
    positions = _as_finite_float32(
        positions, label="quality_contact_position_m"
    )
    normals = _as_finite_float32(
        normals, label="quality_contact_normal"
    )
    return found, forces, positions, normals


def validate_first_contact_raw_quality(
    trace: Mapping, *, physical_sample_count: int | None = None
) -> int:
    """Validate raw quality geometry and return its dense sample count."""

    physical = trace.get("physical")
    if not isinstance(physical, Mapping):
        raise ValueError("physical quality channels are missing")
    found, _, _, _ = _validated_raw_quality_arrays(
        physical, physical_sample_count=physical_sample_count
    )
    return int(found.shape[0])


def _validate_quality_latch_streams(
    event: Mapping,
    first: Mapping,
    *,
    onset: int | None,
    sample_count: int,
) -> None:
    """Require exact initialized-then-immutable float32 tracker latches."""

    event_started = _quality_bool_stream(
        event.get("tracker_started"),
        label="tracker_started",
        sample_count=sample_count,
    )
    expected_started = [False] * sample_count
    if onset is not None:
        expected_started[onset:] = [True] * (sample_count - onset)
    if event_started != expected_started:
        raise ValueError("quality tracker_started latch is inconsistent")

    scalar_streams = {
        "contact_error_m": "tracker_contact_error_m",
        "contact_quality": "tracker_contact_quality",
        "contact_normal_axiality": "tracker_contact_normal_axiality",
    }
    for first_key, event_key in scalar_streams.items():
        first_value = _finite_quality_scalar(
            first.get(first_key), label=first_key
        )
        _as_finite_float32(np.asarray(first_value), label=first_key)
        values = _numeric_quality_array(
            event.get(event_key), label=event_key
        )
        if values.shape != (sample_count,):
            raise ValueError(f"event quality stream shape mismatch: {event_key}")
        _as_finite_float32(values, label=event_key)
        expected = np.zeros(sample_count, dtype=float)
        if onset is not None:
            expected[onset:] = first_value
        if not np.array_equal(values, expected):
            raise ValueError(
                f"quality event/snapshot mismatch for {first_key}"
            )

    first_point = _numeric_quality_array(
        first.get("contact_point_w"), label="contact_point_w"
    )
    if first_point.shape != (3,):
        raise ValueError("contact_point_w must have shape (3,)")
    _as_finite_float32(first_point, label="contact_point_w")
    event_points = _numeric_quality_array(
        event.get("tracker_contact_point_w"), label="tracker_contact_point_w"
    )
    if event_points.shape != (sample_count, 3):
        raise ValueError("event quality stream shape mismatch")
    _as_finite_float32(event_points, label="tracker_contact_point_w")
    expected_points = np.zeros((sample_count, 3), dtype=float)
    if onset is not None:
        expected_points[onset:] = first_point
    if not np.array_equal(event_points, expected_points):
        raise ValueError("quality event/snapshot mismatch for contact_point_w")

    first_valid = first.get("contact_quality_valid")
    if type(first_valid) is not bool:
        raise ValueError("contact_quality_valid must be a JSON boolean")
    event_valid = _quality_bool_stream(
        event.get("tracker_contact_quality_valid"),
        label="tracker_contact_quality_valid",
        sample_count=sample_count,
    )
    expected_valid = [False] * sample_count
    if onset is not None:
        expected_valid[onset:] = [first_valid] * (sample_count - onset)
    if event_valid != expected_valid:
        raise ValueError(
            "quality event/snapshot mismatch for contact_quality_valid"
        )

    first_time = _finite_quality_scalar(
        first.get("first_contact_time_s"), label="first_contact_time_s"
    )
    _as_finite_float32(
        np.asarray(first_time), label="first_contact_time_s"
    )
    event_time = _numeric_quality_array(
        event.get("tracker_first_contact_time_s"),
        label="tracker_first_contact_time_s",
    )
    if event_time.shape != (sample_count,):
        raise ValueError("event quality stream shape mismatch")
    _as_finite_float32(event_time, label="tracker_first_contact_time_s")
    expected_time = np.zeros(sample_count, dtype=float)
    if onset is not None:
        expected_time[onset:] = first_time
    if not np.array_equal(event_time, expected_time):
        raise ValueError(
            "quality event/snapshot mismatch for first_contact_time_s"
        )


def _validate_quality_overflow_latch(
    event: Mapping,
    first: Mapping,
    *,
    onset: int | None,
    sample_count: int,
) -> bool:
    """Validate overflow without coupling it to other quality sentinels."""

    first_overflow = first.get("contact_quality_overflow")
    if type(first_overflow) is not bool:
        raise ValueError("contact_quality_overflow must be a JSON boolean")
    event_overflow = _quality_bool_stream(
        event.get("tracker_contact_quality_overflow"),
        label="tracker_contact_quality_overflow",
        sample_count=sample_count,
    )
    expected_overflow = [False] * sample_count
    if onset is not None:
        expected_overflow[onset:] = [first_overflow] * (sample_count - onset)
    if event_overflow != expected_overflow:
        raise ValueError("quality overflow event/snapshot mismatch")
    return first_overflow


def _quality_overflow_at_onset(
    event: Mapping,
    first: Mapping,
    *,
    onset: int | None,
    sample_count: int,
    found: np.ndarray,
) -> bool:
    snapshot_overflow = _validate_quality_overflow_latch(
        event, first, onset=onset, sample_count=sample_count
    )
    # ``found`` stores the sensor-reported contact count. Compare it with the
    # actual, already-validated slot width; ``>`` is intentional because a
    # count equal to capacity still fits without overflow.
    raw_onset_overflow = (
        False
        if onset is None
        else bool(np.any(found[onset] > found.shape[1]))
    )
    if onset is not None and raw_onset_overflow != snapshot_overflow:
        raise ValueError(
            "raw/event/snapshot overflow disagree at the accepted onset"
        )
    return onset is not None and snapshot_overflow


def first_contact_overflow_at_onset(
    trace: Mapping, *, physical_sample_count: int
) -> bool:
    """Return genuine aligned accepted-onset overflow, independently of liveness."""

    physical = trace.get("physical")
    event = trace.get("event_trace")
    first = trace.get("first_strike")
    if not isinstance(physical, Mapping):
        raise ValueError("physical quality channels are missing")
    if not isinstance(event, Mapping):
        raise ValueError("event quality channels are missing")
    if not isinstance(first, Mapping):
        raise ValueError("first-strike quality snapshot is missing")
    onset = validate_first_contact_snapshot(
        trace, physical_sample_count=physical_sample_count
    )
    found, _, _, _ = _validated_raw_quality_arrays(
        physical, physical_sample_count=physical_sample_count
    )
    return _quality_overflow_at_onset(
        event,
        first,
        onset=onset,
        sample_count=physical_sample_count,
        found=found,
    )


def _quality_snapshot_from_raw_slots(
    physical: Mapping,
    event: Mapping,
    first: Mapping,
    *,
    onset: int | None,
    sample_count: int,
    nail_geometry: Mapping,
) -> dict:
    """Validate raw slots and recompute the immutable onset-quality snapshot."""

    found, forces, positions, normals = _validated_raw_quality_arrays(
        physical, physical_sample_count=sample_count
    )
    _validate_quality_latch_streams(
        event, first, onset=onset, sample_count=sample_count
    )
    if _quality_overflow_at_onset(
        event,
        first,
        onset=onset,
        sample_count=sample_count,
        found=found,
    ):
        raise QualityOverflowError("quality instrumentation overflow")

    axis = _as_finite_float32(
        np.asarray(nail_geometry.get("nail_axis"), dtype=float),
        label="nail_axis",
    )
    nail_xy = _as_finite_float32(
        np.asarray(nail_geometry.get("nail_xy_m"), dtype=float),
        label="nail_xy_m",
    )
    radius = _as_finite_float32(
        np.asarray(
            _finite_quality_scalar(
            nail_geometry.get("nail_radius_m"),
            label="nail_radius_m",
            )
        ),
        label="nail_radius_m",
    ).item()
    if (
        axis.shape != (3,)
        or nail_xy.shape != (2,)
        or not np.isfinite(axis).all()
        or not np.isfinite(nail_xy).all()
        or not np.isfinite(radius)
    ):
        raise ValueError("nail geometry exceeds the float32 range")
    axis_t = torch.from_numpy(axis).unsqueeze(0)
    axis_norm = torch.linalg.vector_norm(axis_t, dim=-1, keepdim=True)
    if radius <= 0.0 or not bool((axis_norm > 0.0).all()):
        raise ValueError("nail geometry is malformed")

    empty = {
        "valid": False,
        "point": np.zeros(3),
        "error": 0.0,
        "quality": 0.0,
        "normal_axiality": 0.0,
    }
    if onset is None:
        recomputed = empty
    else:
        # The tracker produced and latched these values in torch float32.
        # Preserve Torch's operation order rather than approximating it with
        # NumPy: even at float32, their reductions need not round identically.
        arrays32 = tuple(
            np.asarray(value[onset], dtype=np.float32)
            for value in (found, forces, positions, normals)
        )
        if any(not np.isfinite(value).all() for value in arrays32):
            raise ValueError("raw quality slots exceed the float32 range")
        found_t, forces_t, positions_t, normals_t = (
            torch.from_numpy(value).unsqueeze(0) for value in arrays32
        )
        nail_top_t = torch.tensor(
            [[nail_xy[0], nail_xy[1], np.float32(0.0)]],
            dtype=torch.float32,
        )
        normalized_axis = axis_t / axis_norm
        weights_t = torch.where(
            found_t > 0.0,
            forces_t.clamp_min(0.0),
            torch.zeros_like(found_t),
        )
        total_weight = weights_t.sum(dim=1)
        valid = bool(total_weight[0] > 0.0)
        if not valid:
            recomputed = empty
        else:
            point_t = (
                positions_t * weights_t.unsqueeze(-1)
            ).sum(dim=1) / total_weight.unsqueeze(-1)
            offset_t = point_t - nail_top_t
            radial_t = offset_t - (
                offset_t * normalized_axis
            ).sum(dim=-1, keepdim=True) * normalized_axis
            error_t = torch.linalg.vector_norm(radial_t, dim=-1)
            radius_t = torch.as_tensor(radius, dtype=torch.float32)
            quality_t = torch.clamp(
                1.0 - (error_t / radius_t) ** 2,
                min=0.0,
                max=1.0,
            )
            weighted_normal_t = (
                normals_t * weights_t.unsqueeze(-1)
            ).sum(dim=1)
            normal_norm_t = torch.linalg.vector_norm(
                weighted_normal_t, dim=-1, keepdim=True
            )
            unit_normal_t = weighted_normal_t / torch.where(
                normal_norm_t > 0.0,
                normal_norm_t,
                torch.ones_like(normal_norm_t),
            )
            axiality_t = (
                unit_normal_t * normalized_axis
            ).sum(dim=-1).clamp(0.0, 1.0)
            axiality_t = torch.where(
                normal_norm_t.squeeze(-1) > 0.0,
                axiality_t,
                torch.zeros_like(axiality_t),
            )
            recomputed = {
                "valid": True,
                "point": point_t[0].numpy(),
                "error": float(error_t[0]),
                "quality": float(quality_t[0]),
                "normal_axiality": float(axiality_t[0]),
            }

    expected = {
        "contact_point_w": recomputed["point"],
        "contact_error_m": recomputed["error"],
        "contact_quality": recomputed["quality"],
        "contact_quality_valid": recomputed["valid"],
        "contact_normal_axiality": recomputed["normal_axiality"],
    }
    for key, expected_value in expected.items():
        if key not in first:
            raise ValueError(f"missing quality field: {key}")
        first_value = first[key]
        if isinstance(expected_value, np.ndarray):
            parsed_first = _numeric_quality_array(first_value, label=key)
            matches_recomputed = (
                np.array_equal(parsed_first, expected_value)
                if not recomputed["valid"]
                else _within_float32_forward_error(
                    parsed_first, expected_value, field=key
                )
            )
        elif isinstance(expected_value, bool):
            if type(first_value) is not bool:
                raise ValueError(f"{key} must be a JSON boolean")
            matches_recomputed = first_value is expected_value
        else:
            parsed_first = _finite_quality_scalar(first_value, label=key)
            matches_recomputed = (
                parsed_first == expected_value
                if not recomputed["valid"]
                else _within_float32_forward_error(
                    parsed_first, expected_value, field=key
                )
            )
        if not matches_recomputed:
            raise ValueError(f"quality snapshot differs from recomputed {key}")
    return recomputed


def validate_first_contact_quality(
    trace: Mapping, *, nail_geometry: Mapping
) -> tuple[int | None, dict]:
    """Pure validation/recomputation of one immutable first-contact snapshot."""

    physical = trace.get("physical")
    event = trace.get("event_trace")
    first = trace.get("first_strike")
    if not isinstance(physical, Mapping):
        raise ValueError("physical quality channels are missing")
    if not isinstance(event, Mapping):
        raise ValueError("event quality channels are missing")

    contact = physical.get("contact")
    if (
        not isinstance(contact, list)
        or not contact
        or any(type(value) is not bool for value in contact)
    ):
        raise ValueError("contact must be a nonempty JSON-boolean series")
    sample_count = len(contact)
    onset = validate_first_contact_snapshot(
        trace, physical_sample_count=sample_count
    )

    event_started = _quality_bool_stream(
        event.get("tracker_started"),
        label="tracker_started",
        sample_count=sample_count,
    )
    started_indices = [
        index for index, started in enumerate(event_started) if started
    ]
    event_onset = started_indices[0] if started_indices else None
    if event_onset != onset:
        raise ValueError("quality event/snapshot onset mismatch")
    if onset is None:
        if any(contact):
            raise ValueError("raw contact exists without an accepted onset")
    elif not contact[onset] or any(contact[:onset]):
        raise ValueError("raw/event/snapshot onset mismatch")

    snapshot = _quality_snapshot_from_raw_slots(
        physical,
        event,
        first,
        onset=onset,
        sample_count=sample_count,
        nail_geometry=nail_geometry,
    )
    return onset, snapshot


def analyze_quality_episode(
    trace: Mapping,
    *,
    nail_geometry: Mapping,
    impulse_limits_n_m_s: Sequence[float] = IMPULSE_LIMITS,
) -> dict:
    """Derive all registered episode-level quality/mechanism channels."""

    try:
        base = legacy.summarize_episode(trace, nail_geometry=nail_geometry)
    except ValueError as error:
        if "finite" in str(error):
            raise ValueError(f"nonfinite physical quality channel: {error}") from error
        raise
    physical = trace.get("physical", {})
    event = trace.get("event_trace", {})
    first = trace.get("first_strike", {})
    contact = np.asarray(physical.get("contact"), dtype=bool)
    position = np.asarray(physical.get("head_position_m"), dtype=float)
    depth = np.asarray(
        physical.get("tracker_depth_post_integration_m"), dtype=float
    )
    force = np.asarray(physical.get("net_axial_force_n"), dtype=float)
    if (
        contact.ndim != 1
        or position.shape != (contact.size, 3)
        or depth.shape != contact.shape
        or force.shape != contact.shape
        or not np.isfinite(position).all()
        or not np.isfinite(depth).all()
        or not np.isfinite(force).all()
    ):
        raise ValueError("physical quality channels are nonfinite or malformed")
    dt_s = float(trace.get("physics_dt_s", 0.0))
    onset, snapshot = validate_first_contact_quality(
        trace, nail_geometry=nail_geometry
    )
    quality = float(snapshot["quality"])
    plane_x = plane_y = radial = 0.0
    if snapshot["valid"]:
        plane_x, plane_y, radial = _nail_plane(
            np.asarray(snapshot["point"], dtype=float), nail_geometry
        )
    axial_speed = lateral_speed = angle = 0.0
    normal_axiality = float(snapshot["normal_axiality"])
    dwell_ms = 0.0
    recontacts = 0
    depth_gain = 0.0
    if onset is not None:
        if onset > 0:
            velocity = (position[onset] - position[onset - 1]) / dt_s
            axis = np.asarray(nail_geometry["nail_axis"], dtype=float)
            axis /= np.linalg.norm(axis)
            signed_axial = float(np.dot(velocity, axis))
            axial_speed = max(0.0, signed_axial)
            lateral_speed = float(
                np.linalg.norm(velocity - signed_axial * axis)
            )
            angle = float(np.degrees(np.arctan2(lateral_speed, axial_speed)))
        finalized = np.asarray(event.get("tracker_finalized"), dtype=bool)
        finalized_indices = np.flatnonzero(finalized)
        stop = int(finalized_indices[0]) if finalized_indices.size else contact.size - 1
        stop = max(onset, stop)
        depth_gain = float(
            max(0.0, np.max(depth[onset : stop + 1]) - depth[onset])
        )
        dwell_ms = float(np.count_nonzero(contact[onset : stop + 1]) * dt_s * 1e3)
        window_contact = contact[onset : stop + 1]
        recontacts = int(
            np.count_nonzero((~window_contact[:-1]) & window_contact[1:])
        )

    peak_lambda = np.asarray(trace.get("episode_peak_lambda"), dtype=float)
    limits = np.asarray(impulse_limits_n_m_s, dtype=float)
    if (
        peak_lambda.shape != (6,)
        or limits.shape != (6,)
        or not np.isfinite(peak_lambda).all()
        or not np.isfinite(limits).all()
        or np.any(peak_lambda < 0.0)
        or np.any(limits <= 0.0)
    ):
        raise ValueError("lambda/cap values are nonfinite or malformed")
    return {
        "first_contact_quality_sampled": quality,
        "first_contact_quality_valid_sampled": bool(snapshot["valid"]),
        "first_contact_radial_error_m_sampled": radial,
        "contact_plane_x_m_sampled": plane_x,
        "contact_plane_y_m_sampled": plane_y,
        "onset_axial_speed_m_s_sampled": axial_speed,
        "onset_lateral_speed_m_s_sampled": lateral_speed,
        "contact_normal_axiality_sampled": normal_axiality,
        "approach_angle_deg_sampled": angle,
        "first_window_useful_speed_m_s_sampled": float(
            first.get("v_precontact_m_s", 0.0)
        ),
        "first_window_success_sampled": bool(
            first.get("reason") == "success"
        ),
        "overall_success_sampled": bool(trace.get("overall_success", False)),
        "event_window_depth_gain_m_sampled": depth_gain,
        "contact_dwell_ms_sampled": dwell_ms,
        "recontact_count_sampled": recontacts,
        "raw_delivered_impulse_n_s_sampled": float(
            first.get("delivered_n_s", 0.0)
        ),
        "episode_id": str(trace.get("episode_id", "")),
        "env_id": int(trace.get("env_id", -1)),
        "episode_ordinal": int(trace.get("episode_ordinal", -1)),
        "nail_asset_sha256": str(nail_geometry.get("source_sha256", "")),
        "peak_qvel_rad_s_sampled": float(base["qvel_max_abs_rad_s"]),
        "qvel_rail_exceeded_sampled": bool(base["qvel_violation"]),
        "qvel_nonfinite_sampled": bool(base["qvel_nonfinite"]),
        "lambda_per_joint_n_m_s_sampled": peak_lambda.tolist(),
        "worst_lambda_cap_ratio_sampled": float(np.max(peak_lambda / limits)),
    }


def aggregate_quality_episode_metrics(
    episodes: Sequence[Mapping], *, expected_episode_count: int | None = None
) -> dict:
    episodes = list(episodes)
    if expected_episode_count is not None and len(episodes) != expected_episode_count:
        raise ValueError(
            f"expected exactly {expected_episode_count} episodes, found {len(episodes)}"
        )
    if not episodes:
        raise ValueError("cannot aggregate zero episodes")

    def mean(key: str) -> float:
        values = np.asarray([row[key] for row in episodes], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite episode metric: {key}")
        return float(values.mean())

    result = {
        "n_episodes_sampled": len(episodes),
        "first_contact_quality_sampled": mean("first_contact_quality_sampled"),
        "first_contact_radial_error_mean_sampled": mean(
            "first_contact_radial_error_m_sampled"
        ),
        "onset_axial_speed_mean_sampled": mean(
            "onset_axial_speed_m_s_sampled"
        ),
        "onset_lateral_speed_mean_sampled": mean(
            "onset_lateral_speed_m_s_sampled"
        ),
        "contact_normal_axiality_mean_sampled": mean(
            "contact_normal_axiality_sampled"
        ),
        "approach_angle_mean_deg_sampled": mean(
            "approach_angle_deg_sampled"
        ),
        "first_window_useful_speed_mean_sampled": mean(
            "first_window_useful_speed_m_s_sampled"
        ),
        "first_window_success_rate_sampled": mean(
            "first_window_success_sampled"
        ),
        "overall_success_rate_sampled": mean("overall_success_sampled"),
        "event_window_depth_gain_mean_sampled": mean(
            "event_window_depth_gain_m_sampled"
        ),
        "contact_dwell_mean_ms_sampled": mean("contact_dwell_ms_sampled"),
        "recontact_count_mean_sampled": mean("recontact_count_sampled"),
        "raw_delivered_impulse_mean_n_s_sampled": mean(
            "raw_delivered_impulse_n_s_sampled"
        ),
        "peak_qvel_rad_s_sampled": max(
            float(row["peak_qvel_rad_s_sampled"]) for row in episodes
        ),
        "qvel_rail_exceeded_rate_sampled": mean(
            "qvel_rail_exceeded_sampled"
        ),
        "qvel_nonfinite_rate_sampled": mean("qvel_nonfinite_sampled"),
        "worst_lambda_cap_ratio_mean_sampled": mean(
            "worst_lambda_cap_ratio_sampled"
        ),
        "worst_lambda_cap_ratio_max_sampled": max(
            float(row["worst_lambda_cap_ratio_sampled"]) for row in episodes
        ),
        "valid_contact_coordinates_sampled": [
            {
                "episode_id": str(row["episode_id"]),
                "env_id": int(row["env_id"]),
                "episode_ordinal": int(row["episode_ordinal"]),
                "nail_asset_sha256": str(row["nail_asset_sha256"]),
                "x_m": float(row["contact_plane_x_m_sampled"]),
                "y_m": float(row["contact_plane_y_m_sampled"]),
            }
            for row in episodes
            if bool(row["first_contact_quality_valid_sampled"])
        ],
    }
    lambdas = np.asarray(
        [row["lambda_per_joint_n_m_s_sampled"] for row in episodes], dtype=float
    )
    if lambdas.shape != (len(episodes), 6) or not np.isfinite(lambdas).all():
        raise ValueError("per-joint lambda matrix is nonfinite or malformed")
    for joint in range(6):
        result[f"lambda_joint{joint + 1}_mean_n_m_s_sampled"] = float(
            lambdas[:, joint].mean()
        )
        result[f"lambda_joint{joint + 1}_max_n_m_s_sampled"] = float(
            lambdas[:, joint].max()
        )
    return result


def _values(by_arm: Mapping[str, Sequence[Mapping]], label: str, key: str) -> np.ndarray:
    rows = list(by_arm[label])
    if all("training_seed" in row for row in rows):
        rows.sort(key=lambda row: int(row["training_seed"]))
        seeds = [int(row["training_seed"]) for row in rows]
        if seeds != list(SEEDS):
            raise ValueError(f"{label}: seed rows must be exactly 8..15")
    elif len(rows) != len(SEEDS):
        raise ValueError(f"{label}: exactly eight seed rows are required")
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _endpoint(
    by_arm: Mapping[str, Sequence[Mapping]],
    treatment: str,
    control: str,
    key: str,
) -> dict:
    treatment_values = _values(by_arm, treatment, key)
    control_values = _values(by_arm, control, key)
    return {
        "arm_means": {
            "treatment": float(treatment_values.mean()),
            "control": float(control_values.mean()),
        },
        "paired_differences": (treatment_values - control_values).tolist(),
        "paired_bootstrap_difference": _paired_bootstrap_difference(
            treatment_values, control_values
        ),
    }


def _practical_ratio(
    treatment_values: np.ndarray, control_values: np.ndarray
) -> dict:
    common = {
        "arm_means": {
            "treatment": float(treatment_values.mean()),
            "control": float(control_values.mean()),
        },
        "paired_differences": (treatment_values - control_values).tolist(),
    }
    try:
        return {
            "valid": True,
            **paired_bootstrap_ratio(treatment_values, control_values),
            **common,
        }
    except ValueError as error:
        return {
            "valid": False,
            "reason": str(error),
            "one_sided_95_lower": None,
            **common,
        }


def _fq_min_practical_acceptance(
    by_arm: Mapping[str, Sequence[Mapping]], *, safety_gates_pass: bool
) -> dict:
    quality = _endpoint(
        by_arm, "FQ-min", "F8", "first_contact_quality_sampled"
    )
    fq_speed = _values(
        by_arm, "FQ-min", "first_window_useful_speed_mean_sampled"
    )
    f8_speed = _values(by_arm, "F8", "first_window_useful_speed_mean_sampled")
    speed = _practical_ratio(fq_speed, f8_speed)
    fq_depth = _values(
        by_arm, "FQ-min", "event_window_depth_gain_mean_sampled"
    )
    f8_depth = _values(
        by_arm, "F8", "event_window_depth_gain_mean_sampled"
    )
    depth = _practical_ratio(fq_depth, f8_depth)
    fq_first = _values(
        by_arm, "FQ-min", "first_window_success_rate_sampled"
    )
    f8_first = _values(by_arm, "F8", "first_window_success_rate_sampled")
    fq_overall = _values(by_arm, "FQ-min", "overall_success_rate_sampled")
    f8_overall = _values(by_arm, "F8", "overall_success_rate_sampled")
    first_success = _endpoint(
        by_arm, "FQ-min", "F8", "first_window_success_rate_sampled"
    )
    overall_success = _endpoint(
        by_arm, "FQ-min", "F8", "overall_success_rate_sampled"
    )
    quality_gain = float(np.mean(quality["paired_differences"]))
    inclusive_tolerance = 1e-12
    gates = {
        "quality_gain_at_least_0_10":
            quality_gain >= 0.10 - inclusive_tolerance,
        "useful_speed_ratio_lower_gt_0_95": bool(
            speed["valid"] and speed["one_sided_95_lower"] > 0.95
        ),
        "first_window_success_at_least_0_90":
            float(fq_first.mean()) >= 0.90 - inclusive_tolerance,
        "first_window_success_drop_at_most_0_05":
            float((fq_first - f8_first).mean())
            >= -0.05 - inclusive_tolerance,
        "overall_success_at_least_0_90":
            float(fq_overall.mean()) >= 0.90 - inclusive_tolerance,
        "overall_success_drop_at_most_0_05":
            float((fq_overall - f8_overall).mean())
            >= -0.05 - inclusive_tolerance,
        "depth_gain_ratio_lower_gt_0_90": bool(
            depth["valid"] and depth["one_sided_95_lower"] > 0.90
        ),
        "all_provenance_safety_sentinel_gates": bool(safety_gates_pass),
    }
    return {
        "quality_gain": quality,
        "useful_speed_ratio": speed,
        "first_window_success": {
            **first_success,
            "fq_min_mean": float(fq_first.mean()),
        },
        "overall_success": {
            **overall_success,
            "fq_min_mean": float(fq_overall.mean()),
        },
        "depth_gain_ratio": depth,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _d0_mechanism(by_arm: Mapping[str, Sequence[Mapping]]) -> dict:
    keys = {
        "event_window_depth_gain": "event_window_depth_gain_mean_sampled",
        "contact_dwell": "contact_dwell_mean_ms_sampled",
        "recontact_count": "recontact_count_mean_sampled",
        "raw_delivered_impulse": "raw_delivered_impulse_mean_n_s_sampled",
        "first_window_success": "first_window_success_rate_sampled",
        "overall_success": "overall_success_rate_sampled",
    }
    endpoints = {
        name: _endpoint(by_arm, "D0", "F8", key)
        for name, key in keys.items()
    }
    try:
        depth_ratio = paired_bootstrap_ratio(
            _values(
                by_arm, "D0", "event_window_depth_gain_mean_sampled"
            ),
            _values(
                by_arm, "F8", "event_window_depth_gain_mean_sampled"
            ),
        )
        depth_preserved = depth_ratio["one_sided_95_lower"] > 0.90
    except ValueError:
        depth_ratio = {"valid": False, "reason": "nonpositive F8 denominator"}
        depth_preserved = False
    d0_first = _values(
        by_arm, "D0", "first_window_success_rate_sampled"
    )
    f8_first = _values(
        by_arm, "F8", "first_window_success_rate_sampled"
    )
    d0_overall = _values(by_arm, "D0", "overall_success_rate_sampled")
    f8_overall = _values(by_arm, "F8", "overall_success_rate_sampled")
    success_guardrails = (
        float(d0_first.mean()) >= 0.90 - 1e-12
        and float((d0_first - f8_first).mean()) >= -0.05 - 1e-12
        and float(d0_overall.mean()) >= 0.90 - 1e-12
        and float((d0_overall - f8_overall).mean()) >= -0.05 - 1e-12
    )
    conditions = {
        "raw_delivered_upper_below_zero": endpoints[
            "raw_delivered_impulse"
        ]["paired_bootstrap_difference"]["one_sided_95_upper"] < -1e-12,
        "dwell_or_recontact_upper_below_zero": (
            endpoints["contact_dwell"]["paired_bootstrap_difference"][
                "one_sided_95_upper"
            ]
            < -1e-12
            or endpoints["recontact_count"]["paired_bootstrap_difference"][
                "one_sided_95_upper"
            ]
            < -1e-12
        ),
        "depth_gain_ratio_lower_gt_0_90": depth_preserved,
        "success_guardrails_pass": success_guardrails,
    }
    claim_allowed = all(conditions.values())
    return {
        "registered_comparison": "D0_minus_F8",
        "interpretation": (
            "mainly selected dwell/recontact"
            if claim_allowed
            else "mechanism conditions not met"
        ),
        "claim_allowed": claim_allowed,
        "conditions": conditions,
        "depth_gain_ratio": depth_ratio,
        "endpoints": endpoints,
    }


def _invalid(*reasons: str) -> dict:
    return {
        "valid": False,
        "invalidation_reasons": [str(reason) for reason in reasons],
    }


def analyze_quality_campaign(
    rows: Sequence[Mapping],
    accepted_evaluation_manifest: Sequence[Mapping],
) -> dict:
    """Validate, recompute, and analyze the complete registered campaign."""

    try:
        validate_quality_campaign_contract(rows, accepted_evaluation_manifest)
    except Exception as error:
        return _invalid(str(error))
    rows = list(rows)
    manifest = list(accepted_evaluation_manifest)
    manifest_index = {
        (str(item["arm"]), int(item["training_seed"])): item
        for item in manifest
    }
    try:
        raw_payloads = []
        for row in rows:
            payload = legacy._sampled_payload_from_bytes(
                Path(row["sampled_trace_path"]).read_bytes()
            )
            if payload.get("schema_version") != 3:
                return _invalid("quality campaign artifacts require schema v3")
            raw_payloads.append(payload)
        common_nail_geometry = _validate_common_nail_geometry(
            [payload["nail_geometry"] for payload in raw_payloads]
        )
        identity_reasons = []
        for row, payload in zip(rows, raw_payloads, strict=True):
            raw = str(row["treatment"])
            label = _LABEL_BY_RAW_TREATMENT[raw]
            accepted = manifest_index[(label, int(row["training_seed"]))]
            identity_reasons.extend(
                _quality_payload_identity_reasons(row, accepted, payload)
            )
        if identity_reasons:
            return _invalid(*identity_reasons)
        payloads = list(legacy._verified_sampled_artifact_payloads(rows))
    except Exception as error:
        return _invalid(str(error))

    seed_aggregates = []
    invalid_reasons: list[str] = []
    for row, payload in zip(rows, payloads, strict=True):
        raw = str(row["treatment"])
        label = _LABEL_BY_RAW_TREATMENT[raw]
        accepted = manifest_index[(label, int(row["training_seed"]))]
        if tuple(payload.get("impulse_limits_n_m_s", ())) != tuple(IMPULSE_LIMITS):
            invalid_reasons.append(
                f"{label}/seed{row['training_seed']}: impulse cap mismatch"
            )
        metrics = []
        for trace in payload["episodes"]:
            try:
                metrics.append(
                    analyze_quality_episode(
                        trace,
                        nail_geometry=payload["nail_geometry"],
                        impulse_limits_n_m_s=payload["impulse_limits_n_m_s"],
                    )
                )
            except Exception as error:
                invalid_reasons.append(
                    f"{label}/seed{row['training_seed']}: {error}"
                )
                break
        if len(metrics) != 512:
            continue
        aggregate = aggregate_quality_episode_metrics(
            metrics, expected_episode_count=512
        )
        aggregate["valid_contact_coordinates_sampled"] = [
            {
                "treatment": label,
                "raw_treatment": raw,
                "training_seed": int(row["training_seed"]),
                **coordinate,
            }
            for coordinate in aggregate["valid_contact_coordinates_sampled"]
        ]
        if aggregate["qvel_nonfinite_rate_sampled"] != 0.0:
            invalid_reasons.append(
                f"{label}/seed{row['training_seed']}: nonfinite qvel"
            )
        seed_aggregates.append(
            {
                "treatment": label,
                "raw_treatment": raw,
                "training_seed": int(row["training_seed"]),
                **aggregate,
            }
        )
    if invalid_reasons:
        return _invalid(*invalid_reasons)

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for aggregate in seed_aggregates:
        by_arm[aggregate["treatment"]].append(aggregate)
    primary = {}
    raw_mwu = {}
    for name, treatment, control in PRIMARY_CONTRASTS:
        treatment_values = _values(
            by_arm, treatment, "first_contact_quality_sampled"
        )
        control_values = _values(by_arm, control, "first_contact_quality_sampled")
        exact = legacy.exact_seed_tests(treatment_values, control_values)
        interval = _paired_bootstrap_difference(
            treatment_values, control_values
        )
        raw_mwu[name] = exact["mwu_two_sided_exact_p"]
        primary[name] = {
            "treatment": treatment,
            "control": control,
            "endpoint": "first_contact_quality_sampled",
            "arm_means": {
                "treatment": float(treatment_values.mean()),
                "control": float(control_values.mean()),
            },
            "paired_differences": (treatment_values - control_values).tolist(),
            "paired_bootstrap_difference": interval,
            "mann_whitney": exact,
            "paired_sign_flip": exact_paired_sign_flip(
                treatment_values, control_values
            ),
            "decision_basis": "holm_mann_whitney_only",
            "decision_gates": ["holm_adjusted_mann_whitney_p_le_0_05"],
        }
    holm = holm_adjust(raw_mwu)
    for name, adjusted in holm.items():
        primary[name]["holm_adjusted_mann_whitney_p"] = adjusted
        primary[name]["statistically_distinguishable"] = adjusted <= 0.05

    safety_gates_pass = all(
        int(item[sentinel]) == 0
        for item in manifest
        for sentinel in SENTINELS
    )
    valid_contact_coordinates = [
        coordinate
        for aggregate in seed_aggregates
        for coordinate in aggregate["valid_contact_coordinates_sampled"]
    ]
    return {
        "valid": True,
        "campaign": CAMPAIGN_NAME,
        "seed_aggregates": seed_aggregates,
        "valid_contact_coordinates": valid_contact_coordinates,
        "nail_geometry": common_nail_geometry,
        "primary_contrasts": primary,
        "holm_mann_whitney_family": holm,
        "fq_min_practical_acceptance": _fq_min_practical_acceptance(
            by_arm, safety_gates_pass=safety_gates_pass
        ),
        "d0_mechanism": _d0_mechanism(by_arm),
        "hardware_qvel_rail_rad_s": legacy.HARDWARE_QVEL_LIMIT_RAD_S,
        "invalidation_reasons": [],
    }
