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

from evaluation.analysis import first_strike_campaign as legacy


CAMPAIGN_NAME = "fq4x8"
LABELS = ("F8", "F0", "D0", "FQ-min")
SEEDS = tuple(range(8, 16))
RAW_TREATMENT = {"F8": "F8", "F0": "F0", "D0": "D0", "FQ-min": "FQ"}
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
    "F8": "actual_event_linear",
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
    if np.any(control <= 0.0):
        raise ValueError("ratio bootstrap requires a positive denominator")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, treatment.size, size=(samples, treatment.size))
    ratios = treatment[indices].mean(axis=1) / control[indices].mean(axis=1)
    return {
        "estimate": float(treatment.mean() / control.mean()),
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

    rows = list(rows)
    manifest = list(accepted_evaluation_manifest)
    if len(rows) != 32:
        raise ValueError("quality campaign requires exactly 32 evaluation rows")
    if len(manifest) != 32:
        raise ValueError("quality campaign requires 32 accepted-evaluation rows")

    row_index = {
        (str(row.get("treatment")), int(row.get("training_seed", -1))): row
        for row in rows
    }
    manifest_index = {
        (str(row.get("treatment")), int(row.get("training_seed", -1))): row
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
        for key in (
            "treatment",
            "raw_treatment",
            "short",
            "task",
            "training_seed",
            "impact_weight",
            "delivered_weight",
            "impact_reader",
            "delivered_reader",
            "speed_normalizer_m_s",
            "fixed_impedance_signature_sha256",
            "fixed_action_signature_sha256",
            "impulse_limits_n_m_s",
            "training_iterations",
            "training_num_envs",
            "checkpoint_filename",
            "evaluation_num_envs",
            "evaluation_episodes_per_env",
            "decision_fields",
        ):
            if accepted.get(key) != frozen[key]:
                category = {
                    "raw_treatment": "treatment",
                    "impact_weight": "weights",
                    "delivered_weight": "weights",
                    "impact_reader": "reader",
                    "delivered_reader": "reader",
                    "speed_normalizer_m_s": "normalizer",
                    "fixed_impedance_signature_sha256": "gains",
                    "fixed_action_signature_sha256": "action",
                    "impulse_limits_n_m_s": "caps",
                    "training_iterations": "training budget",
                    "training_num_envs": "training budget",
                    "checkpoint_filename": "checkpoint",
                    "evaluation_num_envs": "quota",
                    "evaluation_episodes_per_env": "quota",
                    "decision_fields": "decision fields must use *_sampled",
                }.get(key, key)
                raise ValueError(f"{prefix}: frozen {category} mismatch")

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
        quota = (
            row.get("num_envs"),
            row.get("episodes_per_env_sampled"),
            row.get("n_episodes_sampled"),
        )
        if quota != (256, 2, 512):
            raise ValueError(f"{prefix}: strict 256x2 quota mismatch")
        if float(row.get("imp_max_p", np.nan)) != 0.0:
            raise ValueError(f"{prefix}: imp_max_p must be zero")
        if Path(str(row.get("checkpoint_path", ""))).name != "model_499.pt":
            raise ValueError(f"{prefix}: checkpoint mismatch")
        for dirty_key in ("git_dirty", "asset_git_dirty"):
            try:
                row_dirty = legacy._parse_bool(row.get(dirty_key))
                accepted_dirty = legacy._parse_bool(accepted.get(dirty_key))
            except ValueError as error:
                raise ValueError(f"{prefix}: invalid dirty provenance") from error
            if row_dirty or accepted_dirty:
                raise ValueError(f"{prefix}: dirty provenance is forbidden")
        for sentinel in SENTINELS:
            if int(accepted.get(sentinel, -1)) != 0:
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
            "campaign_config_sha256",
            "treatment_config_sha256",
            "nail_asset_sha256",
            "sampled_trace_digest",
            "sampled_trace_artifact_sha256",
        ):
            if not legacy._is_hex_digest(accepted.get(key), 64):
                raise ValueError(f"{prefix}: invalid manifest hash for {key}")
            if accepted[key] != row[key]:
                raise ValueError(f"{prefix}: manifest binding mismatch for {key}")
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
        # Code revision is independently pinned, not required to equal the
        # training revision: a persistence/provenance-only fix can land in
        # the evaluation checkout after training froze. The evaluation
        # checkout itself must still match the accepted attempt's pinned
        # revision (revision == accepted_revision) for BOTH code and asset;
        # only code additionally may differ from the training checkout --
        # asset (nail/scene geometry) may not.
        for prefix_key, training_key, require_training_match in (
            ("git", "training_code_revision", False),
            ("asset_git", "training_asset_revision", True),
        ):
            revision = row.get(f"{prefix_key}_revision")
            training_revision = row.get(training_key)
            accepted_revision = accepted.get(f"{prefix_key}_revision")
            if not all(
                legacy._is_hex_digest(value, 40)
                for value in (revision, training_revision, accepted_revision)
            ):
                raise ValueError(f"{prefix}: invalid {prefix_key} revision")
            if revision != accepted_revision:
                raise ValueError(f"{prefix}: {prefix_key} revision binding mismatch")
            if require_training_match and revision != training_revision:
                raise ValueError(f"{prefix}: {prefix_key} revision binding mismatch")
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
    for label in LABELS:
        accepted_arm = [
            item for item in manifest if item.get("treatment") == label
        ]
        raw_arm = [
            row for row in rows if row.get("treatment") == RAW_TREATMENT[label]
        ]
        for key in (
            "training_config_sha256",
            "evaluation_config_sha256",
            "training_treatment_reward_sha256",
            "evaluation_treatment_reward_sha256",
            "campaign_config_sha256",
            "treatment_config_sha256",
        ):
            source = (
                raw_arm
                if key in {"campaign_config_sha256", "treatment_config_sha256"}
                else accepted_arm
            )
            if len({str(item.get(key)) for item in source}) != 1:
                raise ValueError(
                    f"{label}: per-treatment {key} identity mismatch"
                )


def _quality_payload_identity_reasons(
    row: Mapping, accepted: Mapping, payload: Mapping
) -> list[str]:
    """Cross-bind schema-v3 payload identities to row and accepted manifest."""

    prefix = f"{accepted.get('treatment')}/seed{accepted.get('training_seed')}"
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
            str(accepted.get("treatment"))
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

    try:
        found = np.asarray(physical["quality_found_count"], dtype=float)
        forces = np.asarray(physical["quality_normal_force_n"], dtype=float)
        positions = np.asarray(
            physical["quality_contact_position_m"], dtype=float
        )
        normals = np.asarray(physical["quality_contact_normal"], dtype=float)
    except KeyError as error:
        raise ValueError(f"missing raw quality slot: {error.args[0]}") from error
    if (
        found.ndim != 2
        or found.shape[0] != sample_count
        or forces.shape != found.shape
        or positions.shape != (*found.shape, 3)
        or normals.shape != (*found.shape, 3)
    ):
        raise ValueError("raw quality slot shape mismatch")
    if (
        not np.isfinite(found).all()
        or not np.isfinite(forces).all()
        or not np.isfinite(positions).all()
        or not np.isfinite(normals).all()
        or np.any(found < 0.0)
        or not np.equal(found, np.floor(found)).all()
    ):
        raise ValueError("raw quality slots contain nonfinite or invalid values")
    slot_count = found.shape[1]
    try:
        event_overflow = np.asarray(
            event["tracker_contact_quality_overflow"], dtype=bool
        )
    except KeyError as error:
        raise ValueError("missing event quality overflow stream") from error
    overflow = (
        bool(np.any(found > slot_count))
        or event_overflow.shape != (sample_count,)
        or bool(np.any(event_overflow))
        or bool(first.get("contact_quality_overflow", False))
    )
    if overflow:
        raise ValueError("quality instrumentation overflow")
    empty = {
        "valid": False,
        "point": np.zeros(3),
        "error": 0.0,
        "quality": 0.0,
        "normal_axiality": 0.0,
    }
    if onset is None:
        return empty

    weights = np.where(
        found[onset] > 0.0, np.maximum(forces[onset], 0.0), 0.0
    )
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        recomputed = empty
    else:
        point = np.sum(positions[onset] * weights[:, None], axis=0) / total_weight
        _, _, error = _nail_plane(point, nail_geometry)
        radius = float(nail_geometry["nail_radius_m"])
        quality = float(np.clip(1.0 - (error / radius) ** 2, 0.0, 1.0))
        weighted_normal = np.sum(
            normals[onset] * weights[:, None], axis=0
        )
        normal_norm = float(np.linalg.norm(weighted_normal))
        axis = np.asarray(nail_geometry["nail_axis"], dtype=float)
        axis /= np.linalg.norm(axis)
        axiality = (
            float(np.clip(np.dot(weighted_normal / normal_norm, axis), 0.0, 1.0))
            if normal_norm > 0.0
            else 0.0
        )
        recomputed = {
            "valid": True,
            "point": point,
            "error": error,
            "quality": quality,
            "normal_axiality": axiality,
        }

    scalar_streams = {
        "contact_error_m": "tracker_contact_error_m",
        "contact_quality": "tracker_contact_quality",
        "contact_normal_axiality": "tracker_contact_normal_axiality",
    }
    event_values: dict[str, object] = {}
    for first_key, event_key in scalar_streams.items():
        values = np.asarray(event.get(event_key), dtype=float)
        if values.shape != (sample_count,) or not np.isfinite(values).all():
            raise ValueError(f"event quality stream shape mismatch: {event_key}")
        event_values[first_key] = float(values[onset])
    event_points = np.asarray(event.get("tracker_contact_point_w"), dtype=float)
    event_valid = np.asarray(
        event.get("tracker_contact_quality_valid"), dtype=bool
    )
    if (
        event_points.shape != (sample_count, 3)
        or not np.isfinite(event_points).all()
        or event_valid.shape != (sample_count,)
    ):
        raise ValueError("event quality stream shape mismatch")
    event_values["contact_point_w"] = event_points[onset]
    event_values["contact_quality_valid"] = bool(event_valid[onset])
    event_time = np.asarray(
        event.get("tracker_first_contact_time_s"), dtype=float
    )
    if event_time.shape != (sample_count,) or not np.isfinite(event_time).all():
        raise ValueError("event quality stream shape mismatch")
    first_time = float(first.get("first_contact_time_s", np.nan))
    if not np.isfinite(first_time) or not np.isclose(
        event_time[onset], first_time, rtol=0.0, atol=1e-12
    ):
        raise ValueError("quality event/snapshot mismatch for first_contact_time_s")
    expected = {
        "contact_point_w": recomputed["point"],
        "contact_error_m": recomputed["error"],
        "contact_quality": recomputed["quality"],
        "contact_quality_valid": recomputed["valid"],
        "contact_normal_axiality": recomputed["normal_axiality"],
    }
    for key, expected_value in expected.items():
        first_value = first.get(key)
        event_value = event_values[key]
        if isinstance(expected_value, np.ndarray):
            matches_recomputed = np.allclose(
                np.asarray(first_value, dtype=float),
                expected_value,
                rtol=0.0,
                atol=1e-9,
            )
            matches_event = np.allclose(
                np.asarray(event_value, dtype=float),
                np.asarray(first_value, dtype=float),
                rtol=0.0,
                atol=1e-9,
            )
        elif isinstance(expected_value, bool):
            matches_recomputed = first_value is expected_value
            matches_event = event_value is first_value
        else:
            matches_recomputed = np.isclose(
                float(first_value), expected_value, rtol=0.0, atol=1e-9
            )
            matches_event = np.isclose(
                float(event_value), float(first_value), rtol=0.0, atol=1e-9
            )
        if not matches_recomputed:
            raise ValueError(f"quality snapshot differs from recomputed {key}")
        if not matches_event:
            raise ValueError(f"quality event/snapshot mismatch for {key}")
    return recomputed


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
    first = trace.get("first_strike", {})
    event = trace.get("event_trace", {})
    required = (
        "contact_error_m",
        "contact_quality",
        "contact_quality_valid",
        "contact_quality_overflow",
        "contact_point_w",
        "first_contact_time_s",
        "contact_normal_axiality",
    )
    missing = [key for key in required if key not in first]
    if missing:
        raise ValueError(f"missing quality field: {missing[0]}")
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
    onset_raw = first.get("accepted_onset_index")
    onset = None if onset_raw is None else int(onset_raw)
    if onset is not None and (onset < 0 or onset >= contact.size):
        raise ValueError("quality onset index is malformed")
    snapshot = _quality_snapshot_from_raw_slots(
        physical,
        event,
        first,
        onset=onset,
        sample_count=contact.size,
        nail_geometry=nail_geometry,
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
        (str(item["treatment"]), int(item["training_seed"])): item
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
            label = "FQ-min" if raw == "FQ" else raw
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
        label = "FQ-min" if raw == "FQ" else raw
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
