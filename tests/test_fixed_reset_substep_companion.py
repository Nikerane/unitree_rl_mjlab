"""Contracts for the deterministic 56-policy 500 Hz companion."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.analysis import fixed_reset_500hz_companion as companion
from evaluation.analysis import fixed_reset_video_library as video_library
from scripts import diag_impulse_trace as diag


INVENTORY_FIELDS = (
    "campaign",
    "arm",
    "short",
    "task",
    "training_seed",
    "checkpoint_path",
    "checkpoint_cache_path",
    "checkpoint_sha256",
    "training_code_revision",
    "training_asset_revision",
    "accepted_training_manifest_sha256",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inventory(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=INVENTORY_FIELDS, delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    return _sha256(path)


def _inventory_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Path], list[dict[str, str]], str]:
    roots = {
        "fq4x8": tmp_path / "checkpoints" / "fq4x8",
        "fq3x8": tmp_path / "checkpoints" / "fq3x8",
    }
    rows: list[dict[str, str]] = []
    for campaign, arms in video_library.EXPECTED.items():
        for arm, seeds in arms.items():
            for seed in seeds:
                relative = Path(f"{arm.lower()}-seed{seed}") / "model_499.pt"
                checkpoint = roots[campaign] / relative
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"{campaign}/{arm}/{seed}".encode())
                rows.append(
                    {
                        "campaign": campaign,
                        "arm": arm,
                        "short": arm.lower(),
                        "task": video_library.expected_task(campaign, arm),
                        "training_seed": str(seed),
                        "checkpoint_path": f"/remote/{relative}",
                        "checkpoint_cache_path": str(relative),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "training_code_revision": "a" * 40,
                        "training_asset_revision": "b" * 40,
                        "accepted_training_manifest_sha256": "c" * 64,
                    }
                )
    inventory = tmp_path / "checkpoint_inventory.tsv"
    digest = _write_inventory(inventory, rows)
    sidecar = inventory.with_suffix(inventory.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {inventory.name}\n", encoding="utf-8")
    return inventory, sidecar, roots, rows, digest


def test_inventory_validation_resolves_only_campaign_cache_paths(
    tmp_path: Path,
) -> None:
    inventory, sidecar, roots, _rows, digest = _inventory_fixture(tmp_path)

    loaded = companion.load_checkpoint_inventory(
        inventory,
        checkpoint_roots=roots,
        expected_sha256=digest,
        sidecar_path=sidecar,
    )

    assert len(loaded) == 56
    assert len({row["checkpoint_sha256"] for row in loaded}) == 56
    first = loaded[0]
    assert first["checkpoint_file"] == str(
        roots[first["campaign"]] / first["checkpoint_cache_path"]
    )
    assert isinstance(first["training_seed"], int)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("absolute_path", "relative"),
        ("traversal", "traversal"),
        ("missing_file", "missing checkpoint"),
        ("hash_drift", "checkpoint SHA"),
        ("task_drift", "task"),
        ("missing_member", "membership"),
        ("duplicate_identity", "membership"),
        ("duplicate_hash", "uniqueness"),
    ],
)
def test_inventory_validation_rejects_path_hash_task_or_membership_mutations(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    inventory, sidecar, roots, rows, _digest = _inventory_fixture(tmp_path)
    if mutation == "absolute_path":
        rows[0]["checkpoint_cache_path"] = "/absolute/model.pt"
    elif mutation == "traversal":
        rows[0]["checkpoint_cache_path"] = "../escape/model.pt"
    elif mutation == "missing_file":
        (roots[rows[0]["campaign"]] / rows[0]["checkpoint_cache_path"]).unlink()
    elif mutation == "hash_drift":
        checkpoint = (
            roots[rows[0]["campaign"]] / rows[0]["checkpoint_cache_path"]
        )
        checkpoint.write_bytes(b"drift")
    elif mutation == "task_drift":
        rows[0]["task"] = rows[1]["task"]
        if rows[0]["task"] == video_library.expected_task(
            rows[0]["campaign"], rows[0]["arm"]
        ):
            rows[0]["task"] = "wrong-task"
    elif mutation == "missing_member":
        rows.pop()
    elif mutation == "duplicate_identity":
        for key in ("campaign", "arm", "training_seed", "task"):
            rows[1][key] = rows[0][key]
    elif mutation == "duplicate_hash":
        first = roots[rows[0]["campaign"]] / rows[0]["checkpoint_cache_path"]
        second = roots[rows[1]["campaign"]] / rows[1]["checkpoint_cache_path"]
        second.write_bytes(first.read_bytes())
        rows[1]["checkpoint_sha256"] = rows[0]["checkpoint_sha256"]
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    digest = _write_inventory(inventory, rows)
    sidecar.write_text(f"{digest}  {inventory.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        companion.load_checkpoint_inventory(
            inventory,
            checkpoint_roots=roots,
            expected_sha256=digest,
            sidecar_path=sidecar,
        )


def test_inventory_sha_and_sidecar_drift_fail_closed(tmp_path: Path) -> None:
    inventory, sidecar, roots, _rows, digest = _inventory_fixture(tmp_path)
    with pytest.raises(ValueError, match="inventory SHA"):
        companion.load_checkpoint_inventory(
            inventory,
            checkpoint_roots=roots,
            expected_sha256="0" * 64,
            sidecar_path=sidecar,
        )

    sidecar.write_text(f"{'f' * 64}  {inventory.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        companion.load_checkpoint_inventory(
            inventory,
            checkpoint_roots=roots,
            expected_sha256=digest,
            sidecar_path=sidecar,
        )


def _payload(
    *,
    substeps: int = 40,
    onset: int | None = 32,
    finalized: int | None = 36,
    quality_available: bool = True,
) -> dict[str, np.ndarray]:
    controls = max(1, substeps // 10)
    assert substeps == controls * 10
    positions = np.zeros((substeps, 3), dtype=np.float64)
    positions[:, 2] = np.linspace(0.2, 0.1, substeps)
    contact = np.zeros(substeps, dtype=bool)
    started = np.zeros(substeps, dtype=bool)
    finalized_stream = np.zeros(substeps, dtype=bool)
    productive = np.zeros(substeps, dtype=bool)
    reason = np.zeros(substeps, dtype=np.int64)
    if onset is not None:
        contact[onset : min(onset + 3, substeps)] = True
        started[onset:] = True
    if finalized is not None:
        finalized_stream[finalized:] = True
        productive[finalized:] = True
        reason[finalized:] = 1
    pre = np.zeros((substeps, 6), dtype=np.float64)
    post = np.zeros_like(pre)
    payload: dict[str, np.ndarray] = {
        "t_s": np.arange(substeps, dtype=np.float64) * 0.002,
        "head_position_m": positions,
        "head_velocity_m_s": np.zeros((substeps, 3), dtype=np.float64),
        "contact": contact,
        "nail_position_m": np.tile([0.0, 0.0, 0.1], (substeps, 1)),
        "nail_depth_m": np.zeros(substeps, dtype=np.float64),
        "joint_velocity_rad_s": pre,
        "joint_position_rad": np.zeros((substeps, 6), dtype=np.float64),
        "joint_velocity_post_integration_rad_s": post,
        "nail_depth_post_integration_m": np.zeros(
            substeps, dtype=np.float64
        ),
        "qfrc_constraint_abs": np.zeros(
            (substeps, 6), dtype=np.float64
        ),
        "lambda_windowed_constraint_read_n_m_s": np.zeros(
            (substeps, 6), dtype=np.float64
        ),
        "delivered_impulse_n_s": np.zeros(substeps, dtype=np.float64),
        "axial_force_n": contact.astype(np.float64) * 10.0,
        "control_step_index": np.repeat(
            np.arange(controls, dtype=np.int64), 10
        ),
        "tracker_started": started,
        "tracker_finalized": finalized_stream,
        "tracker_productive": productive,
        "tracker_reason": reason,
        "tracker_v_precontact_m_s": np.where(started, 1.0, 0.0),
        "tracker_delivered_n_s": np.zeros(substeps, dtype=np.float64),
        "tracker_delivered_transverse_n_s": np.zeros(
            substeps, dtype=np.float64
        ),
        "tracker_peak_depth_m": np.zeros(substeps, dtype=np.float64),
        "tracker_depth_at_contact_m": np.zeros(
            substeps, dtype=np.float64
        ),
        "tracker_first_contact_time_s": np.where(
            started, (0 if onset is None else onset + 1) * 0.002, 0.0
        ),
        "control_step": np.arange(controls, dtype=np.int64),
        "action": np.zeros((controls, 3), dtype=np.float32),
        "strike_phase_control": np.zeros(controls, dtype=np.float64),
        "strike_ref_error_control_m": np.zeros(
            (controls, 3), dtype=np.float64
        ),
        "first_strike_available": np.asarray(True),
        "quality_available": np.asarray(quality_available),
    }
    if quality_available:
        payload.update(
            {
                "tracker_contact_point_w": np.zeros(
                    (substeps, 3), dtype=np.float64
                ),
                "tracker_contact_error_m": np.where(
                    started, 0.002, 0.0
                ),
                "tracker_contact_quality": np.where(
                    started, 0.75, 0.0
                ),
                "tracker_contact_quality_valid": started.copy(),
                "tracker_contact_quality_overflow": np.zeros(
                    substeps, dtype=bool
                ),
                "tracker_contact_normal_axiality": np.where(
                    started, 0.9, 0.0
                ),
            }
        )
    return payload


def _row(tmp_path: Path) -> dict[str, object]:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    return {
        "campaign": "fq3x8",
        "arm": "FQ",
        "training_seed": 23,
        "task": video_library.expected_task("fq3x8", "FQ"),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_cache_path": "model.pt",
        "checkpoint_file": str(checkpoint),
        "training_asset_revision": "d" * 40,
    }


def _leaf_identity(row: dict[str, object]) -> dict[str, object]:
    return {
        "mode": "checkpoint",
        "campaign": row["campaign"],
        "arm": row["arm"],
        "training_seed": row["training_seed"],
        "task": row["task"],
        "checkpoint_file": row["checkpoint_file"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "fixed_reset_envelope": "fixed-reset.json",
        "reset_state_digest": video_library.APPROVED_FIXED_RESET_DIGEST,
        "code_revision": "c" * 40,
        "asset_revision": "d" * 40,
        "terminal_reason": "step_limit",
        "j_limit_n_m_s": [1.64, 3.28, 1.64, 1.64, 1.64, 1.64],
    }


def test_valid_resume_leaf_is_identity_bound_and_complete(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    leaf = tmp_path / "leaf"
    diag._write_trace_leaf(
        leaf,
        _payload(),
        identity=_leaf_identity(row),
        physics_dt=0.002,
    )

    metadata = companion.validate_resume_leaf(
        leaf,
        row,
        fixed_reset_envelope="fixed-reset.json",
        code_revision="c" * 40,
        asset_revision="d" * 40,
        requested_controls=4,
    )

    assert metadata["timing"]["physics_dt_s"] == 0.002
    assert metadata["timing"]["control_decimation"] == 10
    assert metadata["timing"]["control_step_count"] == 4


@pytest.mark.parametrize("field", ["checkpoint_sha256", "code_revision"])
def test_resume_leaf_rejects_digest_consistent_identity_drift(
    tmp_path: Path,
    field: str,
) -> None:
    row = _row(tmp_path)
    identity = _leaf_identity(row)
    identity[field] = (
        "e" * 64 if field == "checkpoint_sha256" else "e" * 40
    )
    leaf = tmp_path / "leaf"
    diag._write_trace_leaf(
        leaf, _payload(), identity=identity, physics_dt=0.002
    )

    with pytest.raises(ValueError, match=field):
        companion.validate_resume_leaf(
            leaf,
            row,
            fixed_reset_envelope="fixed-reset.json",
            code_revision="c" * 40,
            asset_revision="d" * 40,
            requested_controls=4,
        )


def _metric_trace(sample_count: int = 8) -> dict[str, np.ndarray]:
    payload = _payload(
        substeps=40,
        onset=3,
        finalized=5,
        quality_available=True,
    )
    payload = {
        key: (
            value[:sample_count]
            if np.asarray(value).ndim >= 1
            and np.asarray(value).shape[0] == 40
            else value
        )
        for key, value in payload.items()
        if key
        not in {
            "control_step",
            "action",
            "strike_phase_control",
            "strike_ref_error_control_m",
            "control_step_index",
        }
    }
    positions = np.zeros((sample_count, 3), dtype=np.float64)
    positions[:4] = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
    ]
    payload["head_position_m"] = positions
    payload["head_velocity_m_s"] = np.zeros((sample_count, 3))
    payload["head_velocity_m_s"][3] = [0.3, 0.4, -2.0]
    payload["nail_position_m"] = np.zeros((sample_count, 3))
    payload["contact"][:] = False
    payload["contact"][3:5] = True
    if sample_count > 7:
        payload["contact"][7] = True
    payload["tracker_started"][:] = False
    payload["tracker_started"][3:] = True
    payload["tracker_finalized"][:] = False
    payload["tracker_finalized"][5:] = True
    payload["tracker_productive"][:] = False
    payload["tracker_productive"][5:] = True
    payload["tracker_reason"][:] = 0
    payload["tracker_reason"][5:] = 1
    payload["tracker_contact_quality_valid"][:] = False
    payload["tracker_contact_quality_valid"][3:] = True
    payload["tracker_contact_quality"][:] = 0.0
    payload["tracker_contact_quality"][3:] = 0.75
    payload["tracker_contact_quality_overflow"][:] = False
    payload["joint_velocity_rad_s"] = np.zeros((sample_count, 6))
    payload["joint_velocity_post_integration_rad_s"] = np.zeros(
        (sample_count, 6)
    )
    payload["joint_velocity_post_integration_rad_s"][-1, 1] = 3.1415
    payload["lambda_windowed_constraint_read_n_m_s"] = np.zeros(
        (sample_count, 6)
    )
    payload["lambda_windowed_constraint_read_n_m_s"][4] = [
        0.82,
        3.608,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    payload["axial_force_n"] = np.zeros(sample_count)
    payload["axial_force_n"][3:5] = [10.0, 5.0]
    return payload


def test_exact_onset_apex_chord_contact_and_descriptor_metrics() -> None:
    metrics = companion.derive_episode_metrics(_metric_trace(), dt_s=0.002)

    assert metrics["accepted_onset_index"] == 3
    assert metrics["finalization_index"] == 5
    assert metrics["realized_apex_index"] == 1
    assert metrics["apex_to_onset_path_ratio_3d"] == pytest.approx(
        np.sqrt(5.0)
    )
    assert metrics["apex_to_onset_chord_rms_m"] == pytest.approx(
        1.0 / np.sqrt(3.0)
    )
    assert metrics["apex_to_onset_chord_max_m"] == pytest.approx(1.0)
    assert metrics["descent_geometry_valid"] is True
    assert metrics["descent_c_rms"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert metrics["descent_b_rms_m"] == pytest.approx(
        1.0 / np.sqrt(2.0)
    )
    assert metrics["descent_c_max"] == pytest.approx(1.0)
    assert metrics["descent_tortuosity"] == pytest.approx(np.sqrt(5.0))
    assert metrics["descent_progress_sign_changes"] == 0
    assert metrics["accepted_contact_run_duration_s"] == pytest.approx(0.004)
    assert metrics["post_event_recontact"] is True
    assert metrics["onset_downward_axial_velocity_m_s"] == pytest.approx(2.0)
    assert metrics["onset_lateral_velocity_m_s"] == pytest.approx(0.5)
    assert metrics["onset_approach_angle_deg"] == pytest.approx(
        np.degrees(np.arctan2(0.5, 2.0))
    )
    assert metrics["descriptor_force_peak_n"] == pytest.approx(10.0)
    assert metrics[
        "descriptor_force_time_concentration_peak_sample_fraction"
    ] == pytest.approx(2.0 / 3.0)
    assert metrics["descriptor_contact_duty_first_strike_window"] == pytest.approx(
        2.0 / 3.0
    )


def test_descent_phenotype_requires_three_distinct_samples_and_50mm_chord() -> None:
    duplicate = _metric_trace()
    duplicate["head_position_m"][2] = duplicate["head_position_m"][1]
    metrics = companion.derive_episode_metrics(duplicate, dt_s=0.002)
    assert metrics["descent_geometry_valid"] is False
    assert metrics["descent_geometry_invalid_reason"] == (
        "fewer_than_three_distinct_samples"
    )
    assert metrics["descent_c_rms"] is None

    short = _metric_trace()
    short["head_position_m"][:4] *= 0.01
    metrics = companion.derive_episode_metrics(short, dt_s=0.002)
    assert metrics["descent_geometry_valid"] is False
    assert metrics["descent_geometry_invalid_reason"] == "chord_below_0.050m"
    # The legacy unweighted absolute output remains available, but is not the
    # primary normalized curvature phenotype.
    assert metrics["apex_to_onset_chord_rms_m"] is not None
    assert metrics["descent_c_rms"] is None


def test_descent_progress_sign_changes_ignore_zero_progress() -> None:
    trace = _metric_trace(sample_count=9)
    trace["tracker_started"][:] = False
    trace["tracker_started"][5:] = True
    trace["contact"][:] = False
    trace["contact"][5:7] = True
    trace["tracker_finalized"][:] = False
    trace["tracker_finalized"][7:] = True
    trace["tracker_productive"][:] = False
    trace["tracker_productive"][7:] = True
    trace["tracker_reason"][:] = 0
    trace["tracker_reason"][7:] = 1
    trace["tracker_contact_quality_valid"][:] = False
    trace["tracker_contact_quality_valid"][5:] = True
    trace["head_position_m"][:6] = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 0.6],
        [0.1, 0.0, 0.6],
        [0.1, 0.0, 0.8],
        [0.0, 0.0, 0.0],
    ]

    metrics = companion.derive_episode_metrics(trace, dt_s=0.002)

    assert metrics["descent_geometry_valid"] is True
    assert metrics["descent_progress_sign_changes"] == 2


def test_accepted_onset_must_be_live_first_raw_contact() -> None:
    trace = _metric_trace()
    trace["contact"][1] = True
    with pytest.raises(ValueError, match="before tracker acceptance"):
        companion.derive_episode_metrics(trace, dt_s=0.002)

    trace = _metric_trace()
    trace["contact"][3] = False
    with pytest.raises(ValueError, match="in-contact"):
        companion.derive_episode_metrics(trace, dt_s=0.002)


def test_qvel_canonical_coherence_boundary_and_positive_epsilon() -> None:
    trace = _metric_trace()
    legal = companion.derive_episode_metrics(trace, dt_s=0.002)
    assert legal["qvel_max_abs_rad_s"] == pytest.approx(3.1415)
    assert legal["qvel_violation"] is False

    trace["joint_velocity_post_integration_rad_s"][-1, 1] = np.nextafter(
        3.1415, np.inf
    )
    illegal = companion.derive_episode_metrics(trace, dt_s=0.002)
    assert illegal["qvel_violation"] is True

    trace = _metric_trace()
    trace["joint_velocity_post_integration_rad_s"][0, 0] = 0.1
    with pytest.raises(ValueError, match="temporally coherent"):
        companion.derive_episode_metrics(trace, dt_s=0.002)


def test_cap_vector_order_and_single_joint_overrun() -> None:
    metrics = companion.derive_episode_metrics(_metric_trace(), dt_s=0.002)

    assert metrics["lambda_cap_ratio_joint1"] == pytest.approx(0.5)
    assert metrics["lambda_cap_ratio_joint2"] == pytest.approx(1.1)
    assert metrics["lambda_cap_ratio_joint3"] == 0.0
    assert metrics["lambda_cap_ratio_worst"] == pytest.approx(1.1)
    assert metrics["lambda_cap_ratio_worst_joint"] == "joint2"


def test_no_contact_one_point_and_nonfinite_traces_are_explicit() -> None:
    no_contact = _metric_trace()
    no_contact["contact"][:] = False
    no_contact["tracker_started"][:] = False
    no_contact["tracker_finalized"][:] = False
    no_contact["tracker_productive"][:] = False
    no_contact["tracker_reason"][:] = 0
    no_contact["tracker_contact_quality_valid"][:] = False
    metrics = companion.derive_episode_metrics(no_contact, dt_s=0.002)
    assert metrics["accepted_onset_index"] is None
    assert metrics["apex_to_onset_path_ratio_3d"] is None

    one = {
        key: (
            np.asarray(value)
            if np.asarray(value).ndim == 0
            else np.asarray(value)[:1]
        )
        for key, value in no_contact.items()
    }
    one["first_strike_available"] = np.asarray(True)
    one["quality_available"] = np.asarray(True)
    metrics = companion.derive_episode_metrics(one, dt_s=0.002)
    assert metrics["trace_sample_count"] == 1

    bad = _metric_trace()
    bad["head_position_m"][2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        companion.derive_episode_metrics(bad, dt_s=0.002)


def test_quality_unavailable_is_distinct_from_measured_zero() -> None:
    unavailable = _metric_trace()
    unavailable["quality_available"] = np.asarray(False)
    for key in tuple(unavailable):
        if key.startswith("tracker_contact_"):
            unavailable.pop(key)
    result = companion.derive_episode_metrics(unavailable, dt_s=0.002)
    assert result["quality_available"] is False
    assert result["quality"] is None
    assert companion.hardware_eligibility(result)[0] is False

    measured = _metric_trace()
    measured["tracker_contact_quality"][3:] = 0.0
    result = companion.derive_episode_metrics(measured, dt_s=0.002)
    assert result["quality_available"] is True
    assert result["quality"] == 0.0


def test_terminal_funnel_uses_exact_onset_positions() -> None:
    trace = _payload(onset=32, finalized=36)
    positions = trace["head_position_m"]
    positions[:, 0] = np.maximum(
        0.0, (32 - np.arange(len(positions))) * 0.001
    )
    positions[:, 1] = 0.0
    trace["head_velocity_m_s"][32] = [0.0, 0.0, -1.0]

    metrics = companion.derive_episode_metrics(trace, dt_s=0.002)

    assert metrics["terminal_funnel_available"] is True
    assert metrics["terminal_error_m"] == pytest.approx(0.0)
    assert metrics["terminal_contraction_ratio"] is not None


def _rank_row(seed: int, curvature: float, eligible: bool = True) -> dict:
    return {
        "campaign": "fq3x8",
        "arm": "FQ",
        "training_seed": seed,
        "checkpoint_sha256": f"{seed:064x}",
        "apex_to_onset_path_ratio_3d": curvature,
        "descent_geometry_valid": True,
        "descent_c_rms": curvature,
        "descent_b_rms_m": curvature,
        "descent_c_max": curvature,
        "descent_tortuosity": 1.0 + curvature / 10.0,
        "descent_progress_sign_changes": 0,
        "accepted_onset_index": 10,
        "tracker_finalized": True,
        "tracker_productive": True,
        "tracker_reason": 1,
        "quality_available": True,
        "quality_valid": True,
        "quality_overflow": False,
        "finite_trace": True,
        "qvel_violation": not eligible,
    }


def _covariates() -> dict:
    values = {
        16: 0.0,
        17: 1.0,
        18: 1.0,
        19: 0.0,
        20: 0.5,
        21: 0.5,
        22: 0.5,
        23: 0.5,
    }
    return {
        "schema_version": 1,
        "campaign": "fq3x8",
        "seeds": list(range(16, 24)),
        "seed_metrics": {
            "FQ": {
                str(seed): {
                    "first_contact_quality_sampled": value,
                    "first_window_useful_speed_mean_sampled": value,
                    "event_window_depth_gain_mean_sampled": value * 0.001,
                    "first_window_success_rate_sampled": value * 0.05,
                    "overall_success_rate_sampled": 1.0,
                }
                for seed, value in values.items()
            }
        },
    }


def test_ranking_and_standardized_margin_pairing_are_deterministic() -> None:
    rows = [
        _rank_row(seed, 0.01 * float(seed - 15))
        for seed in range(16, 20)
    ]
    first = companion.rank_and_pair_fq(rows, _covariates())
    second = companion.rank_and_pair_fq(list(reversed(rows)), _covariates())

    assert first == second
    assert [row["training_seed"] for row in first["simulation_ranking"]] == [
        16,
        17,
        18,
        19,
    ]
    assert [row["training_seed"] for row in first["hardware_ranking"]] == [
        16,
        17,
        18,
        19,
    ]
    assert first["pairing"]["status"] == "SELECTED"
    assert {
        (pair["lower_seed"], pair["higher_seed"])
        for pair in first["pairing"]["pairs"]
    } == {(16, 19), (17, 18)}
    assert first["pairing"]["population_covariates_sha256"] == (
        companion.POPULATION_COVARIATES_SHA256
    )


def test_pairing_objective_tie_breaks_by_paired_seed_tuples() -> None:
    rows = [
        _rank_row(seed, 0.01 * float(seed - 15))
        for seed in range(16, 20)
    ]
    checkpoint_hashes = {
        16: "a" * 64,
        17: "b" * 64,
        18: "f" * 64,
        19: "0" * 64,
    }
    for row in rows:
        row["checkpoint_sha256"] = checkpoint_hashes[
            int(row["training_seed"])
        ]
    covariates = _covariates()
    for record in covariates["seed_metrics"]["FQ"].values():
        for field in record:
            record[field] = 0.0

    result = companion.rank_and_pair_fq(rows, covariates)

    assert [
        (pair["lower_seed"], pair["higher_seed"])
        for pair in result["pairing"]["pairs"]
    ] == [(16, 18), (17, 19)]
    assert result["pairing"]["selection_objective"][3] == (
        (16, 18),
        (17, 19),
    )


def test_no_hardware_quartet_is_explicit_but_simulation_ranking_is_banked() -> None:
    rows = [_rank_row(seed, float(seed)) for seed in range(16, 20)]
    rows[-1]["qvel_violation"] = True

    result = companion.rank_and_pair_fq(rows, _covariates())

    assert len(result["simulation_ranking"]) == 4
    assert len(result["hardware_ranking"]) == 3
    assert (
        result["pairing"]["status"]
        == companion.NO_HARDWARE_LEGAL_QUARTET
    )
    assert result["pairing"]["pairs"] == []


def test_hardware_ranking_retains_unsupported_descent_geometry() -> None:
    rows = [_rank_row(seed, float(seed)) for seed in range(16, 20)]
    unsupported = rows[1]
    unsupported["descent_geometry_valid"] = False
    unsupported["descent_geometry_invalid_reason"] = (
        "fewer_than_3_distinct_samples"
    )
    for field in (
        "descent_c_rms",
        "descent_b_rms_m",
        "descent_c_max",
        "descent_tortuosity",
    ):
        unsupported[field] = None

    result = companion.rank_and_pair_fq(rows, _covariates())

    assert len(result["hardware_ranking"]) == 4
    retained = next(
        row
        for row in result["hardware_ranking"]
        if row["training_seed"] == 17
    )
    assert retained["hardware_eligible"] is True
    assert retained["hardware_rank"] is None
    assert retained["descent_c_rms"] is None
    assert retained["selection_exclusion_reason"] == (
        "unsupported_descent_phenotype:fewer_than_3_distinct_samples"
    )
    assert result["pairing"]["geometry_eligible_count"] == 3
    assert (
        result["pairing"]["status"]
        == companion.NO_DISTINCT_CURVATURE_QUARTET
    )


def test_odd_eligible_median_is_unassigned_before_pairing() -> None:
    rows = [
        _rank_row(seed, 0.01 * float(seed - 15))
        for seed in range(16, 21)
    ]

    result = companion.rank_and_pair_fq(rows, _covariates())

    assert result["pairing"]["lower_half_seeds"] == [16, 17]
    assert result["pairing"]["higher_half_seeds"] == [19, 20]
    assert result["pairing"]["unassigned_middle_seeds"] == [18]


def test_no_distinct_curvature_quartet_is_explicit() -> None:
    rows = [_rank_row(seed, 0.01) for seed in range(16, 20)]
    for index, row in enumerate(rows):
        row["descent_b_rms_m"] = 0.001 * index

    result = companion.rank_and_pair_fq(rows, _covariates())

    assert result["pairing"]["status"] == "NO DISTINCT-CURVATURE QUARTET"
    assert result["pairing"]["pairs"] == []


def test_no_margin_matched_quartet_is_explicit() -> None:
    rows = [
        _rank_row(seed, 0.01 * float(seed - 15))
        for seed in range(16, 20)
    ]
    covariates = _covariates()
    for seed in (16, 17):
        covariates["seed_metrics"]["FQ"][str(seed)][
            "first_contact_quality_sampled"
        ] = 0.0
    for seed in (18, 19):
        covariates["seed_metrics"]["FQ"][str(seed)][
            "first_contact_quality_sampled"
        ] = 0.5

    result = companion.rank_and_pair_fq(rows, covariates)

    assert result["pairing"]["status"] == "NO MARGIN-MATCHED QUARTET"
    assert result["pairing"]["pairs"] == []


def test_population_covariate_bytes_and_exact_fq_schema_are_frozen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "covariates.json"
    path.write_text(json.dumps(_covariates(), sort_keys=True), encoding="utf-8")
    digest = _sha256(path)
    loaded = companion.load_population_covariates(
        path, expected_sha256=digest
    )
    assert loaded["seeds"] == list(range(16, 24))

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        companion.load_population_covariates(path, expected_sha256=digest)

    malformed = _covariates()
    malformed["seed_metrics"]["FQ"].pop("23")
    path.write_text(json.dumps(malformed, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="exact FQ seeds"):
        companion.load_population_covariates(
            path, expected_sha256=_sha256(path)
        )


def test_supplied_code_revision_must_equal_current_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = "a" * 40
    monkeypatch.setattr(companion, "_git_revision", lambda: current)

    assert companion.resolve_code_revision(None) == current
    assert companion.resolve_code_revision(current) == current
    with pytest.raises(ValueError, match="current source revision"):
        companion.resolve_code_revision("b" * 40)


def test_batch_command_is_exact_cpu_fixed_reset_companion(tmp_path: Path) -> None:
    row = _row(tmp_path)
    command = companion.build_recorder_command(
        row,
        leaf=tmp_path / "out",
        python_executable="/python",
        recorder_script=Path("/repo/scripts/diag_impulse_trace.py"),
        fixed_reset_envelope=Path("/repo/reset.json"),
        code_revision="c" * 40,
        asset_revision="d" * 40,
    )
    joined = " ".join(command)

    assert command[0:2] == [
        "/python",
        "/repo/scripts/diag_impulse_trace.py",
    ]
    assert "--device cpu" in joined
    assert "--play" in command
    assert "--num-envs 1" in joined
    assert "--nsteps 80" in joined
    assert "--j-limit 1.64,3.28,1.64,1.64,1.64,1.64" in joined
    assert "set_gains" not in joined


def test_all_56_status_rows_survive_one_child_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inventory, _sidecar, _roots, raw_rows, _digest = _inventory_fixture(
        tmp_path
    )
    rows: list[dict[str, object]] = []
    output = tmp_path / "output"
    for raw in raw_rows:
        row = dict(raw)
        row["training_seed"] = int(row["training_seed"])
        row["checkpoint_file"] = str(
            tmp_path / "checkpoints" / row["campaign"] / row[
                "checkpoint_cache_path"
            ]
        )
        rows.append(row)
        if int(row["training_seed"]) == 8 and row["arm"] == "F8":
            continue
        leaf = (
            output
            / str(row["campaign"])
            / str(row["arm"])
            / str(row["training_seed"])
        )
        leaf.mkdir(parents=True)
        (leaf / "trace.npz").touch()
        (leaf / "metadata.json").touch()

    def validate(leaf, *_args, **_kwargs):
        assert Path(leaf).joinpath("trace.npz").is_file()
        return {"timing": {"control_step_count": 1}}

    monkeypatch.setattr(companion, "validate_resume_leaf", validate)

    def runner(command, **_kwargs):
        assert "--training-seed" in command
        return SimpleNamespace(returncode=9, stdout="synthetic child failure")

    with pytest.raises(companion.BatchFailure, match="1 of 56"):
        companion.run_batch(
            rows,
            output_root=output,
            python_executable="/python",
            recorder_script=Path("/repo/scripts/diag_impulse_trace.py"),
            fixed_reset_envelope=Path("fixed-reset.json"),
            code_revision="c" * 40,
            asset_revision="d" * 40,
            runner=runner,
            progress=lambda _message: None,
        )

    with (output / "run_status.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        status = list(csv.DictReader(handle))
    assert len(status) == 56
    failed = [row for row in status if row["status"] == "failed"]
    assert len(failed) == 1
    assert "synthetic child failure" in failed[0]["error"]
    assert sum(row["status"] == "reused" for row in status) == 55


def test_metric_failure_marks_status_and_continues_all_56(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inventory, _sidecar, _roots, raw_rows, _digest = _inventory_fixture(
        tmp_path
    )
    rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for raw in raw_rows:
        row: dict[str, object] = dict(raw)
        row["training_seed"] = int(row["training_seed"])
        row["checkpoint_file"] = str(
            tmp_path / "checkpoints" / row["campaign"] / row[
                "checkpoint_cache_path"
            ]
        )
        rows.append(row)
        status_rows.append(
            {
                "campaign": row["campaign"],
                "arm": row["arm"],
                "training_seed": row["training_seed"],
                "task": row["task"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "status": "completed",
                "error": "",
                "elapsed_s": 0.0,
            }
        )
    output = tmp_path / "output"
    companion._write_csv(output / "run_status.csv", status_rows)
    seen: list[tuple[str, str, int]] = []
    failed_identity = ("fq4x8", "F8", 8)

    monkeypatch.setattr(
        companion, "validate_resume_leaf", lambda *_args, **_kwargs: {}
    )

    def load(leaf):
        path = Path(leaf)
        return {
            "identity": (
                path.parents[1].name,
                path.parent.name,
                int(path.name),
            )
        }

    def derive(payload):
        identity = payload["identity"]
        seen.append(identity)
        if identity == failed_identity:
            raise ValueError("scientific metric failure")
        return {}

    monkeypatch.setattr(companion, "load_trace_payload", load)
    monkeypatch.setattr(companion, "derive_episode_metrics", derive)

    with pytest.raises(companion.BatchFailure, match="analysis"):
        companion.analyze_completed_leaves(
            rows,
            output_root=output,
            fixed_reset_envelope="fixed-reset.json",
            code_revision="c" * 40,
            asset_revision="b" * 40,
        )

    assert len(seen) == 56
    with (output / "run_status.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        status = list(csv.DictReader(handle))
    failed = [
        row
        for row in status
        if (
            row["campaign"],
            row["arm"],
            int(row["training_seed"]),
        )
        == failed_identity
    ]
    assert failed[0]["status"] == "analysis_failed"
    assert "scientific metric failure" in failed[0]["error"]


def test_leaf_revalidation_failure_marks_status_and_continues_all_56(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inventory, _sidecar, _roots, raw_rows, _digest = _inventory_fixture(
        tmp_path
    )
    rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for raw in raw_rows:
        row: dict[str, object] = dict(raw)
        row["training_seed"] = int(row["training_seed"])
        row["checkpoint_file"] = str(
            tmp_path / "checkpoints" / row["campaign"] / row[
                "checkpoint_cache_path"
            ]
        )
        rows.append(row)
        status_rows.append(
            {
                "campaign": row["campaign"],
                "arm": row["arm"],
                "training_seed": row["training_seed"],
                "task": row["task"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "status": "completed",
                "error": "",
                "elapsed_s": 0.0,
            }
        )
    output = tmp_path / "output"
    companion._write_csv(output / "run_status.csv", status_rows)
    validated: list[tuple[str, str, int]] = []
    loaded: list[tuple[str, str, int]] = []
    failed_identity = ("fq4x8", "F8", 8)

    def identity_from_leaf(leaf) -> tuple[str, str, int]:
        path = Path(leaf)
        return (
            path.parents[1].name,
            path.parent.name,
            int(path.name),
        )

    def validate(leaf, *_args, **_kwargs):
        identity = identity_from_leaf(leaf)
        validated.append(identity)
        if identity == failed_identity:
            raise ValueError("resume identity drift")
        return {}

    def load(leaf):
        identity = identity_from_leaf(leaf)
        loaded.append(identity)
        return {}

    monkeypatch.setattr(companion, "validate_resume_leaf", validate)
    monkeypatch.setattr(companion, "load_trace_payload", load)
    monkeypatch.setattr(
        companion, "derive_episode_metrics", lambda _payload: {}
    )

    with pytest.raises(companion.BatchFailure, match="analysis"):
        companion.analyze_completed_leaves(
            rows,
            output_root=output,
            fixed_reset_envelope="fixed-reset.json",
            code_revision="c" * 40,
            asset_revision="b" * 40,
        )

    assert len(validated) == 56
    assert len(loaded) == 55
    with (output / "run_status.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        status = list(csv.DictReader(handle))
    failed = [
        row
        for row in status
        if (
            row["campaign"],
            row["arm"],
            int(row["training_seed"]),
        )
        == failed_identity
    ]
    assert failed[0]["status"] == "analysis_failed"
    assert "resume identity drift" in failed[0]["error"]
