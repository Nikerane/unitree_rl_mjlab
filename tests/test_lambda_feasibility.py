from __future__ import annotations

import json
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.whip.lambda_feasibility import (
    ALLOWED_DELTAS,
    BindingClass,
    LEGACY_WHIP_FILES,
    NONFACE_FORCE_TOL,
    POST_TAPE_ZERO_ACTIONS,
    RELEASE_CONFIRM_SAMPLES,
    RESET_BANK_SIZE,
    RESET_BANK_SEED,
    _assert_authoritative_outside_repositories,
    _continuation_label,
    _determinism_qualification,
    _git_provenance,
    _max_nonface_efc_force,
    _new_result,
    _reference_calibration_qualification,
    _requires_solref_replay,
    _stage0_env_cfg,
    actuator_saturation_summary,
    aggregate_results,
    assert_legacy_realized_digest,
    assert_deterministic_replay,
    assert_exact_zero_off_raw_contact,
    assert_ordinary_shadow_prefix_parity,
    classify_binding,
    compare_solref_results,
    contact_signal_liveness,
    deterministic_trace_digest,
    first_impact_lobe_end,
    first_release,
    load_saved_tapes,
    onset_window_impulses,
    reset_bank_digest,
    run_isolated_replay,
    sliding_window_impulses,
    stage0_action_sequence,
    strict_qpos_legal,
    strict_qvel_legal,
    summarize_trace,
    validate_reset_bank,
    write_frozen_json,
)
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT


def _completed_result(row: dict) -> dict:
    row["realized_action_tape"] = [[0.0, 0.0, 0.0]] * 30
    row["realized_action_digest"] = str(row["action_digest"])
    return row


def _successful_solref_pair(baseline: dict) -> dict:
    baseline["solref_comparison"] = {"test_pair": True}
    identity = {
        key: baseline.get(key)
        for key in (
            "source_kind",
            "source_id",
            "source_path",
            "source_field",
            "source_aliases",
            "delta",
            "action_digest",
            "reset_digest",
            "reset_role",
        )
    }
    softened = _new_result(
        {**identity, "solref_scale": 2.0}, status="ok"
    )
    softened["solref_comparison"] = {"test_pair": True}
    return softened


def test_release_is_backdated_after_two_off_samples() -> None:
    contact = np.array([False, True, True, False, False, True])
    assert first_release(contact, onset=1, confirm_off=2) == 3


def test_one_sample_flicker_is_not_release() -> None:
    contact = np.array([False, True, True, False, True, True, False, False])
    assert first_release(contact, onset=1, confirm_off=2) == 6


def test_unreleased_contact_has_no_release() -> None:
    contact = np.array([False, True, True, True, True])
    assert first_release(contact, onset=1, confirm_off=2) is None


def test_first_lobe_ends_at_first_in_contact_non_downward_sample() -> None:
    contact = np.array([False, True, True, True, False])
    downward_speed = np.array([0.0, 1.2, 0.4, -0.1, -0.2])
    assert first_impact_lobe_end(contact, downward_speed, onset=1) == 3


def test_lobe_does_not_end_on_off_contact_sample() -> None:
    contact = np.array([False, True, True, False, False])
    downward_speed = np.array([0.0, 1.2, 0.4, -0.1, -0.2])
    assert first_impact_lobe_end(contact, downward_speed, onset=1) is None


def test_lobe_search_cannot_jump_across_release_to_a_later_contact() -> None:
    contact = np.array([False, True, True, False, False, True, True])
    downward_speed = np.array([0.0, 1.2, 0.4, -0.1, -0.2, 0.2, -0.1])
    assert (
        first_impact_lobe_end(contact, downward_speed, onset=1, stop_exclusive=3)
        is None
    )


def test_release_and_lobe_require_contact_at_accepted_onset() -> None:
    contact = np.array([False, False, True, False, False])
    speed = np.zeros(5)
    with pytest.raises(ValueError, match="not in contact"):
        first_release(contact, onset=1)
    with pytest.raises(ValueError, match="not in contact"):
        first_impact_lobe_end(contact, speed, onset=1)


def test_lobe_rejects_nonfinite_speed_trace() -> None:
    contact = np.array([False, True, True, False])
    speed = np.array([0.0, 1.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        first_impact_lobe_end(contact, speed, onset=1)


def test_onset_windows_are_13_14_and_25_substeps() -> None:
    contrib = np.ones((30, 2), dtype=np.float64)
    out = onset_window_impulses(contrib, onset=2)
    np.testing.assert_array_equal(out[13], [13.0, 13.0])
    np.testing.assert_array_equal(out[14], [14.0, 14.0])
    np.testing.assert_array_equal(out[25], [25.0, 25.0])


def test_onset_window_fails_closed_when_trace_is_too_short() -> None:
    with pytest.raises(ValueError, match="too short"):
        onset_window_impulses(np.ones((10, 2)), onset=2)


def test_sliding_window_reports_perjoint_peak_and_earliest_tie() -> None:
    contrib = np.zeros((8, 2), dtype=np.float64)
    contrib[1:4, 0] = 1.0
    contrib[4:7, 0] = 1.0
    contrib[2:5, 1] = 2.0
    out = sliding_window_impulses(contrib, length=3)
    np.testing.assert_array_equal(out["peak_perjoint"], [3.0, 6.0])
    np.testing.assert_array_equal(out["start_perjoint"], [1, 2])


def test_sliding_window_rejects_negative_or_short_trace() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        sliding_window_impulses(np.array([[0.0], [-1.0]]), length=1)
    with pytest.raises(ValueError, match="too short"):
        sliding_window_impulses(np.zeros((2, 1)), length=3)


def test_qvel_legality_catches_precontact_violation_and_nonfinite() -> None:
    legal, peak = strict_qvel_legal(
        np.array([[0.0, 0.0], [3.2, 0.1], [0.2, 0.2]]), rail=3.1415
    )
    assert legal is False
    assert peak == pytest.approx(3.2)

    legal, peak = strict_qvel_legal(
        np.array([[0.0, np.nan], [0.2, 0.2]]), rail=3.1415
    )
    assert legal is False
    assert np.isnan(peak)


def test_qvel_legality_checks_the_final_post_tape_sample() -> None:
    qvel = np.zeros((400, 6))
    qvel[-1, 4] = 3.1416
    legal, peak = strict_qvel_legal(qvel, rail=3.1415)
    assert legal is False
    assert peak == pytest.approx(3.1416)


def test_qpos_legality_checks_all_six_joints_over_full_horizon() -> None:
    qpos = np.zeros((400, 6))
    limits = np.tile(np.array([[-1.0, 1.0]]), (6, 1))
    qpos[-1, 5] = 1.01
    assert strict_qpos_legal(qpos, limits=limits) is False
    qpos[-1, 5] = 1.0
    assert strict_qpos_legal(qpos, limits=limits) is True


def test_exact_zero_invariant_uses_raw_face_mask_not_accepted_onset() -> None:
    raw = np.array([False, True, False, False])
    exact = np.zeros((4, 2))
    exact[1, 0] = 0.1
    assert_exact_zero_off_raw_contact(raw, exact)
    exact[2, 0] = 1e-5
    with pytest.raises(RuntimeError, match="raw face contact"):
        assert_exact_zero_off_raw_contact(raw, exact, tol=1e-8)


@pytest.mark.parametrize(
    ("ship", "row26", "row28", "eligible", "expected"),
    [
        (0.8, 0.8, 0.9, True, BindingClass.NO_BIND),
        (1.1, 0.8, 0.9, True, BindingClass.SHIPPED_ONLY),
        (0.8, 1.1, 1.2, True, BindingClass.EXACT_ONLY),
        (1.1, 1.1, 1.2, True, BindingClass.DUAL_BIND),
        (1.1, 0.9, 1.1, True, BindingClass.SHIPPED_TIME_BOUNDARY),
        (0.9, 0.9, 1.1, True, BindingClass.TIME_BOUNDARY),
        (1.1, 1.1, 1.2, False, BindingClass.INELIGIBLE),
    ],
)
def test_binding_interpretation_matrix(
    ship: float,
    row26: float,
    row28: float,
    eligible: bool,
    expected: BindingClass,
) -> None:
    assert (
        classify_binding(
            rho_ship50=ship,
            rho_row26=row26,
            rho_row28=row28,
            eligible=eligible,
        )
        is expected
    )


def test_saved_tape_inventory_is_finite_bounded_and_does_not_trust_old_label() -> None:
    root = Path(__file__).parents[1] / "evaluation" / "whip" / "data"
    tapes = load_saved_tapes(root)
    assert len(tapes) == 8
    assert len({t.digest for t in tapes}) == 8
    assert {t.source_field for t in tapes} == {"best_actions", "best_actions_hw"}
    assert {t.delta for t in tapes} == {0.15, 0.30, 0.45}
    assert ALLOWED_DELTAS == (0.15, 0.30, 0.45)
    for tape in tapes:
        assert tape.actions.shape == (30, 3)
        assert np.isfinite(tape.actions).all()
        assert np.abs(tape.actions).max() <= 1.0
        assert not hasattr(tape, "hardware_legal")


def test_saved_tape_loader_rejects_out_of_range_action(tmp_path: Path) -> None:
    payload = {
        "delta": 0.15,
        "T": 1,
        "best_actions": [[1.01, 0.0, 0.0]],
        "best_actions_hw": [[0.0, 0.0, 0.0]],
    }
    filename = "whip_bad.json"
    (tmp_path / filename).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        load_saved_tapes(tmp_path, filenames=(filename,))


def test_saved_tape_loader_rejects_unregistered_delta(tmp_path: Path) -> None:
    payload = {
        "delta": 0.20,
        "T": 1,
        "best_actions": [[0.0, 0.0, 0.0]],
        "best_actions_hw": [[0.0, 0.0, 0.0]],
    }
    filename = "whip_bad_delta.json"
    (tmp_path / filename).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="allowed"):
        load_saved_tapes(tmp_path, filenames=(filename,))


def test_default_saved_tape_inventory_is_explicit_and_rejects_missing_file(
    tmp_path: Path,
) -> None:
    payload = {
        "delta": 0.15,
        "T": 1,
        "best_actions": [[0.0, 0.0, 0.0]],
        "best_actions_hw": [[0.0, 0.0, -1.0]],
    }
    for filename in LEGACY_WHIP_FILES[:-1]:
        (tmp_path / filename).write_text(json.dumps(payload))
    (tmp_path / "whip_unregistered.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing required whip JSON"):
        load_saved_tapes(tmp_path)


def test_saved_tape_deduplication_preserves_every_source_alias(tmp_path: Path) -> None:
    payload = {
        "delta": 0.15,
        "T": 1,
        "best_actions": [[0.0, 0.0, -1.0]],
        "best_actions_hw": [[0.0, 0.0, -1.0]],
    }
    filename = "whip_same.json"
    (tmp_path / filename).write_text(json.dumps(payload))
    tapes = load_saved_tapes(tmp_path, filenames=(filename,))
    assert len(tapes) == 1
    assert tapes[0].aliases == (
        (filename, "best_actions"),
        (filename, "best_actions_hw"),
    )


def test_identical_action_bytes_at_different_delta_are_not_deduplicated(
    tmp_path: Path,
) -> None:
    filenames = []
    for index, delta in enumerate((0.15, 0.30)):
        payload = {
            "delta": delta,
            "T": 1,
            "best_actions": [[0.0, 0.0, -1.0]],
            "best_actions_hw": [[0.0, 0.0, -1.0]],
        }
        filename = f"whip_{index}.json"
        filenames.append(filename)
        (tmp_path / filename).write_text(json.dumps(payload))
    tapes = load_saved_tapes(tmp_path, filenames=tuple(filenames))
    assert len(tapes) == 2
    assert {t.delta for t in tapes} == {0.15, 0.30}


def test_classifier_uses_unchanged_manufacturer_caps() -> None:
    assert IMP_J_LIMIT == [1.640, 3.280, 1.640, 1.640, 1.640, 1.640]


def test_trace_summary_rejects_negative_impulse_contribution() -> None:
    n, joints, onset = 40, 6, 5
    contact = np.zeros(n, dtype=bool)
    contact[onset : onset + 3] = True
    exact = np.zeros((n, joints))
    exact[onset, 0] = -1e-6
    with pytest.raises(ValueError, match="non-negative"):
        summarize_trace(
            contact=contact,
            downward_speed=np.zeros(n),
            depth=np.zeros(n),
            exact_contrib=exact,
            shipped_peak=np.zeros(joints),
            qvel=np.zeros((n, joints)),
            qpos_legal=True,
            quality_valid=True,
            quality_overflow=False,
            nonface_contact=np.zeros(n, dtype=bool),
            caps=np.asarray(IMP_J_LIMIT),
            rail=3.1415,
            accepted_onset=onset,
        )


def test_trace_summary_rejects_negative_shipped_peak() -> None:
    n, joints, onset = 40, 6, 5
    contact = np.zeros(n, dtype=bool)
    contact[onset : onset + 3] = True
    with pytest.raises(ValueError, match="non-negative"):
        summarize_trace(
            contact=contact,
            downward_speed=np.zeros(n),
            depth=np.zeros(n),
            exact_contrib=np.zeros((n, joints)),
            shipped_peak=np.array([-1e-6, 0, 0, 0, 0, 0]),
            qvel=np.zeros((n, joints)),
            qpos_legal=True,
            quality_valid=True,
            quality_overflow=False,
            nonface_contact=np.zeros(n),
            caps=np.asarray(IMP_J_LIMIT),
            rail=3.1415,
            accepted_onset=onset,
        )


def test_stage0_action_sequence_appends_ten_zero_control_actions() -> None:
    actions = np.arange(90, dtype=np.float32).reshape(30, 3) / 90.0
    sequence = stage0_action_sequence(actions)
    assert POST_TAPE_ZERO_ACTIONS == 10
    assert sequence.shape == (40, 3)
    np.testing.assert_array_equal(sequence[:30], actions)
    np.testing.assert_array_equal(sequence[30:], np.zeros((10, 3)))


def test_stage0_config_removes_only_success_termination() -> None:
    cfg = _stage0_env_cfg(0.15)
    assert "nail_driven" not in cfg.terminations
    assert "time_out" in cfg.terminations
    assert cfg.rewards["completion"].weight == pytest.approx(100.0)
    assert cfg.auto_reset is False
    assert cfg.metrics["substep_impulse_rows"].params["enabled"] is True


def test_stage0_config_asserts_complete_fixed_plant_contract() -> None:
    for delta in ALLOWED_DELTAS:
        cfg = _stage0_env_cfg(delta)
        action = cfg.actions["ik_hammer_head"]
        assert action.delta_pos_scale == pytest.approx(delta)
        assert action.max_dq == pytest.approx(0.5)
        actuators = cfg.scene.entities["robot"].articulation.actuators
        assert [
            (
                tuple(a.target_names_expr),
                float(a.stiffness),
                float(a.damping),
                float(a.effort_limit),
                float(a.armature),
            )
            for a in actuators
        ] == [
            (("joint1", "joint3", "joint4", "joint5", "joint6"), 1000, 100, 30, 0.01),
            (("joint2",), 1500, 150, 60, 0.02),
            (("jointGripper",), 100, 20, 30, 0.005),
        ]


def test_uniform_record_all_schema_for_ok_no_contact_and_error() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "x",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "solref_scale": 1.0,
    }
    no_contact = _new_result(identity, status="no_contact")
    error = _new_result(identity, status="error", error="boom")
    assert no_contact.keys() == error.keys()
    assert no_contact["qvel_legal"] is None
    assert no_contact["qvel_trace_rad_s"] == []
    assert error["error"] == "boom"
    assert error["binding_class"] == BindingClass.INELIGIBLE.value
    for key in (
        "rho_row50",
        "row50_perjoint",
        "compiled_solref",
        "quality_contact_point_w",
        "quality_error_m",
        "quality_first_contact_time_s",
        "quality_normal_axiality",
        "head_twist_linear_angular",
    ):
        assert key in no_contact


def test_deterministic_trace_digest_excludes_operational_wrapper_fields() -> None:
    result = _new_result(
        {
            "source_kind": "legacy",
            "source_id": "x",
            "delta": 0.15,
            "action_digest": "a" * 64,
            "reset_digest": "b" * 64,
            "solref_scale": 1.0,
        },
        status="ok",
    )
    result["qvel_peak_rad_s"] = 1.25
    a = deterministic_trace_digest(result)
    result["worker_pid"] = 999
    result["wall_time_s"] = 12.0
    assert deterministic_trace_digest(result) == a


def _fixed_reset_state() -> dict:
    return _reset_state(0.0, 0, seed=0)


def _reset_state(
    value: float, call: int, *, seed: int = RESET_BANK_SEED
) -> dict:
    realized = {
        "robot_joint_pos": [value] * 7,
        "robot_joint_vel": [0.0] * 7,
        "nail_joint_pos": [0.0],
        "nail_joint_vel": [0.0],
    }
    return {
        "reset_contract_version": 1,
        "reset_state": {
            "regeneration_inputs": {
                "reset_rng_seed": seed,
                "reset_rng_call_index": call,
            },
            "realized": realized,
        },
        "reset_state_digest": reset_bank_digest(realized),
    }


def test_reset_bank_is_exactly_16_seeded_unique_states() -> None:
    assert RESET_BANK_SEED == 2026072901
    assert RESET_BANK_SIZE == 16
    bank = [
        _reset_state((float(i) - 7.5) * 0.005, i)
        for i in range(RESET_BANK_SIZE)
    ]
    validated = validate_reset_bank(bank, fixed_reset=_fixed_reset_state())
    assert len(validated) == 16
    assert len({row["reset_state_digest"] for row in validated}) == 16


def test_reset_bank_rejects_duplicate_or_wrong_seed() -> None:
    bank = [
        _reset_state((float(i) - 7.5) * 0.005, i)
        for i in range(RESET_BANK_SIZE)
    ]
    bank[-1] = json.loads(json.dumps(bank[0]))
    bank[-1]["reset_state"]["regeneration_inputs"]["reset_rng_call_index"] = (
        RESET_BANK_SIZE - 1
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_reset_bank(bank, fixed_reset=_fixed_reset_state())
    bank = [
        _reset_state((float(i) - 7.5) * 0.005, i)
        for i in range(RESET_BANK_SIZE)
    ]
    bank[0]["reset_state"]["regeneration_inputs"]["reset_rng_seed"] = 3
    with pytest.raises(ValueError, match="seed"):
        validate_reset_bank(bank, fixed_reset=_fixed_reset_state())


def test_reset_bank_enforces_training_distribution_contract() -> None:
    bank = [
        _reset_state((float(i) - 7.5) * 0.005, i)
        for i in range(RESET_BANK_SIZE)
    ]
    bank[0] = _reset_state(0.051, 0)
    with pytest.raises(ValueError, match="0.05"):
        validate_reset_bank(bank, fixed_reset=_fixed_reset_state())

    bank = [
        _reset_state((float(i) - 7.5) * 0.005, i)
        for i in range(RESET_BANK_SIZE)
    ]
    bank[0]["reset_state"]["realized"]["robot_joint_vel"][0] = 1e-3
    bank[0]["reset_state_digest"] = reset_bank_digest(
        bank[0]["reset_state"]["realized"]
    )
    with pytest.raises(ValueError, match="velocity"):
        validate_reset_bank(bank, fixed_reset=_fixed_reset_state())

    bank = [
        _reset_state((float(i) - 7.5) * 0.005, i)
        for i in range(RESET_BANK_SIZE)
    ]
    bank[0]["reset_state"]["realized"]["nail_joint_pos"][0] = 1e-3
    bank[0]["reset_state_digest"] = reset_bank_digest(
        bank[0]["reset_state"]["realized"]
    )
    with pytest.raises(ValueError, match="nail"):
        validate_reset_bank(bank, fixed_reset=_fixed_reset_state())


def test_legacy_realized_digest_cannot_silently_replace_frozen_tape() -> None:
    assert_legacy_realized_digest(
        action_mode="tape", expected="a" * 64, realized="a" * 64
    )
    with pytest.raises(RuntimeError, match="digest"):
        assert_legacy_realized_digest(
            action_mode="tape", expected="a" * 64, realized="b" * 64
        )
    # The scripted reference has no frozen open-loop tape before playback.
    assert_legacy_realized_digest(
        action_mode="reference", expected="a" * 64, realized="b" * 64
    )


def test_write_frozen_json_is_write_once_hash_sidecar_and_read_only(
    tmp_path: Path,
) -> None:
    target = tmp_path / "evidence.json"
    digest = write_frozen_json(target, {"b": 2, "a": 1})
    sidecar = target.with_suffix(target.suffix + ".sha256")
    assert len(digest) == 64
    assert sidecar.read_text().strip() == f"{digest}  {target.name}"
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        write_frozen_json(target, {"a": 1})


def test_authoritative_output_must_be_outside_code_and_asset_repositories(
    tmp_path: Path,
) -> None:
    code = tmp_path / "code"
    asset = tmp_path / "asset"
    outside = tmp_path / "evidence"
    code.mkdir()
    asset.mkdir()
    outside.mkdir()
    _assert_authoritative_outside_repositories(
        outside / "result.json", code_repo=code, asset_repo=asset
    )
    with pytest.raises(ValueError, match="outside"):
        _assert_authoritative_outside_repositories(
            code / "data" / "result.json", code_repo=code, asset_repo=asset
        )
    with pytest.raises(ValueError, match="outside"):
        _assert_authoritative_outside_repositories(
            asset / "result.json", code_repo=code, asset_repo=asset
        )


def test_asset_provenance_scopes_cleanliness_to_loaded_assets(
    monkeypatch,
) -> None:
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="b58ccd2\n")
        if "--" in args:
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout="?? .claude/\n?? CLAUDE.md\n")

    monkeypatch.setattr(
        "evaluation.whip.lambda_feasibility.subprocess.run", fake_run
    )
    out = _git_provenance(
        Path("/asset-repo"), scope=("hammer_z1_env/assets",)
    )
    assert out["dirty"] is False
    assert out["asset_scope_clean"] is True
    assert out["repository_dirty"] is True
    assert "?? .claude/" in out["repository_status"]
    assert any(
        args[-2:] == ["--", "hammer_z1_env/assets"] for args in calls
    )


def test_release_sensitivity_and_qvel_margin_are_reported() -> None:
    assert RELEASE_CONFIRM_SAMPLES == (2, 5, 10)
    n, joints, onset = 50, 6, 5
    contact = np.zeros(n, dtype=bool)
    contact[onset : onset + 3] = True
    speed = np.zeros(n)
    speed[onset : onset + 3] = [1.0, 0.4, 0.0]
    depth = np.zeros(n)
    depth[onset + 2 :] = 0.031
    exact = np.zeros((n, joints))
    exact[onset : onset + 3, 0] = [0.7, 0.6, 0.4]
    summary = summarize_trace(
        contact=contact,
        downward_speed=speed,
        depth=depth,
        exact_contrib=exact,
        shipped_peak=np.array([1.7, 0, 0, 0, 0, 0]),
        qvel=np.zeros((n + 1, joints)),
        qpos_legal=True,
        quality_valid=True,
        quality_overflow=False,
        nonface_contact=np.zeros(n),
        caps=np.asarray(IMP_J_LIMIT),
        rail=3.1415,
        accepted_onset=onset,
    )
    assert summary["release_index_by_confirm"] == {"2": 8, "5": 8, "10": 8}
    assert summary["qvel_legal_95pct"] is True
    assert summary["row50_onset_perjoint"][0] == pytest.approx(1.7)
    assert summary["row50_sliding_peak_perjoint"][0] == pytest.approx(1.7)


def test_aggregate_sentinels_exclude_reference_and_do_not_count_resets_as_independent() -> None:
    base = {
        "source_kind": "legacy",
        "source_id": "tape-a",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    fixed = _new_result(base, status="ok")
    fixed.update(eligible=True, binding_class=BindingClass.DUAL_BIND.value)
    robust = _new_result(
        {**base, "reset_digest": "c" * 64, "reset_role": "robustness"},
        status="ok",
    )
    robust.update(eligible=False)
    reference = _new_result(
        {
            **base,
            "source_kind": "reference",
            "source_id": "scripted-reference",
            "action_digest": "d" * 64,
        },
        status="ok",
    )
    reference.update(eligible=True, binding_class=BindingClass.DUAL_BIND.value)
    aggregate = aggregate_results([fixed, robust, reference])
    assert aggregate["impossible_success_n"] == 0
    assert aggregate["lambda_dead_n"] == 0
    assert aggregate["fixed_reset_existence"]["dual_bind_n"] == 1
    assert aggregate["fixed_reset_existence"]["denominator_n"] == 1
    assert aggregate["reference_control_n"] == 1
    assert aggregate["robustness_replay_n"] == 1
    assert aggregate["delta_counts_descriptive_only"]["0.15"]["total_n"] == 2
    assert (
        aggregate["stratified_counts"]["0.15"]["fixed"]["1.0"]["total_n"]
        == 1
    )
    assert (
        aggregate["stratified_counts"]["0.15"]["robustness"]["1.0"][
            "total_n"
        ]
        == 1
    )
    assert aggregate["operational_failure_n"] == 0


def test_aggregate_counts_operational_failures_explicitly() -> None:
    failed = _new_result(
        {
            "source_kind": "legacy",
            "source_id": "failed",
            "delta": 0.45,
            "action_digest": "a" * 64,
            "reset_digest": "b" * 64,
            "reset_role": "fixed",
            "solref_scale": 1.0,
        },
        status="timeout",
    )
    aggregate = aggregate_results([failed])
    assert aggregate["operational_failure_n"] == 1
    assert aggregate["fixed_baseline_failure_n"] == 1


def test_continuation_uses_fixed_reset_existence_not_robustness_votes() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "tape-a",
        "delta": 0.30,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    fixed = _completed_result(_new_result(identity, status="ok"))
    fixed.update(
        eligible=True,
        qvel_legal=True,
        qpos_legal=True,
        rho_row28=1.1,
        binding_class=BindingClass.DUAL_BIND.value,
    )
    fixed_soft = _successful_solref_pair(fixed)
    robustness_failures = [
        _new_result(
            {
                **identity,
                "reset_digest": f"{index + 1:064x}",
                "reset_role": "robustness",
            },
            status="no_contact",
        )
        for index in range(16)
    ]
    deterministic = _determinism_qualification(fixed, json.loads(json.dumps(fixed)))
    out = _continuation_label(
        [fixed, fixed_soft, *robustness_failures],
        qualifications=[deterministic],
    )
    assert out["label"] == "simulator_existence_proven"
    assert out["stage1"] == "not_required_for_existence"


def test_continuation_near_lead_and_no_lead_are_not_physical_boundaries() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "tape-a",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    near = _completed_result(_new_result(identity, status="ok"))
    near.update(
        eligible=True,
        qvel_legal=True,
        qpos_legal=True,
        rho_row28=0.70,
        binding_class=BindingClass.NO_BIND.value,
    )
    near_soft = _successful_solref_pair(near)
    near_q = _determinism_qualification(near, json.loads(json.dumps(near)))
    assert _continuation_label(
        [near, near_soft], qualifications=[near_q]
    )["label"] == "near_lead"

    far = _completed_result(_new_result(identity, status="ok"))
    far.update(
        eligible=True,
        qvel_legal=True,
        qpos_legal=True,
        rho_row28=0.69,
        binding_class=BindingClass.NO_BIND.value,
    )
    far_soft = _successful_solref_pair(far)
    far_q = _determinism_qualification(far, json.loads(json.dumps(far)))
    out = _continuation_label([far, far_soft], qualifications=[far_q])
    assert out["label"] == "no_stage1_lead"
    assert out["not_a_physical_impossibility_claim"] is True


def test_continuation_is_ambiguous_if_any_fixed_baseline_failed() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "stable",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    stable = _completed_result(_new_result(identity, status="ok"))
    stable.update(qvel_legal=True, qpos_legal=True, rho_row28=0.1)
    failed = _new_result(
        {**identity, "source_id": "failed", "action_digest": "c" * 64},
        status="crash",
        error="native crash",
    )
    stable_q = _determinism_qualification(
        stable, json.loads(json.dumps(stable))
    )
    failed_q = _determinism_qualification(
        failed, json.loads(json.dumps(failed))
    )
    out = _continuation_label(
        [stable, failed], qualifications=[stable_q, failed_q]
    )
    assert out["label"] == "ambiguous"
    assert "fixed baseline" in out["basis"]


def test_continuation_is_ambiguous_on_missing_or_failed_mandatory_solref() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "eligible",
        "delta": 0.30,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    eligible = _completed_result(_new_result(identity, status="ok"))
    eligible.update(
        eligible=True,
        qvel_legal=True,
        qpos_legal=True,
        rho_row28=0.8,
        binding_class=BindingClass.NO_BIND.value,
    )
    deterministic = _determinism_qualification(
        eligible, json.loads(json.dumps(eligible))
    )
    missing = _continuation_label(
        [eligible], qualifications=[deterministic]
    )
    assert missing["label"] == "ambiguous"
    assert "solref" in missing["basis"]

    softened = _new_result(
        {**identity, "solref_scale": 2.0},
        status="error",
        error="comparison failed",
    )
    failed = _continuation_label(
        [eligible, softened], qualifications=[deterministic]
    )
    assert failed["label"] == "ambiguous"
    assert "solref" in failed["basis"]


def test_actuator_saturation_is_full_horizon_per_joint_occupancy() -> None:
    effort = np.array(
        [
            [0.0, 0.0],
            [30.0, 59.0],
            [-30.0, 60.0],
        ]
    )
    out = actuator_saturation_summary(effort, limits=np.array([30.0, 60.0]))
    np.testing.assert_allclose(out["perjoint"], [2 / 3, 1 / 3])
    assert out["overall"] == pytest.approx(0.5)


def test_determinism_requires_same_identity_and_trace_digest() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "x",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    a = _new_result(identity, status="ok")
    b = _new_result(identity, status="ok")
    a["qvel_trace_rad_s"] = [[1.0]]
    b["qvel_trace_rad_s"] = [[1.0]]
    assert_deterministic_replay(a, b)
    b["qvel_trace_rad_s"] = [[1.01]]
    with pytest.raises(RuntimeError, match="deterministic"):
        assert_deterministic_replay(a, b)


def test_determinism_qualification_fails_closed_on_incomplete_pair() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "x",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    first = _new_result(identity, status="crash", error="native crash")
    repeat = _new_result(identity, status="crash", error="native crash")
    qualification = _determinism_qualification(first, repeat)
    assert qualification["status"] == "error"
    assert qualification["qualification"] == "deterministic_replay"
    assert "first=crash" in qualification["error"]


def test_determinism_qualification_passes_only_two_complete_equal_replays() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "x",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    first = _new_result(identity, status="ok")
    repeat = _new_result(identity, status="ok")
    for row in (first, repeat):
        row["realized_action_tape"] = [[0.0, 0.0, 0.0]] * 30
        row["realized_action_digest"] = "a" * 64
    qualification = _determinism_qualification(first, repeat)
    assert qualification["status"] == "ok"
    assert qualification["deterministic_trace_digest"]


def test_reference_calibration_is_non_numeric_and_fails_if_impact_eligible() -> None:
    reference = _completed_result(
        _new_result(
            {
                "source_kind": "reference",
                "source_id": "SingleStrikeReference-default",
                "delta": 0.15,
                "action_digest": "a" * 64,
                "reset_digest": "b" * 64,
                "reset_role": "fixed",
                "solref_scale": 1.0,
            },
            status="ok",
        )
    )
    reference.update(
        contact_seen=True,
        accepted_onset_index=10,
        productive_30mm_50ms=True,
        qvel_legal=True,
        qpos_legal=True,
        exact_live_contact_substeps=2,
        object_live_contact_substeps=2,
        aligned_live_contact_substeps=2,
        release_by_28ms=False,
        quality_valid=True,
        quality_overflow=False,
        eligible=False,
        binding_class=BindingClass.INELIGIBLE.value,
        solref_comparison={"calibration": True},
    )
    calibration = _reference_calibration_qualification(reference)
    assert calibration["status"] == "ok"
    assert calibration["observed_reference_class"] == BindingClass.INELIGIBLE.value

    reference["eligible"] = True
    reference["binding_class"] = BindingClass.DUAL_BIND.value
    unexpected = _reference_calibration_qualification(reference)
    assert unexpected["status"] == "error"
    assert "manual review" in unexpected["error"]

    reference.update(
        eligible=False,
        binding_class=BindingClass.INELIGIBLE.value,
        release_by_28ms=True,
    )
    short_but_ineligible = _reference_calibration_qualification(reference)
    assert short_but_ineligible["status"] == "error"
    assert "broad-contact" in short_but_ineligible["error"]

    reference.update(release_by_28ms=False, quality_valid=False)
    invalid_quality = _reference_calibration_qualification(reference)
    assert invalid_quality["status"] == "error"
    assert "quality invalid" in invalid_quality["error"]

    reference.update(quality_valid=True, quality_overflow=True)
    overflowed_quality = _reference_calibration_qualification(reference)
    assert overflowed_quality["status"] == "error"
    assert "quality overflow" in overflowed_quality["error"]


def test_ordinary_shadow_prefix_parity_is_exact_through_success() -> None:
    ordinary = {
        "actions": [[0.0, 0.0, -1.0]] * 3,
        "qpos": [[0.0], [0.1], [0.2], [0.3]],
        "qvel": [[0.0], [1.0], [1.0], [1.0]],
        "head_position": [[0.0, 0.0, 0.2]] * 4,
        "nail_depth": [0.0, 0.0, 0.01, 0.03],
        "success_control_index": 2,
    }
    shadow = json.loads(json.dumps(ordinary))
    shadow["qpos"].append([0.4])
    assert_ordinary_shadow_prefix_parity(ordinary, shadow)
    shadow["qvel"][2][0] = 0.9
    with pytest.raises(RuntimeError, match="prefix parity"):
        assert_ordinary_shadow_prefix_parity(ordinary, shadow)


class _FakeQueue:
    def __init__(self) -> None:
        self.value = None

    def put(self, value) -> None:
        self.value = value

    def get(self, timeout=None):
        del timeout
        if self.value is None:
            import queue

            raise queue.Empty
        return self.value


class _FakeProcess:
    def __init__(self, *, target, args, run=True, exitcode=0) -> None:
        self.target = target
        self.args = args
        self._run = run
        self.exitcode = exitcode
        self.pid = 123
        self.terminated = False

    def start(self) -> None:
        if self._run:
            self.target(*self.args)

    def join(self, timeout=None) -> None:
        del timeout

    def is_alive(self) -> bool:
        return not self._run and not self.terminated

    def terminate(self) -> None:
        self.terminated = True
        self.exitcode = -15


class _FakeContext:
    def __init__(self, *, run=True, exitcode=0) -> None:
        self.queue = _FakeQueue()
        self.run = run
        self.exitcode = exitcode

    def Queue(self):
        return self.queue

    def Process(self, *, target, args):
        return _FakeProcess(
            target=target, args=args, run=self.run, exitcode=self.exitcode
        )


def test_isolated_replay_records_timeout_in_uniform_schema(monkeypatch) -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "x",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    result = run_isolated_replay(
        {"identity": identity}, timeout_s=0.01, context=_FakeContext(run=False)
    )
    assert result["status"] == "timeout"
    assert result["binding_class"] == BindingClass.INELIGIBLE.value


def test_isolated_replay_records_worker_error_without_aborting(monkeypatch) -> None:
    import evaluation.whip.lambda_feasibility as module

    identity = {
        "source_kind": "legacy",
        "source_id": "x",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }

    def fail(_request):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(module, "_execute_replay_request", fail)
    result = run_isolated_replay(
        {"identity": identity}, timeout_s=1.0, context=_FakeContext()
    )
    assert result["status"] == "error"
    assert "synthetic failure" in result["error"]


def test_solref_comparison_reports_perjoint_and_binding_joint_switch() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "x",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
    }
    baseline = _new_result({**identity, "solref_scale": 1.0}, status="ok")
    softened = _new_result({**identity, "solref_scale": 2.0}, status="ok")
    baseline.update(
        binding_joint_index=0,
        row26_perjoint=[1, 2, 3, 4, 5, 6],
        row28_perjoint=[2, 2, 3, 4, 5, 6],
        full_lobe_perjoint=[3, 2, 3, 4, 5, 6],
    )
    softened.update(
        binding_joint_index=1,
        row26_perjoint=[0.9, 2.2, 3, 4, 5, 6],
        row28_perjoint=[1.8, 2.2, 3, 4, 5, 6],
        full_lobe_perjoint=[2.7, 2.2, 3, 4, 5, 6],
    )
    out = compare_solref_results(baseline, softened)
    assert out["binding_joint_switched"] is True
    assert out["baseline_binding_joint_relative_change"]["row26"] == pytest.approx(
        -0.1
    )
    assert out["necessary_model_robustness_only"] is True
    assert out["proves_ballistic_or_hardware_fidelity"] is False


def test_solref_scheduler_always_includes_fixed_reference_calibration() -> None:
    identity = {
        "source_kind": "reference",
        "source_id": "reference",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    reference = _new_result(identity, status="ok")
    assert reference["eligible"] is False
    assert _requires_solref_replay(reference) is True
    reference["reset_role"] = "robustness"
    assert _requires_solref_replay(reference) is False

    legacy = _new_result(
        {**identity, "source_kind": "legacy", "reset_role": "fixed"},
        status="ok",
    )
    assert _requires_solref_replay(legacy) is False
    legacy["eligible"] = True
    assert _requires_solref_replay(legacy) is True


def test_nonface_gate_uses_maximum_active_efc_force_and_frozen_tolerance() -> None:
    force = _max_nonface_efc_force(
        geom=np.array([[1, 2], [7, 8], [1, 9]], dtype=np.int64),
        contact_type=np.array([1, 1, 1], dtype=np.int64),
        world_id=np.array([0, 0, 0], dtype=np.int64),
        efc_address=np.array([[0, -1], [1, -1], [2, -1]], dtype=np.int64),
        efc_force=np.array([[1e3, 2e3, 2e-6]], dtype=np.float64),
        robot_geom_ids={1},
        hammer_geom_ids={1},
        nail_geom_ids={2},
        constraint_bit=1,
    )
    assert NONFACE_FORCE_TOL == pytest.approx(1e-6)
    assert force == pytest.approx(2e-6)


def test_contact_signal_liveness_requires_aligned_but_not_equal_signals() -> None:
    contact = np.array([False, True, True, False])
    exact = np.zeros((4, 6))
    exact[2, 0] = 1.0
    object_total = np.array([0.0, 0.0, 100.0, 0.0])
    liveness = contact_signal_liveness(
        contact=contact,
        exact_contrib=exact,
        object_axial_contrib=object_total,
        onset=1,
    )
    assert liveness == {
        "exact_live_contact_substeps": 1,
        "object_live_contact_substeps": 1,
        "aligned_live_contact_substeps": 1,
    }


def test_contact_signal_liveness_fails_when_channels_never_overlap() -> None:
    contact = np.array([False, True, True, False])
    exact = np.zeros((4, 6))
    exact[1, 0] = 1.0
    object_total = np.array([0.0, 0.0, 2.0, 0.0])
    with pytest.raises(RuntimeError, match="never live on the same"):
        contact_signal_liveness(
            contact=contact,
            exact_contrib=exact,
            object_axial_contrib=object_total,
            onset=1,
        )


def test_contact_signal_liveness_cannot_use_a_later_recontact() -> None:
    contact = np.array([False, True, True, False, False, True, False])
    exact = np.zeros((7, 6))
    exact[1, 0] = 1.0
    exact[5, 0] = 1.0
    object_axial = np.zeros(7)
    object_axial[5] = 2.0
    with pytest.raises(RuntimeError, match="object-side.*dead"):
        contact_signal_liveness(
            contact=contact,
            exact_contrib=exact,
            object_axial_contrib=object_axial,
            onset=1,
            stop_exclusive=3,
        )


def test_trace_summary_requires_short_releasing_productive_legal_impact() -> None:
    n, joints, onset = 40, 6, 5
    contact = np.zeros(n, dtype=bool)
    contact[onset : onset + 3] = True
    speed = np.zeros(n)
    speed[onset : onset + 3] = [1.0, 0.4, 0.0]
    depth = np.zeros(n)
    depth[onset + 2 :] = 0.031
    exact = np.zeros((n, joints))
    exact[onset : onset + 3, 0] = [0.7, 0.6, 0.4]
    summary = summarize_trace(
        contact=contact,
        downward_speed=speed,
        depth=depth,
        exact_contrib=exact,
        shipped_peak=np.array([1.7, 0, 0, 0, 0, 0]),
        qvel=np.zeros((n, joints)),
        qpos_legal=True,
        quality_valid=True,
        quality_overflow=False,
        nonface_contact=np.zeros(n, dtype=bool),
        caps=np.asarray(IMP_J_LIMIT),
        rail=3.1415,
        accepted_onset=onset,
    )
    assert summary["eligible"] is True
    assert summary["release_index"] == onset + 3
    assert summary["lobe_end_index"] == onset + 2
    assert summary["rho_row26"] > 1.0
    assert summary["lobe_fraction"] == pytest.approx(1.0)
    assert summary["binding_class"] == BindingClass.DUAL_BIND.value


def test_lobe_share_is_bounded_to_the_onset_50ms_denominator() -> None:
    n, joints, onset = 50, 6, 5
    contact = np.zeros(n, dtype=bool)
    contact[onset : onset + 35] = True
    speed = np.ones(n)
    speed[onset + 30] = 0.0
    depth = np.zeros(n)
    depth[onset + 2 :] = 0.031
    exact = np.zeros((n, joints))
    exact[onset : onset + 31, 0] = 0.1
    summary = summarize_trace(
        contact=contact,
        downward_speed=speed,
        depth=depth,
        exact_contrib=exact,
        shipped_peak=np.zeros(joints),
        qvel=np.zeros((n + 1, joints)),
        qpos_legal=True,
        quality_valid=True,
        quality_overflow=False,
        nonface_contact=np.zeros(n),
        caps=np.asarray(IMP_J_LIMIT),
        rail=3.1415,
        accepted_onset=onset,
    )
    assert summary["lobe_fraction"] == pytest.approx(1.0)
    assert summary["full_lobe_perjoint"][0] == pytest.approx(3.1)


def test_uniform_schema_banks_all_raw_500hz_rederivation_channels() -> None:
    result = _new_result(
        {
            "source_kind": "legacy",
            "source_id": "x",
            "delta": 0.15,
            "action_digest": "a" * 64,
            "reset_digest": "b" * 64,
            "reset_role": "fixed",
            "solref_scale": 1.0,
        },
        status="error",
    )
    required = {
        "exact_contrib_perjoint_trace",
        "shipped_rolling_perjoint_trace",
        "shipped_contrib_perjoint_trace",
        "object_axial_force_n_trace",
        "object_axial_contrib_n_s_trace",
        "object_total_contrib_n_s_trace",
        "nonface_efc_force_trace",
        "tracker_started_trace",
    }
    assert required <= set(result)
    assert all(result[key] == [] for key in required)


def test_deterministic_digest_binds_raw_rederivation_channels() -> None:
    identity = {
        "source_kind": "legacy",
        "source_id": "x",
        "delta": 0.15,
        "action_digest": "a" * 64,
        "reset_digest": "b" * 64,
        "reset_role": "fixed",
        "solref_scale": 1.0,
    }
    first = _new_result(identity, status="ok")
    second = json.loads(json.dumps(first))
    first["exact_contrib_perjoint_trace"] = [[0.0] * 6]
    second["exact_contrib_perjoint_trace"] = [[0.1] + [0.0] * 5]
    assert deterministic_trace_digest(first) != deterministic_trace_digest(second)


@pytest.mark.parametrize("failure", ["no_release", "late_release", "recontact", "qvel"])
def test_trace_summary_rejects_press_recontact_and_illegal_motion(failure: str) -> None:
    n, joints, onset = 50, 6, 5
    contact = np.zeros(n, dtype=bool)
    contact[onset : onset + 3] = True
    speed = np.zeros(n)
    speed[onset : onset + 3] = [1.0, 0.4, 0.0]
    depth = np.zeros(n)
    depth[onset + 2 :] = 0.031
    exact = np.zeros((n, joints))
    exact[onset : onset + 3, 0] = [0.7, 0.6, 0.4]
    qvel = np.zeros((n, joints))
    if failure == "no_release":
        contact[onset:] = True
    elif failure == "late_release":
        contact[onset : onset + 16] = True
    elif failure == "recontact":
        contact[onset + 8 : onset + 10] = True
    elif failure == "qvel":
        qvel[1, 0] = 3.2
    summary = summarize_trace(
        contact=contact,
        downward_speed=speed,
        depth=depth,
        exact_contrib=exact,
        shipped_peak=np.array([1.7, 0, 0, 0, 0, 0]),
        qvel=qvel,
        qpos_legal=True,
        quality_valid=True,
        quality_overflow=False,
        nonface_contact=np.zeros(n, dtype=bool),
        caps=np.asarray(IMP_J_LIMIT),
        rail=3.1415,
        accepted_onset=onset,
    )
    assert summary["eligible"] is False
    assert summary["binding_class"] == BindingClass.INELIGIBLE.value
