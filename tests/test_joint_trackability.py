"""Joint command-trackability formula, calibration, and artifact contracts."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.joint_position.calibrate_trackability import (
    _require_accepted_terminal,
    _scheduled_replay_tape,
    build_trackability_payload,
    canonical_sha256,
    derive_k_tt,
    write_canonical_json,
)
from src.tasks.hammer.config.z1.joint_position_contract import (
    JOINT_NAMES,
    load_joint_trackability_contract,
)
from src.tasks.hammer.mdp.trackability import (
    joint_target_rmse,
    joint_target_squared_error,
    joint_trackability_cost,
)


def _reader_env(*, terminal: bool = False, action_name: str = "joint_position"):
    q_des = torch.tensor(
        [[0.2, 0.0, -0.1, 0.3, 0.1, 0.0, 99.0]], dtype=torch.float32
    )
    q_next = torch.tensor(
        [[0.1, 0.0, -0.1, 0.1, 0.0, 0.0, -99.0]], dtype=torch.float32
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos_target=q_des, joint_pos=q_next)
    )
    action_term = SimpleNamespace(
        target_ids=torch.arange(6), target_names=list(JOINT_NAMES)
    )
    action_manager = SimpleNamespace(
        active_terms=[action_name], get_term=lambda name: action_term
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        action_manager=action_manager,
        cfg=SimpleNamespace(auto_reset=False),
        reset_buf=torch.tensor([terminal]),
        step_dt=0.02,
    )
    arm_cfg = SimpleNamespace(
        name="robot", joint_ids=list(range(6)), joint_names=list(JOINT_NAMES)
    )
    return env, arm_cfg


def test_readers_use_public_applied_target_and_only_six_ordered_arm_joints() -> None:
    """Reading realized q, a seventh joint, or a reordered action must change this result."""
    env, arm_cfg = _reader_env()
    expected = torch.tensor([0.06])
    torch.testing.assert_close(joint_target_squared_error(env, arm_cfg), expected)
    torch.testing.assert_close(
        joint_target_rmse(env, arm_cfg), torch.sqrt(expected / 6.0)
    )
    torch.testing.assert_close(
        joint_trackability_cost(env, arm_cfg, 2.0), 2.0 * expected
    )


def test_reader_keeps_accepted_contact_terminal_transition_with_auto_reset_disabled() -> None:
    """A terminal mask must not zero or replace the accepted-contact post-step sample."""
    env, arm_cfg = _reader_env(terminal=True)
    torch.testing.assert_close(
        joint_target_squared_error(env, arm_cfg), torch.tensor([0.06])
    )


def test_reward_manager_dt_is_external_to_raw_trackability_cost() -> None:
    """Accidentally multiplying the raw term by dt would apply RewardManager scaling twice."""
    env, arm_cfg = _reader_env()
    raw = joint_trackability_cost(env, arm_cfg, 2.0)
    torch.testing.assert_close(raw, torch.tensor([0.12]))
    torch.testing.assert_close(raw * env.step_dt, torch.tensor([0.0024]))


@pytest.mark.parametrize("k_tt", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_trackability_cost_rejects_nonpositive_or_nonfinite_gain(k_tt: float) -> None:
    env, arm_cfg = _reader_env()
    with pytest.raises(ValueError, match="k_tt"):
        joint_trackability_cost(env, arm_cfg, k_tt)


def test_reader_rejects_shape_mismatch_and_non_joint_action_task() -> None:
    env, arm_cfg = _reader_env()
    env.scene["robot"].data.joint_pos = torch.zeros(2, 7)
    with pytest.raises(ValueError, match="shape"):
        joint_target_squared_error(env, arm_cfg)

    env, arm_cfg = _reader_env(action_name="ik_hammer_head")
    with pytest.raises(ValueError, match="joint-position"):
        joint_target_squared_error(env, arm_cfg)


def test_derive_k_tt_uses_one_trajectory_q90_and_preregistered_target() -> None:
    """Pooling 16 duplicated trajectories would alter neither this API nor its denominator."""
    errors = np.asarray([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
    q90 = float(np.quantile(errors, 0.90))
    assert derive_k_tt(errors) == pytest.approx(0.1 / q90)


@pytest.mark.parametrize(
    "errors",
    [
        [0.0, 0.0],
        [1e-10, 1e-10],
        [0.1, math.nan],
        [0.1, math.inf],
        [],
    ],
)
def test_derive_k_tt_rejects_degenerate_or_nonfinite_calibration(errors) -> None:
    with pytest.raises(ValueError):
        derive_k_tt(errors)


def _runs(*, changed_witness: bool = False):
    targets = [[0.2] * 6, [0.3] * 6, [0.4] * 6]
    qualified_target_hash = canonical_sha256(targets)
    errors = [0.01, 0.02, 0.04]
    runs = []
    for seed in range(1000, 1016):
        run_errors = list(errors)
        if changed_witness and seed == 1007:
            run_errors[-1] = 0.041
        runs.append(
            {
                "seed": seed,
                "applied_target_tape_rad": copy.deepcopy(targets),
                "qualified_applied_target_tape_sha256": qualified_target_hash,
                "squared_errors_rad2": run_errors,
                "terminal_transition_captured": True,
            }
        )
    return runs


def _build_payload(*, runs=None):
    return build_trackability_payload(
        runs=_runs() if runs is None else runs,
        source_qualification_payload_sha256="a" * 64,
        source_qualification_code_revision="b" * 40,
        source_asset_revision="c" * 40,
        calibration_code_revision="d" * 40,
    )


def test_calibration_uses_seed_1000_once_and_15_runs_only_as_witnesses() -> None:
    """Duplicating samples into a pooled population must never define q90 or cumulative dose."""
    payload = _build_payload()
    canonical_errors = np.asarray([0.01, 0.02, 0.04])
    q90 = float(np.quantile(canonical_errors, 0.90))
    k_tt = 0.1 / q90
    assert payload["canonical_seed"] == 1000
    assert payload["canonical_sample_count"] == 3
    assert payload["q90_squared_error_rad2"] == pytest.approx(q90)
    assert payload["k_tt"] == pytest.approx(k_tt)
    assert payload["canonical_pre_dt_cumulative_cost"] == pytest.approx(
        k_tt * canonical_errors.sum()
    )
    assert payload["canonical_returned_dose"] == pytest.approx(
        0.02 * k_tt * canonical_errors.sum()
    )
    assert len(payload["repeatability_rows"]) == 16
    assert payload["decision"] == "PASS"


def test_calibration_reconstructs_hold_targets_before_replaying_to_terminal() -> None:
    """Source may terminate one interval before replay, so its tape cannot be used raw."""
    source_tape = np.arange(8 * 6, dtype=np.float64).reshape(8, 6)
    contract = SimpleNamespace(
        source_target_tape_rad=source_tape,
        per_seed_replay_rows=(
            {
                "source": {"playback_length": 6},
                "replay": {"target_count": 9},
            },
        ),
    )
    scheduled = _scheduled_replay_tape(contract)
    assert scheduled.shape == (16, 6)
    np.testing.assert_array_equal(scheduled[:8], source_tape)
    np.testing.assert_array_equal(scheduled[8:], np.repeat(source_tape[-1:], 8, axis=0))


def test_calibration_requires_the_banked_productive_success_terminal() -> None:
    """An unrelated termination at the same target count is not calibration evidence."""
    env = SimpleNamespace(
        reset_buf=torch.tensor([True]),
        reset_terminated=torch.tensor([True]),
        reset_time_outs=torch.tensor([False]),
        _hammer_first_strike=SimpleNamespace(
            finalized=torch.tensor([True]),
            productive=torch.tensor([True]),
            reason=torch.tensor([1]),
        ),
    )
    _require_accepted_terminal(env, {"terminal_reason": "success"})
    env._hammer_first_strike.reason[0] = 2
    with pytest.raises(RuntimeError, match="productive success"):
        _require_accepted_terminal(env, {"terminal_reason": "success"})


def test_calibration_rejects_witness_drift_and_excessive_cumulative_cost() -> None:
    with pytest.raises(ValueError, match="repeatability"):
        _build_payload(runs=_runs(changed_witness=True))
    expensive = _runs()
    for run in expensive:
        run["squared_errors_rad2"] = [1e-8] * 1000 + [1.0]
        run["applied_target_tape_rad"] = [[0.2] * 6] * 1001
        run["qualified_applied_target_tape_sha256"] = canonical_sha256(
            run["applied_target_tape_rad"]
        )
    with pytest.raises(ValueError, match="cumulative"):
        _build_payload(runs=expensive)


def test_calibration_rejects_mutually_repeatable_live_tape_not_banked_per_seed() -> None:
    """Sixteen identical live runs are invalid when they do not replay the qualified tape."""
    runs = _runs()
    different_live_tape = [[0.25] * 6, [0.35] * 6, [0.45] * 6]
    for run in runs:
        run["applied_target_tape_rad"] = copy.deepcopy(different_live_tape)
    with pytest.raises(ValueError, match="qualified replay"):
        _build_payload(runs=runs)


def _source_contract():
    qualified_target_hash = canonical_sha256([[0.2] * 6, [0.3] * 6, [0.4] * 6])
    return SimpleNamespace(
        payload_sha256="a" * 64,
        source_code_revision="b" * 40,
        source_asset_revision="c" * 40,
        joint_names=JOINT_NAMES,
        physics_dt_s=0.002,
        control_decimation=10,
        per_seed_replay_rows=tuple(
            {
                "replay": {
                    "replay_applied_target_tape_sha256": qualified_target_hash,
                    "target_count": 3,
                }
            }
            for _ in range(16)
        ),
    )


def test_trackability_loader_is_strict_identity_bound_and_deeply_immutable(
    tmp_path: Path,
) -> None:
    payload = _build_payload()
    artifact = tmp_path / "trackability.json"
    write_canonical_json(artifact, payload)

    loaded = load_joint_trackability_contract(
        artifact, source_contract=_source_contract()
    )
    assert loaded.k_tt == pytest.approx(payload["k_tt"])
    assert loaded.repeatability_rows[0]["seed"] == 1000
    with pytest.raises(TypeError):
        loaded.repeatability_rows[0]["seed"] = 42


def test_trackability_loader_rejects_self_consistent_truncated_sample_set(
    tmp_path: Path,
) -> None:
    """A rehashed two-step calibration cannot stand in for the three-step qualified replay."""
    payload = _build_payload()
    errors = payload["canonical_squared_errors_rad2"][:-1]
    q90 = float(np.quantile(errors, 0.90))
    k_tt = 0.1 / q90
    error_hash = canonical_sha256(errors)
    pre_dt_cost = k_tt * float(np.sum(errors, dtype=np.float64))
    payload.update(
        canonical_sample_count=len(errors),
        canonical_squared_errors_rad2=errors,
        canonical_squared_errors_sha256=error_hash,
        q90_squared_error_rad2=q90,
        k_tt=k_tt,
        canonical_pre_dt_cumulative_cost=pre_dt_cost,
        canonical_returned_dose=pre_dt_cost * 0.02,
    )
    for row in payload["repeatability_rows"]:
        row["sample_count"] = len(errors)
        row["squared_errors_sha256"] = error_hash
    unsigned = dict(payload)
    unsigned.pop("payload_sha256")
    payload["payload_sha256"] = canonical_sha256(unsigned)
    artifact = tmp_path / "truncated-trackability.json"
    write_canonical_json(artifact, payload)

    with pytest.raises(ValueError, match="qualified replay target_count"):
        load_joint_trackability_contract(
            artifact, source_contract=_source_contract()
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("k_tt"),
        lambda p: p.update(k_tt=math.nan),
        lambda p: p.update(decision="FAIL"),
        lambda p: p.update(source_qualification_payload_sha256="f" * 64),
        lambda p: p["repeatability_rows"].pop(),
    ],
)
def test_trackability_loader_fails_closed_on_mutated_artifact(
    tmp_path: Path, mutate
) -> None:
    payload = _build_payload()
    mutate(payload)
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    try:
        payload["payload_sha256"] = canonical_sha256(unsigned)
    except ValueError:
        # Non-finite JSON is rejected before its otherwise impossible canonical digest.
        pass
    artifact = tmp_path / "trackability.json"
    artifact.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError):
        load_joint_trackability_contract(
            artifact, source_contract=_source_contract()
        )
