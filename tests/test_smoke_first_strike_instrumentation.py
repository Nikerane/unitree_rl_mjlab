"""Pure schema and population-gate tests for the CUDA smoke record."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest
import torch

from mjlab.managers.reward_manager import RewardManager, RewardTermCfg

from scripts import smoke_first_strike_instrumentation as smoke


ARM_TASKS = {
    "C": "Unitree-Z1-Hammer-CaT-Impulse",
    "D-prime": "Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy",
    "F": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    "E": "Unitree-Z1-Hammer-CaT-Impulse-Event",
}
# fq4x8 quality-conditioned campaign arms. F8 is intentionally absent here:
# its registered task is byte-identical to ARM_TASKS["F"] (smoke.ARM_TASKS),
# so the existing "F" contract already covers every F8 predicate.
QUALITY_ARM_TASKS = {
    "F0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
    "D0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
    "FQ-min": "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
}

COMMON_PREDICATES = (
    "finite_signals",
    "hammer_nail_contact",
    "normal_success",
    "lambda_live",
    "delivered_live",
    "hardware_qvel",
)

TRACKER_COMMON_PREDICATES = (
    "tracker_exists",
    "tracker_started",
    "productive_success",
    "terminal_reason_success",
    "delivered_event_impulse",
)
TRACKER_SPEED_PULSE_PREDICATES = (
    "impact_no_early_pulse",
    "impact_exactly_one_terminal_pulse",
)
TRACKER_DELIVERED_PULSE_PREDICATES = (
    "delivered_no_early_pulse",
    "delivered_exactly_one_terminal_pulse",
)
QUALITY_PREDICATES = ("quality_snapshot_valid", "quality_no_overflow")


def valid_record(*, arm: str, num_envs: int, device_type: str) -> dict[str, object]:
    """A literal record satisfying the frozen all-environment contract."""
    contract = smoke.ARM_CONTRACTS[arm]
    predicate_names = list(COMMON_PREDICATES)
    predicate_names.append(
        "manager_impact_zero" if contract.speed_zero else "manager_impact_positive"
    )
    predicate_names.append(
        "manager_delivered_zero"
        if contract.delivered_zero
        else "manager_delivered_positive"
    )
    if contract.speed_zero or contract.delivered_zero:
        if not contract.speed_zero:
            predicate_names.append("raw_impact_reader_finite")
        if not contract.delivered_zero:
            predicate_names.append("raw_delivered_reader_finite")
    else:
        predicate_names.append("raw_reward_readers_finite")
    if arm != "C":
        predicate_names.extend(TRACKER_COMMON_PREDICATES)
        if not contract.speed_zero:
            predicate_names.extend(TRACKER_SPEED_PULSE_PREDICATES)
        if not contract.delivered_zero:
            predicate_names.extend(TRACKER_DELIVERED_PULSE_PREDICATES)
    if contract.quality_required:
        predicate_names.extend(QUALITY_PREDICATES)
    actual_device = "cuda:0" if device_type == "cuda" else "cpu"
    return {
        "schema_version": "four-task-cuda-smoke-v1",
        "arm": arm,
        "task": contract.task,
        "num_envs": num_envs,
        "device_type": device_type,
        "visible_cuda_device_count": 1,
        "actual_tensor_device": actual_device,
        "environment_device": actual_device,
        "physical_device_identity": "NVIDIA-A100-0" if device_type == "cuda" else "cpu",
        "max_pre_arm_qvel_rad_s": 3.1415,
        "max_post_arm_qvel_rad_s": 3.1415,
        "predicate_counts": {
            name: {"passed": num_envs, "total": num_envs}
            for name in predicate_names
        },
        "impossible_success_n": 0,
        "lambda_dead_n": 0,
        "provenance": {
            "code": {
                "canonical_path": "/deployment/code",
                "expected_revision": "a" * 40,
                "pre_revision": "a" * 40,
                "post_revision": "a" * 40,
                "pre_dirty": False,
                "post_dirty": False,
            },
            "assets": {
                "canonical_path": "/deployment/assets",
                "expected_revision": "b" * 40,
                "pre_revision": "b" * 40,
                "post_revision": "b" * 40,
                "pre_dirty": False,
                "post_dirty": False,
            },
        },
    }


def test_valid_c_record_passes_without_tracker():
    """Removing C's documented tracker exemption must fail this test."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is True
    assert result["hardware_transport_qualified"] is True
    assert result["hardware_transport_label"] == "within_observed_qvel_rail"
    assert result["failed_predicates"] == []


@pytest.mark.parametrize("arm", ("F0", "D0", "FQ-min"))
def test_valid_quality_campaign_record_passes(arm):
    """A frozen fq4x8 record satisfying its own payout contract must qualify."""
    record = valid_record(arm=arm, num_envs=256, device_type="cuda")

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is True
    assert result["failed_predicates"] == []


def test_f0_speed_payout_must_be_exactly_zero_not_the_legacy_positive_predicate():
    """F0's zeroed speed term must be gated by manager_impact_zero, not positive."""
    record = valid_record(arm="F0", num_envs=256, device_type="cuda")
    assert "manager_impact_positive" not in record["predicate_counts"]
    record["predicate_counts"]["manager_impact_zero"] = {"passed": 200, "total": 256}  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["cuda_qualification_pass"] is False
    assert "manager_impact_zero" in result["failed_predicates"]
    assert "manager_impact_positive" not in result["failed_predicates"]


@pytest.mark.parametrize("arm", ("D0", "FQ-min"))
def test_delivered_payout_must_be_exactly_zero_for_d0_and_fq_min(arm):
    """D0/FQ-min's zeroed delivered term must be gated by manager_delivered_zero."""
    record = valid_record(arm=arm, num_envs=256, device_type="cuda")
    assert "manager_delivered_positive" not in record["predicate_counts"]
    record["predicate_counts"]["manager_delivered_zero"] = {"passed": 0, "total": 256}  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["cuda_qualification_pass"] is False
    assert "manager_delivered_zero" in result["failed_predicates"]


@pytest.mark.parametrize("predicate", QUALITY_PREDICATES)
def test_fq_min_requires_the_quality_snapshot_predicates(predicate):
    """A dead or overflowing quality snapshot must not silently qualify FQ-min."""
    record = valid_record(arm="FQ-min", num_envs=256, device_type="cuda")
    record["predicate_counts"][predicate] = {"passed": 0, "total": 256}  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["cuda_qualification_pass"] is False
    assert predicate in result["failed_predicates"]


@pytest.mark.parametrize("arm", ("F0", "D0"))
def test_non_quality_arms_never_require_the_quality_snapshot_predicates(arm):
    """F0/D0 have no quality sensor wired; requiring it would fail them forever."""
    record = valid_record(arm=arm, num_envs=256, device_type="cuda")

    assert "quality_snapshot_valid" not in record["predicate_counts"]
    result = smoke.evaluate_gate(record)
    assert result["cuda_qualification_pass"] is True


@pytest.mark.parametrize("arm", ("F0", "D0", "FQ-min"))
def test_a_zeroed_sides_pulse_predicates_are_not_required(arm):
    """A weight-0 term's func is never invoked, so its pulse-timing predicates
    must never gate qualification (they would fail forever, unrelated to
    whether the strike itself was productive)."""
    contract = smoke.ARM_CONTRACTS[arm]
    record = valid_record(arm=arm, num_envs=256, device_type="cuda")
    zeroed_pulse_predicates = (
        TRACKER_SPEED_PULSE_PREDICATES
        if contract.speed_zero
        else TRACKER_DELIVERED_PULSE_PREDICATES
    )
    for predicate in zeroed_pulse_predicates:
        assert predicate not in record["predicate_counts"]

    result = smoke.evaluate_gate(record)

    assert result["cuda_qualification_pass"] is True


def test_predicate_count_derives_failed_from_passed_and_total():
    """Making a count's failed value independent from its totals must fail."""
    count = smoke.PredicateCount(passed=255, total=256)

    assert count.failed == 1


@pytest.mark.parametrize("arm", ("D-prime", "F", "E"))
def test_first_strike_arm_requires_productive_success_and_one_terminal_pulse(arm):
    """Accepting an incomplete first-strike event must fail qualification."""
    record = valid_record(arm=arm, num_envs=256, device_type="cuda")
    record["predicate_counts"]["productive_success"] = {"passed": 255, "total": 256}  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["cuda_qualification_pass"] is False
    assert "productive_success" in result["failed_predicates"]


def test_finite_overspeed_is_transport_label_not_instrumentation_failure():
    """Conflating finite overspeed with instrumentation failure must fail."""
    record = valid_record(arm="E", num_envs=256, device_type="cuda")
    record["predicate_counts"]["hardware_qvel"] = {"passed": 255, "total": 256}  # type: ignore[index]
    record["max_post_arm_qvel_rad_s"] = 3.2

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is True
    assert result["hardware_transport_qualified"] is False
    assert result["hardware_transport_label"] == (
        "finite_qvel_rail_exceedance_simulation_only"
    )
    assert "hardware_qvel" not in result["failed_predicates"]


def test_cpu_can_pass_integration_but_never_cuda_qualification():
    """Treating CPU success as CUDA evidence must fail this test."""
    record = valid_record(arm="E", num_envs=8, device_type="cpu")

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is False


@pytest.mark.parametrize(
    ("predicate", "failed_name"),
    (
        ("finite_signals", "finite_signals"),
        ("hammer_nail_contact", "hammer_nail_contact"),
        ("lambda_live", "lambda_live"),
        ("delivered_live", "delivered_live"),
        ("raw_reward_readers_finite", "raw_reward_readers_finite"),
        ("manager_impact_positive", "manager_impact_positive"),
        ("manager_delivered_positive", "manager_delivered_positive"),
        ("normal_success", "normal_success"),
    ),
)
def test_common_physical_and_reward_predicates_require_every_environment(
    predicate, failed_name
):
    """Relaxing any common predicate from all environments must fail."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    record["predicate_counts"][predicate] = {"passed": 0, "total": 256}  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert failed_name in result["failed_predicates"]


@pytest.mark.parametrize("sentinel", ("impossible_success_n", "lambda_dead_n"))
def test_nonzero_sentinel_fails_gate(sentinel):
    """Ignoring either sampled-evaluator sentinel must fail this test."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    record[sentinel] = 1

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert sentinel in result["failed_predicates"]


@pytest.mark.parametrize(
    "predicate",
    (
        "tracker_exists",
        "tracker_started",
        "terminal_reason_success",
        "delivered_event_impulse",
        "impact_no_early_pulse",
        "delivered_no_early_pulse",
        "impact_exactly_one_terminal_pulse",
        "delivered_exactly_one_terminal_pulse",
    ),
)
def test_first_strike_terminal_and_pulse_predicates_are_universal(predicate):
    """Accepting the wrong terminal reason or pulse timing must fail."""
    record = valid_record(arm="E", num_envs=256, device_type="cuda")
    record["predicate_counts"][predicate] = {"passed": 255, "total": 256}  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert predicate in result["failed_predicates"]


@pytest.mark.parametrize(
    "mutation, failed_name",
    (
        (
            lambda record: record["provenance"]["code"].__setitem__("pre_dirty", True),  # type: ignore[index]
            "code_provenance",
        ),
        (
            lambda record: record["provenance"]["assets"].__setitem__("post_revision", "c" * 40),  # type: ignore[index]
            "assets_provenance",
        ),
    ),
)
def test_dirty_or_changed_provenance_fails_integration(mutation, failed_name):
    """Dropping clean, unchanged provenance validation must fail this test."""
    record = valid_record(arm="E", num_envs=256, device_type="cuda")
    mutation(record)

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert failed_name in result["failed_predicates"]


def test_schema_task_arm_and_cuda_residency_mismatches_fail_closed():
    """Accepting an invalid identity or false CUDA residency must fail."""
    cases = []
    wrong_task = valid_record(arm="E", num_envs=256, device_type="cuda")
    wrong_task["task"] = ARM_TASKS["F"]
    cases.append((wrong_task, "task_arm_pairing"))
    wrong_schema = valid_record(arm="E", num_envs=256, device_type="cuda")
    wrong_schema["schema_version"] = "unknown"
    cases.append((wrong_schema, "schema_version"))
    wrong_tensor = valid_record(arm="E", num_envs=256, device_type="cuda")
    wrong_tensor["actual_tensor_device"] = "cpu"
    cases.append((wrong_tensor, "actual_tensor_device"))

    for record, failed_name in cases:
        result = smoke.evaluate_gate(record)
        assert result["cuda_qualification_pass"] is False
        assert failed_name in result["failed_predicates"]


@pytest.mark.parametrize(
    ("field", "invalid_device"),
    (
        ("actual_tensor_device", "cuda"),
        ("actual_tensor_device", "cuda:1"),
        ("environment_device", "cuda"),
        ("environment_device", "cuda:1"),
    ),
)
def test_cuda_gate_requires_exact_cuda_zero_residency(field, invalid_device):
    """A CUDA type-only or wrong-index record must not qualify."""
    record = valid_record(arm="E", num_envs=256, device_type="cuda")
    record[field] = invalid_device

    result = smoke.evaluate_gate(record)

    assert result["cuda_qualification_pass"] is False
    assert field in result["failed_predicates"]


def test_cuda_qualification_requires_exact_campaign_population_and_single_gpu():
    """Relaxing CUDA's 256-env or one-GPU shape must fail qualification."""
    wrong_count = valid_record(arm="E", num_envs=8, device_type="cuda")
    multiple_gpus = valid_record(arm="E", num_envs=256, device_type="cuda")
    multiple_gpus["visible_cuda_device_count"] = 2

    for record, failed_name in ((wrong_count, "cuda_num_envs"), (multiple_gpus, "visible_cuda_device_count")):
        result = smoke.evaluate_gate(record)
        assert result["integration_pass"] is True
        assert result["cuda_qualification_pass"] is False
        assert failed_name in result["failed_predicates"]


def test_gate_preserves_counts_and_sorts_failures():
    """Dropping counts or returning nondeterministic diagnostics must fail."""
    record = valid_record(arm="E", num_envs=256, device_type="cuda")
    record["predicate_counts"]["lambda_live"] = {"passed": 255, "total": 256}  # type: ignore[index]
    record["predicate_counts"]["hardware_qvel"] = {"passed": 0, "total": 256}  # type: ignore[index]
    original_counts = deepcopy(record["predicate_counts"])

    result = smoke.evaluate_gate(record)

    assert result["predicate_counts"] == original_counts
    assert result["failed_predicates"] == sorted(result["failed_predicates"])


@pytest.mark.parametrize("field", ("max_pre_arm_qvel_rad_s", "max_post_arm_qvel_rad_s"))
def test_finite_qvel_maxima_above_rail_only_disqualify_hardware_transport(field):
    """Finite overspeed must remain visible without failing instrumentation."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    record[field] = 3.1416
    record["predicate_counts"]["hardware_qvel"] = {"passed": 0, "total": 256}  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is True
    assert result["hardware_transport_qualified"] is False
    assert "hardware_qvel" not in result["failed_predicates"]


@pytest.mark.parametrize("field", ("max_pre_arm_qvel_rad_s", "max_post_arm_qvel_rad_s"))
def test_nonfinite_qvel_maxima_fail_integration_and_cuda(field):
    """Nonfinite pre/post qvel must remain a hard instrumentation failure."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    record[field] = float("nan")

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert result["cuda_qualification_pass"] is False
    assert "qvel_nonfinite" in result["failed_predicates"]


def test_hardware_transport_predicate_count_must_be_well_formed():
    """Malformed transport evidence is schema failure, not finite overspeed."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    record["predicate_counts"]["hardware_qvel"] = {
        "passed": 255.0,
        "total": 256,
    }

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert "hardware_qvel_schema" in result["failed_predicates"]


@pytest.mark.parametrize("invalid_count", (True, 1.0))
def test_gpu_count_accepts_only_json_integer_one(invalid_count):
    """Treating JSON bool/float GPU counts as one must fail qualification."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    record["visible_cuda_device_count"] = invalid_count

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is False
    assert "visible_cuda_device_count" in result["failed_predicates"]


@pytest.mark.parametrize("field", ("impossible_success_n", "lambda_dead_n"))
@pytest.mark.parametrize("invalid_count", (False, 0.0))
def test_sentinels_accept_only_json_integer_zero(field, invalid_count):
    """Treating JSON bool/float sentinels as zero must fail integration."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    record[field] = invalid_count

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert field in result["failed_predicates"]


def test_provenance_requires_the_canonical_assets_key():
    """Accepting an undocumented singular asset key must fail integration."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    assets = record["provenance"].pop("assets")  # type: ignore[index]
    record["provenance"]["asset"] = assets  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert "assets_provenance" in result["failed_predicates"]


@pytest.mark.parametrize("repository", ("code", "assets"))
@pytest.mark.parametrize("invalid_path", (None, "relative/repo"))
def test_gate_requires_banked_absolute_canonical_repository_paths(
    repository, invalid_path
):
    """Evidence without an absolute canonical source path must fail closed."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    if invalid_path is None:
        del record["provenance"][repository]["canonical_path"]  # type: ignore[index]
    else:
        record["provenance"][repository]["canonical_path"] = invalid_path  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert f"{repository}_provenance" in result["failed_predicates"]


@pytest.mark.parametrize(
    ("arm", "impact_reader", "delivered_reader", "tracker_required", "i_ref", "saturate"),
    (
        ("C", "ImpactProgressTerm", "DeliveredImpulseTerm", False, 0.6094, None),
        (
            "D-prime",
            "FirstStrikeLegacyImpactRewardTerm",
            "FirstStrikeLegacyDeliveredRewardTerm",
            True,
            0.6094,
            None,
        ),
        (
            "F",
            "FirstStrikeImpactRewardTerm",
            "FirstStrikeDeliveredRewardTerm",
            True,
            0.3088,
            False,
        ),
        (
            "E",
            "FirstStrikeImpactRewardTerm",
            "FirstStrikeDeliveredRewardTerm",
            True,
            0.3088,
            True,
        ),
    ),
)
def test_arm_contracts_reject_drift_in_reader_reference_or_saturation(
    arm, impact_reader, delivered_reader, tracker_required, i_ref, saturate
):
    """Changing any frozen per-arm semantic must fail this literal contract test."""
    contract = smoke.ARM_CONTRACTS[arm]

    assert contract.task == ARM_TASKS[arm]
    assert contract.impact_reader == impact_reader
    assert contract.delivered_reader == delivered_reader
    assert contract.tracker_required is tracker_required
    assert contract.event_i_ref_n_s == i_ref
    assert contract.delivered_saturate is saturate


@pytest.mark.parametrize(
    (
        "arm", "impact_reader", "speed_zero", "delivered_zero",
        "quality_required", "impact_v_expected_n_s",
    ),
    (
        ("F0", "FirstStrikeImpactRewardTerm", True, False, False, None),
        ("D0", "FirstStrikeImpactRewardTerm", False, True, False, None),
        (
            "FQ-min", "FirstStrikeQualityImpactRewardTerm", False, True, True,
            1.4598331451416016,
        ),
    ),
)
def test_arm_contracts_pin_the_fq4x8_payout_semantics(
    arm, impact_reader, speed_zero, delivered_zero, quality_required,
    impact_v_expected_n_s,
):
    """Changing any fq4x8 arm's payout shape must fail this literal contract test."""
    contract = smoke.ARM_CONTRACTS[arm]

    assert contract.task == QUALITY_ARM_TASKS[arm]
    assert contract.impact_reader == impact_reader
    assert contract.delivered_reader == "FirstStrikeDeliveredRewardTerm"
    assert contract.tracker_required is True
    assert contract.event_i_ref_n_s == 0.3088
    assert contract.delivered_saturate is False
    assert contract.speed_zero is speed_zero
    assert contract.delivered_zero is delivered_zero
    assert contract.quality_required is quality_required
    assert contract.impact_v_expected_n_s == impact_v_expected_n_s


@pytest.mark.parametrize(
    ("field", "invalid_value", "failed_name"),
    (
        ("device_type", "cpu", "device_type"),
        ("visible_cuda_device_count", 0, "visible_cuda_device_count"),
        ("actual_tensor_device", "cpu", "actual_tensor_device"),
        ("environment_device", "cpu", "environment_device"),
        ("physical_device_identity", "", "physical_device_identity"),
        ("num_envs", 8, "cuda_num_envs"),
    ),
)
def test_each_cuda_qualification_requirement_fails_independently(
    field, invalid_value, failed_name
):
    """Removing any CUDA-residency/shape requirement must fail this test."""
    record = valid_record(
        arm="C",
        num_envs=8 if field == "num_envs" else 256,
        device_type="cuda",
    )
    record[field] = invalid_value

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is False
    assert failed_name in result["failed_predicates"]


LITERAL_IMPULSE_LIMITS = (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)


def _live_configs(task: str):
    """Return independent registered configurations for a deliberate drift."""
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    import src.tasks  # noqa: F401  (register hammer tasks)

    return deepcopy(load_env_cfg(task, play=False)), deepcopy(load_rl_cfg(task))


def _install_live_config(monkeypatch, task, training_cfg, rl_cfg):
    monkeypatch.setattr(
        smoke, "load_env_cfg", lambda requested, play=False: training_cfg, raising=False
    )
    monkeypatch.setattr(smoke, "load_rl_cfg", lambda requested: rl_cfg, raising=False)


@pytest.mark.parametrize("task", tuple(ARM_TASKS.values()))
def test_live_contract_matches_each_registered_task_and_disables_only_row_diagnostic(task):
    """Accepting a drifted registered config or a live row metric must fail."""
    training_cfg, rl_cfg, digest = smoke.validate_live_contract(task)

    assert digest["impact_weight"] == 8.0
    assert digest["delivered_weight"] == 2.0
    assert digest["imp_max_p"] == 0.0
    assert tuple(digest["impulse_limits_n_m_s"]) == LITERAL_IMPULSE_LIMITS
    assert digest["rl_clip_actions"] == 1.0
    assert digest["registered_substep_impulse_rows_enabled"] is True
    assert digest["event_i_ref_n_s"] == smoke.TASK_CONTRACTS[task].event_i_ref_n_s

    diagnostic_cfg = smoke.make_diagnostic_cfg(training_cfg, num_envs=8)

    assert diagnostic_cfg.metrics["substep_impulse_rows"].params["enabled"] is False
    assert diagnostic_cfg.scene.num_envs == 8
    assert diagnostic_cfg.events["reset_robot_joints"].params["position_range"] == (0.0, 0.0)
    assert diagnostic_cfg.observations["actor"].enable_corruption is False
    assert diagnostic_cfg.observations["critic"].enable_corruption is False
    assert diagnostic_cfg.events["reset_robot_joints"].params["position_range"] != training_cfg.events["reset_robot_joints"].params["position_range"]
    assert training_cfg.metrics["substep_impulse_rows"].params["enabled"] is True


@pytest.mark.parametrize(
    ("arm", "expected_impact_weight", "expected_delivered_weight"),
    (
        ("F0", 0.0, 2.0),
        ("D0", 8.0, 0.0),
        ("FQ-min", 8.0, 0.0),
    ),
)
def test_live_contract_accepts_each_quality_arm_with_its_own_weight_contract(
    arm, expected_impact_weight, expected_delivered_weight
):
    """Silently defaulting F0/D0/FQ-min's maximize weights to 8/2 must fail."""
    task = QUALITY_ARM_TASKS[arm]

    training_cfg, rl_cfg, digest = smoke.validate_live_contract(task)

    assert digest["impact_weight"] == expected_impact_weight
    assert digest["delivered_weight"] == expected_delivered_weight
    assert digest["imp_max_p"] == 0.0
    assert tuple(digest["impulse_limits_n_m_s"]) == LITERAL_IMPULSE_LIMITS
    assert digest["rl_clip_actions"] == 1.0
    assert digest["registered_substep_impulse_rows_enabled"] is True
    assert digest["event_i_ref_n_s"] == smoke.TASK_CONTRACTS[task].event_i_ref_n_s

    diagnostic_cfg = smoke.make_diagnostic_cfg(training_cfg, num_envs=8)

    assert diagnostic_cfg.metrics["substep_impulse_rows"].params["enabled"] is False


def test_live_contract_rejects_fq_min_v_expected_drift(monkeypatch):
    """Silently accepting a drifted quality-speed normalizer must fail this test."""
    task = QUALITY_ARM_TASKS["FQ-min"]
    training_cfg, rl_cfg = _live_configs(task)
    training_cfg.rewards["impact_progress"].params["v_expected"] = 1.0
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)

    with pytest.raises(ValueError, match="v_expected"):
        smoke.validate_live_contract(task)


@pytest.mark.parametrize(
    ("task", "saturate"),
    (
        (ARM_TASKS["F"], True),
        (ARM_TASKS["E"], False),
    ),
)
def test_live_contract_rejects_swapped_event_saturation(monkeypatch, task, saturate):
    """Swapping F/E delivered payout saturation must fail validation."""
    training_cfg, rl_cfg = _live_configs(task)
    training_cfg.rewards["delivered_impulse"].params["saturate"] = saturate
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)

    with pytest.raises(ValueError, match="saturate"):
        smoke.validate_live_contract(task)


@pytest.mark.parametrize("task", tuple(ARM_TASKS.values()))
@pytest.mark.parametrize(
    ("reward_name", "expected"),
    (
        ("impact_progress", "impact reader"),
        ("delivered_impulse", "delivered reader"),
    ),
)
def test_live_contract_rejects_wrong_reward_reader(
    monkeypatch, task, reward_name, expected
):
    """Replacing either configured reward reader must fail validation."""
    training_cfg, rl_cfg = _live_configs(task)
    training_cfg.rewards[reward_name].func = object
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)

    with pytest.raises(ValueError, match=expected):
        smoke.validate_live_contract(task)


@pytest.mark.parametrize("task", (ARM_TASKS["D-prime"], ARM_TASKS["F"], ARM_TASKS["E"]))
@pytest.mark.parametrize("mutation", ("missing", "wrong", "per_substep", "reduce"))
def test_live_contract_requires_exact_first_strike_tracker(monkeypatch, task, mutation):
    """Removing or replacing a first-event tracker must fail validation."""
    training_cfg, rl_cfg = _live_configs(task)
    if mutation == "missing":
        del training_cfg.metrics["first_strike"]
    elif mutation == "wrong":
        training_cfg.metrics["first_strike"].func = object
    elif mutation == "per_substep":
        training_cfg.metrics["first_strike"].per_substep = False
    else:
        training_cfg.metrics["first_strike"].reduce = "mean"
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)

    with pytest.raises(ValueError, match="FirstStrikeEventTracker"):
        smoke.validate_live_contract(task)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    (
        (lambda cfg: cfg.rewards["impact_progress"].__setattr__("weight", 7.0), "weights"),
        (lambda cfg: cfg.rewards["delivered_impulse"].__setattr__("weight", 1.0), "weights"),
        (lambda cfg: cfg.rewards["delivered_impulse"].params.__setitem__("i_ref", 99.0), "i_ref"),
        (lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("imp_max_p", 0.5), "imp_max_p"),
        (lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("imp_limit", [1.0] * 6), "impulse limits"),
    ),
)
def test_live_contract_rejects_reward_or_cap_drift(monkeypatch, mutate, expected):
    """Changing any live reward/cap safety value must fail validation."""
    task = ARM_TASKS["E"]
    training_cfg, rl_cfg = _live_configs(task)
    mutate(training_cfg)
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)

    with pytest.raises(ValueError, match=expected):
        smoke.validate_live_contract(task)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("entity_name", "wrong_robot"),
        ("clip", (-1.0, 1.0)),
        ("actuator_names", ("joint6",)),
        ("frame_type", "body"),
        ("frame_name", "wrong_site"),
        ("use_relative_mode", False),
        ("delta_pos_scale", 0.1),
        ("delta_ori_scale", 0.5),
        ("damping", 0.1),
        ("max_dq", 0.4),
        ("position_weight", 0.5),
        ("orientation_weight", 1.0),
        ("joint_limit_weight", 1.0),
        ("posture_weight", 1.0),
        ("posture_target", (0.0,) * 6),
    ),
)
def test_live_contract_rejects_every_diffik_field_drift(monkeypatch, field, value):
    """Changing any frozen Differential-IK field must fail validation."""
    task = ARM_TASKS["C"]
    training_cfg, rl_cfg = _live_configs(task)
    setattr(training_cfg.actions["ik_hammer_head"], field, value)
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)

    with pytest.raises(ValueError, match="action signature"):
        smoke.validate_live_contract(task)


@pytest.mark.parametrize("field", ("stiffness", "damping", "effort_limit", "armature"))
def test_live_contract_rejects_every_fixed_actuator_field_drift(monkeypatch, field):
    """Changing a fixed-impedance actuator value must fail validation."""
    task = ARM_TASKS["C"]
    training_cfg, rl_cfg = _live_configs(task)
    actuator = training_cfg.scene.entities["robot"].articulation.actuators[0]
    setattr(actuator, field, getattr(actuator, field) + 1.0)
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)

    with pytest.raises(ValueError, match="actuator signature"):
        smoke.validate_live_contract(task)


def test_live_contract_rejects_imported_impulse_limit_drift(monkeypatch):
    """Mutating the imported cap constant must fail against literal manufacturer caps."""
    task = ARM_TASKS["C"]
    training_cfg, rl_cfg = _live_configs(task)
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)
    monkeypatch.setattr(smoke, "IMP_J_LIMIT", (1.0,) * 6)

    with pytest.raises(ValueError, match="manufacturer"):
        smoke.validate_live_contract(task)


def test_live_contract_rejects_rl_action_clip_drift(monkeypatch):
    """Allowing policy actions beyond the frozen clip must fail validation."""
    task = ARM_TASKS["C"]
    training_cfg, rl_cfg = _live_configs(task)
    rl_cfg.clip_actions = 0.5
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)

    with pytest.raises(ValueError, match="clip_actions"):
        smoke.validate_live_contract(task)


def test_diagnostic_config_rejects_enabled_row_metric(monkeypatch):
    """Keeping expensive row reconstruction enabled in the smoke config must fail."""
    task = ARM_TASKS["C"]
    training_cfg, rl_cfg = _live_configs(task)
    _install_live_config(monkeypatch, task, training_cfg, rl_cfg)

    validated_cfg, _, _ = smoke.validate_live_contract(task)
    diagnostic_cfg = smoke.make_diagnostic_cfg(validated_cfg, num_envs=8)

    assert diagnostic_cfg.metrics["substep_impulse_rows"].params["enabled"] is False


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def init_clean_repo(path: Path) -> Path:
    """Build a minimal committed repository for provenance behavior tests."""
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "smoke@example.test")
    _git(path, "config", "user.name", "Smoke Test")
    (path / "tracked.txt").write_text("clean\n")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-qm", "initial")
    return path


def git_head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def test_atomic_writer_rejects_repo_local_and_existing_paths(tmp_path):
    """Writing inside a source repo or over an artifact must be rejected."""
    code_repo = init_clean_repo(tmp_path / "code")
    asset_repo = init_clean_repo(tmp_path / "assets")

    with pytest.raises(ValueError, match="outside both repositories"):
        smoke.validate_output_path(code_repo / "result.json", code_repo, asset_repo)

    out = tmp_path / "external" / "result.json"
    out.parent.mkdir()
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        smoke.write_json_atomic(out, {"schema_version": smoke.SCHEMA_VERSION})


def test_output_path_resolves_symlinks_before_rejecting_repo_descendants(tmp_path):
    """A symlink route into either canonical repository must be rejected."""
    code_repo = init_clean_repo(tmp_path / "code")
    asset_repo = init_clean_repo(tmp_path / "assets")
    indirect = tmp_path / "indirect"
    indirect.symlink_to(code_repo, target_is_directory=True)

    with pytest.raises(ValueError, match="outside both repositories"):
        smoke.validate_output_path(indirect / "result.json", code_repo, asset_repo)


def test_output_path_does_not_follow_a_dangling_leaf_symlink(tmp_path):
    """Validation must retain an external leaf name instead of resolving its target."""
    code_repo = init_clean_repo(tmp_path / "code")
    asset_repo = init_clean_repo(tmp_path / "assets")
    out = tmp_path / "external" / "result.json"
    out.parent.mkdir()
    target = code_repo / "must-remain-untouched.json"
    out.symlink_to(target)

    validated = smoke.validate_output_path(out, code_repo, asset_repo)

    assert validated == out.parent.resolve() / out.name
    assert out.is_symlink()
    assert not target.exists()


def test_provenance_rejects_dirty_repo(tmp_path):
    """Untracked deployment state must make provenance fail closed."""
    repo = init_clean_repo(tmp_path / "repo")
    (repo / "untracked.txt").write_text("dirty")

    with pytest.raises(ValueError, match="dirty"):
        smoke.read_repo_provenance(repo, git_head(repo))


def test_provenance_rejects_wrong_expected_revision(tmp_path):
    """A clean but different revision must not count as the deployment revision."""
    repo = init_clean_repo(tmp_path / "repo")

    with pytest.raises(ValueError, match="expected revision"):
        smoke.read_repo_provenance(repo, "0" * 40)


def test_provenance_returns_canonical_clean_matching_snapshot(tmp_path):
    """A clean repository returns a full expected/observed canonical snapshot."""
    repo = init_clean_repo(tmp_path / "repo")
    head = git_head(repo)

    provenance = smoke.read_repo_provenance(repo, head)

    assert provenance == {
        "canonical_path": str(repo.resolve()),
        "expected_revision": head,
        "revision": head,
        "dirty": False,
    }


def test_provenance_rejects_noncanonical_revision_format(tmp_path):
    """Abbreviated or uppercase revision inputs cannot qualify provenance."""
    repo = init_clean_repo(tmp_path / "repo")

    with pytest.raises(ValueError, match="full 40-character lowercase hexadecimal"):
        smoke.read_repo_provenance(repo, git_head(repo)[:12])


def test_provenance_rejects_repository_subdirectory(tmp_path):
    """A path below, rather than equal to, the canonical Git root must fail."""
    repo = init_clean_repo(tmp_path / "repo")
    nested = repo / "nested"
    nested.mkdir()

    with pytest.raises(ValueError, match="repository root"):
        smoke.read_repo_provenance(nested, git_head(repo))


@pytest.mark.parametrize("invalid_revision", ("a" * 39, "A" * 40, "g" * 40))
def test_gate_rejects_noncanonical_literal_provenance_revisions(invalid_revision):
    """Malformed but equal expected/pre/post revisions cannot qualify a record."""
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    for revision_field in ("expected_revision", "pre_revision", "post_revision"):
        record["provenance"]["code"][revision_field] = invalid_revision  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["integration_pass"] is False
    assert "code_provenance" in result["failed_predicates"]


def test_post_run_revision_or_dirty_state_invalidates_payload():
    """A changed post-run code repository must invalidate the gated record."""
    record = valid_record(arm="E", num_envs=256, device_type="cuda")
    record["provenance"]["code"]["post_dirty"] = True  # type: ignore[index]

    result = smoke.evaluate_gate(record)

    assert result["cuda_qualification_pass"] is False
    assert "code_provenance" in result["failed_predicates"]


def test_atomic_writer_publishes_with_exclusive_link_not_replace(tmp_path, monkeypatch):
    """Publication must avoid replace so a final path can never be overwritten."""
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        smoke.os,
        "replace",
        lambda source, target: (_ for _ in ()).throw(AssertionError("forbidden")),
    )

    smoke.write_json_atomic(out, {"schema_version": smoke.SCHEMA_VERSION})

    assert json.loads(out.read_text()) == {"schema_version": smoke.SCHEMA_VERSION}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_writer_rejects_dangling_symlink_collision(tmp_path):
    """A dangling final symlink is an occupied name and must not be replaced."""
    out = tmp_path / "result.json"
    out.symlink_to(tmp_path / "missing.json")

    with pytest.raises(FileExistsError):
        smoke.write_json_atomic(out, {"schema_version": smoke.SCHEMA_VERSION})

    assert out.is_symlink()
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_writer_rejects_concurrent_creator_and_cleans_temp(tmp_path, monkeypatch):
    """A creator winning between prepare and publish cannot be overwritten."""
    out = tmp_path / "result.json"

    real_link = smoke.os.link

    def concurrent_creator(source, destination):
        Path(destination).write_text('{"winner": true}\n')
        real_link(source, destination)

    monkeypatch.setattr(smoke.os, "link", concurrent_creator)

    with pytest.raises(FileExistsError):
        smoke.write_json_atomic(out, {"schema_version": smoke.SCHEMA_VERSION})

    assert json.loads(out.read_text()) == {"winner": True}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_writer_fsyncs_destination_directory_after_publish(tmp_path, monkeypatch):
    """Successful publication must sync the destination directory entry."""
    out = tmp_path / "result.json"
    original_fsync = smoke.os.fsync
    synced_directory: list[bool] = []

    def record_fsync(fd):
        synced_directory.append(stat.S_ISDIR(smoke.os.fstat(fd).st_mode))
        return original_fsync(fd)

    monkeypatch.setattr(smoke.os, "fsync", record_fsync)

    smoke.write_json_atomic(out, {"schema_version": smoke.SCHEMA_VERSION})

    assert any(synced_directory)
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_writer_propagates_directory_fsync_failure_after_publication(
    tmp_path, monkeypatch
):
    """A durability failure reports failure without retracting the complete publication."""
    out = tmp_path / "result.json"
    original_fsync = smoke.os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(smoke.os.fstat(fd).st_mode):
            raise OSError("directory fsync injected")
        return original_fsync(fd)

    monkeypatch.setattr(smoke.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="directory fsync injected"):
        smoke.write_json_atomic(out, {"schema_version": smoke.SCHEMA_VERSION})

    assert json.loads(out.read_text()) == {"schema_version": smoke.SCHEMA_VERSION}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_writer_propagates_generic_link_failure_and_cleans_temp(
    tmp_path, monkeypatch
):
    """Unsupported hard links fail closed without a replace fallback or partial final."""
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        smoke.os,
        "link",
        lambda source, destination: (_ for _ in ()).throw(OSError("link injected")),
    )

    with pytest.raises(OSError, match="link injected"):
        smoke.write_json_atomic(out, {"schema_version": smoke.SCHEMA_VERSION})

    assert not out.exists()
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


class CountingReward:
    """Stateful reward fake whose returned tensor and reset calls are observable."""

    def __init__(self, value):
        self.value = value
        self.calls = 0
        self.reset_calls = []

    def __call__(self, env, **params):
        del env, params
        self.calls += 1
        return self.value

    def reset(self, env_ids):
        self.reset_calls.append(env_ids.clone())


def test_raw_reward_tap_delegates_exactly_once_and_preserves_nonfinite_raw():
    """Calling or sanitizing the original reward more than once must fail."""
    original_value = torch.tensor([float("nan"), 2.0])
    original = CountingReward(original_value)
    tap = smoke.RawRewardTap(
        original, num_envs=2, device=torch.device("cpu")
    )

    returned = tap(SimpleNamespace())

    assert returned is original_value
    assert original.calls == 1
    assert torch.isnan(returned[0])
    assert torch.isnan(tap.last_raw[0])
    torch.testing.assert_close(tap.last_raw[1], torch.tensor(2.0))
    assert tap.all_finite.tolist() == [False, True]
    assert tap.positive_count.tolist() == [0, 1]
    assert tap.first_positive_step.tolist() == [-1, 0]
    assert tap.last_positive_step.tolist() == [-1, 0]


def test_raw_reward_tap_reset_delegates_once_and_clears_only_selected_envs():
    """Resetting one environment must not erase another environment's tap state."""
    original = CountingReward(torch.tensor([1.0, 2.0]))
    tap = smoke.RawRewardTap(
        original, num_envs=2, device=torch.device("cpu")
    )
    tap(SimpleNamespace())
    original.value = torch.tensor([3.0, 0.0])
    tap(SimpleNamespace())
    untouched_last_raw = tap.last_raw[1].clone()
    untouched_count = tap.positive_count[1].clone()

    env_ids = torch.tensor([0])
    tap.reset(env_ids)

    assert len(original.reset_calls) == 1
    torch.testing.assert_close(original.reset_calls[0], env_ids)
    assert tap.last_raw[0] == 0.0
    assert tap.all_finite[0]
    assert tap.positive_count[0] == 0
    assert tap.call_count[0] == 0
    assert tap.first_positive_step[0] == -1
    torch.testing.assert_close(tap.last_raw[1], untouched_last_raw)
    torch.testing.assert_close(tap.positive_count[1], untouched_count)


def _make_taps():
    impact = CountingReward(torch.zeros(2))
    delivered = CountingReward(torch.zeros(2))
    return (
        impact,
        delivered,
        smoke.RawRewardTap(impact, num_envs=2, device=torch.device("cpu")),
        smoke.RawRewardTap(delivered, num_envs=2, device=torch.device("cpu")),
    )


def _capture_rewards(impact_original, delivered_original, impact_tap, delivered_tap, impact, delivered):
    impact_original.value = torch.tensor(impact)
    delivered_original.value = torch.tensor(delivered)
    impact_tap(SimpleNamespace())
    delivered_tap(SimpleNamespace())


def _control_payload(**overrides):
    payload = {
        "reset_buf": torch.tensor([True, False]),
        "terminal_depth": torch.tensor([0.031, 0.010]),
        "terminal_success": torch.tensor([True, False]),
        "episode_lambda": torch.tensor(
            [[0.11, 0.12, 0.13, 0.14, 0.15, 0.16], [0.01] * 6]
        ),
        "episode_delivered": torch.tensor([0.61, 0.10]),
        "tracker_exists": torch.tensor([True, True]),
        "tracker_started": torch.tensor([True, False]),
        "tracker_finalized": torch.tensor([True, False]),
        "tracker_productive": torch.tensor([True, False]),
        "tracker_reason": torch.tensor([1, 0]),
        "tracker_delivered": torch.tensor([0.55, 0.02]),
        "tracker_contact_quality_valid": torch.tensor([False, False]),
        "tracker_contact_quality_overflow": torch.tensor([False, False]),
        "manager_impact": torch.tensor([8.0, 0.0]),
        "manager_delivered": torch.tensor([2.0, 0.0]),
    }
    payload.update(overrides)
    return payload


def test_terminal_latch_captures_each_environment_once_with_episode_aggregates():
    """A later episode or another environment's terminal must not overwrite fields."""
    impact, delivered, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [1.0, 0.0], [0.5, 0.0]
    )
    recorder.capture_substep(
        contact=torch.tensor([True, False]),
        axial_force=torch.tensor([120.0, 10.0]),
        pre_arm_qvel=torch.tensor([[1.0] * 6, [0.1] * 6]),
        post_arm_qvel=torch.tensor([[1.5] * 6, [0.2] * 6]),
        pre_depth=torch.tensor([0.030, 0.009]),
        post_depth=torch.tensor([0.031, 0.010]),
    )
    recorder.capture_control_step(**_control_payload())
    first_env_terminal = {
        name: getattr(recorder, name)[0].clone()
        for name in (
            "terminal_depth",
            "terminal_success",
            "terminal_lambda",
            "terminal_delivered",
            "terminal_tracker_started",
            "terminal_tracker_finalized",
            "terminal_tracker_productive",
            "terminal_tracker_reason",
            "terminal_tracker_delivered",
            "terminal_raw_impact_finite",
            "terminal_raw_delivered_finite",
            "terminal_manager_impact",
            "terminal_manager_delivered",
            "terminal_control_step",
            "terminal_impact_positive_count",
            "terminal_delivered_positive_count",
            "terminal_impact_first_positive_step",
            "terminal_delivered_first_positive_step",
            "terminal_contact_seen",
            "terminal_force_max",
            "terminal_pre_qvel_max",
            "terminal_post_qvel_max",
            "terminal_pre_depth_max",
            "terminal_post_depth_max",
        )
    }

    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [9.0, 1.5], [9.0, 0.75]
    )
    recorder.capture_substep(
        contact=torch.tensor([True, True]),
        axial_force=torch.tensor([999.0, 80.0]),
        pre_arm_qvel=torch.tensor([[3.0] * 6, [1.1] * 6]),
        post_arm_qvel=torch.tensor([[3.1] * 6, [1.2] * 6]),
        pre_depth=torch.tensor([0.001, 0.029]),
        post_depth=torch.tensor([0.002, 0.031]),
    )
    recorder.capture_control_step(
        **_control_payload(
            reset_buf=torch.tensor([True, True]),
            terminal_depth=torch.tensor([0.001, 0.031]),
            terminal_success=torch.tensor([False, True]),
            episode_lambda=torch.tensor([[9.0] * 6, [0.21] * 6]),
            episode_delivered=torch.tensor([9.0, 0.72]),
            tracker_started=torch.tensor([False, True]),
            tracker_finalized=torch.tensor([False, True]),
            tracker_productive=torch.tensor([False, True]),
            tracker_reason=torch.tensor([0, 1]),
            tracker_delivered=torch.tensor([9.0, 0.66]),
            manager_impact=torch.tensor([99.0, 12.0]),
            manager_delivered=torch.tensor([99.0, 4.0]),
        )
    )

    assert recorder.terminal_seen.tolist() == [True, True]
    for name, expected in first_env_terminal.items():
        torch.testing.assert_close(getattr(recorder, name)[0], expected)
    torch.testing.assert_close(recorder.terminal_depth[1], torch.tensor(0.031))
    assert recorder.terminal_success.tolist() == [True, True]
    torch.testing.assert_close(recorder.terminal_lambda[1], torch.tensor([0.21] * 6))
    torch.testing.assert_close(recorder.terminal_delivered[1], torch.tensor(0.72))
    assert recorder.terminal_tracker_started.tolist() == [True, True]
    assert recorder.terminal_tracker_finalized.tolist() == [True, True]
    assert recorder.terminal_tracker_productive.tolist() == [True, True]
    assert recorder.terminal_tracker_reason.tolist() == [1, 1]
    torch.testing.assert_close(recorder.terminal_tracker_delivered[1], torch.tensor(0.66))
    assert recorder.terminal_raw_impact_finite.tolist() == [True, True]
    assert recorder.terminal_raw_delivered_finite.tolist() == [True, True]
    torch.testing.assert_close(recorder.terminal_manager_impact, torch.tensor([8.0, 12.0]))
    torch.testing.assert_close(recorder.terminal_manager_delivered, torch.tensor([2.0, 4.0]))
    assert recorder.terminal_control_step.tolist() == [0, 1]
    assert recorder.terminal_impact_positive_count.tolist() == [1, 1]
    assert recorder.terminal_delivered_positive_count.tolist() == [1, 1]
    assert recorder.terminal_impact_first_positive_step.tolist() == [0, 1]
    assert recorder.terminal_delivered_first_positive_step.tolist() == [0, 1]
    assert recorder.terminal_impact_last_positive_step.tolist() == [0, 1]
    assert recorder.terminal_delivered_last_positive_step.tolist() == [0, 1]
    assert recorder.terminal_contact_seen.tolist() == [True, True]
    torch.testing.assert_close(recorder.terminal_force_max, torch.tensor([120.0, 80.0]))
    torch.testing.assert_close(recorder.terminal_pre_qvel_max, torch.tensor([1.0, 1.1]))
    torch.testing.assert_close(recorder.terminal_post_qvel_max, torch.tensor([1.5, 1.2]))
    torch.testing.assert_close(recorder.terminal_pre_depth_max, torch.tensor([0.030, 0.029]))
    torch.testing.assert_close(recorder.terminal_post_depth_max, torch.tensor([0.031, 0.031]))


def test_terminal_latch_preserves_raw_nonfinite_status_per_environment():
    """RewardManager sanitization must not turn a raw NaN into a finite terminal record."""
    impact, delivered, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    _capture_rewards(
        impact,
        delivered,
        impact_tap,
        delivered_tap,
        [float("nan"), 1.0],
        [0.5, float("inf")],
    )

    recorder.capture_control_step(
        **_control_payload(reset_buf=torch.tensor([True, True]))
    )

    assert recorder.terminal_raw_impact_finite.tolist() == [False, True]
    assert recorder.terminal_raw_delivered_finite.tolist() == [True, False]


def test_terminal_latch_rejects_a_single_positive_pulse_before_terminal():
    """A one-shot pulse on an earlier control step must not count as terminal."""
    impact, delivered, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [1.0, 1.0], [1.0, 1.0]
    )
    recorder.capture_control_step(
        **_control_payload(
            reset_buf=torch.tensor([False, False]),
            manager_impact=torch.tensor([8.0, 8.0]),
            manager_delivered=torch.tensor([2.0, 2.0]),
        )
    )
    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [0.0, 0.0], [0.0, 0.0]
    )
    recorder.capture_control_step(
        **_control_payload(
            reset_buf=torch.tensor([True, True]),
            manager_impact=torch.tensor([0.0, 0.0]),
            manager_delivered=torch.tensor([0.0, 0.0]),
        )
    )

    result = recorder.finalize()

    assert result["predicate_counts"]["impact_no_early_pulse"]["passed"] == 0
    assert result["predicate_counts"]["delivered_no_early_pulse"]["passed"] == 0
    assert (
        result["predicate_counts"]["impact_exactly_one_terminal_pulse"]["passed"]
        == 0
    )
    assert (
        result["predicate_counts"]["delivered_exactly_one_terminal_pulse"][
            "passed"
        ]
        == 0
    )


@pytest.mark.parametrize("missing_reader", ("impact", "delivered"))
def test_terminal_latch_requires_both_raw_readers_to_have_been_called(
    missing_reader,
):
    """Default-finite tap buffers must not let an unwired reader pass C."""
    _, _, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    if missing_reader == "impact":
        delivered_tap(SimpleNamespace())
    else:
        impact_tap(SimpleNamespace())

    recorder.capture_control_step(
        **_control_payload(
            reset_buf=torch.tensor([True, True]),
            manager_impact=torch.tensor([8.0, 8.0]),
            manager_delivered=torch.tensor([2.0, 2.0]),
        )
    )

    result = recorder.finalize()

    assert (
        result["predicate_counts"]["raw_reward_readers_finite"]["passed"] == 0
    )


def test_terminal_latch_splits_raw_reader_liveness_per_side():
    """A weight-0 side's reader is never invoked (mjlab skips it); its
    liveness must be checkable independently of the other, active side."""
    _, _, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    delivered_tap(SimpleNamespace())  # F0-style: only delivered is ever called

    recorder.capture_control_step(
        **_control_payload(
            reset_buf=torch.tensor([True, True]),
            manager_impact=torch.tensor([0.0, 0.0]),
            manager_delivered=torch.tensor([2.0, 2.0]),
        )
    )

    result = recorder.finalize()

    assert result["predicate_counts"]["raw_impact_reader_finite"]["passed"] == 0
    assert result["predicate_counts"]["raw_delivered_reader_finite"]["passed"] == 2


def test_finalize_computes_exact_zero_manager_payout_predicates():
    """A zero-weighted term's payout must register as exactly zero, distinct
    from merely "not positive" (which would also admit a NaN or negative)."""
    impact, delivered, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [1.0, 1.0], [1.0, 1.0]
    )
    recorder.capture_control_step(
        **_control_payload(
            reset_buf=torch.tensor([True, True]),
            manager_impact=torch.tensor([0.0, 8.0]),
            manager_delivered=torch.tensor([2.0, 0.0]),
        )
    )

    result = recorder.finalize()

    assert result["predicate_counts"]["manager_impact_zero"]["passed"] == 1
    assert result["predicate_counts"]["manager_impact_positive"]["passed"] == 1
    assert result["predicate_counts"]["manager_delivered_zero"]["passed"] == 1
    assert result["predicate_counts"]["manager_delivered_positive"]["passed"] == 1


def test_finalize_computes_quality_snapshot_predicates_from_the_tracker_latch():
    """A dead or overflowing quality snapshot must not silently register valid."""
    impact, delivered, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [1.0, 1.0], [1.0, 1.0]
    )
    recorder.capture_control_step(
        **_control_payload(
            reset_buf=torch.tensor([True, True]),
            tracker_contact_quality_valid=torch.tensor([True, False]),
            tracker_contact_quality_overflow=torch.tensor([False, True]),
        )
    )

    result = recorder.finalize()

    assert result["predicate_counts"]["quality_snapshot_valid"]["passed"] == 1
    assert result["predicate_counts"]["quality_no_overflow"]["passed"] == 1


def test_terminal_latch_includes_transient_depth_nonfinite_in_finite_signals():
    """A transient NaN in either coherent depth channel must fail finiteness."""
    impact, delivered, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [1.0, 1.0], [1.0, 1.0]
    )
    recorder.capture_substep(
        contact=torch.tensor([True, True]),
        axial_force=torch.tensor([1.0, 1.0]),
        pre_arm_qvel=torch.zeros(2, 6),
        post_arm_qvel=torch.zeros(2, 6),
        pre_depth=torch.tensor([float("nan"), 0.01]),
        post_depth=torch.tensor([0.02, 0.02]),
    )
    recorder.capture_control_step(
        **_control_payload(reset_buf=torch.tensor([True, True]))
    )

    result = recorder.finalize()

    assert result["predicate_counts"]["finite_signals"]["passed"] == 1


class StatefulManagerReward:
    """Real RewardManager term used to verify live config replacement/reset."""

    def __init__(self, cfg, env):
        del cfg, env
        self.calls = 0
        self.reset_calls = []

    def __call__(self, env, raw):
        del env
        self.calls += 1
        return raw

    def reset(self, env_ids):
        self.reset_calls.append(
            env_ids.clone() if isinstance(env_ids, torch.Tensor) else env_ids
        )


def test_install_reward_taps_integrates_with_real_reward_manager_compute_and_reset():
    """Installing anywhere but live term configs breaks compute/reset delegation."""
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        max_episode_length_s=1.0,
        scene={},
    )
    manager = RewardManager(
        {
            "impact_progress": RewardTermCfg(
                func=StatefulManagerReward,
                weight=8.0,
                params={"raw": torch.tensor([float("nan"), 2.0])},
            ),
            "delivered_impulse": RewardTermCfg(
                func=StatefulManagerReward,
                weight=2.0,
                params={"raw": torch.tensor([1.0, float("inf")])},
            ),
        },
        env,
    )
    env.reward_manager = manager
    impact_original = manager.get_term_cfg("impact_progress").func
    delivered_original = manager.get_term_cfg("delivered_impulse").func

    impact_tap, delivered_tap = smoke.install_reward_taps(env)
    contribution = manager.compute(dt=0.1)

    assert manager.get_term_cfg("impact_progress").func is impact_tap
    assert manager.get_term_cfg("delivered_impulse").func is delivered_tap
    assert impact_original.calls == 1
    assert delivered_original.calls == 1
    assert torch.isnan(impact_tap.last_raw[0])
    assert torch.isinf(delivered_tap.last_raw[1])
    torch.testing.assert_close(contribution, torch.tensor([0.2, 1.6]))
    torch.testing.assert_close(
        manager._step_reward, torch.tensor([[0.0, 2.0], [16.0, 0.0]])
    )

    env_ids = torch.tensor([0])
    manager.reset(env_ids)

    assert len(impact_original.reset_calls) == 1
    assert len(delivered_original.reset_calls) == 1
    torch.testing.assert_close(impact_original.reset_calls[0], env_ids)
    torch.testing.assert_close(delivered_original.reset_calls[0], env_ids)
    assert impact_tap.call_count.tolist() == [0, 1]
    assert delivered_tap.call_count.tolist() == [0, 1]


def test_finalize_uses_existing_invariant_logic_on_first_terminal_records(monkeypatch):
    """Replacing evaluator sentinels with a parallel formula must fail this test."""
    impact, delivered, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [1.0, 1.0], [1.0, 1.0]
    )
    recorder.capture_control_step(
        **_control_payload(reset_buf=torch.tensor([True, True]))
    )
    observed = {}

    def invariant_spy(records):
        observed.update(records)
        return 7, 9

    monkeypatch.setattr(smoke.eval_impulse, "_invariant_violations", invariant_spy)

    result = recorder.finalize()

    assert result["impossible_success_n"] == 7
    assert result["lambda_dead_n"] == 9
    assert len(observed["lam"]) == 2
    assert observed["succ"] == [True, False]
    assert observed["delivered"] == pytest.approx([0.61, 0.10])


def test_no_host_actions_in_substep_and_control_callbacks(monkeypatch):
    """Adding any host read, synchronization, tensor branch, or forward call must fail."""
    impact, delivered, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [1.0, 1.0], [1.0, 1.0]
    )
    substep = {
        "contact": torch.tensor([True, False]),
        "axial_force": torch.tensor([12.0, 3.0]),
        "pre_arm_qvel": torch.ones(2, 6),
        "post_arm_qvel": torch.full((2, 6), 2.0),
        "pre_depth": torch.tensor([0.01, 0.02]),
        "post_depth": torch.tensor([0.02, 0.03]),
    }
    control = _control_payload()

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("forbidden host action")

    with monkeypatch.context() as guarded:
        guarded.setattr(torch.Tensor, "item", forbidden)
        guarded.setattr(torch.Tensor, "cpu", forbidden)
        guarded.setattr(torch.Tensor, "numpy", forbidden)
        guarded.setattr(torch.Tensor, "__bool__", forbidden)
        guarded.setattr(torch.cuda, "synchronize", forbidden)
        guarded.setattr(smoke.warp, "synchronize", forbidden)
        guarded.setattr(smoke.warp, "synchronize_device", forbidden)

        recorder.capture_substep(**substep)
        recorder.capture_control_step(**control)
        with pytest.raises(AssertionError, match="forbidden host action"):
            substep["axial_force"][0].item()


def test_no_host_installed_callbacks_preserve_runtime_order_and_terminal_state(
    monkeypatch,
):
    """Reordering original callbacks or reading state after reset must fail."""
    impact, delivered, impact_tap, delivered_tap = _make_taps()
    _capture_rewards(
        impact, delivered, impact_tap, delivered_tap, [1.0, 1.0], [1.0, 1.0]
    )
    order = []
    robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel=torch.tensor([[1.0] * 6, [0.5] * 6]))
    )
    nail = SimpleNamespace(
        data=SimpleNamespace(joint_pos=torch.tensor([[0.010], [0.020]]))
    )
    contact = SimpleNamespace(
        data=SimpleNamespace(found=torch.tensor([[0], [0]]))
    )
    impulse_sensor = SimpleNamespace(
        data=SimpleNamespace(force=torch.zeros(2, 1, 3))
    )
    accumulator = SimpleNamespace(_episode_peak_perjoint=torch.zeros(2, 6))
    delivered_accumulator = SimpleNamespace(delivered=torch.zeros(2))
    tracker = SimpleNamespace(
        started=torch.zeros(2, dtype=torch.bool),
        finalized=torch.zeros(2, dtype=torch.bool),
        productive=torch.zeros(2, dtype=torch.bool),
        reason=torch.zeros(2, dtype=torch.long),
        delivered=torch.zeros(2),
        contact_quality_valid=torch.zeros(2, dtype=torch.bool),
        contact_quality_overflow=torch.zeros(2, dtype=torch.bool),
    )

    class FakeEntityCfg:
        def __init__(self, name, **kwargs):
            del name, kwargs
            self.joint_ids = None

        def resolve(self, scene):
            del scene
            self.joint_ids = list(range(6))

    monkeypatch.setattr(smoke, "SceneEntityCfg", FakeEntityCfg)

    def simulation_step():
        order.append("sim")
        robot.data.joint_vel[:] = 2.0
        nail.data.joint_pos[:] = torch.tensor([[0.031], [0.021]])
        contact.data.found[:] = torch.tensor([[1], [0]])
        impulse_sensor.data.force[:, :, 2] = torch.tensor([[-100.0], [-5.0]])

    def original_substep():
        order.append("substep")

    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        step_dt=0.02,
        scene={
            "robot": robot,
            "nail_block": nail,
            "hammer_nail_contact": contact,
            "hammer_nail_impulse": impulse_sensor,
        },
        sim=SimpleNamespace(
            step=simulation_step,
            forward=lambda: (_ for _ in ()).throw(
                AssertionError("sim.forward forbidden")
            ),
        ),
        reset_buf=torch.tensor([True, False]),
        reward_manager=SimpleNamespace(
            active_terms=["impact_progress", "delivered_impulse"],
            _step_reward=torch.tensor([[400.0, 100.0], [0.0, 0.0]]),
        ),
        _hammer_substep_impulse=accumulator,
        _hammer_substep_delivered=delivered_accumulator,
        _hammer_first_strike=tracker,
    )

    def original_compute():
        order.append("compute")
        accumulator._episode_peak_perjoint[:] = 0.2
        delivered_accumulator.delivered[:] = torch.tensor([0.7, 0.1])
        tracker.started[:] = torch.tensor([True, False])
        tracker.finalized[:] = torch.tensor([True, False])
        tracker.productive[:] = torch.tensor([True, False])
        tracker.reason[:] = torch.tensor([1, 0])
        tracker.delivered[:] = torch.tensor([0.6, 0.0])
        tracker.contact_quality_valid[:] = torch.tensor([True, False])
        tracker.contact_quality_overflow[:] = torch.tensor([False, False])

    env.metrics_manager = SimpleNamespace(
        compute_substep=original_substep,
        compute=original_compute,
    )
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    original_sim_step = env.sim.step
    original_metrics_substep = env.metrics_manager.compute_substep
    original_metrics_compute = env.metrics_manager.compute

    recorder.install(env)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("forbidden host action")

    with monkeypatch.context() as guarded:
        guarded.setattr(torch.Tensor, "item", forbidden)
        guarded.setattr(torch.Tensor, "cpu", forbidden)
        guarded.setattr(torch.Tensor, "numpy", forbidden)
        guarded.setattr(torch.Tensor, "__bool__", forbidden)
        guarded.setattr(torch.cuda, "synchronize", forbidden)
        guarded.setattr(smoke.warp, "synchronize", forbidden)
        guarded.setattr(smoke.warp, "synchronize_device", forbidden)
        env.sim.step()
        env.metrics_manager.compute_substep()
        env.metrics_manager.compute()

    assert order == ["sim", "substep", "compute"]
    assert recorder.terminal_seen.tolist() == [True, False]
    torch.testing.assert_close(recorder.terminal_pre_qvel_max[0], torch.tensor(1.0))
    torch.testing.assert_close(recorder.terminal_post_qvel_max[0], torch.tensor(2.0))
    torch.testing.assert_close(recorder.terminal_lambda[0], torch.tensor([0.2] * 6))
    torch.testing.assert_close(recorder.terminal_delivered[0], torch.tensor(0.7))
    torch.testing.assert_close(recorder.terminal_manager_impact[0], torch.tensor(8.0))
    torch.testing.assert_close(recorder.terminal_manager_delivered[0], torch.tensor(2.0))
    assert recorder.terminal_success[0]
    assert recorder.terminal_tracker_started[0]
    assert recorder.terminal_tracker_finalized[0]
    assert recorder.terminal_tracker_contact_quality_valid[0]
    assert not recorder.terminal_tracker_contact_quality_overflow[0]

    recorder.uninstall()

    assert env.sim.step is original_sim_step
    assert env.metrics_manager.compute_substep is original_metrics_substep
    assert env.metrics_manager.compute is original_metrics_compute


def _exception_test_env(monkeypatch, *, failing_callback):
    """Small complete environment for callback restoration behavior."""
    robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel=torch.zeros(2, 6))
    )
    nail = SimpleNamespace(
        data=SimpleNamespace(joint_pos=torch.zeros(2, 1))
    )
    contact = SimpleNamespace(
        data=SimpleNamespace(found=torch.zeros(2, 1, dtype=torch.long))
    )
    impulse_sensor = SimpleNamespace(
        data=SimpleNamespace(force=torch.zeros(2, 1, 3))
    )

    class FakeEntityCfg:
        def __init__(self, name, **kwargs):
            del name, kwargs
            self.joint_ids = None

        def resolve(self, scene):
            del scene
            self.joint_ids = list(range(6))

    monkeypatch.setattr(smoke, "SceneEntityCfg", FakeEntityCfg)

    def ok():
        return None

    def fail():
        raise RuntimeError(f"{failing_callback} injected")

    sim_step = fail if failing_callback == "sim" else ok
    substep = fail if failing_callback == "substep" else ok
    compute = fail if failing_callback == "compute" else ok
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        step_dt=0.02,
        scene={
            "robot": robot,
            "nail_block": nail,
            "hammer_nail_contact": contact,
            "hammer_nail_impulse": impulse_sensor,
        },
        sim=SimpleNamespace(step=sim_step),
        metrics_manager=SimpleNamespace(
            compute_substep=substep,
            compute=compute,
        ),
        reset_buf=torch.zeros(2, dtype=torch.bool),
        reward_manager=SimpleNamespace(
            active_terms=["impact_progress", "delivered_impulse"],
            _step_reward=torch.zeros(2, 2),
        ),
        _hammer_substep_impulse=SimpleNamespace(
            _episode_peak_perjoint=torch.zeros(2, 6)
        ),
        _hammer_substep_delivered=SimpleNamespace(delivered=torch.zeros(2)),
    )
    return env, sim_step, substep, compute


@pytest.mark.parametrize(
    ("failing_callback", "invoke"),
    (
        ("sim", lambda env: env.sim.step()),
        ("substep", lambda env: env.metrics_manager.compute_substep()),
        ("compute", lambda env: env.metrics_manager.compute()),
    ),
)
def test_installed_callback_exception_restores_every_original_method(
    monkeypatch, failing_callback, invoke
):
    """Any wrapped callback exception must leave no instrumentation installed."""
    _, _, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    env, sim_step, substep, compute = _exception_test_env(
        monkeypatch, failing_callback=failing_callback
    )
    recorder.install(env)

    with pytest.raises(RuntimeError, match=f"{failing_callback} injected"):
        invoke(env)

    assert env.sim.step is sim_step
    assert env.metrics_manager.compute_substep is substep
    assert env.metrics_manager.compute is compute


def test_installed_context_restores_callbacks_when_rollout_body_raises(monkeypatch):
    """An exception outside callbacks must still restore through the context boundary."""
    _, _, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    env, sim_step, substep, compute = _exception_test_env(
        monkeypatch, failing_callback="none"
    )

    with pytest.raises(RuntimeError, match="rollout injected"):
        with recorder.installed(env):
            raise RuntimeError("rollout injected")

    assert env.sim.step is sim_step
    assert env.metrics_manager.compute_substep is substep
    assert env.metrics_manager.compute is compute


def test_partial_installation_failure_restores_methods_already_patched(monkeypatch):
    """A late attribute-assignment failure must roll back earlier wrappers."""
    _, _, impact_tap, delivered_tap = _make_taps()
    recorder = smoke.DeviceSmokeRecorder(impact_tap, delivered_tap)
    env, sim_step, substep, compute = _exception_test_env(
        monkeypatch, failing_callback="none"
    )

    class FailComputeAssignment:
        def __init__(self):
            self.compute_substep = substep
            self._compute = compute

        @property
        def compute(self):
            return self._compute

        @compute.setter
        def compute(self, value):
            if value is not compute:
                raise RuntimeError("compute assignment injected")
            self._compute = value

    env.metrics_manager = FailComputeAssignment()

    with pytest.raises(RuntimeError, match="compute assignment injected"):
        recorder.install(env)

    assert env.sim.step is sim_step
    assert env.metrics_manager.compute_substep is substep
    assert env.metrics_manager.compute is compute


def _valid_cli_argv(tmp_path, *, task=ARM_TASKS["C"], device="cpu", num_envs=1):
    return [
        "--task",
        task,
        "--device",
        device,
        "--num-envs",
        str(num_envs),
        "--expected-code-revision",
        "a" * 40,
        "--expected-asset-revision",
        "b" * 40,
        "--asset-repo",
        str(tmp_path / "assets"),
        "--out",
        str(tmp_path / "external" / "result.json"),
    ]


def test_cli_parses_the_exact_required_command_shape(tmp_path):
    """Removing or renaming any frozen deployment argument must fail."""
    args = smoke.parse_args(_valid_cli_argv(tmp_path))

    assert args.task == ARM_TASKS["C"]
    assert args.device == "cpu"
    assert args.num_envs == 1
    assert args.expected_code_revision == "a" * 40
    assert args.expected_asset_revision == "b" * 40
    assert args.asset_repo == tmp_path / "assets"
    assert args.out == tmp_path / "external" / "result.json"


def test_cli_module_has_an_executable_main_guard():
    """Running the script directly must propagate main's qualification exit code."""
    source = Path(smoke.__file__).read_text()

    assert 'if __name__ == "__main__":' in source
    assert "raise SystemExit(main())" in source


@pytest.mark.parametrize(
    "argv",
    (
        ["--task", "unknown"],
        ["--device", "cuda"],
        ["--num-envs", "0"],
        ["--expected-code-revision", "a" * 39],
        ["--expected-asset-revision", "B" * 40],
    ),
)
def test_cli_rejects_unknown_task_device_count_or_revision(tmp_path, argv):
    """Loosening task/device/count/revision validation must fail this test."""
    valid = _valid_cli_argv(tmp_path)
    option = argv[0]
    index = valid.index(option)
    valid[index : index + 2] = argv

    with pytest.raises(SystemExit):
        smoke.parse_args(valid)


def test_cli_requires_256_environments_for_a_cuda_request(tmp_path):
    """A smaller CUDA invocation is CPU-style evidence, never qualification."""
    with pytest.raises(SystemExit):
        smoke.parse_args(
            _valid_cli_argv(tmp_path, device="cuda:0", num_envs=255)
        )


def test_cli_rejects_missing_expected_revision(tmp_path):
    """A smoke cannot run before both deployment revisions are supplied."""
    argv = _valid_cli_argv(tmp_path)
    index = argv.index("--expected-code-revision")
    del argv[index : index + 2]

    with pytest.raises(SystemExit):
        smoke.parse_args(argv)


def test_cli_rejects_output_collision_before_running_physics(
    tmp_path, monkeypatch, capsys
):
    """An occupied evidence path must fail before constructing an environment."""
    out = tmp_path / "external" / "result.json"
    out.parent.mkdir()
    out.write_text("{}")
    monkeypatch.setattr(
        smoke,
        "run_smoke",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("physics must not run after an output collision")
        ),
        raising=False,
    )

    exit_code = smoke.main(_valid_cli_argv(tmp_path))

    diagnostic = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert diagnostic["failed_predicates"] == ["output_validation_exception"]
    assert diagnostic["error_type"] == "FileExistsError"
    assert out.read_text() == "{}"


def test_cli_repo_local_output_emits_validation_json_before_physics(
    tmp_path, monkeypatch, capsys
):
    """A repository-local destination must fail as one compact diagnostic."""
    fake_script = tmp_path / "code" / "scripts" / "smoke.py"
    out = tmp_path / "code" / "result.json"
    argv = _valid_cli_argv(tmp_path)
    argv[argv.index("--out") + 1] = str(out)
    monkeypatch.setattr(smoke, "__file__", str(fake_script))
    monkeypatch.setattr(
        smoke,
        "run_smoke",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("physics must not run for repo-local output")
        ),
    )

    exit_code = smoke.main(argv)

    lines = capsys.readouterr().out.splitlines()
    assert exit_code != 0
    assert len(lines) == 1
    diagnostic = json.loads(lines[0])
    assert diagnostic["failed_predicates"] == ["output_validation_exception"]
    assert diagnostic["error_type"] == "ValueError"
    assert not out.exists()


def test_cli_invalid_output_emits_validation_json_before_physics(
    tmp_path, monkeypatch, capsys
):
    """Any ordinary output validation error must use the machine boundary."""
    monkeypatch.setattr(
        smoke,
        "validate_output_path",
        lambda *args: (_ for _ in ()).throw(
            ValueError("invalid output injected")
        ),
    )
    monkeypatch.setattr(
        smoke,
        "run_smoke",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("physics must not run for invalid output")
        ),
    )

    exit_code = smoke.main(_valid_cli_argv(tmp_path))

    diagnostic = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert diagnostic["failed_predicates"] == ["output_validation_exception"]
    assert "invalid output injected" in diagnostic["error"]


def test_cli_binds_clean_pre_and_post_provenance_then_writes_once(
    tmp_path, monkeypatch, capsys
):
    """Banking an unbound runtime record or skipping post-run Git checks must fail."""
    code_repo = init_clean_repo(tmp_path / "code")
    asset_repo = init_clean_repo(tmp_path / "assets")
    fake_script = code_repo / "scripts" / "smoke_first_strike_instrumentation.py"
    fake_script.parent.mkdir()
    fake_script.write_text("# location sentinel\n")
    _git(code_repo, "add", "scripts/smoke_first_strike_instrumentation.py")
    _git(code_repo, "commit", "-qm", "add script")
    code_head = git_head(code_repo)
    asset_head = git_head(asset_repo)
    out = tmp_path / "external" / "result.json"
    out.parent.mkdir()
    runtime = valid_record(arm="C", num_envs=1, device_type="cpu")
    runtime["visible_cuda_device_count"] = 0
    monkeypatch.setattr(smoke, "__file__", str(fake_script))
    monkeypatch.setattr(smoke, "run_smoke", lambda **kwargs: deepcopy(runtime))

    exit_code = smoke.main(
        [
            "--task",
            ARM_TASKS["C"],
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--expected-code-revision",
            code_head,
            "--expected-asset-revision",
            asset_head,
            "--asset-repo",
            str(asset_repo),
            "--out",
            str(out),
        ]
    )

    payload = json.loads(out.read_text())
    assert exit_code == 0
    assert payload["integration_pass"] is True
    assert payload["cuda_qualification_pass"] is False
    assert payload["provenance"]["code"] == {
        "canonical_path": str(code_repo.resolve()),
        "expected_revision": code_head,
        "pre_revision": code_head,
        "post_revision": code_head,
        "pre_dirty": False,
        "post_dirty": False,
    }
    assert payload["provenance"]["assets"] == {
        "canonical_path": str(asset_repo.resolve()),
        "expected_revision": asset_head,
        "pre_revision": asset_head,
        "post_revision": asset_head,
        "pre_dirty": False,
        "post_dirty": False,
    }
    summary = json.loads(capsys.readouterr().out)
    assert summary["integration_pass"] is True
    assert summary["cuda_qualification_pass"] is False


def test_cli_rejects_noncanonical_asset_repository_before_physics(
    tmp_path, monkeypatch, capsys
):
    """A Git subdirectory cannot stand in for the canonical asset repository."""
    code_repo = init_clean_repo(tmp_path / "code")
    asset_repo = init_clean_repo(tmp_path / "assets")
    nested_asset = asset_repo / "nested"
    nested_asset.mkdir()
    fake_script = code_repo / "scripts" / "smoke_first_strike_instrumentation.py"
    fake_script.parent.mkdir()
    fake_script.write_text("# sentinel\n")
    _git(code_repo, "add", "scripts/smoke_first_strike_instrumentation.py")
    _git(code_repo, "commit", "-qm", "add script")
    monkeypatch.setattr(smoke, "__file__", str(fake_script))
    monkeypatch.setattr(
        smoke,
        "run_smoke",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("physics must not run for noncanonical provenance")
        ),
    )

    exit_code = smoke.main(
        [
            "--task",
            ARM_TASKS["C"],
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--expected-code-revision",
            git_head(code_repo),
            "--expected-asset-revision",
            git_head(asset_repo),
            "--asset-repo",
            str(nested_asset),
            "--out",
            str(tmp_path / "external" / "result.json"),
        ]
    )

    diagnostic = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert diagnostic["failed_predicates"] == ["pre_provenance_exception"]
    assert "repository root" in diagnostic["error"]


def _mock_clean_provenance(path, expected_revision):
    return {
        "canonical_path": str(Path(path).resolve()),
        "expected_revision": expected_revision,
        "revision": expected_revision,
        "dirty": False,
    }


def _install_successful_cli_runtime(tmp_path, monkeypatch):
    fake_script = tmp_path / "code" / "scripts" / "smoke.py"
    runtime = valid_record(arm="C", num_envs=1, device_type="cpu")
    runtime["visible_cuda_device_count"] = 0
    monkeypatch.setattr(smoke, "__file__", str(fake_script))
    monkeypatch.setattr(smoke, "read_repo_provenance", _mock_clean_provenance)
    monkeypatch.setattr(smoke, "run_smoke", lambda **kwargs: deepcopy(runtime))
    out = tmp_path / "external" / "result.json"
    out.parent.mkdir()
    return out


@pytest.mark.parametrize("failure", ("collision", "link", "directory_fsync"))
def test_cli_publish_failure_emits_one_failure_json_and_no_false_success(
    tmp_path, monkeypatch, capsys, failure
):
    """Publication failure must never be preceded by a success announcement."""
    out = _install_successful_cli_runtime(tmp_path, monkeypatch)

    def fail_publish(destination, payload):
        if failure == "collision":
            raise FileExistsError("concurrent creator injected")
        if failure == "link":
            raise OSError("hard-link publish injected")
        destination.write_text(json.dumps(payload, sort_keys=True))
        raise OSError("directory fsync injected")

    monkeypatch.setattr(smoke, "write_json_atomic", fail_publish)

    exit_code = smoke.main(_valid_cli_argv(tmp_path))

    lines = capsys.readouterr().out.splitlines()
    assert exit_code != 0
    assert len(lines) == 1
    diagnostic = json.loads(lines[0])
    assert diagnostic["integration_pass"] is False
    assert diagnostic["cuda_qualification_pass"] is False
    assert diagnostic["failed_predicates"] == ["output_publish_exception"]
    assert diagnostic["error_type"] in ("FileExistsError", "OSError")
    if failure == "directory_fsync":
        payload = json.loads(out.read_text())
        assert payload["integration_pass"] is True
    else:
        assert not out.exists()


def test_cli_prints_success_only_after_successful_publication(
    tmp_path, monkeypatch, capsys
):
    """A success line before durable publication would make this observation nonempty."""
    out = _install_successful_cli_runtime(tmp_path, monkeypatch)
    real_writer = smoke.write_json_atomic
    stdout_before_publish = []

    def observed_publish(destination, payload):
        stdout_before_publish.append(capsys.readouterr().out)
        real_writer(destination, payload)

    monkeypatch.setattr(smoke, "write_json_atomic", observed_publish)

    exit_code = smoke.main(_valid_cli_argv(tmp_path))

    assert exit_code == 0
    assert stdout_before_publish == [""]
    summary = json.loads(capsys.readouterr().out)
    assert summary["integration_pass"] is True
    assert out.exists()


@pytest.mark.parametrize("exception", (KeyboardInterrupt(), SystemExit(9)))
def test_cli_publish_boundary_does_not_swallow_process_control(
    tmp_path, monkeypatch, capsys, exception
):
    """Publication catches Exception only, preserving interruption semantics."""
    out = _install_successful_cli_runtime(tmp_path, monkeypatch)

    def interrupt(*args, **kwargs):
        del args, kwargs
        raise exception

    monkeypatch.setattr(smoke, "write_json_atomic", interrupt)

    with pytest.raises(type(exception)):
        smoke.main(_valid_cli_argv(tmp_path))

    assert capsys.readouterr().out == ""
    assert not out.exists()


def test_cli_runtime_failure_prints_json_returns_nonzero_and_banks_nothing(
    tmp_path, monkeypatch, capsys
):
    """A contract/runtime exception must be diagnostic, never silent evidence."""
    fake_script = tmp_path / "code" / "scripts" / "smoke.py"
    out = tmp_path / "external" / "result.json"
    out.parent.mkdir()
    monkeypatch.setattr(smoke, "__file__", str(fake_script))
    monkeypatch.setattr(smoke, "read_repo_provenance", _mock_clean_provenance)
    monkeypatch.setattr(
        smoke,
        "run_smoke",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("live contract drift injected")
        ),
    )

    exit_code = smoke.main(_valid_cli_argv(tmp_path))

    diagnostic = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert diagnostic["integration_pass"] is False
    assert diagnostic["cuda_qualification_pass"] is False
    assert diagnostic["failed_predicates"] == ["runtime_exception"]
    assert diagnostic["error_type"] == "ValueError"
    assert "live contract drift injected" in diagnostic["error"]
    assert not out.exists()


def test_cli_post_provenance_failure_prints_json_and_banks_nothing(
    tmp_path, monkeypatch, capsys
):
    """A repository change during physics must invalidate and suppress the artifact."""
    fake_script = tmp_path / "code" / "scripts" / "smoke.py"
    out = tmp_path / "external" / "result.json"
    out.parent.mkdir()
    calls = 0

    def provenance(path, expected_revision):
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise ValueError("post-run repository drift injected")
        return _mock_clean_provenance(path, expected_revision)

    runtime = valid_record(arm="C", num_envs=1, device_type="cpu")
    runtime["visible_cuda_device_count"] = 0
    monkeypatch.setattr(smoke, "__file__", str(fake_script))
    monkeypatch.setattr(smoke, "read_repo_provenance", provenance)
    monkeypatch.setattr(smoke, "run_smoke", lambda **kwargs: deepcopy(runtime))

    exit_code = smoke.main(_valid_cli_argv(tmp_path))

    diagnostic = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert diagnostic["integration_pass"] is False
    assert diagnostic["cuda_qualification_pass"] is False
    assert diagnostic["failed_predicates"] == ["post_provenance_exception"]
    assert diagnostic["error_type"] == "ValueError"
    assert "post-run repository drift injected" in diagnostic["error"]
    assert not out.exists()


@pytest.mark.parametrize("exception", (KeyboardInterrupt(), SystemExit(7)))
def test_cli_does_not_swallow_process_control_exceptions(
    tmp_path, monkeypatch, exception
):
    """The diagnostic boundary must catch ordinary failures, not process control."""
    fake_script = tmp_path / "code" / "scripts" / "smoke.py"
    out = tmp_path / "external" / "result.json"
    out.parent.mkdir()
    monkeypatch.setattr(smoke, "__file__", str(fake_script))
    monkeypatch.setattr(smoke, "read_repo_provenance", _mock_clean_provenance)

    def interrupt(**kwargs):
        del kwargs
        raise exception

    monkeypatch.setattr(smoke, "run_smoke", interrupt)

    with pytest.raises(type(exception)):
        smoke.main(_valid_cli_argv(tmp_path))

    assert not out.exists()


def test_fixed_rollout_forbids_host_conversion_sync_tensor_bool_and_forward(
    monkeypatch,
):
    """The runner loop must remain device-only and fixed-length."""
    actions = []

    class FakeReference:
        def __init__(self):
            self.calls = []

        def playback_target(self, step):
            self.calls.append(step)
            return torch.full((2, 3), float(step))

    class FakeWrappedEnv:
        def __init__(self):
            self.env = SimpleNamespace(
                sim=SimpleNamespace(
                    forward=lambda: (_ for _ in ()).throw(
                        AssertionError("sim.forward forbidden")
                    )
                )
            )

        def step(self, action):
            actions.append(action.clone())

    reference = FakeReference()
    wrapped = FakeWrappedEnv()
    head = lambda: torch.zeros(2, 3)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("forbidden host action")

    with monkeypatch.context() as guarded:
        guarded.setattr(torch.Tensor, "item", forbidden)
        guarded.setattr(torch.Tensor, "cpu", forbidden)
        guarded.setattr(torch.Tensor, "numpy", forbidden)
        guarded.setattr(torch.Tensor, "__bool__", forbidden)
        guarded.setattr(torch.cuda, "synchronize", forbidden)
        guarded.setattr(smoke.warp, "synchronize", forbidden)
        guarded.setattr(smoke.warp, "synchronize_device", forbidden)
        smoke._run_fixed_rollout(
            wrapped,
            reference,
            head_position=head,
            playback_length=3,
        )

    assert reference.calls == [1, 2, 3, 3, 3, 3, 3, 3, 3]
    assert len(actions) == 9
    torch.testing.assert_close(actions[0], torch.full((2, 3), 1.0).clamp(-1, 1))


def test_runner_relies_on_wrapper_constructor_for_exactly_one_reset(
    monkeypatch,
):
    """An explicit reset after wrapper construction must fail this integration."""
    real_wrapper = smoke.RslRlVecEnvWrapper
    raw_reset_calls = []

    class OneResetWrapper(real_wrapper):
        def __init__(self, env, clip_actions=None):
            original_reset = env.reset

            def counted_reset(*args, **kwargs):
                raw_reset_calls.append("reset")
                return original_reset(*args, **kwargs)

            env.reset = counted_reset
            super().__init__(env, clip_actions=clip_actions)

        def reset(self):
            raise AssertionError("second environment reset")

    monkeypatch.setattr(smoke, "RslRlVecEnvWrapper", OneResetWrapper)

    result = smoke.run_smoke(
        task=ARM_TASKS["C"], device="cpu", num_envs=1
    )

    assert result["integration_pass"] is True
    assert raw_reset_calls == ["reset"]


@pytest.mark.parametrize("task", tuple(ARM_TASKS.values()))
def test_cpu_integration_runs_the_fixed_reference_for_each_task(task):
    """Breaking any registered arm's live instrumentation must fail locally."""
    result = smoke.run_smoke(task=task, device="cpu", num_envs=1)

    assert result["integration_pass"] is True, result["failed_predicates"]
    assert result["cuda_qualification_pass"] is False
    assert result["predicate_counts"]["normal_success"]["passed"] == 1
    assert result["predicate_counts"]["lambda_live"]["passed"] == 1
    assert result["predicate_counts"]["delivered_live"]["passed"] == 1
    assert result["max_pre_arm_qvel_rad_s"] <= 3.1415
    assert result["max_post_arm_qvel_rad_s"] <= 3.1415

    if task == ARM_TASKS["C"]:
        assert result["predicate_counts"]["manager_impact_positive"]["passed"] == 1
        assert result["predicate_counts"]["manager_delivered_positive"]["passed"] == 1
    else:
        assert result["predicate_counts"]["productive_success"]["passed"] == 1
        assert result["predicate_counts"]["terminal_reason_success"]["passed"] == 1
        assert (
            result["predicate_counts"]["impact_exactly_one_terminal_pulse"][
                "passed"
            ]
            == 1
        )
        assert (
            result["predicate_counts"]["delivered_exactly_one_terminal_pulse"][
                "passed"
            ]
            == 1
        )


@pytest.mark.parametrize("arm", ("F0", "D0", "FQ-min"))
def test_cpu_integration_pays_out_each_quality_arm_treatment_correctly(arm):
    """Prove each fq4x8 arm's reward pays out exactly as its treatment specifies.

    F8 is not repeated here: its registered task is byte-identical to
    ARM_TASKS["F"], already exercised above by
    test_cpu_integration_runs_the_fixed_reference_for_each_task with the same
    raw-speed/raw-delivered assertions this test applies to F0/D0/FQ-min.
    """
    contract = smoke.ARM_CONTRACTS[arm]

    result = smoke.run_smoke(task=contract.task, device="cpu", num_envs=1)

    assert result["integration_pass"] is True, result["failed_predicates"]
    assert result["cuda_qualification_pass"] is False
    assert result["predicate_counts"]["normal_success"]["passed"] == 1
    assert result["predicate_counts"]["productive_success"]["passed"] == 1
    assert result["predicate_counts"]["terminal_reason_success"]["passed"] == 1
    # The zeroed side's reward-manager func is never invoked (mjlab skips
    # weight==0 terms), so its pulse-timing predicate reads 0/1 -- proving
    # both that the physical strike is unaffected AND that evaluate_gate
    # correctly excludes this predicate from qualification for this arm.
    assert result["predicate_counts"]["impact_exactly_one_terminal_pulse"][
        "passed"
    ] == (0 if contract.speed_zero else 1)
    assert result["predicate_counts"]["delivered_exactly_one_terminal_pulse"][
        "passed"
    ] == (0 if contract.delivered_zero else 1)

    if arm == "F0":
        assert result["predicate_counts"]["manager_impact_zero"]["passed"] == 1
        assert result["predicate_counts"]["manager_delivered_positive"]["passed"] == 1
        assert result["predicate_counts"]["raw_delivered_reader_finite"]["passed"] == 1
    elif arm == "D0":
        assert result["predicate_counts"]["manager_impact_positive"]["passed"] == 1
        assert result["predicate_counts"]["manager_delivered_zero"]["passed"] == 1
        assert result["predicate_counts"]["raw_impact_reader_finite"]["passed"] == 1
    else:
        assert result["predicate_counts"]["manager_impact_positive"]["passed"] == 1
        assert result["predicate_counts"]["manager_delivered_zero"]["passed"] == 1
        assert result["predicate_counts"]["raw_impact_reader_finite"]["passed"] == 1
        assert result["predicate_counts"]["quality_snapshot_valid"]["passed"] == 1
        assert result["predicate_counts"]["quality_no_overflow"]["passed"] == 1
