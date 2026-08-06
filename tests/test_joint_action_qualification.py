"""Behavioral contract for the causal joint-position qualification artifact."""

from __future__ import annotations

import copy
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.joint_position.qualify_joint_action import (
    derive_scale_by_joint,
    normalized_action_for_target,
    qualification_passes,
    reduce_first_target_per_interval,
)
from src.tasks.hammer.config.z1.joint_position_contract import (
    load_joint_position_contract,
)


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
PHYSICAL_CLIPS = [
    [-2.61799, 2.61799],
    [0.0, 2.96706],
    [-2.87979, 0.0],
    [-1.51844, 1.51844],
    [-1.34390, 1.34390],
    [-2.79253, 2.79253],
]
TAPE = [
    [0.0, 0.5, -0.5, 0.0, 0.1, -0.1],
    [0.1, 0.6, -0.6, 0.1, 0.2, -0.2],
]
SOURCE_TASK_CONFIG_PROJECTION = {
    "action": {"name": "hammer_ik", "dimension": 3},
    "observations": {"policy": {"dimension": 44}},
    "rewards": {"nail_depth_delta": {"weight": 600.0}},
    "metrics": {"first_strike": {"enabled": True}},
    "events": {"reset": {"mode": "fixed"}},
    "actuators": {"arm": {"joint_names": JOINT_NAMES}},
    "timing": {"physics_dt_s": 0.002, "control_decimation": 10},
    "reset": {"joint_pos_offset": [0.0, 0.0]},
}


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


TAPE_SHA256 = _canonical_sha256(TAPE)


def _canonical_payload_sha256(payload: dict[str, object]) -> str:
    """Independent implementation of the documented non-self-referential digest."""
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _passing_rows() -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "passed": True,
            "source_target_tape_sha256": TAPE_SHA256,
        }
        for seed in range(1000, 1016)
    ]


def _passing_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "action_semantics": "absolute_default_offset_joint_position",
        "source_task_id": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4",
        "source_code_revision": "a" * 40,
        "source_asset_revision": "b" * 40,
        "source_task_config_projection": copy.deepcopy(SOURCE_TASK_CONFIG_PROJECTION),
        "source_task_config_sha256": _canonical_sha256(SOURCE_TASK_CONFIG_PROJECTION),
        "joint_names": copy.deepcopy(JOINT_NAMES),
        "actuator_names": copy.deepcopy(JOINT_NAMES),
        "default_joint_pos_rad": [0.0, 0.5, -0.5, 0.0, 0.1, -0.1],
        "physical_clip_rad": copy.deepcopy(PHYSICAL_CLIPS),
        "scale_rad": [0.165, 0.165, 0.165, 0.165, 0.165, 0.165],
        "physics_dt_s": 0.002,
        "control_decimation": 10,
        "post_reference_hold_control_steps": 10,
        "seeds": list(range(1000, 1016)),
        "source_target_tape_rad": copy.deepcopy(TAPE),
        "source_target_tape_sha256": TAPE_SHA256,
        "per_seed_replay_rows": _passing_rows(),
        "decision": "PASS",
    }
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    return payload


def _add_source_task_config_projection(payload: dict[str, object]) -> None:
    projection = copy.deepcopy(SOURCE_TASK_CONFIG_PROJECTION)
    payload["source_task_config_projection"] = projection
    payload["source_task_config_sha256"] = _canonical_sha256(projection)
    payload["payload_sha256"] = _canonical_payload_sha256(payload)


def _refresh_tape_binding(payload: dict[str, object]) -> None:
    tape_sha256 = _canonical_sha256(payload["source_target_tape_rad"])
    payload["source_target_tape_sha256"] = tape_sha256
    for row in payload["per_seed_replay_rows"]:
        row["source_target_tape_sha256"] = tape_sha256
    payload["payload_sha256"] = _canonical_payload_sha256(payload)


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")


def test_reduction_uses_first_target_not_interval_future() -> None:
    """Selecting an interval's final DiffIK target would make this fail."""
    targets = np.arange(20 * 6, dtype=np.float64).reshape(20, 6)

    reduced = reduce_first_target_per_interval(targets, decimation=10)

    np.testing.assert_array_equal(reduced, targets[[0, 10]])


def test_scale_has_exploration_floor_and_margin() -> None:
    """Dropping either the 0.05-rad floor or 1.10 margin would make this fail."""
    tape = np.array([[0.2, -0.1], [0.4, 0.3]])
    default = np.array([0.1, 0.0])
    expected = 1.10 * (np.array([0.3, 0.3]) + 0.05)

    np.testing.assert_allclose(derive_scale_by_joint(tape, default), expected)


def test_normalized_tape_round_trips_without_saturation() -> None:
    """Using current state or a mismatched affine sign would make this fail."""
    tape = np.array([[0.2, -0.1], [0.4, 0.3]])
    default = np.array([0.1, 0.0])
    scale = np.array([0.4, 0.4])

    action = normalized_action_for_target(tape, default, scale)

    assert np.max(np.abs(action)) < 1.0
    np.testing.assert_allclose(default + scale * action, tape)


def test_qualification_requires_exact_passing_seed_sequence() -> None:
    """Accepting a missing seed or a failed replay row would make this fail."""
    rows = _passing_rows()

    assert qualification_passes(rows) is True
    assert qualification_passes(rows[:-1]) is False
    rows[7]["passed"] = False
    assert qualification_passes(rows) is False


def test_loader_accepts_complete_canonical_contract(tmp_path: Path) -> None:
    """Rejecting a complete, internally consistent artifact would make this fail."""
    artifact = tmp_path / "joint_position.json"
    _write_payload(artifact, _passing_payload())

    contract = load_joint_position_contract(artifact)

    assert contract.joint_names == tuple(JOINT_NAMES)
    assert contract.actuator_names == tuple(JOINT_NAMES)
    np.testing.assert_allclose(contract.default_joint_pos_rad, [0.0, 0.5, -0.5, 0.0, 0.1, -0.1])
    assert contract.payload_sha256 == _passing_payload()["payload_sha256"]


def test_loader_returns_read_only_scientific_arrays(tmp_path: Path) -> None:
    """Mutating validated evidence after its digest check would make this fail."""
    artifact = tmp_path / "joint_position.json"
    _write_payload(artifact, _passing_payload())

    contract = load_joint_position_contract(artifact)

    for array in (
        contract.default_joint_pos_rad,
        contract.physical_clip_rad,
        contract.scale_rad,
        contract.source_target_tape_rad,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 123.0


@pytest.mark.parametrize(
    "field",
    (
        "default_joint_pos_rad",
        "physical_clip_rad",
        "scale_rad",
        "source_target_tape_rad",
    ),
)
def test_loader_scientific_arrays_cannot_be_made_writeable(
    tmp_path: Path, field: str
) -> None:
    """Backing validated arrays with owned mutable memory would make this fail."""
    artifact = tmp_path / "joint_position.json"
    _write_payload(artifact, _passing_payload())
    array = getattr(load_joint_position_contract(artifact), field)

    try:
        array.setflags(write=True)
    except ValueError:
        assert array.flags.writeable is False
    else:
        before = float(array.flat[0])
        array.flat[0] = before + 1.0
        assert float(array.flat[0]) != before
        pytest.fail(f"{field} accepted setflags(write=True) and mutation")


def test_loader_accepts_projection_bound_config_identity(tmp_path: Path) -> None:
    """Ignoring the explicit source-config projection would make this fail."""
    payload = _passing_payload()
    _add_source_task_config_projection(payload)
    artifact = tmp_path / "projected_config.json"
    _write_payload(artifact, payload)

    contract = load_joint_position_contract(artifact)

    assert contract.source_task_config_projection["timing"]["control_decimation"] == 10
    assert contract.source_task_config_sha256 == _canonical_sha256(
        SOURCE_TASK_CONFIG_PROJECTION
    )
    with pytest.raises(TypeError):
        contract.source_task_config_projection["timing"]["control_decimation"] = 11


def test_loader_rejects_arbitrary_rehashed_task_config_digest(tmp_path: Path) -> None:
    """Trusting a well-formed but projection-unbound config digest would make this fail."""
    payload = _passing_payload()
    _add_source_task_config_projection(payload)
    payload["source_task_config_sha256"] = "d" * 64
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "unbound_config_digest.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="does not match source_task_config_projection"):
        load_joint_position_contract(artifact)


@pytest.mark.parametrize(
    ("field", "index"),
    (
        ("default_joint_pos_rad", 0),
        ("physical_clip_rad", (0, 0)),
        ("scale_rad", 0),
        ("source_target_tape_rad", (0, 0)),
    ),
)
def test_loader_rejects_json_booleans_in_numeric_arrays(
    tmp_path: Path, field: str, index: int | tuple[int, int]
) -> None:
    """Letting NumPy coerce JSON booleans into physical numbers would make this fail."""
    payload = _passing_payload()
    if isinstance(index, tuple):
        payload[field][index[0]][index[1]] = False
    else:
        payload[field][index] = False
    _refresh_tape_binding(payload)
    artifact = tmp_path / f"boolean_{field}.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="must not contain JSON booleans"):
        load_joint_position_contract(artifact)


def test_loader_returns_deeply_immutable_replay_rows(tmp_path: Path) -> None:
    """Mutating validated replay evidence through nested mappings would make this fail."""
    payload = _passing_payload()
    payload["per_seed_replay_rows"][0]["metrics"] = {"waypoint_hits": [1, 2, 3]}
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "nested_rows.json"
    _write_payload(artifact, payload)

    contract = load_joint_position_contract(artifact)
    row = contract.per_seed_replay_rows[0]

    assert isinstance(row, Mapping)
    assert row.get("seed") == 1000
    assert row["metrics"]["waypoint_hits"] == (1, 2, 3)
    with pytest.raises(TypeError):
        row["passed"] = False
    with pytest.raises(TypeError):
        row["metrics"]["waypoint_hits"] = ()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.update(default_joint_pos_rad=[0.0] * 5),
        lambda payload: payload.update(joint_names=JOINT_NAMES[:-1]),
        lambda payload: payload.update(joint_names=list(reversed(JOINT_NAMES))),
    ),
    ids=("wrong-width", "missing-joint", "permuted-joints"),
)
def test_loader_rejects_wrong_joint_contract_width_or_order(
    tmp_path: Path, mutate
) -> None:
    """Weakening the ordered six-joint identity check would make this fail."""
    payload = _passing_payload()
    mutate(payload)
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "invalid.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_missing_seed(tmp_path: Path) -> None:
    """Allowing incomplete deterministic-repeatability evidence would make this fail."""
    payload = _passing_payload()
    payload["per_seed_replay_rows"] = _passing_rows()[:-1]
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "missing_seed.json"
    _write_payload(artifact, payload)

    assert qualification_passes(payload["per_seed_replay_rows"]) is False
    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_unknown_schema_field(tmp_path: Path) -> None:
    """Silently accepting an uninterpreted artifact field would make this fail."""
    payload = _passing_payload()
    payload["scale_derivation"] = "hand_tuned"
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "unknown_field.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="unexpected fields"):
        load_joint_position_contract(artifact)


def test_loader_rejects_boolean_schema_version(tmp_path: Path) -> None:
    """Treating JSON true as integer schema version one would make this fail."""
    payload = _passing_payload()
    payload["schema_version"] = True
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "boolean_schema.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="schema_version"):
        load_joint_position_contract(artifact)


def test_loader_rejects_nonhex_task_config_digest(tmp_path: Path) -> None:
    """Accepting a signed integer-shaped string as a SHA-256 digest would make this fail."""
    payload = _passing_payload()
    payload["source_task_config_sha256"] = "-" + "c" * 63
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "nonhex_digest.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_joint_position_contract(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_code_revision", "main"),
        ("source_asset_revision", "b" * 39),
    ),
)
def test_loader_rejects_unpinned_source_revision(
    tmp_path: Path, field: str, value: str
) -> None:
    """Accepting a symbolic or truncated revision would make this fail."""
    payload = _passing_payload()
    payload[field] = value
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "unpinned_revision.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="40-character lowercase Git commit"):
        load_joint_position_contract(artifact)


def test_loader_rejects_wrong_source_task_identity(tmp_path: Path) -> None:
    """Loading evidence captured from a different task would make this fail."""
    payload = _passing_payload()
    payload["source_task_id"] = "Unitree-Z1-Hammer-Other-Task"
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "wrong_source_task.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="source_task_id"):
        load_joint_position_contract(artifact)


def test_loader_rejects_nonfinite_json_values(tmp_path: Path) -> None:
    """Permitting JSON NaN in a scientific artifact would make this fail."""
    artifact = tmp_path / "nonfinite.json"
    artifact.write_text('{"scale_rad":[NaN]}', encoding="utf-8")

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_nonphysical_clips(tmp_path: Path) -> None:
    """Accepting an inverted physical interval would make this fail."""
    payload = _passing_payload()
    payload["physical_clip_rad"] = copy.deepcopy(PHYSICAL_CLIPS)
    payload["physical_clip_rad"][2] = [0.0, -2.87979]
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "inverted_clip.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_scale_that_violates_frozen_derivation(tmp_path: Path) -> None:
    """Accepting hand-tuned scales instead of the frozen formula would make this fail."""
    payload = _passing_payload()
    payload["scale_rad"] = [0.2] * 6
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "hand_tuned_scale.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="frozen derivation"):
        load_joint_position_contract(artifact)


def test_loader_rejects_inconsistent_source_tape_hashes(tmp_path: Path) -> None:
    """Not binding every replay row to the recorded causal tape would make this fail."""
    payload = _passing_payload()
    payload["per_seed_replay_rows"] = _passing_rows()
    payload["per_seed_replay_rows"][3]["source_target_tape_sha256"] = "d" * 64
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "drifted_tape.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_nonpassing_decision(tmp_path: Path) -> None:
    """Treating a FAIL artifact as banked evidence would make this fail."""
    payload = _passing_payload()
    payload["decision"] = "FAIL"
    payload["payload_sha256"] = _canonical_payload_sha256(payload)
    artifact = tmp_path / "failed.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)


def test_loader_rejects_mismatched_canonical_payload_sha256(tmp_path: Path) -> None:
    """Skipping the non-self-referential canonical digest check would make this fail."""
    payload = _passing_payload()
    payload["scale_rad"][0] = 0.23
    artifact = tmp_path / "tampered.json"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError):
        load_joint_position_contract(artifact)
