"""Public runner and CLI contract for the primary controlled-drop calibration."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess

import pytest

from scripts import calibrate_controlled_drop as calibration_cli
from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML
from src.tasks.hammer.calibration import controlled_drop
from src.tasks.hammer.calibration.controlled_drop import run_primary_calibration
from src.tasks.hammer.calibration.controlled_drop_contract import (
    ControlledDropExecution,
    ControlledDropResult,
    ControlledDropTrial,
    build_primary_result,
)


def _trial(trial_index: int, *, impulse: float) -> ControlledDropTrial:
    return ControlledDropTrial(
        trial_index=trial_index,
        h0_m=0.150,
        release_velocity_m_s=0.0,
        precontact_velocity_m_s=1.70,
        impulse_n_s=impulse,
        contacted=True,
        finalized=True,
        productive=True,
        reason="success",
        depth_at_contact_m=0.0,
        peak_depth_m=0.031,
    )


def _git_boundary(
    *,
    code_revisions: list[str],
    asset_revisions: list[str],
    code_statuses: list[str],
    asset_statuses: list[str],
):
    code_root = Path(controlled_drop.__file__).resolve().parents[4]
    asset_root = Z1_HAMMER_XML.parents[2]
    outputs = {
        (str(code_root), ("rev-parse", "HEAD")): iter(code_revisions),
        (
            str(code_root),
            ("status", "--porcelain", "--untracked-files=no"),
        ): iter(code_statuses),
        (str(asset_root), ("rev-parse", "HEAD")): iter(asset_revisions),
        (
            str(asset_root),
            ("status", "--porcelain", "--untracked-files=no"),
        ): iter(asset_statuses),
    }

    def fake_run(args, *, check, capture_output, text):
        assert args[:2] == ["git", "-C"]
        assert check is True
        assert capture_output is True
        assert text is True
        key = (args[2], tuple(args[3:]))
        assert key in outputs
        stdout = next(outputs[key])
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    return fake_run


def _result() -> ControlledDropResult:
    execution = ControlledDropExecution(
        code_revision="a" * 40,
        asset_revision="b" * 40,
        device="cpu",
        backend="mujoco-warp",
        mujoco_version="3.8.1",
        mujoco_warp_version="3.8.1",
        mjlab_version="1.4.0",
        physics_dt_s=0.002,
        h0_m=0.150,
        mass_kg=0.200,
        radius_m=0.012,
        half_height_m=0.004,
        friction=(1.5, 0.02, 0.002),
        slide_axis=(0.0, 0.0, -1.0),
        slide_damping=0.0,
        slide_frictionloss=0.0,
        tracker_axis=(0.0, 0.0, -1.0),
        tracker_window_substeps=25,
        tracker_progress_eps=5e-4,
    )
    trials = tuple(
        _trial(index, impulse=impulse)
        for index, impulse in enumerate((0.1, 0.2, 0.3, 0.4, 0.5), start=1)
    )
    return build_primary_result(execution, trials)


def test_primary_calibration_calls_five_ordered_fresh_trials_and_keeps_every_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run_one(*, device, trial_index):
        calls.append((device, trial_index))
        return _trial(trial_index, impulse=trial_index / 10)

    monkeypatch.setattr(controlled_drop, "run_one_primary_drop", fake_run_one)
    monkeypatch.setattr(
        controlled_drop,
        "capture_clean_execution_identity",
        lambda **kwargs: ("a" * 40, "b" * 40),
    )
    result = run_primary_calibration(
        device="cpu",
        expected_code_revision="a" * 40,
        expected_asset_revision="b" * 40,
    )

    assert calls == [
        ("cpu", 1),
        ("cpu", 2),
        ("cpu", 3),
        ("cpu", 4),
        ("cpu", 5),
    ]
    assert [row.impulse_n_s for row in result.summary.trials] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    assert result.summary.i_ref_mean_n_s == 0.3


@pytest.mark.parametrize(
    (
        "observed_code",
        "observed_asset",
        "code_status",
        "asset_status",
        "message",
    ),
    [
        ("c" * 40, "b" * 40, "", "", "code revision differs"),
        ("a" * 40, "c" * 40, "", "", "asset revision differs"),
        (
            "a" * 40,
            "b" * 40,
            " M tracked.py\n",
            "",
            "code repository has tracked changes",
        ),
        (
            "a" * 40,
            "b" * 40,
            "",
            " M hammer.xml\n",
            "asset repository has tracked changes",
        ),
    ],
)
def test_primary_calibration_rejects_revision_mismatch_or_tracked_dirt_before_any_drop(
    monkeypatch: pytest.MonkeyPatch,
    observed_code: str,
    observed_asset: str,
    code_status: str,
    asset_status: str,
    message: str,
) -> None:
    trial_calls = []
    monkeypatch.setattr(
        controlled_drop,
        "run_one_primary_drop",
        lambda **kwargs: trial_calls.append(kwargs),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        _git_boundary(
            code_revisions=[observed_code],
            asset_revisions=[observed_asset],
            code_statuses=[code_status],
            asset_statuses=[asset_status],
        ),
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        run_primary_calibration(
            device="cpu",
            expected_code_revision="a" * 40,
            expected_asset_revision="b" * 40,
        )

    assert trial_calls == []


def test_primary_calibration_rejects_repository_change_after_fifth_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial_calls = []

    def fake_run_one(*, device, trial_index):
        trial_calls.append((device, trial_index))
        return _trial(trial_index, impulse=trial_index / 10)

    monkeypatch.setattr(controlled_drop, "run_one_primary_drop", fake_run_one)
    monkeypatch.setattr(
        subprocess,
        "run",
        _git_boundary(
            code_revisions=["a" * 40, "c" * 40],
            asset_revisions=["b" * 40, "b" * 40],
            code_statuses=["", ""],
            asset_statuses=["", ""],
        ),
    )

    with pytest.raises((RuntimeError, ValueError)):
        run_primary_calibration(
            device="cpu",
            expected_code_revision="a" * 40,
            expected_asset_revision="b" * 40,
        )

    assert trial_calls == [
        ("cpu", 1),
        ("cpu", 2),
        ("cpu", 3),
        ("cpu", 4),
        ("cpu", 5),
    ]


def test_cli_accepts_only_output_device_and_expected_revisions(tmp_path: Path) -> None:
    output = tmp_path / "primary.json"
    base = [
        "--device",
        "cpu",
        "--output",
        str(output),
        "--code-revision",
        "a" * 40,
        "--asset-revision",
        "b" * 40,
    ]

    args = calibration_cli.parse_args(base)

    assert args.device == "cpu"
    assert args.output == output
    assert args.code_revision == "a" * 40
    assert args.asset_revision == "b" * 40
    with pytest.raises(SystemExit):
        calibration_cli.parse_args([*base, "--device", "cuda:1"])
    for forbidden in (
        "--mass",
        "--height",
        "--runs",
        "--radius",
        "--friction",
        "--window-substeps",
    ):
        with pytest.raises(SystemExit):
            calibration_cli.parse_args([*base, forbidden, "1"])


def test_writer_publishes_nested_complete_json_without_clobber(
    tmp_path: Path,
) -> None:
    result = _result()
    output = tmp_path / "primary.json"

    calibration_cli.write_primary_result(output, result)

    encoded = output.read_text(encoding="utf-8")
    assert encoded.startswith('{\n  "execution": {')
    assert encoded.endswith("\n")
    payload = json.loads(encoded)
    assert set(payload) == {"execution", "schema_version", "summary"}
    assert [row["trial_index"] for row in payload["summary"]["trials"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert (
        math.fsum(row["impulse_n_s"] for row in payload["summary"]["trials"]) / 5
        == payload["summary"]["i_ref_mean_n_s"]
    )
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        calibration_cli.write_primary_result(output, result)
    assert output.read_bytes() == before

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target.json")
    assert os.path.lexists(dangling)
    with pytest.raises(FileExistsError):
        calibration_cli.write_primary_result(dangling, result)

    invalid = replace(
        result,
        summary=replace(result.summary, i_ref_mean_n_s=float("nan")),
    )
    invalid_output = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        calibration_cli.write_primary_result(invalid_output, invalid)
    assert not os.path.lexists(invalid_output)


def test_cli_failure_leaves_no_final_or_temporary_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "primary.json"

    def fail_calibration(**kwargs):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(
        calibration_cli,
        "run_primary_calibration",
        fail_calibration,
    )

    with pytest.raises(RuntimeError, match="calibration failed"):
        calibration_cli.main(
            [
                "--device",
                "cpu",
                "--output",
                str(output),
                "--code-revision",
                "a" * 40,
                "--asset-revision",
                "b" * 40,
            ]
        )

    assert not os.path.lexists(output)
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []
