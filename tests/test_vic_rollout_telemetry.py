"""Checkpoint telemetry for the qualified Z1 variable-impedance rollout."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from mjlab.rl.runner import MjlabOnPolicyRunner

import src.tasks.hammer.rl.runner as runner_module
from scripts.smoke_cat_soft import (
    DIRECT_FICTT_TASK,
    VIC_TT_TASK,
    _assert_checkpoint_rollout_telemetry,
    run_smoke,
)
from src.tasks.hammer.config.z1.joint_position_contract import JOINT_NAMES
from src.tasks.hammer.rl.runner import (
    HammerOnPolicyRunner,
    _summarize_vic_rollout_telemetry,
)


VIC_ACTION_TERMS = ("joint_position", "joint_stiffness")


def _literal_storage() -> SimpleNamespace:
    actions = torch.zeros(2, 2, 12)
    actions[:, :, 6:] = torch.tensor(
        [
            [[-2.0, -1.0, 0.0, 1.0, 2.0, 0.5], [-1.0, 0.0, 1.0, 2.0, -2.0, -0.5]],
            [[1.0, 1.0, 2.0, -2.0, -1.0, 1.5], [2.0, 2.0, -2.0, -1.0, 0.0, -1.5]],
        ]
    )
    means = torch.zeros(2, 2, 12)
    means[:, :, 6:] = torch.tensor(
        [
            [[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], [-0.5, 0.0, 0.5, 1.0, 1.5, -1.0]],
            [[0.5, 0.5, 1.0, 1.5, -1.0, 0.0], [1.0, 1.0, 1.5, -1.0, -0.5, 0.5]],
        ]
    )
    std = torch.ones(2, 2, 12)
    std[:, :, 6:] = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
            [[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]],
        ]
    )
    return SimpleNamespace(
        actions=actions,
        distribution_params=(means, std),
        num_transitions_per_env=2,
        num_envs=2,
    )


def test_summarizer_reports_independently_worked_gain_statistics() -> None:
    """Wrong slicing, clipping, quantiles, or occupancy must change a literal result."""
    telemetry = _summarize_vic_rollout_telemetry(
        _literal_storage(),
        action_terms=VIC_ACTION_TERMS,
        raw_action_clip=1.0,
    )

    assert telemetry is not None
    assert telemetry["schema_version"] == 1
    assert telemetry["source"] == "cat_rollout_storage"
    assert telemetry["action_terms"] == list(VIC_ACTION_TERMS)
    assert telemetry["joint_names"] == list(JOINT_NAMES)
    assert telemetry["gain_action_indices"] == [6, 7, 8, 9, 10, 11]
    assert telemetry["raw_action_clip"] == 1.0
    assert telemetry["sample_count"] == 4

    sampled_raw = telemetry["sampled_action"]["raw"]
    assert sampled_raw["mean"] == pytest.approx([0.0, 0.5, 0.25, 0.0, -0.25, 0.0])
    assert sampled_raw["std"] == pytest.approx(
        [math.sqrt(2.5), math.sqrt(1.25), math.sqrt(2.1875), math.sqrt(2.5), math.sqrt(2.1875), math.sqrt(1.25)]
    )
    assert sampled_raw["minimum"] == [-2.0, -1.0, -2.0, -2.0, -2.0, -1.5]
    assert sampled_raw["p05"] == pytest.approx([-1.85, -0.85, -1.7, -1.85, -1.85, -1.35])
    assert sampled_raw["median"] == pytest.approx([0.0, 0.5, 0.5, 0.0, -0.5, 0.0])
    assert sampled_raw["p95"] == pytest.approx([1.85, 1.85, 1.85, 1.85, 1.7, 1.35])
    assert sampled_raw["maximum"] == [2.0, 2.0, 2.0, 2.0, 2.0, 1.5]
    assert sampled_raw["lower_bound_occupancy"] == [0.5, 0.25, 0.25, 0.5, 0.5, 0.25]
    assert sampled_raw["upper_bound_occupancy"] == [0.5, 0.5, 0.5, 0.5, 0.25, 0.25]

    sampled_clipped = telemetry["sampled_action"]["clipped"]
    assert sampled_clipped["mean"] == pytest.approx([0.0, 0.25, 0.25, 0.0, -0.25, 0.0])
    assert sampled_clipped["std"] == pytest.approx(
        [1.0, math.sqrt(0.6875), math.sqrt(0.6875), 1.0, math.sqrt(0.6875), math.sqrt(0.625)]
    )
    assert sampled_clipped["minimum"] == [-1.0] * 6
    assert sampled_clipped["p05"] == pytest.approx([-1.0, -0.85, -0.85, -1.0, -1.0, -0.925])
    assert sampled_clipped["median"] == pytest.approx([0.0, 0.5, 0.5, 0.0, -0.5, 0.0])
    assert sampled_clipped["p95"] == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.85, 0.925])
    assert sampled_clipped["maximum"] == [1.0] * 6

    mean_raw = telemetry["deterministic_gaussian_mean"]["raw"]
    assert mean_raw["mean"] == pytest.approx([0.0, 0.25, 0.75, 0.5, 0.25, 0.25])
    assert mean_raw["lower_bound_occupancy"] == [0.25, 0.0, 0.0, 0.25, 0.25, 0.25]
    assert mean_raw["upper_bound_occupancy"] == [0.25, 0.25, 0.5, 0.5, 0.5, 0.25]
    assert telemetry["gaussian_exploration_std"] == pytest.approx(
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    encoded = json.dumps(telemetry, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == telemetry


@pytest.mark.parametrize("field", ("actions", "mean", "std"))
def test_summarizer_rejects_nonfinite_rollout_storage(field: str) -> None:
    """A NaN in any consumed rollout channel must fail before checkpointing."""
    storage = _literal_storage()
    if field == "actions":
        storage.actions[0, 0, 6] = torch.nan
    else:
        params = list(storage.distribution_params)
        params[0 if field == "mean" else 1][0, 0, 6] = torch.nan
        storage.distribution_params = tuple(params)

    with pytest.raises(ValueError, match="finite storage"):
        _summarize_vic_rollout_telemetry(
            storage,
            action_terms=VIC_ACTION_TERMS,
            raw_action_clip=1.0,
        )


def _fake_runner(
    tmp_path, monkeypatch: pytest.MonkeyPatch, *, action_terms: tuple[str, ...], storage
) -> HammerOnPolicyRunner:
    runner = HammerOnPolicyRunner.__new__(HammerOnPolicyRunner)
    runner.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            action_manager=SimpleNamespace(active_terms=list(action_terms))
        ),
        clip_actions=1.0,
    )
    runner.alg = SimpleNamespace(storage=storage)
    runner.logger = SimpleNamespace(logger_type="tensorboard")
    runner.cfg = {"upload_model": False}
    runner._get_export_paths = lambda path: (
        tmp_path,
        "policy.onnx",
        tmp_path / "policy.onnx",
    )
    runner.export_policy_to_onnx = lambda *args, **kwargs: None

    def _save_base(_self, path: str, infos=None) -> None:
        torch.save({"infos": infos}, path)

    monkeypatch.setattr(MjlabOnPolicyRunner, "save", _save_base)
    monkeypatch.setattr(runner_module, "_get_hammer_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner_module, "attach_metadata_to_onnx", lambda *args, **kwargs: None)
    return runner


def test_fic_save_preserves_infos_without_inspecting_rollout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixed-impedance checkpoint infos must remain byte-semantically upstream-owned."""
    malformed = _literal_storage()
    malformed.actions[0, 0, 6] = torch.nan
    runner = _fake_runner(
        tmp_path,
        monkeypatch,
        action_terms=("joint_position",),
        storage=malformed,
    )
    checkpoint = tmp_path / "fic.pt"
    infos = {"existing": [1, 2, 3]}

    runner.save(str(checkpoint), infos)

    assert torch.load(checkpoint, weights_only=False)["infos"] == infos
    assert "vic_rollout_telemetry" not in infos


def test_manual_empty_vic_save_preserves_none_infos(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid pre-rollout VIC save must not invent an empty telemetry record."""
    storage = _literal_storage()
    storage.distribution_params = None
    runner = _fake_runner(
        tmp_path,
        monkeypatch,
        action_terms=VIC_ACTION_TERMS,
        storage=storage,
    )
    checkpoint = tmp_path / "vic-empty.pt"

    runner.save(str(checkpoint))

    assert torch.load(checkpoint, weights_only=False)["infos"] is None


def test_malformed_populated_vic_storage_fails_before_checkpoint_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A populated VIC rollout with misaligned Gaussian storage must fail closed."""
    storage = _literal_storage()
    storage.distribution_params = (
        storage.distribution_params[0][..., :-1],
        storage.distribution_params[1],
    )
    runner = _fake_runner(
        tmp_path,
        monkeypatch,
        action_terms=VIC_ACTION_TERMS,
        storage=storage,
    )
    checkpoint = tmp_path / "vic-malformed.pt"

    with pytest.raises(ValueError, match="aligned nonempty 12D storage"):
        runner.save(str(checkpoint))

    assert not checkpoint.exists()


def test_one_iteration_vic_catppo_checkpoint_contains_rollout_telemetry() -> None:
    """The real rollout/update/save path must persist the latest VIC gain record."""
    checkpoint = run_smoke(task=VIC_TT_TASK, device="cpu", num_envs=1, iters=1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    telemetry = state["infos"]["vic_rollout_telemetry"]
    assert telemetry["schema_version"] == 1
    assert telemetry["sample_count"] == 24
    assert telemetry["action_terms"] == list(VIC_ACTION_TERMS)
    assert telemetry["gain_action_indices"] == [6, 7, 8, 9, 10, 11]
    assert telemetry["joint_names"] == list(JOINT_NAMES)
    assert telemetry["raw_action_clip"] == 1.0
    json.dumps(telemetry, allow_nan=False)


def test_smoke_requires_exact_telemetry_only_for_vic() -> None:
    """The smoke gate must reject missing VIC telemetry without changing FIC."""
    _assert_checkpoint_rollout_telemetry(
        DIRECT_FICTT_TASK, {"infos": None}, expected_sample_count=4
    )

    with pytest.raises(AssertionError, match="VIC checkpoint telemetry"):
        _assert_checkpoint_rollout_telemetry(
            VIC_TT_TASK, {"infos": None}, expected_sample_count=4
        )

    valid = _summarize_vic_rollout_telemetry(
        _literal_storage(),
        action_terms=VIC_ACTION_TERMS,
        raw_action_clip=1.0,
    )
    assert valid is not None
    _assert_checkpoint_rollout_telemetry(
        VIC_TT_TASK,
        {"infos": {"vic_rollout_telemetry": valid}},
        expected_sample_count=4,
    )

    invalid = dict(valid)
    invalid["gain_action_indices"] = [0, 1, 2, 3, 4, 5]
    with pytest.raises(AssertionError, match="gain action indices"):
        _assert_checkpoint_rollout_telemetry(
            VIC_TT_TASK,
            {"infos": {"vic_rollout_telemetry": invalid}},
            expected_sample_count=4,
        )
