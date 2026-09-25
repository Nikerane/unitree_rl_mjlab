import pytest

from scripts.vega_three_route_impact_campaign import (
    TASK,
    evaluation_command,
    pair_reset_evidence,
    run_spec,
    training_config,
)


def test_six_run_matrix_pairs_speed_modes_with_variable_impedance():
    runs = [run_spec(index) for index in range(6)]
    assert [(run["speed_mode"], run["seed"]) for run in runs] == [
        ("external", 10), ("policy", 10),
        ("external", 11), ("policy", 11),
        ("external", 12), ("policy", 12),
    ]
    assert all(run["gain_mode"] == "variable" for run in runs)
    assert all(run["iterations"] == 50 and run["worlds"] == 1024 for run in runs)
    for invalid in (-1, 6, 1.0, True):
        with pytest.raises(ValueError):
            run_spec(invalid)


@pytest.mark.parametrize("index", range(6))
def test_training_config_uses_fresh_three_route_speed_experiment(index):
    spec = run_spec(index)
    cfg = training_config(spec)
    assert TASK == "Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed"
    assert cfg.sampled_pd500 and cfg.sampled_pd500_gain_mode == "variable"
    assert cfg.speed_mode == spec["speed_mode"]
    assert cfg.speed_bounds == (1.5, 2.2) and cfg.speed_cost == .8493676
    assert cfg.agent.seed == spec["seed"] and cfg.agent.max_iterations == 50
    assert cfg.env.scene.num_envs == 1024
    assert cfg.env.events["sample_strike_route_signs"].params.get("fixed_route_sign") is None
    assert len(cfg.env.events["reset_strike_route_joints"].params["route_joint_positions"]) == 3
    assert cfg.qualification_checkpoint is None


@pytest.mark.parametrize("mode", ["external", "policy"])
def test_evaluation_command_uses_held_out_three_route_protocol(mode, tmp_path):
    from scripts.evaluate_impact_speed import parse_args

    command = evaluation_command(
        tmp_path / "model_49.pt", tmp_path / "eval", mode, "common-window", smoke=False
    )
    args = parse_args(command[2:])
    assert args.task == TASK
    assert args.seed == 20260925 and args.num_envs == 384
    assert args.speed_bounds == [1.5, 2.2] and args.speed_cost == .8493676
    assert args.mode == mode
    assert args.goals == ([1.5, 1.85, 2.2] if mode == "external" else None)
    assert args.sampled_pd500 and args.sampled_pd500_gain_mode == "variable"
    assert args.measurement_protocol == "common-window"


def test_pairing_report_requires_reset_hash_and_route_identity_evidence():
    def condition(routes, hashes):
        return {"first_episode_route_signs": routes, "reset_hashes": hashes}

    policy = {"conditions": [condition([-1, 0, 1], ["a", "b", "c"])]}
    external = {"conditions": [
        condition([-1, 0, 1], ["a", "b", "c"]),
        condition([-1, 0, 1], ["a", "b", "c"]),
        condition([-1, 0, 1], ["a", "b", "c"]),
    ]}
    report = pair_reset_evidence(external, policy)
    assert report["claim"] == "paired_initial_evaluation_states"
    assert all(row["reset_hashes_match"] and row["route_signs_match"] for row in report["goal_conditions"])

    external["conditions"][1]["first_episode_route_signs"] = [1, 0, -1]
    report = pair_reset_evidence(external, policy)
    assert report["claim"] == "matched_training_seeds_only"
    assert report["goal_conditions"][1]["route_signs_match"] is False


@pytest.mark.parametrize(("mode", "selections", "valid"), [
    ("external", 0, True), ("external", -1, False),
    ("policy", 1, True), ("policy", 0, False),
])
def test_training_rollout_selector_evidence_is_mode_specific(mode, selections, valid, tmp_path):
    import torch

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"iter": 7, "infos": {"impact_speed_rollout_telemetry": {
        "selection_count": selections,
        "selection_source": "saved_distribution_activity_mask",
        "selected_speed_mps": {},
        "speed_bounds_mps": [1.5, 2.2],
    }}}, checkpoint)
    spec = {"speed_mode": mode}
    if valid:
        evidence = campaign()._rollout_evidence(checkpoint, spec)
        assert evidence["inspected_saved_windows_selection_count"] == selections
        assert evidence["inspected_saved_windows"] == [{
            "checkpoint": "checkpoint.pt", "iteration": 7,
            "rollout_window": "rollout_storage_saved_at_iteration_7",
            "selection_count": selections,
            "selected_speed_mps": {},
        }]
    else:
        with pytest.raises(ValueError, match="selector telemetry"):
            campaign()._rollout_evidence(checkpoint, spec)


@pytest.mark.parametrize(("selection_source", "bounds"), [
    ("action_tensor", [1.5, 2.2]),
    ("saved_distribution_activity_mask", [1.0, 2.0]),
])
def test_training_rollout_evidence_requires_mask_source_and_campaign_bounds(
    selection_source, bounds, tmp_path,
):
    import torch

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"iter": 7, "infos": {"impact_speed_rollout_telemetry": {
        "selection_count": 0,
        "selection_source": selection_source,
        "selected_speed_mps": {},
        "speed_bounds_mps": bounds,
    }}}, checkpoint)
    with pytest.raises(ValueError, match="selector telemetry|speed bounds"):
        campaign()._rollout_evidence(checkpoint, {"speed_mode": "external"})


def _save_rollout_window(path, *, iteration, selections):
    import torch

    torch.save({"iter": iteration, "infos": {"impact_speed_rollout_telemetry": {
        "selection_count": selections,
        "selection_source": "saved_distribution_activity_mask",
        "selected_speed_mps": {},
        "speed_bounds_mps": [1.5, 2.2],
    }}}, path)


def test_policy_selection_in_first_saved_window_passes_when_final_window_has_zero(tmp_path):
    first, final = tmp_path / "model_0.pt", tmp_path / "model_1.pt"
    _save_rollout_window(first, iteration=0, selections=32)
    _save_rollout_window(final, iteration=1, selections=0)

    evidence = campaign()._rollout_evidence(final, {"speed_mode": "policy"}, first)

    assert evidence["inspected_saved_windows_selection_count"] == 32
    assert [(row["checkpoint"], row["iteration"], row["selection_count"])
            for row in evidence["inspected_saved_windows"]] == [
                ("model_0.pt", 0, 32), ("model_1.pt", 1, 0)]


@pytest.mark.parametrize(("mode", "first_count", "final_count"), [
    ("policy", 0, 0), ("external", 1, 0),
])
def test_inspected_windows_reject_wrong_aggregate_or_earlier_external_selection(
    tmp_path, mode, first_count, final_count,
):
    first, final = tmp_path / "model_0.pt", tmp_path / "model_1.pt"
    _save_rollout_window(first, iteration=0, selections=first_count)
    _save_rollout_window(final, iteration=1, selections=final_count)
    with pytest.raises(ValueError, match="selector telemetry"):
        campaign()._rollout_evidence(final, {"speed_mode": mode}, first)


def test_external_all_zero_saved_windows_pass(tmp_path):
    first, final = tmp_path / "model_0.pt", tmp_path / "model_1.pt"
    _save_rollout_window(first, iteration=0, selections=0)
    _save_rollout_window(final, iteration=1, selections=0)
    evidence = campaign()._rollout_evidence(final, {"speed_mode": "external"}, first)
    assert evidence["inspected_saved_windows_selection_count"] == 0
    assert len(evidence["inspected_saved_windows"]) == 2


def test_duplicate_saved_window_is_not_double_counted(tmp_path):
    checkpoint = tmp_path / "model_0.pt"
    _save_rollout_window(checkpoint, iteration=0, selections=3)
    evidence = campaign()._rollout_evidence(checkpoint, {"speed_mode": "policy"}, checkpoint)
    assert evidence["inspected_saved_windows_selection_count"] == 3
    assert len(evidence["inspected_saved_windows"]) == 1


@pytest.mark.parametrize("invalid", [True, -1, 1.5, "3", None])
def test_saved_window_rejects_malformed_selection_count(tmp_path, invalid):
    checkpoint = tmp_path / "model_0.pt"
    _save_rollout_window(checkpoint, iteration=0, selections=invalid)
    with pytest.raises(ValueError, match="selector telemetry"):
        campaign()._rollout_evidence(checkpoint, {"speed_mode": "policy"})


def test_train_and_evaluate_inspects_initial_and_final_saved_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = campaign()
    spec = {"speed_mode": "policy", "iterations": 2}
    captured = []

    def fake_train(_task, _cfg, train_dir):
        train_dir.mkdir()
        _save_rollout_window(train_dir / "model_0.pt", iteration=0, selections=1)
        _save_rollout_window(train_dir / "model_1.pt", iteration=1, selections=0)

    original_rollout = module._rollout_evidence
    def inspect_rollout(final, run_spec, initial):
        captured.append((final, initial, run_spec))
        return original_rollout(final, run_spec, initial)

    monkeypatch.setattr(module, "training_config", lambda _spec: SimpleNamespace())
    monkeypatch.setattr("scripts.train.run_train", fake_train)
    monkeypatch.setattr(module, "_selector_preflight", lambda *_: {"strict_reload": "passed"})
    monkeypatch.setattr(module, "_rollout_evidence", inspect_rollout)
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: None)
    result = module.train_and_evaluate(spec, tmp_path / "run", smoke=True)
    assert captured == [(tmp_path / "run/train/model_1.pt", tmp_path / "run/train/model_0.pt", spec)]
    assert result["training_selector_telemetry"]["inspected_saved_windows_selection_count"] == 1


def campaign():
    import importlib
    return importlib.import_module("scripts.vega_three_route_impact_campaign")
