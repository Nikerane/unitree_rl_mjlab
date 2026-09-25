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
    torch.save({"infos": {"impact_speed_rollout_telemetry": {
        "selection_count": selections,
        "selection_source": "saved_distribution_activity_mask",
        "selected_speed_mps": {},
        "speed_bounds_mps": [1.5, 2.2],
    }}}, checkpoint)
    spec = {"speed_mode": mode}
    if valid:
        assert campaign()._rollout_evidence(checkpoint, spec)["selection_count"] == selections
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
    torch.save({"infos": {"impact_speed_rollout_telemetry": {
        "selection_count": 0,
        "selection_source": selection_source,
        "selected_speed_mps": {},
        "speed_bounds_mps": bounds,
    }}}, checkpoint)
    with pytest.raises(ValueError, match="selector telemetry|speed bounds"):
        campaign()._rollout_evidence(checkpoint, {"speed_mode": "external"})


def campaign():
    import importlib
    return importlib.import_module("scripts.vega_three_route_impact_campaign")
