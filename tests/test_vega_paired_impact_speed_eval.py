"""Contract tests for the bounded, evaluation-only Vega checkpoint array."""

import importlib
import json

import pytest


def campaign():
    return importlib.import_module("scripts.vega_paired_impact_speed_eval")


def frozen_contract():
    return {
        "speed_mode": "policy", "speed_bounds_mps": [1.5, 2.2],
        "speed_cost": {"coefficient": .8493676},
        "controller_identity": "sampled_vic500_v1",
        "action_type": "joint_position_variable_impedance_sampled_pd",
        "sampled_pd": {"gain_mode": "variable", "version": "variable_impedance_pd500_v1"},
        "safety": {
            "cat_soft": {
                "imp_limit": [.369, .246, .738, .369, .246, .0164],
                "imp_max_p": .2, "limit": 3.1415,
                "use_vel": True, "use_impulse": True, "vel_detection": "substep",
            },
            "substep_impulse": {"event_window_substeps": 200},
        },
    }


def write_valid_checkpoint(tmp_path, monkeypatch):
    module = campaign()
    spec = module.checkpoint_spec(0)
    checkpoint = tmp_path / "training" / "runs" / f"run-{spec['run_index']}" / "train" / "model_49.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen policy checkpoint")
    specs = (dict(spec, expected_sha256=module.file_sha256(checkpoint)), *module.CHECKPOINTS[1:])
    monkeypatch.setattr(module, "CHECKPOINTS", specs)
    evaluator_script = tmp_path / "evaluate_impact_speed.py"
    evaluator_script.write_text("# frozen evaluator overlay\n")
    return checkpoint, evaluator_script, specs[0]


def valid_summary(spec, *, seed=20260926, task=None, contract=None):
    shared_contract = frozen_contract() if contract is None else contract
    goals = [None, 1.5, 1.85, 2.2]
    roles = ["autonomous_policy_selector", "preaction_fixed_target",
             "preaction_fixed_target", "preaction_fixed_target"]
    return {
        "task": campaign().TASK if task is None else task,
        "seed": seed, "actor_status": "checkpoint", "mode": "paired",
        "learning_updates": 0,
        "checkpoint_sha256": spec["expected_sha256"],
        "initial_state_pairing": "verified_policy_vs_preaction_fixed_targets",
        "conditions": [
            {"goal_mps": goal, "condition_role": role, "contract": shared_contract,
             "measurement": {"protocol": "common-window"}}
            for goal, role in zip(goals, roles)
        ],
        "comparisons": [{} for _ in range(6)],
    }


def test_array_slots_are_only_existing_policy_checkpoints():
    specs = [campaign().checkpoint_spec(index) for index in range(3)]

    assert [(row["run_index"], row["seed"]) for row in specs] == [(1, 10), (3, 11), (5, 12)]
    assert all(row["expected_sha256"] for row in specs)
    with pytest.raises(ValueError, match="0..2"):
        campaign().checkpoint_spec(3)
    with pytest.raises(ValueError, match="0..2"):
        campaign().checkpoint_spec(True)


def test_command_is_fixed_contract_paired_evaluation_not_training(tmp_path):
    command = campaign().evaluation_command(
        tmp_path / "model_49.pt", tmp_path / "seed-10", tmp_path / "evaluate_impact_speed.py"
    )

    assert command[:2] == [campaign().sys.executable, str(tmp_path / "evaluate_impact_speed.py")]
    assert command[command.index("--mode") + 1] == "paired"
    assert command[command.index("--num-envs") + 1] == "384"
    assert command[command.index("--goals") + 1:command.index("--goals") + 4] == ["1.5", "1.85", "2.2"]
    assert command[command.index("--measurement-protocol") + 1] == "common-window"
    assert command[command.index("--sampled-pd500-gain-mode") + 1] == "variable"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert "--sampled-pd500" in command
    assert "train" not in command


def test_remote_campaign_runner_rejects_missing_or_modified_checkpoint(tmp_path, monkeypatch):
    spec = campaign().checkpoint_spec(0)
    checkpoint = tmp_path / "training" / "runs" / f"run-{spec['run_index']}" / "train" / "model_49.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not the frozen policy checkpoint")
    monkeypatch.setattr(campaign(), "CHECKPOINTS", (spec, *campaign().CHECKPOINTS[1:]))

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        campaign().run_evaluation(0, tmp_path / "training", tmp_path / "evaluation", tmp_path / "evaluate.py")

    assert not (tmp_path / "evaluation").exists()


def test_remote_campaign_runner_rejects_missing_checkpoint(tmp_path, monkeypatch):
    spec = campaign().checkpoint_spec(0)
    monkeypatch.setattr(campaign(), "CHECKPOINTS", (spec, *campaign().CHECKPOINTS[1:]))

    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        campaign().run_evaluation(0, tmp_path / "missing-training", tmp_path / "evaluation", tmp_path / "evaluate.py")


def test_remote_campaign_runner_fails_closed_on_wrong_task_seed_goals_or_contract(tmp_path, monkeypatch):
    module = campaign()
    checkpoint, evaluator_script, spec = write_valid_checkpoint(tmp_path, monkeypatch)
    result_dir = tmp_path / "evaluation" / "evaluations" / "seed-10"

    def fake_run(command, *, check):
        assert check is True
        result_dir.mkdir(parents=True)
        summary = valid_summary(spec, seed=20260925)
        (result_dir / "summary.json").write_text(json.dumps(summary))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="frozen run contract"):
        module.run_evaluation(0, tmp_path / "training", tmp_path / "evaluation", evaluator_script)

    assert not (result_dir / "completion.json").exists()


@pytest.mark.parametrize("mutation", [
    "task", "seed", "goal", "actor", "protocol", "missing_measurement",
    "bounds", "cost", "controller", "caps",
])
def test_summary_contract_validation_rejects_experiment_drift(tmp_path, monkeypatch, mutation):
    module = campaign()
    _, _, spec = write_valid_checkpoint(tmp_path, monkeypatch)
    summary = valid_summary(spec)
    if mutation == "task":
        summary["task"] = "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed"
    elif mutation == "seed":
        summary["seed"] += 1
    elif mutation == "goal":
        summary["conditions"][1]["goal_mps"] = 1.6
    elif mutation == "actor":
        summary["actor_status"] = "untrained software smoke only"
    elif mutation == "protocol":
        for condition in summary["conditions"]:
            condition["measurement"]["protocol"] = "native"
    elif mutation == "missing_measurement":
        summary["conditions"][0].pop("measurement")
    elif mutation == "bounds":
        for condition in summary["conditions"]:
            condition["contract"]["speed_bounds_mps"] = [.5, 1.]
    elif mutation == "cost":
        for condition in summary["conditions"]:
            condition["contract"]["speed_cost"]["coefficient"] = .4
    elif mutation == "controller":
        for condition in summary["conditions"]:
            condition["contract"]["sampled_pd"]["gain_mode"] = "fixed"
    else:
        for condition in summary["conditions"]:
            condition["contract"]["safety"]["cat_soft"]["imp_limit"][1] = .5

    with pytest.raises(ValueError, match="frozen run contract"):
        module.validate_summary(summary, spec["expected_sha256"])


def test_remote_campaign_runner_writes_verified_manifest_and_refuses_reuse(tmp_path, monkeypatch):
    module = campaign()
    checkpoint, evaluator_script, spec = write_valid_checkpoint(tmp_path, monkeypatch)
    output_root = tmp_path / "evaluation"
    result_dir = output_root / "evaluations" / "seed-10"
    launched = []

    def fake_run(command, *, check):
        launched.append(command)
        assert check is True
        result_dir.mkdir(parents=True)
        (result_dir / "summary.json").write_text(json.dumps(valid_summary(spec)))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    manifest = module.run_evaluation(0, tmp_path / "training", output_root, evaluator_script)

    assert manifest["status"] == "completed"
    assert manifest["checkpoint_sha256"] == spec["expected_sha256"]
    assert manifest["learning_updates"] == 0
    assert len(launched) == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.run_evaluation(0, tmp_path / "training", output_root, evaluator_script)
    assert len(launched) == 1
