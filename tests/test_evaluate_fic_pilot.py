"""Public contract tests for the matched 64-episode FIC pilot evaluator."""

from types import SimpleNamespace

import pytest
import torch

from evaluation.joint_position import evaluate_fic_pilot as pilot


def _cfg(*, task=pilot.FIC0_TASK):
    names = pilot.JOINT_NAMES
    action = SimpleNamespace(
        actuator_names=names,
        scale={name: 0.1 for name in names},
        clip={name: (-1.0, 1.0) for name in names},
        use_default_offset=True,
    )
    rewards = {
        "delivered_impulse": SimpleNamespace(
            weight=4.0, params={"i_ref": pilot.I_REF_N_S}
        ),
        "r_waypoint_progress": SimpleNamespace(weight=8.0, params={}),
    }
    if task == pilot.FICTT_TASK:
        rewards["r_tt"] = SimpleNamespace(weight=-1.0, params={"k_tt": 1.0})
    return SimpleNamespace(
        actions={"joint_position": action},
        rewards=rewards,
        metrics={
            "cat_soft": SimpleNamespace(
                params={
                    "use_vel": True,
                    "use_impulse": True,
                    "vel_detection": "substep",
                    "imp_max_p": 0.0,
                }
            ),
            "substep_impulse_rows": SimpleNamespace(params={"enabled": False}),
        },
    )


def test_public_evaluator_seams_are_exposed() -> None:
    assert callable(pilot.validate_fic_contract)
    assert callable(pilot.evaluate_checkpoint)


@pytest.mark.parametrize("task", pilot.TASKS)
def test_contract_admits_only_exact_fic_treatments(task):
    contract = pilot.validate_fic_contract(
        task,
        _cfg(task=task),
        SimpleNamespace(clip_actions=1.0, actor=SimpleNamespace(obs_normalization=True)),
    )
    assert contract["delivered_impulse_i_ref_n_s"] == pilot.I_REF_N_S
    assert contract["r_tt_enabled"] is (task == pilot.FICTT_TASK)


def test_contract_rejects_cartesian_task_and_treatment_drift():
    with pytest.raises(ValueError, match="unsupported"):
        pilot.validate_fic_contract(
            "cartesian",
            _cfg(),
            SimpleNamespace(clip_actions=1.0, actor=SimpleNamespace(obs_normalization=True)),
        )
    drifted = _cfg()
    drifted.rewards["delivered_impulse"].params["i_ref"] = 0.3088
    with pytest.raises(ValueError, match="i_ref drift"):
        pilot.validate_fic_contract(
            pilot.FIC0_TASK,
            drifted,
            SimpleNamespace(clip_actions=1.0, actor=SimpleNamespace(obs_normalization=True)),
        )
    tt = _cfg(task=pilot.FICTT_TASK)
    tt.rewards["r_tt"].params["k_tt"] = 2.0
    with pytest.raises(ValueError, match="r_tt contract"):
        pilot.validate_fic_contract(
            pilot.FICTT_TASK,
            tt,
            SimpleNamespace(clip_actions=1.0, actor=SimpleNamespace(obs_normalization=True)),
        )


def test_contract_rejects_action_order_and_normalizer_drift():
    action_drift = _cfg()
    action_drift.actions["joint_position"].actuator_names = tuple(reversed(pilot.JOINT_NAMES))
    agent = SimpleNamespace(clip_actions=1.0, actor=SimpleNamespace(obs_normalization=True))
    with pytest.raises(ValueError, match="joint action"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, action_drift, agent)
    with pytest.raises(ValueError, match="normalization"):
        pilot.validate_fic_contract(
            pilot.FIC0_TASK,
            _cfg(),
            SimpleNamespace(clip_actions=1.0, actor=SimpleNamespace(obs_normalization=False)),
        )


def test_cli_requires_exact_positional_treatment_and_launch_shape(tmp_path):
    args = pilot._parse_args(
        [
            pilot.FIC0_TASK,
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--output",
            str(tmp_path / "result.json"),
            "--device",
            "cuda:0",
        ]
    )
    assert args.task == pilot.FIC0_TASK
    assert args.device == "cuda:0"
    with pytest.raises(SystemExit):
        pilot._parse_args(["--task", pilot.FIC0_TASK])


def test_evaluator_protocol_constants_are_frozen():
    assert pilot.SEED == 2026081202
    assert pilot.NUM_ENVS == 64
    assert pilot.EPISODE_LENGTH_S == 4.0


def test_capture_records_each_first_terminal_before_tracker_mutation():
    tracker = SimpleNamespace(
        started=torch.tensor([True, True]),
        finalized=torch.tensor([True, True]),
        productive=torch.tensor([True, False]),
        reason=torch.tensor([1, 2]),
        v_precontact=torch.tensor([0.7, 0.8]),
        delivered=torch.tensor([0.25, 0.5]),
    )
    records = {}
    done = torch.tensor([True, True])
    ids = pilot.capture_first_terminals(
        records=records,
        done=done,
        terminated=torch.tensor([True, False]),
        timed_out=torch.tensor([False, True]),
        steps=torch.tensor([9, 10]),
        tracker=tracker,
        nail_depth=torch.tensor([0.032, 0.01]),
        joint_impulse_peak=torch.ones(2, 6),
    )
    tracker.delivered.zero_()
    pilot.capture_first_terminals(
        records=records,
        done=done,
        terminated=torch.tensor([True, False]),
        timed_out=torch.tensor([False, True]),
        steps=torch.tensor([99, 99]),
        tracker=tracker,
        nail_depth=torch.zeros(2),
        joint_impulse_peak=torch.zeros(2, 6),
    )
    assert ids == (0, 1)
    assert [records[index]["first_event_impulse_n_s"] for index in (0, 1)] == [0.25, 0.5]
    assert [records[index]["episode_steps"] for index in (0, 1)] == [9, 10]


def test_new_json_writer_refuses_existing_output_and_never_writes_partial(tmp_path):
    output = tmp_path / "fic.json"
    pilot._publish_new_json(output, {"ok": True})
    assert output.read_text().strip()
    with pytest.raises(FileExistsError):
        pilot._publish_new_json(output, {"ok": False})


def test_incomplete_population_is_rejected_before_json_can_be_published():
    with pytest.raises(RuntimeError, match="incomplete"):
        pilot._ordered_complete_records({0: {"env_id": 0}})


def test_missing_checkpoint_fails_before_loading_any_live_configuration(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        pilot.evaluate_checkpoint(pilot.FIC0_TASK, tmp_path / "missing.pt", tmp_path / "out.json")


def test_existing_output_is_refused_before_loading_any_live_configuration(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not loaded")
    output = tmp_path / "out.json"
    output.write_text("already present")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pilot.evaluate_checkpoint(pilot.FIC0_TASK, checkpoint, output)
