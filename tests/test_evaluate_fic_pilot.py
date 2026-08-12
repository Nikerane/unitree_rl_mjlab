"""Public contract tests for the matched 64-episode FIC pilot evaluator."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from evaluation.joint_position import evaluate_fic_pilot as pilot


def _live_configs(task=pilot.FIC0_TASK):
    return pilot.load_env_cfg(task, play=False), pilot.load_rl_cfg(task)


def test_public_evaluator_seams_are_exposed() -> None:
    assert callable(pilot.validate_fic_contract)
    assert callable(pilot.evaluate_checkpoint)


@pytest.mark.parametrize("task", pilot.TASKS)
def test_contract_admits_only_exact_fic_treatments(task):
    env_cfg, agent_cfg = _live_configs(task)
    contract = pilot.validate_fic_contract(task, env_cfg, agent_cfg)
    assert contract["delivered_impulse_i_ref_n_s"] == pilot.I_REF_N_S
    assert contract["r_tt_enabled"] is (task == pilot.FICTT_TASK)


def test_contract_rejects_cartesian_task_and_treatment_drift():
    env_cfg, agent_cfg = _live_configs()
    with pytest.raises(ValueError, match="unsupported"):
        pilot.validate_fic_contract("cartesian", env_cfg, agent_cfg)
    drifted, agent_cfg = _live_configs()
    drifted.rewards["delivered_impulse"].params["i_ref"] = 0.3088
    with pytest.raises(ValueError, match="i_ref drift"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, drifted, agent_cfg)
    tt, agent_cfg = _live_configs(pilot.FICTT_TASK)
    tt.rewards["r_tt"].params["k_tt"] = 2.0
    with pytest.raises(ValueError, match="r_tt contract"):
        pilot.validate_fic_contract(pilot.FICTT_TASK, tt, agent_cfg)


def test_contract_rejects_action_order_and_normalizer_drift():
    action_drift, agent = _live_configs()
    action_drift.actions["joint_position"].actuator_names = tuple(reversed(pilot.JOINT_NAMES))
    with pytest.raises(ValueError, match="joint action"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, action_drift, agent)
    env_cfg, agent = _live_configs()
    agent.actor.obs_normalization = False
    with pytest.raises(ValueError, match="normalization"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent)


def test_contract_rejects_qualified_joint_scale_value_drift():
    env_cfg, agent_cfg = _live_configs()
    env_cfg.actions["joint_position"].scale["joint1"] *= 2.0

    with pytest.raises(ValueError, match="joint action"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_qualified_mapping_order_drift():
    env_cfg, agent_cfg = _live_configs()
    action = env_cfg.actions["joint_position"]
    action.scale = dict(reversed(tuple(action.scale.items())))

    with pytest.raises(ValueError, match="joint action"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_qualified_joint_physical_clip_drift():
    env_cfg, agent_cfg = _live_configs()
    env_cfg.actions["joint_position"].clip["joint2"] = (-1.0, 2.96706)

    with pytest.raises(ValueError, match="joint action"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_joint_action_preserve_order_drift():
    env_cfg, agent_cfg = _live_configs()
    env_cfg.actions["joint_position"].preserve_order = False

    with pytest.raises(ValueError, match="joint action"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_rtt_function_drift():
    env_cfg, agent_cfg = _live_configs(pilot.FICTT_TASK)
    env_cfg.rewards["r_tt"].func = lambda *args, **kwargs: torch.zeros(1)

    with pytest.raises(ValueError, match="r_tt contract"):
        pilot.validate_fic_contract(pilot.FICTT_TASK, env_cfg, agent_cfg)


def test_contract_rejects_rtt_selected_joint_drift():
    env_cfg, agent_cfg = _live_configs(pilot.FICTT_TASK)
    env_cfg.rewards["r_tt"].params["robot_cfg"].joint_names = pilot.JOINT_NAMES[:-1]

    with pytest.raises(ValueError, match="r_tt contract"):
        pilot.validate_fic_contract(pilot.FICTT_TASK, env_cfg, agent_cfg)


def test_contract_rejects_rtt_joint_order_semantics_drift():
    env_cfg, agent_cfg = _live_configs(pilot.FICTT_TASK)
    env_cfg.rewards["r_tt"].params["robot_cfg"].preserve_order = False

    with pytest.raises(ValueError, match="r_tt contract"):
        pilot.validate_fic_contract(pilot.FICTT_TASK, env_cfg, agent_cfg)


def test_contract_rejects_contact_gating_added_to_rtt():
    env_cfg, agent_cfg = _live_configs(pilot.FICTT_TASK)
    env_cfg.rewards["r_tt"].params["contact_only"] = True

    with pytest.raises(ValueError, match="r_tt contract"):
        pilot.validate_fic_contract(pilot.FICTT_TASK, env_cfg, agent_cfg)


def test_contract_rejects_rtt_robot_entity_drift():
    env_cfg, agent_cfg = _live_configs(pilot.FICTT_TASK)
    env_cfg.rewards["r_tt"].params["robot_cfg"].name = "nail_block"

    with pytest.raises(ValueError, match="r_tt contract"):
        pilot.validate_fic_contract(pilot.FICTT_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize("field", ("physics_dt", "decimation", "decimation_type"))
def test_contract_rejects_control_timing_drift(field):
    env_cfg, agent_cfg = _live_configs()
    if field == "physics_dt":
        env_cfg.sim.mujoco.timestep = 0.001
    elif field == "decimation":
        env_cfg.decimation = 5
    else:
        env_cfg.decimation = 10.0

    with pytest.raises(ValueError, match="timing"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_fixed_actuator_signature_drift():
    env_cfg, agent_cfg = _live_configs()
    env_cfg.scene.entities["robot"].articulation.actuators[1].stiffness = 1499.0

    with pytest.raises(ValueError, match="actuator"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize("field", ("offset", "entity"))
def test_contract_rejects_default_offset_action_semantics_drift(field):
    env_cfg, agent_cfg = _live_configs()
    action = env_cfg.actions["joint_position"]
    if field == "offset":
        action.offset = 0.1
    else:
        action.entity_name = "nail_block"

    with pytest.raises(ValueError, match="joint action"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("limit", 3.0),
        ("max_p", 0.4),
        ("min_p", 0.1),
        ("tau", 0.9),
        ("imp_limit", [1.0] * 6),
        ("imp_seed", 0.01),
    ),
)
def test_contract_rejects_velocity_cat_hook_parameter_drift(key, value):
    env_cfg, agent_cfg = _live_configs()
    env_cfg.metrics["cat_soft"].params[key] = value

    with pytest.raises(ValueError, match="CaT contract"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_velocity_cat_hook_function_drift():
    env_cfg, agent_cfg = _live_configs()
    env_cfg.metrics["cat_soft"].func = lambda *args, **kwargs: torch.zeros(1)

    with pytest.raises(ValueError, match="CaT contract"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_unknown_velocity_cat_hook_parameter():
    env_cfg, agent_cfg = _live_configs()
    env_cfg.metrics["cat_soft"].params["contact_only"] = True

    with pytest.raises(ValueError, match="CaT contract"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_cat_negative_term_split_drift(monkeypatch):
    env_cfg, agent_cfg = _live_configs(pilot.FICTT_TASK)
    monkeypatch.setattr(pilot, "_NEG_TERMS", ("action_rate", "joint_pos_limits"), raising=False)

    with pytest.raises(ValueError, match="negative-term split"):
        pilot.validate_fic_contract(pilot.FICTT_TASK, env_cfg, agent_cfg)


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


def test_partial_reset_preserves_unfinished_inputs_and_torch_rng(monkeypatch):
    observations = TensorDict(
        {
            "actor": torch.tensor([[10.0], [20.0]]),
            "critic": torch.tensor([[30.0], [40.0]]),
        },
        batch_size=[2],
    )
    cpu_state = torch.get_rng_state().clone()
    cuda_rng = {"state": torch.tensor([7], dtype=torch.uint8)}
    restored_devices = []

    def get_cuda_rng_state(device):
        assert torch.device(device) == torch.device("cuda:0")
        return cuda_rng["state"].clone()

    def set_cuda_rng_state(state, device):
        restored_devices.append(torch.device(device))
        cuda_rng["state"] = state.clone()

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_cuda_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_cuda_rng_state)

    class ResettingEnv:
        num_envs = 2
        device = torch.device("cuda:0")

        def reset(self, *, env_ids):
            assert env_ids.tolist() == [0]
            torch.rand(8)
            cuda_rng["state"].fill_(99)
            return {
                "actor": torch.tensor([[100.0], [200.0]]),
                "critic": torch.tensor([[300.0], [400.0]]),
            }, {}

    merged = pilot.reset_done_envs_preserving_unfinished(
        ResettingEnv(), observations, torch.tensor([0])
    )

    assert merged["actor"].tolist() == [[100.0], [20.0]]
    assert merged["critic"].tolist() == [[300.0], [40.0]]
    assert observations["actor"].tolist() == [[10.0], [20.0]]
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert cuda_rng["state"].tolist() == [7]
    assert restored_devices == [torch.device("cuda:0")]


def test_new_json_writer_refuses_existing_output_and_never_writes_partial(tmp_path):
    output = tmp_path / "fic.json"
    pilot._publish_new_json(output, {"z": [1, 2], "a": {"ok": True}})
    assert output.read_bytes() == b'{"a":{"ok":true},"z":[1,2]}\n'
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
