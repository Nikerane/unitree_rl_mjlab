"""Public contract tests for the matched 64-episode FIC pilot evaluator."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from evaluation.joint_position import evaluate_fic_pilot as pilot
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT


DIRECT_FIC0_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-Fixed"
)
DIRECT_FICTT_TASK = f"{DIRECT_FIC0_TASK}-TT"
BANKED_FIC0_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-"
    "Delivered4-JointPosition-Fixed"
)
DIRECT_OBSERVATIONS = (
    "joint_pos",
    "joint_vel",
    "ee_pos",
    "ee_vel",
    "head_pos",
    "head_vel",
    "nail_top_pos",
    "nail_depth",
    "strike_phase",
    "strike_ref_error",
    "actions",
)
IMITATION_STAGES = [
    {"step": 0, "weight": 0.10},
    {"step": 1200, "weight": 0.08},
    {"step": 2400, "weight": 0.06},
    {"step": 3600, "weight": 0.04},
    {"step": 4800, "weight": 0.02},
    {"step": 6000, "weight": 0.00},
]


def _live_configs(task=pilot.FIC0_TASK):
    return pilot.load_env_cfg(task, play=False), pilot.load_rl_cfg(task)


def test_public_evaluator_seams_are_exposed() -> None:
    assert callable(pilot.validate_fic_contract)
    assert callable(pilot.evaluate_checkpoint)


def test_evaluator_uses_the_authoritative_production_impulse_caps() -> None:
    caps = pilot.JOINT_IMPULSE_CAP_N_M_S
    assert isinstance(caps, tuple)
    assert caps is not IMP_J_LIMIT
    assert caps == tuple(IMP_J_LIMIT)
    assert caps == (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
    with pytest.raises(TypeError):
        caps[0] = 99.0


def test_evaluator_targets_only_direct_reference_fic_treatments() -> None:
    assert pilot.TASKS == (DIRECT_FIC0_TASK, DIRECT_FICTT_TASK)
    assert pilot.FIC0_TASK == DIRECT_FIC0_TASK
    assert pilot.FICTT_TASK == DIRECT_FICTT_TASK
    assert BANKED_FIC0_TASK not in pilot.TASKS


@pytest.mark.parametrize("task", pilot.TASKS)
def test_contract_admits_only_exact_fic_treatments(task):
    env_cfg, agent_cfg = _live_configs(task)
    contract = pilot.validate_fic_contract(task, env_cfg, agent_cfg)
    assert contract["delivered_impulse_i_ref_n_s"] == pilot.I_REF_N_S
    assert contract["observation_names"] == list(DIRECT_OBSERVATIONS)
    assert contract["observation_width"] == 40
    assert contract["r_imit_weight"] == 0.10
    assert contract["r_imit_sigma_m"] == 0.05
    assert contract["r_imit_curriculum"] == IMITATION_STAGES
    assert contract["r_tt_enabled"] is (task == pilot.FICTT_TASK)


def test_contract_rejects_cartesian_task_and_treatment_drift():
    env_cfg, agent_cfg = _live_configs()
    with pytest.raises(ValueError, match="unsupported"):
        pilot.validate_fic_contract("cartesian", env_cfg, agent_cfg)
    with pytest.raises(ValueError, match="unsupported"):
        pilot.validate_fic_contract(BANKED_FIC0_TASK, env_cfg, agent_cfg)
    drifted, agent_cfg = _live_configs()
    drifted.rewards["delivered_impulse"].params["i_ref"] = 0.3088
    with pytest.raises(ValueError, match="delivered_impulse contract"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, drifted, agent_cfg)
    tt, agent_cfg = _live_configs(pilot.FICTT_TASK)
    tt.rewards["r_tt"].params["k_tt"] = 2.0
    with pytest.raises(ValueError, match="r_tt contract"):
        pilot.validate_fic_contract(pilot.FICTT_TASK, tt, agent_cfg)


@pytest.mark.parametrize(
    "mutation",
    ("function", "weight", "i_ref", "eps", "saturate", "nail", "extra"),
)
def test_contract_rejects_delivered_impulse_scientific_drift(mutation):
    env_cfg, agent_cfg = _live_configs()
    reward = env_cfg.rewards["delivered_impulse"]
    if mutation == "function":
        reward.func = object
    elif mutation == "weight":
        reward.weight = 3.0
    elif mutation == "i_ref":
        reward.params["i_ref"] = 0.3088
    elif mutation == "eps":
        reward.params["eps"] = 0.1
    elif mutation == "saturate":
        reward.params["saturate"] = True
    elif mutation == "nail":
        reward.params["nail_cfg"].joint_names = ("other_joint",)
    else:
        reward.params["unexpected"] = 1.0

    with pytest.raises(ValueError, match="delivered_impulse contract"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_action_order_and_normalizer_drift():
    action_drift, agent = _live_configs()
    action_drift.actions["joint_position"].actuator_names = tuple(reversed(pilot.JOINT_NAMES))
    with pytest.raises(ValueError, match="joint action"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, action_drift, agent)
    env_cfg, agent = _live_configs()
    agent.actor.obs_normalization = False
    with pytest.raises(ValueError, match="normalization"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent)
    env_cfg, agent = _live_configs()
    agent.critic.obs_normalization = False
    with pytest.raises(ValueError, match="normalization"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent)


def test_contract_rejects_stock_ppo_that_would_bypass_soft_cat():
    env_cfg, agent_cfg = _live_configs()
    agent_cfg.algorithm.class_name = "PPO"

    with pytest.raises(ValueError, match="CatPPO"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize("mutation", ("function", "weight", "params"))
def test_contract_rejects_action_rate_scientific_drift(mutation):
    env_cfg, agent_cfg = _live_configs()
    action_rate = env_cfg.rewards["action_rate"]
    if mutation == "function":
        action_rate.func = object
    elif mutation == "weight":
        action_rate.weight = -0.02
    else:
        action_rate.params["scale"] = 1.0

    with pytest.raises(ValueError, match="action_rate"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize(
    "reward_name",
    (
        "approach",
        "nail_driven",
        "nail_depth_delta",
        "impact_progress",
        "completion",
        "joint_pos_limits",
    ),
)
@pytest.mark.parametrize("mutation", ("function", "weight", "params"))
def test_contract_rejects_baseline_reward_scientific_drift(reward_name, mutation):
    env_cfg, agent_cfg = _live_configs()
    reward = env_cfg.rewards[reward_name]
    if mutation == "function":
        reward.func = object
    elif mutation == "weight":
        reward.weight += 0.001
    else:
        reward.params["unexpected"] = 1.0

    with pytest.raises(ValueError, match=f"{reward_name} contract"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_non_timeout",
        "timeout_function",
        "timeout_flag",
        "timeout_params",
        "success_function",
        "success_flag",
        "success_depth",
        "success_nail",
    ),
)
def test_contract_rejects_termination_semantic_drift(mutation):
    env_cfg, agent_cfg = _live_configs()
    time_out = env_cfg.terminations["time_out"]
    nail_driven = env_cfg.terminations["nail_driven"]
    if mutation == "extra_non_timeout":
        env_cfg.terminations["failure"] = nail_driven
    elif mutation == "timeout_function":
        time_out.func = object
    elif mutation == "timeout_flag":
        time_out.time_out = False
    elif mutation == "timeout_params":
        time_out.params["limit"] = 1
    elif mutation == "success_function":
        nail_driven.func = object
    elif mutation == "success_flag":
        nail_driven.time_out = True
    elif mutation == "success_depth":
        nail_driven.params["success_depth"] = 0.031
    else:
        nail_driven.params["nail_cfg"].joint_names = ("other_joint",)

    with pytest.raises(ValueError, match="termination"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize("group", ("actor", "critic"))
def test_contract_rejects_direct_observation_name_or_order_drift(group):
    env_cfg, agent_cfg = _live_configs()
    moved = env_cfg.observations[group].terms.pop("actions")
    env_cfg.observations[group].terms = {
        "actions": moved,
        **env_cfg.observations[group].terms,
    }

    with pytest.raises(ValueError, match="observation"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_observation_group_and_waypoint_behavior_drift():
    env_cfg, agent_cfg = _live_configs()
    env_cfg.observations["privileged"] = env_cfg.observations["critic"]
    with pytest.raises(ValueError, match="observation"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)

    env_cfg, agent_cfg = _live_configs()
    env_cfg.metrics["waypoint_progress"] = env_cfg.metrics["first_strike"]
    with pytest.raises(ValueError, match="waypoint"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)

    env_cfg, agent_cfg = _live_configs()
    env_cfg.rewards["r_gate"] = env_cfg.rewards["approach"]
    with pytest.raises(ValueError, match="waypoint"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_fixed_joint_reset_drift():
    env_cfg, agent_cfg = _live_configs()
    env_cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)

    with pytest.raises(ValueError, match="reset"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize("mutation", ("function", "weight", "sigma", "sensor", "head", "nail"))
def test_contract_rejects_imitation_prior_drift(mutation):
    env_cfg, agent_cfg = _live_configs()
    term = env_cfg.rewards["r_imit"]
    if mutation == "function":
        term.func = object
    elif mutation == "weight":
        term.weight = 0.1000001
    elif mutation == "sigma":
        term.params["sigma"] = 0.0500001
    elif mutation == "sensor":
        term.params["sensor_name"] = "other_contact"
    elif mutation == "head":
        term.params["robot_cfg"].site_names = ("other_head",)
    else:
        term.params["nail_cfg"].site_names = ("other_nail",)

    with pytest.raises(ValueError, match="r_imit"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize("mutation", ("function", "name", "stages", "extra"))
def test_contract_rejects_imitation_curriculum_drift(mutation):
    env_cfg, agent_cfg = _live_configs()
    term = env_cfg.curriculum["r_imit_anneal"]
    if mutation == "function":
        term.func = object
    elif mutation == "name":
        term.params["reward_name"] = "other"
    elif mutation == "stages":
        term.params["stages"][1]["weight"] = 0.0800001
    else:
        env_cfg.curriculum["extra"] = term

    with pytest.raises(ValueError, match="curriculum"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_live_observation_contract_requires_exact_width_40() -> None:
    manager = SimpleNamespace(
        active_terms={
            "actor": list(DIRECT_OBSERVATIONS),
            "critic": list(DIRECT_OBSERVATIONS),
        },
        group_obs_dim={"actor": (40,), "critic": (40,)},
    )
    pilot._validate_live_observation_contract(SimpleNamespace(observation_manager=manager))
    manager.group_obs_dim["critic"] = (41,)
    with pytest.raises(RuntimeError, match="width"):
        pilot._validate_live_observation_contract(SimpleNamespace(observation_manager=manager))


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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("delay_min_lag", 1),
        ("frictionloss", 0.01),
        ("viscous_damping", 0.01),
        ("transmission_type", "joint"),
    ),
)
def test_contract_rejects_actuator_delay_and_transmission_drift(field, value):
    env_cfg, agent_cfg = _live_configs()
    actuator = env_cfg.scene.entities["robot"].articulation.actuators[0]
    setattr(actuator, field, value)

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


@pytest.mark.parametrize("mutation", ("cfg_class", "transmission"))
def test_contract_rejects_joint_action_type_drift(mutation):
    env_cfg, agent_cfg = _live_configs()
    action = env_cfg.actions["joint_position"]
    if mutation == "cfg_class":
        env_cfg.actions["joint_position"] = SimpleNamespace(**vars(action))
    else:
        action.transmission_type = "joint"

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


@pytest.mark.parametrize("mutation", ("per_substep", "reduce_value", "reduce_type"))
def test_contract_rejects_cat_soft_scheduling_drift(mutation):
    env_cfg, agent_cfg = _live_configs()
    cat = env_cfg.metrics["cat_soft"]
    if mutation == "per_substep":
        cat.per_substep = True
    elif mutation == "reduce_value":
        cat.reduce = "last"
    else:
        class MeanString(str):
            pass

        cat.reduce = MeanString("mean")

    with pytest.raises(ValueError, match="CaT scheduling"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize("mutation", ("name", "joints", "preserve_order"))
def test_contract_rejects_cat_soft_robot_selection_drift(mutation):
    env_cfg, agent_cfg = _live_configs()
    robot_cfg = env_cfg.metrics["cat_soft"].params["robot_cfg"]
    if mutation == "name":
        robot_cfg.name = "nail_block"
    elif mutation == "joints":
        robot_cfg.joint_names = pilot.JOINT_NAMES[:-1]
    else:
        robot_cfg.preserve_order = True

    with pytest.raises(ValueError, match="CaT contract"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


def test_contract_rejects_unknown_velocity_cat_hook_parameter():
    env_cfg, agent_cfg = _live_configs()
    env_cfg.metrics["cat_soft"].params["contact_only"] = True

    with pytest.raises(ValueError, match="CaT contract"):
        pilot.validate_fic_contract(pilot.FIC0_TASK, env_cfg, agent_cfg)


@pytest.mark.parametrize(
    ("name", "mutation"),
    (
        ("substep_peak_qv", "per_substep"),
        ("substep_peak_qv", "function"),
        ("substep_peak_qv", "reduce"),
        ("substep_peak_qv", "params"),
        ("substep_impulse", "per_substep"),
        ("substep_impulse", "function"),
        ("substep_impulse", "reduce"),
        ("substep_impulse", "sensor"),
        ("substep_impulse", "joints"),
        ("substep_impulse", "baseline"),
        ("substep_impulse", "window"),
        ("first_strike", "per_substep"),
        ("first_strike", "function"),
        ("first_strike", "reduce"),
        ("first_strike", "contact_sensor"),
        ("first_strike", "impulse_sensor"),
        ("first_strike", "head"),
        ("first_strike", "nail_joint"),
        ("first_strike", "nail_site"),
        ("first_strike", "axis"),
        ("first_strike", "window"),
        ("first_strike", "progress_eps"),
    ),
)
def test_contract_rejects_endpoint_producer_metric_drift(name, mutation):
    env_cfg, agent_cfg = _live_configs()
    metric = env_cfg.metrics[name]
    if mutation == "per_substep":
        metric.per_substep = False
    elif mutation == "function":
        metric.func = object
    elif mutation == "reduce":
        metric.reduce = "last" if name == "substep_peak_qv" else "mean"
    elif mutation == "params":
        metric.params["unexpected"] = True
    elif mutation in ("sensor", "contact_sensor"):
        key = "sensor_name" if name == "substep_impulse" else "contact_sensor_name"
        metric.params[key] = "other_contact"
    elif mutation == "impulse_sensor":
        metric.params["impulse_sensor_name"] = "other_impulse"
    elif mutation == "joints":
        metric.params["robot_cfg"].joint_names = pilot.JOINT_NAMES[:-1]
    elif mutation == "baseline":
        metric.params["subtract_baseline"] = False
    elif mutation == "head":
        metric.params["robot_cfg"].site_names = ("other_head",)
    elif mutation == "nail_joint":
        metric.params["nail_cfg"].joint_names = ("other_joint",)
    elif mutation == "nail_site":
        metric.params["nail_cfg"].site_names = ("other_site",)
    elif mutation == "axis":
        metric.params["axis"] = (0.0, 0.0, 1.0)
    elif mutation == "progress_eps":
        metric.params["progress_eps"] = 0.001
    else:
        key = "event_window_substeps" if name == "substep_impulse" else "window_substeps"
        metric.params[key] = 24

    with pytest.raises(ValueError, match="producer"):
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
            str(tmp_path / "model_499.pt"),
            "--output",
            str(tmp_path / "result.json"),
            "--device",
            "cuda:0",
            "--training-seed",
            "3",
        ]
    )
    assert args.task == pilot.FIC0_TASK
    assert args.device == "cuda:0"
    assert args.training_seed == 3
    with pytest.raises(SystemExit):
        pilot._parse_args(
            [
                pilot.FIC0_TASK,
                "--checkpoint",
                str(tmp_path / "model_499.pt"),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
    with pytest.raises(SystemExit):
        pilot._parse_args(
            [
                pilot.FIC0_TASK,
                "--checkpoint",
                str(tmp_path / "model_499.pt"),
                "--output",
                str(tmp_path / "result.json"),
                "--training-seed",
                "1",
            ]
        )
    with pytest.raises(SystemExit):
        pilot._parse_args(["--task", pilot.FIC0_TASK])


def test_evaluator_protocol_constants_are_frozen():
    assert pilot.SEED == 2026081202
    assert pilot.NUM_ENVS == 64
    assert pilot.EPISODE_LENGTH_S == 4.0


def _accumulate(
    acc,
    qvel_peak,
    target_error,
    reference_error,
    reference_eligible,
):
    pilot._accumulate_episode_measurements(
        acc,
        qvel_peak=torch.tensor(qvel_peak),
        target_error=torch.tensor(target_error),
        reference_error=torch.tensor(reference_error),
        reference_eligible=torch.tensor(reference_eligible),
    )


def test_episode_measurements_hold_500hz_per_joint_peaks_and_legality_boundary():
    acc = pilot._new_episode_accumulators(2, torch.device("cpu"))
    _accumulate(
        acc,
        [[3.1415, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1] * 6],
        [[0.0] * 6, [0.0] * 6],
        [0.01, 0.02],
        [True, True],
    )
    above = torch.nextafter(torch.tensor(3.1415), torch.tensor(float("inf"))).item()
    _accumulate(
        acc,
        [[0.1, 2.0, 0.2, 1.0, 0.1, 0.2], [above, 0.2, 0.3, 0.4, 0.5, 0.6]],
        [[0.0] * 6, [0.0] * 6],
        [0.03, 0.04],
        [True, True],
    )

    legal = pilot._finalize_episode_measurements(acc, 0, 2)
    illegal = pilot._finalize_episode_measurements(acc, 1, 2)
    assert legal["joint_velocity_peak_rad_s"] == pytest.approx(
        [3.1415, 2.0, 0.3, 1.0, 0.5, 0.6]
    )
    assert legal["joint_velocity_legal"] is True
    assert illegal["joint_velocity_legal"] is False


@pytest.mark.parametrize("task", (DIRECT_FIC0_TASK, DIRECT_FICTT_TASK))
def test_target_error_reduces_over_time_and_six_joint_axes_for_both_arms(task):
    assert task in pilot.TASKS
    acc = pilot._new_episode_accumulators(2, torch.device("cpu"))
    _accumulate(
        acc,
        [[0.0] * 6, [0.0] * 6],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0] * 6],
        [0.01, 0.02],
        [True, True],
    )
    _accumulate(
        acc,
        [[0.0] * 6, [0.0] * 6],
        [[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0, 8.0]],
        [0.03, 0.04],
        [True, True],
    )

    first = pilot._finalize_episode_measurements(acc, 0, 2)
    second = pilot._finalize_episode_measurements(acc, 1, 2)
    assert first["joint_target_rmse_rad"] == pytest.approx((182.0 / 12.0) ** 0.5)
    assert first["joint_target_error_max_rad"] == 6.0
    assert second["joint_target_rmse_rad"] == pytest.approx((108.0 / 12.0) ** 0.5)
    assert second["joint_target_error_max_rad"] == 8.0


def test_reference_samples_follow_live_imitation_contact_latch() -> None:
    num_envs = 2
    robot = SimpleNamespace(
        data=SimpleNamespace(site_pos_w=torch.zeros(num_envs, 1, 3))
    )
    nail = SimpleNamespace(
        data=SimpleNamespace(site_pos_w=torch.zeros(num_envs, 1, 3))
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            found=torch.tensor([[1.0], [0.0]]),
            current_contact_time=torch.zeros(num_envs, 1),
            last_contact_time=torch.tensor([[0.0], [0.004]]),
        )
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={
            "robot": robot,
            "nail_block": nail,
            "hammer_nail_contact": sensor,
        },
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )
    term = pilot.ImitationPriorTerm(cfg=None, env=env)
    term(
        env,
        sensor_name="hammer_nail_contact",
        robot_cfg=SimpleNamespace(name="robot", site_ids=[0]),
        nail_cfg=SimpleNamespace(name="nail_block", site_ids=[0]),
        sigma=0.05,
    )
    assert term._contacted.tolist() == [True, True]

    acc = pilot._new_episode_accumulators(num_envs, torch.device("cpu"))
    _accumulate(
        acc,
        [[0.0] * 6, [0.0] * 6],
        [[0.0] * 6, [0.0] * 6],
        [0.25, 0.5],
        (~term._contacted).tolist(),
    )
    assert acc.reference_sum.tolist() == [0.0, 0.0]
    assert acc.reference_max.tolist() == [0.0, 0.0]
    assert acc.reference_samples.tolist() == [0, 0]
    with pytest.raises(RuntimeError, match="reference samples"):
        pilot._finalize_episode_measurements(acc, 0, 1)


def test_reset_episode_accumulators_clears_only_done_local_slices():
    acc = pilot._new_episode_accumulators(2, torch.device("cpu"))
    _accumulate(
        acc,
        [[1.0] * 6, [2.0] * 6],
        [[3.0] * 6, [4.0] * 6],
        [0.1, 0.2],
        [True, True],
    )

    pilot._reset_episode_accumulators(acc, torch.tensor([0]))

    assert acc.qvel_peak[0].tolist() == [0.0] * 6
    assert acc.target_samples.tolist() == [0, 1]
    assert acc.reference_samples.tolist() == [0, 1]
    assert acc.qvel_peak[1].tolist() == [2.0] * 6


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
    measurements = pilot._new_episode_accumulators(2, torch.device("cpu"))
    measurements.qvel_peak[:] = torch.tensor([[1.0] * 6, [2.0] * 6])
    measurements.target_sq_sum[:] = torch.tensor([54.0, 240.0])
    measurements.target_abs_max[:] = torch.tensor([0.5, 0.75])
    measurements.target_samples[:] = torch.tensor([9, 10])
    measurements.reference_sum[:] = torch.tensor([0.9, 2.0])
    measurements.reference_max[:] = torch.tensor([0.2, 0.4])
    measurements.reference_samples[:] = torch.tensor([9, 10])
    ids = pilot.capture_first_terminals(
        records=records,
        done=done,
        nail_driven=torch.tensor([True, False]),
        timed_out=torch.tensor([False, True]),
        steps=torch.tensor([9, 10]),
        tracker=tracker,
        nail_depth=torch.tensor([0.032, 0.01]),
        joint_impulse_peak=torch.ones(2, 6),
        episode_accumulators=measurements,
    )
    tracker.delivered.zero_()
    pilot.capture_first_terminals(
        records=records,
        done=done,
        nail_driven=torch.tensor([True, False]),
        timed_out=torch.tensor([False, True]),
        steps=torch.tensor([99, 99]),
        tracker=tracker,
        nail_depth=torch.zeros(2),
        joint_impulse_peak=torch.zeros(2, 6),
        episode_accumulators=measurements,
    )
    assert ids == (0, 1)
    assert [records[index]["first_event_impulse_n_s"] for index in (0, 1)] == [0.25, 0.5]
    assert [records[index]["episode_steps"] for index in (0, 1)] == [9, 10]
    assert records[0]["joint_velocity_peak_rad_s"] == [1.0] * 6
    assert records[1]["joint_target_rmse_rad"] == pytest.approx(2.0)
    assert records[0]["reference_error_mean_m"] == pytest.approx(0.1)
    assert records[0]["joint_impulse_utilization"] == pytest.approx(
        [1.0 / 1.64, 1.0 / 3.28, 1.0 / 1.64, 1.0 / 1.64, 1.0 / 1.64, 1.0 / 1.64]
    )


def test_capture_does_not_label_an_aggregate_failure_termination_as_success():
    tracker = SimpleNamespace(
        started=torch.tensor([False]),
        finalized=torch.tensor([False]),
        productive=torch.tensor([False]),
        reason=torch.tensor([0]),
        v_precontact=torch.tensor([0.0]),
        delivered=torch.tensor([0.0]),
    )
    measurements = pilot._new_episode_accumulators(1, torch.device("cpu"))
    measurements.target_samples[:] = 1
    measurements.reference_samples[:] = 1
    aggregate_non_timeout = torch.tensor([True])

    records = {}
    pilot.capture_first_terminals(
        records=records,
        done=aggregate_non_timeout,
        nail_driven=torch.tensor([False]),
        timed_out=torch.tensor([False]),
        steps=torch.tensor([1]),
        tracker=tracker,
        nail_depth=torch.tensor([0.0]),
        joint_impulse_peak=torch.zeros(1, 6),
        episode_accumulators=measurements,
    )

    assert records[0]["success"] is False


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


def _summary_row(*, qvel, target_rmse, target_max, reference_mean, reference_max, impulse):
    caps = pilot.JOINT_IMPULSE_CAP_N_M_S
    return {
        "success": True,
        "first_strike_productive": True,
        "first_event_impulse_n_s": 0.25,
        "joint_impulse_peak_n_m_s": list(impulse),
        "joint_impulse_utilization": [
            value / cap for value, cap in zip(impulse, caps, strict=True)
        ],
        "joint_velocity_peak_rad_s": list(qvel),
        "joint_velocity_legal": all(
            value <= pilot.JOINT_VELOCITY_LIMIT_RAD_S for value in qvel
        ),
        "joint_target_rmse_rad": target_rmse,
        "joint_target_error_max_rad": target_max,
        "reference_error_mean_m": reference_mean,
        "reference_error_max_m": reference_max,
    }


def test_schema2_summary_reports_exact_axes_quantiles_and_inclusive_caps():
    caps = pilot.JOINT_IMPULSE_CAP_N_M_S
    rows = [
        _summary_row(
            qvel=[3.1415, 1.0, 1.0, 1.0, 1.0, 1.0],
            target_rmse=0.1,
            target_max=0.2,
            reference_mean=0.01,
            reference_max=0.02,
            impulse=caps,
        ),
        _summary_row(
            qvel=[3.1416, 2.0, 2.0, 2.0, 2.0, 2.0],
            target_rmse=0.3,
            target_max=0.5,
            reference_mean=0.03,
            reference_max=0.05,
            impulse=(*caps[:-1], 1.65),
        ),
    ]

    summary = pilot._summary(rows)

    assert summary["joint_velocity_peak_rad_s"]["p95"] == pytest.approx(
        [3.141595, 1.95, 1.95, 1.95, 1.95, 1.95]
    )
    assert summary["joint_velocity_peak_rad_s"]["max"] == pytest.approx(
        [3.1416, 2.0, 2.0, 2.0, 2.0, 2.0]
    )
    assert summary["joint_velocity_peak_rad_s"]["all_joints_legal_n"] == 1
    assert summary["joint_velocity_peak_rad_s"]["all_joints_legal_rate"] == 0.5
    assert summary["joint_target_rmse_rad"] == pytest.approx(
        {"mean": 0.2, "p95": 0.29, "max": 0.3}
    )
    assert summary["joint_target_error_max_rad"] == pytest.approx(
        {"mean": 0.35, "p95": 0.485, "max": 0.5}
    )
    assert summary["reference_error_mean_m"] == pytest.approx(
        {"mean": 0.02, "p95": 0.029, "max": 0.03}
    )
    assert summary["reference_error_max_m"] == pytest.approx(
        {"mean": 0.035, "p95": 0.0485, "max": 0.05}
    )
    assert summary["joint_impulse_utilization"]["p95"][-1] == pytest.approx(
        0.95 * (1.65 / 1.64) + 0.05
    )
    assert summary["joint_impulse_utilization"]["max"][-1] == pytest.approx(
        1.65 / 1.64
    )
    assert summary["joint_impulse_utilization"]["all_joints_at_or_below_cap_n"] == 1
    assert summary["joint_impulse_utilization"]["all_joints_at_or_below_cap_rate"] == 0.5


def test_schema2_summary_rejects_nonfinite_endpoint_values():
    row = _summary_row(
        qvel=[float("nan"), 1.0, 1.0, 1.0, 1.0, 1.0],
        target_rmse=0.1,
        target_max=0.2,
        reference_mean=0.01,
        reference_max=0.02,
        impulse=pilot.JOINT_IMPULSE_CAP_N_M_S,
    )

    with pytest.raises(RuntimeError, match="finite"):
        pilot._summary([row])


def test_incomplete_population_is_rejected_before_json_can_be_published():
    with pytest.raises(RuntimeError, match="incomplete"):
        pilot._ordered_complete_records({0: {"env_id": 0}})


def test_missing_checkpoint_fails_before_loading_any_live_configuration(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        pilot.evaluate_checkpoint(
            pilot.FIC0_TASK,
            tmp_path / "model_499.pt",
            tmp_path / "out.json",
            training_seed=2,
        )


@pytest.mark.parametrize("training_seed", (True, 1, 4.0, 5))
def test_callable_rejects_nonqualified_training_seed_before_live_config(
    tmp_path, training_seed
):
    with pytest.raises(ValueError, match="training seed"):
        pilot.evaluate_checkpoint(
            pilot.FIC0_TASK,
            tmp_path / "model_499.pt",
            tmp_path / "out.json",
            training_seed=training_seed,
        )


def test_callable_rejects_nonfinal_checkpoint_basename_before_live_config(tmp_path):
    checkpoint = tmp_path / "model_498.pt"
    checkpoint.write_bytes(b"not loaded")

    with pytest.raises(ValueError, match="model_499.pt"):
        pilot.evaluate_checkpoint(
            pilot.FIC0_TASK,
            checkpoint,
            tmp_path / "out.json",
            training_seed=2,
        )


def test_existing_output_is_refused_before_loading_any_live_configuration(tmp_path):
    checkpoint = tmp_path / "model_499.pt"
    checkpoint.write_bytes(b"not loaded")
    output = tmp_path / "out.json"
    output.write_text("already present")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pilot.evaluate_checkpoint(
            pilot.FIC0_TASK, checkpoint, output, training_seed=2
        )
