"""Contracts for the native Z1 variable-impedance action."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from src.tasks.hammer.mdp.variable_impedance import (
    JointStiffnessActionCfg,
    JointStiffnessTelemetry,
    VariableImpedanceGains,
    expand_variable_impedance_model_fields,
    variable_impedance_gains,
)


JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def test_author_v1_gain_map_clips_and_matches_known_values() -> None:
    p = torch.tensor(
        [[-2.0, -1.0, -0.5, 0.0, 1.0, 2.0]], dtype=torch.float64
    )
    nominal_kp = torch.tensor(
        [100.0, 200.0, 300.0, 400.0, 500.0, 600.0], dtype=torch.float64
    )
    nominal_kd = torch.tensor(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=torch.float64
    )

    gains = variable_impedance_gains(p, nominal_kp, nominal_kd, C=4.0)

    assert isinstance(gains, VariableImpedanceGains)
    assert gains.p.dtype == p.dtype
    assert gains.p.device == p.device
    torch.testing.assert_close(
        gains.p,
        torch.tensor(
            [[-1.0, -1.0, -0.5, 0.0, 1.0, 1.0]], dtype=torch.float64
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        gains.multiplier,
        torch.tensor([[0.25, 0.25, 0.5, 1.0, 4.0, 4.0]], dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )
    torch.testing.assert_close(
        gains.kp,
        torch.tensor(
            [[25.0, 50.0, 150.0, 400.0, 2000.0, 2400.0]],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        gains.kd,
        torch.tensor(
            [[5.0, 10.0, 15.0 * 2**0.5, 40.0, 100.0, 120.0]],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_gain_map_is_monotone_has_nominal_parity_and_is_frozen() -> None:
    p = torch.tensor([[-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]])
    nominal_kp = torch.tensor([1000.0, 1500.0, 1000.0, 1000.0, 1000.0, 1000.0])
    nominal_kd = torch.tensor([100.0, 150.0, 100.0, 100.0, 100.0, 100.0])

    gains = variable_impedance_gains(p, nominal_kp, nominal_kd, C=1.25)

    assert gains.p.shape == p.shape
    assert torch.all(gains.multiplier[:, 1:] > gains.multiplier[:, :-1])
    assert gains.multiplier[0, 2].item() == pytest.approx(1.0)
    assert gains.kp[0, 2].item() == pytest.approx(nominal_kp[2].item())
    assert gains.kd[0, 2].item() == pytest.approx(nominal_kd[2].item())
    with pytest.raises(FrozenInstanceError):
        gains.kp = torch.zeros_like(gains.kp)  # type: ignore[misc]


@pytest.mark.parametrize("C", [1.0, 0.0, -2.0, math.inf, -math.inf, math.nan])
def test_gain_map_rejects_invalid_base(C: float) -> None:
    p = torch.zeros((2, 6))
    nominal = torch.ones(6)
    with pytest.raises(ValueError, match="C"):
        variable_impedance_gains(p, nominal, nominal, C=C)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("p", math.nan),
        ("p", math.inf),
        ("nominal_kp", math.nan),
        ("nominal_kp", 0.0),
        ("nominal_kd", math.inf),
        ("nominal_kd", -1.0),
    ],
)
def test_gain_map_rejects_nonfinite_or_nonpositive_inputs(
    field: str, bad_value: float
) -> None:
    tensors = {
        "p": torch.zeros((2, 6)),
        "nominal_kp": torch.ones(6),
        "nominal_kd": torch.ones(6),
    }
    tensors[field].flatten()[0] = bad_value
    with pytest.raises(ValueError, match=field):
        variable_impedance_gains(**tensors, C=1.25)


def test_gain_map_rejects_shape_dtype_and_device_contract_violations() -> None:
    p = torch.zeros((2, 6))
    nominal = torch.ones(6)

    with pytest.raises(ValueError, match="shape"):
        variable_impedance_gains(p[:, :5], nominal, nominal, C=1.25)
    with pytest.raises(ValueError, match="one-dimensional"):
        variable_impedance_gains(p, nominal.unsqueeze(0), nominal, C=1.25)
    with pytest.raises(ValueError, match="dtype"):
        variable_impedance_gains(p.double(), nominal, nominal, C=1.25)
    with pytest.raises(ValueError, match="floating"):
        variable_impedance_gains(p.long(), nominal.long(), nominal.long(), C=2.0)

    if torch.backends.mps.is_available():
        with pytest.raises(ValueError, match="device"):
            variable_impedance_gains(p.to("mps"), nominal, nominal, C=1.25)


def test_gain_map_rejects_nonfinite_outputs() -> None:
    p = torch.ones((1, 6), dtype=torch.float32)
    nominal = torch.full((6,), torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match="finite output"):
        variable_impedance_gains(p, nominal, nominal, C=2.0)


@pytest.fixture(scope="module")
def live_stiffness_env():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg

    from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg

    cfg = z1_hammer_env_cfg(play=True)
    cfg.scene.num_envs = 2
    cfg.events["expand_variable_impedance_model_fields"] = EventTermCfg(
        func=expand_variable_impedance_model_fields,
        mode="startup",
    )
    cfg.actions = {
        "joint_stiffness": JointStiffnessActionCfg(
            entity_name="robot",
            joint_names=JOINT_NAMES,
            C=1.25,
        )
    }
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        yield env
    finally:
        env.close()


@pytest.mark.integration
def test_live_action_resolves_canonical_native_control_contract(
    live_stiffness_env,
) -> None:
    env = live_stiffness_env
    term = env.action_manager.get_term("joint_stiffness")

    assert expand_variable_impedance_model_fields.model_fields == (
        "actuator_gainprm",
        "actuator_biasprm",
    )
    assert {"actuator_gainprm", "actuator_biasprm"}.issubset(
        env.sim.expanded_fields
    )
    assert env.sim.model.actuator_gainprm.stride(0) != 0
    assert env.sim.model.actuator_biasprm.stride(0) != 0
    assert term.action_dim == 6
    assert tuple(term.joint_names) == JOINT_NAMES
    assert tuple(term.control_ids.tolist()) == (0, 5, 1, 2, 3, 4)

    telemetry = term.telemetry
    assert isinstance(telemetry, JointStiffnessTelemetry)
    assert telemetry.mapping_family == "author_v1_exponential"
    assert telemetry.C == pytest.approx(1.25)
    assert telemetry.p_bounds == (-1.0, 1.0)
    assert telemetry.joint_names == JOINT_NAMES
    torch.testing.assert_close(
        telemetry.nominal_kp,
        torch.tensor([1000.0, 1500.0, 1000.0, 1000.0, 1000.0, 1000.0]),
    )
    torch.testing.assert_close(
        telemetry.nominal_kd,
        torch.tensor([100.0, 150.0, 100.0, 100.0, 100.0, 100.0]),
    )


@pytest.mark.integration
def test_live_action_writes_all_native_pd_fields_without_touching_limits_or_gripper(
    live_stiffness_env,
) -> None:
    env = live_stiffness_env
    term = env.action_manager.get_term("joint_stiffness")
    term.reset()

    gain_before = env.sim.model.actuator_gainprm.clone()
    bias_before = env.sim.model.actuator_biasprm.clone()
    force_before = env.sim.model.actuator_forcerange.clone()
    force_limited_before = env.sim.model.actuator_forcelimited.clone()

    term.process_actions(
        torch.tensor(
            [[-1.0] * 6, [1.0] * 6],
            dtype=term.nominal_kp.dtype,
            device=env.device,
        )
    )
    term.apply_actions()

    expected_kp = torch.stack((term.nominal_kp / 1.25, term.nominal_kp * 1.25))
    expected_kd = torch.stack(
        (term.nominal_kd / math.sqrt(1.25), term.nominal_kd * math.sqrt(1.25))
    )
    ctrl = term.control_ids
    expected_gain = gain_before.clone()
    expected_bias = bias_before.clone()
    expected_gain[:, ctrl, 0] = expected_kp
    expected_bias[:, ctrl, 1] = -expected_kp
    expected_bias[:, ctrl, 2] = -expected_kd
    torch.testing.assert_close(
        env.sim.model.actuator_gainprm[:], expected_gain, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        env.sim.model.actuator_biasprm[:], expected_bias, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        env.sim.model.actuator_forcerange[:],
        force_before,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        env.sim.model.actuator_forcelimited[:],
        force_limited_before,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.integration
def test_live_action_partial_reset_restores_only_selected_world(
    live_stiffness_env,
) -> None:
    env = live_stiffness_env
    term = env.action_manager.get_term("joint_stiffness")
    term.process_actions(torch.tensor([[-1.0] * 6, [1.0] * 6], device=env.device))
    term.apply_actions()

    term.reset(torch.tensor([0], device=env.device))

    ctrl = term.control_ids
    torch.testing.assert_close(
        env.sim.model.actuator_gainprm[0, ctrl, 0], term.nominal_kp
    )
    torch.testing.assert_close(
        env.sim.model.actuator_biasprm[0, ctrl, 1], -term.nominal_kp
    )
    torch.testing.assert_close(
        env.sim.model.actuator_biasprm[0, ctrl, 2], -term.nominal_kd
    )
    torch.testing.assert_close(
        env.sim.model.actuator_gainprm[1, ctrl, 0], term.nominal_kp * 1.25
    )
    torch.testing.assert_close(term.raw_action[0], torch.zeros(6))
    torch.testing.assert_close(term.raw_action[1], torch.ones(6))
    torch.testing.assert_close(term.telemetry.p[0], torch.zeros(6))
    torch.testing.assert_close(term.telemetry.multiplier[0], torch.ones(6))


@pytest.mark.integration
def test_live_action_fails_closed_without_independent_model_fields(
    live_stiffness_env,
) -> None:
    env = live_stiffness_env
    fields = env.sim.expanded_fields
    fields.remove("actuator_gainprm")
    try:
        with pytest.raises(ValueError, match="expanded per world"):
            JointStiffnessActionCfg(
                entity_name="robot", joint_names=JOINT_NAMES, C=1.25
            ).build(env)
    finally:
        fields.add("actuator_gainprm")


@pytest.mark.integration
@pytest.mark.parametrize(
    "joint_names",
    (
        tuple(reversed(JOINT_NAMES)),
        JOINT_NAMES[:-1],
        ("jointGripper",),
        (*JOINT_NAMES, "jointGripper"),
        (*JOINT_NAMES[:-1], "missing_joint"),
    ),
    ids=("reversed", "subset", "gripper", "superset", "unknown"),
)
def test_live_action_rejects_every_noncanonical_z1_joint_contract(
    live_stiffness_env, joint_names: tuple[str, ...]
) -> None:
    env = live_stiffness_env
    with pytest.raises(ValueError, match="canonical Z1 arm"):
        JointStiffnessActionCfg(
            entity_name="robot",
            joint_names=joint_names,
            C=1.25,
        ).build(env)


@pytest.mark.integration
def test_live_action_rejects_invalid_policy_sample(live_stiffness_env) -> None:
    env = live_stiffness_env
    term = env.action_manager.get_term("joint_stiffness")
    with pytest.raises(ValueError, match="shape"):
        term.process_actions(torch.zeros((env.num_envs, 5), device=env.device))


@pytest.mark.integration
def test_live_policy_step_map_extracts_no_tensor_scalars(live_stiffness_env) -> None:
    from torch.profiler import ProfilerActivity, profile

    term = live_stiffness_env.action_manager.get_term("joint_stiffness")
    action = torch.zeros(
        (live_stiffness_env.num_envs, term.action_dim), device=live_stiffness_env.device
    )
    with profile(activities=[ProfilerActivity.CPU]) as trace:
        term.process_actions(action)

    scalar_ops = {
        event.key
        for event in trace.key_averages()
        if event.key in {"aten::item", "aten::_local_scalar_dense"}
    }
    assert scalar_ops == set()


@pytest.mark.integration
def test_single_world_action_accepts_declared_expansion_with_shared_stride() -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg

    from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg

    cfg = z1_hammer_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.events["expand_variable_impedance_model_fields"] = EventTermCfg(
        func=expand_variable_impedance_model_fields,
        mode="startup",
    )
    cfg.actions = {
        "joint_stiffness": JointStiffnessActionCfg(
            entity_name="robot", joint_names=JOINT_NAMES, C=1.25
        )
    }
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        assert env.action_manager.get_term("joint_stiffness").action_dim == 6
        assert env.sim.expanded_fields.issuperset(
            {"actuator_gainprm", "actuator_biasprm"}
        )
    finally:
        env.close()
