"""The experimental controller must hold torque, not resample hidden native PD."""
import importlib.util

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata, validate_impact_speed_metadata


def subject():
    name = "src.tasks.hammer.mdp.sampled_pd500"
    assert importlib.util.find_spec(name) is not None, "sampled controller integration missing"
    return __import__(name, fromlist=["configure_sampled_pd500"])


def config():
    cfg = _impact_speed_env_cfg(play=True)
    cfg.scene.num_envs = 2
    cfg.actions["strike_speed"].bounds = (1.5, 2.2)
    cfg.actions["strike_speed"].mode = "external"
    cfg.rewards["speed_goal"].weight = -0.8493676
    return cfg


def _variable_controller():
    api = subject()
    cfg = config()
    api.configure_sampled_pd500(cfg, gain_mode="variable")
    env = ManagerBasedRlEnv(cfg, device="cpu")
    return env, api.install_sampled_pd500(env, gain_mode="variable")


def test_live_only_effort_limit_changes_real_force_and_is_rejected():
    env, controller = _variable_controller()
    try:
        env.reset()
        j1 = int(controller.ids[0])
        host = env.sim.mj_model.actuator_forcerange.copy()
        env.sim.model.actuator_forcerange[0, j1, 1] = 1.
        env.sim.data.qpos[:, controller.qids] = 0.
        env.sim.data.qvel[:, controller.vids] = 0.
        env.sim.data.ctrl[:] = .01
        env.sim.step()
        assert (env.sim.mj_model.actuator_forcerange == host).all()
        torch.testing.assert_close(controller.torque[:, j1], torch.full((2,), 2.56))
        torch.testing.assert_close(env.sim.data.actuator_force[:, j1], torch.ones(2))
        with pytest.raises(ValueError, match="runtime.*actuator_forcerange"):
            build_impact_speed_metadata(env, 1.)
    finally:
        env.close()


@pytest.mark.parametrize("field,index,value", [
    ("actuator_gear", (0, 0, 0), 2.),
    ("actuator_trnid", (0, 0), 1),
    ("actuator_trntype", (0,), 1),
    ("actuator_dyntype", (0,), 1),
    ("actuator_gaintype", (0,), 1),
    ("actuator_biastype", (0,), 0),
    ("actuator_forcelimited", (0,), 0),
    ("jnt_qposadr", (0,), 1),
    ("jnt_dofadr", (0,), 1),
])
def test_live_only_actuator_contract_drift_is_rejected(field, index, value):
    env, _ = _variable_controller()
    try:
        host = getattr(env.sim.mj_model, field).copy()
        getattr(env.sim.model, field)[index] = value
        assert (getattr(env.sim.mj_model, field) == host).all()
        with pytest.raises(ValueError, match="runtime.*" + field):
            build_impact_speed_metadata(env, 1.)
    finally:
        env.close()


@pytest.mark.parametrize("field", ["actuator_gear", "actuator_forcerange", "dof_damping"])
def test_live_heterogeneous_mechanics_and_damping_rejected(field):
    env, _ = _variable_controller()
    try:
        env.sim.expand_model_fields((field,))
        values = getattr(env.sim.model, field)
        values[1].view(-1)[0] += 1
        with pytest.raises(ValueError, match="heterogeneous"):
            build_impact_speed_metadata(env, 1.)
    finally:
        env.close()


def test_live_only_homogeneous_damping_changes_checkpoint_identity():
    env, _ = _variable_controller()
    try:
        expected = build_impact_speed_metadata(env, 1.)
        host = env.sim.mj_model.dof_damping.copy()
        env.sim.model.dof_damping[0, 0] += .1
        assert (env.sim.mj_model.dof_damping == host).all()
        changed = build_impact_speed_metadata(env, 1.)
        with pytest.raises(ValueError, match="mismatch"):
            validate_impact_speed_metadata(expected, changed)
    finally:
        env.close()


def test_variable_sample_maps_each_joint_and_environment_into_applied_torque():
    env, controller = _variable_controller()
    try:
        env.reset()
        gain = env.action_manager.get_term("joint_stiffness")
        p = torch.tensor([[-1., 0., 1., -.5, .5, 0.], [1., -1., 0., .5, -.5, 1.]])
        gain.process_actions(p)
        env.sim.data.qpos[:, controller.qids] = 0.
        env.sim.data.qvel[:, controller.vids] = .25
        env.sim.data.ctrl[:] = 0.
        env.sim.data.ctrl[:, controller.ids] = .02
        position_error = env.sim.data.ctrl[:, controller.ids] - env.sim.data.qpos[:, controller.qids[controller.ids]]
        joint_velocity = env.sim.data.qvel[:, controller.vids[controller.ids]]
        nominal_kp = p.new_tensor([256.,384.,384.,256.,192.,128.])
        nominal_kd = p.new_tensor([6.4]*6)
        expected_kp = nominal_kp * torch.pow(p.new_tensor(1.25), p)
        expected_kd = nominal_kd * torch.sqrt(torch.pow(p.new_tensor(1.25), p))
        expected_request = expected_kp * position_error - expected_kd * joint_velocity
        upper = p.new_tensor([30.,60.,30.,30.,30.,30.])
        lower = -upper
        expected_torque = torch.maximum(torch.minimum(expected_request, upper), lower)
        env.sim.step()
        torch.testing.assert_close(controller.applied_kp, expected_kp)
        torch.testing.assert_close(controller.applied_kd, expected_kd)
        torch.testing.assert_close(controller.requested_torque[:, controller.ids], expected_request)
        torch.testing.assert_close(controller.torque[:, controller.ids], expected_torque)
        torch.testing.assert_close(env.sim.data.actuator_force[:, controller.ids], expected_torque)
        torch.testing.assert_close(controller.saturated[:, controller.ids], expected_request.ne(expected_torque))
        assert controller.updates == 1
        assert controller.metadata()["version"] == "variable_impedance_pd500_v1"
        assert "ignored_action_indices" not in controller.metadata()
    finally:
        env.close()


def test_variable_sample_clamps_effort_and_holds_old_request_for_eight_steps():
    env, controller = _variable_controller()
    try:
        env.reset()
        gain = env.action_manager.get_term("joint_stiffness")
        gain.process_actions(torch.ones(2, 6))
        env.sim.data.ctrl[:] = 10.
        env.sim.step()
        first = controller.torque.clone()
        first_request = controller.requested_torque.clone()
        first_kp = controller.applied_kp.clone()
        assert bool(controller.saturated[:, controller.ids].all())
        assert bool((first[:, controller.ids].abs() <= first.new_tensor([30.,60.,30.,30.,30.,30.])).all())
        gain.process_actions(-torch.ones(2, 6))
        env.sim.data.ctrl[:] = -10.
        for _ in range(7):
            env.sim.step()
            torch.testing.assert_close(controller.torque, first)
            torch.testing.assert_close(controller.requested_torque, first_request)
            torch.testing.assert_close(controller.applied_kp, first_kp)
        assert controller.updates == 1
        env.sim.step()
        assert controller.updates == 2
        assert bool((controller.torque[:, controller.ids] < 0).all())
        assert bool((controller.applied_kp < first_kp).all())
    finally:
        env.close()


def test_variable_nominal_matches_fixed_at_identical_state_and_full_policy_interval():
    api = subject()
    results = []
    for mode in ("fixed", "variable"):
        cfg = config()
        api.configure_sampled_pd500(cfg, gain_mode=mode)
        env = ManagerBasedRlEnv(cfg, device="cpu")
        try:
            controller = api.install_sampled_pd500(env, gain_mode=mode)
            env.reset()
            env.action_manager.get_term("joint_stiffness").process_actions(torch.zeros(2, 6))
            env.sim.data.qpos[:, controller.qids] = 0.
            env.sim.data.qvel[:, controller.vids] = 0.
            env.sim.data.ctrl[:] = 0.
            env.sim.data.ctrl[:, controller.ids] = .01
            env.sim.step()
            results.append(controller.torque.clone())
            for _ in range(79):
                env.sim.step()
            assert controller.updates == 10
            assert controller.count == 80
        finally:
            env.close()
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def test_variable_sample_uses_calibrated_nominals_not_different_native_nominals():
    env, controller = _variable_controller()
    try:
        env.reset()
        gain = env.action_manager.get_term("joint_stiffness")
        assert not torch.equal(gain.nominal_kp, controller.arm_kp)
        assert not torch.equal(gain.nominal_kd, controller.arm_kd)
        gain.process_actions(torch.zeros(2, 6))
        env.sim.data.qpos[:, controller.qids] = 0.
        env.sim.data.qvel[:, controller.vids] = 0.
        env.sim.data.ctrl[:] = 0.
        env.sim.data.ctrl[:, controller.ids] = .01
        env.sim.step()
        expected = torch.tensor([2.56, 3.84, 3.84, 2.56, 1.92, 1.28])
        torch.testing.assert_close(controller.applied_kp, torch.tensor([256.,384.,384.,256.,192.,128.]).expand(2,-1))
        torch.testing.assert_close(controller.torque[:, controller.ids], expected.expand(2,-1))
        torch.testing.assert_close(env.sim.data.actuator_force[:, controller.ids], expected.expand(2,-1))
    finally:
        env.close()


def test_fixed_subset_reset_preserves_held_torque_until_next_sample():
    api = subject()
    cfg = config()
    api.configure_sampled_pd500(cfg)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        controller = api.install_sampled_pd500(env)
        env.reset()
        env.sim.data.ctrl[:] = 10.
        env.sim.step()
        held = controller.torque.clone()
        applied_kp = controller.applied_kp.clone()
        env.action_manager.get_term("joint_stiffness").reset(torch.tensor([0]))
        for _ in range(7):
            env.sim.step()
            torch.testing.assert_close(controller.torque, held)
            torch.testing.assert_close(controller.applied_kp, applied_kp)
            torch.testing.assert_close(env.sim.data.actuator_force[:], held)
        assert controller.updates == 1
    finally:
        env.close()


def test_variable_subset_and_full_reset_invalidate_applied_snapshots():
    env, controller = _variable_controller()
    try:
        env.reset()
        gain = env.action_manager.get_term("joint_stiffness")
        gain.process_actions(torch.ones(2, 6))
        env.sim.data.ctrl[:] = 10.
        env.sim.step()
        gain.reset(torch.tensor([0]))
        assert not bool(controller.torque[0].any())
        assert not bool(controller.requested_torque[0].any())
        assert not bool(controller.saturated[0].any())
        assert not bool(controller.applied_kp[0].any())
        assert bool(controller.torque[1].any())
        for _ in range(7):
            env.sim.step()
        assert controller.updates == 1
        env.sim.step()
        assert controller.updates == 2
        torch.testing.assert_close(controller.applied_kp[0], controller.arm_kp)
        assert bool(controller.torque[0].any())
        env.reset()
        assert controller.count == controller.updates == 0
        assert controller.torque is None
        assert controller.applied_kp is None
        assert controller.applied_kd is None
        assert controller.requested_torque is None
        assert controller.saturated is None
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["bad", "VARIABLE", ""])
def test_sampled_mode_rejects_unknown_values(mode):
    api = subject()
    with pytest.raises(ValueError, match="gain_mode"):
        api.configure_sampled_pd500(config(), gain_mode=mode)


def test_configuration_preserves_wall_time_and_safety():
    cfg = config()
    before = dict(cfg.metrics["cat_soft"].params)
    subject().configure_sampled_pd500(cfg)
    assert cfg.sim.mujoco.timestep == .00025
    assert cfg.decimation == 80
    assert cfg.metrics["cat_soft"].params == before
    assert cfg.metrics["first_strike"].params["window_substeps"] == 200
    assert cfg.metrics["substep_impulse"].params["event_window_substeps"] == 200
    assert cfg.metrics["substep_delivered"].params["rearm_gap_substeps"] == 200


def test_live_hold_limits_gains_reset_and_distinct_checkpoint_contract():
    cfg = config()
    api = subject()
    api.configure_sampled_pd500(cfg)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        controller = api.install_sampled_pd500(env)
        env.reset()
        metadata = build_impact_speed_metadata(env, 1.)
        assert metadata["action_type"] == "joint_position_fixed_sampled_pd"
        assert "variable_impedance" not in metadata
        assert metadata["sampled_pd"]["pd_dt_s"] == .002
        assert metadata["sampled_pd"]["kp"] == [256., 384., 384., 256., 192., 128.]
        assert metadata["sampled_pd"]["effort_limits_nm"] == [30., 60., 30., 30., 30., 30.]
        validate_impact_speed_metadata(metadata, metadata)
        wrong = dict(metadata, sampled_pd={**metadata["sampled_pd"], "pd_dt_s": .001})
        with pytest.raises(ValueError, match="mismatch"):
            validate_impact_speed_metadata(wrong, metadata)
        gain = env.action_manager.get_term("joint_stiffness")
        env.action_manager.process_action(torch.ones(2, 13))
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step()
        first = env.sim.data.actuator_force[:].clone()
        env.sim.data.ctrl[:] = -10.
        for _ in range(7):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step()
            torch.testing.assert_close(env.sim.data.actuator_force[:], first, rtol=0, atol=1e-5)
        assert controller.updates == 1
        env.sim.step()
        assert controller.updates == 2
        limits = env.sim.data.actuator_force.new_tensor([30.,60.,30.,30.,30.,30.])
        assert bool((first[:, gain.control_ids].abs() <= limits).all())
        gain.reset(torch.tensor([0]))
        torch.testing.assert_close(gain.telemetry.kp, torch.tensor([[256.,384.,384.,256.,192.,128.]]).expand(2,-1))
        torch.testing.assert_close(gain.telemetry.kd, torch.full((2,6),6.4))
        env.sim.step = controller.original_step
        with pytest.raises(ValueError, match="installed"):
            build_impact_speed_metadata(env, 1.)
    finally:
        env.close()


@pytest.mark.parametrize("field", ["kp", "kd", "limits", "qids", "vids", "window"])
def test_metadata_rejects_drift_of_applied_controller(field):
    api = subject()
    cfg = config()
    api.configure_sampled_pd500(cfg)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        controller = api.install_sampled_pd500(env)
        if field == "window":
            env._hammer_first_strike._window = 25
        else:
            getattr(controller, field).flatten()[0] += 1
        with pytest.raises(ValueError, match="controller|window|mapping"):
            build_impact_speed_metadata(env, 1.)
    finally:
        env.close()


def test_metadata_rejects_live_execution_clock_drift_with_config_unchanged():
    env, controller = _variable_controller()
    try:
        assert controller.metadata()["physics_dt_s"] == .00025
        env.sim.model.opt.timestep[0] = .0005
        assert env.cfg.sim.mujoco.timestep == .00025
        assert env.sim.mj_model.opt.timestep == .00025
        with pytest.raises(ValueError, match="timestep"):
            controller.metadata()
    finally:
        env.close()


def test_checkpoint_roundtrip_and_native_controller_rejection(tmp_path):
    from copy import deepcopy
    from dataclasses import asdict
    from src.tasks.hammer.config.z1.rl_cfg import z1_impact_speed_ppo_runner_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    api = subject()
    cfg = config()
    api.configure_sampled_pd500(cfg)
    agent = asdict(z1_impact_speed_ppo_runner_cfg())
    agent["logger"] = "tensorboard"
    agent["actor"]["hidden_dims"] = agent["critic"]["hidden_dims"] = [16,8]
    env = ManagerBasedRlEnv(cfg, device="cpu")
    native = None
    try:
        controller = api.install_sampled_pd500(env)
        wrapped = wrap_impact_speed_env(env, 1.)
        runner = HammerOnPolicyRunner(wrapped, deepcopy(agent), str(tmp_path), "cpu")
        runner.logger.init_logging_writer()
        obs, _ = wrapped.reset()
        with torch.inference_mode():
            for _ in range(2):
                obs, *_ = wrapped.step(torch.zeros(2,13))
        assert controller.count == 160 and controller.updates == 20
        with torch.inference_mode():
            wrapped.reset()
        assert controller.count == 0 and controller.torque is None
        path = tmp_path / "model_0.pt"
        runner.save(str(path))
        runner.load(str(path), strict=True)
        assert list(tmp_path.rglob("*.onnx")), "sampled controller ONNX export missing"
        native = ManagerBasedRlEnv(config(), device="cpu")
        old_runner = HammerOnPolicyRunner(wrap_impact_speed_env(native,1.),agent)
        with pytest.raises(ValueError, match="mismatch"):
            old_runner.load(str(path), strict=True)
    finally:
        env.close()
        if native is not None:
            native.close()
