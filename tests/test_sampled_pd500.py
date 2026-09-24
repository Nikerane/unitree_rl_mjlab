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
