"""No-learning integration evidence for the free-time spline treatment."""

from dataclasses import asdict

import pytest
import torch


@pytest.mark.parametrize("gain_mode", ["fixed", "variable"])
def test_cpu_sampled_three_routes_async_reset_and_pre_reset_measurement(gain_mode):
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.mdp.strike_evaluation import configure_strike_evaluation
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
    from src.tasks.hammer.rl.impact_speed_wrapper import configure_impact_speed_controller, install_impact_speed_controller
    from scripts.evaluate_impact_speed import PreResetSnapshot

    cfg = _impact_speed_env_cfg(play=False, route_mode="three_starts")
    cfg.scene.num_envs = 12
    cfg.seed = 7
    cfg.actions["strike_speed"].bounds = (1.5, 2.2)
    cfg.rewards["speed_goal"].weight = -.4
    configure_impact_speed_controller(cfg, sampled_pd500=True, gain_mode=gain_mode)
    configure_strike_evaluation(cfg)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        controller = install_impact_speed_controller(env, sampled_pd500=True, gain_mode=gain_mode)
        wrapped = wrap_impact_speed_env(env, 1.)
        obs, _ = wrapped.reset()
        route_signs = env._strike_reference.route_signs()
        assert set(route_signs.tolist()) == {-1, 0, 1}
        route_params = cfg.events["reset_strike_route_joints"].params
        route_joints = [env.scene["robot"].joint_names.index(name)
                        for name in route_params["asset_cfg"].joint_names]
        expected_poses = torch.tensor(route_params["route_joint_positions"])
        torch.testing.assert_close(env.scene["robot"].data.joint_pos[:, route_joints],
                                   expected_poses[route_signs.long() + 1], rtol=0, atol=1e-6)
        assert obs["actor"].shape == obs["critic"].shape == (12, 42)
        env.episode_length_buf[0] = env.max_episode_length - 1
        before = env._strike_reference.route_signs().clone()
        action = torch.zeros(12, 13)
        action[:, 6:12] = .25
        with PreResetSnapshot(env) as capture:
            obs, _, done, extras = wrapped.step(action)
        assert done.tolist() == [True] + [False] * 11
        assert extras["impact_speed_outcome"]["charged"].tolist() == [True] + [False] * 11
        assert extras["impact_speed_outcome"]["goal_mps"].tolist() == pytest.approx([1.85] * 12)
        assert env.action_manager.get_term("strike_speed").state.pending.tolist() == [1.] + [0.] * 11
        assert torch.equal(env._strike_reference.route_signs()[1:], before[1:])
        assert capture.latest_evaluation is not None
        assert capture.latest_evaluation[0]["observed_samples"] == 80
        assert torch.isfinite(capture.latest["kp"]).all()
        assert torch.isfinite(capture.latest["kd"]).all()
        assert torch.isfinite(controller.applied_kp).all()
        assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
    finally:
        env.close()


@pytest.mark.parametrize("gain_mode", [None, "fixed", "variable"])
def test_diagnostics_leave_finite_staggered_reset_sequence_unchanged(gain_mode):
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.mdp.strike_evaluation import configure_strike_evaluation
    from src.tasks.hammer.rl.impact_speed_wrapper import (wrap_impact_speed_env,
        configure_impact_speed_controller, install_impact_speed_controller)
    from scripts.evaluate_impact_speed import PreResetSnapshot

    def sequence(diagnostic):
        cfg = _impact_speed_env_cfg(play=False, route_mode="three_starts")
        cfg.scene.num_envs = 3
        cfg.seed = 13
        cfg.actions["strike_speed"].bounds = (1.5, 2.2)
        cfg.rewards["speed_goal"].weight = -.4
        if gain_mode is not None:
            configure_impact_speed_controller(cfg, sampled_pd500=True, gain_mode=gain_mode)
        if diagnostic:
            configure_strike_evaluation(cfg)
        env = ManagerBasedRlEnv(cfg, device="cpu")
        try:
            controller = None
            if gain_mode is not None:
                controller = install_impact_speed_controller(env, sampled_pd500=True, gain_mode=gain_mode)
            wrapped = wrap_impact_speed_env(env, 1.)
            obs, _ = wrapped.reset()
            records = [(obs["actor"].clone(), obs["critic"].clone(),
                        env._strike_reference.route_signs().clone(),
                        torch.get_rng_state().clone())]
            for reset_world in (0, 1, 2):
                env.episode_length_buf[reset_world] = env.max_episode_length - 1
                action = torch.zeros(3, 13)
                action[:, 6:12] = .25
                with PreResetSnapshot(env) as capture:
                    obs, reward, done, extras = wrapped.step(action)
                assert torch.isfinite(obs["actor"]).all()
                assert torch.isfinite(reward).all()
                assert done.tolist() == [index == reset_world for index in range(3)]
                if diagnostic:
                    assert capture.latest_evaluation[reset_world]["observed_samples"] > 0
                records.append((obs["actor"].clone(), obs["critic"].clone(), reward.clone(),
                                done.clone(), extras["impact_speed_outcome"]["goal_mps"].clone(),
                                env.scene["robot"].data.joint_pos.clone(),
                                env.scene["robot"].data.joint_vel.clone(),
                                env._strike_reference.route_signs().clone(),
                                env.action_manager.get_term("strike_speed").state.pending.clone(),
                                capture.latest["kp"].clone(), capture.latest["kd"].clone(),
                                torch.get_rng_state().clone(),
                                controller.torque.clone() if controller is not None else torch.empty(0)))
                if controller is not None:
                    assert torch.isfinite(controller.applied_kp).all()
            return records
        finally:
            env.close()

    plain, instrumented = sequence(False), sequence(True)
    for left, right in zip(plain, instrumented):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_deterministic_post_latch_actor_recomputes_motor_and_selects_once(tmp_path, monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.config.z1.rl_cfg import z1_impact_speed_ppo_runner_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from mjlab.tasks import registry
    from scripts import evaluate_impact_speed as evaluation

    cfg = _impact_speed_env_cfg(play=False)
    cfg.scene.num_envs = 2
    cfg.seed = 7
    cfg.actions["strike_speed"].bounds = (1.5, 2.2)
    cfg.rewards["speed_goal"].weight = -.4
    env = ManagerBasedRlEnv(cfg, device="cpu")
    checkpoint = tmp_path / "untrained-fixture.pt"
    try:
        agent_cfg = asdict(z1_impact_speed_ppo_runner_cfg())
        agent_cfg["upload_model"] = False
        agent_cfg["logger"] = "tensorboard"
        HammerOnPolicyRunner(wrap_impact_speed_env(env, 1.), agent_cfg).save(str(checkpoint))
    finally:
        env.close()

    original = registry.load_env_cfg
    def short_episode(task, *, play):
        result = original(task, play=play)
        result.episode_length_s = .06
        return result
    monkeypatch.setattr(registry, "load_env_cfg", short_episode)

    def deterministic_policy(self, device=None):
        def actor(obs):
            normalized_goal = obs["actor"][:, -2]
            raw = torch.zeros(normalized_goal.numel(), 13, device=normalized_goal.device)
            raw[:, :12] = normalized_goal[:, None] * (1 - obs["actor"][:, -1, None])
            return raw
        return actor
    monkeypatch.setattr(HammerOnPolicyRunner, "get_inference_policy", deterministic_policy)
    args = evaluation.parse_args([
        "--task", "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed", "--checkpoint", str(checkpoint),
        "--speed-bounds", "1.5", "2.2", "--speed-cost", ".4", "--seed", "7", "--num-envs", "2",
        "--mode", "intervention", "--goals", "1.5", "1.85", "2.2", "--output", str(tmp_path / "paired")])
    summary = evaluation.run(args)
    assert summary["learning_updates"] == 0
    assert len({c["first_action_hash"] for c in summary["conditions"]}) == 1
    assert [c["selection_count"] for c in summary["conditions"]] == [2, 2, 2]
    low, center, high = [c["second_motor_action"][0][0] for c in summary["conditions"]]
    assert low == pytest.approx(-1., abs=1e-6)
    assert center == pytest.approx(0., abs=1e-6)
    assert high == pytest.approx(1., abs=1e-6)
    assert summary["comparisons"][-1]["second_motor_action_rms_difference"] > 0
    assert summary["comparisons"][-1]["responsive"] is False
