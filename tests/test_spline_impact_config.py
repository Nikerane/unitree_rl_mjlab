"""Free-time quintic route and guidance configuration contract."""

import math

import pytest
import torch

from mjlab.tasks.registry import load_env_cfg

import src.tasks  # noqa: F401 - register tasks
import src.tasks.hammer.config.z1 as z1


LEFT = "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed"
THREE = "Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed"


@pytest.mark.parametrize("play", [False, True])
def test_left_task_keeps_baseline_and_three_starts_only_changes_route(play):
    left = load_env_cfg(LEFT, play=play)
    three = load_env_cfg(THREE, play=play)
    assert left.events["sample_strike_route_signs"].params["fixed_route_sign"] == -1
    assert "fixed_route_sign" not in three.events["sample_strike_route_signs"].params
    assert (three.events["reset_strike_route_joints"].params["route_joint_positions"]
            == left.events["reset_strike_route_joints"].params["route_joint_positions"])
    for cfg in (left, three):
        assert cfg.rewards["r_imit"].weight == .2
        assert cfg.rewards["r_imit"].params["sigma"] == .05
        assert "r_imit_anneal" not in cfg.curriculum
        assert cfg.metrics["cat_soft"].params["imp_max_p"] == .2
        assert cfg.metrics["cat_soft"].params["imp_limit"] == [.369, .246, .738, .369, .246, .0164]
        consumers = [cfg.events["sample_strike_route_signs"].params,
                     cfg.rewards["r_imit"].params]
        for group in cfg.observations.values():
            consumers.extend(group.terms[n].params for n in ("strike_phase", "strike_ref_error"))
        for params in consumers:
            assert params["path_shape"] == "quintic"
            assert params["overshoot"] == .02
            assert params["followthrough_mode"] == "vertical"
        assert tuple(cfg.actions) == tuple(left.actions)
        for group in ("actor", "critic"):
            assert tuple(cfg.observations[group].terms) == tuple(left.observations[group].terms)


@pytest.mark.parametrize("weight,width", [(float("nan"), .05), (-.1, .05),
                                              (float("inf"), .05), (.2, 0.),
                                              (.2, float("nan")), (.2, float("inf"))])
def test_guidance_rejects_invalid_overrides(weight, width):
    with pytest.raises(ValueError):
        z1._free_time_quintic_env_cfg(play=False, route_mode="three_starts",
                                   guidance_weight=weight, guidance_width=width)


def test_guidance_overrides_are_controlled_and_only_change_requested_fields():
    cfg = z1._free_time_quintic_env_cfg(play=False, route_mode="three_starts",
                                     guidance_weight=.35, guidance_width=.07)
    assert cfg.rewards["r_imit"].weight == .35
    assert cfg.rewards["r_imit"].params["sigma"] == .07
    assert "r_imit_anneal" not in cfg.curriculum
    assert cfg.metrics["cat_soft"].params["imp_max_p"] == .2
    assert math.isfinite(cfg.rewards["r_imit"].weight)


@pytest.mark.integration
def test_three_start_live_reset_geometry_and_observations():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg(THREE, play=True)
    cfg.scene.num_envs = 24
    cfg.actions["strike_speed"].bounds = (.2, .8)
    cfg.rewards["speed_goal"].weight = -.4
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        observations, _ = env.reset(seed=20260826)
        ref = env._strike_reference
        signs = ref.route_signs()
        assert set(signs.tolist()) == {-1, 0, 1}
        robot = env.scene["robot"]
        joint_ids = env.action_manager.get_term("joint_position").target_ids
        actual = robot.data.joint_pos[:, joint_ids]
        table = torch.tensor(cfg.events["reset_strike_route_joints"].params["route_joint_positions"])
        torch.testing.assert_close(actual, table[signs.long() + 1], atol=1e-6, rtol=0)
        head_ids, _ = robot.find_sites(("hammer_head_site",))
        head = robot.data.site_pos_w[:, head_ids].squeeze(1)
        nail_ids, _ = env.scene["nail_block"].find_sites(("nail_top",))
        nail = env.scene["nail_block"].data.site_pos_w[:, nail_ids].squeeze(1)
        expected_x = .500001967 + signs.to(head.dtype) * .020
        torch.testing.assert_close(head[:, 0], expected_x, atol=2e-6, rtol=0)
        for i in range(env.num_envs):
            path = ref.reference_polyline(i, 65)
            torch.testing.assert_close(path[0], head[i], atol=0, rtol=0)
            contact_phase = (head[i, 2] - nail[i, 2]) / (head[i, 2] - (nail[i, 2] - .02))
            contact = ref.waypoint(torch.full((env.num_envs,), contact_phase))[i]
            torch.testing.assert_close(contact, nail[i], atol=1e-6, rtol=0)
            torch.testing.assert_close(path[-1, :2], nail[i, :2], atol=1e-6, rtol=0)
            assert path[-1, 2].item() == pytest.approx(nail[i, 2].item() - .02, abs=1e-6)
            before = ref.waypoint(torch.full((env.num_envs,), contact_phase - .001))[i]
            assert abs((contact[0] - before[0]).item()) < 1e-6
        # Paired episode starts at the same nail pose expose distinct actor inputs.
        actor = observations["actor"]
        assert actor.shape == (24, 42)
        for a, b in ((-1, 0), (0, 1), (-1, 1)):
            first = actor[(signs == a).nonzero()[0, 0]]
            second = actor[(signs == b).nonzero()[0, 0]]
            assert not torch.equal(first, second)
        old_signs = signs.clone()
        old_phase = ref.peek()
        old_joints = actual.clone()
        subset = torch.tensor([0, 2, 4])
        sampler = env.event_manager.get_term_cfg("sample_strike_route_signs")
        sampler.func(env, subset, **sampler.params)
        route_reset = env.event_manager.get_term_cfg("reset_strike_route_joints")
        route_reset.func(env, subset, **route_reset.params)
        assert torch.equal(ref.route_signs()[1::2], old_signs[1::2])
        torch.testing.assert_close(actual[1::2], old_joints[1::2])
        assert torch.equal(ref.peek()[1::2], old_phase[1::2])
    finally:
        env.close()
