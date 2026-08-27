"""Unit tests for the direct, one-segment hammer strike reference.

Pure torch — no MuJoCo/Warp.  The reference anchors the realized reset head
and advances only through spatial progress on the finite follow-through
segment; episode time must not create motion.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp import references as reference_mdp
from src.tasks.hammer.mdp.references import (
    SingleStrikeReference,
    get_strike_reference,
)


B = 3
HEAD0 = torch.tensor([[0.50, 0.00, 0.20]]).repeat(B, 1)
NAIL = torch.tensor([[0.50, 0.00, 0.102]]).repeat(B, 1)


def _ref(**kw) -> SingleStrikeReference:
    return SingleStrikeReference(B, "cpu", overshoot=0.05, descent_speed=0.03, axis_tol=0.01, **kw)


def _steps(v: int) -> torch.Tensor:
    return torch.full((B,), v, dtype=torch.long)


def _target(nail: torch.Tensor = NAIL) -> torch.Tensor:
    return nail - torch.tensor([0.0, 0.0, 0.05])


def test_reference_is_the_reset_head_to_follow_through_segment():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))

    assert torch.equal(ref._head0, HEAD0)
    assert torch.equal(ref._target, _target())
    assert torch.allclose(ref.waypoint(torch.zeros(B)), HEAD0)
    midpoint = torch.tensor([[0.50, 0.00, 0.126]]).repeat(B, 1)
    assert torch.allclose(ref.waypoint(torch.full((B,), 0.5)), midpoint, atol=1e-6)
    assert torch.equal(ref.waypoint(torch.ones(B)), _target())


def test_strike_axis_followthrough_is_one_straight_line_through_the_nail():
    starts = torch.tensor(
        [
            [-0.03, 0.00, 0.14],
            [0.00, 0.00, 0.15],
            [0.03, 0.00, 0.14],
        ]
    )
    nail = torch.tensor([[0.00, 0.00, 0.10]]).repeat(B, 1)
    expected = torch.tensor(
        [
            [
                [-0.03, 0.00, 0.14],
                [-0.015, 0.00, 0.12],
                [0.00, 0.00, 0.10],
                [0.015, 0.00, 0.08],
                [0.03, 0.00, 0.06],
            ],
            [
                [0.00, 0.00, 0.15],
                [0.00, 0.00, 0.125],
                [0.00, 0.00, 0.10],
                [0.00, 0.00, 0.075],
                [0.00, 0.00, 0.05],
            ],
            [
                [0.03, 0.00, 0.14],
                [0.015, 0.00, 0.12],
                [0.00, 0.00, 0.10],
                [-0.015, 0.00, 0.08],
                [-0.03, 0.00, 0.06],
            ],
        ]
    )
    ref = SingleStrikeReference(
        B,
        "cpu",
        overshoot=0.05,
        followthrough_mode="strike_axis",
    )
    ref.update(starts, nail, _steps(0))

    phases = (0.0, 0.25, 0.5, 0.75, 1.0)
    for index, phase in enumerate(phases):
        waypoint = ref.waypoint(torch.full((B,), phase))
        torch.testing.assert_close(waypoint, expected[:, index], rtol=0.0, atol=1e-7)
    for env_id in range(B):
        torch.testing.assert_close(
            ref.reference_polyline(env_id=env_id, num_points=5),
            expected[env_id],
            rtol=0.0,
            atol=1e-7,
        )


def test_waypoints_never_rise_above_the_frozen_reset_head():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))

    for phi in (0.0, 0.25, 0.5, 0.75, 1.0):
        waypoint = ref.waypoint(torch.full((B,), phi))
        assert torch.all(waypoint[:, 2] <= HEAD0[:, 2])


def test_low_reset_clamps_the_target_height_without_changing_xy_target():
    low_head = torch.tensor([[0.30, -0.20, 0.00]]).repeat(B, 1)
    nail = torch.tensor([[0.80, 0.40, 0.10]]).repeat(B, 1)
    ref = _ref()
    ref.update(low_head, nail, _steps(0))

    expected_target = torch.tensor([[0.80, 0.40, 0.00]]).repeat(B, 1)
    assert torch.equal(ref._target, expected_target)
    for phi in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert torch.all(ref.waypoint(torch.full((B,), phi))[:, 2] <= low_head[:, 2])
    for k in range(ref.playback_length() + 1):
        assert torch.all(ref.playback_target(k)[:, 2] <= low_head[:, 2])


def test_stationary_head_does_not_advance_from_episode_clock():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))

    for step in (1, 7, 100):
        assert torch.allclose(ref.update(HEAD0, NAIL, _steps(step)), torch.zeros(B))


def test_phase_is_projection_gated_and_monotone_latched():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    halfway = torch.tensor([[0.50, 0.00, 0.126]]).repeat(B, 1)
    assert torch.allclose(ref.update(halfway, NAIL, _steps(1)), torch.full((B,), 0.5))

    off_axis = torch.tensor([[0.512, 0.00, 0.052]]).repeat(B, 1)
    assert torch.allclose(ref.update(off_axis, NAIL, _steps(2)), torch.full((B,), 0.5))
    assert torch.allclose(ref.update(_target(), NAIL, _steps(3)), torch.ones(B))
    assert torch.allclose(ref.update(halfway, NAIL, _steps(4)), torch.ones(B))


def test_preview_is_current_kinematics_only_and_peek_is_defensive():
    ref = _ref()
    assert torch.allclose(ref.peek(), torch.zeros(B))
    assert not bool(ref._anchored.any())
    ref.update(HEAD0, NAIL, _steps(0))
    halfway = torch.tensor([[0.50, 0.00, 0.126]]).repeat(B, 1)

    state = (ref._phi.clone(), ref._anchored.clone())
    assert torch.allclose(ref.preview(halfway, _steps(5)), torch.full((B,), 0.5))
    assert torch.equal(ref._phi, state[0])
    assert torch.equal(ref._anchored, state[1])
    peeked = ref.peek()
    peeked += 1.0
    assert torch.allclose(ref.peek(), torch.zeros(B))


def test_reanchor_and_batching_follow_each_environment_reset_state():
    heads = HEAD0.clone()
    heads[1, 0] += 0.04
    nails = NAIL.clone()
    nails[1, 0] += 0.04
    ref = _ref()
    ref.update(heads, nails, _steps(0))

    assert torch.equal(ref._head0, heads)
    assert torch.equal(ref._target, _target(nails))
    assert not torch.allclose(ref.waypoint(torch.full((B,), 0.5))[0], ref.waypoint(torch.full((B,), 0.5))[1])

    halfway = (heads + _target(nails)) / 2
    ref.update(halfway, nails, _steps(1))
    assert torch.allclose(ref.peek(), torch.tensor([0.5, 0.5, 0.5]))

    reset_heads = heads.clone()
    reset_heads[0, 1] += 0.02
    reset_nails = nails.clone()
    reset_nails[0, 1] += 0.02
    phi = ref.update(reset_heads, reset_nails, torch.tensor([0, 2, 2]))
    assert torch.allclose(phi, torch.tensor([0.0, 0.5, 0.5]))
    assert torch.equal(ref._head0[0], reset_heads[0])
    assert torch.equal(ref._head0[1:], heads[1:])
    assert torch.equal(ref._target[0], _target(reset_nails)[0])
    assert torch.equal(ref._target[1:], _target(nails)[1:])

    ref.reset(torch.tensor([2]))
    rearmed_heads = reset_heads.clone()
    rearmed_heads[2, 0] += 0.03
    rearmed_nails = reset_nails.clone()
    rearmed_nails[2, 0] += 0.03
    phi = ref.update(rearmed_heads, rearmed_nails, torch.tensor([1, 3, 3]))
    assert torch.allclose(phi, torch.tensor([0.0, 0.5, 0.0]))
    assert torch.equal(ref._head0[2], rearmed_heads[2])
    assert torch.equal(ref._target[2], _target(rearmed_nails)[2])


def test_zero_length_segment_is_safe():
    nail_at_target = HEAD0 + torch.tensor([0.0, 0.0, 0.05])
    ref = _ref()
    assert torch.allclose(ref.update(HEAD0, nail_at_target, _steps(0)), torch.zeros(B))
    assert torch.allclose(ref.update(HEAD0, nail_at_target, _steps(4)), torch.zeros(B))
    assert torch.equal(ref.waypoint(torch.ones(B)), HEAD0)
    assert ref.playback_length() == 0
    assert torch.equal(ref.playback_target(1), HEAD0)


def test_playback_advances_along_the_direct_segment_at_descent_speed():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))

    assert ref.playback_length() == 5
    previous = ref.playback_target(0)
    for k in range(1, 6):
        current = ref.playback_target(k)
        assert torch.all((current[:, 2] <= previous[:, 2]))
        assert float((current - previous).norm(dim=-1).max()) <= 0.03 + 1e-6
        previous = current
    assert torch.equal(ref.playback_target(5), _target())
    assert torch.equal(ref.playback_target(99), _target())


def test_horizontal_routes_use_the_frozen_world_x_detour_only():
    ref = _ref(horizontal_detour_m=0.020)
    ref.update(HEAD0, NAIL, _steps(0))
    ref.set_route_signs(torch.tensor([-1, 0, 1]))

    phi = torch.tensor([0.25, 0.25, 0.25])
    waypoint = ref.waypoint(phi)
    direct = torch.tensor([[0.50, 0.00, 0.163]]).repeat(B, 1)
    expected = direct + torch.tensor(
        [[-0.020, 0.0, 0.0], [0.0, 0.0, 0.0], [0.020, 0.0, 0.0]]
    )
    assert torch.allclose(waypoint, expected, atol=1e-7, rtol=0.0)

    for phase in (0.0, 0.125, 0.5, 0.75, 1.0):
        routed = ref.waypoint(torch.full((B,), phase))
        straight = (HEAD0 * (1.0 - phase)) + (_target() * phase)
        assert torch.equal(routed[:, 1:], straight[:, 1:])
    assert torch.equal(ref.waypoint(torch.zeros(B)), HEAD0)
    assert torch.equal(ref.waypoint(torch.ones(B)), _target())
    assert torch.equal(ref.waypoint(torch.full((B,), 0.5))[:, 0], HEAD0[:, 0])
    assert torch.equal(
        ref.waypoint(torch.full((B,), 0.75)),
        (HEAD0 * 0.25) + (_target() * 0.75),
    )


def test_zero_horizontal_amplitude_is_bit_identical_to_direct_reference():
    direct = _ref()
    routed = _ref(horizontal_detour_m=0.0)
    direct.update(HEAD0, NAIL, _steps(0))
    routed.update(HEAD0, NAIL, _steps(0))
    routed.set_route_signs(torch.tensor([-1, 0, 1]))

    for phase in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
        phi = torch.full((B,), phase)
        assert torch.equal(routed.waypoint(phi), direct.waypoint(phi))
    for step in range(direct.playback_length() + 3):
        assert torch.equal(routed.playback_target(step), direct.playback_target(step))


def test_reference_polyline_samples_the_commanded_route_without_mutating_state():
    ref = _ref(horizontal_detour_m=0.020)
    ref.update(HEAD0, NAIL, _steps(0))
    ref.set_route_signs(torch.tensor([-1, 0, 1]))
    state_before = {
        "head0": ref._head0.clone(),
        "target": ref._target.clone(),
        "phi": ref._phi.clone(),
        "anchored": ref._anchored.clone(),
        "route_sign": ref._route_sign.clone(),
    }

    path = ref.reference_polyline(env_id=2, num_points=5)

    assert path.shape == (5, 3)
    assert torch.equal(path[0], HEAD0[2])
    assert torch.equal(path[-1], _target()[2])
    assert path[1, 0] == pytest.approx(float(HEAD0[2, 0] + 0.020))
    assert torch.equal(ref._head0, state_before["head0"])
    assert torch.equal(ref._target, state_before["target"])
    assert torch.equal(ref._phi, state_before["phi"])
    assert torch.equal(ref._anchored, state_before["anchored"])
    assert torch.equal(ref._route_sign, state_before["route_sign"])


def test_reference_polyline_uses_the_new_live_anchor_after_reset():
    ref = _ref()
    ref.update(HEAD0, NAIL, _steps(0))
    first = ref.reference_polyline(env_id=0, num_points=3)

    shifted_head = HEAD0.clone()
    shifted_nail = NAIL.clone()
    shifted_head[0, 1] += 0.04
    shifted_nail[0, 1] += 0.04
    ref.reset(torch.tensor([0]))
    ref.update(shifted_head, shifted_nail, torch.tensor([0, 1, 1]))
    second = ref.reference_polyline(env_id=0, num_points=3)

    assert torch.equal(first[:, 1], torch.zeros(3))
    assert torch.equal(second[:, 1], torch.full((3,), 0.04))


_ROBOT_CFG = SimpleNamespace(name="robot", site_ids=[0])
_NAIL_CFG = SimpleNamespace(name="nail_block", site_ids=[0])


def _stub_env(num_envs: int = B):
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=HEAD0.clone().unsqueeze(1)))
    nail = SimpleNamespace(data=SimpleNamespace(site_pos_w=NAIL.clone().unsqueeze(1)))
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"robot": robot, "nail_block": nail},
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )


def test_observations_anchor_and_stay_stationary_without_spatial_progress():
    from src.tasks.hammer.mdp.observations import strike_phase, strike_ref_error

    env = _stub_env()
    assert tuple(strike_phase(env, robot_cfg=_ROBOT_CFG, nail_cfg=_NAIL_CFG).shape) == (B, 1)
    env.episode_length_buf += 7
    assert torch.allclose(strike_phase(env, robot_cfg=_ROBOT_CFG, nail_cfg=_NAIL_CFG), torch.zeros(B, 1))
    assert torch.allclose(strike_ref_error(env, robot_cfg=_ROBOT_CFG, nail_cfg=_NAIL_CFG), torch.zeros(B, 3))


def test_get_strike_reference_is_cached_per_env():
    env = _stub_env()
    assert get_strike_reference(env) is get_strike_reference(env)


def test_reset_event_samples_only_requested_envs_and_signs_stay_episode_stable():
    env = _stub_env()
    ref = get_strike_reference(env, horizontal_detour_m=0.020)
    ref.set_route_signs(torch.tensor([1, 1, 1]))

    torch.manual_seed(20260821)
    reference_mdp.sample_strike_route_signs(
        env, torch.tensor([0, 2]), horizontal_detour_m=0.020
    )
    sampled = ref.route_signs()
    assert get_strike_reference(env, horizontal_detour_m=0.020) is ref
    assert torch.equal(sampled, torch.tensor([-1, 1, -1], dtype=torch.int8))

    ref.update(HEAD0, NAIL, _steps(0))
    ref.update(HEAD0, NAIL, _steps(9))
    ref.waypoint(torch.full((B,), 0.25))
    assert torch.equal(ref.route_signs(), sampled)

    reference_mdp.sample_strike_route_signs(
        env, torch.tensor([1]), horizontal_detour_m=0.020
    )
    resampled = ref.route_signs()
    assert torch.equal(resampled[[0, 2]], sampled[[0, 2]])
    assert torch.equal(resampled, torch.tensor([-1, 0, -1], dtype=torch.int8))


@pytest.mark.parametrize("fixed_sign", (-1, 0, 1))
def test_reset_event_can_fix_one_route_without_changing_rng_consumption(fixed_sign):
    fixed_env = _stub_env()
    fixed_ref = get_strike_reference(
        fixed_env, horizontal_detour_m=0.0, followthrough_mode="strike_axis"
    )
    env_ids = torch.tensor([0, 2])

    torch.manual_seed(20260827)
    reference_mdp.sample_strike_route_signs(
        fixed_env,
        env_ids,
        horizontal_detour_m=0.0,
        followthrough_mode="strike_axis",
        fixed_route_sign=fixed_sign,
    )
    fixed_next_draw = torch.randint(0, 2**16, (8,))

    sampled_env = _stub_env()
    torch.manual_seed(20260827)
    reference_mdp.sample_strike_route_signs(
        sampled_env,
        env_ids,
        horizontal_detour_m=0.0,
        followthrough_mode="strike_axis",
    )
    sampled_next_draw = torch.randint(0, 2**16, (8,))

    assert torch.equal(
        fixed_ref.route_signs()[env_ids],
        torch.full((2,), fixed_sign, dtype=torch.int8),
    )
    assert torch.equal(fixed_next_draw, sampled_next_draw)


def test_route_start_reset_writes_the_joint_pose_selected_by_cached_sign():
    class RecordingRobot:
        def __init__(self):
            self.calls = []

        def write_joint_state_to_sim(
            self, joint_pos, joint_vel, *, joint_ids, env_ids
        ):
            self.calls.append(
                (
                    joint_pos.clone(),
                    joint_vel.clone(),
                    tuple(joint_ids),
                    env_ids.clone(),
                )
            )

    ref = SingleStrikeReference(B, "cpu", followthrough_mode="strike_axis")
    ref.set_route_signs(torch.tensor([-1, 0, 1]))
    robot = RecordingRobot()
    env = SimpleNamespace(
        num_envs=B,
        device="cpu",
        scene={"robot": robot},
        _strike_reference=ref,
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=list(range(6)))
    route_joint_positions = (
        (0.0, 1.1, -0.1, -1.0, 0.0, 1.5),
        (0.0, 1.2, -0.2, -1.0, 0.0, 1.5),
        (0.0, 1.3, -0.3, -1.0, 0.0, 1.5),
    )

    reference_mdp.reset_joints_to_strike_route_starts(
        env,
        torch.tensor([0, 2]),
        route_joint_positions=route_joint_positions,
        asset_cfg=asset_cfg,
    )

    assert len(robot.calls) == 1
    joint_pos, joint_vel, joint_ids, env_ids = robot.calls[0]
    torch.testing.assert_close(
        joint_pos,
        torch.tensor([route_joint_positions[0], route_joint_positions[2]]),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(joint_vel) == 0
    assert joint_ids == tuple(range(6))
    assert torch.equal(env_ids, torch.tensor([0, 2]))
