"""Layer 4 — Full environment construction and physics sanity tests.

These tests require the complete mjlab + Warp stack and will compile
GPU/CPU kernels on the first run (slow). Run with:

    pytest tests/test_env.py -v                    # requires Warp
    pytest tests/ -v -m "not integration"          # skip this file

All tests share a single env instance (module scope) to avoid recompiling
Warp kernels between tests.
"""

import pytest
import torch

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nail_depth(env) -> float:
    """Return current nail slide qpos for env 0."""
    nail_entity = env.scene["nail_block"]
    joint_ids, _ = nail_entity.find_joints(("nail_slide",))
    return nail_entity.data.joint_pos[0, joint_ids[0]].item()


def _get_site_z(entity, site_name: str) -> float:
    """Return world-frame Z position of a named site for env 0."""
    site_ids, _ = entity.find_sites((site_name,))
    return entity.data.site_pos_w[0, site_ids[0], 2].item()


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


class TestEnvConstruction:
    def test_env_is_not_none(self, env_cpu):
        assert env_cpu is not None

    def test_action_shape(self, env_cpu):
        """DifferentialIK produces 3D delta-position actions."""
        assert env_cpu.action_manager.action.shape == (1, 3)

    def test_num_envs(self, env_cpu):
        assert env_cpu.cfg.scene.num_envs == 1


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


class TestObservations:
    def test_reset_returns_dict(self, env_cpu):
        obs, _ = env_cpu.reset()
        assert isinstance(obs, dict)

    def test_obs_has_actor_key(self, obs_after_reset):
        assert "actor" in obs_after_reset

    def test_obs_has_critic_key(self, obs_after_reset):
        assert "critic" in obs_after_reset

    def test_actor_obs_shape(self, obs_after_reset):
        # 7 joint_pos + 7 joint_vel + 3 ee_pos + 3 ee_vel
        # + 3 head_pos + 3 head_vel + 3 nail_top_pos + 1 nail_depth + 3 actions = 33
        assert obs_after_reset["actor"].shape == (1, 33), (
            f"Expected actor shape (1, 33), got {obs_after_reset['actor'].shape}"
        )

    def test_critic_obs_shape(self, obs_after_reset):
        assert obs_after_reset["critic"].shape == (1, 33)

    def test_obs_finite(self, obs_after_reset):
        for key, tensor in obs_after_reset.items():
            assert torch.isfinite(tensor).all(), (
                f"Non-finite values in obs['{key}']"
            )


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_runs(self, env_cpu):
        env_cpu.reset()
        action = torch.zeros(1, 3)
        result = env_cpu.step(action)
        assert result is not None

    def test_step_returns_5_tuple(self, env_cpu):
        env_cpu.reset()
        action = torch.zeros(1, 3)
        result = env_cpu.step(action)
        assert len(result) == 5

    def test_reward_shape(self, env_cpu):
        env_cpu.reset()
        _, reward, _, _, _ = env_cpu.step(torch.zeros(1, 3))
        assert reward.shape == (1,), f"Expected reward shape (1,), got {reward.shape}"

    def test_done_is_bool_tensor(self, env_cpu):
        env_cpu.reset()
        _, _, done, _, _ = env_cpu.step(torch.zeros(1, 3))
        assert done.dtype == torch.bool

    def test_reward_is_finite(self, env_cpu):
        env_cpu.reset()
        _, reward, _, _, _ = env_cpu.step(torch.zeros(1, 3))
        assert torch.isfinite(reward).all()

    def test_obs_after_step_is_finite(self, env_cpu):
        env_cpu.reset()
        obs, _, _, _, _ = env_cpu.step(torch.zeros(1, 3))
        for key, tensor in obs.items():
            assert torch.isfinite(tensor).all(), (
                f"Non-finite values in obs['{key}'] after step"
            )


# ---------------------------------------------------------------------------
# Physics sanity at reset
# ---------------------------------------------------------------------------


class TestPhysicsAtReset:
    def test_nail_depth_zero_at_reset(self, env_cpu):
        env_cpu.reset()
        depth = _get_nail_depth(env_cpu)
        assert abs(depth) < 1e-4, (
            f"Nail depth should be ~0 at reset, got {depth:.6f}"
        )

    def test_hammer_head_above_nail_at_reset(self, env_cpu):
        """Hammer head must start above the nail so it can strike downward."""
        env_cpu.reset()
        robot = env_cpu.scene["robot"]
        nail = env_cpu.scene["nail_block"]

        head_z = _get_site_z(robot, "hammer_head_site")
        nail_top_z = _get_site_z(nail, "nail_top")

        assert head_z > nail_top_z, (
            f"Hammer head z={head_z:.4f} is not above nail top z={nail_top_z:.4f}"
        )

    def test_neutral_joint_positions_at_reset(self, env_cpu):
        """Joint positions after reset must match NEUTRAL_JOINT_POS within 1e-3 rad."""
        from src.assets.robots.unitree_z1.z1_constants import NEUTRAL_JOINT_POS

        env_cpu.reset()
        robot = env_cpu.scene["robot"]
        joint_pos = robot.data.joint_pos[0]  # shape: (njoint,)

        for i, name in enumerate(robot.joint_names):
            expected = NEUTRAL_JOINT_POS[name]
            actual = joint_pos[i].item()
            assert abs(actual - expected) < 1e-3, (
                f"Joint '{name}': expected {expected:.6f} rad, got {actual:.6f} rad"
            )

    def test_nail_position_in_valid_range(self, env_cpu):
        """Nail qpos must be within its slide joint range [0, 0.075]."""
        env_cpu.reset()
        depth = _get_nail_depth(env_cpu)
        assert 0.0 <= depth <= 0.075, (
            f"Nail depth {depth:.6f} is outside valid range [0, 0.075]"
        )


# ---------------------------------------------------------------------------
# Termination logic
# ---------------------------------------------------------------------------


class TestTermination:
    def test_no_termination_at_reset(self, env_cpu):
        """Episode should not be done immediately after reset."""
        env_cpu.reset()
        _, _, done, _, _ = env_cpu.step(torch.zeros(1, 3))
        # At rest with nail at 0, done should be False (nail_driven condition)
        # (time_out won't trigger on step 1 of a 20 s episode)
        assert not done[0].item(), (
            "Episode terminated on step 1 — check nail_fully_driven threshold"
        )

    def test_termination_when_nail_fully_driven(self, env_cpu):
        """Manually driving the nail past the success threshold must trigger done."""
        from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD

        env_cpu.reset()
        nail_entity = env_cpu.scene["nail_block"]
        joint_ids, _ = nail_entity.find_joints(("nail_slide",))

        # Set nail qpos just above the success threshold
        target_depth = NAIL_SUCCESS_THRESHOLD + 0.001
        pos = torch.tensor(
            [[target_depth]], dtype=torch.float32, device="cpu"
        )
        nail_entity.data.write_joint_position(
            pos, joint_ids=torch.tensor(joint_ids, dtype=torch.long)
        )

        _, _, done, _, _ = env_cpu.step(torch.zeros(1, 3))
        assert done[0].item(), (
            f"Expected done=True with nail at {target_depth:.4f} m "
            f"(threshold={NAIL_SUCCESS_THRESHOLD})"
        )
