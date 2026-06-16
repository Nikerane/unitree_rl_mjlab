"""Layer 2 — Python config construction validation.

Tests that all config objects are correctly constructed and wired.
No MuJoCo model compilation, no mjlab env creation needed.
"""

import pytest

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg

from src.assets.robots.unitree_z1.z1_constants import (
    ARM_ACTUATOR_NAMES,
    ARM_JOINT_NAMES,
    EE_SITE_NAME,
    GRIPPER_JOINT_NAME,
    HAMMER_HEAD_SITE_NAME,
    NEUTRAL_JOINT_POS,
    Z1_ARTICULATION,
    Z1_HAMMER_DELTA_POS_SCALE,
    _Z1_ARM_J2,
    _Z1_ARM_STANDARD,
    _Z1_GRIPPER,
    get_z1_hammer_robot_cfg,
)
from src.tasks.hammer.hammer_env_cfg import make_hammer_env_cfg
from src.tasks.hammer.nail_block import (
    NAIL_GOAL_DEPTH,
    NAIL_SUCCESS_THRESHOLD,
    get_nail_block_entity_cfg,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg


# ---------------------------------------------------------------------------
# Actuator configs
# ---------------------------------------------------------------------------


class TestActuatorConfigs:
    def test_standard_arm_stiffness(self):
        assert _Z1_ARM_STANDARD.stiffness == 1000.0

    def test_standard_arm_damping(self):
        assert _Z1_ARM_STANDARD.damping == 100.0

    def test_standard_arm_effort_limit(self):
        assert _Z1_ARM_STANDARD.effort_limit == 30.0

    def test_standard_arm_armature(self):
        assert _Z1_ARM_STANDARD.armature == 0.01

    def test_standard_arm_covers_5_joints(self):
        assert len(_Z1_ARM_STANDARD.target_names_expr) == 5

    def test_standard_arm_excludes_joint2(self):
        assert "joint2" not in _Z1_ARM_STANDARD.target_names_expr

    def test_j2_stiffness(self):
        assert _Z1_ARM_J2.stiffness == 1500.0

    def test_j2_damping(self):
        assert _Z1_ARM_J2.damping == 150.0

    def test_j2_effort_limit(self):
        assert _Z1_ARM_J2.effort_limit == 60.0

    def test_j2_armature(self):
        assert _Z1_ARM_J2.armature == 0.02

    def test_j2_targets_only_joint2(self):
        assert _Z1_ARM_J2.target_names_expr == ("joint2",)

    def test_gripper_stiffness(self):
        # Softened 1000->100 (2026-06-16): the gripper is vestigial (hammer is rigidly
        # attached to ee_center_body, policy never actuates the gripper), and the stiff
        # actuator chattered numerically on the near-limit joint once the arm became
        # gravity-compensated, leaving a phantom ~-4.59 rad/s in the gripper qvel obs.
        assert _Z1_GRIPPER.stiffness == 100.0

    def test_gripper_damping(self):
        assert _Z1_GRIPPER.damping == 20.0

    def test_gripper_effort_limit(self):
        assert _Z1_GRIPPER.effort_limit == 30.0

    def test_gripper_armature_smaller_than_arm(self):
        assert _Z1_GRIPPER.armature < _Z1_ARM_STANDARD.armature

    def test_gripper_armature_value(self):
        assert _Z1_GRIPPER.armature == 0.005

    def test_all_are_builtin_position(self):
        for cfg in (_Z1_ARM_STANDARD, _Z1_ARM_J2, _Z1_GRIPPER):
            assert isinstance(cfg, BuiltinPositionActuatorCfg)


# ---------------------------------------------------------------------------
# Articulation config
# ---------------------------------------------------------------------------


class TestArticulationConfig:
    def test_has_3_actuator_groups(self):
        assert len(Z1_ARTICULATION.actuators) == 3

    def test_all_groups_cover_7_joints(self):
        all_targets = []
        for act in Z1_ARTICULATION.actuators:
            all_targets.extend(act.target_names_expr)
        # 5 standard + 1 joint2 + 1 gripper = 7
        assert len(all_targets) == 7


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------


class TestNamedConstants:
    def test_arm_joint_names_count(self):
        assert len(ARM_JOINT_NAMES) == 6

    def test_arm_joint_names_correct(self):
        assert set(ARM_JOINT_NAMES) == {
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        }

    def test_arm_actuator_names_match_joint_names(self):
        assert set(ARM_ACTUATOR_NAMES) == set(ARM_JOINT_NAMES)

    def test_arm_actuator_names_excludes_gripper(self):
        assert GRIPPER_JOINT_NAME not in ARM_ACTUATOR_NAMES

    def test_arm_actuator_names_has_6_entries(self):
        assert len(ARM_ACTUATOR_NAMES) == 6

    def test_ee_site_name(self):
        assert EE_SITE_NAME == "ee_center_site"

    def test_hammer_head_site_name(self):
        assert HAMMER_HEAD_SITE_NAME == "hammer_head_site"

    def test_delta_pos_scale(self):
        assert Z1_HAMMER_DELTA_POS_SCALE == 0.05

    def test_neutral_joint_pos_has_all_joints(self):
        expected = set(ARM_JOINT_NAMES) | {GRIPPER_JOINT_NAME}
        assert set(NEUTRAL_JOINT_POS.keys()) == expected

    def test_neutral_joint_pos_values_in_radians(self):
        # All values should be in a plausible joint range [-π, π]
        import math
        for name, val in NEUTRAL_JOINT_POS.items():
            assert abs(val) <= math.pi * 2, (
                f"Joint {name} value {val} seems too large for radians"
            )


# ---------------------------------------------------------------------------
# Nail block constants
# ---------------------------------------------------------------------------


class TestNailBlockConstants:
    def test_goal_depth(self):
        # 0.075 -> 0.032 (2026-06-10, geometry fix 6a: head flush with block top).
        assert NAIL_GOAL_DEPTH == pytest.approx(0.032)

    def test_success_threshold(self):
        # Re-pinned 0.030 -> 0.027 (2026-06-17, real claw-hammer): best clean single
        # strike = 28.3 mm < old 0.030, so success must be single-strike-reachable.
        assert NAIL_SUCCESS_THRESHOLD == pytest.approx(0.027)

    def test_threshold_less_than_goal(self):
        assert NAIL_SUCCESS_THRESHOLD < NAIL_GOAL_DEPTH

    def test_threshold_close_to_goal(self):
        # Success should still mean "mostly driven" (>= 80% of full depth). Relaxed from
        # 90% on 2026-06-17 because the real hammer's best single strike (88% of goal)
        # leaves no room above the 90% line; 0.027 = 84% of goal.
        assert NAIL_SUCCESS_THRESHOLD >= 0.8 * NAIL_GOAL_DEPTH


# ---------------------------------------------------------------------------
# EntityCfg factory functions
# ---------------------------------------------------------------------------


class TestEntityCfgFactories:
    def test_get_z1_robot_cfg_returns_fresh_instances(self):
        cfg1 = get_z1_hammer_robot_cfg()
        cfg2 = get_z1_hammer_robot_cfg()
        assert cfg1 is not cfg2, "Factory must return new instances to avoid mutation bugs"

    def test_get_nail_block_cfg_returns_fresh_instances(self):
        cfg1 = get_nail_block_entity_cfg()
        cfg2 = get_nail_block_entity_cfg()
        assert cfg1 is not cfg2

    def test_z1_robot_cfg_has_articulation(self):
        cfg = get_z1_hammer_robot_cfg()
        assert cfg.articulation is not None

    def test_nail_block_cfg_has_no_articulation(self):
        # Nail block has no BuiltinActuators — it relies on XmlActuator via articulation=None
        cfg = get_nail_block_entity_cfg()
        assert cfg.articulation is None


# ---------------------------------------------------------------------------
# Base env config structure
# ---------------------------------------------------------------------------


class TestBaseEnvCfgStructure:
    def test_observation_groups(self):
        cfg = make_hammer_env_cfg()
        assert "actor" in cfg.observations
        assert "critic" in cfg.observations

    def test_actor_obs_terms(self):
        cfg = make_hammer_env_cfg()
        actor_terms = cfg.observations["actor"].terms
        for key in ("joint_pos", "joint_vel", "ee_pos", "ee_vel",
                    "head_pos", "head_vel", "nail_top_pos", "nail_depth", "actions"):
            assert key in actor_terms, f"Missing actor obs term: '{key}'"

    def test_critic_obs_terms_match_actor(self):
        cfg = make_hammer_env_cfg()
        actor_keys = set(cfg.observations["actor"].terms.keys())
        critic_keys = set(cfg.observations["critic"].terms.keys())
        assert actor_keys == critic_keys

    def test_action_key(self):
        cfg = make_hammer_env_cfg()
        assert "ik_hammer_head" in cfg.actions

    def test_action_is_differential_ik(self):
        cfg = make_hammer_env_cfg()
        assert isinstance(cfg.actions["ik_hammer_head"], DifferentialIKActionCfg)

    def test_reward_keys(self):
        cfg = make_hammer_env_cfg()
        for key in ("approach", "nail_driven", "action_rate", "joint_pos_limits"):
            assert key in cfg.rewards, f"Missing reward term: '{key}'"

    def test_termination_keys(self):
        cfg = make_hammer_env_cfg()
        assert "time_out" in cfg.terminations
        assert "nail_driven" in cfg.terminations

    def test_time_out_is_time_out_type(self):
        cfg = make_hammer_env_cfg()
        assert cfg.terminations["time_out"].time_out is True

    def test_reward_weights_sign(self):
        cfg = make_hammer_env_cfg()
        assert cfg.rewards["approach"].weight > 0
        assert cfg.rewards["nail_driven"].weight > 0
        assert cfg.rewards["action_rate"].weight < 0
        assert cfg.rewards["joint_pos_limits"].weight < 0

    def test_event_keys(self):
        cfg = make_hammer_env_cfg()
        assert "reset_robot_joints" in cfg.events
        assert "reset_nail" in cfg.events

    def test_decimation(self):
        cfg = make_hammer_env_cfg()
        # 10 × 0.002 s = 0.02 s control period
        assert cfg.decimation == 10

    def test_episode_length(self):
        cfg = make_hammer_env_cfg()
        assert cfg.episode_length_s == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Z1-specific env config wiring
# ---------------------------------------------------------------------------


class TestZ1EnvCfgWiring:
    def test_ik_action_actuator_names(self):
        cfg = z1_hammer_env_cfg()
        ik = cfg.actions["ik_hammer_head"]
        assert ik.actuator_names == ARM_ACTUATOR_NAMES

    def test_ik_action_frame_name(self):
        cfg = z1_hammer_env_cfg()
        ik = cfg.actions["ik_hammer_head"]
        assert ik.frame_name == HAMMER_HEAD_SITE_NAME

    def test_ik_action_delta_pos_scale(self):
        cfg = z1_hammer_env_cfg()
        ik = cfg.actions["ik_hammer_head"]
        assert ik.delta_pos_scale == pytest.approx(Z1_HAMMER_DELTA_POS_SCALE)

    def test_ik_action_uses_relative_mode(self):
        cfg = z1_hammer_env_cfg()
        ik = cfg.actions["ik_hammer_head"]
        assert ik.use_relative_mode is True

    def test_ik_action_position_only(self):
        cfg = z1_hammer_env_cfg()
        ik = cfg.actions["ik_hammer_head"]
        assert ik.orientation_weight == pytest.approx(0.0)

    def test_viewer_body_name(self):
        cfg = z1_hammer_env_cfg()
        assert cfg.viewer.body_name == "link00"

    def test_ee_obs_site_name_wired_in_actor(self):
        cfg = z1_hammer_env_cfg()
        actor_terms = cfg.observations["actor"].terms
        ee_site = actor_terms["ee_pos"].params["asset_cfg"].site_names
        assert ee_site == (EE_SITE_NAME,)

    def test_ee_obs_site_name_wired_in_critic(self):
        cfg = z1_hammer_env_cfg()
        critic_terms = cfg.observations["critic"].terms
        ee_site = critic_terms["ee_pos"].params["asset_cfg"].site_names
        assert ee_site == (EE_SITE_NAME,)

    def test_head_obs_site_name_wired(self):
        cfg = z1_hammer_env_cfg()
        actor_terms = cfg.observations["actor"].terms
        head_site = actor_terms["head_pos"].params["asset_cfg"].site_names
        assert head_site == (HAMMER_HEAD_SITE_NAME,)

    def test_approach_reward_site_name_wired(self):
        cfg = z1_hammer_env_cfg()
        site_names = cfg.rewards["approach"].params["robot_cfg"].site_names
        assert site_names == (HAMMER_HEAD_SITE_NAME,)

    def test_entities_registered(self):
        cfg = z1_hammer_env_cfg()
        assert "robot" in cfg.scene.entities
        assert "nail_block" in cfg.scene.entities

    def test_robot_entity_has_articulation(self):
        cfg = z1_hammer_env_cfg()
        robot_cfg = cfg.scene.entities["robot"]
        assert robot_cfg.articulation is not None


# ---------------------------------------------------------------------------
# Play mode overrides
# ---------------------------------------------------------------------------


class TestZ1EnvCfgPlayMode:
    def test_play_mode_disables_actor_corruption(self):
        cfg = z1_hammer_env_cfg(play=True)
        actor_group = cfg.observations["actor"]
        assert isinstance(actor_group, ObservationGroupCfg)
        assert actor_group.enable_corruption is False

    def test_play_mode_disables_critic_corruption(self):
        cfg = z1_hammer_env_cfg(play=True)
        critic_group = cfg.observations["critic"]
        assert critic_group.enable_corruption is False

    def test_play_mode_large_episode_length(self):
        cfg = z1_hammer_env_cfg(play=True)
        assert cfg.episode_length_s > 1000

    def test_train_mode_actor_has_corruption_enabled(self):
        cfg = z1_hammer_env_cfg(play=False)
        actor_group = cfg.observations["actor"]
        assert actor_group.enable_corruption is True
