"""Layer 2 — Python config construction validation.

Tests that all config objects are correctly constructed and wired.
No MuJoCo model compilation, no mjlab env creation needed.
"""

import copy
import dataclasses
import inspect
import pickle

import pytest
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

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
from src.tasks.hammer.config.z1.env_cfgs import (
    I_REF_DELIVERED,
    I_REF_FIRST_STRIKE_SUCCESS,
    _guideline_observation,
    z1_hammer_env_cfg,
)
from src.tasks.hammer.mdp.first_strike import FirstStrikeEventTracker
from src.tasks.hammer.mdp.guideline import (
    WaypointProgressTracker,
    completed_gate_fraction,
    guideline_perpendicular_error,
    next_gate_vector,
    ordered_gate_progress_reward,
    ordered_waypoint_progress_reward,
    waypoint_progress_state,
)
from src.tasks.hammer.mdp.rewards import (
    DeliveredImpulseTerm,
    FirstStrikeDeliveredRewardTerm,
    FirstStrikeBoundedImpactRewardTerm,
    FirstStrikeImpactRewardTerm,
    FirstStrikeQualityDeliveredRewardTerm,
    FirstStrikeQualityImpactRewardTerm,
    FirstStrikeLegacyDeliveredRewardTerm,
    FirstStrikeLegacyImpactRewardTerm,
    ImpactProgressTerm,
)


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

    def test_arm_joint_names_single_source_no_drift(self):
        # ORDER-sensitive: IMP_J_LIMIT (and every per-joint metric/cap) keys on this tuple's
        # POSITION, so a silent divergence between the two physical definitions would misassign
        # joint2's 2x cap. velocity_bound._ARM_CFG feeds the accumulators/hook/contact-row;
        # z1_constants.ARM_JOINT_NAMES feeds env_cfgs. Pin them equal (ORDER, not just set) so a
        # rename/reorder in one cannot drift from the other. (2026-07-14 dedup: contact_row and the
        # env_cfgs imp_peak loop now derive from these instead of hardcoding their own copies.)
        from src.assets.robots.unitree_z1.z1_constants import _Z1_VELOCITY_LIMIT
        from src.tasks.hammer.mdp.velocity_bound import Z1_JOINT_VEL_LIMIT, _ARM_CFG
        assert tuple(_ARM_CFG.joint_names) == tuple(ARM_JOINT_NAMES)
        # Same drift guard for the scalar Z1 joint-velocity limit (two physical definitions).
        assert Z1_JOINT_VEL_LIMIT == _Z1_VELOCITY_LIMIT

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
        # 0.05 -> 0.15 (2026-06-17): raised for a genuine ~1.35 m/s strike (terminal
        # head speed ~linear in this scale); self-limiting under the 3.1415 rad/s joint
        # limit (peak ~2.41 rad/s). See the strike-not-press design record.
        assert Z1_HAMMER_DELTA_POS_SCALE == 0.15

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
        # Re-pinned 0.027 -> 0.030 (2026-07-15, post neck-fix): the strengthened reference
        # drives the nail to its 0.032 cap in a single strike (playback_reference PASS at
        # 0.030, all heights), so the bar rises to ~0.94x cap and stays single-strike-
        # reachable. Lower again if a face-only retrain caps below ~30 mm.
        assert NAIL_SUCCESS_THRESHOLD == pytest.approx(0.030)

    def test_threshold_less_than_goal(self):
        assert NAIL_SUCCESS_THRESHOLD < NAIL_GOAL_DEPTH

    def test_threshold_close_to_goal(self):
        # Success should mean "mostly driven" (>= 80% of full depth). 0.030 = 94% of goal:
        # the strengthened reference reaches the 0.032 cap, so the bar no longer sits low.
        # The >= 80% floor is kept loose to allow lowering if a face-only retrain caps low.
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

    def test_no_terminate_variant(self):
        # Option B: no_terminate pops the success termination + retunes the un-latched completion.
        cfg = z1_hammer_env_cfg(cat_impulse=True, no_terminate=True)
        base = z1_hammer_env_cfg(cat_impulse=True)
        assert "nail_driven" not in cfg.terminations
        assert "time_out" in cfg.terminations and cfg.terminations["time_out"].time_out is True
        assert cfg.rewards["completion"].weight == pytest.approx(1.0)
        # nail_driven reward stays; no hard 2.0 pin (Option A may later promote 0.5 as the code default).
        assert cfg.rewards["nail_driven"].weight == base.rewards["nail_driven"].weight
        assert "nail_driven" in base.terminations and base.rewards["completion"].weight == pytest.approx(100.0)

    def test_nail_driven_not_farmable(self):
        # ANTI-FARM invariant (2026-07-16): nail_driven is a per-step Gaussian on the RATCHETING nail
        # depth, so holding just below the 0.030 success line pays w_nd*G(d) FOREVER, while completing
        # terminates + forfeits that stream for the one-time completion bonus. The weight must be low
        # enough that the max discounted sub-threshold hold value stays below completion, else the mean
        # policy parks (the Vega nf1 bug at weight 2.0: hold ~205 > completion 100). Cut to 0.5 fixed it.
        import math
        cfg = z1_hammer_env_cfg(cat_impulse=True)
        w_nd = cfg.rewards["nail_driven"].weight
        w_ap = cfg.rewards["approach"].weight          # approach is also per-step (farmable)
        w_cp = cfg.rewards["completion"].weight
        goal = cfg.rewards["nail_driven"].params["goal_depth"]
        std = cfg.rewards["nail_driven"].params["std"]
        thresh = cfg.terminations["nail_driven"].params["success_depth"]
        gamma = 0.99  # rl_cfg.py; effective horizon 1/(1-gamma)=100 steps
        # Worst case: the Gaussian is largest at the sub-threshold depth CLOSEST to its 0.032 centre,
        # i.e. d -> thresh (0.030). Hold-forever discounted value must stay below completion.
        g_max = math.exp(-((goal - thresh) ** 2) / (std ** 2))
        hold_value = (w_nd * g_max + w_ap) / (1.0 - gamma)
        assert hold_value < w_cp, (
            f"nail_driven weight {w_nd} is FARMABLE: sub-threshold hold value {hold_value:.1f} "
            f">= completion {w_cp} -> the policy will park below {thresh} instead of completing."
        )

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


class TestArmCompositionGuards:
    def test_vel_penalty_cannot_compose_with_soft_cat_hook_arms(self):
        # 2026-07-14 audit: vel_penalty's vel_excess weight starts 0.0 and is curriculum-ramped
        # NEGATIVE at step 1500, bypassing CatSoftHook's one-shot _NEG_TERMS guard — the
        # penalty-evasion exploit Decision 1 forbids. The factory must refuse the composition.
        with pytest.raises(ValueError, match="vel_penalty"):
            z1_hammer_env_cfg(vel_penalty=True, cat_soft=True)
        with pytest.raises(ValueError, match="vel_penalty"):
            z1_hammer_env_cfg(vel_penalty=True, cat_impulse=True)
        # The lone arms stay constructible.
        z1_hammer_env_cfg(vel_penalty=True)
        z1_hammer_env_cfg(cat_soft=True)

    def test_event_correct_requires_impulse_arm(self):
        with pytest.raises(ValueError, match="event_correct.*cat_impulse"):
            z1_hammer_env_cfg(event_correct=True)

    def test_event_linear_requires_event_correct_arm(self):
        with pytest.raises(ValueError, match="event_linear.*event_correct"):
            z1_hammer_env_cfg(cat_impulse=True, event_linear=True)

    def test_event_quality_requires_event_correct_and_instrumentation(self):
        with pytest.raises(ValueError, match="event_quality.*event_correct"):
            z1_hammer_env_cfg(
                cat_impulse=True,
                event_quality=True,
                quality_instrumentation=True,
            )
        with pytest.raises(ValueError, match="event_quality.*quality_instrumentation"):
            z1_hammer_env_cfg(cat_impulse=True, event_correct=True, event_quality=True)

    def test_event_quality_and_linear_are_separate_treatments(self):
        with pytest.raises(ValueError, match="event_quality.*event_linear"):
            z1_hammer_env_cfg(
                cat_impulse=True,
                event_correct=True,
                event_linear=True,
                event_quality=True,
                quality_instrumentation=True,
            )

    def test_first_strike_legacy_rejects_event_composition(self):
        with pytest.raises(ValueError, match="first_strike_legacy.*event_correct"):
            z1_hammer_env_cfg(
                cat_impulse=True, event_correct=True, first_strike_legacy=True
            )
        with pytest.raises(ValueError, match="first_strike_legacy.*event_linear"):
            z1_hammer_env_cfg(
                cat_impulse=True, event_correct=True, event_linear=True,
                first_strike_legacy=True,
            )

    def test_event_flag_preserves_legacy_positional_dcmotor_slot(self):
        cfg = z1_hammer_env_cfg(
            False, False, False, False, False, False, False, False, True
        )
        actuator_types = [
            type(act).__name__
            for act in cfg.scene.entities["robot"].articulation.actuators
        ]
        assert actuator_types[:2] == ["DcMotorActuatorCfg", "DcMotorActuatorCfg"]

    def test_linear_flag_is_appended_after_legacy_event_positional_slot(self):
        cfg = z1_hammer_env_cfg(
            False, False, False, False, False, False, False, True, False, False, True
        )
        assert "first_strike" in cfg.metrics
        assert cfg.rewards["delivered_impulse"].params["saturate"] is True

    def test_gate_reward_requires_guideline_tracker(self):
        with pytest.raises(ValueError, match="gate_reward.*guideline"):
            z1_hammer_env_cfg(gate_reward=True)

    def test_guideline_requires_first_strike_tracker(self):
        with pytest.raises(ValueError, match="guideline.*first_strike"):
            z1_hammer_env_cfg(guideline=True)


class TestCartesianGuidelineStudy:
    _C0 = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0"
    _C_GATE = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate"
    _C_PROGRESS = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress"
    _F8 = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear"
    _GUIDELINE_OBSERVATIONS = {
        "next_gate_vector": (next_gate_vector, 3),
        "completed_gate_fraction": (completed_gate_fraction, 1),
        "guideline_perpendicular_error": (guideline_perpendicular_error, 1),
        "waypoint_progress_state": (waypoint_progress_state, 2),
    }
    _F8_REWARDS = {
        "approach": 0.1,
        "nail_driven": 0.5,
        "nail_depth_delta": 600.0,
        "impact_progress": 8.0,
        "completion": 100.0,
        "action_rate": -0.01,
        "joint_pos_limits": -10.0,
        "delivered_impulse": 2.0,
    }
    _LEGACY_FACTORY_PARAMETERS = (
        "play",
        "imitation",
        "vel_penalty",
        "cat_vel",
        "cat_substep",
        "vel_hard_term",
        "cat_soft",
        "cat_impulse",
        "dcmotor",
        "no_terminate",
        "event_correct",
        "event_linear",
        "first_strike_legacy",
        "event_quality",
        "quality_instrumentation",
    )
    _LEGACY_TASK_IDS = {
        "Unitree-Z1-Hammer",
        "Unitree-Z1-Hammer-Track",
        "Unitree-Z1-Hammer-VPenalty",
        "Unitree-Z1-Hammer-CaT",
        "Unitree-Z1-Hammer-DcMotor",
        "Unitree-Z1-Hammer-CaT-Substep",
        "Unitree-Z1-Hammer-VelHardTerm",
        "Unitree-Z1-Hammer-CaT-Soft",
        "Unitree-Z1-Hammer-CaT-Impulse",
        "Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy",
        "Unitree-Z1-Hammer-CaT-Impulse-Event",
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
        "Unitree-Z1-Hammer-CaT-Impulse-Track",
        "Unitree-Z1-Hammer-CaT-Impulse-NoTerm",
    }

    @staticmethod
    def _actuator_signature(cfg):
        return tuple(
            (
                type(act).__name__,
                act.stiffness,
                act.damping,
                act.effort_limit,
                act.armature,
                tuple(act.target_names_expr),
            )
            for act in cfg.scene.entities["robot"].articulation.actuators
        )

    @staticmethod
    def _normalize(value):
        if dataclasses.is_dataclass(value):
            return {
                field.name: TestCartesianGuidelineStudy._normalize(
                    getattr(value, field.name)
                )
                for field in dataclasses.fields(value)
            }
        if isinstance(value, dict):
            return {
                key: TestCartesianGuidelineStudy._normalize(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return tuple(
                TestCartesianGuidelineStudy._normalize(item) for item in value
            )
        if callable(value):
            return (value.__module__, value.__qualname__)
        return value

    def test_factory_appends_flags_without_changing_legacy_signature(self):
        parameters = inspect.signature(z1_hammer_env_cfg).parameters
        assert tuple(parameters) == self._LEGACY_FACTORY_PARAMETERS + (
            "guideline",
            "gate_reward",
            "progress_reward",
        )
        assert all(
            parameters[name].default is False
            for name in self._LEGACY_FACTORY_PARAMETERS
        )
        assert parameters["guideline"].default is False
        assert parameters["gate_reward"].default is False
        assert parameters["progress_reward"].default is False

    def test_progress_reward_requires_guideline_tracker(self):
        with pytest.raises(ValueError, match="progress_reward.*guideline"):
            z1_hammer_env_cfg(progress_reward=True)

    def test_new_registration_adds_only_three_task_ids(self):
        registered_z1 = {
            task for task in list_tasks() if task.startswith("Unitree-Z1-Hammer")
        }
        assert registered_z1 == self._LEGACY_TASK_IDS | {
            self._C0,
            self._C_GATE,
            self._C_PROGRESS,
        }

    @pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
    @pytest.mark.parametrize("task_id", (_C0, _C_GATE, _C_PROGRESS))
    def test_all_arms_share_exact_guideline_state_and_f8_contract(
        self, task_id, play
    ):
        cfg = load_env_cfg(task_id, play=play)
        f8 = load_env_cfg(self._F8, play=play)

        for group_name in ("actor", "critic"):
            terms = cfg.observations[group_name].terms
            added = set(terms) - set(f8.observations[group_name].terms)
            assert added == set(self._GUIDELINE_OBSERVATIONS)
            for name, (reader, width) in self._GUIDELINE_OBSERVATIONS.items():
                assert terms[name].func is _guideline_observation
                assert terms[name].params == {"reader": reader, "width": width}

        added_metrics = set(cfg.metrics) - set(f8.metrics)
        assert added_metrics == {"waypoint_progress"}
        metric_names = list(cfg.metrics)
        assert metric_names.index("waypoint_progress") == metric_names.index("first_strike") + 1
        waypoint = cfg.metrics["waypoint_progress"]
        assert waypoint.func is WaypointProgressTracker
        assert waypoint.per_substep is True
        assert waypoint.params["robot_cfg"].site_names == (HAMMER_HEAD_SITE_NAME,)
        assert waypoint.params["nail_cfg"].site_names == ("nail_top",)

        expected_reward_keys = set(self._F8_REWARDS)
        if task_id == self._C_GATE:
            expected_reward_keys.add("r_gate")
        if task_id == self._C_PROGRESS:
            expected_reward_keys.add("r_waypoint_progress")
        assert set(cfg.rewards) == expected_reward_keys
        assert {
            name: cfg.rewards[name].weight for name in self._F8_REWARDS
        } == self._F8_REWARDS
        for name in self._F8_REWARDS:
            assert cfg.rewards[name].func is f8.rewards[name].func
            assert cfg.rewards[name].params == f8.rewards[name].params
        assert "r_imit" not in cfg.rewards

        assert set(cfg.actions) == {"ik_hammer_head"}
        ik = cfg.actions["ik_hammer_head"]
        assert isinstance(ik, DifferentialIKActionCfg)
        assert repr(cfg.actions) == repr(f8.actions)
        assert ik.delta_pos_scale == pytest.approx(0.15)
        assert ik.orientation_weight == pytest.approx(0.0)
        assert self._actuator_signature(cfg) == self._actuator_signature(f8) == (
            (
                "BuiltinPositionActuatorCfg",
                1000.0,
                100.0,
                30.0,
                0.01,
                ("joint1", "joint3", "joint4", "joint5", "joint6"),
            ),
            ("BuiltinPositionActuatorCfg", 1500.0, 150.0, 60.0, 0.02, ("joint2",)),
            ("BuiltinPositionActuatorCfg", 100.0, 20.0, 30.0, 0.005, ("jointGripper",)),
        )
        assert cfg.metrics["cat_soft"].params["imp_max_p"] == 0.0
        assert tuple(cfg.metrics["cat_soft"].params["imp_limit"]) == (
            1.64,
            3.28,
            1.64,
            1.64,
            1.64,
            1.64,
        )
        assert cfg.metrics["cat_soft"].params["use_vel"] is False
        assert "cat_vel" not in cfg.terminations
        assert "r_imit" not in cfg.rewards
        assert "set_gains" not in cfg.actions

    def test_constructor_probe_is_narrow_and_runtime_reader_still_fails_closed(self):
        from types import SimpleNamespace

        cfg = load_env_cfg(self._C0)
        term = cfg.observations["actor"].terms["next_gate_vector"]
        constructing = SimpleNamespace(num_envs=2, device="cpu")
        assert term.func(constructing, **term.params).shape == (2, 3)
        assert torch.count_nonzero(term.func(constructing, **term.params)) == 0

        runtime_without_tracker = SimpleNamespace(
            num_envs=2, device="cpu", observation_manager=object()
        )
        with pytest.raises(RuntimeError, match="needs WaypointProgressTracker"):
            term.func(runtime_without_tracker, **term.params)

    def test_guideline_observation_term_survives_pickle_round_trip(self):
        cfg = load_env_cfg(self._C0)
        term = cfg.observations["actor"].terms["next_gate_vector"]
        restored = pickle.loads(pickle.dumps(term))
        assert restored.func is term.func
        assert restored.params["reader"] is next_gate_vector
        assert restored.params["width"] == 3

    def test_only_cgate_enables_ordered_gate_reward(self):
        c0 = load_env_cfg(self._C0)
        cgate = load_env_cfg(self._C_GATE)
        assert "r_gate" not in c0.rewards
        reward = cgate.rewards["r_gate"]
        assert reward.func is ordered_gate_progress_reward
        assert reward.weight == pytest.approx(8.0)
        assert reward.params == {}

    def test_only_cprogress_enables_ordered_waypoint_progress_reward(self):
        c0 = load_env_cfg(self._C0)
        cprogress = load_env_cfg(self._C_PROGRESS)
        assert "r_waypoint_progress" not in c0.rewards
        reward = cprogress.rewards["r_waypoint_progress"]
        assert reward.func is ordered_waypoint_progress_reward
        assert reward.weight == pytest.approx(8.0)
        assert reward.params == {}

    @pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
    def test_c0_and_cgate_whole_configs_differ_only_by_gate_reward(self, play):
        c0 = load_env_cfg(self._C0, play=play)
        cgate = load_env_cfg(self._C_GATE, play=play)
        cgate.rewards.pop("r_gate")
        assert self._normalize(cgate) == self._normalize(c0)

    @pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
    def test_c0_and_cprogress_whole_configs_differ_only_by_waypoint_reward(self, play):
        c0 = load_env_cfg(self._C0, play=play)
        cprogress = load_env_cfg(self._C_PROGRESS, play=play)
        cprogress.rewards.pop("r_waypoint_progress")
        assert self._normalize(cprogress) == self._normalize(c0)

    @pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
    def test_cgate_and_cprogress_exchange_only_their_reward_readers(self, play):
        cgate = load_env_cfg(self._C_GATE, play=play)
        cprogress = load_env_cfg(self._C_PROGRESS, play=play)
        cgate.rewards.pop("r_gate")
        cprogress.rewards.pop("r_waypoint_progress")
        assert self._normalize(cgate) == self._normalize(cprogress)

    @pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
    def test_c0_is_f8_plus_only_shared_guideline_state(self, play):
        c0 = load_env_cfg(self._C0, play=play)
        f8 = load_env_cfg(self._F8, play=play)
        for group in c0.observations.values():
            for name in self._GUIDELINE_OBSERVATIONS:
                group.terms.pop(name)
        c0.metrics.pop("waypoint_progress")
        c0.events["reset_robot_joints"].params["position_range"] = (
            f8.events["reset_robot_joints"].params["position_range"]
        )
        assert self._normalize(c0) == self._normalize(f8)

    @pytest.mark.parametrize("play", (False, True), ids=("train", "play"))
    @pytest.mark.parametrize("task_id", (_C0, _C_GATE, _C_PROGRESS))
    def test_guideline_arms_use_a_fixed_nominal_joint_reset(self, task_id, play):
        cfg = load_env_cfg(task_id, play=play)
        assert cfg.events["reset_robot_joints"].params["position_range"] == (
            0.0,
            0.0,
        )

    def test_legacy_f8_remains_exact(self):
        f8 = load_env_cfg(self._F8)
        assert f8.events["reset_robot_joints"].params["position_range"] == (
            -0.05,
            0.05,
        )
        assert set(f8.rewards) == set(self._F8_REWARDS)
        assert {name: term.weight for name, term in f8.rewards.items()} == self._F8_REWARDS
        assert set(f8.observations["actor"].terms) == {
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
        }
        assert set(f8.observations["critic"].terms) == set(
            f8.observations["actor"].terms
        )
        assert "waypoint_progress" not in f8.metrics
        assert "r_gate" not in f8.rewards
        assert "r_imit" not in f8.rewards
        assert f8.metrics["cat_soft"].params["imp_max_p"] == 0.0


class TestFirstStrikeEventArm:
    def test_registered_event_task_is_isolated_from_legacy_arm(self):
        task_id = "Unitree-Z1-Hammer-CaT-Impulse-Event"
        assert task_id in list_tasks()
        event = load_env_cfg(task_id)
        legacy = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse")

        tracker_cfg = event.metrics["first_strike"]
        assert tracker_cfg.func is FirstStrikeEventTracker
        assert tracker_cfg.per_substep is True
        assert tracker_cfg.params["contact_sensor_name"] == "hammer_nail_contact"
        assert tracker_cfg.params["impulse_sensor_name"] == "hammer_nail_impulse"
        assert (
            tracker_cfg.params["contact_sensor_name"]
            != tracker_cfg.params["impulse_sensor_name"]
        )
        assert event.rewards["impact_progress"].func is FirstStrikeImpactRewardTerm
        assert event.rewards["delivered_impulse"].func is FirstStrikeDeliveredRewardTerm
        assert event.rewards["delivered_impulse"].params["saturate"] is True

        assert "first_strike" not in legacy.metrics
        assert legacy.rewards["impact_progress"].func is ImpactProgressTerm
        assert legacy.rewards["delivered_impulse"].func is DeliveredImpulseTerm
        assert legacy.rewards["delivered_impulse"].params["i_ref"] == pytest.approx(0.6094)
        assert legacy.rewards["delivered_impulse"].params["i_ref"] == I_REF_DELIVERED
        assert "saturate" not in legacy.rewards["delivered_impulse"].params

        # The event horizon ends inclusively at first success, before the legacy
        # full contact-window tail. Its independently provisioned default-reference
        # normalizer therefore must not alias the legacy 0.6094 N.s value.
        assert I_REF_FIRST_STRIKE_SUCCESS == pytest.approx(0.3088)
        assert event.rewards["delivered_impulse"].params["i_ref"] == pytest.approx(0.3088)
        assert (
            event.rewards["delivered_impulse"].params["i_ref"]
            == I_REF_FIRST_STRIKE_SUCCESS
        )
        assert I_REF_FIRST_STRIKE_SUCCESS != I_REF_DELIVERED

        # The event treatment leaves enforcement and plant authority unchanged.
        assert event.metrics["cat_soft"].params["imp_max_p"] == 0.0
        assert (
            event.metrics["cat_soft"].params["imp_limit"]
            == legacy.metrics["cat_soft"].params["imp_limit"]
        )
        assert (
            event.actions["ik_hammer_head"].delta_pos_scale
            == legacy.actions["ik_hammer_head"].delta_pos_scale
        )
        event_gains = [
            (act.stiffness, act.damping)
            for act in event.scene.entities["robot"].articulation.actuators
        ]
        legacy_gains = [
            (act.stiffness, act.damping)
            for act in legacy.scene.entities["robot"].articulation.actuators
        ]
        assert event_gains == legacy_gains

    def test_quality_instrumentation_is_passive_for_event_reward_and_policy_config(self):
        base = z1_hammer_env_cfg(
            cat_impulse=True, event_correct=True, event_linear=True
        )
        instrumented = z1_hammer_env_cfg(
            cat_impulse=True,
            event_correct=True,
            event_linear=True,
            quality_instrumentation=True,
        )

        assert "hammer_nail_quality" not in {sensor.name for sensor in base.scene.sensors}
        assert "quality_sensor_name" not in base.metrics["first_strike"].params
        assert "hammer_nail_quality" in {sensor.name for sensor in instrumented.scene.sensors}
        assert (
            instrumented.metrics["first_strike"].params["quality_sensor_name"]
            == "hammer_nail_quality"
        )
        assert repr(base.actions) == repr(instrumented.actions)
        assert [
            (act.stiffness, act.damping, act.effort_limit, act.armature)
            for act in base.scene.entities["robot"].articulation.actuators
        ] == [
            (act.stiffness, act.damping, act.effort_limit, act.armature)
            for act in instrumented.scene.entities["robot"].articulation.actuators
        ]
        for reward_name in base.rewards:
            assert base.rewards[reward_name].func is instrumented.rewards[reward_name].func
            assert base.rewards[reward_name].weight == instrumented.rewards[reward_name].weight
            assert base.rewards[reward_name].params == instrumented.rewards[reward_name].params

    def test_registered_linear_event_task_changes_only_delivered_payout_shape(self):
        saturated = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse-Event")
        task_id = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear"
        assert task_id in list_tasks()
        linear = load_env_cfg(task_id)

        assert linear.metrics["first_strike"].func is FirstStrikeEventTracker
        assert linear.rewards["impact_progress"].func is FirstStrikeImpactRewardTerm
        assert linear.rewards["delivered_impulse"].func is FirstStrikeDeliveredRewardTerm
        assert linear.rewards["delivered_impulse"].params["saturate"] is False
        assert saturated.rewards["delivered_impulse"].params["saturate"] is True
        assert linear.rewards["impact_progress"].weight == pytest.approx(8.0)
        assert saturated.rewards["impact_progress"].weight == pytest.approx(8.0)
        assert linear.rewards["delivered_impulse"].weight == pytest.approx(2.0)
        assert saturated.rewards["delivered_impulse"].weight == pytest.approx(2.0)
        assert (
            linear.rewards["delivered_impulse"].params["i_ref"]
            == saturated.rewards["delivered_impulse"].params["i_ref"]
            == I_REF_FIRST_STRIKE_SUCCESS
        )

        # Arm F is still the fixed-impedance, log-only treatment: its constraint
        # caps, action authority and physical gains are identical to Arm E.
        assert set(linear.rewards) == set(saturated.rewards)
        assert set(linear.metrics) == set(saturated.metrics)
        assert set(linear.actions) == set(saturated.actions)
        assert linear.metrics["cat_soft"].params["imp_max_p"] == 0.0
        assert (
            linear.metrics["cat_soft"].params["imp_limit"]
            == saturated.metrics["cat_soft"].params["imp_limit"]
        )
        assert (
            linear.actions["ik_hammer_head"].delta_pos_scale
            == saturated.actions["ik_hammer_head"].delta_pos_scale
        )
        linear_gains = [
            (act.stiffness, act.damping)
            for act in linear.scene.entities["robot"].articulation.actuators
        ]
        saturated_gains = [
            (act.stiffness, act.damping)
            for act in saturated.scene.entities["robot"].articulation.actuators
        ]
        assert linear_gains == saturated_gains

    def test_quality_treatments_change_only_the_preregistered_reward_surface(self):
        f8 = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear")
        f0 = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0")
        d0 = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0")
        fq = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse-Event-Quality")

        def params_signature(params):
            return tuple(sorted(params.items(), key=lambda item: item[0]))

        def signature(cfg):
            return {
                "rewards": {
                    name: (
                        term.func,
                        term.weight,
                        params_signature(term.params),
                    )
                    for name, term in cfg.rewards.items()
                },
                "actions": repr(cfg.actions),
                "actuators": tuple(
                    (act.stiffness, act.damping, act.effort_limit, act.armature)
                    for act in cfg.scene.entities["robot"].articulation.actuators
                ),
                "tracker": (
                    cfg.metrics["first_strike"].func,
                    cfg.metrics["first_strike"].per_substep,
                    cfg.metrics["first_strike"].reduce,
                    params_signature(cfg.metrics["first_strike"].params),
                ),
                "sensors": tuple(
                    (sensor.name, sensor.fields, sensor.reduce, sensor.num_slots)
                    for sensor in cfg.scene.sensors
                ),
                "caps": tuple(cfg.metrics["cat_soft"].params["imp_limit"]),
                "cat": params_signature(cfg.metrics["cat_soft"].params),
                "budget": (
                    cfg.episode_length_s,
                    cfg.decimation,
                    cfg.sim.mujoco.timestep,
                ),
            }

        assert {
            "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
            "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
            "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
        }.issubset(set(list_tasks()))
        assert f0.rewards["impact_progress"].weight == pytest.approx(0.0)
        assert f0.rewards["delivered_impulse"].weight == pytest.approx(2.0)
        assert d0.rewards["impact_progress"].weight == pytest.approx(8.0)
        assert d0.rewards["delivered_impulse"].weight == pytest.approx(0.0)
        assert fq.rewards["impact_progress"].weight == 8.0
        assert fq.rewards["delivered_impulse"].weight == 0.0
        assert fq.rewards["impact_progress"].func is FirstStrikeQualityImpactRewardTerm
        assert (
            fq.rewards["impact_progress"].params["v_expected"]
            == 1.4598331451416016
        )
        assert all(
            term.func is not FirstStrikeQualityDeliveredRewardTerm
            for term in fq.rewards.values()
        )

        f8_sig = signature(f8)
        f0_sig = signature(f0)
        d0_sig = signature(d0)
        fq_sig = signature(fq)
        for key in ("actions", "actuators", "caps", "cat", "budget"):
            assert f0_sig[key] == d0_sig[key] == fq_sig[key] == f8_sig[key]
        assert fq_sig["caps"] == (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
        assert all(
            cfg.metrics["cat_soft"].params["imp_max_p"] == 0.0
            for cfg in (f8, f0, d0, fq)
        )

        assert f0_sig["tracker"] == d0_sig["tracker"] == f8_sig["tracker"]
        assert f0_sig["sensors"] == d0_sig["sensors"] == f8_sig["sensors"]

        for altered, reward_name in ((f0_sig, "impact_progress"), (d0_sig, "delivered_impulse")):
            expected = dict(f8_sig["rewards"])
            expected[reward_name] = altered["rewards"][reward_name]
            assert altered["rewards"] == expected

        expected_fq_rewards = dict(d0_sig["rewards"])
        expected_fq_rewards["impact_progress"] = fq_sig["rewards"]["impact_progress"]
        assert fq_sig["rewards"] == expected_fq_rewards

        d0_impact_params = dict(d0_sig["rewards"]["impact_progress"][2])
        fq_impact_params = dict(fq_sig["rewards"]["impact_progress"][2])
        assert d0_impact_params.pop("v_expected") == 1.0
        assert fq_impact_params.pop("v_expected") == 1.4598331451416016
        assert fq_impact_params == d0_impact_params

        d0_tracker = d0.metrics["first_strike"]
        fq_tracker = fq.metrics["first_strike"]
        assert (
            fq_tracker.func,
            fq_tracker.per_substep,
            fq_tracker.reduce,
        ) == (
            d0_tracker.func,
            d0_tracker.per_substep,
            d0_tracker.reduce,
        )
        fq_tracker_params = dict(fq_tracker.params)
        assert fq_tracker_params.pop("quality_sensor_name") == "hammer_nail_quality"
        assert fq_tracker_params == d0_tracker.params

        d0_sensors = {
            name: (fields, reduce, num_slots)
            for name, fields, reduce, num_slots in d0_sig["sensors"]
        }
        fq_sensors = {
            name: (fields, reduce, num_slots)
            for name, fields, reduce, num_slots in fq_sig["sensors"]
        }
        assert fq_sensors.pop("hammer_nail_quality") == (
            ("found", "force", "pos", "normal"),
            "maxforce",
            8,
        )
        assert fq_sensors == d0_sensors

    def test_bounded_and_quality_tasks_differ_only_in_the_impact_reader(self):
        """Any B8/FQ drift beyond their preregistered speed-reader treatment must fail."""
        b8_id = "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded"
        fq_id = "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality"
        v_fq = 1.4598331451416016

        assert {b8_id, fq_id}.issubset(set(list_tasks()))
        b8 = load_env_cfg(b8_id)
        fq = load_env_cfg(fq_id)
        linear = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear")

        assert b8.rewards["impact_progress"].func is FirstStrikeBoundedImpactRewardTerm
        assert fq.rewards["impact_progress"].func is FirstStrikeQualityImpactRewardTerm
        for cfg in (b8, fq):
            assert cfg.rewards["impact_progress"].params["v_expected"] == v_fq
            assert cfg.rewards["impact_progress"].weight == pytest.approx(8.0)
            assert cfg.rewards["delivered_impulse"].weight == pytest.approx(0.0)
            assert cfg.rewards["delivered_impulse"].params["saturate"] is False
            assert cfg.metrics["cat_soft"].params["imp_max_p"] == 0.0
            assert tuple(cfg.metrics["cat_soft"].params["imp_limit"]) == (
                1.64, 3.28, 1.64, 1.64, 1.64, 1.64,
            )
            quality_sensor = next(
                sensor for sensor in cfg.scene.sensors if sensor.name == "hammer_nail_quality"
            )
            assert quality_sensor.num_slots == 8

        def actuator_signature(cfg):
            return tuple(
                (act.stiffness, act.damping, act.effort_limit, act.armature)
                for act in cfg.scene.entities["robot"].articulation.actuators
            )

        assert repr(b8.actions) == repr(fq.actions) == repr(linear.actions)
        assert actuator_signature(b8) == actuator_signature(fq) == actuator_signature(linear)
        assert (
            tuple(b8.metrics["cat_soft"].params["imp_limit"])
            == tuple(fq.metrics["cat_soft"].params["imp_limit"])
            == tuple(linear.metrics["cat_soft"].params["imp_limit"])
        )

        def normalize(value):
            if dataclasses.is_dataclass(value):
                return {
                    field.name: normalize(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                }
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return tuple(normalize(item) for item in value)
            if callable(value):
                return (value.__module__, value.__qualname__)
            return value

        normalized_b8 = copy.deepcopy(b8)
        normalized_fq = copy.deepcopy(fq)
        sentinel = object()
        normalized_b8.rewards["impact_progress"].func = sentinel
        normalized_fq.rewards["impact_progress"].func = sentinel
        assert normalize(normalized_b8) == normalize(normalized_fq)

    def test_registered_first_strike_legacy_task_keeps_legacy_readout(self):
        task_id = "Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy"
        assert task_id in list_tasks()
        dprime = load_env_cfg(task_id)
        control = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse")
        saturated = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse-Event")
        linear = load_env_cfg("Unitree-Z1-Hammer-CaT-Impulse-Event-Linear")

        assert dprime.metrics["first_strike"].func is FirstStrikeEventTracker
        assert dprime.rewards["impact_progress"].func is FirstStrikeLegacyImpactRewardTerm
        assert (
            dprime.rewards["delivered_impulse"].func
            is FirstStrikeLegacyDeliveredRewardTerm
        )
        assert dprime.rewards["impact_progress"].weight == pytest.approx(8.0)
        assert dprime.rewards["delivered_impulse"].weight == pytest.approx(2.0)
        assert dprime.rewards["impact_progress"].params["v_expected"] == pytest.approx(1.0)
        assert dprime.rewards["delivered_impulse"].params["i_ref"] == I_REF_DELIVERED
        assert "saturate" not in dprime.rewards["delivered_impulse"].params

        assert "first_strike" not in control.metrics
        for event in (saturated, linear):
            assert (
                event.rewards["delivered_impulse"].params["i_ref"]
                == I_REF_FIRST_STRIKE_SUCCESS
            )
            assert event.rewards["impact_progress"].weight == pytest.approx(8.0)
            assert event.rewards["delivered_impulse"].weight == pytest.approx(2.0)
        assert saturated.rewards["delivered_impulse"].params["saturate"] is True
        assert linear.rewards["delivered_impulse"].params["saturate"] is False


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


# ---------------------------------------------------------------------------
# Imitation flag (A-TRACK arm)
# ---------------------------------------------------------------------------


class TestImitationFlag:
    def test_base_has_no_r_imit_or_curriculum(self):
        base = z1_hammer_env_cfg(imitation=False)
        assert "r_imit" not in base.rewards
        assert not base.curriculum  # A-BASE unchanged: empty curriculum

    def test_track_adds_only_r_imit(self):
        base = z1_hammer_env_cfg(imitation=False)
        track = z1_hammer_env_cfg(imitation=True)
        # Exactly one new reward term, nothing else changed.
        assert set(track.rewards) - set(base.rewards) == {"r_imit"}
        assert track.rewards["r_imit"].weight == pytest.approx(0.1)

    def test_track_anneal_curriculum_present(self):
        track = z1_hammer_env_cfg(imitation=True)
        assert "r_imit_anneal" in track.curriculum
        stages = track.curriculum["r_imit_anneal"].params["stages"]
        assert stages[0]["weight"] == pytest.approx(0.1)
        assert stages[-1]["weight"] == pytest.approx(0.0)
        assert stages[-1]["step"] == 6000  # 250 iters * 24 steps/iter

    def test_track_head_site_wired(self):
        track = z1_hammer_env_cfg(imitation=True)
        assert track.rewards["r_imit"].params["robot_cfg"].site_names == (HAMMER_HEAD_SITE_NAME,)

    def test_track_play_mode_clears_curriculum(self):
        # Play/validation mode keeps r_imit but drops the anneal (weight fixed at 0.1).
        track_play = z1_hammer_env_cfg(play=True, imitation=True)
        assert "r_imit" in track_play.rewards
        assert not track_play.curriculum
