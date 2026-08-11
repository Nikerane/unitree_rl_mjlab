"""Compiled public contract for the fixed controlled-drop calibration fixture."""

from __future__ import annotations

import inspect
import math

import mujoco
import numpy as np
import pytest

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg

import src.tasks.hammer.calibration as calibration
from src.tasks.hammer.calibration.controlled_drop import run_one_primary_drop
from src.tasks.hammer.mdp.first_strike import FirstStrikeEventTracker
from src.tasks.hammer.nail_block import get_nail_block_entity_cfg


FIC0_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
    "CProgress-Vel-Delivered4-JointPosition-Fixed"
)


def _names_with_prefix(
    model: mujoco.MjModel, objtype: mujoco.mjtObj, count: int, prefix: str
) -> list[str]:
    return [
        name
        for idx in range(count)
        if (name := mujoco.mj_id2name(model, objtype, idx)) is not None
        and name.startswith(prefix)
    ]


def _axial_face_center(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    direction_w: tuple[float, float, float],
) -> np.ndarray:
    """Return the cylinder face center furthest along ``direction_w``."""
    direction = np.asarray(direction_w, dtype=np.float64)
    cylinder_axis = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
    if float(np.dot(cylinder_axis, direction)) < 0.0:
        cylinder_axis = -cylinder_axis
    return data.geom_xpos[geom_id] + model.geom_size[geom_id, 1] * cylinder_axis


def _assert_production_nail_unchanged(
    fixture_model: mujoco.MjModel,
    fixture_data: mujoco.MjData,
) -> None:
    production_spec = get_nail_block_entity_cfg().spec_fn()
    production_model = production_spec.compile()
    production_data = mujoco.MjData(production_model)
    mujoco.mj_forward(production_model, production_data)

    fixture_joint_id = fixture_model.joint("nail_block/nail_slide").id
    production_joint_id = production_model.joint("nail_slide").id
    fixture_dof_id = fixture_model.jnt_dofadr[fixture_joint_id]
    production_dof_id = production_model.jnt_dofadr[production_joint_id]
    for field in (
        "jnt_type",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "jnt_limited",
        "jnt_solref",
        "jnt_solimp",
        "jnt_stiffness",
        "jnt_margin",
    ):
        np.testing.assert_array_equal(
            getattr(fixture_model, field)[fixture_joint_id],
            getattr(production_model, field)[production_joint_id],
        )
    for field in ("dof_damping", "dof_frictionloss", "dof_armature"):
        np.testing.assert_array_equal(
            getattr(fixture_model, field)[fixture_dof_id],
            getattr(production_model, field)[production_dof_id],
        )

    fixture_geom_id = fixture_model.geom("nail_block/nail_head").id
    production_geom_id = production_model.geom("nail_head").id
    for field in (
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_priority",
        "geom_solmix",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
    ):
        np.testing.assert_array_equal(
            getattr(fixture_model, field)[fixture_geom_id],
            getattr(production_model, field)[production_geom_id],
        )

    fixture_body_id = fixture_model.body("nail_block/nail").id
    production_body_id = production_model.body("nail").id
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_gravcomp",
    ):
        np.testing.assert_array_equal(
            getattr(fixture_model, field)[fixture_body_id],
            getattr(production_model, field)[production_body_id],
        )

    np.testing.assert_array_equal(
        fixture_data.geom_xpos[fixture_geom_id],
        production_data.geom_xpos[production_geom_id],
    )
    np.testing.assert_array_equal(
        fixture_data.geom_xmat[fixture_geom_id],
        production_data.geom_xmat[production_geom_id],
    )


def _assert_compiled_options_match(
    fixture_model: mujoco.MjModel, production_model: mujoco.MjModel
) -> None:
    scalar_fields = (
        "timestep",
        "integrator",
        "solver",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "ccd_iterations",
        "impratio",
        "cone",
        "jacobian",
        "disableflags",
        "enableflags",
    )
    for field in scalar_fields:
        assert getattr(fixture_model.opt, field) == getattr(
            production_model.opt, field
        )
    np.testing.assert_array_equal(
        fixture_model.opt.gravity, production_model.opt.gravity
    )

    assert fixture_model.opt.timestep == 0.002
    assert fixture_model.opt.iterations == 10
    assert fixture_model.opt.ls_iterations == 20
    assert fixture_model.opt.impratio == 10.0
    assert fixture_model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC


@pytest.mark.integration
def test_primary_fixture_compiles_exact_protocol() -> None:
    maker = getattr(calibration, "make_controlled_drop_env_cfg", None)
    assert callable(maker), "make_controlled_drop_env_cfg is absent"
    assert calibration.PRIMARY_DROP_H0_M == 0.150
    assert calibration.PRIMARY_DROP_MASS_KG == 0.200
    assert calibration.PRIMARY_DROP_RADIUS_M == 0.012
    assert calibration.PRIMARY_DROP_HALF_HEIGHT_M == 0.004
    assert calibration.PRIMARY_DROP_FRICTION == (1.5, 0.02, 0.002)
    assert calibration.PRIMARY_AXIS == (0.0, 0.0, -1.0)
    assert calibration.PRIMARY_PHYSICS_DT_S == 0.002
    assert calibration.PRIMARY_WINDOW_SUBSTEPS == 25
    assert calibration.PRIMARY_PROGRESS_EPS == 5e-4
    assert inspect.signature(maker).parameters == {}

    production_cfg = load_env_cfg(FIC0_TASK, play=True)
    production_cfg.scene.num_envs = 1
    cfg = maker()

    assert cfg.decimation == 1
    assert cfg.scene.num_envs == 1
    assert cfg.actions == {}
    assert cfg.observations == {}
    assert cfg.rewards == {}
    assert cfg.terminations == {}
    assert cfg.commands == {}
    assert cfg.curriculum == {}
    assert tuple(cfg.events) == ("reset_scene_to_default",)
    reset_cfg = cfg.events["reset_scene_to_default"]
    assert reset_cfg.func is reset_scene_to_default
    assert reset_cfg.mode == "reset"
    assert cfg.sim == production_cfg.sim
    assert cfg.sim is not production_cfg.sim
    assert cfg.sim.nconmax == 64
    assert cfg.sim.njmax == 300
    assert cfg.sim.mujoco.timestep == calibration.PRIMARY_PHYSICS_DT_S

    robot_cfg = cfg.scene.entities["robot"]
    assert robot_cfg.init_state.pos == (0.0, 0.0, 0.0)
    assert robot_cfg.init_state.joint_pos == {".*": 0.0}
    assert robot_cfg.init_state.joint_vel == {".*": 0.0}
    assert cfg.scene.entities["nail_block"].spec_fn is not None

    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    assert tuple(sensors) == ("hammer_nail_contact", "hammer_nail_impulse")
    contact_cfg = sensors["hammer_nail_contact"]
    impulse_cfg = sensors["hammer_nail_impulse"]
    assert isinstance(contact_cfg, ContactSensorCfg)
    assert isinstance(impulse_cfg, ContactSensorCfg)
    expected_primary = ContactMatch(
        mode="geom", pattern="dropper_face", entity="robot"
    )
    expected_secondary = ContactMatch(
        mode="body", pattern="nail", entity="nail_block"
    )
    assert contact_cfg.primary == expected_primary
    assert contact_cfg.secondary == expected_secondary
    assert contact_cfg.fields == ("found", "force")
    assert contact_cfg.reduce == "maxforce"
    assert contact_cfg.track_air_time is True
    assert impulse_cfg.primary == expected_primary
    assert impulse_cfg.secondary == expected_secondary
    assert impulse_cfg.fields == ("found", "force")
    assert impulse_cfg.reduce == "netforce"
    assert impulse_cfg.track_air_time is False

    assert tuple(cfg.metrics) == ("first_strike",)
    tracker_cfg = cfg.metrics["first_strike"]
    assert isinstance(tracker_cfg, MetricsTermCfg)
    assert tracker_cfg.func is FirstStrikeEventTracker
    assert tracker_cfg.per_substep is True
    assert tracker_cfg.reduce == "last"
    assert set(tracker_cfg.params) == {
        "contact_sensor_name",
        "impulse_sensor_name",
        "robot_cfg",
        "nail_cfg",
        "axis",
        "window_substeps",
        "progress_eps",
    }
    assert tracker_cfg.params["contact_sensor_name"] == "hammer_nail_contact"
    assert tracker_cfg.params["impulse_sensor_name"] == "hammer_nail_impulse"
    assert tracker_cfg.params["axis"] == calibration.PRIMARY_AXIS
    assert tracker_cfg.params["window_substeps"] == calibration.PRIMARY_WINDOW_SUBSTEPS
    assert tracker_cfg.params["progress_eps"] == calibration.PRIMARY_PROGRESS_EPS
    assert tracker_cfg.params["robot_cfg"] == SceneEntityCfg(
        "robot", site_names=("dropper_face_site",)
    )
    assert tracker_cfg.params["nail_cfg"] == SceneEntityCfg(
        "nail_block",
        joint_names=("nail_slide",),
        site_names=("nail_top",),
    )

    fixture_env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    production_env = ManagerBasedRlEnv(cfg=production_cfg, device="cpu")
    try:
        assert fixture_env.action_manager.action.shape == (1, 0)
        model = fixture_env.sim.mj_model
        data = fixture_env.sim.mj_data

        robot_joints = _names_with_prefix(
            model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt, "robot/"
        )
        assert robot_joints == ["robot/drop_axis"]
        drop_joint_id = model.joint("robot/drop_axis").id
        assert model.jnt_type[drop_joint_id] == mujoco.mjtJoint.mjJNT_SLIDE
        np.testing.assert_array_equal(
            model.jnt_axis[drop_joint_id], np.asarray(calibration.PRIMARY_AXIS)
        )
        drop_dof_id = model.jnt_dofadr[drop_joint_id]
        assert model.dof_damping[drop_dof_id] == 0.0
        assert model.dof_frictionloss[drop_dof_id] == 0.0
        assert all(
            model.jnt_type[model.joint(name).id]
            not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE)
            for name in robot_joints
        )

        drop_body_id = model.body("robot/dropper").id
        assert model.body_mass[drop_body_id] == calibration.PRIMARY_DROP_MASS_KG
        drop_geom_id = model.geom("robot/dropper_face").id
        assert model.geom_type[drop_geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        np.testing.assert_array_equal(
            model.geom_size[drop_geom_id],
            np.asarray(
                (
                    calibration.PRIMARY_DROP_RADIUS_M,
                    calibration.PRIMARY_DROP_HALF_HEIGHT_M,
                    0.0,
                )
            ),
        )
        np.testing.assert_array_equal(
            model.geom_friction[drop_geom_id],
            np.asarray(calibration.PRIMARY_DROP_FRICTION),
        )
        assert model.geom_contype[drop_geom_id] == 3
        assert model.geom_conaffinity[drop_geom_id] == 3
        np.testing.assert_array_equal(
            model.geom_solref[drop_geom_id], np.asarray((0.02, 1.0))
        )
        np.testing.assert_array_equal(
            model.geom_solimp[drop_geom_id],
            np.asarray((0.9, 0.95, 0.001, 0.5, 2.0)),
        )

        drop_qpos_adr = model.jnt_qposadr[drop_joint_id]
        assert data.qpos[drop_qpos_adr] == 0.0
        assert data.qvel[drop_dof_id] == 0.0

        nail_geom_id = model.geom("nail_block/nail_head").id
        drop_lower_face = _axial_face_center(
            model, data, drop_geom_id, calibration.PRIMARY_AXIS
        )
        nail_upper_face = _axial_face_center(
            model,
            data,
            nail_geom_id,
            tuple(-component for component in calibration.PRIMARY_AXIS),
        )
        np.testing.assert_allclose(
            data.geom_xpos[drop_geom_id, :2],
            data.geom_xpos[nail_geom_id, :2],
            rtol=0.0,
            atol=1e-12,
        )
        clearance = float(
            np.dot(
                drop_lower_face - nail_upper_face,
                -np.asarray(calibration.PRIMARY_AXIS),
            )
        )
        assert clearance == pytest.approx(
            calibration.PRIMARY_DROP_H0_M, rel=0.0, abs=1e-12
        )

        _assert_production_nail_unchanged(model, data)
        _assert_compiled_options_match(model, production_env.sim.mj_model)
    finally:
        production_env.close()
        fixture_env.close()


@pytest.mark.integration
def test_one_primary_drop_contacts_and_finalizes_with_finite_positive_measurements():
    trial = run_one_primary_drop(device="cpu", trial_index=1)

    assert trial.trial_index == 1
    assert trial.release_velocity_m_s == 0.0
    assert trial.contacted is True
    assert trial.finalized is True
    assert trial.productive is True
    assert trial.reason in {"success", "window"}
    assert math.isfinite(trial.precontact_velocity_m_s)
    assert trial.precontact_velocity_m_s > 0.0
    assert math.isfinite(trial.impulse_n_s)
    assert trial.impulse_n_s > 0.0
