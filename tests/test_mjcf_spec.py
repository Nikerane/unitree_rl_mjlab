"""Layer 3 — MuJoCo MjSpec compilation tests.

Tests that spec factory functions return valid, compilable MjSpec objects
with the correct model structure (joint names, site names, zero XML actuators).
Requires mujoco and mjlab to be installed; does NOT create an RL environment.
"""

import mujoco
import pytest

from src.assets.robots.unitree_z1.z1_constants import (
    ARM_JOINT_NAMES,
    EE_SITE_NAME,
    GRIPPER_JOINT_NAME,
    HAMMER_HEAD_SITE_NAME,
    get_spec,
)
from src.tasks.hammer.nail_block import get_nail_block_spec


# ---------------------------------------------------------------------------
# Z1 robot spec
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def z1_spec():
    return get_spec()


@pytest.fixture(scope="module")
def z1_model(z1_spec):
    return z1_spec.compile()


class TestZ1Spec:
    def test_returns_mjspec(self, z1_spec):
        assert isinstance(z1_spec, mujoco.MjSpec)

    def test_compiles_without_error(self, z1_model):
        assert z1_model is not None

    def test_no_xml_actuators(self, z1_model):
        """XML has no <actuator> block — mjlab adds them via BuiltinPositionActuatorCfg."""
        assert z1_model.nu == 0, (
            f"Expected nu=0, got {z1_model.nu}. "
            "Remove the <actuator> block from z1_hammer_robot.xml."
        )

    def test_joint_count(self, z1_model):
        assert z1_model.njnt == 7

    def test_arm_joint_names_present(self, z1_model):
        model_joints = {z1_model.joint(i).name for i in range(z1_model.njnt)}
        for name in ARM_JOINT_NAMES:
            assert name in model_joints, f"Joint '{name}' missing from compiled model"

    def test_gripper_joint_present(self, z1_model):
        model_joints = {z1_model.joint(i).name for i in range(z1_model.njnt)}
        assert GRIPPER_JOINT_NAME in model_joints

    def test_ee_site_present(self, z1_model):
        site_names = {z1_model.site(i).name for i in range(z1_model.nsite)}
        assert EE_SITE_NAME in site_names, (
            f"Site '{EE_SITE_NAME}' missing. Available: {site_names}"
        )

    def test_hammer_head_site_present(self, z1_model):
        site_names = {z1_model.site(i).name for i in range(z1_model.nsite)}
        assert HAMMER_HEAD_SITE_NAME in site_names, (
            f"Site '{HAMMER_HEAD_SITE_NAME}' missing. Available: {site_names}"
        )

    def test_has_expected_dof(self, z1_model):
        # nq = nv = 7 (all revolute joints)
        assert z1_model.nq == 7
        assert z1_model.nv == 7

    def test_mesh_assets_loaded(self, z1_spec):
        """Assets dict must be populated by get_spec() before compile."""
        assert len(z1_spec.assets) > 0, "No mesh assets attached to MjSpec"

    def test_get_spec_returns_fresh_each_call(self):
        """Each call must return a new MjSpec instance (not a cached singleton)."""
        spec1 = get_spec()
        spec2 = get_spec()
        assert spec1 is not spec2


# ---------------------------------------------------------------------------
# Nail + block spec
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nail_spec():
    return get_nail_block_spec()


@pytest.fixture(scope="module")
def nail_model(nail_spec):
    return nail_spec.compile()


class TestNailBlockSpec:
    def test_returns_mjspec(self, nail_spec):
        assert isinstance(nail_spec, mujoco.MjSpec)

    def test_compiles_without_error(self, nail_model):
        assert nail_model is not None

    def test_has_one_joint(self, nail_model):
        assert nail_model.njnt == 1
        assert nail_model.nq == 1
        assert nail_model.nv == 1

    def test_nail_slide_joint_name(self, nail_model):
        assert nail_model.joint(0).name == "nail_slide"

    def test_nail_slide_is_slide_type(self, nail_model):
        # mujoco.mjtJoint.mjJNT_SLIDE == 2
        assert nail_model.joint(0).type == mujoco.mjtJoint.mjJNT_SLIDE

    def test_nail_top_site_present(self, nail_model):
        site_names = {nail_model.site(i).name for i in range(nail_model.nsite)}
        assert "nail_top" in site_names

    def test_nail_no_xml_actuators(self, nail_model):
        assert nail_model.nu == 0
