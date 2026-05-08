"""Layer 1 — XML asset and filesystem validation.

Tests that all required files exist and that the XML scenes compile
correctly in MuJoCo. No mjlab or Warp required.
"""

import mujoco
import pytest

from src.assets.robots.unitree_z1.z1_constants import (
    Z1_HAMMER_XML,
    _MESH_DIR,
    ARM_JOINT_NAMES,
    EE_SITE_NAME,
    HAMMER_HEAD_SITE_NAME,
)
from src.tasks.hammer.nail_block import _SCENE_XML


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_xml(xml_path, assets=None):
    """Load and compile an XML, optionally providing binary assets."""
    spec = mujoco.MjSpec.from_file(str(xml_path), assets=assets or {})
    return spec.compile()


def _mesh_assets():
    """Return a dict of {filename: bytes} for all STL meshes."""
    return {f.name: f.read_bytes() for f in _MESH_DIR.glob("*.stl")}


# ---------------------------------------------------------------------------
# Z1 robot XML — path and filesystem
# ---------------------------------------------------------------------------


def test_z1_robot_xml_exists():
    assert Z1_HAMMER_XML.exists(), f"Z1 robot XML not found: {Z1_HAMMER_XML}"


def test_mesh_dir_exists():
    assert _MESH_DIR.exists(), f"Mesh directory not found: {_MESH_DIR}"


def test_mesh_dir_has_stl_files():
    stl_files = list(_MESH_DIR.glob("*.stl"))
    assert len(stl_files) > 0, f"No STL files in {_MESH_DIR}"


# ---------------------------------------------------------------------------
# Z1 robot XML — MuJoCo compilation
# ---------------------------------------------------------------------------


def test_z1_xml_compiles():
    m = _compile_xml(Z1_HAMMER_XML, _mesh_assets())
    # 7 single-DOF joints (joint1-6 + gripperJoint) → nq=nv=7
    assert m.nq == 7, f"Expected nq=7, got {m.nq}"
    assert m.nv == 7, f"Expected nv=7, got {m.nv}"


def test_z1_xml_has_no_xml_actuators():
    """The <actuator> block must be absent — mjlab adds actuators programmatically."""
    m = _compile_xml(Z1_HAMMER_XML, _mesh_assets())
    assert m.nu == 0, (
        f"Expected nu=0 (no XML actuators), got {m.nu}. "
        "The <actuator> block should not be present in z1_hammer_robot.xml."
    )


def test_z1_xml_has_all_arm_joints():
    m = _compile_xml(Z1_HAMMER_XML, _mesh_assets())
    model_joints = {m.joint(i).name for i in range(m.njnt)}
    for name in ARM_JOINT_NAMES:
        assert name in model_joints, f"Expected joint '{name}' not found in model"


def test_z1_xml_has_gripper_joint():
    m = _compile_xml(Z1_HAMMER_XML, _mesh_assets())
    model_joints = {m.joint(i).name for i in range(m.njnt)}
    assert "jointGripper" in model_joints


def test_z1_xml_total_joint_count():
    m = _compile_xml(Z1_HAMMER_XML, _mesh_assets())
    # 6 arm joints + 1 gripper
    assert m.njnt == 7, f"Expected 7 joints, got {m.njnt}"


def test_z1_xml_has_ee_center_site():
    m = _compile_xml(Z1_HAMMER_XML, _mesh_assets())
    site_names = {m.site(i).name for i in range(m.nsite)}
    assert EE_SITE_NAME in site_names, (
        f"Site '{EE_SITE_NAME}' not found. Available: {site_names}"
    )


def test_z1_xml_has_hammer_head_site():
    m = _compile_xml(Z1_HAMMER_XML, _mesh_assets())
    site_names = {m.site(i).name for i in range(m.nsite)}
    assert HAMMER_HEAD_SITE_NAME in site_names, (
        f"Site '{HAMMER_HEAD_SITE_NAME}' not found. Available: {site_names}"
    )


# ---------------------------------------------------------------------------
# Nail + block scene XML — path and compilation
# ---------------------------------------------------------------------------


def test_nail_block_xml_exists():
    assert _SCENE_XML.exists(), f"Nail/block scene XML not found: {_SCENE_XML}"


def test_nail_block_xml_compiles():
    m = _compile_xml(_SCENE_XML)
    # 1 slide joint for nail
    assert m.njnt == 1, f"Expected 1 joint (nail_slide), got {m.njnt}"
    assert m.nq == 1
    assert m.nv == 1


def test_nail_block_xml_has_nail_slide_joint():
    m = _compile_xml(_SCENE_XML)
    joint_names = {m.joint(i).name for i in range(m.njnt)}
    assert "nail_slide" in joint_names


def test_nail_block_xml_has_nail_top_site():
    m = _compile_xml(_SCENE_XML)
    site_names = {m.site(i).name for i in range(m.nsite)}
    assert "nail_top" in site_names, (
        f"Site 'nail_top' not found. Available: {site_names}"
    )


def test_nail_slide_joint_range():
    """Nail slide range must be [0, 0.075] — the full driving depth."""
    m = _compile_xml(_SCENE_XML)
    nail_joint = m.joint("nail_slide")
    assert pytest.approx(nail_joint.range[0], abs=1e-6) == 0.0
    assert pytest.approx(nail_joint.range[1], abs=1e-4) == 0.075
