"""Unitree Z1 arm + hammer constants and EntityCfg for mjlab.

The robot XML (z1_hammer_robot.xml) is the mocap-free version of the
Z1 scene: joints and geometry are intact; the <actuator> section has
been removed because mjlab's BuiltinPositionActuatorCfg adds actuators
programmatically, which is required for DifferentialIKActionCfg to
resolve joint IDs at environment construction time.

Actuator parameters are derived from the original XML defaults:
  <general biastype="affine" gainprm="1000" biasprm="0 -1000 -100" forcerange="-30 30"/>
  joint2 override: gainprm="1500" biasprm="0 -1500 -150" forcerange="-60 60"
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# Paths — meshes live next to the XML in safe_impact_manipulation.
##

_ASSETS_DIR: Path = (
    Path(__file__).resolve().parents[5]  # up to ~/repos
    / "safe_impact_manipulation"
    / "hammer_z1_env"
    / "assets"
)

Z1_HAMMER_XML: Path = _ASSETS_DIR / "z1_hammer_robot.xml"
_MESH_DIR: Path = _ASSETS_DIR / "meshes"

assert Z1_HAMMER_XML.exists(), f"Z1 robot XML not found: {Z1_HAMMER_XML}"
assert _MESH_DIR.exists(), f"Z1 mesh directory not found: {_MESH_DIR}"


##
# Spec factory.
##


def get_spec() -> mujoco.MjSpec:
    """Load the Z1+hammer MJCF and inject mesh binary assets."""
    spec = mujoco.MjSpec.from_file(str(Z1_HAMMER_XML))
    assets: dict[str, bytes] = {}
    for mesh_file in _MESH_DIR.glob("*.stl"):
        assets[mesh_file.name] = mesh_file.read_bytes()
    spec.assets = assets
    return spec


##
# Builtin actuator configs — replaces XML <actuator> section.
# stiffness = gainprm, damping = |biasprm[2]|, effort_limit = forcerange max.
##

# Joints 1, 3-6: default PD gains from <default class="z1">
# armature=0.01 adds rotor inertia (reflected motor inertia to joint space).
# Z1 motor datasheets are not public; 0.01 kg·m² is the same ballpark used
# for Go2 and H2 in this repo. Tune once real motor specs are available.
_Z1_ARM_STANDARD = BuiltinPositionActuatorCfg(
    target_names_expr=("joint1", "joint3", "joint4", "joint5", "joint6"),
    stiffness=1000.0,
    damping=100.0,
    effort_limit=30.0,
    armature=0.01,
)

# Joint 2 (shoulder lift): higher gains + slightly heavier motor (2× torque limit).
_Z1_ARM_J2 = BuiltinPositionActuatorCfg(
    target_names_expr=("joint2",),
    stiffness=1500.0,
    damping=150.0,
    effort_limit=60.0,
    armature=0.02,
)

# Gripper: lighter mechanism, smaller armature.
_Z1_GRIPPER = BuiltinPositionActuatorCfg(
    target_names_expr=("jointGripper",),
    stiffness=1000.0,
    damping=100.0,
    effort_limit=30.0,
    armature=0.005,
)

Z1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(_Z1_ARM_STANDARD, _Z1_ARM_J2, _Z1_GRIPPER),
)


##
# Initial joint configuration (radians).
# From the Gym env neutral pose, matches hammer head above the nail.
##

NEUTRAL_JOINT_POS: dict[str, float] = {
    "joint1": -0.000297861,
    "joint2":  1.30495,
    "joint3": -1.54711,
    "joint4":  0.197456,
    "joint5":  0.000311167,
    "joint6":  1.57079632679,
    "jointGripper": -0.000964725,
}

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos=NEUTRAL_JOINT_POS,
    joint_vel={".*": 0.0},
)


##
# Named model elements — used by SceneEntityCfg in env_cfgs.
# ARM_ACTUATOR_NAMES now matches the joint names used in BuiltinPositionActuatorCfg
# (actuators are keyed by their target joint name in mjlab's actuator registry).
##

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINT_NAME = "jointGripper"

# Used by DifferentialIKActionCfg.actuator_names — must match target_names_expr.
ARM_ACTUATOR_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

EE_SITE_NAME = "ee_center_site"
HAMMER_HEAD_SITE_NAME = "hammer_head_site"

# DifferentialIK scale: max 5 cm per policy step (matches Gym env action_scale).
Z1_HAMMER_DELTA_POS_SCALE: float = 0.05


##
# Public robot config factory.
##


def get_z1_hammer_robot_cfg() -> EntityCfg:
    """Return a fresh Z1+hammer EntityCfg with BuiltinPositionActuatorCfg.

    Returns a new instance each call to avoid shared-state mutation bugs
    when the config is used in multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=Z1_ARTICULATION,
    )
