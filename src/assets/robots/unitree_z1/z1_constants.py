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

from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
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
    # Real claw-hammer meshes (SimToolReal, MIT) live in a subdir and are .obj;
    # key them by the path the XML references (relative to meshdir="meshes/").
    for mesh_file in _MESH_DIR.glob("claw_hammer/*.obj"):
        assets[f"claw_hammer/{mesh_file.name}"] = mesh_file.read_bytes()
    spec.assets = assets
    # Gravity compensation on every robot body (arm links + gripper + hammer).
    # The real Unitree Z1 runs gravity compensation in firmware, and every major
    # MuJoCo-based manipulation RL stack does the same: robosuite adds MuJoCo's
    # qfrc_bias to the controller output so the position-controlled arm HOLDS its
    # commanded pose against gravity instead of sagging (verified deep dive,
    # 2026-06-16; robosuite Controller.torque_compensation = qfrc_bias, added in
    # both OSC and joint-PD controllers). Without it the DiffIK "hold current pose"
    # action ratchets the arm downward under zero action and the hammer creeps
    # onto the nail at rest. gravcomp cancels only the static gravity FORCE — masses,
    # inertia and contacts are untouched, so the strike impulse (m_eff*v) is
    # preserved. Consistent with the nail body, which already uses gravcomp.
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
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

# Gripper: vestigial for this task — the hammer is rigidly attached to ee_center_body,
# so the gripper only holds a non-functional jaw at jointGripper≈-0.001. The original
# stiff+light config (k=1000, armature=0.005) chattered numerically once the arm became
# gravity-compensated: ~25% of resets left a phantom ~-4.59 rad/s in the gripper qvel
# (qpos frozen) that polluted the joint_vel observation (sim-audit 2026-06-16). Softer
# gains + larger armature lower the actuator's natural frequency well below the
# substep-stability limit, so the jaw holds quietly. No functional effect (the jaw
# grasps nothing).
_Z1_GRIPPER = BuiltinPositionActuatorCfg(
    target_names_expr=("jointGripper",),
    stiffness=100.0,
    damping=20.0,
    effort_limit=30.0,
    armature=0.005,
)

Z1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(_Z1_ARM_STANDARD, _Z1_ARM_J2, _Z1_GRIPPER),
)


##
# A4 (ablation): DC-motor torque-speed envelope variant.
##
# Same PD gains/armature as the position actuators, but a velocity-dependent torque
# ceiling makes >velocity_limit physically unreachable to the MOTOR (back-EMF model,
# the MuJoCo-maintainer-endorsed alternative to a non-physical clip). The velocity bound
# comes from the torque-speed CURVE (torque_speed_top = saturation*(1 - |q̇|/velocity_limit)
# -> 0 at q̇=velocity_limit), NOT the continuous-torque clamp: with effort_limit==
# saturation_effort the horizontal clamp is inert for forward motion, so it's left equal
# to the URDF rating (no separate continuous datasheet value exists). NOTE this routes the
# arm through mjlab's Python torque-level IdealPdActuator (a <motor> + explicit PD) instead
# of the native <position> affine PD; gravcomp (get_spec) is unaffected. CAVEAT (research
# docs/research/reward-design/CONSTRAINED_RL_LANDSCAPE.md): the curve limits each joint's OWN motor, so it removes
# the actuator-driven windup but cannot brake chain-coupled momentum delivered through the
# linkage -- so it may reduce, not fully eliminate, the 4.3-4.65 rad/s overshoot.
_Z1_VELOCITY_LIMIT: float = 3.1415  # rad/s, URDF no-load speed (all joints)

_Z1_ARM_STANDARD_DC = DcMotorActuatorCfg(
    target_names_expr=("joint1", "joint3", "joint4", "joint5", "joint6"),
    stiffness=1000.0,
    damping=100.0,
    effort_limit=30.0,          # == saturation_effort (no separate continuous rating)
    saturation_effort=30.0,     # stall (peak) torque
    velocity_limit=_Z1_VELOCITY_LIMIT,
    armature=0.01,
)

_Z1_ARM_J2_DC = DcMotorActuatorCfg(
    target_names_expr=("joint2",),
    stiffness=1500.0,
    damping=150.0,
    effort_limit=60.0,
    saturation_effort=60.0,
    velocity_limit=_Z1_VELOCITY_LIMIT,
    armature=0.02,
)

# Gripper stays a position actuator (vestigial; must not change behavior).
Z1_ARTICULATION_DC = EntityArticulationInfoCfg(
    actuators=(_Z1_ARM_STANDARD_DC, _Z1_ARM_J2_DC, _Z1_GRIPPER),
)


##
# Initial joint configurations (radians).
##

# Neutral/home pose from the original Gym env.
NEUTRAL_JOINT_POS: dict[str, float] = {
    "joint1": -0.000297861,
    "joint2":  1.30495,
    "joint3": -1.54711,
    "joint4":  0.197456,
    "joint5":  0.000311167,
    "joint6":  1.57079632679,
    "jointGripper": -0.000964725,
}

# RE-SOLVED 2026-07-06 for the NEW hammer grasp (SimplifiedLink06 holder; hammer body rotated
# -90 deg about Z so the handle is coaxial in the holder bore -> the poll FACE points perpendicular
# to the handle). With this grasp a perfectly vertical strike AT the floor nail is kinematically
# impossible (in-plane face-down only becomes reachable above z~0.20). So this is a RESET poised
# ~15 cm ABOVE the (still-on-floor) nail, at the one place an IN-PLANE (joint1=0) DEAD-VERTICAL pose
# IS reachable: face target world (0.5, 0, 0.25), tilt 0 deg. The policy drives DOWN from here to
# the nail and discovers the (necessarily oblique near the floor) contact angle itself. Block stays
# on the FLOOR (no raise / no pedestal). Solved via the in-plane windup sweep (scratchpad). The old
# value (solved for the pre-2026-07-06 grasp) is in git history.
# NOTE (open thread): posture at contact is an ~8.5x impact lever (effective-mass analysis), but the
# position-only DiffIK action space collapses the redundancy, so bracing must EMERGE via the
# delivered-impulse reward + (future) null-space access, not be baked into this reset.
# (sweep solver not preserved; pose verified by playback_reference.py gate, 2026-07-10)
NEAR_NAIL_JOINT_POS: dict[str, float] = {
    "joint1":  0.00000000,
    "joint2":  1.60600000,
    "joint3": -0.43010000,
    "joint4": -1.19760000,
    "joint5": -0.00130000,
    "joint6":  1.55440000,
    "jointGripper": -0.001,
}

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos=NEAR_NAIL_JOINT_POS,
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

# DifferentialIK scale: max position delta the policy commands per control step.
# 0.05 -> 0.15 (2026-06-17): terminal hammer-head speed is ~linear in this scale
# (~0.18 * delta_pos_scale / dt), so 0.05 capped the strike at ~0.45 m/s -- a gentle
# servo push, not a momentum blow. 0.15 -> ~1.35 m/s (max joint ~2.2 rad/s, under the
# 3.1415 rad/s velocity rail; head KE ~0.45 J) enabling a genuine single strike. The
# rail (env_cfgs.py max_dq) keeps the achievable speed hardware-faithful.
# See docs/archive/2026-06-17-z1-strike-not-press-redesign-design.md.
Z1_HAMMER_DELTA_POS_SCALE: float = 0.15


##
# Public robot config factory.
##


def get_z1_hammer_robot_cfg(dcmotor: bool = False) -> EntityCfg:
    """Return a fresh Z1+hammer EntityCfg.

    Returns a new instance each call to avoid shared-state mutation bugs
    when the config is used in multiple places.

    Args:
      dcmotor: if True, use the DC-motor torque-speed-envelope arm actuators (A4
        ablation; velocity_limit=3.1415 rad/s) instead of the default
        BuiltinPositionActuatorCfg. The gripper stays a position actuator either way.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=Z1_ARTICULATION_DC if dcmotor else Z1_ARTICULATION,
    )
