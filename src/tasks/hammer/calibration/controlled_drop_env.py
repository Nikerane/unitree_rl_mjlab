"""Fixed passive-drop fixture for the primary delivered-impulse calibration."""

from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg

from src.tasks.hammer.mdp.first_strike import FirstStrikeEventTracker
from src.tasks.hammer.nail_block import get_nail_block_entity_cfg


PRIMARY_DROP_H0_M: float = 0.150
PRIMARY_DROP_MASS_KG: float = 0.200
PRIMARY_DROP_RADIUS_M: float = 0.012
PRIMARY_DROP_HALF_HEIGHT_M: float = 0.00875
PRIMARY_DROP_FRICTION: tuple[float, float, float] = (1.5, 0.02, 0.002)
PRIMARY_AXIS: tuple[float, float, float] = (0.0, 0.0, -1.0)
PRIMARY_PHYSICS_DT_S: float = 0.002
PRIMARY_WINDOW_SUBSTEPS: int = 25
PRIMARY_PROGRESS_EPS: float = 5e-4

_FIC0_TASK = (
  "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
  "CProgress-Vel-Delivered4-JointPosition-Fixed"
)


def _production_nail_head_geometry() -> tuple[np.ndarray, float]:
  """Resolve the production nail-head center and upper axial surface."""
  nail_spec = get_nail_block_entity_cfg().spec_fn()
  nail_model = nail_spec.compile()
  nail_data = mujoco.MjData(nail_model)
  mujoco.mj_forward(nail_model, nail_data)

  nail_head_id = nail_model.geom("nail_head").id
  if nail_model.geom_type[nail_head_id] != mujoco.mjtGeom.mjGEOM_CYLINDER:
    raise ValueError("production nail_head must remain a cylinder")

  center_w = nail_data.geom_xpos[nail_head_id].copy()
  cylinder_axis_w = nail_data.geom_xmat[nail_head_id].reshape(3, 3)[:, 2].copy()
  upward_w = -np.asarray(PRIMARY_AXIS, dtype=np.float64)
  if float(np.dot(cylinder_axis_w, upward_w)) < 0.0:
    cylinder_axis_w = -cylinder_axis_w
  upper_face_w = (
    center_w + nail_model.geom_size[nail_head_id, 1] * cylinder_axis_w
  )
  return center_w, float(upper_face_w[2])


def _make_dropper_spec() -> mujoco.MjSpec:
  nail_head_center_w, nail_contact_surface_z_m = _production_nail_head_geometry()
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(
    name="dropper",
    pos=(
      float(nail_head_center_w[0]),
      float(nail_head_center_w[1]),
      nail_contact_surface_z_m
      + PRIMARY_DROP_H0_M
      + PRIMARY_DROP_HALF_HEIGHT_M,
    ),
  )
  body.add_joint(
    name="drop_axis",
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=PRIMARY_AXIS,
    damping=0.0,
    frictionloss=0.0,
  )
  body.add_geom(
    name="dropper_face",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(PRIMARY_DROP_RADIUS_M, PRIMARY_DROP_HALF_HEIGHT_M, 0.0),
    mass=PRIMARY_DROP_MASS_KG,
    friction=PRIMARY_DROP_FRICTION,
    contype=3,
    conaffinity=3,
  )
  body.add_site(
    name="dropper_face_site",
    pos=(0.0, 0.0, -PRIMARY_DROP_HALF_HEIGHT_M),
  )
  return spec


def make_controlled_drop_env_cfg() -> ManagerBasedRlEnvCfg:
  """Build the exact fixed-0.150 m passive-drop calibration fixture."""
  contact_sensor = ContactSensorCfg(
    name="hammer_nail_contact",
    primary=ContactMatch(
      mode="geom", pattern="dropper_face", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="nail", entity="nail_block"),
    fields=("found", "force"),
    reduce="maxforce",
    track_air_time=True,
  )
  impulse_sensor = ContactSensorCfg(
    name="hammer_nail_impulse",
    primary=ContactMatch(
      mode="geom", pattern="dropper_face", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="nail", entity="nail_block"),
    fields=("found", "force"),
    reduce="netforce",
    track_air_time=False,
  )
  tracker_cfg = MetricsTermCfg(
    func=FirstStrikeEventTracker,
    per_substep=True,
    reduce="last",
    params={
      "contact_sensor_name": "hammer_nail_contact",
      "impulse_sensor_name": "hammer_nail_impulse",
      "robot_cfg": SceneEntityCfg(
        "robot", site_names=("dropper_face_site",)
      ),
      "nail_cfg": SceneEntityCfg(
        "nail_block",
        joint_names=("nail_slide",),
        site_names=("nail_top",),
      ),
      "axis": PRIMARY_AXIS,
      "window_substeps": PRIMARY_WINDOW_SUBSTEPS,
      "progress_eps": PRIMARY_PROGRESS_EPS,
    },
  )

  production_sim = deepcopy(load_env_cfg(_FIC0_TASK, play=True).sim)
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=1,
      entities={
        "robot": EntityCfg(spec_fn=_make_dropper_spec, articulation=None),
        "nail_block": get_nail_block_entity_cfg(),
      },
      sensors=(contact_sensor, impulse_sensor),
    ),
    observations={},
    actions={},
    rewards={},
    terminations={},
    commands={},
    curriculum={},
    metrics={"first_strike": tracker_cfg},
    sim=production_sim,
    decimation=1,
  )
