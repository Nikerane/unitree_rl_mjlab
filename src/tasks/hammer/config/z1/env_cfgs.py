"""Unitree Z1 hammer-nail environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots.unitree_z1.z1_constants import (
  ARM_ACTUATOR_NAMES,
  EE_SITE_NAME,
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
  get_z1_hammer_robot_cfg,
)
from src.tasks.hammer.hammer_env_cfg import make_hammer_env_cfg
from src.tasks.hammer.nail_block import get_nail_block_entity_cfg


def z1_hammer_env_cfg(play: bool = False, imitation: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Z1 hammer-nail task configuration."""
  cfg = make_hammer_env_cfg(imitation=imitation)

  # --- Scene entities ---
  cfg.scene.entities = {
    "robot": get_z1_hammer_robot_cfg(),
    "nail_block": get_nail_block_entity_cfg(),
  }

  # --- Contact sensor: hammer_head geom vs. nail body ---
  hammer_nail_contact = ContactSensorCfg(
    name="hammer_nail_contact",
    # Real-hammer head collision pieces: hammer_head_0 = c4 (the striking FACE,
    # rounded poll) and hammer_head_1 = c3 (neck). The forked claw (hammer_claw_0/1
    # = c1/c2) and the shaft (hammer_handle_col = c0) are deliberately NOT named
    # hammer_head_* so this fullmatch regex excludes them -- only the striking head
    # counts as a strike (the FACE leads contact by 56 mm; diag_collision_recheck.py).
    primary=ContactMatch(mode="geom", pattern="hammer_head_.*", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="nail", entity="nail_block"),
    fields=("found", "force"),
    reduce="maxforce",
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (hammer_nail_contact,)

  # --- Wire DifferentialIK to the Z1 arm actuators and hammer_head_site ---
  ik_action = cfg.actions["ik_hammer_head"]
  assert isinstance(ik_action, DifferentialIKActionCfg)
  ik_action.actuator_names = ARM_ACTUATOR_NAMES
  ik_action.frame_name = HAMMER_HEAD_SITE_NAME
  ik_action.delta_pos_scale = Z1_HAMMER_DELTA_POS_SCALE
  # Joint-velocity headroom: at delta_pos_scale=0.15 the max straight-down command
  # drives the dominant joint to ~2.41 rad/s -- under the real Z1 limit of 3.1415 rad/s
  # (z1_description URDF) -- so the action scale is self-limiting and no max_dq rail is
  # needed here. (max_dq is NOT a clean velocity limit: it bounds the per-substep IK
  # step, and the PD then settles at ~(kp/kd)*max_dq, so a dt-based value craters motion.)
  # If delta_pos_scale is ever raised toward ~2.5 m/s, add a calibrated rail
  # (max_dq ~= 0.29 caps joint speed near 3.14 rad/s) and re-verify with diag_strike_probe.

  # --- Wire observation site names ---
  # EE site observations.
  for obs_key in ("ee_pos", "ee_vel"):
    for group in cfg.observations.values():
      assert isinstance(group, ObservationGroupCfg)
      if obs_key in group.terms:
        group.terms[obs_key].params["asset_cfg"].site_names = (EE_SITE_NAME,)

  # Hammer head site observations.
  for obs_key in ("head_pos", "head_vel"):
    for group in cfg.observations.values():
      assert isinstance(group, ObservationGroupCfg)
      if obs_key in group.terms:
        group.terms[obs_key].params["asset_cfg"].site_names = (HAMMER_HEAD_SITE_NAME,)

  # Strike-reference observations (T1): wire the hammer head site.
  for obs_key in ("strike_phase", "strike_ref_error"):
    for group in cfg.observations.values():
      assert isinstance(group, ObservationGroupCfg)
      if obs_key in group.terms:
        group.terms[obs_key].params["robot_cfg"].site_names = (HAMMER_HEAD_SITE_NAME,)

  # --- Wire approach reward site name ---
  cfg.rewards["approach"].params["robot_cfg"].site_names = (HAMMER_HEAD_SITE_NAME,)

  # --- Wire impact_progress reward head site name (mirrors approach) ---
  cfg.rewards["impact_progress"].params["robot_cfg"].site_names = (HAMMER_HEAD_SITE_NAME,)

  # --- Wire r_imit reward head site name (A-TRACK arm only; mirrors approach) ---
  if imitation:
    cfg.rewards["r_imit"].params["robot_cfg"].site_names = (HAMMER_HEAD_SITE_NAME,)

  # --- Viewer ---
  cfg.viewer.body_name = "link00"

  # --- Play mode overrides ---
  if play:
    cfg.episode_length_s = int(1e9)
    # Deterministic resets for validation/play; training keeps ±0.05 rad noise.
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    for group in cfg.observations.values():
      assert isinstance(group, ObservationGroupCfg)
      group.enable_corruption = False
    cfg.curriculum = {}

  return cfg
