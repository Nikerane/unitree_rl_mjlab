"""Base hammer-nail task configuration (robot-agnostic).

Mirrors the pattern of mjlab's lift_cube_env_cfg.py:
  - make_hammer_env_cfg() returns a ManagerBasedRlEnvCfg skeleton.
  - Robot-specific overrides (entity, site names, action scale) are applied
    in config/<robot>/env_cfgs.py.

Task semantics (matches the Gym Z1HammerEnv):
  - Action:   3-D delta position of the hammer head site (DifferentialIK).
  - Obs:      ee_pos, ee_vel, head_pos, head_vel, nail_top_pos, nail_depth.
  - Reward:   Staged — approach reward + nail-driven reward + action-rate penalty.
  - Success:  nail_slide qpos >= NAIL_SUCCESS_THRESHOLD (0.07 m).
  - Episode:  20 s; terminates early on success.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab.envs import mdp as envs_mdp
from src.tasks.hammer import mdp as hammer_mdp
from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH, NAIL_SUCCESS_THRESHOLD


def make_hammer_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base hammer-nail task configuration.

  Robot entity and site names must be filled in by the robot-specific
  config (config/z1/env_cfgs.py).
  """

  # --- Observations ---
  # Per-robot site names are placeholders (empty tuples); filled in per-robot.
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "ee_pos": ObservationTermCfg(
      func=hammer_mdp.ee_pos_b,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "ee_vel": ObservationTermCfg(
      func=hammer_mdp.ee_vel_b,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "head_pos": ObservationTermCfg(
      func=hammer_mdp.hammer_head_pos_b,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "head_vel": ObservationTermCfg(
      func=hammer_mdp.hammer_head_vel_b,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "nail_top_pos": ObservationTermCfg(
      func=hammer_mdp.nail_top_pos_w,
      params={"asset_cfg": SceneEntityCfg("nail_block", site_names=("nail_top",))},
      noise=Unoise(n_min=-0.002, n_max=0.002),
    ),
    "nail_depth": ObservationTermCfg(
      func=hammer_mdp.nail_depth,
      params={"asset_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",))},
      noise=Unoise(n_min=-0.001, n_max=0.001),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  critic_terms = {**actor_terms}

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  # --- Actions ---
  # DifferentialIK: 3-D delta position of the hammer head site.
  # frame_name and delta_pos_scale are overridden per-robot.
  actions: dict[str, ActionTermCfg] = {
    "ik_hammer_head": DifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=(),        # filled in per-robot (arm joints only)
      frame_type="site",
      frame_name="",            # filled in per-robot: "hammer_head_site"
      use_relative_mode=True,
      delta_pos_scale=0.05,
      orientation_weight=0.0,   # position-only, 3-D action
    ),
  }

  # --- Events ---
  events = {
    # FUTURE_UPDATES #2a (applied 2026-06-10): ±0.05 rad (~3°) joint noise at
    # reset so the policy must close the loop on joint_pos instead of replaying
    # one memorised trajectory. Zeroed in play mode (config/z1/env_cfgs.py) so
    # validation scripts stay deterministic.
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_nail": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
      },
    ),
  }

  # --- Rewards ---
  rewards = {
    # Approach: guide arm near nail, but keep weight low so hovering is never
    # a stable optimum. Tighter std (0.08) means reward only within ~8 cm.
    "approach": RewardTermCfg(
      func=hammer_mdp.hammer_approach_reward,
      weight=0.1,
      params={
        "std": 0.08,
        "robot_cfg": SceneEntityCfg("robot", site_names=()),   # head site, per-robot
        "nail_cfg": SceneEntityCfg("nail_block", site_names=("nail_top",)),
      },
    ),
    # Gaussian on how far nail has been driven. Still near-zero at 0 mm but
    # provides pull once nail starts moving.
    # std 0.03 -> 0.013 (2026-06-10): scaled with the 6a range change
    # (goal 0.075 -> 0.032) to preserve the design intent. At std=0.03 with the
    # new goal, the term paid ~0.32 at depth 0 (~0.68 reward/step), turning
    # episode survival into a farm that beats the +100 completion bonus.
    "nail_driven": RewardTermCfg(
      func=hammer_mdp.nail_driven_reward,
      weight=2.0,
      params={
        "goal_depth": NAIL_GOAL_DEPTH,
        "std": 0.013,
        "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
      },
    ),
    # Progress reward: max(0, depth - max_depth_so_far). Provides a non-zero
    # gradient from 0mm onward, where the Gaussian nail_driven_reward above
    # is near-zero. Tracks per-episode state via ManagerTermBase.reset().
    "nail_depth_delta": RewardTermCfg(
      func=hammer_mdp.NailDepthDeltaTerm,
      weight=600.0,  # 2000 -> 600 (A1 rebalance); A0 baseline = 2000. Full-drive cumulative ~30-40.
      params={
        "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
      },
    ),
    # Double-gated momentum reward (NEW, #2): pays axial (downward) impact speed
    # only on a fresh hammer->nail contact that advances the nail past its max
    # depth. On the position-only DiffIK action space the controllable impact
    # lever is end-effector momentum, not contact force. v_axial is normalised by
    # ~1 m/s so the per-strike bonus is O(1) before weighting.
    "impact_progress": RewardTermCfg(
      func=hammer_mdp.ImpactProgressTerm,
      weight=8.0,
      params={
        "sensor_name": "hammer_nail_contact",
        "robot_cfg": SceneEntityCfg("robot", site_names=()),  # head site, per-robot
        "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
        "axis": (0.0, 0.0, -1.0),
        "eps": 5e-4,
        "v_expected": 1.0,
      },
    ),
    # Sparse task-completion reward. Fires once when nail crosses success_depth.
    # This is the actual task signal (not shaping); episode terminates on success
    # via the nail_driven TerminationTermCfg.
    "completion": RewardTermCfg(
      func=hammer_mdp.completion_bonus,
      weight=100.0,
      params={
        "success_depth": NAIL_SUCCESS_THRESHOLD,
        "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
      },
    ),
    "action_rate": RewardTermCfg(
      func=hammer_mdp.action_rate_penalty,
      weight=-0.01,
    ),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  # --- Terminations ---
  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "nail_driven": TerminationTermCfg(
      func=hammer_mdp.nail_fully_driven,
      params={
        "success_depth": NAIL_SUCCESS_THRESHOLD,
        "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=2.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",          # filled in per-robot: "link00"
      distance=2.0,
      elevation=-20.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=64,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=10,        # 10 × 0.002 s = 0.02 s control period (50 Hz)
    episode_length_s=20.0,
  )
