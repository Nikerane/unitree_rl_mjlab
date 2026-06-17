"""Unitree Z1 hammer-nail environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.envs.mdp.curriculums import reward_curriculum
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots.unitree_z1.z1_constants import (
  ARM_ACTUATOR_NAMES,
  ARM_JOINT_NAMES,
  EE_SITE_NAME,
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
  get_z1_hammer_robot_cfg,
)
from src.tasks.hammer import mdp as hammer_mdp
from src.tasks.hammer.cat import CatSoftHook
from src.tasks.hammer.hammer_env_cfg import make_hammer_env_cfg
from src.tasks.hammer.nail_block import get_nail_block_entity_cfg


def z1_hammer_env_cfg(
  play: bool = False,
  imitation: bool = False,
  vel_penalty: bool = False,
  cat_vel: bool = False,
  cat_substep: bool = False,
  vel_hard_term: bool = False,
  cat_soft: bool = False,
  dcmotor: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Z1 hammer-nail task configuration.

  Velocity-bound ablation arms (keep delta_pos_scale=0.15; bound joint velocity to
  the real 3.1415 rad/s Z1 limit as a constraint, not by lowering the action scale):
    vel_penalty=True  -- A2: annealed squared-excess reward penalty (joint_vel_excess_penalty).
    cat_vel=True      -- A3: Constraints-as-Terminations on joint velocity (CaTJointVelConstraint).
    dcmotor=True      -- A4: DC-motor torque-speed-envelope arm actuators (plant-level bound).
  All default False -> byte-identical A-BASE.
  """
  cfg = make_hammer_env_cfg(imitation=imitation)

  # --- Scene entities ---
  cfg.scene.entities = {
    "robot": get_z1_hammer_robot_cfg(dcmotor=dcmotor),
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

  # --- Velocity-bound ablation arms (A2/A3) ---
  # Keep delta_pos_scale=0.15; bound arm joint velocity to the real 3.1415 rad/s Z1
  # limit as a constraint. Wave-1 signal = control-rate joint_vel (consistent with the
  # diag_policy_trace eval); substep-peak is the v2 refinement.
  vb_robot_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
  if vel_penalty:
    # A2: squared excess over 0.9*limit, ANNEALED in after the strike is learned so the
    # penalty never blocks learning to strike ("agent refuses to move" trap).
    cfg.rewards["vel_excess"] = RewardTermCfg(
      func=hammer_mdp.joint_vel_excess_penalty,
      weight=0.0,  # ramped negative by the curriculum below
      params={"limit": hammer_mdp.Z1_JOINT_VEL_LIMIT, "beta": 0.9, "robot_cfg": vb_robot_cfg},
    )
    cfg.curriculum["vel_excess_anneal"] = CurriculumTermCfg(
      func=reward_curriculum,
      params={
        "reward_name": "vel_excess",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 1500, "weight": -0.1},
          {"step": 3000, "weight": -0.3},
          {"step": 4500, "weight": -0.5},
        ],
      },
    )
  if cat_vel or cat_substep:
    # A3 / A3-substep: Constraints-as-Terminations. time_out defaults False -> counts as
    # `terminated`, so PPO does not bootstrap it; the policy sees the lost completion bonus.
    # cat_substep reads the within-window substep PEAK (de-confounds A3's control-rate aliasing);
    # p_max kept at 0.5 (same as A3) so ONLY the detection axis changes.
    cfg.terminations["cat_vel"] = TerminationTermCfg(
      func=hammer_mdp.CaTJointVelConstraint,
      params={
        "limit": hammer_mdp.Z1_JOINT_VEL_LIMIT,
        "p_max": 0.5,
        "tau": 0.95,
        "robot_cfg": vb_robot_cfg,
        "detection": "substep" if cat_substep else "control_rate",
      },
    )
  if vel_hard_term:
    # Deterministic hard cap: any arm joint over the limit ends the episode (strongest learned
    # enforcement; the top of the soft->hard sweep). Warmup-gated so the strike forms first.
    cfg.terminations["vel_hard"] = TerminationTermCfg(
      func=hammer_mdp.joint_vel_hard_termination,
      params={
        "limit": hammer_mdp.Z1_JOINT_VEL_LIMIT,
        "warmup_steps": 3600,  # ~30% of a 500-iter run (24 steps/iter); avoids all-hard collapse
        "robot_cfg": vb_robot_cfg,
        "detection": "substep",
      },
    )

  if cat_substep or vel_hard_term:
    # per_substep metric: peak-hold |q̇| inside the decimation loop so the constraint reads the
    # true 500 Hz peak instead of the aliased post-decimation sample. Stashes itself on env.
    cfg.metrics["substep_peak_qv"] = MetricsTermCfg(
      func=hammer_mdp.SubstepPeakJointVel,
      per_substep=True,
    )

  if cat_soft:
    # C3: faithful soft γ(1−δ) CaT. Full-step MetricsTerm computes δ + r_pos and writes env.extras
    # for CatPPO. It is a METRICS term, so it can NEVER feed reset_buf (Decision 5 — a soft violation
    # discounts the value target, it does not end the episode). MUST be paired with the CatPPO rl_cfg
    # (z1_hammer_ppo_runner_cfg(cat_soft=True)) or δ is computed but never consumed (silent no-op).
    cfg.metrics["cat_soft"] = MetricsTermCfg(
      func=CatSoftHook,
      per_substep=False,
      params={
        "limit": hammer_mdp.Z1_JOINT_VEL_LIMIT,
        "max_p": 0.5,
        "min_p": 0.0,
        "tau": 0.95,
        "robot_cfg": vb_robot_cfg,
      },
    )

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
