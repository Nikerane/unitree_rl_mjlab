"""Unitree Z1 hammer-nail environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.envs.mdp.curriculums import reward_curriculum
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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

# Fixture-era per-joint impulse caps, MEASURED by derive_impulse_thresholds.py on 2026-07-06 (windup
# NEAR_NAIL reset, oblique contact; committed record: docs/results/2026-07-10_c2_enforcement_record.md).
# J_limit_j = tau_rated_j x 2 (HD Repeated-Peak) x Delta_t_impact at the 2026-07-06 measured
# Delta_t ~= 27.3 ms — NOTE the shipped sliding window integrates 50 ms and the post-reference-fix
# impact window measures ~44 ms; whether to re-derive at a different Delta_t is the window/cap
# pairing inside Khadiv decision (e). Values stay FIXED until that decision. Module-level (not
# function-local) so downstream tooling (e.g. scripts/eval_impulse.py) IMPORTS this instead of
# hardcoding a second copy that could drift.
IMP_J_LIMIT: list[float] = [1.640, 3.280, 1.640, 1.640, 1.640, 1.640]  # N·m·s


def _wire_site(cfg, obs_keys: tuple[str, ...], param_key: str, site_name: str) -> None:
  """Set ``site_names=(site_name,)`` on the given obs terms' ``param_key`` SceneEntityCfg, across
  every observation group that carries the term (actor + critic share the term objects)."""
  for obs_key in obs_keys:
    for group in cfg.observations.values():
      assert isinstance(group, ObservationGroupCfg)
      if obs_key in group.terms:
        group.terms[obs_key].params[param_key].site_names = (site_name,)


def z1_hammer_env_cfg(
  play: bool = False,
  imitation: bool = False,
  vel_penalty: bool = False,
  cat_vel: bool = False,
  cat_substep: bool = False,
  vel_hard_term: bool = False,
  cat_soft: bool = False,
  cat_impulse: bool = False,
  dcmotor: bool = False,
  no_terminate: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Z1 hammer-nail task configuration.

  Velocity-bound ablation arms (keep delta_pos_scale=0.15; bound joint velocity to
  the real 3.1415 rad/s Z1 limit as a constraint, not by lowering the action scale):
    vel_penalty=True  -- A2: annealed squared-excess reward penalty (joint_vel_excess_penalty).
    cat_vel=True      -- A3: Constraints-as-Terminations on joint velocity (CaTJointVelConstraint).
    dcmotor=True      -- A4: DC-motor torque-speed-envelope arm actuators (plant-level bound).
  All default False -> byte-identical A-BASE.
  """
  # Guard (2026-07-14 audit): vel_penalty's `vel_excess` term starts at weight 0.0 and is ramped
  # NEGATIVE by curriculum at env-step 1500 — but CatSoftHook's _NEG_TERMS penalty-evasion guard
  # inspects weights ONCE, lazily, at its first __call__. Composing vel_penalty with a soft-CaT
  # hook arm would pass the guard at step 0 and then silently (1-δ)-discount the ramped penalty:
  # the exact exploit Decision 1 (scale-positives) exists to prevent. No registered arm composes
  # them; fail loudly if one ever does (fix = add vel_excess to _NEG_TERMS + re-check the guard
  # after curriculum mutations, then relax this).
  if vel_penalty and (cat_soft or cat_impulse):
    raise ValueError(
      "z1_hammer_env_cfg: vel_penalty cannot be combined with cat_soft/cat_impulse — the "
      "curriculum-ramped negative vel_excess weight bypasses CatSoftHook's one-shot _NEG_TERMS "
      "guard (penalty-evasion exploit). See the guard comment in env_cfgs.py."
    )
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
    primary=ContactMatch(mode="geom", pattern="hammer_head_0", entity="robot"),
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
  _wire_site(cfg, ("ee_pos", "ee_vel"), "asset_cfg", EE_SITE_NAME)
  _wire_site(cfg, ("head_pos", "head_vel"), "asset_cfg", HAMMER_HEAD_SITE_NAME)
  # Strike-reference obs (T1) carry the head site under "robot_cfg", not "asset_cfg".
  _wire_site(cfg, ("strike_phase", "strike_ref_error"), "robot_cfg", HAMMER_HEAD_SITE_NAME)

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

  if cat_soft and not cat_impulse:
    # C3: faithful soft γ(1−δ) CaT. Full-step MetricsTerm computes δ + r_pos and writes env.extras
    # for CatPPO. It is a METRICS term, so it can NEVER feed reset_buf (Decision 5 — a soft violation
    # discounts the value target, it does not end the episode). MUST be paired with the CatPPO rl_cfg
    # (z1_hammer_ppo_runner_cfg(cat_soft=True)) or δ is computed but never consumed (silent no-op).
    # (When cat_impulse is ALSO set, the cat_impulse block below installs ONE hook with BOTH
    # constraints — velocity ∪ impulse soft-OR, the C5 composition — instead of overwriting this.)
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

  if cat_impulse:
    # C0 IMPULSE arm (Unitree-Z1-Hammer-CaT-Impulse): the thesis headline -- BOUND the robot-side
    # per-joint reaction impulse (soft-CaT, LOG-ONLY at C0) WHILE MAXIMIZING the object-side delivered
    # impulse (reward). Run as a SEPARATE arm from velocity (use_vel=False) for clean attribution;
    # velocity∪impulse combined is the trivial C5 step. See IMPULSE_CAT_IMPL_PLAN.md C0.

    # Object-side net contact wrench in the WORLD frame (reduce=netforce): the delivered-impulse
    # signal AND the weld/friction-immune ground truth for the C0 quantity gate (the sensor sees ONLY
    # the hammer<->nail contact). Separate from hammer_nail_contact (maxforce, used by impact_progress).
    hammer_nail_impulse = ContactSensorCfg(
      name="hammer_nail_impulse",
      primary=ContactMatch(mode="geom", pattern="hammer_head_0", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="nail", entity="nail_block"),
      fields=("found", "force"),
      reduce="netforce",
      track_air_time=False,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (hammer_nail_impulse,)

    # Robot-side per-joint reaction impulse Λ_j = Σ|qfrc_constraint_j|·dt, contact-anchored, 500 Hz.
    cfg.metrics["substep_impulse"] = MetricsTermCfg(
      func=hammer_mdp.SubstepImpulseAccumulator,
      per_substep=True,
      reduce="last",  # the term returns the EPISODE-PEAK worst-joint Λ — log the final value
      params={
        "sensor_name": "hammer_nail_contact",
        "robot_cfg": vb_robot_cfg,
        "subtract_baseline": True,  # C2 (2026-07-06): enforce the baseline-subtracted Λ_j — removes
        # the dominant dof-friction share; residual quantified by the Track-2 contact-row metric.
        # Sliding-window length (2026-07-13): Λ_j = reaction impulse over the most recent 25
        # substeps (50 ms) — explicit here (not the class default) so the C0 gate mirror can never
        # silently desync from the shipped value. NOTE: scales with physics_dt if that ever changes.
        "event_window_substeps": 25,
      },
    )
    # Track-2 RIGOROUS metric: contact-row-only Λ (JᵀF over hammer↔nail efc rows) — validates the
    # enforced baseline-subtracted Λ; LOG-ONLY, never feeds joint_impulse_excess/δ.
    # "enabled" is the per-run perf toggle (I6): the per-contact Python loop is untested at GPU
    # scale — disable via --env.metrics.substep-impulse-rows.params.enabled False if profiling
    # shows it dominating the step budget (diagnostic-only metric, nothing consumes it).
    cfg.metrics["substep_impulse_rows"] = MetricsTermCfg(
      func=hammer_mdp.ContactRowImpulseAccumulator,
      per_substep=True,
      reduce="last",
      # No robot_cfg param: the class hardcodes the arm joints via arm_dof_cols/ARM_JOINT_NAMES
      # (a robot_cfg passed here was silently ignored — removed 2026-07-14, audit finding).
      params={"sensor_name": "hammer_nail_contact", "enabled": True},
    )
    # Object-side delivered axial impulse (episode-cumulative, per-event capped — see the class).
    cfg.metrics["substep_delivered"] = MetricsTermCfg(
      func=hammer_mdp.SubstepDeliveredImpulse,
      per_substep=True,
      reduce="last",  # cumulative signal — log the episode-final total, not a time-average
      params={"sensor_name": "hammer_nail_impulse", "axis": (0.0, 0.0, -1.0),
              "event_window_substeps": 25,  # explicit (2026-07-13): keep gate mirror in sync
              "rearm_gap_substeps": 25},  # flicker-re-arm debounce (2026-07-14, C1 farm fix)
    )
    # I3 fix (2026-07-14): the AUTHORITATIVE episode-total delivered impulse. The per-substep
    # metric above logs the TERMINAL step's substep-MEAN of the ramping cumulative signal
    # (metrics_manager reduce="last" reads _step_values = the within-step average), undercounting
    # strike-terminated episodes arm-dependently — unusable for the C3 cross-arm comparison.
    # This full-step reader logs the exact buffer value (same idiom as imp_peak_* below).
    cfg.metrics["delivered_total"] = MetricsTermCfg(
      func=hammer_mdp.delivered_impulse_total, per_substep=False, reduce="last", params={},
    )
    # soft-CaT hook with the IMPULSE constraint, LOG-ONLY (imp_max_p=0 ⇒ δ≡0). MUST pair with the
    # CatPPO rl_cfg (z1_hammer_ppo_runner_cfg(cat_soft=True)). Do NOT raise imp_max_p until (a) the C0
    # quantity gate passes, (b) Khadiv confirms soft-CaT (vs CMDP/Lagrangian), (c) the local gate is green.
    # use_vel follows the cat_soft flag: cat_impulse alone = impulse-only (clean attribution);
    # cat_soft + cat_impulse = ONE hook with BOTH constraints (velocity ∪ impulse soft-OR, C5) —
    # previously this block silently overwrote the velocity hook (2026-07 review finding #8).
    # IMP_J_LIMIT is the module-level constant defined above.
    cfg.metrics["cat_soft"] = MetricsTermCfg(
      func=CatSoftHook,
      per_substep=False,
      params={
        "use_vel": bool(cat_soft),
        "use_impulse": True,
        "imp_limit": IMP_J_LIMIT,  # measured fixture-era per-joint caps (see above)
        "imp_max_p": 0.0,  # log-only default; C2/C3 raise it per-run via
        #   --env.metrics.cat-soft.params.imp-max-p (verified tyro flag)
        "imp_seed": 1e-3,  # normalizer DECAY FLOOR only — never a p95/excess statistic (hook.py)
        "robot_cfg": vb_robot_cfg,
        "limit": hammer_mdp.Z1_JOINT_VEL_LIMIT,
        "max_p": 0.5,
        "min_p": 0.0,
        "tau": 0.95,
      },
    )
    # Per-joint episode-peak Λ (TB: Episode_Metrics/imp_peak_joint1..6) — the authoritative peaks;
    # J_limit differs 2× across joints (joint2 τ_rated=60), so worst-joint alone can't be compared
    # to the cap vector. cat_delta_peak: peak binding pressure (episode-mean δ dilutes strikes).
    for _j, _jn in enumerate(ARM_JOINT_NAMES):  # single-sourced; positional order pinned by tests
      cfg.metrics[f"imp_peak_{_jn}"] = MetricsTermCfg(
        func=hammer_mdp.joint_impulse_peak, per_substep=False, reduce="last", params={"joint": _j},
      )
    cfg.metrics["cat_delta_peak"] = MetricsTermCfg(
      func=hammer_mdp.CatDeltaPeak, per_substep=False, reduce="last", params={},
    )
    # Live instrumentation SENTINELS (Tier-3 safety net, 2026-07-14): TB alarms that make a
    # dead measurement instrument visible DURING training, not after (the first GPU smoke logged
    # Λ≡0 while success=1.0 and no curve revealed it). contact_seen ≈ 1.0 healthy (independent
    # object-side liveness); impossible_success flat 0.0 healthy (success-with-zero-Λ = dead qfrc
    # path). Both arms carry them (pair integrity: only {r_imit, r_imit_anneal} differs).
    cfg.metrics["contact_seen"] = MetricsTermCfg(
      func=hammer_mdp.contact_seen, per_substep=False, reduce="last",
      params={"sensor_name": "hammer_nail_contact"},
    )
    cfg.metrics["impossible_success"] = MetricsTermCfg(
      func=hammer_mdp.impossible_success, per_substep=False, reduce="last", params={},
    )
    # Ordering contract → guard (2026-07-14 audit): the metrics manager evaluates full-step terms
    # in dict-insertion order, and cat_delta_peak reads the δ that cat_soft wrote to env.extras in
    # the SAME compute pass. Until now this was guaranteed only by the line order in this function.
    _mk = list(cfg.metrics)
    assert _mk.index("cat_soft") < _mk.index("cat_delta_peak"), (
      "cat_delta_peak must be registered AFTER cat_soft (insertion-order evaluation; it reads "
      "this step's env.extras['cat_delta'])."
    )
    # MAXIMIZE objective: object-side delivered impact impulse (positive term; rides the (1−δ) discount
    # so an over-limit strike's delivered-impulse reward is worth less -- the two-sides interplay).
    # i_ref contract: the reference strike's delivered impulse, MEASURED via the C0 gate
    # (derive_impulse_thresholds.py section [3] mean). 0.0811 (2026-07-06, endpoint-servo-era
    # reference) -> 0.6094 (2026-07-13): the F3 follow-through fix turned the scripted reference
    # into a genuine 1.37 m/s in-script strike, so the reference-level impulse baseline rose ~7.5x.
    # delivered_impulse SHARE (26.5% of positive) was measured at C2 against the OLD normalizer
    # (committed record: docs/results/2026-07-10_c2_enforcement_record.md, incl. the staleness
    # note); re-check the share on the first post-fix training run before trusting the weight.
    cfg.rewards["delivered_impulse"] = RewardTermCfg(
      func=hammer_mdp.DeliveredImpulseTerm,
      weight=2.0,
      params={
        "i_ref": 0.6094,
        "eps": 5e-4,
        "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
        # Phase-2 maximization sweep: keep True (shipped). Flip via
        #   --env.rewards.delivered-impulse.params.depth-gate False
        # to pay delivered impulse into a seated nail (impulse ungated from depth progress).
        "depth_gate": True,
      },
    )

  # --- Non-terminating (DAPG-style) variant: anti-parking (Option B) ---
  if no_terminate:
    # Success no longer ENDS the episode (time_out still truncates at 20 s), so completing no longer
    # forfeits the per-step reward streams -- the ratcheted nail (frictionloss=30, no spring, gravcomp)
    # otherwise farms nail_driven just under the 0.030 success line (hold ~193 discounted > one-time
    # completion +100 at weight 2.0), parking the mean policy at ~28 mm. completion_bonus is UN-LATCHED
    # (rewards.py fires every step depth>=success_depth); the termination was the only thing making it
    # one-shot, so its weight MUST become a per-step stream: 1.0/step == the old one-time 100 in
    # discounted value (1/(1-gamma)=100 at gamma=0.99).
    cfg.terminations.pop("nail_driven")
    cfg.rewards["completion"].weight = 1.0

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
