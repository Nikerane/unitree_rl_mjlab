"""Z1 hammer-nail task configurations."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.hammer.config.z1.env_cfgs import (
    install_z1_joint_position_action,
    z1_hammer_env_cfg,
)
from src.tasks.hammer.config.z1.joint_position_contract import JOINT_NAMES
from src.tasks.hammer.mdp.rewards import FirstStrikeBoundedImpactRewardTerm
from src.tasks.hammer.mdp.trackability import joint_trackability_cost
from src.tasks.hammer.config.z1.rl_cfg import z1_hammer_ppo_runner_cfg
from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

register_mjlab_task(
    task_id="Unitree-Z1-Hammer",
    env_cfg=z1_hammer_env_cfg(),
    play_env_cfg=z1_hammer_env_cfg(play=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)

# A-TRACK arm: A-BASE + the weak-annealed tracking prior r_imit (plan T2).
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-Track",
    env_cfg=z1_hammer_env_cfg(imitation=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, imitation=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)

# Velocity-bound ablation arms (keep delta_pos_scale=0.15; bound joint velocity to the
# real 3.1415 rad/s Z1 limit as a constraint). See docs/research/reward-design/
# CONSTRAINED_RL_LANDSCAPE.md. A1 (delta->0.10) needs no task -- it is a CLI override.
# A2: annealed squared joint-velocity-excess reward penalty.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-VPenalty",
    env_cfg=z1_hammer_env_cfg(vel_penalty=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, vel_penalty=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)

# A3: Constraints-as-Terminations on joint velocity (CaT, Chane-Sane et al. IROS 2024).
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT",
    env_cfg=z1_hammer_env_cfg(cat_vel=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, cat_vel=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)

# A4: DC-motor torque-speed-envelope arm (plant-level velocity bound; trains inside the
# real motor envelope so >3.1415 rad/s is unreachable to the actuator).
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-DcMotor",
    env_cfg=z1_hammer_env_cfg(dcmotor=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, dcmotor=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)

# A3-substep: CaT on the SUBSTEP-PEAK |q̇| (de-confounds A3's control-rate aliasing; same
# p_max=0.5). Tests whether catching the 500 Hz peak tightens the worst-case bound.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Substep",
    env_cfg=z1_hammer_env_cfg(cat_substep=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, cat_substep=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)

# Hard cap: deterministic episode termination on substep-peak |q̇| > limit (strongest learned
# enforcement; warmup-gated). Tests whether fixed PD can learn a limit-respecting hard strike,
# or can only comply by striking softly (the strike-vs-honesty tension that motivates VIC).
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-VelHardTerm",
    env_cfg=z1_hammer_env_cfg(vel_hard_term=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, vel_hard_term=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)

# Faithful soft γ(1−δ) CaT (FAITHFUL_SOFT_CAT_IMPL_PLAN.md): the soft termination probability δ
# discounts the value-target bootstrap (CatPPO dual-mask GAE) and the positive reward
# (scale-positives, Decision 1) -- it NEVER ends the episode (Decision 5). The primary fixed-impedance
# thesis arm. NOTE: cat_soft=True must be set on BOTH env_cfg (installs the δ/r_pos hook) AND rl_cfg
# (selects CatPPO); one without the other is a silent no-op.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Soft",
    env_cfg=z1_hammer_env_cfg(cat_soft=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, cat_soft=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# C0 IMPULSE arm (IMPULSE_CAT_IMPL_PLAN.md): the thesis headline -- BOUND the robot-side per-joint
# reaction impulse Λ_j via the SAME soft-CaT machinery (LOG-ONLY at C0, imp_max_p=0) WHILE MAXIMIZING
# the object-side delivered impulse (DeliveredImpulseTerm reward). Reuses CatPPO (cat_soft=True rl_cfg):
# at imp_max_p=0, δ≡0 so the discount + dual-mask GAE are exact no-ops -- it trains identically to the
# baseline until enforcement is turned on. Separate from the velocity arm for clean attribution.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse",
    env_cfg=z1_hammer_env_cfg(cat_impulse=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, cat_impulse=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# D-prime comparator: the shared first-event boundary from F/E with the
# unchanged 50 Hz legacy maximize measurements, emitted once at finalization.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy",
    env_cfg=z1_hammer_env_cfg(cat_impulse=True, first_strike_legacy=True),
    play_env_cfg=z1_hammer_env_cfg(
        play=True, cat_impulse=True, first_strike_legacy=True
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# Experimental first-strike event-credit arm. The shipped impulse task above remains
# the control; this arm changes only the shared event tracker and the two maximize readers.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event",
    env_cfg=z1_hammer_env_cfg(cat_impulse=True, event_correct=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, cat_impulse=True, event_correct=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# Arm F: the same first-strike event semantics as the saturated Event arm above,
# but with linear delivered-impulse payout beyond the event reference.  This is
# still fixed-impedance and log-only (imp_max_p=0); only the reward payout shape
# differs.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    env_cfg=z1_hammer_env_cfg(
        cat_impulse=True, event_correct=True, event_linear=True
    ),
    play_env_cfg=z1_hammer_env_cfg(
        play=True, cat_impulse=True, event_correct=True, event_linear=True
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# Cartesian straight-waypoint study: fresh matched F8 controls share the same
# always-on tracker and policy observations; only their reward reader differs.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
    env_cfg=z1_hammer_env_cfg(
        cat_impulse=True, event_correct=True, event_linear=True, guideline=True
    ),
    play_env_cfg=z1_hammer_env_cfg(
        play=True,
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        guideline=True,
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate",
    env_cfg=z1_hammer_env_cfg(
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        guideline=True,
        gate_reward=True,
    ),
    play_env_cfg=z1_hammer_env_cfg(
        play=True,
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        guideline=True,
        gate_reward=True,
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress",
    env_cfg=z1_hammer_env_cfg(
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        guideline=True,
        progress_reward=True,
    ),
    play_env_cfg=z1_hammer_env_cfg(
        play=True,
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        guideline=True,
        progress_reward=True,
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# P+V: the CProgress arm plus faithful soft velocity-CaT on the per-joint 500 Hz substep peak.
# Wave-2 showed all 18 C0/G/P policies exceed the 3.1415 rad/s Z1 joint-velocity limit while the
# impulse constraint stayed at 0.128 of cap — but velocity enforcement was OFF, so that is not
# evidence that CaT fails. This arm turns it on (max_p=0.5, min_p=0.0, tau=0.95, no curriculum) and
# changes NOTHING else: the impulse constraint stays log-only (imp_max_p=0), the caps are unchanged,
# and there is no deterministic termination or action clipping — legality must come from behaviour.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel",
    env_cfg=z1_hammer_env_cfg(
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        guideline=True,
        progress_reward=True,
        cat_soft=True,
        vel_cat_substep=True,
    ),
    play_env_cfg=z1_hammer_env_cfg(
        play=True,
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        guideline=True,
        progress_reward=True,
        cat_soft=True,
        vel_cat_substep=True,
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)


def _presentation_i_off_env_cfg(
    *,
    play: bool = False,
    velocity_cat: bool = False,
    delivered_weight: float,
):
    """Build one fixed-plant, impulse-log-only presentation arm.

    The helper isolates only the delivered-reward dose and optional velocity CaT.
    Active impulse CaT is excluded until the current sliding-window semantics receive
    a new C2 dose calibration.
    """
    cfg = z1_hammer_env_cfg(
        play=play,
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        guideline=True,
        progress_reward=True,
        cat_soft=velocity_cat,
        vel_cat_substep=velocity_cat,
    )
    # Explicitly pin the study arm even though the base factory is also log-only.
    cfg.metrics["cat_soft"].params["imp_max_p"] = 0.0
    cfg.rewards["delivered_impulse"].weight = float(delivered_weight)
    return cfg


# Presentation Phase 1 contains the matched velocity-on/off D2/D4 cells. The
# velocity-on D0 contingency below adds the missing zero-dose comparison without
# activating or retuning impulse CaT.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Delivered4",
    env_cfg=_presentation_i_off_env_cfg(delivered_weight=4.0),
    play_env_cfg=_presentation_i_off_env_cfg(play=True, delivered_weight=4.0),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

register_mjlab_task(
    task_id=(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered0"
    ),
    env_cfg=_presentation_i_off_env_cfg(
        velocity_cat=True, delivered_weight=0.0
    ),
    play_env_cfg=_presentation_i_off_env_cfg(
        play=True, velocity_cat=True, delivered_weight=0.0
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

register_mjlab_task(
    task_id=(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered4"
    ),
    env_cfg=_presentation_i_off_env_cfg(
        velocity_cat=True, delivered_weight=4.0
    ),
    play_env_cfg=_presentation_i_off_env_cfg(
        play=True, velocity_cat=True, delivered_weight=4.0
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)


def _install_joint_trackability_cost(cfg):
    cfg.rewards["r_tt"] = RewardTermCfg(
        func=joint_trackability_cost,
        weight=-1.0,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES, preserve_order=True
            ),
            "k_tt": 1.0,
        },
    )
    return cfg


_FIC_CONTROLLED_DROP_I_REF_N_S = 0.2799950838088989


def _joint_position_fixed_env_cfg(*, play: bool, trackability: bool):
    cfg = _presentation_i_off_env_cfg(
        play=play, velocity_cat=True, delivered_weight=4.0
    )
    # FIC-only calibration: exact five-drop mean from the corrected 17.5 mm
    # controlled-drop fixture. Cartesian registrations retain their historical
    # first-strike normalizer through z1_hammer_env_cfg.
    cfg.rewards["delivered_impulse"].params["i_ref"] = (
        _FIC_CONTROLLED_DROP_I_REF_N_S
    )
    # This diagnostic-only contact-row decomposition is not qualified at the
    # 4,096-environment pilot scale. The production impulse accumulator remains on.
    cfg.metrics["substep_impulse_rows"].params["enabled"] = False
    install_z1_joint_position_action(cfg)
    if trackability:
        _install_joint_trackability_cost(cfg)
    return cfg


def _direct_reference_joint_position_fixed_env_cfg(*, play: bool, trackability: bool):
    """Build the direct-reference FIC treatment without waypoint guidance."""
    cfg = z1_hammer_env_cfg(
        play=play,
        imitation=True,
        cat_impulse=True,
        event_correct=True,
        event_linear=True,
        cat_soft=True,
        vel_cat_substep=True,
    )
    # FIC-only calibration and production-scale diagnostic setting match the
    # banked joint-position pair.  The production accumulator remains enabled.
    cfg.rewards["delivered_impulse"].weight = 4.0
    cfg.rewards["delivered_impulse"].params["i_ref"] = (
        _FIC_CONTROLLED_DROP_I_REF_N_S
    )
    cfg.metrics["substep_impulse_rows"].params["enabled"] = False
    reset = cfg.events["reset_robot_joints"].params
    reset["position_range"] = (0.0, 0.0)
    reset["velocity_range"] = (0.0, 0.0)
    install_z1_joint_position_action(cfg)
    if trackability:
        _install_joint_trackability_cost(cfg)
    return cfg


register_mjlab_task(
    task_id=(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered4-JointPosition-Fixed"
    ),
    env_cfg=_joint_position_fixed_env_cfg(play=False, trackability=False),
    play_env_cfg=_joint_position_fixed_env_cfg(play=True, trackability=False),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

register_mjlab_task(
    task_id=(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered4-JointPosition-Fixed-TT"
    ),
    env_cfg=_joint_position_fixed_env_cfg(play=False, trackability=True),
    play_env_cfg=_joint_position_fixed_env_cfg(play=True, trackability=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# Direct-reference FIC pair: use the weak annealed task-space prior, but do not
# install the Cartesian waypoint guidance observations, metric, or reward.
register_mjlab_task(
    task_id=(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
        "JointPosition-Fixed"
    ),
    env_cfg=_direct_reference_joint_position_fixed_env_cfg(
        play=False, trackability=False
    ),
    play_env_cfg=_direct_reference_joint_position_fixed_env_cfg(
        play=True, trackability=False
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

register_mjlab_task(
    task_id=(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
        "JointPosition-Fixed-TT"
    ),
    env_cfg=_direct_reference_joint_position_fixed_env_cfg(
        play=False, trackability=True
    ),
    play_env_cfg=_direct_reference_joint_position_fixed_env_cfg(
        play=True, trackability=True
    ),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# F0 and D0 are fresh copies of F8 with exactly one maximize-term weight
# removed.  They retain the linear event readers and fixed-impedance plant.
_f0_env_cfg = z1_hammer_env_cfg(
    cat_impulse=True, event_correct=True, event_linear=True
)
_f0_env_cfg.rewards["impact_progress"].weight = 0.0
_f0_play_env_cfg = z1_hammer_env_cfg(
    play=True, cat_impulse=True, event_correct=True, event_linear=True
)
_f0_play_env_cfg.rewards["impact_progress"].weight = 0.0
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
    env_cfg=_f0_env_cfg,
    play_env_cfg=_f0_play_env_cfg,
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

_d0_env_cfg = z1_hammer_env_cfg(
    cat_impulse=True, event_correct=True, event_linear=True
)
_d0_env_cfg.rewards["delivered_impulse"].weight = 0.0
_d0_play_env_cfg = z1_hammer_env_cfg(
    play=True, cat_impulse=True, event_correct=True, event_linear=True
)
_d0_play_env_cfg.rewards["delivered_impulse"].weight = 0.0
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
    env_cfg=_d0_env_cfg,
    play_env_cfg=_d0_play_env_cfg,
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# FQ-min adds one bounded quality-conditioned speed reader to D0's disabled
# delivered treatment. Its only added sensor is passive contact-quality
# instrumentation.
_fq_env_cfg = z1_hammer_env_cfg(
    cat_impulse=True,
    event_correct=True,
    event_quality=True,
    quality_instrumentation=True,
)
_fq_env_cfg.rewards["delivered_impulse"].weight = 0.0
_fq_play_env_cfg = z1_hammer_env_cfg(
    play=True,
    cat_impulse=True,
    event_correct=True,
    event_quality=True,
    quality_instrumentation=True,
)
_fq_play_env_cfg.rewards["delivered_impulse"].weight = 0.0
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
    env_cfg=_fq_env_cfg,
    play_env_cfg=_fq_play_env_cfg,
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# B8 is FQ-min's center-blind bounded-speed control: retain its passive
# eight-slot quality instrumentation while replacing only the impact reader.
_b8_env_cfg = z1_hammer_env_cfg(
    cat_impulse=True,
    event_correct=True,
    quality_instrumentation=True,
)
_b8_env_cfg.rewards["impact_progress"].func = FirstStrikeBoundedImpactRewardTerm
_b8_env_cfg.rewards["impact_progress"].params["v_expected"] = 1.4598331451416016
_b8_env_cfg.rewards["delivered_impulse"].weight = 0.0
_b8_env_cfg.rewards["delivered_impulse"].params["saturate"] = False
_b8_play_env_cfg = z1_hammer_env_cfg(
    play=True,
    cat_impulse=True,
    event_correct=True,
    quality_instrumentation=True,
)
_b8_play_env_cfg.rewards["impact_progress"].func = FirstStrikeBoundedImpactRewardTerm
_b8_play_env_cfg.rewards["impact_progress"].params["v_expected"] = 1.4598331451416016
_b8_play_env_cfg.rewards["delivered_impulse"].weight = 0.0
_b8_play_env_cfg.rewards["delivered_impulse"].params["saturate"] = False
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
    env_cfg=_b8_env_cfg,
    play_env_cfg=_b8_play_env_cfg,
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# C0 IMPULSE + reference-guided arm: the -CaT-Impulse measurement arm (cat_impulse=True) PLUS the
# weak-annealed ante-impact tracking prior r_imit (imitation=True) -- i.e. "reference-guided online
# RL". The scripted SingleStrikeReference bootstraps the approach (r_imit weight 0.1 -> 0 by iter 250,
# ante-impact gated), then hands off to free RL for the impact. The prior never touches the Λ
# measurement MACHINERY (substep accumulators/hook are byte-identical to -CaT-Impulse and never read
# r_imit or phase) -- but the trained policy remains a prior-shaped policy after the anneal; that
# path dependence is precisely the treatment the prior-vs-none pair (this vs -CaT-Impulse) isolates.
# The weak-prior form is the light-touch reference guidance that SURVIVED the DeepMimic/
# generate-then-track walk-back -- heavier persistent tracking would need Khadiv re-confirmation.
# imitation+cat_impulse are independent factory blocks (validate_rewards Phase K / Phase M exercise
# them separately). rl_cfg mirrors -CaT-Impulse (CatPPO, imp_max_p=0 -> δ≡0, so log-only).
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-Track",
    env_cfg=z1_hammer_env_cfg(cat_impulse=True, imitation=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, cat_impulse=True, imitation=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)

# DAPG-style NON-TERMINATING variant of -CaT-Impulse (Option B, anti-parking): success no longer
# ends the episode (time_out still truncates at 20 s), so holding just under the 0.030 threshold can
# no longer out-earn completing (the terminate-on-success forfeit parked the mean policy at ~28 mm).
# The un-latched completion bonus is retuned 100 -> 1.0/step (the popped termination was what made it
# one-shot). Λ machinery byte-identical to -CaT-Impulse; log-only (imp_max_p=0 -> δ≡0). The PLAY cfg
# deliberately KEEPS the success termination (eval keys success on reset_terminated / the
# Episode_Termination-nail_driven panel; a frozen policy's success rate is unaffected by eval-time
# termination). Do NOT pair a NoTerm arm against a terminating one in a prior-vs-none study.
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-CaT-Impulse-NoTerm",
    env_cfg=z1_hammer_env_cfg(cat_impulse=True, no_terminate=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, cat_impulse=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(cat_soft=True),
    runner_cls=HammerOnPolicyRunner,
)
