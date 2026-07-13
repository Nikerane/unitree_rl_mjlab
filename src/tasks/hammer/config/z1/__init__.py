"""Z1 hammer-nail task configurations."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
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
