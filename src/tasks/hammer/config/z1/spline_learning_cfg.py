"""Opt-in A/N/R task factories; legacy O remains its original 42-input task."""

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hammer.config.z1.rl_cfg import z1_impact_speed_ppo_runner_cfg
from src.tasks.hammer.mdp.spline_learning_variants import (
    TASKS, CREDIT_ARMS, ONSET_FRACTIONS, PATH_WEIGHTS, startup_age,
)


def spline_learning_env_cfg(*, play: bool, variant: str):
    from src.tasks.hammer.config.z1 import _spline_speed_variant_env_cfg

    if variant not in ("A", "N", "R"):
        raise ValueError("learning treatment must be A, N or R")
    cfg = _spline_speed_variant_env_cfg(play=play, variant="O")
    for group in cfg.observations.values():
        terms = {}
        for name, term in group.terms.items():
            if name == "impact_speed_goal":
                terms["startup_age"] = ObservationTermCfg(func=startup_age, params={"variant": variant})
            terms[name] = term
        group.terms = terms
    return cfg


def spline_learning_rl_cfg(variant: str):
    if variant not in ("A", "N", "R"):
        raise ValueError("learning treatment must be A, N or R")
    cfg = z1_impact_speed_ppo_runner_cfg()
    prefix = "src.tasks.hammer.rl.impact_speed_policy:"
    cfg.actor.class_name = prefix + ("ImpactSpeedLowNoiseModel" if variant == "N" else "ImpactSpeedAgeModel")
    cfg.critic.class_name = prefix + "ImpactSpeedAgeModel"
    return cfg


def spline_credit_env_cfg(*, play: bool, arm: str):
    """Isolate stronger path weight or later credit on the unchanged N recipe."""
    from src.tasks.hammer.mdp.spline_boundary_credit import BoundarySplineStrikeReward

    if arm not in CREDIT_ARMS:
        raise ValueError("credit treatment must be C, W, S or D")
    cfg = spline_learning_env_cfg(play=play, variant="N")
    cfg.rewards["spline_path_cost"].weight = PATH_WEIGHTS[arm]
    if arm in ("S", "D"):
        reward = cfg.rewards["accepted_spline_strike"]
        reward.func = BoundarySplineStrikeReward
        reward.params["onset_fraction"] = ONSET_FRACTIONS[arm]
    return cfg


def spline_credit_rl_cfg(arm: str):
    if arm not in CREDIT_ARMS:
        raise ValueError("credit treatment must be C, W, S or D")
    return spline_learning_rl_cfg("N")


def register_spline_learning_tasks():
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

    for variant in ("A", "N", "R"):
        register_mjlab_task(
            task_id=TASKS[variant],
            env_cfg=spline_learning_env_cfg(play=False, variant=variant),
            play_env_cfg=spline_learning_env_cfg(play=True, variant=variant),
            rl_cfg=spline_learning_rl_cfg(variant),
            runner_cls=HammerOnPolicyRunner,
        )
    for arm in ("W", "S", "D"):
        register_mjlab_task(
            task_id=TASKS[arm],
            env_cfg=spline_credit_env_cfg(play=False, arm=arm),
            play_env_cfg=spline_credit_env_cfg(play=True, arm=arm),
            rl_cfg=spline_credit_rl_cfg(arm),
            runner_cls=HammerOnPolicyRunner,
        )
