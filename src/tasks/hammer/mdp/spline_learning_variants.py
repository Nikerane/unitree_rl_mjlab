"""Observable physical startup clock and identities for isolated A/N/R learning."""

from __future__ import annotations

import torch

from src.tasks.hammer.mdp.spline_speed_variants import TASKS as SPLINE_TASKS


TASKS = {
    "O-long": SPLINE_TASKS["O"],
    "A": SPLINE_TASKS["O"] + "-Age",
    "N": SPLINE_TASKS["O"] + "-Age-LowNoise",
    "R": SPLINE_TASKS["O"] + "-Age-StartupRamp",
}
# C is exactly the existing N task. Reward profiles distinguish the new arms
# without changing N's observation marker, model, or checkpoint identity.
TASKS.update({"C": TASKS["N"], "W": TASKS["N"] + "-PathWeight1",
              "S": TASKS["N"] + "-SplitCredit", "D": TASKS["N"] + "-DeferredCredit"})
CREDIT_ARMS = ("C", "W", "S", "D")
ONSET_FRACTIONS = {"C": 1., "W": 1., "S": .2, "D": 0.}
PATH_WEIGHTS = {"C": -.2, "W": -1., "S": -.2, "D": -.2}
AGE_TAIL = ("startup_age", "impact_speed_goal", "impact_speed_pending")
_VARIANTS = ("A", "N", "R")


def startup_age(env, *, variant: str) -> torch.Tensor:
    """Read the controller's actual per-world reset clock, never episode bookkeeping.

    The observation manager probes shapes during environment construction,
    before the sampled controller is installed. Only that pre-install read is
    zero-filled; the live treatment contract requires the installed controller.
    """
    if variant not in _VARIANTS:
        raise ValueError("startup age requires an A, N or R treatment")
    controller = getattr(env, "_sampled_pd500", None)
    if controller is None:
        if hasattr(env, "observation_manager"):
            raise RuntimeError("startup age requires its sampled controller after manager construction")
        return torch.zeros((env.num_envs, 1), device=env.device)
    return controller.startup_age.unsqueeze(-1)


def learning_variant(cfg) -> str | None:
    """Resolve serialized actor/critic markers without adding hidden cfg fields."""
    from src.tasks.hammer.mdp.spline_exploration_pilot import pilot_setting
    if pilot_setting(cfg) is not None:
        return None
    terms = [cfg.observations[group].terms.get("startup_age") for group in ("actor", "critic")]
    if all(term is None for term in terms):
        return None
    if any(term is None or term.func is not startup_age or set(term.params) != {"variant"}
           for term in terms):
        raise ValueError("learning treatment requires matching startup-age observations")
    variant = terms[0].params["variant"]
    if variant not in _VARIANTS or terms[1].params["variant"] != variant:
        raise ValueError("learning treatment requires the same A/N/R identity for actor and critic")
    return variant


def learning_metadata(env) -> dict | None:
    """Bind live controller delivery and policy initialization to exact reloads."""
    variant = learning_variant(env.cfg)
    if variant is None:
        return None
    controller = getattr(env, "_sampled_pd500", None)
    if controller is None or controller.gain_mode != "variable":
        raise ValueError("learning treatment requires sampled variable-impedance control")
    ramp = .1 if variant == "R" else 0.
    if controller.startup_ramp_s != ramp:
        raise ValueError("learning treatment startup ramp differs from its live controller")
    age = controller.startup_age
    if age.shape != (env.num_envs,) or not torch.isfinite(age).all() or not ((age >= 0) & (age <= 1)).all():
        raise ValueError("learning treatment requires finite normalized per-world startup age")
    for group in ("actor", "critic"):
        term = env.observation_manager.get_term_cfg(group, "startup_age")
        if term.func is not startup_age or term.params != {"variant": variant}:
            raise ValueError("live startup-age observation differs from the learning treatment")
    return {
        "schema_version": "spline_learning_v1",
        "variant": variant,
        "startup_age": {
            "source": "sampled_pd500_physics_substeps_since_own_reset",
            "normalization_s": .1,
            "clip": [0., 1.],
            "observation_index": 40,
        },
        "initial_exploration_std": ([.3] * 12 + [1.]) if variant == "N" else [1.] * 13,
        "startup_target_ramp_s": ramp,
    }
