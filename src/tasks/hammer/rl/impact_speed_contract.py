"""Versioned contracts for the experimental 13-action impact-speed policy.

The exported actor emits raw deterministic actions. Its consumer still owns the
episode goal latch, reset, clipping, and goal/pending observation feedback.
"""

from __future__ import annotations

import json

import torch

from src.tasks.hammer.mdp.first_strike import FirstStrikeEventTracker, _ENV_FIRST_STRIKE_ATTR
from src.tasks.hammer.mdp.impact_speed import (
    ImpactSpeedAction,
    ImpactSpeedCost,
    ImpactSpeedState,
    impact_speed_goal,
    impact_speed_pending,
    validate_speed_goal_weight,
)
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedModel, ImpactSpeedNormalization
from src.tasks.hammer.rl.impact_speed_wrapper import IMPACT_SPEED_ACTION_SIGNATURE


_SCHEMA = "impact_speed_v1"
_TAIL = ("impact_speed_goal", "impact_speed_pending")
_OBSERVATION_WIDTHS = (7, 7, 3, 3, 3, 3, 3, 1, 1, 3, 6, 1, 1)


def _observation_metadata(env) -> dict:
    # Shared validation is imported lazily to keep the legacy runner's public
    # dispatch small without introducing a second copy of its controller rules.
    from src.tasks.hammer.rl.runner import _DIRECT_JOINT_OBSERVATIONS, _joint_observation_metadata

    metadata = _joint_observation_metadata(env, scalar_tail=_TAIL)
    if tuple(metadata["observation_names"]) != (*_DIRECT_JOINT_OBSERVATIONS, *_TAIL):
        raise ValueError("impact speed requires the direct-reference observation order")
    for group in ("actor", "critic"):
        widths = env.observation_manager.group_obs_term_dim[group]
        if tuple(widths) != tuple((width,) for width in _OBSERVATION_WIDTHS):
            raise ValueError("impact speed observation term widths must match the 40+1+1 schema")
        for name, func in zip(_TAIL, (impact_speed_goal, impact_speed_pending), strict=True):
            cfg = env.observation_manager.get_term_cfg(group, name)
            if (
                cfg.func is not func or cfg.params or cfg.noise is not None
                or cfg.clip is not None or cfg.scale is not None
                or cfg.history_length != 0 or cfg.delay_min_lag != 0
                or cfg.delay_max_lag != 0 or cfg.delay_hold_prob != 0
                or cfg.delay_update_period != 0
            ):
                raise ValueError("impact speed observations must preserve raw goal/pending semantics")
    return {
        **metadata,
        "observation_term_widths": list(_OBSERVATION_WIDTHS),
        "observation_normalization": {
            "type": "physical40_empirical_fixed_tail_v1",
            "empirical_width": 40,
            "fixed_tail": list(_TAIL),
            "empirical_eps": 0.01,
        },
    }


def build_impact_speed_metadata(env, raw_policy_clip: float) -> dict:
    """Validate live experimental semantics and freeze controller provenance."""
    from src.tasks.hammer.config.z1.joint_position_contract import load_joint_position_contract
    from src.tasks.hammer.rl.runner import (
        _JOINT_POSITION_CONTRACT_PATH,
        _joint_position_metadata,
        _vic_controller_metadata,
        _with_vic_metadata,
    )

    if tuple(env.action_manager.active_terms) != IMPACT_SPEED_ACTION_SIGNATURE:
        raise ValueError("impact speed requires the exact ordered action signature")
    if list(env.action_manager.action_term_dim) != [6, 6, 1] or env.action_manager.total_action_dim != 13:
        raise ValueError("impact speed requires action dimensions [6, 6, 1]")
    speed = env.action_manager.get_term("strike_speed")
    if not isinstance(speed, ImpactSpeedAction):
        raise ValueError("impact speed requires its typed speed action")
    # Reuse the strict scalar domain checks and catch stale cfg versus live state.
    # Use policy mode for this validation-only state: external mode would sample
    # a goal and advance the training RNG merely by saving/checking metadata.
    checked = ImpactSpeedState(1, "cpu", speed.cfg.bounds, "policy")
    if speed.cfg.mode not in ("external", "policy"):
        raise ValueError("impact speed mode must be external or policy")
    if checked.bounds != speed.state.bounds or speed.cfg.mode != speed.state.mode:
        raise ValueError("impact speed bounds/mode differ between config and live state")
    tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR, None)
    if not isinstance(tracker, FirstStrikeEventTracker):
        raise ValueError("impact speed requires the first-strike onset-speed tracker")
    axis = tracker._axis.detach().cpu().tolist()
    if axis != [0.0, 0.0, -1.0]:
        raise ValueError("impact speed axis must be the world-frame nail approach axis [0, 0, -1]")
    cost = env.reward_manager.get_term_cfg("speed_goal")
    validate_speed_goal_weight(cost.weight)
    if type(cost.func) is not ImpactSpeedCost or cost.params:
        raise ValueError("impact speed cost must use the exact event-unit implementation")

    metadata = _joint_position_metadata(
        env, "", raw_policy_clip=raw_policy_clip, observation_metadata=_observation_metadata(env)
    )
    controller = _vic_controller_metadata(
        env, env.scene["robot"], env.action_manager.get_term("joint_position"),
        load_joint_position_contract(_JOINT_POSITION_CONTRACT_PATH), action_term_dims=(6, 6, 1),
    )
    metadata = _with_vic_metadata(env, metadata, controller)
    metadata.pop("run_path")
    # Freeze the safety knobs rather than runtime EMA state. Preserve configured
    # doses/limits; constructing metadata never changes the treatment.
    safety = {}
    for name in ("cat_soft", "substep_impulse"):
        cfg = env.cfg.metrics[name]
        safety[name] = {
            key: value for key, value in cfg.params.items() if key != "robot_cfg"
        }
    metadata.update({
        "schema_version": _SCHEMA,
        "distribution_parameters": ["mean", "std", "activity_mask"],
        "speed_bounds_mps": list(checked.bounds),
        "speed_axis_world": axis,
        "speed_mode": speed.cfg.mode,
        "speed_measurement": "first_accepted_contact_onset_previous_free_flight_difference_mps",
        "speed_cost": {
            "coefficient": -float(cost.weight),
            "units": "dimensionless_per_event",
            "reward_term_units": "cost_per_second_before_manager_dt",
            "timeout_semantics": "reset_boundary_shaping",
        },
        "safety": safety,
    })
    # Make PT and ONNX metadata use the same JSON-safe list/scalar vocabulary.
    return json.loads(json.dumps(metadata, allow_nan=False))


def validate_impact_speed_models(actor, critic) -> None:
    """Reject a plain 42-feature empirical normalizer before runner use."""
    for group, model in (("actor", actor), ("critic", critic)):
        if not isinstance(model, ImpactSpeedModel):
            raise ValueError("impact speed requires ImpactSpeedModel for actor and critic")
        normalizer = model.obs_normalizer
        if (
            not isinstance(normalizer, ImpactSpeedNormalization)
            or normalizer.physical.mean.shape != (40,)
            or normalizer.physical.eps != 0.01
            or tuple(model.obs_groups) != (group,)
        ):
            raise ValueError("impact speed requires fixed-tail observation normalization and matching groups")


def validate_impact_speed_metadata(
    saved: dict, expected: dict, *, allow_phase_transfer: bool = False
) -> None:
    """Normal resume is exact; only an explicit external→policy path may differ."""
    if not isinstance(saved, dict) or saved.get("schema_version") != _SCHEMA:
        raise ValueError("impact speed checkpoint metadata is missing or has the wrong schema")
    if not isinstance(expected, dict) or expected.get("schema_version") != _SCHEMA:
        raise ValueError("expected impact speed metadata has the wrong schema")
    candidate = dict(saved)
    if allow_phase_transfer and candidate.get("speed_mode") == "external" and expected.get("speed_mode") == "policy":
        candidate["speed_mode"] = "policy"
    # JSON comparison also distinguishes true from 1 and rejects nonfinite data.
    if json.dumps(candidate, allow_nan=False, sort_keys=True) != json.dumps(expected, allow_nan=False, sort_keys=True):
        different = sorted(key for key in set(candidate) | set(expected) if candidate.get(key) != expected.get(key))
        raise ValueError(f"impact speed checkpoint metadata mismatch: {different}")


def summarize_impact_speed_rollout(storage, bounds: tuple[float, float]) -> dict | None:
    """Summarize gains and only genuinely selected (saved-mask) speed actions."""
    from src.tasks.hammer.rl.runner import _summarize_gain_rollout

    checked = ImpactSpeedState(1, "cpu", bounds, "policy")
    if isinstance(storage.step, bool) or storage.step != 0:
        raise ValueError("impact speed telemetry requires post-update storage cursor 0")
    telemetry = _summarize_gain_rollout(
        storage, action_terms=IMPACT_SPEED_ACTION_SIGNATURE, raw_action_clip=1.0,
        action_dim=13, distribution_param_count=3, schema_version=_SCHEMA,
    )
    if telemetry is None:
        return None
    masks = storage.distribution_params[2]
    if (
        not isinstance(masks, torch.Tensor) or masks.shape != storage.actions.shape
        or not torch.all((masks == 0) | (masks == 1))
        or not torch.all(masks[..., :12] == 1)
    ):
        raise ValueError("impact speed telemetry requires aligned binary masks with all motor actions active")
    selected = masks[..., 12].bool()
    sampled = storage.actions[..., 12][selected].detach().to(device="cpu", dtype=torch.float64).clamp(-1, 1)
    lo, hi = checked.bounds
    physical = lo + (sampled + 1) * 0.5 * (hi - lo)
    count = int(selected.sum())
    summary = dict.fromkeys(("mean", "std", "minimum", "p05", "median", "p95", "maximum"))
    if count:
        quantiles = torch.quantile(physical, physical.new_tensor([0.05, 0.5, 0.95]))
        summary.update({
            "mean": float(physical.mean()), "std": float(physical.std(correction=0)),
            "minimum": float(physical.min()), "p05": float(quantiles[0]),
            "median": float(quantiles[1]), "p95": float(quantiles[2]), "maximum": float(physical.max()),
        })
    telemetry.update({
        "speed_action_index": 12,
        "speed_bounds_mps": list(checked.bounds),
        "selection_count": count,
        "selected_speed_mps": summary,
        "selection_source": "saved_distribution_activity_mask",
    })
    json.dumps(telemetry, allow_nan=False)
    return telemetry
