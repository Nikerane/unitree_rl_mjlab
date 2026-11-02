"""Versioned contracts for the experimental 13-action impact-speed policy.

The exported actor emits raw deterministic actions. Its consumer still owns the
episode goal latch, reset, clipping, and goal/pending observation feedback.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass

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
from src.tasks.hammer.rl.impact_speed_policy import (
    ImpactSpeedModel, ImpactSpeedNormalization, ImpactSpeedAgeModel, ImpactSpeedLowNoiseModel,
)
from src.tasks.hammer.rl.impact_speed_wrapper import IMPACT_SPEED_ACTION_SIGNATURE


_SCHEMA = "impact_speed_v1"
_TAIL = ("impact_speed_goal", "impact_speed_pending")
_OBSERVATION_WIDTHS = (7, 7, 3, 3, 3, 3, 3, 1, 1, 3, 6, 1, 1)


def _spline_reward_profile(env) -> dict:
    """Freeze live reward functions, weights/parameters and native end semantics."""
    def identity(func):
        target = func if hasattr(func, "__qualname__") else type(func)
        return f"{target.__module__}.{target.__qualname__}"

    def serial(value):
        if is_dataclass(value):
            return serial(asdict(value))
        if isinstance(value, dict):
            return {key: serial(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serial(item) for item in value]
        if isinstance(value, slice):
            return {"slice": [value.start, value.stop, value.step]}
        return value

    terms = {}
    for name in env.reward_manager.active_terms:
        cfg = env.reward_manager.get_term_cfg(name)
        terms[name] = {"impl": identity(cfg.func), "weight": float(cfg.weight),
                       "params": serial(cfg.params)}
    termination = env.termination_manager.get_term_cfg("nail_driven")
    success_semantics = "accepted_strike_50ms_window_complete_not_nail_depth"
    if termination.params.get("window_s") == .2:
        success_semantics = "accepted_strike_200ms_window_complete_not_nail_depth"
    profile = {"name": "bounded_spline_speed_v1", "terms": terms,
            "progress_units": "dimensionless_per_episode_total_at_most_one",
            "accepted_strike_units": "dimensionless_once_per_accepted_first_contact",
            "path_cost_units": "dimensionless_per_second_before_manager_dt",
            "completion_semantics": "accepted_downward_first_contact_within_10mm_of_frozen_nail_and_spline",
            "termination": {"impl": identity(termination.func), "params": serial(termination.params),
                            "success_semantics": success_semantics}}
    from src.tasks.hammer.mdp.spline_boundary_credit import BoundarySplineStrikeReward
    credit = env.reward_manager.get_term_cfg("accepted_spline_strike")
    if isinstance(credit.func, BoundarySplineStrikeReward):
        fraction = credit.func.onset_fraction
        if (type(credit.func) is not BoundarySplineStrikeReward or credit.weight != 1.
                or fraction not in (0., .2) or credit.params.get("onset_fraction") != fraction):
            raise ValueError("boundary credit must preserve its constructed unit and onset_fraction")
        profile["credit_timing"] = {
            "schema_version": "spline_boundary_credit_v1",
            "onset_fraction": fraction,
            "boundary_fraction": 1. - fraction,
            "raw_accepted_event_units": 1.,
            "eligibility": "original_first_onset_consumed_including_rejected_contact",
            "boundary": "native_reset_buf_after_final_physics_before_reset",
            "eligible_failure_and_timeout": "pay_remaining_balance_once",
            "timeout_bootstrap": "unchanged",
            "evaluator_cutoff": "no_payment",
            "continuing_contact_required": False,
            "reset": "clear_eligibility_and_unpaid_credit_per_world",
            "observability": "unchanged_N_partial_observability",
        }
    from src.tasks.hammer.rl.spline_credit_critic_contract import credit_critic_metadata
    experiment = credit_critic_metadata(env)
    if experiment is not None:
        profile["credit_timing"] = {
            "schema_version": "spline_credit_deadline_v1",
            "credit_deadline_ms": experiment["credit_deadline_ms"],
            "boundary": "first_policy_boundary_at_deadline_or_earlier_native_reset",
            "eligibility": "original_first_onset_consumed_including_rejected_contact",
            "raw_accepted_event_units": 1.,
            "eligible_failure_and_timeout": "pay_remaining_balance_once",
            "external_reset": "clear_without_payment",
            "late_safety_reversal": False,
            "timeout_bootstrap": "unchanged",
            "observability": experiment["critic_mode"],
        }
    from src.tasks.hammer.rl.spline_split_credit_contract import split_metadata, credit_timing
    from src.tasks.hammer.rl.spline_split_event036_contract import split_event_metadata
    from src.tasks.hammer.rl.spline_reward037_contract import reward_metadata
    split_event = reward_metadata(env)
    if split_event is None:
        split_event = split_event_metadata(env)
    if split_event is not None:
        profile["credit_timing"] = split_event["credit_timing"]
    else:
        split = split_metadata(env)
        if split is not None:
            profile["credit_timing"] = credit_timing(split["early_fraction"])
    return profile


def _sampled_vic_treatment_metadata(env) -> dict:
    """Freeze variable-controller plant and the path that supplies its policy inputs."""
    def executed_field(name: str) -> list:
        # The host MjModel is a template. The Warp bridge exposes the arrays
        # that simulation steps read, including expanded per-world fields.
        values = getattr(env.sim.model, name).detach().cpu()
        if values.ndim < 2 or values.shape[0] not in (1, env.num_envs):
            raise ValueError(f"sampled VIC runtime load {name} has unexpected world shape")
        if not torch.isfinite(values).all():
            raise ValueError(f"sampled VIC runtime load {name} must be finite")
        if values.shape[0] > 1 and not torch.equal(values, values[:1].expand_as(values)):
            raise ValueError(f"sampled VIC runtime load {name} is heterogeneous across worlds; unsupported checkpoint identity")
        return values[0].tolist()

    reward = env.reward_manager.get_term_cfg("r_imit")
    reference_keys = ("path_shape", "overshoot", "followthrough_mode", "horizontal_detour_m")
    reference = {key: reward.params.get(key, 0.0 if key == "horizontal_detour_m" else None)
                 for key in reference_keys}
    if (not math.isfinite(reward.weight) or reward.weight < 0
        or env.cfg.rewards["r_imit"].weight != reward.weight or reference != {
        "path_shape": "quintic", "overshoot": .02,
        "followthrough_mode": "vertical", "horizontal_detour_m": 0.0,
    } or "r_imit_anneal" in env.cfg.curriculum):
        raise ValueError("sampled VIC guidance must preserve the persistent quintic treatment")
    for group in ("actor", "critic"):
        for name in ("strike_phase", "strike_ref_error"):
            params = env.observation_manager.get_term_cfg(group, name).params
            if any(params.get(key, 0.0 if key == "horizontal_detour_m" else None) != value
                   for key, value in reference.items()):
                raise ValueError("sampled VIC observation path differs from guidance")
    sampler = env.cfg.events["sample_strike_route_signs"].params
    if any(sampler.get(key, 0.0 if key == "horizontal_detour_m" else None) != value
           for key, value in reference.items()):
        raise ValueError("sampled VIC reset path differs from guidance")
    fixed_route_sign = sampler.get("fixed_route_sign")
    if fixed_route_sign not in (-1, None):
        raise ValueError("sampled VIC route must be left-only or three-start sampled")
    width = reward.params.get("sigma", .05)
    if (not math.isfinite(width) or width <= 0
        or env.cfg.rewards["r_imit"].params.get("sigma", .05) != width):
        raise ValueError("sampled VIC guidance width must be finite and positive")
    route_positions = env.cfg.events["reset_strike_route_joints"].params["route_joint_positions"]
    return {
        "reference": {
            "path_shape": reference["path_shape"], "overshoot_m": reference["overshoot"],
            "followthrough_mode": reference["followthrough_mode"],
            "horizontal_detour_m": reference["horizontal_detour_m"],
            "fixed_route_sign": fixed_route_sign,
            "route_joint_positions_rad": route_positions,
        },
        "guidance": {"weight": float(reward.weight), "width_m": float(width),
                     "impl": f"{type(reward.func).__module__}.{type(reward.func).__qualname__}"},
        "load": {
            "body_mass_kg": executed_field("body_mass"),
            "body_inertia_kg_m2": executed_field("body_inertia"),
            "body_gravcomp": executed_field("body_gravcomp"),
            "dof_frictionloss_nm": executed_field("dof_frictionloss"),
            "dof_damping": executed_field("dof_damping"),
            "jnt_stiffness": executed_field("jnt_stiffness"),
        },
    }


def _observation_metadata(env) -> dict:
    # Shared validation is imported lazily to keep the legacy runner's public
    # dispatch small without introducing a second copy of its controller rules.
    from src.tasks.hammer.rl.runner import _DIRECT_JOINT_OBSERVATIONS, _joint_observation_metadata

    from src.tasks.hammer.mdp.spline_learning_variants import AGE_TAIL, learning_variant, startup_age

    from src.tasks.hammer.mdp.spline_exploration_pilot import pilot_setting, pilot_startup_age
    pilot = pilot_setting(env.cfg)
    variant = learning_variant(env.cfg)
    has_age = variant is not None or pilot is not None
    tail = AGE_TAIL if has_age else _TAIL
    widths_expected = (*_OBSERVATION_WIDTHS[:-2], 1, 1, 1) if has_age else _OBSERVATION_WIDTHS
    age_func = pilot_startup_age if pilot is not None else startup_age
    functions = (age_func, impact_speed_goal, impact_speed_pending) if has_age else (
        impact_speed_goal, impact_speed_pending)
    from src.tasks.hammer.rl.spline_credit_critic_contract import credit_critic_metadata
    experiment = credit_critic_metadata(env)
    if experiment is None:
        from src.tasks.hammer.rl.spline_reward037_contract import reward_metadata
        experiment = reward_metadata(env)
        if experiment is None:
            from src.tasks.hammer.rl.spline_split_event036_contract import split_event_metadata
            experiment = split_event_metadata(env)
    extra = tuple(experiment["critic_event_fields"]) if experiment is not None else ()
    metadata = _joint_observation_metadata(env, scalar_tail=tail, critic_extra_tail=extra)
    if tuple(metadata["observation_names"]) != (*_DIRECT_JOINT_OBSERVATIONS, *tail):
        raise ValueError("impact speed requires the direct-reference observation order")
    for group in ("actor", "critic"):
        widths = env.observation_manager.group_obs_term_dim[group]
        group_widths = (*widths_expected, *((1,) * len(extra))) if group == "critic" else widths_expected
        if tuple(widths) != tuple((width,) for width in group_widths):
            raise ValueError("impact speed observation term widths must match the declared fixed-tail schema")
        for name, func in zip(tail, functions, strict=True):
            cfg = env.observation_manager.get_term_cfg(group, name)
            params = ({"setting": pilot} if pilot is not None else {"variant": variant}) if name == "startup_age" else {}
            if (
                cfg.func is not func or cfg.params != params or cfg.noise is not None
                or cfg.clip is not None or cfg.scale is not None
                or cfg.history_length != 0 or cfg.delay_min_lag != 0
                or cfg.delay_max_lag != 0 or cfg.delay_hold_prob != 0
                or cfg.delay_update_period != 0
            ):
                raise ValueError("impact speed observations must preserve raw fixed-tail semantics")
    return {
        **metadata,
        "observation_term_widths": list(widths_expected),
        **({"critic_observation_term_widths": [*widths_expected, *([1] * len(extra))]} if extra else {}),
        "observation_normalization": {
            "type": "physical40_empirical_fixed_tail_v1",
            "empirical_width": 40,
            "fixed_tail": list(tail),
            **({"critic_fixed_tail": [*tail, *extra]} if extra else {}),
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
    sampled = getattr(env, "_sampled_pd500", None)
    if sampled is not None:
        metadata.pop("variable_impedance")
        metadata["action_type"] = (
            "joint_position_variable_impedance_sampled_pd"
            if sampled.gain_mode == "variable" else "joint_position_fixed_sampled_pd"
        )
        metadata["sampled_pd"] = sampled.metadata()
        if sampled.gain_mode == "variable":
            metadata["controller_identity"] = "sampled_vic500_v1"
            metadata["sampled_vic_treatment"] = _sampled_vic_treatment_metadata(env)
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
    if "spline_path_cost" in env.reward_manager.active_terms:
        metadata["reward_profile"] = _spline_reward_profile(env)
        from src.tasks.hammer.mdp.spline_speed_variants import variant_metadata
        variant = variant_metadata(env)
        if variant is not None:
            metadata["spline_variant"] = variant
    from src.tasks.hammer.mdp.spline_learning_variants import learning_metadata
    learning = learning_metadata(env)
    if learning is not None:
        metadata["spline_learning_variant"] = learning
    from src.tasks.hammer.rl.spline_credit_critic_contract import credit_critic_metadata
    experiment = credit_critic_metadata(env)
    if experiment is not None:
        metadata["credit_critic_experiment"] = experiment
    from src.tasks.hammer.rl.spline_reward037_contract import reward_metadata
    reward037 = reward_metadata(env)
    if reward037 is not None:
        metadata["spline_reward037"] = reward037
        metadata["split_physical_treatment"] = _sampled_vic_treatment_metadata(env)
        return json.loads(json.dumps(metadata, allow_nan=False))
    from src.tasks.hammer.rl.spline_split_event036_contract import split_event_metadata
    split_event = split_event_metadata(env)
    if split_event is not None:
        metadata["spline_split_event036"] = split_event
        metadata["split_physical_treatment"] = _sampled_vic_treatment_metadata(env)
        return json.loads(json.dumps(metadata, allow_nan=False))
    from src.tasks.hammer.rl.spline_split_credit_contract import split_metadata
    split = split_metadata(env)
    from src.tasks.hammer.rl.reward_exploration032_contract import metadata032
    combined = metadata032(env)
    if combined is not None:
        metadata["spline_reward_exploration032"] = combined
        metadata["split_physical_treatment"] = _sampled_vic_treatment_metadata(env)
        return json.loads(json.dumps(metadata, allow_nan=False))
    from src.tasks.hammer.rl.depth_reward031_contract import depth_metadata031
    depth = depth_metadata031(env)
    if split is not None:
        metadata["spline_split_credit"] = split
        metadata["split_physical_treatment"] = _sampled_vic_treatment_metadata(env)
        if depth is not None:
            metadata["spline_depth_reward031"] = depth
        return json.loads(json.dumps(metadata, allow_nan=False))
    from src.tasks.hammer.rl.spline_exploration_pilot_contract import pilot_metadata
    pilot = pilot_metadata(env)
    if pilot is not None:
        metadata["spline_exploration_pilot"] = pilot
        metadata["pilot_physical_treatment"] = _sampled_vic_treatment_metadata(env)
    # Make PT and ONNX metadata use the same JSON-safe list/scalar vocabulary.
    return json.loads(json.dumps(metadata, allow_nan=False))


def validate_impact_speed_models(actor, critic, expected_metadata: dict | None = None) -> None:
    """Require exact observation width, model treatment and fixed-tail normalization."""
    if "spline_reward037" in (expected_metadata or {}):
        from src.tasks.hammer.rl.spline_reward037_contract import validate_reward_models
        validate_reward_models(actor, critic, expected_metadata)
        return
    if "spline_split_event036" in (expected_metadata or {}):
        from src.tasks.hammer.rl.spline_split_event036_contract import validate_split_event_models
        validate_split_event_models(actor, critic, expected_metadata)
        return
    if "spline_reward_exploration032" in (expected_metadata or {}):
        from src.tasks.hammer.rl.reward_exploration032_contract import validate_models
        validate_models(actor, critic, expected_metadata)
        return
    if "spline_depth_reward031" in (expected_metadata or {}):
        from src.tasks.hammer.rl.depth_reward031_contract import validate_depth_models031
        validate_depth_models031(actor, critic, expected_metadata)
        return
    if "spline_split_credit" in (expected_metadata or {}):
        from src.tasks.hammer.rl.spline_split_credit_contract import validate_split_models
        validate_split_models(actor, critic, expected_metadata)
        return
    if (expected_metadata or {}).get("spline_exploration_pilot") is not None:
        from src.tasks.hammer.rl.spline_exploration_pilot_contract import validate_pilot_models
        validate_pilot_models(actor, critic, expected_metadata)
        return
    learning = (expected_metadata or {}).get("spline_learning_variant")
    variant = learning["variant"] if learning is not None else None
    classes = (ImpactSpeedModel, ImpactSpeedModel)
    if variant is not None:
        classes = (ImpactSpeedLowNoiseModel if variant == "N" else ImpactSpeedAgeModel, ImpactSpeedAgeModel)
    from src.tasks.hammer.rl.spline_credit_critic_contract import validate_experiment_identity
    from src.tasks.hammer.rl.spline_credit_critic import CreditEventCritic
    experiment = (expected_metadata or {}).get("credit_critic_experiment")
    if experiment is not None:
        experiment = validate_experiment_identity(experiment)
        if variant != "N":
            raise ValueError("credit critic screen requires unchangedN actor treatment")
        if experiment["critic_mode"] == "event":
            classes = (ImpactSpeedLowNoiseModel, CreditEventCritic)
    for group, model in (("actor", actor), ("critic", critic)):
        expected_class = classes[0 if group == "actor" else 1]
        width = experiment[group + "_width"] if experiment is not None else (43 if variant is not None else 42)
        if type(model) is not expected_class or model.obs_dim != width:
            raise ValueError("impact speed requires the exact ImpactSpeedModel treatment for actor and critic")
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
    if "spline_reward037" in saved or "spline_reward037" in expected:
        from src.tasks.hammer.rl.spline_reward037_contract import validate_reward_metadata
        for value in (saved, expected):
            validate_reward_metadata(value)
    if "spline_depth_reward031" in saved or "spline_depth_reward031" in expected:
        from src.tasks.hammer.rl.depth_reward031_contract import validate_depth_identity031, same
        for value in (saved, expected):
            marker = validate_depth_identity031(value.get("spline_depth_reward031"))
            if not same(value.get("spline_split_credit"), marker["baseline_split"]):
                raise ValueError("depth031 checkpoint baseline split differs")
    if "spline_reward_exploration032" in saved or "spline_reward_exploration032" in expected:
        from src.tasks.hammer.rl.reward_exploration032_contract import validate_identity
        for value in (saved, expected):
            validate_identity(value.get("spline_reward_exploration032"))
    if "spline_split_event036" in saved or "spline_split_event036" in expected:
        from src.tasks.hammer.rl.spline_split_event036_contract import validate_split_event_identity
        for value in (saved, expected):
            validate_split_event_identity(value.get("spline_split_event036"))
    candidate = dict(saved)
    if allow_phase_transfer and candidate.get("speed_mode") == "external" and expected.get("speed_mode") == "policy":
        candidate["speed_mode"] = "policy"
    # JSON comparison also distinguishes true from 1 and rejects nonfinite data.
    if json.dumps(candidate, allow_nan=False, sort_keys=True) != json.dumps(expected, allow_nan=False, sort_keys=True):
        different = sorted(key for key in set(candidate) | set(expected) if candidate.get(key) != expected.get(key))
        raise ValueError(f"impact speed checkpoint metadata mismatch: {different}")


def summarize_impact_speed_rollout(storage, bounds: tuple[float, float], *, pilot_metadata=None) -> dict | None:
    """Summarize gains and only genuinely selected (saved-mask) speed actions."""
    from src.tasks.hammer.rl.runner import _summarize_gain_rollout

    pilot = None
    if pilot_metadata is not None:
        from src.tasks.hammer.rl.spline_exploration_pilot_contract import validate_pilot_identity
        pilot = validate_pilot_identity(pilot_metadata)
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
        or not torch.all(masks[..., :6] == 1)
        or not torch.all(masks[..., 6:12] == (pilot["likelihood_gain_mask"] if pilot is not None else 1))
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
    if pilot is not None:
        telemetry["requested_gain_actions"] = {name: telemetry.pop(name) for name in (
            "deterministic_gaussian_mean", "sampled_action", "gaussian_exploration_std")}
        telemetry["requested_gain_semantics"] = "POLICY_REQUESTS_NOT_APPLIED_GAIN_MEASUREMENTS"
        telemetry["spline_exploration_pilot"] = pilot
        if pilot["setting"] == "fixed_nominal":
            telemetry["applied_fixed_nominal_gains_kp"] = pilot["fixed_nominal_kp"]
            telemetry["applied_fixed_nominal_gains_kd"] = pilot["fixed_nominal_kd"]
    json.dumps(telemetry, allow_nan=False)
    return telemetry
