"""On-policy runner for the hammer-nail task."""

import json
import math
from pathlib import Path

import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import DifferentialIKAction, JointPositionAction
from mjlab.envs.mdp.observations import last_action
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner

from src.tasks.hammer.config.z1.joint_position_contract import (
    JOINT_NAMES,
    JointPositionContract,
    load_joint_position_contract,
)
from src.tasks.hammer.mdp.rewards import action_rate_penalty
from src.tasks.hammer.mdp.trackability import joint_trackability_cost
from src.tasks.hammer.mdp.variable_impedance import (
    VARIABLE_IMPEDANCE_MAPPING_FAMILY,
    VARIABLE_IMPEDANCE_P_BOUNDS,
    JointStiffnessAction,
    expand_variable_impedance_model_fields,
)


_JOINT_POSITION_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/z1/data/z1_joint_position_stage1.json"
)

_DIRECT_JOINT_OBSERVATIONS = (
    "joint_pos",
    "joint_vel",
    "ee_pos",
    "ee_vel",
    "head_pos",
    "head_vel",
    "nail_top_pos",
    "nail_depth",
    "strike_phase",
    "strike_ref_error",
    "actions",
)
_WAYPOINT_JOINT_OBSERVATIONS = (
    *_DIRECT_JOINT_OBSERVATIONS,
    "next_gate_vector",
    "completed_gate_fraction",
    "guideline_perpendicular_error",
    "waypoint_progress_state",
)
_WAYPOINT_OBSERVATION_TERMS = frozenset(
    _WAYPOINT_JOINT_OBSERVATIONS[len(_DIRECT_JOINT_OBSERVATIONS) :]
)
_GUIDANCE_REWARD_KEYS = frozenset(("r_imit", "r_gate", "r_waypoint_progress"))
_BANKED_ACTUATOR_FIELDS = (
    "target_names_expr",
    "transmission_type",
    "armature",
    "frictionloss",
    "viscous_damping",
    "delay_min_lag",
    "delay_max_lag",
    "delay_hold_prob",
    "delay_update_period",
    "delay_per_env_phase",
    "stiffness",
    "damping",
    "effort_limit",
)
_GUIDANCE_SCHEMAS = {
    _DIRECT_JOINT_OBSERVATIONS: {
        "width": 40,
        "guidance_type": "direct_reference",
        "reward_key": "r_imit",
        "reward_impl": "src.tasks.hammer.mdp.rewards.ImitationPriorTerm",
        "waypoint_tracker": False,
    },
    _WAYPOINT_JOINT_OBSERVATIONS: {
        "width": 47,
        "guidance_type": "waypoint_progress",
        "reward_key": "r_waypoint_progress",
        "reward_impl": (
            "src.tasks.hammer.mdp.guideline.ordered_waypoint_progress_reward"
        ),
        "waypoint_tracker": True,
    },
}


def _finite_row(value: torch.Tensor | float, *, name: str) -> list[float]:
    """Return the first resolved action row after checking the full live tensor."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 2 or value.shape[1] != len(JOINT_NAMES):
            raise ValueError(f"{name} must contain one value for each Z1 arm joint")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return [float(item) for item in value[0].detach().cpu().tolist()]
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return [float(value)] * len(JOINT_NAMES)


def _finite_vector(value: torch.Tensor, *, name: str) -> list[float]:
    """Serialize one immutable six-joint controller vector without rounding."""
    if value.ndim != 1 or value.shape[0] != len(JOINT_NAMES):
        raise ValueError(f"{name} must contain one value for each Z1 arm joint")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return [float(item) for item in value.detach().cpu().tolist()]


def _fixed_actuator_signature(env) -> list[dict[str, object]]:
    """Serialize the fixed plant's resolved actuator configuration safely."""
    actuators = env.cfg.scene.entities["robot"].articulation.actuators
    signature = []
    for actuator in actuators:
        values = {
            "stiffness": float(actuator.stiffness),
            "damping": float(actuator.damping),
            "effort_limit": float(actuator.effort_limit),
            "armature": float(actuator.armature),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("fixed actuator signature must contain only finite values")
        signature.append(
            {
                "type": type(actuator).__name__,
                "target_names_expr": list(actuator.target_names_expr),
                **values,
            }
        )
    return signature


def _callable_identity(value: object) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _normalized_actuator_field(name: str, value: object) -> object:
    if name == "target_names_expr":
        return tuple(value)
    if name == "transmission_type":
        return getattr(value, "value", value)
    return value


def _require_banked_vic_actuators(
    env, contract: JointPositionContract
) -> None:
    """Require the exact actuator rows from the immutable qualification artifact."""
    expected_rows = contract.source_task_config_projection["actuators"]["robot"]
    actuators = env.cfg.scene.entities["robot"].articulation.actuators
    if len(actuators) != len(expected_rows) or any(
        type(actuator) is not BuiltinPositionActuatorCfg for actuator in actuators
    ):
        raise ValueError("VIC metadata requires the banked actuator contract")

    expected = tuple(
        tuple(
            _normalized_actuator_field(name, row[name])
            for name in _BANKED_ACTUATOR_FIELDS
        )
        for row in expected_rows
    )
    actual = tuple(
        tuple(
            _normalized_actuator_field(name, getattr(actuator, name))
            for name in _BANKED_ACTUATOR_FIELDS
        )
        for actuator in actuators
    )
    if actual != expected:
        raise ValueError("VIC metadata requires the banked actuator contract")


def _ordered_banked_nominal_gains(
    contract: JointPositionContract,
) -> tuple[list[float], list[float]]:
    by_joint: dict[str, tuple[float, float]] = {}
    for row in contract.source_task_config_projection["actuators"]["robot"]:
        for joint_name in row["target_names_expr"]:
            if joint_name not in JOINT_NAMES:
                continue
            if joint_name in by_joint:
                raise ValueError("VIC banked actuator contract contains duplicate joints")
            by_joint[joint_name] = (float(row["stiffness"]), float(row["damping"]))
    if set(by_joint) != set(JOINT_NAMES):
        raise ValueError("VIC banked actuator contract is missing canonical joints")
    return (
        [by_joint[name][0] for name in JOINT_NAMES],
        [by_joint[name][1] for name in JOINT_NAMES],
    )


def _require_banked_vic_position_action(
    action: JointPositionAction, contract: JointPositionContract
) -> None:
    """Reject any VIC position mapping that differs from its qualified FIC seam."""
    expected_scale = action.scale.new_tensor(contract.scale_rad).expand_as(action.scale)
    expected_offset = action.offset.new_tensor(contract.default_joint_pos_rad).expand_as(
        action.offset
    )
    expected_clip = action._clip.new_tensor(contract.physical_clip_rad).expand_as(
        action._clip
    )
    if not (
        torch.equal(action.scale, expected_scale)
        and torch.equal(action.offset, expected_offset)
        and torch.equal(action._clip, expected_clip)
    ):
        raise ValueError("VIC metadata requires the banked joint-position contract")


def _joint_observation_metadata(env) -> dict[str, object]:
    """Resolve one of the two qualified joint-policy schemas from live managers."""
    observation_manager = env.observation_manager
    active_terms = observation_manager.active_terms
    if set(active_terms) != {"actor", "critic"}:
        raise ValueError("joint metadata requires exactly actor and critic observations")
    actor_terms = tuple(active_terms["actor"])
    critic_terms = tuple(active_terms["critic"])
    if actor_terms != critic_terms:
        raise ValueError("joint metadata requires identical actor and critic observation terms")

    group_obs_dim = observation_manager.group_obs_dim
    if set(group_obs_dim) != {"actor", "critic"}:
        raise ValueError("joint metadata requires exactly actor and critic observation widths")
    actor_dim = tuple(group_obs_dim["actor"])
    critic_dim = tuple(group_obs_dim["critic"])
    if actor_dim != critic_dim:
        raise ValueError("joint metadata requires identical actor and critic observation widths")
    if len(actor_dim) != 1:
        raise ValueError("joint metadata requires flat actor and critic observations")

    schema = _GUIDANCE_SCHEMAS.get(actor_terms)
    if schema is None:
        raise ValueError("unsupported joint observation schema")
    expected_width = schema["width"]
    if actor_dim != (expected_width,):
        raise ValueError(
            f"joint metadata requires observation width {expected_width} for "
            f"{schema['guidance_type']}"
        )

    reward_terms = set(env.reward_manager.active_terms)
    guidance_reward_key = schema["reward_key"]
    if reward_terms & _GUIDANCE_REWARD_KEYS != {guidance_reward_key}:
        raise ValueError(
            f"joint metadata guidance reward identity must be {guidance_reward_key}"
        )

    metrics_terms = set(env.metrics_manager.active_terms)
    has_waypoint_tracker = "waypoint_progress" in metrics_terms
    if has_waypoint_tracker is not schema["waypoint_tracker"]:
        expectation = "require" if schema["waypoint_tracker"] else "forbid"
        raise ValueError(
            f"joint metadata {expectation}s the waypoint tracker for "
            f"{schema['guidance_type']}"
        )
    if schema["guidance_type"] == "direct_reference" and (
        set(actor_terms) & _WAYPOINT_OBSERVATION_TERMS
    ):
        raise ValueError("direct-reference metadata forbids waypoint observations")

    reward_impl = _callable_identity(
        env.reward_manager.get_term_cfg(guidance_reward_key).func
    )
    if reward_impl != schema["reward_impl"]:
        raise ValueError(
            f"joint metadata guidance reward implementation must be {schema['reward_impl']}"
        )
    return {
        "observation_names": list(actor_terms),
        "observation_widths": {
            "actor": actor_dim[0],
            "critic": critic_dim[0],
        },
        "guidance_type": schema["guidance_type"],
        "guidance_reward_key": guidance_reward_key,
        "guidance_reward_impl": reward_impl,
    }


def _vic_controller_metadata(
    env,
    robot: Entity,
    position_action: JointPositionAction,
    contract: JointPositionContract,
) -> dict[str, object]:
    """Validate and serialize the immutable live VIC controller seams."""
    _require_banked_vic_actuators(env, contract)
    _require_banked_vic_position_action(position_action, contract)
    action_dims = list(env.action_manager.action_term_dim)
    if action_dims != [len(JOINT_NAMES), len(JOINT_NAMES)]:
        raise ValueError("VIC metadata requires action dimensions [6, 6]")
    if env.action_manager.total_action_dim != 2 * len(JOINT_NAMES):
        raise ValueError("VIC metadata requires exactly twelve action dimensions")

    stiffness = env.action_manager.get_term("joint_stiffness")
    if not isinstance(stiffness, JointStiffnessAction):
        raise ValueError("VIC metadata requires a typed joint stiffness action")
    telemetry = stiffness.telemetry
    if telemetry.mapping_family != VARIABLE_IMPEDANCE_MAPPING_FAMILY:
        raise ValueError("VIC metadata requires the approved gain mapping family")
    if telemetry.C != 1.25:
        raise ValueError("VIC metadata requires C=1.25")
    if telemetry.p_bounds != VARIABLE_IMPEDANCE_P_BOUNDS:
        raise ValueError("VIC metadata requires policy bounds [-1, 1]")
    if telemetry.joint_names != JOINT_NAMES:
        raise ValueError("VIC metadata requires the canonical Z1 stiffness order")

    control_id_by_joint: dict[str, int] = {}
    for actuator in robot.actuators:
        for joint_name, control_id in zip(
            actuator.target_names, actuator.global_ctrl_ids.tolist(), strict=True
        ):
            if joint_name in JOINT_NAMES:
                if joint_name in control_id_by_joint:
                    raise ValueError(
                        f"VIC metadata found duplicate actuator mapping for {joint_name}"
                    )
                control_id_by_joint[joint_name] = int(control_id)
    if set(control_id_by_joint) != set(JOINT_NAMES):
        raise ValueError("VIC metadata requires all canonical Z1 actuator mappings")
    expected_control_ids = [control_id_by_joint[name] for name in JOINT_NAMES]
    control_ids = telemetry.control_ids.detach().cpu().tolist()
    if control_ids != expected_control_ids:
        raise ValueError("VIC metadata control IDs must match the live Z1 actuators")

    nominal_kp = _finite_vector(telemetry.nominal_kp, name="nominal Kp")
    nominal_kd = _finite_vector(telemetry.nominal_kd, name="nominal Kd")
    expected_kp, expected_kd = _ordered_banked_nominal_gains(contract)
    if nominal_kp != expected_kp or nominal_kd != expected_kd:
        raise ValueError("VIC metadata requires the exact banked nominal Kp/Kd")

    native_model_fields = tuple(expand_variable_impedance_model_fields.model_fields)
    if native_model_fields != ("actuator_gainprm", "actuator_biasprm"):
        raise ValueError("VIC metadata requires the approved native model fields")
    if not set(native_model_fields).issubset(env.sim.expanded_fields):
        raise ValueError("VIC metadata requires expanded native model fields")

    for group_name in ("actor", "critic"):
        term_cfg = env.observation_manager.get_term_cfg(group_name, "actions")
        if term_cfg.func is not last_action or term_cfg.params != {
            "action_name": "joint_position"
        }:
            raise ValueError(
                "VIC action observation must be exact position-only last_action"
            )

    action_rate_cfg = env.reward_manager.get_term_cfg("action_rate")
    if (
        action_rate_cfg.func is not action_rate_penalty
        or action_rate_cfg.weight != -0.01
        or action_rate_cfg.params != {"action_name": "joint_position"}
    ):
        raise ValueError("VIC action-rate cost must be exact and position-only")

    if "r_tt" not in env.reward_manager.active_terms:
        raise ValueError("VIC metadata requires r_tt")
    r_tt_cfg = env.reward_manager.get_term_cfg("r_tt")
    if (
        r_tt_cfg.func is not joint_trackability_cost
        or r_tt_cfg.weight != -1.0
        or set(r_tt_cfg.params) != {"robot_cfg", "k_tt"}
        or r_tt_cfg.params["k_tt"] != 1.0
    ):
        raise ValueError("VIC r_tt must use the exact implementation, weight, and k_tt=1")
    r_tt_robot_cfg = r_tt_cfg.params["robot_cfg"]
    if (
        r_tt_robot_cfg.name != "robot"
        or tuple(r_tt_robot_cfg.joint_names or ()) != JOINT_NAMES
        or list(r_tt_robot_cfg.joint_ids) != position_action.target_ids.detach().cpu().tolist()
        or r_tt_robot_cfg.preserve_order is not True
    ):
        raise ValueError("VIC r_tt must use the canonical ordered Z1 robot selector")

    return {
        "action_observation_impl": _callable_identity(last_action),
        "action_observation_source": "joint_position",
        "action_rate_impl": _callable_identity(action_rate_penalty),
        "action_rate_source": "joint_position",
        "action_rate_weight": float(action_rate_cfg.weight),
        "r_tt_weight": float(r_tt_cfg.weight),
        "r_tt_impl": _callable_identity(r_tt_cfg.func),
        "variable_impedance": {
            "mapping_family": telemetry.mapping_family,
            "C": float(telemetry.C),
            "p_bounds": [float(value) for value in telemetry.p_bounds],
            "joint_names": list(telemetry.joint_names),
            "control_ids": control_ids,
            "nominal_kp": nominal_kp,
            "nominal_kd": nominal_kd,
            "native_model_fields": list(native_model_fields),
        },
    }


def _get_hammer_metadata(env, run_path: str, *, raw_policy_clip: float) -> dict:
    action_terms = tuple(env.action_manager.active_terms)
    qualified_signatures = (
        ("ik_hammer_head",),
        ("joint_position",),
        ("joint_position", "joint_stiffness"),
    )
    if action_terms not in qualified_signatures:
        if len(action_terms) == 1:
            raise ValueError(
                "unsupported action term for hammer ONNX metadata: "
                f"{action_terms[0]}"
            )
        raise ValueError(
            "hammer ONNX metadata requires exactly one supported action or the "
            "exact VIC action signature ('joint_position', 'joint_stiffness')"
        )
    is_vic = action_terms == ("joint_position", "joint_stiffness")
    action_term = action_terms[0]

    robot: Entity = env.scene["robot"]
    action = env.action_manager.get_term(action_term)
    if isinstance(action, DifferentialIKAction):
        if action_term != "ik_hammer_head":
            raise ValueError(
                f"unsupported action term for Cartesian metadata: {action_term}"
            )
        return {
            "run_path": run_path,
            "action_type": "ik_delta_pos",
            "frame_name": action.cfg.frame_name,
            "delta_pos_scale": action.cfg.delta_pos_scale,
            "joint_names": list(robot.joint_names),
            "observation_names": env.observation_manager.active_terms["actor"],
        }
    if not isinstance(action, JointPositionAction):
        raise ValueError(f"unsupported action type for hammer ONNX metadata: {type(action).__name__}")
    if action_term != "joint_position":
        raise ValueError(f"unsupported action term for joint metadata: {action_term}")
    if action.action_dim != len(JOINT_NAMES):
        raise ValueError("joint metadata must contain exactly six action dimensions")
    if tuple(action.target_names) != JOINT_NAMES:
        raise ValueError("joint metadata requires the canonical Z1 joint order")
    target_ids = action.target_ids.detach().cpu().tolist()
    expected_ids, expected_names = robot.find_joints(JOINT_NAMES)
    if target_ids != expected_ids or tuple(expected_names) != JOINT_NAMES:
        raise ValueError("joint metadata requires the canonical Z1 joint order")
    if tuple(action.cfg.actuator_names) != JOINT_NAMES:
        raise ValueError("joint metadata requires the canonical Z1 actuator order")
    if not action.cfg.use_default_offset:
        raise ValueError("joint metadata requires default-offset action semantics")
    physics_dt_s = env.cfg.sim.mujoco.timestep
    if (
        isinstance(physics_dt_s, bool)
        or not isinstance(physics_dt_s, (int, float))
        or not math.isfinite(float(physics_dt_s))
        or physics_dt_s != 0.002
    ):
        raise ValueError("joint metadata requires physics timestep 0.002 s")
    control_decimation = env.cfg.decimation
    if (
        isinstance(control_decimation, bool)
        or not isinstance(control_decimation, int)
        or control_decimation != 10
    ):
        raise ValueError("joint metadata requires control decimation 10")
    if (
        isinstance(raw_policy_clip, bool)
        or not isinstance(raw_policy_clip, (int, float))
        or not math.isfinite(float(raw_policy_clip))
        or raw_policy_clip != 1.0
    ):
        raise ValueError("joint metadata requires raw policy clip 1.0")
    scales = _finite_row(action.scale, name="joint action scale")
    offsets = _finite_row(action.offset, name="joint action default offsets")
    clips = action._clip
    if clips.ndim != 3 or clips.shape[1:] != (len(JOINT_NAMES), 2):
        raise ValueError("joint physical clips must contain six [min, max] pairs")
    if not torch.isfinite(clips).all() or not torch.all(clips[:, :, 0] < clips[:, :, 1]):
        raise ValueError("joint physical clips must be finite increasing intervals")
    contract = load_joint_position_contract(_JOINT_POSITION_CONTRACT_PATH)
    observation_metadata = _joint_observation_metadata(env)

    delivered_i_ref = env.reward_manager.get_term_cfg(
        "delivered_impulse"
    ).params["i_ref"]
    if (
        isinstance(delivered_i_ref, bool)
        or not isinstance(delivered_i_ref, (int, float))
        or not math.isfinite(float(delivered_i_ref))
        or float(delivered_i_ref) <= 0.0
    ):
        raise ValueError("delivered impulse i_ref must be finite and strictly positive")

    r_tt_enabled = "r_tt" in env.reward_manager.active_terms
    if r_tt_enabled:
        r_tt_k_tt = float(env.reward_manager.get_term_cfg("r_tt").params["k_tt"])
        if not math.isfinite(r_tt_k_tt):
            raise ValueError("r_tt k_tt must be finite")
    else:
        r_tt_k_tt = "not_applicable"
    metadata = {
        "run_path": run_path,
        "action_type": "joint_position",
        "action_term": action_term,
        "action_dim": action.action_dim,
        "target_names": list(action.target_names),
        "target_ids": target_ids,
        "actuator_names": list(action.cfg.actuator_names),
        "use_default_offset": action.cfg.use_default_offset,
        "default_offsets": offsets,
        "action_scale": scales,
        "physical_clips": clips[0].detach().cpu().tolist(),
        "raw_policy_clip": float(raw_policy_clip),
        "physics_dt_s": float(physics_dt_s),
        "control_decimation": control_decimation,
        "fixed_actuator_signature": _fixed_actuator_signature(env),
        "joint_action_qualification_payload_sha256": contract.payload_sha256,
        "delivered_impulse_i_ref_n_s": float(delivered_i_ref),
        **observation_metadata,
        "r_tt_enabled": r_tt_enabled,
        "r_tt_k_tt": r_tt_k_tt,
    }
    if not is_vic:
        return metadata

    controller_metadata = _vic_controller_metadata(env, robot, action, contract)
    nominal_actuator_signature = metadata.pop("fixed_actuator_signature")
    metadata.pop("action_term")
    metadata.update(
        {
            "action_type": "joint_position_variable_impedance",
            "action_terms": list(action_terms),
            "action_term_dims": list(env.action_manager.action_term_dim),
            "action_dim": env.action_manager.total_action_dim,
            "position_action_dim": action.action_dim,
            "nominal_actuator_signature": nominal_actuator_signature,
            **controller_metadata,
        }
    )
    return metadata


def _metadata_for_onnx(metadata: dict) -> dict:
    """Encode structured joint-policy fields without upstream CSV rounding."""
    if metadata.get("action_type") not in (
        "joint_position",
        "joint_position_variable_impedance",
    ):
        return metadata
    return {
        key: json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if isinstance(value, (list, dict))
        else value
        for key, value in metadata.items()
    }


def _vic_per_joint_summary(
    values: torch.Tensor, *, raw_action_clip: float
) -> dict[str, list[float]]:
    """Summarize one flattened population of six ordered gain coordinates."""
    quantiles = torch.quantile(
        values,
        values.new_tensor((0.05, 0.5, 0.95)),
        dim=0,
    )

    def _list(tensor: torch.Tensor) -> list[float]:
        return [float(item) for item in tensor.tolist()]

    return {
        "mean": _list(values.mean(dim=0)),
        "std": _list(values.std(dim=0, correction=0)),
        "minimum": _list(values.amin(dim=0)),
        "p05": _list(quantiles[0]),
        "median": _list(quantiles[1]),
        "p95": _list(quantiles[2]),
        "maximum": _list(values.amax(dim=0)),
        "lower_bound_occupancy": _list(
            (values <= -raw_action_clip).to(dtype=values.dtype).mean(dim=0)
        ),
        "upper_bound_occupancy": _list(
            (values >= raw_action_clip).to(dtype=values.dtype).mean(dim=0)
        ),
    }


def _summarize_vic_rollout_telemetry(
    storage,
    *,
    action_terms: tuple[str, ...],
    raw_action_clip: float,
) -> dict[str, object] | None:
    """Return JSON-safe gain telemetry from the latest qualified VIC rollout."""
    if action_terms != ("joint_position", "joint_stiffness"):
        raise ValueError("VIC rollout telemetry requires the exact ordered action pair")
    if (
        isinstance(raw_action_clip, bool)
        or not isinstance(raw_action_clip, (int, float))
        or not math.isfinite(float(raw_action_clip))
        or raw_action_clip != 1.0
    ):
        raise ValueError("VIC rollout telemetry requires raw action clip 1.0")

    distribution_params = storage.distribution_params
    if distribution_params is None:
        return None
    if not isinstance(distribution_params, tuple) or len(distribution_params) != 2:
        raise ValueError("VIC rollout telemetry requires Gaussian mean and std")

    actions = storage.actions
    means, exploration_std = distribution_params
    expected_shape = (
        storage.num_transitions_per_env,
        storage.num_envs,
        2 * len(JOINT_NAMES),
    )
    if (
        not isinstance(actions, torch.Tensor)
        or not isinstance(means, torch.Tensor)
        or not isinstance(exploration_std, torch.Tensor)
        or tuple(actions.shape) != expected_shape
        or tuple(means.shape) != expected_shape
        or tuple(exploration_std.shape) != expected_shape
        or expected_shape[0] <= 0
        or expected_shape[1] <= 0
    ):
        raise ValueError("VIC rollout telemetry requires aligned nonempty 12D storage")
    if not (
        torch.isfinite(actions).all()
        and torch.isfinite(means).all()
        and torch.isfinite(exploration_std).all()
    ):
        raise ValueError("VIC rollout telemetry requires finite storage")
    if not torch.all(exploration_std > 0.0):
        raise ValueError("VIC rollout telemetry requires positive Gaussian std")

    gain_slice = slice(len(JOINT_NAMES), 2 * len(JOINT_NAMES))

    def _gain_population(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., gain_slice].reshape(-1, len(JOINT_NAMES)).detach().to(
            device="cpu", dtype=torch.float64
        )

    sampled_raw = _gain_population(actions)
    mean_raw = _gain_population(means)
    std_raw = _gain_population(exploration_std)
    telemetry: dict[str, object] = {
        "schema_version": 1,
        "source": "cat_rollout_storage",
        "action_terms": list(action_terms),
        "joint_names": list(JOINT_NAMES),
        "gain_action_indices": list(range(len(JOINT_NAMES), 2 * len(JOINT_NAMES))),
        "raw_action_clip": float(raw_action_clip),
        "sample_count": int(sampled_raw.shape[0]),
        "deterministic_gaussian_mean": {
            "raw": _vic_per_joint_summary(
                mean_raw, raw_action_clip=float(raw_action_clip)
            ),
            "clipped": _vic_per_joint_summary(
                mean_raw.clamp(-raw_action_clip, raw_action_clip),
                raw_action_clip=float(raw_action_clip),
            ),
        },
        "sampled_action": {
            "raw": _vic_per_joint_summary(
                sampled_raw, raw_action_clip=float(raw_action_clip)
            ),
            "clipped": _vic_per_joint_summary(
                sampled_raw.clamp(-raw_action_clip, raw_action_clip),
                raw_action_clip=float(raw_action_clip),
            ),
        },
        "gaussian_exploration_std": [
            float(item) for item in std_raw.mean(dim=0).tolist()
        ],
    }
    json.dumps(telemetry, allow_nan=False)
    return telemetry


class HammerOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    action_terms = tuple(self.env.unwrapped.action_manager.active_terms)
    if action_terms == ("joint_position", "joint_stiffness"):
      telemetry = _summarize_vic_rollout_telemetry(
        self.alg.storage,
        action_terms=action_terms,
        raw_action_clip=self.env.clip_actions,
      )
      if telemetry is not None:
        infos = {} if infos is None else dict(infos)
        infos["vic_rollout_telemetry"] = telemetry
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    is_wandb = self.logger.logger_type == "wandb"
    if is_wandb:
      import wandb  # lazy: a tensorboard run (the Lightning default) must not require wandb installed
      run_name = wandb.run.name if wandb.run else "local"
    else:
      run_name = "local"
    metadata = _get_hammer_metadata(
      self.env.unwrapped, run_name, raw_policy_clip=self.env.clip_actions
    )
    metadata_for_onnx = _metadata_for_onnx(metadata)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      attach_metadata_to_onnx(str(onnx_path), metadata_for_onnx)
      if is_wandb and self.cfg.get("upload_model"):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
