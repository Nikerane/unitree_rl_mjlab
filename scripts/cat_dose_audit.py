"""Frozen-trace attribution for an offline CaT velocity-dose counterfactual.

The replay records the running velocity normalizer after the environment's normal
CaT update. The 0.75 dose is then reconstructed from the same raw margins and
normalizers; no hook state, policy, dynamics, or live safety limit is modified.
"""

from __future__ import annotations

import math

import torch


SCHEMA_VERSION = "cat_dose_trace_v1"
WINNER_NONE = 0
WINNER_VELOCITY = 1
WINNER_LOAD = 2
WINNER_TIE = 3
WINNER_LABELS = {
    WINNER_NONE: "none",
    WINNER_VELOCITY: "velocity",
    WINNER_LOAD: "load",
    WINNER_TIE: "tie",
}
JOINT_CAPS_NMS = [.369, .246, .738, .369, .246, .0164]


def _float_tensor(value, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _probability_winner(velocity_per_joint: torch.Tensor,
                        load_per_joint: torch.Tensor) -> tuple[torch.Tensor, ...]:
    delta_velocity = velocity_per_joint.max(dim=-1).values
    delta_load = load_per_joint.max(dim=-1).values
    combined = torch.maximum(delta_velocity, delta_load)
    active = combined > 0.
    velocity_wins = active & (delta_velocity > delta_load)
    load_wins = active & (delta_load > delta_velocity)
    tie = active & (delta_velocity == delta_load)
    winner = torch.full_like(delta_velocity, WINNER_NONE, dtype=torch.long)
    winner = torch.where(velocity_wins, WINNER_VELOCITY, winner)
    winner = torch.where(load_wins, WINNER_LOAD, winner)
    winner = torch.where(tie, WINNER_TIE, winner)
    velocity_joint = velocity_per_joint.argmax(dim=-1)
    load_joint = load_per_joint.argmax(dim=-1)
    winner_joint = torch.full_like(velocity_joint, -1)
    winner_joint = torch.where(velocity_wins, velocity_joint, winner_joint)
    winner_joint = torch.where(load_wins, load_joint, winner_joint)
    return delta_velocity, delta_load, combined, winner, winner_joint


def counterfactual_components(
    raw_velocity_margin,
    velocity_normalizer,
    load_probability_per_joint,
    *,
    baseline_max_p: float,
    target_max_p: float,
    min_p: float,
    observed_velocity_per_joint=None,
    observed_combined_delta=None,
) -> dict[str, torch.Tensor]:
    """Reconstruct both velocity doses, then apply CaT's max across terms/joints.

    Inputs may have any leading dimensions, with joints on the final axis. The
    denominator may broadcast over leading dimensions but must resolve to the
    same shape as the raw margin. ``load_probability_per_joint`` is intentionally
    passed through unchanged, not rescaled with velocity.
    """
    margin = _float_tensor(raw_velocity_margin, "raw velocity margin")
    load = _float_tensor(load_probability_per_joint, "load probability")
    if margin.ndim < 1 or margin.shape != load.shape:
        raise ValueError("velocity margins and per-joint load probabilities must have matching joint shapes")
    try:
        denominator = torch.broadcast_to(_float_tensor(velocity_normalizer, "velocity normalizer"), margin.shape)
    except RuntimeError as exc:
        raise ValueError("velocity normalizer is not broadcastable to the raw margins") from exc
    if bool((denominator <= 0.).any()):
        raise ValueError("velocity normalizer must be strictly positive")
    if bool(((load < 0.) | (load > 1.)).any()):
        raise ValueError("load probabilities must lie in [0, 1]")
    params = (baseline_max_p, target_max_p, min_p)
    if any(not math.isfinite(float(value)) for value in params):
        raise ValueError("CaT probability settings must be finite")
    if (not 0. < baseline_max_p <= 1. or not baseline_max_p < target_max_p <= 1.
            or not 0. <= min_p <= baseline_max_p):
        raise ValueError("require 0 <= min_p <= baseline_max_p < target_max_p <= 1")

    normalized_margin = (margin / denominator).clamp(0., 1.)

    def velocity_dose(max_p):
        return torch.where(
            margin > 0., min_p + normalized_margin * (max_p - min_p), torch.zeros_like(margin)
        )

    velocity_baseline_reconstructed = velocity_dose(baseline_max_p)
    velocity_target = velocity_dose(target_max_p)
    velocity_baseline = velocity_baseline_reconstructed
    if observed_velocity_per_joint is not None:
        observed = _float_tensor(observed_velocity_per_joint, "observed velocity probability")
        if observed.shape != velocity_baseline_reconstructed.shape or not torch.allclose(
                observed, velocity_baseline_reconstructed, rtol=1e-5, atol=1e-6):
            raise ValueError("recorded velocity probabilities do not reconstruct from margin/normalizer")
        # The recorded component values and winner labels were produced together
        # on the simulator device. Use those authoritative values for the baseline
        # winner after checking the CPU reconstruction within tolerance; otherwise
        # a one-ULP CPU/GPU division difference can turn a recorded winner into a tie.
        velocity_baseline = observed
    baseline_velocity, baseline_load, baseline_delta, baseline_winner, baseline_joint = (
        _probability_winner(velocity_baseline, load)
    )
    target_velocity, target_load, target_delta, target_winner, target_joint = (
        _probability_winner(velocity_target, load)
    )
    if observed_combined_delta is not None:
        observed = _float_tensor(observed_combined_delta, "observed combined CaT delta")
        if observed.shape != baseline_delta.shape or not torch.allclose(
                observed, baseline_delta, rtol=1e-5, atol=1e-6):
            raise ValueError("recorded combined delta does not equal max(velocity, load)")
    return {
        "velocity_baseline_per_joint": velocity_baseline,
        "velocity_baseline_reconstructed_per_joint": velocity_baseline_reconstructed,
        "velocity_target_per_joint": velocity_target,
        "load_per_joint": load,
        "delta_velocity_baseline": baseline_velocity,
        "delta_velocity_target": target_velocity,
        "delta_load": baseline_load,
        "delta_baseline": baseline_delta,
        "delta_target": target_delta,
        "winner_baseline": baseline_winner,
        "winner_target": target_winner,
        "winner_joint_baseline": baseline_joint,
        "winner_joint_target": target_joint,
        "delta_load_target": target_load,
    }


def validate_baseline_contract(contract: dict, *, target_max_p: float) -> dict:
    """Fail closed unless the trace was produced with the expected live CaT contract."""
    try:
        cat = contract["safety"]["cat_soft"]
        impulse = contract["safety"]["substep_impulse"]
        caps = cat["imp_limit"]
        matches = (
            math.isclose(float(cat["max_p"]), .5, rel_tol=0., abs_tol=1e-12)
            and math.isclose(float(cat["min_p"]), 0., rel_tol=0., abs_tol=1e-12)
            and math.isclose(float(cat["tau"]), .95, rel_tol=0., abs_tol=1e-12)
            and math.isclose(float(cat["imp_max_p"]), .2, rel_tol=0., abs_tol=1e-12)
            and math.isclose(float(cat["limit"]), 3.1415, rel_tol=0., abs_tol=1e-12)
            and cat["use_vel"] is True and cat["use_impulse"] is True
            and cat["vel_detection"] == "substep"
            and all(math.isclose(float(left), right, rel_tol=0., abs_tol=1e-12)
                    for left, right in zip(caps, JOINT_CAPS_NMS))
            and list(caps) == list(JOINT_CAPS_NMS)
            and impulse["event_window_substeps"] == 200
        )
        if not matches:
            raise ValueError("live CaT or impulse-window settings differ from the frozen baseline contract")
        if not math.isfinite(float(target_max_p)) or not .5 < target_max_p <= 1.:
            raise ValueError("offline target max_p must be greater than .5 and no larger than 1")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid CaT dose audit contract: {exc}") from exc
    return {
        "baseline_velocity_max_p": .5,
        "offline_counterfactual_velocity_max_p": float(target_max_p),
        "min_p": 0., "tau": .95, "load_max_p": .2,
        "load_component_changed": False,
        "live_safety_limits_changed": False,
        "learning_updates": 0,
    }


class CatDoseTrace:
    """Compact first-episode control-step trace captured before native autoreset."""

    _MATRIX_FIELDS = {
        "velocity_margin": "cat_velocity_margin_per_joint",
        "velocity_per_joint": "delta_velocity_per_joint",
        "load_per_joint": "delta_impulse_per_joint",
    }
    _VECTOR_FIELDS = {
        "combined_delta": "cat_delta",
        "winner_constraint": "cat_winner_constraint",
        "winner_joint": "cat_winner_joint",
        "contact_started": "started",
    }

    def __init__(self, num_envs: int):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self._normalizers: list[torch.Tensor] = []
        self._worlds = [{
            "world": index, "route_sign": None, "complete": False,
            "contact_transition": None, "global_steps": [],
            **{name: [] for name in (*self._MATRIX_FIELDS, *self._VECTOR_FIELDS)},
        } for index in range(num_envs)]

    def record_step(self, physical: dict, *, done, first_episode_mask) -> None:
        active = torch.as_tensor(first_episode_mask, dtype=torch.bool).detach().cpu()
        terminal = torch.as_tensor(done, dtype=torch.bool).detach().cpu()
        if active.shape != (self.num_envs,) or terminal.shape != (self.num_envs,):
            raise ValueError("first-episode and done masks must contain one value per world")
        if not bool(active.any()):
            raise ValueError("cannot record a CaT trace step after all first episodes finish")
        matrices, vectors = {}, {}
        for output_name, input_name in self._MATRIX_FIELDS.items():
            value = _float_tensor(physical[input_name], input_name).detach().cpu()
            if value.ndim != 2 or value.shape[0] != self.num_envs:
                raise ValueError(f"{input_name} must have shape (num_envs, joints)")
            matrices[output_name] = value
        for output_name, input_name in self._VECTOR_FIELDS.items():
            value = torch.as_tensor(physical[input_name]).detach().cpu()
            if value.shape != (self.num_envs,):
                raise ValueError(f"{input_name} must contain one scalar per world")
            if output_name == "contact_started":
                value = value.bool()
            elif output_name in ("winner_constraint", "winner_joint"):
                value = value.long()
            else:
                value = _float_tensor(value, input_name)
            vectors[output_name] = value
        denominator = _float_tensor(
            physical["cat_velocity_normalizer_per_joint"], "velocity normalizer"
        ).detach().cpu()
        if denominator.shape != matrices["velocity_margin"].shape:
            raise ValueError("velocity normalizer must match the batch and joint dimensions")
        if not torch.allclose(denominator, denominator[:1].expand_as(denominator),
                              rtol=1e-6, atol=1e-8):
            raise ValueError("velocity normalizer must be one shared per-joint batch value")
        baseline_delta = torch.maximum(
            matrices["velocity_per_joint"].max(dim=-1).values,
            matrices["load_per_joint"].max(dim=-1).values,
        )
        if not torch.allclose(vectors["combined_delta"], baseline_delta, rtol=1e-5, atol=1e-6):
            raise ValueError("combined CaT delta is inconsistent with component telemetry")
        route = torch.as_tensor(physical["route_sign"]).detach().cpu()
        if route.shape != (self.num_envs,) or bool(~torch.isin(route, torch.tensor([-1, 0, 1])).all()):
            raise ValueError("route_sign must identify each world as -1, 0, or +1")

        global_step = len(self._normalizers)
        self._normalizers.append(denominator[0].clone())
        for world_index, world in enumerate(self._worlds):
            if not bool(active[world_index]):
                continue
            sign = int(route[world_index])
            if world["route_sign"] is None:
                world["route_sign"] = sign
            elif world["route_sign"] != sign:
                raise ValueError("route sign changed before the first episode completed")
            local_step = len(world["global_steps"])
            world["global_steps"].append(global_step)
            for name, value in matrices.items():
                world[name].append(value[world_index].clone())
            for name, value in vectors.items():
                world[name].append(value[world_index].clone())
            if world["contact_transition"] is None and bool(vectors["contact_started"][world_index]):
                world["contact_transition"] = local_step
            if bool(terminal[world_index]):
                world["complete"] = True

    def to_payload(self) -> dict:
        worlds = []
        for source in self._worlds:
            row = {key: value for key, value in source.items()
                   if key not in (*self._MATRIX_FIELDS, *self._VECTOR_FIELDS, "global_steps")}
            row["global_steps"] = torch.tensor(source["global_steps"], dtype=torch.long)
            for name in self._MATRIX_FIELDS:
                values = source[name]
                joints = len(values[0]) if values else 6
                row[name] = torch.stack(values) if values else torch.empty((0, joints), dtype=torch.float32)
            for name in self._VECTOR_FIELDS:
                values = source[name]
                dtype = torch.bool if name == "contact_started" else torch.long if name.startswith("winner") else torch.float32
                row[name] = torch.stack(values).to(dtype=dtype) if values else torch.empty((0,), dtype=dtype)
            worlds.append(row)
        normalizers = (torch.stack(self._normalizers) if self._normalizers
                       else torch.empty((0, 6), dtype=torch.float32))
        return {
            "schema_version": SCHEMA_VERSION,
            "normalizer_semantics": "actual post-update CaT velocity EMA shared across the fresh replay batch; persists across environment resets",
            "contact_transition_semantics": "first post-metrics control transition with the accepted first-strike tracker latched; its CaT value can include post-contact physics substeps",
            "delta_semantics": "per-joint velocity/load probabilities; aggregate is max across both components and all joints",
            "worlds": worlds,
            "velocity_normalizer_by_global_step": normalizers,
        }


def _metric(values: torch.Tensor) -> dict[str, float | None]:
    values = values.to(torch.float64)
    return {
        "mean": float(values.mean()) if values.numel() else None,
        "maximum": float(values.max()) if values.numel() else None,
    }


def _contact_weights(delta: torch.Tensor, contact_transition: int, *, label: str) -> dict:
    def log_weight(end_exclusive):
        log_weight = torch.log1p(-delta[:end_exclusive].to(torch.float64)).sum()
        return float(log_weight)

    before = log_weight(contact_transition)
    through = log_weight(contact_transition + 1)
    return {
        f"{label}_log_before_contact_interval": before,
        f"{label}_before_contact_interval": math.exp(before),
        f"{label}_log_through_contact_interval": through,
        f"{label}_through_contact_interval": math.exp(through),
    }


def summarize_cat_dose_trace(
    payload: dict, *, baseline_max_p: float, target_max_p: float,
) -> dict:
    """Validate and summarize a serialized fixed-policy component trace."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown or missing CaT trace schema")
    normalizers = _float_tensor(payload["velocity_normalizer_by_global_step"], "velocity normalizers")
    if normalizers.ndim != 2 or normalizers.shape[-1] != 6 or bool((normalizers <= 0.).any()):
        raise ValueError("velocity normalizers must be a finite positive [time, 6] tensor")
    episodes = []
    winner_counts = {"baseline": {label: 0 for label in WINNER_LABELS.values()},
                     "target": {label: 0 for label in WINNER_LABELS.values()}}
    all_velocity_baseline, all_velocity_target, all_load = [], [], []
    complete_count = contact_count = 0
    for world in payload["worlds"]:
        steps = int(world["combined_delta"].numel())
        if steps == 0:
            raise ValueError("every world must have at least one first-episode transition")
        global_steps = torch.as_tensor(world["global_steps"], dtype=torch.long)
        if global_steps.shape != (steps,) or bool((global_steps < 0).any()) or bool(
                (global_steps >= normalizers.shape[0]).any()):
            raise ValueError("world trace references an unavailable normalizer step")
        if any(world[name].shape != (steps, 6) for name in
               ("velocity_margin", "velocity_per_joint", "load_per_joint")):
            raise ValueError("world component traces must have shape [steps, 6]")
        baseline_delta = _float_tensor(world["combined_delta"], "combined delta")
        if baseline_delta.shape != (steps,) or world["winner_constraint"].shape != (steps,):
            raise ValueError("per-transition combined delta and winner must have shape [steps]")
        if world["winner_joint"].shape != (steps,):
            raise ValueError("per-transition winning joint must have shape [steps]")
        contact = world["contact_transition"]
        if contact is not None and (type(contact) is not int or not 0 <= contact < steps):
            raise ValueError("contact transition must be null or an observed transition index")
        result = counterfactual_components(
            world["velocity_margin"], normalizers[global_steps], world["load_per_joint"],
            baseline_max_p=baseline_max_p, target_max_p=target_max_p, min_p=0.,
            observed_velocity_per_joint=world["velocity_per_joint"],
            observed_combined_delta=baseline_delta,
        )
        if not torch.equal(result["winner_baseline"].cpu(), world["winner_constraint"].long().cpu()):
            raise ValueError("recorded baseline winning constraint differs from reconstructed components")
        if not torch.equal(result["winner_joint_baseline"].cpu(), world["winner_joint"].long().cpu()):
            raise ValueError("recorded baseline winning joint differs from reconstructed components")
        if bool(world["complete"]):
            complete_count += 1
        if contact is not None:
            contact_count += 1
            base_contact = _contact_weights(result["delta_baseline"], contact, label="baseline")
            target_contact = _contact_weights(result["delta_target"], contact, label="target")
            contact_summary = {
                **base_contact,
                **target_contact,
                "contact_transition": contact,
                "semantics": "CaT continuation weight on fixed replay observations; not physical survival or full return attenuation; through-contact includes post-contact physics substeps in the same control transition",
            }
        else:
            contact_summary = None
        per_joint = []
        for joint in range(6):
            per_joint.append({
                "joint_index": joint,
                "velocity_margin": _metric(world["velocity_margin"][:, joint]),
                "velocity_probability_baseline": _metric(result["velocity_baseline_per_joint"][:, joint]),
                "velocity_probability_target": _metric(result["velocity_target_per_joint"][:, joint]),
                "load_probability_unchanged": _metric(result["load_per_joint"][:, joint]),
            })
        for name, winners in (("baseline", result["winner_baseline"]),
                              ("target", result["winner_target"])):
            for code, label in WINNER_LABELS.items():
                winner_counts[name][label] += int((winners == code).sum())
        all_velocity_baseline.append(result["delta_velocity_baseline"])
        all_velocity_target.append(result["delta_velocity_target"])
        all_load.append(result["delta_load"])
        episodes.append({
            "world": int(world["world"]), "route_sign": int(world["route_sign"]),
            "steps": steps, "complete": bool(world["complete"]),
            "contact_transition": contact,
            "winner_counts_baseline": {label: int((result["winner_baseline"] == code).sum())
                                        for code, label in WINNER_LABELS.items()},
            "winner_counts_target": {label: int((result["winner_target"] == code).sum())
                                      for code, label in WINNER_LABELS.items()},
            "target_winner_changed_steps": int((result["winner_baseline"] != result["winner_target"]).sum()),
            "delta_baseline": _metric(result["delta_baseline"]),
            "delta_target": _metric(result["delta_target"]),
            "delta_velocity_baseline": _metric(result["delta_velocity_baseline"]),
            "delta_velocity_target": _metric(result["delta_velocity_target"]),
            "delta_load_unchanged": _metric(result["delta_load"]),
            "per_joint": per_joint,
            "contact_continuation_weight": contact_summary,
        })
    contact_log_ratios = []
    for episode in episodes:
        weights = episode["contact_continuation_weight"]
        if weights is None:
            continue
        baseline_log = weights["baseline_log_through_contact_interval"]
        target_log = weights["target_log_through_contact_interval"]
        contact_log_ratios.append(target_log - baseline_log)
    velocity_baseline = torch.cat(all_velocity_baseline)
    velocity_target = torch.cat(all_velocity_target)
    load = torch.cat(all_load)
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_velocity_max_p": float(baseline_max_p),
        "offline_counterfactual_velocity_max_p": float(target_max_p),
        "load_max_p_changed": False,
        "dynamics_or_policy_changed": False,
        "learning_updates": 0,
        "interpretation": "counterfactual CaT reweighting on unchanged frozen-policy observations under the fresh replay's batch-dependent normalizers; not a prediction of behavior or safety after training at the higher dose",
        "winner_counts_by_control_transition": winner_counts,
        "aggregate_component_probabilities": {
            "velocity_baseline": _metric(velocity_baseline),
            "velocity_target": _metric(velocity_target),
            "load_unchanged": _metric(load),
        },
        "mean_log_contact_through_interval_weight_ratio_target_vs_baseline": (
            sum(contact_log_ratios) / len(contact_log_ratios) if contact_log_ratios else None
        ),
        "coverage": {
            "worlds": len(episodes), "complete_episodes": complete_count,
            "incomplete_episodes": len(episodes) - complete_count,
            "contact_episodes": contact_count,
            "no_contact_episodes": len(episodes) - contact_count,
        },
        "episodes": episodes,
    }
