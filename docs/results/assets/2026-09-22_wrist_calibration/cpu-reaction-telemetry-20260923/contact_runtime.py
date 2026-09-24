"""Shared experiment-only state capture and native substep observations."""
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp


def runtime_identity():
    import importlib.metadata
    import subprocess
    import scripts.evaluate_impact_speed as evaluator
    root = Path(evaluator.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if not commit.startswith("cc46d99"):
        raise RuntimeError("diagnostic requires source revision cc46d99")
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")) + [root / "scripts/evaluate_impact_speed.py"]:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"commit": commit, "source_sha256": digest.hexdigest(),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("mjlab", "mujoco", "mujoco_warp", "torch", "warp-lang")}}


def sync():
    wp.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def data_arrays(obj, prefix=""):
    """Enumerate every MJWarp Data array; unknown leaves fail closed."""
    if not is_dataclass(obj):
        raise RuntimeError(f"unsupported MJWarp state container: {prefix}")
    arrays, scalars = {}, {}
    for f in fields(obj):
        value, name = getattr(obj, f.name), prefix + f.name
        if isinstance(value, wp.array):
            arrays[name] = value
        elif is_dataclass(value):
            a, s = data_arrays(value, name + ".")
            arrays.update(a)
            scalars.update(s)
        elif value is None or isinstance(value, (bool, int, float, str)):
            scalars[name] = value
        else:
            raise RuntimeError(f"unsupported MJWarp state leaf {name}: {type(value)}")
    return arrays, scalars


def cpu_array(array):
    # Empty arrays have no state; wp.to_torch on empty CPU arrays is unsupported.
    return None if array.size == 0 else wp.to_torch(array).detach().cpu().clone()


def tensor_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        if value is None:
            digest.update(b"empty")
        else:
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_hash(env):
    """MJB identity normalized only for the requested replay timestep."""
    import mujoco
    model = env.sim.mj_model
    old = model.opt.timestep
    try:
        model.opt.timestep = .002
        data = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=data)
    finally:
        model.opt.timestep = old
    return hashlib.sha256(data.tobytes()).hexdigest()


def initial_state(env):
    sync()
    arrays, scalars = data_arrays(env.sim.wp_data)
    saved = {name: cpu_array(value) for name, value in arrays.items()}
    encoder_bias = env.scene["robot"].data.encoder_bias.detach().cpu().clone()
    expanded = {}
    for name in sorted(env.sim.expanded_fields):
        value = getattr(env.sim.wp_model, name)
        if not isinstance(value, wp.array):
            raise RuntimeError(f"unsupported expanded model field {name}")
        expanded[name] = cpu_array(value)
    return {"data": saved, "data_scalars": scalars, "expanded": expanded,
            "encoder_bias": encoder_bias,
            "model_hash": model_hash(env),
            "state_hash": tensor_hash({**saved, "entity.encoder_bias": encoder_bias}),
            "expanded_hash": tensor_hash(expanded),
            "world_ids": list(range(env.num_envs))}


def restore_initial(env, saved):
    if model_hash(env) != saved["model_hash"]:
        raise RuntimeError("compiled model identity mismatch (only timestep may differ)")
    arrays, scalars = data_arrays(env.sim.wp_data)
    if set(arrays) != set(saved["data"]) or scalars != saved["data_scalars"]:
        raise RuntimeError("MJWarp data inventory/capacity mismatch")
    if set(env.sim.expanded_fields) != set(saved["expanded"]):
        raise RuntimeError("expanded/randomized model inventory mismatch")
    if not torch.equal(env.scene["robot"].data.encoder_bias.detach().cpu(), saved["encoder_bias"]):
        raise RuntimeError("source/replay reset encoder bias differs")
    sync()
    for current, payload in ((arrays, saved["data"]),
                             ({k: getattr(env.sim.wp_model, k) for k in saved["expanded"]}, saved["expanded"])):
        for name, value in payload.items():
            dest = current[name]
            if value is None:
                if dest.size != 0:
                    raise RuntimeError(f"empty/nonempty state mismatch: {name}")
                continue
            target = wp.to_torch(dest)
            if target.shape != value.shape or target.dtype != value.dtype:
                raise RuntimeError(f"state shape/dtype mismatch: {name}")
            target.copy_(value.to(target.device))
    sync()
    restored = initial_state(env)
    for key in ("state_hash", "expanded_hash"):
        if restored[key] != saved[key]:
            raise RuntimeError(f"restored state mismatch: {key}")


def preflight(env):
    """Reject hidden physics changes and unavailable validity channels."""
    if env.num_envs != 128:
        raise RuntimeError("this bounded diagnostic requires the full 128-world cohort")
    if any(mode in env.event_manager.available_modes for mode in ("step", "interval")):
        raise RuntimeError("replay does not reproduce step/interval randomization events")
    tracker = env._hammer_first_strike
    if tracker._contact_sensor.cfg.reduce != "maxforce" or tracker._contact_sensor.cfg.num_slots < 1:
        raise RuntimeError("expected existing maxforce sensor for binary contact gating")
    if tracker._impulse_sensor.cfg.reduce != "netforce" or tracker._impulse_sensor.cfg.num_slots != 1:
        raise RuntimeError("delivered impulse requires existing one-slot netforce aggregate sensor")
    data = env.sim.wp_data
    for name in ("nacon", "naconmax", "nefc", "njmax", "qacc_warmstart", "act", "ctrl"):
        if not hasattr(data, name):
            raise RuntimeError(f"missing state/overflow API: {name}")
    # A wrong order silently mislabels whole-arm results.
    gains = env.action_manager.get_term("joint_stiffness")
    if list(gains.joint_names) != [f"joint{i}" for i in range(1, 7)]:
        raise RuntimeError("unexpected controlled joint order")
    match_capacity = env.sim.wp_model.opt.contact_sensor_maxmatch
    if match_capacity < 1:
        raise RuntimeError("missing positive native contact matching capacity")
    quality = tracker._quality_sensor
    return {"contact_gate_reducer": "maxforce", "delivered_reducer": "netforce",
            "point_slot_quality": "unavailable" if quality is None else "available",
            "point_slot_num_slots": None if quality is None else quality.cfg.num_slots,
            "contact_match_capacity": int(match_capacity)}


def contact_capacity(tracker, nacon, naconmax, nefc, njmax, nmatch, maxmatch):
    """Aggregation is not slot loss; guard actual simulation/matching capacities."""
    sensor_fields = [tracker._contact_sensor.data.found,
                     tracker._impulse_sensor.data.found, tracker._impulse_sensor.data.force]
    quality = tracker._quality_sensor
    point_overflow = torch.full_like(nefc, -1, dtype=torch.int8)
    if quality is not None:
        qdata = quality.data
        sensor_fields.extend(getattr(qdata, key) for key in ("found", "force", "pos", "normal"))
        point_overflow = (qdata.found.amax(dim=1) > quality.cfg.num_slots).to(torch.int8)
    if any(value is None or not bool(torch.isfinite(value).all()) for value in sensor_fields):
        raise RuntimeError("missing/nonfinite native contact sensor channel")
    global_overflow = (nefc > njmax) | bool((nacon > naconmax).any())
    match_overflow = (nmatch > maxmatch).any(dim=1)
    return {"overflow": global_overflow | match_overflow | (point_overflow == 1),
            "global_capacity_overflow": global_overflow,
            "contact_match_overflow": match_overflow,
            "point_slot_overflow": point_overflow}


def arm_velocity(env):
    """Use exactly the native velocity metric's resolved joint columns."""
    metric = env._hammer_substep_peak_qv
    value = metric._robot.data.joint_vel[:, metric._joint_ids]
    if value.ndim != 2 or value.shape[1] != 6:
        raise RuntimeError("native velocity metric must resolve exactly six arm joints")
    return value.clone()


def check_native_target(actual_target, actual_ctrl, processed, bias, expected_bias):
    """Validate effective commands after native apply/write, including encoder bias."""
    if not np.array_equal(bias, expected_bias):
        raise RuntimeError("encoder bias differs from frozen initial state")
    effective = processed - bias
    if not np.array_equal(actual_target, effective):
        raise RuntimeError("native joint position target differs from processed target minus encoder bias")
    if not np.array_equal(actual_ctrl, effective):
        raise RuntimeError("native actuator ctrl differs from effective position target")


def position_command(env):
    position = env.action_manager.get_term("joint_position")
    gains = env.action_manager.get_term("joint_stiffness")
    data = position._entity.data
    return {"native_target": data.joint_pos_target[:, position._target_ids].detach().cpu().numpy().copy(),
            "ctrl": env.sim.data.ctrl[:, gains.control_ids].detach().cpu().numpy().copy(),
            "encoder_bias": data.encoder_bias[:, position._target_ids].detach().cpu().numpy().copy()}


def sample(env):
    """Called after the native metrics substep, before any extra forward call."""
    tracker = env._hammer_first_strike
    gains = env.action_manager.get_term("joint_stiffness")
    data = env.sim.data
    contact = (tracker._contact_sensor.data.found > 0).any(dim=-1)
    force = tracker._impulse_sensor.data.force.sum(dim=-2)
    axis_force = (force * tracker._axis).sum(dim=-1).clamp_min(0)
    wp_data = env.sim.wp_data
    # MJWarp nmatch is temporary kernel scratch, not a Data field. The native
    # found output exposes that full match count for each monitored sensor.
    counts = [tracker._contact_sensor.data.found, tracker._impulse_sensor.data.found]
    if tracker._quality_sensor is not None:
        counts.append(tracker._quality_sensor.data.found)
    capacity = contact_capacity(tracker, wp.to_torch(wp_data.nacon), wp_data.naconmax,
                                wp.to_torch(wp_data.nefc), wp_data.njmax,
                                torch.cat(counts, dim=1),
                                env.sim.wp_model.opt.contact_sensor_maxmatch)
    result = {
        "qpos": data.qpos.clone(), "qvel": data.qvel.clone(),
        "velocity": arm_velocity(env),
        "reaction": env._hammer_substep_impulse.impulse.clone(),
        "delivered_native": tracker.delivered.clone(),
        "onset_s": tracker.first_contact_time_s.clone(),
        "finalized": tracker.finalized.clone(), "reason": tracker.reason.clone(),
        "contact": contact, "force_axial": axis_force,
        "actuator_force": data.actuator_force[:, gains.control_ids].clone(),
        "nail_depth": tracker._nail.data.joint_pos[:, tracker._joint_ids].squeeze(1).clone(),
        **capacity,
    }
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("nonfinite physical observation; refusing numerical attribution")
    if bool(result["overflow"].any()):
        raise RuntimeError("contact/constraint capacity overflow; refusing numerical attribution")
    result = {k: v.detach().cpu().numpy().copy() for k, v in result.items()}
    command = position_command(env)
    processed = env.action_manager.get_term("joint_position")._processed_actions.detach().cpu().numpy()
    check_native_target(command["native_target"], command["ctrl"], processed,
                        command["encoder_bias"], command["encoder_bias"])
    result.update(command)
    return result


def stacked(rows):
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
