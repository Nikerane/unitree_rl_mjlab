"""Explicit prospective engineering-equivalence profile; never a strict pass.

Pure NumPy checker. model_layout reads actual compiled-model joint metadata;
no simulator, checkpoint, or execution environment is constructed here.
"""
import numpy as np


PROFILE = "2026-09-22-prospective-engineering-v1"
ARM_VELOCITY_CAP = 3.1415


def model_layout(model, arm_joint_names, velocity_cap):
    if float(velocity_cap) != ARM_VELOCITY_CAP:
        raise ValueError("profile requires existing arm velocity cap 3.1415")
    types = np.asarray(model.jnt_type, dtype=int)
    qadr, vadr = np.asarray(model.jnt_qposadr, int), np.asarray(model.jnt_dofadr, int)
    if types.shape != (model.njnt,) or not np.isin(types, (2, 3)).all():
        raise ValueError("only compiled MuJoCo slide=2 and hinge=3 joints are supported")
    if (model.nq != model.njnt or model.nv != model.njnt or
        sorted(qadr.tolist()) != list(range(model.nq)) or sorted(vadr.tolist()) != list(range(model.nv))):
        raise ValueError("ambiguous/non-bijective hinge/slide joint addresses")
    try:
        arm_ids = [int(model.joint(name).id) for name in arm_joint_names]
    except (KeyError, IndexError) as exc:
        raise ValueError("canonical arm joint missing from compiled model") from exc
    if len(arm_ids) != 6 or len(set(arm_ids)) != 6 or any(i < 0 or i >= len(types) or types[i] != 3 for i in arm_ids):
        raise ValueError("exactly six unique compiled hinge arm joints required")
    qp = np.empty(model.nq)
    qv = np.full(model.nv, .005)
    for kind, qa in zip(types, qadr):
        qp[qa] = .0002 if kind == 3 else .0001
    arm_dof = vadr[arm_ids]
    qv[arm_dof] = .005 * ARM_VELOCITY_CAP
    return {"joint_types": types.tolist(), "qpos_addresses": qadr.tolist(),
            "dof_addresses": vadr.tolist(), "arm_joint_names": list(arm_joint_names),
            "arm_dof_ids": arm_dof.tolist(), "qpos_budget": qp.tolist(), "qvel_budget": qv.tolist()}


def engineering_check(actual, expected, terminal_steps, layout, impulse_caps, velocity_cap):
    """Compare every eligible native 2ms sample, with rtol=0 and peak flip guards."""
    caps = np.asarray(impulse_caps, float)
    if caps.shape != (6,) or not np.isfinite(caps).all() or np.any(caps <= 0):
        raise ValueError("six finite positive unchanged impulse caps required")
    if float(velocity_cap) != ARM_VELOCITY_CAP:
        raise ValueError("profile requires existing arm velocity cap 3.1415")
    terminal = np.asarray(terminal_steps, int)
    if terminal.ndim != 1 or not terminal.size or np.any(terminal < 0):
        raise ValueError("one first-terminal step per world required")
    actual_steps, expected_steps = len(actual["qpos"]), len(expected["qpos"])
    required = int((terminal.max() + 1) * 10)
    if expected_steps < required or actual_steps < required:
        raise ValueError("incomplete source or replay prefix")
    n = min(actual_steps, expected_steps)
    active = np.arange(n)[:, None] // 10 <= terminal[None, :]
    budgets = {"qpos": np.asarray(layout["qpos_budget"]),
               "qvel": np.asarray(layout["qvel_budget"]),
               "reaction": .005 * caps, "velocity": np.full(6, .005 * velocity_cap),
               "delivered_native": .001}
    exact = ("contact", "finalized", "reason", "native_target", "ctrl", "encoder_bias")
    failures, maxima, first = [], {}, {}
    finite = {}
    for key in (*budgets, *exact):
        a, b = np.asarray(actual[key])[:n], np.asarray(expected[key])[:n]
        if a.shape != b.shape or a.shape[:2] != active.shape:
            raise ValueError(f"unaligned channel shape: {key}")
        left, right = a[active], b[active]
        finite[key] = bool(np.isfinite(left).all() and np.isfinite(right).all())
        if not finite[key]:
            failures.append("nonfinite:" + key)
            maxima[key] = None
            continue
        if key in budgets:
            budget = np.asarray(budgets[key])
            if a.shape[2:] != budget.shape:
                raise ValueError(f"channel units/budget shape mismatch: {key}")
            error = np.abs(left - right)
            maxima[key] = np.max(error, axis=0).tolist()
            bad = np.abs(a - b) > budget
        else:
            bad = a != b
        masked = bad & active.reshape(active.shape + (1,) * (bad.ndim - 2))
        if masked.any():
            failures.append(key)
            idx = tuple(int(i) for i in np.argwhere(masked)[0])
            first[key] = {"physics_sample": idx[0], "world": idx[1],
                          "component": list(idx[2:]), "actual": a[idx].item(), "expected": b[idx].item()}
    flips, peaks = {}, {}
    for key, cap in (("reaction", caps), ("velocity", velocity_cap)):
        if not finite[key]:
            continue
        values = []
        for traces in (expected, actual):
            array = np.asarray(traces[key])[:n]
            if key == "velocity":
                array = np.abs(array)
            values.append(np.where(active[:, :, None], array, 0).max(axis=0))
        source_peak, replay_peak = values
        changed = (source_peak > cap) != (replay_peak > cap)
        flips[key] = np.argwhere(changed).tolist()
        peaks[key] = {"source": source_peak.tolist(), "replay": replay_peak.tolist()}
        if changed.any():
            failures.append(key + "_cap_classification")
    return {"profile": PROFILE, "status": "failed" if failures else "passed",
            "interpretation": "engineering equivalence only; does not alter strict verdict or authorize refinement",
            "failures": failures, "first_violations": first,
            "max_absolute_error_per_component": maxima,
            "budgets": {key: np.asarray(value).tolist() for key, value in budgets.items()},
            "rtol": 0, "exact_channels": list(exact), "layout": layout,
            "cap_classification_flips_world_joint": flips, "eligible_prefix_peaks": peaks,
            "active_samples_per_world": active.sum(axis=0).tolist(),
            "impulse_caps_unchanged": caps.tolist(), "velocity_cap_unchanged": velocity_cap}
