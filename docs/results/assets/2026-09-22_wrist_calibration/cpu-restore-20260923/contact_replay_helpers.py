"""Simulator-free physical-time and frozen-tape contracts."""
import math
import numpy as np


def duration_steps(duration, dt):
    if not all(math.isfinite(v) and v > 0 for v in (duration, dt)):
        raise ValueError("duration and timestep must be finite positive")
    n = round(duration / dt)
    if n < 1 or not math.isclose(n * dt, duration, abs_tol=1e-10, rel_tol=0):
        raise ValueError("timestep must divide physical duration exactly")
    return n


def command_at(tape, terminal_steps, step):
    tape, terminal = np.asarray(tape), np.asarray(terminal_steps, dtype=int)
    if tape.ndim < 2 or terminal.shape != (tape.shape[1],):
        raise ValueError("tape must have time/world axes and one terminal index per world")
    if step < 0 or np.any(terminal < 0) or np.any(terminal >= len(tape)):
        raise ValueError("missing or out-of-range first termination")
    return tape[np.minimum(step, terminal), np.arange(len(terminal))].copy()


def prefix_mask(terminal_steps, step):
    return step <= np.asarray(terminal_steps)


def horizon_steps(onset_times, control_dt=.02):
    onset = np.asarray(onset_times)
    if not onset.size or not np.isfinite(onset).all() or np.any(onset <= 0):
        raise ValueError("fixed source horizon requires an observed contact in every world")
    return math.ceil((float(onset.max()) + .05 + .02) / control_dt - 1e-10)


def passive_window(contact, force_axial, dt, duration=.05):
    """Uncensored wall-time window, inclusive onset, even across contact gaps."""
    contact, force = np.asarray(contact, bool), np.asarray(force_axial)
    if contact.ndim != 2 or force.shape != contact.shape or not np.isfinite(force).all():
        raise ValueError("require finite, matching [substep, world] sensor arrays")
    n = duration_steps(duration, dt)
    started = contact.any(axis=0)
    onset = contact.argmax(axis=0)
    slots = np.arange(len(contact))[:, None]
    inside = started & (slots >= onset) & (slots < onset + n)
    return {"started": started, "onset_s": np.where(started, (onset + 1) * dt, 0),
            "complete": started & (onset + n <= len(contact)),
            "impulse": (np.maximum(force, 0) * contact * inside).sum(axis=0) * dt}


def require_pair(left, right):
    for key in ("world_ids", "state_hash"):
        if left[key] != right[key]:
            raise ValueError(f"replay pairing mismatch: {key}")


def check_fidelity(actual, expected, active, tolerances):
    errors = {}
    if not np.any(active):
        return errors
    for key, atol in tolerances.items():
        left, right = actual[key][active], expected[key][active]
        errors[key] = float(np.max(np.abs(left - right)))
        if not np.allclose(left, right, rtol=1e-5, atol=atol):
            raise ValueError(f"native replay fidelity failed: {key} maxabs={errors[key]}")
    for key in ("contact", "reason", "finalized"):
        if key in expected and not np.array_equal(actual[key][active], expected[key][active]):
            raise ValueError(f"native discrete-event fidelity failed: {key}")
    return errors


def fidelity_detail(actual, expected, active, tolerances):
    """Audit the unchanged fidelity gate without throwing away later mismatches."""
    active = np.asarray(active, bool)
    arrays, first = {"active": active.copy()}, None
    for key, atol in tolerances.items():
        left, right = np.asarray(actual[key]), np.asarray(expected[key])
        error = np.abs(left - right).reshape(len(active), -1)
        budget = (atol + 1e-5 * np.abs(right)).reshape(len(active), -1)
        bad = active & (error > budget).any(axis=1)
        arrays["abs_" + key] = np.where(active, error.max(axis=1), 0)
        arrays["bad_" + key] = bad
        if first is None and bad.any():
            world = int(np.flatnonzero(bad)[0])
            first = {"key": key, "world": world, "actual": left[world].tolist(),
                     "expected": right[world].tolist(), "atol": atol, "rtol": 1e-5,
                     "first_violating_flat_component": int(np.flatnonzero(error[world] > budget[world])[0])}
    for key in ("contact", "reason", "finalized"):
        if key not in expected:
            continue
        left, right = np.asarray(actual[key]), np.asarray(expected[key])
        bad = active & (left != right).reshape(len(active), -1).any(axis=1)
        arrays["bad_" + key] = bad
        if first is None and bad.any():
            world = int(np.flatnonzero(bad)[0])
            first = {"key": key, "world": world, "actual": left[world].tolist(),
                     "expected": right[world].tolist(), "exact": True}
    return {"arrays": arrays, "first_failure": first}


def replay_schedule(native_audit=False):
    if native_audit:
        return ((.002, "native-audit-2ms"),)
    return ((.002, "dt-2ms"), (.001, "dt-1ms"), (.0005, "dt-0p5ms"))


def native_audit_verdict(first_failure, arming_match):
    reasons = []
    if first_failure is not None:
        reasons.append("unchanged source-prefix fidelity gate failed")
    if not arming_match:
        reasons.append("raw contact onset differs from accepted first-strike onset")
    return ("failed" if reasons else "passed"), reasons
