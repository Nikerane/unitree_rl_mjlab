"""Read-only, same-timestep reaction telemetry for the two authorized candidates.

This wraps the configured accumulator's existing call and replay's existing sample.
It neither changes the simulator path nor computes a replacement reaction signal.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch

from contact_refinement import validate_reports, verify_identity
from contact_runtime import write_json


CANONICAL_JOINTS = [f"joint{i}" for i in range(1, 7)]
EXACT_CHANNELS = ("native_target", "ctrl", "encoder_bias", "contact", "finalized", "reason")
HEALTH_CHANNELS = ("overflow", "global_capacity_overflow", "contact_match_overflow")


def _copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.array(value, copy=True)


class _Observer:
    def __init__(self):
        self.pending = None
        self.config = None
        self.layout = None
        self.samples = 0

    def before(self, metric, env):
        if self.pending is not None:
            raise RuntimeError("unconsumed accumulator call before next substep")
        if metric is not env._hammer_substep_impulse:
            raise RuntimeError("call is not the configured accumulator instance")
        gains = env.action_manager.get_term("joint_stiffness")
        velocity = env._hammer_substep_peak_qv
        robot = env.scene["robot"]
        ids = metric._joint_ids
        all_names = list(robot.joint_names)

        def resolved(selected):
            return list(range(len(all_names)))[selected] if isinstance(selected, slice) else [int(i) for i in selected]

        selected_ids = resolved(ids)
        if (list(gains.joint_names) != CANONICAL_JOINTS or metric._robot is not robot
                or velocity._robot is not robot or selected_ids != resolved(velocity._joint_ids)
                or len(selected_ids) != 6 or len(set(selected_ids)) != 6
                or [all_names[i] for i in selected_ids] != CANONICAL_JOINTS):
            raise RuntimeError("reaction joint order differs from canonical velocity/gain order")
        qfrc = metric._robot.data._joint_dof_field("qfrc_constraint")[:, ids]
        baseline = metric._last_off_qfrc
        gate = (metric._sensor.data.found > 0).any(dim=-1)
        if qfrc.shape != baseline.shape or qfrc.ndim != 2 or qfrc.shape[1] != 6 or gate.shape != qfrc.shape[:1]:
            raise RuntimeError("unexpected configured accumulator input shape")
        if not bool(torch.isfinite(qfrc).all()) or not bool(torch.isfinite(baseline).all()):
            raise RuntimeError("nonfinite raw reaction or baseline at accumulator call")
        cfg = {"subtract_baseline": bool(metric._subtract_baseline),
               "window_substeps": int(metric._window), "decimation": int(metric._dec),
               "joint_names": CANONICAL_JOINTS}
        if self.config is None:
            self.config = cfg
            if hasattr(env, "sim"):
                from engineering_fidelity import model_layout
                self.layout = model_layout(env.sim.mj_model,
                                           [f"robot/{name}" for name in CANONICAL_JOINTS], 3.1415)
        elif cfg != self.config:
            raise RuntimeError("configured reaction accumulator changed during replay")
        if metric._i != self.samples:
            raise RuntimeError("reaction metric counter differs from observed substep count")
        self.pending = {"reaction_raw_qfrc": _copy(qfrc),
                        "reaction_baseline_before": _copy(baseline),
                        "reaction_contact_gate": _copy(gate),
                        "reaction_metric_i_before": np.int64(metric._i),
                        "_env": env, "_metric": metric}

    def after(self, metric, env, returned):
        if self.pending is None or self.pending["_env"] is not env or self.pending["_metric"] is not metric:
            raise RuntimeError("reaction call/sample association lost")
        slot = (metric._buf_i - 1) % metric._window
        self.pending.update({
            "reaction_baseline_after": _copy(metric._last_off_qfrc),
            "reaction_contribution": _copy(metric._buf[:, :, slot]),
            "reaction_rolling": _copy(metric._rolling),
            "reaction_pulse": _copy(metric._pulse),
            "reaction_impulse_after": _copy(metric.impulse if hasattr(metric, "impulse") else torch.maximum(metric._pulse, metric._rolling)),
            "reaction_metric_return": _copy(returned),
            "reaction_metric_i_after": np.int64(metric._i),
            "reaction_buf_i_after": np.int64(metric._buf_i),
        })
        if metric._i != self.samples + 1 or not np.isfinite(self.pending["reaction_contribution"]).all():
            raise RuntimeError("reaction accumulator call did not advance exactly one finite substep")

    def sample(self, env, original):
        row = original(env)
        pending = self.pending
        if pending is None or pending["_env"] is not env or "reaction_contribution" not in pending:
            raise RuntimeError("replay sample does not follow the configured reaction call")
        if "contact" not in row or not np.array_equal(row["contact"], pending["reaction_contact_gate"]):
            raise RuntimeError("sample contact differs from accumulator's same-substep gate")
        if any(key in row for key in pending if not key.startswith("_")):
            raise RuntimeError("reaction telemetry collides with existing replay channel")
        row.update({k: v for k, v in pending.items() if not k.startswith("_")})
        self.pending = None
        self.samples += 1
        return row


@contextmanager
def observe_replay(accumulator_class, replay_module):
    observer = _Observer()
    original_call = accumulator_class.__call__
    original_sample = replay_module.sample

    def call(metric, env, **params):
        observer.before(metric, env)
        result = original_call(metric, env, **params)
        observer.after(metric, env, result)
        return result

    def sample(env):
        return observer.sample(env, original_sample)

    accumulator_class.__call__ = call
    replay_module.sample = sample
    try:
        yield observer
        if observer.pending is not None:
            raise RuntimeError("unconsumed reaction substep at replay exit")
    finally:
        accumulator_class.__call__ = original_call
        replay_module.sample = original_sample


def write_report(output, level_name, report):
    write_json(Path(output) / f"noninterference-{level_name}.json", report)


def telemetry_basis(accepted_basis, reference_root):
    return {**accepted_basis,
            "protocol": "task-5-brief.md: same-dt reaction telemetry",
            "timesteps_s": [.00025, .000125],
            "scope": "read-only reaction observer; same .25/.125ms pair, 128 worlds, 0.28s",
            "reference_root": str(Path(reference_root).resolve()),
            "authorization": "root: candidate arms 1 and 3 only; no extra timestep levels"}


def compare_same_dt(reference, observed, reference_summary, observed_summary, commands_a,
                    commands_b, terminal_steps, caps, layout):
    """Prospective non-interference gate, full horizon plus native terminal prefixes."""
    for key in ("world_ids", "state_hash", "model_hash", "expanded_hash", "runtime_identity",
                "command_hash", "horizon_s", "window_s", "control_dt", "dt"):
        if reference_summary[key] != observed_summary[key]:
            raise ValueError(f"same-dt identity mismatch: {key}")
    if set(commands_a) != set(commands_b) or any(not np.array_equal(commands_a[k], commands_b[k]) for k in commands_a):
        raise ValueError("full frozen target/Kp/Kd command tapes differ")
    if set(commands_a) != {"target", "kp", "kd"}:
        raise ValueError("unexpected frozen command tape channels")
    if set(reference) != {k for k in observed if not k.startswith("reaction_")}:
        raise ValueError("instrumented replay changed existing trace channel inventory")
    dt = float(reference_summary["dt"])
    n = round(float(reference_summary["horizon_s"]) / dt)
    terminal = np.asarray(terminal_steps, dtype=int)
    caps = np.asarray(caps, dtype=float)
    if (n < 1 or terminal.ndim != 1 or terminal.size != 128 or caps.shape != (6,)
            or not np.isfinite(caps).all() or np.any(caps <= 0)
            or layout["joint_types"] and any(t not in (2, 3) for t in layout["joint_types"])):
        raise ValueError("unsupported shape, cap, or free/ball layout")
    budget = {"reaction": .005 * caps, "delivered_native": .001,
              "velocity": np.full(6, .005 * 3.1415),
              "qpos": np.full(len(layout["qpos_addresses"]), .0001),
              "qvel": np.full(len(layout["dof_addresses"]), .005)}
    failures, maxima, first = [], {}, {}
    for key in ("nonfinite_count", "global_capacity_overflow_count",
                "monitored_contact_match_buffer_overflow_count", "point_slot_overflow_count"):
        for summary in (reference_summary, observed_summary):
            expected = None if key == "point_slot_overflow_count" else 0
            if summary.get(key) not in (0, expected):
                if "reported_health:" + key not in failures:
                    failures.append("reported_health:" + key)
    for key in reference:
        a, b = np.asarray(reference[key]), np.asarray(observed[key])
        if a.shape != b.shape or a.shape[:2] != (n, 128):
            raise ValueError(f"unaligned full-horizon channel: {key}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            failures.append("nonfinite:" + key)
        if key in (*HEALTH_CHANNELS, "point_slot_overflow"):
            if not np.array_equal(a, b) or (key == "point_slot_overflow" and np.any(a > 0)) or (key != "point_slot_overflow" and np.any(a)):
                failures.append("capacity:" + key)
        if key in EXACT_CHANNELS and not np.array_equal(a, b):
            failures.append("exact:" + key)
        if key in budget:
            limit = np.asarray(budget[key])
            if a.shape[2:] != limit.shape:
                raise ValueError(f"budget/layout mismatch: {key}")
            diff = np.abs(a - b)
            maxima[key] = np.max(diff, axis=(0, 1)).tolist()
            if np.any(diff > limit):
                failures.append("budget:" + key)
                first[key] = np.argwhere(diff > limit)[0].tolist()
    active = np.arange(n)[:, None] // round(.02 / dt) <= terminal[None, :]
    flips = {}
    for key, cap in (("reaction", caps), ("velocity", 3.1415)):
        flips[key] = {}
        for scope, mask in (("full_horizon", np.ones((n, 128), bool)), ("source_terminal_prefix", active)):
            peaks = []
            for traces in (reference, observed):
                values = np.asarray(traces[key])
                if key == "velocity":
                    values = np.abs(values)
                peaks.append(np.where(mask[:, :, None], values, 0).max(axis=0))
            changed = (peaks[0] > cap) != (peaks[1] > cap)
            flips[key][scope] = np.argwhere(changed).tolist()
            if changed.any():
                failures.append(f"cap_classification:{key}:{scope}")
    return {"status": "failed" if failures else "passed", "failures": failures,
            "first_budget_violation": first, "max_absolute_error_per_component": maxima,
            "budgets": {k: np.asarray(v).tolist() for k, v in budget.items()},
            "exact_channels": list(EXACT_CHANNELS), "cap_classification_flips": flips,
            "scope": "full 0.28 s same-dt trace; caps also checked through each source-native terminal prefix",
            "rtol": 0, "attribution_allowed": not failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, type=int, choices=(1, 3))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    capture, accepted, basis = validate_reports(args.validation_root, args.arm)
    import contact_replay
    from src.tasks.hammer.mdp.impulse_bound import SubstepImpulseAccumulator

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "basis.json", telemetry_basis(basis, args.reference_root))
    with np.load(capture / "tape.npz") as tape:
        terminal = tape["terminal_steps"].copy()
    with (capture / "capture.json").open() as stream:
        meta = json.load(stream)
    for dt, name in ((.00025, "dt-0p25ms"), (.000125, "dt-0p125ms")):
        try:
            reference_path = args.reference_root / f"arm-{args.arm}" / name
            reference_summary = json.loads((reference_path / "summary.json").read_text())
            verify_identity(accepted, reference_summary)
            if reference_summary["dt"] != dt:
                raise ValueError("reference is not the original same-dt level")
            with np.load(reference_path / "substeps.npz") as archive:
                reference = dict(archive)
            with np.load(reference_path / "commands.npz") as archive:
                commands_a = dict(archive)
            output = args.output / name
            with observe_replay(SubstepImpulseAccumulator, contact_replay) as observer:
                observed_summary = contact_replay.replay(capture, output, dt, args.device, horizon_steps=14)
            if observer.layout is None or observer.config is None or observer.samples != round(.28 / dt):
                raise RuntimeError("incomplete reaction observer coverage or unsupported model layout")
            if (observer.config["window_substeps"] != round(.05 / dt)
                    or observer.config["decimation"] != round(.02 / dt)):
                raise RuntimeError("configured reaction window/decimation differs from replay")
            with np.load(output / "substeps.npz") as archive:
                observed = dict(archive)
            with np.load(output / "commands.npz") as archive:
                commands_b = dict(archive)
            report = compare_same_dt(reference, observed, reference_summary, observed_summary,
                                     commands_a, commands_b, terminal, meta["impulse_caps"], observer.layout)
            report.update({"arm": args.arm, "dt": dt, "reference": str(reference_path.resolve()),
                           "telemetry": str(output.resolve()), "accumulator_config": observer.config,
                           "observed_substeps": observer.samples,
                           "limitations": ["generalized qfrc includes non-contact constraints",
                                           "nonlinear abs(raw-baseline) prevents additive absolute attribution",
                                           "no contact-row isolation or convergence verdict"]})
            write_report(args.output, name, report)
            if report["status"] != "passed":
                raise RuntimeError(f"same-dt non-interference failed at {name}; attribution blocked")
        except Exception as exc:
            write_json(args.output / "failure.json", {"dt": dt, "error": str(exc),
                       "status": "data retained; attribution blocked; no later level executed"})
            raise
    write_json(args.output / "verification.json", {"status": "passed",
               "meaning": "same-dt non-interference only; offline reconstruction and attribution remain root work",
               "levels_s": [.00025, .000125], "horizon_s": .28})


if __name__ == "__main__":
    main()
