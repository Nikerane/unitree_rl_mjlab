"""Explicit candidate-only .25/.125ms diagnostic; never adds another level."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_refinement import quantiles
from contact_refinement import validate_reports, verify_identity


def startup_comparison(a, dt_a, b, dt_b):
    peaks = []
    for trace, dt in ((a, dt_a), (b, dt_b)):
        n = round(.06 / dt)
        velocity, contact = trace["velocity"], trace["contact"]
        if (not np.isclose(n * dt, .06, rtol=0, atol=1e-12)
                or velocity.ndim != 3 or velocity.shape[2] != 6
                or velocity.shape[0] < n or contact.shape != velocity.shape[:2]):
            raise ValueError("startup requires complete 60ms six-joint traces")
        if not np.isfinite(velocity[:n]).all() or np.any(contact[:n]):
            raise ValueError("startup contains contact or nonfinite velocity")
        peaks.append(np.abs(velocity[:n]).max(axis=0))
    if peaks[0].shape != peaks[1].shape or not peaks[0].shape[0]:
        raise ValueError("startup world dimensions differ or are empty")
    q = quantiles(np.abs(peaks[0] - peaks[1]) / 3.1415)
    passed = (np.max(q["median"]) <= .05 and np.max(q["p95"]) <= .05
              and np.max(q["max"]) <= .10)
    return {"status": "passed" if passed else "failed", "window_s": .06,
            "timesteps_s": [dt_a, dt_b], "velocity_cap_rad_s": 3.1415,
            "peak_difference_cap_ratio": q, "no_contact": True,
            "peak_rad_s": [p.tolist() for p in peaks],
            "criteria": {"median": .05, "p95": .05, "max": .10},
            "scope": "startup only; does not establish full-contact convergence"}


def full_contact_comparison(a, b):
    """Same paired screen as analyze_refinement, reported independently of startup."""
    metrics, failures = {}, []
    for key in ("native_success_censored_delivered_ns", "passive_50ms_delivered_ns",
                "reaction_cap_ratio", "velocity_cap_ratio", "first_contact_s"):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"invalid full-contact metric: {key}")
        diff = np.abs(x - y)
        if "delivered" in key:
            diff /= np.maximum(np.abs(y), .001)
        q = quantiles(diff)
        metrics[key] = q
        if key == "first_contact_s":
            passed = q["max"] <= .002 + 1e-7
        elif "delivered" in key:
            passed = q["median"] <= .05 and q["p95"] <= .05
        else:
            passed = (np.max(q["median"]) <= .05 and np.max(q["p95"]) <= .05
                      and np.max(q["max"]) <= .10)
        if not passed:
            failures.append(key)
    if not all(all(s["passive_started"]) and all(s["passive_complete"]) for s in (a, b)):
        failures.append("miss_or_incomplete_window")
    success_counts = []
    for summary in (a, b):
        reasons = np.asarray(summary.get("native_finalize_reason", []))
        world_count = len(summary["first_contact_s"])
        valid = reasons.shape == (world_count,) and world_count > 0
        success_counts.append(int(np.sum(reasons == 1)) if valid else None)
        if not valid or not np.all(reasons == 1):
            if "native_success_incomplete_or_invalid" not in failures:
                failures.append("native_success_incomplete_or_invalid")
    return {"status": "failed" if failures else "passed", "failures": failures,
            "native_success_counts": success_counts,
            "metrics": metrics, "timesteps_s": [a["dt"], b["dt"]],
            "relative_delivered_denominator": "max(abs(finer value), 0.001 N s)",
            "scope": "full 280ms replay; 50ms passive impulse, including held-command tail"}


def verify_health(summary):
    for key in ("nonfinite_count", "global_capacity_overflow_count",
                "monitored_contact_match_buffer_overflow_count"):
        if summary.get(key) != 0:
            raise ValueError(f"invalid replay health: {key}")
    if summary.get("point_slot_overflow_count") not in (None, 0):
        raise ValueError("point slot overflow")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path,
                        help="previous refinement root containing arm-N/dt-0p5ms")
    parser.add_argument("--arm", required=True, type=int, choices=(1, 3))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    capture, baseline, basis = validate_reports(args.validation_root, args.arm)
    reference_path = args.reference_root / f"arm-{args.arm}/dt-0p5ms"
    previous = json.loads((reference_path / "summary.json").read_text())
    if previous["dt"] != .0005:
        raise ValueError("reference must be 0.5ms")
    verify_identity(baseline, previous)
    verify_health(previous)
    with np.load(reference_path / "substeps.npz") as data:
        previous_trace = {k: data[k] for k in ("velocity", "contact")}
    with np.load(reference_path / "commands.npz") as data:
        reference_commands = dict(data)
    startup_comparison(previous_trace, .0005, previous_trace, .0005)
    from contact_replay import replay
    from contact_runtime import write_json

    args.output.mkdir(parents=True, exist_ok=False)
    basis.update({"timesteps_s": [.00025, .000125], "reference": str(reference_path.resolve()),
                  "authorization": "root: candidate arms 1,3 only; no additional levels",
                  "startup_window_s": .06, "startup_peak_difference_cap_ratio_limits":
                  {"median": .05, "p95": .05, "max": .10},
                  "full_contact_verdict": "separate from startup; never inferred"})
    write_json(args.output / "basis.json", basis)
    comparisons = []
    for dt, name in ((.00025, "dt-0p25ms"), (.000125, "dt-0p125ms")):
        try:
            output = args.output / name
            current = replay(capture, output, dt, args.device, horizon_steps=14)
            verify_identity(baseline, current)
            verify_identity(previous, current)
            verify_health(current)
            with np.load(output / "commands.npz") as commands:
                if set(commands.files) != set(reference_commands) or any(
                    not np.array_equal(commands[k], reference_commands[k]) for k in commands.files
                ):
                    raise ValueError("command tape differs from 0.5ms reference")
            with np.load(output / "substeps.npz") as data:
                current_trace = {k: data[k] for k in ("velocity", "contact")}
            comparison = {"startup": startup_comparison(previous_trace, previous["dt"], current_trace, dt),
                          "full_contact": full_contact_comparison(previous, current)}
            comparisons.append(comparison)
            write_json(args.output / "comparisons.json", comparisons)
            previous, previous_trace = current, current_trace
        except Exception as exc:
            write_json(args.output / "failure.json", {"dt": dt, "error": str(exc),
                       "status": "stopped; no further timestep executed"})
            raise
    write_json(args.output / "verification.json", {
        "identity_checks": "passed against original accepted 2ms and all refinements",
        "commands": "exact arrays equal to 0.5ms reference; native application checked by replay",
        "strict_native_verdict_preserved": "failed", "timesteps_s": [.00025, .000125],
        "horizon_s": .28, "additional_levels_authorized": False,
        "startup_finest_pair_status": comparisons[-1]["startup"]["status"],
        "full_contact_finest_pair_status": comparisons[-1]["full_contact"]["status"],
        "qualification": "diagnostic only; no training or safety qualification"})


if __name__ == "__main__":
    main()
