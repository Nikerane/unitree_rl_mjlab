"""Two full-cohort CPU replays at 0.25 ms; exact within-CPU repeatability only."""
import argparse
import json
from pathlib import Path

import numpy as np

from contact_refinement import validate_reports, verify_identity
from contact_runtime import write_json
from reaction_telemetry import compare_same_dt


DT = .00025
HORIZON_STEPS = 14


def exact_saved_arrays(first, second, *, samples, worlds):
    """Exact numeric-value comparison of every saved full-horizon channel."""
    if set(first) != set(second):
        raise ValueError("saved array inventory differs between CPU repeats")
    counts, errors, locations = {}, {}, {}
    for key in sorted(first):
        a, b = np.asarray(first[key]), np.asarray(second[key])
        if a.shape != b.shape or a.shape[:2] != (samples, worlds) or a.dtype != b.dtype:
            raise ValueError(f"saved array shape/dtype differs: {key}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"nonfinite saved array: {key}")
        changed = a != b
        counts[key] = int(np.count_nonzero(changed))
        errors[key] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        if counts[key]:
            locations[key] = np.argwhere(changed)[0].tolist()
    return {"status": "failed" if any(counts.values()) else "passed",
            "mismatch_count": counts, "max_absolute_error": errors,
            "first_mismatch": locations, "scope": "every saved value at every full-horizon sample"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, type=int, choices=(1, 3))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    capture, accepted, accepted_basis = validate_reports(args.validation_root, args.arm)
    engineering_path = (args.validation_root / f"arm-{args.arm}" /
                        "native-audit-2ms" / "engineering-equivalence.json")
    engineering = json.loads(engineering_path.read_text())
    if engineering["status"] != "passed" or engineering["profile"] != accepted_basis["profile"]:
        raise ValueError("compiled-layout report is not the accepted engineering validation")
    capture_meta = json.loads((capture / "capture.json").read_text())
    with np.load(capture / "tape.npz") as archive:
        terminal = archive["terminal_steps"].copy()

    from contact_replay import replay

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "basis.json", {
        **accepted_basis, "protocol": "cpu-repeat-brief.md: two exact CPU repeats",
        "timesteps_s": [DT], "horizon_steps": HORIZON_STEPS, "horizon_s": .28,
        "control_dt_s": .02, "window_s": .05, "num_worlds": 128,
        "device": "cpu", "observer_enabled": False, "repeat_count": 2,
        "engineering_layout": str(engineering_path.resolve()),
        "interpretation": "within-process fresh-env CPU repeatability only; no cross-process/host, GPU, convergence, utility, or safety qualification",
    })
    try:
        first = replay(capture, args.output / "repeat-1", DT, "cpu", horizon_steps=HORIZON_STEPS)
        verify_identity(accepted, first)
        second = replay(capture, args.output / "repeat-2", DT, "cpu", horizon_steps=HORIZON_STEPS)
        verify_identity(accepted, second)
        verify_identity(first, second)
        with np.load(args.output / "repeat-1/substeps.npz") as archive:
            first_trace = dict(archive)
        with np.load(args.output / "repeat-2/substeps.npz") as archive:
            second_trace = dict(archive)
        with np.load(args.output / "repeat-1/commands.npz") as archive:
            first_commands = dict(archive)
        with np.load(args.output / "repeat-2/commands.npz") as archive:
            second_commands = dict(archive)

        gate = compare_same_dt(first_trace, second_trace, first, second,
                               first_commands, second_commands, terminal,
                               capture_meta["impulse_caps"], engineering["layout"])
        exact_trace = exact_saved_arrays(first_trace, second_trace, samples=1120, worlds=128)
        exact_commands = exact_saved_arrays(first_commands, second_commands, samples=14, worlds=128)
        status = "passed" if all(result["status"] == "passed" for result in
                                 (gate, exact_trace, exact_commands)) else "failed"
        write_json(args.output / "comparison.json", {
            "status": status, "arm": args.arm, "device": "cpu", "dt": DT,
            "engineering_gate": gate, "exact_substeps": exact_trace,
            "exact_commands": exact_commands,
            "interpretation": "within-process fresh-env CPU only; no cross-process/host or GPU qualification, no automatic finer timestep"})
        if status != "passed":
            raise RuntimeError("CPU repeats failed existing engineering or exact full-array gate")
    except Exception as exc:
        write_json(args.output / "failure.json", {
            "status": "failed; saved data retained", "arm": args.arm, "device": "cpu",
            "dt": DT, "error": str(exc), "no_finer_stage_executed": True})
        raise


if __name__ == "__main__":
    main()
