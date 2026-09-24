"""One 0.125 ms CPU replay after both exact 0.25 ms CPU-repeat gates passed."""
import argparse
import json
from pathlib import Path

import numpy as np

from contact_refinement import validate_reports, verify_identity
from contact_runtime import write_json
from contact_startup_refinement import startup_comparison, full_contact_comparison, verify_health


COARSE_DT = .00025
FINE_DT = .000125


def require_cpu_repeats(root):
    evidence = {}
    for arm in (1, 3):
        path = Path(root) / f"arm-{arm}" / "comparison.json"
        report = json.loads(path.read_text())
        if (report.get("status") != "passed" or report.get("device") != "cpu"
                or report.get("dt") != COARSE_DT):
            raise ValueError(f"arm {arm} CPU 0.25 ms repeat gate is not passed")
        for key in ("engineering_gate", "exact_substeps", "exact_commands"):
            part = report.get(key, {})
            if part.get("status") != "passed":
                raise ValueError(f"arm {arm} {key} gate is not passed")
            if key.startswith("exact_"):
                counts = part.get("mismatch_count")
                if not isinstance(counts, dict) or not counts or any(value != 0 for value in counts.values()):
                    raise ValueError(f"arm {arm} {key} has unverified exact mismatches")
        evidence[arm] = str(path.resolve())
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--cpu-repeat-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, type=int, choices=(1, 3))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repeat_evidence = require_cpu_repeats(args.cpu_repeat_root)
    capture, accepted, accepted_basis = validate_reports(args.validation_root, args.arm)
    reference_path = args.cpu_repeat_root / f"arm-{args.arm}" / "repeat-1"
    repeat2_path = args.cpu_repeat_root / f"arm-{args.arm}" / "repeat-2"
    previous = json.loads((reference_path / "summary.json").read_text())
    repeat2 = json.loads((repeat2_path / "summary.json").read_text())
    if previous["dt"] != COARSE_DT or repeat2["dt"] != COARSE_DT:
        raise ValueError("CPU repeat reference must be the verified 0.25 ms pair")
    verify_identity(accepted, previous)
    verify_identity(accepted, repeat2)
    verify_identity(previous, repeat2)
    verify_health(previous)
    verify_health(repeat2)
    with np.load(reference_path / "substeps.npz") as archive:
        previous_trace = {key: archive[key] for key in ("velocity", "contact")}
    with np.load(reference_path / "commands.npz") as archive:
        reference_commands = dict(archive)
    if set(reference_commands) != {"target", "kp", "kd"}:
        raise ValueError("unexpected CPU reference command channels")

    from contact_replay import replay

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "basis.json", {
        **accepted_basis,
        "protocol": "cpu-timestep-brief.md: conditional single finer CPU level",
        "device": "cpu", "timesteps_s": [COARSE_DT, FINE_DT],
        "horizon_s": .28, "control_dt_s": .02, "window_s": .05, "num_worlds": 128,
        "capture": str(capture.resolve()), "reference": str(reference_path.resolve()),
        "repeat2": str(repeat2_path.resolve()), "cpu_repeat_evidence": repeat_evidence,
        "held_tail_limit": "full 0.28 s fixed-command horizon includes postterminal held-command tail",
        "interpretation": "bounded two-timestep CPU comparison only; no GPU, gain-safety or convergence qualification",
    })
    try:
        current = replay(capture, args.output / "dt-0p125ms", FINE_DT, "cpu", horizon_steps=14)
        verify_identity(accepted, current)
        verify_identity(previous, current)
        verify_health(current)
        with np.load(args.output / "dt-0p125ms/commands.npz") as archive:
            commands = dict(archive)
        if set(commands) != set(reference_commands) or any(
            not np.array_equal(commands[key], reference_commands[key]) for key in reference_commands
        ):
            raise ValueError("full target/Kp/Kd command tape differs from verified CPU 0.25 ms repeat")
        with np.load(args.output / "dt-0p125ms/substeps.npz") as archive:
            current_trace = {key: archive[key] for key in ("velocity", "contact")}
        comparison = {
            "startup": startup_comparison(previous_trace, COARSE_DT, current_trace, FINE_DT),
            "full_contact": full_contact_comparison(previous, current),
        }
        scientific_pass = all(value["status"] == "passed" for value in comparison.values())
        write_json(args.output / "comparison.json", {
            "status": "passed" if scientific_pass else "scientific_failed",
            "arm": args.arm, "device": "cpu", "timesteps_s": [COARSE_DT, FINE_DT],
            "commands_exact": True, "startup": comparison["startup"],
            "full_contact": comparison["full_contact"],
            "source_terminal_prefix_reaction_analysis": "still required for interpretation",
            "held_tail_limit": "full 0.28 s includes postterminal held-command tail",
            "interpretation": "fixed-command candidates only; no GPU or safe-gain qualification",
        })
    except Exception as exc:
        write_json(args.output / "failure.json", {
            "status": "process_error", "arm": args.arm, "dt": FINE_DT,
            "error": str(exc), "saved_data_retained": True, "no_further_level": True})
        raise
    if not scientific_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
