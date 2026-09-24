"""Bounded CPU reaction observation against split, exact same-dt CPU references."""
import argparse
import json
from pathlib import Path

import numpy as np

from contact_refinement import validate_reports, verify_identity
from contact_runtime import write_json
from contact_startup_refinement import verify_health
from cpu_repeat import exact_saved_arrays
from reaction_telemetry import compare_same_dt, observe_replay, write_report


LEVELS = ((.00025, "dt-0p25ms"), (.000125, "dt-0p125ms"))


def reference_path(coarse_root, fine_root, arm, dt):
    if dt == .00025:
        return Path(coarse_root) / f"arm-{arm}" / "repeat-1"
    if dt == .000125:
        return Path(fine_root) / f"arm-{arm}" / "dt-0p125ms"
    raise ValueError("only the already executed 0.25/0.125 ms CPU levels are authorized")


def exact_original_channels(reference, observed, *, samples, worlds):
    if set(reference) != {key for key in observed if not key.startswith("reaction_")}:
        raise ValueError("original trace channel inventory differs")
    return exact_saved_arrays(reference, {key: observed[key] for key in reference},
                              samples=samples, worlds=worlds)


def require_coarse_gate(output, arm):
    path = Path(output) / "noninterference-dt-0p25ms.json"
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("passed coarse noninterference gate is required before finer replay") from exc
    if (report.get("status") != "passed" or report.get("arm") != arm
            or report.get("device") != "cpu" or report.get("dt") != .00025
            or any(report.get(key, {}).get("status") != "passed" for key in
                   ("engineering_gate", "exact_original_substeps", "exact_commands"))):
        raise ValueError("passed coarse noninterference gate is required before finer replay")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--coarse-reference-root", required=True, type=Path)
    parser.add_argument("--fine-reference-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, type=int, choices=(1, 3))
    parser.add_argument("--dt", required=True, type=float, choices=(.00025, .000125))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    name = dict(LEVELS)[args.dt]

    capture, accepted, accepted_basis = validate_reports(args.validation_root, args.arm)
    with np.load(capture / "tape.npz") as archive:
        terminal = archive["terminal_steps"].copy()
    meta = json.loads((capture / "capture.json").read_text())
    import contact_replay
    from src.tasks.hammer.mdp.impulse_bound import SubstepImpulseAccumulator

    args.output.mkdir(parents=True, exist_ok=True)
    if args.dt == .000125:
        require_coarse_gate(args.output, args.arm)
    write_json(args.output / f"basis-{name}.json", {
        **accepted_basis,
        "protocol": "2026-09-23 wrist reaction decision goal: exact CPU telemetry",
        "device": "cpu", "timestep_s": args.dt,
        "horizon_steps": 14, "horizon_s": .28, "control_dt_s": .02,
        "window_s": .05, "num_worlds": 128, "scheduler_limit_s": 600,
        "capture": str(capture.resolve()),
        "coarse_reference_root": str(args.coarse_reference_root.resolve()),
        "fine_reference_root": str(args.fine_reference_root.resolve()),
        "observer": "configured SubstepImpulseAccumulator at actual call; unchanged replay",
        "interpretation": "raw generalized force, not isolated contact load; exact CPU noninterference required before attribution",
    })
    dt = args.dt
    try:
        reference_dir = reference_path(args.coarse_reference_root, args.fine_reference_root,
                                       args.arm, dt)
        reference_summary = json.loads((reference_dir / "summary.json").read_text())
        verify_identity(accepted, reference_summary)
        verify_health(reference_summary)
        if reference_summary["dt"] != dt:
            raise ValueError("CPU reference is not the matching original timestep")
        with np.load(reference_dir / "substeps.npz") as archive:
            reference = dict(archive)
        with np.load(reference_dir / "commands.npz") as archive:
            reference_commands = dict(archive)

        with observe_replay(SubstepImpulseAccumulator, contact_replay) as observer:
            current = contact_replay.replay(capture, args.output / name, dt, "cpu",
                                            horizon_steps=14)
        verify_identity(accepted, current)
        verify_identity(reference_summary, current)
        verify_health(current)
        if (observer.layout is None or observer.config is None
                or observer.samples != round(.28 / dt)
                or observer.config["window_substeps"] != round(.05 / dt)
                or observer.config["decimation"] != round(.02 / dt)):
            raise RuntimeError("incomplete or mismatched configured accumulator observation")
        with np.load(args.output / name / "substeps.npz") as archive:
            observed = dict(archive)
        with np.load(args.output / name / "commands.npz") as archive:
            commands = dict(archive)
        engineering = compare_same_dt(reference, observed, reference_summary, current,
                                      reference_commands, commands, terminal,
                                      meta["impulse_caps"], observer.layout)
        exact_trace = exact_original_channels(reference, observed,
                                              samples=round(.28 / dt), worlds=128)
        exact_commands = exact_saved_arrays(reference_commands, commands,
                                            samples=14, worlds=128)
        status = "passed" if all(item["status"] == "passed" for item in
                                 (engineering, exact_trace, exact_commands)) else "failed"
        write_report(args.output, name, {
            "status": status, "arm": args.arm, "device": "cpu", "dt": dt,
            "reference": str(reference_dir.resolve()),
            "telemetry": str((args.output / name).resolve()),
            "observed_substeps": observer.samples,
            "accumulator_config": observer.config,
            "engineering_gate": engineering,
            "exact_original_substeps": exact_trace,
            "exact_commands": exact_commands,
            "attribution_allowed": status == "passed",
            "limitations": ["raw qfrc_constraint includes multiple constraint types",
                            "postterminal held-command tail is not a source-terminal prefix",
                            "no GPU or gain-safety qualification"],
        })
        if status != "passed":
            raise RuntimeError(f"CPU telemetry changed the original {name} output; attribution blocked")
    except Exception as exc:
        write_json(args.output / f"failure-{name}.json", {
            "status": "stopped; evidence retained; attribution blocked",
            "arm": args.arm, "dt": dt, "error": str(exc),
            "no_later_level_or_retry": True})
        raise
    write_json(args.output / f"verification-{name}.json", {
        "status": "passed", "meaning": "CPU same-dt noninterference only; reconstruction and attribution remain separate",
        "timestep_s": dt, "device": "cpu", "horizon_s": .28})


if __name__ == "__main__":
    main()
