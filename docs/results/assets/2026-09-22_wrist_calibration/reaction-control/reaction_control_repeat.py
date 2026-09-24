"""One unobserved 0.25 ms control repeat against the original same-dt replay."""
import argparse
import json
from pathlib import Path

import numpy as np

from contact_refinement import validate_reports, verify_identity
from contact_runtime import write_json
from reaction_telemetry import compare_same_dt


DT = .00025
LEVEL = "dt-0p25ms"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, type=int, choices=(1, 3))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    capture, accepted, accepted_basis = validate_reports(args.validation_root, args.arm)
    engineering_path = (args.validation_root / f"arm-{args.arm}" /
                        "native-audit-2ms" / "engineering-equivalence.json")
    engineering = json.loads(engineering_path.read_text())
    if engineering["status"] != "passed" or engineering["profile"] != accepted_basis["profile"]:
        raise ValueError("compiled-layout report is not the accepted engineering validation")
    layout = engineering["layout"]
    reference_path = args.reference_root / f"arm-{args.arm}" / LEVEL
    reference_summary = json.loads((reference_path / "summary.json").read_text())
    verify_identity(accepted, reference_summary)
    if reference_summary["dt"] != DT:
        raise ValueError("reference must be original 0.25 ms replay")
    with np.load(reference_path / "substeps.npz") as archive:
        reference = dict(archive)
    with np.load(reference_path / "commands.npz") as archive:
        reference_commands = dict(archive)
    with np.load(capture / "tape.npz") as archive:
        terminal = archive["terminal_steps"].copy()
    capture_meta = json.loads((capture / "capture.json").read_text())

    from contact_replay import replay

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "basis.json", {
        **accepted_basis,
        "protocol": "reaction-control-brief.md: single uninstrumented same-dt repeat",
        "timesteps_s": [DT], "horizon_s": .28, "control_dt": .02, "window_s": .05,
        "reference": str(reference_path.resolve()), "engineering_layout": str(engineering_path.resolve()),
        "observer_enabled": False,
        "interpretation": "control only; no convergence, noise-floor, or contact-safety inference",
        "authorization": "one unobserved 0.25 ms repeat for candidate arms 1 and 3 only",
    })
    try:
        current = replay(capture, args.output / LEVEL, DT, args.device, horizon_steps=14)
        verify_identity(accepted, current)
        verify_identity(reference_summary, current)
        with np.load(args.output / LEVEL / "substeps.npz") as archive:
            observed = dict(archive)
        with np.load(args.output / LEVEL / "commands.npz") as archive:
            observed_commands = dict(archive)
        report = compare_same_dt(reference, observed, reference_summary, current,
                                 reference_commands, observed_commands, terminal,
                                 capture_meta["impulse_caps"], layout)
        report.update({"arm": args.arm, "dt": DT, "observer_enabled": False,
                       "reference": str(reference_path.resolve()),
                       "control": str((args.output / LEVEL).resolve()),
                       "interpretation": "single control comparison, not a statistical noise floor"})
        write_json(args.output / "noninterference-dt-0p25ms.json", report)
        if report["status"] != "passed":
            raise RuntimeError("unobserved same-dt control failed prospective non-interference gate")
    except Exception as exc:
        write_json(args.output / "failure.json", {"dt": DT, "error": str(exc),
                   "status": "data retained; control failed; no attribution authorized"})
        raise


if __name__ == "__main__":
    main()
