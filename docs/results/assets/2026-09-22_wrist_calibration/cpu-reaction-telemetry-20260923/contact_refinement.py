"""Root-authorized 1/.5ms diagnostic after all four prospective native checks."""
import argparse
import hashlib
import json
from pathlib import Path


PROFILE = "2026-09-22-prospective-engineering-v1"
IDENTITY_KEYS = ("world_ids", "state_hash", "model_hash", "expanded_hash", "command_hash",
                 "runtime_identity", "horizon_s", "window_s", "control_dt")


def verify_identity(reference, current):
    for key in IDENTITY_KEYS:
        if reference[key] != current[key]:
            raise ValueError(f"refinement identity mismatch: {key}")


def validate_reports(root, selected_arm):
    if selected_arm not in range(4):
        raise ValueError("arm must be 0 through 3")
    root = Path(root).resolve()
    evidence, selected = [], None
    for arm in range(4):
        capture = root / f"contact-{arm}-capture"
        report_path = root / f"arm-{arm}" / "native-audit-2ms/engineering-equivalence.json"
        summary_path = report_path.with_name("summary.json")
        report = json.loads(report_path.read_text())
        summary = json.loads(summary_path.read_text())
        source = json.loads((capture / "capture.json").read_text())
        required = {"profile": PROFILE, "status": "passed", "failures": [],
                    "strict_verdict_preserved": "failed", "horizon_s": .28, "native_dt": .002,
                    "source_model_state_identity_gates": "passed",
                    "effective_target_ctrl_gain_bias_gates": "passed every substep",
                    "physical_finiteness_and_monitored_capacity_gates": "passed",
                    "cap_classification_flips_world_joint": {"reaction": [], "velocity": []}}
        for key, expected in required.items():
            if report.get(key) != expected:
                raise ValueError(f"arm {arm} engineering acceptance missing/invalid: {key}")
        if Path(report["source_capture"]).resolve() != capture:
            raise ValueError(f"arm {arm} report belongs to another source capture")
        for key, expected in {"dt": .002, "horizon_s": .28, "window_s": .05, "control_dt": .02,
                              "native_prefix_fidelity": "failed", "native_audit_only": True,
                              "engineering_equivalence_status": "passed",
                              "raw_contact_onset_matches_accepted": True}.items():
            if summary.get(key) != expected:
                raise ValueError(f"arm {arm} native summary inconsistent: {key}")
        for key in ("world_ids", "state_hash", "model_hash", "runtime_identity"):
            if summary[key] != source[key]:
                raise ValueError(f"arm {arm} capture identity differs from accepted native replay: {key}")
        evidence.append({"arm": arm, "report": str(report_path),
                         "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                         "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()})
        if arm == selected_arm:
            selected = (capture, summary)
    basis = {"basis": "explicit root authorization after all four fresh engineering checks passed",
             "protocol": "engineering-fidelity-protocol.md", "profile": PROFILE,
             "validation_job": "45737812", "strict_native_verdict": "failed",
             "strict_failure_cause": "unresolved; engineering acceptance is not strict reproduction",
             "engineering_acceptance": "all four passed", "arm": selected_arm,
             "horizon_steps": 14, "horizon_s": .28, "timesteps_s": [.001, .0005],
             "evidence": evidence, "qualification": "bounded numerical diagnostic only"}
    return *selected, basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, type=int, choices=range(4))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    capture, baseline, basis = validate_reports(args.validation_root, args.arm)
    # Import the existing simulator runner only after the read-only cohort gate.
    from contact_replay import replay
    from contact_runtime import write_json

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "basis.json", basis)
    previous = baseline
    for dt, name in ((.001, "dt-1ms"), (.0005, "dt-0p5ms")):
        try:
            result = replay(capture, args.output / name, dt, args.device, horizon_steps=14)
            verify_identity(baseline, result)
            verify_identity(previous, result)
            previous = result
        except Exception as exc:
            write_json(args.output / "failure.json", {"dt": dt, "error": str(exc),
                       "status": "refinement stopped; no further timestep executed"})
            raise
    write_json(args.output / "verification.json", {
        "identity_checks": "passed against accepted 2ms summary and between both refinements",
        "strict_native_verdict_preserved": "failed", "engineering_basis": PROFILE,
        "timesteps_s": [.001, .0005], "horizon_s": .28,
        "numerical_convergence": "not adjudicated; inspect physical results against independent budget",
    })
    print(json.dumps({"output": str(args.output), "status": "bounded refinement executed",
                      "strict_native_verdict": "failed", "basis": "engineering equivalence accepted"}))


if __name__ == "__main__":
    main()
