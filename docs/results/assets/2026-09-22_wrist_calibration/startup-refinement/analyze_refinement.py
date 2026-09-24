"""Offline, paired summary of the frozen contact refinement; no simulation."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3] / "docs/results/assets/2026-09-22_wrist_calibration"
CAPS = np.array([.369, .246, .738, .369, .246, .0164])


def quantiles(values):
    return dict(zip(("median", "p95", "max"), np.quantile(values, [.5, .95, 1], axis=0).tolist()))


def load(path):
    summary = json.loads((path / "summary.json").read_text())
    trace = np.load(path / "substeps.npz")
    return summary, trace


def main():
    output = {"comparison": "paired absolute differences, 1ms versus 0.5ms",
              "relative_delivered_denominator": "max(abs(0.5ms value), 0.001 N s)",
              "qualification": "diagnostic engineering screen, not hardware safety", "arms": []}
    for arm in range(4):
        paths = [ROOT / f"engineering-validation/arm-{arm}/native-audit-2ms",
                 ROOT / f"refinement/arm-{arm}/dt-1ms", ROOT / f"refinement/arm-{arm}/dt-0p5ms"]
        loaded = [load(path) for path in paths]
        entry = {"arm": arm, "controller": "original" if arm % 2 == 0 else "candidate128/25.6",
                 "dose": .2 if arm < 2 else .3, "levels": []}
        for d, z in loaded:
            dt = d["dt"]
            time = (np.arange(z["velocity"].shape[0]) + 1) * dt
            terminal = np.array(d["native_terminal_s"])
            # Shared per-world precontact cut uses the earlier onset across all dt levels.
            onset = np.min([np.array(s["first_contact_s"]) for s, _ in loaded], axis=0)
            masks = {"common_precontact": time[:, None] < onset[None, :] - 1e-7,
                     "native_prefix": time[:, None] <= terminal[None, :] + 1e-7,
                     "held_tail": time[:, None] > terminal[None, :] + 1e-7}
            level = {"dt": dt, "hits": sum(d["passive_started"]),
                     "complete_50ms": sum(d["passive_complete"]),
                     "nonfinite": d["nonfinite_count"],
                     "capacity_overflows": [d["global_capacity_overflow_count"], d["monitored_contact_match_buffer_overflow_count"]],
                     "native_success_count": int(np.sum(np.array(d["native_finalize_reason"]) == 1)),
                     "metrics": {}, "cuts": {}}
            for key in ("native_success_censored_delivered_ns", "passive_50ms_delivered_ns",
                        "reaction_cap_ratio", "velocity_cap_ratio", "contact_duration_s",
                        "peak_nail_depth_m", "saturation_s", "saturation_fraction", "force_sign_changes_per_s"):
                level["metrics"][key] = quantiles(d[key])
            for key in ("reaction_cap_ratio", "velocity_cap_ratio"):
                level[key + "_exceedance_counts"] = (np.array(d[key]) > 1).sum(axis=0).tolist()
            for cut, mask in masks.items():
                level["cuts"][cut] = {}
                for key, cap in (("reaction", CAPS), ("velocity", 3.1415)):
                    peak = np.max(np.where(mask[..., None], np.abs(z[key]), 0), axis=0) / cap
                    level["cuts"][cut][key] = {"quantiles": quantiles(peak),
                                                          "exceedance_counts": (peak > 1).sum(axis=0).tolist()}
            entry["levels"].append(level)
        a, b = [item[0] for item in loaded[1:]]
        entry["finest_pair"] = {}
        failures = []
        for key in ("native_success_censored_delivered_ns", "passive_50ms_delivered_ns",
                    "reaction_cap_ratio", "velocity_cap_ratio", "first_contact_s"):
            x, y = np.array(a[key]), np.array(b[key])
            diff = np.abs(x-y)
            if "delivered" in key:
                diff /= np.maximum(np.abs(y), .001)
            q = quantiles(diff)
            entry["finest_pair"][key] = q
            if key == "first_contact_s":
                passed = q["max"] <= .002 + 1e-7
            elif "delivered" in key:
                passed = q["median"] <= .05 and q["p95"] <= .05
            else:
                passed = (np.max(q["median"]) <= .05 and np.max(q["p95"]) <= .05 and np.max(q["max"]) <= .10)
            if not passed:
                failures.append(key)
        if not all(all(d["passive_started"]) and all(d["passive_complete"]) for d, _ in loaded):
            failures.append("miss_or_incomplete_window")
        entry["screen_failures"] = failures
        entry["screen_verdict"] = "failed" if failures else "passed"
        output["arms"].append(entry)
    path = ROOT / "refinement-analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(path), "verdicts": [a["screen_verdict"] for a in output["arms"]]}))


if __name__ == "__main__":
    main()
