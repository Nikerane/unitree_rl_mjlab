"""Phase D: the preregistered pooled analysis over 18 policies, 6 seeds per arm.

Endpoints and thresholds are read from the preregistration and applied as written.
Nothing here is chosen after seeing the data: the three straight-label clauses and
the primary endpoint were fixed in docs/results/2026-08-03_wave2_waypoint_replication_prereg.md
before Wave 2 trained. Counts are exact out of six. No significance claim is made
at n=6, and no policy is dropped.
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("evaluation/results/2026-08-03_wave2_waypoint")
TABLE = ROOT / "tables" / "wave1_wave2_eighteen_policy_comparison.csv"
ARMS = ("C0", "G", "P")
QVEL_LIMIT = 3.1415
# PREDECLARED, before Wave-2 data existed. Do not tune to the observed values.
STRAIGHT_MAX_PERP_MM = 15.0
STRAIGHT_MAX_PATH_RATIO = 1.15
REQUIRED_GATES = 6


def load():
    with TABLE.open() as fh:
        return list(csv.DictReader(fh))


def num(row, key):
    value = row[key]
    return float("nan") if value == "" else float(value)


def summarize(values):
    values = [v for v in values if v == v]
    if not values:
        return "n/a"
    return "%.2f  [%.2f, %.2f]" % (np.median(values), min(values), max(values))


def main() -> None:
    rows = load()
    if len(rows) != 18:
        raise SystemExit(f"expected 18 pooled rows, got {len(rows)}")

    for row in rows:
        row["_gates"] = int(row["gates_at_contact_onset"])
        row["_perp"] = num(row, "precontact_segment_perp_max_mm")
        row["_rms"] = num(row, "precontact_segment_perp_rms_mm")
        row["_ratio"] = num(row, "precontact_path_ratio")
        row["_straight"] = (
            row["_gates"] >= REQUIRED_GATES
            and row["_perp"] <= STRAIGHT_MAX_PERP_MM
            and row["_ratio"] <= STRAIGHT_MAX_PATH_RATIO
        )

    print("=== PRIMARY ENDPOINT: all six gates crossed by first contact onset ===")
    print("%-4s %-8s %s" % ("arm", "count", "per-seed gates@onset (seeds 2,3 | 4,5,6,7)"))
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        arm_rows.sort(key=lambda r: int(r["training_seed"]))
        hits = sum(1 for r in arm_rows if r["_gates"] >= REQUIRED_GATES)
        per = " ".join(
            ("%d" % r["_gates"]) + ("|" if r["training_seed"] == "3" else "")
            for r in arm_rows
        )
        print("%-4s %-8s %s" % (arm, "%d/6" % hits, per))

    print()
    print("=== PREDECLARED STRAIGHT LABEL (all three clauses) ===")
    print("  six gates by onset AND pre-contact max <= %.0f mm AND ratio <= %.2f"
          % (STRAIGHT_MAX_PERP_MM, STRAIGHT_MAX_PATH_RATIO))
    for arm in ARMS:
        arm_rows = sorted(
            (r for r in rows if r["arm"] == arm), key=lambda r: int(r["training_seed"])
        )
        hits = [r for r in arm_rows if r["_straight"]]
        print("  %-3s %s   straight seeds: %s" % (
            arm, "%d/6" % len(hits),
            ", ".join(r["training_seed"] for r in hits) or "none"))

    print()
    print("=== GATE-COMPLETERS THAT FAIL A STRAIGHTNESS CLAUSE ===")
    near = [r for r in rows if r["_gates"] >= REQUIRED_GATES and not r["_straight"]]
    if not near:
        print("  none")
    for r in near:
        why = []
        if r["_perp"] > STRAIGHT_MAX_PERP_MM:
            why.append("perp %.2f > %.0f mm" % (r["_perp"], STRAIGHT_MAX_PERP_MM))
        if r["_ratio"] > STRAIGHT_MAX_PATH_RATIO:
            why.append("ratio %.4f > %.2f" % (r["_ratio"], STRAIGHT_MAX_PATH_RATIO))
        print("  %s/%s  %s" % (r["arm"], r["training_seed"], "; ".join(why)))

    print()
    print("=== SECONDARY ENDPOINTS: median [min, max] over 6 seeds ===")
    header = ("arm", "perp_max_mm", "perp_rms_mm", "path_ratio",
              "depth_mm", "qvel_rad_s", "delivered_N_s", "max_cap_ratio")
    print("  " + "".join("%-22s" % h for h in header))
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        print("  " + "".join("%-22s" % v for v in (
            arm,
            summarize([r["_perp"] for r in arm_rows]),
            summarize([r["_rms"] for r in arm_rows]),
            summarize([r["_ratio"] for r in arm_rows]),
            summarize([num(r, "traj_nail_depth_physical_mm") for r in arm_rows]),
            summarize([num(r, "traj_peak_qvel_rad_s") for r in arm_rows]),
            summarize([num(r, "imp_delivered_n_s") for r in arm_rows]),
            summarize([num(r, "imp_max_cap_ratio") for r in arm_rows]),
        )))

    print()
    print("=== TASK OUTCOME AND SAFETY (all 18) ===")
    contact = sum(1 for r in rows if r["traj_has_contact"] == "True")
    success = sum(1 for r in rows if r["traj_success"] == "True")
    at_stop = sum(1 for r in rows
                  if abs(num(r, "traj_nail_depth_physical_mm") - 32.0) < 1e-6)
    legal = sum(1 for r in rows if r["traj_qvel_legal"] == "True")
    qvels = [num(r, "traj_peak_qvel_rad_s") for r in rows]
    caps = [num(r, "imp_max_cap_ratio") for r in rows]
    worst = max(rows, key=lambda r: num(r, "imp_max_cap_ratio"))
    print("  contact           : %d/18" % contact)
    print("  terminated success: %d/18" % success)
    print("  nail at 32 mm stop: %d/18" % at_stop)
    print("  qvel LEGAL        : %d/18   (limit %.4f rad/s; observed %.4f-%.4f)"
          % (legal, QVEL_LIMIT, min(qvels), max(qvels)))
    print("  max Lambda/cap    : %.6f  (%s/%s, joint %s)   over all 18: %.6f-%.6f"
          % (num(worst, "imp_max_cap_ratio"), worst["arm"], worst["training_seed"],
             worst["imp_max_cap_joint"], min(caps), max(caps)))

    payload = {
        "policies": len(rows),
        "seeds_per_arm": 6,
        "predeclared": {
            "required_gates": REQUIRED_GATES,
            "max_precontact_perp_mm": STRAIGHT_MAX_PERP_MM,
            "max_precontact_path_ratio": STRAIGHT_MAX_PATH_RATIO,
        },
        "primary_gate_completion": {
            arm: sum(1 for r in rows
                     if r["arm"] == arm and r["_gates"] >= REQUIRED_GATES)
            for arm in ARMS
        },
        "straight_label": {
            arm: sorted(int(r["training_seed"]) for r in rows
                        if r["arm"] == arm and r["_straight"])
            for arm in ARMS
        },
        "qvel_legal": legal,
        "max_cap_ratio": max(caps),
    }
    out = ROOT / "tables" / "wave2_pooled_summary.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print()
    print("wrote", out)


if __name__ == "__main__":
    main()
