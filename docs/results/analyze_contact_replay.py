"""Offline descriptive analysis of the approved first-episode contact replay.

Reads only immutable trace/evaluation artifacts; emits JSON to stdout. No
simulation, learning, policy changes, or inferred causal attribution.
"""
import argparse
import json
from pathlib import Path

import torch


def first_true(mask):
    indices = mask.nonzero().flatten()
    return int(indices[0]) + 1 if indices.numel() else None


def interval_stats(data, world, mask):
    count = int(mask.sum())
    if not count:
        return {"samples": 0}
    result = {"samples": count}
    for field in ("kp", "kd", "saturated", "joint_velocity_rad_s"):
        result[field + "_mean"] = data[field][:, world][mask].double().mean(0).tolist()
    return result


def analyze_condition(path, rows):
    data = torch.load(path / "contact-trace.pt", map_location="cpu", weights_only=True)
    meta = json.loads((path / "contact-trace.json").read_text())
    dt = meta["physics_dt_s"]
    assert dt == .00025
    n = data["valid"].shape[1]
    assert len(rows) == n and {r["world"] for r in rows} == set(range(n))
    result = []
    for row in sorted(rows, key=lambda r: r["world"]):
        w = row["world"]
        valid = data["valid"][:, w]
        count = int(valid.sum())
        assert valid[:count].all() and not valid[count:].any()
        measure = row["measurement"]
        assert count == measure["observed_samples"]
        onset = first_true(data["accepted_onset"][:, w] & valid)
        assert int((data["accepted_onset"][:, w] & valid).sum()) <= 1
        success = int(data["native_success_sample"][w]) or None
        assert success == measure["native_success_sample"]
        load = data["joint_load_nms"][:count, w].double()
        rolling = data["rolling_joint_load_nms"][:count, w].double()
        peak = data["control_step_peak_load_nms"][:count, w].double()
        torch.testing.assert_close(load, torch.maximum(rolling, peak), rtol=0, atol=0)
        torch.testing.assert_close(load.amax(0), torch.tensor(
            measure["loads"]["episode"]["peak_nms"], dtype=torch.float64), rtol=0, atol=0)
        caps = torch.tensor(measure["loads"]["cap_nms"], dtype=torch.float64)
        exceed = load > caps
        roll_exceed = rolling > caps
        assert exceed.any(0).tolist() == measure["loads"]["episode"]["exceeded"]
        joints = []
        for j in range(6):
            first = first_true(exceed[:, j])
            joints.append({
                "joint": j + 1, "peak_nms": float(load[:, j].max()),
                "utilization": float((load[:, j] / caps[j]).max()),
                "first_exceedance_sample": first,
                "first_rolling_exceedance_sample": first_true(roll_exceed[:, j]),
                "first_exceedance_relative_contact_ms": (first - onset) * dt * 1000 if first and onset else None,
                "first_exceedance_relative_success_ms": (first - success) * dt * 1000 if first and success else None,
                "exceeded_by_native_success": bool(exceed[:success, j].any()) if success else None,
                "first_exceedance_after_native_success": bool(first > success) if first and success else False,
                "held_only_exceedance_samples": int((exceed[:, j] & ~roll_exceed[:, j]).sum()),
                "exceedance_samples": int(exceed[:, j].sum()),
                "reaction_abs_peak_nm": float(data["qfrc_constraint_nm"][:count, w, j].abs().max()),
                "saturated_sample_fraction": float(data["saturated"][:count, w, j].double().mean()),
            })
        record = {"world": w, "goal_mps": row["goal_mps"],
                  "measured_mps": row["measured_mps"], "selection_count": row["selection_count"],
                  "onset_sample": onset, "native_success_sample": success,
                  "observed_samples": count, "joints": joints,
                  "common_window": measure["common_window"], "gain_intervals": {}}
        if onset:
            assert abs(measure["time_to_contact_s"] - onset * dt) < 1e-10
            relative = torch.arange(data["valid"].shape[0]) + 1 - onset
            for name, lo, hi in (("pre_20ms", -80, 0), ("contact_0_20ms", 0, 80),
                                 ("contact_20_50ms", 80, 200)):
                record["gain_intervals"][name] = interval_stats(
                    data, w, valid & (relative >= lo) & (relative < hi))
            record["baseline_at_onset_nm"] = data["off_contact_baseline_nm"][onset-1, w].tolist()
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    output = {"schema": "contact_replay_analysis_v1", "runs": []}
    for run in range(6):
        directory = args.campaign / "results" / f"run-{run}"
        rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
        first = [r for r in rows if r["episode"] == 0]
        conditions = []
        for condition in range(3):
            selected = [r for r in first if r["condition"] == condition]
            conditions.append({"condition": condition, "episodes": analyze_condition(
                directory / f"condition-{condition}", selected)})
        output["runs"].append({"run": run, "gain_mode": "variable" if run % 2 else "fixed",
                               "training_seed": 10 + run // 2, "conditions": conditions})
    print(json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
