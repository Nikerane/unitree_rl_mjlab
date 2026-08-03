"""Pooled 18-policy table: Wave-1 (seeds 2-3) + Wave-2 (seeds 4-7), 6 seeds per arm.

Extends `2026-08-02_wave1_waypoint/build_comparison_table.py` from 6 to 18 rows.
The arithmetic for every pre-existing column is carried over unchanged, so the
Wave-1 six rows project back to the frozen CSV
(4ab58b67658ef6c899a21ffbd200d4c568327352c72da0dd1c386437dda86268) BYTE-FOR-BYTE.
tests/test_wave2_pooled_table.py enforces that; a diff there means this extension
perturbed a measurement and must be fixed, never accepted as a re-measurement.

The one added measurement is `gates_at_contact_onset` -- the preregistered PRIMARY
endpoint. Wave 1's `traj_gates_completed` was a full-rollout maximum; it is kept
untouched for the projection, and both are reported so any disagreement is visible
rather than pooled away.

BOTH halves of every row are CPU fixed-reset runs of the SAME checkpoint from the
SAME realized reset digest, so the trajectory/impulse join is physically coherent.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

WAVE1_ROOT = Path("evaluation/results/2026-08-02_wave1_waypoint")
WAVE2_ROOT = Path("evaluation/results/2026-08-03_wave2_waypoint")
WAVE1_ORDER = [("C0", 2), ("C0", 3), ("G", 2), ("G", 3), ("P", 2), ("P", 3)]
WAVE2_ORDER = [(arm, seed) for arm in ("C0", "G", "P") for seed in (4, 5, 6, 7)]
EXPECTED_POOLED_ROWS = len(WAVE1_ORDER) + len(WAVE2_ORDER)
SHORT = {"C0": "c0", "G": "g", "P": "p"}
TREATMENT = {"C0": None, "G": "r_gate", "P": "r_waypoint_progress"}
QVEL_LIMIT = 3.1415
NAIL_STOP_M = 0.032

FROZEN_COLUMNS = (
    "arm", "training_seed", "checkpoint_sha256",
    "traj_has_contact", "traj_contact_substeps", "traj_contact_dwell_ms",
    "traj_success", "traj_terminal_reason", "traj_control_steps",
    "traj_nail_depth_physical_mm", "traj_nail_depth_raw_mm",
    "traj_gates_completed", "traj_first_contact_s",
    "precontact_segment_perp_max_mm", "precontact_segment_perp_rms_mm",
    "full_rollout_segment_perp_max_mm", "full_rollout_segment_perp_rms_mm",
    "precontact_window_substeps", "traj_path_len_ratio",
    "traj_backward_travel_mm", "traj_lateral_excursion_mm",
    "traj_peak_qvel_rad_s", "traj_qvel_legal", "traj_treatment_payout",
    "imp_v_precontact_m_s", "imp_delivered_n_s", "imp_axial_force_peak_n",
    "imp_lambda_max_j1", "imp_lambda_max_j2", "imp_lambda_max_j3",
    "imp_lambda_max_j4", "imp_lambda_max_j5", "imp_lambda_max_j6",
    "imp_max_cap_ratio", "imp_max_cap_joint", "imp_contact_substeps",
    "imp_npz_sha256",
)
# Appended, never inserted: a pre-existing column must not move or change.
POOLED_COLUMNS = FROZEN_COLUMNS + (
    "gates_at_contact_onset", "precontact_path_ratio", "campaign",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def precontact_slice(contact) -> slice:
    """Substeps [0 .. first contact onset], INCLUSIVE.

    No contact anywhere -> the whole rollout is pre-contact by construction.
    """
    contact = np.asarray(contact, dtype=bool)
    onset = int(np.argmax(contact)) if contact.any() else len(contact) - 1
    return slice(0, onset + 1)


def gates_at_contact_onset(gate_index, contact) -> int:
    """Gates crossed no later than the first contact-onset substep.

    The preregistered PRIMARY endpoint. Guidance is a claim about the APPROACH:
    a gate swept during follow-through, after the nail is already struck, is not
    guidance and must not be scored as such.
    """
    gate_index = np.asarray(gate_index)
    return int(gate_index[precontact_slice(contact)].max())


def precontact_path_ratio(pos, contact, entry) -> float:
    """Path length over straight-line chord, across the shared pre-contact window.

    Superset of the frozen `traj_path_len_ratio`: identical wherever that column
    is defined, and ALSO defined when a policy never contacts the nail -- where
    the frozen column emits an empty cell because it hardcodes a zero chord.
    That blank is not harmless: a policy can cross all six gates and stop short
    of the nail, which is exactly the guidance-succeeds/task-fails outcome this
    study exists to detect, and the predeclared straight label needs a ratio for
    it. The frozen column is left untouched; this one is appended beside it.
    """
    pos = np.asarray(pos, dtype=float)
    window = precontact_slice(contact)
    last = window.stop - 1
    path_len = float(np.linalg.norm(np.diff(pos[: last + 1], axis=0), axis=1).sum())
    chord = float(np.linalg.norm(pos[last] - np.asarray(entry, dtype=float)))
    return path_len / chord if chord > 0 else float("nan")


def assert_join_identity(rmeta, dmeta, *, campaign, arm, seed, checkpoint_sha256, leaf):
    """Verify both halves of a row describe the SAME policy and the SAME reset.

    The docstring claim that the trajectory and impulse halves are the same
    physical rollout is the load-bearing assumption of every joined row, and both
    metadata files already carry the identity to check it with. Trusting the
    directory name instead means one mis-named render out of twelve joins one
    policy's trajectory to another's provenance, silently.
    """
    expected = {
        "campaign": campaign,
        "arm": arm,
        "training_seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for name, meta in (("renderer", rmeta), ("impulse", dmeta)):
        for key, want in expected.items():
            got = meta.get(key)
            if key == "training_seed":
                got = int(got) if got is not None else None
            if got != want:
                raise ValueError(
                    f"join identity mismatch in {name} half of {leaf}: "
                    f"{key}={got!r}, expected {want!r}"
                )
    if rmeta.get("reset_state_digest") != dmeta.get("reset_state_digest"):
        raise ValueError(
            f"join identity mismatch in {leaf}: the two halves used different "
            f"fixed resets ({rmeta.get('reset_state_digest')!r} vs "
            f"{dmeta.get('reset_state_digest')!r})"
        )


def _campaign_paths(campaign: str, arm: str, seed: int):
    root = WAVE1_ROOT if campaign == "wave1" else WAVE2_ROOT
    leaf = f"{campaign}_{SHORT[arm]}_seed{seed}"
    return root / "videos" / leaf, root / "impulse" / leaf


def _checkpoint_sha(campaign: str, arm: str, seed: int) -> str:
    if campaign == "wave1":
        man = json.loads((WAVE1_ROOT / "artifact_manifest.json").read_text())
        rows = {(p["arm"], p["seed"]): p["checkpoint_sha256"] for p in man["policies"]}
    else:
        man = json.loads((WAVE2_ROOT / "wave2_training_inventory.json").read_text())
        rows = {(r["arm"], r["seed"]): r["checkpoint_sha256"] for r in man["runs"]}
    return rows[(arm, seed)]


def build_rows(order, campaign: str) -> list[dict]:
    """Measure one campaign's policies. Column arithmetic is frozen; do not edit."""
    out_rows = []
    for arm, seed in order:
        vleaf, ileaf = _campaign_paths(campaign, arm, seed)
        r = dict(np.load(vleaf / "trace.npz"))
        rmeta = json.loads((vleaf / "metadata.json").read_text())
        d = dict(np.load(ileaf / "trace.npz"))
        dmeta = json.loads((ileaf / "metadata.json").read_text())
        checkpoint_sha256 = _checkpoint_sha(campaign, arm, seed)
        assert_join_identity(
            rmeta, dmeta, campaign=campaign, arm=arm, seed=seed,
            checkpoint_sha256=checkpoint_sha256, leaf=vleaf.name,
        )

        pos = r["substep_head_position_m"]
        contact = r["substep_contact"].astype(bool)
        entry = r["guideline_entry_m"]
        nail = r["guideline_nail_m"]
        axis = nail - entry
        axis_u = axis / np.linalg.norm(axis)

        # Perpendicular error: ONLY the PRODUCTION-recorded field. Never re-derived.
        # guideline.py:project_to_reference() computes it against the FINITE
        # entry->nail SEGMENT (progress clamped to [0,1]).
        rel = pos - entry
        along = rel @ axis_u
        perp = r["substep_perpendicular_error_m"]

        # ONE pre-contact window, from the unit-tested helper, shared by every
        # pre-contact metric. Deriving it three times invites silent divergence.
        window = precontact_slice(contact)
        onset = window.stop - 1
        perp_pre = perp[window]

        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        if contact.any():
            first_c = onset
            path_len = float(seg[:first_c].sum())
            direct = float(np.linalg.norm(pos[first_c] - entry))
            ratio = path_len / direct if direct > 0 else float("nan")
            # dt from the trace, not a literal: a new campaign must not inherit
            # Wave-1's timing by assumption. (Both record 0.002.)
            t_first = first_c * float(r["physics_dt_s"])
        else:
            first_c, path_len, direct, ratio, t_first = (
                -1, float(seg.sum()), 0.0, float("nan"), float("nan"))

        dalong = np.diff(along)
        backward = float(-dalong[dalong < 0].sum())
        # lateral excursion / backward travel deliberately use the UNCLAMPED axis:
        # they measure sideways drift and reversal along the guideline direction,
        # which are NOT the finite-segment perpendicular error and feed no
        # preregistered endpoint. The "never re-derived" rule above scopes to the
        # perpendicular-error columns only. (Carried over verbatim from the frozen
        # Wave-1 generator; dropping this note made the file read as self-contradictory.)
        lateral = float(np.abs(rel - np.outer(along, axis_u))[:, 1].max())

        peak_qvel = float(np.abs(r["substep_arm_qvel_rad_s"]).max())

        lam = np.abs(d["lambda_windowed_constraint_read_n_m_s"])
        caps = np.array(dmeta["j_limit_n_m_s"], dtype=float)
        lam_peak = lam.max(axis=0)
        lam_ratio = lam_peak / caps

        payouts = r["control_step_reward_terms"]
        names = [str(n) for n in r["control_step_reward_term_names"]]
        treat = TREATMENT[arm]
        treat_payout = float(payouts[:, names.index(treat)].sum()) if treat else 0.0

        out_rows.append({
            "arm": arm, "training_seed": seed,
            "checkpoint_sha256": checkpoint_sha256[:16],
            "traj_has_contact": bool(contact.any()),
            "traj_contact_substeps": int(contact.sum()),
            "traj_contact_dwell_ms": round(float(contact.sum()) * 2.0, 1),
            "traj_success": rmeta["rollout"]["terminal_boundary"]["reason"] == "terminated",
            "traj_terminal_reason": rmeta["rollout"]["terminal_boundary"]["reason"],
            "traj_control_steps": int(r["executed_control_steps"]),
            "traj_nail_depth_physical_mm": round(
                min(float(r["substep_nail_depth_m"].max()), NAIL_STOP_M) * 1e3, 3),
            "traj_nail_depth_raw_mm": round(float(r["substep_nail_depth_m"].max()) * 1e3, 3),
            # full-rollout maximum: the Wave-1 column, kept for the frozen projection
            "traj_gates_completed": int(r["substep_gate_index"].max()),
            "traj_first_contact_s": round(t_first, 4) if t_first == t_first else "",
            "precontact_segment_perp_max_mm": round(float(perp_pre.max()) * 1e3, 2),
            "precontact_segment_perp_rms_mm": round(
                float(np.sqrt((perp_pre ** 2).mean())) * 1e3, 2),
            "full_rollout_segment_perp_max_mm": round(float(perp.max()) * 1e3, 2),
            "full_rollout_segment_perp_rms_mm": round(
                float(np.sqrt((perp ** 2).mean())) * 1e3, 2),
            "precontact_window_substeps": int(onset + 1),
            "traj_path_len_ratio": round(ratio, 4) if ratio == ratio else "",
            "traj_backward_travel_mm": round(backward * 1e3, 2),
            "traj_lateral_excursion_mm": round(lateral * 1e3, 2),
            "traj_peak_qvel_rad_s": round(peak_qvel, 4),
            "traj_qvel_legal": bool(peak_qvel <= QVEL_LIMIT),
            "traj_treatment_payout": round(treat_payout, 6),
            "imp_v_precontact_m_s": round(float(np.max(d["tracker_v_precontact_m_s"])), 4),
            "imp_delivered_n_s": round(float(np.abs(d["delivered_impulse_n_s"]).max()), 6),
            "imp_axial_force_peak_n": round(float(np.abs(d["axial_force_n"]).max()), 3),
            "imp_lambda_max_j1": round(float(lam_peak[0]), 6),
            "imp_lambda_max_j2": round(float(lam_peak[1]), 6),
            "imp_lambda_max_j3": round(float(lam_peak[2]), 6),
            "imp_lambda_max_j4": round(float(lam_peak[3]), 6),
            "imp_lambda_max_j5": round(float(lam_peak[4]), 6),
            "imp_lambda_max_j6": round(float(lam_peak[5]), 6),
            "imp_max_cap_ratio": round(float(lam_ratio.max()), 6),
            "imp_max_cap_joint": int(lam_ratio.argmax()) + 1,
            "imp_contact_substeps": int(d["contact"].astype(bool).sum()),
            "imp_npz_sha256": dmeta["npz_sha256"][:16],
            # --- added in Wave 2 ---
            "gates_at_contact_onset": gates_at_contact_onset(
                r["substep_gate_index"], contact),
            "precontact_path_ratio": round(
                precontact_path_ratio(pos, contact, entry), 4),
            "campaign": campaign,
        })
    return out_rows


def write_csv(path, rows, columns) -> None:
    """Write `columns` only, dropping any extras -- this is what makes the
    Wave-1 projection byte-comparable to the frozen CSV.

    A MISSING key is an error, never a blank cell: DictWriter's default would
    emit "" for a typo'd column name, which reads downstream as a measured
    value rather than a bug.
    """
    columns = list(columns)
    for index, row in enumerate(rows):
        missing = [name for name in columns if name not in row]
        if missing:
            raise ValueError(f"row {index} missing declared columns: {missing}")
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows(WAVE1_ORDER, "wave1") + build_rows(WAVE2_ORDER, "wave2")
    if len(rows) != EXPECTED_POOLED_ROWS:
        raise SystemExit(
            f"pooled table must carry all {EXPECTED_POOLED_ROWS} policies without "
            f"selection; got {len(rows)}"
        )
    tables = WAVE2_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    out = tables / "wave1_wave2_eighteen_policy_comparison.csv"
    write_csv(out, rows, POOLED_COLUMNS)

    # Re-prove the frozen projection on every run, not only in the test suite.
    projection = tables / "wave1_projection_check.csv"
    write_csv(projection, rows[: len(WAVE1_ORDER)], FROZEN_COLUMNS)
    frozen = (
        WAVE1_ROOT / "tables" / "wave1_six_policy_comparison.csv"
    ).read_bytes()
    if projection.read_bytes() != frozen:
        raise SystemExit(
            "Wave-1 projection is no longer byte-identical to the frozen CSV; "
            "the extension perturbed a measurement"
        )
    projection.unlink()
    print("Wave-1 projection: byte-identical to the frozen CSV  OK")

    print("=== pooled 18-policy table ===")
    hdr = ("arm/seed", "camp", "contact", "succ", "depth_mm", "gates@onset",
           "gates_full", "preP_max", "preP_rms", "pathR", "qvel", "legal", "payout")
    print("  " + "".join(f"{h:>12}" for h in hdr))
    for x in rows:
        print("  " + "".join(f"{v:>12}" for v in (
            f"{x['arm']}/{x['training_seed']}", x["campaign"],
            x["traj_contact_substeps"], str(x["traj_success"])[:5],
            x["traj_nail_depth_physical_mm"], x["gates_at_contact_onset"],
            x["traj_gates_completed"], x["precontact_segment_perp_max_mm"],
            x["precontact_segment_perp_rms_mm"], x["traj_path_len_ratio"],
            x["traj_peak_qvel_rad_s"], str(x["traj_qvel_legal"])[:5],
            x["traj_treatment_payout"])))

    disagree = [x for x in rows
                if x["gates_at_contact_onset"] != x["traj_gates_completed"]]
    print()
    print("policies where onset and full-rollout gate counts DISAGREE:", len(disagree))
    for x in disagree:
        print("  %s/%s  onset=%d  full=%d" % (
            x["arm"], x["training_seed"], x["gates_at_contact_onset"],
            x["traj_gates_completed"]))

    print()
    print("rows:", len(rows), " table sha256:", sha256(out)[:16])


if __name__ == "__main__":
    main()
